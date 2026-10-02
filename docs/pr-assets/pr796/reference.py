# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Reference from PR #796 at 7ff0fd60; subcell=False reproduces the pre-PR RK1/RK3 RHS.

The legacy path uses its original absolute Peng epsilon. RK2 legacy differs at the clamp,
so comparisons in these figures use RK3 only.
"""

import torch
import fvdb

_FACE_OFFSETS = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))


def reference_redistance(
    grid: "fvdb.Grid", field: torch.Tensor, band: int, iters: int, order: int, subcell: bool = True
) -> torch.Tensor:
    """Float64 torch transcription of the CUDA redistance, the generator for the pinned values below.

    Mirrors ReinitializeSdf.cu step for step: inactive faces read as +/-band*vx with the sign of the
    voxel doing the reading; interface cells (a strict sign change across an active face in the input)
    relax to the Russo-Smereka distance D with denominator max(central gradient norm, largest
    one-sided slope, eps), shared with the steepest crossing neighbour; all other cells take the Godunov update with the frozen Peng sign; RK1,
    SSP Heun, and Shu-Osher RK3 with dt = 0.4 vx and a band clamp after every stage."""
    vx = float(grid.voxel_size[0])
    band_width = band * vx
    dt = 0.4 * vx
    ijk = grid.ijk
    face_index = torch.stack(
        [grid.ijk_to_index(ijk + torch.tensor(o, device=ijk.device, dtype=ijk.dtype)) for o in _FACE_OFFSETS], dim=1
    )
    active = face_index >= 0

    def faces(phi: torch.Tensor, sign_source: torch.Tensor) -> torch.Tensor:
        # Build the inactive value in phi's dtype; torch.where on two Python floats would yield float32.
        inactive = torch.where(sign_source < 0, -1.0, 1.0).to(phi.dtype) * band_width
        out = inactive[:, None].expand(-1, 6).clone()
        out[active] = phi[face_index[active]]
        return out

    phi0 = field.reshape(-1).double()
    phi0_faces = faces(phi0, phi0)
    center = phi0[:, None]
    interface = (phi0_faces * center < 0).any(dim=1)

    forward = torch.where(active[:, 1::2], (phi0_faces[:, 1::2] - center).abs(), torch.zeros_like(center))
    backward = torch.where(active[:, 0::2], (center - phi0_faces[:, 0::2]).abs(), torch.zeros_like(center))
    both = active[:, 1::2] & active[:, 0::2]
    axis_gradient = torch.where(
        both, (phi0_faces[:, 1::2] - phi0_faces[:, 0::2]).abs() / 2, torch.maximum(forward, backward)
    )
    denominator = torch.maximum(torch.maximum(forward, backward).max(dim=1).values, axis_gradient.norm(dim=1))
    denominator = denominator.clamp(min=1e-6 * vx)
    # Each interface cell shares the larger denominator with its steepest crossing neighbour so both
    # ends of a crossing edge scale alike and the interpolated crossing stays put.
    crossing = phi0_faces * center < 0
    edge_slope = torch.where(crossing, (phi0_faces - center).abs(), torch.full_like(phi0_faces, -1.0))
    steepest = face_index.gather(1, edge_slope.argmax(dim=1, keepdim=True)).squeeze(1).clamp(min=0)
    denominator = torch.where(interface, torch.maximum(denominator, denominator[steepest]), denominator)
    distance = vx * phi0 / denominator

    central = (phi0_faces[:, 1::2] - phi0_faces[:, 0::2]) / (2 * vx)
    frozen_sign = phi0 / torch.sqrt(
        phi0 * phi0 + central.pow(2).sum(dim=1) * vx * vx + (1e-10 * vx * vx if subcell else 1e-12)
    )

    def rhs(phi: torch.Tensor) -> torch.Tensor:
        f = faces(phi, frozen_sign)
        back = (phi[:, None] - f[:, 0::2]) / vx
        fwd = (f[:, 1::2] - phi[:, None]) / vx
        positive = torch.maximum(back.clamp(min=0) ** 2, fwd.clamp(max=0) ** 2)
        negative = torch.maximum(back.clamp(max=0) ** 2, fwd.clamp(min=0) ** 2)
        gradient = torch.where(frozen_sign[:, None] > 0, positive, negative).sum(dim=1).sqrt()
        return (
            torch.where(interface, (distance - phi) / vx, frozen_sign * (1 - gradient))
            if subcell
            else frozen_sign * (1 - gradient)
        )

    def clamp(phi: torch.Tensor) -> torch.Tensor:
        return phi.clamp(-band_width, band_width)

    phi = phi0.clone()
    for _ in range(iters):
        stage1 = clamp(phi + dt * rhs(phi))
        if order == 1:
            phi = stage1
        elif order == 2:
            phi = clamp(0.5 * phi + 0.5 * stage1 + 0.5 * dt * rhs(stage1))
        else:
            stage2 = clamp(0.75 * phi + 0.25 * stage1 + 0.25 * dt * rhs(stage1))
            phi = clamp(phi / 3 + 2 * stage2 / 3 + 2 * dt * rhs(stage2) / 3)
    return phi.to(field.dtype)

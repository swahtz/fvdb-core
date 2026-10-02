# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Ray-casting helpers from the original chair reproduction."""

import math
import torch


def look_at_rays(eye, target, up, res, fov_deg, device):
    """Perspective camera rays. Returns origins (res*res, 3) and unit directions (res*res, 3)."""
    eye = torch.as_tensor(eye, dtype=torch.float32, device=device)
    target = torch.as_tensor(target, dtype=torch.float32, device=device)
    up = torch.as_tensor(up, dtype=torch.float32, device=device)

    fwd = torch.nn.functional.normalize(target - eye, dim=0)
    right = torch.nn.functional.normalize(torch.linalg.cross(fwd, up), dim=0)
    cam_up = torch.linalg.cross(right, fwd)

    half = math.tan(math.radians(fov_deg) / 2)
    lin = torch.linspace(-half, half, res, device=device)
    v, u = torch.meshgrid(-lin, lin, indexing="ij")
    dirs = fwd[None, None] + u[..., None] * right[None, None] + v[..., None] * cam_up[None, None]
    dirs = torch.nn.functional.normalize(dirs.reshape(-1, 3), dim=1)
    origins = eye[None].expand_as(dirs).contiguous()
    return origins, dirs


def render_sdf(grid, sdf, origins, dirs, res, light_dir):
    """Ray-cast the zero level-set. Returns (shaded rgb HxWx3, depth HxW with NaN where no hit)."""
    t = grid.ray_implicit_intersection(origins, dirs, sdf.reshape(-1))
    hit = t >= 0

    depth = torch.full((res * res,), float("nan"), device=sdf.device)
    depth[hit] = t[hit]

    rgb = torch.ones((res * res, 3), device=sdf.device) * 0.12
    if hit.any():
        pts = origins[hit] + dirs[hit] * t[hit, None]
        _, grad = grid.sample_trilinear_with_grad(pts, sdf.reshape(-1, 1))
        n = torch.nn.functional.normalize(grad.reshape(-1, 3), dim=1)
        # Flip normals toward the viewer so both sign conventions shade the same.
        flip = (n * dirs[hit]).sum(1, keepdim=True) > 0
        n = torch.where(flip, -n, n)
        l = torch.nn.functional.normalize(torch.as_tensor(light_dir, dtype=torch.float32, device=sdf.device), dim=0)
        diffuse = (n @ l).clamp(min=0.0)
        shade = 0.25 + 0.75 * diffuse
        rgb[hit] = shade[:, None] * torch.tensor([0.85, 0.75, 0.55], device=sdf.device)

    return rgb.reshape(res, res, 3).cpu().numpy(), depth.reshape(res, res).cpu().numpy()

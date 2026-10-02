# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Generate numerical PR #796 figures. Run outside the repo root with installed fvdb.

python /path/to/generate.py --chair /path/to/chair.nvdb
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

import fvdb
from reference import reference_redistance
from render import look_at_rays, render_sdf

OUT = Path(__file__).resolve().parent
INK, MUTED, ORANGE, TEAL = "#182b45", "#607086", "#d65a32", "#008b82"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#ccd5df",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def save(fig, name):
    fig.savefig(OUT / name, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Saved {name}", flush=True)


def mechanism():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))
    fig.subplots_adjust(left=0.06, right=0.975, bottom=0.35, top=0.72, wspace=0.32)
    fig.text(0.06, 0.94, "Anchor the interface before redistancing", fontsize=23, weight="bold")
    fig.text(
        0.06,
        0.87,
        "A zero crossing lies between samples. Freezing their signs alone does not freeze its position.",
        color=MUTED,
    )
    panels = [
        (-1, 3, INK, "1  Input field", "Crossing at 0.25 voxel"),
        (-0.2, 1, ORANGE, "2  Unequal endpoint changes", "Crossing moves to 0.167 voxel"),
        (-0.25, 0.75, TEAL, "3  Shared anchor scale", "Crossing stays at 0.25 voxel"),
    ]
    for ax, (left, right, color, title, subtitle) in zip(axes, panels):
        crossing = -left / (right - left)
        ax.axhspan(-1.3, 0, color="#f2f5f9")
        ax.axhline(0, color=MUTED, lw=1)
        ax.axvline(0.25, color=MUTED, ls="--", lw=1.2)
        ax.plot([0, 1], [-1, 3], color="#bbc5d0", ls=":", lw=2)
        ax.plot([0, 1], [left, right], "o-", color=color, lw=2.7, ms=8)
        ax.plot(crossing, 0, "D", color=color, ms=9, zorder=5)
        ax.annotate(f"{left:g}", (0, left), xytext=(8, 7), textcoords="offset points", color=color, weight="bold")
        ax.annotate(
            f"+{right:g}",
            (1, right),
            xytext=(-8, 9),
            ha="right",
            textcoords="offset points",
            color=color,
            weight="bold",
        )
        ax.set(
            xlim=(-0.08, 1.08),
            ylim=(-1.35, 3.7),
            xticks=[0, 0.25, 1],
            yticks=[0],
            xlabel="Position along one voxel edge",
        )
        ax.set_title(title + "\n" + subtitle, loc="left", fontsize=12, pad=18, color=color)
    axes[0].set_ylabel("Field value")
    fig.text(0.06, 0.18, r"Interface cells:  $\partial\phi/\partial\tau=(D-\phi)/\Delta x$", fontsize=16, weight="bold")
    fig.text(
        0.06,
        0.105,
        "Compute D from the input; reuse it throughout the solve. Other cells keep the Godunov update.",
        fontsize=11,
    )
    fig.text(
        0.06,
        0.055,
        "Illustrative edge values, not measured output. Here, dividing both endpoints by 4 preserves the zero-crossing ratio.",
        fontsize=10,
        color=MUTED,
    )
    save(fig, "01-subcell-mechanism.png")


def rod(metrics):
    size, length = 10, 24
    grid = fvdb.Grid.from_dense_axis_aligned_bounds(
        [size, size, length], [0, 0, 0], [size, size, length], device="cuda"
    )
    coords = grid.ijk
    x, y = coords[:, 0].double() - (size - 1) / 2, coords[:, 1].double() - (size - 1) / 2
    dx, dy = x.abs() - 1, y.abs() - 1
    field = (torch.sqrt(dx.clamp(min=0) ** 2 + dy.clamp(min=0) ** 2) + torch.maximum(dx, dy).clamp(max=0)).clamp(-3, 3)
    middle = coords[:, 2] == length // 2

    def section(values):
        out = np.zeros((size, size))
        c = coords[middle].cpu().numpy()
        out[c[:, 1], c[:, 0]] = values[middle].cpu().numpy()
        return out

    initial = section(field)
    axis = np.arange(size) - (size - 1) / 2
    iterations = [3, 6, 12, 48]
    fig, axes = plt.subplots(2, 4, figsize=(12, 7.2))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.15, top=0.77, wspace=0.14, hspace=0.32)
    fig.text(0.045, 0.945, "The anchor prevents thin-feature collapse", fontsize=20, weight="bold")
    fig.text(
        0.045,
        0.89,
        "Cross-section of an exact SDF for a 2-voxel-wide rod. Same input and settings in every panel.",
        color=MUTED,
    )
    rod_stats = []
    for col, iters in enumerate(iterations):
        legacy = reference_redistance(grid, field, 3, iters, 3, subcell=False)
        fixed = grid.reinitialize_sdf(field, band=3, order=3, smooth=0, redistance_iters=iters)
        reference = reference_redistance(grid, field, 3, iters, 3)
        parity = float((fixed - reference).abs().max())
        assert parity < 1e-9, parity
        row = {"iterations": iters, "fixed_reference_max_error_vox": parity}
        for r, (values, color) in enumerate([(legacy, ORANGE), (fixed, TEAL)]):
            ax = axes[r, col]
            data = section(values)
            ax.contourf(axis, axis, data, levels=[-4, 0, 4], colors=[color, "white"], alpha=0.17)
            if data.min() < 0:
                ax.contour(axis, axis, data, levels=[0], colors=[color], linewidths=2.5)
            else:
                ax.text(0, 0, "No interior\nremains", ha="center", va="center", color=color, weight="bold")
            ax.contour(axis, axis, initial, levels=[0], colors=[INK], linewidths=1.2, linestyles="--")
            xx, yy = np.meshgrid(axis, axis)
            ax.scatter(xx, yy, s=9, color="#a7b3c1", alpha=0.65)
            count = int((data < 0).sum())
            row["legacy_inside_samples" if r == 0 else "fixed_inside_samples"] = count
            ax.set(xlim=(-2, 2), ylim=(-2, 2), aspect="equal", xticks=[-2, 0, 2], yticks=[-2, 0, 2])
            ax.set_title(f"{iters} iterations", fontsize=12, pad=8)
        rod_stats.append(row)
    fig.text(0.035, 0.61, "Before\nsubcell fix", color=ORANGE, weight="bold", va="center", fontsize=13)
    fig.text(0.035, 0.31, "PR #796", color=TEAL, weight="bold", va="center", fontsize=13)
    fig.legend(
        handles=[
            Line2D([0], [0], color=INK, ls="--", label="Input zero contour"),
            Line2D([0], [0], color=ORANGE, lw=2.5, label="Unanchored result"),
            Line2D([0], [0], color=TEAL, lw=2.5, label="Anchored result"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.55, 0.066),
        ncol=3,
        frameon=False,
    )
    fig.text(
        0.045,
        0.035,
        "Axes in voxels. Band = 3; RK3; no smoothing. Before: float64 legacy reference. After: CUDA, checked against the PR reference.",
        fontsize=9,
        color=MUTED,
    )
    save(fig, "02-thin-rod.png")
    metrics["rod"] = rod_stats


def chair(path, metrics):
    grid, values, _ = fvdb.Grid.from_nanovdb(path, device="cuda")
    field = values.reshape(-1).float()
    vx = float(grid.voxel_size[0])
    counts, fields = [], {}
    for iters in [3, 12, 24, 48]:
        legacy = reference_redistance(grid, field.double(), 3, iters, 3, subcell=False).float()
        fixed = grid.reinitialize_sdf(field, band=3, order=3, smooth=0, redistance_iters=iters)
        expected = reference_redistance(grid, field.double(), 3, iters, 3).float()
        parity = float((fixed - expected).abs().max()) / vx
        assert parity < 1e-4, parity
        counts.append(
            {
                "iterations": iters,
                "legacy_sign_flips": int(((legacy < 0) != (field < 0)).sum()),
                "fixed_sign_flips": int(((fixed < 0) != (field < 0)).sum()),
                "fixed_reference_max_error_vox": parity,
            }
        )
        if iters == 48:
            fields = {"Input SDF": field, "Before subcell fix": legacy, "PR #796": fixed}
    bbox = grid.bbox.float()
    low = grid.voxel_to_world(bbox[0].reshape(1, 3)).reshape(3)
    high = grid.voxel_to_world((bbox[1] + 1).reshape(1, 3)).reshape(3)
    center = ((low + high) / 2).cpu()
    fov, res = 35.0, 800
    distance = float((high - low).norm() / 2) / math.tan(math.radians(fov) / 2) * 1.0
    view = np.array([1.0, 0.7, 1.4])
    eye = (center + torch.tensor(view / np.linalg.norm(view), dtype=torch.float32) * distance).tolist()
    origins, directions = look_at_rays(eye, center.tolist(), [0, 1, 0], res, fov, "cuda")
    light = (view / np.linalg.norm(view) + 0.6 * np.array([0, 1, 0]) + 0.3 * np.cross([0, 1, 0], view)).tolist()
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.2))
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.15, top=0.78, wspace=0.045)
    fig.text(0.035, 0.94, "Thin chair features survive additional redistancing", fontsize=23, weight="bold")
    fig.text(
        0.035, 0.87, "One call, 48 iterations, identical settings. Only the redistance scheme changes.", color=MUTED
    )
    depths = []
    for ax, (name, data), color in zip(axes, fields.items(), [INK, ORANGE, TEAL]):
        rgb, depth = render_sdf(grid, data, origins, directions, res, light)
        depths.append(depth)
        # Use a common crop determined from the original image, keeping framing identical.
        if len(depths) == 1:
            rows, cols = np.where(np.isfinite(depth))
            pad = 28
            crop = (
                slice(max(0, rows.min() - pad), min(res, rows.max() + pad)),
                slice(max(0, cols.min() - pad), min(res, cols.max() + pad)),
            )
        ax.imshow(np.clip(rgb[crop], 0, 1))
        ax.axis("off")
        flips = int(((data < 0) != (field < 0)).sum())
        ax.set_title(f"{name}\n{flips:,} voxel sign flips", color=color, weight="bold", fontsize=13, pad=12)
    for i, name in enumerate(fields):
        counts[-1][name + "_silhouette_changed_pixels"] = int((np.isfinite(depths[0]) != np.isfinite(depths[i])).sum())
    fig.text(
        0.035,
        0.10,
        "Band = 3; RK3; no smoothing. Sign flips count voxel samples that change inside/outside classification; they are not a volume measure.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.035,
        0.05,
        "Before: float64 transcription of the pre-PR solver. After: CUDA at 7ff0fd60, checked against its float64 reference. Same camera and lighting.",
        fontsize=9.5,
        color=MUTED,
    )
    save(fig, "03-chair-comparison.png")
    metrics["chair"] = {"input": path.name, "num_voxels": grid.num_voxels, "voxel_size": vx, "sweep": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chair", type=Path, required=True)
    args = parser.parse_args()
    metrics = {
        "pr_head": "7ff0fd6089790a6de4460b61ffd749bf9eedffc0",
        "legacy_commit": "3c9e2727678faafa88a0e9589de831c290c37133",
        "settings": {"band": 3, "order": 3, "smooth": 0},
        "device": torch.cuda.get_device_name(),
    }
    mechanism()
    rod(metrics)
    chair(args.chair, metrics)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

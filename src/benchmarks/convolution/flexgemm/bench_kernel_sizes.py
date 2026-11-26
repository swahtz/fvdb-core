# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""pred_gather_igemm vs masked_igemm vs default (gather-scatter), fp32 with TF32 on.

Covers PredGatherIGemm's whole admission range: batch 1, uniform k in {3,5,7}, stride in {1,2},
channels divisible by 32, float32. Times the forward alone and forward+backward of one layer
with the plan built once, and the plan build itself. Error is ||out - ref|| / ||ref|| against
the default backend in float64.
"""

import argparse
import json
import sys

import torch

sys.path.insert(0, sys.argv.pop(1))  # flexgemm benchmark dir, for common.py
from common import coords_to_jagged, sphere_coords, time_cuda  # noqa: E402

import fvdb  # noqa: E402

BACKENDS = ("pred_gather_igemm", "masked_igemm", "default")


def rel_err(a, ref):
    return float((a.double() - ref).norm() / ref.norm().clamp_min(1e-300))


def run_case(grid, feats, C, k, s, iters):
    torch.manual_seed(0)
    w = torch.randn(C, C, k, k, k, device="cuda") / (C * k**3) ** 0.5
    ref_plan = fvdb.ConvolutionPlan.from_grid_batch(k, s, grid, channel_pairs=((C, C),))
    ref = ref_plan.execute(feats.double(), w.double())
    out = {"N_in": int(feats.shape[0]), "N_out": int(ref.shape[0])}
    for b in BACKENDS:

        def build():
            return fvdb.ConvolutionPlan.from_grid_batch(
                k, s, grid, expert_config={"backend": b}, channel_pairs=((C, C),)
            )

        try:
            plan = build()
            y = plan.execute(feats, w)
            err = rel_err(y, ref)
            fwd_ms, _ = time_cuda(lambda: plan.execute(feats, w), iters=iters)
            x = feats.clone().requires_grad_(True)
            wg = w.clone().requires_grad_(True)

            def step():
                x.grad = None
                wg.grad = None
                y = plan.execute(x, wg)
                y.backward(torch.ones_like(y))

            step_ms, peak = time_cuda(step, iters=iters)
            build_ms, _ = time_cuda(build, iters=5, warmup=1)
            out[b] = {"fwd_ms": fwd_ms, "step_ms": step_ms, "build_ms": build_ms, "err": err, "peak_gb": peak}
        except Exception as e:  # noqa: BLE001
            out[b] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        torch.cuda.empty_cache()
    return out


def main():
    global BACKENDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--tf32", type=int, default=1)
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--workloads", default=None, help="kind:res:C,...")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    BACKENDS = tuple(args.backends.split(","))
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)

    workloads = [("shell", 128, 128), ("shell", 256, 64), ("shell", 256, 128), ("shell", 512, 32),
                 ("shell", 512, 64), ("shell", 1024, 32), ("solid", 64, 64), ("solid", 128, 32)]
    if args.workloads:
        workloads = [(k, int(r), int(c)) for k, r, c in (w.split(":") for w in args.workloads.split(","))]
    rows = {}
    print(f"{'case':>26} {'N_in':>8} | {'backend':>17} {'fwd ms':>8} {'step ms':>8} {'build':>7} {'err':>8}")
    for kind, res, C in workloads:
        feats, coords, _ = sphere_coords(res, C, solid=(kind == "solid"))
        grid = fvdb.GridBatch.from_ijk(coords_to_jagged(coords, 1))
        feats = torch.randn(grid.total_voxels, C, device="cuda")
        for k in (3, 5, 7):
            for s in (1, 2):
                key = f"{kind},{res},{C},k{k},s{s}"
                r = run_case(grid, feats, C, k, s, args.iters)
                rows[key] = r
                for b in BACKENDS:
                    if b not in r:
                        continue
                    v = r[b]
                    if "error" in v:
                        print(f"{key:>26} {r['N_in']:>8} | {b:>17} ERR {v['error']}")
                    else:
                        print(f"{key:>26} {r['N_in']:>8} | {b:>17} {v['fwd_ms']:8.3f} {v['step_ms']:8.3f} "
                              f"{v['build_ms']:7.2f} {v['err']:8.1e}", flush=True)
        del feats, coords, grid
        torch.cuda.empty_cache()
    meta = {"gpu": torch.cuda.get_device_name(), "tf32": bool(args.tf32), "dtype": "float32", "batch": 1}
    json.dump({"meta": meta, "rows": rows}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()

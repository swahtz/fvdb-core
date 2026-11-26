# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
import sys
import torch
sys.path.insert(0, "/home/jswartz/Development/fvdb-core-conv/src/benchmarks/convolution/flexgemm")
import fvdb  # noqa: E402
from common import coords_to_jagged, sphere_coords  # noqa: E402

def tensor_bytes(obj, seen=None):
    seen = seen if seen is not None else set()
    if obj is None or id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    total = 0
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            v = getattr(obj, name)
        except Exception:
            continue
        if isinstance(v, torch.Tensor) or type(v).__module__.startswith("fvdb"):
            total += tensor_bytes(v, seen)
    return total

kind, res, k, s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
_, coords, _ = sphere_coords(res, 1, solid=(kind == "solid"))
grid = fvdb.GridBatch.from_ijk(coords_to_jagged(coords, 1))
out = [f"{kind},{res},k{k},s{s} N={grid.total_voxels}"]
for b in ("masked_igemm", "default"):
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    plan = fvdb.ConvolutionPlan.from_grid_batch(k, s, grid, expert_config={"backend": b})
    torch.cuda.synchronize()
    topo = plan._backend.topology if b == "masked_igemm" else plan._backend.topology
    held = torch.cuda.memory_allocated() - base
    peak = torch.cuda.max_memory_allocated() - base
    out.append(f"{b}: holds {held/2**30:.2f} GiB, build peak {peak/2**30:.2f} GiB")
    del plan, topo
print(" | ".join(out))

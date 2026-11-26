# Kernel-size benchmarks (masked_igemm, gather-scatter default, pred_gather_igemm)

RTX PRO 6000 Blackwell (SM120), float32 with TF32, batch 1, one layer, plan built once.
`bench_kernel_sizes.py <flexgemm dir> --json out.json [--backends ...] [--workloads kind:res:C,...]`
times forward, forward + backward and plan build for kernel sizes 3, 5, 7 and strides 1, 2 on
sphere shells (RES 128-1024) and solid balls (RES 64, 128). Plans are built with
`from_grid_batch(k, s, grid)` and no target, so stride-1 plans dilate the grid.

- `k3_k5_k7_before_large_kernels.json`: all three backends before masked_igemm supported more
  than 32 taps (masked_igemm rows for 3x3x3 only).
- `k3_k5_k7_dense_maps.json`: all three backends with large kernels on dense neighbor maps
  (7x7x7 at RES 1024 stride 1 ran out of memory).
- `k3_k5_k7_compact_maps_masked_igemm_only.json`: masked_igemm with compact maps for large
  kernels, the code in the PR; compare its rows with the default and pred_gather_igemm rows
  of `k3_k5_k7_dense_maps.json`, which those changes do not affect.

## Plan memory (`plan_memory.py kind res k s`, GiB held after building the plan)

| Plan | masked_igemm, dense maps | masked_igemm, compact maps | gather-scatter |
|---|---|---|---|
| 5x5x5, stride 1, shell RES 512 (1.0M voxels) | 4.35 | 2.19 | 0.96 |
| 7x7x7, stride 1, shell RES 512 | 15.15 | 6.14 | 2.63 |
| 7x7x7, stride 2, shell RES 512 | 3.08 | 0.85 | 0.33 |
| 7x7x7, stride 1, solid RES 128 (1.1M voxels) | 6.05 | 5.84 | 2.81 |
| 5x5x5, stride 1, shell RES 1024 (4.1M voxels) | 17.43 | 8.77 | 3.85 |
| 7x7x7, stride 2, shell RES 1024 | 12.35 | 3.40 | 1.32 |
| 7x7x7, stride 1, shell RES 1024 | out of memory | 24.59 | 10.53 |

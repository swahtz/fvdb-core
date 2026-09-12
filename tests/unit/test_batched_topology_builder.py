# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Equivalence tests for the batched leaf-mask topology builder (issue #755).

On CUDA, ``refined_grid`` / ``coarsened_grid`` (and through them ``conv_grid`` /
``conv_transpose_grid`` for K == S), and ``from_ijk`` (and through it ``from_points``,
``from_mesh``, ``from_nearest_voxels_to_points`` and the non-power-of-two ``conv_grid`` fallback)
build all batch members in a single batched pass instead of one NanoVDB build + merge per member.
These tests pin the batched results against:

- ``from_ijk`` (PointsToGrid) grids built from independently computed expected coordinates,
  compared **elementwise** (``torch.equal`` on ``ijk.jdata``), which pins the canonical NanoVDB
  node ordering (root tiles by offset-shifted key, then x-major upper/lower child offsets), and
- the CPU implementations of the same ops, compared as per-member coordinate sets.

Emphasis is on the cases the batched pipeline must get right and a per-member loop got for free:
batches with empty members (first/middle/last/all), members of wildly unequal sizes, coordinates
straddling root-tile boundaries (+-4096) and negative octants (where the sort-key encoding differs
from the stored ``Tile::key`` encoding), and multi-pass factors (4 = two chained passes).
"""

import unittest

import torch
from parameterized import parameterized

import fvdb
from fvdb import GridBatch, JaggedTensor


def _build(ijks, device):
    jt = JaggedTensor([t.to(device=device, dtype=torch.int32) for t in ijks])
    return GridBatch.from_ijk(jt, voxel_sizes=1.0, origins=0.0)


def _expected_refine_ijk(ijk: torch.Tensor, factor: int) -> torch.Tensor:
    """All fine coordinates of ``ijk`` subdivided by ``factor`` (unique by construction)."""
    if ijk.numel() == 0:
        return ijk.reshape(0, 3)
    offsets = torch.stack(
        torch.meshgrid(torch.arange(factor), torch.arange(factor), torch.arange(factor), indexing="ij"),
        dim=-1,
    ).reshape(-1, 3)
    fine = ijk.to(torch.int64)[:, None, :] * factor + offsets[None, :, :].to(ijk.device)
    return fine.reshape(-1, 3).to(torch.int32)


def _expected_coarsen_ijk(ijk: torch.Tensor, factor: int) -> torch.Tensor:
    """Unique coarse coordinates floor(ijk / factor)."""
    if ijk.numel() == 0:
        return ijk.reshape(0, 3)
    coarse = torch.div(ijk.to(torch.int64), factor, rounding_mode="floor")
    return torch.unique(coarse, dim=0).to(torch.int32)


def _ijk_sets(grid: GridBatch):
    return [set(map(tuple, t.cpu().to(torch.int64).tolist())) for t in grid.ijk.unbind()]


# Coordinate batches exercising the tricky regimes of the batched builder.
def _tricky_batches():
    torch.manual_seed(42)
    return {
        "mixed_sizes": [
            torch.randint(-8, 8, (200, 3), dtype=torch.int32),
            torch.tensor([[0, 0, 0]], dtype=torch.int32),
            torch.randint(20, 60, (500, 3), dtype=torch.int32),
        ],
        "empty_members": [
            torch.empty((0, 3), dtype=torch.int32),
            torch.randint(-10, 10, (100, 3), dtype=torch.int32),
            torch.empty((0, 3), dtype=torch.int32),
            torch.tensor([[5, 5, 5], [5, 5, 6]], dtype=torch.int32),
            torch.empty((0, 3), dtype=torch.int32),
        ],
        "all_empty": [
            torch.empty((0, 3), dtype=torch.int32),
            torch.empty((0, 3), dtype=torch.int32),
        ],
        # Straddles the +-4096 root-tile boundaries and negative octants: multiple root tiles per
        # grid and coordinates where the offset-shifted sort key and stored Tile::key encodings
        # order tiles differently.
        "tile_boundaries": [
            torch.tensor(
                [
                    [-4097, -4097, -4097],
                    [-4096, -4096, -4096],
                    [-2049, 0, 0],
                    [-1, -1, -1],
                    [0, 0, 0],
                    [4095, 4095, 4095],
                    [4096, 4096, 4096],
                    [4096, -4097, 0],
                ],
                dtype=torch.int32,
            ),
            torch.randint(-4200, -3900, (300, 3), dtype=torch.int32),
            torch.randint(3900, 4200, (300, 3), dtype=torch.int32),
        ],
        "single_grid": [
            torch.randint(-16, 16, (400, 3), dtype=torch.int32),
        ],
        "larger_batch": [torch.randint(-32, 32, (50 + 37 * i, 3), dtype=torch.int32) for i in range(16)],
    }


# Additional coordinate batches for the from_ijk (Coords) pass, whose slots are coordinates rather
# than leaves: heavy duplication (many coordinates per output leaf, repeated coordinates) and
# coordinates straddling the +-4096 root-tile boundaries with negative components.
def _from_ijk_batches():
    batches = _tricky_batches()
    torch.manual_seed(7)
    batches.update(
        {
            "heavy_duplicates": [
                torch.randint(-4, 4, (5000, 3), dtype=torch.int32),
                torch.tensor([[3, -5, 7]], dtype=torch.int32).repeat(1000, 1),
                torch.tensor(
                    [[7, 7, 7], [8, 8, 8], [7, 7, 7], [8, 7, 8], [-1, 0, 0], [-1, 0, 0]], dtype=torch.int32
                ).repeat(50, 1),
                torch.randint(-4100, -4090, (3000, 3), dtype=torch.int32),
            ],
            "neg_tile_straddle": [
                torch.tensor(
                    [
                        [-4096, -4097, -4095],
                        [-4097, -4096, -4096],
                        [-1, -4096, 4096],
                        [-8193, 0, -4097],
                        [-8192, -8192, -8192],
                        [-8193, -8193, -8193],
                        [4096, -1, -4097],
                        [-4096, -4097, -4095],
                        [-4097, -4096, -4096],
                        [0, -1, 0],
                        [-1, 0, -1],
                        [-4095, 4095, -4096],
                    ],
                    dtype=torch.int32,
                ),
                torch.randint(-4110, -4080, (400, 3), dtype=torch.int32),
                torch.cat(
                    [
                        torch.randint(-4110, -4080, (200, 3), dtype=torch.int32),
                        torch.randint(4080, 4110, (200, 3), dtype=torch.int32),
                        torch.randint(-20, 20, (200, 3), dtype=torch.int32),
                    ]
                ),
                torch.stack(
                    [
                        torch.randint(-8200, -8180, (300,), dtype=torch.int32),
                        torch.randint(4090, 4100, (300,), dtype=torch.int32),
                        torch.randint(-5, 5, (300,), dtype=torch.int32),
                    ],
                    dim=-1,
                ),
            ],
            "int64_input": [
                torch.randint(-100, 100, (700, 3), dtype=torch.int64),
                torch.randint(-5000, 5000, (300, 3), dtype=torch.int64),
            ],
        }
    )
    return batches


# Canonical NanoVDB voxel enumeration order (root tiles by offset-shifted sort key, then x-major
# upper/lower child offsets, then leaf-local offset) of the hand-written members above, as produced
# by the per-member PointsToGrid path from_ijk used before the batched Coords pass. Pins the
# elementwise ``ijk.jdata`` order of the batched pass to that reference.
_FROM_IJK_CANONICAL_ORDER = {
    ("tile_boundaries", 0): [
        [-4097, -4097, -4097],
        [-4096, -4096, -4096],
        [-1, -1, -1],
        [-2049, 0, 0],
        [0, 0, 0],
        [4095, 4095, 4095],
        [4096, -4097, 0],
        [4096, 4096, 4096],
    ],
    ("neg_tile_straddle", 0): [
        [-8193, -8193, -8193],
        [-8193, 0, -4097],
        [-8192, -8192, -8192],
        [-4097, -4096, -4096],
        [-4096, -4097, -4095],
        [-1, -4096, 4096],
        [-4095, 4095, -4096],
        [-1, 0, -1],
        [0, -1, 0],
        [4096, -1, -4097],
    ],
    ("heavy_duplicates", 2): [[-1, 0, 0], [7, 7, 7], [8, 7, 8], [8, 8, 8]],
}


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required for the batched builder")
class TestBatchedTopologyBuilder(unittest.TestCase):
    def _check_against_expected(self, result: GridBatch, expected_ijks, msg: str):
        """Elementwise-pin `result` against a from_ijk build of the expected coordinates."""
        expected = _build(expected_ijks, result.device)
        self.assertEqual(result.grid_count, expected.grid_count, msg)
        self.assertTrue(torch.equal(result.num_voxels, expected.num_voxels), msg)
        self.assertTrue(
            torch.equal(result.ijk.jdata, expected.ijk.jdata),
            f"{msg}: voxel enumeration (canonical node order) differs from PointsToGrid",
        )
        for b in range(result.grid_count):
            self.assertTrue(
                torch.equal(result.bbox_at(b).cpu(), expected.bbox_at(b).cpu()),
                f"{msg}: bbox of member {b}",
            )

    def _check_against_cpu(self, result: GridBatch, cpu_result: GridBatch, msg: str):
        self.assertEqual(result.grid_count, cpu_result.grid_count, msg)
        self.assertTrue(torch.equal(result.num_voxels.cpu(), cpu_result.num_voxels), msg)
        self.assertEqual(_ijk_sets(result), _ijk_sets(cpu_result), msg)

    @parameterized.expand([(name,) for name in _tricky_batches().keys()])
    def test_refined_grid_matches_expected_and_cpu(self, name):
        coords = _tricky_batches()[name]
        for factor in (2, 4):
            grid = _build(coords, "cuda")
            result = grid.refined_grid(factor)
            expected = [_expected_refine_ijk(c, factor) for c in coords]
            self._check_against_expected(result, expected, f"{name} refine x{factor}")
            cpu_result = _build(coords, "cpu").refined_grid(factor)
            self._check_against_cpu(result, cpu_result, f"{name} refine x{factor} vs CPU")

    @parameterized.expand([(name,) for name in _tricky_batches().keys()])
    def test_coarsened_grid_matches_expected_and_cpu(self, name):
        coords = _tricky_batches()[name]
        for factor in (2, 4):
            grid = _build(coords, "cuda")
            result = grid.coarsened_grid(factor)
            expected = [_expected_coarsen_ijk(c, factor) for c in coords]
            self._check_against_expected(result, expected, f"{name} coarsen x{factor}")
            cpu_result = _build(coords, "cpu").coarsened_grid(factor)
            self._check_against_cpu(result, cpu_result, f"{name} coarsen x{factor} vs CPU")

    def test_conv_grid_k2s2_multi_grid_matches_per_member(self):
        coords = _tricky_batches()["mixed_sizes"]
        full = _build(coords, "cuda")
        conv = full.conv_grid(kernel_size=2, stride=2)
        convt = full.conv_transpose_grid(kernel_size=2, stride=2)
        for b, c in enumerate(coords):
            single = _build([c], "cuda")
            conv_single = single.conv_grid(kernel_size=2, stride=2)
            convt_single = single.conv_transpose_grid(kernel_size=2, stride=2)
            self.assertTrue(
                torch.equal(conv.ijk.unbind()[b], conv_single.ijk.jdata),
                f"conv_grid k2s2 member {b}",
            )
            self.assertTrue(
                torch.equal(convt.ijk.unbind()[b], convt_single.ijk.jdata),
                f"conv_transpose_grid k2s2 member {b}",
            )

    @parameterized.expand([(name,) for name in _tricky_batches().keys()])
    def test_conv_stride1_and_k3s2_match_cpu(self, name):
        # Stride-1 uniform K routes through batched box-dilate passes (odd K: [-1,1]^3 per pass;
        # even K: one-sided {-1,0}^3 / {0,1}^3 passes), and k3s2 transpose through refine + one
        # negative pad pass. Pin against the CPU implementation (coordinate sets) and against a
        # from_ijk build of the CPU coordinates (elementwise: canonical node order).
        coords = _tricky_batches()[name]
        cuda_grid = _build(coords, "cuda")
        cpu_grid = _build(coords, "cpu")
        cases = (
            [("conv_grid", k, 1) for k in (2, 3, 4, 5)]
            + [("conv_transpose_grid", k, 1) for k in (2, 3, 4, 5)]
            + [("conv_transpose_grid", 3, 2)]
        )
        for op, k, s in cases:
            msg = f"{name} {op} k{k}s{s}"
            result = getattr(cuda_grid, op)(kernel_size=k, stride=s)
            reference = getattr(cpu_grid, op)(kernel_size=k, stride=s)
            self._check_against_cpu(result, reference, msg)
            expected = _build(list(reference.ijk.unbind()), "cuda")
            self.assertTrue(
                torch.equal(result.ijk.jdata, expected.ijk.jdata),
                f"{msg}: voxel enumeration (canonical node order) differs from PointsToGrid",
            )

    def test_masked_refine_multi_grid(self):
        # Masked subdivision routes through pruneGrid then the batched refine.
        coords = _tricky_batches()["mixed_sizes"]
        grid = _build(coords, "cuda")
        mask = JaggedTensor([((t.sum(-1) % 2) == 0) for t in grid.ijk.unbind()])
        result = grid.refined_grid(2, mask=mask)
        cpu_grid = _build(coords, "cpu")
        cpu_mask = JaggedTensor([m.cpu() for m in mask.unbind()])
        cpu_result = cpu_grid.refined_grid(2, mask=cpu_mask)
        self._check_against_cpu(result, cpu_result, "masked refine x2 vs CPU")

    @parameterized.expand([(name,) for name in _from_ijk_batches().keys()])
    def test_from_ijk_matches_cpu_per_member_and_canonical_order(self, name):
        coords = _from_ijk_batches()[name]
        result = _build(coords, "cuda")
        cpu_result = _build(coords, "cpu")
        self._check_against_cpu(result, cpu_result, f"{name} from_ijk vs CPU")
        for b, c in enumerate(coords):
            self.assertTrue(torch.equal(result.bbox_at(b).cpu(), cpu_result.bbox_at(b)), f"{name} bbox {b}")
            expected_count = 0 if c.numel() == 0 else torch.unique(c, dim=0).shape[0]
            self.assertEqual(int(result.num_voxels[b]), expected_count, f"{name} unique count {b}")
            # Multi-member build equals the single-member build elementwise (same canonical order).
            single = _build([c], "cuda")
            self.assertTrue(torch.equal(result.ijk.unbind()[b], single.ijk.jdata), f"{name} member {b} vs B=1")
            self.assertTrue(torch.equal(result.bbox_at(b), single.bbox_at(0)), f"{name} member {b} bbox vs B=1")
            pinned = _FROM_IJK_CANONICAL_ORDER.get((name, b))
            if pinned is not None:
                self.assertEqual(result.ijk.unbind()[b].cpu().tolist(), pinned, f"{name} member {b} canonical order")
        # Idempotence: rebuilding from the enumerated voxels reproduces the enumeration.
        rebuilt = GridBatch.from_ijk(result.ijk, voxel_sizes=1.0, origins=0.0)
        self.assertTrue(torch.equal(rebuilt.ijk.jdata, result.ijk.jdata), f"{name} from_ijk idempotence")
        self.assertTrue(torch.equal(rebuilt.num_voxels, result.num_voxels), f"{name} from_ijk idempotence counts")

    def _check_multi_member_equals_per_member(self, multi: GridBatch, singles, msg: str):
        self.assertEqual(multi.grid_count, len(singles), msg)
        for b, single in enumerate(singles):
            self.assertEqual(single.grid_count, 1, msg)
            self.assertTrue(torch.equal(multi.num_voxels[b : b + 1], single.num_voxels), f"{msg} member {b} count")
            self.assertTrue(torch.equal(multi.ijk.unbind()[b], single.ijk.jdata), f"{msg} member {b} ijk")
            self.assertTrue(torch.equal(multi.bbox_at(b), single.bbox_at(0)), f"{msg} member {b} bbox")

    def test_from_points_multi_member_matches_per_member_and_cpu(self):
        torch.manual_seed(3)
        pts = [
            torch.rand(2000, 3) * 4 - 2,
            torch.empty(0, 3),
            torch.rand(50, 3) * 0.1 + 5.0,
            torch.rand(3000, 3) * 40 - 20,
        ]
        multi = GridBatch.from_points(JaggedTensor([p.cuda() for p in pts]), voxel_sizes=0.1, origins=0.0)
        singles = [GridBatch.from_points(JaggedTensor([p.cuda()]), voxel_sizes=0.1, origins=0.0) for p in pts]
        self._check_multi_member_equals_per_member(multi, singles, "from_points")
        cpu = GridBatch.from_points(JaggedTensor(pts), voxel_sizes=0.1, origins=0.0)
        self._check_against_cpu(multi, cpu, "from_points vs CPU")

        nearest = GridBatch.from_nearest_voxels_to_points(
            JaggedTensor([p.cuda() for p in pts]), voxel_sizes=0.1, origins=0.0
        )
        nearest_singles = [
            GridBatch.from_nearest_voxels_to_points(JaggedTensor([p.cuda()]), voxel_sizes=0.1, origins=0.0) for p in pts
        ]
        self._check_multi_member_equals_per_member(nearest, nearest_singles, "from_nearest_voxels_to_points")
        nearest_cpu = GridBatch.from_nearest_voxels_to_points(JaggedTensor(pts), voxel_sizes=0.1, origins=0.0)
        self._check_against_cpu(nearest, nearest_cpu, "from_nearest_voxels_to_points vs CPU")

    def test_from_mesh_multi_member_matches_per_member_and_cpu(self):
        try:
            from fvdb.utils.examples import load_bunny_mesh

            vertices, faces = load_bunny_mesh(device="cpu")
        except Exception as e:  # example data not available offline
            self.skipTest(f"bunny mesh example data unavailable: {e}")
        vertices = vertices.to(torch.float32)
        faces = faces.to(torch.int64)
        verts = [vertices * s + o for s, o in ((1.0, 0.0), (2.5, -3.0), (0.4, 7.0))]
        multi = GridBatch.from_mesh(
            JaggedTensor([v.cuda() for v in verts]),
            JaggedTensor([faces.cuda() for _ in verts]),
            voxel_sizes=0.02,
            origins=0.0,
        )
        singles = [
            GridBatch.from_mesh(JaggedTensor([v.cuda()]), JaggedTensor([faces.cuda()]), voxel_sizes=0.02, origins=0.0)
            for v in verts
        ]
        self._check_multi_member_equals_per_member(multi, singles, "from_mesh")
        cpu = GridBatch.from_mesh(
            JaggedTensor(verts), JaggedTensor([faces for _ in verts]), voxel_sizes=0.02, origins=0.0
        )
        self._check_against_cpu(multi, cpu, "from_mesh vs CPU")

    def test_conv_grid_k3s3_multi_member_matches_per_member_and_cpu(self):
        # Non-power-of-two stride: coordinate-list fallback through from_ijk.
        coords = _tricky_batches()["mixed_sizes"] + _from_ijk_batches()["neg_tile_straddle"][:2]
        full = _build(coords, "cuda")
        conv = full.conv_grid(kernel_size=3, stride=3)
        singles = [_build([c], "cuda").conv_grid(kernel_size=3, stride=3) for c in coords]
        self._check_multi_member_equals_per_member(conv, singles, "conv_grid k3s3")
        cpu = _build(coords, "cpu").conv_grid(kernel_size=3, stride=3)
        self._check_against_cpu(conv, cpu, "conv_grid k3s3 vs CPU")

    def test_refine_coarsen_roundtrip_grid_count(self):
        # Serialization-visible metadata: grid_count/address arithmetic must be consistent for a
        # multi-grid handle produced by the batched builder.
        coords = _tricky_batches()["empty_members"]
        grid = _build(coords, "cuda")
        fine = grid.refined_grid(2)
        back = fine.coarsened_grid(2)
        self.assertEqual(fine.grid_count, grid.grid_count)
        self.assertEqual(back.grid_count, grid.grid_count)
        self.assertEqual(_ijk_sets(back), _ijk_sets(grid))
        self.assertTrue(torch.equal(back.num_voxels, grid.num_voxels))


if __name__ == "__main__":
    unittest.main()

// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_BUILDGRIDFROMIJK_H
#define FVDB_DETAIL_OPS_BUILDGRIDFROMIJK_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace batched {
struct BatchedTopologyResult;
} // namespace batched

namespace ops {

// Internal helper used by other grid-building ops (BuildCoarseGridFromFine, BuildGridFromPoints,
// etc.)
nanovdb::GridHandle<TorchDeviceBuffer> _createNanoGridFromIJK(const JaggedTensor &ijk);

// CUDA only. Builds the from_ijk grid batch and returns the raw batched-pass result (one buffer
// holding every member, plus the per-member layout) instead of a GridHandle, so callers can chain
// further batched passes onto it through `batched::sourceFromResult` (e.g. the {0,1}^3 pad of
// from_nearest_voxels_to_points). Multi-member batches run the batched Coords pass; a
// single-member batch runs NanoVDB's PointsToGrid (same output, lower peak memory) and is wrapped
// with `batched::resultFromGridHandle`. Performs the same argument validation as
// `_createNanoGridFromIJK`; `ijk` must live on a CUDA device (not PrivateUse1). Callers must
// include BatchedTopologyBuilder.cuh to use the result.
batched::BatchedTopologyResult batchedCoordsPassFromIJK(const JaggedTensor &ijk);

c10::intrusive_ptr<GridBatchData>
createNanoGridFromIJK(const JaggedTensor &ijk,
                      const std::vector<nanovdb::Vec3d> &voxelSizes,
                      const std::vector<nanovdb::Vec3d> &origins);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDGRIDFROMIJK_H

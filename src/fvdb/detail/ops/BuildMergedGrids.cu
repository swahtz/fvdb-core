// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/TorchDeviceBuffer.h>
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/BuildMergedGrids.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/nanovdb/BatchedTopologyBuilder.cuh>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace fvdb::detail::ops {

template <torch::DeviceType>
nanovdb::GridHandle<TorchDeviceBuffer> dispatchMergeGrids(const GridBatchData &gridBatch1,
                                                          const GridBatchData &gridBatch2);

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchMergeGrids<torch::kCUDA>(const GridBatchData &gridBatch1, const GridBatchData &gridBatch2) {
    c10::cuda::CUDAGuard deviceGuard(gridBatch1.device());
    TORCH_CHECK_VALUE(gridBatch1.device() == gridBatch2.device(),
                      "All arguments to MergeGrids must be on the same device");
    TORCH_CHECK_VALUE(gridBatch1.batchSize() == gridBatch2.batchSize(),
                      "GridBatches to merge should have the same batch size");

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(gridBatch1.device().index());

    // All batch members are merged together in one batched Union pass: every leaf of either
    // source batch is an emission slot carrying its origin and activity mask, coincident leaves
    // OR-combine in the dedup stage, a single output buffer, one stream synchronization -- no
    // per-member nanovdb::tools::cuda::MergeGrids builds or handle merging (issue #775). Members
    // empty on one side pass the other side through; members empty on both sides become valid
    // empty grids inline. The output headers are seeded from the first batch.
    batched::BatchedTopologyResult result =
        batched::runBatchedTopologyPass(batched::sourceFromGridBatchPair(gridBatch1, gridBatch2),
                                        batched::TopologyPassSpec::unionOf(),
                                        stream);
    return nanovdb::GridHandle<TorchDeviceBuffer>(std::move(result.buffer));
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchMergeGrids<torch::kCPU>(const GridBatchData &gridBatch1, const GridBatchData &gridBatch2) {
    using GridT     = nanovdb::ValueOnIndex;
    using IndexTree = nanovdb::NanoTree<GridT>;
    TORCH_CHECK(gridBatch1.device().is_cpu(), "All arguments to MergeGrids must be on the CPU");
    TORCH_CHECK(gridBatch2.device().is_cpu(), "All arguments to MergeGrids must be on the CPU");
    TORCH_CHECK_VALUE(gridBatch1.batchSize() == gridBatch2.batchSize(),
                      "GridBatches to merge should have the same batch size");

    const nanovdb::GridHandle<TorchDeviceBuffer> &gridHdl1 = gridBatch1.nanoGridHandle();
    const nanovdb::GridHandle<TorchDeviceBuffer> &gridHdl2 = gridBatch2.nanoGridHandle();

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> gridHandles;
    gridHandles.reserve(gridHdl1.gridCount());
    for (uint32_t bidx = 0; bidx < gridHdl1.gridCount(); bidx += 1) {
        const nanovdb::OnIndexGrid *grid1 = gridHdl1.template grid<GridT>(bidx);
        const nanovdb::OnIndexGrid *grid2 = gridHdl2.template grid<GridT>(bidx);
        TORCH_CHECK(grid1, "Failed to get pointer to nanovdb index grid (first argument to merge)");
        TORCH_CHECK(grid2,
                    "Failed to get pointer to nanovdb index grid (second argument to merge)");
        const IndexTree &tree1 = grid1->tree();
        const IndexTree &tree2 = grid2->tree();

        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        const int64_t joffset = gridBatch1.cumVoxelsAt(bidx);
        for (auto it = ActiveVoxelIterator<-1>(tree1); it.isValid(); it++) {
            proxyGridAccessor.setValue(it->first, 1);
        }
        for (auto it = ActiveVoxelIterator<-1>(tree2); it.isValid(); it++) {
            proxyGridAccessor.setValue(it->first, 1);
        }

        proxyGridAccessor.merge();
        auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridT, TorchDeviceBuffer>(
            *proxyGrid, 0u, false, false);
        ret.buffer().to(torch::kCPU);
        gridHandles.push_back(std::move(ret));
    }

    if (gridHandles.size() == 1) {
        return std::move(gridHandles[0]);
    } else {
        return nanovdb::mergeGrids(gridHandles);
    }
}

c10::intrusive_ptr<GridBatchData>
mergeGrids(const GridBatchData &gridBatch1, const GridBatchData &gridBatch2) {
    TORCH_CHECK_VALUE(gridBatch1.batchSize() == gridBatch2.batchSize(),
                      "GridBatches to merge should have same batch size");
    TORCH_CHECK_VALUE(gridBatch1.device() == gridBatch2.device(),
                      "GridBatches to merge should be on same device/host");
    std::vector<nanovdb::Vec3d> voxS, voxO;
    gridBatch1.gridVoxelSizesAndOrigins(voxS, voxO);
    auto hdl = FVDB_DISPATCH_KERNEL_DEVICE(gridBatch1.device(), [&]() {
        return dispatchMergeGrids<DeviceTag>(gridBatch1, gridBatch2);
    });
    return makeGridBatchData(std::move(hdl), voxS, voxO);
}

} // namespace fvdb::detail::ops

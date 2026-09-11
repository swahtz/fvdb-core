// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/TorchDeviceBuffer.h>
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/BuildPrunedGrid.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/nanovdb/BatchedTopologyBuilder.cuh>
#include <fvdb/detail/utils/nanovdb/CreateEmptyGridHandle.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>

namespace fvdb::detail::ops {

template <torch::DeviceType>
nanovdb::GridHandle<TorchDeviceBuffer> dispatchPruneGrid(const GridBatchData &gridBatch,
                                                         const JaggedTensor &mask);

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchPruneGrid<torch::kCUDA>(const GridBatchData &gridBatch, const JaggedTensor &mask) {
    c10::cuda::CUDAGuard deviceGuard(gridBatch.device());

    TORCH_CHECK_VALUE(mask.rdim() == 1, "Mask must be a one-dimensional boolean tensor");
    TORCH_CHECK_VALUE(mask.scalar_type() == torch::kBool, "Mask must be a boolean tensor");
    TORCH_CHECK_VALUE(gridBatch.device() == mask.device(), "Grid and mask must be on same device");
    TORCH_CHECK_VALUE(mask.element_count() == gridBatch.totalVoxels(),
                      "Mask has ",
                      mask.element_count(),
                      " entries but the grid batch has ",
                      gridBatch.totalVoxels(),
                      " voxels");

    if (gridBatch.batchSize() == 0) {
        return createEmptyGridHandle(gridBatch.device());
    }

    // All batch members are pruned in one batched leaf-mask pass: a single output buffer and one
    // stream synchronization, no per-member builds or handle merging. The mask's own joffsets
    // locate each grid's voxels, so sliced (non-contiguous) batch views work unchanged.
    const torch::Tensor keep    = mask.jdata().contiguous();
    const torch::Tensor offsets = mask.joffsets().contiguous();
    TORCH_CHECK(offsets.scalar_type() == torch::kInt64 &&
                    offsets.numel() == gridBatch.batchSize() + 1,
                "Unexpected mask offsets layout");

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(gridBatch.device().index());
    const std::vector<batched::TopologyPassSpec> passes{
        batched::TopologyPassSpec::prune(keep.data_ptr<bool>(), offsets.data_ptr<int64_t>())};
    return batched::batchedTopologyHandle(gridBatch, passes, stream.stream());
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchPruneGrid<torch::kCPU>(const GridBatchData &gridBatch, const JaggedTensor &mask) {
    using GridT     = nanovdb::ValueOnIndex;
    using IndexTree = nanovdb::NanoTree<GridT>;

    TORCH_CHECK_VALUE(mask.rdim() == 1, "Mask must be a one-dimensional boolean tensor");
    TORCH_CHECK_VALUE(mask.scalar_type() == torch::kBool, "Mask must be a boolean tensor");
    TORCH_CHECK_VALUE(gridBatch.device() == mask.device(), "Grid and mask must be on same device");

    const nanovdb::GridHandle<TorchDeviceBuffer> &gridHdl = gridBatch.nanoGridHandle();

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> gridHandles;
    gridHandles.reserve(gridHdl.gridCount());
    for (int64_t bidx = 0; bidx < gridBatch.batchSize(); bidx += 1) {
        const nanovdb::OnIndexGrid *grid = gridBatch.hostGridPtrAt(bidx);
        if (!grid) {
            throw std::runtime_error("Failed to get pointer to nanovdb index grid");
        }
        const IndexTree &tree = grid->tree();

        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        const torch::Tensor maskI = mask.index(bidx).jdata().reshape({-1});
        const int64_t joffset     = gridBatch.cumVoxelsAt(bidx);
        const auto maskIacc       = maskI.accessor<bool, 1>();
        for (auto it = ActiveVoxelIterator<-1>(tree); it.isValid(); it++) {
            const nanovdb::Coord baseIjk = it->first;
            const auto index             = it->second;
            if (maskIacc[index]) {
                proxyGridAccessor.setValue(baseIjk, 1);
            }
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
pruneGrid(const GridBatchData &gridBatch, const JaggedTensor &mask) {
    TORCH_CHECK_VALUE(mask.ldim() == 1, "Mask should be a list of tensors");
    TORCH_CHECK_VALUE(gridBatch.batchSize() == mask.num_tensors(),
                      "Cardinality of masks should match gridbatch size");
    TORCH_CHECK_VALUE(gridBatch.device() == mask.device(),
                      "GridBatch and mask should be on same device/host");
    std::vector<nanovdb::Vec3d> voxS, voxO;
    gridBatch.gridVoxelSizesAndOrigins(voxS, voxO);
    auto hdl = FVDB_DISPATCH_KERNEL_DEVICE(
        gridBatch.device(), [&]() { return dispatchPruneGrid<DeviceTag>(gridBatch, mask); });
    return makeGridBatchData(std::move(hdl), voxS, voxO);
}

} // namespace fvdb::detail::ops

// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/GridBatchData.h>
#include <fvdb/TorchDeviceBuffer.h>
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/ops/BuildPaddedGrid.h>
#include <fvdb/detail/ops/MakeContiguous.h>
#include <fvdb/detail/ops/PopulateGridMetadata.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>
#include <fvdb/detail/utils/cuda/GridDim.h>
#include <fvdb/detail/utils/nanovdb/BatchedTopologyBuilder.cuh>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildPaddedGrid(const GridBatchData &baseBatchHdl, int bmin, int bmax, bool excludeBorder);

__device__ inline void
copyCoords(const fvdb::JIdxType bidx,
           const int64_t base,
           const nanovdb::Coord &ijk0,
           const nanovdb::CoordBBox &bbox,
           TorchRAcc64<int32_t, 2> outIJK,
           TorchRAcc64<fvdb::JIdxType, 1> outIJKBIdx) {
    static_assert(sizeof(nanovdb::Coord) == 3 * sizeof(int32_t));
    nanovdb::Coord ijk;
    int32_t count = 0;
    for (int di = bbox.min()[0]; di <= bbox.max()[0]; di += 1) {
        for (int dj = bbox.min()[1]; dj <= bbox.max()[1]; dj += 1) {
            for (int dk = bbox.min()[2]; dk <= bbox.max()[2]; dk += 1) {
                ijk                      = ijk0 + nanovdb::Coord(di, dj, dk);
                outIJK[base + count][0]  = ijk[0];
                outIJK[base + count][1]  = ijk[1];
                outIJK[base + count][2]  = ijk[2];
                outIJKBIdx[base + count] = bidx;
                count += 1;
            }
        }
    }
}

__device__ inline void
copyCoords(const fvdb::JIdxType bidx,
           const int64_t base,
           const nanovdb::Coord size,
           const nanovdb::Coord &ijk0,
           TorchRAcc64<int32_t, 2> outIJK,
           TorchRAcc64<fvdb::JIdxType, 1> outIJKBIdx) {
    return copyCoords(bidx,
                      base,
                      ijk0,
                      nanovdb::CoordBBox(nanovdb::Coord(0), size - nanovdb::Coord(1)),
                      outIJK,
                      outIJKBIdx);
}

__device__ void
ijkForGridVoxelCallback(int32_t bidx,
                        int32_t lidx,
                        int32_t vidx,
                        int32_t cidx,
                        const GridBatchData::Accessor batchAcc,
                        const nanovdb::CoordBBox bbox,
                        TorchRAcc64<int32_t, 2> outIJKData,
                        TorchRAcc64<fvdb::JIdxType, 1> outIJKBIdx) {
    const int32_t totalPadAmount = static_cast<int32_t>(bbox.volume());

    const nanovdb::OnIndexGrid *gridPtr = batchAcc.grid(bidx);
    const int64_t totalVoxels           = gridPtr->activeVoxelCount();
    const typename nanovdb::OnIndexGrid::LeafNodeType &leaf =
        gridPtr->tree().template getFirstNode<0>()[lidx];
    const int64_t baseOffset = batchAcc.voxelOffset(bidx);

    if (leaf.isActive(vidx)) {
        const int64_t value       = ((int64_t)leaf.getValue(vidx)) - 1;
        const int64_t base        = (baseOffset + value) * totalPadAmount;
        const nanovdb::Coord ijk0 = leaf.offsetToGlobalCoord(vidx);
        copyCoords(bidx, base, ijk0, bbox, outIJKData, outIJKBIdx);
    }
}

template <torch::DeviceType DeviceTag>
JaggedTensor
paddedIJKForGrid(const GridBatchData &batchHdl, const nanovdb::CoordBBox &bbox) {
    TORCH_CHECK(batchHdl.device().is_cuda() || batchHdl.device().is_privateuseone(),
                "GridBatchData must be on CUDA or PrivateUse1 device");
    TORCH_CHECK(batchHdl.device().has_index(), "GridBatchData must have a valid index");

    const int32_t totalPadAmount = static_cast<int32_t>(bbox.volume());

    const torch::TensorOptions optsData =
        torch::TensorOptions().dtype(torch::kInt32).device(batchHdl.device());
    const torch::TensorOptions optsBIdx =
        torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(batchHdl.device());
    torch::Tensor outIJK     = torch::empty({batchHdl.totalVoxels() * totalPadAmount, 3}, optsData);
    torch::Tensor outIJKBIdx = torch::empty({batchHdl.totalVoxels() * totalPadAmount},
                                            optsBIdx); // TODO: Don't populate for single batch

    auto outIJKAcc = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
    auto outIJKBIdxAcc =
        outIJKBIdx.packed_accessor64<fvdb::JIdxType, 1, torch::RestrictPtrTraits>();

    auto cb = [=] __device__(int32_t bidx,
                             int32_t lidx,
                             int32_t vidx,
                             int32_t cidx,
                             GridBatchData::Accessor bacc) {
        ijkForGridVoxelCallback(bidx, lidx, vidx, cidx, bacc, bbox, outIJKAcc, outIJKBIdxAcc);
    };

    if constexpr (DeviceTag == torch::kCUDA) {
        forEachVoxelCUDA(1, batchHdl, cb);
    } else if constexpr (DeviceTag == torch::kPrivateUse1) {
        forEachVoxelPrivateUse1(1, batchHdl, cb);
    }

    return JaggedTensor::from_data_offsets_and_list_ids(
        outIJK, batchHdl.voxelOffsets() * totalPadAmount, batchHdl.jlidx());
}

nanovdb::GridHandle<TorchDeviceBuffer>
buildPaddedGridFromGridWithoutBorderCPU(const GridBatchData &baseBatchHdl, int BMIN, int BMAX) {
    using GridT = nanovdb::ValueOnIndex;

    TORCH_CHECK(BMIN <= BMAX, "BMIN must be less than BMAX");

    const nanovdb::GridHandle<TorchDeviceBuffer> &baseGridHdl = baseBatchHdl.nanoGridHandle();

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(baseGridHdl.gridCount());
    for (int64_t i = 0; i < baseBatchHdl.batchSize(); i += 1) {
        // View-aware byte-offset accessor: the i-th *logical* grid (correct for sliced/
        // non-contiguous batches, unlike grid<GridT>(i) which indexes physically).
        const nanovdb::OnIndexGrid *baseGrid = baseBatchHdl.hostGridPtrAt(i);
        if (!baseGrid) {
            throw std::runtime_error("Failed to get pointer to nanovdb index grid");
        }
        auto baseGridAccessor = baseGrid->getAccessor();

        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        for (auto it = ActiveVoxelIterator(baseGrid->tree()); it.isValid(); it++) {
            nanovdb::Coord ijk0 = it->first;
            bool active         = true;
            for (int di = BMIN; di <= BMAX && active; di += 1) {
                for (int dj = BMIN; dj <= BMAX && active; dj += 1) {
                    for (int dk = BMIN; dk <= BMAX && active; dk += 1) {
                        const nanovdb::Coord ijk = ijk0 + nanovdb::Coord(di, dj, dk);
                        if (ijk != ijk0) {
                            active = active && baseGridAccessor.isActive(
                                                   ijk); // if any surrounding is off, turn it off.
                        }
                    }
                }
            }
            if (active) {
                proxyGridAccessor.setValue(ijk0, 1.0f);
            }
        }

        proxyGridAccessor.merge();
        auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridT, TorchDeviceBuffer>(
            *proxyGrid, 0u, false, false);
        ret.buffer().to(torch::kCPU);
        batchHandles.push_back(std::move(ret));
    }

    if (batchHandles.size() == 1) {
        return std::move(batchHandles[0]);
    } else {
        return nanovdb::mergeGrids(batchHandles);
    }
}

nanovdb::GridHandle<TorchDeviceBuffer>
buildPaddedGridFromGridCPU(const GridBatchData &baseBatchHdl, int BMIN, int BMAX) {
    using GridT = nanovdb::ValueOnIndex;

    TORCH_CHECK(BMIN <= BMAX, "BMIN must be less than BMAX");

    const nanovdb::GridHandle<TorchDeviceBuffer> &baseGridHdl = baseBatchHdl.nanoGridHandle();

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(baseGridHdl.gridCount());
    for (int64_t i = 0; i < baseBatchHdl.batchSize(); i += 1) {
        // View-aware byte-offset accessor: the i-th *logical* grid (correct for sliced/
        // non-contiguous batches, unlike grid<GridT>(i) which indexes physically).
        const nanovdb::OnIndexGrid *baseGrid = baseBatchHdl.hostGridPtrAt(i);
        if (!baseGrid) {
            throw std::runtime_error("Failed to get pointer to nanovdb index grid");
        }

        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        for (auto it = ActiveVoxelIterator(baseGrid->tree()); it.isValid(); it++) {
            nanovdb::Coord ijk0 = it->first;
            for (int di = BMIN; di <= BMAX; di += 1) {
                for (int dj = BMIN; dj <= BMAX; dj += 1) {
                    for (int dk = BMIN; dk <= BMAX; dk += 1) {
                        const nanovdb::Coord ijk = ijk0 + nanovdb::Coord(di, dj, dk);
                        proxyGridAccessor.setValue(ijk, 1.0f);
                    }
                }
            }
        }

        proxyGridAccessor.merge();
        auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridT, TorchDeviceBuffer>(
            *proxyGrid, 0u, false, false);
        ret.buffer().to(torch::kCPU);
        batchHandles.push_back(std::move(ret));
    }

    if (batchHandles.size() == 1) {
        return std::move(batchHandles[0]);
    } else {
        return nanovdb::mergeGrids(batchHandles);
    }
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildPaddedGrid<torch::kCUDA>(const GridBatchData &baseBatchHdl,
                                      int bmin,
                                      int bmax,
                                      bool excludeBorder) {
    c10::cuda::CUDAGuard deviceGuard(baseBatchHdl.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(baseBatchHdl.device().index());

    // Pad by [bmin, bmax]^3 = (bmax positive unit passes) followed by (-bmin negative unit
    // passes). Minkowski sums / erosions by boxes compose, so the order is immaterial.
    // dual_grid (0, 1) is exactly one positive pass.
    const int numPositive = bmax;
    const int numNegative = -bmin;
    const int totalPasses = numPositive + numNegative;

    // Identity case (bmin == bmax == 0): no morphology passes run. For a *contiguous* batch the
    // whole underlying handle is exactly the result, so copy it in one shot. A sliced /
    // non-contiguous batch shares a handle holding MORE grids than batchSize(), so compact just the
    // selected grids into a fresh contiguous handle -- a byte copy with header fix-up, no radix
    // sort and no joffsets().cpu() sync. This is rare and degenerate -- only build_padded_grid(0,
    // 0) reaches it; dual_grid, being (0, 1), never does. The tail then applies the transform
    // fix-up (dual swap or verbatim copy, per `dualTransform`).
    if (totalPasses == 0) {
        if (baseBatchHdl.isContiguous()) {
            // The guide buffer carries the device (with index) into the copy's allocation; the
            // copied buffer inherits it.
            const TorchDeviceBuffer guide(0, baseBatchHdl.device());
            return baseBatchHdl.nanoGridHandle().copy<TorchDeviceBuffer>(guide);
        }
        return ops::contiguousGridHandle(baseBatchHdl);
    }

    // The whole batch is built in `totalPasses` chained batched passes via
    // batched::batchedTopologyHandle (issue #775): one output buffer, one stream synchronization
    // per pass, no per-member builds or handle merging; members that end up empty (including
    // members eroded to nothing) become valid empty grids inline. Sliced / non-contiguous batches
    // are handled by the view-aware source pointers.
    //
    //   plain padding   : bmax BoxDilate({0,1}^3) passes, then -bmin BoxDilate({-1,0}^3) passes
    //                     (Minkowski sum with [bmin, bmax]^3);
    //   exclude_border  : the same octants as Erode passes (a voxel survives iff its whole
    //                     [bmin, bmax]^3 neighborhood is active), matching
    //                     buildPaddedGridFromGridWithoutBorderCPU.
    //
    // dual_grid is exactly one {0,1}^3 pass.
    const nanovdb::Coord zero(0), one(1), minusOne(-1);
    std::vector<batched::TopologyPassSpec> passes;
    passes.reserve(totalPasses);
    for (int p = 0; p < numPositive; ++p) {
        passes.push_back(excludeBorder ? batched::TopologyPassSpec::erode(zero, one)
                                       : batched::TopologyPassSpec::boxDilate(zero, one));
    }
    for (int p = 0; p < numNegative; ++p) {
        passes.push_back(excludeBorder ? batched::TopologyPassSpec::erode(minusOne, zero)
                                       : batched::TopologyPassSpec::boxDilate(minusOne, zero));
    }
    return batched::batchedTopologyHandle(baseBatchHdl, passes, stream.stream());
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildPaddedGrid<torch::kPrivateUse1>(const GridBatchData &baseBatchHdl,
                                             int bmin,
                                             int bmax,
                                             bool excludeBorder) {
    // Multi-GPU / PrivateUse1 keeps the coordinate-list path (TopologyBuilder-based morphology
    // is single-device). The exclude-border variant was never supported here (its old code path
    // called a CUDA-only helper that asserts is_cuda), so reject it explicitly rather than
    // crash obscurely.
    TORCH_CHECK(!excludeBorder,
                "dual_grid/build_padded_grid with exclude_border=True is not supported on "
                "PrivateUse1 (multi-GPU) devices");
    nanovdb::CoordBBox bbox = nanovdb::CoordBBox::createCube(bmin, bmax);
    JaggedTensor coords     = paddedIJKForGrid<torch::kPrivateUse1>(baseBatchHdl, bbox);
    return ops::_createNanoGridFromIJK(coords);
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildPaddedGrid<torch::kCPU>(const GridBatchData &baseBatchHdl,
                                     int bmin,
                                     int bmax,
                                     bool excludeBorder) {
    if (excludeBorder) {
        return buildPaddedGridFromGridWithoutBorderCPU(baseBatchHdl, bmin, bmax);
    } else {
        return buildPaddedGridFromGridCPU(baseBatchHdl, bmin, bmax);
    }
}

c10::intrusive_ptr<GridBatchData>
buildPaddedGrid(
    const GridBatchData &baseBatchHdl, int bmin, int bmax, bool excludeBorder, bool dualTransform) {
    // The structuring element [bmin, bmax]^3 must contain the origin so the padded grid is a
    // superset of the primal (and the erosion a subset); a box excluding the origin would be a
    // translation, which this op does not model. This also lets the CUDA path decompose the box
    // into unit positive/negative octant passes.
    TORCH_CHECK_VALUE(bmin <= 0 && bmax >= 0,
                      "buildPaddedGrid requires bmin <= 0 <= bmax, got bmin=",
                      bmin,
                      ", bmax=",
                      bmax);
    std::vector<nanovdb::Vec3d> voxS, voxO;
    baseBatchHdl.gridVoxelSizesAndOrigins(voxS, voxO);
    auto hdl = FVDB_DISPATCH_KERNEL(baseBatchHdl.device(), [&]() {
        return dispatchBuildPaddedGrid<DeviceTag>(baseBatchHdl, bmin, bmax, excludeBorder);
    });

    const int64_t bs           = hdl.gridCount();
    const torch::Device device = hdl.buffer().device();

    GridBatchData::GridMetadata *hostMeta   = nullptr;
    GridBatchData::GridMetadata *deviceMeta = nullptr;
    if (device.is_cpu() || device.is_cuda()) {
        hostMeta = allocateHostGridMetadata(bs);
        if (device.is_cuda()) {
            deviceMeta = allocateDeviceGridMetadata(device, bs);
        }
    } else if (device.is_privateuseone()) {
        deviceMeta = allocateUnifiedMemoryGridMetadata(bs);
        hostMeta   = deviceMeta;
    }

    torch::Tensor batchOffsets;
    GridBatchData::GridBatchMetadata batchMeta;
    ops::populateGridMetadata(hdl, voxS, voxO, batchOffsets, hostMeta, deviceMeta, &batchMeta);
    batchMeta.mIsContiguous = true;

    // Fix up the per-grid transforms. populateGridMetadata already recomputed primal/dual
    // transforms from the source's (voxelSize, origin), so here we just carry the source's stored
    // transforms over verbatim -- either swapped (dual) or as-is (plain pad).
    for (int64_t i = 0; i < bs; i++) {
        const auto &srcMeta = baseBatchHdl.mHostGridMetadata[i];
        if (dualTransform) {
            // dual_grid: result voxels sit at the *corners* of the source voxels, so the source's
            // dual (corner-aligned) transform becomes the result's primal (center-aligned)
            // transform, and vice versa. This shifts the origin by half a voxel.
            hostMeta[i].mPrimalTransform = srcMeta.mDualTransform;
            hostMeta[i].mDualTransform   = srcMeta.mPrimalTransform;
        } else {
            // Plain padded grid: same lattice as the source, so keep its transforms unchanged.
            hostMeta[i].mPrimalTransform = srcMeta.mPrimalTransform;
            hostMeta[i].mDualTransform   = srcMeta.mDualTransform;
        }
        hostMeta[i].mVoxelSize = srcMeta.mVoxelSize;
    }
    syncMetadataToDevice(hostMeta, deviceMeta, bs, device, true);

    const torch::Tensor listIndices =
        torch::empty({0, 1}, torch::TensorOptions().dtype(fvdb::JLIdxScalarType).device(device));
    std::vector<torch::Tensor> leafBatchIdxs;
    leafBatchIdxs.reserve(bs);
    for (int64_t i = 0; i < bs; i += 1) {
        leafBatchIdxs.push_back(
            torch::full({hostMeta[i].mNumLeaves},
                        static_cast<fvdb::JIdxType>(i),
                        torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(device)));
    }
    torch::Tensor leafBatchIndices = torch::cat(leafBatchIdxs, 0);

    auto gridHdlPtr = std::make_shared<nanovdb::GridHandle<TorchDeviceBuffer>>(std::move(hdl));

    return c10::make_intrusive<GridBatchData>(std::move(gridHdlPtr),
                                              hostMeta,
                                              deviceMeta,
                                              bs,
                                              std::move(batchMeta),
                                              std::move(leafBatchIndices),
                                              std::move(batchOffsets),
                                              std::move(listIndices));
}

} // namespace ops
} // namespace detail
} // namespace fvdb

// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/BuilderResource.h>
#include <fvdb/GridBatchData.h>
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>
#include <fvdb/detail/utils/nanovdb/BatchedTopologyBuilder.cuh>
#include <fvdb/detail/utils/nanovdb/CreateEmptyGridHandle.h>

#if CCCL_DEVICE_MERGE_SUPPORTED
#include <nanovdb/tools/cuda/DistributedPointsToGrid.cuh>
#else
#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#endif

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>

#include <limits>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
nanovdb::GridHandle<TorchDeviceBuffer> dispatchCreateNanoGridFromIJK(const JaggedTensor &ijk);

// NanoVDB's PointsToGrid / DistributedPointsToGrid radix sort casts the per-grid coordinate count
// to int32 (PointsToGrid.cuh:645), which silently corrupts the grid above 2^31 candidates, and the
// batched Coords pass indexes its emission slots (one per coordinate) with int32 as well. Reject
// that here (on every CUDA path: single-member PointsToGrid and multi-member Coords) rather than
// return a garbage grid. Takes an already-on-host joffsets accessor so each
// dispatch reuses the host copy it makes anyway -- no extra device sync.
static void
checkCandidateCountsFitInt32(const torch::TensorAccessor<fvdb::JOffsetsType, 1> &joffsetsAcc) {
    for (int64_t gi = 0; gi + 1 < joffsetsAcc.size(0); gi += 1) {
        const int64_t nCoords = joffsetsAcc[gi + 1] - joffsetsAcc[gi];
        TORCH_CHECK(nCoords <= std::numeric_limits<int32_t>::max(),
                    "Grid ",
                    gi,
                    " would be built from ",
                    nCoords,
                    " candidate coordinates, which exceeds the ",
                    std::numeric_limits<int32_t>::max(),
                    "-coordinate limit of the NanoVDB grid builder. Reduce the grid size.");
    }
}

// Shared argument validation of `_createNanoGridFromIJK` and `batchedCoordsPassFromIJK`.
static void
checkIjkArguments(const JaggedTensor &ijk) {
    TORCH_CHECK_VALUE(
        ijk.ldim() == 1,
        "Expected coords to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        ijk.ldim(),
        "list dimensions");
    TORCH_CHECK_TYPE(at::isIntegralType(ijk.scalar_type(), false), "ijk must have an integer type");
    TORCH_CHECK_VALUE(ijk.rdim() == 2,
                      std::string("Expected ijk to have 2 dimensions (shape (n, 3)) but got ") +
                          std::to_string(ijk.rdim()) + " dimensions");
    TORCH_CHECK_VALUE(ijk.rsize(1) == 3,
                      "Expected 3 dimensional coords but got ijk.rshape[1] = " +
                          std::to_string(ijk.rsize(1)));
    TORCH_CHECK(ijk.num_tensors() == ijk.num_outer_lists(),
                "If this happens, Francis' paranoia was justified. File a bug");
    TORCH_CHECK_VALUE(ijk.num_outer_lists() <= GridBatchData::MAX_GRIDS_PER_BATCH,
                      "Cannot create a batch of grids with more than ",
                      GridBatchData::MAX_GRIDS_PER_BATCH,
                      " grids in it. ",
                      "You passed in ",
                      ijk.num_outer_lists(),
                      " ijk sets.");
    const int64_t numGrids = ijk.joffsets().size(0) - 1;
    TORCH_CHECK(numGrids == ijk.num_outer_lists(),
                "If this happens, Francis' paranoia was justified. File a bug");
}

namespace {

// Validated CUDA from_ijk input: int32 contiguous (N, 3) coordinates, the device jagged offsets,
// and the host-side per-member coordinate counts (from the one host copy of joffsets that the
// int32 candidate-count guard makes anyway).
struct CudaIjkInput {
    torch::Tensor ijkData;     // int32, contiguous, (N, 3)
    torch::Tensor joffsetsDev; // int64, contiguous, (B + 1)
    std::vector<int64_t> coordCounts;
    at::cuda::CUDAStream stream;
};

CudaIjkInput
prepareCudaIjk(const JaggedTensor &ijk) {
    TORCH_CHECK(ijk.is_contiguous(), "ijk must be contiguous");
    TORCH_CHECK(ijk.device().is_cuda(), "device must be cuda");
    TORCH_CHECK(ijk.device().has_index(), "device must have index");
    TORCH_CHECK(ijk.scalar_type() == torch::kInt32 || ijk.scalar_type() == torch::kInt64,
                "ijk must be int32 or int64");

    static_assert(sizeof(nanovdb::Coord) == 3 * sizeof(int32_t), "nanovdb::Coord must be 3 ints");
    static_assert(std::is_same_v<fvdb::JOffsetsType, int64_t>,
                  "the batched Coords pass reads int64 jagged offsets");

    CudaIjkInput in{
        torch::Tensor(), torch::Tensor(), {}, at::cuda::getCurrentCUDAStream(ijk.device().index())};

    torch::Tensor ijkBOffsetTensor = ijk.joffsets().cpu();
    auto ijkBOffset                = ijkBOffsetTensor.accessor<fvdb::JOffsetsType, 1>();
    checkCandidateCountsFitInt32(ijkBOffset);
    const int64_t numGrids = ijkBOffset.size(0) - 1;
    in.coordCounts.resize(numGrids);
    for (int64_t gi = 0; gi < numGrids; gi += 1) {
        in.coordCounts[gi] = ijkBOffset[gi + 1] - ijkBOffset[gi];
    }

    in.ijkData = ijk.jdata();
    if (in.ijkData.scalar_type() != torch::kInt32) {
        in.ijkData = in.ijkData.to(torch::kInt32);
    }
    TORCH_CHECK(in.ijkData.is_contiguous(), "ijk must be contiguous");
    TORCH_CHECK(in.ijkData.dim() == 2, "ijk must have shape (N, 3)");
    TORCH_CHECK(in.ijkData.size(1) == 3, "ijk must have shape (N, 3)");
    in.joffsetsDev = ijk.joffsets().contiguous();
    return in;
}

// Single-member build: NanoVDB's PointsToGrid, as before the batched pass. It is kept for B == 1
// because the Coords pass gains almost nothing in time there (there is no per-member overhead to
// remove) while holding more scratch per coordinate (about 148 MiB vs 88 MiB of torch-visible
// peak for 2M coordinates).
nanovdb::GridHandle<TorchDeviceBuffer>
pointsToGridSingle(const CudaIjkInput &in, const torch::Device &device) {
    using GridT = nanovdb::ValueOnIndex;
    TORCH_CHECK(in.coordCounts.size() == 1, "pointsToGridSingle expects exactly one member");
    const int64_t nVoxels = in.coordCounts[0];
    if (nVoxels == 0) {
        return createEmptyGridHandle(device);
    }
    // The guide buffer carries the device (with index) into TorchDeviceBuffer::create.
    TorchDeviceBuffer guide(0, device);
    auto handle = nanovdb::tools::cuda::
        voxelsToGrid<GridT, nanovdb::Coord *, TorchDeviceBuffer, BuilderResource>(
            reinterpret_cast<nanovdb::Coord *>(in.ijkData.data_ptr<int32_t>()),
            nVoxels,
            1.0,
            guide);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return handle;
}

// Multi-member build: all batch members together in one batched Coords pass -- one emission slot
// per coordinate (leaf origin + single-voxel mask bit), duplicates and coordinates sharing a leaf
// OR-combined, a single output buffer, one stream synchronization; no per-member PointsToGrid
// builds or handle merging (issue #775). Empty members become valid empty grids inline.
batched::BatchedTopologyResult
coordsPassMulti(const CudaIjkInput &in, const torch::Device &device) {
    const batched::TopologyPassSpec pass = batched::TopologyPassSpec::coords(
        in.ijkData.data_ptr<int32_t>(), in.joffsetsDev.data_ptr<fvdb::JOffsetsType>());
    return batched::runBatchedTopologyPass(
        batched::sourceFromCoordCounts(in.coordCounts, device), pass, in.stream.stream());
}

} // namespace

batched::BatchedTopologyResult
batchedCoordsPassFromIJK(const JaggedTensor &ijk) {
    checkIjkArguments(ijk);
    c10::cuda::CUDAGuard deviceGuard(ijk.device());
    const CudaIjkInput in = prepareCudaIjk(ijk);
    if (in.coordCounts.size() == 1) {
        return batched::resultFromGridHandle(pointsToGridSingle(in, ijk.device()),
                                             in.stream.stream());
    }
    return coordsPassMulti(in, ijk.device());
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchCreateNanoGridFromIJK<torch::kCUDA>(const JaggedTensor &ijk) {
    c10::cuda::CUDAGuard deviceGuard(ijk.device());
    const CudaIjkInput in = prepareCudaIjk(ijk);
    if (in.coordCounts.size() == 1) {
        return pointsToGridSingle(in, ijk.device());
    }
    batched::BatchedTopologyResult result = coordsPassMulti(in, ijk.device());
    return nanovdb::GridHandle<TorchDeviceBuffer>(std::move(result.buffer));
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchCreateNanoGridFromIJK<torch::kPrivateUse1>(const JaggedTensor &ijk) {
    using GridT = nanovdb::ValueOnIndex;

#if CCCL_DEVICE_MERGE_SUPPORTED
    TORCH_CHECK(ijk.is_contiguous(), "ijk must be contiguous");
    TORCH_CHECK(ijk.device().is_privateuseone(), "device must be privateuseone");
    TORCH_CHECK(ijk.device().has_index(), "device must have index");
    TORCH_CHECK(ijk.scalar_type() == torch::kInt32, "ijk must be int32");

    static_assert(sizeof(nanovdb::Coord) == 3 * sizeof(int32_t), "nanovdb::Coord must be 3 ints");

    // This guide buffer is a hack to pass in a device with an index to the
    // cudaCreateNanoGrid function. We can't pass in a device directly but we can pass in a
    // buffer which gets passed to TorchDeviceBuffer::create. The guide buffer holds the
    // device and effectively passes it to the created buffer.
    TorchDeviceBuffer guide(0, ijk.device());

    // FIXME: This is slow because we have to copy this data to the host and then build the
    // grids. Ideally we want to do this in a single invocation.
    torch::Tensor ijkBOffsetTensor = ijk.joffsets().cpu();
    auto ijkBOffset                = ijkBOffsetTensor.accessor<fvdb::JOffsetsType, 1>();
    checkCandidateCountsFitInt32(ijkBOffset);
    torch::Tensor ijkData = ijk.jdata();
    TORCH_CHECK(ijkData.is_contiguous(), "ijk must be contiguous");
    TORCH_CHECK(ijkData.dim() == 2, "ijk must have shape (N, 3)");
    TORCH_CHECK(ijkData.size(1) == 3, "ijk must have shape (N, 3)");

    for (const auto device_index: c10::irange(c10::cuda::device_count())) {
        c10::cuda::getCurrentCUDAStream(device_index).synchronize();
    }

    // Create a grid for each batch item and store the handles
    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> handles;
    for (int i = 0; i < (ijkBOffset.size(0) - 1); i += 1) {
        const int64_t startIdx = ijkBOffset[i];
        const int64_t nVoxels  = ijkBOffset[i + 1] - startIdx;

        if (!nVoxels) {
            auto handle = createEmptyGridHandle(ijk.device());
            handles.emplace_back(std::move(handle));
        } else {
            int32_t *dataPtr = ijkData.data_ptr<int32_t>() + ijkData.stride(0) * startIdx;
            auto coordPtr    = reinterpret_cast<nanovdb::Coord *>(dataPtr);

            nanovdb::cuda::DeviceMesh mesh;
            nanovdb::tools::cuda::DistributedPointsToGrid<GridT> converter(mesh);
            auto handle =
                converter.getHandle<nanovdb::Coord *, TorchDeviceBuffer>(coordPtr, nVoxels, guide);
            handles.emplace_back(std::move(handle));
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    for (const auto device_index: c10::irange(c10::cuda::device_count())) {
        c10::cuda::getCurrentCUDAStream(device_index).synchronize();
    }

    if (handles.size() == 1) {
        // If there's only one handle, just return it
        return std::move(handles[0]);
    } else {
        // This copies all the handles into a single handle -- only do it if there are multiple
        // grids
        return nanovdb::cuda::mergeGridHandles(handles, &guide);
    }
#else
    TORCH_CHECK(false, "Distributed creation of grids requires CUDA 12.8 or later");
    return nanovdb::GridHandle<TorchDeviceBuffer>();
#endif
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchCreateNanoGridFromIJK<torch::kCPU>(const JaggedTensor &jaggedCoords) {
    using GridT = nanovdb::ValueOnIndex;

    return AT_DISPATCH_V2(
        jaggedCoords.scalar_type(),
        "buildPaddedGridFromCoords",
        AT_WRAP([&]() {
            using ScalarT = scalar_t;
            jaggedCoords.check_valid();

            static_assert(std::is_integral<ScalarT>::value,
                          "Invalid type for coords, must be integral");

            using ProxyGridT = nanovdb::tools::build::Grid<float>;

            const torch::TensorAccessor<ScalarT, 2> &coordsAcc =
                jaggedCoords.jdata().accessor<ScalarT, 2>();
            const torch::TensorAccessor<fvdb::JOffsetsType, 1> &coordsBOffsetsAcc =
                jaggedCoords.joffsets().accessor<fvdb::JOffsetsType, 1>();
            checkCandidateCountsFitInt32(coordsBOffsetsAcc);

            std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
            batchHandles.reserve(coordsBOffsetsAcc.size(0) - 1);
            for (int bi = 0; bi < (coordsBOffsetsAcc.size(0) - 1); bi += 1) {
                auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
                auto proxyGridAccessor = proxyGrid->getWriteAccessor();

                const int64_t start = coordsBOffsetsAcc[bi];
                const int64_t end   = coordsBOffsetsAcc[bi + 1];

                for (unsigned ci = start; ci < end; ci += 1) {
                    nanovdb::Coord ijk(coordsAcc[ci][0], coordsAcc[ci][1], coordsAcc[ci][2]);
                    proxyGridAccessor.setValue(ijk, 11);
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
        }),
        AT_EXPAND(AT_INTEGRAL_TYPES));
}

nanovdb::GridHandle<TorchDeviceBuffer>
_createNanoGridFromIJK(const JaggedTensor &ijk) {
    checkIjkArguments(ijk);

    // The >2^31-candidate overflow guard runs inside each dispatch (checkCandidateCountsFitInt32),
    // reusing the host joffsets copy those paths already make -- no extra device sync here.
    return FVDB_DISPATCH_KERNEL(ijk.device(),
                                [&]() { return dispatchCreateNanoGridFromIJK<DeviceTag>(ijk); });
}

c10::intrusive_ptr<GridBatchData>
createNanoGridFromIJK(const JaggedTensor &ijk,
                      const std::vector<nanovdb::Vec3d> &voxelSizes,
                      const std::vector<nanovdb::Vec3d> &origins) {
    auto handle = _createNanoGridFromIJK(ijk);
    return makeGridBatchData(std::move(handle), voxelSizes, origins);
}

} // namespace ops
} // namespace detail
} // namespace fvdb

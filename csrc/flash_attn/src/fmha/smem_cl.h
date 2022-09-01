/******************************************************************************
 * Copyright (c) 2022, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_thread_map_tensor_op.h"
#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/layout/layout.h"
#include <cutlass/arch/mma.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include <fmha/gemm.h>

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile_o>
struct FMHAEpilogue {

    // The MMA tile.
    using Mma_tile = fmha::Hmma_tile<Cta_tile_o>;

    // using Shape = cutlass::gemm::GemmShape<16, Mma_tile::MMAS_N * 16, 16>;
    using ThreadblockShapePV = cutlass::gemm::GemmShape<Cta_tile_o::M, Cta_tile_o::N, Cta_tile_o::K>;
    using WarpShapePV = cutlass::gemm::GemmShape<Cta_tile_o::M, Cta_tile_o::N, ThreadblockShapePV::kK / Cta_tile_o::WARPS_K>;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
#else
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    // TD [2022-06-02] We don't support Volta (SM70) yet.
#endif
    using Element = cutlass::half_t;
    using ElementC = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using WarpMma = typename cutlass::gemm::warp::DefaultMmaTensorOp<
        WarpShapePV, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
        LayoutC, cutlass::arch::OpMultiplyAdd, 1, true>::Type;

    static constexpr int kPartitionsK = Cta_tile_o::WARPS_K;

    using AccumulatorFragmentIterator = cutlass::epilogue::warp::FragmentIteratorTensorOp<
                                    typename WarpMma::Shape,
                                    typename WarpMma::Policy::Operator::Shape,
                                    typename WarpMma::Policy::Operator::ElementC,
                                    typename WarpMma::Policy::Operator::FragmentC,
                                    LayoutC>;
    static_assert(AccumulatorFragmentIterator::AccumulatorTile::kElements == Mma_tile::MMAS_M * Mma_tile::MMAS_N * 8);
    using AccumulatorTile = typename AccumulatorFragmentIterator::AccumulatorTile;
    static constexpr int kIterationsStore = AccumulatorFragmentIterator::kIterations;

    // using WarpTileIterator = cutlass::epilogue::warp::TileIteratorTensorOp<
    //     typename WarpMma::Shape, typename WarpMma::Policy::Operator::Shape,
    //     typename WarpMma::Policy::Operator::ElementC, LayoutC>;

    // TODO: looks like elementsPerAccess should vary: 4 for d=64, 2 for d=32?
    using OutputTileThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
        ThreadblockShapePV, typename WarpMma::Shape, kPartitionsK, Element, /*ElementsPerAccess=*/4>::Type;
    using OutputTileThreadMapAccum = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
        ThreadblockShapePV, typename WarpMma::Shape, kPartitionsK, ElementC, /*ElementsPerAccess=*/4>::Type;

    using GmemIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
        OutputTileThreadMap,
        Element
    >;
    // TODO: which ThreadMap should we use?
    using GmemIteratorAccum = cutlass::epilogue::threadblock::PredicatedTileIterator<
        // OutputTileThreadMapAccum,
        OutputTileThreadMap,
        ElementC
    >;


    using DefaultIterators = cutlass::epilogue::threadblock::detail::DefaultIteratorsTensorOp<
        Element, ElementC, /*ElementsPerAccess=*/4, ThreadblockShapePV, typename WarpMma::Shape,
        typename WarpMma::Policy::Operator::Shape, typename OutputTileThreadMap::CompactedThreadMap>;
    using WarpTileIterator = typename DefaultIterators::WarpTileIterator;
    static_assert(WarpTileIterator::kIterations == kIterationsStore);
    using SharedLoadIterator = typename DefaultIterators::SharedLoadIterator;
    using OutputFragment = typename SharedLoadIterator::Fragment;

    // using Padding = cutlass::MatrixShape<0, 0>;
    using Padding = cutlass::MatrixShape<0, 64 / cutlass::sizeof_bits<ElementC>::value * 4>;
    static constexpr int kFragmentsPerIteration = kIterationsStore;  // TODO: could be 1 for Volta?
    /*Using kIterationsStore here so that we get the right storage size*/
    using EpilogueBase = typename cutlass::epilogue::threadblock::EpilogueBase<
        ThreadblockShapePV, typename WarpMma::Shape, kPartitionsK, AccumulatorFragmentIterator, WarpTileIterator,
        Padding, kIterationsStore>;

    using SharedStorage = typename EpilogueBase::SharedStorage;
    static constexpr int kSmemTiles = EpilogueBase::kFragmentsPerIteration;
    static constexpr int kSmemPointerOffset = SharedStorage::StorageShape::kCount / kSmemTiles;
    static constexpr int kSmemPointerOffsetPerWarp = SharedStorage::StorageShape::kCount / (kSmemTiles * kPartitionsK);

    SharedStorage *shared_storage;
    WarpTileIterator warp_tile_iterator;

    inline __device__ FMHAEpilogue(void *smem, const int tidx)
        : shared_storage(reinterpret_cast<SharedStorage *>(smem))
        , warp_tile_iterator(shared_storage->reference(), cutlass::arch::LaneId()) {

        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y) == 0) {
        //     printf("smem_o_cl, shared_storage = 0x%p\n", shared_storage);
        // }

        // const int warp_idx = tidx / 32;
        // Broadcast the warp_id computed by lane 0 to ensure dependent code
        // is compiled as warp-uniform.
        // https://github.com/NVIDIA/cutlass/blob/e66bfcb1f880792caa46b1e983c4114e23afa5f3/include/cutlass/gemm/kernel/gemm_with_fused_epilogue.h#L520
        const int warp_idx = __shfl_sync(0xffffffff, tidx / 32, 0);

        // Compute warp location within threadblock tile by mapping the warp_id to three coordinates:
        //
        //   _m: the warp's position within the threadblock along the M dimension
        //   _n: the warp's position within the threadblock along the N dimension
        //   _k: the warp's position within the threadblock along the K dimension

        // int warp_k = warp_idx / (EpilogueBase::WarpCount::kM * EpilogueBase::WarpCount::kN);
        // int warp_mn = warp_idx % (EpilogueBase::WarpCount::kM * EpilogueBase::WarpCount::kN);
        // int warp_m = warp_mn % EpilogueBase::WarpCount::kM;
        // int warp_n = warp_mn / EpilogueBase::WarpCount::kM;

        // cutlass::MatrixCoord warp_offset{kIterationsStore * warp_k * EpilogueBase::WarpCount::kM + warp_m, warp_n};
        cutlass::MatrixCoord warp_offset{kIterationsStore * warp_idx, 0};

        warp_tile_iterator.add_tile_offset(warp_offset);
    }

    // Store the accumulators.
    static inline __device__ void store_static(const AccumulatorTile &acc,
                                        void *smem, const int tidx) {
        // const int lane_idx = tidx % 32;
        const int lane_idx = cutlass::arch::LaneId();
        // const int warp_idx = tidx / 32;
        // Broadcast the warp_id computed by lane 0 to ensure dependent code
        // is compiled as warp-uniform.
        // https://github.com/NVIDIA/cutlass/blob/e66bfcb1f880792caa46b1e983c4114e23afa5f3/include/cutlass/gemm/kernel/gemm_with_fused_epilogue.h#L520
        const int warp_idx = __shfl_sync(0xffffffff, tidx / 32, 0);

        SharedStorage *shared_storage = reinterpret_cast<SharedStorage *>(smem);
        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
        //     printf("smem_o_cl, shared_storage = 0x%p\n", shared_storage);
        // }
        WarpTileIterator warp_tile_iterator(shared_storage->reference(), lane_idx);

        // Compute warp location within threadblock tile by mapping the warp_id to three coordinates:
        //
        //   _m: the warp's position within the threadblock along the M dimension
        //   _n: the warp's position within the threadblock along the N dimension
        //   _k: the warp's position within the threadblock along the K dimension

        // int warp_k = warp_idx / (EpilogueBase::WarpCount::kM * EpilogueBase::WarpCount::kN);
        // int warp_mn = warp_idx % (EpilogueBase::WarpCount::kM * EpilogueBase::WarpCount::kN);
        // int warp_m = warp_mn % EpilogueBase::WarpCount::kM;
        // int warp_n = warp_mn / EpilogueBase::WarpCount::kM;

        // cutlass::MatrixCoord warp_offset{kIterationsStore * warp_k * EpilogueBase::WarpCount::kM + warp_m, warp_n};
        cutlass::MatrixCoord warp_offset{kIterationsStore * warp_idx, 0};

        warp_tile_iterator.add_tile_offset(warp_offset);

        AccumulatorFragmentIterator accum_fragment_iterator(acc);

        CUTLASS_PRAGMA_UNROLL
        for (int p = 0; p < kIterationsStore; ++p) {
            typename AccumulatorFragmentIterator::Fragment accum_fragment;
            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;

            warp_tile_iterator.store(accum_fragment);
            if (p < kIterationsStore - 1) {
                // warp_tile_iterator.add_pointer_offset(kSmemPointerOffset);
                warp_tile_iterator.add_pointer_offset(kSmemPointerOffsetPerWarp);
            }
        }
    }

    // Store the accumulators.
    inline __device__ void store(const AccumulatorTile &acc) {
        AccumulatorFragmentIterator accum_fragment_iterator(acc);

        // if ((threadIdx.x % 32 == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
        //     printf("tidx = %d, smem_o_cl, kIterationsStore = %d\n", threadIdx.x, kIterationsStore);
        //     printf("tidx = %d, Fragment::kElements = %d\n", threadIdx.x, AccumulatorFragmentIterator::Fragment::kElements);
        //     printf("tidx = %d, kSmemPointerOffsetPerWarp = %d\n", threadIdx.x, kSmemPointerOffsetPerWarp);
        // }
        CUTLASS_PRAGMA_UNROLL
        for (int p = 0; p < kIterationsStore; ++p) {
            typename AccumulatorFragmentIterator::Fragment accum_fragment;
            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;

            warp_tile_iterator.store(accum_fragment);
            if (p < kIterationsStore - 1) {
                // warp_tile_iterator.add_pointer_offset(kSmemPointerOffset);
                warp_tile_iterator.add_pointer_offset(kSmemPointerOffsetPerWarp);
            }
        }
        if (kIterationsStore > 1) {
            warp_tile_iterator.add_pointer_offset((1 - kIterationsStore) * kSmemPointerOffsetPerWarp);
        }
    }

    // Load the accumulators
    template<bool zero_init=true>
    inline __device__ void load(OutputFragment (&out)[kFragmentsPerIteration],
                                const int tidx, int l=0) {
        SharedLoadIterator shared_load_iterator(shared_storage->reference(), tidx);
        CUTLASS_PRAGMA_UNROLL
        for (int p = 0; p < EpilogueBase::kFragmentsPerIteration; ++p) {
            OutputFragment aligned_accum_fragment[kPartitionsK];
            shared_load_iterator.load(aligned_accum_fragment[0]);
            // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0)) {
            //     printf("Smem o loading, iter %d, OutputFragment::kElements = %d\n", 0, (int)OutputFragment::kElements);
            //     auto tmp = aligned_accum_fragment[0];
            //     printf("%f %f %f %f\n", tmp[0], tmp[1], tmp[2], tmp[3]);
            // }
            cutlass::plus<OutputFragment> add_fragments;
            if (kPartitionsK > 1) {
                CUTLASS_PRAGMA_UNROLL
                for ( int i = 1; i < kPartitionsK; ++i) {
                    shared_load_iterator.add_pointer_offset(kSmemPointerOffsetPerWarp * kIterationsStore);
                    shared_load_iterator.load(aligned_accum_fragment[i]);
                    aligned_accum_fragment[0] = add_fragments(aligned_accum_fragment[0], aligned_accum_fragment[i]);
                    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0)) {
                    //     printf("Smem o loading, iter %d\n", i);
                    //     auto tmp1 = aligned_accum_fragment[0];
                    //     printf("%f %f %f %f\n", tmp1[0], tmp1[1], tmp1[2], tmp1[3]);
                    // }
                }
                shared_load_iterator.add_pointer_offset((1 - kPartitionsK) * kSmemPointerOffsetPerWarp * kIterationsStore);
            }
            if (p < EpilogueBase::kFragmentsPerIteration - 1) {
                shared_load_iterator.add_pointer_offset(kSmemPointerOffsetPerWarp);
            }

            out[p] = zero_init ? aligned_accum_fragment[0] : add_fragments(out[p], aligned_accum_fragment[0]);
        }
    }

};

}  // namespace fmha

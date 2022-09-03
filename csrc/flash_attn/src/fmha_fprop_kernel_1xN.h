/***************************************************************************************************
 * Copyright (c) 2022, Tri Dao.
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include "fmha_kernel.h"
#include <fmha/kernel_traits.h>
#include <fmha/gemm.h>
#include <fmha/utils.h>
#include <fmha/smem_cl.h>

#include <fmha/mma_core_sm75.h>

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include "cutlass/numeric_conversion.h"
#include <cutlass/arch/mma.h>
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_thread_map_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, typename WarpMma>
struct Gemm_Q_K_base {
    using Smem_tile_o = typename Kernel_traits::Smem_tile_o;
    using Smem_tile_q = typename Kernel_traits::Smem_tile_q;
    using Smem_tile_k = typename Kernel_traits::Smem_tile_k;
    using Fragment_q = typename Smem_tile_q::Fragment;
    using Fragment_k = typename Smem_tile_k::Fragment;
    using Smem_O_cl = fmha::FMHAEpilogue<typename Kernel_traits::Cta_tile_o>;

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = fmha::Hmma_tile<Cta_tile_p>;

    static constexpr int SMEM_BYTES_SOFTMAX = Cta_tile_p::M * Cta_tile_p::WARPS_N * sizeof(float) * 2;

    __device__ inline Gemm_Q_K_base(char * smem_ptr_q, char * smem_ptr_k, const int tidx) 
        : smem_q(smem_ptr_q, tidx)
        , smem_k(smem_ptr_k, tidx)
        , smem_q_ptr(smem_ptr_q)
        , smem_k_ptr(smem_ptr_k) {

    }

    // __device__ inline void load_q() {
    //     smem_q.load(frag_q[0], 0);
    // }
    __device__ inline void load_q(int byte_offset=0) {
        typename WarpMma::LayoutA layout_A = WarpMma::LayoutA::packed({Cta_tile_p::M, Cta_tile_p::K});
        typename WarpMma::IteratorA iter_A({reinterpret_cast<typename WarpMma::ElementA *>(smem_q_ptr + byte_offset), layout_A}, cutlass::arch::LaneId());
        iter_A.load(frag_q_cl[0]);
    }


    // __device__ inline void reload_q() {
    //     smem_q.load(frag_q[0], 0);
    // }

    __device__ inline void reload_q(int byte_offset=0) {
        typename WarpMma::LayoutA layout_A = WarpMma::LayoutA::packed({Cta_tile_p::M, Cta_tile_p::K});
        typename WarpMma::IteratorA iter_A({reinterpret_cast<typename WarpMma::ElementA *>(smem_q_ptr + byte_offset), layout_A}, cutlass::arch::LaneId());
        iter_A.load(frag_q_cl[0]);
    }

    Fragment_q frag_q[2][Mma_tile_p::MMAS_M];
    Smem_tile_q smem_q;
    Smem_tile_k smem_k;
    typename WarpMma::FragmentA frag_q_cl[2];
    static_assert(WarpMma::FragmentA::kStorageElements == 4 || WarpMma::FragmentA::kStorageElements == 2);
    char *smem_q_ptr;
    char *smem_k_ptr;
};

template<typename Kernel_traits, typename WarpMma, bool K_in_regs, typename elem_type_=__half>
struct Gemm_Q_K : public Gemm_Q_K_base<Kernel_traits, WarpMma> {

    using Base = Gemm_Q_K_base<Kernel_traits, WarpMma>;
    using Smem_tile_o = typename Base::Smem_tile_o;
    using Smem_tile_q = typename Base::Smem_tile_q;
    using Smem_tile_k = typename Base::Smem_tile_k;
    using Fragment_k = typename Base::Fragment_k;
    using Mma_tile_p = typename Base::Mma_tile_p;
    using Cta_tile_p = typename Base::Cta_tile_p;
    using elem_type = elem_type_;
    using Smem_O_cl = typename Base::Smem_O_cl;

    static constexpr int kIterations = WarpMma::Shape::kK / WarpMma::InstructionShape::kK;

    static constexpr bool SHARE_SMEM_FOR_K_AND_V = Kernel_traits::SHARE_SMEM_FOR_K_AND_V;
    // If V is stored in shared memory, we can't load K using the same shared memory.
    static_assert(Kernel_traits::V_IN_REGS);

    static constexpr int SMEM_OFFSET_O = Smem_tile_q::BYTES_PER_TILE;
    // static constexpr int SMEM_OFFSET_SOFTMAX = SMEM_OFFSET_O + Smem_tile_o::BYTES_PER_TILE;
    static constexpr int SMEM_OFFSET_SOFTMAX = SMEM_OFFSET_O + sizeof(typename Smem_O_cl::SharedStorage);
    static constexpr int SMEM_OFFSET_V = Smem_tile_q::BYTES_PER_TILE + (SHARE_SMEM_FOR_K_AND_V ? 0 : Smem_tile_k::BYTES_PER_TILE);

    // Q | K / V
    //   | O | SOFTMAX
    static constexpr int SMEM_BYTES = Smem_tile_q::BYTES_PER_TILE 
                                    + std::max((SHARE_SMEM_FOR_K_AND_V ? 1 : 2) * Smem_tile_k::BYTES_PER_TILE,
                                               // Smem_tile_o::BYTES_PER_TILE + Base::SMEM_BYTES_SOFTMAX);
                                               (int)sizeof(typename Smem_O_cl::SharedStorage) + Base::SMEM_BYTES_SOFTMAX);

    __device__ inline Gemm_Q_K(char * smem_, const int tidx) 
        : Base(smem_, smem_ + Smem_tile_q::BYTES_PER_TILE, tidx) {
    }

    // __device__ inline void load_k(){
    //     #pragma unroll
    //     for( int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki ) {
    //         Base::smem_k.load(frag_k[ki], ki);
    //     }
    // }

    __device__ inline void load_k(){
        // #pragma unroll
        // for( int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki ) {
        //     Base::smem_k.load(frag_k[ki], ki);
        // }
        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
        //     printf("smem_k address = %p\n", smem);
        // }
        typename WarpMma::LayoutB layout_B = WarpMma::LayoutB::packed({Cta_tile_p::K, Cta_tile_p::N});
        typename WarpMma::IteratorB iter_B({reinterpret_cast<typename WarpMma::ElementB *>(Base::smem_k_ptr), layout_B}, cutlass::arch::LaneId());
        int warp_idx = threadIdx.x / 32;
        iter_B.add_tile_offset({0, warp_idx});
        #pragma unroll
        for( int ki = 0; ki < kIterations; ++ki ) {
            iter_B.load(frag_k_cl[ki]);
            ++iter_B;
        }
        // if ((threadIdx.x == 1) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
        //     uint32_t * frag_k_ptr = frag_k_cl[0].raw_data();
        //     printf("frag_k_cl = \n");
        //     for (int i = 0; i < 8; ++i) {
        //         float2 tmp = __half22float2(reinterpret_cast<__half2 &>(frag_k_ptr[i]));
        //         printf("%f, %f, ", tmp.x, tmp.y);
        //     }
        //     printf("\n");
        //     printf("frag_k = \n");
        //     for (int i = 0; i < 2; ++i) {
        //         for (int j = 0; j < 4; ++j) {
        //             float2 tmp = __half22float2(reinterpret_cast<__half2 &>(frag_k[0][i].reg(j)));
        //             printf("%f, %f, ", tmp.x, tmp.y);
        //         }
        //     }
        //     printf("\n");
        // }
    }

    // template<typename Acc, int M, int N>
    // __device__ inline void operator()(Acc (&acc_p)[M][N]){
    //     // Do this part of P^T = (Q * K^T)^T.
    //     #pragma unroll
    //     for( int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki ) {
    //         // Trigger the load from shared memory for the next series of Q values.
    //         Base::smem_q.load(Base::frag_q[ki & 1], ki);
    //         // Do the math for the values already in registers.
    //         fmha::gemm_cl<elem_type>(acc_p, Base::frag_q[(ki - 1) & 1], frag_k[(ki - 1)]);
    //     }
    //     // Do the final stage of math.
    //     {
    //         int ki = Mma_tile_p::MMAS_K;
    //         fmha::gemm_cl<elem_type>(acc_p, Base::frag_q[(ki - 1) & 1], frag_k[(ki - 1)]);
    //     }
    // }

    __device__ inline void operator()(WarpMma warp_mma, typename WarpMma::FragmentC &acc_p, int byte_offset_q=0){
        typename WarpMma::LayoutA layout_A = WarpMma::LayoutA::packed({Base::Cta_tile_p::M, Base::Cta_tile_p::K});
        typename WarpMma::IteratorA iter_A({reinterpret_cast<typename WarpMma::ElementB *>(Base::smem_q_ptr + byte_offset_q), layout_A}, cutlass::arch::LaneId());
        static_assert(WarpMma::FragmentA::kStorageElements == 4 || WarpMma::FragmentA::kStorageElements == 2);
        ++iter_A;
        // Do this part of P^T = (Q * K^T)^T.
        #pragma unroll
        for( int ki = 0; ki < kIterations; ++ki ) {
            // Trigger the load from shared memory for the next series of Q values.
            if (ki + 1 < kIterations) { iter_A.load(Base::frag_q_cl[(ki + 1) % 2]); ++iter_A; }
            // Do the math for the values already in registers.
            warp_mma(acc_p, Base::frag_q_cl[ki % 2], frag_k_cl[ki], acc_p);
        }
    }

    __device__ inline void reload_k(){
        // Noop.
    }

    Fragment_k frag_k[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_N];
    // typename Mma_tile_p::WarpMma::FragmentB frag_k_cl[Mma_tile_p::MMAS_K];
    // static_assert(Mma_tile_p::WarpMma::FragmentB::kStorageElements == 4 * Mma_tile_p::MMAS_N);
    // typename WarpMma::FragmentB frag_k_cl[Mma_tile_p::MMAS_K];
    typename WarpMma::FragmentB frag_k_cl[kIterations];
    static_assert(WarpMma::FragmentB::kStorageElements == 4 * Mma_tile_p::MMAS_N || WarpMma::FragmentB::kStorageElements == 2 * Mma_tile_p::MMAS_N);
};


template<typename Kernel_traits, typename WarpMma, typename elem_type_>
struct Gemm_Q_K<Kernel_traits, WarpMma, false, elem_type_> : public Gemm_Q_K_base<Kernel_traits, WarpMma> {
    using Base = Gemm_Q_K_base<Kernel_traits, WarpMma>;
    using Smem_tile_o = typename Base::Smem_tile_o;
    using Smem_tile_q = typename Base::Smem_tile_q;
    using Smem_tile_k = typename Base::Smem_tile_k;
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;
    using Fragment_k = typename Base::Fragment_k;
    using Cta_tile_p = typename Base::Cta_tile_p;
    using Mma_tile_p = typename Base::Mma_tile_p;
    using elem_type = elem_type_;
    Fragment_k frag_k[2][Mma_tile_p::MMAS_N];
    using Smem_O_cl = typename Base::Smem_O_cl;

    static constexpr bool SHARE_SMEM_FOR_K_AND_V = Kernel_traits::SHARE_SMEM_FOR_K_AND_V;
    static constexpr bool V_IN_REGS = Kernel_traits::V_IN_REGS;
    static_assert(V_IN_REGS || !SHARE_SMEM_FOR_K_AND_V);

    static constexpr int SMEM_OFFSET_V = Smem_tile_q::BYTES_PER_TILE + (SHARE_SMEM_FOR_K_AND_V ? 0 : Smem_tile_k::BYTES_PER_TILE);
    static_assert(Smem_tile_v::BYTES_PER_TILE == (int) Smem_tile_k::BYTES_PER_TILE);
    static constexpr int SMEM_OFFSET_O = SMEM_OFFSET_V + Smem_tile_v::BYTES_PER_TILE;
    // static constexpr int SMEM_OFFSET_SOFTMAX = SMEM_OFFSET_O + Smem_tile_o::BYTES_PER_TILE;
    static constexpr int SMEM_OFFSET_SOFTMAX = SMEM_OFFSET_O + sizeof(typename Smem_O_cl::SharedStorage);

    // If V_IN_REGS and SHARE_SMEM_FOR_K_AND_V:      Q | K/V | O | SOFTMAX
    // If !V_IN_REGS (then !SHARE_SMEM_FOR_K_AND_V): Q | K   | V | O | SOFTMAX
    static constexpr int SMEM_BYTES = Smem_tile_q::BYTES_PER_TILE
                                    + (SHARE_SMEM_FOR_K_AND_V ? 1 : 2) * Smem_tile_k::BYTES_PER_TILE 
                                    // + Smem_tile_o::BYTES_PER_TILE + Base::SMEM_BYTES_SOFTMAX;
                                    + (int)sizeof(typename Smem_O_cl::SharedStorage) + Base::SMEM_BYTES_SOFTMAX;

    __device__ inline Gemm_Q_K(char * smem_, const int tidx) 
      : Base(smem_, smem_ + Smem_tile_q::BYTES_PER_TILE, tidx) {
    }

    __device__ inline void load_k(){
        // Base::smem_k.load(frag_k[0], 0);
        typename WarpMma::LayoutB layout_B = WarpMma::LayoutB::packed({Cta_tile_p::K, Cta_tile_p::N});
        typename WarpMma::IteratorB iter_B({reinterpret_cast<typename WarpMma::ElementB *>(Base::smem_k_ptr), layout_B}, cutlass::arch::LaneId());
        int warp_idx = threadIdx.x / 32;
        iter_B.add_tile_offset({0, warp_idx});
        iter_B.load(frag_k_cl[0]);
    }

    // template<typename Acc, int M, int N>
    // __device__ inline void operator()(Acc (&acc_p)[M][N]){
    //     // Do this part of P^T = (Q * K^T)^T.
    //     #pragma unroll
    //     for( int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki ) {
    //         // Trigger the load from shared memory for the next series of Q values.
    //         Base::smem_q.load(Base::frag_q[ki & 1], ki);
    //         Base::smem_k.load(frag_k[ki & 1], ki);
    //         // Do the math for the values already in registers.
    //         fmha::gemm_cl<elem_type>(acc_p, Base::frag_q[(ki - 1) & 1], frag_k[(ki - 1) & 1]);
    //     }
    //     // Do the final stage of math.
    //     {
    //         int ki = Mma_tile_p::MMAS_K;
    //         fmha::gemm_cl<elem_type>(acc_p, Base::frag_q[(ki - 1) & 1], frag_k[(ki - 1) & 1]);
    //     }
    // }

    __device__ inline void operator()(WarpMma warp_mma, typename WarpMma::FragmentC &acc_p, int byte_offset_q=0){
        typename WarpMma::LayoutA layout_A = WarpMma::LayoutA::packed({Base::Cta_tile_p::M, Base::Cta_tile_p::K});
        typename WarpMma::IteratorA iter_A({reinterpret_cast<typename WarpMma::ElementA *>(Base::smem_q_ptr + byte_offset_q), layout_A}, cutlass::arch::LaneId());
        static_assert(WarpMma::FragmentA::kStorageElements == 4 || WarpMma::FragmentA::kStorageElements == 2);
        ++iter_A;
        typename WarpMma::LayoutB layout_B = WarpMma::LayoutB::packed({Cta_tile_p::K, Cta_tile_p::N});
        typename WarpMma::IteratorB iter_B({reinterpret_cast<typename WarpMma::ElementB *>(Base::smem_k_ptr), layout_B}, cutlass::arch::LaneId());
        int warp_idx = threadIdx.x / 32;
        iter_B.add_tile_offset({0, warp_idx});
        ++iter_B;
        // Do this part of P^T = (Q * K^T)^T.
        constexpr int kIterations = WarpMma::Shape::kK / WarpMma::InstructionShape::kK;
        #pragma unroll
        for( int ki = 0; ki < kIterations; ++ki ) {
            // Trigger the load from shared memory for the next series of Q values.
            if (ki + 1 < kIterations) {
                iter_A.load(Base::frag_q_cl[(ki + 1) % 2]); ++iter_A;
                iter_B.load(frag_k_cl[(ki + 1) % 2]); ++iter_B;
            }
            // Do the math for the values already in registers.
            warp_mma(acc_p, Base::frag_q_cl[ki % 2], frag_k_cl[ki % 2], acc_p);
        }
    }
    __device__ inline void reload_k(){
        // Base::smem_k.load(frag_k[0], 0);
        typename WarpMma::LayoutB layout_B = WarpMma::LayoutB::packed({Cta_tile_p::K, Cta_tile_p::N});
        typename WarpMma::IteratorB iter_B({reinterpret_cast<typename WarpMma::ElementB *>(Base::smem_k_ptr), layout_B}, cutlass::arch::LaneId());
        int warp_idx = threadIdx.x / 32;
        iter_B.add_tile_offset({0, warp_idx});
        iter_B.load(frag_k_cl[0]);
    }

    typename WarpMma::FragmentB frag_k_cl[2];
};

template<typename Kernel_traits>
constexpr size_t get_dynamic_smem_size(){
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    using Mma_tile_p = fmha::Hmma_tile<Cta_tile_p>;

    using InstructionShape = typename Kernel_traits::MmaInstructionShape;
    using Element = cutlass::half_t;
    using ElementAccum = float;

    using ThreadblockShapeQK = cutlass::gemm::GemmShape<Cta_tile_p::M, Cta_tile_p::N, Cta_tile_p::K>;
    using WarpShapeQK = cutlass::gemm::GemmShape<Cta_tile_p::M, 16 * Mma_tile_p::MMAS_N, Cta_tile_p::K>;
    using LayoutP = cutlass::layout::RowMajor;
    // Cutlass's Crosswise only supports at most 64
    constexpr int kCrosswise = std::min(ThreadblockShapeQK::kK, 64);
    using SmemLayoutQ = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
        cutlass::sizeof_bits<Element>::value, kCrosswise>;
    using SmemLayoutK = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
        cutlass::sizeof_bits<Element>::value, kCrosswise>;
    using WarpMmaQK = typename cutlass::gemm::warp::DefaultMmaTensorOp<
        WarpShapeQK, InstructionShape, Element, SmemLayoutQ, Element, SmemLayoutK, ElementAccum,
        LayoutP, cutlass::arch::OpMultiplyAdd, 1, true>::Type;
    return Gemm_Q_K<Kernel_traits, WarpMmaQK, Kernel_traits::K_IN_REGS>::SMEM_BYTES;
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Return_softmax, bool Is_first, bool Is_last, typename Params, typename Prng>
inline __device__ void device_1xN_(const Params &params, const int bidb, const int bidh, int begin, int steps, Prng &ph0, Prng &ph1, const int loop_step_idx) {

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using elem_type = typename Kernel_traits::elem_type;
#else
    constexpr bool is_fp16_type = std::is_same<typename Kernel_traits::elem_type, __half>::value;
    assert(is_fp16_type);
    using elem_type = __half;
#endif

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // The description of the CTA tile for the 2nd batched GEMM.
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = fmha::Hmma_tile<Cta_tile_p>;
    // The MMA tile for the 2nd GEMM.
    using Mma_tile_o = fmha::Hmma_tile<Cta_tile_o>;

    // The global memory tile to load Q.
    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q;

    // The global memory tile to load K.
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_k;

    // The global memory tile to load V.
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle V.
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;

    // The global memory tile to store O.
    using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o;
    using Gmem_tile_o_tmp = fmha::Gmem_tile_o<Cta_tile_o, 4>;
    // The shared memory tile to swizzle O.
    using Smem_tile_o = typename Kernel_traits::Smem_tile_o;

    using Gmem_tile_s = typename Kernel_traits::Gmem_tile_s;

    using Gmem_softmax_sum = typename Kernel_traits::Gmem_softmax_sum;

    using Smem_softmax_sum = typename Kernel_traits::Smem_dp_sum;

    using InstructionShape = typename Kernel_traits::MmaInstructionShape;
    using Element = cutlass::half_t;
    using ElementAccum = float;

    using ThreadblockShapeQK = cutlass::gemm::GemmShape<Cta_tile_p::M, Cta_tile_p::N, Cta_tile_p::K>;
    using WarpShapeQK = cutlass::gemm::GemmShape<Cta_tile_p::M, ThreadblockShapeQK::kN / Cta_tile_p::WARPS_N, Cta_tile_p::K>;
    using LayoutQ = cutlass::layout::RowMajor;
    using LayoutK = cutlass::layout::ColumnMajor;
    using LayoutP = cutlass::layout::RowMajor;
    using MmaCoreQK = typename fmha::FMHAMmaCore<
        ThreadblockShapeQK, WarpShapeQK, InstructionShape, Element, LayoutQ,
        Element, LayoutK, ElementAccum, LayoutP,
        cutlass::arch::OpClassTensorOp>;
    using WarpMmaQK = typename MmaCoreQK::MmaTensorOp;
    using SmemLayoutQ = typename MmaCoreQK::SmemLayoutA;
    using SmemLayoutK = typename MmaCoreQK::SmemLayoutB;
    using SmemIteratorQ = typename MmaCoreQK::SmemIteratorA;
    using SmemIteratorK = typename MmaCoreQK::SmemIteratorB;

    using ThreadblockShapePV = cutlass::gemm::GemmShape<Cta_tile_o::M, Cta_tile_o::N, Cta_tile_o::K>;
    using WarpShapePV = cutlass::gemm::GemmShape<Cta_tile_o::M, Cta_tile_o::N, ThreadblockShapePV::kK / Cta_tile_o::WARPS_K>;
    using LayoutV = cutlass::layout::RowMajor;
    using LayoutO = cutlass::layout::RowMajor;
    using MmaCorePV = typename fmha::FMHAMmaCore<
        ThreadblockShapePV, WarpShapePV, InstructionShape, Element, LayoutP,
        Element, LayoutV, ElementAccum, LayoutO,
        cutlass::arch::OpClassTensorOp>;
    using WarpMmaPV = typename MmaCorePV::MmaTensorOp;
    using WarpIteratorV = typename WarpMmaPV::IteratorB;
    using SmemLayoutV = typename MmaCorePV::SmemLayoutB;
    using SmemIteratorV = typename MmaCorePV::SmemIteratorB;
    constexpr int kIterationsPV = WarpMmaPV::Shape::kK / WarpMmaPV::InstructionShape::kK;

    using Gemm1 = Gemm_Q_K<Kernel_traits, WarpMmaQK, Kernel_traits::K_IN_REGS, elem_type>;

    using Softmax = fmha::Softmax<Cta_tile_p, Kernel_traits>;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    const BlockInfoPadded<Kernel_traits::THREADS> binfo(params, bidb, bidh, tidx);
    // if( binfo.stop_early() ) return;
    if( binfo.stop_early(loop_step_idx * Cta_tile_p::N) ) return;

    Gemm1 gemm_q_k(smem_, tidx);
    // Allocate the global memory tile loader for Q.
    Gmem_tile_q gmem_q(params.q_ptr, params.q_row_stride_in_elts, params.q_head_stride_in_elts, binfo, tidx, true);
    // Allocate the global memory tile loader for O.
    Gmem_tile_o gmem_o(params.o_ptr, params.o_row_stride_in_elts, params.o_head_stride_in_elts, binfo, tidx);
    Gmem_tile_o_tmp gmem_o_tmp(params.o_tmp_ptr, params.o_row_stride_in_elts, params.o_head_stride_in_elts, binfo, tidx);
    // Allocate the global memory tile loader for S.
    Gmem_tile_s gmem_s(params, binfo, tidx);
    Gmem_softmax_sum gmem_softmax_lse(params.softmax_lse_ptr, params, tidx);

    // Wind gmem tiles to the correct position.
    static_assert(Cta_tile_p::N % Cta_tile_p::M == 0);
    const int begin_og = begin;
    begin = Is_causal ? std::max(begin, loop_step_idx * Cta_tile_p::N / Cta_tile_p::M) : begin;
    const int steps_og = steps;
    steps -= begin - begin_og;
    gmem_q.move(begin);
    gmem_o.move(begin);
    gmem_o_tmp.move(begin);
    if (Return_softmax) { gmem_s.move(begin); }
    gmem_softmax_lse.move(begin);
    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("begin = %d, steps = %d\n", begin, steps);
    // }

    fmha::Mask<Cta_tile_p, Is_causal> mask(binfo, tidx, loop_step_idx);

    // Allocate the global memory tile loader for K.
    Gmem_tile_k gmem_k(params.k_ptr, params.k_row_stride_in_elts, params.k_head_stride_in_elts, binfo, tidx, false);
    // Allocate the global memory tile loader for V.
    Gmem_tile_v gmem_v(params.v_ptr, params.v_row_stride_in_elts, params.v_head_stride_in_elts, binfo, tidx, false);
    // The base pointer of smem_v;
    char *smem_v_ = &smem_[Gemm1::SMEM_OFFSET_V];
    
    // Allocate the shared memory tile loader for V. We use the same as K so be careful!!!
    Smem_tile_v smem_v(smem_v_, tidx);

    // Allocate the shared memory tile loader for O. We use the same as K so be careful!!!
    Smem_tile_o smem_o(&smem_[Gemm1::SMEM_OFFSET_O], tidx);

    SmemLayoutQ layout_Q = SmemLayoutQ::packed({ThreadblockShapeQK::kM, ThreadblockShapeQK::kK});
    SmemIteratorQ smem_q_cl({reinterpret_cast<Element *>(smem_), layout_Q}, tidx);
    SmemLayoutK layout_K = SmemLayoutK::packed({ThreadblockShapeQK::kK, ThreadblockShapeQK::kN});
    SmemIteratorK smem_k_cl({reinterpret_cast<Element *>(smem_ + Gemm1::Smem_tile_q::BYTES_PER_TILE), layout_K}, tidx);
    SmemLayoutV layout_V = SmemLayoutV::packed({ThreadblockShapePV::kK, ThreadblockShapePV::kN});
    SmemIteratorV smem_v_cl({reinterpret_cast<Element *>(smem_v_), layout_V}, tidx);
    WarpIteratorV iter_V({reinterpret_cast<Element *>(smem_v_), layout_V}, cutlass::arch::LaneId());

    using Smem_O_cl = fmha::FMHAEpilogue<Cta_tile_o>;
    Smem_O_cl smem_o_cl(&smem_[Gemm1::SMEM_OFFSET_O], tidx);
    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("smem_o address = 0x%p\n", smem_ + Gemm1::SMEM_OFFSET_O);
    // }

    // Copy from mma_piplined_testbed.h
    using GmemIteratorQ = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<ThreadblockShapeQK::kM, ThreadblockShapeQK::kK>,
      Element,
      LayoutQ,
      0,
      typename MmaCoreQK::IteratorThreadMapA
    >;
    LayoutQ gmem_layout_Q(params.q_row_stride_in_elts);
    typename GmemIteratorQ::Params gmem_Q_params(gmem_layout_Q);
    const uint32_t row_offset_q = (binfo.sum_s_q + begin * ThreadblockShapeQK::kM) * params.q_row_stride_in_elts + binfo.bidh * params.q_head_stride_in_elts;
    const int actual_seqlen_q = binfo.actual_seqlen_q - begin * ThreadblockShapeQK::kM;
    const int seqlen_q_remainder = actual_seqlen_q % ThreadblockShapeQK::kM;
    const int extent_q = ((actual_seqlen_q <= ThreadblockShapeQK::kM) || (seqlen_q_remainder == 0)) ? actual_seqlen_q : actual_seqlen_q + ThreadblockShapeQK::kM - seqlen_q_remainder;
    GmemIteratorQ gmem_q_cl(gmem_Q_params,
                            reinterpret_cast<Element *>(params.q_ptr) + row_offset_q,
                            // {extent_q, ThreadblockShapeQK::kK},
                            {extent_q, params.d},
                            tidx);

    using GmemIteratorK = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<ThreadblockShapeQK::kK, ThreadblockShapeQK::kN>,
      Element,
      LayoutK,
      1,
      typename MmaCoreQK::IteratorThreadMapB
    >;
    LayoutK gmem_layout_K(params.k_row_stride_in_elts);
    typename GmemIteratorK::Params gmem_K_params(gmem_layout_K);
    const uint32_t row_offset_k = (binfo.sum_s_k + loop_step_idx * ThreadblockShapeQK::kN) * params.k_row_stride_in_elts + binfo.bidh * params.k_head_stride_in_elts;
    const int extent_k = min(binfo.actual_seqlen_k - loop_step_idx * ThreadblockShapeQK::kN, ThreadblockShapeQK::kN);
    GmemIteratorK gmem_k_cl(gmem_K_params,
                            reinterpret_cast<Element *>(params.k_ptr) + row_offset_k,
                            // {ThreadblockShapeQK::kK, extent_k},
                            {params.d, extent_k},
                            tidx);

    using GmemIteratorV = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<ThreadblockShapePV::kK, ThreadblockShapePV::kN>,
      Element,
      LayoutV,
      0,
      typename MmaCorePV::IteratorThreadMapB
    >;
    LayoutV gmem_layout_V(params.v_row_stride_in_elts);
    typename GmemIteratorV::Params gmem_V_params(gmem_layout_V);
    const uint32_t row_offset_v = (binfo.sum_s_k + loop_step_idx * ThreadblockShapePV::kK) * params.v_row_stride_in_elts + binfo.bidh * params.v_head_stride_in_elts;
    // extent_v is the same as extent_k
    GmemIteratorV gmem_v_cl(gmem_V_params,
                            reinterpret_cast<Element *>(params.v_ptr) + row_offset_v,
                            // {extent_k, ThreadblockShapePV::kN},
                            {extent_k, params.d},
                            tidx);

    // using GmemThreadMapO = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
    //     ThreadblockShapePV,
    //     WarpShapePV,
    //     /*kPartitionsK=*/Cta_tile_o::WARPS_K,
    //     Element,
    //     /*ElementsPerAccess=*/4
    // >::Type;
    // using GmemIteratorO = cutlass::epilogue::threadblock::PredicatedTileIterator<
    //   GmemThreadMapO,
    //   Element
    // >;
    // using GmemThreadMapOAccum = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
    //     ThreadblockShapePV,
    //     WarpShapePV,
    //     /*kPartitionsK=*/Cta_tile_o::WARPS_K,
    //     ElementAccum,
    //     /*ElementsPerAccess=*/4
    // >::Type;
    // using GmemIteratorOAccum = cutlass::epilogue::threadblock::PredicatedTileIterator<
    //   GmemThreadMapOAccum,
    //   ElementAccum
    // >;
    using GmemIteratorO = typename fmha::FMHAEpilogue<Cta_tile_o>::GmemIterator;
    using GmemIteratorOAccum = typename fmha::FMHAEpilogue<Cta_tile_o>::GmemIteratorAccum;

    LayoutO gmem_layout_O(params.o_row_stride_in_elts);
    typename GmemIteratorO::Params gmem_O_params(gmem_layout_O);
    const uint32_t row_offset_o = (binfo.sum_s_q + begin * ThreadblockShapeQK::kM) * params.o_row_stride_in_elts + binfo.bidh * params.o_head_stride_in_elts;
    GmemIteratorO gmem_o_cl(gmem_O_params,
                            reinterpret_cast<Element *>(params.o_ptr) + row_offset_o,
                            // {extent_q, ThreadblockShapePV::kN},
                            {actual_seqlen_q, params.d},
                            tidx);

    typename GmemIteratorOAccum::Params gmem_Oaccum_params(gmem_layout_O);
    GmemIteratorOAccum gmem_o_accum_cl(gmem_Oaccum_params,
                                       reinterpret_cast<ElementAccum *>(params.o_tmp_ptr) + row_offset_o,
                                       {actual_seqlen_q, params.d},
                                       tidx);

    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("ThreadmapO kThreads = %d, kWarpCount = %d\n", OutputTileThreadMap::kThreads, OutputTileThreadMap::kWarpCount);
    //     printf("ThreadmapO kColumn = %d, kRow = %d, kGroup = %d, kCluster = %d\n", OutputTileThreadMap::Iterations::kColumn, OutputTileThreadMap::Iterations::kRow, OutputTileThreadMap::Iterations::kGroup, OutputTileThreadMap::Iterations::kCluster);
    //     printf("ThreadmapO kTile= %d\n", OutputTileThreadMap::Count::kTile);
    //     printf("gmem_o_cl kITerations = %d\n", GmemIteratorO::kIterations);
    //     printf("gmem_o_cl, Fragment::kElements = %d, kThreads = %d, kIterations = %d\n",
    //            GmemIteratorO::Fragment::kElements, GmemIteratorO::kThreads, GmemIteratorO::kIterations);
    //     printf("gmem_o STGS_PER_LOOP = %d\n", Gmem_tile_o::STGS_PER_LOOP);
    // }

    // #if 0
    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y >= 0)) {
    //     printf("GmemK Params k_row_stride_in_elts = %d\n", params.k_row_stride_in_elts);
    //     printf("GmemK row_offsets in elts = %d\n", row_offset);
    // }
    // #endif

    if (!Is_first) {
        gmem_k.move(loop_step_idx);
        gmem_v.move(loop_step_idx);
        if (Return_softmax) { gmem_s.move(loop_step_idx * steps_og); }
    }

    // Trigger the loads for K.
    // gmem_k.load();
    static_assert(GmemIteratorK::Fragment::kElements == decltype(gmem_k)::LDGS * 8);
    typename GmemIteratorK::Fragment gmem_frag_k;
    gmem_frag_k.clear();
    gmem_k_cl.load(gmem_frag_k);

    // #if 0
    // if ((threadIdx.x == 97) && (blockIdx.x == 1) && (blockIdx.y == 3)) {
    //     printf("gmem_frag_k = \n");
    //     uint32_t *ptr = gmem_frag_k.raw_data();
    //     for (int i = 0; i < 32; ++i) {
    //         float2 tmp = __half22float2(reinterpret_cast<__half2 *>(ptr)[i]);
    //         printf("%f %f ", tmp.x, tmp.y);
    //     }
    //     printf("\n");
    //     printf("gmem_k.fetch_ = \n");
    //     for (int i = 0; i < 8; ++i) {
    //         for (int j = 0; j < 4; ++ j) {
    //             float2 tmp = __half22float2(reinterpret_cast<__half2 *>(&gmem_k.fetch_[i])[j]);
    //             printf("%f %f ", tmp.x, tmp.y);
    //         }
    //     }
    //     printf("\n");
    // }
    // #endif

    // Trigger the loads for Q.
    // gmem_q.load();

    static_assert(GmemIteratorQ::Fragment::kElements == decltype(gmem_q)::LDGS * 8);
    typename GmemIteratorQ::Fragment gmem_frag_q;
    gmem_frag_q.clear();
    gmem_q_cl.load(gmem_frag_q);

    // #if 1
    // // if ((threadIdx.x == 37) && (blockIdx.x == 1) && (blockIdx.y == 3)) {
    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("gmem_frag_q = \n");
    //     uint32_t *ptr = gmem_frag_q.raw_data();
    //     for (int i = 0; i < 4; ++i) {
    //         float2 tmp = __half22float2(reinterpret_cast<__half2 *>(ptr)[i]);
    //         printf("%f %f ", tmp.x, tmp.y);
    //     }
    //     printf("\n");
    //     printf("gmem_q.fetch_ = \n");
    //     for (int i = 0; i < 1; ++i) {
    //         for (int j = 0; j < 4; ++ j) {
    //             float2 tmp = __half22float2(reinterpret_cast<__half2 *>(&gmem_q.fetch_[i])[j]);
    //             printf("%f %f ", tmp.x, tmp.y);
    //         }
    //     }
    //     printf("\n");
    // }
    // #endif

    // gmem_q.move();
    // gmem_q.load();

    // ++gmem_q_cl;
    // gmem_frag_q.clear();
    // gmem_q_cl.load(gmem_frag_q);

    // #if 1
    // // if ((threadIdx.x == 37) && (blockIdx.x == 1) && (blockIdx.y == 3)) {
    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("gmem_frag_q = \n");
    //     uint32_t *ptr = gmem_frag_q.raw_data();
    //     for (int i = 0; i < 4; ++i) {
    //         float2 tmp = __half22float2(reinterpret_cast<__half2 *>(ptr)[i]);
    //         printf("%f %f ", tmp.x, tmp.y);
    //     }
    //     printf("\n");
    //     printf("gmem_q.fetch_ = \n");
    //     for (int i = 0; i < 1; ++i) {
    //         for (int j = 0; j < 4; ++ j) {
    //             float2 tmp = __half22float2(reinterpret_cast<__half2 *>(&gmem_q.fetch_[i])[j]);
    //             printf("%f %f ", tmp.x, tmp.y);
    //         }
    //     }
    //     printf("\n");
    // }
    // #endif

    // Trigger the loads for V.
    // gmem_v.load();

    static_assert(GmemIteratorV::Fragment::kElements == decltype(gmem_v)::LDGS * 8);
    typename GmemIteratorV::Fragment gmem_frag_v;
    gmem_frag_v.clear();
    gmem_v_cl.load(gmem_frag_v);

    // #if 1
    // if ((threadIdx.x == 97) && (blockIdx.x == 1) && (blockIdx.y == 3)) {
    // // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("gmem_frag_v = \n");
    //     uint32_t *ptr = gmem_frag_v.raw_data();
    //     for (int i = 0; i < 32; ++i) {
    //         float2 tmp = __half22float2(reinterpret_cast<__half2 *>(ptr)[i]);
    //         printf("%f %f ", tmp.x, tmp.y);
    //     }
    //     printf("\n");
    //     printf("gmem_v.fetch_ = \n");
    //     for (int i = 0; i < 8; ++i) {
    //         for (int j = 0; j < 4; ++ j) {
    //             float2 tmp = __half22float2(reinterpret_cast<__half2 *>(&gmem_v.fetch_[i])[j]);
    //             printf("%f %f ", tmp.x, tmp.y);
    //         }
    //     }
    //     printf("\n");
    // }
    // #endif

    if (!Is_first) { __syncthreads(); }

    float p_prev_lse[Mma_tile_p::MMAS_M * 2];
    if (!Is_first) {
        gmem_softmax_lse.load(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_prev_lse));
    }

    // Commit the data for Q and V to shared memory.
    // gmem_q.commit(gemm_q_k.smem_q);
    // smem_q_cl.store(reinterpret_cast<typename SmemIteratorQ::Fragment(&)>(gmem_q.fetch_));
    smem_q_cl.store(gmem_frag_q);
    // gmem_v.commit(smem_v);
    // smem_v_cl.store(reinterpret_cast<typename SmemIteratorV::Fragment(&)>(gmem_v.fetch_));
    smem_v_cl.store(gmem_frag_v);

    // const uint32_t scale_bmm1 = reinterpret_cast<const uint32_t&>(params.scale_bmm1);
    // #pragma unroll
    // for(int it=0;it < Gmem_tile_k::LDGS;it++){
    //     gmem_k.fetch_[it] = fmha::hmul8(scale_bmm1, gmem_k.fetch_[it]);
    // }

    // Commit the data for K to shared memory.
    if( !Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
        // gmem_k.commit(gemm_q_k.smem_k);
        // smem_k_cl.store(reinterpret_cast<typename SmemIteratorK::Fragment(&)>(gmem_k.fetch_));
        smem_k_cl.store(gmem_frag_k);
    }

    __syncthreads();

    // Load the fragments for Q.
    // gemm_q_k.load_q();
    gemm_q_k.load_q(0 * Gemm1::Smem_tile_q::BYTES_PER_BUFFER);

    // // Load the fragments for V. We keep the data in registers during the entire kernel.
    // typename Smem_tile_v::Fragment frag_v[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_N];
    // #pragma unroll
    // for( int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki ) {
    //     smem_v.load(frag_v[ki], ki);
    // }

    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("smem_v_ address: %p\n", smem_v_);
    // }
    int warp_idx = threadIdx.x / 32;
    // copied from mma_pipelined.h
    // iter_V.add_tile_offset({Mma_tile_o::MMAS_K * warp_idx, 0});
    // typename WarpIteratorV::Fragment frag_v_cl[Mma_tile_o::MMAS_K];
    iter_V.add_tile_offset({kIterationsPV * warp_idx, 0});
    typename WarpIteratorV::Fragment frag_v_cl[kIterationsPV];
    static_assert(WarpIteratorV::Fragment::kStorageElements == 4 * Mma_tile_o::MMAS_N || WarpIteratorV::Fragment::kStorageElements == 2 * Mma_tile_o::MMAS_N );
    #pragma unroll
    for( int ki = 0; ki < kIterationsPV; ++ki ) {
        iter_V.load(frag_v_cl[ki]);
        ++iter_V;
    }

    // for (int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki) {
    //     if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //         uint32_t * frag_v_ptr = frag_v_cl[ki].raw_data();
    //         printf("frag_v_cl = \n");
    //         for (int i = 0; i < 4 * Mma_tile_o::MMAS_N; ++i) {
    //             float2 tmp = __half22float2(reinterpret_cast<__half2 &>(frag_v_ptr[i]));
    //             printf("%f, %f, ", tmp.x, tmp.y);
    //         }
    //         printf("\n");
    //         printf("frag_v = \n");
    //         for (int i = 0; i < Mma_tile_o::MMAS_N; ++i) {
    //             for (int j = 0; j < 4; ++j) {
    //                 float2 tmp = __half22float2(reinterpret_cast<__half2 &>(frag_v[ki][i].reg(j)));
    //                 printf("%f, %f, ", tmp.x, tmp.y);
    //             }
    //         }
    //         printf("\n");
    //     }
    // }

    // Commit the data for K to shared memory if it has not been done already.
    if( Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
        // Make sure we are done loading the fragments for K.
        __syncthreads();

        // Commit the data to shared memory for V.
        // gmem_k.commit(gemm_q_k.smem_k);
        // smem_k_cl.store(reinterpret_cast<typename SmemIteratorK::Fragment(&)>(gmem_k.fetch_));
        smem_k_cl.store(gmem_frag_k);

        // Make sure the data is in shared memory.
        __syncthreads();
    }

    // Load the fragments for K. 
    // gemm_q_k.load_k();
    gemm_q_k.load_k();

    // Create the object to do the softmax.
    Softmax softmax(params, &smem_[Gemm1::SMEM_OFFSET_SOFTMAX], tidx);

    Smem_softmax_sum smem_softmax_lse(reinterpret_cast<float *>(&smem_[Gemm1::SMEM_BYTES]), tidx);

    // Load over the entire sequence length.
    for( int l = 0; l < steps; l++ ) {
        if((begin + l) * Cta_tile_p::M >= binfo.actual_seqlen_q) break;

        // Declare the accumulators for the 1st gemm.
        // fmha::Fragment_accumulator acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
        // fmha::Clear_accumulator<typename fmha::Accumulator_type, Cta_tile_p::WARPS_K>::apply(acc_p);
        // using WarpMmaQK = typename Mma_tile_p::WarpMma;
        WarpMmaQK mma_qk;
        typename WarpMmaQK::FragmentC acc_p;
        acc_p.clear();

        // Do this part of P = Q * K^T.
        // gemm_q_k(acc_p);
        // gemm_q_k(mma_qk, acc_p, (l % 2) * Gemm1::Smem_tile_q::BYTES_PER_BUFFER);
        gemm_q_k(mma_qk, acc_p);

        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
        //     printf("acc_p=%.6f, %.6f\n", acc_p[0][0].elt(0), acc_p[0][0].elt(1));
        // }

        uint4 out[Gmem_tile_o::STGS_PER_LOOP];
        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0)) {
        //     printf("out STGS_PER_LOOP = %d\n", Gmem_tile_o::STGS_PER_LOOP);
        // }
        typename Smem_O_cl::OutputFragment out_cl[Smem_O_cl::kIterationsStore];
        static_assert(GmemIteratorOAccum::kIterations == Smem_O_cl::kIterationsStore);
        static_assert(GmemIteratorO::kIterations == Smem_O_cl::kIterationsStore);
        // if (!Is_first) { gmem_o_tmp.load(out, 0); }
        if (!Is_first) {
            #pragma unroll
            for (int iter = 0; iter < GmemIteratorOAccum::kIterations; ++iter) {
                // out_cl = reinterpret_cast<cutlass::Array<ElementAccum, GmemIteratorOAccum::Fragment::kElements>(&)>(out[iter]);
                // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
                //     printf("out before: %d, %d, %d, %d\n", out[iter].x, out[iter].y, out[iter].z, out[iter].w);
                // }
                // gmem_o_accum_cl.load(reinterpret_cast<cutlass::Array<ElementAccum, GmemIteratorOAccum::Fragment::kElements>(&)>(out[iter]));
                gmem_o_accum_cl.load(out_cl[iter]);
                gmem_o_accum_cl.move();
                // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
                //     printf("byte offset = %d\n", iter * params.o_row_stride_in_elts * GmemIteratorOAccum::ThreadMap::Shape::kRow * sizeof(ElementAccum));
                //     printf("out after: %d, %d, %d, %d\n", out[iter].x, out[iter].y, out[iter].z, out[iter].w);
                // }
            }
        }

        // Trigger the load for the next Q values.
        if( l < steps - 1) {
            gemm_q_k.smem_q.move_to_next_write_buffer();
            // gmem_q.move();
            // gmem_q.load();
            ++gmem_q_cl;
            // if ((l + 1 == steps - 1) && (binfo.actual_seqlen_q % ThreadblockShapeQK::kM != 0)) {
            if ((l + 1 == steps - 1) && (actual_seqlen_q % ThreadblockShapeQK::kM != 0)) {
                // TODO: this probably only works for head_dim = 64 and head_dim = 128, which is
                // what we have right now. Maybe for head_dim = 32 or 96, this could be different.
                const int row_idx = tidx / (GmemIteratorQ::Shape::kColumn / GmemIteratorQ::Fragment::kElements);
                // if (row_idx >= binfo.actual_seqlen_q - (l + 1) * ThreadblockShapeQK::kM) {
                if (row_idx >= actual_seqlen_q - (l + 1) * ThreadblockShapeQK::kM) {
                    gmem_q_cl.clear_mask();
                    // typename GmemIteratorQ::Mask mask;
                    // gmem_q_cl.get_mask(mask);
                    // printf("cleared mask, tidx = %d, row_idx = %d, mask = %x\n", tidx, row_idx, mask[0]);
                }
            //     const uint32_t row_offset_q_last = (binfo.sum_s_q + (l + 1) * ThreadblockShapeQK::kM) * params.q_row_stride_in_elts + binfo.bidh * params.q_head_stride_in_elts;
            //     GmemIteratorQ gmem_q_last(gmem_Q_params,
            //                               reinterpret_cast<Element *>(params.q_ptr) + row_offset_q_last,
            //                               {binfo.actual_seqlen_q - l * ThreadblockShapeQK::kM, params.d},
            //                               tidx);
            //     gmem_q_last.load(gmem_frag_q);
            //     if ((blockIdx.x == 0) && (blockIdx.y == 0)) {
            //         typename GmemIteratorQ::Mask mask;
            //         typename GmemIteratorQ::Mask mask_last;
            //         gmem_q_cl.get_mask(mask);
            //         gmem_q_last.get_mask(mask_last);
            //         printf("tidx = %d, row_idx = %d, mask = %x, mask_last = %x\n", tidx, row_idx, mask[0], mask_last[0]);
            //     }
            // } else {
            //     gmem_q_cl.load(gmem_frag_q);
            }
            gmem_q_cl.load(gmem_frag_q);
        }

        // Load the mask for that iteration.
        mask.load(begin + l);

        // Convert from the accumulator type to FP32 for Softmax.
        softmax.unpack_noscale(acc_p);

        // Apply the mask.
        softmax.apply_mask(mask);

        if( Kernel_traits::SHARE_SMEM_FOR_K_AND_V && l == 0 ) {
            // if we share K and V, it could be that V was not fully read yet but we write into smem for reduction
            __syncthreads();
        }
        // if (!Is_first) {
        //     if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
        //         printf("p_prev_lse=%.6f, %.6f\n", p_prev_lse[0], p_prev_lse[1]);
        //     }
        // }
        // Compute the max.
        float p_max[Mma_tile_p::MMAS_M * 2];
        if (!Is_first) {
            smem_softmax_lse.store_pair(p_prev_lse, l % 2);
            // for (int mi = 0; mi < Mma_tile_p::MMAS_M * 2; mi++) { p_max[mi] = p_prev_lse[mi]; }
            for (int mi = 0; mi < Mma_tile_p::MMAS_M * 2; mi++) { p_max[mi] = p_prev_lse[mi] / params.scale_bmm1f; }
        }

        // Trigger the load for the next LSE values.
        if( l < steps - 1) {
            if (!Is_first) {
                gmem_softmax_lse.load_next(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_prev_lse));
            }
        }

        softmax.template reduce_max</*zero_init=*/Is_first>(p_max);

        // if ((threadIdx.x == 0) && (l == 38)) {
        //     printf("loop_step_idx %d, p_max = %.6f, %.6f., p_prev_lse = %.6f, %.6f\n", loop_step_idx, p_max[0], p_max[1], Is_first ? -10000.f : p_prev_lse[0], Is_first ? -10000.f : p_prev_lse[1]);
        // }

        // if (!Is_first) {
        //     if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
        //         printf("after reduce_max=%.6f, %.6f\n", softmax.elt_[0][0], softmax.elt_[0][1]);
        //     }
        // }

        // Compute the exponential value.
        // softmax.apply_exp(p_max);
        softmax.scale_apply_exp(p_max, params.scale_bmm1f);

        // if (!Is_first) {
        //     if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
        //         printf("after apply_exp=%.6f, %.6f\n", softmax.elt_[0][0], softmax.elt_[0][1]);
        //     }
        // }

        // Compute the sum.
        float p_sum[Mma_tile_p::MMAS_M * 2];
        // if (!Is_first) {
        //     int warp = tidx / Cta_tile_p::THREADS_PER_WARP;
        //     int lane = tidx % Cta_tile_p::THREADS_PER_WARP;
        //     for (int mi = 0; mi < Mma_tile_p::MMAS_M * 2; mi++) {
        //         p_sum[mi] = ((warp == 0) && (lane % 4 == 0)) ? expf(p_prev_lse[mi] - p_max[mi]) : 0;
        //     }
        // }
        // softmax.reduce_sum(p_sum);
        softmax.reduce_sum_before_sync_(p_sum);
        // softmax.template reduce_sum_before_sync_</*zero_init=*/Is_first>(p_sum);

        // float p_sum_log[Mma_tile_p::MMAS_M * 2];
        // for (int mi = 0; mi  < Mma_tile_p::MMAS_M * 2; ++mi) {
        //     float sum = p_sum[mi];
        //     // p_sum_log[mi] = (sum == 0.f || sum != sum) ? INFINITY : p_max[mi] + __logf(sum);
        //     constexpr float kLog2e = M_LOG2E;
        //     p_sum_log[mi] = (sum == 0.f || sum != sum) ? INFINITY : p_max[mi] * kLog2e + __log2f(sum);
        // }
        // // gmem_softmax_lse.store(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_sum));
        // gmem_softmax_lse.store(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_sum_log));
        // gmem_softmax_lse.move();

        // // Finalize softmax on the accumulators of P^T.
        // softmax.scale(p_sum);

        constexpr bool encode_dropout_in_sign_bit = Return_softmax;
        if (Is_dropout) {
            // softmax.template apply_dropout<encode_dropout_in_sign_bit>(ph0, params.p_dropout_in_uint);
            // softmax.template apply_dropout<encode_dropout_in_sign_bit>(ph0, ph1, params.p_dropout_in_uint);
            softmax.template apply_dropout_16bits<encode_dropout_in_sign_bit>(ph0, ph1, params.p_dropout_in_uint16_t);
        }

        // using Frag_p = fmha::Fragment_a<fmha::Row>;
        static_assert(Mma_tile_o::MMAS_M == Mma_tile_p::MMAS_M);
        static_assert(Mma_tile_o::MMAS_K == Mma_tile_p::MMAS_N);
        softmax.pack_noconvert(acc_p);
        cutlass::NumericArrayConverter<Element, ElementAccum, decltype(acc_p)::kElements, cutlass::FloatRoundStyle::round_to_nearest> convert_p;
        auto frag_p = convert_p(acc_p);

        if (Return_softmax) {
            // gmem_s.store(frag_p, mask);
            // gmem_s.template store<Mma_tile_o::MMAS_K, Mma_tile_o::MMAS_M>(frag_p, mask);
            gmem_s.store_cl(reinterpret_cast<const cutlass::Array<Element, 8>(&)[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_M]>(frag_p), mask);
            gmem_s.move();
        }

        // Commit the values for Q into shared memory.
        if(l < steps - 1) {
            // gmem_q.commit(gemm_q_k.smem_q);
            // smem_q_cl.store(reinterpret_cast<typename SmemIteratorQ::Fragment(&)>(gmem_q.fetch_));
            // smem_q_cl.store_with_pointer_offset(reinterpret_cast<typename SmemIteratorQ::Fragment(&)>(gmem_q.fetch_), ((l + 1) % 2) * Gemm1::Smem_tile_q::BYTES_PER_BUFFER / sizeof(Element));
            // smem_q_cl.store_with_pointer_offset(gmem_frag_q, ((l + 1) % 2) * Gemm1::Smem_tile_q::BYTES_PER_BUFFER / sizeof(Element));
            smem_q_cl.store(gmem_frag_q);
        }

        if (Is_dropout && encode_dropout_in_sign_bit) {
            // #pragma unroll
            // for( int ki = 0; ki < Mma_tile_o::MMAS_K; ki++ ) {
            //     #pragma unroll
            //     for( int mi = 0; mi < Mma_tile_o::MMAS_M; mi++ ) {
            //         frag_p[ki][mi].template hrelu_<elem_type>();
            //     }
            // }
            cutlass::epilogue::thread::ReLu<decltype(frag_p)> relu;
            frag_p = relu(frag_p);
        }

        // Declare the accumulators for the 2nd gemm.
        fmha::Fragment_accumulator acc_o[Mma_tile_o::MMAS_M][Mma_tile_o::MMAS_N];
        // fmha::Clear_accumulator<typename fmha::Accumulator_type, Cta_tile_o::WARPS_K>::apply(acc_o);

        // // Do this part of O = P^T * V^T.
        // #pragma unroll
        // for( int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki ) {
        //     fmha::gemm_cl<elem_type>(acc_o, frag_p[ki], frag_v[ki]);
        //     // if ((threadIdx.x == 4) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
        //     //     float2 tmp_p = __half22float2(reinterpret_cast<__half2 &>(frag_p[ki]));
        //     //     float2 tmp_v = __half22float2(reinterpret_cast<__half2 &>(frag_v[ki]));
        //     //     printf("Per warp, threadIdx.x = %d, frag_p = %.6f, %.6f, frag_v = %.6f, %.6f, acc_o=%.6f\n", threadIdx.x, tmp_p.x, tmp_p.y, tmp_v.x, tmp_v.y, acc_o[0][0].elt(0));
        //     // }
        // }

        WarpMmaPV mma_pv;
        typename WarpMmaPV::FragmentC acc_o_cl;
        static_assert(WarpMmaPV::FragmentC::kElements == Mma_tile_o::MMAS_M * Mma_tile_o::MMAS_N * 8);
        acc_o_cl.clear();

        // For some reason, WarpMmaPV::FragmentA has length K * N * (8|4) instead of just N * (8|4).
        // We we have to first cast frag_p to be array of k x (N * (8|4)), then cast each row to be
        // an array of WarpMmaPV::FragmentA (which is what mma_pv expects).
        static_assert(decltype(frag_p)::kElements == kIterationsPV * Mma_tile_o::MMAS_M * WarpMmaPV::FragmentA::kElements);
        const auto frag_p_reshaped = reinterpret_cast<const cutlass::Array<Element, WarpMmaPV::FragmentA::kElements> (&)[kIterationsPV]>(frag_p);
        #pragma unroll
        for( int ki = 0; ki < kIterationsPV; ++ki ) {
            mma_pv(acc_o_cl, reinterpret_cast<const typename WarpMmaPV::FragmentA(&)>(frag_p_reshaped[ki]), frag_v_cl[ki], acc_o_cl);
        }
        smem_o_cl.store(acc_o_cl);
        // #pragma unroll
        // for( int mi = 0; mi < Mma_tile_o::MMAS_M; ++mi ) {
        //     #pragma unroll
        //     for( int ni = 0; ni < Mma_tile_o::MMAS_N; ++ni ) {
        //         #pragma unroll
        //         for (int i = 0; i < 8; ++i) {
        //             acc_o[mi][ni].elt(i) = acc_o_cl[mi * Mma_tile_o::MMAS_M * 8 + ni * 8 + i];
        //         }
        //     }
        // }

        // if ((threadIdx.x % 32 == 16) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
        //     printf("Per warp, threadIdx.x = %d, acc_o=%.6f\n", threadIdx.x, acc_o[0][2].elt(0));
        // }

        // The mapping from tidx to rows changes between the softmax and the
        // O-reduction. So we recalculate the max.
        float p_max_o[Gmem_tile_o::STGS_PER_LOOP][Mma_tile_o::MMAS_M];
        int rows[Gmem_tile_o::STGS_PER_LOOP];
        // for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
        //     rows[jj] = tidx / Gmem_tile_o::THREADS_PER_ROW + jj * Gmem_tile_o::ROWS_PER_STG;
        // }
        using OutputTileThreadMap = typename Smem_O_cl::OutputTileThreadMap;
        constexpr int kRowsPerThread = OutputTileThreadMap::Iterations::kRow * Smem_O_cl::kIterationsStore;
        static_assert(Gmem_tile_o::STGS_PER_LOOP == kRowsPerThread);
        cutlass::MatrixCoord output_thread_offset = OutputTileThreadMap::initial_offset(tidx);
        const int output_thread_start_row = output_thread_offset.row();
        const int output_thread_start_column = output_thread_offset.column();
        for (int iter = 0; iter < Smem_O_cl::kIterationsStore; ++iter) {
            for (int row = 0; row < OutputTileThreadMap::Iterations::kRow; ++row) {
                rows[iter * OutputTileThreadMap::Iterations::kRow + row] = output_thread_start_row + iter * OutputTileThreadMap::Shape::kRow + row;
            }
        }
        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
        //     printf("rows calculation: ");
        //     for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
        //         printf("%d ", rows[jj]);
        //     }
        //     printf("\n");
        // }
        // When d = 16, O only has 16 x 16 = 256 elements, and each of the 128 threads wants
        // to write 4 elements, so only half of the thread should deal with O.
        bool o_rows_are_valid =
            (Kernel_traits::THREADS <= Gmem_tile_o::THREADS_PER_ROW * Gmem_tile_o::ROWS)
            || (tidx / Gmem_tile_o::THREADS_PER_ROW < Gmem_tile_o::ROWS);
        if (o_rows_are_valid) {
            softmax.reduce_max_after_sync_(p_max_o, rows);
        }
        static_assert(Mma_tile_o::MMAS_M == 1);
        for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
            p_max_o[jj][0] *= params.scale_bmm1f;
        }
        float p_prev_scale_o[Gmem_tile_o::STGS_PER_LOOP];
        if ((!Is_first) && o_rows_are_valid) {
            smem_softmax_lse.load(p_prev_scale_o, rows, l % 2);
        }
        // if (!Is_first) {
        //     if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
        //         printf("p_prev_scale_o=%.6f\n", p_prev_scale_o[0]);
        //     }
        // }

        static_assert(Gmem_tile_o::LOOPS == 1);

        // Swizzle the elements and do the final reduction.
        // smem_o.store(acc_o, 0);

        // __syncthreads();
        // smem_o_cl.store(acc_o_cl);

        // __syncthreads();
        // Smem_O_cl::store_static(
        //     reinterpret_cast<typename Smem_O_cl::AccumulatorTile (&)>(acc_o),
        //     &smem_[Gemm1::SMEM_OFFSET_O], tidx);

        // // __syncthreads();
        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0)) {
        //     printf("Smem_o content from new Smem write\n");
        //     for (int row = 0; row < 1; ++row) {
        //         for (int col = 0; col < 64; ++col) {
        //             printf("%f ", reinterpret_cast<float *>(&smem_[Gemm1::SMEM_OFFSET_O])[row * 64 + col]);
        //         }
        //         printf("\n");
        //     }
        // }

        // __syncthreads();

        // Make sure the data is in shared memory.
        __syncthreads();

        static_assert(Mma_tile_o::MMAS_M == 1);
        float p_sum_o[Gmem_tile_o::STGS_PER_LOOP][Mma_tile_o::MMAS_M];
        if (o_rows_are_valid) {
            softmax.reduce_sum_after_sync_(p_sum_o, rows);
        }
        if (!Is_first) {
            for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
                p_prev_scale_o[jj] = expf(p_prev_scale_o[jj] - p_max_o[jj][0]);
                p_sum_o[jj][0] += p_prev_scale_o[jj];
            }
        }

        float p_sum_log[Gmem_tile_o::STGS_PER_LOOP][Mma_tile_o::MMAS_M];
        #pragma unroll
        for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
            float sum = p_sum_o[jj][0];
            p_sum_log[jj][0] = (sum == 0.f || sum != sum) ? -INFINITY : p_max_o[jj][0] + __logf(sum);
            // if (sum == 0.f || sum != sum) {
            //     printf("loop_step_idx = %d, l = %d, tidx = %d, sum = %.6f, p_max_o = %.6f\n", loop_step_idx, l, tidx, sum, p_max_o[jj][0]);
            // }
            // if (Is_first) {
            //     if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
            //         printf("p_sum_log=%.6f\n", p_sum_log[jj][0]);
            //     }
            // }
            // TODO
            // if ((tidx % Gmem_tile_o::THREADS_PER_ROW == 0) && o_rows_are_valid) {
            if ((output_thread_start_column == 0) && o_rows_are_valid) {
                // if ((blockIdx.x == 0) && (blockIdx.y == 0)) {
                //     printf("thread %d storing lse, jj = %d, rows[jj] = %d, p_sum_log[jj] = %f\n", tidx, jj, rows[jj], p_sum_log[jj][0]);
                // }
                gmem_softmax_lse.store_row(
                    reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M]>(p_sum_log[jj]), rows[jj]);
            }
        }
        gmem_softmax_lse.move();

        // Load from shared memory.
        using ArrayTypeO = cutlass::Array<ElementAccum, OutputTileThreadMap::kElementsPerAccess>;
        static_assert(OutputTileThreadMap::kElementsPerAccess * kRowsPerThread == Smem_O_cl::kIterationsStore * Smem_O_cl::OutputFragment::kElements);
        cutlass::multiplies<ArrayTypeO> multiply_fragments;
        if (!Is_first) {
            auto out_cl_reshaped = reinterpret_cast<ArrayTypeO (&)[kRowsPerThread]>(out_cl);
            for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
                out_cl_reshaped[jj] = multiply_fragments(out_cl_reshaped[jj], p_prev_scale_o[jj]);
            }
            // for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
            //     out[jj] = fmha::fmul4(out[jj], p_prev_scale_o[jj]);
            // }
        }
        // smem_o.template load</*zero_init=*/Is_first>(out);
        // FMHAEpilogue::template load</*zero_init=*/Is_first>(
        //     reinterpret_cast<typename FMHAEpilogue::OutputFragment (&)[FMHAEpilogue::kFragmentsPerIteration]>(out),
        //     &smem_[Gemm1::SMEM_OFFSET_O], tidx, l);
        // smem_o_cl.template load</*zero_init=*/Is_first>(reinterpret_cast<typename Smem_O_cl::OutputFragment (&)[Smem_O_cl::kFragmentsPerIteration]>(out), tidx, l);
        smem_o_cl.template load</*zero_init=*/Is_first>(out_cl, tidx, l);

        const bool is_final_write =
            Is_last
            || ((loop_step_idx + 1) * Cta_tile_p::N >= binfo.actual_seqlen_k)
            || ((Is_causal) && ((begin + l) * Cta_tile_p::M < (loop_step_idx + 1) * Cta_tile_p::N));
        #pragma unroll
        // for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
        //     float sum = p_sum_o[jj][0];
        //     float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
        //     if (Is_dropout && is_final_write) {
        //         inv_sum *= params.rp_dropout;
        //     }
        //     out[jj] = fmha::fmul4(out[jj], inv_sum);
        // }
        auto out_cl_reshaped = reinterpret_cast<ArrayTypeO (&)[kRowsPerThread]>(out_cl);
        for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
            float sum = p_sum_o[jj][0];
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            if (Is_dropout && is_final_write) {
                inv_sum *= params.rp_dropout;
            }
            out_cl_reshaped[jj] = multiply_fragments(out_cl_reshaped[jj], inv_sum);
        }

        // if (Is_dropout && Is_last) {
        //     for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
        //         out[jj] = fmha::fmul4(out[jj], params.rp_dropout);
        //     }
        // }

        // Output the values.
        if (is_final_write) {
            // gmem_o.template store<elem_type>(out, 0);
            gmem_o.move();
            typename GmemIteratorO::Fragment out_cl_converted;
            cutlass::NumericArrayConverter<Element, ElementAccum, decltype(out_cl_converted)::kElements, cutlass::FloatRoundStyle::round_to_nearest> convert_o;
            #pragma unroll
            for (int iter = 0; iter < GmemIteratorO::kIterations; ++iter) {
                // out_cl = convert_o(reinterpret_cast<cutlass::Array<ElementAccum, GmemIteratorO::Fragment::kElements>(&)>(out[iter]));
                out_cl_converted = convert_o(out_cl[iter]);
                gmem_o_cl.store(out_cl_converted);
                gmem_o_cl.move();
                // ++gmem_o_cl;
            }
            // We also need to move gmem_o_accum_cl. For example, if Is_causal=true and seqlen=512,
            // in the first loop, we write the first 256 rows to gmem_o_cl and the last 256 rows to gmem_o_accum_cl.
            if (Is_first && !Is_last) { gmem_o_accum_cl.move(GmemIteratorOAccum::kIterations); }
        } else {
            // gmem_o_tmp.store(out, 0);
            // typename GmemIteratorOAccum::Fragment out_cl;
            #pragma unroll
            if (!Is_first) { gmem_o_accum_cl.move(-GmemIteratorOAccum::kIterations); }
            for (int iter = 0; iter < GmemIteratorOAccum::kIterations; ++iter) {
                // out_cl = reinterpret_cast<cutlass::Array<ElementAccum, GmemIteratorOAccum::Fragment::kElements>(&)>(out[iter]);
                // gmem_o_accum_cl.store(reinterpret_cast<cutlass::Array<ElementAccum, GmemIteratorOAccum::Fragment::kElements>(&)>(out[iter]));
                // gmem_o_accum_cl.store(out_cl);
                gmem_o_accum_cl.store(out_cl[iter]);
                gmem_o_accum_cl.move();
            }
        }

        // Move to the next part of the output.
        if (!(Is_first && Is_last)) { gmem_o_tmp.move(); }
        gemm_q_k.reload_k();

        // Make sure we are reading from the correct buffer.
        gemm_q_k.smem_q.move_to_next_read_buffer();
        // Trigger the load from shared memory for the next series of Q values.
        if(l < steps - 1) {
            gemm_q_k.reload_q();
            // gemm_q_k.reload_q(((l + 1) % 2) * Gemm1::Smem_tile_q::BYTES_PER_BUFFER);
        }

    }  // Outer loop over the sequence length.
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Return_softmax, typename Params>
inline __device__ void device_1xN_loop(const Params &params) {

    // The block index for the batch.
    const int bidb = blockIdx.x;
    // The block index for the head.
    const int bidh = blockIdx.y;
    // The thread index.
    const int tidx = threadIdx.x;

    const int tidx_global = (bidb * params.h + bidh) * blockDim.x * 2 + tidx;
    auto seeds = at::cuda::philox::unpack(params.philox_args);
    Philox ph0(std::get<0>(seeds), tidx_global, std::get<1>(seeds));
    Philox ph1(std::get<0>(seeds), tidx_global + blockDim.x, std::get<1>(seeds));
    constexpr int M = Kernel_traits::Cta_tile_p::M;
    const int STEPS = (params.seqlen_q + M - 1) / M;

    constexpr int blocksize_c = Kernel_traits::Cta_tile_p::N;
    if (params.seqlen_k == blocksize_c) {
        fmha::device_1xN_<Kernel_traits, Is_dropout, Is_causal, Return_softmax, true, true>(params, bidb, bidh, 0, STEPS, ph0, ph1, 0);
    } else {
        const int max_loop_steps = (params.seqlen_k + blocksize_c - 1) / blocksize_c;
        fmha::device_1xN_<Kernel_traits, Is_dropout, Is_causal, Return_softmax, true, false>(params, bidb, bidh, 0, STEPS, ph0, ph1, 0);
        for (int loop_step_idx = 1; loop_step_idx < max_loop_steps - 1; loop_step_idx++) {
            fmha::device_1xN_<Kernel_traits, Is_dropout, Is_causal, Return_softmax, false, false>(params, bidb, bidh, 0, STEPS, ph0, ph1, loop_step_idx);
        }
        fmha::device_1xN_<Kernel_traits, Is_dropout, Is_causal, Return_softmax, false, true>(params, bidb, bidh, 0, STEPS, ph0, ph1, max_loop_steps - 1);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha


// src/kernels/cuda/dsa_topk.cuh - the DSA indexer's choice of the top pools in O(n_vis) (one block, 1024 threads).
//
// The selection is the set the O(n_vis^2) rank loop chose: the `top` highest scores, ties by the lower pool index.
// It is written in ASCENDING POOL ORDER, not by rank.  Nothing after the selection reads the order (the attention is an
// online softmax over the cells, KV streaming resolves each cell on its own); the bitonic sort that kept the rank order
// cost ~20 us a call at any context (RTX 3090, 2048 winners, x21 full layers of glm-dsa a token); and pool order is the
// same whichever kernel chose the set - this one block, or the decode's several-block path for long contexts
// (glm_fast.cu, dsa_select) - so a verify window and its token path stay bit for bit the same.
//   0. top >= n_vis: every pool, no scores read;
//   1. a radix select (4 passes of 8 bits) over the scores' order-preserving uint32 keys finds the cut-off key - the
//      top-th largest - and how many of the keys equal to it are taken (the first ones in pool order);
//   2. one pass in pool order (each thread a contiguous run) writes the winners: an exclusive scan of the keys equal to
//      the cut-off gives each thread its share of them, a second one of the winners its offset.
// kMaxTop 2048: glm-dsa selects 2048 TOKENS (its indexer pools nothing); Flash's 2048 / 4 = 512 pools run as before.
#pragma once

#include <cuda_runtime.h>

namespace strata::kernels::dsa {

// a float's order-preserving key: larger float, larger key (-0 is +0; the scores are never NaN)
__device__ __forceinline__ unsigned score_key(float f) {
    const unsigned u = __float_as_uint(f == 0.0f ? 0.0f : f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

constexpr int kMaxTop = 2048;

// The exclusive prefix sum of v over the block (NT threads, a multiple of 32, at most 1024) and the block's total.
// s_warp: NT / 32 ints of shared memory.  Every thread of the block must call it.
template <int NT>
static __device__ __forceinline__ int block_excl_scan(int v, int* s_warp, int& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = lane < NT / 32 ? s_warp[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += y;
        }
        if (lane < NT / 32) s_warp[lane] = w;
    }
    __syncthreads();
    const int before = warp ? s_warp[warp - 1] : 0;
    total = s_warp[NT / 32 - 1];
    __syncthreads();   // s_warp is free for the next scan
    return before + x - v;
}

// cells[r * kpool + m] = p * kpool + m for the top `top` pools p of sc[0, n_vis), r-th in ascending p; 1 <= top <=
// kMaxTop, top <= n_vis, blockDim.x == 1024.  Every thread of the block must call it.  (static: each kernel file that
// includes this gets its own copy - an external symbol would be defined twice at link time)
static __device__ void select_top_pools(const float* __restrict__ sc, int n_vis, int top, int kpool, int* __restrict__ cells) {
    constexpr int NT = 1024;
    const int tid = threadIdx.x;
    if (top >= n_vis) {   // 0. every pool
        for (int i = tid; i < n_vis * kpool; i += NT) cells[i] = i;
        return;
    }
    __shared__ unsigned s_hist[256];
    __shared__ unsigned s_prefix, s_mask;
    __shared__ int s_need;
    __shared__ int s_warp[NT / 32];
    if (tid == 0) {
        s_prefix = 0u;
        s_mask = 0u;
        s_need = top;
    }
    __syncthreads();
    // 1. the cut-off key, a byte at a time from the top
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (tid < 256) s_hist[tid] = 0u;
        __syncthreads();
        const unsigned prefix = s_prefix, mask = s_mask;
        for (int i = tid; i < n_vis; i += NT) {
            const unsigned k = score_key(sc[i]);
            if ((k & mask) == prefix) atomicAdd(&s_hist[(k >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            int need = s_need;
            for (int b = 255; b >= 0; --b) {
                const int c = (int) s_hist[b];
                if (c >= need) {
                    s_prefix = prefix | ((unsigned) b << shift);
                    s_mask = mask | (255u << shift);
                    s_need = need;
                    break;
                }
                need -= c;
            }
        }
        __syncthreads();
    }
    const unsigned cut = s_prefix;
    const int need_eq = s_need;
    // 2. the winners, in pool order
    const int chunk = (n_vis + NT - 1) / NT;
    const int lo = min(n_vis, tid * chunk), hi = min(n_vis, lo + chunk);
    int gt = 0, eq = 0;
    for (int i = lo; i < hi; ++i) {
        const unsigned k = score_key(sc[i]);
        gt += k > cut;
        eq += k == cut;
    }
    int total = 0;
    int e = block_excl_scan<NT>(eq, s_warp, total);   // the keys equal to the cut-off before this run
    const int take_eq = max(0, min(eq, need_eq - e));
    int out = block_excl_scan<NT>(gt + take_eq, s_warp, total);
    for (int i = lo; i < hi; ++i) {
        const unsigned k = score_key(sc[i]);
        bool win = k > cut;
        if (k == cut) win = e++ < need_eq;
        if (win) {
            for (int m = 0; m < kpool; ++m) cells[out * kpool + m] = i * kpool + m;
            ++out;
        }
    }
}

}  // namespace strata::kernels::dsa

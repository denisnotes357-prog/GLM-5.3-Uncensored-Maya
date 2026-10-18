// src/kernels/glm_dsa_select_parity.cpp - the DSA indexer's selection (dsa_topk.cuh) against a CPU reference (GPU,
// synthetic, no model).
//
// For glm-dsa's shape (kpool 1, the top 2048 tokens) and Flash's (kpool 4, the top 512 pools and the tail), over
// contexts from one pool to past 256K and three kinds of scores (random, heavily tied with many zeros, all equal):
//   1. the decode's one-block selection (gf::dsa_select without a workspace),
//   2. the decode's several-block one (with a workspace, STRATA_GLM_SEL_MULTI=1 so it runs on every size it takes),
//   3. the prompt path's (gb::dsa_select, one row)
// all write exactly the reference: the `top` highest scores, ties by the lower pool index, in ascending pool order,
// expanded to cells, then the tail cells (-1 where masked).  Then the cost of 1 and 2 at kpool 1 for a range of
// contexts (the table STRATA_GLM_SEL_MULTI's default comes from).
#include "strata/kernels/glm_batch.hpp"
#include "strata/kernels/glm_fast.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace gf = strata::kernels::glmf;
namespace gb = strata::kernels::glmb;

namespace {
int g_fail = 0;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

// the reference cells: the top `top` of sc (descending, ties by the lower index) in ascending order, expanded, then
// the tail - as dsa_select_kernel lays them out (n_sel entries)
std::vector<int> reference(const std::vector<float>& sc, int n_vis, int kpool, int top_pools, int n_sel, int pos) {
    std::vector<int> cells((size_t) n_sel, -1);
    const int top = std::min(top_pools, n_vis);
    std::vector<int> idx((size_t) n_vis);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return sc[(size_t) a] > sc[(size_t) b]; });
    idx.resize((size_t) top);
    std::sort(idx.begin(), idx.end());
    for (int r = 0; r < top; ++r)
        for (int m = 0; m < kpool; ++m) cells[(size_t) r * kpool + m] = idx[(size_t) r] * kpool + m;
    for (int m = 0; m < kpool - 1; ++m) {
        const int cell = n_vis * kpool + m;
        if (cell <= pos && top_pools * kpool + m < n_sel) cells[(size_t) top_pools * kpool + m] = cell;
    }
    return cells;
}

std::vector<float> scores(int n, int kind, std::mt19937& rng) {
    std::vector<float> s((size_t) n);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& x : s) {
        if (kind == 0) x = nd(rng);                                             // random
        else if (kind == 1) x = std::max(0.0f, std::round(nd(rng) * 2.0f) / 8.0f);   // ties, half of them 0
        else x = 0.5f;                                                          // all equal
    }
    return s;
}

const char* kind_name(int k) { return k == 0 ? "random" : k == 1 ? "tied" : "equal"; }

bool same(const std::vector<int>& a, const std::vector<int>& b, int n, const char* what, int kpool, int n_vis,
          int kind) {
    for (int i = 0; i < n; ++i)
        if (a[(size_t) i] != b[(size_t) i]) {
            std::printf("FAIL %s kpool %d n_vis %d %s: cell %d is %d, reference %d\n", what, kpool, n_vis,
                        kind_name(kind), i, a[(size_t) i], b[(size_t) i]);
            ++g_fail;
            return false;
        }
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    const bool selftest = argc > 1 && std::strcmp(argv[1], "--selftest") == 0;
    setenv("STRATA_GLM_SEL_MULTI", "1", 1);   // the several-block path wherever a workspace is given
    const int max_n = 262147;
    float* d_score = nullptr;
    int *d_cells = nullptr, *d_nsel = nullptr;
    void* d_ws = nullptr;
    ck(cudaMalloc(&d_score, (size_t) max_n * sizeof(float)), "malloc score");
    ck(cudaMalloc(&d_cells, 8192 * sizeof(int)), "malloc cells");
    ck(cudaMalloc(&d_nsel, sizeof(int)), "malloc n_sel");
    ck(cudaMalloc(&d_ws, gf::kSelWsBytes), "malloc ws");
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    std::mt19937 rng(1234);
    struct Shape { int kpool, top, tail; } shapes[] = {{1, 2048, 0}, {4, 512, 1}};
    const int sizes[] = {1, 7, 100, 511, 512, 513, 2047, 2048, 2049, 5000, 16383, 40000, 131072, 262147};
    int checked = 0;
    for (const Shape& sh : shapes)
        for (int n_vis : sizes)
            for (int kind = 0; kind < 3; ++kind) {
                const std::vector<float> sc = scores(n_vis, kind, rng);
                ck(cudaMemcpy(d_score, sc.data(), sc.size() * sizeof(float), cudaMemcpyHostToDevice), "h2d");
                const int top_pools = std::min(sh.top, n_vis);
                const int n_sel = sh.kpool * top_pools + (sh.tail ? sh.kpool - 1 : 0);
                const int pos = sh.kpool == 1 ? n_vis - 1 : n_vis * sh.kpool + sh.kpool - 2;   // n_vis pools done, the whole tail
                const std::vector<int> ref = reference(sc, n_vis, sh.kpool, top_pools, n_sel, pos);
                std::vector<int> got((size_t) n_sel);
                for (int path = 0; path < 3; ++path) {
                    ck(cudaMemset(d_cells, 0x7f, 8192 * sizeof(int)), "memset");
                    if (path == 0) gf::dsa_select(d_score, n_vis, sh.kpool, top_pools, n_sel, pos, d_cells, s);
                    else if (path == 1) gf::dsa_select(d_score, n_vis, sh.kpool, top_pools, n_sel, pos, d_cells, s, d_ws);
                    else gb::dsa_select(d_score, max_n, pos, sh.kpool, sh.top, sh.tail, 1, 8192, d_cells, d_nsel, s);
                    ck(cudaStreamSynchronize(s), "select");
                    ck(cudaMemcpy(got.data(), d_cells, (size_t) n_sel * sizeof(int), cudaMemcpyDeviceToHost), "d2h");
                    if (path == 2) {
                        int ns = 0;
                        ck(cudaMemcpy(&ns, d_nsel, sizeof(int), cudaMemcpyDeviceToHost), "d2h n_sel");
                        if (ns != n_sel) {
                            std::printf("FAIL prompt path kpool %d n_vis %d: n_sel %d, decode %d\n", sh.kpool, n_vis,
                                        ns, n_sel);
                            ++g_fail;
                        }
                    }
                    same(got, ref, n_sel, path == 0 ? "one block" : path == 1 ? "blocks" : "prompt path", sh.kpool,
                         n_vis, kind);
                    ++checked;
                }
            }
    std::printf("selection: %d cases checked, %d failed\n", checked, g_fail);

    // the cost, kpool 1, top 2048, random scores
    std::printf("%10s %12s %12s\n", "n_vis", "one block us", "blocks us");
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    const int tsizes[] = {4096, 8192, 12288, 16384, 24576, 32768, 65536, 131072, 262144};
    for (int n : tsizes) {
        const std::vector<float> sc = scores(n, 0, rng);
        ck(cudaMemcpy(d_score, sc.data(), sc.size() * sizeof(float), cudaMemcpyHostToDevice), "h2d");
        float us[2];
        for (int path = 0; path < 2; ++path) {
            const int reps = selftest ? 20 : 200;
            for (int r = 0; r < 3; ++r)
                gf::dsa_select(d_score, n, 1, 2048, 2048, n - 1, d_cells, s, path ? d_ws : nullptr);
            cudaEventRecord(a, s);
            for (int r = 0; r < reps; ++r)
                gf::dsa_select(d_score, n, 1, 2048, 2048, n - 1, d_cells, s, path ? d_ws : nullptr);
            cudaEventRecord(b, s);
            ck(cudaEventSynchronize(b), "time");
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, a, b);
            us[path] = ms * 1000.0f / (float) reps;
        }
        std::printf("%10d %12.1f %12.1f\n", n, us[0], us[1]);
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}

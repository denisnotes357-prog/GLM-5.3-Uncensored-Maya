// tests/glm_dsa_geometry_test.cpp - the port's P1: Glm5Geometry::from_gguf on BOTH architectures
// (docs/GLM53-ENGINE-PORT.md).  Parsing only: no weights are uploaded and no step is taken.
//
//   glm_dsa_geometry_test MODEL.gguf [--expect-tiny | --expect-flash]
//
//   (no flag)        parse and print one "geometry: ..." line; exit 0/1 on parse success/failure
//   --expect-tiny    also require the synthetic GLM-5.3 that tools/glm53/make_tiny.py writes
//                    (arch Dsa + the six-layer geometry the checks pin below)
//   --expect-flash   also require the committed Flash fixture data/glm5-synth.gguf to parse as
//                    Flash with hc 4 - the regression gate that P1 changed no Flash bit
//
// The last line is always "glm_dsa_geometry_test: PASS" or "glm_dsa_geometry_test: FAIL: ...".
#include "strata/core/glm_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using strata::core::Glm5Geometry;   // the struct lives in strata::core (glm_model.hpp:46)

std::vector<std::string> problems;

void want(bool ok, const std::string& what) {
    if (!ok) problems.push_back(what);
}

template <class T>
void want_eq(const T& got, const T& want_v, const std::string& what) {
    if (!(got == want_v)) {
        char b[160];
        std::snprintf(b, sizeof(b), "%s: got %lld want %lld", what.c_str(),
                      (long long) got, (long long) want_v);
        problems.push_back(b);
    }
}

const char* arch_name(Glm5Geometry::Glm5Arch a) {
    return a == Glm5Geometry::Glm5Arch::Dsa ? "Dsa" : "Flash";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: glm_dsa_geometry_test MODEL.gguf [--expect-tiny | --expect-flash]\n");
        return 1;
    }
    const std::string path = argv[1];
    const bool expect_tiny = argc > 2 && std::string(argv[2]) == "--expect-tiny";
    const bool expect_flash = argc > 2 && std::string(argv[2]) == "--expect-flash";

    strata::core::Glm5Geometry g;
    std::string err;
    try {
        strata::GgufFile f(path);
        if (!strata::core::Glm5Geometry::from_gguf(f, g, err)) {
            std::printf("glm_dsa_geometry_test: FAIL: %s\n", err.c_str());
            return 1;
        }
    } catch (const std::exception& e) {
        std::printf("glm_dsa_geometry_test: FAIL: cannot read %s: %s\n", path.c_str(), e.what());
        return 1;
    }

    // one quotable line with every geometry field the two readers produce
    std::string idx;
    for (size_t i = 0; i < g.indexer_full.size(); ++i) idx += (i ? "," : "") + std::to_string(g.indexer_full[i]);
    std::printf("geometry: %s n_embd=%d n_layers=%d nextn=%d n_head=%d q_lora=%d kv_lora=%d "
                "qk_nope=%d qk_rope=%d v_head=%d lat_row=%d rope_base=%g idx_heads=%d idx_key=%d "
                "idx_top_k=%d idx_kpool=%d select_tail=%d mhc=%d hc=%d sinkhorn=%d n_expert=%d "
                "n_exp_used=%d n_ff_exp=%d n_shared=%d dense_lead=%d n_ff_dense=%d w_scale=%g "
                "norm_w=%d swiglu=%g/%g n_vocab=%d full=%zu/%zu indexer_full=[%s]\n",
                arch_name(g.arch), g.n_embd, g.n_layers, g.nextn, g.n_head, g.q_lora, g.kv_lora,
                g.qk_nope, g.qk_rope, g.v_head, g.lat_row(), (double) g.rope_base, g.idx_heads,
                g.idx_key, g.idx_top_k, g.idx_kpool, g.idx_select_tail, (int) g.mhc, g.hc,
                g.sinkhorn_iters, g.n_expert, g.n_exp_used, g.n_ff_exp, g.n_shared, g.dense_lead,
                g.n_ff_dense, (double) g.w_scale, g.norm_w, (double) g.swiglu_exp,
                (double) g.swiglu_shexp, g.n_vocab,
                (size_t) std::count(g.indexer_full.begin(), g.indexer_full.end(), 1),
                g.indexer_full.size(), idx.c_str());

    if (expect_tiny) {
        want(g.arch == Glm5Geometry::Glm5Arch::Dsa, "arch == Dsa");
        want_eq(g.n_embd, 512, "n_embd");
        want_eq(g.n_layers, 6, "n_layers");
        want(g.nextn == 0 || g.nextn == 1, "nextn is 0 or 1 as the file says");  // 0 for an HF export
        want_eq(g.n_head, 4, "n_head");
        want_eq(g.q_lora, 512, "q_lora");
        want_eq(g.kv_lora, 512, "kv_lora");
        want_eq(g.qk_nope, 192, "qk_nope");
        want_eq(g.qk_rope, 64, "qk_rope");
        want_eq(g.v_head, 256, "v_head");
        want_eq(g.lat_row(), 576, "lat_row");
        want(std::fabs((double) g.rope_base - 8e6) < 1.0, "rope_base == 8e6");
        want_eq(g.idx_heads, 32, "idx_heads");
        want_eq(g.idx_key, 128, "idx_key");
        want_eq(g.idx_top_k, 16, "idx_top_k");
        want_eq(g.idx_kpool, 1, "idx_kpool");
        want(g.indexer_full == std::vector<int>({1, 1, 0, 0, 1, 0}), "indexer_full == 1,1,0,0,1,0");
        want(!g.mhc, "mhc == false");
        want_eq(g.hc, 1, "hc");
        want_eq(g.n_expert, 16, "n_expert");
        want_eq(g.n_exp_used, 8, "n_exp_used");
        want_eq(g.n_ff_exp, 512, "n_ff_exp");
        want_eq(g.dense_lead, 1, "dense_lead");
        want_eq(g.n_ff_dense, 1024, "n_ff_dense");
        want(std::fabs((double) g.w_scale - 2.5) < 1e-6, "w_scale == 2.5");
        want(std::isinf(g.swiglu_exp) && std::isinf(g.swiglu_shexp), "swiglu inf/inf (unclamped)");
    }
    if (expect_flash) {
        want(g.arch == Glm5Geometry::Glm5Arch::Flash, "arch == Flash");
        want_eq(g.hc, 4, "hc (the mHC stack the Flash kernels pin)");
        want(g.mhc, "mhc == true");   // the Flash reader never touches the new flags
    }

    if (!problems.empty()) {
        for (const auto& p : problems) std::printf("  FAIL: %s\n", p.c_str());
        std::printf("glm_dsa_geometry_test: FAIL: %zu check(s): first: %s\n", problems.size(),
                    problems[0].c_str());
        return 1;
    }
    std::printf("glm_dsa_geometry_test: PASS\n");
    return 0;
}
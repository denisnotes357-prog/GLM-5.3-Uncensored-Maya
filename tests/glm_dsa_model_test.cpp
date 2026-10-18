// tests/glm_dsa_model_test.cpp - the glm-dsa reference step (docs/GLM53-ENGINE-PORT.md P2) against transformers.
//
//   glm_dsa_model_test <tiny-f32.gguf | pack dir> <ref dir> [--tol 1e-3] [--save F] [--compare F --cmin 0.999]
//                      [--prefill N]
//
// A pack directory (tools/iq_pack.py over a quantized GGUF) loads through load_pack_env: the token fast path, or the
// reference step with STRATA_GLM_SLOW=1.  --save writes every position's logits (float32 [T, vocab]); --compare reads
// such a file and gates the MEDIAN over positions of the logits' correlation with it at --cmin (fast vs slow on the SAME
// pack: the fast path's q8_1 activations differ, and on a random tiny model they flip an expert or a selected token now
// and then, so a max |d| or a minimum is noise - tiny-iq4 2026-10-15: median 0.99978, min 0.984; with q's RoPE
// dropped, median 0.99453).  A quantized pack is
// far from the f32 reference too (the reference step itself is 1.0 off on tools/glm53/recipes/tiny-iq4.json), so pass a
// --tol to suit and --no-top1.  --prefill N reads the first N tokens through the prompt path (prefill_run: the batched
// layers) and streams the rest; only positions N.. are compared - the prompt path must leave the caches the token path
// would have left.
//
// <ref dir> is tools/glm53/make_tiny.py's output: tokens.npy (int32 [T]) and logits.npy (float32 [T, vocab]) -
// GlmMoeDsaForCausalLM's eager float32 pass over the same bf16-rounded weights.  The runner streams the T
// tokens one at a time (decode is streaming) and every position's logits must match the reference: max |d|
// under --tol and the same argmax.  The tiny model's index_topk (16) is far below T (96), so most positions
// attend a sparse selection, and three of its six layers reuse a previous layer's selection.
#include "strata/core/glm_model.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

[[noreturn]] void die(const std::string& what) {
    std::fprintf(stderr, "glm_dsa_model_test: FAIL: %s\n", what.c_str());
    std::exit(1);
}

// a version-1/2 .npy: the header's shape, then raw little-endian data of `want_descr`
template <class T>
std::vector<T> load_npy(const std::string& path, const char* want_descr, std::vector<size_t>& shape) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) die("cannot open " + path);
    unsigned char pre[10];
    if (std::fread(pre, 1, 10, f) != 10 || std::memcmp(pre, "\x93NUMPY", 6) != 0) die(path + ": not a .npy");
    size_t hlen = pre[8] | (pre[9] << 8);
    if (pre[6] >= 2) {
        unsigned char ext[2];
        if (std::fread(ext, 1, 2, f) != 2) die(path + ": short header");
        hlen |= (size_t) ext[0] << 16 | (size_t) ext[1] << 24;
    }
    std::string h(hlen, '\0');
    if (std::fread(h.data(), 1, hlen, f) != hlen) die(path + ": short header");
    if (h.find(want_descr) == std::string::npos) die(path + ": dtype is not " + want_descr + " (" + h + ")");
    if (h.find("'fortran_order': False") == std::string::npos) die(path + ": fortran order");
    const size_t a = h.find('(', h.find("'shape'")), b = h.find(')', a);
    shape.clear();
    size_t n = 1;
    for (size_t i = a + 1; i < b;) {
        while (i < b && (h[i] == ' ' || h[i] == ',')) ++i;
        if (i >= b) break;
        const size_t d = std::strtoull(h.c_str() + i, nullptr, 10);
        shape.push_back(d);
        n *= d;
        while (i < b && h[i] != ',') ++i;
    }
    std::vector<T> v(n);
    if (std::fread(v.data(), sizeof(T), n, f) != n) die(path + ": short data");
    std::fclose(f);
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: glm_dsa_model_test <tiny-f32.gguf> <ref dir> [--tol 1e-3]\n");
        return 2;
    }
    double tol = 1e-3, cmin = 0.999;
    std::string save, compare;
    bool need_top1 = true;
    int prefill_n = 0;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--tol") == 0 && i + 1 < argc) tol = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--cmin") == 0 && i + 1 < argc) cmin = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--save") == 0 && i + 1 < argc) save = argv[++i];
        else if (std::strcmp(argv[i], "--compare") == 0 && i + 1 < argc) compare = argv[++i];
        else if (std::strcmp(argv[i], "--no-top1") == 0) need_top1 = false;
        else if (std::strcmp(argv[i], "--prefill") == 0 && i + 1 < argc) prefill_n = std::atoi(argv[++i]);
    }
    setenv("STRATA_GLM_DSA_DEV", "1", 1);   // the forward guard: this IS the development check

    std::vector<size_t> ts, ls;
    const auto tokens = load_npy<int32_t>(std::string(argv[2]) + "/tokens.npy", "<i4", ts);
    const auto ref = load_npy<float>(std::string(argv[2]) + "/logits.npy", "<f4", ls);
    if (ls.size() != 2 || ls[0] != tokens.size()) die("logits.npy is not [T, vocab] for tokens.npy's T");
    const size_t T = tokens.size(), V = ls[1];

    strata::core::Glm5Model m;
    std::string err;
    const std::string path = argv[1];
    const bool is_pack = path.size() < 5 || path.compare(path.size() - 5, 5, ".gguf") != 0;
    if (is_pack ? !m.load_pack_env(path, (int64_t) T + 8, err) : !m.load(path, (int64_t) T + 8, err)) die("load: " + err);
    std::vector<float> other;
    if (!compare.empty()) {
        FILE* f = std::fopen(compare.c_str(), "rb");
        if (f == nullptr) die("cannot open " + compare);
        std::fseek(f, 0, SEEK_END);
        const long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        other.resize((size_t) n / sizeof(float));
        if (std::fread(other.data(), sizeof(float), other.size(), f) != other.size()) die("short read on " + compare);
        std::fclose(f);
    }
    std::vector<float> all;
    if (m.geometry().arch != strata::core::Glm5Geometry::Glm5Arch::Dsa) die("the GGUF is not glm-dsa");
    if ((size_t) m.geometry().n_vocab != V) die("vocab mismatch");

    double worst = 0.0, worst_ref = 0.0, cworst = 0.0;
    std::vector<double> corrs;
    size_t worst_t = 0, top1 = 0, ctop1 = 0;
    std::vector<float> lg;
    size_t t_first = 0;
    if (prefill_n > 0) {
        const std::vector<int32_t> head(tokens.begin(), tokens.begin() + std::min<size_t>((size_t) prefill_n, T));
        if (!m.prefill_run(head, err)) die("prefill: " + (err.empty() ? std::string("no prompt path") : err));
        t_first = head.size();
        std::printf("glm_dsa_model_test: %zu tokens through the prompt path\n", t_first);
    }
    for (size_t t = t_first; t < T; ++t) {
        if (!m.forward({tokens[t]}, lg, err)) die("forward at " + std::to_string(t) + ": " + err);
        const float* r = ref.data() + t * V;
        double d = 0.0, rmax = 0.0;
        size_t am = 0, ar = 0;
        for (size_t v = 0; v < V; ++v) {
            d = std::max(d, (double) std::fabs(lg[v] - r[v]));
            rmax = std::max(rmax, (double) std::fabs(r[v]));
            if (lg[v] > lg[am]) am = v;
            if (r[v] > r[ar]) ar = v;
        }
        top1 += am == ar;
        if (!save.empty()) all.insert(all.end(), lg.begin(), lg.begin() + (long) V);
        if (!other.empty()) {
            if (other.size() < (t + 1) * V) die(compare + " holds fewer positions");   // (its row t: position t)
            const float* o = other.data() + t * V;
            size_t ao = 0;
            double sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0;
            for (size_t v = 0; v < V; ++v) {
                cworst = std::max(cworst, (double) std::fabs(lg[v] - o[v]));
                if (o[v] > o[ao]) ao = v;
                sa += lg[v];
                sb += o[v];
                sab += (double) lg[v] * o[v];
                saa += (double) lg[v] * lg[v];
                sbb += (double) o[v] * o[v];
            }
            const double n = (double) V, cov = sab - sa * sb / n;
            const double corr = cov / std::sqrt((saa - sa * sa / n) * (sbb - sb * sb / n));
            corrs.push_back(corr);
            ctop1 += am == ao;
        }
        if (d > worst) {
            worst = d;
            worst_t = t;
            worst_ref = rmax;
        }
        if (t < 3 || t == 15 || t == 16 || t + 1 == T)
            std::printf("t=%3zu  max|d| %.3e  (max|ref| %.3f)  argmax %zu / ref %zu\n", t, d, rmax, am, ar);
    }
    std::printf("glm_dsa_model_test: %zu positions, worst max|d| %.3e at t=%zu (max|ref| %.3f), top-1 %zu/%zu, tol %.1e\n",
                T - t_first, worst, worst_t, worst_ref, top1, T - t_first, tol);
    if (!save.empty()) {
        FILE* f = std::fopen(save.c_str(), "wb");
        if (f == nullptr || std::fwrite(all.data(), sizeof(float), all.size(), f) != all.size()) die("cannot write " + save);
        std::fclose(f);
    }
    if (!other.empty()) {
        std::vector<double> sorted = corrs;
        std::sort(sorted.begin(), sorted.end());
        const double median = sorted[sorted.size() / 2];
        std::printf("glm_dsa_model_test: against %s: median correlation %.6f (cmin %.4f), min %.6f, worst max|d| %.3e, "
                    "top-1 %zu/%zu\n", compare.c_str(), median, cmin, sorted.front(), cworst, ctop1, T - t_first);
        if (median < cmin) die("logits differ from " + compare);
    }
    if (worst > tol || (need_top1 && top1 != T - t_first)) die("logits differ from the transformers reference");
    std::printf("glm_dsa_model_test: PASS\n");
    return 0;
}

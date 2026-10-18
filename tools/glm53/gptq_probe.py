"""tools/glm53/gptq_probe.py - does the calibration beat round-to-nearest?  Three roundings of the same weights.

    python gptq_probe.py --layer 40 [--experts 12] [--prior 4096,16384,65536] [--damp 0.05,0.1]

Every variant goes through llama.cpp's own quantizer (ggml_quantize_chunk) with the recipe's type for that
tensor, and they differ only in what they are told about the row:
    RTN    no importance vector                  - what quantize_rtn.py wrote
    IMX    calib.py's mean x^2 per input channel - what quantize.py writes for every tensor but gate/up
    GPTQ   error feedback over 256-column superblocks (gptq.py), gate/up only; down keeps IMX as at Maya
scored against the FP8 checkpoint:
    wre    sqrt( sum im (Wq - W)^2 / sum im W^2 )   imatrix-weighted relative weight error (the metric)
    rel    ||Wq - W|| / ||W||                       the number check_quant.py reports
    xwerr  ||X Wq^T - X W^T|| / ||X W^T||           on the ffn_in tokens routed to that expert (gate+up)
    yerr   ||w (y_q - y)||^2 / ||w y||^2            the expert's output on held-out ffn_in tokens
plus one dense tensor (the shared expert's gate, Q4_K) for RTN vs IMX, and the --prior/--damp sweep of the GPTQ
hyperparameters: the last table is what quantize.py should be launched with.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
LLAMA_CPP = pathlib.Path("/data/glm53/llama.cpp")
GGML_LIB = str(LLAMA_CPP / "build/bin/libggml-base.so")
MODEL_DIR = "/data/models/GLM-5.3-FP8"
STATS_DIR = "/data/glm53/calib/stats"

sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))
sys.path.insert(0, str(LLAMA_CPP))
sys.path.insert(0, str(HERE))

import gguf  # noqa: E402
import quantize_rtn as qr  # noqa: E402
import gptq as GQ  # noqa: E402

QT = gguf.GGMLQuantizationType


def quant_deq(G, t: int, w: np.ndarray, im: np.ndarray | None, pool=None) -> np.ndarray:
    """float32 [rows, n] -> the same rows quantized and dequantized back (with or without an importance vector).
    The rows are split over `pool` the way quantize_rtn.py splits them (ggml quantizes row chunks in parallel)."""
    rows, n = w.shape
    w = np.ascontiguousarray(w, dtype=np.float32)
    dst = np.empty((rows, G.row_size(t, n)), dtype=np.uint8)
    G.lib.ggml_quantize_init(t)
    ime = None if im is None else np.ascontiguousarray(im, np.float32)
    ptr = None if ime is None else ime.ctypes.data
    k = 1 if pool is None else max(1, min(pool._max_workers * 4, rows // 16 or 1))
    cuts = [round(i * rows / k) for i in range(k + 1)]

    def work(i: int) -> None:
        r0, r1 = cuts[i], cuts[i + 1]
        if r1 > r0:
            G.lib.ggml_quantize_chunk(t, w.ctypes.data, dst.ctypes.data, r0 * n, r1 - r0, n, ptr)

    if k == 1:
        work(0)
    else:
        list(pool.map(work, range(k)))
    q = gguf.quants.dequantize(np.ascontiguousarray(dst), QT(t))
    return np.asarray(q, dtype=np.float32).reshape(rows, n)


def wre(q: np.ndarray, w: np.ndarray, im: np.ndarray) -> tuple[float, float]:
    """(imatrix-weighted relative error, plain relative error); `im` is one value per input column."""
    q, w, ww = q.astype(np.float64), w.astype(np.float64), im.astype(np.float64)
    num = float((ww * (q - w) ** 2).sum())
    den = float((ww * w ** 2).sum())
    return (num / max(den, 1e-30)) ** 0.5, float(np.linalg.norm(q - w) / (np.linalg.norm(w) + 1e-30))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", default=STATS_DIR)
    ap.add_argument("--fp8", default=MODEL_DIR)
    ap.add_argument("--recipe", default=str(HERE / "recipes/g-m24.json"))
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--experts", type=int, default=12, help="as many hottest as coldest, the rest random")
    ap.add_argument("--prior", default="4096,16384,65536")
    ap.add_argument("--damp", default="0.05,0.1")
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--ggml", default=GGML_LIB)
    a = ap.parse_args()

    dev = torch.device("cuda")
    rules, default = qr.load_recipe(pathlib.Path(a.recipe))
    cfg = json.load(open(pathlib.Path(a.fp8) / "config.json"))
    cfg = cfg.get("text_config", cfg)
    E, H, I = int(cfg["n_routed_experts"]), int(cfg["hidden_size"]), int(cfg["moe_intermediate_size"])
    stats_dir = pathlib.Path(a.stats)
    il = a.layer
    st = torch.load(stats_dir / f"stats_{il:02d}.pt")
    moe, lin = st.get("moe"), st["linear"]
    if moe is None:
        raise SystemExit(f"layer {il} has no MoE statistics (it is a dense layer? use 3..77)")
    pick = lambda part, shape: qr.pick_type(f"blk.{il}.{part}", shape, rules, default)  # noqa: E731
    t_gu = pick("ffn_gate_exps.weight", (I, H))
    t_dn = pick("ffn_down_exps.weight", (H, I))
    t_sh = pick("ffn_gate_shexp.weight", (I, H))
    G = qr.Rtg(a.ggml, a.threads).G
    ck = GQ.Checkpoint(a.fp8)
    pool = cf.ThreadPoolExecutor(a.threads)

    cnt = moe["n"].numpy()
    k = a.experts // 3
    hot = [int(e) for e in np.argsort(-cnt)[:k]]
    cold = [int(e) for e in np.argsort(cnt)[:k]]
    mid = [int(e) for e in np.random.default_rng(0).choice([e for e in range(E) if e not in hot + cold],
                                                           a.experts - len(hot) - len(cold), replace=False)]
    experts = hot + mid + cold
    print(f"layer {il}: gate/up {t_gu.name}, down {t_dn.name}, shared gate {t_sh.name}; "
          f"{len(experts)} experts, their calibration tokens {cnt[experts].astype(int).tolist()}", flush=True)

    t0 = time.time()
    layer = GQ.Layer(ck, cfg, stats_dir, il, dev)
    n = layer.X.shape[0]
    sel_tr = torch.zeros(n, dtype=torch.bool, device=dev)
    sel_tr[: n * 8 // 10 // 2048 * 2048] = True
    sel_ho = ~sel_tr
    ref_g = torch.stack([layer.weights(e)[0] for e in experts])          # [C, I, H] fp32 on the GPU
    ref_u = torch.stack([layer.weights(e)[1] for e in experts])
    ref_d = torch.stack([layer.weights(e)[2] for e in experts])          # [C, H, I]
    im_g = torch.stack([(moe["imat_gu"][e].double() / moe["n"][e].clamp(min=1).double()).float()
                        for e in experts]).to(dev)                      # [C, H]
    im_d = torch.stack([(moe["imat_d"][e].double() / moe["n"][e].clamp(min=1).double()).float()
                        for e in experts]).to(dev)                      # [C, I]
    ref_gu = torch.cat([ref_g, ref_u], 1)                                # [C, 2I, H]
    print(f"ffn_in {n} tokens ({int(sel_ho.sum())} held out), routing in {time.time() - t0:.1f} s", flush=True)

    # ---------- RTN / IMX through ggml, the recipe's types ----------
    Wgu = ref_gu.cpu().numpy()
    Wdn = ref_d.cpu().numpy()
    v = {}
    for tag, use_im in (("RTN", False), ("IMX", True)):
        t1 = time.time()
        v[(tag, "gu")] = np.stack([quant_deq(G, int(t_gu), Wgu[i],
                                             im_g[i].cpu().numpy() if use_im else None, pool)
                                   for i in range(len(experts))])
        print(f"  {tag} gate+up {len(experts)} experts in {time.time() - t1:.1f} s", flush=True)
        t1 = time.time()
        v[(tag, "dn")] = np.stack([quant_deq(G, int(t_dn), Wdn[i],
                                             im_d[i].cpu().numpy() if use_im else None, pool)
                                   for i in range(len(experts))])
        print(f"  {tag} down    {len(experts)} experts in {time.time() - t1:.1f} s", flush=True)

    cg = GQ.Codec(G.lib, int(t_gu))
    cu = GQ.Codec(G.lib, int(t_gu))

    def gptq_deq(prior: float, damp: float):
        """gate/up rounded with error feedback: [C, 2I, H] dequantized (and the bytes' shapes checked)."""
        out = []
        for c0 in range(0, len(experts), a.chunk):
            qg, qu, dgu, _ = GQ.round_gate_up(layer, experts[c0:c0 + a.chunk], cg, cu, pool, prior, damp,
                                                  sel_tr)
            want = (len(experts[c0:c0 + a.chunk]), I, H // 256 * cg.bpb)
            if qg.shape != want:
                raise SystemExit(f"gptq bytes {qg.shape}, planned {want}")
            out.append(dgu.cpu())
            del dgu, qg, qu
            torch.cuda.empty_cache()
        return torch.cat(out, 0)

    # ---------- metrics ----------
    def w_err_gu(V: np.ndarray) -> tuple[float, float]:
        rs = [wre(V[i], Wgu[i], im_g[i].cpu().numpy()) for i in range(len(experts))]
        return float(np.mean([r[0] for r in rs])), float(np.mean([r[1] for r in rs]))

    def w_err_dn(V: np.ndarray) -> tuple[float, float]:
        rs = [wre(V[i], Wdn[i], im_d[i].cpu().numpy()) for i in range(len(experts))]
        return float(np.mean([r[0] for r in rs])), float(np.mean([r[1] for r in rs]))

    def xw_err(V: torch.Tensor, sel=None) -> float:                        # ||X Wq^T - X W^T|| / ||X W^T||
        tot = []
        for i, e in enumerate(experts):
            x, _ = layer.tokens(e, sel)
            if x.shape[0] == 0:
                continue
            gq, uq = V[i, :I].to(dev), V[i, I:].to(dev)
            g, u, _ = layer.weights(e)
            z, zq = torch.cat([x @ g.t(), x @ u.t()], 1), torch.cat([x @ gq.t(), x @ uq.t()], 1)
            tot.append(float(((zq - z).square().sum() / (z.square().sum() + 1e-30)).sqrt()))
        return float(np.mean(tot))

    def y_err(gu, dn) -> tuple[float, int]:                                # the whole expert's output, held out
        tot = []
        for i, e in enumerate(experts):
            deq = (torch.as_tensor(gu[i, :I]).to(dev).float(), torch.as_tensor(gu[i, I:]).to(dev).float(),
                   torch.as_tensor(dn[i]).to(dev).float())
            r = GQ.expert_err(layer, e, deq, sel_ho)
            if np.isfinite(r):
                tot.append(r)                      # a cold expert with no held-out row would poison the mean
        return (float(np.mean(tot)), len(tot)) if tot else (float("nan"), 0)

    def y_err_each(gu, dn) -> str:
        out = []
        for i, e in enumerate(experts):
            deq = (torch.as_tensor(gu[i, :I]).to(dev).float(), torch.as_tensor(gu[i, I:]).to(dev).float(),
                   torch.as_tensor(dn[i]).to(dev).float())
            r = GQ.expert_err(layer, e, deq, sel_ho)
            if np.isfinite(r):
                out.append(f"{e}:{r:.4f}")
        return "  ".join(out)

    def t2n(V: torch.Tensor) -> np.ndarray:
        return V.cpu().numpy()

    # ---------- the table ----------
    print("=" * 96, flush=True)
    print(f"weight error vs FP8, mean over {len(experts)} experts (wre is the imatrix-weighted metric)", flush=True)
    for tag in ("RTN", "IMX"):
        r = w_err_gu(v[(tag, "gu")])
        print(f"  gate+up [{t_gu.name}] {tag}: wre {r[0]:.5f}  rel {r[1]:.5f}", flush=True)
    for tag in ("RTN", "IMX"):
        r = w_err_dn(v[(tag, "dn")])
        print(f"  down    [{t_dn.name}] {tag}: wre {r[0]:.5f}  rel {r[1]:.5f}", flush=True)
    print(f"  gate+up output error on the expert's ffn_in rows: RTN {xw_err(torch.as_tensor(v[('RTN', 'gu')])):.5f}"
          f"   IMX {xw_err(torch.as_tensor(v[('IMX', 'gu')])):.5f}", flush=True)
    print(f"  ... the same on the HELD-OUT rows only:           RTN "
          f"{xw_err(torch.as_tensor(v[('RTN', 'gu')]), sel_ho):.5f}"
          f"   IMX {xw_err(torch.as_tensor(v[('IMX', 'gu')]), sel_ho):.5f}", flush=True)
    yr, kn = y_err(v[("RTN", "gu")], v[("RTN", "dn")])
    yi, _ = y_err(v[("IMX", "gu")], v[("IMX", "dn")])
    print(f"  expert output error, held-out tokens ({kn} experts have any): RTN {yr:.5f}   IMX {yi:.5f}", flush=True)
    print(f"    per expert RTN: {y_err_each(v[('RTN', 'gu')], v[('RTN', 'dn')])}", flush=True)
    print(f"    per expert IMX: {y_err_each(v[('IMX', 'gu')], v[('IMX', 'dn')])}", flush=True)
    wsh = ck.get(f"model.layers.{il}.mlp.shared_experts.gate_proj.weight").numpy()
    im_sh = (moe["sh_gu"] / max(1, moe["sh_n"])).numpy().astype(np.float32)
    for tag, imv in (("RTN", None), ("IMX", im_sh)):
        r = wre(quant_deq(G, int(t_sh), wsh, imv, pool), wsh, im_sh)
        print(f"  shared gate [{t_sh.name}] {tag}: wre {r[0]:.5f}  rel {r[1]:.5f}", flush=True)

    print("-" * 96, flush=True)
    print("GPTQ sweep of gate/up (down = IMX, as in production). The sweep fits only the train rows of ffn_in "
          "and is scored on the held-out ones; quantize.py fits all of them.", flush=True)
    best = None
    for prior in [float(p) for p in str(a.prior).split(",")]:
        for damp in [float(d) for d in str(a.damp).split(",")]:
            t1 = time.time()
            V = gptq_deq(prior, damp)
            r = w_err_gu(t2n(V))
            xw, ye = xw_err(V), y_err(t2n(V), v[("IMX", "dn")])[0]
            xwh = xw_err(V, sel_ho)
            print(f"  prior {prior:7.0f} damp {damp:.3f}: wre {r[0]:.5f} rel {r[1]:.5f} xwerr {xw:.5f} "
                  f"(held-out {xwh:.5f}) yerr {ye:.5f}  ({time.time() - t1:.0f} s)", flush=True)
            if best is None or ye < best[0]:
                best = (ye, prior, damp)
    rtn_y, kn = y_err(v[("RTN", "gu")], v[("RTN", "dn")])
    imx_y, _ = y_err(v[("IMX", "gu")], v[("IMX", "dn")])
    print(f"best held-out expert output error: prior {best[1]:.0f} damp {best[2]:.3f} -> {best[0]:.5f} "
          f"(RTN {rtn_y:.5f}, IMX {imx_y:.5f}, {kn} experts with held-out rows)", flush=True)
    print(f"per expert, best setting: {y_err_each(t2n(gptq_deq(best[1], best[2])), v[('IMX', 'dn')])}", flush=True)
    print(f"probe done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
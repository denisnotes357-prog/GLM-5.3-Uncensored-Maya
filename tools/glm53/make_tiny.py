#!/usr/bin/env python3
"""tools/glm53/make_tiny.py - a small random-weight GLM-5.3 (GlmMoeDsa) + its reference outputs.

The model keeps GLM-5.3's real shape where the port cares about it (MLA head dims, indexer dims, RoPE)
and shrinks everything else.  The HF export is bf16 on disk; the reference is computed by RELOADING that
bf16 file into float32, so the reference matches the bytes llama.cpp will read.

    python make_tiny.py [--out /data/glm53/tiny]

Writes: hf/ (save_pretrained + tokenizer), ref/tokens.npy, ref/logits.npy, ref/hidden_*.npy, ref/ppl.txt,
        ref/sparse_check.json.  CPU only, --threads defaults to 8 (a quantization job owns the box).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import numpy as np
import torch

SRC = "/data/models/GLM-5.3-FP8"
N_TOK = 96
TOK_SEED = 7
MODEL_SEED = 1234

# the tiny geometry; everything not listed keeps the original GLM-5.3 value
SHRINK = {
    "num_hidden_layers": 6,
    "hidden_size": 512,
    "intermediate_size": 1024,
    "moe_intermediate_size": 512,
    "n_routed_experts": 16,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "first_k_dense_replace": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "q_lora_rank": 512,
    "index_topk": 16,
    "num_nextn_predict_layers": 1,
    "max_position_embeddings": 4096,
}
INDEXER_TYPES = ["full", "full", "shared", "shared", "full", "shared"]
MLP_LAYER_TYPES = ["dense"] + ["sparse"] * 5

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                   "special_tokens_map.json", "generation_config.json")


def build_config():
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(SRC)
    # "убрать" = нет ключа в сохранённом config.json (иначе converter увидит null)
    cfg.__dict__.pop("quantization_config", None)
    cfg.__dict__.pop("hf_quant_config", None)
    for k, v in SHRINK.items():
        setattr(cfg, k, v)
    # index_topk_pattern/frequency/offset are only read by __post_init__ when indexer_types is None;
    # setting the field explicitly pins it (and __post_init__ never recomputes it)
    cfg.indexer_types = list(INDEXER_TYPES)
    cfg.mlp_layer_types = list(MLP_LAYER_TYPES)
    cfg.layer_types = ["indexed_attention"] * len(INDEXER_TYPES)
    # stale schedule keys would contradict the pinned lists in a later reload
    for k in ("index_topk_freq", "index_skip_topk_offset", "index_topk_pattern"):
        if hasattr(cfg, k):
            setattr(cfg, k, None)
    return cfg


def save_model_bf16(model, hf_dir: pathlib.Path) -> dict:
    info = {}
    model = model.to(torch.bfloat16)
    model.save_pretrained(hf_dir, safe_serialization=True)
    for f in sorted(hf_dir.glob("*.safetensors")):
        info[f.name] = f.stat().st_size
    return info


def apply_norms_and_bias(model) -> dict:
    """RMSNorm weights away from 1.0 (so norm bugs are visible) + a non-zero noaux_tc bias."""
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRMSNorm, GlmMoeDsaTopkRouter

    gen = torch.Generator().manual_seed(MODEL_SEED + 1)
    n_norm, n_router = 0, 0
    for m in model.modules():
        if isinstance(m, GlmMoeDsaRMSNorm):
            with torch.no_grad():
                m.weight.copy_(1.0 + 0.1 * torch.randn(m.weight.shape, generator=gen))
            n_norm += 1
        elif isinstance(m, GlmMoeDsaTopkRouter):
            with torch.no_grad():
                # default _init_weights zeroes it (modeling_glm_moe_dsa.py:646) -> noaux_tc would be inert
                m.e_score_correction_bias.copy_(0.05 * torch.randn(m.e_score_correction_bias.shape, generator=gen))
            n_router += 1
    return {"rmsnorm_touched": n_norm, "router_biases_touched": n_router}


def make_tokens():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TINY_HF))
    gmask = tok.convert_tokens_to_ids("[gMASK]")
    if not isinstance(gmask, int) or gmask < 0:
        raise SystemExit(f"tokenizer has no [gMASK]: {gmask!r}")
    g = torch.Generator().manual_seed(TOK_SEED)
    ids = torch.randint(0, tok.vocab_size, (N_TOK,), generator=g, dtype=torch.int64)
    ids[0] = int(gmask)
    return tok, ids.to(torch.int32), int(gmask)


def stats(x: np.ndarray) -> dict:
    x = x.astype(np.float64)
    return {"std": float(x.std()), "min": float(x.min()), "max": float(x.max()),
            "nan": bool(np.isnan(x).any())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/glm53/tiny")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    global TINY_HF, TINY_REF
    out = pathlib.Path(a.out)
    TINY_HF = out / "hf"
    TINY_REF = out / "ref"
    TINY_HF.mkdir(parents=True, exist_ok=True)
    TINY_REF.mkdir(parents=True, exist_ok=True)

    report: dict = {"threads": a.threads}
    cfg = build_config()
    key = {k: getattr(cfg, k, None) for k in
           sorted(set(SHRINK) | {"kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
                                  "index_n_heads", "index_head_dim", "vocab_size", "rope_parameters",
                                  "rms_norm_eps", "routed_scaling_factor", "indexer_types",
                                  "mlp_layer_types", "layer_types", "head_dim", "qk_head_dim",
                                  "index_topk_freq", "index_skip_topk_offset", "index_topk_pattern",
                                  "indexer_types"})}
    print("CONFIG " + json.dumps(key, ensure_ascii=False), flush=True)
    assert cfg.indexer_types == INDEXER_TYPES, cfg.indexer_types
    assert cfg.mlp_layer_types == MLP_LAYER_TYPES, cfg.mlp_layer_types

    from transformers import GlmMoeDsaForCausalLM

    torch.manual_seed(MODEL_SEED)
    with torch.no_grad():
        model = GlmMoeDsaForCausalLM(cfg)
    model.eval()
    report["params"] = int(sum(p.numel() for p in model.parameters()))
    report["init"] = apply_norms_and_bias(model)
    print(f"MODEL params={report['params']} init={report['init']}", flush=True)

    report["safetensors"] = save_model_bf16(model, TINY_HF)
    del model
    for f in TOKENIZER_FILES:
        s = pathlib.Path(SRC) / f
        if s.exists():
            shutil.copy2(s, TINY_HF / f)
    print("COPIED " + " ".join(sorted(p.name for p in TINY_HF.iterdir() if p.is_file())), flush=True)

    # ---- reference: reload the bf16 bytes we just wrote, run once in float32
    ref_model = GlmMoeDsaForCausalLM.from_pretrained(
        str(TINY_HF), dtype=torch.float32, attn_implementation="eager",
        use_safetensors=True)
    ref_model.eval()
    report["reload_dtypes"] = sorted({str(p.dtype) for p in ref_model.parameters()})
    tok, ids, gmask = make_tokens()
    report["tokens"] = {"n": int(ids.numel()), "first": int(ids[0]), "gMASK": gmask,
                        "min": int(ids.min()), "max": int(ids.max())}
    print(f"TOKENS n={report['tokens']['n']} [0]={ids[0]} gMASK={gmask} "
          f"range=[{int(ids.min())}..{int(ids.max())}] vocab={tok.vocab_size}", flush=True)

    def run(model, idx):
        with torch.no_grad():
            o = model(input_ids=idx.view(1, -1), use_cache=False, output_hidden_states=True,
                      return_dict=True)
        return o

    o1 = run(ref_model, ids)
    logits = o1.logits[0].float().numpy().astype(np.float32)
    np.save(TINY_REF / "tokens.npy", ids.numpy().astype(np.int32))
    np.save(TINY_REF / "logits.npy", logits)
    for i, h in enumerate(o1.hidden_states):
        np.save(TINY_REF / f"hidden_{i}.npy", h[0].float().numpy().astype(np.float32))
    report["hidden_n"] = len(o1.hidden_states)
    report["logits"] = stats(logits)
    print(f"LOGITS {json.dumps(report['logits'])} hidden_states={len(o1.hidden_states)} "
          f"hidden_0={list(o1.hidden_states[0].shape)}", flush=True)

    # ---- perplexity over the 96 tokens (predict t+1 from t)
    lp = torch.log_softmax(torch.from_numpy(logits[:-1]), dim=-1)
    tgt = ids[1:].long()
    nll = -lp[torch.arange(tgt.numel()), tgt]
    ppl = float(torch.exp(nll.mean()))
    (TINY_REF / "ppl.txt").write_text(
        f"tokens: {N_TOK} (seed {TOK_SEED}), predictions: {tgt.numel()}\n"
        f"mean_nll: {float(nll.mean()):.6f}\nppl: {ppl:.6f}\n")
    report["ppl"] = ppl
    report["mean_nll"] = float(nll.mean())
    print(f"PPL {ppl:.6f} mean_nll {float(nll.mean()):.6f} over {tgt.numel()} predictions", flush=True)

    # ---- the sparse indexer must actually change something: 16 vs whole context
    idx_mods = [m for m in ref_model.modules()
                if m.__class__.__name__ == "GlmMoeDsaIndexer" and m is not None]
    saved = [m.index_topk for m in idx_mods]
    for m in idx_mods:
        m.index_topk = 4096
    with torch.no_grad():
        o_full = run(ref_model, ids)
    for m, s in zip(idx_mods, saved):
        m.index_topk = s
    full = o_full.logits[0].float().numpy().astype(np.float32)
    d = np.abs(full - logits)
    sparse = {
        "indexers_patched": len(idx_mods),
        "topk_used": 16,
        "control_topk": int(min(4096, N_TOK)),
        "max_abs_diff_positions_0_15": float(d[:16].max()),
        "max_abs_diff_positions_16_95": float(d[16:].max()),
        "mean_abs_diff_positions_16_95": float(d[16:].mean()),
        "max_abs_diff_position_16": float(d[16].max()),
        "argmax_changed_rows_beyond_16": int((logits[16:].argmax(-1) != full[16:].argmax(-1)).sum()),
        "rows_compared_beyond_16": int(N_TOK - 16),
    }
    (TINY_REF / "sparse_check.json").write_text(json.dumps(sparse, indent=2) + "\n")
    print("SPARSE " + json.dumps(sparse), flush=True)

    # ---- sanity: finite, non-degenerate logits
    ok = (not report["logits"]["nan"]) and report["logits"]["std"] > 0.1
    report["logits_ok"] = bool(ok)
    report["sparse"] = sparse
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"REPORT {out / 'report.json'} logits_ok={ok}", flush=True)
    if not ok:
        raise SystemExit("logits are degenerate: " + json.dumps(report["logits"]))


TINY_HF = pathlib.Path("/data/glm53/tiny/hf")
TINY_REF = pathlib.Path("/data/glm53/tiny/ref")

if __name__ == "__main__":
    main()
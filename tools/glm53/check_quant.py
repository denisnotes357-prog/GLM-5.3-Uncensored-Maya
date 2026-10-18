"""tools/glm53/check_quant.py - a GLM-5.3 GGUF's tensors against the FP8 checkpoint they were quantised from.

Each checked GGUF tensor is dequantised with gguf-py and compared with the block-FP8 weight dequantised straight from
the safetensors (e4m3 x weight_scale_inv per 128x128 block): the relative error ||q - ref|| / ||ref|| and the cosine.
A routed-expert tensor is checked on a few experts (its slices), not whole. Each quant type has a ceiling for the
relative error; a tensor above it, a wrong shape or a missing tensor fails the run (exit 1).

    python check_quant.py GGUF [--fp8 DIR] [--experts 0,17,128,255]

attn_k_b / attn_v_b (kv_b split and transposed by the converter) are not compared here; the end-to-end run checks them.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, "/data/glm53/llama.cpp/gguf-py")
import gguf  # noqa: E402

# a ceiling for ||q - ref|| / ||ref|| per quant type, about 1.2x what each gives on GLM-5.3's weights without an imatrix
# (layers 0-3, 2026-10-15: Q8_0 0.006-0.008, Q6_K 0.017-0.024, Q5_K 0.042, IQ3_S 0.169-0.171 - near the 0.19 of an
# optimal 3-bit scalar quantiser on Gaussian weights); Q4_K / IQ4_XS from their bits per weight
CEIL = {"F32": 1e-6, "F16": 1e-3, "BF16": 1e-2, "Q8_0": 0.01, "Q6_K": 0.03, "Q5_K": 0.055, "Q4_K": 0.11,
        "IQ4_XS": 0.12, "IQ3_S": 0.205}

DIRECT = [  # GGUF name pattern -> checkpoint name ({0} the layer)
    (r"blk\.(\d+)\.attn_q_a\.weight", "model.layers.{0}.self_attn.q_a_proj.weight"),
    (r"blk\.(\d+)\.attn_q_b\.weight", "model.layers.{0}.self_attn.q_b_proj.weight"),
    (r"blk\.(\d+)\.attn_kv_a_mqa\.weight", "model.layers.{0}.self_attn.kv_a_proj_with_mqa.weight"),
    (r"blk\.(\d+)\.attn_output\.weight", "model.layers.{0}.self_attn.o_proj.weight"),
    (r"blk\.(\d+)\.attn_q_a_norm\.weight", "model.layers.{0}.self_attn.q_a_layernorm.weight"),
    (r"blk\.(\d+)\.attn_kv_a_norm\.weight", "model.layers.{0}.self_attn.kv_a_layernorm.weight"),
    (r"blk\.(\d+)\.attn_norm\.weight", "model.layers.{0}.input_layernorm.weight"),
    (r"blk\.(\d+)\.ffn_norm\.weight", "model.layers.{0}.post_attention_layernorm.weight"),
    (r"blk\.(\d+)\.ffn_(gate|up|down)\.weight", "model.layers.{0}.mlp.{1}_proj.weight"),
    (r"blk\.(\d+)\.ffn_(gate|up|down)_shexp\.weight", "model.layers.{0}.mlp.shared_experts.{1}_proj.weight"),
    (r"blk\.(\d+)\.ffn_gate_inp\.weight", "model.layers.{0}.mlp.gate.weight"),
    (r"blk\.(\d+)\.exp_probs_b\.bias", "model.layers.{0}.mlp.gate.e_score_correction_bias"),
    (r"blk\.(\d+)\.indexer\.attn_k\.weight", "model.layers.{0}.self_attn.indexer.wk.weight"),
    (r"blk\.(\d+)\.indexer\.attn_q_b\.weight", "model.layers.{0}.self_attn.indexer.wq_b.weight"),
    (r"blk\.(\d+)\.indexer\.proj\.weight", "model.layers.{0}.self_attn.indexer.weights_proj.weight"),
    (r"blk\.(\d+)\.indexer\.k_norm\.weight", "model.layers.{0}.self_attn.indexer.k_norm.weight"),
    (r"blk\.(\d+)\.indexer\.k_norm\.bias", "model.layers.{0}.self_attn.indexer.k_norm.bias"),
    (r"blk\.(\d+)\.nextn\.eh_proj\.weight", "model.layers.{0}.eh_proj.weight"),
    (r"token_embd\.weight", "model.embed_tokens.weight"),
    (r"output\.weight", "lm_head.weight"),
    (r"output_norm\.weight", "model.norm.weight"),
]
EXPERTS = r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight"


class Fp8:
    def __init__(self, root: str):
        self.root = pathlib.Path(root)
        self.map = json.load(open(self.root / "model.safetensors.index.json"))["weight_map"]
        self.files: dict[str, object] = {}

    def raw(self, name: str) -> torch.Tensor:
        shard = self.map[name]
        if shard not in self.files:
            self.files[shard] = safe_open(str(self.root / shard), framework="pt")
        return self.files[shard].get_tensor(name)

    def get(self, name: str) -> np.ndarray:
        w = self.raw(name)
        s = name[: -len(".weight")] + ".weight_scale_inv" if name.endswith(".weight") else None
        if s is not None and s in self.map:
            sc = self.raw(s).float()
            w = w.float() * sc.repeat_interleave(128, 0)[: w.shape[0]].repeat_interleave(128, 1)[:, : w.shape[1]]
        return w.float().numpy()


def compare(q: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    q, ref = q.astype(np.float64).ravel(), ref.astype(np.float64).ravel()
    nr = np.linalg.norm(ref)
    rel = float(np.linalg.norm(q - ref) / nr) if nr else float(np.linalg.norm(q))
    cos = float(q @ ref / (np.linalg.norm(q) * nr)) if nr and np.linalg.norm(q) else 1.0
    return rel, cos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf")
    ap.add_argument("--fp8", default="/data/models/GLM-5.3-FP8")
    ap.add_argument("--experts", default="0,17,128,255")
    a = ap.parse_args()
    ck = Fp8(a.fp8)
    experts = [int(e) for e in a.experts.split(",")]
    paths = sorted(pathlib.Path(a.gguf).parent.glob(re.sub(r"-00001-of-", "-*-of-", pathlib.Path(a.gguf).name)))
    bad, rows = 0, []
    for p in paths:
        for t in gguf.GGUFReader(str(p)).tensors:
            qt = gguf.GGMLQuantizationType(t.tensor_type).name
            if (m := re.fullmatch(EXPERTS, t.name)) is not None:
                for e in experts:
                    ref = ck.get(f"model.layers.{m[1]}.mlp.experts.{e}.{m[2]}_proj.weight")
                    q = gguf.quants.dequantize(np.ascontiguousarray(t.data[e]), t.tensor_type).reshape(ref.shape)
                    rows.append((f"{t.name}[{e}]", qt, *compare(q, ref)))
                continue
            for rx, hf in DIRECT:
                if (m := re.fullmatch(rx, t.name)) is not None:
                    ref = ck.get(hf.format(*m.groups()))
                    q = gguf.quants.dequantize(np.asarray(t.data), t.tensor_type)
                    if q.size != ref.size:
                        rows.append((t.name, qt, float("inf"), 0.0))
                    else:
                        rows.append((t.name, qt, *compare(q.reshape(ref.shape), ref)))
                    break
            else:
                rows.append((t.name, qt, None, None))
    for name, qt, rel, cos in rows:
        if rel is None:
            print(f"  skip  {name:44s} {qt:7s}")
            continue
        ok = rel <= CEIL.get(qt, 0.2)
        bad += not ok
        print(f"{'  ok ' if ok else ' FAIL'}  {name:44s} {qt:7s} rel {rel:.5f}  cos {cos:.6f}")
    checked = sum(r[2] is not None for r in rows)
    print(f"{checked} checked, {len(rows) - checked} skipped, {bad} over the ceiling")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

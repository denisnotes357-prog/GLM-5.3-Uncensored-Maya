"""tools/glm53/calib_check.py - the independent yardstick for calib.py: transformers' own GLM-5.3 layers.

Runs the first --n sequences of calib.npy through layers 0..--layers with the STOCK `GlmMoeDsaDecoderLayer`
(real DSA indexer with its top-k propagated, real GlmMoeDsaExperts, ordinary residual, MLA + RoPE) and compares
the hidden states after the last of them with the ones calib.py left in its stream buffer:

    python calib_check.py --fp8 DIR --calib calib.npy --streams DIR --buf a --layers 3 --n 4

The compared quantities are the post-residual hidden states (calib.py writes them, calib_check.py ends on them).
Both sides read the same checkpoint and share only two deliberate choices that are not byte-identical to
`from_pretrained` in bf16: the router weight/bias stay fp32 (the recipe quantises the router F32) and the MoE
output is accumulated in fp32 instead of rounding each expert's addition to bf16.  Everything structural - norms,
MLA + RoPE, indexer and its top-k, FFN, residuals - is transformers' own code on both sides, so a broken split
in calib.py shows up as a correlation below the threshold.

Prints one line per sequence and `PASS min-correlation X` / `FAIL ...` (threshold 0.999).
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False

P = "model."
DT = torch.bfloat16


class Checkpoint:
    def __init__(self, root: str):
        self.root = pathlib.Path(root)
        cfg = json.load(open(self.root / "config.json"))
        self.cfg = cfg.get("text_config", cfg)
        self.map = json.load(open(self.root / "model.safetensors.index.json"))["weight_map"]
        self._open: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self.map

    def raw(self, name: str):
        shard = self.map[name]
        f = self._open.get(shard)
        if f is None:
            from safetensors import safe_open
            f = safe_open(str(self.root / shard), framework="pt")
            self._open[shard] = f
        return f.get_tensor(name)

    def get(self, name: str, dev: torch.device | None = None, dtype: torch.dtype = DT) -> torch.Tensor:
        t = self.raw(name)
        if dev is not None:
            t = t.to(dev)
        if t.dtype == torch.float8_e4m3fn:
            w = t.float()
            s = self.raw(name + "_scale_inv")
            s = s.to(dev).float() if dev is not None else s.float()
            s = s.repeat_interleave(128, 0)[: w.shape[0]].repeat_interleave(128, 1)[:, : w.shape[1]]
            return (w * s).to(dtype)
        return t.to(dtype)

    @property
    def n_layers(self) -> int:
        return int(self.cfg["num_hidden_layers"])

    def is_moe(self, il: int) -> bool:
        return il >= int(self.cfg.get("first_k_dense_replace", 3))


def build_layer(ck: Checkpoint, cfg, il: int, dev):
    """The stock decoder layer (mlp included) with the checkpoint's weights."""
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer
    prev_dtype = torch.get_default_dtype()          # the stock module is a bf16 one: build it that way, so
    torch.set_default_dtype(DT)                     #   to_empty does not stage 39 GB of fp32 experts first
    try:
        with torch.device("meta"):
            layer = GlmMoeDsaDecoderLayer(cfg, il)
    finally:
        torch.set_default_dtype(prev_dtype)
    layer = layer.to_empty(device=dev)
    if getattr(layer.self_attn, "indexer", None) is not None:
        layer.self_attn.indexer.weights_proj.to(torch.float32)
    pref = P + f"layers.{il}."
    L = pref + "mlp."
    with torch.no_grad():
        if ck.is_moe(il):
            E, I = int(cfg.n_routed_experts), int(cfg.moe_intermediate_size)
            experts = layer.mlp.experts
            for e in range(E):
                experts.gate_up_proj[e, :I].copy_(ck.get(f"{L}experts.{e}.gate_proj.weight", dev=dev))
                experts.gate_up_proj[e, I:].copy_(ck.get(f"{L}experts.{e}.up_proj.weight", dev=dev))
                experts.down_proj[e].copy_(ck.get(f"{L}experts.{e}.down_proj.weight", dev=dev))
            # router fp32, as calib.py's StatMoE and the F32 recipe take it
            layer.mlp.gate.weight.data = ck.get(L + "gate.weight", dev=dev, dtype=torch.float32)
            layer.mlp.gate.e_score_correction_bias.data = ck.get(
                L + "gate.e_score_correction_bias", dev=dev, dtype=torch.float32)
        for name, p in layer.named_parameters():
            if ck.is_moe(il) and (name.startswith("mlp.gate.") or name.startswith("mlp.experts.")):
                continue
            k = pref + name
            if not ck.has(k):
                raise KeyError(f"checkpoint has no {k}")
            p.copy_(ck.get(k, dev=dev, dtype=p.dtype))
        for name, b in layer.named_buffers():
            k = pref + name
            if ck.has(k):
                v = ck.get(k, dev=dev, dtype=DT)
                if name.endswith("e_score_correction_bias"):
                    continue                       # already fp32 above
                b.copy_(v.to(b.dtype))
    layer.eval()
    return layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp8", default="/data/models/GLM-5.3-FP8")
    ap.add_argument("--calib", default="/data/glm53/calib/calib.npy")
    ap.add_argument("--streams", required=True, help="calib.py's --out directory")
    ap.add_argument("--buf", default=None, help="streams_a|b (default: the one layer L+1 wrote)")
    ap.add_argument("--layers", default="3", help="last layer of the compared stack, e.g. 3")
    ap.add_argument("--n", type=int, default=4, help="first N sequences")
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()

    from transformers import AutoConfig
    from transformers.masking_utils import create_causal_mask
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaDecoderLayer,
        GlmMoeDsaRotaryEmbedding,
    )

    dev = torch.device("cuda")
    ck = Checkpoint(a.fp8)
    cfg = AutoConfig.from_pretrained(a.fp8)
    cfg._attn_implementation = "sdpa"
    last = int(a.layers) if "-" not in a.layers else int(a.layers.split("-")[1])
    n = a.n
    seqs = np.load(a.calib)[:n].astype(np.int64)
    T = seqs.shape[1]
    H = int(cfg.hidden_size)
    out_dir = pathlib.Path(a.streams)
    buf = a.buf or ("a" if (last + 1) % 2 == 0 else "b")
    st_path = out_dir / f"streams_{buf}.bin"
    rows = st_path.stat().st_size // (T * H * 2)
    ref = np.memmap(st_path, dtype=np.uint16, mode="r", shape=(rows, T, H))
    pos = torch.arange(T, device=dev).unsqueeze(0)
    rope_mod = GlmMoeDsaRotaryEmbedding(cfg).to(dev)
    layers, topk = [], None
    for il in range(0, last + 1):
        layers.append(build_layer(ck, cfg, il, dev))
    emb = ck.get(P + "embed_tokens.weight", dtype=DT)
    worst = 1.0
    with torch.no_grad():
        for b0 in range(0, n, a.batch):
            ids = list(range(b0, min(n, b0 + a.batch)))
            bs = len(ids)
            x = torch.stack([emb[torch.from_numpy(seqs[s])] for s in ids]).to(dev)
            rope = rope_mod(x, position_ids=pos)
            mask = create_causal_mask(config=cfg, inputs_embeds=x, attention_mask=None, past_key_values=None,
                                      position_ids=pos, allow_is_causal_skip=False)
            topk = None
            for il, layer in enumerate(layers):
                x, topk = layer(x, attention_mask=mask, position_ids=pos, past_key_values=None, use_cache=False,
                                position_embeddings=rope, prev_topk_indices=topk)
            for i, s in enumerate(ids):
                r = np.asarray(torch.from_numpy(np.asarray(ref[s])).view(DT).float().reshape(-1))
                r = r - r.mean()
                m = x[i].float().reshape(-1).cpu().numpy() - float(x[i].float().mean())
                rho = float(np.dot(r, m) / (np.linalg.norm(r) * np.linalg.norm(m) + 1e-30))
                d = float(np.abs(x[i].float().cpu().numpy() - np.asarray(
                    torch.from_numpy(np.asarray(ref[s])).view(DT).float().numpy())).max())
                worst = min(worst, rho)
                print(f"seq {s:3d}: pearson {rho:.6f}  max|d| {d:.4g}", flush=True)
    print(f"{'PASS' if worst > 0.999 else 'FAIL'} min-correlation {worst:.6f} "
          f"(stack 0-{last}, buffer streams_{buf}.bin)", flush=True)


if __name__ == "__main__":
    main()
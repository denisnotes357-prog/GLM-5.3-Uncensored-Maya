"""tools/glm53/calib.py - the reference and statistics pass for the full GLM-5.3 (glm_moe_dsa), the stage the
GPTQ quants are built on (tools/maya_quant/calib.py does the same for GLM-5.3-Flash; this is the port, with the
format of stats_XX.pt / ffn_in_XX.bin / ref_topk.npz kept identical).

The official block-FP8 checkpoint (/data/models/GLM-5.3-FP8, 78 trunk layers) runs layer by layer on ONE GPU: each
layer is transformers' own `GlmMoeDsaDecoderLayer` (ordinary residual, MLA + RoPE, DSA indexer), with the routed
experts replaced by a module that also records, per expert:
    n          tokens routed to it
    imat_gu    sum over its tokens of x^2 per input channel (the imatrix of gate/up: the diagonal of X^T X)
    imat_d     the same for the down projection's input h = swiglu(gate, up)
    sal        sum of routing weight * ||expert output||  (REAP saliency)
while the shared expert and every dense linear's imatrix (and the router input's) are recorded too.  The hidden
streams of every sequence live in two memmaps (bf16) on disk between layers, so the pass resumes at the first
layer whose stats file is missing.  After the last layer the eval sequences' logits give the reference: top-64
log-probs and the logsumexp at every position (ref_topk.npz) - the yardstick the quants are scored against.  Each
MoE layer's input over the first --ffn-in-seqs calibration sequences is kept too (ffn_in_XX.bin, bf16
[n, T, hidden]): routing and every expert's X^T X are recomputed from it by the error-feedback rounding in gptq.

GLM-5.3 differences this pass encodes (docs/GLM53-ENGINE-PORT.md):
  * 78 trunk layers 0..77, 3 dense (0-2), 75 MoE layers: 256 experts, top-8, sigmoid router + noaux_tc bias,
    norm_topk_prob, routed_scaling_factor 2.5, SwiGLU WITHOUT a clamp, 1 shared expert (moe_intermediate_size).
    The NextN block `layers.78` is not part of this pass.  No mHC: streams are [n_seq, T, hidden].
  * The DSA indexer selects min(index_topk, T) tokens per query and the sequences are T = 2048 with
    index_topk = 2048, so its top-2048 covers EVERY position and the attention is dense and causal anyway.
    The indexer is still computed (its projections get their imatrix), but its top-k is never propagated: every
    layer gets prev_topk_indices = arange(T), i.e. the same token set, which also keeps resume independent of
    the skipped layers.  --skip-indexer drops that computation (with it the indexer's imatrix).
  * FP8 weights are 128 x 128 block-scaled: dequantised in fp32 with their *_scale_inv on the GPU and stored in
    bf16 (config dtype), one expert tensor at a time.

python calib.py [--fp8 DIR] [--calib calib.npy] [--eval eval.npy] [--out DIR] [--layers 0-44] [--batch 4]
                [--ffn-in-seqs 32] [--n-cal N] [--n-eval N] [--skip-indexer]

calib.npy / eval.npy: int32 [n_seq, T] token ids (tools/glm53 prep, CALIB.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False

P = "model."
DT = torch.bfloat16   # config dtype


class Checkpoint:
    """The block-FP8 GLM-5.3 checkpoint, tensor by tensor; `get` dequantises on CPU or GPU."""

    def __init__(self, root: str):
        self.root = pathlib.Path(root)
        cfg = json.load(open(self.root / "config.json"))
        self.cfg = cfg.get("text_config", cfg)
        self.map = json.load(open(self.root / "model.safetensors.index.json"))["weight_map"]
        self._open: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self.map

    def raw(self, name: str) -> torch.Tensor:
        shard = self.map[name]
        f = self._open.get(shard)
        if f is None:
            f = safe_open(str(self.root / shard), framework="pt")
            self._open[shard] = f
        return f.get_tensor(name)

    def get(self, name: str, dev: torch.device | None = None, dtype: torch.dtype = DT) -> torch.Tensor:
        """The tensor in `dtype`; block-FP8 weights are multiplied by their 128 x 128 block scales (in fp32)."""
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


def load_layer(ck: Checkpoint, cfg, il: int, dev) -> torch.nn.Module:
    """One decoder layer's own weights on the GPU (bf16); the routed experts are replaced by Identity."""
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer
    with torch.device("meta"):
        layer = GlmMoeDsaDecoderLayer(cfg, il)
    moe = ck.is_moe(il)
    if moe:
        layer.mlp = torch.nn.Identity()   # replaced by StatMoE (the stock experts module would take 19 GB bf16
    layer = layer.to_empty(device=dev)    #   in another layout, and would not record anything)
    layer = layer.to(DT)
    if getattr(layer.self_attn, "indexer", None) is not None:
        layer.self_attn.indexer.weights_proj.to(torch.float32)   # transformers keeps it fp32 (and the recipe does)
    pref = P + f"layers.{il}."
    with torch.no_grad():
        for name, p in layer.named_parameters():
            k = pref + name
            if not ck.has(k):
                raise KeyError(f"checkpoint has no {k}")
            p.copy_(ck.get(k, dev=dev, dtype=p.dtype))
        for name, b in layer.named_buffers():
            k = pref + name
            if ck.has(k):
                v = ck.get(k, dev=dev, dtype=DT)
                if name.endswith("e_score_correction_bias"):
                    b.data = v.float()
                else:
                    b.copy_(v.to(b.dtype))
    layer.eval()
    return layer


class Stat:
    """Per-input-channel sums of squares and a token count (an imatrix entry)."""

    def __init__(self, n_in: int, dev):
        self.ss = torch.zeros(n_in, dtype=torch.float64, device=dev)
        self.n = 0

    def add(self, x: torch.Tensor):
        self.ss += x.double().square().sum(0)
        self.n += x.shape[0]


class StatMoE(torch.nn.Module):
    """GLM-5.3's GlmMoeDsaMoE (sigmoid + noaux_tc, norm, x2.5, no SwiGLU clamp) recording its statistics."""

    def __init__(self, ck: Checkpoint, cfg, il: int, dev):
        super().__init__()
        L = P + f"layers.{il}.mlp."
        self.E = int(cfg.n_routed_experts)
        self.k = int(cfg.num_experts_per_tok)
        self.scale = float(cfg.routed_scaling_factor)
        self.norm = bool(cfg.norm_topk_prob)
        self.ng, self.tg = int(cfg.n_group), int(cfg.topk_group)
        H, I = int(cfg.hidden_size), int(cfg.moe_intermediate_size)
        self.H, self.I = H, I
        # the router stays fp32 (config moe_router_dtype: float32), like the recipe's F32 router
        self.router = ck.get(L + "gate.weight", dev=dev, dtype=torch.float32)
        self.bias = ck.get(L + "gate.e_score_correction_bias", dev=dev, dtype=torch.float32)
        # the routed experts as bf16 on the device: 256 x 3 x 2048 x 6144 x 2 B = 19.4 GB
        self.wgu = torch.empty(self.E, 2 * I, H, dtype=DT, device=dev)
        self.wd = torch.empty(self.E, H, I, dtype=DT, device=dev)
        for e in range(self.E):
            g = ck.get(f"{L}experts.{e}.gate_proj.weight", dev=dev)
            u = ck.get(f"{L}experts.{e}.up_proj.weight", dev=dev)
            self.wgu[e, :I].copy_(g)
            self.wgu[e, I:].copy_(u)
            self.wd[e].copy_(ck.get(f"{L}experts.{e}.down_proj.weight", dev=dev))
        S = L + "shared_experts."     # n_shared_experts = 1 -> moe_intermediate_size rows
        self.sg = ck.get(S + "gate_proj.weight", dev=dev)
        self.su = ck.get(S + "up_proj.weight", dev=dev)
        self.sd = ck.get(S + "down_proj.weight", dev=dev)
        self.collect = True
        self.ffn = None                # (ffn_in memmap, global sequence ids per row, or -1)
        self.n = torch.zeros(self.E, dtype=torch.float64, device=dev)
        self.sal = torch.zeros(self.E, dtype=torch.float64, device=dev)
        self.imat_gu = torch.zeros(self.E, H, dtype=torch.float64, device=dev)
        self.imat_d = torch.zeros(self.E, I, dtype=torch.float64, device=dev)
        self.sh_gu = torch.zeros(H, dtype=torch.float64, device=dev)
        self.sh_d = torch.zeros(I, dtype=torch.float64, device=dev)
        self.sh_n = 0
        self.router_in = torch.zeros(H, dtype=torch.float64, device=dev)

    @staticmethod
    def _swiglu(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return F.silu(g) * u        # GLM-5.3: no clamp (config has no swiglu_limit)

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        shape = hs.shape
        xb = hs.reshape(-1, shape[-1]).contiguous()      # bf16, as transformers feeds its experts
        if self.ffn is not None:
            mm, rows = self.ffn
            xnp = xb.view(torch.uint16).cpu().numpy()
            for i, s in enumerate(rows):
                if s >= 0:
                    mm[s] = xnp[i]
        xf = xb.float()
        logits = F.linear(xf, self.router)
        scores = logits.sigmoid()
        sc = scores + self.bias
        if self.ng > 1:                                   # GlmMoeDsaTopkRouter's group mask (1 group: identity)
            gs = sc.view(-1, self.ng, self.E // self.ng).topk(min(2, self.E // self.ng), -1)[0].sum(-1)
            gidx = torch.topk(gs, self.tg, -1, sorted=False)[1]
            gm = torch.zeros_like(gs)
            gm.scatter_(1, gidx, 1)
            keep = gm.unsqueeze(-1).expand(-1, self.ng, self.E // self.ng).reshape(-1, self.E)
            sc = sc.masked_fill(~keep.bool(), float("-inf"))
        idx = torch.topk(sc, self.k, dim=-1, sorted=False)[1]
        w = scores.gather(1, idx)
        if self.norm:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        w = w * self.scale
        flat = idx.reshape(-1)
        order = torch.argsort(flat)
        counts = torch.bincount(flat, minlength=self.E).tolist()
        tok = order // self.k
        wk = w.reshape(-1)[order]
        out = torch.zeros_like(xf)                       # fp32 accumulation (transformers rounds each add to bf16)
        pos = 0
        for e in range(self.E):
            c = counts[e]
            if c == 0:
                continue
            t = tok[pos:pos + c]
            we = wk[pos:pos + c]
            pos += c
            xe = xb[t]
            gu = (xe @ self.wgu[e].t()).float()
            h = self._swiglu(gu[:, : self.I], gu[:, self.I:])
            y = (h.to(DT) @ self.wd[e].t()).float()
            if self.collect:
                self.n[e] += c
                self.imat_gu[e] += xe.double().square().sum(0)
                self.imat_d[e] += h.double().square().sum(0)
                self.sal[e] += (we.double() * y.double().norm(dim=1)).sum()
            out.index_add_(0, t, y * we[:, None])
        gsu, gu_s = xb @ self.sg.t(), xb @ self.su.t()    # the shared expert (same act, no clamp)
        hs_ = self._swiglu(gsu, gu_s)
        ys = (hs_.to(DT) @ self.sd.t()).float()
        if self.collect:
            self.sh_gu += xf.double().square().sum(0)
            self.sh_d += hs_.double().square().sum(0)
            self.sh_n += xf.shape[0]
            self.router_in += xf.double().square().sum(0)
        out += ys
        return out.reshape(shape).to(hs.dtype)

    def stats(self):
        return {"n": self.n.cpu(), "sal": self.sal.cpu(),
                "imat_gu": self.imat_gu.float().cpu(), "imat_d": self.imat_d.float().cpu(),
                "sh_gu": self.sh_gu.float().cpu(), "sh_d": self.sh_d.float().cpu(),
                "sh_n": self.sh_n, "router_in": self.router_in.float().cpu()}


def hook_linear_stats(layer, collect):
    """imatrix sums for every nn.Linear of the layer (attention, indexer, dense FFN) by module name."""
    stats, hooks = {}, []
    for name, m in layer.named_modules():
        if isinstance(m, torch.nn.Linear):
            def h(mod, inp, out, name=name):
                if not collect[0]:
                    return
                x = inp[0].reshape(-1, inp[0].shape[-1])
                s = stats.get(name)
                if s is None:
                    s = stats[name] = Stat(x.shape[-1], x.device)
                s.add(x)
            hooks.append(m.register_forward_hook(h))
    return stats, hooks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp8", default="/data/models/GLM-5.3-FP8")
    ap.add_argument("--calib", default="/data/glm53/calib/calib.npy")
    ap.add_argument("--eval", default="/data/glm53/calib/eval.npy")
    ap.add_argument("--out", default="/data/glm53/calib/stats")
    ap.add_argument("--layers", default=None, help="a-b (default: all, resuming)")
    ap.add_argument("--batch", type=int, default=4, help="sequences per forward (attention is quadratic in it)")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--ffn-in-seqs", type=int, default=32,
                    help="MoE input kept for gptq (bf16 ffn_in_XX.bin), 0 = keep none; 32 x 2048 x 6144 x 2 B"
                         " = 0.8 GB a layer")
    ap.add_argument("--n-cal", type=int, default=None, help="first N calibration sequences (default: all)")
    ap.add_argument("--n-eval", type=int, default=None, help="first N eval sequences (default: all)")
    ap.add_argument("--skip-indexer", action="store_true",
                    help="do not compute the DSA indexer (at T=2048 it selects every token anyway)")
    a = ap.parse_args()
    from transformers import AutoConfig
    from transformers.masking_utils import create_causal_mask
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRotaryEmbedding

    dev = torch.device("cuda")
    ck = Checkpoint(a.fp8)
    cfg = AutoConfig.from_pretrained(a.fp8)
    cfg._attn_implementation = "sdpa"
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calib = np.load(a.calib)
    evals = np.load(a.eval)
    if a.n_cal is not None:
        calib = calib[: a.n_cal]
    if a.n_eval is not None:
        evals = evals[: a.n_eval]
    if len(evals) and calib.shape[1] != evals.shape[1]:
        raise SystemExit("calib and eval sequences must share their length")
    seqs = np.concatenate([calib, evals], 0).astype(np.int64)
    n_cal, n_seq, T = calib.shape[0], seqs.shape[0], seqs.shape[1]
    H = int(cfg.hidden_size)
    n_layers = ck.n_layers
    n_ffn = min(a.ffn_in_seqs, n_cal)
    first, last = 0, n_layers - 1
    if a.layers:
        first, last = (int(v) for v in a.layers.split("-"))
    while first <= last and (out / f"stats_{first:02d}.pt").exists():
        first += 1
    print(f"GLM-5.3 calibration: layers {first}-{last} of 0-{n_layers - 1}, {n_cal} cal + {n_seq - n_cal} eval"
          f" x {T} tokens, batch {a.batch}", flush=True)
    print("  3 dense layers (0-2), 75 MoE layers 256 x top-8 sigmoid + noaux_tc x 2.5, no SwiGLU clamp,"
          " ordinary residual (no mHC), NextN layer 78 skipped", flush=True)
    print("  DSA indexer: index_topk 2048 = T, so its top-2048 selects every token and the attention is dense;"
          " it is computed for its imatrix but never propagated - each layer gets prev_topk_indices = arange(T)"
          + (" (--skip-indexer: not computed)" if a.skip_indexer else ""), flush=True)
    bufs = [out / "streams_a.bin", out / "streams_b.bin"]
    nbytes = n_seq * T * H * 2
    for b in bufs:
        if b.exists() and b.stat().st_size != nbytes:
            raise SystemExit(f"{b} holds {b.stat().st_size} bytes, expected {nbytes} - use an empty --out")
    if first > 0 and not bufs[0].exists():
        raise SystemExit("resuming but the hidden streams are gone - use an empty --out instead")
    mm = [np.memmap(b, dtype=np.uint16, mode="r+" if b.exists() else "w+", shape=(n_seq, T, H)) for b in bufs]
    if first == 0:
        emb = ck.get(P + "embed_tokens.weight", dtype=DT)
        for s in range(n_seq):
            mm[0][s] = emb[torch.from_numpy(seqs[s])].view(torch.uint16).numpy()
        mm[0].flush()
        del emb
    rotary = GlmMoeDsaRotaryEmbedding(cfg).to(dev)
    pos = torch.arange(T, device=dev).unsqueeze(0)
    contexts: dict[int, tuple] = {}

    def context(bs: int):
        """(causal mask, position ids, rope, fixed prev_topk_indices) for a batch of `bs` sequences."""
        if bs not in contexts:
            rope = rotary(torch.empty(1, 1, H, device=dev, dtype=DT), position_ids=pos)
            mask = create_causal_mask(config=cfg,
                                      inputs_embeds=torch.empty(bs, T, H, device=dev, dtype=DT),
                                      attention_mask=None, past_key_values=None, position_ids=pos,
                                      allow_is_causal_skip=False)
            # the indexer's real top-2048 would be the same token set (all of them), so this is what every
            # layer sees - resume-safe and bit-identical to propagating the computed indices
            sel = torch.arange(T, device=dev, dtype=torch.int32).view(1, 1, -1).expand(bs, T, T)
            contexts[bs] = (mask, pos, rope, sel)
        return contexts[bs]

    for il in range(first, last + 1):
        t0 = time.time()
        src, dst = mm[il % 2], mm[(il + 1) % 2]
        layer = load_layer(ck, cfg, il, dev)
        moe = None
        if ck.is_moe(il):
            moe = StatMoE(ck, cfg, il, dev)
            layer.mlp = moe
        if a.skip_indexer and getattr(layer.self_attn, "indexer", None) is not None:
            layer.self_attn.indexer = None     # full layers fall back to prev_topk_indices (the same arange)
        ffn = None
        if moe is not None and n_ffn:
            ffn = np.memmap(out / f"ffn_in_{il:02d}.bin", dtype=np.uint16, mode="w+", shape=(n_ffn, T, H))
        collect = [False]
        lin_stats, hooks = hook_linear_stats(layer, collect)
        t_load = time.time() - t0
        t_last = time.time() - t_load
        for b0 in range(0, n_seq, a.batch):
            bs_ids = list(range(b0, min(n_seq, b0 + a.batch)))
            collect[0] = bs_ids[0] < n_cal          # statistics from the calibration sequences only
            if moe is not None:
                moe.collect = collect[0]
                moe.ffn = None if ffn is None else (ffn, [s if s < n_ffn else -1 for s in bs_ids])
            x = torch.stack([torch.from_numpy(np.asarray(src[s])) for s in bs_ids])
            x = x.view(DT).to(dev)
            mask, pp, rope, sel = context(len(bs_ids))
            with torch.no_grad():
                y, _ = layer(x, attention_mask=mask, position_ids=pp, past_key_values=None, use_cache=False,
                             position_embeddings=rope, prev_topk_indices=sel)
            a_np = y.contiguous().cpu().view(torch.uint16).numpy()
            for i, s in enumerate(bs_ids):
                dst[s] = a_np[i]
            t_last = time.time() - t0 - t_load
        for hk in hooks:
            hk.remove()
        dst.flush()
        if ffn is not None:
            ffn.flush()
            del ffn
        st = {"layer": il, "linear": {k: {"ss": v.ss.float().cpu(), "n": v.n} for k, v in lin_stats.items()}}
        if moe is not None:
            st["moe"] = moe.stats()
        torch.save(st, out / f"stats_{il:02d}.pt")
        r = st.get("moe")
        extra = ""
        if r is not None:
            extra = (f", tokens {int(r['n'].sum())}, sal {float(r['sal'].sum()):.4g}, experts used"
                     f" {int((r['n'] > 0).sum())}")
        print(f"layer {il:2d}: {time.time() - t0:6.1f} s (load {t_load:.1f} s, fwd+save {t_last:.1f} s),"
              f" {len(lin_stats)} linears{extra}", flush=True)
        del layer, moe, lin_stats
        torch.cuda.empty_cache()
    print(f"gpu peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB", flush=True)
    if last == n_layers - 1:
        reference(ck, cfg, mm[(last + 1) % 2], n_cal, n_seq, T, a.topk, out, dev)
        print("CALIB DONE", flush=True)


def reference(ck, cfg, streams, n_cal, n_seq, T, topk, out, dev):
    """The eval sequences' next-token distributions: the final norm and lm_head over the kept hidden states."""
    if n_seq == n_cal:
        print("reference: no eval sequences (-n-eval 0), skipped", flush=True)
        return
    norm_w = ck.get(P + "norm.weight", dtype=torch.float32).to(dev)
    lm = ck.get("lm_head.weight", dev=dev)              # not tied (config tie_word_embeddings: False)
    vals, idxs, lses = [], [], []
    with torch.no_grad():
        for s in range(n_cal, n_seq):
            h = torch.from_numpy(np.asarray(streams[s])).view(DT).to(dev).float()
            h = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + float(cfg.rms_norm_eps)) * norm_w
            lg = (h.to(DT) @ lm.t()).float()            # logits fp32 from the bf16 product, as the quant pass sees
            lse = torch.logsumexp(lg, -1)
            v, i = torch.topk(lg, topk, -1)
            vals.append((v - lse[:, None]).cpu().numpy())
            idxs.append(i.int().cpu().numpy())
            lses.append(lse.cpu().numpy())
    np.savez(out / "ref_topk.npz", logprob=np.stack(vals), ids=np.stack(idxs), lse=np.stack(lses))
    print(f"reference: {n_seq - n_cal} sequences x {T} positions, top-{topk}", flush=True)


if __name__ == "__main__":
    main()
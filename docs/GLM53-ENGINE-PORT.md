# Porting the Maya engine to GLM-5.3 (`glm-dsa`)

Math references: llama.cpp `src/models/glm-dsa.cpp` (the commit Maya pins) and `transformers`' `glm_moe_dsa`.

## Design: one model with flags

The port lives inside Maya's existing `Glm5Model`. `Glm5Geometry` gets architecture flags and GLM-5.3-Flash keeps working. GLM-5.3 thereby inherits everything Maya already has: the VRAM / RAM / disk expert tiers and the service thread, splitting layers across GPUs, MTP and the draft-verification window, prompts on MMQ, KV streaming, slots and checkpoints, and the server. Upstream updates still merge in. New code goes into separate files where it can; shared files get short branches on the flags.

```cpp
struct Glm5Geometry {            // new fields
  bool  mhc;                     // Flash: true; GLM-5.3: false -> plain residual (h += attn; h += ffn), hc = 1
  int   qk_rope;                 // Flash: 0;   GLM-5.3: 64 (RoPE tail of q and k, theta = rope.freq_base = 8e6)
  float rope_base;
  int   lat_row;                 // floats in a latent-cache row: kv_lora + qk_rope (512 / 576)
  std::vector<uint8_t> idx_own;  // 1 = own indexer ("full" layer), 0 = reuses the previous full layer's top-k
  int   idx_kpool;               // Flash: 4; GLM-5.3: 1 (no pooling - the indexer selects tokens)
  int   idx_rope;                // indexer: RoPE on the first 64 of 128 head dims (GLM-5.3), 0 for Flash
  bool  swiglu_clamp;            // Flash: true (10); GLM-5.3: false
};
```

The architecture comes from `general.architecture`: `glm5-next` → Flash, `glm-dsa` → GLM-5.3, anything else is refused. The key prefix comes from the same place — **no silent Flash defaults**. The GLM-5.3 path is gated by `STRATA_GLM_DSA_DEV=1`.

## One GLM-5.3 layer

```
x  = rms(h) * attn_norm
qr = rms(q_a · x) * q_a_norm                                   [2048]
if own indexer:
   iq = idx_q_b · qr -> [32, 128], RoPE on [0:64)
   ik = LayerNorm(idx_k · x; k_norm.w, k_norm.b) -> [128], RoPE on [0:64); into the indexer cache
   w  = idx_proj · x / sqrt(128 · 32)                         [32]
   score[t] = sum_h relu(iq_h · ik_t) · w_h   (t <= pos)  -> top-2048 tokens -> sel
else: sel = the previous own-indexer layer's sel (same token)
q  = q_b · qr -> [64, 256] = [nope 192 | rope 64]; RoPE on the rope part
kv = kv_a · x -> [latent 512 | k_rope 64]; latent = rms(latent) * kv_a_norm; RoPE on k_rope
cache[pos] = [latent 512 | k_rope 64]                           (576 floats: FP16 or INT8 + FP16)
q_abs = q_nope · k_b                                            [64, 512]
score = (q_abs · latent_t + q_rope · k_rope_t) / sqrt(256),   t in sel
o = sum softmax · latent_t -> v_b -> [64, 256] -> attn_output -> h += o
FFN: h += MoE(rms(h) * ffn_norm)   (sigmoid + noaux_tc, top-8, x2.5, normalized, SwiGLU without clamp; + shared expert)
```

llama.cpp's Hadamard rotation of the indexer (`self_k_rot_lid`) does not change dot products and is not repeated. RoPE is llama.cpp's NORM (adjacent pairs) for both MLA and the indexer.

## Stages

| # | Stage | Check |
| --- | --- | --- |
| P0 | Synthetic GLM-5.3 (6 layers, hidden 512, 16 experts, indexer top-k 16) in transformers -> GGUF | transformers logits = llama.cpp |
| P1 | `glm-dsa` in `from_gguf`; `iq_pack.py` for `glm-dsa` | pack of the synthetic and the real model; geometry printed right |
| P2 | Slow reference path (`STRATA_GLM_SLOW=1`, `step_layers_dsa`) | synthetic logits = transformers (F32 max abs diff ~3e-5) |
| P3 | Fast token path (`fast_dsa53`; kernels `dsa53_prep`, `rope_norm`, `mla53`) | fast = slow (median correlation >= 0.999); real model's first 18 greedy tokens = llama.cpp |
| P4 | Batched prompt (`prefill_half`, tiles of 64, `mla53_attn` on tensor cores) | prefill + decode = decode only |
| P5 | NextN with its own indexer; the MTP window (`decode_mtp_pipe`) | window = token path, token for token |
| P6 | CPU lane up to `n_embd` 8192 | experts on CPU = on GPU |
| P7 | 7 GPUs, 256K-512K context | speed, comparison with llama.cpp |

## Pitfalls found on the way

- `GgufFile::get()` keeps only 64 array elements — long arrays need `get_full`.
- The split placement checked layer bounds against Flash's 45 layers before reading the geometry.
- `keep16` keeps BF16 only from 65536 elements — for `glm-dsa` a rule by name (router, `indexer.proj`).
- Flash kernels with 4096 hard-wired for `n_embd` silently truncate: `rms_rows`, `mtp_in_rows`, `rms_q8`, `mtp_in`, `head_prep` (`sm[4096]`), `embed_streams` — all widened.
- `dsa_topk` kept at most 1024 winners -> illegal memory access past 1024 tokens of context; now 2048.
- A100s are compute-bound when unpacking K/IQ quants; split-K does not help, Q8_0 does.
- The NextN cache for `glm-dsa` is written by `mtp_cache_rows` through `dsa53_prep` without q (Flash's layout crashed).

## Testing a change without seven GPUs

- Split the synthetic model on one GPU: `STRATA_GLM_SPLIT=1,4 STRATA_GLM_DEVS=0,0,0 STRATA_GLM_SPLIT_SAME_DEV=1` (+ `STRATA_GLM_CPU_LANE=0 STRATA_GLM_COMPLETE_POOL=1 STRATA_GLM_RESIDENT_SKIP=1`).
- Oracle drafts: `STRATA_GLM_MTP_PIPE_ORACLE=<file with the plain decode's ids>` + `STRATA_GLM_MTP_PIPE_ORACLE_ERR=k` (every k-th draft wrong) — the answer must match token for token.

```bash
# slow vs fast on the synthetic pack (gate: median correlation >= 0.999)
STRATA_GLM_SLOW=1 build/glm_dsa_model_test tiny/iq4/pack tiny/ref --tol 2 --no-top1 --save slow.f32
build/glm_dsa_model_test tiny/iq4/pack tiny/ref --tol 2 --no-top1 --compare slow.f32
# F32 against transformers
build/glm_dsa_model_test tiny/tiny-f32.gguf tiny/ref
```

The synthetic model is built by `tools/glm53/make_tiny.py`.

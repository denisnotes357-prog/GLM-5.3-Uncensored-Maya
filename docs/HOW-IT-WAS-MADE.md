# How GLM-5.3 Uncensored on Maya was made

## The idea

[Project Maya](https://github.com/mw00/project-maya) is the fastest way to run GLM-5.3-**Flash** (321B) on your own GPUs: an engine built around one model, with experts tiered across VRAM, RAM and disk, multi-token prediction and a polished server. The **full** GLM-5.3 (753B) is a different architecture, and nothing ran it fast on a home multi-GPU box: llama.cpp works but does not split layers across GPUs with Maya's tricks, and vLLM needs the model to fit in FP8/INT4 across identical GPUs.

The goal was simple to state:

1. run the full GLM-5.3 on one workstation — 5× RTX 4090 48 GB + 2× A100 40 GB (320 GB VRAM), 503 GB RAM;
2. as fast as the hardware allows, with long context;
3. without the refusals of the stock model;
4. without forking Maya into an unmaintainable copy — the port lives inside Maya's own model class behind flags, so upstream updates keep flowing in.

It took two days (15–16 October 2026). Below is what was done, in order, with the numbers and the dead ends.

## 1. Mapping Flash's assumptions

Maya is written for one model. Before any code, every place that hard-codes Flash was listed: 4 hyper-connection streams (mHC), indexer pooling over 4 tokens, 45 layers, `n_embd` 4096 in kernels with fixed shared-memory arrays, a SwiGLU clamp. Several kernels silently truncate at 4096 — they were found here, not in a crash at 2 a.m.

## 2. One model with flags, not a copy of the engine

`Glm5Geometry` got architecture flags; `general.architecture` in the GGUF picks Flash (`glm5-next`) or GLM-5.3 (`glm-dsa`), and an unknown architecture is refused instead of falling back to Flash defaults.

| | Flash | GLM-5.3 |
| --- | --- | --- |
| residual | 4 mHC streams | plain residual (`h += attn; h += ffn`) |
| attention | MLA, latent 512 | MLA + RoPE tail: q = [nope 192 \| rope 64], cache row 576 |
| DSA indexer | pools 4 tokens | selects single tokens, RoPE on 64 of 128 dims, LayerNorm with bias on k |
| own indexer | every layer | 21 "full" layers; the other 57 reuse the previous full layer's selection |
| layers | 45 | 78 + NextN |
| SwiGLU clamp | 10 | none (+inf — a 0 would have zeroed ~35 gates) |

The math reference was llama.cpp's `glm-dsa` (the commit Maya pins already supports it) and `transformers`' `glm_moe_dsa`. Details: [ENGINE-PORT.md](GLM53-ENGINE-PORT.md).

## 3. Proving it right before making it fast

Stage by stage, each closed by a check:

- **P0** — a synthetic GLM-5.3 (6 layers, hidden 512, 16 experts) built with `transformers`, converted to GGUF, logits saved as the reference.
- **P1** — the loader and `iq_pack.py` for `glm-dsa`.
- **P2** — a slow reference path (`STRATA_GLM_SLOW=1`): matches `transformers` on the synthetic model (max |Δ| ~3e-5 in F32).
- **P3** — the fast token path: fast = slow (median logit correlation ≥ 0.999 on a quantized synthetic pack).
- **First real answer**: the full GLM-5.3 (a large RTN quant, 362 GB) answered through the engine on one GPU + RAM at 3.6 tok/s; the first 18 greedy tokens matched llama.cpp exactly. All 1591 quantized tensors were checked against FP8.
- **P4** — batched prompt processing on tensor cores: a 1970-token prompt in 9.3 s instead of 135 s; a 9318-token prompt 25.5 → 12.4 s; on long prompts ~890 tokens/s.
- **Needle test**: a code hidden at token ~28K of a 71,274-token prompt came back exactly.
- **P7** — seven GPUs, layers split `10,22,34,46,58,70` (every part must start on a layer with its own indexer), the A100s at the ends of the pipeline.

## 4. Making it fast

- **Quant size decides speed.** With layers split across GPUs, a token visits the cards one after another, so speed ≈ bytes read per token ÷ one card's bandwidth. GLM-5.3 reads ~15 GB of dense weights per token in Q6_K against ~8 GB of experts, so the attention projections and shared expert went to **Q4_K** (recipe G-M24). Everything — all 256 experts of every layer — then fits in VRAM.
- **Calibration, like Maya.** FP8 calibration statistics over 128 sequences of chat, multilingual chat, reasoning traces, code and tool calls → imatrix for every tensor and GPTQ for the experts' gate/up. Against FP8 on 8 held-out texts: **GPTQ KL 0.207, top-1 agreement 88.3%, ppl 3.37** versus round-to-nearest 0.262 / 86.5% / 3.55 — better on every text. On one probe layer the expert output error went RTN 0.147 → imatrix 0.108 → GPTQ 0.034. RTN also fell into a loop on a poem where GPTQ did not.
- **The A100s are dequant-bound, not bandwidth-bound.** On the layers the split puts on the A100s, the dense tensors went to Q8_0, which an A100 unpacks at bandwidth (G-M24A; KL 0.204, top-1 88.4%).
- **A pipelined MTP window.** Maya's multi-token drafting only worked on a single GPU; on a split it drafted one token per step. Here the NextN head drafts a chain of up to 7 tokens on the last GPU and the whole chain is verified as one window flowing through all seven GPUs (`STRATA_GLM_MTP_PIPE=1`). The answer is token-for-token the same as plain decoding. **32 → 45–57 tok/s.**

Speed on that machine, 400-token answers:

| Text | tok/s |
| --- | ---: |
| English technical | 55–56.8 |
| Russian prose | 45 |
| Chat at temperature 1.0 | 43–44 |
| Without MTP | 24 |

First token after a short prompt: 0.5–0.7 s.

### What did not pay off (and why)

- **Tensor parallelism** (all GPUs on every layer): the 48 GB RTX 4090s have no peer-to-peer, so every all-reduce goes through pinned RAM — measured 76 µs per round, ×156 rounds per token. Slower than the layer pipeline.
- **The tinygrad P2P driver patch**: needs BAR1 ≥ VRAM; the 48 GB 4090s have a 32 GB BAR1 fixed in the VBIOS. Not possible.
- **`k_b` / `v_b` in Q8 for the attention kernel**: no speedup (the cost is kernel launches, not bytes), slightly worse accuracy — reverted.
- **1M context**: works, but the KV cache (12 GB per card) pushes ~40% of one card's experts to RAM: ~30 tok/s. 512K fits with no loss, so 512K is the default.

## 5. Removing the refusals

The stock model refused **39 of 40** JailbreakBench requests (thinking off; the judge was the stock model itself, asked for one word: REFUSED / COMPLIED / DISPUTED).

**Step 1 — donor weights.** [`dealignai/GLM-5.3-UNCENSORED-FP8`](https://huggingface.co/dealignai/GLM-5.3-UNCENSORED-FP8) (MIT) was compared byte by byte with `zai-org/GLM-5.3`: only `self_attn.o_proj` and `mlp.shared_experts.down_proj` differ (155 tensors). Instead of downloading 756 GB, just those tensors (8.9 GB) were fetched with HTTP range requests, quantized to the types our quant already uses (worst relative error 0.08, as for the originals), and swapped in — `tools/glm53/make_gm24u.py`. Speed unchanged. **39 → 6 refusals.**

**Step 2 — a runtime refusal vector.** The remaining refusals were the heaviest topics. Following [alesha-pro's method for GLM-5.3-Flash](https://github.com/alesha-pro/project-maya/blob/glm-ablate/docs/GLM-ABLATE.md), measured anew for the full model:

- 256 harmful and 100 harmless prompts; the engine dumps the last-position residual of every layer (`STRATA_GLM_ABLATE_CAPTURE`) in three modes — thinking off, prompt ending on an open thinking block, and the model's own thinking followed by the closing marker;
- per layer and mode: mean(harmful) − mean(harmless), minus its component along mean(harmless), normalized; the three directions orthonormalized (Gram–Schmidt); layer 0 left alone → `dir3.f32`, 78 × 3 × 6144;
- while generating, after every layer's FFN write, the residual loses its components along that layer's three directions (`src/core/glm_ablate.cuh`). No weights change; the vector is a config switch.

**6 → 2 refusals**, speed 44.8 / 54.1 tok/s (Russian / English). Strength 1.5 was worse (5 / 40). The two left are harm to children and eating disorders, where the model steers to a safe answer.

## 6. Context and serving

- 256K: 44.8 / 54.1 tok/s; **512K: 42.6 / 53.7, no experts leave VRAM** (KV 7.3 GB per card) — the default;
- 1M: works, 25 → 30–33 tok/s as the expert cache warms.
- The server is Maya's: browser chat, OpenAI and Anthropic APIs, a live monitor. One small addition for the capture: `chat_template_kwargs.maya_prefill` makes the answer continue a given text.

## Open ends

- Quality with the vector on (KL against the FP8 original with the same projection) was not measured.
- Speculative sampling of the drafts at temperature > 0 (`STRATA_GLM_SPEC_SAMPLE=1`, +5–21% in chat) is not on by default until a larger quality check.
- The FP8 original is no longer on the build machine, so the quant cannot be rebuilt from scratch there; the tools here rebuild it from `zai-org/GLM-5.3` + the donor tensors.

---

The engineering was done by Denis with AI coding agents (Claude Code as the main engineer, a local model for mechanical work), every engine change checked against the references above.

# GLM-5.3 Uncensored on Maya

**The full 753B-parameter GLM-5.3, uncensored, running at 43–57 tokens/s on seven consumer-and-datacenter GPUs, with 512K context.**

This repository is a fork of [Project Maya](https://github.com/mw00/project-maya) (MIT), the engine that runs GLM-5.3-*Flash* on your own GPUs. Maya only supported the Flash model. Here the engine is ported to the **full GLM-5.3** (`glm-dsa`: MLA attention with RoPE, the token-level DSA indexer, 78 layers + NextN), a calibrated 2–3 bit quant is built for it, and refusals are removed in two steps: donor weights, then a runtime refusal vector.

[Русская версия](README.ru.md) · [How it was made](docs/HOW-IT-WAS-MADE.md) · [Engine port](docs/GLM53-ENGINE-PORT.md) · [Model on Hugging Face](#download-the-model) · [Original Project Maya README](README.project-maya.md)

---

## Results

| | Stock GLM-5.3 (our quant) | **GLM-5.3 Uncensored (this repo)** |
| --- | ---: | ---: |
| Refusals on 40 JailbreakBench requests | 39 / 40 | **2 / 40** |
| Decode speed, English | 55–57 tok/s | **54 tok/s** |
| Decode speed, Russian | 45 tok/s | **43–45 tok/s** |
| Context | 256K | **512K** (1M works, ~30 tok/s) |
| Size | 267 GB | 267 GB GGUF + 2.8 GB pack |
| Quality vs the FP8 original (KL, 8 texts) | 0.204 (top-1 agreement 88.4%) | same weights except 155 donor tensors |

Hardware: 5× RTX 4090 48 GB + 2× A100 40 GB (320 GB VRAM total), 503 GB RAM. All experts live in VRAM; nothing is read from disk while generating.

Speed history of the port on that machine: 14.5 tok/s (first working build, larger quant) → 20 (MTP) → 38.6 (smaller calibrated quant) → **45–57 (pipelined MTP window)**. Without MTP: 24 tok/s.

## What is in here

- **Engine port to `glm-dsa`** (`src/core/glm_*.cu`, `src/kernels/cuda/glm_*.cu`): the full GLM-5.3 inside Maya's existing `Glm5Model`, behind architecture flags, so Flash keeps working and GLM-5.3 inherits everything Maya already had — the VRAM/RAM/disk expert tiers, splitting layers across GPUs, multi-token prediction, batched prompt processing, KV streaming, the OpenAI/Anthropic-compatible server and the browser chat.
- **Pipelined MTP window** (`STRATA_GLM_MTP_PIPE=1`, `decode_mtp_pipe` in `src/core/glm_mtp.cu`): the NextN head drafts a chain of up to 7 tokens on the last GPU, and the chain is verified as a window flowing through all seven GPUs. Output is token-for-token identical to plain decoding; speed goes from ~32 to 45–57 tok/s.
- **Runtime refusal vector** (`src/core/glm_ablate.cuh`, `STRATA_GLM_ABLATE*`): three directions per layer projected out of the residual stream after every layer. No requantization — switch it off, or change its strength, in the config.
- **Quantization toolchain** (`tools/glm53/`): FP8 calibration statistics, imatrix + GPTQ for the experts, the recipes, the donor merge (`make_gm24u.py`), the refusal-direction capture (`tools/glm53/ablate/`).
- **Tests**: a 6-layer synthetic GLM-5.3 checked against `transformers` (`tests/glm_dsa_model_test.cpp`), slow-vs-fast path parity, the geometry loader test.

## Fixes of 2026-10-20 (update if you cloned earlier)

- **Agent sessions could turn into garbage.** When one client's new conversation was dropped while its prompt was being read, the engine kept the previous conversation's snapshot record although it had already reset the caches; that conversation's next turn could resume over reset state and answer with nonsense. A reset now invalidates the snapshot (found and proven on the sister MiMo-V2.6-Pro port, where the same code path broke agent sessions).
- **Repetition penalties work.** `presence_penalty`, `frequency_penalty` and `repetition_penalty` were silently ignored on the GLM path; a request that sets them now runs through the token loop with its history (slower: no MTP drafts for that request).
- The server removes stray thinking tags from reasoning that agents send back in the history.

## Quick start

### 1. Build the engine

The same installer as upstream Maya (Linux x86-64, NVIDIA driver, CUDA 12.x or 13, Python 3.10+; see
[Project Maya's requirements](README.project-maya.md)):

```bash
git clone https://github.com/denisnotes357-prog/GLM-5.3-Uncensored-Maya
cd GLM-5.3-Uncensored-Maya
./setup.sh
```

It creates the private Python environment `.venv`, fetches ggml at the pinned llama.cpp commit and builds `build/strata`.
When it offers to download a model, answer **no** — it would fetch GLM-5.3-Flash, not this model. Rebuilding later:
`cmake --build build -j --target strata`. The port was built and tested for `sm_80` (A100) and `sm_89` (RTX 4090).

### 2. Download the model

```bash
pip install -U huggingface_hub
hf download nofacedfl/GLM-5.3-Uncensored-Maya --local-dir models/GLM-5.3-Uncensored-Maya
```

The download is ~270 GB: six GGUF shards and `pack/` (the engine's index of those shards plus the dense weights; it finds the shards next to itself).

### 3. Run

```bash
tools/glm53/run-glm53.sh                                     # uncensored: donor weights + refusal vector
tools/glm53/run-glm53.sh configs/glm53-uncensored-novector.json   # donor weights only
```

- Chat in the browser: http://127.0.0.1:8095/
- OpenAI API: `http://127.0.0.1:8095/v1/chat/completions`, model `glm-5.3`
- Anthropic API: `http://127.0.0.1:8095/v1/messages` (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8095`)
- Thinking: `reasoning_effort` = `none` / `low` / `high` (default) / `max`
- Live speed monitor: http://127.0.0.1:8095/#monitor

Loading takes about 3 minutes (the experts are warmed into VRAM). The server log should contain
`glm ablate: 1437696 values from tools/glm53/ablate/dir3.f32 ... per layer, scale 1.00`.

### Fitting your GPUs

The shipped config is for 7 GPUs with ~320 GB of VRAM. The keys that matter:

| Key | Meaning |
| --- | --- |
| `gpu: [5, 0, 2, 3, 4, 7, 6]` | the order of the cards along the layer pipeline (the slower A100s at the ends) |
| `STRATA_GLM_SPLIT=10,22,34,46,58,70` | first layer of each next GPU; every part must start on a layer with its own DSA indexer |
| `--max-context 512000` | 512K fits without moving experts out of VRAM on this machine; 1M works with ~30% of experts in RAM |
| `STRATA_GLM_RAM_RESIDENT=1` | whatever does not fit in VRAM is served from RAM, never from disk |
| `STRATA_GLM_MTP_PIPE=1`, `STRATA_GLM_MTP_DRAFT=7` | the pipelined MTP window |
| `STRATA_GLM_ABLATE_SCALE` | strength of the refusal vector (default 1.0; 1.5 was worse) |
| `STRATA_GLM_DSA_DEV=1` | the GLM-5.3 path is marked experimental; the engine refuses `glm-dsa` without it |

With less VRAM, Maya's tiers still work: experts that do not fit are served from RAM (slower, see the 1M-context numbers).

## How the uncensoring works

1. **Donor weights.** [`dealignai/GLM-5.3-UNCENSORED-FP8`](https://huggingface.co/dealignai/GLM-5.3-UNCENSORED-FP8) (MIT) differs from the original only in `self_attn.o_proj` and `mlp.shared_experts.down_proj` — 155 tensors, 8.9 GB. We downloaded just those tensors, quantized them to the same types our quant uses, and swapped them in (`tools/glm53/make_gm24u.py`). Refusals: 39 → 6 of 40.
2. **Runtime refusal vector.** For the remaining refusals, three directions per layer (no thinking / open thinking / end of the model's own thinking) are measured on 256 harmful vs 100 harmless prompts and projected out of the residual stream after every layer while generating. Refusals: 6 → 2 of 40. The method and the original engine hook come from [alesha-pro](https://github.com/alesha-pro/project-maya/blob/glm-ablate/docs/GLM-ABLATE.md), who did it for GLM-5.3-Flash; here it is measured anew for the full 78-layer model.

The two remaining refusals are the heaviest topics (harm to children, eating disorders), where the model turns the answer into a safe one. Details, numbers and dead ends: [docs/HOW-IT-WAS-MADE.md](docs/HOW-IT-WAS-MADE.md).

## Download the model

**Hugging Face:** [nofacedfl/GLM-5.3-Uncensored-Maya](https://huggingface.co/nofacedfl/GLM-5.3-Uncensored-Maya)

| File | Size |
| --- | ---: |
| `GLM-5.3-Maya-G-M24U-GPTQ-0000{1..6}-of-00006.gguf` | 267 GB |
| `pack/` (engine index + dense weights + tokenizer + chat template) | 2.8 GB |
| `dir3.f32` (refusal vector, 78 × 3 × 6144 float32; also in this repo) | 5.8 MB |

Quant recipe (`tools/glm53/recipes/g-m24a.json`): experts IQ2_S gate/up and IQ3_XXS down (IQ3_S down in layers 3–6 and 74–77), GPTQ-rounded against FP8 calibration statistics; attention projections and the shared expert Q4_K, on the A100 layers Q8_0; NextN Q3_K/Q4_K; router and norms F32. The tensor layout follows llama.cpp's `glm-dsa`; the tested runtime is this engine.

## Responsible use

This model will answer requests that the original refuses. It is published for research, red-teaming, private use and creative work by adults. You are responsible for how you use it and for complying with the laws that apply to you. The output can be wrong or harmful; do not rely on it for medical, legal, safety-critical or other high-stakes decisions.

## Credits and licenses

- **Project Maya** by Niko1221 and the Strata contributors — the engine, server and quant tools this repo is built on. MIT, see [LICENSE](LICENSE). Fork base: upstream commit `d187195` (2026-10-15).
- **GLM-5.3** by Z.AI — the model. [GLM-5.3 License](LICENSES/GLM-5.3-LICENSE.txt) (permissive; includes a security-review condition for very large Model-as-a-Service operators).
- **dealignai** — the uncensored donor tensors, [GLM-5.3-UNCENSORED-FP8](https://huggingface.co/dealignai/GLM-5.3-UNCENSORED-FP8), MIT.
- **alesha-pro** — the runtime ablation hook for Maya and the three-direction method, [project-maya `glm-ablate`](https://github.com/alesha-pro/project-maya/blob/glm-ablate/docs/GLM-ABLATE.md).
- **llama.cpp** and **transformers** — the reference implementations the port was checked against.

See [NOTICE.md](NOTICE.md) for the full list. If this project is useful to you, a ⭐ helps others find it.

---
license: other
license_name: glm-5.3
license_link: LICENSE
base_model:
- zai-org/GLM-5.3
- dealignai/GLM-5.3-UNCENSORED-FP8
tags:
- glm
- glm-5.3
- glm-dsa
- moe
- gguf
- uncensored
- abliterated
- maya
language:
- en
- zh
- ru
pipeline_tag: text-generation
---

# GLM-5.3 Uncensored — Maya quant (G-M24U-GPTQ)

The full **GLM-5.3** (753B MoE) quantized to ~2.5 bits with GPTQ, with refusals removed, built to run on the
[GLM-5.3 Uncensored on Maya](https://github.com/denisnotes357-prog/GLM-5.3-Uncensored-Maya) engine — a fork of
[Project Maya](https://github.com/mw00/project-maya) ported to the full GLM-5.3.

**43–57 tokens/s on 7 GPUs (5× RTX 4090 48 GB + 2× A100 40 GB), 512K context, 2 / 40 refusals on JailbreakBench (stock: 39 / 40).**

## Files

| File | Size | What |
| --- | ---: | --- |
| `GLM-5.3-Maya-G-M24U-GPTQ-0000{1..6}-of-00006.gguf` | 267 GB | the quantized model (llama.cpp `glm-dsa` tensor layout) |
| `pack/` | 2.8 GB | the Maya engine's index of the shards + dense weights + tokenizer + chat template; it finds the shards next to itself |
| `dir3.f32` | 5.8 MB | the refusal vector, 78 layers × 3 directions × 6144, float32 (applied at runtime) |

## Run

```bash
git clone https://github.com/denisnotes357-prog/GLM-5.3-Uncensored-Maya && cd GLM-5.3-Uncensored-Maya
./setup.sh                      # builds the engine (decline the Flash model download it offers)
hf download nofacedfl/GLM-5.3-Uncensored-Maya --local-dir models/GLM-5.3-Uncensored-Maya
tools/glm53/run-glm53.sh        # chat at http://127.0.0.1:8095, OpenAI and Anthropic APIs
```

The shipped config needs ~320 GB of VRAM across 7 GPUs for full speed; with less, the engine serves the experts that
do not fit from RAM (slower). The tested runtime is the Maya engine; llama.cpp compatibility of this exact file was not tested.

## How it was made

1. **Quant (G-M24A-GPTQ).** FP8 calibration statistics on 128 sequences (chat, multilingual, reasoning, code, tool
   calls) → imatrix for every tensor and GPTQ for the experts' gate/up. Experts IQ2_S gate/up and IQ3_XXS down
   (IQ3_S down in layers 3–6 and 74–77); attention and shared expert Q4_K, Q8_0 on the layers placed on the A100s;
   NextN Q3_K/Q4_K; router and norms F32. Against FP8 on 8 held-out texts: **KL 0.204, top-1 agreement 88.4%**.
2. **Donor (G-M24U).** `dealignai/GLM-5.3-UNCENSORED-FP8` changes only `self_attn.o_proj` and
   `mlp.shared_experts.down_proj` (155 tensors); those were quantized to the same types and swapped in.
   Refusals 39 → 6 of 40, speed unchanged.
3. **Refusal vector (`dir3.f32`).** Three directions per layer measured on 256 harmful / 100 harmless prompts
   (method of [alesha-pro](https://github.com/alesha-pro/project-maya/blob/glm-ablate/docs/GLM-ABLATE.md)),
   projected out of the residual stream after every layer by the engine. Refusals 6 → 2 of 40.

Full story, numbers and dead ends: [HOW-IT-WAS-MADE.md](https://github.com/denisnotes357-prog/GLM-5.3-Uncensored-Maya/blob/main/docs/HOW-IT-WAS-MADE.md).

## Speed (7 GPUs, pipelined MTP window, 400-token answers)

| Text | tok/s |
| --- | ---: |
| English | 54–57 |
| Russian | 43–45 |
| Chat at temperature 1.0 | 43–44 |
| 1M context | ~30 |

## Responsible use

This model answers requests the original refuses. It is published for research, red-teaming, private and creative use
by adults. You are responsible for how you use it and for following the laws that apply to you. Outputs can be wrong
or harmful; do not rely on them for medical, legal or other high-stakes decisions.

## License and credits

- Model: **GLM-5.3 by Z.AI** under the [GLM-5.3 License](LICENSE) (included in this repo; permissive, with a
  security-review condition for Model-as-a-Service operators above US$10B revenue).
- Donor tensors: [dealignai/GLM-5.3-UNCENSORED-FP8](https://huggingface.co/dealignai/GLM-5.3-UNCENSORED-FP8), MIT.
- Engine: [Project Maya](https://github.com/mw00/project-maya) by Niko1221 and the Strata contributors, MIT;
  the GLM-5.3 port in [GLM-5.3-Uncensored-Maya](https://github.com/denisnotes357-prog/GLM-5.3-Uncensored-Maya), MIT.

# Notice

GLM-5.3 Uncensored on Maya combines work from several projects. Their licenses apply to their parts.

| Part | Author | License | Where |
| --- | --- | --- | --- |
| The engine, server, dashboard and quant tools (everything not listed below) | Niko1221 and the Strata contributors — [Project Maya](https://github.com/mw00/project-maya) | MIT | [LICENSE](LICENSE). Forked at upstream commit `d187195` (2026-10-15). |
| The GLM-5.3 port (`glm-dsa` in `src/`, `include/`, `tests/glm_dsa_*`, `tools/glm53/`), the MTP window, the configs and docs of this repo | denisnotes357-prog | MIT | this repository |
| The runtime ablation hook (`src/core/glm_ablate.cuh`, first written for GLM-5.3-Flash) and the three-direction refusal method | [alesha-pro](https://github.com/alesha-pro/project-maya/blob/glm-ablate/docs/GLM-ABLATE.md) | MIT (fork of Project Maya) | adapted here to the full GLM-5.3 |
| GLM-5.3 model weights (the quantized model on Hugging Face) | Z.AI — [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) | GLM-5.3 License | [LICENSES/GLM-5.3-LICENSE.txt](LICENSES/GLM-5.3-LICENSE.txt) |
| The 155 uncensored donor tensors (`self_attn.o_proj`, `mlp.shared_experts.down_proj`) | [dealignai/GLM-5.3-UNCENSORED-FP8](https://huggingface.co/dealignai/GLM-5.3-UNCENSORED-FP8) | MIT | merged by `tools/glm53/make_gm24u.py` |
| ggml (fetched at build time at the commit Maya pins) and the `glm-dsa` reference | [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | not vendored |

The refusal vector `tools/glm53/ablate/dir3.f32` was measured from the model by this project.

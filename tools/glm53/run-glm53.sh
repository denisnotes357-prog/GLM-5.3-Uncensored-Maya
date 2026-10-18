#!/usr/bin/env bash
# tools/glm53/run-glm53.sh - GLM-5.3 Uncensored (G-M24U) on the 7 GPUs (5x 4090 + 2x A100; not the 3090) through Maya's server:
# the browser chat and the OpenAI / Anthropic APIs on http://127.0.0.1:8095 (OpenAI: /v1/chat/completions, Anthropic:
# /v1/messages).  any other GPU job (e.g. a vLLM server) must be stopped first.  Ctrl+C here stops GLM-5.3.
#   tools/glm53/run-glm53.sh [config]   (default configs/glm53-uncensored.json; without the refusal vector:
#   configs/glm53-uncensored-novector.json)
set -euo pipefail
cd "$(dirname "$0")/../.."
cfg=${1:-configs/glm53-uncensored.json}
if pgrep -f 'vllm.entrypoints.cli.main serve' >/dev/null; then
    echo "vLLM is running on the GPUs GLM-5.3 needs - stop it first (its window, Ctrl+C)" >&2
    exit 1
fi
py=python3; [ -x .venv/bin/python ] && py=.venv/bin/python
exec "$py" serve/server.py --engine strata --config "$cfg"

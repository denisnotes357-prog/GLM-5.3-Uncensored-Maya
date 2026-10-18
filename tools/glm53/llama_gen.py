#!/usr/bin/env python3
"""tools/glm53/llama_gen.py - ONE greedy generation through llama.cpp's llama-server, on token ids.

The same contract as engine_gen.py (ids, chat-template prompt, n tokens), so the two runs can be
diffed token by token.  It boots

    llama-server -m GGUF [--host 127.0.0.1 --port P] + shlex.split(--server-args)

waits for GET /health == 200, then POST /completion with

    {"prompt": [ids...], "n_predict": n, "temperature": 0, "top_k": 1, "ignore_eos": true,
     "n_probs": 0, "cache_prompt": false, "return_tokens": true}

The generated ids are `response["tokens"]` (server-task.cpp to_json_non_oaicompat) - only filled when
`return_tokens` is asked for (server-context.cpp:1839), which is why the payload carries it.
The server is stopped (SIGTERM, then KILL) when the run ends.

CUDA: the run asks for `-ngl 0` (CPU), yet llama.cpp's CUDA backend still enumerates EVERY card and
calls cudaSetDevice on it while loading - on this box GPU 2-7 are held by vLLM (GPU5 answered
`CUDA error: out of memory` in ggml_cuda_set_device), so the server starts with
CUDA_VISIBLE_DEVICES="" (no GPU at all - which is what `-ngl 0` wants).  --cuda-devices 0 (or any
list) puts the visible cards back for a run that really wants them.

    python llama_gen.py --gguf model.gguf [--ids toks.npy | --ids 1,2,3 | --prompt "text"]
        [--n 16] [--port 18093] [--server-args "-ngl 0 -c 512 -t 8"]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

DEFAULT_EXE = "/data/glm53/llama.cpp/build/bin/llama-server"
DEFAULT_TOK = "/data/glm53/tokenizer"   # GLM-5.3's HF tokenizer and chat template (the FP8 checkpoint is gone)
DEFAULT_LOG = "/data/glm53/llama-gen.log"


def load_ids(spec: str) -> list[int] | None:
    p = pathlib.Path(spec)
    if p.exists():
        return [int(x) for x in np.load(p).reshape(-1)]
    if "," in spec or spec.strip().isdigit():
        return [int(x) for x in spec.split(",")]
    return None


NO_THINK = False


def render_prompt(text: str, tok_dir: str) -> tuple[list[int], str, str]:
    """Identical to engine_gen.render_prompt: GLM-5.3's template has no reasoning=off (its line 2
    maps anything outside ['low','high'] - so also our 'none' - onto 'max'), so we ask for 'none',
    notice it in the rendered string and re-render without the kwarg."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir)
    msgs = [{"role": "user", "content": text}]
    try:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    reasoning_effort="none")
        asked = "reasoning_effort='none'"
    except Exception:
        s = None
        asked = "reasoning_effort rejected by apply_chat_template"
    if s is None or "Reasoning Effort: " in s:
        fallback_err = "" if s is not None else "apply_chat_template raised"
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        asked = f"{asked} -> NOT HONOURED by the template; rendered WITHOUT the kwarg" \
                f"{' (' + fallback_err + ')' if fallback_err else ''}"
    if NO_THINK:   # the generation prompt ends with "<|assistant|><think>": close it, as Maya's server does for "off"
        s += "</think>"
    ids = tok(s, add_special_tokens=False)["input_ids"]
    return ids, s, asked


def get(url: str, timeout: float) -> int:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, OSError):
        return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", default=DEFAULT_EXE)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ids", help=".npy file or a comma list of token ids")
    ap.add_argument("--prompt", help="tokenized through --tok's chat template instead")
    ap.add_argument("--prompt-file", help="the --prompt text from a file (a long prompt: argv holds 128 KB an argument)")
    ap.add_argument("--tok", default=DEFAULT_TOK)
    ap.add_argument("--no-think", action="store_true", help="--prompt: close the <think> block (answer directly)")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18093)
    ap.add_argument("--server-args", default="-ngl 0 -c 512 -t 8",
                    help="extra llama-server flags as ONE string")
    ap.add_argument("--cuda-devices", default="",
                    help='CUDA_VISIBLE_DEVICES for the server (empty = no GPU, what -ngl 0 means; 0 = GPU0)')
    ap.add_argument("--log", default=DEFAULT_LOG, help="the server's stderr/stdout")
    ap.add_argument("--ready-timeout", type=float, default=600.0)
    ap.add_argument("--keep", action="store_true", help="leave the server running")
    a = ap.parse_args()

    global NO_THINK
    NO_THINK = a.no_think
    if getattr(a, "prompt_file", None):
        a.prompt = open(a.prompt_file, encoding="utf-8").read()
    ids = load_ids(a.ids) if a.ids else None
    if a.prompt:
        if ids is not None:
            raise SystemExit("--ids and --prompt are exclusive")
        ids, template, reasoning = render_prompt(a.prompt, a.tok)
        print(f"template: {template!r}")
        print(f"reasoning: {reasoning}")
        print(f"ids: n={len(ids)} first30={ids[:30]}")
    elif ids is not None:
        print(f"ids: n={len(ids)} first30={ids[:30]}")
    else:
        raise SystemExit("one of --ids / --prompt is required")
    sys.stdout.flush()

    cmd = [a.exe, "-m", a.gguf, "--host", a.host, "--port", str(a.port)] + shlex.split(a.server_args)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = a.cuda_devices
    print(f"server: {' '.join(cmd)}")
    print(f"cuda: CUDA_VISIBLE_DEVICES={a.cuda_devices!r}")
    print(f"log: {a.log}")
    log = open(a.log, "a", encoding="utf-8")
    log.write(f"\n----- llama_gen.py {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)} "
              f"CUDA_VISIBLE_DEVICES={a.cuda_devices!r}\n")
    log.flush()
    proc = subprocess.Popen(cmd, stdout=log, stderr=log, text=True, env=env)
    base = f"http://{a.host}:{a.port}"
    t0 = time.perf_counter()
    try:
        while True:
            if get(f"{base}/health", 5.0) == 200:
                break
            if proc.poll() is not None:
                print(f"FAIL: the server exited before /health (rc={proc.returncode}); see {a.log}")
                sys.exit(1)
            if time.perf_counter() - t0 > a.ready_timeout:
                print(f"FAIL: /health did not answer 200 in {a.ready_timeout:.0f} s; see {a.log}")
                sys.exit(1)
    except KeyboardInterrupt:
        print("FAIL: interrupted")
        sys.exit(1)
    print(f"ready: /health 200 in {time.perf_counter() - t0:.1f} s")

    payload = {"prompt": [int(t) for t in ids], "n_predict": int(a.n), "temperature": 0,
               "top_k": 1, "ignore_eos": True, "n_probs": 0, "cache_prompt": False,
               "return_tokens": True}
    sent = time.perf_counter()
    req = urllib.request.Request(f"{base}/completion", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            resp = json.loads(r.read().decode())
    except Exception as e:
        print(f"FAIL: /completion: {type(e).__name__}: {e}")
        sys.exit(1)
    elapsed = time.perf_counter() - sent

    gen = resp.get("tokens") or []
    if not gen:
        # the ids are the only source we accept as equivalent (the old builds omit the field without
        # return_tokens; ours asked for it - say so rather than reconstructing from text)
        print(f"WARN: no ids in the response (keys: {sorted(resp.keys())})")
    text = resp.get("content", "")
    tim = resp.get("timings", {})
    print(f"ids: n={len(ids)} first30={ids[:30]}")
    print(f"gen: n={len(gen)} {[int(x) for x in gen]}")
    print(f"text: {text!r}")
    print(f"time: {elapsed * 1000:.1f} ms wall, server timings: "
          f"prompt {tim.get('prompt_ms', '?')} ms / predicted {tim.get('predicted_ms', '?')} ms "
          f"= {tim.get('predicted_per_second', '?')} tok/s, "
          f"evaluated {resp.get('tokens_evaluated', '?')} predicted {resp.get('tokens_predicted', '?')}")

    if not a.keep:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=20)
        print(f"server stopped (rc={proc.returncode})")
    else:
        print(f"server left running (pid {proc.pid})")
    log.close()
    sys.exit(0 if gen else 1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""tools/glm53/engine_gen.py - ONE greedy generation through the Maya engine's --serve wire protocol.

The engine side is src/program/generate.cpp (glm_pack_generate, the --serve block: it parses
`GEN <max_new> [key=val ...] <id,id,...>`), the client side is serve/server.py::StrataEngine:

    us ->    READY <ctx> [stop] [pause] [multi=N]      (engine ready: the model is loaded)
    us ->    GEN <max_new>[ temperature=0 ...] <id,id,...>
    engine -> T <id>                                  (one per token; PP ... while the prompt is read,
                                                        STAT/NOTE/INFO in between)
    engine -> DONE <generated> <prompt> <prompt_ms> <decode_ms> <finish> ...
    us ->    QUIT                                     (the engine frees its GPU memory itself)
    engine -> ERR <text>                              (a request error: printed and exit 1)

Greedy = `temperature=0` (serve/server.py sampling_keys: temperature <= 0 sends only that key, and
"the engine reads temperature=0 as that whatever else the line says", no other sampler key goes).

    python engine_gen.py --pack /path/to/pack (--ids toks.npy | --ids 1,2,3 | --prompt "text")
        [--exe build/strata] [--n 32] [--max-context 4096]
        [--tok /data/glm53/tokenizer] [--env KEY=VAL ...] [--log /data/glm53/engine-gen.log]

Stderr of the engine goes to --log, the protocol (stdout) is read here.  Always prints ids/gen/text
and the time to the first token; exits 0 only on DONE.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import queue
import subprocess
import sys
import threading
import time

import numpy as np

DEFAULT_EXE = "build/strata"
DEFAULT_TOK = "/data/glm53/tokenizer"   # GLM-5.3's HF tokenizer and chat template (the FP8 checkpoint is gone)
DEFAULT_LOG = "/data/glm53/engine-gen.log"


def load_ids(spec: str) -> list[int] | None:
    """"a.npy" -> its values; otherwise a comma list.  None: not an ids spec."""
    p = pathlib.Path(spec)
    if p.exists():
        return [int(x) for x in np.load(p).reshape(-1)]
    if "," in spec or spec.strip().isdigit():
        return [int(x) for x in spec.split(",")]
    return None


NO_THINK = False


def render_prompt(text: str, tok_dir: str) -> tuple[list[int], str, str]:
    """chat template -> ids, the rendered template, and which reasoning setting was used.

    GLM-5.3's chat_template.jinja has ONE reasoning knob and no "off":
        line 2: reasoning_effort if defined and in ['low','high'] else 'max'
    so 'none' is NOT honoured (it becomes 'max'), and leaving it out also renders
    `Reasoning Effort: Max` - the least reasoning we can ask for through the template is 'low'.
    We follow the recipe anyway: ask for 'none', detect the fallback in the rendered string, and if
    the template ignored it, re-render WITHOUT the kwarg and say so (the facts go into the output).
    """
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
        # the template ignored it (or refused): render without the kwarg, which is what it does anyway
        fallback_err = "" if s is not None else "apply_chat_template raised"
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        asked = f"{asked} -> NOT HONOURED by the template; rendered WITHOUT the kwarg" \
                f"{' (' + fallback_err + ')' if fallback_err else ''}"
    if NO_THINK:   # the generation prompt ends with "<|assistant|><think>": close it, as Maya's server does for "off"
        s += "</think>"
    ids = tok(s, add_special_tokens=False)["input_ids"]
    return ids, s, asked


def pump(proc: subprocess.Popen, lines: queue.Queue) -> None:
    try:
        for line in proc.stdout:
            lines.put(line)
    finally:
        lines.put(None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", default=DEFAULT_EXE)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ids", help=".npy file or a comma list of token ids")
    ap.add_argument("--prompt", help="tokenized through --tok's chat template instead")
    ap.add_argument("--prompt-file", help="the --prompt text from a file (a long prompt: argv holds 128 KB an argument)")
    ap.add_argument("--tok", default=DEFAULT_TOK)
    ap.add_argument("--no-think", action="store_true", help="--prompt: close the <think> block (answer directly)")
    ap.add_argument("--n", type=int, default=32, help="tokens to generate")
    ap.add_argument("--temp", type=float, default=0.0, help="> 0: a sampled run (with --seed and --top-p) instead of greedy")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--log", default=DEFAULT_LOG, help="the engine's stderr (protocol stays on stdout)")
    ap.add_argument("--ready-timeout", type=float, default=600.0)
    ap.add_argument("--gen-timeout", type=float, default=1800.0)
    a = ap.parse_args()

    global NO_THINK
    NO_THINK = a.no_think
    if getattr(a, "prompt_file", None):
        a.prompt = open(a.prompt_file, encoding="utf-8").read()
    ids = load_ids(a.ids) if a.ids else None
    template = ""
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

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)

    env = dict(os.environ)
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    log = open(a.log, "a", encoding="utf-8")
    log.write(f"\n----- engine_gen.py {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"{a.exe} --serve --glm-pack {a.pack} --max-context {a.max_context} "
              f"env={dict((kv.partition('=')[0], kv.partition('=')[2]) for kv in a.env)}\n")
    log.flush()
    print(f"engine: {a.exe} --serve --glm-pack {a.pack} --max-context {a.max_context}"
          + (f" env={a.env}" if a.env else ""))
    print(f"log: {a.log}")

    proc = subprocess.Popen([a.exe, "--serve", "--glm-pack", a.pack,
                             "--max-context", str(a.max_context)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                            text=True, encoding="utf-8", bufsize=1, env=env)
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=pump, args=(proc, lines), daemon=True).start()

    def take(timeout: float) -> str | None:
        try:
            return lines.get(timeout=timeout)
        except queue.Empty:
            return ""

    # ---- READY
    t0 = time.perf_counter()
    line, seen = None, []
    while time.perf_counter() - t0 < a.ready_timeout:
        line = take(30.0)
        if line == "":
            if proc.poll() is not None:
                print(f"FAIL: the engine exited before READY (rc={proc.returncode}); see {a.log}")
                sys.exit(1)
            continue
        if line is None:
            print(f"FAIL: the engine closed its stdout before READY; see {a.log}")
            sys.exit(1)
        line = line.rstrip("\n")
        if line.startswith(("NOTE ", "INFO ", "VRAM ", "STAT ")):
            seen.append(line)
            continue
        break
    if not line or not line.startswith("READY"):
        print(f"FAIL: no READY line (last: {line!r}); see {a.log}")
        sys.exit(1)
    print(f"ready: {line} ({time.perf_counter() - t0:.1f} s)")

    # ---- GEN, greedy
    samp = f"temperature={a.temp!r} top_p={a.top_p!r} seed={int(a.seed)}" if a.temp > 0 else "temperature=0"
    head = f"GEN {int(a.n)} {samp} {','.join(str(int(t)) for t in ids)}"
    sent = time.perf_counter()
    proc.stdin.write(head + "\n")
    proc.stdin.flush()
    gen: list[int] = []
    first = None
    done_fields: dict = {}
    try:
        while True:
            line = take(30.0)
            if line == "":
                if proc.poll() is not None:
                    print(f"FAIL: the engine exited mid-request (rc={proc.returncode}); see {a.log}")
                    sys.exit(1)
                if time.perf_counter() - sent > a.gen_timeout:
                    print(f"FAIL: no DONE in {a.gen_timeout:.0f} s; see {a.log}")
                    sys.exit(1)
                continue
            if line is None:
                print(f"FAIL: the engine closed its stdout mid-request; see {a.log}")
                sys.exit(1)
            line = line.rstrip("\n")
            if line.startswith("T "):
                if first is None:
                    first = time.perf_counter()
                gen.append(int(line[2:]))
            elif line.startswith("PP "):
                pass                      # prompt progress: the prompt is being read
            elif line.startswith(("STAT ", "NOTE ", "INFO ")):
                if line.startswith("NOTE "):
                    print("[engine] " + line[5:])
            elif line.startswith("ERR"):
                print("FAIL: " + line[4:].strip())
                print(f"sent: {head[:80]} ...")
                proc.stdin.write("QUIT\n")
                proc.stdin.flush()
                sys.exit(1)
            elif line.startswith("DONE"):
                f = line.split()
                done_fields = {"generated": int(f[1]), "prompt": int(f[2]), "prompt_ms": float(f[3]),
                               "decode_ms": float(f[4]), "finish": f[5] if len(f) > 5 else "?"}
                break
            else:
                pass
    finally:
        try:
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            proc.wait(timeout=60)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            try:
                proc.kill()
                proc.wait(timeout=20)
            except (OSError, subprocess.TimeoutExpired):
                pass
    log.close()

    # ---- report
    text = tok.decode(gen, skip_special_tokens=True) if gen else ""
    ttf = (first - sent) * 1000.0 if first is not None else float("nan")
    span = (time.perf_counter() - sent) if gen else 0.0
    engine_tps = (1000.0 * done_fields.get("generated", 0) / done_fields["decode_ms"]
                  if done_fields.get("decode_ms") else float("nan"))
    print(f"ids: n={len(ids)} first30={ids[:30]}")
    print(f"gen: n={len(gen)} {gen}")
    print(f"text: {text!r}")
    print(f"first_token: {ttf:.1f} ms")
    print(f"speed: {len(gen) / span if span else float('nan'):.2f} tok/s over {span:.2f} s "
          f"(engine: {done_fields})")
    sys.exit(0 if gen else 1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""tools/glm53/llama_logits.py - compare a HF reference's next-token distribution with llama.cpp.

Boots llama-server on a local port with a GGUF, posts the SAME token ids as an array prompt
(server-common.cpp documents `"prompt": [12, 34, 56]`), and compares the predicted top-1 and its
probability against softmax(reference logits) at the last position of each prefix.

    python llama_logits.py --gguf /data/glm53/tiny/tiny-f32.gguf \
        --tokens /data/glm53/tiny/ref/tokens.npy \
        --ref /data/glm53/tiny/ref/logits.npy \
        --prefixes 16,32,64,96 [--port 18091] [--threads 8]

Stops the server afterwards (unless --keep).  CPU only: -ngl 0.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

SERVER = "/data/glm53/llama.cpp/build/bin/llama-server"


def post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(url: str, timeout: float = 5.0) -> int:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, OSError):
        return 0   # not listening yet


def wait_health(host: str, port: int, timeout: float) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if get(f"http://{host}:{port}/health") == 200:
            return
        time.sleep(0.5)
    raise SystemExit(f"llama-server /health did not become ready in {timeout}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--prefixes", default="16,32,64,96")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18091)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default="")
    ap.add_argument("--keep", action="store_true", help="do not stop the server")
    a = ap.parse_args()

    ids = np.load(a.tokens).astype(np.int64)
    logits = np.load(a.ref).astype(np.float64)   # [96, vocab] from the HF reference
    prefixes = [int(x) for x in a.prefixes.split(",")]
    assert logits.shape[0] == ids.shape[0], (logits.shape, ids.shape)

    cmd = [SERVER, "-m", a.gguf, "--host", a.host, "--port", str(a.port),
           "-ngl", "0", "-c", str(a.ctx), "-t", str(a.threads), "--no-warmup"]
    print("SERVER " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    rows = []
    try:
        wait_health(a.host, a.port, 180.0)
        for L in prefixes:
            pref = ids[:L]
            payload = {"prompt": [int(t) for t in pref], "n_predict": 1, "n_probs": a.top,
                       "temperature": 0, "top_k": 1, "ignore_eos": True, "seed": 1}
            resp = post(f"http://{a.host}:{a.port}/completion", payload)
            # llama.cpp /completion answers with "completion_probabilities": this build emits
            # [{id, token, logprob, top_logprobs: [{id, token, logprob}, ..]}, ..] (one entry per
            # generated token); accept the older shapes as well
            probs = []
            block = resp.get("completion_probabilities")
            if isinstance(block, list) and block:
                first = block[0]
                if isinstance(first, dict) and "top_logprobs" in first:
                    probs = first["top_logprobs"]
                elif isinstance(first, dict) and isinstance(first.get("probs"), list):
                    probs = first["probs"]
                elif isinstance(first, dict) and ("id" in first or "token" in first):
                    probs = block
                elif isinstance(first, list):
                    probs = first
            if not probs:
                probs = resp.get("probs") or []
            if not rows:
                print("RAW completion_probabilities = " + json.dumps(block)[:400], flush=True)
            if probs and isinstance(probs[0], dict):
                def norm(p: dict) -> dict:
                    pid = int(p.get("id", p.get("index", -1)))
                    if "prob" in p:
                        pr = float(p["prob"])
                    elif "logprob" in p:
                        pr = float(np.exp(float(p["logprob"])))
                    elif "probability" in p:
                        pr = float(p["probability"])
                    else:
                        pr = 0.0
                    return {"id": pid, "prob": pr}
                probs = [norm(p) for p in probs]
            got_top = None
            got_p = None
            if probs:
                best = max(probs, key=lambda p: p["prob"])
                got_top, got_p = int(best["id"]), float(best["prob"])
            # the reference's own prediction after this prefix
            lp = np.exp(logits[L - 1] - logits[L - 1].max())
            lp = lp / lp.sum()
            ref_order = np.argsort(-lp)[:a.top]
            ref_top = int(ref_order[0])
            ref_p = float(lp[ref_top])
            top5 = [[int(i), round(float(lp[i]), 6)] for i in ref_order[:5]]
            rows.append({
                "prefix": L,
                "llama_top1": got_top, "llama_top1_prob": got_p,
                "ref_top1": ref_top, "ref_top1_prob": ref_p,
                "ref_prob_of_llama_top1": float(lp[got_top]) if got_top is not None else None,
                "top1_match": bool(got_top == ref_top),
                "n_probs_returned": len(probs),
                "ref_top5": top5,
                "server_content": (resp.get("content") or "")[:20],
            })
            print(f"PREFIX {L} llama_top1={got_top} p={got_p} ref_top1={ref_top} p={ref_p:.6f} "
                  f"match={rows[-1]['top1_match']} n_probs={len(probs)}", flush=True)
    finally:
        if not a.keep:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            print("SERVER stopped", flush=True)
        err = ""
        if proc.stderr is not None:
            try:
                err = proc.stderr.read()
            except Exception:
                pass
        tail = "\n".join(err.splitlines()[-15:])
        if tail or proc.returncode not in (0, -signal.SIGTERM, None):
            print(f"SERVER rc={proc.returncode}\nSERVER STDERR (tail)\n" + tail, flush=True)

    out = pathlib.Path(a.out) if a.out else None
    if out:
        out.write_text(json.dumps(rows, indent=2) + "\n")
    print("TABLE")
    print(f"{'prefix':>6} {'llama_top1':>11} {'p_llama':>9} {'ref_top1':>9} {'p_ref':>9} {'match':>6} {'|dp|':>9}")
    for r in rows:
        dp = (abs(r["llama_top1_prob"] - r["ref_top1_prob"])
              if r["llama_top1_prob"] is not None else float("nan"))
        print(f"{r['prefix']:>6} {str(r['llama_top1']):>11} "
              f"{(r['llama_top1_prob'] if r['llama_top1_prob'] is not None else float('nan')):>9.6f} "
              f"{r['ref_top1']:>9} {r['ref_top1_prob']:>9.6f} {str(r['top1_match']):>6} {dp:>9.6f}")
    ok = sum(1 for r in rows if r["top1_match"])
    print(f"MATCH {ok}/{len(rows)}")
    sys.exit(0 if ok == len(rows) else 2)


if __name__ == "__main__":
    main()
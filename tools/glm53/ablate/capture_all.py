"""capture_all.py - run derive prompts through http://127.0.0.1:8095/v1/chat/completions (temperature 0, no proxy)
while the engine appends ALL layers' last-token residuals to one append-only capture file: each record is int32 layer
+ 6144 float32.  Before and after every request the script takes os.path.getsize(cap) (0 when absent) and writes the
index: a list of {"tag","start","end","mode"} byte offsets ("closed" too in endthink).

--mode:
  noth      reasoning_effort "none": the prompt ends with the thinking block CLOSED empty (the default render).
  think     reasoning_effort "high", max_tokens 1: the prompt ends on the OPEN thinking marker.
  endthink  reasoning_effort "high", max_tokens 1, chat_template_kwargs {"maya_prefill": thinking + CLOSE}: the model
            thought in an earlier pass, here the prompt is request + that thinking + the closing marker and the
            capture sits on that last marker.  Such prompts come from --think-file (think_gen.py's JSONL).

--limit N runs only the first N prompts (trial).  Defaults: --cap cap-<mode>.bin, --index cap-index-<mode>.json."""
import argparse, json, os, time, urllib.request

O = "/data/glm53/uncensored"
CLOSE = "<" + "/" + "think" + ">"          # what the pack's template closes a thinking block with (serve.THINK_END)

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=("noth", "think", "endthink"), default="noth")
p.add_argument("--cap", default=None)
p.add_argument("--idx", "--index", dest="idx", default=None)
p.add_argument("--limit", type=int, default=0)
p.add_argument("--prompts", default=O + "/derive-prompts.jsonl")
p.add_argument("--think-file", default=O + "/think-gen.jsonl")
a = p.parse_args()
a.cap = a.cap or f"{O}/cap-{a.mode}.bin"
a.idx = a.idx or f"{O}/cap-index-{a.mode}.json"

if a.mode == "endthink":
    rows = [json.loads(l) for l in open(a.think_file)]
else:
    rows = [json.loads(l) for l in open(a.prompts)]
if a.limit > 0:
    rows = rows[:a.limit]

effort = "none" if a.mode == "noth" else "high"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # no proxy, no env
index = []
t0 = time.time()
for i, r in enumerate(rows):
    start = os.path.getsize(a.cap) if os.path.exists(a.cap) else 0
    body = {"model": "glm-5.3", "messages": [{"role": "user", "content": r["prompt"]}], "max_tokens": 1,
            "temperature": 0.0, "reasoning_effort": effort, "stream": False}
    if a.mode == "endthink":
        body["chat_template_kwargs"] = {"maya_prefill": (r.get("thinking") or "").split(CLOSE)[0].rstrip() + CLOSE}
    req = urllib.request.Request("http://127.0.0.1:8095/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = json.loads(opener.open(req, timeout=900).read())
    pn = (resp.get("timings") or {}).get("prompt_n")   # prompt tokens the engine actually ran (the rest was cached)
    # the engine appends its records a moment after the reply: wait until the file stops growing and holds
    # 78 records per prompt token run (when the server says how many), at most 60 s
    R = 4 + 6144 * 4
    want = start + 78 * R * pn if pn else None
    last, still, t1 = -1, 0, time.time()
    while time.time() - t1 < 60:
        end = os.path.getsize(a.cap) if os.path.exists(a.cap) else 0
        still = still + 1 if end == last else 0
        last = end
        if (want is not None and end >= want and still >= 1) or (want is None and end > start and still >= 4):
            break
        time.sleep(0.25)
    ent = {"tag": r["tag"], "start": start, "end": end, "mode": a.mode, "prompt_n": pn,
           "ok": want is None or end == want}
    if a.mode == "endthink":
        ent["closed"] = bool(r.get("closed"))
    index.append(ent)
    with open(a.idx, "w") as f:
        json.dump(index, f)
    if i % 25 == 0:
        print(f"[{a.mode}] {i}/{len(rows)} {end} bytes {time.time()-t0:.0f}s", flush=True)
print(f"CAPTURE DONE mode={a.mode} prompts={len(index)} bytes="
      f"{os.path.getsize(a.cap) if os.path.exists(a.cap) else 0} index={a.idx}", flush=True)
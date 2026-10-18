"""think_gen.py - the thinking pass for the endthink capture: the first 96 H and first 96 B derive prompts, each
asked once with reasoning_effort "high", max_tokens 900, temperature 0, stream off (no proxy), through
http://127.0.0.1:8095/v1/chat/completions.  The model's reasoning comes back as message.reasoning_content
(serve/server.py: openai_collect sets it when the reasoning was non-empty; there is no plain "reasoning" field), the
answer as message.content.  Writes think-gen.jsonl: {"tag","prompt","thinking","closed"} - closed = the thinking block
was closed (the closing marker is in the reasoning, or an answer followed) before max_tokens cut it.
--limit N runs only the first N of those 192 (trial)."""
import argparse, json, os, time, urllib.request

O = "/data/glm53/uncensored"
CLOSE = "<" + "/" + "think" + ">"          # serve.frontend.THINK_END for this pack's template

p = argparse.ArgumentParser()
p.add_argument("--prompts", default=O + "/derive-prompts.jsonl")
p.add_argument("--out", default=O + "/think-gen.jsonl")
p.add_argument("--limit", type=int, default=0)
p.add_argument("--per-class", type=int, default=96)
a = p.parse_args()

rows = [json.loads(l) for l in open(a.prompts)]
picked, counts = [], {"H": 0, "B": 0}
for r in rows:
    t = r.get("tag")
    if t in counts and counts[t] < a.per_class:
        picked.append(r)
        counts[t] += 1
    if all(counts[t] >= a.per_class for t in counts):
        break
if a.limit > 0:
    picked = picked[:a.limit]
print(f"think_gen: {len(picked)} prompts (H={sum(1 for r in picked if r['tag'] == 'H')} "
      f"B={sum(1 for r in picked if r['tag'] == 'B')})", flush=True)

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # no proxy, no env
done, closed_n = [], 0
t0 = time.time()
for i, r in enumerate(picked):
    body = {"model": "glm-5.3", "messages": [{"role": "user", "content": r["prompt"]}], "max_tokens": 900,
            "temperature": 0.0, "reasoning_effort": "high", "stream": False}
    req = urllib.request.Request("http://127.0.0.1:8095/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = json.loads(opener.open(req, timeout=900).read())
    msg = resp["choices"][0]["message"]
    fin = resp["choices"][0].get("finish_reason")
    think = msg.get("reasoning_content") or msg.get("reasoning") or ""
    content = msg.get("content") or ""
    closed = CLOSE in think or bool(content)
    closed_n += int(closed)
    done.append({"tag": r["tag"], "prompt": r["prompt"], "thinking": think, "closed": closed})
    with open(a.out, "w") as f:
        for x in done:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    if i % 16 == 0 or not closed:
        print(f"{i}/{len(picked)} closed={closed} finish={fin} think={len(think)} chars {time.time()-t0:.0f}s",
              flush=True)
print(f"THINK GEN DONE {len(done)} rows, closed={closed_n}, open={len(done) - closed_n}, out={a.out}", flush=True)
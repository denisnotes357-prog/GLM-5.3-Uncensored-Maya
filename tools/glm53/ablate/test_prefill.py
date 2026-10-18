"""test_prefill.py - the GLM-5.3 chat template rendered WITHOUT a model, exactly the way serve/server.py renders it
(ChatTemplate with transformers' Jinja settings, then render_ids' closure of an empty ).  Prints the last 120
characters of the prompt for: a) reasoning_effort none, b) high, c) high + maya_prefill, and checks that
Service.prepare pops maya_prefill out of the template kwargs and keeps it as the answer's prefill."""
import sys
from pathlib import Path

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
from serve.frontend import THINK_END, ChatTemplate, effort_kwargs             # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service                 # noqa: E402

TPL = Path("models/GLM-5.3-Uncensored-Maya/pack/tokenizer/chat_template.jinja")
print(f"template exists: {TPL.exists()} ({TPL.stat().st_size if TPL.exists() else 0} bytes)")
tpl = ChatTemplate(TPL)
print(f"glm_effort template: {tpl.glm_effort}")

msgs = [{"role": "user", "content": "The user asks X."}]
PREFILL = "Hmm, the user asks X." + THINK_END   # thinking over the request, then the end-of-thinking marker


THINK_START = "<" + "think" + ">"     # what the prompt ends with when the template opened a thinking block


def tail(label, kwargs, suffix=""):
    out = tpl.render([dict(m) for m in msgs], **dict(kwargs))
    if kwargs.get("enable_thinking") is False and out.endswith(THINK_START):   # render_ids closes the open block
        out += THINK_END
    out = out + suffix
    print(f"{label}")
    print(f"  kwargs={kwargs}")
    print(f"  suffix={suffix!r}")
    print(f"  len={len(out)} tail120={out[-120:]!r}")


a = dict(effort_kwargs("none"))
b = dict(effort_kwargs("high"))
tail("a) reasoning_effort none", a)
tail("b) reasoning_effort high", b)
tail("c) reasoning_effort high + maya_prefill", b, suffix=PREFILL)

# the same path the server takes: Service.prepare with maya_prefill in the template kwargs
tok = ByteTokenizer()
svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(TPL))
kw = {"reasoning_effort": "high", "maya_prefill": PREFILL}
ids, thinking, max_new = svc.prepare([dict(m) for m in msgs], None, kw, 0)
print("prepare():")
print(f"  thinking={thinking} max_new={max_new} ids={len(ids)}")
print(f"  kw keys={sorted(kw)}")
print(f"  kw['_prefill']={kw.get('_prefill')!r}")
print(f"  maya_prefill still in kw: {'maya_prefill' in kw}")
print("TEST PREFILL DONE")
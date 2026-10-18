#!/usr/bin/env python3
"""tools/glm53/make_gm24u.py - G-M24U-GPTQ: G-M24A-GPTQ with the refusal-edited tensors of a weight-level uncensored donor.

The donor, HF dealignai/GLM-5.3-UNCENSORED-FP8 (a permanent bf16 edit of zai-org/GLM-5.3's "residual writers", no
LoRA, no runtime hooks), differs from the official FP8 checkpoint in exactly two tensor kinds a layer:
self_attn.o_proj and mlp.shared_experts.down_proj - checked byte for byte on layers 3 / 40 / 77: every routed
expert, q/kv projection and shared gate/up is identical.  Those 155 tensors (blk.0-78 attn_output, blk.3-78
ffn_down_shexp; FP8 e4m3 with 128 x 128 block scales, 8.9 GB) were fetched by HTTP range (donor/fetch.py ->
/data/models/donor-glm53-uncensored), so the 756 GB checkpoint never had to be downloaded.

This script rebuilds G-M24A with them in place:

  * every other tensor is COPIED byte for byte out of G-M24A-GPTQ (the GPTQ experts, the indexers, the routers,
    the Q8_0 A100 layers);
  * each donor tensor is dequantised (w * scale_inv, as calib.py does) and rounded to the SAME ggml type the source
    holds for it (Q4_K on blk.10-69, Q8_0 on the A100 layers 0-9 / 70-78) with the same importance vector
    quantize.py used (calib.py's statistics; blk.78 has none and rounds plain) - so the only difference between
    G-M24A and G-M24U is the donor's edit.

The container is the source's: the same KV block (maya.rounding reworded), tensor order, types, sizes and offsets.

    python make_gm24u.py [--dry-run] [--threads 20] [--src DIR] [--donor DIR] [--stats DIR] [--out DIR]
        [--only-shard N] [--no-check]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_gm24a as mk  # noqa: E402  (Shard / TInfo / header_of / copy_into / pad / shards_of / GGML_LIB)
import quantize_rtn as qr  # noqa: E402
from quantize import Imx, Stats  # noqa: E402

gguf = mk.gguf
QT = mk.QT
SRC = pathlib.Path("/data/models/GLM-5.3-Maya-G-M24A-GPTQ")
SRC_STEM = "GLM-5.3-Maya-G-M24A-GPTQ"
DONOR = pathlib.Path("/data/models/donor-glm53-uncensored")
STATS = pathlib.Path("/data/glm53/calib/stats")
OUT = pathlib.Path("/data/models/GLM-5.3-Maya-G-M24U-GPTQ")
STEM = "GLM-5.3-Maya-G-M24U-GPTQ"
DONOR_REPO = "dealignai/GLM-5.3-UNCENSORED-FP8"

GGUF_OF = {"self_attn.o_proj.weight": "attn_output.weight",
           "mlp.shared_experts.down_proj.weight": "ffn_down_shexp.weight"}


def donor_map(donor: pathlib.Path) -> tuple[dict, dict[str, str]]:
    """meta.json, and GGUF name -> donor tensor name for the edited kinds."""
    meta = json.load(open(donor / "meta.json"))
    out = {}
    for name in meta:
        m = re.match(r"model\.layers\.(\d+)\.(.+)$", name)
        if m and m.group(2) in GGUF_OF:
            out[f"blk.{m.group(1)}.{GGUF_OF[m.group(2)]}"] = name
    return meta, out


def dequant(donor: pathlib.Path, meta: dict, name: str) -> np.ndarray:
    """the donor's FP8 e4m3 tensor x its 128 x 128 block scales -> float32 [out, in] (torch's layout = GGUF's rows)"""
    m, ms = meta[name], meta[name + "_scale_inv"]
    if m["dtype"] != "F8_E4M3" or ms["dtype"] != "F32":
        raise SystemExit(f"{name}: unexpected dtypes {m['dtype']} / {ms['dtype']}")
    w = torch.frombuffer(bytearray((donor / f"{name}.bin").read_bytes()), dtype=torch.float8_e4m3fn)
    w = w.reshape(m["shape"]).float()
    s = torch.frombuffer(bytearray((donor / f"{name}_scale_inv.bin").read_bytes()), dtype=torch.float32)
    s = s.reshape(ms["shape"])
    s = s.repeat_interleave(128, 0)[: w.shape[0]].repeat_interleave(128, 1)[:, : w.shape[1]]
    out = (w * s).numpy()
    if not np.isfinite(out).all():
        raise SystemExit(f"{name}: non-finite values after dequantisation")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(SRC), help="byte source of every unchanged tensor (read only)")
    ap.add_argument("--donor", default=str(DONOR), help="the donor's edited tensors (donor/fetch.py)")
    ap.add_argument("--stats", default=str(STATS), help="calib.py's statistics (the importance vectors)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    ap.add_argument("--only-shard", type=int, default=None, help="write one shard (1-based), for testing")
    ap.add_argument("--no-check", action="store_true", help="skip the dequantise-and-compare of each written tensor")
    a = ap.parse_args()

    src_dir, donor, out_dir = pathlib.Path(a.src), pathlib.Path(a.donor), pathlib.Path(a.out)
    srcs = [mk.Shard(p, SRC_STEM) for p in mk.shards_of(src_dir, SRC_STEM)]
    print(f"src: {len(srcs)} shards, {sum(len(s.tensors) for s in srcs)} tensors, "
          f"{sum(t.nbytes for s in srcs for t in s.tensors) / 1e9:.3f} GB", flush=True)
    meta, dmap = donor_map(donor)
    src_by_name = {t.name: (s, t) for s in srcs for t in s.tensors}
    missing = [n for n in dmap if n not in src_by_name]
    if missing:
        raise SystemExit(f"donor tensors absent from the source: {missing[:5]} ...")
    changed = {n: src_by_name[n][1] for n in dmap}
    for n, ti in changed.items():
        fs = tuple(ti.float_shape)
        ds = tuple(meta[dmap[n]]["shape"])
        if fs != ds:
            raise SystemExit(f"{n}: GGUF float shape {fs} != donor {ds}")
        p = donor / f"{dmap[n]}.bin"
        if not p.exists() or p.stat().st_size != meta[dmap[n]]["bytes"]:
            raise SystemExit(f"{n}: donor bytes not fetched ({p})")
    kinds: dict[str, int] = {}
    for n, ti in changed.items():
        k = f"{n.split('.', 2)[2]} {ti.qt.name}"
        kinds[k] = kinds.get(k, 0) + 1
    print(f"donor {DONOR_REPO}: {len(changed)} tensors replaced, types kept: {kinds}", flush=True)
    if a.dry_run:
        print("(dry run, nothing written)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    stats = Stats(a.stats)
    imx = Imx(mk.GGML_LIB, a.threads, stats, lambda: None)
    kv_patches = {
        "maya.rounding": (f"gptq prior=65536.0 damp=0.1 (copied experts); the A100 dense tensors Q8_0 from G-L RTN Q6_K; "
                          f"attn_output (blk.0-78) and ffn_down_shexp (blk.3-78) from {DONOR_REPO} (FP8 dequantised, "
                          f"the same type and importance vector as the source) - the refusal-removed residual writers"),
    }
    n_im = n_plain = 0
    worst = (0.0, "")
    for i, s in enumerate(srcs, 1):
        if a.only_shard and i != a.only_shard:
            continue
        plan = [mk.TInfo(t.name, t.dtype, t.ne, t.nbytes, t.src_off) for t in s.tensors]
        present = {kp.key for kp in s.kv_pairs}
        hdr, offs, end = mk.header_of(s, plan, {k: v for k, v in kv_patches.items() if k in present})
        # the reworded maya.rounding moves the data offset by a few alignment steps; the payload itself must not change
        if end - len(hdr) != s.path.stat().st_size - s.data_offset:
            raise SystemExit(f"{s.path.name}: payload {end - len(hdr)} vs the source's {s.path.stat().st_size - s.data_offset}"
                             f" - types changed?")
        todo = [t for t in plan if t.name in changed]
        name = out_dir / f"{STEM}-{i:05d}-of-{len(srcs):05d}.gguf"
        print(f"shard {i}/{len(srcs)}: {len(plan)} tensors, {len(todo)} from the donor -> {name.name}", flush=True)
        t0 = time.time()
        with open(name, "wb") as fo:
            fo.write(hdr)
            for k, ti in enumerate(plan, 1):
                if ti.name in changed:
                    fl = dequant(donor, meta, dmap[ti.name])
                    rows, n = fl.shape
                    im = stats.get(ti.name, (rows, n))
                    if im is not None:
                        q = imx.eager_im(fl, ti.qt, im)
                        n_im += 1
                        how = "imatrix"
                    else:
                        q = imx.eager(fl, ti.qt)
                        n_plain += 1
                        how = "plain"
                    q = np.ascontiguousarray(q).reshape(-1)
                    if q.nbytes != ti.nbytes:
                        raise SystemExit(f"{ti.name}: quantised to {q.nbytes} bytes, the plan says {ti.nbytes}")
                    fo.write(q.tobytes())
                    note = ""
                    if not a.no_check:
                        bs, ts = gguf.GGML_QUANT_SIZES[ti.qt]
                        back = gguf.quants.dequantize(q.reshape(rows, n // bs * ts), ti.qt)
                        rel = float(np.linalg.norm(back - fl) / max(1e-12, np.linalg.norm(fl)))
                        note = f" rel {rel:.4f}"
                        if rel > worst[0]:
                            worst = (rel, ti.name)
                    print(f"    [{k}/{len(plan)}] {ti.qt.name} {how} {ti.name} {ti.nbytes / 1e6:8.1f} MB{note} "
                          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
                    del fl, q
                else:
                    mk.copy_into(fo, s, ti)
                    if k % 200 == 0:
                        print(f"    [{k}/{len(plan)}] copy ... ({(time.time() - t0) / 60:.1f} min)", flush=True)
                p = mk.pad(ti.nbytes, s.align)
                if p != ti.nbytes:
                    fo.write(bytes(p - ti.nbytes))
        size = name.stat().st_size
        if size != end:
            raise SystemExit(f"{name}: wrote {size}, planned {end}")
        print(f"  {name.name} {size / 1e9:.3f} GB in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"donor tensors: {n_im} with an importance vector, {n_plain} plain; worst rel error {worst[0]:.4f} ({worst[1]})")
    print("MAKE DONE", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""tools/glm53/check_gm24a.py - G-M24A-GPTQ against the two tensors' sources.

FP8 is gone (deleted 2026-10-16 10:18), so the old check_quant.py comparison cannot run.  What has to hold for
the A100-Q8_0 rebuild is this instead, and it is what this script checks, tensor by tensor, in every shard:

  * CHANGED tensor (the 153 the recipe turned into Q8_0):
        rel = || Q8_0 - dequant(G-L RTN Q6_K) || / || dequant(G-L RTN Q6_K) ||
    i.e. only the rounding error of Q8_0 on the finer Q6_K weights: about 0.005-0.007, ceiling 0.01.
  * UNCHANGED tensor: byte for byte IDENTICAL to /data/models/GLM-5.3-Maya-G-M24-GPTQ (experts, indexers,
    routers, blk.10-69 included).
  * container: the tensor table has the source's names, order and shapes, and the only KV values that differ
    from the source are maya.recipe / maya.rounding.

    python check_gm24a.py /data/models/GLM-5.3-Maya-G-M24A-GPTQ [--threads 20]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_gm24a as M  # noqa: E402
import gguf  # noqa: E402

QT = M.QT
CEIL = {"Q8_0": 0.01}
CHUNK = 64 << 20


def rel_vs(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / nb) if nb else float(np.linalg.norm(a))


def read(shard: M.Shard, ti: M.TInfo) -> np.ndarray:
    with open(shard.path, "rb") as f:
        f.seek(shard.offsets[ti.name])
        b = f.read(ti.nbytes)
    if len(b) != ti.nbytes:
        raise SystemExit(f"{shard.path.name}: {ti.name} is {len(b)} bytes, table says {ti.nbytes}")
    return np.frombuffer(b, dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", default=str(M.OUT))
    ap.add_argument("--src", default=str(M.SRC))
    ap.add_argument("--gl", default=str(M.GL))
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--limit-bytes", type=int, default=0, help="compare only tensors up to this many bytes")
    a = ap.parse_args()

    outs = [M.Shard(p, M.STEM) for p in M.shards_of(pathlib.Path(a.out), M.STEM)]
    srcs = [M.Shard(p) for p in M.shards_of(pathlib.Path(a.src), M.SRC_STEM)]
    gls = [M.Shard(p, M.GL_STEM) for p in M.shards_of(pathlib.Path(a.gl), M.GL_STEM)]
    src_map = {t.name: (s, t) for s in srcs for t in s.tensors}
    gl_map = {t.name: (s, t) for s in gls for t in s.tensors}
    print(f"out {len(outs)} shards / src {len(srcs)} / gl {len(gls)}", flush=True)

    n_changed = n_same = n_skip = bad = 0
    worst: list[tuple[float, str]] = []
    for oi, o in enumerate(outs):
        s = srcs[oi]
        # container: same tensor set, order and shapes; only the changed types/sizes differ
        if [t.name for t in o.tensors] != [t.name for t in s.tensors]:
            raise SystemExit(f"{o.path.name}: tensor order differs from {s.path.name}")
        if [(tuple(t.ne), ) for t in o.tensors] != [(tuple(t.ne),) for t in s.tensors]:
            raise SystemExit(f"{o.path.name}: a tensor's dims differ from the source")
        ok_kv = {kp.key for kp in o.kv_pairs} == {kp.key for kp in s.kv_pairs}
        s_vals = {x.key: x.value for x in s.kv_pairs}
        diff_kv = [kp.key for kp in o.kv_pairs if kp.value != s_vals[kp.key]]
        want_diff = sorted(k for k in ("maya.recipe", "maya.rounding") if k in s_vals)
        if not ok_kv or sorted(diff_kv) != want_diff:
            raise SystemExit(f"{o.path.name}: KV keys changed {sorted(diff_kv)}, want {want_diff}")
        print(f"{o.path.name}: order/shapes/KV ok (KV values changed: {diff_kv})", flush=True)

        for ti in o.tensors:
            ssh, st = src_map[ti.name]
            if ti.qt == st.qt and ti.nbytes == st.nbytes:
                n_same += 1
                if a.limit_bytes and ti.nbytes > a.limit_bytes:
                    n_skip += 1
                    continue
                # byte-for-byte with the G-M24-GPTQ original
                new, old = read(o, ti), read(ssh, st)
                if not np.array_equal(new, old):
                    print(f" FAIL {ti.name}: bytes differ from G-M24-GPTQ ({ti.qt.name})")
                    bad += 1
                continue
            # changed: Q8_0 of the same weight as G-L RTN Q6_K
            n_changed += 1
            if ti.qt != QT.Q8_0:
                print(f" FAIL {ti.name}: {st.qt.name} -> {ti.qt.name}, expected Q8_0")
                bad += 1
                continue
            gsh, gt = gl_map[ti.name]
            rows = 1
            for d in ti.ne[1:]:
                rows *= d
            n = ti.ne[0]
            rb_src = n // gguf.GGML_QUANT_SIZES[gt.qt][0] * gguf.GGML_QUANT_SIZES[gt.qt][1]
            rb_new = n // 32 * 34          # Q8_0
            with open(o.path, "rb") as f, open(gsh.path, "rb") as g:
                f.seek(o.offsets[ti.name])
                g.seek(gsh.offsets[gt.name])
                q = np.frombuffer(f.read(ti.nbytes), np.uint8).reshape(rows, rb_new)
                r = np.frombuffer(g.read(gt.nbytes), np.uint8).reshape(rows, rb_src)
            dq = gguf.quants.dequantize(r, gt.qt)
            dq2 = gguf.quants.dequantize(q, QT.Q8_0)
            rel = rel_vs(dq2, dq)
            worst.append((rel, ti.name))
            ok = rel <= CEIL["Q8_0"]
            bad += not ok
            print(f"{' ok ' if ok else 'FAIL'}  {ti.name:44s} Q8_0 vs G-L Q6_K rel {rel:.5f}", flush=True)
    print(f"{n_changed} changed tensors checked against G-L RTN, {n_same} unchanged tensors byte-compared "
          f"with G-M24-GPTQ ({n_skip} skipped by --limit-bytes), {bad} problems")
    if worst:
        worst.sort(reverse=True)
        print("worst changed:", ", ".join(f"{n} {r:.5f}" for r, n in worst[:5]))
        print("best  changed:", ", ".join(f"{n} {r:.5f}" for r, n in worst[-3:]))
    print("CHECK PASS" if not bad else "CHECK FAIL")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
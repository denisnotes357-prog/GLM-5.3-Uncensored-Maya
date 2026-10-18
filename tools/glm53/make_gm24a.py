#!/usr/bin/env python3
"""tools/glm53/make_gm24a.py - G-M24A-GPTQ: the A100 layers of G-M24-GPTQ in Q8_0.

The 7-way split puts blk.[0,10) on GPU5 and blk.[70,78) + blk.78 + output on GPU6, and those two A100s are bound
by the *count* of K/IQ dequant steps, not by bandwidth (a 4090 clocks ~2x higher), while Q8_0 unpacks as a plain
byte stream (~1.55 TB/s on A100).  The recipe (recipes/g-m24a.json, first matching rule wins) therefore turns the
dense 2-D tensors of blk.0-9 / blk.70-78 plus output.weight into Q8_0; the routed experts, the indexers, the
routers and blk.10-69 keep G-M24 exactly.

Nothing is re-quantised from scratch and GPTQ is NOT re-run:

  * every tensor the new recipe types exactly as the source GGUF is COPIED byte for byte out of
    /data/models/GLM-5.3-Maya-G-M24-GPTQ (experts included - they carry the GPTQ rounding);
  * every tensor whose type changed is taken from /data/models/GLM-5.3-Maya-G-L-RTN, where all dense 2-D
    tensors are Q6_K (finer than the Q4_K G-M24 held), dequantised to float with gguf-py and rounded to Q8_0 by
    ggml (ggml_quantize_chunk, no imatrix - the a100q8-fix correction says Q8_0 needs none).

The output keeps the source's metadata (the KV block, with only maya.recipe / maya.rounding reworded), tensor
order and shard assignment; only the type/size/position of the changed tensors differ.

    python make_gm24a.py [--check-template] [--dry-run] [--threads 20]
        [--src DIR] [--gl DIR] [--out DIR] [--recipe recipes/g-m24a.json] [--only-shard N]

--check-template rebuilds the six headers from the SOURCE plan and compares them byte for byte with the source
files, and checks that the payload plan lands exactly on each file's size: the proof that this writer produces
the same container as the converter did.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import struct
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
LLAMA_CPP = pathlib.Path("/data/glm53/llama.cpp")
GGML_LIB = str(LLAMA_CPP / "build/bin/libggml-base.so")
sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))
sys.path.insert(0, str(LLAMA_CPP))
sys.path.insert(0, str(HERE))

import gguf  # noqa: E402
import quantize_rtn as qr  # noqa: E402  (load_recipe / pick_type / Rtg - the converter's own recipe logic)

QT = gguf.GGMLQuantizationType
SRC = pathlib.Path("/data/models/GLM-5.3-Maya-G-M24-GPTQ")
GL = pathlib.Path("/data/models/GLM-5.3-Maya-G-L-RTN")
OUT = pathlib.Path("/data/models/GLM-5.3-Maya-G-M24A-GPTQ")
RECIPE = HERE / "recipes/g-m24a.json"
STEM = "GLM-5.3-Maya-G-M24A-GPTQ"
SRC_STEM = "GLM-5.3-Maya-G-M24-GPTQ"
GL_STEM = "GLM-5.3-Maya-G-L-RTN"

_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}  # GGUFValueType -> bytes
_VT_STR, _VT_ARR = 8, 9


def _u64s(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def pad(n: int, align: int) -> int:
    return (n + align - 1) // align * align


class KVPair:
    __slots__ = ("key", "vtype", "value")

    def __init__(self, key: str, vtype: int, value: bytes):
        self.key, self.vtype, self.value = key, vtype, value


class TInfo:
    """one tensor of the container: name, ggml dims (dim0 = the quantised row), type, bytes"""

    __slots__ = ("name", "dtype", "ne", "nbytes", "src_off")

    def __init__(self, name: str, dtype: int, ne: tuple[int, ...], nbytes: int, src_off: int):
        self.name, self.dtype, self.ne, self.nbytes, self.src_off = name, dtype, ne, nbytes, src_off

    @property
    def qt(self) -> QT:
        return QT(self.dtype)

    @property
    def float_shape(self) -> tuple[int, ...]:
        return tuple(reversed(self.ne))

    def packed(self, off: int) -> bytes:
        b = bytearray(_u64s(self.name))
        b += struct.pack("<I", len(self.ne))
        for d in self.ne:
            b += struct.pack("<Q", d)
        b += struct.pack("<I", self.dtype)
        b += struct.pack("<Q", off)
        return bytes(b)


def nbytes_of(ti: TInfo) -> int:
    rows = 1
    for d in ti.ne[1:]:
        rows *= d
    n = ti.ne[0]
    if ti.qt in (QT.F32, QT.F16, QT.BF16):
        return rows * n * (4 if ti.qt == QT.F32 else 2)
    bs, ts = gguf.GGML_QUANT_SIZES[ti.qt]
    if n % bs:
        raise SystemExit(f"{ti.name}: row {n} is not a whole {ti.qt.name} block")
    return rows * (n // bs * ts)


class Shard:
    """one GGUF file's header: the KV list and the tensor table, plus where each tensor's bytes live."""

    def __init__(self, path: pathlib.Path, stem: str | None = None):
        self.path = pathlib.Path(path)
        self.stem = stem or self.path.name.split("-00001-of-")[0]
        self.kv_pairs: list[KVPair] = []
        with open(self.path, "rb") as f:
            hdr = f.read(24)
            if hdr[:4] != b"GGUF":
                raise SystemExit(f"{self.path}: not a GGUF file")
            self.version = struct.unpack_from("<I", hdr, 4)[0]
            n_tensors, self.n_kv = struct.unpack_from("<QQ", hdr, 8)
            kv = bytearray()
            for _ in range(self.n_kv):
                ln = struct.unpack("<Q", f.read(8))[0]
                key = f.read(ln).decode()
                vtype = struct.unpack("<I", f.read(4))[0]
                vstart = f.tell()
                vend = self._skip_value(f, vtype)
                f.seek(vstart)
                value = f.read(vend - vstart)
                kv += _u64s(key) + struct.pack("<I", vtype) + value
                self.kv_pairs.append(KVPair(key, vtype, value))
            self.tensors: list[TInfo] = []
            for _ in range(n_tensors):
                ln = struct.unpack("<Q", f.read(8))[0]
                name = f.read(ln).decode()
                nd = struct.unpack("<I", f.read(4))[0]
                ne = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(nd))
                dtype = struct.unpack("<I", f.read(4))[0]
                off = struct.unpack("<Q", f.read(8))[0]
                ti = TInfo(name, dtype, ne, 0, off)
                ti.nbytes = nbytes_of(ti)
                self.tensors.append(ti)
            ti_end = f.tell()
            self.kv_bytes = bytes(kv)
            self.align = 32
            for kp in self.kv_pairs:
                if kp.key == "general.alignment" and kp.vtype == 5:
                    self.align = struct.unpack("<I", kp.value)[0]
            self.header_len = ti_end
            self.data_offset = pad(ti_end, self.align)
            # payload cursor: the table's offsets must be the padded sizes we recompute
            cur = self.data_offset
            self.offsets: dict[str, int] = {}
            for ti in self.tensors:
                if cur - self.data_offset != ti.src_off:
                    raise SystemExit(f"{self.path}: {ti.name}: table offset {ti.src_off} vs planned "
                                     f"{cur - self.data_offset} - layout not as expected")
                self.offsets[ti.name] = cur
                cur = pad(cur + ti.nbytes, self.align)
            if cur != self.path.stat().st_size:
                raise SystemExit(f"{self.path}: layout ends at {cur}, file is {self.path.stat().st_size}")

    @staticmethod
    def _skip_value(f, vtype: int) -> int:
        if vtype == _VT_ARR:
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            if et == _VT_STR:
                for _ in range(n):
                    ln = struct.unpack("<Q", f.read(8))[0]
                    f.seek(ln, 1)
            else:
                f.seek(n * _FIXED[et], 1)
        elif vtype == _VT_STR:
            ln = struct.unpack("<Q", f.read(8))[0]
            f.seek(ln, 1)
        else:
            f.seek(_FIXED[vtype], 1)
        return f.tell()

    def kv_patch(self, patches: dict[str, str]) -> bytes:
        out = bytearray()
        seen = set()
        for kp in self.kv_pairs:
            val = kp.value
            if kp.key in patches:
                seen.add(kp.key)
                if kp.vtype != _VT_STR:
                    raise SystemExit(f"{kp.key} is not a GGUF string")
                s = patches[kp.key].encode()
                val = struct.pack("<Q", len(s)) + s
            out += _u64s(kp.key) + struct.pack("<I", kp.vtype) + val
        if set(patches) - seen:
            raise SystemExit(f"KV to patch not found: {set(patches) - seen}")
        return bytes(out)


def shards_of(root: pathlib.Path, stem: str) -> list[pathlib.Path]:
    ps = sorted(root.glob(f"{stem}-*-of-*.gguf"))
    if not ps:
        raise SystemExit(f"{root}: no {stem} shards")
    return ps


# ===== payloads =====
def to_q8_from_q6(shard: Shard, ti: TInfo, rtg: qr.Rtg, chunk_target: int = 192 << 20) -> np.ndarray:
    """G-L Q6_K bytes -> float32 -> ggml Q8_0, in row chunks (a 780 MB tensor stays ~1 GB in RAM)."""
    if ti.qt != QT.Q6_K:
        raise SystemExit(f"{ti.name}: {shard.path.parent.name} holds {ti.qt.name}, expected Q6_K")
    rows = 1
    for d in ti.ne[1:]:
        rows *= d
    n = ti.ne[0]
    bs, ts = gguf.GGML_QUANT_SIZES[ti.qt]
    rb = n // bs * ts
    chunk = max(16, chunk_target // max(1, rb))
    out = None
    with open(shard.path, "rb") as f:
        base = shard.offsets[ti.name]
        for r0 in range(0, rows, chunk):
            r1 = min(rows, r0 + chunk)
            f.seek(base + r0 * rb)
            raw = np.frombuffer(f.read((r1 - r0) * rb), dtype=np.uint8).reshape(r1 - r0, rb)
            fl = gguf.quants.dequantize(raw, ti.qt)
            if fl.shape != (r1 - r0, n):
                raise SystemExit(f"{ti.name}: dequant gave {fl.shape}, want {(r1 - r0, n)}")
            q = rtg.eager(fl, QT.Q8_0)
            if out is None:
                out = np.empty((rows, q.shape[1]), dtype=np.uint8)
            out[r0:r1] = q
    return out.reshape(-1) if out is not None else np.empty(0, np.uint8)


def copy_into(fo, shard: Shard, ti: TInfo, chunk: int = 64 << 20) -> None:
    left, written = ti.nbytes, 0
    with open(shard.path, "rb") as f:
        f.seek(shard.offsets[ti.name])
        while left:
            b = f.read(min(chunk, left))
            if not b:
                raise SystemExit(f"{shard.path}: {ti.name} ended {left} bytes early")
            fo.write(b)
            left -= len(b)
            written += len(b)
    if written != ti.nbytes:
        raise SystemExit(f"{ti.name}: copied {written}, plan {ti.nbytes}")


# ===== writer =====
def plan_offsets(plan: list[TInfo], data_offset: int, align: int) -> tuple[dict[str, int], int]:
    offs, cur = {}, data_offset
    for ti in plan:
        offs[ti.name] = cur - data_offset
        cur = pad(cur + ti.nbytes, align)
    return offs, cur


def header_of(shard: Shard, plan: list[TInfo], kv_patches: dict[str, str]) -> tuple[bytes, dict[str, int], int]:
    kv = shard.kv_patch(kv_patches)
    hdr = b"GGUF" + struct.pack("<I", shard.version) + struct.pack("<QQ", len(plan), shard.n_kv)
    ti_len = sum(len(t.packed(0)) for t in plan)
    data_offset = pad(len(hdr) + len(kv) + ti_len, shard.align)
    offs, end = plan_offsets(plan, data_offset, shard.align)
    ti_data = bytes().join(t.packed(offs[t.name]) for t in plan)
    assert len(ti_data) == ti_len
    return hdr + kv + ti_data + bytes(data_offset - len(hdr) - len(kv) - len(ti_data)), offs, end


def check_template(srcs: list[Shard]) -> None:
    """rebuild every header from the SOURCE plan and diff it against the source file (container proof)."""
    bad = 0
    for s in srcs:
        plan = [TInfo(t.name, t.dtype, t.ne, t.nbytes, t.src_off) for t in s.tensors]
        hdr, offs, end = header_of(s, plan, {})
        with open(s.path, "rb") as f:
            want = f.read(len(hdr))
        size = s.path.stat().st_size
        ok = hdr == want and end == size
        bad += not ok
        print(f"  {s.path.name}: header+kv+table {'IDENTICAL' if hdr == want else 'DIFFERS'}, plan end {end} "
              f"vs file {size} -> {'ok' if ok else 'FAIL'}", flush=True)
    if bad:
        raise SystemExit("template check FAILED: the builder would not reproduce the source container")
    print("TEMPLATE OK", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(SRC), help="byte source of every unchanged tensor (read only)")
    ap.add_argument("--gl", default=str(GL), help="Q6_K dense weights for the changed tensors (read only)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--recipe", default=str(RECIPE))
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--check-template", action="store_true",
                    help="rebuild the six headers from the source plan and diff them with the source files")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and size delta, write nothing")
    ap.add_argument("--only-shard", type=int, default=None, help="write one shard (1-based), for testing")
    a = ap.parse_args()

    src_dir, gl_dir, out_dir = pathlib.Path(a.src), pathlib.Path(a.gl), pathlib.Path(a.out)
    srcs = [Shard(p) for p in shards_of(src_dir, SRC_STEM)]
    print(f"src: {len(srcs)} shards, {sum(len(s.tensors) for s in srcs)} tensors, "
          f"{sum(t.nbytes for s in srcs for t in s.tensors) / 1e9:.3f} GB", flush=True)
    if a.check_template:
        check_template(srcs)
        return

    rules, default = qr.load_recipe(pathlib.Path(a.recipe))
    src_by_name = {t.name: (s, t) for s in srcs for t in s.tensors}

    plans: list[tuple[Shard, list[TInfo]]] = []
    changed: dict[str, tuple[TInfo, TInfo]] = {}  # name -> (old, new)
    for s in srcs:
        plan = []
        for ti in s.tensors:
            nt = new_type(ti, rules, default)
            t2 = TInfo(ti.name, int(nt), ti.ne, 0, ti.src_off)
            t2.nbytes = nbytes_of(t2)
            plan.append(t2)
            if nt != ti.qt or t2.nbytes != ti.nbytes:
                changed[ti.name] = (ti, t2)
        plans.append((s, plan))
    print(f"recipe {a.recipe}: {len(changed)} tensors change", flush=True)

    gls = [Shard(p, GL_STEM) for p in shards_of(gl_dir, GL_STEM)]
    gl_map = {t.name: (s, t) for s in gls for t in s.tensors}
    old_tot = sum(o.nbytes for o, _ in changed.values())
    new_tot = sum(n.nbytes for _, n in changed.values())
    parts: dict[str, float] = {}
    for name, (o, n) in changed.items():
        if name not in gl_map:
            raise SystemExit(f"{name}: absent from {gl_dir} (the correction expects every changed tensor in G-L)")
        gsh, gt = gl_map[name]
        if gt.qt != QT.Q6_K:
            raise SystemExit(f"{name}: {gt.qt.name} in G-L RTN, expected Q6_K")
        if gt.float_shape != o.float_shape:
            raise SystemExit(f"{name}: G-L shape {gt.float_shape} != G-M24 {o.float_shape}")
        m = re.match(r"blk\.(\d+)\.", name)
        part = (f"GPU5 blk.[0,10)" if m and 0 <= int(m.group(1)) <= 9 else
                f"GPU6 blk.[70,78)+draft" if m and 70 <= int(m.group(1)) <= 78 else
                "GPU6 output head" if name == "output.weight" else "OTHER")
        parts[part] = parts.get(part, 0.0) + (n.nbytes - o.nbytes)
    print(f"  {old_tot / 1e9:.3f} GB -> {new_tot / 1e9:.3f} GB  (delta {(new_tot - old_tot) / 1e9:+.3f} GB)")
    for k, v in sorted(parts.items()):
        print(f"    {k}: +{v / 1e9:.3f} GB")
    for name in sorted(changed):
        o, n = changed[name]
        print(f"    {name:44s} {o.qt.name:5s} -> {n.qt.name:5s} {o.nbytes / 1e6:9.2f} -> {n.nbytes / 1e6:9.2f} MB")
    for s, plan in plans:
        _, end = plan_offsets(plan, s.data_offset, s.align)
        print(f"  shard {s.path.name}: {sum(t.nbytes for t in plan) / 1e9:.3f} GB -> end {end} "
              f"(source {s.path.stat().st_size})")
    if a.dry_run:
        print("(dry run, nothing written)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    rtg = qr.Rtg(GGML_LIB, a.threads)
    kv_patches = {
        "maya.recipe": json.dumps(json.load(open(a.recipe))),
        "maya.rounding": ("gptq prior=65536.0 damp=0.1 (copied experts); the A100 dense tensors are ggml Q8_0 "
                          "rounded from the same weights in G-L RTN Q6_K"),
    }
    for i, (s, plan) in enumerate(plans, 1):
        if a.only_shard and i != a.only_shard:
            continue
        todo = [t for t in plan if t.name in changed]
        present = {kp.key for kp in s.kv_pairs}
        patches = {k: v for k, v in kv_patches.items() if k in present}
        hdr, offs, end = header_of(s, plan, patches)
        print(f"shard {i}/{len(plans)}: {len(plan)} tensors, {len(todo)} re-quantised, "
              f"{end / 1e9:.3f} GB -> {STEM}-{i:05d}-of-{len(plans):05d}.gguf", flush=True)
        name = out_dir / f"{STEM}-{i:05d}-of-{len(plans):05d}.gguf"
        t0 = time.time()
        with open(name, "wb") as fo:
            fo.write(hdr)
            for k, ti in enumerate(plan, 1):
                if ti.name in changed:
                    gsh, gt = gl_map[ti.name]
                    payload = to_q8_from_q6(gsh, gt, rtg)
                    if payload.nbytes != ti.nbytes:
                        raise SystemExit(f"{ti.name}: wrote {payload.nbytes}, plan says {ti.nbytes}")
                    fo.write(payload)
                    del payload
                    kind = "q8_0"
                else:
                    copy_into(fo, s, ti)
                    kind = "copy"
                p = pad(ti.nbytes, s.align)
                if p != ti.nbytes:
                    fo.write(bytes(p - ti.nbytes))
                if kind == "q8_0" or k % 200 == 0:
                    print(f"    [{k}/{len(plan)}] {kind} {ti.name} {ti.nbytes / 1e6:9.1f} MB "
                          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        size = name.stat().st_size
        if size != end:
            raise SystemExit(f"{name}: wrote {size}, planned {end}")
        print(f"  {name.name} {size / 1e9:.3f} GB in {(time.time() - t0) / 60:.1f} min", flush=True)
    print("MAKE DONE", flush=True)


def new_type(ti: TInfo, rules, default) -> QT:
    """the converter's decision, replayed: the recipe, or its own forced F32 for 1-D / norms."""
    if len(ti.ne) <= 1 or ti.name.endswith("_norm.weight"):
        return ti.qt
    return qr.pick_type(ti.name, ti.float_shape, rules, default)


if __name__ == "__main__":
    main()
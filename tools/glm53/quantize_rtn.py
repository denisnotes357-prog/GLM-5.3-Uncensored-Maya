#!/usr/bin/env python3
"""tools/glm53/quantize_rtn.py - GLM-5.3 FP8 -> GGUF without calibration (round-to-nearest, RTN).

The model is converted by llama.cpp's own converter (conversion/glm.py's GlmMoeDsaModel) with the default file
type ALL_F32, so every tensor the recipe does not cover stays F32 (1-D tensors and *_norm.weight always do -
the converter forces those itself).  The 2-D tensors are typed by a recipe of regex rules on the GGUF tensor
name (first match wins, otherwise the recipe's "default"); a type whose block does not fit the row falls back
to Q8_0, or to F32 when the row is not even a multiple of 32.

The chosen types are quantized by ggml's own code through libggml-base.so (ggml_quantize_chunk - the same call
llama-quantize makes, no imatrix, no calibration, straight rounding), in row chunks of a ThreadPoolExecutor.
gguf.quants.quantize is replaced so the conversion writes through that path: for F32/F16/BF16 the original is
used, for anything else our function, wrapped with LazyNumpyTensor._wrap_fn (meta_noop = uint8 + the byte
shape) exactly like gguf-py's own quantizers, so a big tensor is quantized while it is written, not in RAM.

    python quantize_rtn.py --recipe recipes/g-l.json --dry-run > plan.txt
        # table of every tensor: name / float shape / type / bytes; nothing written
    python quantize_rtn.py --recipe recipes/g-l.json --outfile OUT.gguf [--only-layers 0-3]
        [--model /data/models/GLM-5.3-FP8] [--threads 28] [--split-max-size 48G] [--ggml PATH]

--only-layers "A-B" keeps model.layers.N.* with A <= N <= B plus embed_tokens / model.norm / lm_head.
Block 78 is the NextN/MTP block (78 trunk layers).  Reads the checkpoint, writes only --outfile.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import ctypes
import json
import logging
import os
import pathlib
import re
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
LLAMA_CPP = pathlib.Path("/data/glm53/llama.cpp")
GGML_LIB = str(LLAMA_CPP / "build/bin/libggml-base.so")
MODEL_DIR = "/data/models/GLM-5.3-FP8"

sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))
sys.path.insert(0, str(LLAMA_CPP))

import gguf  # noqa: E402
from gguf.lazy import LazyNumpyTensor  # noqa: E402
from conversion import (  # noqa: E402
    ModelBase, ModelType, get_model_architecture, get_model_class, load_all_models,
    logger as conv_logger,
)
from conversion.glm import GlmMoeDsaModel  # noqa: E402  (the class the converter registers, and we subclass it)
from convert_hf_to_gguf import split_str_to_n_bytes  # noqa: E402  (same "--split-max-size N(K|M|G)" parsing

QT = gguf.GGMLQuantizationType
_PASSTHROUGH = (QT.F32, QT.F16, QT.BF16)  # the original gguf.quants.quantize handles these


# ===== ggml through ctypes (copied from tools/maya_quant/quantize.py, lines ~39-62) =====
class Ggml:
    def __init__(self, path: str):
        self.lib = ctypes.CDLL(path)
        L = self.lib
        L.ggml_quantize_chunk.restype = ctypes.c_size_t
        L.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                          ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        L.ggml_row_size.restype = ctypes.c_size_t
        L.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        L.ggml_quantize_init.argtypes = [ctypes.c_int]
        L.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
        L.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]

    def row_size(self, t: int, n: int) -> int:
        return int(self.lib.ggml_row_size(t, n))


def _shape_to_bytes(shape: tuple[int, ...], qtype: QT) -> tuple[int, ...]:
    """(..., n) float -> (..., n // block_size * type_size) bytes."""
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    if shape[-1] % block_size != 0:
        raise gguf.QuantError(
            f"Can't quantize tensor with shape {tuple(shape)} to {qtype.name}: "
            f"row size {shape[-1]} is not a multiple of its block size {block_size}")
    return (*shape[:-1], shape[-1] // block_size * type_size)


class Rtg:
    """gguf.quants.quantize replaced: ggml (libggml-base.so) in row chunks, lazily for LazyNumpyTensor."""

    def __init__(self, lib: str, threads: int):
        self.G = Ggml(lib)
        self.threads = threads
        self.pool = cf.ThreadPoolExecutor(threads)
        self._wrapped: dict[QT, object] = {}
        self.orig = gguf.quants.quantize

    def patch(self) -> None:
        gguf.quants.quantize = self.quantize
        if getattr(gguf, "quantize", None) is self.orig:  # gguf/__init__ re-exports it under this name too
            gguf.quantize = self.quantize

    # --- the actual quantization: float32 [*, n] -> uint8 [*, row_size] ---
    def eager(self, arr: np.ndarray, qtype: QT) -> np.ndarray:
        if self.G.lib.ggml_quantize_requires_imatrix(int(qtype)):
            raise RuntimeError(f"{qtype.name} needs an imatrix; this RTN recipe passes none")
        a = np.ascontiguousarray(arr, dtype=np.float32)
        rows, n = (int(np.prod(a.shape[:-1])), int(a.shape[-1])) if a.ndim > 1 else (1, a.shape[0])
        flat = a.reshape(rows, n)
        row_size = self.G.row_size(int(qtype), n)
        dst = np.empty((rows, row_size), dtype=np.uint8)
        # one init upfront; ggml_quantize_chunk re-runs it per call as a noop when the type is unchanged
        self.G.lib.ggml_quantize_init(int(qtype))
        k = max(1, min(self.threads * 4, rows // 16 or 1))
        cuts = [round(i * rows / k) for i in range(k + 1)]

        def work(i: int) -> None:
            r0, r1 = cuts[i], cuts[i + 1]
            if r1 > r0:
                # ggml writes dst at start_row * row_size: src/dst are the BASE pointers, start is the row offset
                self.G.lib.ggml_quantize_chunk(int(qtype), flat.ctypes.data, dst.ctypes.data,
                                               r0 * n, r1 - r0, n, None)

        list(self.pool.map(work, range(k)))
        return dst.reshape(_shape_to_bytes(tuple(a.shape), qtype))

    def quantize(self, data, qtype: QT):
        """drop-in for gguf.quants.quantize (F32/F16/BF16 fall through to it untouched)"""
        if qtype in _PASSTHROUGH:
            return self.orig(data, qtype)
        if isinstance(data, LazyNumpyTensor):
            _shape_to_bytes(tuple(data.shape), qtype)  # raises QuantError before the lazy node is built
            wrapped = self._wrapped.get(qtype)
            if wrapped is None:
                wrapped = LazyNumpyTensor._wrap_fn(
                    self.eager, meta_noop=(np.uint8, lambda s, t=qtype: _shape_to_bytes(tuple(s), t)))
                self._wrapped[qtype] = wrapped
            return wrapped(data, qtype)
        return self.eager(data, qtype)


# ===== recipe: {"rules": [{"match": regex, "type": "Q6_K"}, ...], "default": ...} =====
def load_recipe(path: pathlib.Path) -> tuple[list[tuple[re.Pattern, QT]], QT]:
    recipe = json.loads(path.read_text())
    rules = [(re.compile(r["match"]), QT[r["type"]]) for r in recipe["rules"]]
    return rules, QT[recipe.get("default", "Q8_0")]


def pick_type(name: str, shape: tuple[int, ...], rules: list[tuple[re.Pattern, QT]], default: QT) -> QT:
    t = default
    for rx, rt in rules:  # first rule that matches wins, else the recipe's default
        if rx.search(name):
            t = rt
            break
    if t == QT.F32:
        return QT.F32
    n = int(shape[-1])  # rows are the leading dims, ggml quantizes each row of n along the last dim
    if n % gguf.GGML_QUANT_SIZES[t][0]:  # rows that are not whole blocks: Q8_0 if its 32 fits, else F32
        t = QT.Q8_0 if n % 32 == 0 else QT.F32
    return t


class RtnQuantModel(GlmMoeDsaModel):
    """GlmMoeDsaModel + the recipe (tensor_force_quant) and record of the emitted tensor shapes."""

    model_arch = GlmMoeDsaModel.model_arch  # TextModel.__init_subclass__ wants it redeclared on each subclass

    _recipe_rules: list[tuple[re.Pattern, QT]] = []
    _recipe_default: QT = QT.F32
    _only_layers: tuple[int, int] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shapes: dict[str, tuple[int, ...]] = {}

    @classmethod
    def filter_tensors(cls, item):
        titem = super().filter_tensors(item)
        if titem is None:
            return None
        if cls._only_layers is not None:
            m = re.match(r"model\.layers\.(\d+)\.", titem[0])
            if m and not (cls._only_layers[0] <= int(m.group(1)) <= cls._only_layers[1]):
                return None  # embed_tokens / model.norm / lm_head are not model.layers.* and stay
        return titem

    def modify_tensors(self, data_torch, name, bid):
        for new_name, tensor in super().modify_tensors(data_torch, name, bid):
            self._shapes[new_name] = tuple(int(x) for x in tensor.shape)
            yield new_name, tensor

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int):
        if n_dims <= 1 or new_name.endswith("_norm.weight"):
            return False  # the converter puts these in F32 itself
        shape = self._shapes.get(new_name)
        if shape is None:
            return False
        return pick_type(new_name, shape, self._recipe_rules, self._recipe_default)


def build_model(args: argparse.Namespace, rules, default) -> RtnQuantModel:
    dir_model = pathlib.Path(args.model)
    outfile = pathlib.Path(args.outfile) if args.outfile else pathlib.Path("/data/glm53/g-l.gguf")
    if args.dry_run:
        outfile = pathlib.Path("/data/glm53/dry-run.gguf")  # nothing is written in a dry run
    only = None
    if args.only_layers:
        lo, _, hi = args.only_layers.partition("-")
        only = (int(lo), int(hi or lo))
    RtnQuantModel._recipe_rules = rules
    RtnQuantModel._recipe_default = default
    RtnQuantModel._only_layers = only

    load_all_models()  # conversion/__init__: import every model module so @ModelBase.register runs
    hparams = ModelBase.load_hparams(dir_model, is_mistral_format=False)
    arch = get_model_architecture(hparams, ModelType.TEXT)
    registered = get_model_class(arch)
    if not issubclass(registered, GlmMoeDsaModel):
        raise SystemExit(f"{arch} -> {registered.__name__}, expected a GlmMoeDsaModel subclass")

    # as convert_hf_to_gguf.py's main() builds it: ALL_F32 = whatever the recipe does not cover stays F32,
    # lazy (not --no-lazy), its own --split-max-size parsing
    with __import__("torch").inference_mode():
        return RtnQuantModel(
            dir_model, gguf.LlamaFileType.ALL_F32, outfile,
            is_big_endian=False, use_temp_file=False, eager=False,
            metadata_override=None, model_name=None, split_max_tensors=0,
            split_max_size=split_str_to_n_bytes(args.split_max_size),
            dry_run=args.dry_run, small_first_shard=False, remote_hf_model_id=None,
        )


def nbytes_of(qtype: QT, shape: tuple[int, ...], G: Ggml) -> int:
    rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
    if qtype in _PASSTHROUGH and qtype != QT.BF16:
        item = 4 if qtype == QT.F32 else 2
        return rows * int(shape[-1]) * item
    return rows * G.row_size(int(qtype), int(shape[-1]))


def run_dry_plan(model: RtnQuantModel, G: Ggml) -> None:
    """walk every tensor (subclass + recipe, all quantization stays lazy) and print the plan"""
    rows: list[tuple[str, tuple[int, ...], QT, int]] = []
    orig = model.gguf_writer.add_tensor

    def spy(name, tensor, raw_shape=None, raw_dtype=None, tensor_endianess=None):
        qtype = raw_dtype if raw_dtype is not None else QT.F32
        shape = model._shapes.get(name)
        if shape is None:  # added straight to the writer, not through modify_tensors
            sh = tuple(int(d) for d in (raw_shape if raw_shape is not None else tensor.shape))
            shape = sh if qtype in _PASSTHROUGH else gguf.quants.quant_shape_from_byte_shape(sh, qtype)
        rows.append((name, shape, qtype, nbytes_of(qtype, shape, G)))
        return orig(name, tensor, raw_shape=raw_shape, raw_dtype=raw_dtype, tensor_endianess=tensor_endianess)

    model.gguf_writer.add_tensor = spy
    conv_logger.setLevel(logging.WARNING)  # keep the plan a table
    model.prepare_tensors()

    by_type: dict[str, list[int]] = {}
    for name, shape, qtype, nb in rows:
        by_type.setdefault(qtype.name, [0, 0])
        by_type[qtype.name][0] += 1
        by_type[qtype.name][1] += nb
        print(f"{name:46s} {str(list(shape)):26s} {qtype.name:7s} {nb / 1e9:9.4f}")
    total = sum(r[3] for r in rows)
    print("-" * 96)
    for t, (c, nb) in sorted(by_type.items(), key=lambda kv: -kv[1][1]):
        print(f"{t:7s} {c:5d} tensors {nb / 1e9:10.3f} GB")
    print("-" * 96)
    print(f"{len(rows)} tensors, {total / 1e9:.2f} GB (plan only, nothing written)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--recipe", default=str(HERE / "recipes/g-l.json"))
    ap.add_argument("--outfile")
    ap.add_argument("--threads", type=int, default=28)
    ap.add_argument("--split-max-size", default="48G")
    ap.add_argument("--only-layers", default=None, help='"A-B": keep model.layers.N with A<=N<=B (+embed/norm/lm_head)')
    ap.add_argument("--dry-run", action="store_true", help="print the tensor plan, quantize and write nothing")
    ap.add_argument("--ggml", default=GGML_LIB, help="libggml-base.so to quantize with")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if not args.dry_run and not args.outfile:
        ap.error("--outfile is required unless --dry-run")

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    rules, default = load_recipe(pathlib.Path(args.recipe))
    rtg = Rtg(args.ggml, args.threads)
    rtg.patch()
    t0 = time.time()
    model = build_model(args, rules, default)
    print(f"{os.path.basename(__file__)}: recipe {args.recipe}, type {QT(default).name} default, "
          f"ftype {model.ftype.name}, threads {rtg.threads}, split {args.split_max_size}, "
          f"only-layers {args.only_layers or 'all'}", flush=True)

    if args.dry_run:
        run_dry_plan(model, rtg.G)
        print(f"planned in {time.time() - t0:.1f} s")
        return

    model.write()
    out = model.fname_out
    size = 0
    if out.is_file():
        size = out.stat().st_size
    else:  # split output
        stem = out.name.replace(".gguf", "")
        size = sum(p.stat().st_size for p in out.parent.glob(f"{stem}*.gguf"))
    print(f"done: {size / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
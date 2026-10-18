"""tools/glm53/quantize.py - GLM-5.3 FP8 -> GGUF with the calibration statistics (imatrix) and optional GPTQ.

The sibling quantize_rtn.py rounds to nearest; this one keeps its output identical in every other respect - the
same llama.cpp converter class (conversion/glm.py's GlmMoeDsaModel), the same tensor order, the same metadata,
the same recipe rules, the same --split-max-size sharding, i.e. exactly what the engine and iq_pack.py were
checked against - and only changes *how* a 2-D tensor is rounded:

  * every tensor is quantized by ggml's own code (ggml_quantize_chunk - the call llama-quantize makes) with the
    importance of its input channels: mean x^2 per channel from calib.py's stats_XX.pt, and per routed expert for
    the *_exps tensors (the pointer is what llama-quantize passes as its imatrix).  A tensor without statistics
    rounds as quantize_rtn.py does; those are the 1-D tensors and norms (forced F32 by the converter anyway),
    embed/output, the attn_k_b slice (its GGUF row is the qk_nope axis, not the latent input) and NextN block 78,
    which the calibration pass did not run;
  * with --gptq, the routed experts' gate/up of every layer whose MoE input calib.py kept (ffn_in_XX.bin) are
    rounded with error feedback on the GPU (gptq.py).  Down keeps the imatrix rounding, which measured better at
    Maya, so only gate/up ever goes through GPTQ.

As in quantize_rtn.py the conversion is lazy: gguf.quants.quantize is patched and returns lazy tensors the writer
evaluates one at a time while it writes the shards.  The GPTQ for a layer therefore runs when its two tensors are
written (about 2 GB of bytes live between them), not while the other tensors wait.

    python quantize.py --stats /data/glm53/calib/stats --recipe recipes/g-m24.json --gptq \
        --outfile /data/glm53/models/GLM-5.3-Maya-G-M24-GPTQ/GLM-5.3-Maya-G-M24-GPTQ.gguf
        [--model /data/models/GLM-5.3-FP8] [--threads 20] [--split-max-size 48G] [--only-layers 40-40]
        [--gptq-prior 16384] [--gptq-damp 0.1] [--gptq-chunk 8] [--dry-run]

The recipe is {"rules": [{"match": regex on the GGUF name, "type": "IQ2_S"}, ...], "default": ...} as in
recipes/g-m24.json (the recipe the RTN model used, so the two files differ only in rounding).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
LLAMA_CPP = pathlib.Path("/data/glm53/llama.cpp")
GGML_LIB = str(LLAMA_CPP / "build/bin/libggml-base.so")
MODEL_DIR = "/data/models/GLM-5.3-FP8"
STATS_DIR = "/data/glm53/calib/stats"

sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))
sys.path.insert(0, str(LLAMA_CPP))
sys.path.insert(0, str(HERE))

import gguf  # noqa: E402
from gguf.lazy import LazyNumpyTensor  # noqa: E402
import quantize_rtn as qr  # noqa: E402  (the RTN sibling: recipe, converter subclass, build_model)
import gptq as GQ  # noqa: E402

QT = gguf.GGMLQuantizationType
GPTQ_RE = re.compile(r"blk\.(\d+)\.ffn_(gate|up)_exps\.weight$")


class Stats:
    """calib.py's per-layer statistics -> the GGUF tensor's importance vector (mean x^2 per input channel)."""

    LIN = {
        "attn_q_a.weight": "self_attn.q_a_proj", "attn_q_b.weight": "self_attn.q_b_proj",
        "attn_kv_a_mqa.weight": "self_attn.kv_a_proj_with_mqa",
        # attn_k_b has no entry on purpose: the converter stores it as [64, 512, 192], so its GGUF row runs along
        # qk_nope (192) while the projection's input channels are the latent (512) - the vectors would not align.
        # attn_v_b is [64, 256, 512]: its row IS the latent, so kv_b_proj's vector applies.
        "attn_v_b.weight": "self_attn.kv_b_proj",
        "attn_output.weight": "self_attn.o_proj",
        "ffn_gate.weight": "mlp.gate_proj", "ffn_up.weight": "mlp.up_proj", "ffn_down.weight": "mlp.down_proj",
        "indexer.attn_q_b.weight": "self_attn.indexer.wq_b",
        "indexer.attn_k.weight": "self_attn.indexer.wk",
        "indexer.proj.weight": "self_attn.indexer.weights_proj",
    }

    def __init__(self, root: str):
        self.root = pathlib.Path(root)
        if not self.root.is_dir():
            raise SystemExit(f"--stats {self.root} is not a directory")
        self._cache: dict[int, dict | None] = {}

    def layer(self, il: int) -> dict | None:
        if il not in self._cache:
            if len(self._cache) > 4:
                self._cache.clear()
            p = self.root / f"stats_{il:02d}.pt"
            self._cache[il] = torch.load(p) if p.exists() else None
        return self._cache[il]

    def get(self, name: str, shape: tuple[int, ...]) -> np.ndarray | None:
        """[n] or [E, n] of mean x^2, or None (no statistics - round as quantize_rtn.py does)."""
        m = re.match(r"blk\.(\d+)\.(.+)$", name)
        if not m:
            return None                          # embed_tokens / lm_head: the calibration has no such rows
        il, part = int(m.group(1)), m.group(2)
        st = self.layer(il)
        if st is None:
            return None
        n_last = int(shape[-1])
        vec = None
        if part in self.LIN:
            s = st["linear"].get(self.LIN[part])
            if s is not None and s["n"]:
                vec = (s["ss"] / s["n"]).numpy().astype(np.float32)
        elif part == "ffn_down_shexp.weight":
            moe = st.get("moe")
            if moe is not None:
                vec = (moe["sh_d"] / max(1, moe["sh_n"])).numpy().astype(np.float32)
        elif part in ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"):
            moe = st.get("moe")
            if moe is not None:
                vec = (moe["sh_gu"] / max(1, moe["sh_n"])).numpy().astype(np.float32)
        elif part in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"):
            moe = st.get("moe")
            if moe is not None:
                ss = moe["imat_d"] if part == "ffn_down_exps.weight" else moe["imat_gu"]
                n = moe["n"].double().clamp(min=1).unsqueeze(1)
                im = (ss.double() / n).float()
                live = moe["n"] > 0
                if live.any():                   # never routed: the layer's own average
                    im[~live] = im[live].mean(0)
                vec = im.numpy().astype(np.float32)
        if vec is None:
            return None
        if vec.shape[-1] != n_last or (vec.ndim == 2 and vec.shape[0] != int(shape[0])):
            print(f"[warn] {name}: stats shape {vec.shape} vs tensor {tuple(shape)} - rounding without imatrix",
                  flush=True)
            return None
        return np.ascontiguousarray(vec)


class Imx(qr.Rtg):
    """qr.Rtg (ggml through ctypes, lazily) plus the tensor's name and, for gate/up experts, GPTQ.

    The name is threaded from the converter: ImxQuantModel.modify_tensors sets `current` for the tensor the
    caller is about to quantize, and quantize() runs in that same loop iteration (the lazy node it returns keeps
    whatever it captured for the write phase).
    """

    def __init__(self, lib: str, threads: int, stats: Stats, name_of, gptq_out=None):
        super().__init__(lib, threads)
        self.stats = stats
        self.name_of = name_of
        self.gptq = gptq_out
        self.plan: dict[str, tuple[QT, tuple[int, ...]]] = {}
        self.n_imat = 0
        self.n_rtn = 0

    def quantize(self, data, qtype: QT):
        name = self.name_of()
        if name is None:
            return super().quantize(data, qtype)
        shape = tuple(int(d) for d in data.shape)        # the float shape (lazy meta or eager)
        self.plan[name] = (qtype, shape)
        if qtype in qr._PASSTHROUGH:
            return self.orig(data, qtype)
        if (self.gptq is not None and (m := GPTQ_RE.match(name)) is not None
                and self.gptq.available(int(m.group(1)))):
            return self.gptq.node(name, qtype, shape)
        im = self.stats.get(name, shape)
        if im is None:
            self.n_rtn += 1
            return super().quantize(data, qtype)         # exactly quantize_rtn.py's rounding
        self.n_imat += 1
        if isinstance(data, LazyNumpyTensor):
            wrapped = LazyNumpyTensor._wrap_fn(
                lambda a, t=qtype, im=im: self.eager_im(a, t, im),
                meta_noop=(np.uint8, lambda s, t=qtype: qr._shape_to_bytes(tuple(s), t)))
            return wrapped(data, qtype)
        return self.eager_im(data, qtype, im)

    # --- ggml with the importance vector (qr.Rtg.eager plus one pointer) ---
    def _rows(self, flat: np.ndarray, dst: np.ndarray, qtype: QT, im: np.ndarray | None) -> None:
        rows, n = flat.shape
        k = max(1, min(self.threads * 4, rows // 16 or 1))
        cuts = [round(i * rows / k) for i in range(k + 1)]
        ime = None if im is None else np.ascontiguousarray(im, np.float32)
        ptr = None if ime is None else ime.ctypes.data

        def work(i: int) -> None:
            r0, r1 = cuts[i], cuts[i + 1]
            if r1 > r0:
                self.G.lib.ggml_quantize_chunk(int(qtype), flat.ctypes.data, dst.ctypes.data,
                                               r0 * n, r1 - r0, n, ptr)

        list(self.pool.map(work, range(k)))

    def eager_im(self, arr: np.ndarray, qtype: QT, im: np.ndarray) -> np.ndarray:
        a = np.ascontiguousarray(arr, dtype=np.float32)
        shape = tuple(a.shape)
        n = int(shape[-1])
        rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
        row_bytes = self.G.row_size(int(qtype), n)
        self.G.lib.ggml_quantize_init(int(qtype))
        out = np.empty(qr._shape_to_bytes(shape, qtype), dtype=np.uint8)
        if im.ndim == 2:                                  # experts: every slice its own vector
            flat_out = out.reshape(int(im.shape[0]), -1, row_bytes)
            for e in range(flat_out.shape[0]):
                src = np.ascontiguousarray(a[e]).reshape(-1, n)
                self._rows(src, flat_out[e], qtype, im[e])
        else:
            self._rows(a.reshape(rows, n), out.reshape(rows, row_bytes), qtype, im)
        return out


class GptqOut:
    """GPTQ bytes of one layer's routed gate/up, computed when the writer reaches those two tensors."""

    def __init__(self, ck: GQ.Checkpoint, cfg: dict, stats: pathlib.Path, lib, plan, pool,
                 prior: float, damp: float, chunk: int):
        self.ck, self.cfg, self.stats, self.lib = ck, cfg, stats, lib
        self.plan, self.pool = plan, pool
        self.prior, self.damp, self.chunk = prior, damp, chunk
        self.cache: dict[str, np.ndarray] = {}
        self.bytes = 0
        self.t0 = time.time()
        self.dev = torch.device("cuda")

    def available(self, il: int) -> bool:
        return (self.stats / f"ffn_in_{il:02d}.bin").exists()

    def node(self, name: str, qtype: QT, shape: tuple[int, ...]):
        meta = LazyNumpyTensor.meta_with_dtype_and_shape(np.uint8, qr._shape_to_bytes(shape, qtype))
        return LazyNumpyTensor(meta=meta, args=(), kwargs={}, func=lambda: self.bytes_for(name, qtype, shape))

    def bytes_for(self, name: str, qtype: QT, shape: tuple[int, ...]) -> np.ndarray:
        if name in self.cache:
            return self.cache.pop(name)
        m = GPTQ_RE.match(name)
        il, part = int(m.group(1)), m.group(2)
        gname, uname = f"blk.{il}.ffn_gate_exps.weight", f"blk.{il}.ffn_up_exps.weight"
        for k in [k for k in self.cache if not k.startswith(f"blk.{il}.")]:
            del self.cache[k]          # a layer whose other tensor went a different way: do not keep its bytes
        if part == "gate":
            qtype_g, qtype_u = qtype, self.plan.get(uname, (qtype, None))[0]
        else:
            qtype_g, qtype_u = self.plan.get(gname, (qtype, None))[0], qtype
        t0 = time.time()
        cg = GQ.Codec(self.lib, int(qtype_g))
        cu = GQ.Codec(self.lib, int(qtype_u))
        layer = GQ.Layer(self.ck, self.cfg, self.stats, il, self.dev)
        qg, qu = [], []
        for c0 in range(0, layer.E, self.chunk):
            ex = list(range(c0, min(layer.E, c0 + self.chunk)))
            g, u, _, _ = GQ.round_gate_up(layer, ex, cg, cu, self.pool, self.prior, self.damp)
            qg.append(g)
            qu.append(u)
        E = layer.E
        del layer
        torch.cuda.empty_cache()
        qg = np.concatenate(qg, 0)
        qu = np.concatenate(qu, 0)
        exp = qr._shape_to_bytes(shape, qtype_g)
        exp_u = qr._shape_to_bytes(shape, qtype_u)
        if qg.shape != exp or qu.shape != exp_u:
            raise SystemExit(f"{name}: gptq wrote {qg.shape}/{qu.shape}, planned {exp}/{exp_u}")
        self.cache[gname], self.cache[uname] = qg, qu
        out = self.cache.pop(name)
        self.bytes += out.nbytes
        print(f"[gptq] blk.{il} {part}: {E} experts {qtype_g.name}/{qtype_u.name} {time.time() - t0:6.1f} s, "
              f"{self.bytes / 1e9:6.2f} GB done, gpu {torch.cuda.max_memory_allocated() / 2 ** 30:4.1f} GiB",
              flush=True)
        return out


class ImxQuantModel(qr.RtnQuantModel):
    """RtnQuantModel + the tensor name the quantizer is being asked about (see Imx)."""

    model_arch = qr.RtnQuantModel.model_arch
    current: str | None = None

    def modify_tensors(self, data_torch, name, bid):
        for new_name, tensor in super().modify_tensors(data_torch, name, bid):
            ImxQuantModel.current = new_name      # prepare_tensors quantizes this tensor before resuming us
            yield new_name, tensor


class LiteReader(gguf.GGUFReader):
    """gguf.GGUFReader that parses the tensor table without materialising the tensors' data: the table of an
    interrupted shard is complete while its data is cut short, and the stock reader raises on those."""

    def _build_tensors(self, start_offs, fields):
        ts = []
        for field in fields:
            _nl, name_data, _nd, dims, raw_dtype, offset_tensor = field.parts
            gt = QT(int(raw_dtype[0]))
            ne = 1
            for d in dims.tolist():
                ne *= int(d)
            bs, tsize = gguf.GGML_QUANT_SIZES[gt]
            ts.append(SimpleNamespace(name=str(bytes(name_data), encoding="utf-8"), tensor_type=gt,
                                      shape=dims, n_bytes=ne * tsize // bs,
                                      data_offset=int(start_offs) + int(offset_tensor[0])))
        self.tensors = ts


class Prev:
    """What an interrupted run already wrote: the plan its shards prove, the tensors they hold, their headers.

    The new run must produce byte-identical shards, so the tensor tables and the data offsets have to match
    exactly; check() verifies that before a single weight is written and restores the old header if they do not.
    """

    def __init__(self, names):
        self.keep: set[str] = set()
        self.old: dict[int, list] = {}
        self.off: dict[tuple[int, str], int] = {}
        self.hdr: dict[int, bytes] = {}
        self.paths: dict[int, pathlib.Path] = {}
        self.size: dict[int, int] = {}
        self.kv_n: dict[int, int] = {}
        self.data_off: dict[int, int] = {}
        self.restored = False
        for i, p in enumerate(names):
            fp = pathlib.Path(p)
            if not fp.exists():
                continue
            r = LiteReader(str(fp))
            size = fp.stat().st_size
            self.paths[i] = fp
            self.kv_n[i] = len(r.fields)          # the KV count the header was written with
            self.old[i] = [(t.name, int(t.tensor_type), tuple(int(x) for x in t.shape), int(t.n_bytes))
                           for t in r.tensors]
            self.data_off[i] = int(r.data_offset)
            self.size[i] = size
            with open(fp, "rb") as f:
                # the header plus enough of the data to put the file back exactly if the check below fails
                self.hdr[i] = f.read(r.data_offset + self.PREFIX)
            for t in r.tensors:
                self.off[(i, t.name)] = int(t.data_offset)
                if t.data_offset + t.n_bytes <= size:
                    self.keep.add(t.name)

    PREFIX = 16 << 20        # header + 16 MB of data kept in RAM to restore a failed check

    @staticmethod
    def _key(name, ti):
        return name, int(ti.dtype), tuple(reversed([int(x) for x in ti.shape])), int(ti.nbytes)

    def check(self, i: int, tensors, data_off: int) -> None:
        """shard i's new plan must be exactly the one the existing bytes were written from"""
        if i not in self.old:
            return
        old, new = self.old[i], [(n, ti) for n, ti in tensors.items()]
        if len(old) != len(new):
            self.restore_all()
            raise SystemExit(f"resume: shard {i} holds {len(old)} tensors, the new plan has {len(new)} - "
                             f"run without --resume to rebuild")
        for (on, ot, osh, ob), (nn, ti) in zip(old, new):
            if (on, ot, osh, ob) != self._key(nn, ti):
                self.restore_all()
                raise SystemExit(f"resume: shard {i}: {nn} {self._key(nn, ti)} does not match the file's "
                                 f"{on} {(ot, osh, ob)} - run without --resume to rebuild")
        if data_off != self.data_off[i]:
            self.restore_all()
            raise SystemExit(f"resume: shard {i} would start its data at {data_off}, the file says "
                             f"{self.data_off[i]} (the header/KV differs) - run without --resume to rebuild")

    def restore_all(self) -> None:
        """put the original headers (and the data prefix behind them) back, and the original file lengths"""
        if self.restored:
            return
        self.restored = True
        for i, hdr in self.hdr.items():
            with open(self.paths[i], "r+b") as f:
                f.seek(0)
                f.write(hdr)
                f.truncate(self.size[i])
        print(f"[resume] {len(self.hdr)} shard(s) restored to their previous bytes", flush=True)


def write_progress(w) -> None:
    """gguf_writer.write_tensors_to_file with per-tensor lines instead of tqdm (copied, progress added, and
    --resume: a tensor whose bytes the previous run already wrote is kept, only the rest is computed)."""
    prev = getattr(w, "_prev", None)
    w.write_ti_data_to_file()
    for fout in w.fout:
        w.write_padding(fout, fout.tell())
    total = sum(ti.nbytes for t in w.tensors for ti in t.values())
    done = kept = n_kept = n_new = 0
    t0 = time.time()
    for si, (fout, tensors) in enumerate(zip(w.fout, w.tensors)):
        if prev is not None:
            prev.check(si, tensors, fout.tell())
        k0 = n_kept
        for name, ti in tensors.items():
            if prev is not None and name in prev.keep:
                s = fout.tell()
                want = prev.off.get((si, name))
                if s != want:
                    raise SystemExit(f"resume: {name} would be written at {s}, the file holds it at {want}")
                fout.seek(s + ti.nbytes)
                w.write_padding(fout, ti.nbytes)
                done += ti.nbytes
                kept += ti.nbytes
                n_kept += 1
                continue
            assert ti.tensor is not None           # can only iterate once over the tensors
            assert ti.tensor.nbytes == ti.nbytes
            s, t1 = fout.tell(), time.time()
            ti.tensor.tofile(fout)
            if fout.tell() - s != ti.nbytes:
                raise ValueError(f"tensor {name!r} wrote {fout.tell() - s} bytes, expected {ti.nbytes}")
            w.write_padding(fout, ti.nbytes)
            done += ti.nbytes
            ti.tensor = None
            n_new += 1
            dt = time.time() - t1
            if dt >= 1.0 or ti.nbytes >= 5e8 or GPTQ_RE.match(name):
                print(f"[{100 * done / total:5.1f}%] {name:44s} {ti.nbytes / 1e6:9.1f} MB {dt:6.1f} s "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if prev is not None and n_kept > k0:
            print(f"[resume] shard {si + 1}: kept {n_kept - k0} tensors, wrote {len(tensors) - n_kept + k0}",
                  flush=True)
        print(f"--- shard {si + 1}/{len(w.fout)} written ({(time.time() - t0) / 60:.1f} min)", flush=True)
    if prev is not None:
        print(f"[resume] {n_kept} tensors ({kept / 1e9:.1f} GB) kept, {n_new} computed "
              f"({(total - kept) / 1e9:.1f} GB)", flush=True)
    w.state = gguf.WriterState.WEIGHTS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--stats", default=STATS_DIR, help="calib.py's output: stats_XX.pt and ffn_in_XX.bin")
    ap.add_argument("--recipe", default=str(HERE / "recipes/g-m24.json"))
    ap.add_argument("--outfile")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--split-max-size", default="48G")
    ap.add_argument("--only-layers", default=None, help='"A-B": keep model.layers.N with A<=N<=B (+embed/norm/lm')
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="keep the tensors an interrupted run already wrote (their shard plan must match)")
    ap.add_argument("--ggml", default=GGML_LIB)
    ap.add_argument("--gptq", action="store_true", help="error-feedback rounding of the experts' gate/up")
    ap.add_argument("--gptq-prior", type=float, default=16384.0,
                    help="tokens' worth of the layer-wide Hessian added to every expert's")
    ap.add_argument("--gptq-damp", type=float, default=0.1)
    ap.add_argument("--gptq-chunk", type=int, default=8, help="experts rounded together (GPU memory)")
    ap.add_argument("--imatrix-dataset", default="GLM-5.3 calibration (calib.py): chat, multilingual chat, "
                                                 "reasoning traces, web/code, tool calls, 128 x 2048 tokens")
    ap.add_argument("--imatrix-chunks", type=int, default=128)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if not args.dry_run and not args.outfile:
        ap.error("--outfile is required unless --dry-run")

    import logging
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    rules, default = qr.load_recipe(pathlib.Path(args.recipe))
    stats = Stats(args.stats)
    imx = Imx(args.ggml, args.threads, stats, name_of=lambda: ImxQuantModel.current)
    if args.gptq:
        ck = GQ.Checkpoint(args.model)
        cfg = ck.cfg
        imx.gptq = GptqOut(ck=ck, cfg=cfg, stats=pathlib.Path(args.stats), lib=imx.G.lib, plan=imx.plan,
                           pool=imx.pool, prior=args.gptq_prior, damp=args.gptq_damp, chunk=args.gptq_chunk)
    imx.patch()
    t0 = time.time()
    qr.RtnQuantModel = ImxQuantModel      # build_model instantiates what the module names
    model = qr.build_model(args, rules, default)
    print(f"quantize.py: recipe {args.recipe}, stats {args.stats}, gptq "
          f"{'on prior %.0f damp %.3f chunk %d' % (args.gptq_prior, args.gptq_damp, args.gptq_chunk) if args.gptq else 'off'}, "
          f"threads {imx.threads}, split {args.split_max_size}, only-layers {args.only_layers or 'all'}",
          flush=True)

    if args.dry_run:
        qr.run_dry_plan(model, imx.G)
        print(f"planned in {time.time() - t0:.1f} s: imatrix on {imx.n_imat} tensors, plain rounding on "
              f"{imx.n_rtn} (quantize_rtn's)")
        return

    # provenance on top of the converter's own metadata (the KV the engine and llama.cpp rely on are untouched)
    w = model.gguf_writer
    S, U32 = gguf.GGUFValueType.STRING, gguf.GGUFValueType.UINT32
    w.add_key_value("quantize.imatrix.dataset", args.imatrix_dataset, S)
    w.add_key_value("quantize.imatrix.chunks_count", int(args.imatrix_chunks), U32)
    w.add_key_value("maya.recipe", json.dumps(json.load(open(args.recipe))), S)
    w.add_key_value("maya.rounding", f"gptq prior={args.gptq_prior} damp={args.gptq_damp}"
                    if args.gptq else "imatrix", S)
    w.write_tensors_to_file = lambda *, progress=True, _w=w: write_progress(_w)
    if args.resume:
        def _open_resume(path=None):
            """gguf_writer.open_output_file, but reopening an existing shard instead of truncating it"""
            import gguf.gguf_writer as gw
            if w.state is gw.WriterState.EMPTY and w.fout is not None and (path is None or path == w.path):
                return
            if w.state is not gw.WriterState.NO_FILE:
                raise ValueError(f"Expected output file to be not yet opened, got {w.state}")
            if path is not None:
                w.path = path
            if w.path is None:
                return
            names = [str(n) for n in w.print_plan()]
            stale = [str(s) for s in pathlib.Path(w.path).parent.glob(pathlib.Path(w.path).stem + "-*.gguf")
                     if str(s) not in names]
            if stale:
                raise SystemExit(f"resume: shards of another plan are in the way: {stale[:3]} - "
                                 f"run without --resume to rebuild")
            prev = Prev(names)
            w._prev = prev
            w.fout = [open(n, "r+b" if pathlib.Path(n).exists() else "wb") for n in names]
            w.state = gw.WriterState.EMPTY
            n_t = sum(len(t) for t in w.tensors)
            print(f"[resume] {len(prev.keep)} of {n_t} tensors keep their bytes "
                  f"({len(prev.old)} existing shard(s))", flush=True)

        w.open_output_file = _open_resume

    model.write()
    out = pathlib.Path(model.fname_out)
    size = 0
    if out.is_file():
        size = out.stat().st_size
    else:
        stem = out.name.replace(".gguf", "")
        size = sum(p.stat().st_size for p in out.parent.glob(f"{stem}*.gguf"))
    print(f"imatrix on {imx.n_imat} tensors, plain rounding on {imx.n_rtn}", flush=True)
    print(f"done: {size / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)
    print("QUANT DONE", flush=True)


if __name__ == "__main__":
    main()
"""directions.py - per-layer refusal directions from THREE capture runs of the same prompts (3 records a layer a
prompt, K=3):

  noth      cap-noth.bin      / cap-index-noth.json       thinking off, prompt closed with an empty block
  think     cap-think.bin     / cap-index-think.json      reasoning_effort high, capture on the open marker
  endthink  cap-endthink.bin  / cap-index-endthink.json   the model's own thinking + closing marker as prefill
                                                          (only prompts the index marks closed=true)

For every prompt the records in its byte range are parsed as int32 layer + 6144 float32 and only the LAST record of
that layer in that prompt is used.  Per layer and mode: d = mean(H) - mean(B), then the component along the normalized
mean(B) of that same layer AND mode is removed (the residual's own drift), then d is normalized to unit length.
The three directions of a layer are orthonormalized by Gram-Schmidt in the order noth, think, endthink; one that
comes out shorter than 1e-3 is replaced by zeros (a warning is printed).  Layer 0 stays all zeros.

Output --out: raw float32 78*3*6144 in layer order (STRATA_GLM_ABLATE_LAYERS=78, K=3 - layer * K * E).
Printed: |mean(H)-mean(B)| per mode, the three pairwise cosines (before Gram-Schmidt) and the missing-layer
histogram - every 6th layer.  numpy + stdlib only."""
import argparse, json, os, sys
import numpy as np

O = "/data/glm53/uncensored"
P = argparse.ArgumentParser()
for _m in ("noth", "think", "endthink"):
    P.add_argument(f"--cap-{_m}", default=f"{O}/cap-{_m}.bin")
    P.add_argument(f"--idx-{_m}", default=f"{O}/cap-index-{_m}.json")
P.add_argument("--out", default=f"{O}/dir-3modes.f32")
P.add_argument("--L", type=int, default=78)
P.add_argument("--E", type=int, default=6144)
a = P.parse_args()
MODES = ("noth", "think", "endthink")
PAIRS = ((0, 1, "noth-think"), (0, 2, "noth-endthink"), (1, 2, "think-endthink"))


def load(mode, cap, idx_path, closed_only):
    """-> counts[N,L], sumH[L,E], cntH[L], sumB[L,E], cntB[L], avg|residual|[L], used"""
    if not os.path.exists(cap) or not os.path.exists(idx_path):
        sys.exit(f"directions: missing capture for mode {mode}: {cap} / {idx_path}")
    idx = json.load(open(idx_path))
    if closed_only:
        missing = [e for e in idx if "closed" not in e]
        if missing:
            sys.exit(f"directions: {idx_path} has no \"closed\" flag ({len(missing)} entries)")
        idx = [e for e in idx if e["closed"]]
    L, E = a.L, a.E
    REC = 4 + E * 4
    N = len(idx)
    counts = np.zeros((N, L), np.int64)
    sumH, sumB = np.zeros((L, E)), np.zeros((L, E))
    cntH, cntB = np.zeros(L), np.zeros(L)
    rsum, rcnt = np.zeros(L), np.zeros(L)
    if N:
        mem = np.memmap(cap, dtype=np.uint8, mode="r")
        for pi, e in enumerate(idx):
            lo, hi = int(e["start"]), int(e["end"])
            if (hi - lo) % REC:
                print(f"  !! mode {mode} prompt {pi} range {lo}:{hi} is not a multiple of {REC}")
            last = [None] * L
            for off in range(lo, lo + (hi - lo) // REC * REC, REC):
                layer = int(np.frombuffer(bytes(mem[off:off + 4]), dtype="<i4")[0])
                if 0 <= layer < L:
                    counts[pi, layer] += 1
                    last[layer] = np.frombuffer(bytes(mem[off + 4:off + REC]), dtype="<f4").astype(np.float64)
            for layer, v in enumerate(last):
                if v is None:
                    continue
                rsum[layer] += float(np.linalg.norm(v))
                rcnt[layer] += 1
                if e["tag"] == "H":
                    sumH[layer] += v
                    cntH[layer] += 1
                elif e["tag"] == "B":
                    sumB[layer] += v
                    cntB[layer] += 1
    return counts, sumH, cntH, sumB, cntB, rsum / np.maximum(rcnt, 1), N


def one_mode(mode, cap, idx_path, closed_only):
    counts, sumH, cntH, sumB, cntB, rmean, used = load(mode, cap, idx_path, closed_only)
    L = a.L
    missp = [i for i in range(used) if (counts[i] == 0).any()]
    print(f"[{mode}] prompts used: {used}; prompts missing a layer: {len(missp)}"
          f"{' -> ' + str(missp[:20]) if missp else ''}")
    d, bm = np.zeros((L, a.E)), np.zeros((L, a.E))
    empty = []
    for l in range(L):
        if cntH[l] and cntB[l]:
            d[l] = sumH[l] / cntH[l] - sumB[l] / cntB[l]
            bm[l] = sumB[l] / cntB[l]
        else:
            empty.append(l)
    if empty:
        print(f"  !! [{mode}] layers without both classes: {empty} - zero direction")
    return d, bm, rmean, counts


def main():
    D, BM, RN, C = [], [], [], []
    for m in MODES:
        d, bm, rn, c = one_mode(m, getattr(a, f"cap_{m}"), getattr(a, f"idx_{m}"), m == "endthink")
        D.append(d)
        BM.append(bm)
        RN.append(rn)
        C.append(c)
        miss = [l for l in range(a.L) if c.size and (c[:, l] == 0).any()]
        print(f"  [{m}] layers with at least one prompt missing: {len(miss)}")

    # per mode and layer: drop the component along the normalized mean(B) of that layer AND mode, then normalize
    unit = []
    for i, m in enumerate(MODES):
        u = np.zeros_like(D[i])
        for l in range(a.L):
            nb = float(np.linalg.norm(BM[i][l]))
            v = D[i][l]
            if nb > 0:
                b = BM[i][l] / nb
                v = v - float(np.dot(v, b)) * b
            n = float(np.linalg.norm(v))
            if n > 0:
                u[l] = v / n
            elif np.linalg.norm(D[i][l]) > 0:
                print(f"  !! [{m}] layer {l}: nothing left after removing mean(B) - zeros")
        unit.append(u)

    show = sorted({l for l in range(0, a.L, 6)} | {a.L - 1, 40})
    print("per-layer (every 6th) raw |mean(H)-mean(B)| by mode, cosines before Gram-Schmidt:")
    for l in show:
        raw = [float(np.linalg.norm(D[i][l])) for i in range(3)]
        cs = []
        for x, y, nm in PAIRS:
            nx, ny = float(np.linalg.norm(unit[x][l])), float(np.linalg.norm(unit[y][l]))
            cs.append(float(np.dot(unit[x][l], unit[y][l]) / (nx * ny)) if nx and ny else float("nan"))
        print(f"  layer {l:2d} " + " ".join(f"|{m}|d|={r:.4f}" for m, r in zip(MODES, raw)) +
              " | " + " ".join(f"{nm}={c:+.4f}" for (x, y, nm), c in zip(PAIRS, cs)))

    # orthonormalize the three of each layer, noth < think < endthink; the too-short one becomes zeros
    out = np.zeros((a.L, 3, a.E), dtype=np.float64)
    warned = []
    for l in range(a.L):
        if l == 0:
            continue
        basis = []
        for i in range(3):
            v = unit[i][l].copy()
            for b in basis:
                v -= float(np.dot(v, b)) * b
            n = float(np.linalg.norm(v))
            if n < 1e-3:
                warned.append((MODES[i], l, n))
            else:
                out[l, i] = v / n
                basis.append(v / n)
    for m, l, n in warned:
        print(f"  !! Gram-Schmidt: {m} layer {l} collapsed to |v|={n:.2e} - zeros written")

    print("raw |mean(H)-mean(B)| and ratio to that mode's avg |residual| (every 6th layer):")
    for l in show:
        parts = []
        for i, m in enumerate(MODES):
            n = float(np.linalg.norm(D[i][l]))
            r = float(RN[i][l])
            parts.append(f"{m} |d|={n:.4f} ratio={n / r if r else float('nan'):.4f}")
        print(f"  layer {l:2d} " + " | ".join(parts))

    with open(a.out, "wb") as f:
        f.write(out.astype(np.float32).tobytes())
    print(f"WROTE {a.out} {os.path.getsize(a.out)} bytes ({a.L}*3*{a.E}*4)")


main()
"""test_directions.py - synthetic check of the three-mode directions.py: in /tmp/abl-test builds cap-<mode>.bin and
cap-index-<mode>.json for noth / think / endthink (10 H + 10 B prompts, 78 layers each; layer 40 of every H prompt
carries that mode's own known ORTHOGONAL unit vector uk; layer 5 is written twice in one noth prompt; two endthink
prompts are marked closed=false and must be skipped), runs directions.py and asserts
  * cosine(out[layer 40, k], uk) > 0.99 for all three modes
  * the output is 78*3*6144*4 bytes
  * layer 0 is all zeros"""
import json, os, struct, subprocess, sys
import numpy as np

D = "/tmp/abl-test"
E, L, U = 6144, 78, 40
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(1234)

# three orthogonal unit vectors, one per mode
Q, _ = np.linalg.qr(rng.standard_normal((E, 3)))
US = [np.ascontiguousarray(Q[:, k]) for k in range(3)]     # US[k] for mode k

MODES = ("noth", "think", "endthink")


def build(mode, k, dup=False, drop_closed=False):
    cap, idxp = f"{D}/cap-{mode}.bin", f"{D}/cap-index-{mode}.json"
    for f in (cap, idxp):
        if os.path.exists(f):
            os.remove(f)
    index, off = [], 0
    with open(cap, "wb") as f:
        for pi, tag in enumerate(["H"] * 10 + ["B"] * 10):
            start = off
            for l in range(L):
                base = rng.standard_normal(E).astype(np.float32) * 0.001
                vec = base + (US[k].astype(np.float32) if (tag == "H" and l == U) else 0.0)
                rec = struct.pack("<i", l) + vec.astype(np.float32).tobytes()
                f.write(rec)
                off += len(rec)
                if dup and pi == 3 and l == 5:
                    r2 = struct.pack("<i", l) + (base * 3.0 + 1.0).astype(np.float32).tobytes()
                    f.write(r2)
                    off += len(r2)
            ent = {"tag": tag, "start": start, "end": off, "mode": mode}
            if mode == "endthink":
                ent["closed"] = not (drop_closed and pi in (0, 11))    # two rows the filter must drop
            index.append(ent)
    json.dump(index, open(idxp, "w"))
    return cap, idxp


paths, args = {}, []
for k, m in enumerate(MODES):
    cap, idxp = build(m, k, dup=(m == "noth"), drop_closed=(m == "endthink"))
    paths[m] = (cap, idxp)
    args += [f"--cap-{m}", cap, f"--idx-{m}", idxp]
out = f"{D}/dir-test.f32"
r = subprocess.run([sys.executable, "/data/glm53/uncensored/directions.py", *args, "--out", out],
                   capture_output=True, text=True)
print(r.stdout, end="")
if r.stderr:
    print(r.stderr, end="", file=sys.stderr)
assert r.returncode == 0, f"directions.py failed rc={r.returncode}"

data = np.fromfile(out, dtype=np.float32).reshape(L, 3, E)
ok_size = os.path.getsize(out) == L * 3 * E * 4
print(f"output size: {os.path.getsize(out)} bytes, expected {L*3*E*4} -> {'OK' if ok_size else 'FAIL'}")
assert ok_size, "bad output size"

worst = 1.0
for k, m in enumerate(MODES):
    d = data[U, k].astype(np.float64)
    cos = float(np.dot(d, US[k]) / (np.linalg.norm(d) * np.linalg.norm(US[k])))
    worst = min(worst, cos)
    print(f"cosine(layer40 [{m}], u{k + 1}) = {cos:.6f} norm={np.linalg.norm(d):.6f} -> {'OK' if cos > 0.99 else 'FAIL'}")
    assert cos > 0.99, f"{m} cosine {cos} <= 0.99"

z0 = float(np.abs(data[0]).max())
print(f"layer 0 max|value| = {z0} -> {'OK' if z0 == 0.0 else 'FAIL'}")
assert z0 == 0.0, "layer 0 is not zero"

# every retained direction is unit, layers are orthonormal where nothing collapsed
g = data[1:].reshape(-1, E).astype(np.float64)
gn = np.linalg.norm(g, axis=1)
print(f"layers 1..77 directions: min|v|={gn.min():.6f} max|v|={gn.max():.6f} (collapsed = 0)")
nonzero = gn[gn > 0]
assert np.allclose(nonzero, 1.0, atol=1e-5), "a kept direction is not unit"

# the two dropped endthink prompts must not be read (their H carries u3 at layer 40 anyway) - counts printed above
used_line = [l for l in r.stdout.splitlines() if "[endthink] prompts used" in l]
print(used_line[0] if used_line else "NO ENDTHINK LINE")
assert used_line and "prompts used: 18" in used_line[0], "closed=false filtering did not drop 2 prompts"
print("TEST PASS")
print(f"worst cosine across modes = {worst:.6f}")
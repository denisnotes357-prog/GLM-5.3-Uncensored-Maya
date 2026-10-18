# Measuring the refusal vector (`dir3.f32`)

`dir3.f32` is raw float32, 78 layers × 3 directions × 6144, layer-major — the engine reads it with
`STRATA_GLM_ABLATE=dir3.f32 STRATA_GLM_ABLATE_LAYERS=78` (strength: `STRATA_GLM_ABLATE_SCALE`, default 1.0).

To measure your own (for example on other prompts):

1. **Prompts** — a JSONL of `{"tag": "H" | "B", "prompt": "..."}`: harmful requests (H) and harmless ones (B).
   The shipped vector used 256 H and 100 B.
2. **The model's own thinking** for the `endthink` mode — start the server with the donor model **without** the vector
   (`configs/glm53-uncensored-novector.json`) and run `think_gen.py` (192 prompts, `reasoning_effort` high).
3. **Capture** — restart the server with `cfg-capall.json` (it sets `STRATA_GLM_ABLATE_CAPTURE`: every layer appends
   `int32 layer + 6144 float32` of the last position's residual; MTP off) and run, with an explicit `--cap`:
   ```bash
   python capture_all.py --cap cap-all.bin --mode noth     --index cap-index-noth.json
   python capture_all.py --cap cap-all.bin --mode think    --index cap-index-think.json
   python capture_all.py --cap cap-all.bin --mode endthink --index cap-index-endthink.json
   ```
   Try `--limit 3` first: each prompt must hold 78 × `prompt_n` records. ~1.5–2 s a prompt; ~43 GB for 900 prompts.
4. **Directions** —
   ```bash
   python directions.py --cap-noth cap-all.bin --idx-noth cap-index-noth.json \
       --cap-think cap-all.bin --idx-think cap-index-think.json \
       --cap-endthink cap-all.bin --idx-endthink cap-index-endthink.json --out dir3.f32
   ```
   Per layer and mode: mean(H) − mean(B), minus its component along mean(B), normalized; the three directions
   orthonormalized in the order noth, think, endthink; layer 0 stays zero. `test_directions.py` checks this on
   synthetic data.
5. **Evaluate** — ask 40 [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) behaviours with thinking off
   and let a judge model answer one word (REFUSED / COMPLIED / DISPUTED). Results for the shipped vector:
   stock 39/40, donor only 6/40, donor + vector 2/40; strength 1.5 gave 5/40.

The method comes from [alesha-pro's GLM-ABLATE](https://github.com/alesha-pro/project-maya/blob/glm-ablate/docs/GLM-ABLATE.md)
for GLM-5.3-Flash; here it runs on the full 78-layer model with one residual stream instead of four.

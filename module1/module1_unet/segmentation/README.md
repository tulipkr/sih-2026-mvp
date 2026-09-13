# Module 1 — U-Net SAR Oil-Spill Segmentation (Tulip)

Implements the attached specification document exactly (architecture,
inputs, preprocessing assumptions, training, validation, inference,
checkpointing, fallback behavior, outputs, JSON schema, folder structure,
integration/handoff — all per that document). This README also records
every place I had to resolve an ambiguity or fill a gap the document left
open, per this session's explicit instruction not to silently paper over those.

## Honest status of this build

No network access in the sandbox I built this in — could not `pip install
torch`, `segmentation-models-pytorch`, or `rasterio`, so could not execute
the full pipeline myself. Exactly what was and wasn't verified:

**Actually executed and passing (pure Python/numpy, no torch/rasterio needed):**
- All normalization logic (`none`, `db_clip_minmax`) — pass-through, midpoint
  math, missing-parameter errors, unknown-method errors.
- Tiling index math (`_tile_indices`) — verified exact/non-exact grid sizes,
  confirmed no tile ever goes out of bounds.
- The full `fallback_detector.py` logic — blob detection, land-pixel
  exclusion (both zero and NaN land markers), small-blob removal, confirmed
  the score stays in [0,1] and never leaks NaN.
- The complete `inference_result.json` schema validator, including the new
  `fallback_used`/`score_type` consistency check — every required-field,
  wrong-type, out-of-range, and mislabeling case correctly rejected.
- **The specification document's own §31 sample output was run through
  `schema.py` and validates cleanly** — direct confirmation the schema
  implementation matches the document exactly.
- Every file syntax-checked with `py_compile` — no syntax errors anywhere.

**Not executed — needs you to run it with the real dependencies installed:**
- Building the actual `segmentation-models-pytorch` U-Net (the channel-count
  assertion in `model.py` needs a real run to confirm smp's own behavior
  matches spec §13's channel-mismatch requirement — see "Ambiguities" below).
- The training loop, loss/metric computation, and checkpointing on real tensors.
- Generating synthetic GeoTIFFs (`synthetic_data.py` calls rasterio).
- `test_model_forward.py`, `test_losses.py` (torch-dependent parts),
  `test_infer_pipeline.py`, and the fixture-dependent tests in
  `test_edge_cases.py`/`test_dataset.py`.

## Ambiguities found in the specification, and how I resolved them

Full detail is in my message before the code, summarized here for reference
alongside the implementation:

1. **Normalization default.** §7 marks the SAR representation `PENDING:
   ZENODO VERIFICATION`; §25 shows a concrete default (`db_clip_minmax`,
   `-30`/`0`). Per your explicit instruction this session, shipped default is
   `method: none`. The `db_clip_minmax` path is fully implemented and uses
   the exact field names from §25 (`db_clip_min`, `db_clip_max`) — flip it on
   in `config.yaml` once verified against a real Zenodo sample.
2. **Fallback detector's "land mask" (§35) vs. Tulip's ownership boundary
   (§2, which assigns land masking to Ishita).** Resolved by having the
   fallback respect land markers already present in Ishita's preprocessed
   input (NaN or exactly 0.0, `config.fallback.land_marker`) rather than
   computing a coastline mask itself. **This assumes Ishita's preprocessing
   marks land that way — needs her confirmation, not just this module's.**
3. **`score_type` for fallback output** — the document's §9 schema shows a
   single fixed value (`raw_sigmoid_output`), but that would misrepresent a
   rule-based result. Added `rule_based_threshold` as a second valid value,
   with `schema.py` enforcing that it's used if and only if `fallback_used`
   is true. Not something the document specifies — flagging it as my own addition.
4. Missing-checkpoint behavior (§15 allows either) → defaults to explicit
   error; `inference.on_missing_checkpoint: fallback` opts into auto-fallback.
5. Dimension-mismatch behavior (§15/edge-case table allows either) →
   defaults to reject; `inference.on_dimension_mismatch: resize` opts into
   auto-resize (with a warning, per §15's requirement).
6. Pretrained-weight availability isn't addressed by the document at all —
   added `model.pretrained: true/false` so you can turn it off if you're
   somewhere without internet access to fetch ImageNet weights.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Google Colab (GPU)

```python
!pip install -q segmentation-models-pytorch rasterio
# then upload/clone this segmentation/ folder, cd into it, and run the same
# commands below — config.yaml's device: auto will pick up the Colab GPU
# automatically, no code change needed.
```

## Exact command: synthetic pipeline test

This is the one to run first — proves the entire wiring (dataset → model →
loss → training → checkpoint → inference → all 3 output files) works before
any real Zenodo data exists.

```bash
python -m src.synthetic_data --output_dir data/synthetic --n_samples 8 --patch_size 256 \
    --lookalike_dir data/test_lookalike --lookalike_samples 4
python -m src.train --config configs/config.yaml
python -m src.infer --config configs/config.yaml --scene_id synthetic_scene_003
```

Where each output lands:
- `checkpoints/best_model.pt` — from `train.py`.
- `outputs/metrics/training_metrics.json` — per-epoch loss/metrics history
  + the look-alike false-positive check result.
- `outputs/prob_maps/synthetic_scene_003.tif`, `outputs/masks/synthetic_scene_003.tif`,
  `outputs/synthetic_scene_003_inference_result.json` — from `infer.py`.

## Exact command: real training

Point `configs/config.yaml`'s `data.manifest_path` at Ishita's real
`manifest.json`, then:

```bash
python -m src.train --config configs/config.yaml
```

**Before doing this**, resolve the normalization question (Ambiguity #1
above) with Ishita and update `config.yaml` accordingly — training on
unverified normalization is exactly the "most dangerous silent failure"
the spec itself warns about (§38).

## Exact command: real inference

```bash
python -m src.infer --config configs/config.yaml --scene_id <real_scene_id>
```

Add `--checkpoint <path>` to point at a specific checkpoint instead of the
one in `config.yaml`.

## Exact command: fallback detector (no trained model)

```bash
# in config.yaml, set: inference.on_missing_checkpoint: fallback
python -m src.infer --config configs/config.yaml --scene_id <scene_id>
```

## Test suite

```bash
pytest tests/ -v
```

## Folder structure

Matches spec §24 exactly, with three additions (marked) that the document
implies but doesn't enumerate a home for:

```
segmentation/
  configs/config.yaml
  data/
    train/ val/ test_lookalike/
    synthetic/                    # ADDED — §34 mock-generator output only
  src/
    dataset.py model.py losses.py metrics.py train.py infer.py
    fallback_detector.py
    schema.py                      # ADDED — implements §9 schema + validator
    synthetic_data.py               # ADDED — implements §34's mock generator
    utils.py
  checkpoints/
  outputs/
    prob_maps/ masks/ metrics/
  tests/
    test_dataset.py test_model_forward.py test_losses.py
    test_infer_pipeline.py test_edge_cases.py
  requirements.txt                  # ADDED — for exact install commands
  README.md
```

## Interface with Ishita (do not change without telling her)

Input manifest entry: `scene_id, patch_path, acquisition_timestamp_utc, crs,
transform, bands` (§7). Training entries additionally need `mask_path` —
this module's own addition for pairing image+label, not part of her
inference-time contract.

Output `inference_result.json` is defined in `src/schema.py`, matches §9
field-for-field (verified directly against the spec's own §31 sample — see
above), and is validated on every inference call. Any change needs to be
communicated to Ishita and Bhumika first, per §33's explicit instruction.

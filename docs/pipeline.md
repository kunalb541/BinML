# BinML — Simulation, Training & Evaluation Pipeline

The v5 pipeline (`pipeline/`) is a ground-up rebuild of the original 3-class classifier into a
**6-class, multi-band model for partial Roman-like seasons**. It is a synthetic streaming-triage
benchmark for the Nancy Grace Roman Galactic Bulge Time-Domain Survey, not a validated autonomous
real-time trigger.

The released model uses a legacy Cycle-7-inspired one-season schedule (15-min F146, 6-h F087/F213),
not the current GBTDS design. Current planning uses approximately 12-min F146 sampling, 66-s
exposures, staggered colour visits, and a multi-season programme.

## What changed from the original (3-class) pipeline

| | original | **v5** |
|---|---|---|
| Classes | 3 (Flat / PSPL / Binary) | **6** — Flat, PSPL, NonPSPL, PeriodicVar, LongPeriodVar, Eruptive |
| Bands | single | **3** under the legacy schedule — F146 (6912 epochs), F087, F213 (288 epochs each) |
| Labelling | by generator intent | **detectability-conditioned** — an event is labelled by what is *observable*, not by what we generated (undetectable event → Flat; undetectable anomaly → PSPL) |
| Partial-season objective | none | **cascade** — under truncation a binary is relabelled Flat→PSPL→NonPSPL using a truth-informed onset proxy (`t_anom`) |
| Headline metric | accuracy / F1 | **completeness at fixed purity** (what a follow-up pipeline is actually specified against) |
| Scale | laptop subset | distributed generation on AWS (millions of events), in-region binning + inference |

The six classes and the labelling rule live in `classes.py` and `assemble.py`.
`NonPSPL` is the binary-lens anomaly class and contains both stellar binaries and
planetary-mass-ratio systems; it is not a planet-only label.

---

## The 6 pipeline stages

Data flows: **generate → bin → memmap/mix → train → evaluate → plot**. Each stage is a
runnable module (`python -m pipeline.<module>`).

```
run_shard        raw light-curve shards  (HDF5, ~312 MB each)
   │
run_bin /        bin to compact cache    (feat/frac tokens, ~46 MB each)
run_bineval      (bineval also runs a checkpoint in-region → compact preds)
   │
to_memmap        cache → fp16 memmap for fast shuffled training
mix_finetune     (or) stratified hard/natural training mix
   │
train         train / warm-start fine-tune → checkpoint .pt
   │
evaluate      checkpoint + test cache → metrics + saved logits artifact
   │
plots         artifact → 14-figure diagnostic suite
plot_evolution_cloud   per-event probability-evolution plots
```

---

## File-by-file

### Core physics / data model (libraries)

| file | what it is |
|---|---|
| `classes.py` | the 6-class registry and label indices |
| `priors.py` | broad analytic training supports for tE, u0, q, s, ρ, blending, and other parameters; these are literature-informed supports, not fitted population distributions |
| `generators.py` | per-class light-curve generators: PSPL, binary lenses via VBBinaryLensing, and analytic/phenomenological variable and eruptive waveforms (not OGLE templates) |
| `photometry.py` | per-band SED, bulge extinction, blending, flux-space noise, detectability |
| `assemble.py` | **the heart** — windows an event, samples all 3 bands, applies noise, and applies the adopted detectability-conditioned label policy; also computes truth-informed, noise-free `t_anom` |
| `writer.py` | HDF5 shard writer; defines `PARAM_FIELDS` (the per-event parameter vector, incl. `t_anom`) |
| `cache.py` | raw shard → compact cache (min/max-pooled tokens that preserve caustic extrema exactly) |
| `model.py` | the classifier — conv stem with non-learned min/max carry lanes → SDPA transformer over 156 tokens, ~505k params |

### Stage modules (CLIs)

| file | run it to… |
|---|---|
| `run_shard.py` | **generate** raw shards. `--regime` selects a hard/OOR regime; `--seed-base` gives a disjoint RNG stream for unseen test data |
| `run_bin.py` | **bin** raw → cache shards (in-region on AWS; idempotent, skips missing/done) |
| `run_bineval.py` | **bin + evaluate** in one pass — bins, runs a checkpoint on CPU, uploads only compact preds (~30 floats/event instead of the light curve) |
| `eval_shard.py` | run a checkpoint on one cache shard → logits npz (the inference core used by `run_bineval`) |
| `to_memmap.py` | cache shards → shuffled fp16 memmap (fast random-access training input) |
| `mix_finetune.py` | build a **stratified** hard/natural training mix (per-class caps; keeps all scarce NonPSPL) |
| `train.py` | **train / fine-tune**. Class-weighted 6-way CE; `--truncate-aug` enables the cascade labelling; `--init-weights` warm-starts; `--resume` continues an interrupted run |
| `evaluate.py` | **evaluate** on a test cache → metrics.json + saved logits/labels/params artifact |
| `plots.py` | the **14-figure diagnostic suite** from a saved eval artifact (confusion, ROC/PR, calibration, efficiency planes, probability-evolution, early-detection…) |
| `plot_evolution_cloud.py` | per-event 3-panel probability-evolution plots (light curve / class probs vs time / commit-time) |
| `agg_stress.py` | aggregate distributed stress-test preds into per-class × per-regime metrics |
| `report.py` | one-shot final report (runs eval + baseline + figures) |
| `baseline.py` | the classical Δχ² baseline the network is compared against |

---

## Quick start — train & evaluate locally

Install the full optional dependency set before running the simulation or paper pipeline:

```bash
pip install -e ".[all]"
```

This includes `numpy`, `scipy`, `h5py`, `torch`, plotting/evaluation packages, and
`VBBinaryLensing`. The simulator and paper build must not proceed with a single-lens fallback.

**1. Generate a little data** (one shard ≈ 7,500 events across all 6 classes):
```bash
python -m pipeline.run_shard --shard 0 --n-shards 1 --out data/raw
```

**2. Bin it to a training cache.** `run_bin`/`run_bineval` are for *distributed* S3 runs (they
probe `aws s3api`); **locally**, call `build_cache` directly:
```bash
python -c "from pipeline.cache import build_cache; import glob; \
build_cache(sorted(glob.glob('data/raw/*.h5')), 'data/cache/shard_00000.h5')"
```

**3. Convert to a training memmap:**
```bash
python -m pipeline.to_memmap --in-dir data/cache --out data/mm
```

**4. Train (or fine-tune):**
```bash
# from scratch
python -m pipeline.train --cache data/mm --out runs/binml.pt \
  --epochs 6 --batch-size 384 --device mps

# warm-start fine-tune with partial-season cascade augmentation
python -m pipeline.train --cache data/mm --out runs/binml.pt \
  --init-weights runs/base.pt --truncate-aug 0.5 --alpha-nonpspl 1.0 \
  --lr 5e-5 --epochs 6 --resume
```

**5. Evaluate on a held-out cache** (must be disjoint — use different shard indices):
```bash
python -m pipeline.evaluate --ckpt runs/binml.pt --cache data/mm_test \
  --out eval/binml --device mps
```

**6. Make the figures:**
```bash
# 14-figure diagnostic suite (pass --ckpt to also get the temporal figures 13/14)
python -m pipeline.plots --preds eval/binml --cache data/mm_test \
  --out figures/binml --ckpt runs/binml.pt

# per-event probability-evolution plots (50 of class 2 = NonPSPL)
python -m pipeline.plot_evolution_cloud --cache data/cache/shard_00000.h5 \
  --ckpt runs/binml.pt --class-idx 2 --n 50 --out figures/evolution
```

---

## Key concepts you need to read the numbers

- **Detectability-conditioned labelling.** `assemble.py` applies an operational label policy to
  each simulated window. An event whose peak falls outside the season, or whose amplitude is
  below the adopted floor, is Flat; a binary below the anomaly criterion is PSPL. The 0.02-mag
  floor is a modelling choice and has not been validated on Roman data.

- **The partial-season cascade (`t_anom`).** For truncation augmentation,
  `train._apply_truncation` relabels a binary PSPL before `t_anom` and NonPSPL after it. This onset
  is computed from injected, noise-free binary-versus-PSPL deviations and is unavailable to a
  live broker. The stored validation scan samples revealed seasons every 0.5 days; among
  nonpremature detections its median lag is +5 days. The matched 400-event ablation is an
  exploratory risk–coverage comparison because it selects thresholds and evaluates outcomes on
  the same events; its conditional McNemar values are not confirmatory inference. The main prefix
  scan contains eligible binaries only and reuses a complete-season threshold, so it does not
  measure streaming false alerts, purity, or workload on contaminant classes.

- **Simulation supports.** The distributions are broad analytic supports in the style of Zhang
  et al., not a measured Roman population model. The `tE` prior is an authored truncated
  lognormal anchored to a literature mean. Variable-star generators are analytic or
  phenomenological shapes rather than samples from OGLE templates.

- **Legacy photometry.** The released model was trained with 46.8-s legacy exposures. Against the
  current Roman calibration, its F087/F213 zeropoints are optimistic by about 0.10/0.14 mag, its
  F087 saturation ordering is wrong at equal exposure, and its colour-band background ratios do
  not match the published thermal backgrounds. Corrected constants are recorded in
  `photometry.py`, but the released checkpoint has not been retrained with them and their effect is
  unquantified.

- **`keep_prob` reweighting is asymmetric.** NonPSPL rows have `keep_prob=1`; the byproduct
  PSPL/Flat rows are subsampled. So **recall must NOT be reweighted, but precision/purity MUST**.
  `evaluate.population_weights` handles this (and uses float64 — float32 cumsum overshoots 1).

- **Completeness at fixed purity** is the headline, not accuracy or F1. Accuracy is meaningless
  (Flat+PSPL are ~61% of events); bare recall is degenerate. The threshold is fixed on a
  validation split and applied frozen to test.

## Distributed runs (AWS)

Generation/binning/inference scale out across free-tier spot instances. The orchestration
scripts (fleet launchers, controller) are operational and account-specific, kept out of the
public tree. The modules themselves take `--bucket`/`--seed-base`/`--worker`/`--workers` so the
same code runs one shard locally or thousands in-region.

## Scope of reproducibility

The commands above exercise the local pipeline. The paper build has a separate five-stage
artifact-validation and rendering workflow documented in [`../paper/README.md`](../paper/README.md).
Stored cloud artifacts have known provenance limits: the published labelling-ablation source hash
was repaired after its run, and the newer content-addressed mechanism has not yet produced that
artifact.

# BinML v5 — Simulation, Training & Evaluation Pipeline

The v5 pipeline (`pipeline/sim_v5/`) is a ground-up rebuild of the original 3-class classifier
into a **6-class, multi-band, real-time-aware** model for the Nancy Grace Roman Galactic Bulge
Time-Domain Survey.

## What changed from the original (3-class) pipeline

| | original | **v5** |
|---|---|---|
| Classes | 3 (Flat / PSPL / Binary) | **6** — Flat, PSPL, NonPSPL, PeriodicVar, LongPeriodVar, Eruptive |
| Bands | single | **3** — F146 (6912 epochs), F087, F213 (288 epochs each) |
| Labelling | by generator intent | **detectability-conditioned** — an event is labelled by what is *observable*, not by what we generated (undetectable event → Flat; undetectable anomaly → PSPL) |
| Real-time behaviour | none | **cascade** — under truncation a binary reads Flat→PSPL→NonPSPL as evidence arrives (`t_anom`), so the model never flags a class before its evidence is on screen |
| Headline metric | accuracy / F1 | **completeness at fixed purity** (what a follow-up pipeline is actually specified against) |
| Scale | laptop subset | distributed generation on AWS (millions of events), in-region binning + inference |

The six classes and the labelling rule live in `classes.py` and `assemble.py`.

---

## The 6 pipeline stages

Data flows: **generate → bin → memmap/mix → train → evaluate → plot**. Each stage is a
runnable module (`python -m pipeline.sim_v5.<module>`).

```
run_shard        raw light-curve shards  (HDF5, ~312 MB each)
   │
run_bin /        bin to compact cache    (feat/frac tokens, ~46 MB each)
run_bineval      (bineval also runs a checkpoint in-region → compact preds)
   │
to_memmap        cache → fp16 memmap for fast shuffled training
mix_finetune     (or) stratified hard/natural training mix
   │
train_v5         train / warm-start fine-tune → checkpoint .pt
   │
evaluate_v5      checkpoint + test cache → metrics + saved logits artifact
   │
plots_v5         artifact → 14-figure diagnostic suite
plot_evolution_cloud   per-event probability-evolution plots
```

---

## File-by-file

### Core physics / data model (libraries)
| file | what it is |
|---|---|
| `classes.py` | the 6-class registry and label indices |
| `priors.py` | parameter priors (tE, u0, q, s, ρ, blending…) from Mróz+2019 / Suzuki+2016 / Penny+2019 |
| `generators.py` | per-class light-curve generators (PSPL, binary via VBBinaryLensing, RR Lyrae/EB/Mira/DN…) |
| `photometry.py` | per-band SED, bulge extinction, blending, flux-space noise, detectability |
| `assemble.py` | **the heart** — windows an event, samples all 3 bands, applies noise, and **labels by observability**; also computes `t_anom` (the day a binary's anomaly first becomes detectable) |
| `writer.py` | HDF5 shard writer; defines `PARAM_FIELDS` (the per-event parameter vector, incl. `t_anom`) |
| `cache.py` | raw shard → compact cache (min/max-pooled tokens that preserve caustic extrema exactly) |
| `model_v5.py` | the classifier — conv stem with non-learned min/max carry lanes → SDPA transformer over 156 tokens, ~505k params |

### Stage modules (CLIs)
| file | run it to… |
|---|---|
| `run_shard.py` | **generate** raw shards. `--regime` selects a hard/OOR regime; `--seed-base` gives a disjoint RNG stream for unseen test data |
| `run_bin.py` | **bin** raw → cache shards (in-region on AWS; idempotent, skips missing/done) |
| `run_bineval.py` | **bin + evaluate** in one pass — bins, runs a checkpoint on CPU, uploads only compact preds (~30 floats/event instead of the light curve) |
| `eval_shard.py` | run a checkpoint on one cache shard → logits npz (the inference core used by `run_bineval`) |
| `to_memmap.py` | cache shards → shuffled fp16 memmap (fast random-access training input) |
| `mix_finetune.py` | build a **stratified** hard/natural training mix (per-class caps; keeps all scarce NonPSPL) |
| `train_v5.py` | **train / fine-tune**. Class-weighted 6-way CE; `--truncate-aug` enables the cascade labelling; `--init-weights` warm-starts; `--resume` continues an interrupted run |
| `evaluate_v5.py` | **evaluate** on a test cache → metrics.json + saved logits/labels/params artifact |
| `plots_v5.py` | the **14-figure diagnostic suite** from a saved eval artifact (confusion, ROC/PR, calibration, efficiency planes, probability-evolution, early-detection…) |
| `plot_evolution_cloud.py` | per-event 3-panel probability-evolution plots (light curve / class probs vs time / commit-time) |
| `agg_stress.py` | aggregate distributed stress-test preds into per-class × per-regime metrics |
| `report_v5.py` | one-shot final report (runs eval + baseline + figures) |
| `baseline_v5.py` | the classical Δχ² baseline the network is compared against |

---

## Quick start — train & evaluate locally

Everything below is CPU/MPS-friendly and needs only `numpy scipy h5py torch` (+ `VBBinaryLensing`
for binary-lens generation). See `environment.yml`.

**1. Generate a little data** (one shard ≈ 7,500 events across all 6 classes):
```bash
python -m pipeline.sim_v5.run_shard --shard 0 --n-shards 1 --out data/raw
```

**2. Bin it to a training cache.** `run_bin`/`run_bineval` are for *distributed* S3 runs (they
probe `aws s3api`); **locally**, call `build_cache` directly:
```bash
python -c "from pipeline.sim_v5.cache import build_cache; import glob; \
build_cache(sorted(glob.glob('data/raw/*.h5')), 'data/cache/shard_00000.h5')"
```

**3. Convert to a training memmap:**
```bash
python -m pipeline.sim_v5.to_memmap --in-dir data/cache --out data/mm
```

**4. Train (or fine-tune):**
```bash
# from scratch
python -m pipeline.sim_v5.train_v5 --cache data/mm --out runs/model.pt \
  --epochs 6 --batch-size 384 --device mps

# warm-start fine-tune with the real-time cascade enabled
python -m pipeline.sim_v5.train_v5 --cache data/mm --out runs/stage6.pt \
  --init-weights runs/stage5.pt --truncate-aug 0.5 --alpha-nonpspl 1.0 \
  --lr 5e-5 --epochs 6 --resume
```

**5. Evaluate on a held-out cache** (must be disjoint — use different shard indices):
```bash
python -m pipeline.sim_v5.evaluate_v5 --ckpt runs/stage6.pt --cache data/mm_test \
  --out eval/stage6 --device mps
```

**6. Make the figures:**
```bash
# 14-figure diagnostic suite (pass --ckpt to also get the temporal figures 13/14)
python -m pipeline.sim_v5.plots_v5 --preds eval/stage6 --cache data/mm_test \
  --out figures/stage6 --ckpt runs/stage6.pt

# per-event probability-evolution plots (50 of class 2 = NonPSPL)
python -m pipeline.sim_v5.plot_evolution_cloud --cache data/cache/shard_00000.h5 \
  --ckpt runs/stage6.pt --class-idx 2 --n 50 --out figures/evolution
```

---

## Key concepts you need to read the numbers

- **Detectability-conditioned labelling.** `assemble.py` labels by what is observable in the
  window. An event whose peak falls outside the season, or whose amplitude is buried by noise,
  is **Flat**. A binary whose anomaly is below the Δχ² floor is **PSPL**. This removes the label
  noise that would otherwise punish the classifier for not seeing what isn't there.

- **The real-time cascade (`t_anom`).** For truncation augmentation, `train_v5._apply_truncation`
  relabels by what is observable *in the revealed window*: a binary reads **PSPL until its
  anomaly onset day `t_anom`**, then NonPSPL. This is what stops the model flagging a binary
  before the caustic is on screen. Measured effect: premature NonPSPL flagging at day 11 dropped
  from 42% to 9%.

- **`keep_prob` reweighting is asymmetric.** NonPSPL rows have `keep_prob=1`; the byproduct
  PSPL/Flat rows are subsampled. So **recall must NOT be reweighted, but precision/purity MUST**.
  `evaluate_v5.population_weights` handles this (and uses float64 — float32 cumsum overshoots 1).

- **Completeness at fixed purity** is the headline, not accuracy or F1. Accuracy is meaningless
  (Flat+PSPL are ~61% of events); bare recall is degenerate. The threshold is fixed on a
  validation split and applied frozen to test.

## Distributed runs (AWS)

Generation/binning/inference scale out across free-tier spot instances. The orchestration
scripts (fleet launchers, controller) are operational and account-specific, kept out of the
public tree. The modules themselves take `--bucket`/`--seed-base`/`--worker`/`--workers` so the
same code runs one shard locally or thousands in-region.

## Verified

All 21 modules import cleanly and every stage CLI responds to `--help`
(checked 2026-07-26). The commands above are the exact invocations used to produce the
`stage6` model.

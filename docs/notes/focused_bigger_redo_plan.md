# Focused Bigger Redo Plan

Date: 2026-07-05

Context: the useful trained checkpoints are gone, so the project needs a retrain anyway. Since the redo cost is unavoidable, the next run should be larger and cleaner than the old 600k-per-prior setup, but it must not repeat the old dense-storage design.

## Objective

Build a new reproducible microlensing training pipeline that can scale beyond the old runs while staying storage-aware.

Primary target:

```text
Generate 100M candidate events.
Store compact metadata for all candidates.
Store dense selected curves for 10M events.
Train final models on 3M -> 10M selected dense events.
```

Stretch target:

```text
Generate 300M-1B candidates.
Store dense selected curves for up to 30M events only if 10M still improves.
```

Non-goal:

```text
Do not store or train 100M-1B dense full lightcurves.
Do not store 10B dense curves.
```

## Core Reason

The old recovered results show that flat-vs-lensing is solved, while PSPL-vs-binary fails under distribution shift:

| Old Scenario | Accuracy | PSPL Recall | Binary Recall |
|---|---:|---:|---:|
| distinct -> distinct | 0.9965 | 0.999 | 0.990 |
| distinct -> general | 0.7072 | 0.999 | 0.122 |
| general -> distinct | 0.7660 | 0.298 | 1.000 |
| general -> general | 0.7040 | 0.295 | 0.817 |

Therefore the redo should spend data budget on:

- PSPL/binary boundary cases
- low-q planetary binaries
- smooth/ambiguous binaries
- calibration regimes
- broad general population

Not on endless easy flat or easy distinct-caustic duplicates.

## Mandatory Storage Change First

Current dense format is too expensive:

```text
flux + delta_t + timestamps ~= 81 KiB/event raw
10M events ~= 810 GB raw
100M events ~= 8.1 TB raw
```

Before any big generation, implement compact storage:

1. Store `time_grid` once per shard, not `timestamps` per event.
2. Store cadence mask or valid indices, not dense `delta_t`.
3. Reconstruct `delta_t` in the loader from `time_grid + mask`.
4. Store `flux` as `float16` first; test `uint16` later.
5. Store per-row metadata:
   - seed
   - label
   - source component
   - binary preset
   - physical params
   - detectability/anomaly metrics
   - simulator version/commit

Expected storage after compacting:

| Dataset | Rough Size |
|---|---:|
| 10M selected dense | ~140-200 GB |
| 30M selected dense | ~420-600 GB |
| 100M metadata-only | ~10-50 GB |
| 1B metadata-only | ~100-500 GB |

## Data Plan

### Candidate Universe

Generate candidates in CPU shards. Candidates can be metadata-only unless selected for dense storage.

Initial target:

```text
100M candidates
```

Candidate mix:

| Component | Candidate Share |
|---|---:|
| Flat | 10% |
| PSPL normal | 20% |
| PSPL hard/mimic | 15% |
| Binary baseline/general | 20% |
| Binary planetary | 20% |
| Binary stellar | 10% |
| Binary distinct/caustic | 5% |

### Dense Selected Pool

First main dense target:

```text
10M selected dense events
```

Recommended 10M mix:

| Component | Dense Count |
|---|---:|
| Flat | 1.0M |
| PSPL normal | 2.0M |
| PSPL hard/mimic | 1.5M |
| Binary baseline/general | 2.0M |
| Binary planetary | 1.5M |
| Binary stellar | 1.0M |
| Binary distinct/caustic | 1.0M |
| **Total** | **10.0M** |

Hold back separate test sets. Do not let train selection contaminate test.

## Training Ladder

### Run 0: Compact Format Validation

```text
300k dense events
```

Purpose:

- verify compact storage
- verify exact `delta_t` reconstruction
- compare float16 vs float32
- validate train/evaluate compatibility

Pass criteria:

- metrics match old-format baseline within noise on a small replicated dataset
- no shape/provenance mismatches
- evaluation can recover per-row params after shuffling

### Run 1: First Bigger Model

```text
3M selected dense events
```

Purpose:

- prove bigger mixed-regime training improves the broad PSPL/binary boundary
- tune class mix and loss weights

Suggested model:

```text
d_model=64
n_layers=4
hierarchical=true
aux_head=true
attention_pooling=true
stage1_weight=0.5
stage2_weight sweep: 1.0, 2.0, 3.0
```

Pass criteria:

- broad-population accuracy >= 0.78
- PSPL recall >= 0.65
- binary recall >= 0.75
- distinct accuracy >= 0.95
- flat recall >= 0.995

### Run 2: Main Model

```text
10M selected dense events
```

Purpose:

- final serious redo target
- train using the best Run 1 settings
- evaluate all stress sets

Pass criteria:

- materially beats Run 1 on broad PSPL/binary metrics
- planetary low-q recall is separately reported
- calibration/ECE is computed numerically
- ambiguity shows up as lower confidence, not random overconfident mistakes

### Run 3: Stretch

```text
30M selected dense events
```

Only run if 10M is still improving. Stop if 10M saturates.

## Evaluation Sets

Keep all tests separate and frozen:

| Test Set | Size | Purpose |
|---|---:|---|
| Broad/general | 500k-1M | Honest population performance |
| Distinct/caustic | 300k | Compare to old 99.65% d2d regime |
| Planetary low-q | 300k-500k | Science-critical exoplanet regime |
| Stellar binary | 300k | Equal/high-q binary stress |
| Hard boundary | 300k-500k | PSPL-like binaries and binary-like PSPL |
| Calibration | 300k | Reliability/ECE only; never train on it |

Every run must report:

- full confusion matrix
- PSPL-vs-binary-only confusion
- per-class precision/recall/F1
- ROC-AUC
- ECE/reliability
- metrics by `u0`, `q`, `s`, `rho`, `tE`, peak magnification, SNR
- inference latency with hardware named
- cost/hardware/provenance note

## Compute Plan

CPU generation:

- local machine first for benchmark
- AWS CPU Spot/free-credit workers for parallel shards
- workers are stateless: generate shard, upload, terminate

GPU training:

- start with one stable SSH GPU VM
- use AWS `g4dn/g6` Spot, Runpod, Lambda, or similar
- avoid notebooks
- avoid A100/H100 until 3M or 10M proves the strategy

## Immediate Implementation Order

1. Add compact shard format.
2. Add shard manifest and per-row provenance.
3. Add compact loader that reconstructs `delta_t`.
4. Add integrity checker.
5. Add multi-shard training loader.
6. Add ECE/calibration metrics.
7. Add latency benchmark.
8. Run 300k compact validation.
9. Generate 10M candidates.
10. Select/densify 3M.
11. Train Run 1.
12. Generate/select/densify 10M.
13. Train Run 2.

## Claude Audit Questions

Ask Claude to audit only these:

1. Is the 10M selected mix right, or should binary planetary get more weight?
2. Is `stage2_weight` sweep `[1,2,3]` enough?
3. Should the model remain 3-class, or add an ambiguity/detectability head?
4. Can `delta_t` be exactly reconstructed from mask + global time grid?
5. Is `float16` flux safe for classification accuracy?
6. What detectability/anomaly metric should decide which candidates become dense?
7. What stop rule should prevent wasting compute beyond 10M?

## Bottom Line

Because checkpoints are gone, redo bigger. But bigger means:

```text
100M generated candidates
10M selected dense training curves
3M -> 10M training ladder
30M only if still improving
```

The project should become a selection and provenance pipeline, not a giant pile of dense HDF5 files.

# Next Training Strategy For Claude Audit

Date: 2026-07-05

Purpose: propose the next best microlensing training strategy after reviewing the active `binml` code, recovered results, old research guide, and thesis audit material. This is intentionally written as an audit target for Claude or another reviewer before implementation.

## Project-Root Inventory Reviewed

Project root:

```text
/Users/kunalbhatia/Desktop/Research/microlensing
```

Relevant sibling material reviewed, not just `binml`:

| Path | Role | Training-strategy relevance |
|---|---|---|
| `binml/` | Active ML code: simulator, model, train, evaluate | Source of truth for what can be retrained now. |
| `binml_recovered_2ea3364/` | Recovered BinML results/checkpoints metadata | Contains old eval outputs and recovered research guide; not active code. |
| `masters-thesis/` | Thesis + paper assets + recovered eval copy | Contains the strongest audit/provenance docs and corrected tables. |
| `masters-thesis/paper_plan.md` | Prior audit plan | Key finding: 99.65% is real but favorable; broad population is ~70%. |
| `masters-thesis/paper_assets.md` | Review assets | Contains corrected simulation table, architecture table, and cross-eval table. |
| `masters-thesis/provenance.md` | Source map | Maps every headline number to recovered files; weights are gone. |
| `masters-thesis/paper_assets/*.csv` | Structured evidence | Useful for any new paper/plan table. |
| `masters-thesis/recovered_eval/` | Four cross-evaluation JSON/report dirs | Duplicates recovered evals in a clean thesis-side location. |
| `_audit/` | Annotation and figure scripts | Includes real caustic/lightcurve figure generator, but paths need current-root fixes. |
| `thesis_feedback_round2.pdf.annots.json` | Advisor comments | Scientific constraints: caustics source plane, correct references, compute transparency. |
| `_backups/` | Tarball backups | Not used for planning unless something is missing. |

Important correction from this root-level pass:

- `masters-thesis/paper_assets.md` already contains a clean corrected simulation-parameter table and should be treated as a high-value source for any next paper/training writeup.
- `masters-thesis/provenance.md` is the strongest traceability document; any new claim should copy that style.
- `_audit/gen_figures.py` is useful but currently hardcodes `/Users/kunalbhatia/Downloads/Research/...`; if reused, update it to `/Users/kunalbhatia/Desktop/Research/...` or make it relative.
- The old `RESEARCH_GUIDE.md` contains a transformer plan and expected numbers; it is useful historically, but active code and recovered evidence supersede it.

## Executive Recommendation

Do not simply generate billions of dense lightcurves and train on all of them.

Train a **mixed-regime, selected 1M-event model first**, then scale to **3M-10M selected dense curves** only if cross-regime validation improves. Generate broader CPU-only candidate pools as metadata/seeds, but keep dense curves only for selected training/evaluation examples.

Recommended first serious training distribution:

| Component | Dense Train Count | Why |
|---|---:|---|
| Flat | 200k | Flat-vs-lensing is already solved; keep enough for calibration without wasting half the data. |
| PSPL easy/normal | 200k | Prevent binary overcall; match binary `u0`, `tE`, `t0` distributions. |
| PSPL hard mimic set | 150k | Smooth, high-magnification, low-SNR, or broad events likely to mimic binaries. |
| Binary baseline/general | 200k | Broad realistic prior, including non-caustic smooth binaries. |
| Binary planetary | 150k | Scientifically important and harder low-q regime. |
| Binary distinct/caustic | 100k | Keeps sensitivity to strong caustic signatures without overfitting to the easy regime. |
| **Total** | **1.0M** | Fits a real first run and avoids the old matched-regime trap. |

Keep validation/test separate and deliberately multi-regime:

| Split | Count | Composition |
|---|---:|---|
| Validation | 300k | Balanced across flat, PSPL, baseline binary, planetary, stellar, distinct. |
| Test A: broad population | 300k-700k | Baseline/general prior, balanced labels. |
| Test B: caustic/distinct | 300k | Matched to old d2d for comparability. |
| Test C: planetary stress | 300k | Low-q planetary regime, binned by `q`, `s`, `u0`, `rho`. |
| Test D: hard boundary | 300k | PSPL-like binaries and binary-like PSPL cases selected by a weak baseline model. |

## Evidence From Existing Folder

### Active Code

- Simulator: `code/simulate.py`
- Trainer: `code/train.py`
- Model: `code/model.py`
- Evaluation: `code/evaluate.py`
- Current architecture: compact hierarchical CNN-GRU with optional attention pooling.
- Current data shape: `6912` time points per event, dense HDF5 arrays.
- Current binary presets: `distinct`, `planetary`, `stellar`, `baseline`.

### Recovered Cross-Evaluation Results

Recovered summaries live under:

```text
../masters-thesis/recovered_eval/
```

The old results show the key failure mode:

| Scenario | Accuracy | Flat Recall | PSPL Recall | Binary Recall | Interpretation |
|---|---:|---:|---:|---:|---|
| distinct -> distinct | 0.9965 | 1.000 | 0.999 | 0.990 | Easy matched caustic regime works extremely well. |
| distinct -> general | 0.7072 | 1.000 | 0.999 | 0.122 | Distinct-trained model misses most general binaries as PSPL. |
| general -> distinct | 0.7660 | 1.000 | 0.298 | 1.000 | General-trained model overcalls binary on distinct-like/non-general boundary. |
| general -> general | 0.7040 | 1.000 | 0.295 | 0.817 | Broad PSPL-vs-binary boundary is unstable. |

Conclusion: the next run must target **PSPL-vs-binary calibration under distribution shift**, not headline distinct accuracy.

### Existing Thesis/Paper Evidence To Preserve

Do not throw away the prior audit work. It already establishes:

- corrected simulation parameters in `masters-thesis/paper_assets/sim_parameters.csv`;
- corrected architecture parameter count in `masters-thesis/paper_assets/architecture.csv`;
- 2x2 cross-eval metrics in `masters-thesis/paper_assets/cross_eval.csv`;
- source provenance in `masters-thesis/provenance.md`;
- scope guardrails in `masters-thesis/paper_assets.md`.

The next training run should produce artifacts that can be dropped into the same provenance pattern:

```text
run_manifest.json
simulation_manifest.jsonl
training_config.json
evaluation_summary.json
classification_report.txt
calibration_metrics.json
latency_benchmark.json
cost_and_hardware.md
```

## What Went Wrong Before

1. The 99.65% result was real but scoped to a favorable matched distinct regime.
2. Broad-population performance stayed near 70%.
3. Flat classification was solved, so adding many more flat examples is low value.
4. The binary class mixed genuinely detectable caustic events with physically PSPL-like smooth binaries.
5. The model was forced to make a hard class decision even when the lightcurve may be scientifically ambiguous.
6. Old research docs mention a transformer architecture; active code is CNN-GRU, so Claude should not rely on the old architecture section.

## Training Objective

Primary objective:

```text
Maximize robust PSPL-vs-binary discrimination across binary morphology regimes,
while preserving near-perfect flat-vs-lensing separation.
```

Secondary objective:

```text
Expose and calibrate ambiguity instead of hiding it.
```

For scientific use, a calibrated "ambiguous / needs follow-up" output is more valuable than forcing smooth binaries into a brittle PSPL/Binary split.

## Recommended Label Strategy

Keep the current three labels for compatibility:

```text
0 = Flat
1 = PSPL
2 = Binary
```

But add evaluation buckets and optionally a future auxiliary target:

```text
detectable_binary = binary with caustic/anomaly signal strong enough to distinguish
smooth_binary = binary parameters but PSPL-like photometric morphology
```

Do not change labels in the first run. Instead:

1. Store morphology metadata and detectability metrics.
2. Report metrics separately for detectable vs smooth binaries.
3. Consider a later 4-head output:
   - flat
   - single-lens
   - detectable binary
   - ambiguous/smooth binary

Claude audit question: should the next architecture add an explicit ambiguity head, or should this remain an evaluation-only concept first?

## Data Generation Plan

### Phase 1: Benchmark Shards

Generate small, reproducible shards:

```text
30k events per shard:
10k flat / 10k PSPL / 10k binary
```

Use these to measure:

- events/sec
- file size/event
- failure rate per preset
- CPU cost per million events
- training throughput
- compression impact

### Phase 2: 1M Dense Training Set

Generate separate shards by component, not one monolith:

```text
train_flat_*.h5
train_pspl_normal_*.h5
train_pspl_hard_*.h5
train_binary_baseline_*.h5
train_binary_planetary_*.h5
train_binary_distinct_*.h5
```

This requires either:

1. a multi-shard training loader, or
2. a controlled merge step that records component provenance.

Preferred: multi-shard loader.

### Phase 3: Candidate Universe

Generate many more events as compact metadata/seeds:

```text
10M-100M candidate parameter rows
1M-10M dense selected rows
```

Do not store dense 10B curves. Use 10B only as a generated/searchable parameter universe if needed.

## Storage Strategy

Before scaling past ~1M dense examples:

1. Store global `time_grid` once, not `timestamps` per event.
2. Stop storing dense `delta_t`; reconstruct it from `time_grid + mask`.
3. Store a bitmask or valid indices for cadence gaps.
4. Test `float16` flux storage against float32 accuracy.
5. Store compact metadata/seeds for all generated candidates.
6. Keep dense lightcurves only for selected training/evaluation examples.

Rough storage:

| Format | Approx/Event | 1M | 10M |
|---|---:|---:|---:|
| Current dense | ~81 KiB | ~81 GB | ~810 GB |
| No per-event timestamps | ~54 KiB | ~54 GB | ~540 GB |
| float16 flux + mask-derived delta_t | ~14-16 KiB | ~14-16 GB | ~140-160 GB |
| metadata/seed only | ~100-500 B | ~0.1-0.5 GB | ~1-5 GB |

## Model Strategy

Start with the current CNN-GRU, not a new transformer.

Baseline run:

```bash
python code/train.py \
  --data <multi-shard-or-merged-train.h5> \
  --output results/checkpoints \
  --epochs 100 \
  --batch-size 128 \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --warmup-epochs 5 \
  --d-model 64 \
  --n-layers 4 \
  --dropout 0.25 \
  --window-size 5 \
  --hierarchical \
  --use-aux-head \
  --attention-pooling \
  --stage1-weight 0.5 \
  --stage2-weight 2.0 \
  --aux-weight 0.5 \
  --stage2-temperature 1.5 \
  --use-amp \
  --save-every 5
```

Why these changes:

- `d_model=64`: more capacity than the recovered `d32`, still small.
- `stage1-weight=0.5`: flat-vs-nonflat is easy; reduce dominance.
- `stage2-weight=2.0`: focus learning on PSPL-vs-binary boundary.
- `stage2-temperature=1.5`: less soft than old 2.0, but not brittle.
- `epochs=100`: enough for convergence diagnostics; early stopping can be added later.

Claude audit question: should `stage2-weight` be swept over `[1.0, 2.0, 3.0]` before the 1M final run?

## Training Ladder

### Run 0: Pipeline Sanity

```text
300k total
100k flat / 100k PSPL / 100k binary mixed
```

Purpose:

- verify storage changes
- verify data loader
- compare float16 vs float32
- ensure recovered d2d/g2g-style eval scripts still work

### Run 1: First Serious Model

```text
1M mixed-regime train
300k validation
4 test sets
```

Success criteria:

- Flat recall >= 0.995 on all tests.
- Broad-population accuracy >= 0.78.
- Broad binary recall >= 0.75 while PSPL recall >= 0.65.
- Planetary binary recall reported by `q` bins, not just macro average.
- Distinct accuracy remains >= 0.95.
- Calibration/ECE computed numerically.

### Run 2: Hard-Example Selected

Generate 10M+ candidate metadata/seeds, densify selected examples:

```text
3M dense train:
1M original mixed
1M hard PSPL/binary boundary
500k planetary low-q
500k smooth-binary/ambiguous regime
```

Success criteria:

- broad-population accuracy improves materially over Run 1.
- PSPL/Binary confusion becomes less one-sided.
- confidence is calibrated; ambiguous cases show lower confidence.

### Run 3: 10M Selected Max

Only if Run 2 still improves:

```text
10M selected dense train
metadata universe much larger
single/multi GPU depending budget
```

Do not proceed if Run 2 saturates.

## Evaluation Plan

Every serious run must produce:

1. 2x2 old-style cross-eval:
   - mixed/general model -> broad test
   - mixed/general model -> distinct test
   - optional distinct-only model -> both tests for comparison
2. Per-class precision/recall/F1.
3. PSPL-vs-binary binary-only confusion.
4. Metrics by:
   - `u0`
   - `q`
   - `s`
   - `rho`
   - `tE`
   - peak magnification
   - anomaly/caustic detectability proxy
   - SNR / baseline magnitude
5. ECE and reliability curves.
6. Early-detection curves only if actually run and saved.
7. Inference latency measurement with script and hardware noted.

Claude audit question: are `q`, `s`, and caustic/anomaly metrics recoverable from current parameter datasets after shuffling, or do we need a more explicit per-row metadata table?

## Compute Plan

Cheap mode:

- Generate CPU shards locally and/or AWS CPU Spot.
- Train first on one GPU.
- Use AWS/Runpod/Vast/Lambda only as stable SSH VMs, not notebooks.

GPU order:

1. Local GPU if available.
2. AWS `g4dn.xlarge`/`g6.xlarge` Spot after quota.
3. Runpod/Lambda stable VM.
4. Avoid A100/H100 until Run 1 proves the data strategy.

The old HPC 40-A100 28-minute number is useful for throughput reference, not as the default plan.

## Implementation Tasks Before Run 1

1. Add a shard manifest writer to `simulate.py`.
2. Store global `time_grid` once instead of per-event timestamps.
3. Add compact mask/valid-index storage, or at least prepare for it.
4. Add a shard integrity checker.
5. Add a multi-shard loader or deterministic merge tool.
6. Add per-row provenance:
   - component/preset
   - seed
   - event type
   - physical params
   - detectability/anomaly metrics
7. Add numeric calibration/ECE output to `evaluate.py`.
8. Add inference latency benchmark.
9. Add a single command/script for the 300k sanity run.
10. Add a reproducibility note with commit, env, hardware, seeds, and cost.

11. Port the useful root-level audit assets into the active workflow:
    - make `_audit/gen_figures.py` path-relative or copy a fixed version into `binml/scripts/`;
    - keep `paper_assets/*.csv` synchronized with newly generated results;
    - update `provenance.md` style source mapping for new runs;
    - preserve old cross-eval rows as baseline comparators, never overwrite them.

## Risks

- More data may not fix physically ambiguous smooth binaries.
- Overweighting binary can destroy PSPL recall, as the old general -> distinct result hints.
- Distinct/caustic examples can inflate headline accuracy and hide broad-population weakness.
- Storage becomes the real bottleneck unless compact formats land first.
- Current trainer loads full arrays; it will not scale to many dense shards without a loader change.
- Old weights are gone, so all next claims must be freshly reproducible.

## What Claude Should Audit

Ask Claude to check:

1. Is the 1M distribution the right balance, or should flat be reduced further?
2. Should PSPL hard examples be generated by parameter heuristics first or by model-mined false positives/false negatives?
3. Should the label space stay 3-class or add a fourth ambiguous/smooth-binary target?
4. Is `stage2-weight=2.0` justified, or should it be swept first?
5. Is `d_model=64` enough, or should the first serious run include `d_model=128` comparison?
6. Can `delta_t` be reconstructed exactly from a bitmask and global time grid for all current evaluation plots?
7. Are parameter datasets correctly aligned with shuffled rows, especially for per-row morphology metrics?
8. What is the minimum storage format change needed before 1M, 3M, and 10M runs?
9. What failure criterion should stop scaling before wasting compute?
10. What results would make this scientifically publishable rather than just bigger?

## Bottom Line

The best next strategy is not "maximum lightcurves." It is:

```text
mixed-regime training + hard-boundary mining + compact storage + calibrated evaluation
```

Start with 300k sanity, then 1M mixed selected, then 3M hard-selected if the PSPL/binary boundary actually improves. Dense 10M is a ceiling, not a starting point. Dense 10B is not the plan.

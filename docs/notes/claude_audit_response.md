# Claude Audit Response

Date: 2026-07-05

Audit of the three GPT-authored planning docs against the **actual** `binml/code` source
(`simulate.py`, `model.py`, `train.py`, `evaluate.py`). Every code claim below was verified
by reading the source, with line references.

Docs audited:

- `docs/next_training_strategy_for_audit.md`
- `docs/focused_bigger_redo_plan.md`
- `docs/aws_cpu_gpu_scale_plan.md`

## Verdict Summary

The strategy is **sound and I endorse the core thesis**: compact storage first, mixed-regime
training with hard-example selection, calibrated evaluation, scale only if the PSPL/binary
boundary actually improves. The docs are internally consistent and correctly diagnose the real
failure mode (PSPL-vs-binary under distribution shift, not headline accuracy).

Two of the plans' assumptions are **wrong or incomplete** against the code, and one claimed gap
is **worse than described**. Details below.

## Code Facts — Verified

| Claim in the plans | Verdict | Evidence |
|---|---|---|
| Simulator writes `flux`, `delta_t`, `timestamps`, `labels`, `m_base`, `params_{class}` | TRUE | `simulate.py:1699-1767` |
| 6912 points/event, 15-min cadence, 72-day season | TRUE | `simulate.py:205-224, 691, 721` |
| `timestamps` is identical for every event (redundant) | TRUE — fully redundant | `simulate.py:1736-1737` writes `np.tile(time_grid, ...)` |
| `delta_t` is exactly reconstructable from `time_grid + mask` | TRUE — that is literally how it is computed | `simulate.py:1387`, `compute_delta_t():1151-1169` |
| Trainer loads full arrays into RAM (won't scale to many shards) | TRUE | `train.py:452-460` does `f['flux'][:]` into `/dev/shm`; `SharedRAMLensingDataset` |
| Presets `baseline`, `planetary`, `stellar`, `distinct` exist | TRUE | `simulate.py:726-770, 1281` |
| Model default is 3-class, hierarchical, aux head, attention pooling | TRUE (as defaults) | `model.py:284-301` |
| Recovered model was `d_model=32` | PARTLY — code **default** is `d_model=16`, not 32 | `model.py:141` `DEFAULT_D_MODEL = 16` |
| `evaluate.py` lacks numeric ECE | TRUE, and worse — see below | `evaluate.py:1634-1667` |

## Three Corrections The Plans Need

### 1. The cadence mask is NOT stored — reconstruction requires adding it

The plans say "reconstruct `delta_t` from `time_grid + mask`." Correct in principle, but the
**mask is currently discarded**. It is generated at `simulate.py:1383`
(`mask = np.random.random(n) > mask_prob`), used to zero masked flux and to compute `delta_t`,
then thrown away. It is never written to HDF5.

Consequence for the compact-storage task: **storing the mask is not optional, it is the
enabling step.** Without it, `delta_t` cannot be regenerated from a stored `time_grid` alone.
The plans list "store cadence mask" as one bullet among many; it should be the *first* one,
because it is the precondition for dropping the dense `delta_t` array.

Cheap recovery note: masked flux is set to `0.0` (`simulate.py:1384`) and flux is a magnitude
(never legitimately 0), so `mask ≈ (flux != 0)` is recoverable from existing files if needed
for back-compat. But for the new format, store an explicit packed bitmask (6912 bits = 864
bytes/event) — clean, unambiguous, and cheap.

### 2. Physics params are NOT row-aligned with the shuffled data — this is the real provenance hole

Audit question "are parameter datasets correctly aligned with shuffled rows?" — **No, not for
the physics params.**

- `m_base` **is** globally aligned to the shuffled `flux`/`labels` rows (`simulate.py:1703, 1722`).
- The rich params (`q`, `s`, `u0`, `rho`, `tE`, `t0`, ...) are stored **per class** as separate
  `params_{class}` struct arrays (`simulate.py:1744-1767`), appended in the order events appear
  in the shuffled stream (`params_by_class[type].append(...)` at `simulate.py:1723`).

So `params_binary[k]` is the k-th *binary* event in shuffled order. To recover
`(q, s, u0, rho)` for global row `i` you must: read `labels[i]`, then count how many events of
that class preceded row `i`. That mapping is derivable **in-file** but is fragile and **breaks
the moment you re-shuffle at train time, subset, or merge shards** — exactly what the multi-shard
plan requires.

This is precisely why "metrics by `q`, `s`, `rho`, `u0`" and per-row detectability selection are
hard today. **Fix: write a single global, row-aligned `params` struct array (or per-field
datasets) indexed by the same global row order as `flux`.** Make this part of the mandatory
storage change, not a later nicety. It is the single highest-leverage provenance fix and unlocks
half the evaluation plan.

### 3. `evaluate.py` computes NO numeric calibration metric at all

The plans call ECE a "gap to add." It is a complete absence: `evaluate.py:1634-1667` draws a
`sklearn` reliability diagram and a confidence histogram, but there is **no scalar ECE, no MCE,
no Brier score** computed or saved anywhere. Confirmed: no `expected_calibration_error`, no
`|prob_true - prob_pred|` aggregation in the file. So "calibration/ECE computed numerically" in
every run's pass criteria is currently unachievable without new code. Add: ECE (equal-width and
equal-mass bins), MCE, Brier, and dump them to `calibration_metrics.json`.

## Answers To The Explicit Audit Questions

**Q: Can `delta_t` be reconstructed exactly from mask + global time grid?**
Yes, exactly — it is defined that way (`compute_delta_t`). Caveat: you must first *store the
mask* (see correction 1). Masked and first-valid points both get `delta_t = 0`; that is by
design and is reproduced deterministically from the mask, so there is no information loss.

**Q: Is `float16` flux safe for classification accuracy?**
Probably yes, but validate it, and store it right. Flux is an AB magnitude (~20-25). float16 has
~3 significant digits; near mag 24 the ULP is ~0.01-0.016 mag, which is marginal at the faint
end and comparable to photometric noise. Safer: store flux **mean-subtracted or as offset from
`m_base`** (small dynamic range → float16 is comfortably precise), not raw magnitudes. Run 0
already A/Bs float16 vs float32 — keep that gate and require ≤ noise-level metric delta before
committing.

**Q: Should the label space stay 3-class or add a 4th ambiguous/smooth-binary head?**
Stay 3-class for Run 0/1. "Ambiguity" has no clean physical ground-truth label, so a 4th
softmax class would be trained on an arbitrary threshold and hurt more than help. The model
already has a 3-class *aux* head (`model.py:181, 288`) — that is not an ambiguity head.
Do it as **evaluation buckets + a stored detectability metric first** (which both plans also land
on). Revisit a 4th head only if the metadata shows a stable, separable smooth-binary population.

**Q: Detectability/anomaly metric to decide which candidates go dense?**
Use a PSPL-deviation statistic: fit the best single-lens (PSPL) model to each binary curve and
record `Δχ²(PSPL vs true)` or `max|A_binary − A_PSPL| / σ`. The simulator already reasons about
caustic crossings via the `require_caustic` flag (`simulate.py:767, 1220, 1336`), so also store a
boolean caustic-crossing flag and the peak anomaly amplitude. Select dense examples to
**oversample the low-Δχ² (smooth/ambiguous) binaries and low-`q` planetary events** — the regimes
the old evals failed on — rather than the easy high-anomaly caustic crossers.

**Q: Is `stage2_weight` sweep `[1, 2, 3]` enough?**
Yes as a coarse first pass. Add `stage2_temperature` interaction awareness — the plan sets temp
1.5 while sweeping weight; if weight=3 overcorrects PSPL recall downward (the `general→distinct`
failure mode: PSPL recall collapsed to 0.298), that is the signal to stop, not to push higher.
Sweep on the 1M/3M rung, not the 10M rung.

**Q: Is `d_model=64` enough, or compare `d_model=128`?**
`d_model=64` is a reasonable 4x jump over the code default of 16. Include a `d_model=128` point
**only at the 3M rung**, where capacity can actually be fed. At 300k-1M, 64 vs 128 will be noise
and waste compute. Note the plan's premise "recovered d32" is off by the code default (16) —
doesn't change the recommendation, but correct the baseline in the writeup.

**Q: Is the 1M / 10M class mix right, or reduce flat further?**
Reduce flat further. Flat-vs-lensing is solved at ~1.0 recall across every old scenario, so flat
is pure calibration ballast. 20% (1M plan) and 10% (10M plan) are both too high — drop to
~5% (50k of 1M, 500k of 10M) and move the freed budget into **PSPL-hard-mimic** and
**smooth/low-q binary**, which is where the boundary is unstable. Everything else in the mix is
well-reasoned.

**Q: What stop rule prevents wasting compute past 10M?**
Stop when a data doubling buys < ~0.5-1% absolute broad-population accuracy **and** the
PSPL/binary confusion has stopped becoming less one-sided (track the ratio of PSPL→binary vs
binary→PSPL error rates; convergence toward 1.0 is the real target, not raw accuracy). If
smooth-binary recall plateaus while planetary-low-`q` is still climbing, that says "more selective
data," not "more data."

## Storage Math — Checked

All storage numbers in the plans are arithmetically correct:
`3 arrays × 6912 × 4 B = 82,944 B ≈ 81 KiB/event`; dropping `timestamps` saves exactly 1/3;
float16 flux + bitmask ≈ 14-15 KiB/event. No corrections.

## Recommended Reordering Of "Immediate Implementation"

The plans' task lists are right but mis-prioritized. Do them in this dependency order:

1. **Write a global, row-aligned `params` table** (correction 2) — unlocks selection + per-`q`/`s` eval.
2. **Store the cadence mask** (correction 1) — precondition for dropping dense `delta_t`.
3. Drop per-event `timestamps`; store `time_grid` once.
4. Compact loader that rebuilds `delta_t` from `time_grid + mask`; verify bit-exactness vs old files.
5. Numeric ECE/MCE/Brier in `evaluate.py` → `calibration_metrics.json` (correction 3).
6. Detectability metric (`Δχ²` PSPL-fit + caustic flag) written per row.
7. Multi-shard streaming loader (replaces the `/dev/shm` full-array load in `train.py`).
8. Shard manifest + integrity checker.
9. Run 0 (300k) compact-format validation with the float16 A/B gate.

Items 1-2 are the ones the plans undersell. Everything after 3 matches the plans.

## Implementation Status (2026-07-05)

Corrections 1-3 are now **implemented and tested** in code:

| Fix | Change | Where |
|---|---|---|
| Store cadence mask | New packed-bit `mask` dataset (864 B/event); `unpack_mask()` helper; `delta_t` proven bit-exact via `compute_delta_t(time_grid, unpack_mask(...))` | `simulate.py` (v4.2.0) |
| Row-aligned params | New global `params` struct dataset in shuffled row order; NaN marks fields not applicable to a class; per-class `params_{class}` kept for back-compat | `simulate.py` (v4.2.0) |
| Numeric calibration | `compute_calibration_metrics()` → ECE (equal-width + adaptive), MCE, multiclass Brier; added to `metrics['calibration']`, logged, and written to `calibration_metrics.json` | `evaluate.py` (v4.2.0) |

Verified by a round-trip test: mask packs/unpacks losslessly and `delta_t` reconstructs
bit-exactly; the global params table places each binary event's `q`/`s`/`rho` at its true
global index (not class-grouped) with NaN for inapplicable fields; ECE is ~0 for a perfectly
calibrated predictor and >0.9 for an overconfident-and-wrong one, with a sane Brier score.

### Beyond the fixes — two upgrades that operationalize the strategy

| Upgrade | Change | Where |
|---|---|---|
| Per-event detectability | Every event now stores `peak_magnification`, `snr`, and (binary only) `anomaly_dchi2` + `max_anomaly` — the Δχ² of the true curve vs the matched single-lens (same u0/tE/t0). This is the per-row score the whole "hard-example selection" plan depends on, and it flows into the aligned `params` table for free (numeric fields). | `simulate.py` (v4.2.0) |
| Per-regime metrics | New `analyze_metrics_by_parameter()` reads the aligned `params` table and reports accuracy + per-class recall binned by `u0/tE/q/s/rho/peak_mag/snr/anomaly_dchi2` → `metrics_by_parameter.json`. Delivers the plans' "metrics by parameter" requirement; gracefully skips pre-v4.2.0 files. | `evaluate.py` (v4.2.0) |

Verified: detectability cleanly separates smooth binaries (`anomaly_dchi2 ≈ 0`) from caustic
binaries (large `Δχ²`); and the per-regime report surfaces the exact failure mode the strategy
targets — in a synthetic test, binary recall was 0.34 in the lowest-anomaly bin vs 1.00 in the
highest. Selection can now oversample low-`anomaly_dchi2` binaries and low-`q` planetary events
directly.

**Still open (bigger changes, correctly scoped in the plans):** drop the redundant per-event
`timestamps` and read `time_grid` once in the loader; add the multi-shard streaming loader to
replace the full-array `/dev/shm` load in `train.py`; post-hoc temperature scaling in
`evaluate.py` (raw logits are already captured). The storage/eval format now supports all three.

## Bottom Line

Endorse the strategy. Before any large generation, land in order: (1) row-aligned param table,
(2) stored mask, (3) drop redundant timestamps, (4) numeric ECE. Reduce flat to ~5%, sweep
`stage2_weight` on the small rungs only, keep 3 classes, and gate float16 on Run 0. The plans'
biggest blind spot is not the model — it is that the current shuffled-but-class-grouped param
storage cannot cleanly answer the very per-regime evaluation questions the strategy depends on.

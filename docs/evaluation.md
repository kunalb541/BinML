# Evaluation methodology

This document describes how BinML is evaluated and how to interpret its synthetic benchmark.
The model is a six-class triage system for partial Roman-like seasons. These experiments do not
establish autonomous real-time triggering or performance on real Roman data.

The complete-season headline and per-class results in Section 2 use the **held-out final-test split
(360,472 events)** of a 450,589-event evaluation pool. The other 90,117 rows fix the operating
threshold and are not used for those final-test metrics. Later sections name their own samples:
the 1,000- and 400-event prefix scans, the 14.9-million-event stress suite, and the 9,000-event
baseline experiment are separate artifacts. The reported generation protocol uses shard indices
disjoint from training; the archived training provenance is not complete enough to reconstruct the
original run bit for bit. See [`leakage_audit.md`](leakage_audit.md). Complete-season scoring is
implemented in [`evaluate.py`](../pipeline/evaluate.py).

---

## 1. Four principles baked into the metrics

1. **The headline is completeness at a FIXED purity — not accuracy, not F1.** Accuracy is
   meaningless (Flat+PSPL are ~61% of events); bare recall is degenerate (predict NonPSPL for
   everything → recall 1.0). Completeness at a stated purity is useful for a constrained triage
   queue. The threshold is fixed on a validation split and applied **frozen** to test.

2. **`keep_prob` reweighting is asymmetric.** Every NonPSPL row has `keep_prob = 1` by
   construction; the PSPL/Flat "byproduct" rows that supply the false positives were subsampled
   6.67×. So **recall must NOT be reweighted, but precision/purity MUST** — otherwise purity is
   systematically optimistic by an amount that grows as the model improves. Both a `_sample` and
   a `_population` number are reported.

3. **Detectability conditioning.** A binary whose anomaly falls below the adopted synthetic
   floor is assigned PSPL. This is a label policy, not a proof that the structure is unknowable.
   Recall is therefore reported binned by anomaly Δχ² and across the (log s, log q) plane rather
   than as one number that conflates model behaviour with the chosen floor.

4. **Zero-support classes yield `None`, not 0.0** — so an absent class can't silently drag
   macro-F1 down.

## 2. Headline results

Per-class recall / precision / F1 (population-weighted, selection-corrected argmax; from
`metrics.json` → `argmax_population`):

| metric | Flat | PSPL | NonPSPL | PeriodicVar | LongPeriodVar | Eruptive |
|---|---|---|---|---|---|---|
| recall    | 0.960 | 0.936 | 0.948 | 0.990 | 0.982 | 0.985 |
| precision | 0.982 | 0.988 | 0.716 | 0.952 | 0.851 | 0.795 |
| **F1**    | 0.971 | 0.962 | 0.816 | 0.971 | 0.912 | 0.880 |

Macro-F1 is 0.919. **Read NonPSPL carefully:** the class contains both stellar binaries and
planetary-mass-ratio binary lenses; it is not a planet label. Its recall is 0.95 and its precision
is 0.72. Of the false positives, 94.7% were generated as binaries but demoted to PSPL by the
adopted detectability floor. They remain false positives for this task. The diagnostic could
reflect sub-threshold structure or correlated simulation properties and is not a second, higher
precision estimate.

- **Completeness @ fixed purity: 0.879.** Average precision (population): 0.9515.
- **Binary anomaly calibration:** weighted ECE 0.0533 and weighted Brier score 0.0209 for
  P(NonPSPL) against the NonPSPL-versus-rest target. These replace the less relevant top-label
  multiclass calibration summary.
- **Confusion structure.** The NonPSPL→PSPL anomaly miss rate is 0.049; the comparison value of
  0.055 comes from a different checkpoint in the training lineage, not from a controlled cascade
  on/off experiment, so read it as descriptive. The LongPeriodVar→PSPL false-microlensing
  impostor is 0.009. No large cross-confusions appear elsewhere; the variable classes are
  near-diagonal.

## 3. Partial-season streaming behaviour

BinML is trained with partially revealed seasons. During truncation augmentation, a simulated
binary is relabelled along **Flat → PSPL → NonPSPL** using the per-event `t_anom`. That onset is
computed from the injected, noise-free binary-versus-PSPL deviation; it is truth-informed and is
not an observable timestamp that a live survey broker could calculate.

The stored scan measures the first threshold crossing on a frozen 1,000-binary sample
(`validation/cascade_trace.py` → `cascade_reduce.py`, artifact
`validation/cascade_reproduce_result.json`), sweeping the revealed season in 0.5 d steps with
F146 only:

| | BinML 1.0 (cascade) |
|---|---|
| detected within the season | 89.0% |
| first crossing precedes truth-informed onset | 1.6% of eligible events (95% CI 1.0–2.6%) |
| median lag, detections that were not premature | +5.0 d |

The +5 d latency is conditional on detections that were not premature. It supports partial-season
triage after anomaly evidence has accumulated; it does not demonstrate triggering during a short
planetary perturbation. Rates also change with grid spacing, persistence, bands, threshold policy,
and onset definition, all of which are recorded in the reduction artifact.

The matched 400-event comparison (`validation/cascade_matched_result.json`) is an exploratory
finite-sample risk–coverage analysis. Each arm's threshold is chosen to attain a target detection
count on those same 400 events, and paired outcomes are then evaluated on them. The reported
conditional McNemar values describe that sample; they are not confirmatory population-level
p-values. Threshold selection needs a disjoint calibration set before inferential use. The
augmented arm has fewer premature crossings through most, but not all, of the measured range, so
a causal benefit is not established.

The 1,000-event streaming scan is also conditional on binary eligibility: it contains no Flat,
PSPL, demoted-binary, or variable-star prefix traces. An every-class scan of two held-out-pool shards (15,016 events; `validation/referee_round.json`, `mixed_class_stream{,_f146}`; shipped model, frozen complete-season threshold) measures the burden on our simulator: with all three bands revealed no flat or variable-star event alerts, and the scan raises 0.77 alerts per 1,000 events per day, 89% of them detectable anomalies at the simulated 5.5% prevalence (58% at 1% and 12% at 0.1% by prior shift); with F146 alone it raises 0.98 per day, 67% of them detectable anomalies (26% at 1%), and 4 eruptive variables alert. With three bands nearly all false alerts come from binaries whose anomaly falls below the detectability policy (29 of 32; 3 of the 1,759 generated single lenses with a PSPL label alert); with F146 alone 51 of those single lenses alert as well. The threshold is still the complete-season one; a streaming threshold calibrated on disjoint prefixes is untested. A deployment-style test
still needs disjoint mixed-class prefix calibration.

The stored scan also has provenance limits. The main trace was generated from a dirty source tree
and records no source hash or diff. The matched trace does not record the code or checkpoint hashes.
The files are reproducible reductions of the stored arrays, not complete evidence that the
original remote executions can be reconstructed bit for bit.

## 4. Checkpoint comparison — read the operating point, not the argmax

On full-season classification the two checkpoints score similarly, which the raw macro-F1
(0.926 → 0.919) hides. The honest, threshold-independent comparison is NonPSPL completeness at
**matched purity**:

| purity | baseline (no cascade) | BinML 1.0 (cascade) |
|---|---|---|
| 0.90 | 0.879 | **0.883** |
| 0.92 | 0.863 | **0.868** |
| 0.95 | 0.832 | 0.831 |
| 0.97 | 0.797 | 0.793 |
| 0.99 | 0.722 | 0.711 |

The curves **cross** (AP tied at 0.950 vs 0.952). The cascade model favours the high-completeness
regime, the baseline the high-purity regime — neither dominates on this comparison.

This table compares two *checkpoints from the shipped training lineage*, not a controlled on/off
experiment. The controlled ablation
(`validation/ablations_result.json`, identical data and recipe, `--truncate-aug` 0.5 vs 0.0) found
that turning truncation augmentation off **also** changes full-season macro-F1, so "adds the
cascade at no full-season cost" is not established. Read this table as a description of the
lineage, not as a causal claim.

## 5. Stress testing — same-prior, targeted and out-of-range arms

The full suite contains 14.9 million events: a 4.5-million-event same-prior subset, 8.7 million events in 12
targeted regimes (mostly enriched pools also used in training, drawn afresh) and 1.7 million in 17 out-of-range
sweeps. It was scored in July 2026 with the **stage-5 checkpoint**, the released model's predecessor
(`paper/results/stress_report.json`; its macro-F1 0.927 is stage 5's). For the released model,
`validation/stress_rescore_local.py` regenerates the first shards of each quoted regime with the suite's seeds,
class mix and (for the sweeps) its generator's peak-time draw, and scores the same events with both checkpoints.
The regenerated events are a new realisation, not the suite's own: the suite's cloud fleet ran an unpinned
software environment, and the regenerated natural tier labels 2.3% fewer detectable anomalies. Stage 5 gives the
suite's macro-F1 on the subset (0.927 vs 0.927) and differs from its other quoted numbers by 0.007 at the median
(at most 0.060, for the wide-separation and faint-source recalls); checkpoint comparisons are made within the
subset. The released model reproduces its held-out macro-F1 on the same-prior part (0.919 vs 0.919).

The targeted arms expose failures at faint magnitudes, wide separations, long periods and sub-day timescales:
sub-day single lenses (tE 0.2-1 d) are classified PSPL only 0.29 of the time and called anomalies 69% of the time
(35% above the frozen threshold); the suite's per-label PSPL recall (0.52) mixed in binaries of natural timescale
demoted to PSPL. Among faint microlensing events without a detectable anomaly the false-anomaly rate rises from
3.3% to 26%; the faint sweep's anomaly precision (0.026) is set mostly by its class mix. The low-q and faint-source
regimes were training pools from stage 4, and stage 6 added pools overlapping the wide-separation and sub-day
sweeps (s 3-8, tE 0.3-10 d), so for the released model these measure how far targeted coverage carries. They test
sensitivity to chosen stressors; they do not define their prevalence in the Roman population.

## 6. Baselines are sanity checks

The classical and learned comparators are useful reference points, not matched contests. The
neural model was trained on up to 1.9 million events per stage and receives the supplied true baseline magnitude;
the gradient-boosted and logistic baselines use a much smaller event set and eight summary
features. Every method, the fitted-PSPL residual included, sees all 6,912 F146 epochs (the residual's AP is 0.545
after the 2026-09-09 full-cadence rescore; an earlier version thinned it to 800 epochs). These differences
in budget and oracle inputs prevent attributing score gaps to architecture alone. Treat all such
comparisons as sanity checks.

## 7. Simulation scope

The simulator uses broad analytic training supports in the style of synthetic microlensing work
such as Zhang et al., not a measured Roman population model. In particular, the `tE` prior is an
authored truncated lognormal anchored to a literature mean. Binary-lens parameters provide broad
coverage, and variable/eruptive light curves use analytic or phenomenological waveform families;
they are not sampled OGLE templates. Conclusions are therefore conditional on these supports,
the photometric model, and the unvalidated 0.02-mag detectability floor.

The cadence and photometry are also legacy assumptions rather than the current survey definition.
The released model uses one 72-day season, 15-min F146 sampling, 46.8-s exposures, and
non-staggered colour grids. Current planning uses approximately 12-min F146, 66-s exposures,
staggered colour visits, and multiple seasons. The released F087/F213 zeropoints, F087 saturation,
and colour-band background ratios have known discrepancies from the current calibration. Measured on our simulator (`validation/referee_round.json`, 15,016 events of two held-out-pool shards): with the audited colour photometry the shipped model's anomaly ranking is unchanged (AP 0.958 vs 0.957), but the variable classes lose precision (periodic 0.949 to 0.908, eruptive 0.806 to 0.776; F1 0.970 to 0.949 and 0.891 to 0.871) because more flat sources are called periodic and more single lenses eruptive; recall is unchanged. Short fine-tunes on each calibration (one seed each) keep AP; the audited-calibration fine-tune falls to periodic-variable F1 0.750 on the old photometry (0.872 at its last epoch), against 0.961 (0.972) for the old-calibration fine-tune; on the audited photometry the two differ at the kept epoch (0.957 vs 0.917) but not at the last (0.967 vs 0.967). The released model has not been retrained with corrected values.

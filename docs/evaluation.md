# Evaluation methodology

This document describes how BinML is evaluated and — more importantly — how its numbers
should be *read*. Getting a headline accuracy is easy; interpreting a 6-class,
detectability-conditioned classifier honestly is the hard part, and most of this page is about
that.

Everything below is on the **held-out final-test split (360,472 events)** of a 450,589-event evaluation pool; the other 90,117 rows are reserved for fixing the operating threshold and are never used for reported metrics or figures (shard indices
disjoint from training — see [`leakage_audit.md`](leakage_audit.md)), scored by
[`evaluate.py`](../pipeline/evaluate.py).

---

## 1. Four principles baked into the metrics

1. **The headline is completeness at a FIXED purity — not accuracy, not F1.** Accuracy is
   meaningless (Flat+PSPL are ~61% of events); bare recall is degenerate (predict NonPSPL for
   everything → recall 1.0). Completeness at a stated purity is what a follow-up-triggering
   pipeline is actually specified against. The threshold is fixed on a validation split and
   applied **frozen** to test.

2. **`keep_prob` reweighting is asymmetric.** Every NonPSPL row has `keep_prob = 1` by
   construction; the PSPL/Flat "byproduct" rows that supply the false positives were subsampled
   6.67×. So **recall must NOT be reweighted, but precision/purity MUST** — otherwise purity is
   systematically optimistic by an amount that grows as the model improves. Both a `_sample` and
   a `_population` number are reported.

3. **Detectability conditioning.** A binary whose anomaly leaves no detectable signature is,
   observationally, a PSPL — no classifier or human can separate them from photometry. Recall is
   therefore reported binned by anomaly Δχ² and across the (log s, log q) plane, never as a
   single population number that conflates model skill with physical degeneracy.

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

Macro-F1 0.919. **Read NonPSPL carefully:** its recall is 0.95 — the model *finds* the anomalies.
The 0.72 precision (and hence the 0.82 F1) is a **labelling artefact**: 71% of its "false
positives" are genuine binaries whose caustic anomaly is below the detectability floor and were
therefore labelled PSPL by the detectability-conditioned labelling (§1). Counting those as
correct would give 0.99, but we do NOT report that as a second precision: under the observational task these are false positives. It is a diagnostic that the model responds to sub-threshold anomaly signal. This is exactly why the headline is
completeness@purity, not F1 — the argmax F1 penalises the model for a labelling choice that makes
the *reported* numbers honest.

- **Completeness @ fixed purity: 0.879.** Average precision (population): 0.9515.
- **Confusion is healthy.** The science-critical error — NonPSPL→PSPL, a *missed planet* — is
  0.048 (down from 0.055 without the cascade). The LongPeriodVar→PSPL false-microlensing impostor is
  0.010. No dangerous cross-confusions elsewhere; the variable classes are near-diagonal.

## 3. The real-time cascade — the headline capability

A Roman follow-up pipeline runs on *partial* seasons. It must not flag a false binary before
the caustic has been seen. BinML is trained (via detectability-conditioned truncation
labelling and a per-binary anomaly-onset day `t_anom`) so class probabilities follow the
cascade of what is observable: **Flat → PSPL → NonPSPL**.

Measured on truncated light curves, fraction flagged NonPSPL *before the anomaly is observable*:

| | baseline (no cascade) | BinML 1.0 (cascade) |
|---|---|---|
| premature NonPSPL flag (day 11) | 42% | **9%** |
| pre-onset P(NonPSPL) | 0.411 | **0.033** |

A **factor-4.7 reduction (79% fewer)** in premature binary flagging (the ~12x factor applies to the mean pre-onset probability, 0.411 -> 0.033) — and it's surgical: the other five classes'
temporal behaviour is unchanged (Flat stays Flat, Periodic commits in one cycle, Eruptive waits
for the outburst).

## 4. Baseline vs cascade model — read the operating point, not the argmax

On full-season classification the two models are **equivalent**, which the raw macro-F1
(0.926 → 0.919) hides. The honest, threshold-independent comparison is NonPSPL completeness at
**matched purity**:

| purity | baseline (no cascade) | BinML 1.0 (cascade) |
|---|---|---|
| 0.90 | 0.879 | **0.883** |
| 0.92 | 0.863 | **0.868** |
| 0.95 | 0.832 | 0.831 |
| 0.97 | 0.797 | 0.793 |
| 0.99 | 0.722 | 0.711 |

The curves **cross** (AP tied at 0.950 vs 0.952). The cascade model favours the high-completeness regime,
the baseline the high-purity regime — neither dominates. The macro-F1 dip is purely where the argmax
cut lands, **not** lost capability. The model ships because it adds the cascade at no full-season
cost.

## 5. Generalisation — the ~19M-event stress test

Fresh parameter draws the model never saw (disjoint RNG seed bases) reproduce the held-out
numbers (natural-population macro-F1 0.927), confirming generalisation rather than memorisation.
Out-of-range sweeps expose the real limits — all rare corners near fundamental physics limits:
faint m>25 (noise-dominated), wide caustics s>5 (rarely crossed), sub-day tE (few epochs on the
spike). These are documented, not hidden.

## 6. Baseline

Every NonPSPL number is compared against the classical Δχ² anomaly detector
([`baseline.py`](../pipeline/baseline.py)) applied to the same noisy data. The
network's margin (AP 0.95 vs the baseline's ~0.34) is margin from reading morphology and colour,
not from privileged access to the labelling rule.

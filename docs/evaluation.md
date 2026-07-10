# Evaluation methodology

This document describes how BinML is evaluated and, more importantly, how its
numbers should be *read*. BinML is a three-class classifier over Roman-cadence
microlensing light curves — **Flat** (no event), **PSPL** (single-lens
microlensing, the "is this microlensing?" detection proxy), and **Binary** (a
planetary or binary lens, the distinct anomalous class). Getting the headline
accuracy is easy; interpreting the Binary channel honestly is the hard part, and
most of this page is about that.

The guiding principle: **a binary lens whose anomaly leaves no detectable
signature in the light curve is, observationally, a PSPL.** No classifier — and no
human modeller — can separate the two from photometry alone. Reporting a raw
population recall for the Binary class therefore penalises the model for failing
to do something physically impossible. Every Binary number below is reported
*conditioned on detectability* wherever the honest metric differs from the raw
one.

All headline results are from a single **full 1,000,000-event held-out
evaluation** (streaming, never loaded into memory at once). Normalization
statistics are fit on the training split only; the evaluation set is disjoint (see
[`docs/leakage_audit.md`](leakage_audit.md)).

---

## 1. Headline results

Two models ship. The **base** model is trained for balanced three-class behaviour;
the **fine-tuned** model (round 1) is warm-started from the base checkpoint on a
bump-rich, low-mass-ratio-targeted set to recover the hard planetary regime. The
fine-tuned model is the package default.

| Metric | Base | Fine-tuned (round 1) |
|---|---|---|
| Overall accuracy | 64.3% | 67.2% |
| Recall — Flat | 100% | 100% |
| Recall — PSPL | 80.0% | 70.9% |
| Recall — Binary (raw population) | 52.2% | 62.3% |
| Binary precision | 81.7% | — |
| Planetary binary recall (q ∈ 1e-4..1e-3) | 41.2% | 57.4% |

Fine-tuning is a deliberate trade: it moves probability mass toward the Binary
class, lifting Binary recall (and especially the planetary sub-regime, +16 points)
at the cost of some PSPL recall. This is the right trade for a planet-finding tool,
where the scientifically valuable events are the anomalous ones.

A **second, more aggressive fine-tune round collapsed** — it drove PSPL recall to
~18% by calling nearly everything Binary — and was discarded. Round 1 is the
shipped fine-tuned model. This failure is the reason the fine-tuning schedule is
conservative (low LR, warm start, targeted data rather than more base training).

The raw Binary recall in the table above is the number **not** to quote in
isolation. Section 2 explains why and gives the honest replacement.

---

## 2. Detectability-conditioned binary recall (the honest metric)

### 2.1 Why raw population recall is misleading

A binary lens produces a detectable anomaly only when the source trajectory
actually probes the caustic structure at the sampled epochs. For a large fraction
of simulated binaries — small mass ratio, unfavourable geometry, or a caustic
approach that falls between observations — the light curve is **identical, within
the noise, to a single-lens (PSPL) curve.** Such an event carries no information
that could distinguish it from a PSPL, so a "misclassification" as PSPL is the
physically correct read of the photometry, not a model error.

We quantify anomaly strength with `anomaly_dchi2`: the χ² of the true binary light
curve against a matched single-lens fit. It is the physical **detectability** of
the anomaly — high `anomaly_dchi2` means the binary signature is strong and
sampled; near-zero means the binary is observationally a PSPL.

### 2.2 Recall rises monotonically with detectability

Binned by `anomaly_dchi2`, the fine-tuned model's Binary recall increases
**monotonically** with anomaly strength, from **~11%** for events with no
detectable signal to **~85%** for events with a strong, well-sampled anomaly. The
model recovers the binaries it is *possible* to recover, and the low recall in the
raw population number is dominated by the tail of intrinsically undetectable
events.

```
Binary recall vs. anomaly detectability (fine-tuned model)

  85% |                                        ####  strong anomaly
      |                                   ####
      |                              ####
      |                        ####
      |                  ####
      |            ####
  11% |  ####                                        no detectable signal
      +----------------------------------------------
       low  anomaly_dchi2  (physical detectability) -> high
```

### 2.3 The numbers to report

At a detection threshold of **Δχ² ≥ 300**:

- **~32% of binaries are physically indistinguishable from PSPL** — their anomaly
  falls below the detection threshold. No method can classify these as Binary from
  the light curve alone.
- **Detectable-only Binary recall is ~78%**, versus the ~62% raw population number.
  This is the metric that reflects the model's actual capability.
- **~60% of the "missed" binaries at this threshold are physically PSPL-like** —
  i.e. below detectability. The majority of apparent Binary errors are not model
  failures; they are the observationally correct answer.

**Report Binary performance conditioned on detectability — the detectable-only
recall, the indistinguishable fraction, and the recall-vs-Δχ² curve — never the raw
population recall by itself.** The raw number understates the model by conflating
its errors with a hard physical limit of photometric microlensing.

`binml.evaluate_dataset` returns both a `ClassificationReport` (raw population
metrics) and a `DetectabilityReport` (the conditioned metrics above) so this
distinction is surfaced automatically — see Section 5.

---

## 3. Temporal-bias test: effect size, not the p-value

A natural worry for a causal, time-ordered classifier is that it might succeed only
on events peaking at convenient positions within the 72-day Roman season — e.g.
events near the middle where the wings are well sampled — and systematically fail on
events peaking near the edges. If real, that would bias any downstream population
inference on peak time `t0`.

We test this with a **two-sample Kolmogorov–Smirnov (KS) test** comparing the `t0`
distribution of correctly classified events against that of incorrectly classified
events. If classification were `t0`-dependent, the two distributions would differ.

The result is a KS statistic of **D ~ 0.01**, i.e. the two `t0` distributions are
essentially identical — a negligible effect size. There is **no meaningful temporal
bias**: the model's success does not depend on where in the season an event peaks.

**Read the effect size, not the p-value.** With ~1e6 events the test has enormous
statistical power, so the p-value is driven to ~0 by any deviation, however tiny,
and is therefore meaningless as a decision criterion — at this sample size a p-value
would flag a difference of no scientific consequence. D ~ 0.01 is the number that
matters, and it says the effect is negligible. This is a general lesson for
evaluating on million-event sets: **significance tests measure sample size as much
as effect, so quote and threshold on effect size.**

---

## 4. Cadence dependence: Roman vs. LSST

BinML is built for **Roman-class cadence** — 6912 points at 15-minute sampling over
a 72-day season. Its ability to characterise anomalies depends critically on this
density, because the diagnostic feature of a binary — the short caustic-crossing
anomaly — is a brief event that only exists in the data if the cadence samples it.
Evaluated zero-shot on sparser, LSST-like cadences, the model degrades in a way that
cleanly separates two distinct tasks.

**Detection** (is this microlensing at all — Flat vs. PSPL/Binary) is comparatively
robust to sparse sampling, because the overall magnification bump is a slow,
days-to-weeks feature:

- At **~1 visit/day**, microlensing detection holds at **~89%**.
- By **~3-day gaps**, detection **collapses to ~4%** — the bump itself is no longer
  reliably sampled.

**Characterisation** (is this a *binary* — the Binary channel) does **not** survive
any LSST cadence:

- Binary characterisation is **~0% at any LSST cadence.** The short caustic anomaly
  is never sampled at LSST visit rates, so the distinguishing signal is simply
  absent from the data.

The practical conclusion: **Roman is the planet-finder; LSST is, at best, a
detector.** BinML should be applied to LSST-like data only as a microlensing
*detector* (and only at the densest visit rates), never as a binary/planet
characteriser. Quoting a Binary metric on LSST cadence would be meaningless — the
information is not in the light curve.

---

## 5. Running an evaluation

### Python API

```python
import binml

clf = binml.Classifier()                       # fine-tuned default
report, det_report = binml.evaluate_dataset(clf, "test.h5")
```

`evaluate_dataset` streams a compact HDF5 test shard and returns a pair:

- **`ClassificationReport`** — the raw population metrics (overall accuracy,
  per-class recall, precision, confusion structure). These are the Section 1
  numbers.
- **`DetectabilityReport`** — the honest, `anomaly_dchi2`-conditioned metrics from
  Section 2: detectable-only Binary recall, the indistinguishable fraction at the
  detection threshold, and the recall-vs-Δχ² curve.

Always read the two together. The `ClassificationReport` alone will understate the
Binary channel for the reasons in Section 2; the `DetectabilityReport` is what makes
the Binary performance interpretable.

For single light curves, `Classifier.predict` exposes the detection/characterisation
split directly:

```python
r = clf.predict(time, mag, mag_err)
r.probabilities      # [P(Flat), P(PSPL), P(Binary)]
r.label              # argmax class
r.is_microlensing    # P(PSPL) + P(Binary)  -> the detection proxy
r.is_anomalous       # P(Binary)            -> the characterisation proxy
```

### Command line

```bash
binml evaluate test.h5
```

This runs the same streaming evaluation and prints both the classification and
detectability reports.

---

## 6. Real-data validation

Beyond the simulated held-out set, the models are checked against **real OGLE-IV EWS
light curves**. The classifier correctly flags known caustic-crossing binaries
(e.g. OGLE-2014-BLG-0289, OGLE-2013-BLG-0578) and clean single-lens events, and the
fine-tuned model recovers real binaries the base model missed (OGLE-2011-BLG-0417,
OGLE-2015-BLG-0060) — consistent with the fine-tune's targeted gain in the hard
low-mass-ratio regime. OGLE cadence is sparser than Roman, so this validation
exercises the detection channel and the strongest, best-sampled anomalies; it is a
sanity check on real photometry, not a substitute for the Roman-cadence evaluation.

---

## Summary — how to quote BinML

- **Do** report overall accuracy and Flat/PSPL recall directly (Section 1).
- **Do** report the Binary channel **conditioned on detectability**: detectable-only
  recall (~78% at Δχ² ≥ 300), the ~32% indistinguishable fraction, and the
  recall-vs-Δχ² curve.
- **Don't** quote the raw Binary population recall (62.3%) in isolation — it
  conflates model error with a physical detectability limit.
- **Do** read effect size (KS D ~ 0.01), not the p-value, at million-event scale.
- **Do** distinguish detection from characterisation when the cadence is not
  Roman-class: BinML detects on dense LSST-like data at best, and characterises
  binaries only at Roman cadence.

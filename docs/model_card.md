# BinML — Model Card

## Overview

- **Model:** BinML 1.0 — a 6-class classifier for Nancy Grace Roman Space Telescope Galactic
  Bulge Time-Domain Survey light curves.
- **Architecture:** convolutional stem (with non-learned min/max carry lanes) → 4-layer, 4-head
  transformer encoder (`d_model=96`), masked attention-pool → flat 6-way head. **505,479 params.**
- **Inputs:** three bands under the released legacy schedule (F146 15-min / 6912 epochs, F087 &
  F213 6-h / 288 epochs) over one 72-day season, binned to 156 tokens × 5 channels (mean, min,
  max, observed-fraction, mask). This is not the current multi-season GBTDS schedule.
- **Outputs:** probabilities over {Flat, PSPL, NonPSPL, PeriodicVar, LongPeriodVar, Eruptive}.

## Intended use
Research triage and vetting of partial Roman-like seasons: separate simulated microlensing from
variable-star contaminants and rank anomalous binary-lens candidates for modelling or human
review. NonPSPL includes both stellar and planetary-mass-ratio binaries; it is not a planet label.
The model has not been validated as an autonomous real-time discovery or follow-up trigger.

## Training data
Millions of simulated Roman-like light curves. The parameter distributions are broad analytic
training supports, in the style of Zhang et al., rather than a fitted Roman population model. The
`tE` distribution is an authored truncated lognormal anchored to a literature mean; it is not the
measured OGLE distribution. Binary lenses use VBBinaryLensing and broad mass-ratio/separation
support. Variable and eruptive classes use analytic or phenomenological waveform families
motivated by their named classes, not sampled OGLE templates. **Labels are
detectability-conditioned:** events below the adopted synthetic floor are reassigned Flat or PSPL.
The 0.02-mag floor has not been validated on real Roman data.

The released simulation predates the current survey definition. Current GBTDS planning uses
approximately 12-min F146 sampling, 66-s exposures, staggered colour visits, and multiple seasons.
An audit against the current calibration also found F087/F213 zeropoints optimistic by about
0.10/0.14 mag, an incorrectly ordered F087 saturation assumption, and colour-band background
ratios inconsistent with the published thermal backgrounds. The resulting effect on contaminant
rejection is unquantified; the checkpoint has not been retrained with corrected photometry.

## Evaluation
The evaluation pool contains 450,589 events: 90,117 fix the operating threshold and the
remaining 360,472 form the final test set; shard indices are disjoint from training.
**Completeness@purity is 0.879 and AP is 0.952**. Per-class F1 is [Flat 0.97, PSPL 0.96,
NonPSPL 0.82, PeriodicVar 0.97, LongPeriodVar 0.91, Eruptive 0.88], with macro-F1 0.919.
NonPSPL-versus-rest calibration has weighted ECE 0.0533 and weighted Brier score 0.0209.

On a frozen 1,000-event F146 scan at 0.5-day spacing, 89.0% of eligible binaries cross the
threshold within the season. Among nonpremature detections, the median lag is +5.0 days relative
to a truth-informed, noise-free onset proxy; 1.6% of eligible events cross before that proxy
(95% CI 1.0–2.6%). These results support partial-season triage, not response during a short
planetary perturbation. The matched 400-event cascade comparison is exploratory risk–coverage:
thresholds and outcomes use the same events, so its conditional McNemar values are not
confirmatory population-level inference.

The prefix scan contains only already-eligible binaries and uses the complete-season operating
threshold. It does not measure repeated-score false alerts, streaming purity, or alert burden on
Flat, PSPL, demoted-binary, and variable-star prefixes.

The 14.9-million-event stress suite contains a 4.5-million-event same-prior subset, on which
macro-F1 is 0.927, and 10.4 million targeted out-of-distribution cases that expose failures. Full
detail: [evaluation.md](evaluation.md).

## Limitations / known failure modes

Documented in targeted out-of-distribution tests (their population frequency is not established):
- **Faint sources (m > 25):** noise-dominated; risk of noise excursions read as anomalies
  (NonPSPL precision collapses). The single most operationally relevant weak spot.
- **Wide caustics (s > 5):** the caustic is rarely crossed, so many are unrecoverable.
- **Sub-day tE:** few epochs sample the peak.
- **Cadence:** trained on a legacy one-season Roman-like schedule; not validated on the current
  multi-season survey design or sparse ground-survey sampling.
- **Oracle baseline:** evaluation supplies the true simulated baseline magnitude; performance with
  an estimated baseline is not yet established.
- **Baseline comparisons:** comparator models use unmatched training/evaluation budgets and input
  information, so their scores are sanity checks rather than architecture-isolating experiments.
- **Not yet validated on real Roman data** (survey not yet flying). Real-data testing is necessary,
  but corrected photometry and current-schedule retraining are also required before operational use.

## Ethical / scientific scope

The model reports probabilities; for the NonPSPL-versus-rest decision, weighted ECE is 0.0533 and
weighted Brier score is 0.0209 on the synthetic final test set. A flagged NonPSPL is a candidate
for human review or modelling, not a discovery claim. Read performance as completeness at a
stated purity, not bare accuracy.

## Provenance

Model line: base training → a curriculum of targeted warm-start fine-tunes on hard regimes →
the partial-season cascade fine-tune (the shipped model). Weights are bundled as
`binml/weights/binml.pt`.

The current validation artifacts do not provide complete end-to-end run provenance. The main
cascade trace records a dirty tree but no source hash/diff; the matched trace lacks code and
checkpoint hashes. The labelling-ablation artifact's source hash was repaired after the run, and
the newer content-addressed provenance mechanism has not yet generated the published artifact.

# BinML — Model Card

## Overview
- **Model:** BinML 1.0 — a 6-class classifier for Nancy Grace Roman Space Telescope Galactic
  Bulge Time-Domain Survey light curves.
- **Architecture:** convolutional stem (with non-learned min/max carry lanes) → 4-layer, 4-head
  transformer encoder (`d_model=96`), masked attention-pool → flat 6-way head. **505,479 params.**
- **Inputs:** three bands (F146 15-min / 6912 epochs, F087 & F213 6-h / 288 epochs) over a
  72-day season, binned to 156 tokens × 5 channels (mean, min, max, observed-fraction, mask).
- **Outputs:** probabilities over {Flat, PSPL, NonPSPL, PeriodicVar, LongPeriodVar, Eruptive}.

## Intended use
Triage / vetting of Roman GBTDS light curves: separate microlensing from variable-star
contaminants and flag the anomalous (binary/planetary) events for follow-up. It is a
follow-up-triggering aid, **not** a substitute for full light-curve modelling of a candidate.
Designed to run on *partial* seasons and only flag a class once its evidence is observable.

## Training data
Simulated Roman light curves (millions of events). Class priors from the literature: tE from
Mróz et al. 2019 (OGLE-IV), planet mass-ratio break q≈1.7×10⁻⁴ from Suzuki et al. 2016, survey
cadence/photometry from Penny et al. 2019. Binary lenses via VBBinaryLensing (Bozza 2010, 2018);
variable/eruptive classes from OGLE catalog morphologies (Mira/SRV/OSARG, RR Lyrae/EB/δ Scuti,
DN/WZSge/Be). **Labels are detectability-conditioned** — an event is labelled by what is
observable, so undetectable events → Flat and undetectable anomalies → PSPL.

## Evaluation
Independent 450,589-event held-out test set (shard indices disjoint from training). Metrics:
completeness at fixed purity (headline), keep_prob-aware precision/purity, detectability-binned
recall. **Completeness@purity 0.879, AP 0.952**; per-class F1 [Flat 0.97, PSPL 0.96, NonPSPL 0.82,
Per 0.97, LPV 0.91, Erup 0.88] (macro 0.919; NonPSPL F1 is precision-limited by the
detectability-floor demotion — recall is 0.95, physical precision 0.99, see evaluation.md).
Real-time cascade: premature NonPSPL flagging 42%→9%. Generalises
on a 12.9M-event unseen-parameter stress test (macro-F1 0.927). Full detail: [evaluation.md](evaluation.md).

## Limitations / known failure modes
Documented at the out-of-range extremes (rare in the real population, near fundamental limits):
- **Faint sources (m > 25):** noise-dominated; risk of noise excursions read as anomalies
  (NonPSPL precision collapses). The single most operationally relevant weak spot.
- **Wide caustics (s > 5):** the caustic is rarely crossed, so many are unrecoverable.
- **Sub-day tE:** few epochs sample the peak.
- **Cadence:** trained on Roman cadence; not validated on sparse ground-survey sampling.
- **Not yet validated on real Roman data** (survey not yet flying) — the outstanding next step.

## Ethical / scientific scope
The model reports calibrated probabilities (ECE ≈ 0.004). It does not make discovery claims; a
flagged NonPSPL is a candidate for human/modelling follow-up. Numbers should be read as
completeness-at-purity, never bare accuracy (Flat+PSPL are ~61% of events).

## Provenance
Model line: base training → a curriculum of targeted warm-start fine-tunes on hard regimes →
the real-time cascade fine-tune (the shipped model). Weights bundled as `binml/weights/binml.pt`.

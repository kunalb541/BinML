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
0.10/0.14 mag, an F087 saturation limit carried from a longer exposure (too faint: at equal
exposure F087 saturates ~1.2 mag brighter than F146), and colour-band background
ratios inconsistent with the published thermal backgrounds. Measured on our simulator (`validation/referee_round.json`, 15,016 events of two held-out-pool shards): with the audited colour photometry the shipped model's anomaly ranking is unchanged (AP 0.958 vs 0.957), but the variable classes lose precision (periodic 0.949 to 0.908, eruptive 0.806 to 0.776; F1 0.970 to 0.949 and 0.891 to 0.871) because more flat sources are called periodic and more single lenses eruptive; recall is unchanged. A short fine-tune on the audited calibration (one seed) keeps AP and restores periodic-variable F1 on the audited photometry (0.957) but collapses on the old one (0.750). The released model has not been retrained with corrected values.

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

The 1,000-event prefix scan contains only already-eligible binaries. An every-class scan of two held-out-pool shards (15,016 events; `validation/referee_round.json`, `mixed_class_stream{,_f146}`; shipped model, frozen complete-season threshold) measures the burden on our simulator: with all three bands revealed no flat or variable-star event alerts, and the scan raises 0.77 alerts per 1,000 events per day, 89% of them detectable anomalies at the simulated 5.5% prevalence (58% at 1% and 12% at 0.1% by prior shift); with F146 alone it raises 0.98 per day, 67% of them detectable anomalies (26% at 1%), and 4 eruptive variables alert. With three bands nearly all false alerts come from binaries whose anomaly falls below the detectability policy (29 of 32; 3 of the 1,759 generated single lenses alert); with F146 alone 51 generated single lenses alert as well. The threshold is still the complete-season one; a streaming threshold calibrated on disjoint prefixes is untested.
A repeat on RMDC26 with single lenses included
(`validation/gulls/cascade_gulls.json`; recommended gap-aware checkpoint, F146 only) measures the
single-lens part: 6.2% of single lenses raise an alert at some point in the season at the frozen threshold
(3.2% at 0.956), so if 1% of events were planetary about one alert in twenty would come from a claimable
anomaly (about one in thirteen at 0.956). RMDC26 contains no variable stars or flat sources, so a real
stream would be less pure.

The 14.9-million-event stress suite (a 4.5-million-event same-prior subset and 10.4 million targeted cases) was
scored with the stage-5 checkpoint, the released model's predecessor; its 0.927 macro-F1 is stage 5's. On the
first shards of each quoted regime, regenerated with the same seeds (255,527 events), the released model reaches
macro-F1 0.919 on the same-prior part, equal to its held-out value
(`validation/stress_rescore_local.json`). Full detail: [evaluation.md](evaluation.md).

## Limitations / known failure modes

Documented in targeted out-of-distribution tests (their population frequency is not established):
- **Continuous F146 required; a gapped schedule breaks the shipped checkpoint.** The training
  grid has no mid-season gaps. The RGES-PIT RMDC26 release (a GULLS simulation of the survey) pauses
  F146 for 43-44 h per season in six or seven pauses whose phases differ between seasons, and the
  shipped checkpoint reads such gaps as evidence against a single lens (single-lens and Flat recall
  collapse; the eruptive and long-period variables and NonPSPL also degrade;
  `validation/gulls/schedule_finetune.json`). One gap of 0.5 h costs nothing; one of 1-2 h at mid-season (day 43) costs
  about 7 points of single-lens recall (`gap_sensitivity.json`, n = 100 per class; the cost depends on the gap's position). **Input contract:**
  continuous F146 within one 72-day season. For a gapped schedule use the recommended gap-aware
  checkpoint `binml/weights/binml-gapaware.pt` (`Classifier(weights="gapaware")`; identical to
  `validation/gulls/weights/ft_fspl5s_seasons_g08.pt`: finite-source single lenses and the measured RMDC26
  pauses in training) at threshold 0.956 (`binml.GAPAWARE_THRESHOLD`) (chosen at 90% purity on our own simulations with
  the measured RMDC26 pauses; 68% slice range 0.947-0.965); the shipped threshold does not apply to it. On
  RMDC26 it flags 2.4% of single lenses at that threshold (3.5% weighted by event rate), with planetary
  recall 0.29 / 0.33; two further training seeds of the recipe, calibrated the same way, give 2.6% / 3.0%
  and 1S2L recall 0.26 / 0.28 (the released run is the best of three on RMDC26). See `paper/REVISION.md` §1½
  and `docs/VERIFICATION_2026-09-1{1,2}.md`.
- **Partial bin occupancy.** Survey schedules in which colour visits displace F146 exposures leave
  bins partly filled (RMDC26: one of eight epochs in about 35% of bins), which training never
  contains; on RMDC26 the recommended gap-aware checkpoint reads it as mild evidence of an anomaly
  (setting it to full lowers its single-lens false alarms from 6.8% to 5.9% and planetary recall from
  0.42 to 0.33 on 6,024 inputs; `validation/gulls/occupancy_sensitivity.json`).
- **Short single lenses.** RMDC26's scored single lenses are much shorter than our training prior
  (38% have t_E < 3 d, 16% weighted by event rate, against 1.8% of our prior's mass), and the gap-aware
  checkpoints' false-alarm rate rises with timescale, so per-event and rate-weighted rates differ
  (`validation/gulls/transfer_tradeoff_all.json`, `weighted` blocks).
- **Faint sources (m = 25-27.5):** noise-dominated; NonPSPL precision 0.026, mostly because detectable anomalies
  are rare there (0.19% of events by weight); at the natural prevalence the same rates would give 0.44, and the
  false-anomaly rate rises from 2.1% to 5.4% of non-anomalous events.
- **Sub-day single lenses (tE 0.2-1 d):** only 0.27 of the detectable ones are classified PSPL; 71% are called
  anomalies (37% above the frozen threshold). The suite's per-label 0.52 mixed in demoted binaries. Stage 6 had
  trained on tE down to 0.3 d.
- **Wide caustics (s = 5-12):** anomaly recall 0.42 on the 60 detectable ones in the regenerated subset (stage 5:
  0.22 on the suite's 438).
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

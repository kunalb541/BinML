# Changelog

## Unreleased — gap sensitivity and first cross-simulator validation (2026-08-23)
- **Shipped checkpoint fails on Roman's planned F146 schedule.** The training grid is
  continuous; the planned GBTDS schedule pauses F146 ~6 h seven times per season. Inserting
  those gaps into in-distribution events drops PSPL recall 0.93 → 0.11 and Flat 1.00 → 0.08
  (`validation/gulls/gap_sensitivity.py`). Reproduced with no external data.
- **`pipeline/train.py --gap-aug`** blanks contiguous Roman-like runs in every band and
  relabels; `tests/test_gap_augmentation.py` pins the contract. A warm-start fine-tune
  (`validation/modal_gap_finetune.py`, checkpoint `validation/gulls/weights/ft_g08e12.pt`)
  restores held-out macro-F1 under the gapped schedule from 0.384 to 0.879.
- **First valid GULLS (RMDC26) transfer numbers.** `validation/gulls_transfer.py` now reads the
  172 GB obs table locally by contiguous `event_id` block (0.09 s/event), pins HF revision
  `a338d5ba` (RGES-PIT re-uploaded on 2026-08-18), measures the baseline empirically (the
  catalogue value is 0.471 mag off in this release), and accepts `--weights`. On 1,286 events
  the fine-tuned model cuts single-lens false alarms at the frozen threshold from 52% to 7%.
- Shipped weights unchanged; submitted numbers unaffected. Write-up plan: `paper/REVISION.md`.

## Unreleased — audit round 3 (streaming analysis corrected)
- **Cascade numbers rebuilt from one stored scan.** `validation/cascade_trace.py` records
  P(NonPSPL) at every 0.5 d cut for the frozen 1,000-event sample, under F146-only and
  all-three-band revealing, plus the anomaly-detectability mask on the same grid;
  `cascade_reduce.py` derives every reported statistic from it. Alert policy and onset definition
  are now knobs in the reduction rather than reasons to re-scan a different sample.
- **Two premature-alert numbers withdrawn.** The previously reported 31.3% scored alerts against
  an onset quantised to a 7.2 d grid (median inflation 3.7 d), and the 1.3 d "median lag after
  onset" was taken over all alerts including the premature ones. Corrected: 1.6% premature
  (1.0-2.6%) under a first-detectable onset, 4.5% under a strict persistent-detectable onset, with
  a median lag of +5.0 d. The old figure is retained in the artifact for comparison.
- **Labelling ablation de-confounded.** `pipeline/train.py` gained `--weight-labels`, because
  `compute_weights` derives class weights from whichever labels it is handed: the original arms
  differed in objective as well as in target. `validation/modal_labelling_ablation.py` pins the
  weights, scores every arm against both ontologies, and reports a paired bootstrap interval.
- **Equal-detection-rate cascade comparison run.** The experiment the paper named as its own
  clearest outstanding weakness. `validation/modal_cascade_matched.py` stores full probability
  traces for both ablation arms on a shared sample; `cascade_matched_reduce.py` sweeps each arm's
  threshold to a common within-season detection rate and compares premature rates there with a
  paired McNemar calculation. The augmented arm is premature less often at every matched detection
  rate from 0.50 to 0.90, but the thresholds and outcomes use the same 400 events and the arms use
  one training seed each. The conditional values are therefore descriptive rather than valid
  confirmatory p-values; the sign also reverses at 0.95. A causal benefit remains unestablished.
- **Prevalence thresholds moved off the final test set.** `validation/prevalence_thresholds.py`
  selects each threshold on the reserved validation rows and reports on the frozen test rows.
- **Provenance.** Future Modal runs key caches and checkpoints by a hash of the simulator/model source
  plus the full config, with a completion marker so an interrupted run cannot be silently reused
  as a finished one, and write a manifest recording git commit, pinned dependency versions, and
  per-artifact hashes. The published labelling-ablation artifact predates a verified execution of
  that mechanism; its source hash was repaired after the run.
- **Build.** `make_macros.py` reads the cascade, prevalence and ablation artifacts directly and
  fail-closed; a new CI job builds the paper end to end from `git archive HEAD`. Packaging moved
  to an SPDX license expression and explicit package-data configuration.

## 1.0.0 — 6-class multi-band model
- **New model: `pipeline/`.** 6 classes (Flat, PSPL, NonPSPL, PeriodicVar,
  LongPeriodVar, Eruptive) from 3 bands (F146/F087/F213). Conv-stem + transformer,
  505,479 params, replacing the 3-class CNN-GRU.
- **Detectability-conditioned labelling** — events labelled by what is observable, not by
  generator intent.
- **Partial-season cascade** — truncation labelling uses a truth-informed, noise-free per-binary anomaly-onset
  day (`t_anom`). Alert timing is measured event-level on a frozen 1,000-binary sample; see
  `validation/cascade_trace.py` and `validation/cascade_reduce.py`. The legacy 42%->9% premature
  flagging figure was untracked, could not be reproduced, and is withdrawn.
- **Distributed pipeline** — AWS spot generation/binning, in-region inference (`run_bineval`,
  `eval_shard`), stratified fine-tune mix (`mix_finetune`), stress-test aggregation (`agg_stress`).
- Headline (a 90,117-event threshold-calibration split plus a disjoint 360,472-event final-test
  split): completeness@purity 0.879, AP 0.952. The 14.9M-event stress suite comprises a 4.5M
  same-prior reproduction and 10.4M targeted out-of-distribution diagnostics; it is not a single
  population-validation set.
- Docs: new `docs/pipeline.md`; `architecture.md`, `evaluation.md`, `data_format.md`,
  `training.md`, and `README.md` updated to v5.


## 0.2.0
- Added detectability-conditioned evaluation (`binml.evaluate`): 3-class
  `classification_report`, `detectability_curve` (binary recall vs Δχ², indistinguishable
  fraction, detectable-only recall), and `evaluate_dataset` for compact HDF5 test sets.
- Added `Classifier.predict_arrays` for batched, GPU-vectorized prediction from stored grids.
- New CLI subcommand `binml evaluate`; added `binml --version`.
- Docs: "report binary performance the honest way" (detectability caveat).

## 0.1.0
- Initial release: `Classifier` (Flat / PSPL / Binary), real-survey preprocessing,
  OGLE/MOA/generic loaders + `fetch_ogle_ews`, probability-evolution, CLI, bundled weights
  (fine-tuned default + base).

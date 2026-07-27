# Changelog

## Unreleased — v5 (6-class multi-band model)
- **New model: `pipeline/sim_v5/`.** 6 classes (Flat, PSPL, NonPSPL, PeriodicVar,
  LongPeriodVar, Eruptive) from 3 bands (F146/F087/F213). Conv-stem + transformer,
  505,479 params, replacing the 3-class CNN-GRU.
- **Detectability-conditioned labelling** — events labelled by what is observable, not by
  generator intent.
- **Real-time cascade** — truncation labelling by observability with a per-binary anomaly-onset
  day (`t_anom`); premature NonPSPL flagging cut from 42% -> 9%.
- **Distributed pipeline** — AWS spot generation/binning, in-region inference (`run_bineval`,
  `eval_shard`), stratified fine-tune mix (`mix_finetune`), stress-test aggregation (`agg_stress`).
- Headline (independent 450,589-event test set): completeness@purity 0.879, AP 0.952; validated
  against a 12.9M-event unseen-parameter stress test.
- Docs: new `docs/pipeline_v5.md`; `architecture.md`, `evaluation.md`, `data_format.md`,
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

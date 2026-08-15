<h1 align="center">BinML</h1>

<p align="center">
  <b>Multi-band, 6-class triage of progressively revealed microlensing and variable-star light curves.</b><br>
  A synthetic Roman-like benchmark for the <i>Nancy Grace Roman Space Telescope</i> Galactic
  Bulge Time-Domain Survey.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/pytorch-2.0%2B-orange.svg" alt="PyTorch">
  <a href="https://github.com/kunalb541/BinML/actions"><img src="https://github.com/kunalb541/BinML/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT"></a>
</p>

---

BinML classifies simulated Roman-like light curves into **six operational classes** from three
photometric bands (F146 at 15-min cadence, F087/F213 at 6-h). It is designed for triage as a
season is revealed. The present evidence is simulation-only and does not establish autonomous
real-time planet triggering.

The released checkpoint uses a **legacy Cycle-7-inspired schedule**, not the current GBTDS
definition: one 72-day season, 15-min F146 sampling, and separate 6-h colour visits. The current
design uses approximately 12-min F146 sampling, 66-s exposures, staggered F087/F213 visits, and a
multi-season programme. Results below describe the released legacy-schedule benchmark.

| class | meaning |
|---|---|
| **Flat** | no detectable event (baseline / noise) |
| **PSPL** | single-lens microlensing |
| **NonPSPL** | detectable non-PSPL structure from stellar- or planetary-mass-ratio binary lenses |
| **PeriodicVar** | short-period variables (RR Lyrae, eclipsing binaries, δ Scuti) |
| **LongPeriodVar** | Miras, semiregulars, OSARGs — the dangerous microlensing impostor |
| **Eruptive** | dwarf novae, Be outbursts |

Two design choices set it apart from a standard classifier:

1. **Detectability-conditioned labelling.** An event is labelled by what is *observable*, not by
   what we simulated. A microlensing event whose peak falls outside the season, or whose
   amplitude is buried by noise, is **Flat**. A binary whose caustic anomaly is below the noise
   floor is assigned **PSPL** under the adopted synthetic detectability rule. This rule prevents
   the classifier from being scored against injected structure that the simulation declares
   undetectable; the floor is a modelling choice, not a theorem about real data.

2. **A partial-season cascade objective.** During truncation training, a simulated binary is
   relabelled **Flat → PSPL → NonPSPL** as more of the season is exposed. The transition uses
   a truth-informed, noise-free anomaly-onset proxy. It therefore tests streaming behaviour in a
   controlled simulation; it is not an observable alert timestamp available to a live broker.

## The model

The network is a **convolutional stem feeding a small transformer encoder** — **505,479
parameters** — deliberately small, because the task is not data-starved (millions of simulated
events) but inference-heavy at survey scale.

- **Input: 3 bands → 156 tokens × 5 channels.** In the released legacy schedule, F146 is the workhorse (15-min cadence, 6912
  epochs / 72-day season); F087 and F213 are the colour bands (6-h, 288 epochs). Each band is
  binned into fixed token slots (F146→864, colour→96) and every bin carries `mean, min, max,
  observed-fraction, observed-mask`.
- **Conv stem with non-learned min/max carry lanes.** The stem downsamples each band to its
  token count (F146 864→108, colour 96→24, total **156 tokens**). A learned averaging filter
  would smear a ~2.9-mag caustic spike down to ~0.36 mag — the size of ordinary PSPL curvature,
  erasing the PSPL-vs-NonPSPL distinction. So two stem channels are reserved as **non-learned
  max/min pooling lanes** carried through every downsampling step: the caustic extremum survives
  *by construction*.
- **Transformer:** 4 pre-norm blocks, 4 heads, `d_model=96`, fused SDPA. Absent colour bands are
  masked out of attention (F087 drops out in ~38% of events, F213 in ~1%, under extinction).
- **Head:** masked attention-pooling → a flat **6-way** classifier. (A hierarchical head was
  tested and shipped worse.)

Details: [`docs/architecture.md`](docs/architecture.md).

## Results (held-out final-test set: 360,472 events)

Per-class F1 (population-weighted, selection-corrected):

| Flat | PSPL | NonPSPL | PeriodicVar | LongPeriodVar | Eruptive |
|---|---|---|---|---|---|
| 0.97 | 0.96 | 0.82 | 0.97 | 0.91 | 0.88 |

> The NonPSPL F1 (0.82) is precision-limited; its recall is 0.95. Of its false positives, 94.7%
> were generated as binaries but demoted to PSPL by the adopted detectability floor. They remain
> false positives for the stated observational task. This diagnostic may reflect sensitivity to
> sub-threshold structure or correlated simulation properties and is not a second precision
> estimate.

- **Completeness at fixed purity: 0.879** (the headline triage metric, rather than accuracy or
  F1). Average precision 0.952. NonPSPL-versus-rest calibration has weighted ECE **0.0533** and
  weighted Brier score **0.0209**.
- **The cascade, measured event by event (n=1000, F146 only, 0.5 d steps):** the model alerts on
  **89.0%** of eligible binaries within the season. Among detections that were not premature,
  the median first crossing is **+5.0 days** after a truth-informed, noise-free anomaly-onset
  proxy. The first crossing is *premature* relative to that proxy for
  **1.6%** of eligible binaries (95% CI 1.0–2.6%), rising to **4.5%** under a stricter definition
  of onset that requires the anomaly to stay detectable. The premature rate depends on the alert
  policy and on the onset definition far more than on the model — see
  `validation/cascade_reproduce_result.json`, which reports it under coarser grids, a
  two-crossing persistence rule, multi-band revealing, and an argmax rule.
- **Cascade ablation:** the matched 400-event comparison is an exploratory risk–coverage curve.
  Its thresholds and paired outcomes were selected on the same events, so its conditional
  McNemar values are descriptive and do not support confirmatory population-level inference.
- **Streaming scope:** the prefix scan contains eligible binaries only and reuses a threshold
  selected on complete seasons. It measures conditional detection timing, not sequential false
  alerts, streaming purity, or broker workload on Flat/PSPL/variable contaminants.
- **Stress testing:** the full suite contains 14.9 million events, but the reported macro-F1
  reproduction applies to its **4.5-million-event same-prior subset**. Separate deliberately
  out-of-distribution arms expose substantial failures; they are diagnostics, not evidence of
  broad population validity.

Full methodology and the honest reading of each number: [`docs/evaluation.md`](docs/evaluation.md).

## Install

BinML is not yet on PyPI; install the inference package (torch ≥ 2.0 + numpy, weights bundled)
from source:
```bash
pip install git+https://github.com/kunalb541/BinML.git
```
For the full simulation, validation, and paper pipeline, install all optional dependencies
(including `VBBinaryLensing`):
```bash
git clone https://github.com/kunalb541/BinML.git && cd BinML
pip install -e ".[all]"
```
The conda environment is an alternative:
```bash
conda env create -f environment.yml && conda activate binml
```

## Classify a light curve (Python)

```python
import binml
clf = binml.Classifier()                      # BinML 1.0 (6-class), CPU, weights bundled

# multi-band: {band: (time_days, magnitude)}; F146 required, colour bands optional
r = clf.predict({"F146": (t146, m146), "F087": (t087, m087), "F213": (t213, m213)},
                m_base_ref=22.1)              # F146 baseline magnitude (recommended)

print(r)                    # <BinML NonPSPL 0.98 | microlensing 0.99 anomalous 0.98>
r.probabilities            # {'Flat':.., 'PSPL':.., 'NonPSPL':.., 'PeriodicVar':.., ...}
r.is_microlensing          # P(PSPL)+P(NonPSPL)
r.is_anomalous             # P(NonPSPL) -- anomalous binary-lens score

# single band (F146 only) is fine:
r = clf.predict(t146, m146, m_base_ref=22.1)

# streaming probabilities as the season is revealed
days, probs = clf.predict_evolution({"F146": (t146, m146)}, m_base_ref=22.1)
```
Command line: `binml classify lc.csv --m-base 22.1`. More: [`docs/usage.md`](docs/usage.md).

## Train / evaluate from scratch

```bash
# one shard of simulated data (all 6 classes) -> cache -> train -> evaluate
python -m pipeline.run_shard --shard 0 --n-shards 1 --out data/raw
python -c "from pipeline.cache import build_cache; import glob; \
build_cache(sorted(glob.glob('data/raw/*.h5')), 'data/cache/shard_00000.h5')"
python -m pipeline.to_memmap --in-dir data/cache --out data/mm
python -m pipeline.train    --cache data/mm --out runs/binml.pt --epochs 6 --device mps
python -m pipeline.evaluate  --ckpt runs/binml.pt --cache data/mm_test --out eval/
```
File-by-file walkthrough: [`docs/pipeline.md`](docs/pipeline.md).

## Simulating data at scale (AWS)

The simulator runs at two scales from one codebase — a single shard on a laptop, or thousands
across a transient cloud fleet — because the modules take `--bucket`/`--prefix` and write locally
if you omit the bucket. Generation is embarrassingly parallel:

- **Index-addressed shards.** One process builds one shard (an HDF5 file of a few thousand light
  curves), with `seed = seed_base + shard * 7919`. Workers HEAD the key
  `prefix/shard_NNNNN.h5` and skip it if present, so a Spot interruption costs at most one shard
  within a frozen run configuration. This is **not content addressing**: the skip check does not
  compare source, simulator configuration, regime, or seed base. Use a new prefix for every such
  change and preserve a run manifest. Work is split by a modulo partition
  (`--worker W --workers N`).
- **`--seed-base` makes evaluation honest.** Train/val/test used base `20260720`; a far-off base
  (e.g. `900000000`) gives a different PCG64 stream — parameter tuples the model did not see in
  training. The full stress suite has 10.4M targeted out-of-distribution events plus a 4.5M
  same-prior subset.
- **Binning & inference run in-region.** Raw shards are ~312 MB each (~125 GB for a full run);
  binning them to compact caches (~46 MB) and running the model *in the S3 region* means the
  light curves never leave — only the compact predictions (~30 floats/event) come back.
- **Self-terminating fleets.** Free-tier Spot instances install deps from a code tarball, refuse
  to start without `VBBinaryLensing` (else binary lenses silently degrade to single-lens and
  ruin the dataset), generate → upload → shut down; a watchdog force-terminates any stragglers.

The `aws/` launch scripts are account-specific infra (hard-coded bucket/region/IAM) and are kept
out of the public tree; the `pipeline` modules underneath them are what run. See
[`docs/pipeline.md`](docs/pipeline.md).

## Repository layout

```
binml/             pip package — the 6-class inference API (Classifier, predict, CLI) + weights
binml/legacy/      the earlier 3-class (Flat/PSPL/Binary) model, preserved for provenance
pipeline/          simulation, training & evaluation modules (run via `python -m pipeline.<mod>`)
docs/              architecture, pipeline, evaluation, data format, usage, model card, leakage audit
paper/             software paper + references
examples/, tests/  usage examples and smoke tests
aws/               (local, gitignored) account-specific fleet-launch scripts
```

## Limitations & intended use

- **Legacy Roman-like cadence.** Trained on dense 15-min F146 sampling for one 72-day season.
  The current survey design differs in cadence, exposure, colour interleaving, and multi-season
  structure. Binary *characterization*
  needs that density — the short caustic anomaly must be observed. On sparse ground-survey
  cadence (LSST, multi-day gaps) detection degrades and characterization is not recoverable.
- **Known weak spots** (targeted out-of-range tests, documented in [`docs/model_card.md`](docs/model_card.md)):
  faint sources m>25 (noise-dominated → false anomalies), wide caustics s>5 (rarely crossed),
  sub-day tE (few epochs on the peak).
- **Synthetic support, not a population forecast.** The simulator uses broad analytic training
  supports, including an authored truncated-lognormal timescale distribution anchored to a
  literature mean. Variable-star curves are analytic or phenomenological shapes, not sampled
  OGLE templates. The 0.02-mag detectability floor has not been validated on Roman data.
- **Known colour-photometry mismatch.** Relative to the current Roman calibration, the released
  simulator's F087 and F213 zeropoints are optimistic by about 0.10 and 0.14 mag, respectively;
  its F087 saturation assumption has the wrong ordering at equal exposure, and its colour-band
  background ratios do not reproduce the published thermal backgrounds. The effect on contaminant
  rejection is unquantified, and the released model has not been retrained with corrected values.
- **Provide `m_base_ref`.** The model input is baseline-relative; give the F146 quiescent
  magnitude when you have it (a catalogue value). The faint-tail estimate is only reliable for
  short, well-sampled events.
- **Intended use:** research triage/vetting of partial Roman-like seasons. The +5-day conditional
  latency does not demonstrate response during a short planetary perturbation. Use the output to
  prioritise modelling and review, not as an autonomous discovery or follow-up trigger.

## Documentation

- [Usage](docs/usage.md) · [Pipeline](docs/pipeline.md) · [Architecture](docs/architecture.md)
- [Evaluation](docs/evaluation.md) · [Data format](docs/data_format.md) · [Model card](docs/model_card.md)
- [Glossary](docs/glossary.md) · [Leakage audit](docs/leakage_audit.md) · [Legacy 3-class model](docs/legacy_3class.md)
- Runnable example: [`examples/quickstart.py`](examples/quickstart.py)

## Citing

See [`CITATION.cff`](CITATION.cff).

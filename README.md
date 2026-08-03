<h1 align="center">BinML</h1>

<p align="center">
  <b>Multi-band, 6-class deep-learning classifier for microlensing & variable light curves.</b><br>
  Built for the <i>Nancy Grace Roman Space Telescope</i> Galactic Bulge Time-Domain Survey.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/pytorch-1.13%2B-orange.svg" alt="PyTorch">
  <a href="https://github.com/kunalb541/BinML/actions"><img src="https://github.com/kunalb541/BinML/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT"></a>
</p>

---

BinML classifies Roman light curves into **six physically-meaningful classes**, from three
photometric bands (F146 at 15-min cadence, F087/F213 at 6-h), and does so **in a way a real
survey can act on**: it only flags a class once that class's evidence is actually observable.

| class | meaning |
|---|---|
| **Flat** | no detectable event (baseline / noise) |
| **PSPL** | single-lens microlensing |
| **NonPSPL** | binary / planetary lens — *the science-critical anomalous class* |
| **PeriodicVar** | short-period variables (RR Lyrae, eclipsing binaries, δ Scuti) |
| **LongPeriodVar** | Miras, semiregulars, OSARGs — the dangerous microlensing impostor |
| **Eruptive** | dwarf novae, Be outbursts |

Two design choices set it apart from a standard classifier:

1. **Detectability-conditioned labelling.** An event is labelled by what is *observable*, not by
   what we simulated. A microlensing event whose peak falls outside the season, or whose
   amplitude is buried by noise, is **Flat**. A binary whose caustic anomaly is below the noise
   floor is, observationally, a **PSPL** — no classifier or human modeller could tell them apart
   from the photometry. This removes the label noise that would otherwise punish the model for
   not seeing what isn't there.

2. **The real-time cascade.** Under a partially-observed season, BinML flags classes only as
   their evidence arrives: **Flat → PSPL → NonPSPL**. A binary reads as a plain PSPL during its
   smooth rise and becomes NonPSPL only when the caustic is on screen. This is what a Roman
   follow-up pipeline needs — **it must not trigger on a false binary before it has seen one.**

## The model

The network is a **convolutional stem feeding a small transformer encoder** — **505,479
parameters** — deliberately small, because the task is not data-starved (millions of simulated
events) but inference-heavy at survey scale.

- **Input: 3 bands → 156 tokens × 5 channels.** F146 is the workhorse (15-min cadence, 6912
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

## Results (independent 450,589-event test set)

Per-class F1 (population-weighted, selection-corrected):

| Flat | PSPL | NonPSPL | PeriodicVar | LongPeriodVar | Eruptive |
|---|---|---|---|---|---|
| 0.97 | 0.96 | 0.82 | 0.97 | 0.91 | 0.88 |

> The NonPSPL F1 (0.82) is held down by *precision*, and that precision is a labelling artefact,
> not model error: **71% of its "false positives" are genuine binaries whose caustic anomaly
> falls below the detectability floor** and were therefore labelled PSPL (see detectability
> conditioning above). Counting those as correct — they *are* physically anomalous — the physical
> precision is 0.99. NonPSPL **recall is 0.95**. This is why completeness@purity, not F1, is the
> headline.

- **Completeness at fixed purity: 0.879** (the headline a follow-up pipeline is specified
  against, not accuracy or F1). Average precision 0.952.
- **The cascade works:** premature NonPSPL flagging (calling a binary before its anomaly is
  observable) dropped **42% → 9%** — a factor of ~4.7 (78% fewer) — with no loss on full-season
  discrimination; the mean pre-onset anomaly probability fell further, 0.411 → 0.033.
- **Generalises to unseen parameter draws:** a ~19-million-event stress set from a disjoint seed
  reproduces the held-out numbers on its natural-population subset; several out-of-range regimes
  degrade, and are documented rather than hidden.

Full methodology and the honest reading of each number: [`docs/evaluation.md`](docs/evaluation.md).

## Install

BinML is not yet on PyPI; install the inference package (torch ≥ 2.0 + numpy, weights bundled)
from source:
```bash
pip install git+https://github.com/kunalb541/BinML.git
```
For the full simulation/training pipeline (adds `h5py`, `scipy`, and `VBBinaryLensing` for
binary-lens generation) use the conda environment:
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
r.is_anomalous             # P(NonPSPL)  -- is it binary/planetary?

# single band (F146 only) is fine:
r = clf.predict(t146, m146, m_base_ref=22.1)

# the real-time cascade: probabilities as the season is revealed
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

- **Content-addressed shards.** One process builds one shard (an HDF5 file of a few thousand
  light curves), seeded purely by its index: `seed = seed_base + shard * 7919`. Workers HEAD each
  shard's S3 key and skip existing ones, so a Spot interruption costs at most one shard, and runs
  are fully resumable. Work is split by a modulo partition (`--worker W --workers N`).
- **`--seed-base` makes evaluation honest.** Train/val/test used base `20260720`; a far-off base
  (e.g. `900000000`) gives a different PCG64 stream — parameter tuples the model *provably* never
  saw. That's how the 12.9M-event unseen-parameter stress test is built.
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

- **Roman-quality cadence.** Trained on dense ~15-min F146 sampling. Binary *characterization*
  needs that density — the short caustic anomaly must be observed. On sparse ground-survey
  cadence (LSST, multi-day gaps) detection degrades and characterization is not recoverable.
- **Known weak spots** (rare out-of-range extremes, documented in [`docs/model_card.md`](docs/model_card.md)):
  faint sources m>25 (noise-dominated → false anomalies), wide caustics s>5 (rarely crossed),
  sub-day tE (few epochs on the peak). These are near fundamental physical limits, not gaps.
- **Provide `m_base_ref`.** The model input is baseline-relative; give the F146 quiescent
  magnitude when you have it (a catalogue value). The faint-tail estimate is only reliable for
  short, well-sampled events.
- **Intended use:** triage/vetting of Roman GBTDS light curves — a follow-up-triggering aid, not
  a substitute for full light-curve modelling of a candidate.

## Documentation

- [Usage](docs/usage.md) · [Pipeline](docs/pipeline.md) · [Architecture](docs/architecture.md)
- [Evaluation](docs/evaluation.md) · [Data format](docs/data_format.md) · [Model card](docs/model_card.md)
- [Glossary](docs/glossary.md) · [Leakage audit](docs/leakage_audit.md) · [Legacy 3-class model](docs/legacy_3class.md)
- Runnable example: [`examples/quickstart.py`](examples/quickstart.py)

## Citing

See [`CITATION.cff`](CITATION.cff).

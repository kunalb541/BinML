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
   floor is, observationally, a **PSPL** — because no classifier or human modeller could tell
   them apart from the photometry. This removes the label noise that would otherwise punish the
   model for not seeing what isn't there.

2. **The real-time cascade.** Under a partially-observed season, BinML flags classes only as
   their evidence arrives: **Flat → PSPL → NonPSPL**. A binary reads as a plain PSPL during its
   smooth rise and only becomes NonPSPL when the caustic is actually on screen. This is what a
   Roman follow-up pipeline needs — **it must not trigger on a false binary before it has seen
   one.**

## Results — the model (independent 450,589-event test set)

Per-class F1 (population-weighted):

| Flat | PSPL | NonPSPL | PeriodicVar | LongPeriodVar | Eruptive |
|---|---|---|---|---|---|
| 0.99 | 0.92 | 0.93 | 0.96 | 0.95 | 0.88 |

- **Completeness at fixed purity: 0.879** (the headline — what a follow-up pipeline is
  specified against, not accuracy or F1). Average precision 0.952.
- **The cascade works:** premature NonPSPL flagging (calling a binary before its anomaly is
  observable) dropped from **42% → 9%** vs the previous model — a 12× reduction — with no loss on
  full-season discrimination (AP tied, completeness-at-matched-purity curves cross).
- **Generalises to unseen parameters:** a 12.9-million-event stress test on parameter draws the
  model never saw reproduces the held-out numbers, with documented failure modes only at the
  out-of-range extremes (faint m>25, wide caustics s>5, sub-day tE).

See [`docs/evaluation.md`](docs/evaluation.md) for the full analysis and the honest reading of
each number.

## Documentation

- **[Pipeline](docs/pipeline.md)** — the current model: simulation, training & evaluation,
  file-by-file, with verified quick-start commands. **Start here.**
- [Architecture](docs/architecture.md) — the conv-stem + transformer, and why it's built this way
- [Evaluation](docs/evaluation.md) — detectability-conditioned, honest metrics
- [Data format](docs/data_format.md) — the compact multi-band cache
- [Leakage audit](docs/leakage_audit.md) — how train/test disjointness is guaranteed

## Quick start

```bash
# one shard of simulated data (all 6 classes) -> cache -> train -> evaluate
python -m pipeline.run_shard --shard 0 --n-shards 1 --out data/raw
python -c "from pipeline.cache import build_cache; import glob; \
build_cache(sorted(glob.glob('data/raw/*.h5')), 'data/cache/shard_00000.h5')"
python -m pipeline.to_memmap --in-dir data/cache --out data/mm
python -m pipeline.train   --cache data/mm --out runs/binml.pt --epochs 6 --device mps
python -m pipeline.evaluate --ckpt runs/binml.pt --cache data/mm_test --out eval/
```

Full walkthrough and every module's role: [`docs/pipeline.md`](docs/pipeline.md).

## Repository layout

```
pipeline/   the current 6-class multi-band pipeline (simulate, train, evaluate, plot)
binml/             legacy installable 3-class inference package (Flat/PSPL/Binary) + weights
docs/              architecture, pipeline, evaluation, data format, leakage audit
paper/             software paper + references
examples/, tests/  usage examples and smoke tests
```

> **Note.** The `binml/` pip package is the earlier **3-class** (Flat/PSPL/Binary) classifier and
> its `pip install binml` API is unchanged. The **6-class multi-band** model described above is
> the current research pipeline in [`pipeline/`](pipeline/).

## Cadence matters

BinML is trained on **Roman-quality cadence** (dense ~15-min F146 sampling). Binary
characterization needs that density — the short caustic anomaly must actually be observed. On
sparse ground-survey cadence (e.g. LSST at multi-day gaps) detection degrades and
characterization is not recoverable. Roman is the planet-finder; LSST is at best a detector.

## Citing

See [`CITATION.cff`](CITATION.cff).

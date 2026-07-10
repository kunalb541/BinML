<h1 align="center">BinML</h1>

<p align="center">
  <b>Deep-learning classifier for gravitational microlensing light curves.</b><br>
  Flat · PSPL · Binary — built for <i>Nancy Grace Roman Space Telescope</i> cadence.
</p>

<p align="center">
  <a href="https://pypi.org/project/binml/"><img src="https://img.shields.io/pypi/v/binml.svg" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/pytorch-1.13%2B-orange.svg" alt="PyTorch">
  <a href="https://github.com/kunalb541/BinML/actions"><img src="https://github.com/kunalb541/BinML/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT"></a>
</p>

---

BinML predicts, from a microlensing light curve, **three physically-meaningful classes**:

| class | meaning |
|---|---|
| **Flat** | no event (baseline / noise) |
| **PSPL** | single-lens microlensing — *"there is a microlensing event"* |
| **Binary** | planetary / binary lens — *the anomalous class (a binary is **not** a PSPL)* |

So every prediction answers two questions: **detection** — is this a microlensing event
(`P(PSPL) + P(Binary)`)? — and **characterization** — is it anomalous, i.e. binary/planetary
rather than a plain single lens (`P(Binary)`)?

It's a modern take on tools like [MicroLIA](https://github.com/Professor-G/MicroLIA): a
causal CNN–GRU trained on Roman-quality cadence, with the weights bundled so it runs out of
the box.

## Install

```bash
pip install binml            # weights bundled — only needs torch + numpy
```

## Quickstart

```python
import binml

clf = binml.Classifier()                                 # fine-tuned model, CPU
t, mag, err = binml.surveys.fetch_ogle_ews(2014, 289)    # a real OGLE event
r = clf.predict(t, mag, err)

print(r)
# <BinML Binary (conf 1.00) | Flat 0.000 PSPL 0.000 Binary 1.000 | microlensing=1.00 anomalous=1.00>
r.is_microlensing   # 1.00  -> yes, a microlensing event
r.is_anomalous      # 1.00  -> binary/planetary, not plain PSPL
r.probabilities     # {'Flat': ..., 'PSPL': ..., 'Binary': ...}
```

Your own light curve — any `time, mag, mag_err` arrays (or `is_flux=True` for fluxes):

```python
r = clf.predict(time, mag, mag_err)
```

## Command line

```bash
binml classify phot.dat --format ogle          # classify a file
binml ogle 2017 482                            # fetch OGLE-2017-BLG-0482 and classify
binml evolution phot.dat -o evolution.png      # probability-evolution plot
binml evaluate test_set.h5                      # detectability-conditioned evaluation
```

## Reporting binary performance honestly

**A binary whose caustic isn't sampled/perturbing is observationally identical to a PSPL** —
calling it "PSPL" is physically correct, not an error. A single population-level binary
recall therefore conflates real model skill with irreducible physical degeneracy. BinML makes
the honest, **detectability-conditioned** metric the default:

```python
report, detect = binml.evaluate_dataset(clf, "test_set.h5")
print(detect)   # binary recall vs Δχ², the indistinguishable fraction, detectable-only recall
```

Report **detectable-only recall + the indistinguishable fraction**, never the raw population
number. See [docs/evaluation.md](docs/evaluation.md).

## Cadence matters

BinML is trained on **Roman-quality cadence** (dense ~15-min sampling). It transfers to the
*shape* of real single-lens events, but **binary characterization needs dense sampling** — the
short caustic anomaly must actually be observed. On sparse ground-survey cadence (e.g. LSST at
multi-day gaps) detection degrades and characterization is not recoverable. Roman is the
planet-finder; LSST is at best a detector.

## Results (full 1,000,000-event held-out evaluation)

| model | accuracy | Flat | PSPL | Binary | notes |
|---|---|---|---|---|---|
| base | 64.3% | 100% | 80.0% | 52.2% | balanced |
| **fine-tuned** (shipped) | 67.2% | 100% | 70.9% | 62.3% | planetary recall 41→57% |

Binary recall **rises monotonically with anomaly strength** (Δχ²) from ~11% (no detectable
signal) to **85%** (strong anomaly). At a Δχ²≥300 detection threshold, ~32% of binaries are
physically indistinguishable from PSPL and the **detectable-only binary recall is ~78%** — the
raw 62% understates the model. Validated on real OGLE-IV events (correctly flags
caustic-crossing binaries such as OGLE-2014-BLG-0289 and OGLE-2013-BLG-0578).

## Repository layout

```
binml/        installable inference package (Classifier, preprocess, evaluate, surveys, cli) + bundled weights
pipeline/     research pipeline — simulate.py, train.py, evaluate.py, select_subset.py,
              train_modal.py (Modal GPU orchestration), analysis/, curricula/
docs/         architecture, training, data format, evaluation, leakage audit
paper/        software paper (JOSS-style) + references
examples/     usage examples          tests/  smoke tests
data/  results/   simulated data and figures
```

## Training from scratch

The full pipeline (simulate → detectability-aware subset → single-GPU streaming training →
warm-start fine-tuning) is in [`pipeline/`](pipeline/) and documented in
[docs/training.md](docs/training.md). It runs locally on one GPU or on
[Modal](https://modal.com) via `pipeline/train_modal.py`. Fine-tuning uses targeted
warm-starting (`train.py --init-weights`) on the hard low-mass-ratio regime — which moves the
needle far more than additional base training.

## Documentation

- [Architecture](docs/architecture.md) — the causal CNN–GRU
- [Training](docs/training.md) — simulate → subset → train → fine-tune
- [Data format](docs/data_format.md) — the compact HDF5 dataset
- [Evaluation](docs/evaluation.md) — detectability-conditioned, honest metrics
- [Leakage audit](docs/leakage_audit.md)

## Citing BinML

If you use BinML, please cite it — see [CITATION.cff](CITATION.cff) (a "Cite this repository"
button appears on GitHub) and the paper in [`paper/`](paper/).

## License

[MIT](LICENSE) © Kunal Bhatia

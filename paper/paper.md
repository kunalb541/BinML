---
title: "BinML: A detectability-honest deep-learning classifier for gravitational microlensing light curves"
tags:
  - Python
  - astronomy
  - microlensing
  - exoplanets
  - deep learning
authors:
  - name: Kunal Bhatia
    affiliation: 1
affiliations:
  - name: University of Heidelberg
    index: 1
date: 10 July 2026
bibliography: paper.bib
---

# Summary

`BinML` is a deep-learning classifier for gravitational microlensing light curves,
designed for the observing cadence of the Nancy Grace Roman Space Telescope
[@roman]. It sorts a photometric time series into three physically motivated
classes: **Flat** (no event), **PSPL** (a single-lens point-source point-lens
event — the proxy for "is this microlensing at all"), and **Binary** (a
planetary or binary lens, the distinct anomalous class that carries the
exoplanet signal). A binary event is treated as its own class rather than a
perturbation of a PSPL event, which lets the model separate the two questions
that matter observationally: *did a microlensing event occur*, and *does it show
a caustic anomaly*.

The model is a compact CNN-GRU (~130K parameters). It consumes two channels
only — the magnification $A = 10^{0.4(m_\mathrm{base}-m)}$ and the time gap
$\Delta t$ since the previous valid observation — and is fully causal: a
depthwise-separable convolutional stack with left-padding only, a
unidirectional GRU, and masked multi-head attention pooling. Causality means a
truncated prefix of a light curve is always a valid input, enabling
probability-evolution and early-detection analyses. Classification is
hierarchical: a stage-1 deviation logit separates Flat from event, and a stage-2
head separates Binary from PSPL, with an auxiliary 3-class head.

`BinML` ships as a pip-installable package (`binml`) with bundled weights,
requiring only `torch` and `numpy` for inference, alongside a full research
pipeline for simulation, training, and evaluation.

# Statement of need

Microlensing is the primary channel through which Roman will detect cold,
low-mass exoplanets, and it will produce far more light curves than can be
inspected by hand. Automated triage is therefore a prerequisite, not a
convenience. The most widely used open-source tool, `MicroLIA` [@microlia],
frames the problem as random-forest classification over engineered statistical
features. `BinML` differs in two ways that we consider its core contributions.

First, the **three-class framing**. Rather than a binary "event / no-event"
decision, `BinML` explicitly models Flat, PSPL, and Binary, so that detection
(PSPL and Binary together) and characterization (Binary alone) are read directly
from the output probabilities. A learned causal sequence model, rather than a
fixed feature set, lets the network attend to the short, sharply localized
caustic crossings that distinguish a binary from a single lens.

Second, and more importantly, **detectability-honest evaluation**. A large
fraction of simulated binaries produce anomalies too weak to be distinguished
from a single-lens fit given the noise and sampling; calling these "missed
binaries" penalizes the classifier for a limit imposed by physics, not by the
model. `BinML` conditions all binary-recovery metrics on the anomaly
$\Delta\chi^2$ — the $\chi^2$ of the true binary relative to a matched
single-lens fit — which quantifies whether the anomaly is present in the data at
all. This turns an inflated "error" rate into an interpretable
detectability curve and is, to our knowledge, not standard practice in
microlensing classification.

# Functionality

The **package** exposes a small, stable API:

```python
import binml
clf = binml.Classifier()                 # fine-tuned default; Classifier("base") for balanced
r = clf.predict(time, mag, mag_err)      # -> .probabilities, .label, .is_microlensing, .is_anomalous
clf.predict_evolution(...)               # class probabilities over growing prefixes
binml.evaluate_dataset(clf, "test.h5")   # -> (ClassificationReport, DetectabilityReport)
binml.surveys.fetch_ogle_ews(2014, 289)  # plus load_ogle / load_moa / load_generic
```

A command-line interface mirrors this: `binml classify`, `binml ogle`,
`binml evolution`, and `binml evaluate`.

The **pipeline** covers the research workflow end to end. `simulate.py` generates
events with `VBBinaryLensing` [@vbbinarylensing] into a compact HDF5 format —
6912 points at 15-minute sampling over a 72-day Roman season, with per-event
physical parameters and a packed observation mask. `select_subset.py` performs
detectability-aware subset selection, stratifying binaries by anomaly
$\Delta\chi^2$ and mass ratio $q$ so the training set is not dominated by
undetectable anomalies. `train.py` is a single-GPU streaming trainer
(block-shuffle sampling, hierarchical loss, AMP, cosine schedule with warmup,
early stopping, atomic checkpointing) built on PyTorch [@pytorch], with
normalization statistics fit on the training split only to avoid leakage.
`train_modal.py` orchestrates the full lifecycle — download, train, fine-tune,
and streaming evaluation — on cloud L4 GPUs. Fine-tuning is a warm-start from an
existing checkpoint (`--init-weights`) on a targeted, anomaly-rich set; this,
rather than additional base training, is the lever that improves the hard
low-mass-ratio regime.

# Results

On a held-out evaluation of 1,000,000 events, the base model reaches 64.3%
overall accuracy, with recall of 100% (Flat), 80.0% (PSPL), and 52.2% (Binary),
and 81.7% binary precision. A single round of targeted fine-tuning raises overall
accuracy to 67.2% (Flat 100%, PSPL 70.9%, Binary 62.3%) and, in the planetary
regime ($q \sim 10^{-4}$–$10^{-3}$), lifts binary recall from 41.2% to 57.4%. A
second, over-aggressive fine-tuning round collapsed — labelling almost everything
Binary and dropping PSPL recall to 18% — and was discarded; round 1 is the
shipped model.

These raw population numbers understate performance because they average over
undetectable binaries. Conditioned on anomaly $\Delta\chi^2$, binary recall rises
monotonically from ~11% (no detectable signal) to 85% (strong anomaly). At a
$\Delta\chi^2 \geq 300$ detection threshold, ~32% of binaries are physically
indistinguishable from PSPL, and the detectable-only binary recall is ~78%,
versus the 62% raw figure; roughly 60% of the "missed" binaries at that threshold
are genuinely PSPL-like rather than model errors. **Binary performance should
always be reported conditioned on detectability.** A KS test on the event time
$t_0$ of correct versus incorrect classifications gives $D \sim 0.01$, indicating
no meaningful temporal bias.

The distinction between **detection** and **characterization** is set by cadence.
With Roman-class ~15-minute sampling the model does both. Applied zero-shot to
LSST-like cadence, microlensing *detection* holds only to about one visit per day
(~89% recall) and collapses by ~3-day gaps (~4%), while binary *characterization*
is ~0% at any LSST cadence because the short caustic anomaly is never sampled.
Roman is the planet-finder; LSST is at best a detector.

Applied to real OGLE-IV Early Warning System data [@ogle_ews], `BinML` correctly
flags caustic-crossing binaries (e.g. OGLE-2014-BLG-0289, OGLE-2013-BLG-0578) and
clean single-lens events, and the fine-tuned model recovers real binaries the
base model missed (OGLE-2011-BLG-0417, OGLE-2015-BLG-0060).

# Availability

`BinML` is MIT-licensed and available at
<https://github.com/kunalb541/BinML>. It requires Python $\geq 3.9$; inference
needs only `torch` and `numpy`, with bundled weights. Documentation, worked
examples, and tests are included in the repository.

# Acknowledgements

This project began as an MSc research effort at the University of Heidelberg and
is released as a standalone open-source tool. We thank the OGLE collaboration for
the public Early Warning System data used in validation.

# References

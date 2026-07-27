"""
Evaluation with **detectability-conditioned** binary scoring.

Why this module exists
----------------------
A binary lens whose caustic is not sampled/perturbing produces *no detectable anomaly* — it
is observationally identical to a single lens (PSPL). Calling such an event "PSPL" is the
physically correct inference, not an error. So a single population-level binary recall
conflates two very different things:

  * genuine model error, and
  * irreducible physical degeneracy (binaries that simply look like PSPL).

The honest way to report binary performance is therefore **conditioned on detectability**:
score only the binaries that carry a real signal (Δχ² of the binary fit vs a matched
single-lens above a threshold), report the *indistinguishable fraction* separately, and show
recall as a function of Δχ² (a detectability–completeness curve). Optimising the raw
population recall instead pushes a model to call everything Binary and destroy PSPL.

Functions here work on already-computed (labels, preds) plus an optional per-event Δχ²
(``anomaly_dchi2``), or end-to-end on a compact HDF5 test set via :func:`evaluate_dataset`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .classifier import CLASS_NAMES

__all__ = ["ClassificationReport", "DetectabilityReport",
           "classification_report", "detectability_curve", "evaluate_dataset"]

DEFAULT_THRESHOLDS = (50.0, 100.0, 300.0, 1000.0)
DEFAULT_DCHI2_BINS = (0, 20, 50, 100, 300, 1000, 1e4, 1e12)
BINARY = 2


@dataclass
class ClassificationReport:
    accuracy: float
    recall: Dict[str, float]
    precision: Dict[str, float]
    f1: Dict[str, float]
    confusion: List[List[int]]           # rows = true, cols = pred (Flat, PSPL, Binary)

    def __repr__(self) -> str:
        rows = "\n".join(
            f"    {c:7s} recall={self.recall[c]:.3f} precision={self.precision[c]:.3f} f1={self.f1[c]:.3f}"
            for c in CLASS_NAMES)
        return f"<ClassificationReport acc={self.accuracy:.3f}\n{rows}>"


@dataclass
class DetectabilityReport:
    """Binary performance as a function of anomaly detectability (Δχ²)."""
    n_binaries: int
    population_recall: float                       # raw, over all binaries (dragged down by physics)
    thresholds: Dict[str, Dict[str, float]]        # per Δχ² threshold: indistinguishable frac, detectable recall, ...
    recall_by_dchi2_bin: Dict[str, Dict[str, float]]

    def summary(self) -> str:
        lines = [f"Binary detectability report ({self.n_binaries} binaries)",
                 f"  raw population recall: {self.population_recall*100:.1f}%  "
                 f"(conflates model skill with physical degeneracy)",
                 "  detectability-conditioned:"]
        for thr, v in self.thresholds.items():
            lines.append(
                f"    Δχ²>={thr:>6}: indistinguishable={v['indistinguishable_frac']*100:4.1f}%  "
                f"detectable-only recall={v['recall_detectable']*100:5.1f}%  "
                f"(of misses, {v['frac_of_missed_indistinguishable']*100:.0f}% are physically PSPL)")
        lines.append("  recall vs Δχ²:")
        for b, v in self.recall_by_dchi2_bin.items():
            lines.append(f"    Δχ² {b:14s} n={v['n']:<7d} recall={v['recall']*100:5.1f}%")
        return "\n".join(lines)

    __repr__ = summary


def classification_report(labels, preds) -> ClassificationReport:
    """Standard 3-class report from integer labels/preds (0=Flat,1=PSPL,2=Binary)."""
    labels = np.asarray(labels); preds = np.asarray(preds)
    conf = np.zeros((3, 3), np.int64)
    for t, p in zip(labels, preds):
        conf[int(t), int(p)] += 1
    rec, prec, f1 = {}, {}, {}
    for i, c in enumerate(CLASS_NAMES):
        r = conf[i, i] / max(conf[i].sum(), 1)
        p = conf[i, i] / max(conf[:, i].sum(), 1)
        rec[c] = float(r); prec[c] = float(p)
        f1[c] = float(2 * r * p / (r + p)) if (r + p) else 0.0
    acc = float(np.trace(conf) / max(conf.sum(), 1))
    return ClassificationReport(acc, rec, prec, f1, conf.tolist())


def detectability_curve(labels, preds, anomaly_dchi2,
                        thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
                        dchi2_bins: Sequence[float] = DEFAULT_DCHI2_BINS) -> DetectabilityReport:
    """Detectability-conditioned binary scoring.

    Parameters
    ----------
    labels, preds : int arrays (2 == Binary)
    anomaly_dchi2 : per-event Δχ² of the true binary vs a matched single-lens. Only the
        entries for true binaries are used. (From simulate.py's ``params['anomaly_dchi2']``.)
    """
    labels = np.asarray(labels); preds = np.asarray(preds)
    an = np.asarray(anomaly_dchi2, dtype=float)
    b = labels == BINARY
    an_b = an[b]; caught = preds[b] == BINARY
    n = int(b.sum())
    pop = float(caught.mean()) if n else float("nan")

    thr_out = {}
    for thr in thresholds:
        det = an_b >= thr; ind = ~det; missed = ~caught
        thr_out[str(int(thr))] = dict(
            indistinguishable_frac=float(ind.mean()) if n else float("nan"),
            recall_detectable=float(caught[det].mean()) if det.sum() else float("nan"),
            recall_indistinguishable=float(caught[ind].mean()) if ind.sum() else float("nan"),
            frac_of_missed_indistinguishable=float(ind[missed].mean()) if missed.sum() else float("nan"),
        )
    bin_out = {}
    for lo, hi in zip(dchi2_bins[:-1], dchi2_bins[1:]):
        m = (an_b >= lo) & (an_b < hi)
        if m.sum():
            bin_out[f"{lo:g}-{hi:g}"] = dict(n=int(m.sum()), recall=float(caught[m].mean()))
    return DetectabilityReport(n, pop, thr_out, bin_out)


def evaluate_dataset(clf, h5_path, batch: int = 256, thresholds=DEFAULT_THRESHOLDS):
    """Run ``clf`` over a compact HDF5 test set and return
    ``(ClassificationReport, DetectabilityReport | None)``.

    The file must have ``flux``, ``delta_t``, ``labels``; if it also has a ``params`` table
    with an ``anomaly_dchi2`` field (BinML's simulated test sets do), the detectability
    report is produced. Real-data sets without truth get only the classification report
    (or nothing, if unlabelled).
    """
    import h5py

    with h5py.File(str(h5_path), "r") as f:
        flux = f["flux"][:]; delta_t = f["delta_t"][:]
        labels = f["labels"][:] if "labels" in f else None
        dchi2 = None
        if "params" in f and "anomaly_dchi2" in f["params"].dtype.names:
            dchi2 = f["params"]["anomaly_dchi2"][:]
    probs = clf.predict_arrays(flux, delta_t, batch=batch)
    preds = probs.argmax(1)
    if labels is None:
        return None, None
    report = classification_report(labels, preds)
    detect = detectability_curve(labels, preds, dchi2, thresholds) if dchi2 is not None else None
    return report, detect

"""Plotting helpers: a prediction summary and the probability-evolution view."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .classifier import CLASS_NAMES, Evolution, Prediction

__all__ = ["plot_prediction", "plot_evolution"]

_COL = ["#9aa0a6", "#edae49", "#2e8b57"]  # Flat, PSPL, Binary


def plot_prediction(pred: Prediction, ax=None, title: Optional[str] = None):
    """Light curve + a probability bar. Returns the matplotlib Figure."""
    import matplotlib.pyplot as plt

    pre = pred._pre
    if pre is None:
        raise ValueError("Prediction has no stored light curve; call predict() directly.")
    if ax is None:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6),
                                     gridspec_kw={"height_ratios": [3, 1]})
    else:
        a1, a2 = ax; fig = a1.figure
    x = pre.time - pre.t0
    a1.errorbar(x, pre.mag, yerr=pre.mag_err, fmt=".", ms=3, elinewidth=0.4,
                alpha=0.6, color="#1f77b4")
    a1.axhline(pre.m_base, ls="--", color="gray", lw=0.8, label=f"baseline {pre.m_base:.2f}")
    a1.invert_yaxis(); a1.set_xlabel("days from peak"); a1.set_ylabel("magnitude")
    a1.set_title(title or f"BinML: {pred.label} ({pred.confidence:.2f})")
    a1.legend(fontsize=8)
    p = [pred.probabilities[c] for c in CLASS_NAMES]
    a2.barh(list(CLASS_NAMES), p, color=_COL, edgecolor="black")
    a2.set_xlim(0, 1); a2.set_xlabel("probability")
    for i, v in enumerate(p):
        a2.text(min(v + 0.02, 0.88), i, f"{v:.2f}", va="center", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_evolution(evo: Evolution, title: Optional[str] = None):
    """3-panel probability-evolution plot (light curve / class prob / confidence)."""
    import matplotlib.pyplot as plt

    pre = evo.final._pre
    x = evo.days_from_peak
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 2, 1.4]})
    ax[0].errorbar(pre.time - pre.t0, pre.mag, yerr=pre.mag_err, fmt=".", ms=3,
                   elinewidth=0.4, alpha=0.6, color="#1f77b4")
    ax[0].axhline(pre.m_base, ls="--", color="gray", lw=0.8)
    ax[0].invert_yaxis(); ax[0].set_ylabel("magnitude")
    ax[0].set_title(title or f"BinML probability evolution -> {evo.final.label} "
                             f"{evo.final.confidence:.2f}")
    for c in range(3):
        ax[1].plot(x, evo.probabilities[:, c], color=_COL[c], lw=2, label=CLASS_NAMES[c])
    ax[1].axhline(1 / 3, ls=":", color="gray", lw=0.8)
    ax[1].set_ylabel("class probability"); ax[1].set_ylim(-0.02, 1.02)
    ax[1].legend(loc="center left", ncol=3, fontsize=9)
    conf = evo.probabilities.max(1)
    ax[2].fill_between(x, conf, color="black", alpha=0.12); ax[2].plot(x, conf, "k", lw=1.5)
    ax[2].set_ylabel("confidence"); ax[2].set_ylim(0.3, 1.02)
    ax[2].set_xlabel("days from peak (latest observation fed)")
    fig.tight_layout()
    return fig

#!/usr/bin/env python3
"""Simulate representative events and render publication figures from raw light curves.

Two figures the evaluation artifact alone cannot produce, both generated from freshly simulated
events (deterministic seeds) run through the shipped model:

  lightcurve_gallery.pdf  -- one representative event per class, as a Roman observer sees it.
  cascade_evolution.pdf   -- the real-time cascade: class probability as the season is revealed,
                             for a clean binary, a subtle binary, and a single-lens control.

Needs the simulation stack (VBBinaryLensing, h5py, scipy) and the binml package. Run from paper/:
    python make_data_figures.py
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.assemble import simulate_event, SurveyConfig
import binml

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.linewidth": 0.7,
    "font.family": "serif", "mathtext.fontset": "cm",
})

CLF = binml.Classifier()
CLASS_NAMES = binml.CLASS_NAMES
CFG = SurveyConfig()
COL = {"Flat": "#7f7f7f", "PSPL": "#1f77b4", "NonPSPL": "#d62728",
       "PeriodicVar": "#2ca02c", "LongPeriodVar": "#9467bd", "Eruptive": "#ff7f0e"}
PRETTY = {"Flat": "Flat", "PSPL": "PSPL (single lens)", "NonPSPL": "NonPSPL (binary/planet)",
          "PeriodicVar": "PeriodicVar", "LongPeriodVar": "LongPeriodVar", "Eruptive": "Eruptive"}


def bands_dict(ev):
    return {b: (ev.bands[b].t, ev.bands[b].mag) for b in ev.bands}


def predict(ev):
    p = CLF.predict(bands_dict(ev), m_base_ref=ev.params["_m_base_ref"], t_start=0.0)
    return p.probabilities, p.label


def find_example(true_class, seeds, want_amp=None, need_correct=True, prefer_conf=True):
    """Search seeds for a clean, representative, correctly-labelled event."""
    best = None
    for s in seeds:
        ev = simulate_event(true_class, np.random.default_rng(s), CFG)
        if ev is None or "F146" not in ev.bands or len(ev.bands["F146"].t) < 50:
            continue
        if need_correct and ev.label != true_class:
            continue
        probs, label = predict(ev)
        if need_correct and label != true_class:
            continue
        amp = ev.bands["F146"].mag.max() - ev.bands["F146"].mag.min()
        if want_amp is not None and amp < want_amp:
            continue
        conf = probs.get(true_class, 0.0)
        score = conf if prefer_conf else amp
        if best is None or score > best[0]:
            best = (score, ev, probs, label)
    return best[1:] if best else (None, None, None)


# ------------------------------------------------------------------ light-curve gallery
def fig_gallery():
    seeds = range(1, 400)
    picks = {}
    want = {"Flat": None, "PSPL": 0.5, "NonPSPL": 0.6, "PeriodicVar": 0.4,
            "LongPeriodVar": 0.3, "Eruptive": 0.5}
    for c in CLASS_NAMES:
        ev, probs, label = find_example(c, seeds, want_amp=want[c])
        if ev is None:
            ev, probs, label = find_example(c, seeds, want_amp=None)
        picks[c] = (ev, probs, label)

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.4))
    folded = set()
    for ax, c in zip(axes.ravel(), CLASS_NAMES):
        ev, probs, label = picks[c]
        b = ev.bands["F146"]
        step = max(1, len(b.t) // 1400)
        # PeriodicVar: phase-fold so the (fast) variability is legible, not aliased to noise
        if c == "PeriodicVar" and ev.params.get("P"):
            P = float(ev.params["P"])
            ph = (b.t % P) / P
            ax.scatter(ph[::step], b.mag[::step], s=1.4, color=COL[c], alpha=0.45, lw=0)
            ax.scatter(ph[::step] + 1.0, b.mag[::step], s=1.4, color=COL[c], alpha=0.45, lw=0)
            ax.set_xlim(0, 2); folded.add(c)
            ax.text(0.5, 0.93, f"folded, $P={P:.1f}$ d", transform=ax.transAxes,
                    fontsize=6.5, ha="center", va="top", color="#555")
        else:
            ax.scatter(b.t[::step], b.mag[::step], s=1.4, color=COL[c], alpha=0.5, lw=0)
        ax.invert_yaxis()
        ax.set_title(f"{PRETTY[c]}", fontsize=8.5)
        ax.text(0.03, 0.06, f"model: {label} ({100*probs[label]:.0f}%)",
                transform=ax.transAxes, fontsize=6.8, va="bottom",
                color="black" if label == c else "#b00")
        ax.tick_params(labelsize=7)
        ax.margins(x=0.02)
    for ax, c in zip(axes[1], CLASS_NAMES[3:]):
        ax.set_xlabel("phase" if c in folded else "days since season start")
    for ax in axes[:, 0]:
        ax.set_ylabel("F146 magnitude")
    fig.suptitle("Representative simulated Roman light curves, one per class",
                 fontsize=9.5, y=1.005)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "lightcurve_gallery.pdf"))
    plt.close(fig)
    print("lightcurve_gallery.pdf")


# --------------------------------------------------------------- cascade / prob evolution
def _prob_evolution(ev, n_steps=24):
    days, probs = CLF.predict_evolution(bands_dict(ev), m_base_ref=ev.params["_m_base_ref"],
                                        t_start=0.0, n_steps=n_steps)
    return days, probs


def fig_cascade():
    # a clean strong binary, a subtle one, and a single-lens control that must stay PSPL
    clean, cp, _ = find_example("NonPSPL", range(1, 300), want_amp=0.8)
    subtle, sp, _ = find_example("NonPSPL", range(300, 900), want_amp=0.25, prefer_conf=False)
    control, ctp, _ = find_example("PSPL", range(1, 300), want_amp=0.6)
    events = [("Clean binary", clean), ("Subtle binary", subtle), ("Single-lens control", control)]

    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.4),
                             gridspec_kw={"height_ratios": [1.0, 1.15]})
    for col, (ttl, ev) in enumerate(events):
        b = ev.bands["F146"]
        t_anom = ev.params.get("t_anom")
        # --- top: light curve
        axt = axes[0, col]
        step = max(1, len(b.t) // 900)
        axt.scatter(b.t[::step], b.mag[::step], s=1.3, color="#333", alpha=0.45, lw=0)
        axt.invert_yaxis()
        axt.set_title(ttl, fontsize=8.8)
        if col == 0:
            axt.set_ylabel("F146 mag")
        if t_anom is not None and ev.true_class == "NonPSPL":
            axt.axvline(t_anom, color=COL["NonPSPL"], lw=1.0, ls="--")
            axt.text(t_anom, axt.get_ylim()[1], " anomaly\n onset", color=COL["NonPSPL"],
                     fontsize=6.3, va="top", ha="left")
        # --- bottom: probability evolution
        axb = axes[1, col]
        days, P = _prob_evolution(ev)
        for c in ["Flat", "PSPL", "NonPSPL"]:
            i = CLASS_NAMES.index(c)
            axb.plot(days, P[:, i], lw=1.6, color=COL[c], label=c)
        if t_anom is not None and ev.true_class == "NonPSPL":
            axb.axvline(t_anom, color=COL["NonPSPL"], lw=1.0, ls="--")
        axb.set_ylim(-0.02, 1.02)
        axb.set_xlabel("days of season revealed")
        if col == 0:
            axb.set_ylabel("class probability")
            axb.legend(loc="center left", frameon=False)
    fig.suptitle("The real-time cascade: Flat $\\rightarrow$ PSPL $\\rightarrow$ NonPSPL as evidence arrives",
                 fontsize=9.5, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cascade_evolution.pdf"))
    plt.close(fig)
    print("cascade_evolution.pdf")


if __name__ == "__main__":
    fig_gallery()
    fig_cascade()

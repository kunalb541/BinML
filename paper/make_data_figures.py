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
# per-band styling (by wavelength: F087 bluest, F213 reddest); colour bands are sparse (6-h)
BANDS = ["F146", "F087", "F213"]
BAND_COL = {"F146": "#333333", "F087": "#2166ac", "F213": "#b2182b"}
BAND_MS = {"F146": 1.6, "F087": 8.0, "F213": 8.0}
BAND_A = {"F146": 0.40, "F087": 0.85, "F213": 0.85}


def _plot_bands(ax, ev, fold_P=None, thin_f146=1400):
    """Scatter every present band onto ax (magnitude, inverted). Returns bands drawn."""
    drawn = []
    for b in BANDS:
        if b not in ev.bands or len(ev.bands[b].t) == 0:
            continue
        t, m = ev.bands[b].t, ev.bands[b].mag
        if fold_P:
            x = (t % fold_P) / fold_P
            step = max(1, len(t) // thin_f146) if b == "F146" else 1
            ax.scatter(x[::step], m[::step], s=BAND_MS[b], color=BAND_COL[b],
                       alpha=BAND_A[b], lw=0)
            ax.scatter(x[::step] + 1.0, m[::step], s=BAND_MS[b], color=BAND_COL[b],
                       alpha=BAND_A[b], lw=0)
        else:
            step = max(1, len(t) // thin_f146) if b == "F146" else 1
            ax.scatter(t[::step], m[::step], s=BAND_MS[b], color=BAND_COL[b],
                       alpha=BAND_A[b], lw=0, label=b)
        drawn.append(b)
    ax.invert_yaxis()
    return drawn
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
        # PeriodicVar: phase-fold so the (fast) variability is legible, not aliased to noise
        if c == "PeriodicVar" and ev.params.get("P"):
            P = float(ev.params["P"])
            _plot_bands(ax, ev, fold_P=P)
            ax.set_xlim(0, 2); folded.add(c)
            ax.text(0.5, 0.93, f"folded, $P={P:.1f}$ d", transform=ax.transAxes,
                    fontsize=6.5, ha="center", va="top", color="#555")
        else:
            _plot_bands(ax, ev)
        ax.set_title(f"{PRETTY[c]}", fontsize=8.5)
        ax.text(0.03, 0.06, f"model: {label} ({100*probs[label]:.0f}%)",
                transform=ax.transAxes, fontsize=6.8, va="bottom",
                color="black" if label == c else "#b00")
        ax.tick_params(labelsize=7)
        ax.margins(x=0.02)
    # one shared band legend (top-left panel)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=4, color=BAND_COL[b], label=b)
               for b in BANDS]
    axes[0, 0].legend(handles=handles, loc="upper right", frameon=False, fontsize=6.5,
                      handletextpad=0.2, borderpad=0.2)
    for ax, c in zip(axes[1], CLASS_NAMES[3:]):
        ax.set_xlabel("phase" if c in folded else "days since season start")
    for ax in axes[:, 0]:
        ax.set_ylabel("magnitude")
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
        t_anom = ev.params.get("t_anom")
        # --- top: light curve (all bands)
        axt = axes[0, col]
        _plot_bands(axt, ev, thin_f146=900)
        axt.set_title(ttl, fontsize=8.8)
        if col == 0:
            axt.set_ylabel("magnitude")
            hs = [plt.Line2D([], [], marker="o", ls="", ms=3.5, color=BAND_COL[b], label=b)
                  for b in BANDS]
            axt.legend(handles=hs, loc="upper left", frameon=False, fontsize=6,
                       handletextpad=0.2, borderpad=0.15)
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


# ------------------------------------------------------ detection latency (censor-aware)
def fig_latency(n_target=200, n_steps=72):
    """Censor-aware detection latency: EVERY eligible detectable binary is kept.

    The earlier version sampled until it had a fixed number of DETECTED binaries and skipped
    late-onset events, which is survivor bias with hidden right-censoring (an audit finding).
    Here every binary whose anomaly is detectable enters the sample; those never flagged within
    the season are recorded as right-censored and reported as a non-detection fraction rather
    than dropped. The reveal grid is 1 d (was 2 d).
    """
    import json
    thr = json.load(open(os.path.join(HERE, "results", "metrics.json")))["headline"]["threshold"]
    lags, n_censored, seed = [], 0, 2000
    while len(lags) + n_censored < n_target and seed < 2000 + 8000:
        seed += 1
        ev = simulate_event("NonPSPL", np.random.default_rng(seed), CFG)
        if ev is None or ev.label != "NonPSPL":
            continue
        t_anom = ev.params.get("t_anom")
        if t_anom is None or not np.isfinite(t_anom):
            continue
        days, P = _prob_evolution(ev, n_steps=n_steps)
        pn = P[:, CLASS_NAMES.index("NonPSPL")]
        crossed = np.where(pn >= thr)[0]
        if not len(crossed):
            n_censored += 1          # never detected within the season: censored, NOT discarded
            continue
        lags.append(float(days[crossed[0]] - t_anom))
    lags = np.array(lags)
    n_total = len(lags) + n_censored
    det_frac = len(lags) / max(n_total, 1)
    med = float(np.median(lags[lags >= 0])) if (lags >= 0).any() else float("nan")
    pre_frac = float((lags < 0).mean()) if lags.size else 0.0

    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    bins = np.linspace(-20, 40, 25)
    ax.hist(lags[lags >= 0], bins=bins, color=COL["NonPSPL"], alpha=0.85,
            label="detected after onset")
    ax.hist(lags[lags < 0], bins=bins, color="#888", alpha=0.85,
            label=f"flagged before onset ({100*pre_frac:.0f}\% of detections)")
    ax.axvline(0, color="k", lw=1.0, ls="--")
    if np.isfinite(med):
        ax.axvline(med, color=COL["PSPL"], lw=1.2)
        ax.text(med + 1.2, ax.get_ylim()[1] * 0.55, f"median\n{med:.1f} d",
                color=COL["PSPL"], fontsize=7, va="center")
    ax.set_xlabel("days between caustic onset and BinML flag")
    ax.set_ylabel("number of binaries")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.set_title(f"Detection latency: {len(lags)}/{n_total} detected "
                 f"({100*det_frac:.0f}\%), {n_censored} censored", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "detection_latency.pdf"))
    plt.close(fig)
    stats = {"n_eligible": n_total, "n_detected": int(len(lags)), "n_censored": int(n_censored),
             "detection_fraction": round(det_frac, 3), "median_lag_detected_days": round(med, 2),
             "pre_onset_fraction_of_detections": round(pre_frac, 3)}
    with open(os.path.join(HERE, "outputs", "latency_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"detection_latency.pdf  {stats}")


# ---------------------------------------------- probability evolution for ALL six classes
def fig_prob_all(n_steps=30):
    """One panel per class: how the model's six class probabilities evolve as the season is
    revealed, for a representative correctly-identified event of that class."""
    seeds = range(1, 500)
    want = {"Flat": None, "PSPL": 0.5, "NonPSPL": 0.7, "PeriodicVar": 0.4,
            "LongPeriodVar": 0.3, "Eruptive": 0.6}
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.4), sharey=True)
    for ax, c in zip(axes.ravel(), CLASS_NAMES):
        ev, probs, label = find_example(c, seeds, want_amp=want[c])
        if ev is None:
            ev, probs, label = find_example(c, seeds, want_amp=None)
        days, P = _prob_evolution(ev, n_steps=n_steps)
        for k in CLASS_NAMES:
            i = CLASS_NAMES.index(k)
            is_true = (k == c)
            ax.plot(days, P[:, i], lw=2.0 if is_true else 0.9,
                    color=COL[k], alpha=1.0 if is_true else 0.5,
                    zorder=3 if is_true else 1, label=k)
        ax.set_title(PRETTY[c], fontsize=8.5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, CFG.window_days)
        ax.tick_params(labelsize=7)
    for ax in axes[1]:
        ax.set_xlabel("days of season revealed")
    for ax in axes[:, 0]:
        ax.set_ylabel("class probability")
    handles = [plt.Line2D([], [], color=COL[k], lw=2, label=k) for k in CLASS_NAMES]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=7.2,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Class probability as the season is revealed, one representative event per class",
                 fontsize=9.5, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "prob_evolution_all.pdf"))
    plt.close(fig)
    print("prob_evolution_all.pdf")


if __name__ == "__main__":
    fig_gallery()
    fig_cascade()
    fig_latency()
    fig_prob_all()

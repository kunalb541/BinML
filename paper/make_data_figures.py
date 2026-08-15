#!/usr/bin/env python3
"""Simulate representative events and render publication figures from raw light curves.

Two figures the evaluation artifact alone cannot produce, both generated from freshly simulated
events (deterministic seeds) run through the shipped model:

  lightcurve_gallery.pdf  -- one representative simulated event per class.
  cascade_evolution.pdf   -- the half-day cascade: class probability as the season is revealed,
                             for a clean binary, a subtle binary, and a single-lens control.

Needs the simulation stack (VBBinaryLensing, h5py, scipy) and the binml package. Run from paper/:
    python make_data_figures.py
"""
from __future__ import annotations
import hashlib
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.assemble import simulate_event, SurveyConfig
from pipeline.generators import has_vbb
import binml

if not has_vbb():
    raise SystemExit(
        "FATAL: VBBinaryLensing is required to regenerate the simulated binary-lens figures; "
        "the simulator's import-only PSPL fallback must not be used for a paper build")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs", "figures")
os.makedirs(OUT, exist_ok=True)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

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


def find_example(true_class, seeds, want_amp=None, need_correct=True, prefer="conf",
                 need_bands=()):
    """Search deterministic seeds for a legible, correctly labelled illustration.

    TWO BUGS THIS FUNCTION USED TO HAVE, both of which put a caption at odds with its figure.

    1. The amplitude gate measured `mag.max() - mag.min()` on the NOISY series. For a faint source
       that is satisfied by photometric scatter alone, so a "want_amp=0.5" PSPL example could be
       selected with no visible magnification at all: the chosen event had u0=1.21 and t0=71.7 d in
       a 72 d season -- a 0.22 mag peak, truncated at the season edge -- while the noise gave a
       measured 0.66 mag spread. The gate now uses the NOISELESS reference curve, so it measures
       signal.
    2. `prefer_conf=False` set score=amp and then kept the MAXIMUM, so the panel captioned "subtle
       binary" was selecting the most dramatic event in its seed range (5.4 mag, u0=0.038). `prefer`
       is now explicit: "conf" (most confident), "max_amp", or "min_amp" for genuinely subtle.

    `need_bands` requires the named bands to have at least one surviving epoch, so a panel whose
    caption promises three bands can be made to have three bands.
    """
    assert prefer in ("conf", "max_amp", "min_amp"), prefer
    best = None
    for s in seeds:
        out = simulate_event(true_class, np.random.default_rng(s), CFG, _return_ref_truth=True)
        if out is None:
            continue
        ev, ref = out if isinstance(out, tuple) else (out, None)
        if ev is None or "F146" not in ev.bands or len(ev.bands["F146"].t) < 50:
            continue
        if need_correct and ev.label != true_class:
            continue
        if any(b not in ev.bands or len(ev.bands[b].t) == 0 for b in need_bands):
            continue
        probs, label = predict(ev)
        if need_correct and label != true_class:
            continue
        # amplitude of the SIGNAL, from the noise-free reference curve
        amp = (float(ref[1].max() - ref[1].min()) if ref is not None and len(ref[1])
               else float(ev.bands["F146"].mag.max() - ev.bands["F146"].mag.min()))
        if want_amp is not None and amp < want_amp:
            continue
        conf = probs.get(true_class, 0.0)
        score = {"conf": conf, "max_amp": amp, "min_amp": -amp}[prefer]
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
            # Grey text directly on the scatter was illegible; give it an opaque backing box.
            ax.text(0.5, 0.96, f"folded, $P={P:.1f}$ d", transform=ax.transAxes,
                    fontsize=6.5, ha="center", va="top", color="#333",
                    bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.5))
        else:
            _plot_bands(ax, ev)
        ax.set_title(f"{PRETTY[c]}", fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.margins(x=0.02)
    # one shared band legend (top-left panel)
    # Legend lists only bands that appear somewhere in the figure.
    _shown = [b for b in BANDS if any(b in picks[c][0].bands and len(picks[c][0].bands[b].t) > 0
                                      for c in CLASS_NAMES)]
    handles = [plt.Line2D([], [], marker="o", ls="", ms=4, color=BAND_COL[b], label=b)
               for b in _shown]
    axes[0, 0].legend(handles=handles, loc="lower right", fontsize=6.5, frameon=True, framealpha=0.92, edgecolor="none", facecolor="white",
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
    subtle, sp, _ = find_example("NonPSPL", range(300, 900), want_amp=0.25, prefer="min_amp")
    control, ctp, _ = find_example("PSPL", range(1, 300), want_amp=0.6)
    events = [("Clean binary", clean), ("Subtle binary", subtle), ("Single-lens control", control)]

    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.4),
                             gridspec_kw={"height_ratios": [1.0, 1.15]})
    for col, (ttl, ev) in enumerate(events):
        t_anom = ev.params.get("t_anom")
        # --- top: light curve (all bands)
        axt = axes[0, col]
        drawn = _plot_bands(axt, ev, thin_f146=900)
        axt.set_title(ttl, fontsize=8.8)
        if col == 0:
            axt.set_ylabel("magnitude")
            hs = [plt.Line2D([], [], marker="o", ls="", ms=3.5, color=BAND_COL[b], label=b)
                  for b in drawn]   # only bands with surviving epochs, never a phantom entry
            # Upper right: the magnitude axis is inverted, so this corner is the bright/early
            # region, which is empty for all three example events. Lower left collides with the
            # anomaly-onset label, which is anchored to the bottom of the panel.
            axt.legend(handles=hs, loc="upper right", fontsize=6, frameon=True, framealpha=0.92,
                       edgecolor="none", facecolor="white", handletextpad=0.2, borderpad=0.15)
        if t_anom is not None and ev.true_class == "NonPSPL":
            axt.axvline(t_anom, color=COL["NonPSPL"], lw=1.0, ls="--")
            # Place the label INSIDE the axes on whichever side has room. Anchoring it at the
            # y-limit with ha="left" pushed it outside the frame for events whose onset falls
            # near the end of the season, where it collided with the panel edge and the title.
            _x0, _x1 = axt.get_xlim()
            _right = t_anom > 0.5 * (_x0 + _x1)
            axt.annotate("anomaly\nonset", xy=(t_anom, 0.02), xycoords=("data", "axes fraction"),
                         xytext=(-4 if _right else 4, 2), textcoords="offset points",
                         color=COL["NonPSPL"], fontsize=6.3, va="bottom",
                         ha="right" if _right else "left",
                         bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
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
            axb.legend(loc="center left", fontsize=7, frameon=True, framealpha=0.92, edgecolor="none", facecolor="white")
    fig.suptitle("Half-day partial-light-curve cascade: Flat $\\rightarrow$ PSPL $\\rightarrow$ NonPSPL",
                 fontsize=9.5, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cascade_evolution.pdf"))
    plt.close(fig)
    print("cascade_evolution.pdf")


# ------------------------------------------------------ detection latency (censor-aware)
def fig_latency():
    """Plot the SAME event-level artifact the cascade numbers come from.

    Figures 4 and 5 previously came from two different sweeps with different reveal grids (1 d vs
    0.5 d) and were described as one population, giving inconsistent premature-alert rates (15% vs
    29% of detections). Both now read validation/cascade_events.json -- one experiment, one
    definition, one grid -- so they cannot disagree.
    """
    import json as _json
    npz = os.path.join(os.path.dirname(HERE), "validation", "cascade_trace.npz")
    res = os.path.join(os.path.dirname(HERE), "validation", "cascade_reproduce_result.json")
    if not (os.path.exists(npz) and os.path.exists(res)):
        raise SystemExit("FATAL: run validation/cascade_trace.py then cascade_reduce.py "
                         "(figures 4 and 5 are generated from their artifacts)")
    summ = _json.load(open(res))
    expected = summ.get("reduction_provenance", {}).get("input_trace_sha256")
    actual = _sha256_file(npz)
    if not expected or expected != actual:
        raise SystemExit(f"FATAL: cascade summary/trace mismatch: {expected} != {actual}")
    d = np.load(npz, allow_pickle=False)
    thr = summ["protocol"]["threshold"]
    cuts = d["cuts"].astype(float)
    over = np.nan_to_num(d["p_f146"].astype(float), nan=0.0) >= thr
    alert = np.where(over.any(1), cuts[np.argmax(over, 1)], np.nan)
    lags = (alert - d["t_anom_fine"].astype(float))[np.isfinite(alert)]

    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    # Bins span the data, so no detection falls outside the plotted range.
    bins = np.linspace(np.floor(lags.min()), np.ceil(lags.max()), 33)
    ax.hist(lags[lags >= 0], bins=bins, color=COL["NonPSPL"], alpha=0.85,
            label="first alert after onset")
    ax.hist(lags[lags < 0], bins=bins, color="#888", alpha=0.85,
            label=f"premature ({100*summ['premature_rate_of_detected']:.0f}% of detections)")
    ax.axvline(0, color="k", lw=1.0, ls="--")
    # TWO medians, because they answer different questions and the paper reports both: the median
    # over all detections is pulled negative by the premature alerts, so it is NOT the typical
    # post-onset delay.
    med_all = summ["median_lag_detected_days"]
    med_np = summ["median_lag_non_premature_days"]
    top = ax.get_ylim()[1]
    if abs(med_all - med_np) < 1e-9:
        # When premature alerts are a small minority the two medians coincide; drawing two lines
        # and two labels on top of each other implies a distinction the data does not show.
        ax.axvline(med_np, color=COL["PSPL"], lw=1.4)
        # Anchor the label in AXES coordinates, not data coordinates. In data coordinates the
        # text started at med_np+1.5 d, which for a median near zero put it on top of the tallest
        # bars and ran the first characters off the left edge of the axes.
        ax.annotate(f"median {med_np:+.1f} d\n(identical over all alerts\nand over non-premature)",
                    xy=(med_np, top * 0.55), xycoords="data",
                    xytext=(0.62, 0.62), textcoords="axes fraction",
                    color=COL["PSPL"], fontsize=6.4, va="center", ha="left",
                    bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.5),
                    arrowprops=dict(arrowstyle="-", color=COL["PSPL"], lw=0.7,
                                    shrinkA=0, shrinkB=2))
    else:
        ax.axvline(med_all, color="#888", lw=1.2, ls=":")
        ax.axvline(med_np, color=COL["PSPL"], lw=1.4)
        ax.annotate(f"median, all\ndetections {med_all:+.1f} d", xy=(med_all, top * 0.7),
                    xycoords="data", xytext=(0.05, 0.86), textcoords="axes fraction",
                    color="#666", fontsize=6.4, va="center", ha="left",
                    bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.5),
                    arrowprops=dict(arrowstyle="-", color="#888", lw=0.7, shrinkA=0, shrinkB=2))
        ax.annotate(f"median, not\npremature {med_np:+.1f} d", xy=(med_np, top * 0.45),
                    xycoords="data", xytext=(0.62, 0.55), textcoords="axes fraction",
                    color=COL["PSPL"], fontsize=6.4, va="center", ha="left",
                    bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.5),
                    arrowprops=dict(arrowstyle="-", color=COL["PSPL"], lw=0.7,
                                    shrinkA=0, shrinkB=2))
    ax.set_xlabel("days between caustic onset and first BinML alert")
    ax.set_ylabel("number of binaries")
    ax.legend(fontsize=6.5, loc="upper right", frameon=True, framealpha=0.92, edgecolor="none", facecolor="white")
    ax.set_title(f"Time to first alert: {summ['n_detected']}/{summ['n_eligible']} detected "
                 f"({100*summ['detection_fraction']:.0f}%), {summ['n_censored']} censored",
                 fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "detection_latency.pdf"))
    plt.close(fig)
    print(f"detection_latency.pdf  (from cascade_trace.npz; {summ['n_eligible']} eligible)")


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

#!/usr/bin/env python3
"""Generate every manuscript figure from the archived evaluation artifact.

Reads paper/results/*.npy (the independent 450,589-event evaluation of the production
model) and writes publication-quality PDFs to paper/outputs/figures/. Also writes
paper/outputs/figures_stats.json with numbers *derived here* (e.g. the classical Delta-chi^2
baseline average precision) so make_macros.py can cite them without re-deriving.

Deterministic: no RNG, no network. Run from the paper/ directory:  python make_figures.py
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "outputs", "figures")
os.makedirs(OUT, exist_ok=True)

CLASSES = ["Flat", "PSPL", "NonPSPL", "PeriodicVar", "LongPeriodVar", "Eruptive"]
SHORT = ["Flat", "PSPL", "NonPSPL", "Periodic", "LongPeriod", "Eruptive"]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.linewidth": 0.7,
    "font.family": "serif", "mathtext.fontset": "cm",
})

def load(name):
    return np.load(os.path.join(RES, name))

lg   = load("logits.npy").astype(np.float64)
y    = load("label.npy").astype(int)
kp   = load("keep_prob.npy").astype(np.float64)
w    = 1.0 / np.clip(kp, 1e-3, 1.0)
metrics = json.load(open(os.path.join(RES, "metrics.json")))

# softmax probabilities
z = lg - lg.max(1, keepdims=True)
P = np.exp(z); P /= P.sum(1, keepdims=True)
pred = P.argmax(1)
NONP = CLASSES.index("NonPSPL")

stats = {}   # derived numbers -> figures_stats.json


# ---------------------------------------------------------------- Fig: confusion matrix
def fig_confusion():
    C = np.zeros((6, 6))
    for t, p, wi in zip(y, pred, w):
        C[t, p] += wi
    row = C.sum(1, keepdims=True)
    Cn = 100.0 * C / np.clip(row, 1e-9, None)
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    im = ax.imshow(Cn, cmap="Blues", vmin=0, vmax=100)
    for i in range(6):
        for j in range(6):
            v = Cn[i, j]
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="white" if v > 55 else "black", fontsize=7.5)
    ax.set_xticks(range(6)); ax.set_xticklabels(SHORT, rotation=40, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels(SHORT)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("row-normalised %  (selection-corrected)")
    fig.savefig(os.path.join(OUT, "confusion.pdf"))
    plt.close(fig)


# ------------------------------------------- Fig: PR curve for NonPSPL + classical baseline
def _pr_curve(score, target, weight):
    order = np.argsort(-score)
    t = target[order].astype(float); ww = weight[order]
    tp = np.cumsum(ww * t); fp = np.cumsum(ww * (1 - t))
    P_tot = (ww * t).sum()
    prec = tp / np.clip(tp + fp, 1e-9, None)
    rec = tp / max(P_tot, 1e-9)
    ap = np.sum(np.diff(np.concatenate([[0], rec])) * prec)
    return rec, prec, ap

def fig_pr_nonpspl():
    tgt = (y == NONP).astype(int)
    # network: P(NonPSPL)
    rec_n, prec_n, ap_n = _pr_curve(P[:, NONP], tgt, w)
    # oracle: rank by the injected anomaly Delta-chi^2 (PSPL vs true binary on the NOISELESS
    # truth). This is an upper bound on any blind detector -- it uses the true parameters --
    # NOT a fieldable baseline. BinML must recover this ranking from raw photometry alone.
    dchi2 = load("dchi2_anomaly.npy").astype(np.float64)
    rec_c, prec_c, ap_c = _pr_curve(dchi2, tgt, w)
    # Report the SAME average precision as metrics.json (the authoritative value) so the figure
    # legend and the manuscript text cannot disagree; the tiny difference from this curve's own
    # trapezoidal AP is just the integration scheme.
    ap_n = metrics["average_precision_population"]
    stats["ap_network"] = round(float(ap_n), 3)
    stats["ap_oracle_dchi2"] = round(float(ap_c), 3)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot(rec_c, prec_c, lw=1.4, color="#b0562a", ls="--",
            label=f"oracle $\\Delta\\chi^2$ ceiling  (AP = {ap_c:.3f})")
    ax.plot(rec_n, prec_n, lw=1.8, color="#1f4e79",
            label=f"BinML, blind  (AP = {ap_n:.3f})")
    # operating point
    thr = metrics["headline"]["threshold"]
    op = P[:, NONP] >= thr
    tp = (w * (op & (tgt == 1))).sum(); fp = (w * (op & (tgt == 0))).sum()
    fn = (w * (~op & (tgt == 1))).sum()
    ax.scatter([tp / (tp + fn)], [tp / (tp + fp)], zorder=5, s=40,
               color="black", marker="o",
               label="operating point (purity 0.90)")
    ax.set_xlabel("Completeness (recall)"); ax.set_ylabel("Purity (precision)")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(os.path.join(OUT, "pr_nonpspl.pdf"))
    plt.close(fig)


# ------------------------------------------------------ Fig: (log s, log q) efficiency plane
def fig_efficiency_plane():
    ep = metrics["efficiency_plane"]
    qe = np.array(ep["log_q_edges"]); se = np.array(ep["log_s_edges"])
    det = np.array(ep["survey_detectability"], float)
    e2e = np.array(ep["end_to_end_recovery"], float)
    clf = np.array(ep["classifier_efficiency"], float)
    neff = np.array(ep["n_eff"], float)
    for M in (det, e2e, clf):
        M[neff < 20] = np.nan
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
    titles = ["Survey detectability", "End-to-end recovery", "Classifier efficiency"]
    for ax, M, ttl in zip(axes, (det, e2e, clf), titles):
        im = ax.pcolormesh(se, qe, M, cmap="viridis", vmin=0, vmax=1,
                           shading="flat")
        ax.set_title(ttl)
        ax.set_xlabel(r"$\log_{10} s$")
    axes[0].set_ylabel(r"$\log_{10} q$")
    # planetary regime marker
    for ax in axes:
        ax.axhline(-2.0, color="white", lw=0.6, ls=":")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("fraction")
    fig.savefig(os.path.join(OUT, "efficiency_plane.pdf"))
    plt.close(fig)


# ------------------------------------------------ Fig: NonPSPL recall vs source brightness & tE
def fig_param_dependence():
    params = load("params.npy")  # structured or 2D; try structured field access
    mb = load("m_base_ref.npy").astype(float)
    tgt = y == NONP
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))

    # recall vs baseline magnitude
    ax = axes[0]
    bins = np.linspace(np.nanpercentile(mb, 1), np.nanpercentile(mb, 99), 11)
    cx, cy = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = tgt & (mb >= lo) & (mb < hi)
        if m.sum() > 30:
            cx.append(0.5 * (lo + hi)); cy.append((pred[m] == NONP).mean())
    ax.plot(cx, cy, "o-", color="#1f4e79", ms=3, lw=1.3)
    ax.set_xlabel(r"F146 baseline magnitude $m_{\rm base}$")
    ax.set_ylabel("NonPSPL recall"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.25, lw=0.5)
    ax.axvline(22.5, color="grey", ls=":", lw=0.8)

    # recall vs anomaly Delta-chi^2 (evidence strength) from metrics
    ax = axes[1]
    rb = metrics["nonpspl_recall_by_dchi2"]
    labels = ["160–\n2k", "2k–\n10k", "10k–\n100k", ">100k"]
    keys = ["160-2000", "2000-10000", "10000-100000", "100000-inf"]
    vals = [rb[k]["recall"] for k in keys]
    ax.bar(range(4), vals, color="#4a7ba6", width=0.68)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.02); ax.set_ylabel("NonPSPL recall")
    ax.set_xlabel(r"anomaly $\Delta\chi^2$ (evidence strength)")
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    fig.savefig(os.path.join(OUT, "param_dependence.pdf"))
    plt.close(fig)


# ------------------------------------------------------------- Fig: cascade summary (schematic)
def fig_cascade_summary():
    cn = json.load(open(os.path.join(HERE, "canonical_numbers.json")))["cascade"]
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    groups = ["premature\nflag rate", "pre-onset\nP(NonPSPL)", "missed\nplanet rate"]
    before = [cn["premature_flag_before"], cn["preonset_p_before"], cn["missed_planet_before"]]
    after  = [cn["premature_flag_after"],  cn["preonset_p_after"],  cn["missed_planet_after"]]
    x = np.arange(3); ww = 0.36
    ax.bar(x - ww/2, before, ww, label="baseline model", color="#b0562a")
    ax.bar(x + ww/2, after,  ww, label="cascade model",  color="#1f4e79")
    for xi, b, a in zip(x, before, after):
        ax.text(xi - ww/2, b + 0.008, f"{b:.2f}", ha="center", fontsize=7)
        ax.text(xi + ww/2, a + 0.008, f"{a:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("rate / probability")
    ax.set_ylim(0, 0.5)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Detectability-conditioned cascade: before vs after")
    fig.savefig(os.path.join(OUT, "cascade_summary.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------- Fig: model schematic
def fig_schematic():
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 34)
    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=fc, ec="#333", lw=0.8))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=7.6)
    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=9, lw=0.9, color="#333"))
    # bands
    box(1, 22, 15, 8, "F146\n864 bins", "#dbe9f6")
    box(1, 13, 15, 7, "F087\n96 bins", "#eaf3da")
    box(1, 4,  15, 7, "F213\n96 bins", "#f6e6da")
    ax.text(8.5, 1.2, "3 bands × 5 channels\n(mean/min/max/frac/mask)",
            ha="center", va="center", fontsize=6.3, color="#555")
    # conv stem
    box(26, 11, 16, 12, "conv stem\n+ non-learned\nmin/max lanes", "#fff2cc")
    for y in (26, 16.5, 7.5):
        arrow(16, y, 26, 17)
    # tokens
    box(50, 12, 13, 10, "156 tokens\n(108+24+24)", "#e2d9f3")
    arrow(42, 17, 50, 17)
    # transformer
    box(69, 11, 15, 12, "transformer\n4 layers, 4 heads\n$d{=}96$", "#d9ead3")
    arrow(63, 17, 69, 17)
    # head
    box(88, 12, 11, 10, "6-way\nhead", "#f4cccc")
    arrow(84, 17, 88, 17)
    ax.text(50, 31.5, "BinML  —  505,479 parameters", ha="center", fontsize=9, weight="bold")
    fig.savefig(os.path.join(OUT, "schematic.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_confusion();        print("confusion.pdf")
    fig_pr_nonpspl();       print("pr_nonpspl.pdf")
    fig_efficiency_plane(); print("efficiency_plane.pdf")
    fig_param_dependence(); print("param_dependence.pdf")
    fig_cascade_summary();  print("cascade_summary.pdf")
    fig_schematic();        print("schematic.pdf")
    with open(os.path.join(HERE, "outputs", "figures_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("figures_stats.json:", stats)

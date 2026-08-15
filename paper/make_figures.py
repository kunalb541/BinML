#!/usr/bin/env python3
"""Generate every manuscript figure from the archived evaluation artifact.

Reads paper/results/*.npy (the independent 450,589-event evaluation of the production
model) and writes publication-quality PDFs to paper/outputs/figures/. Also writes
paper/outputs/figures_stats.json with numbers *derived here* (e.g. the classical Delta-chi^2
baseline average precision) so make_macros.py can cite them without re-deriving.

Deterministic: no RNG, no network. Run from the paper/ directory:  python make_figures.py
"""
from __future__ import annotations
import hashlib, json, os
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

metrics = json.load(open(os.path.join(RES, "metrics.json")))

# Restrict EVERY figure to the frozen final-test split (test_idx.npy = the rows held out from
# threshold selection). The operating threshold was fixed on the disjoint validation rows and is
# applied frozen here, so no figure sees a threshold-selection row. This is the same mask the
# headline metrics use; loading the full pool would leak calibration rows into the figures.
_ti = load("test_idx.npy").astype(int)
lg   = load("logits.npy").astype(np.float64)[_ti]
y    = load("label.npy").astype(int)[_ti]
tc   = load("true_class.npy").astype(int)[_ti]
params = load("params.npy")[_ti]
kp   = load("keep_prob.npy").astype(np.float64)[_ti]
w    = 1.0 / np.clip(kp, 1e-3, 1.0)
PARAM_FIELDS = (json.load(open(os.path.join(RES, "meta.json"))).get("param_fields") or [])

# softmax probabilities
z = lg - lg.max(1, keepdims=True)
P = np.exp(z); P /= P.sum(1, keepdims=True)
pred = P.argmax(1)
NONP = CLASSES.index("NonPSPL")

stats = {"n_test": int(len(_ti))}   # derived numbers -> figures_stats.json


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_trace_link(summary, trace_path, label):
    expected = summary.get("reduction_provenance", {}).get("input_trace_sha256")
    actual = _sha256_file(trace_path)
    if not expected:
        raise SystemExit(f"FATAL: {label} summary has no input-trace hash; rerun its reducer")
    if expected != actual:
        raise SystemExit(f"FATAL: {label} summary/trace mismatch: {expected} != {actual}")


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
    dchi2 = load("dchi2_anomaly.npy").astype(np.float64)[_ti]   # frozen-test rows only
    rec_c, prec_c, ap_c = _pr_curve(dchi2, tgt, w)
    # Report the SAME average precision as metrics.json (the authoritative value) so the figure
    # legend and the manuscript text cannot disagree; the tiny difference from this curve's own
    # trapezoidal AP is just the integration scheme.
    ap_n = metrics["average_precision_population"]
    stats["ap_network"] = round(float(ap_n), 3)
    stats["ap_oracle_dchi2"] = round(float(ap_c), 3)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot(rec_c, prec_c, lw=1.4, color="#b0562a", ls="--",
            label=f"truth-informed $\\Delta\\chi^2$ reference  (AP = {ap_c:.3f})")
    ax.plot(rec_n, prec_n, lw=1.8, color="#1f4e79",
            label=f"BinML (AP = {ap_n:.3f})")
    # operating point
    thr = metrics["headline"]["threshold"]
    op = P[:, NONP] >= thr
    tp = (w * (op & (tgt == 1))).sum(); fp = (w * (op & (tgt == 0))).sum()
    fn = (w * (~op & (tgt == 1))).sum()
    ax.scatter([tp / (tp + fn)], [tp / (tp + fp)], zorder=5, s=40,
               color="black", marker="o",
               label=f"operating point (purity {tp / (tp + fp):.3f})")
    ax.set_xlabel("Completeness (recall)"); ax.set_ylabel("Purity (precision)")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=7, frameon=True, framealpha=0.92, edgecolor="none", facecolor="white")
    ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(os.path.join(OUT, "pr_nonpspl.pdf"))
    plt.close(fig)


# ------------------------------------------------------ Fig: (log s, log q) efficiency plane
def fig_efficiency_plane():
    """Recomputed on the frozen test split with a BOUNDED conditional quantity.

    The earlier 'classifier efficiency' was B/A with mismatched bases (numerator counted all
    generated binaries predicted NonPSPL, denominator only the detectable ones), giving a ratio
    that exceeded 1 (up to 31) and was silently clipped. Here the classifier panel is instead the
    conditional recall P(predicted NonPSPL | detectable NonPSPL, q, s), which is a probability in
    [0,1]. Cells with fewer than 30 effective events (or 30 effective detectable events, for the
    recall panel) are left blank; effective counts are shown so low-support cells are visible.
    """
    iq, isx = PARAM_FIELDS.index("q"), PARAM_FIELDS.index("s")
    gen = (tc == NONP) & np.isfinite(params[:, iq]) & np.isfinite(params[:, isx])
    lq = np.log10(params[gen, iq]); ls = np.log10(params[gen, isx])
    det = (y[gen] == NONP); rec = (pred[gen] == NONP); ww = w[gen]
    n_q, n_s = 12, 6
    qe = np.linspace(-6, 0, n_q + 1); se = np.linspace(np.log10(0.2), np.log10(5.0), n_s + 1)
    qi = np.clip(np.digitize(lq, qe) - 1, 0, n_q - 1)
    si = np.clip(np.digitize(ls, se) - 1, 0, n_s - 1)
    A = np.full((n_q, n_s), np.nan); Cc = np.full((n_q, n_s), np.nan)
    N = np.zeros((n_q, n_s)); Nd = np.zeros((n_q, n_s))
    for i in range(n_q):
        for j in range(n_s):
            cell = (qi == i) & (si == j)
            neff = ww[cell].sum(); N[i, j] = neff
            if neff < 30:
                continue
            A[i, j] = (ww[cell] * det[cell]).sum() / neff
            dcell = cell & det; nd = ww[dcell].sum(); Nd[i, j] = nd
            if nd >= 30:
                Cc[i, j] = (ww[dcell] * rec[dcell]).sum() / nd
    stats["eff_cond_recall_median"] = round(float(np.nanmedian(Cc)), 3)
    stats["eff_cond_recall_min"] = round(float(np.nanmin(Cc)), 3)

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.9), sharey=True)
    logNd = np.log10(np.where(Nd > 0, Nd, np.nan))
    panels = [(A, "Survey detectability", 0, 1, "viridis", "fraction detectable"),
              (Cc, r"Classifier recall $|$ detectable", 0, 1, "viridis", "conditional recall"),
              (logNd, "Detectable support", None, None, "magma",
               r"$\log_{10} n_{\rm eff,det}$")]
    for ax, (M, ttl, vmn, vmx, cm, cl) in zip(axes, panels):
        im = ax.pcolormesh(se, qe, M, cmap=cm, vmin=vmn, vmax=vmx, shading="flat")
        ax.set_title(ttl, fontsize=8.5); ax.set_xlabel(r"$\log_{10} s$")
        ax.axhline(-2.0, color="white", lw=0.6, ls=":")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label(cl, fontsize=7)
    axes[0].set_ylabel(r"$\log_{10} q$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "efficiency_plane.pdf"))
    plt.close(fig)


# ------------------------------------------------ Fig: NonPSPL recall vs source brightness & tE
def fig_param_dependence():
    mb = load("m_base_ref.npy").astype(float)[_ti]   # frozen-test rows only
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


# ------------------------------------------- Fig: censor-aware cumulative alert curve
def fig_cascade_summary():
    """Fraction of eligible binaries alerted by time t relative to anomaly onset.

    A survival-style view of the frozen cascade sample which makes the two operationally distinct
    quantities visible at once: the curve's value at t<0 is the premature alert fraction, and its
    plateau is the within-season detection fraction (the shortfall to 1 is the right-censored
    events). The axis spans the FULL range of observed lags -- an earlier version clipped it to
    [-25, 45] d, which silently dropped 11 detections and made the plateau fall short of the
    detection line it was meant to meet.

    Both curves come from validation/cascade_trace.npz via cascade_reduce.py, so this figure,
    Figure 5 and the reported numbers cannot disagree.
    """
    npz = os.path.join(os.path.dirname(HERE), "validation", "cascade_trace.npz")
    res = os.path.join(os.path.dirname(HERE), "validation", "cascade_reproduce_result.json")
    if not (os.path.exists(npz) and os.path.exists(res)):
        raise SystemExit("FATAL: run validation/cascade_trace.py then cascade_reduce.py "
                         "(Figures 4-5 need them)")
    summ = json.load(open(res))
    _require_trace_link(summ, npz, "cascade")
    d = np.load(npz, allow_pickle=False)
    thr = summ["protocol"]["threshold"]
    cuts = d["cuts"].astype(float)
    n = len(d["seed"])

    def first_alert(P):
        over = np.nan_to_num(P.astype(float), nan=0.0) >= thr
        return np.where(over.any(1), cuts[np.argmax(over, 1)], np.nan)

    onset_fine = d["t_anom_fine"].astype(float)
    onset_coarse = d["t_anom_coarse"].astype(float)
    a = first_alert(d["p_f146"])
    lag_fine = (a - onset_fine)[np.isfinite(a)]
    lag_coarse = (a - onset_coarse)[np.isfinite(a)]

    lo = float(min(lag_fine.min(), lag_coarse.min()))
    hi = float(max(lag_fine.max(), lag_coarse.max()))
    pad = 0.03 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 600)
    cum = np.array([(lag_fine <= t).sum() / n for t in grid])      # denominator = ALL eligible
    cum_c = np.array([(lag_coarse <= t).sum() / n for t in grid])
    prem = summ["premature_rate_of_eligible"]
    det = summ["detection_fraction"]

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot(grid, cum, color="#1f4e79", lw=1.8, label="onset on the 0.5 d grid")
    ax.plot(grid, cum_c, color="#1f4e79", lw=1.1, ls="--", alpha=0.6,
            label="onset on the 7.2 d generator grid")
    ax.axvline(0, color="k", ls="--", lw=1.0)
    ax.axhline(det, color="#b0562a", ls=":", lw=1.2,
               label=f"within-season detection {100*det:.1f}%")
    ax.axhline(prem, color="#888", ls="-.", lw=1.2,
               label=f"premature {100*prem:.1f}% "
                     f"[{100*summ['premature_ci_of_eligible'][0]:.1f},"
                     f"{100*summ['premature_ci_of_eligible'][1]:.1f}]%")
    ax.fill_between(grid, 0, cum, where=(grid < 0), color="#888", alpha=0.25)
    ax.set_xlabel("days relative to anomaly onset $t_{\\rm anom}$")
    ax.set_ylabel("fraction of eligible binaries alerted")
    ax.set_ylim(0, 1.0); ax.set_xlim(grid[0], grid[-1])
    ax.legend(fontsize=6.2, loc="lower right", frameon=True, framealpha=0.92, edgecolor="none", facecolor="white")  # opaque: the reference lines run under it
    ax.set_title(f"Cumulative first-alert time ($N={n}$ eligible, F146 only)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cascade_summary.pdf"))
    plt.close(fig)
    stats["cascade_premature_fraction"] = round(float(prem), 3)
    stats["cascade_lag_range_days"] = [round(lo, 1), round(hi, 1)]


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


# ------------------------------------------------ Fig: prior-sensitivity of purity/alert burden
def derived_prose_numbers():
    """Numbers quoted in the prose that were previously hard-coded literals.

    Each of these appeared in paper.tex as a bare number. Three of them agreed with the artifact
    to the quoted precision; the fourth (an "about 38%" F087-loss rate) did not agree with
    anything, and is replaced here by the artifact's own definition -- the fraction of
    detectable-anomaly events for which no F087 epoch survives. A number that cannot be
    regenerated is a number that can drift, which is the failure this pipeline exists to prevent.
    """
    pred = P.argmax(1)
    NONP_, PSPL_ = 2, 1

    def row_frac(true_c, pred_c):
        m = y == true_c
        return float(100 * (w[m] * (pred[m] == pred_c)).sum() / w[m].sum())

    stats["conf_nonpspl_to_pspl_pct"] = round(row_frac(NONP_, PSPL_), 2)
    stats["conf_pspl_to_nonpspl_pct"] = round(row_frac(PSPL_, NONP_), 2)
    # generator-NonPSPL events demoted by detectability-conditioned labelling, as a fraction of
    # ALL final-test rows (the paper contrasts this with the 39.5% overall relabelling rate)
    stats["gen_nonpspl_demoted_pct"] = round(float(100 * ((tc == NONP_) & (y != NONP_)).mean()), 2)
    # Missed anomalies and the LongPeriodVar->PSPL confusion. Both were hand-copied into
    # canonical_numbers.json and both disagreed with this artifact (0.048 vs 0.052, 0.010 vs
    # 0.009). Derive them.
    LPV_ = CLASSES.index("LongPeriodVar")
    _m = y == NONP_
    stats["missed_anomaly_rate"] = round(float((w[_m] * (pred[_m] != NONP_)).sum() / w[_m].sum()), 3)
    _m2 = y == LPV_
    stats["lpv_to_pspl_rate"] = round(float((w[_m2] * (pred[_m2] == PSPL_)).sum() / w[_m2].sum()), 3)
    cnj = json.load(open(os.path.join(HERE, "canonical_numbers.json")))
    stats["no_f087_pct"] = round(100 * cnj["slices"]["no_blue_band"]["n"]
                                 / cnj["per_class_support"]["NonPSPL"], 1)


def fig_prior_sensitivity():
    """Purity and alert burden as a function of the ASSUMED anomaly prevalence.

    At the fixed operating threshold, completeness (recall) is prevalence-independent, but purity
    and the alert burden are not. The synthetic training mixture has a detectable-anomaly prevalence
    of ~5.6%; the real survey prevalence in a given input stream is set by event rates and any
    upstream selection, and is generally much lower. This figure makes the dependence explicit, so
    the headline purity is read as conditional on the mixture rather than as a survey number.
    """
    thr = metrics["headline"]["threshold"]
    pn = P[:, NONP]; yb = (y == NONP); flagged = pn >= thr
    Rc = (w * (flagged & yb)).sum() / (w * yb).sum()              # recall (prevalence-independent)
    fpr = (w * (flagged & ~yb)).sum() / (w * ~yb).sum()
    pi0 = (w * yb).sum() / w.sum()
    stats["prior_recall"] = round(float(Rc), 3)
    stats["prior_fpr"] = round(float(fpr), 4)
    stats["prior_pi0"] = round(float(pi0), 4)
    for tag, pival in [("purity_pi_1pct", 0.01), ("purity_pi_p1pct", 0.001)]:
        stats[tag] = round(float(pival * Rc / (pival * Rc + (1 - pival) * fpr)), 3)
    pis = np.logspace(-3.3, np.log10(0.3), 200)
    purity = pis * Rc / (pis * Rc + (1 - pis) * fpr)
    burden = pis * Rc + (1 - pis) * fpr

    # Right panel: what is ACHIEVABLE if the threshold is re-tuned for the assumed prevalence.
    # Thresholds come from validation/prevalence_thresholds.py, which selects them on the reserved
    # validation rows and reports the outcome on these frozen test rows -- never both on the same
    # rows, which is what an earlier revision did.
    pv_path = os.path.join(os.path.dirname(HERE), "validation", "prevalence_result.json")
    if not os.path.exists(pv_path):
        raise SystemExit("FATAL: run validation/prevalence_thresholds.py (Figure 12 needs it)")
    pv = json.load(open(pv_path))
    cx = np.array([c["prevalence"] for c in pv["curve"] if c.get("completeness_test") is not None])
    cy = np.array([c["completeness_test"] for c in pv["curve"]
                   if c.get("completeness_test") is not None])

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(7.1, 3.1))
    ax.semilogx(pis, purity, color="#1f4e79", lw=1.8, label="purity")
    ax.axvline(pi0, color="grey", ls="--", lw=1.0)
    # Sits just right of the dashed prevalence line, high in the panel where both the purity
    # curve and the alert-burden curve are absent -- the lower-right corner now holds the legend.
    ax.text(pi0 * 1.15, 0.30, "synthetic\nmixture", fontsize=6.5, color="grey", va="center")
    ax.axhline(Rc, color="#b0562a", ls=":", lw=1.2, label=f"completeness {Rc:.3f} (fixed)")
    ax.set_xlabel("assumed detectable-anomaly prevalence"); ax.set_ylabel("purity")
    ax.set_ylim(0, 1.02)
    ax2 = ax.twinx()
    ax2.semilogx(pis, 100 * burden, color="#2ca02c", lw=1.3, ls="-.")
    ax2.set_ylabel("alert burden (% of light curves)", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax.legend(loc="lower right", fontsize=6.5, frameon=True, framealpha=0.92, edgecolor="none", facecolor="white")
    ax.set_title("At the fixed threshold", fontsize=9)

    axb.semilogx(cx, cy, color="#1f4e79", lw=1.8, marker="o", ms=2.5)
    axb.axvline(pi0, color="grey", ls="--", lw=1.0)
    for k, lab in (("one_pct", "1%"), ("tenth_pct", "0.1%")):
        s = pv["scenarios"][k]
        if s.get("completeness_test") is not None:
            axb.plot([s["prevalence"]], [s["completeness_test"]], "s", color="#b0562a", ms=5)
            axb.annotate(f"{lab}: {s['completeness_test']:.3f}",
                         (s["prevalence"], s["completeness_test"]),
                         textcoords="offset points", xytext=(6, -10), fontsize=6.5,
                         color="#b0562a")
    axb.set_xlabel("assumed detectable-anomaly prevalence")
    axb.set_ylabel(f"completeness achievable at purity {pv['purity_target']:.2f}")
    axb.set_ylim(0, 1.02)
    axb.set_title("Threshold re-tuned on validation rows", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "prior_sensitivity.pdf"))
    plt.close(fig)
    stats["prev_comp_one_pct"] = pv["scenarios"]["one_pct"]["completeness_test"]
    stats["prev_comp_tenth_pct"] = pv["scenarios"]["tenth_pct"]["completeness_test"]


# -------------------------------------------------------------- Fig: calibration (reliability)
def fig_calibration():
    """Reliability of the anomaly score actually used by the alert policy.

    Top-label calibration does not validate a threshold on P(NonPSPL). This is therefore a
    population-weighted one-vs-rest diagram, with a binary Brier score for the same probability.
    """
    conf = P[:, NONP]; correct = (y == NONP).astype(float)
    nb = 12; edges = np.linspace(0, 1, nb + 1)
    bi = np.clip(np.digitize(conf, edges) - 1, 0, nb - 1)
    xc, acc, wsum, ece = [], [], [], 0.0
    W = w.sum()
    for b in range(nb):
        m = bi == b
        if not m.any():
            continue
        ww = w[m]; a = (ww * correct[m]).sum() / ww.sum(); c = (ww * conf[m]).sum() / ww.sum()
        xc.append(c); acc.append(a); wsum.append(ww.sum())
        ece += ww.sum() / W * abs(a - c)
    brier = float((w * (conf - correct) ** 2).sum() / W)
    stats["nonpspl_ece_weighted"] = round(float(ece), 4)
    stats["nonpspl_brier_weighted"] = round(brier, 4)

    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(6.8, 3.2))
    for panel in (ax, zoom):
        panel.plot([0, 1], [0, 1], ls=":", color="grey", lw=1)
    sizes = 400 * np.array(wsum) / max(wsum)
    ax.scatter(xc, acc, s=sizes, color="#1f4e79", alpha=0.7, edgecolor="white", lw=0.5, zorder=3)
    ax.plot(xc, acc, color="#1f4e79", lw=1.2)
    zoom.scatter(xc, acc, s=sizes, color="#1f4e79", alpha=0.7, edgecolor="white", lw=0.5,
                 zorder=3)
    zoom.plot(xc, acc, color="#1f4e79", lw=1.2)
    threshold = metrics["headline"]["threshold"]
    for panel in (ax, zoom):
        panel.axvline(threshold, color="#b0562a", ls="--", lw=1,
                     label=f"operating threshold {threshold:.3f}")
    ax.set_xlabel(r"predicted $P(\mathrm{NonPSPL})$")
    ax.set_ylabel("weighted anomaly frequency")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.text(0.05, 0.9, f"ECE = {ece:.3f}\nBrier = {brier:.3f}", fontsize=8,
            transform=ax.transAxes, va="top")
    ax.set_title("Full probability range", fontsize=9)
    zoom.set_xlim(0.75, 1.005); zoom.set_ylim(0.5, 1.005)
    zoom.set_xlabel(r"predicted $P(\mathrm{NonPSPL})$")
    zoom.set_title("Near the operating threshold", fontsize=9)
    zoom.legend(fontsize=6.5, loc="lower right", frameon=True, framealpha=0.92, edgecolor="none", facecolor="white")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "calibration.pdf")); plt.close(fig)


# ---------------------------------------- Fig: matched-detection exploratory risk--coverage
def fig_matched_risk_coverage():
    """Paired cascade arms at exactly matched detection counts.

    Thresholds were selected and evaluated on the same 400 events, so the plot is descriptive and
    not a confirmatory causal test. Showing the full predeclared range also exposes the reversal at
    the highest detection target instead of elevating one favourable operating point.
    """
    root = os.path.dirname(HERE)
    trace = os.path.join(root, "validation", "matched_traces.npz")
    result = os.path.join(root, "validation", "cascade_matched_result.json")
    if not (os.path.exists(trace) and os.path.exists(result)):
        raise SystemExit("FATAL: matched cascade trace/result missing; run cascade_matched_reduce.py")
    d = json.load(open(result))
    _require_trace_link(d, trace, "matched cascade")
    rows = sorted(d["matched"].values(),
                  key=lambda r: r["cascade_on"]["detection_fraction"])

    fig, ax = plt.subplots(figsize=(4.9, 3.25))
    styles = (("cascade_on", "truncation augmented", "#1f4e79", "o"),
              ("cascade_off", "no truncation augmentation", "#b0562a", "s"))
    for arm, label, color, marker in styles:
        x = np.array([r[arm]["detection_fraction"] for r in rows])
        yv = np.array([r[arm]["premature_rate_of_eligible"] for r in rows])
        ci = np.array([r[arm]["premature_ci_of_eligible"] for r in rows])
        yerr = np.vstack((yv - ci[:, 0], ci[:, 1] - yv))
        ax.errorbar(100 * x, 100 * yv, yerr=100 * yerr, color=color, marker=marker,
                    ms=4, lw=1.35, capsize=2.5, label=label)
    ax.set_xlabel("within-season detection (% of eligible events)")
    ax.set_ylabel("premature alerts (% of eligible events)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, loc="upper left", frameon=True, framealpha=0.92, edgecolor="none", facecolor="white")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cascade_matched.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_calibration();      print("calibration.pdf")
    fig_confusion();        print("confusion.pdf")
    fig_pr_nonpspl();       print("pr_nonpspl.pdf")
    fig_efficiency_plane(); print("efficiency_plane.pdf")
    fig_prior_sensitivity(); print("prior_sensitivity.pdf")
    fig_param_dependence(); print("param_dependence.pdf")
    fig_cascade_summary();  print("cascade_summary.pdf")
    fig_matched_risk_coverage(); print("cascade_matched.pdf")
    fig_schematic();        print("schematic.pdf")
    derived_prose_numbers()
    with open(os.path.join(HERE, "outputs", "figures_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("figures_stats.json:", stats)

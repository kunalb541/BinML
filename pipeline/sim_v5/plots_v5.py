"""
BinML v5 — the full diagnostic plot suite.

Ports every figure the v4 ``evaluate.py`` produced, adapted to 6 classes and 3 bands, and adds
the v5-specific ones (detectability conditioning, the (log s, log q) efficiency planes, the
population-vs-sample distinction, and the classical-baseline comparison).

Everything here reads the SAVED PREDICTION ARTIFACT, not the model. That is deliberate: once
``evaluate_v5`` has written per-event logits, every figure, slice, threshold and bootstrap is a
pure numpy operation, so the whole suite regenerates in seconds and none of it requires a GPU
or re-running the network.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np

from .classes import CLASS_NAMES, N_CLASSES

I_NON = CLASS_NAMES.index("NonPSPL")
COL = ["#7f7f7f", "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
        "savefig.facecolor": "white", "axes.grid": True, "grid.alpha": 0.25,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8,
    })
    return plt


def softmax(lg):
    z = lg - lg.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Preds:
    """The saved artifact, plus everything derived from it."""

    def __init__(self, d: str):
        self.d = d
        self.logits = np.load(f"{d}/logits.npy")
        self.p = softmax(self.logits)
        self.pred = self.logits.argmax(1)
        self.y = np.load(f"{d}/label.npy").astype(int)
        self.tc = np.load(f"{d}/true_class.npy").astype(int)
        self.kp = np.load(f"{d}/keep_prob.npy")
        # float64: see evaluate_v5.population_weights -- float32 cumsum overshoots 1.
        self.w = 1.0 / np.clip(self.kp.astype(np.float64), 1e-3, 1.0)
        self.dca = np.load(f"{d}/dchi2_anomaly.npy")
        self.dce = np.load(f"{d}/dchi2_event.npy")
        self.mb = np.load(f"{d}/m_base_ref.npy")
        self.aks = np.load(f"{d}/a_ks.npy")
        meta = json.load(open(f"{d}/meta.json"))
        self.pf = meta.get("param_fields")
        pp = f"{d}/params.npy"
        self.par = np.load(pp) if os.path.exists(pp) else None

    def col(self, name) -> Optional[np.ndarray]:
        if self.par is None or not self.pf or name not in self.pf:
            return None
        return self.par[:, self.pf.index(name)]


# ---------------------------------------------------------------------------------
def fig_confusion(P: Preds, out: str):
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, wt, title in ((axes[0], None, "sample (raw counts)"),
                          (axes[1], P.w, "population (1/keep_prob)")):
        k = P.y * N_CLASSES + P.pred
        cm = np.bincount(k, weights=wt, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
        rn = cm / np.maximum(cm.sum(1, keepdims=True), 1e-9)
        im = ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"Confusion — {title}"); ax.grid(False)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                if rn[i, j] > 0.004:
                    ax.text(j, i, f"{rn[i,j]:.3f}", ha="center", va="center", fontsize=7,
                            color="white" if rn[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Row-normalised confusion. NonPSPL→PSPL is a MISSED PLANET; "
                 "LongPeriodVar→PSPL is a FALSE microlensing detection.", fontsize=9)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_roc_pr(P: Preds, baseline: Optional[str], out: str):
    plt = _style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ROC one-vs-rest
    ax = axes[0]
    for c in range(N_CLASSES):
        pos = (P.y == c); s = P.p[:, c]
        o = np.argsort(-s); pw = P.w[o] * pos[o]; nw = P.w[o] * (~pos[o])
        # Prepend the origin. np.cumsum starts at the first sample, so the curve begins at
        # (fpr_1, tpr_1) rather than (0, 0) and the trapezoid integral over-counts -- which
        # showed up as an impossible AUC of 1.0005 for PeriodicVar.
        tpr = np.concatenate([[0.0], np.cumsum(pw) / max(pw.sum(), 1e-9)])
        fpr = np.concatenate([[0.0], np.cumsum(nw) / max(nw.sum(), 1e-9)])
        auc = float(np.trapezoid(tpr, fpr))
        ax.plot(fpr, tpr, color=COL[c], lw=1.4, label=f"{CLASS_NAMES[c]} ({auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC one-vs-rest (population-weighted, AUC)"); ax.legend(loc="lower right")

    # PR for NonPSPL, model vs baseline
    ax = axes[1]
    pos = (P.y == I_NON)
    o = np.argsort(-P.p[:, I_NON])
    tp = np.cumsum(P.w[o] * pos[o]); fp = np.cumsum(P.w[o] * (~pos[o]))
    prec = tp / np.maximum(tp + fp, 1e-9); rec = tp / max((P.w * pos).sum(), 1e-9)
    ap = float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))
    ax.plot(rec, prec, color="#d62728", lw=1.8, label=f"BinML v5 (AP={ap:.4f})")
    if baseline and os.path.exists(baseline):
        d = np.load(baseline)
        by = d["label"]; bw = 1.0 / np.clip(d["keep_prob"], 1e-3, 1.0)
        bs = np.where(d["dchi2_event"] > 500, d["dchi2_anomaly"], -np.inf)
        bo = np.argsort(-bs); bpos = (by == I_NON)
        btp = np.cumsum(bw[bo] * bpos[bo]); bfp = np.cumsum(bw[bo] * (~bpos[bo]))
        bp = btp / np.maximum(btp + bfp, 1e-9); br = btp / max((bw * bpos).sum(), 1e-9)
        bap = float(np.sum(np.diff(np.concatenate([[0], br])) * bp))
        ax.plot(br, bp, color="#1f77b4", lw=1.6, ls="--",
                label=f"classical $\\Delta\\chi^2$ (AP={bap:.4f})")
    ax.axhline(pos.mean(), color="k", ls=":", lw=0.8, label=f"prevalence {pos.mean():.3f}")
    ax.set_xlabel("completeness (recall)"); ax.set_ylabel("purity (precision)")
    ax.set_title("NonPSPL precision–recall: network vs classical"); ax.legend(loc="lower left")

    # operating point curve
    ax = axes[2]
    thr = np.linspace(0.02, 0.98, 60)
    comp, pur, deep = [], [], []
    q = P.col("q")
    dm = pos & np.isfinite(q) & (q < 1e-3) if q is not None else pos
    for t in thr:
        sel = P.p[:, I_NON] >= t
        tpw = (P.w * (sel & pos)).sum(); fpw = (P.w * (sel & ~pos)).sum()
        comp.append(tpw / max((P.w * pos).sum(), 1e-9))
        pur.append(tpw / max(tpw + fpw, 1e-9))
        deep.append((P.p[dm, I_NON] >= t).mean())
    ax.plot(thr, comp, label="completeness (all NonPSPL)", lw=1.6)
    ax.plot(thr, pur, label="purity", lw=1.6)
    ax.plot(thr, deep, label="deep planetary q<1e-3 recall", lw=1.6, ls="--")
    ax.axvline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xlabel("decision threshold on P(NonPSPL)"); ax.set_ylabel("value")
    ax.set_title("Operating point — argmax is only one choice"); ax.legend()
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_calibration(P: Preds, out: str):
    plt = _style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    conf = P.p.max(1); correct = (P.pred == P.y)

    ax = axes[0]
    edges = np.linspace(0, 1, 16); ece = 0.0
    xs, ys, ns = [], [], []
    for i in range(len(edges) - 1):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        if m.sum() < 20:
            continue
        xs.append(conf[m].mean()); ys.append(correct[m].mean()); ns.append(int(m.sum()))
        ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    ax.plot(xs, ys, "o-", color="#d62728", label=f"model (ECE={ece:.4f})")
    ax.set_xlabel("mean predicted confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_title("Calibration"); ax.legend()

    ax = axes[1]
    ax.hist(conf[correct], bins=40, alpha=0.65, label="correct", color="#2ca02c", density=True)
    ax.hist(conf[~correct], bins=40, alpha=0.65, label="wrong", color="#d62728", density=True)
    ax.set_xlabel("max softmax confidence"); ax.set_ylabel("density")
    ax.set_title("Confidence distribution"); ax.legend()

    ax = axes[2]
    for c in range(N_CLASSES):
        m = P.y == c
        if m.sum() > 50:
            ax.hist(P.p[m, I_NON], bins=40, histtype="step", lw=1.4,
                    color=COL[c], label=CLASS_NAMES[c], density=True)
    ax.set_yscale("log"); ax.set_xlabel("P(NonPSPL)"); ax.set_ylabel("density (log)")
    ax.set_title("Anomaly score by true class"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_per_class(P: Preds, out: str):
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, wt, title in ((axes[0], None, "sample"), (axes[1], P.w, "population")):
        k = P.y * N_CLASSES + P.pred
        cm = np.bincount(k, weights=wt, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
        rec = np.diag(cm) / np.maximum(cm.sum(1), 1e-9)
        pre = np.diag(cm) / np.maximum(cm.sum(0), 1e-9)
        f1 = 2 * rec * pre / np.maximum(rec + pre, 1e-9)
        x = np.arange(N_CLASSES); wdt = 0.27
        ax.bar(x - wdt, rec, wdt, label="recall", color="#1f77b4")
        ax.bar(x, pre, wdt, label="precision", color="#ff7f0e")
        ax.bar(x + wdt, f1, wdt, label="F1", color="#2ca02c")
        for i in range(N_CLASSES):
            ax.text(i, min(rec[i], pre[i]) - 0.06, f"{f1[i]:.3f}", ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
        ax.set_ylim(0, 1.05); ax.set_title(f"Per-class metrics — {title}"); ax.legend()
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_parameter_dependence(P: Preds, out: str):
    """NonPSPL recall as a function of every physical parameter that could drive it."""
    plt = _style()
    specs = [("q", "mass ratio q", True), ("s", "separation s", True),
             ("tE", "$t_E$ (days)", True), ("u0", "impact parameter $u_0$", False),
             (None, "baseline mag (F146)", False), (None, "extinction $A_{Ks}$", False),
             (None, "anomaly $\\Delta\\chi^2$", True), (None, "event $\\Delta\\chi^2$", True),
             ("rho", "source size $\\rho$", True)]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    pos = (P.y == I_NON); ok = (P.pred == I_NON)
    extras = {4: P.mb, 5: P.aks, 6: P.dca, 7: P.dce}
    for k, (name, label, logx) in enumerate(specs):
        ax = axes[k // 3][k % 3]
        v = P.col(name) if name else extras.get(k)
        if v is None:
            ax.text(0.5, 0.5, f"{label}: unavailable", ha="center", transform=ax.transAxes)
            ax.set_axis_off(); continue
        m = pos & np.isfinite(v) & (v > 0 if logx else np.isfinite(v))
        if m.sum() < 50:
            ax.text(0.5, 0.5, "too few", ha="center", transform=ax.transAxes); continue
        vv = np.log10(v[m]) if logx else v[m]
        edges = np.linspace(np.percentile(vv, 0.5), np.percentile(vv, 99.5), 13)
        cen, rr, nn = [], [], []
        for i in range(len(edges) - 1):
            sel = (vv >= edges[i]) & (vv < edges[i + 1])
            if sel.sum() < 25:
                continue
            cen.append(0.5 * (edges[i] + edges[i + 1]))
            rr.append(ok[m][sel].mean()); nn.append(int(sel.sum()))
        ax.plot(cen, rr, "o-", color="#d62728", lw=1.5)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(("log$_{10}$ " if logx else "") + label)
        ax.set_ylabel("NonPSPL recall")
        ax2 = ax.twinx(); ax2.bar(cen, nn, width=(edges[1] - edges[0]) * 0.85,
                                  alpha=0.15, color="grey"); ax2.set_yticks([])
        if name == "q":
            ax.axvline(np.log10(1.7e-4), color="b", ls=":", lw=1, label="Suzuki break")
            ax.axvline(-2, color="g", ls="--", lw=1, label="planetary q<1e-2"); ax.legend(fontsize=6)
        if name == "s":
            ax.axvline(0, color="b", ls=":", lw=1, label="resonant s=1"); ax.legend(fontsize=6)
    fig.suptitle("NonPSPL recall vs physical parameters (grey = event count per bin)", y=0.995)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_light_curves(P: Preds, cache: str, out: str, n_per: int = 2):
    """Example curves per class: correctly classified and, where they exist, failures."""
    plt = _style()
    from .model_v5 import BAND_BINS
    n = len(P.y)
    feats = {b: np.memmap(f"{cache}/feat_{b}.f16", dtype=np.float16, mode="r",
                          shape=(json.load(open(f"{cache}/meta.json"))["n_events"], L, 3))
             for b, L in BAND_BINS.items()}
    ti = np.load(f"{P.d}/test_idx.npy") if os.path.exists(f"{P.d}/test_idx.npy") else np.arange(n)
    rows = N_CLASSES; cols = n_per * 2
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.3 * rows))
    rng = np.random.default_rng(0)
    colours = {"F146": "#1f77b4", "F087": "#2ca02c", "F213": "#d62728"}
    for c in range(N_CLASSES):
        good = np.nonzero((P.y == c) & (P.pred == c))[0]
        bad = np.nonzero((P.y == c) & (P.pred != c))[0]
        picks = ([(i, True) for i in rng.choice(good, min(n_per, len(good)), replace=False)] +
                 [(i, False) for i in rng.choice(bad, min(n_per, len(bad)), replace=False)])
        for j in range(cols):
            ax = axes[c][j]
            if j >= len(picks):
                ax.set_axis_off(); continue
            i, okflag = picks[j]
            src = int(ti[i]) if len(ti) == n else int(i)
            for b in BAND_BINS:
                f = np.asarray(feats[b][src, :, 0], dtype=np.float32)
                t = np.linspace(0, 72, len(f), endpoint=False)
                v = np.isfinite(f)
                if v.any():
                    ax.plot(t[v], P.mb[i] + f[v], ".", ms=1.6, color=colours[b], label=b)
            ax.invert_yaxis()
            pr = CLASS_NAMES[P.pred[i]]
            ax.set_title(f"{CLASS_NAMES[c]} → {pr} (p={P.p[i,P.pred[i]]:.2f})",
                         fontsize=7, color="green" if okflag else "red")
            if c == 0 and j == 0:
                ax.legend(fontsize=6, markerscale=4)
            ax.tick_params(labelsize=6)
    fig.suptitle("Example light curves — green: correct, red: misclassified "
                 "(binned F146/F087/F213, magnitudes)", y=0.997)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_slices(P: Preds, out: str):
    plt = _style()
    q = P.col("q"); pos = (P.y == I_NON); ok = (P.pred == I_NON)
    sl = {}
    if q is not None:
        sl["planetary\nq<1e-2"] = pos & np.isfinite(q) & (q < 1e-2)
        sl["deep planetary\nq<1e-3"] = pos & np.isfinite(q) & (q < 1e-3)
        sl["Suzuki break\n1e-5<q<1e-3"] = pos & np.isfinite(q) & (q > 1e-5) & (q < 1e-3)
    sl["faint\nm>22.5"] = pos & (P.mb > 22.5)
    sl["bright\nm<21.5"] = pos & (P.mb < 21.5)
    sl["high ext\nA_Ks>0.7"] = pos & (P.aks > 0.7)
    sl["near threshold\n160<dchi2<2000"] = pos & (P.dca >= 160) & (P.dca < 2000)
    sl["strong\ndchi2>1e4"] = pos & (P.dca >= 1e4)
    sl["ALL NonPSPL"] = pos
    fig, ax = plt.subplots(figsize=(13, 5))
    names = list(sl); vals = [ok[sl[k]].mean() if sl[k].sum() else 0 for k in names]
    ns = [int(sl[k].sum()) for k in names]
    bars = ax.bar(range(len(names)), vals,
                  color=["#d62728" if v < 0.9 else "#2ca02c" for v in vals])
    for i, (v, nn) in enumerate(zip(vals, ns)):
        ax.text(i, v + 0.012, f"{v:.3f}\nn={nn:,}", ha="center", fontsize=7)
    ax.axhline(ok[pos].mean(), color="k", ls="--", lw=1,
               label=f"overall {ok[pos].mean():.3f}")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=7)
    ax.set_ylim(0, 1.12); ax.set_ylabel("NonPSPL recall")
    ax.set_title("Recall by stratified slice — red marks where the model is weakest")
    ax.legend()
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_failure_analysis(P: Preds, out: str):
    """Where do the errors come from, and how many are label noise rather than mistakes?"""
    plt = _style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    pos = (P.y == I_NON); fp = (P.pred == I_NON) & ~pos; fn = pos & (P.pred != I_NON)

    ax = axes[0]
    byp = fp & (P.tc == I_NON)
    parts = [int(byp.sum()), int((fp & (P.tc != I_NON)).sum())]
    ax.pie(parts, labels=[f"genuine binary,\nanomaly below cut\n({parts[0]:,})",
                          f"true error\n({parts[1]:,})"],
           colors=["#ff7f0e", "#d62728"], autopct="%1.1f%%", startangle=90)
    ax.set_title("False positives: label noise vs real error")

    ax = axes[1]
    lab = [CLASS_NAMES[c] for c in range(N_CLASSES) if c != I_NON]
    cnt = [int((fn & (P.pred == c)).sum()) for c in range(N_CLASSES) if c != I_NON]
    ax.bar(lab, cnt, color="#d62728")
    for i, v in enumerate(cnt):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("count"); ax.set_title("Missed NonPSPL — predicted as what?")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[2]
    b = np.array([0, 160, 500, 2e3, 1e4, 1e5, 1e12])
    cen = np.arange(len(b) - 1)
    rec, nn = [], []
    for i in range(len(b) - 1):
        m = pos & (P.dca >= b[i]) & (P.dca < b[i + 1])
        rec.append((P.pred[m] == I_NON).mean() if m.sum() > 10 else np.nan)
        nn.append(int(m.sum()))
    ax.bar(cen, rec, color="#1f77b4")
    for i, (v, k) in enumerate(zip(rec, nn)):
        if np.isfinite(v):
            ax.text(i, v + 0.015, f"{v:.3f}\nn={k:,}", ha="center", fontsize=7)
    ax.set_xticks(cen)
    ax.set_xticklabels([f"{b[i]:.0f}–{b[i+1]:.0f}" for i in range(len(b) - 1)],
                       rotation=35, ha="right", fontsize=7)
    ax.set_ylim(0, 1.12); ax.set_ylabel("recall")
    ax.set_title("Recall vs anomaly $\\Delta\\chi^2$ (detectability)")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_efficiency_planes(P: Preds, out: str, n_q: int = 12, n_s: int = 6):
    """Three-panel (log s, log q) detection efficiency — the canonical microlensing figure."""
    plt = _style()
    q = P.col("q"); s = P.col("s")
    if q is None or s is None:
        return
    gen = (P.tc == I_NON) & np.isfinite(q) & np.isfinite(s)
    lq = np.log10(q[gen]); ls = np.log10(s[gen])
    det = (P.y[gen] == I_NON).astype(float)
    rec = (P.pred[gen] == I_NON).astype(float)
    ww = P.w[gen]
    qe = np.linspace(-6, 0, n_q + 1); se = np.linspace(np.log10(0.2), np.log10(5.0), n_s + 1)
    qi = np.clip(np.digitize(lq, qe) - 1, 0, n_q - 1)
    si = np.clip(np.digitize(ls, se) - 1, 0, n_s - 1)
    A = np.full((n_q, n_s), np.nan); B = np.full((n_q, n_s), np.nan)
    C = np.full((n_q, n_s), np.nan); N = np.zeros((n_q, n_s))
    for i in range(n_q):
        for j in range(n_s):
            m = (qi == i) & (si == j); neff = ww[m].sum(); N[i, j] = neff
            if neff < 30:
                continue
            A[i, j] = (det[m] * ww[m]).sum() / neff
            B[i, j] = (rec[m] * ww[m]).sum() / neff
            if A[i, j] > 0.02:
                C[i, j] = min(B[i, j] / A[i, j], 1.5)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    for ax, M, ttl, vmax in ((axes[0], A, "A: survey detectability\n(labelled NonPSPL / generated)", 1),
                             (axes[1], B, "B: end-to-end recovery\n(predicted NonPSPL / generated)", 1),
                             (axes[2], C, "C: classifier efficiency B/A\n(>1 = flags sub-threshold binaries)", 1.5),
                             (axes[3], np.log10(np.maximum(N, 1)), "log$_{10}$ effective count", None)):
        im = ax.pcolormesh(se, qe, M, cmap="viridis", shading="auto",
                           vmin=0, vmax=vmax if vmax else None)
        ax.axvline(0.0, color="w", ls="--", lw=1)
        ax.axhline(np.log10(1.7e-4), color="r", ls=":", lw=1.2)
        ax.axhline(-2, color="w", ls="-.", lw=0.9)
        ax.set_xlabel("log$_{10}$ s"); ax.set_ylabel("log$_{10}$ q")
        ax.set_title(ttl, fontsize=9); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Detection efficiency in the (s, q) plane. white dashed s=1 resonant; "
                 "red dotted Suzuki+2016 break q=1.7e-4; white dash-dot planetary q=1e-2", y=1.0)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_training(logs: dict, out: str):
    plt = _style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for name, rows in logs.items():
        if not rows:
            continue
        ep = [r[0] for r in rows]
        axes[0].plot(ep, [r[1] for r in rows], "-o", ms=4, label=f"{name} F1")
        axes[1].plot(ep, [r[2] for r in rows], "-o", ms=4, label=f"{name} precision")
        axes[1].plot(ep, [r[3] for r in rows], "--s", ms=3, alpha=0.6, label=f"{name} recall")
        axes[2].plot(ep, [r[4] for r in rows], "-o", ms=4, label=f"{name} val loss")
    for ax, t, yl in ((axes[0], "NonPSPL F1", "F1"),
                      (axes[1], "precision vs recall — stage 1 collapsed", "value"),
                      (axes[2], "validation loss", "loss")):
        ax.set_xlabel("epoch"); ax.set_ylabel(yl); ax.set_title(t); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def _scan_rows(P, n_per_class: int = 1200, seed: int = 7) -> np.ndarray:
    """Class-balanced row sample, so the truncation figures are not dominated by Flat+PSPL."""
    rng = np.random.default_rng(seed)
    out = []
    for c in range(N_CLASSES):
        idx = np.nonzero(P.y == c)[0]
        if len(idx):
            out.append(rng.choice(idx, size=min(n_per_class, len(idx)), replace=False))
    return np.sort(np.concatenate(out))


def make_all(preds_dir: str, cache: str, out_dir: str, baseline: Optional[str] = None,
             logs: Optional[dict] = None, ckpt: Optional[str] = None,
             device: str = "mps", n_scan: int = 1200) -> list:
    os.makedirs(out_dir, exist_ok=True)
    P = Preds(preds_dir)
    made = []
    jobs = [("01_confusion.png", lambda p: fig_confusion(P, p)),
            ("02_roc_pr_operating.png", lambda p: fig_roc_pr(P, baseline, p)),
            ("03_calibration.png", lambda p: fig_calibration(P, p)),
            ("04_per_class.png", lambda p: fig_per_class(P, p)),
            ("05_parameter_dependence.png", lambda p: fig_parameter_dependence(P, p)),
            ("06_slices.png", lambda p: fig_slices(P, p)),
            ("07_failure_analysis.png", lambda p: fig_failure_analysis(P, p)),
            ("08_efficiency_planes.png", lambda p: fig_efficiency_planes(P, p)),
            ("09_light_curves.png", lambda p: fig_light_curves(P, cache, p))]
    if logs:
        jobs.append(("10_training.png", lambda p: fig_training(logs, p)))
    jobs.append(("11_class_distributions.png", lambda p: fig_class_distributions(P, p)))
    jobs.append(("12_temporal_bias.png", lambda p: fig_temporal_bias(P, p)))
    # The temporal pair needs a forward pass over progressively truncated curves, so it only
    # runs when a checkpoint is supplied. These were previously unreachable: the __main__
    # block sat ABOVE their definitions, so at the moment it executed the names did not yet
    # exist and make_all could only ever emit 10 figures.
    if ckpt:
        rows = _scan_rows(P, n_per_class=n_scan)
        probs, fracs = temporal_scan(ckpt, cache, rows, device=device)
        jobs.append(("13_probability_evolution.png",
                     lambda p: fig_probability_evolution(P, probs, fracs, rows, p)))
        jobs.append(("14_early_detection.png",
                     lambda p: fig_early_detection(P, probs, fracs, rows, p)))
    for name, fn in jobs:
        p = os.path.join(out_dir, name)
        try:
            fn(p)
            if os.path.exists(p):
                made.append(p); print(f"  ok   {name}")
        except Exception as e:
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    return made


# ---------------------------------------------------------------------------------
# Temporal analysis: what does the model know, and WHEN?
# ---------------------------------------------------------------------------------
def _truncated_batch(cache, n_all, rows, frac_seen: float):
    """Build model input with only the first `frac_seen` of the season revealed.

    Later bins are marked UNOBSERVED rather than zeroed -- the model already has an explicit
    observed-mask channel, so this is exactly the situation it sees when a band drops out,
    not an out-of-distribution input.
    """
    import torch
    from .model_v5 import BAND_BINS
    out = {}
    for b, L in BAND_BINS.items():
        f = np.memmap(f"{cache}/feat_{b}.f16", dtype=np.float16, mode="r", shape=(n_all, L, 3))
        fr = np.memmap(f"{cache}/frac_{b}.f16", dtype=np.float16, mode="r", shape=(n_all, L))
        x = np.asarray(f[rows], dtype=np.float32)
        g = np.asarray(fr[rows], dtype=np.float32)
        cut = int(round(frac_seen * L))
        obs = np.isfinite(x[:, :, 0]).astype(np.float32)
        obs[:, cut:] = 0.0
        g = g.copy(); g[:, cut:] = 0.0
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x[:, cut:, :] = 0.0
        out[b] = torch.tensor(np.concatenate([x, g[:, :, None], obs[:, :, None]], 2),
                              dtype=torch.float32)
    return out


def temporal_scan(ckpt: str, cache: str, rows: np.ndarray, device: str = "mps",
                  n_steps: int = 16) -> np.ndarray:
    """(len(rows), n_steps, 6) softmax probabilities as the season is progressively revealed."""
    import torch
    from .model_v5 import BAND_BINS, BinMLv5, ModelConfigV5
    ck = torch.load(ckpt, map_location="cpu")
    cfg = ModelConfigV5(**{k: v for k, v in ck["config"].items()
                           if k in ModelConfigV5.__dataclass_fields__})
    m = BinMLv5(cfg).to(device); m.load_state_dict(ck["model"]); m.eval()
    n_all = json.load(open(f"{cache}/meta.json"))["n_events"]
    fracs = np.linspace(1.0 / n_steps, 1.0, n_steps)
    out = np.zeros((len(rows), n_steps, N_CLASSES), dtype=np.float32)
    with torch.inference_mode():
        for k, fr in enumerate(fracs):
            for s in range(0, len(rows), 512):
                sl = rows[s:s + 512]
                feats = _truncated_batch(cache, n_all, sl, float(fr))
                feats = {kk: v.to(device) for kk, v in feats.items()}
                pres = {b: (feats[b][..., 4].sum(dim=1) > 0) for b in BAND_BINS}
                lg = m(feats, pres).float().cpu().numpy()
                out[s:s + len(sl), k] = softmax(lg)
    return out, fracs


def fig_probability_evolution(P: Preds, probs, fracs, rows, out: str):
    """Per-class probability tracks as the season is revealed — one panel per class."""
    plt = _style()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    days = fracs * 72.0
    for c in range(N_CLASSES):
        ax = axes[c // 3][c % 3]
        m = P.y[rows] == c
        if m.sum() < 5:
            ax.set_axis_off(); continue
        for k in range(N_CLASSES):
            mean = probs[m, :, k].mean(0)
            lo = np.percentile(probs[m, :, k], 25, axis=0)
            hi = np.percentile(probs[m, :, k], 75, axis=0)
            ax.plot(days, mean, color=COL[k], lw=1.6, label=CLASS_NAMES[k])
            ax.fill_between(days, lo, hi, color=COL[k], alpha=0.12)
        ax.set_ylim(0, 1); ax.set_xlabel("days of season observed")
        ax.set_ylabel("P(class)")
        ax.set_title(f"true = {CLASS_NAMES[c]}  (n={int(m.sum())})")
        if c == 0:
            ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Probability evolution as the season is revealed. Stage 4 was TRAINED with\n"
                 "truncation augmentation (50%), so partial seasons are in-distribution: with\n"
                 "nothing yet visible the model correctly reports Flat and commits as evidence arrives.",
                 y=1.0,
                 fontsize=8)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_early_detection(P: Preds, probs, fracs, rows, out: str):
    plt = _style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    days = fracs * 72.0
    y = P.y[rows]

    ax = axes[0]
    for c in range(N_CLASSES):
        m = y == c
        if m.sum() < 10:
            continue
        acc = (probs[m].argmax(2) == c).mean(0)
        ax.plot(days, acc, color=COL[c], lw=1.6, label=CLASS_NAMES[c])
    ax.set_xlabel("days observed"); ax.set_ylabel("recall")
    ax.set_ylim(0, 1); ax.set_title("Recall vs observing time")
    ax.legend(fontsize=7)

    ax = axes[1]
    pos = y == I_NON
    if pos.sum() > 10:
        for thr, ls in ((0.5, "-"), (0.8, "--"), (0.9, ":")):
            ax.plot(days, (probs[pos, :, I_NON] >= thr).mean(0), ls, lw=1.6,
                    label=f"P(NonPSPL) $\\geq$ {thr}")
        ax.set_xlabel("days observed"); ax.set_ylabel("fraction of anomalies flagged")
        ax.set_ylim(0, 1)
        ax.set_title("Anomaly flagging vs time")
        ax.legend(fontsize=7)

    ax = axes[2]
    # When does the model FIRST commit to the right answer and keep it?
    first = np.full(len(rows), np.nan)
    for i in range(len(rows)):
        pr = probs[i].argmax(1)
        ok = pr == y[i]
        if ok.any():
            # last index where it was wrong, +1 => first index of a run that persists
            wrong = np.nonzero(~ok)[0]
            j = (wrong[-1] + 1) if len(wrong) else 0
            if j < len(ok):
                first[i] = days[j]
    for c in range(N_CLASSES):
        m = (y == c) & np.isfinite(first)
        if m.sum() > 20:
            ax.hist(first[m], bins=np.linspace(0, 72, 17), histtype="step", lw=1.5,
                    color=COL[c], label=f"{CLASS_NAMES[c]} (med {np.median(first[m]):.0f}d)")
    ax.set_xlabel("days until the model commits to the correct class (and stays)")
    ax.set_ylabel("count"); ax.set_title("Time to stable correct classification")
    ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_class_distributions(P: Preds, out: str):
    """Physical parameter distributions per class — a sanity check on the simulation."""
    plt = _style()
    fields = [("dchi2_event", np.log10(np.maximum(P.dce, 1)), "log$_{10}$ event $\\Delta\\chi^2$"),
              ("dchi2_anom", np.log10(np.maximum(P.dca, 1)), "log$_{10}$ anomaly $\\Delta\\chi^2$"),
              ("m_base", P.mb, "baseline mag F146"), ("a_ks", P.aks, "extinction $A_{Ks}$")]
    q = P.col("q"); tE = P.col("tE"); u0 = P.col("u0")
    if q is not None:
        fields.append(("q", np.log10(np.where(np.isfinite(q) & (q > 0), q, np.nan)), "log$_{10}$ q"))
    if tE is not None:
        fields.append(("tE", np.log10(np.where(np.isfinite(tE) & (tE > 0), tE, np.nan)), "log$_{10}$ $t_E$"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for k, (_, v, lab) in enumerate(fields[:6]):
        ax = axes[k // 3][k % 3]
        for c in range(N_CLASSES):
            m = (P.y == c) & np.isfinite(v)
            if m.sum() > 50:
                ax.hist(v[m], bins=40, histtype="step", lw=1.4, density=True,
                        color=COL[c], label=CLASS_NAMES[c])
        ax.set_xlabel(lab); ax.set_ylabel("density")
        if k == 0:
            ax.legend(fontsize=6)
    fig.suptitle("Parameter distributions by observed class", y=0.995)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_temporal_bias(P: Preds, out: str):
    """Does performance depend on WHERE in the window the event peaks? It should not much."""
    plt = _style()
    t0 = P.col("t0")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    pos = P.y == I_NON; ok = P.pred == I_NON
    ax = axes[0]
    if t0 is not None:
        m = pos & np.isfinite(t0)
        edges = np.linspace(-40, 112, 20); cen, rr, nn = [], [], []
        for i in range(len(edges) - 1):
            sel = m & (t0 >= edges[i]) & (t0 < edges[i + 1])
            if sel.sum() < 30:
                continue
            cen.append(0.5 * (edges[i] + edges[i + 1])); rr.append(ok[sel].mean())
            nn.append(int(sel.sum()))
        ax.plot(cen, rr, "o-", color="#d62728")
        ax.axvspan(0, 72, alpha=0.12, color="green", label="inside the season")
        ax.set_ylim(0, 1.05); ax.set_xlabel("$t_0$ (days from season start)")
        ax.set_ylabel("NonPSPL recall")
        ax.set_title("Temporal bias: recall vs peak time"); ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "t0 unavailable", ha="center", transform=ax.transAxes)
    ax = axes[1]
    if t0 is not None:
        for c in range(N_CLASSES):
            m = (P.y == c) & np.isfinite(t0)
            if m.sum() > 50:
                ax.hist(t0[m], bins=40, histtype="step", lw=1.4, density=True,
                        color=COL[c], label=CLASS_NAMES[c])
        ax.axvspan(0, 72, alpha=0.12, color="green")
        ax.set_xlabel("$t_0$ (days)"); ax.set_ylabel("density")
        ax.set_title("Peak-time distribution (padded beyond the window by design)")
        ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


if __name__ == "__main__":
    import argparse, glob
    from .report_v5 import parse_log
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--ckpt", default=None, help="enables figures 13/14 (temporal scan)")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    v5 = os.path.expanduser("~/Desktop/Research/microlensing/v5runs")
    lg = {}
    # stage3/stage4 were missing here, so the fine-tuning history -- the part that actually
    # produced the shipped model -- never appeared in the training figure.
    for tag, pat in (("stage1", "train_2026*.log"), ("stage2", "stage2_*.log"),
                     ("stage3", "stage3_*.log"), ("stage4", "stage4_*.log")):
        f = sorted(glob.glob(os.path.join(v5, pat)))
        if f:
            lg[tag] = parse_log(f[-1])
    made = make_all(a.preds, a.cache, a.out, a.baseline, lg,
                    ckpt=a.ckpt, device=a.device)
    print(f"\n{len(made)} figures -> {a.out}")


def _event_prob_track(model, cache: str, n_all: int, row: int, device: str,
                      n_steps: int = 36):
    """Probabilities for ONE event as its season is progressively revealed.

    Returns (days, probs[n_steps, 6]). The reveal is expressed in DAYS of the 72-day season
    rather than fraction of bins, so the x-axis matches the light-curve panel above it.
    """
    import torch
    from .model_v5 import BAND_BINS
    fracs = np.linspace(1.0 / n_steps, 1.0, n_steps)
    rows = np.array([row])
    out = np.zeros((n_steps, N_CLASSES), dtype=np.float32)
    with torch.inference_mode():
        for k, fr in enumerate(fracs):
            feats = _truncated_batch(cache, n_all, rows, float(fr))
            feats = {kk: v.to(device) for kk, v in feats.items()}
            pres = {b: (feats[b][..., 4].sum(dim=1) > 0) for b in BAND_BINS}
            out[k] = softmax(model(feats, pres).float().cpu().numpy())[0]
    return fracs * 72.0, out


def fig_event_evolution(P: Preds, cache: str, ckpt: str, out_dir: str, n_per: int = 3,
                        device: str = "mps", n_steps: int = 36, pick: str = "mixed",
                        seed: int = 3) -> list:
    """The v4 three-panel per-event figure, extended to all six classes.

    Panel 1  the light curve (all three bands, magnitudes)
    Panel 2  every class probability as the season is revealed
    Panel 3  max probability -- how confident, and when it commits

    fig 13 shows the MEAN track per class, which is a different object: it answers "what does
    the average PSPL look like over time" and hides the per-event dynamics -- the moment a
    caustic crossing arrives and P(NonPSPL) jumps, or a curve that commits early and then
    changes its mind. This one keeps that, one figure per event, for every class including
    the ones v4 never had (PeriodicVar, LongPeriodVar, Eruptive).
    """
    import torch
    from .model_v5 import BAND_BINS, BinMLv5, ModelConfigV5
    plt = _style()
    os.makedirs(out_dir, exist_ok=True)
    ck = torch.load(ckpt, map_location="cpu")
    cfg = ModelConfigV5(**{k: v for k, v in ck["config"].items()
                           if k in ModelConfigV5.__dataclass_fields__})
    model = BinMLv5(cfg).to(device); model.load_state_dict(ck["model"]); model.eval()

    n_all = json.load(open(f"{cache}/meta.json"))["n_events"]
    feats = {b: np.memmap(f"{cache}/feat_{b}.f16", dtype=np.float16, mode="r",
                          shape=(n_all, L, 3)) for b, L in BAND_BINS.items()}
    ti = np.load(f"{P.d}/test_idx.npy") if os.path.exists(f"{P.d}/test_idx.npy") \
        else np.arange(len(P.y))
    colours = {"F146": "#1f77b4", "F087": "#2ca02c", "F213": "#d62728"}
    rng = np.random.default_rng(seed)
    made = []
    for c in range(N_CLASSES):
        # one guaranteed failure per class where one exists: the interesting dynamics are
        # usually in the events the model gets wrong or changes its mind about
        if pick == "random":
            # uniform draw from the class, so the sample reflects what the model actually
            # does on a typical event rather than over-representing failures
            idx = np.nonzero(P.y == c)[0]
            picks = list(rng.choice(idx, min(n_per, len(idx)), replace=False))
        else:
            good = np.nonzero((P.y == c) & (P.pred == c))[0]
            bad = np.nonzero((P.y == c) & (P.pred != c))[0]
            picks = list(rng.choice(good, min(n_per - 1, len(good)), replace=False))
            if len(bad):
                picks.append(int(rng.choice(bad)))
        for rank, i in enumerate(picks):
            i = int(i)
            src = int(ti[i]) if len(ti) == len(P.y) else i
            days, probs = _event_prob_track(model, cache, n_all, src, device, n_steps)
            fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9, 7.6), sharex=True)
            for b in BAND_BINS:
                f = np.asarray(feats[b][src, :, 0], dtype=np.float32)
                t = np.linspace(0, 72, len(f), endpoint=False)
                v = np.isfinite(f)
                if v.any():
                    a1.plot(t[v], P.mb[i] + f[v], ".", ms=2.0, color=colours[b], label=b)
            a1.invert_yaxis(); a1.set_ylabel("AB magnitude")
            a1.legend(fontsize=7, markerscale=4, ncol=3)
            ok = "correct" if P.pred[i] == c else "MISCLASSIFIED"
            a1.set_title(f"{CLASS_NAMES[c]} -> {CLASS_NAMES[P.pred[i]]} "
                         f"(p={P.p[i, P.pred[i]]:.2f}) — {ok}",
                         color="green" if P.pred[i] == c else "red")
            for k in range(N_CLASSES):
                a2.plot(days, probs[:, k], "-", lw=1.6, color=COL[k], label=CLASS_NAMES[k])
            a2.axhline(1.0 / N_CLASSES, color="gray", ls="--", lw=0.8, alpha=0.6)
            a2.set_ylabel("class probability"); a2.set_ylim(0, 1.05)
            a2.legend(fontsize=7, ncol=3, loc="best")
            conf = probs.max(1)
            a3.plot(days, conf, "-", color="black", lw=1.6)
            a3.fill_between(days, 0, conf, alpha=0.25, color="gray")
            # when it first commits to its final answer and never leaves
            fin = probs[-1].argmax()
            agree = probs.argmax(1) == fin
            stable = len(agree) - 1
            while stable > 0 and agree[stable - 1]:
                stable -= 1
            a3.axvline(days[stable], color="#d62728", ls=":", lw=1.2,
                       label=f"commits to {CLASS_NAMES[fin]} at day {days[stable]:.0f}")
            a3.set_ylabel("max probability"); a3.set_xlabel("days of season observed")
            a3.set_ylim(0, 1.05); a3.legend(fontsize=7)
            p = os.path.join(out_dir, f"evolution_{CLASS_NAMES[c]}_{i}.png")
            fig.tight_layout(); fig.savefig(p); plt.close(fig)
            made.append(p)
    return made

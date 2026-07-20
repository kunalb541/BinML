"""
BinML v5 — final report: metrics, baseline comparison, and figures.

Runs the trained model over the independent test set, computes the protocol defined in
``evaluate_v5``, compares against the classical Delta-chi^2 baseline, and writes the figures.

The comparison against the baseline is the scientifically load-bearing part. The baseline
applies the SAME decision rule the labels were defined by (delta-chi^2 vs flat, then vs a
refit PSPL) to noisy observed data, so any margin the network shows is margin from reading
morphology and colour rather than from privileged access to the labelling rule.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

from .classes import CLASS_NAMES
from .evaluate_v5 import I_NON, average_precision, evaluate_checkpoint, population_weights


def baseline_pr(npz_path: str):
    d = np.load(npz_path)
    y = d["label"]; kp = d["keep_prob"]
    w = population_weights(kp)
    # The classical anomaly score. An event must first BE an event, so combine: a large
    # anomaly residual only counts when the event itself was detected.
    score = np.where(d["dchi2_event"] > 500, d["dchi2_anomaly"], -np.inf)
    ap = average_precision(score, y, w)
    return {"average_precision_population": float(ap),
            "n": int(len(y)), "prevalence": float((y == I_NON).mean()),
            "median_dchi2_anomaly_by_label": {
                CLASS_NAMES[c]: float(np.median(d["dchi2_anomaly"][y == c]))
                for c in range(len(CLASS_NAMES)) if (y == c).sum() > 5}}


def figures(res: dict, base: dict, curves: list, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 11))

    # 1. training curves
    ax = fig.add_subplot(2, 3, 1)
    for name, rows, style in curves:
        if not rows:
            continue
        ep = [r[0] for r in rows]
        ax.plot(ep, [r[1] for r in rows], style, label=f"{name} NonPSPL F1")
        ax.plot(ep, [r[2] for r in rows], style, alpha=0.45, label=f"{name} precision")
    ax.set_xlabel("epoch"); ax.set_ylabel("score"); ax.set_title("training")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 2. detectability-conditioned recall
    ax = fig.add_subplot(2, 3, 2)
    d = res.get("nonpspl_recall_by_dchi2", {})
    ks = [k for k in d if d[k].get("recall") is not None]
    ax.bar(range(len(ks)), [d[k]["recall"] for k in ks], color="#1f77b4")
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, rotation=40, ha="right", fontsize=7)
    ax.set_ylim(0, 1); ax.set_ylabel("NonPSPL recall")
    ax.set_title("recall vs anomaly $\\Delta\\chi^2$"); ax.grid(alpha=0.3, axis="y")
    for i, k in enumerate(ks):
        ax.text(i, 0.02, f"n={d[k]['n']}", ha="center", fontsize=6, rotation=90)

    # 3. confusion (population-weighted, row-normalised)
    ax = fig.add_subplot(2, 3, 3)
    cm = np.array(res["argmax_population"].get("confusion", res.get("confusion", [])), dtype=float) \
        if "confusion" in res.get("argmax_population", {}) else None
    if cm is None:
        cm = np.zeros((6, 6))
    rn = cm / np.maximum(cm.sum(1, keepdims=True), 1e-9)
    im = ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(CLASS_NAMES, fontsize=6)
    ax.set_title("confusion (pop-weighted, row-norm)")
    for i in range(6):
        for j in range(6):
            if rn[i, j] > 0.005:
                ax.text(j, i, f"{rn[i,j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if rn[i, j] > 0.5 else "black")

    # 4-6. the (log s, log q) planes
    pl = res.get("efficiency_plane")
    titles = [("survey_detectability", "A: survey detectability"),
              ("end_to_end_recovery", "B: end-to-end recovery"),
              ("classifier_efficiency", "C: classifier efficiency (B/A)")]
    for k, (key, title) in enumerate(titles):
        ax = fig.add_subplot(2, 3, 4 + k)
        if pl is None:
            ax.text(0.5, 0.5, "no params in cache", ha="center", transform=ax.transAxes)
            ax.set_title(title); continue
        M = np.array(pl[key], dtype=float)
        qe = pl["log_q_edges"]; se = pl["log_s_edges"]
        im = ax.pcolormesh(se, qe, M, cmap="viridis", vmin=0, vmax=1, shading="auto")
        ax.axvline(0.0, color="w", lw=0.8, ls="--")               # s = 1, resonant caustic
        ax.axhline(np.log10(1.7e-4), color="r", lw=0.8, ls=":")   # Suzuki+2016 break
        ax.set_xlabel("log$_{10}$ s"); ax.set_ylabel("log$_{10}$ q")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("BinML v5 — Roman multi-band 6-class classifier, independent test set", y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"figures -> {out_png}")


_EP_RE = re.compile(
    r"^ep\s+(\d+)\s+loss\s+([\d.naif]+)\s+val\s+([\d.naif]+)\s+NonPSPL f1\s+([\d.]+)\s+"
    r"\(r\s+([\d.]+)\s+p\s+([\d.]+)\)\s+macroF1\s+([\d.]+)")


def parse_log(path: str):
    """(epoch, f1, precision, recall, val_loss) per epoch.

    Uses a regex, not str.split. Splitting on "p " looks fine until you notice that the line
    STARTS with "ep ", so `line.split("p ")[1]` returns the epoch number and every parse
    silently failed -- leaving the training-curve panel empty with no error.
    """
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path):
        m = _EP_RE.match(line)
        if m:
            try:
                rows.append((int(m.group(1)), float(m.group(4)), float(m.group(6)),
                             float(m.group(5)), float(m.group(3))))
            except ValueError:
                pass
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-events", type=int, default=0)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    res = evaluate_checkpoint(args.ckpt, args.cache, os.path.join(args.out, "preds"),
                              device=args.device, batch=1024, max_events=args.max_events)
    base = baseline_pr(args.baseline) if args.baseline and os.path.exists(args.baseline) else None

    import glob
    v5 = os.path.expanduser("~/Desktop/Research/microlensing/v5runs")
    curves = [("stage1", parse_log(sorted(glob.glob(f"{v5}/train_2026*.log"))[-1] if glob.glob(f"{v5}/train_2026*.log") else ""), "-o"),
              ("stage2", parse_log(sorted(glob.glob(f"{v5}/stage2_*.log"))[-1] if glob.glob(f"{v5}/stage2_*.log") else ""), "-s")]
    figures(res, base or {}, curves, os.path.join(args.out, "binml_v5_report.png"))

    summary = {"model": res, "baseline": base}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2, default=str)

    print("\n================ BinML v5 — FINAL ================")
    hl = res.get("headline")
    if hl:
        print(f"completeness @ {hl['target_purity']:.0%} purity : {hl['completeness_at_fixed_purity']:.4f}")
        print(f"purity achieved                    : {hl['purity_achieved']:.4f}")
    ci = res.get("headline_ci95")
    if ci:
        print(f"  95% CI                           : [{ci['lo']:.4f}, {ci['hi']:.4f}]")
    print(f"average precision (population)     : {res['average_precision_population']:.4f}")
    if base:
        print(f"  BASELINE (classical dchi2)       : {base['average_precision_population']:.4f}")
        print(f"  improvement                      : "
              f"{res['average_precision_population']/max(base['average_precision_population'],1e-9):.2f}x")
    for conv in ("argmax_sample", "argmax_population"):
        m = res[conv]
        print(f"\n{conv}:  macro F1 {m['macro_f1']:.4f}")
        for c in CLASS_NAMES:
            r, p = m["recall"][c], m["precision"][c]
            print(f"   {c:14s} r={r if r is None else round(r,4)}  p={p if p is None else round(p,4)}")
    print("\nslices (NonPSPL recall):")
    for k, v in res.get("slices", {}).items():
        print(f"   {k:28s} n={v['n_true_nonpspl']:6d}  recall={v['recall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

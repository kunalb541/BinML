#!/usr/bin/env python3
"""What a survey could achieve at fixed purity if it re-tuned the threshold for its own prevalence.

AUDIT FINDING THIS FIXES.  The manuscript previously reported "re-optimising on the frozen
artifact": the prevalence-specific thresholds were chosen from the FINAL-TEST precision-recall
curve and then evaluated on those same rows.  That is threshold selection on the test set, and it
biases the reported completeness upward.  Here every threshold is chosen on the reserved
validation rows (the 90,117 rows never used for reporting) and applied unchanged to the 360,472
final-test rows, so the reported completeness is an honest held-out number.

WHAT IS ASSUMED.  Only the anomaly prevalence changes; the class-conditional score distributions
are held fixed.  This is the standard label-shift assumption, and it is exactly what lets a
threshold chosen at one prevalence be re-derived for another: the reweighting rescales the negative
mass relative to the positive mass without touching either score distribution.  If a real stream's
non-anomalous population also LOOKS different (different cadence, different blending, genuine
variables the simulator does not contain), this analysis does not cover it -- that is domain shift,
not prior shift, and it is treated separately in the paper's limitations.

Writes validation/prevalence_result.json; consumed directly by paper/make_macros.py and
paper/make_figures.py.  Usage:  python validation/prevalence_thresholds.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RES = os.path.join(REPO, "paper", "results")
NON = 2                                     # NonPSPL class index
PURITY_TARGET = 0.90
SCENARIOS = [("synthetic", None), ("one_pct", 0.01), ("tenth_pct", 0.001)]


def reweight(w, pos, prevalence):
    """Scale positive weights so the anomaly class carries `prevalence` of the total mass."""
    if prevalence is None:
        return w
    wp, wn = w[pos].sum(), w[~pos].sum()
    scale = prevalence / (1.0 - prevalence) * wn / wp
    out = w.copy()
    out[pos] *= scale
    return out


def threshold_for_purity(score, pos, w, target=PURITY_TARGET):
    """Lowest threshold whose weighted purity is at least `target` (i.e. best completeness
    subject to the purity constraint).  Returns None if no threshold reaches the target."""
    order = np.argsort(-score)
    s, p, ww = score[order], pos[order], w[order]
    tp = np.cumsum(ww * p)
    sel = np.cumsum(ww)
    prec = tp / np.maximum(sel, 1e-12)
    # Only consider distinct score values, and only points where the constraint is met.
    last = np.r_[s[1:] != s[:-1], True]
    ok = last & (prec >= target)
    if not ok.any():
        return None
    return float(s[np.max(np.flatnonzero(ok))])


def at_threshold(score, pos, w, thr):
    sel = score >= thr
    tp = (w * (sel & pos)).sum(); fp = (w * (sel & ~pos)).sum(); fn = (w * (~sel & pos)).sum()
    return (float(tp / max(tp + fn, 1e-12)), float(tp / max(tp + fp, 1e-12)))


def counts(score, pos, thr):
    """Raw (unweighted) row counts above threshold.  At low assumed prevalence the constraint is
    met far out in the score tail, where few rows live and the selected threshold is noisy; the
    counts are what tells a reader how much sample supports each scenario."""
    sel = score >= thr
    return {"n_selected": int(sel.sum()), "n_selected_true_anomaly": int((sel & pos).sum()),
            "n_anomaly_rows": int(pos.sum())}


def main():
    lg = np.load(os.path.join(RES, "logits.npy")).astype(np.float64)
    y = np.load(os.path.join(RES, "label.npy")).astype(int)
    kp = np.load(os.path.join(RES, "keep_prob.npy")).astype(np.float64)
    ti = np.load(os.path.join(RES, "test_idx.npy")).astype(int)
    n = len(y)
    # The reserved threshold-selection rows are the complement of the frozen final-test rows.
    vi = np.setdiff1d(np.arange(n), ti, assume_unique=False)
    assert len(vi) + len(ti) == n and not np.intersect1d(vi, ti).size, "val/test overlap"

    z = lg - lg.max(1, keepdims=True)
    e = np.exp(z)
    score = (e / e.sum(1, keepdims=True))[:, NON]
    pos = y == NON
    w = 1.0 / np.clip(kp, 1e-3, 1.0)

    pi0 = float(w[ti][pos[ti]].sum() / w[ti].sum())
    frozen = json.load(open(os.path.join(RES, "metrics.json")))["headline"]["threshold"]

    out = {"_doc": __doc__.split("\n")[0],
           "purity_target": PURITY_TARGET,
           "n_validation": int(len(vi)), "n_final_test": int(len(ti)),
           "synthetic_prevalence": round(pi0, 4),
           "frozen_threshold": frozen,
           "selection": "threshold chosen on validation rows, reported on final-test rows",
           "assumption": "label shift only: class-conditional score distributions held fixed",
           "scenarios": {}}

    for name, prev in SCENARIOS:
        p_use = pi0 if prev is None else prev
        wv = reweight(w[vi], pos[vi], p_use)
        wt = reweight(w[ti], pos[ti], p_use)
        thr = threshold_for_purity(score[vi], pos[vi], wv)
        rec = {"prevalence": round(p_use, 5)}
        if thr is None:
            rec.update({"threshold": None, "note": f"purity {PURITY_TARGET} unreachable"})
        else:
            comp, pur = at_threshold(score[ti], pos[ti], wt, thr)
            comp_v, pur_v = at_threshold(score[vi], pos[vi], wv, thr)
            rec.update({"threshold": round(thr, 6),
                        "completeness_test": round(comp, 3), "purity_test": round(pur, 3),
                        "completeness_val": round(comp_v, 3), "purity_val": round(pur_v, 3),
                        "counts_val": counts(score[vi], pos[vi], thr),
                        "counts_test": counts(score[ti], pos[ti], thr)})
        # what the SHIPPED fixed threshold delivers at this prevalence, for contrast
        comp_f, pur_f = at_threshold(score[ti], pos[ti], wt, frozen)
        rec["completeness_test_at_frozen_thr"] = round(comp_f, 3)
        rec["purity_test_at_frozen_thr"] = round(pur_f, 3)
        out["scenarios"][name] = rec

    # Curve for the figure: threshold selected on validation at each prevalence, completeness
    # reported on final test.
    curve = []
    for p_use in np.logspace(np.log10(1e-4), np.log10(0.2), 30):
        wv = reweight(w[vi], pos[vi], p_use)
        wt = reweight(w[ti], pos[ti], p_use)
        thr = threshold_for_purity(score[vi], pos[vi], wv)
        if thr is None:
            curve.append({"prevalence": round(float(p_use), 6), "completeness_test": None})
            continue
        comp, pur = at_threshold(score[ti], pos[ti], wt, thr)
        curve.append({"prevalence": round(float(p_use), 6), "threshold": round(thr, 6),
                      "completeness_test": round(comp, 4), "purity_test": round(pur, 4)})
    out["curve"] = curve

    dest = os.path.join(HERE, "prevalence_result.json")
    json.dump(out, open(dest, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "curve"}, indent=2))
    print(f"-> {dest}  (curve has {len(curve)} points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

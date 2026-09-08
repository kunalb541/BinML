"""Reduce GULLS transfer row files into the paper's tables and response curves.

Pure reduction: takes the per-event rows written by validation/gulls_transfer.py for two
checkpoints (shipped, gap-aware), enforces a matched event set, and emits every number the
cross-simulator section reports -- per-class rates with Wilson intervals, GULLS
final_weight-weighted rates (the population-realistic version), response curves in t_E,
baseline magnitude, |u0| and log10(q), and the shipped->fine-tuned flip matrix.  Nothing here
touches the network or the model; rerunning it on the same rows is free and deterministic,
which is the property that let cascade_reduce.py close the streaming-analysis audit.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

LABELS = {"RMDC26_1S1L_ML": ("PSPL", "single lens"),
          "RMDC26_1S2L_ML": ("NonPSPL", "planetary lens"),
          "RMDC26_2S2L_ML": ("NonPSPL", "planetary lens AND binary source")}

TE_BINS = [1, 3, 10, 30, 100, 300]
MB_BINS = [16, 19, 20, 21, 22, 23, 26]
U0_BINS = [0.0, 0.1, 0.3, 0.6, 1.0, 3.0]
LOGQ_BINS = [-7, -5, -4.5, -4, -3.5, -3, -1.5]


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(c - h, 0.0), 4), round(min(c + h, 1.0), 4)]


def rate(flags, weights=None):
    """Rate with a Wilson interval; weighted rates use the effective sample size."""
    n = len(flags)
    if n == 0:
        return {"n": 0}
    f = np.asarray(flags, float)
    if weights is None:
        k = int(f.sum())
        return {"n": n, "rate": round(k / n, 4), "ci95": wilson(k, n)}
    w = np.asarray(weights, float)
    r = float((w * f).sum() / w.sum())
    n_eff = float(w.sum() ** 2 / (w * w).sum())
    return {"n": n, "n_eff": round(n_eff, 1), "rate": round(r, 4),
            "ci95": wilson(r * n_eff, n_eff)}


def slice_curve(rows, key_fn, edges, thr):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = [r for r in rows if lo <= key_fn(r) < hi]
        cell = {"bin": [lo, hi],
                "over_thr": rate([r["p_nonpspl"] >= thr for r in sub]),
                "argmax_nonpspl": rate([r["pred"] == "NonPSPL" for r in sub])}
        out.append(cell)
    return out


def load_rows(path):
    rows = [r for r in json.load(open(path)) if "pred" in r and r.get("dense")]
    return {r["event_id"]: r for r in rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows-a", required=True, help="rows JSON for the shipped checkpoint")
    ap.add_argument("--rows-b", required=True, help="rows JSON for the gap-aware checkpoint")
    ap.add_argument("--label-a", default="shipped")
    ap.add_argument("--label-b", default="ft_g08e12")
    ap.add_argument("--out", default=os.path.join(HERE, "transfer_reduced.json"))
    args = ap.parse_args(argv)

    thr = json.load(open(os.path.join(REPO, "paper", "results",
                                      "metrics.json")))["headline"]["threshold"]
    A, B = load_rows(args.rows_a), load_rows(args.rows_b)
    common = sorted(set(A) & set(B))
    # The matched set is load-bearing: rates for the two checkpoints are only comparable on
    # identical events, and the curve cache guarantees identical inputs for them.
    dropped = {"only_a": len(A) - len(common), "only_b": len(B) - len(common)}

    out = {"_doc": __doc__.split("\n")[0], "threshold": thr,
           "n_matched_dense": len(common), "dropped_unmatched": dropped,
           "models": {"a": args.label_a, "b": args.label_b}, "by_class": {}}

    for lab, (truth, desc) in LABELS.items():
        ra = [A[e] for e in common if A[e]["sim_label"] == lab]
        rb = [B[e] for e in common if B[e]["sim_label"] == lab]
        if not ra:
            continue
        blk = {"truth": truth, "description": desc, "n": len(ra), "per_model": {}}
        for tag, rr in ((args.label_a, ra), (args.label_b, rb)):
            w = [r["weight"] for r in rr]
            pm = {"over_thr": rate([r["p_nonpspl"] >= thr for r in rr]),
                  "over_thr_weighted": rate([r["p_nonpspl"] >= thr for r in rr], w),
                  "argmax_nonpspl": rate([r["pred"] == "NonPSPL" for r in rr]),
                  "argmax_nonpspl_weighted": rate([r["pred"] == "NonPSPL" for r in rr], w),
                  "argmax_distribution": {},
                  "curves": {
                      "tE": slice_curve(rr, lambda r: r["tE"], TE_BINS, thr),
                      "m_base": slice_curve(rr, lambda r: r["m_base"], MB_BINS, thr),
                      "abs_u0": slice_curve(rr, lambda r: abs(r["u0"]), U0_BINS, thr)}}
            preds = [r["pred"] for r in rr]
            for c in sorted(set(preds)):
                pm["argmax_distribution"][c] = round(preds.count(c) / len(preds), 4)
            if lab != "RMDC26_1S1L_ML":
                pm["curves"]["log10_q"] = slice_curve(
                    rr, lambda r: math.log10(max(float(r["planet_q"]), 1e-9)),
                    LOGQ_BINS, thr)
            blk["per_model"][tag] = pm
        # who flipped where, shipped -> fine-tuned
        flips = {}
        for e in common:
            if A[e]["sim_label"] != lab:
                continue
            key = f"{A[e]['pred']}->{B[e]['pred']}"
            flips[key] = flips.get(key, 0) + 1
        blk["flip_matrix"] = dict(sorted(flips.items(), key=lambda kv: -kv[1]))
        if lab == "RMDC26_2S2L_ML":
            for tag, rr in ((args.label_a, ra), (args.label_b, rb)):
                tb = [r for r in rr if (r.get("source_is_binary") or 0) > 0]
                blk["per_model"][tag]["binary_source_half"] = {
                    "over_thr": rate([r["p_nonpspl"] >= thr for r in tb]),
                    "argmax_nonpspl": rate([r["pred"] == "NonPSPL" for r in tb])}
        out["by_class"][lab] = blk

    json.dump(out, open(args.out, "w"), indent=1)
    print(f"matched dense events: {len(common)}  (unmatched dropped: {dropped})")
    for lab, blk in out["by_class"].items():
        print(f"\n{lab} (truth {blk['truth']}, n={blk['n']})")
        for tag, pm in blk["per_model"].items():
            print(f"  {tag:10s} over_thr {pm['over_thr']['rate']:.4f} {pm['over_thr']['ci95']}"
                  f"  weighted {pm['over_thr_weighted']['rate']:.4f}"
                  f"  argmaxNP {pm['argmax_nonpspl']['rate']:.4f}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Sub-day-t_E single lenses (the FFP-like third of RMDC26 1S1L) as an explicit OUT-OF-SUPPORT row.

BinML's t_E prior is truncated to [1, 300] d; the GULLS transfer applies the same support cut, so
sub-day events never entered any reported number. Roman will find them. This summarises how each
checkpoint reads them: single-lens false-alarm rate at the frozen and (where available) gapped-
calibrated thresholds, the argmax class distribution, and the false-alarm rate by t_E bin --
from the rows written by gulls_transfer.py --min-te 0 --max-te 1 into the sub-day curve cache.

Usage:
  python validation/gulls/subday_summary.py --models shipped ft_g08e12 fspl_g08 fspl5_g08 fspl5s_g08
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.expanduser("~/Desktop/Research/microlensing/gulls_curve_cache_subday")
FROZEN = 0.9042405486106873
L1 = "RMDC26_1S1L_ML"
TE_BINS = [(0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--rows-dir", default=SUB)
    ap.add_argument("--out", default=os.path.join(HERE, "transfer_subday.json"))
    args = ap.parse_args(argv)
    rows = {m: {r["event_id"]: r for r in json.load(open(os.path.join(args.rows_dir, f"rows_subday_{m}.json")))
                if r.get("dense") and "pred" in r and r["sim_label"] == L1} for m in args.models}
    common = sorted(set.intersection(*[set(v) for v in rows.values()]))
    te = np.array([rows[args.models[0]][i]["tE"] for i in common])
    w = np.array([rows[args.models[0]][i]["weight"] for i in common])
    out = {"_doc": __doc__.split("\n")[0], "n_matched_dense_1S1L_subday": len(common),
           "te_days": {"median": round(float(np.median(te)), 3), "p10": round(float(np.percentile(te, 10)), 3),
                       "p90": round(float(np.percentile(te, 90)), 3)},
           "note": "OUT OF SUPPORT: t_E below the [1, 300] d training prior; a 0.16 d event spans two of the 864 F146 bins",
           "models": {}}
    for m in args.models:
        p = np.array([rows[m][i]["p_nonpspl"] for i in common])
        pred = np.array([rows[m][i]["pred"] for i in common])
        thrs = {"frozen": FROZEN}
        f = os.path.join(HERE, f"gapped_threshold_{m}_seasons.json")
        if os.path.exists(f):
            fp = json.load(open(f))["arms"]["rmdc26_gapped"].get("pool", {}).get("full_pool", {})
            if fp.get("achievable"):
                thrs["calibrated_seasons_fullpool"] = float(fp["threshold"])
        blk = {"fa_at": {k: {"threshold": round(t, 4), "fa": round(float((p >= t).mean()), 4),
                             "fa_weighted": round(float(w[p >= t].sum() / w.sum()), 4)} for k, t in thrs.items()},
               "argmax_distribution": {c: round(float((pred == c).mean()), 4) for c in sorted(set(pred))},
               "fa_frozen_by_te": []}
        blk["sample"] = ("a random contiguous id window of eligible events per class (gulls_transfer.py --per-class 1500 "
                         "--min-te 0 --max-te 1); dense in-season single lenses only")
        for lo, hi in TE_BINS:
            sel = (te >= lo) & (te < hi)
            blk["fa_frozen_by_te"].append({"te_bin": [lo, hi], "n": int(sel.sum()),
                                           "fa": round(float((p[sel] >= FROZEN).mean()), 4) if sel.sum() else None,
                                           "frac_argmax_pspl": round(float((pred[sel] == "PSPL").mean()), 4) if sel.sum() else None})
        out["models"][m] = blk
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"n={len(common)} sub-day 1S1L, tE median {out['te_days']['median']} d")
    print(f"{'model':12s} {'FA frozen':>9} {'FA calib':>9}  argmax")
    for m, b in out["models"].items():
        c = b["fa_at"].get("calibrated_seasons_fullpool", {}).get("fa")
        print(f"{m:12s} {b['fa_at']['frozen']['fa']:9.4f} {(c if c is not None else float('nan')):9.4f}  "
              + " ".join(f"{k}={v:.2f}" for k, v in b["argmax_distribution"].items()))
    print("->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

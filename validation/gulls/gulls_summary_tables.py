"""GULLS/RMDC26 summary tables across checkpoints, from the per-event rows -- one command, no typing.

Reads the rows files written by validation/gulls_transfer.py (default: the curve-cache directory),
restricts every comparison to the events ALL listed models scored (identical inputs, identical set),
and writes two artifacts:

  transfer_tradeoff_all.json   -- per model: false alarms at the frozen threshold; 1S1L false-alarm
                                  rate by rho/|u0| (finite-source strength, from the RMDC26 metadata);
                                  planetary recall at MATCHED single-lens false-alarm budgets; mean
                                  1S2L recall over FA <= 0.3.  Matched budgets are the comparison that
                                  cannot be gamed by a threshold shift.
  transfer_colour_ablation.json -- for each checkpoint scored with --bands F146, the same quantities
                                  next to its three-band run.

Usage:
  python validation/gulls/gulls_summary_tables.py \\
      --models shipped=rows_full_shipped_v2.json ft_g08e12=rows_full_ft_g08e12_v2.json \\
               fspl_g08=rows_full_fspl_g08.json fspl5_g08=rows_full_fspl5_g08.json fspl5s_g08=rows_full_fspl5s_g08.json \\
      --f146 fspl_g08=rows_full_fspl_g08_f146only.json ft_g08e12=rows_full_ft_g08e12_f146only.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser("~/Desktop/Research/microlensing/gulls_curve_cache")
META = "/tmp/rmdc26_meta.parquet"
FROZEN = 0.9042405486106873
BINS = [(0, 0.03), (0.03, 0.1), (0.1, 0.3), (0.3, 1), (1, 3), (3, float("inf"))]
BUDGETS = (0.02, 0.031, 0.052, 0.117, 0.20)
L1, L2, L3 = "RMDC26_1S1L_ML", "RMDC26_1S2L_ML", "RMDC26_2S2L_ML"


def load_rows(path):
    if not os.path.isabs(path):
        path = os.path.join(CACHE, path)
    return {r["event_id"]: (r["sim_label"], float(r["p_nonpspl"]))
            for r in json.load(open(path)) if r.get("dense") and "pred" in r}


def load_meta(ids):
    import pyarrow.parquet as pq
    m = pq.read_table(META, columns=["event_id", "rho", "u0lens1"]).to_pydict()
    d = {int(e): (float(r), abs(float(u))) for e, r, u in zip(m["event_id"], m["rho"], m["u0lens1"])}
    return np.array([d[i][0] / max(d[i][1], 1e-6) for i in ids])


def summarise(p, lab, ratio):
    s1, s2, s3 = lab == L1, lab == L2, lab == L3
    ths = np.unique(np.round(p, 4))[::-1]
    fa = np.array([(p[s1] >= t).mean() for t in ths])
    r2 = np.array([(p[s2] >= t).mean() for t in ths])
    r3 = np.array([(p[s3] >= t).mean() for t in ths])
    o = np.argsort(fa); k = fa[o] <= 0.3
    out = {"frozen_threshold": {"fa_1S1L": round(float((p[s1] >= FROZEN).mean()), 4),
                                "recall_1S2L": round(float((p[s2] >= FROZEN).mean()), 4),
                                "recall_2S2L": round(float((p[s3] >= FROZEN).mean()), 4)},
           "fa_1S1L_by_rho_over_u0": [], "recall_at_matched_fa": [],
           "mean_recall_1S2L_fa_le_0p3": round(float(np.trapezoid(r2[o][k], fa[o][k]) / 0.3), 4),
           "mean_recall_2S2L_fa_le_0p3": round(float(np.trapezoid(r3[o][k], fa[o][k]) / 0.3), 4)}
    for lo, hi in BINS:
        sel = s1 & (ratio >= lo) & (ratio < hi)
        out["fa_1S1L_by_rho_over_u0"].append({"bin": [lo, None if hi == float("inf") else hi], "n": int(sel.sum()),
                                              "fa": round(float((p[sel] >= FROZEN).mean()), 4) if sel.sum() else None})
    for tgt in BUDGETS:
        i = int(np.argmin(np.abs(fa - tgt)))
        out["recall_at_matched_fa"].append({"fa_target": tgt, "threshold": float(ths[i]), "fa": round(float(fa[i]), 4),
                                            "recall_1S2L": round(float(r2[i]), 4), "recall_2S2L": round(float(r3[i]), 4)})
    return out


def parse_pairs(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1); out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", required=True, help="name=rows.json (three-band runs)")
    ap.add_argument("--f146", nargs="*", default=[], help="name=rows.json (F146-only runs of the same checkpoints)")
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args(argv)
    models = {k: load_rows(v) for k, v in parse_pairs(args.models).items()}
    f146 = {k: load_rows(v) for k, v in parse_pairs(args.f146).items()}
    common = sorted(set.intersection(*[set(v) for v in list(models.values()) + list(f146.values())]))
    lab = np.array([next(iter(models.values()))[i][0] for i in common])
    ratio = load_meta(common)
    trade = {"_doc": __doc__.split("\n")[0], "n_matched_dense": len(common), "frozen_threshold": FROZEN,
             "budgets": list(BUDGETS), "models": {}}
    for name, d in models.items():
        p = np.array([d[i][1] for i in common])
        trade["models"][name] = summarise(p, lab, ratio)
    json.dump(trade, open(os.path.join(args.out_dir, "transfer_tradeoff_all.json"), "w"), indent=1)
    print(f"matched dense events: {len(common)}")
    print(f"{'model':12s} {'FA@frozen':>9} {'rec 1S2L':>9} {'rec 2S2L':>9} {'rec@5.2%FA':>11} {'mean rec<=0.3':>13}   FA by rho/|u0| bins")
    for name, m in trade["models"].items():
        f = m["frozen_threshold"]; at = next(a for a in m["recall_at_matched_fa"] if a["fa_target"] == 0.052)
        bins = " ".join(f"{b['fa']:.3f}" if b["fa"] is not None else "  -  " for b in m["fa_1S1L_by_rho_over_u0"])
        print(f"{name:12s} {f['fa_1S1L']:9.4f} {f['recall_1S2L']:9.4f} {f['recall_2S2L']:9.4f} {at['recall_1S2L']:11.3f} {m['mean_recall_1S2L_fa_le_0p3']:13.3f}   {bins}")
    if f146:
        col = {"_doc": "Colour ablation on GULLS: three-band vs F146-only inputs for the same checkpoint on identical events.",
               "n_matched_dense": len(common), "models": {}}
        for name, d in f146.items():
            if name not in models:
                continue
            pa = np.array([models[name][i][1] for i in common]); pb = np.array([d[i][1] for i in common])
            col["models"][name] = {"three_band": summarise(pa, lab, ratio), "f146_only": summarise(pb, lab, ratio),
                                   "per_event_abs_dP_median": round(float(np.median(np.abs(pa - pb))), 4),
                                   "per_event_abs_dP_p90": round(float(np.percentile(np.abs(pa - pb), 90)), 4),
                                   "frac_decisions_flipped_at_frozen": round(float(((pa >= FROZEN) != (pb >= FROZEN)).mean()), 4)}
        json.dump(col, open(os.path.join(args.out_dir, "transfer_colour_ablation.json"), "w"), indent=1)
        print("\ncolour ablation (recall at 5.2% FA, 1S2L/2S2L; mean recall FA<=0.3):")
        for name, c in col["models"].items():
            for tag in ("three_band", "f146_only"):
                at = next(a for a in c[tag]["recall_at_matched_fa"] if a["fa_target"] == 0.052)
                print(f"  {name:12s} {tag:10s} {at['recall_1S2L']:.3f}/{at['recall_2S2L']:.3f}   {c[tag]['mean_recall_1S2L_fa_le_0p3']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

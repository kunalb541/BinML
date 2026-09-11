"""Sensitivity of the GULLS relabelling and recall to the adopted 0.02 mag detectability floor.

The floor is a modelling choice (pipeline/assemble.SurveyConfig.min_amplitude_mag) and the
pre-submission referee round asked how results depend on it. The truth cache written by
detectability_relabel.py stores, per GULLS event, the noise-free statistics the label policy
uses (delta-chi^2 vs flat, max excursion, delta-chi^2 vs the best PSPL, peak deviation), so the
LABEL side of that question is a reduction: re-derive label_detect at each floor and re-score
recall on detectable binaries and the flag rate on undetectable ones for each checkpoint. The
TRAINING side (retraining with another floor) is not covered here.

Usage:
  python validation/gulls/floor_sensitivity.py --models fspl5s_g08=rows_full_fspl5s_g08.json ft_g08e12=rows_full_ft_g08e12_v2.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from detectability_relabel import _load_truth, L1, L2, L3, CURVES, FROZEN, CFG  # noqa: E402

FLOORS = (0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1)


def label_at(r, floor):
    """The training label rule at a given floor. Single lenses are never NonPSPL (the rule refits only
    binaries), exactly as detectability_relabel._stats labels them."""
    if r["dchi2_event"] < CFG.dchi2_event or r["max_amp_mag"] < floor:
        return "Flat"
    if r["sim_label"] != L1 and r["dchi2_anomaly"] >= CFG.dchi2_anomaly and r["anomaly_amp_mag"] >= floor:
        return "NonPSPL"
    return "PSPL"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--truth-cache", default=os.path.expanduser("~/Desktop/Research/microlensing/gulls_truth_cache_v2"))
    ap.add_argument("--ref-rows", default="rows_full_fspl5s_g08.json")
    ap.add_argument("--cap-1s1l", type=int, default=1600); ap.add_argument("--cap-binary", type=int, default=2500)
    ap.add_argument("--out", default=os.path.join(HERE, "floor_sensitivity.json"))
    args = ap.parse_args(argv)
    from detectability_relabel import selected_ids
    truth = _load_truth(args.truth_cache)
    sel = selected_ids(args.ref_rows, args.cap_1s1l, args.cap_binary)
    want = set(x for v in sel.values() for x in v)
    truth = {k: v for k, v in truth.items() if k in want}
    bad = [k for k, v in truth.items() if label_at(v, CFG.min_amplitude_mag) != v["label_detect"]]
    assert not bad, f"label_at disagrees with the stored labels at the adopted floor for {len(bad)} events"
    models = {}
    for it in args.models:
        k, v = it.split("=", 1)
        models[k] = {r["event_id"]: float(r["p_nonpspl"]) for r in json.load(open(os.path.join(CURVES, v))) if r.get("dense") and "pred" in r}
    ids = sorted(set(truth) & set.intersection(*[set(v) for v in models.values()]))
    T = [truth[i] for i in ids]; lab = np.array([t["sim_label"] for t in T])
    calib = {}
    for name in models:
        f = os.path.join(HERE, f"gapped_threshold_{name}.json")
        if os.path.exists(f):
            calib[name] = float(json.load(open(f))["arms"]["rmdc26_gapped"]["threshold_at_target_purity"])
    out = {"_doc": __doc__.split("\n")[0], "n_events": len(ids), "adopted_floor": CFG.min_amplitude_mag,
           "precision_note": ("the flagged set is fixed and the NonPSPL label sets are nested as the floor falls, so precision is "
                              "non-increasing in the floor for ANY classifier; it is reported for completeness, not as a test"),
           "dchi2_event": CFG.dchi2_event, "dchi2_anomaly": CFG.dchi2_anomaly, "floors": list(FLOORS), "by_floor": []}
    for fl in FLOORS:
        det = np.array([label_at(t, fl) for t in T])
        row = {"floor_mag": fl, "relabelling": {}, "models": {}}
        for L in (L1, L2, L3):
            s = lab == L
            row["relabelling"][L] = {k: round(float((det[s] == k).mean()), 4) for k in ("Flat", "PSPL", "NonPSPL")}
        bin_det = ((lab == L2) | (lab == L3)) & (det == "NonPSPL"); bin_undet = ((lab == L2) | (lab == L3)) & (det != "NonPSPL")
        for name, d in models.items():
            p = np.array([d[i] for i in ids]); thrs = {"frozen": FROZEN, **({"calibrated_gapped": calib[name]} if name in calib else {})}
            row["models"][name] = {}
            for tn, thr in thrs.items():
                row["models"][name][tn] = {"recall_detectable_binaries": round(float((p[bin_det] >= thr).mean()), 4) if bin_det.sum() else None,
                                           "n_detectable": int(bin_det.sum()),
                                           "flag_rate_undetectable_binaries": round(float((p[bin_undet] >= thr).mean()), 4) if bin_undet.sum() else None,
                                           "fa_1S1L_policy_pspl": round(float((p[(lab == L1) & (det == "PSPL")] >= thr).mean()), 4),
                                           "ontology_precision_sample_mix": round(float((p[det == "NonPSPL"] >= thr).sum() / max((p >= thr).sum(), 1)), 4)}
        out["by_floor"].append(row)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"n={len(ids)}  (adopted floor {CFG.min_amplitude_mag})")
    print(f"{'floor':>6} {'1S2L det':>9} {'2S2L det':>9} | " + " | ".join(f"{m}: rec det / flag undet / precision" for m in models))
    for row in out["by_floor"]:
        cells = []
        for m in models:
            a = row["models"][m]["frozen"]; cells.append(f"{a['recall_detectable_binaries']:.3f} / {a['flag_rate_undetectable_binaries']:.3f} / {a['ontology_precision_sample_mix']:.3f}")
        print(f"{row['floor_mag']:6.3f} {row['relabelling'][L2]['NonPSPL']:9.3f} {row['relabelling'][L3]['NonPSPL']:9.3f} | " + " | ".join(cells))
    print("->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

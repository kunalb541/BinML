"""GULLS/RMDC26 per-epoch F146 noise vs pipeline.photometry.photometric_sigma, from the truth cache.

Reads the detectability_relabel.py rows (which carry each event's median quiescent F146 error, from
GULLS' own flux_err_uJy) and compares it with our noise model at the same baseline magnitude.
Writes gulls_noise_model.json: ratio GULLS/ours by baseline-magnitude bin, and the background
multiplier (SurveyConfig.bkg_mult) and global multiplier (noise_mult) that best reproduce GULLS.
This is the measurement the noise-model ablation (REVISION.md S1.5, experiment 5) is calibrated on.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from pipeline.photometry import ROMAN_BANDS, photometric_sigma  # noqa: E402

TRUTH = os.path.expanduser("~/Desktop/Research/microlensing/gulls_truth_cache")
EDGES = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--truth-cache", default=TRUTH)
    ap.add_argument("--out", default=os.path.join(HERE, "gulls_noise_model.json"))
    args = ap.parse_args(argv)
    m, s = [], []
    for f in sorted(os.listdir(args.truth_cache)):
        if f.startswith("t_") and f.endswith(".json"):
            for r in json.load(open(os.path.join(args.truth_cache, f))):
                q = r.get("sigma_f146_quiescent_median")
                if q and "F146" in r.get("m_base_true", {}):
                    m.append(r["m_base_true"]["F146"]); s.append(q)
    m, s = np.array(m), np.array(s)
    b = ROMAN_BANDS["F146"]
    ours = photometric_sigma(b, m)
    ratio = s / ours
    out = {"_doc": __doc__.split("\n")[0], "n_events": int(m.size), "by_mag": []}
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        sel = (m >= lo) & (m < hi)
        if sel.sum() >= 5:
            out["by_mag"].append({"m_base_bin": [lo, hi], "n": int(sel.sum()),
                                  "gulls_sigma_median": round(float(np.median(s[sel])), 5),
                                  "ours_sigma_median": round(float(np.median(ours[sel])), 5),
                                  "ratio_median": round(float(np.median(ratio[sel])), 3),
                                  "ratio_p16_p84": [round(float(np.percentile(ratio[sel], 16)), 3), round(float(np.percentile(ratio[sel], 84)), 3)]})
    # best single multipliers (least squares in log sigma), for the two knobs
    def fit(kind):
        grid = np.geomspace(0.5, 100, 400)
        best = None
        for k in grid:
            if kind == "bkg":
                import dataclasses
                bb = dataclasses.replace(b, background_e2=b.background_e2 * k)
                pred = photometric_sigma(bb, m)
            else:
                pred = photometric_sigma(b, m, k)
            err = float(np.mean((np.log(s) - np.log(pred)) ** 2))
            if best is None or err < best[1]:
                best = (float(k), err)
        return {"mult": round(best[0], 3), "rms_log_resid": round(float(np.sqrt(best[1])), 4)}
    out["best_fit"] = {"bkg_mult": fit("bkg"), "noise_mult": fit("global")}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"n={m.size}")
    for r in out["by_mag"]:
        print(f"  m_base {r['m_base_bin'][0]}-{r['m_base_bin'][1]}: n={r['n']:5d}  GULLS {r['gulls_sigma_median']:.4f}  ours {r['ours_sigma_median']:.4f}  ratio {r['ratio_median']:.2f}  [{r['ratio_p16_p84'][0]:.2f}, {r['ratio_p16_p84'][1]:.2f}]")
    print("best fit:", out["best_fit"]); print("->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

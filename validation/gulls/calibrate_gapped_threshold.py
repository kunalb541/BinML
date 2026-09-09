"""Re-calibrate the anomaly threshold for Roman's gapped schedule WITHOUT touching GULLS.

The frozen threshold 0.9042 was chosen on gap-free, point-source data. For a gap-aware checkpoint
used on the planned schedule the honest operating point is a threshold chosen the same way the
paper chooses it -- on OUR simulated events, at the target purity -- but with the schedule's gaps
and finite-source single lenses present. Calibrating on GULLS itself would turn the transfer test
into a fit; this script never reads GULLS to choose anything, it only REPORTS what the chosen
threshold does there.

Steps:
  1. copy a held-out memmap and blank the RMDC26 seven-gap schedule into it (bins whose centre
     falls inside a ~6.2 h pause; colour bins by any-overlap, as train._apply_gaps does);
  2. run pipeline.evaluate on the gapped copy for each checkpoint -> headline threshold at the
     target purity (chosen on evaluate's validation rows);
  3. apply that threshold to the GULLS matched rows and report single-lens false alarms and
     planetary recall, next to the frozen-threshold numbers.

Usage:
  python validation/gulls/calibrate_gapped_threshold.py --heldout ~/Desktop/Research/microlensing/fspl_local_work/mm_heldout \
      --ckpt validation/gulls/weights/ft_fspl_g08.pt --tag fspl_g08 --rows ~/Desktop/Research/microlensing/gulls_curve_cache/rows_full_fspl_g08.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

RMDC26_GAPS_D = [0.98, 2.48, 21.49, 31.48, 35.23, 62.23, 69.48]   # validation/gulls/gap_sensitivity.py
RMDC26_GAP_H = 6.2
BAND_BINS = {"F146": 864, "F087": 96, "F213": 96}
FROZEN = 0.9042405486106873


def gap_mask_f146():
    centres = (np.arange(864) + 0.5) * 72.0 / 864
    m = np.zeros(864, bool)
    for g in RMDC26_GAPS_D:
        m |= (centres >= g) & (centres < g + RMDC26_GAP_H / 24.0)
    return m


def make_gapped(src, dst):
    if os.path.exists(os.path.join(dst, "meta.json")):
        return
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith(".npy") or f == "meta.json":
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    n = json.load(open(os.path.join(src, "meta.json")))["n_events"]
    m146 = gap_mask_f146()
    for b, L in BAND_BINS.items():
        mask = m146 if L == 864 else m146.reshape(L, 864 // L).any(axis=1)
        feat = np.memmap(os.path.join(src, f"feat_{b}.f16"), np.float16, "r", shape=(n, L, 3))
        frac = np.memmap(os.path.join(src, f"frac_{b}.f16"), np.float16, "r", shape=(n, L))
        fo = np.memmap(os.path.join(dst, f"feat_{b}.f16"), np.float16, "w+", shape=(n, L, 3))
        fr = np.memmap(os.path.join(dst, f"frac_{b}.f16"), np.float16, "w+", shape=(n, L))
        step = 4096
        for i in range(0, n, step):
            x = np.array(feat[i:i + step]); y = np.array(frac[i:i + step])
            x[:, mask, :] = np.nan; y[:, mask] = 0.0      # an empty bin, exactly as the cache encodes one
            fo[i:i + step] = x; fr[i:i + step] = y
        fo.flush(); fr.flush()
    meta = json.load(open(os.path.join(dst, "meta.json")))
    meta["gapped"] = {"schedule": "RMDC26 seven ~6.2 h pauses", "gaps_d": RMDC26_GAPS_D,
                      "f146_bins_blanked": int(m146.sum())}
    json.dump(meta, open(os.path.join(dst, "meta.json"), "w"), indent=1)


def gulls_at(rows_path, thr):
    rows = [r for r in json.load(open(rows_path)) if r.get("dense") and "pred" in r]
    out = {}
    for lab, key in (("RMDC26_1S1L_ML", "fa_1S1L"), ("RMDC26_1S2L_ML", "recall_1S2L"),
                     ("RMDC26_2S2L_ML", "recall_2S2L")):
        p = np.array([r["p_nonpspl"] for r in rows if r["sim_label"] == lab])
        out[key] = round(float((p >= thr).mean()), 4); out[key + "_n"] = int(p.size)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--heldout", required=True, help="held-out memmap dir (our own simulations)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rows", required=True, help="GULLS rows for this checkpoint (from gulls_transfer.py)")
    ap.add_argument("--target-purity", type=float, default=0.90)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--work", default=None)
    args = ap.parse_args(argv)
    work = args.work or os.path.join(os.path.dirname(args.heldout.rstrip("/")), "calib")
    gapped = os.path.join(work, "mm_heldout_gapped")
    make_gapped(args.heldout, gapped)
    res = {"_doc": __doc__.split("\n")[0], "checkpoint": args.ckpt, "target_purity": args.target_purity,
           "heldout_source": args.heldout, "frozen_threshold": FROZEN, "arms": {}}
    for arm, mm in (("clean", args.heldout), ("rmdc26_gapped", gapped)):
        ev = os.path.join(work, f"eval_{args.tag}_{arm}")
        if not os.path.exists(os.path.join(ev, "metrics.json")):
            subprocess.run([sys.executable, "-m", "pipeline.evaluate", "--ckpt", args.ckpt, "--cache", mm,
                            "--out", ev, "--device", args.device, "--target-purity", str(args.target_purity)],
                           check=True, cwd=REPO, env=dict(os.environ, PYTHONPATH=REPO))
        m = json.load(open(os.path.join(ev, "metrics.json")))
        thr = float(m["headline"]["threshold"])
        res["arms"][arm] = {"threshold_at_target_purity": thr,
                            "our_heldout": {"completeness": round(m["headline"]["completeness_at_fixed_purity"], 4),
                                            "purity": round(m["headline"]["purity_achieved"], 4),
                                            "ap": round(m["average_precision_population"], 4)},
                            "gulls_at_this_threshold": gulls_at(args.rows, thr)}
    res["gulls_at_frozen_threshold"] = gulls_at(args.rows, FROZEN)
    out = os.path.join(HERE, f"gapped_threshold_{args.tag}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k in ("arms", "gulls_at_frozen_threshold")}, indent=1))
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

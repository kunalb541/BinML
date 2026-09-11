"""How much of the RMDC26 transfer result is due to partial bin occupancy -> occupancy_sensitivity.json.

RMDC26's colour visits displace one of eight 15-min F146 epochs in about a third of the model's 2-h
bins (occupancy 7/8), and its ~6.9 h colour cadence leaves about half of the 18-h colour bins at 2/3.
Training never contains either (validation/gulls/rmdc26_schedule.json). This rescoring keeps every
feature of the real RMDC26 input and sets only the occupancy channel of partially filled bins to 1,
then compares the scores. (Imposing the pattern on OUR binned held-out instead is not faithful: it
caps the occupancy without removing an epoch from the bin's features; see pipeline/train.py
_apply_schedule_template.)

Usage:  python validation/gulls/occupancy_sensitivity.py [--every 8] [--max-events 6000] [--models name=weights.pt:rows.json ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
import binml  # noqa: E402
from binml.preprocess import BAND_BINS, to_tokens  # noqa: E402

CURVES = os.path.expanduser("~/Desktop/Research/microlensing/gulls_curve_cache")
FROZEN = 0.9042405486106873
L1, L2, L3 = "RMDC26_1S1L_ML", "RMDC26_1S2L_ML", "RMDC26_2S2L_ML"
MODELS = {"fspl5s_seasons_g08": ("validation/gulls/weights/ft_fspl5s_seasons_g08.pt", "rows_full_fspl5s_seasons_g08.json"),
          "fspl5s_g08": ("validation/gulls/weights/ft_fspl5s_g08.pt", "rows_full_fspl5s_g08.json"),
          "ft_g08e12": ("validation/gulls/weights/ft_g08e12.pt", "rows_full_ft_g08e12_v2.json")}


def wilson(k, n, z=1.96):
    ph = k / n; d = 1 + z * z / n; c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--every", type=int, default=8, help="use every Nth curve-cache chunk")
    ap.add_argument("--max-events", type=int, default=6000)
    ap.add_argument("--out", default=os.path.join(HERE, "occupancy_sensitivity.json"))
    ap.add_argument("--models", nargs="*", default=[], help="name=weights.pt:rows.json (default: the three in MODELS)")
    args = ap.parse_args(argv)
    if args.models:
        MODELS.clear()
        for it in args.models:
            k, v = it.split("=", 1); ck, rowsf = v.split(":", 1); MODELS[k] = (ck, rowsf)
    out = {"_doc": __doc__.split("\n")[0], "command": " ".join(sys.argv), "models": {}}
    for name, (ck, rowsf) in MODELS.items():
        rows = {r["event_id"]: r for r in json.load(open(os.path.join(CURVES, rowsf))) if r.get("dense") and "pred" in r}
        clf = binml.Classifier(weights=os.path.join(REPO, ck)); inon = clf.class_names.index("NonPSPL")
        p0, p1, lab, dev, fr146, frcol = [], [], [], [], [], []
        for f in sorted(glob.glob(os.path.join(CURVES, "c_*.npz")))[::args.every]:
            z = np.load(f, allow_pickle=False)
            for k in z.files:
                if not k.startswith("mb|"):
                    continue
                e = int(k.split("|")[1])
                if e not in rows:
                    continue
                bands = {b: (np.asarray(z[f"b|{e}|{b}|t"], float), np.asarray(z[f"b|{e}|{b}|m"], float))
                         for b in BAND_BINS if f"b|{e}|{b}|t" in z.files}
                tok = to_tokens(bands, m_base_ref=float(z[k][0]), t_start=0.0)
                feats = {b: tok.feat[b][None] for b in BAND_BINS}; fr = {b: tok.frac[b][None] for b in BAND_BINS}
                a = float(clf._forward(feats, fr)[0, inon])
                fr1 = {b: np.where((fr[b] > 0) & (fr[b] < 1), 1.0, fr[b]).astype(fr[b].dtype) for b in BAND_BINS}
                p0.append(a); p1.append(float(clf._forward(feats, fr1)[0, inon])); lab.append(rows[e]["sim_label"])
                dev.append(abs(a - rows[e]["p_nonpspl"]))
                fr146.append(float(((fr["F146"] > 0) & (fr["F146"] < 1)).sum() / max((fr["F146"] > 0).sum(), 1)))
                frcol.append(float(np.mean([((fr[b] > 0) & (fr[b] < 1)).sum() / max((fr[b] > 0).sum(), 1) for b in ("F087", "F213")])))
            if len(p0) >= args.max_events:
                break
        p0, p1, lab = np.array(p0), np.array(p1), np.array(lab)
        blk = {"n": int(p0.size), "max_abs_dev_from_committed_rows": round(float(np.max(dev)), 6),
               "median_share_partial_bins": {"F146": round(float(np.median(fr146)), 4), "colour": round(float(np.median(frcol)), 4)}}
        for L, key in ((L1, "fa_1S1L"), (L2, "recall_1S2L"), (L3, "recall_2S2L")):
            s = lab == L; k0, k1, n = int((p0[s] >= FROZEN).sum()), int((p1[s] >= FROZEN).sum()), int(s.sum())
            blk[key] = {"n": n, "k_as_observed": k0, "k_occupancy_set_to_1": k1,
                        "as_observed": round(k0 / n, 4), "as_observed_wilson95": wilson(k0, n),
                        "occupancy_set_to_1": round(k1 / n, 4), "occupancy_set_to_1_wilson95": wilson(k1, n),
                        "median_p_as_observed": round(float(np.median(p0[s])), 4), "median_p_occupancy_1": round(float(np.median(p1[s])), 4)}
        out["models"][name] = blk
        print(name, json.dumps(blk))
    json.dump(out, open(args.out, "w"), indent=1)
    print("->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

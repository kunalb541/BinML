#!/usr/bin/env python3
"""EXPERIMENTAL -- replay RMDC26 as if it arrived in daily batches, the way IPAC's GBTDS photometry pipeline
plans to update light curves (daily, within 48 h; see experimental/deployment/README.md).

For a sample of scored RMDC26 events (the local curve cache; no network), each simulated day reveals one more
day of each event's season, rescores every event that received data with the recommended gap-aware
checkpoint, and puts an event on the watchlist the first day its P(NonPSPL) reaches the threshold. Reports
the daily alert load per 1,000 monitored events, the share of single-lens and planetary events alerted by
season end, the median alert day, and the CPU time per daily batch. This is the operational framing of the
cascade analysis in validation/gulls/cascade_gulls.py (which uses half-day cuts and onsets from the truth
curves); nothing here is part of the paper.

Usage:  python experimental/deployment/replay_rmdc26.py [--per-class 300] [--threshold 0.9563]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
CACHE = os.path.expanduser("~/Desktop/Research/microlensing/gulls_curve_cache")
WEIGHTS = os.path.join(REPO, "validation/gulls/weights/ft_fspl5s_seasons_g08.pt")
ROWS = os.path.join(CACHE, "rows_full_fspl5s_seasons_g08.json")
CLASSES = ("RMDC26_1S1L_ML", "RMDC26_1S2L_ML", "RMDC26_2S2L_ML")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--threshold", type=float, default=None, help="default: the recommended checkpoint's calibrated threshold")
    ap.add_argument("--bands", default="F146", help="F146 (the cascade's primary protocol) or F146,F087,F213")
    ap.add_argument("--out", default=os.path.join(REPO, "experimental/deployment/replay_results.json"))
    args = ap.parse_args(argv)
    import torch
    import binml
    from binml.preprocess import BAND_BINS, to_tokens
    torch.set_num_threads(1)
    thr = args.threshold
    if thr is None:
        cal = json.load(open(os.path.join(REPO, "validation/gulls/gapped_threshold_fspl5s_seasons_g08_seasons.json")))
        thr = float(cal["arms"]["rmdc26_gapped"]["pool"]["full_pool"]["threshold"])
    lab = {r["event_id"]: r["sim_label"] for r in json.load(open(ROWS)) if r.get("dense") and "pred" in r}
    want = {L: sorted(e for e, l in lab.items() if l == L)[:args.per_class] for L in CLASSES}
    ids = {e for v in want.values() for e in v}
    curves = {}
    for f in sorted(glob.glob(os.path.join(CACHE, "c_*.npz"))):
        z = np.load(f, allow_pickle=False)
        for k in z.files:
            if k.startswith("mb|") and int(k.split("|")[1]) in ids:
                e = int(k.split("|")[1])
                curves[e] = ({b: (np.asarray(z[f"b|{e}|{b}|t"], float), np.asarray(z[f"b|{e}|{b}|m"], float))
                              for b in args.bands.split(",") if f"b|{e}|{b}|t" in z.files}, float(z[k][0]))
    clf = binml.Classifier(weights=WEIGHTS, device="cpu"); inon = clf.class_names.index("NonPSPL")
    first = {e: None for e in curves}; cpu = []
    for day in range(1, 73):
        t0 = time.process_time(); feats = {b: [] for b in BAND_BINS}; fracs = {b: [] for b in BAND_BINS}; batch = []
        for e, (bands, mb) in curves.items():
            if first[e] is not None:
                continue                                     # already on the watchlist
            rev = {b: (t[t <= day], m[t <= day]) for b, (t, m) in bands.items()}
            if rev["F146"][0].size < 10 or not np.any(bands["F146"][0][(bands["F146"][0] > day - 1) & (bands["F146"][0] <= day)].size):
                continue                                     # no new data today (pause, or season not started)
            tok = to_tokens(rev, m_base_ref=mb, t_start=0.0)
            for b in BAND_BINS:
                feats[b].append(tok.feat[b]); fracs[b].append(tok.frac[b])
            batch.append(e)
        if batch:
            p = clf._forward({b: np.stack(feats[b]) for b in BAND_BINS}, {b: np.stack(fracs[b]) for b in BAND_BINS})[:, inon]
            for e, pe in zip(batch, p):
                if pe >= thr:
                    first[e] = day
        cpu.append({"day": day, "n_scored": len(batch), "cpu_s": round(time.process_time() - t0, 3)})
    out = {"_doc": __doc__.split("\n")[0], "threshold": thr, "bands": args.bands, "n_events": {L: sum(1 for e in curves if lab[e] == L) for L in CLASSES},
           "alerted_by_season_end": {}, "median_alert_day": {}, "alerts_per_1000_per_day_single_lens": None, "cpu": cpu}
    for L in CLASSES:
        d = [first[e] for e in curves if lab[e] == L]
        a = [x for x in d if x is not None]
        out["alerted_by_season_end"][L] = round(len(a) / max(len(d), 1), 4)
        out["median_alert_day"][L] = float(np.median(a)) if a else None
    n1 = out["n_events"]["RMDC26_1S1L_ML"]
    out["alerts_per_1000_per_day_single_lens"] = round(1000 * out["alerted_by_season_end"]["RMDC26_1S1L_ML"] / 72.0, 3) if n1 else None
    out["cpu_s_per_1000_rescored"] = round(1000 * sum(c["cpu_s"] for c in cpu) / max(sum(c["n_scored"] for c in cpu), 1), 2)
    out["command"] = " ".join(sys.argv)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "cpu"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())

"""The RMDC26 observing schedule on BinML's grid, season by season -> rmdc26_schedule.json.

Why this exists. The first schedule work in this repo hard-coded ONE set of seven F146 pauses
(pipeline/train.py RMDC26_GAPS_D), measured on the first high-cadence season and described as
"the" schedule at "fixed season phases". The 2026-09-11 verification found that the pause phases
differ from season to season (only the pauses near days 1.0, 35.2 and 69.5 recur; three seasons
merge two pauses into one ~12 h pause), so that mask matches only the first season, which holds
16% of the scored RMDC26 events. It also found that RMDC26's colour visits displace F146 epochs,
leaving one of eight 15-min epochs empty in about a third of F146 bins (frac 0.875), a value the
training data never contains.

What it writes, per dense season (0-indexed as in the epoch table):
  pauses          gaps > 1 h between consecutive epochs (any band), in days from season start
                  and hours of duration (from the pinned epoch table);
  empty_bins      per band, the bins of BinML's grid (F146 864, F087/F213 96) that contain no
                  observation in ANY of the sampled events of that season (so per-event SNR and
                  saturation drop-outs do not count);
  frac_template   per band, the maximum over sampled events of the per-bin occupancy fraction
                  that binml.preprocess.bin_band produces (1.0 on a continuous 15-min grid);
  n_events        how many cached events were used.
The templates come from the curve cache written by validation/gulls_transfer.py (dense events,
windows starting at the season start, as the model sees them).

Usage:  python validation/gulls/rmdc26_schedule.py [--per-season 40]
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
from binml.preprocess import BAND_BINS, bin_band  # noqa: E402

CURVES = os.path.expanduser("~/Desktop/Research/microlensing/gulls_curve_cache")
EPOCH = "/tmp/rmdc26_epoch.parquet"
META = "/tmp/rmdc26_meta.parquet"
SEASON_GAP_D = 5.0
PAUSE_MIN_H = 1.0


def seasons(bjd):
    b = np.sort(np.unique(np.round(bjd, 6)))
    cut = np.flatnonzero(np.diff(b) > SEASON_GAP_D)
    starts, ends = np.r_[b[0], b[cut + 1]], np.r_[b[cut], b[-1]]
    out = []
    for i, (a, e) in enumerate(zip(starts, ends)):
        s = b[(b >= a) & (b <= e)]
        d = np.diff(s)
        k = np.flatnonzero(d > PAUSE_MIN_H / 24.0)
        out.append({"index": i, "start_bjd": float(a), "end_bjd": float(e), "length_days": round(float(e - a), 3),
                    "n_unique_epochs": int(s.size), "dense": bool(s.size > 3000),
                    "pauses": [{"start_day": round(float(s[j] - a), 3), "hours": round(float(d[j] * 24), 2)} for j in k]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-season", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(HERE, "rmdc26_schedule.json"))
    args = ap.parse_args(argv)
    import pyarrow.parquet as pq
    ep = pq.read_table(EPOCH, columns=["bjd"]).to_pydict()["bjd"]
    S = seasons(np.asarray(ep, float))
    meta = pq.read_table(META, columns=["event_id", "t0lens1"]).to_pydict()
    t0 = dict(zip(meta["event_id"], meta["t0lens1"]))
    dense = [s for s in S if s["dense"]]

    def season_of(t):
        for s in S:
            if s["start_bjd"] <= t <= s["end_bjd"]:
                return s["index"]
        return None

    got = {s["index"]: [] for s in dense}
    for f in sorted(glob.glob(os.path.join(CURVES, "c_*.npz"))):
        if all(len(v) >= args.per_season for v in got.values()):
            break
        z = np.load(f, allow_pickle=False)
        for k in z.files:
            if not k.startswith("mb|"):
                continue
            e = int(k.split("|")[1]); si = season_of(float(t0[e]))
            if si not in got or len(got[si]) >= args.per_season or f"b|{e}|F146|t" not in z.files:
                continue
            bands = {bd: np.asarray(z[f"b|{e}|{bd}|t"], float) for bd in BAND_BINS if f"b|{e}|{bd}|t" in z.files}
            if bands["F146"].size < 1000:
                continue
            got[si].append(bands)
    for s in dense:
        evs = got[s["index"]]
        s["n_events"] = len(evs)
        s["empty_bins"], s["frac_template"] = {}, {}
        for bd, L in BAND_BINS.items():
            fr = np.zeros((max(len(evs), 1), L), np.float32)
            for i, bands in enumerate(evs):
                if bd in bands:
                    _, fr[i], _ = bin_band(bands[bd], np.zeros_like(bands[bd]), bd, 0.0, 0.0)
            tmpl = fr.max(axis=0)
            s["frac_template"][bd] = [round(float(x), 4) for x in tmpl]
            s["empty_bins"][bd] = [int(i) for i in np.flatnonzero(tmpl == 0)]
        f146 = np.asarray(s["frac_template"]["F146"])
        s["summary"] = {"f146_empty_bins": int((f146 == 0).sum()), "f146_bins_frac_lt_1": int(((f146 > 0) & (f146 < 1)).sum()),
                        "f087_empty_bins": len(s["empty_bins"]["F087"]), "f213_empty_bins": len(s["empty_bins"]["F213"]),
                        "pause_hours_total": round(sum(p["hours"] for p in s["pauses"]), 2), "n_pauses": len(s["pauses"])}
    out = {"_doc": __doc__.split("\n")[0], "epoch_table": "RGES-PIT/MachineLearning RMDC26_ML_Data_epoch.parquet @ a338d5ba",
           "pause_min_hours": PAUSE_MIN_H, "grid": {"window_days": 72.0, "bins": BAND_BINS}, "seasons": S,
           "dense_season_indices": [s["index"] for s in dense]}
    json.dump(out, open(args.out, "w"), indent=1)
    for s in S:
        line = f"season {s['index']}: {s['length_days']:6.2f} d, {s['n_unique_epochs']:6d} epochs, {'dense' if s['dense'] else 'low-cadence'}"
        if s["dense"]:
            p = "; ".join(f"{q['start_day']:.2f}d/{q['hours']:.1f}h" for q in s["pauses"])
            line += f" | pauses {p} | {s['summary']}"
        print(line)
    print("->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

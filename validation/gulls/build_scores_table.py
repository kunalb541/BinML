"""Compact, committed per-event RMDC26 score table -> validation/gulls/rmdc26_scores.csv.gz.

The transfer tables were computed from ~43 MB per-checkpoint rows files kept OUTSIDE the repository
(the curve cache), so the committed artifacts could not be regenerated from a clone (2026-09-11
verification). This writes one row per scored (dense, in-season) event with the metadata the tables
use and P(NonPSPL) for every checkpoint scored so far, rounded as the rows store it (4 d.p.; newer
rows 6), with rho and u0 at full precision so the rho/|u0| bins are reproduced exactly.
`gulls_summary_tables.py --scores validation/gulls/rmdc26_scores.csv.gz` regenerates transfer_tradeoff_all.json and
transfer_colour_ablation.json from it alone.

Usage:  python validation/gulls/build_scores_table.py
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CURVES = os.path.expanduser("~/Desktop/Research/microlensing/gulls_curve_cache")
META = "/tmp/rmdc26_meta.parquet"
CHECKPOINTS = {
    "shipped": "rows_full_shipped_v2.json", "ft_g08e12": "rows_full_ft_g08e12_v2.json",
    "fspl_g08": "rows_full_fspl_g08.json", "fspl5_g08": "rows_full_fspl5_g08.json", "fspl5s_g08": "rows_full_fspl5s_g08.json",
    "fspl5s_noisy_g08": "rows_full_fspl5s_noisy_g08.json", "fspl5s_v2_g08": "rows_full_fspl5s_v2_g08.json",
    "pspl5s_ctrl_g08": "rows_full_pspl5s_ctrl_g08.json", "fspl5s_espl_g08": "rows_full_fspl5s_espl_g08.json",
    "fspl5s_seasons_g08": "rows_full_fspl5s_seasons_g08.json",
    "sched_rand": "rows_full_sched_rand.json", "sched_sched": "rows_full_sched_sched.json",
    "sched_sched_norelabel": "rows_full_sched_sched_norelabel.json", "sched_rand_norelabel": "rows_full_sched_rand_norelabel.json",
    "sched_sched_seasons": "rows_full_sched_sched_seasons.json",
    "fspl_g08_f146only": "rows_full_fspl_g08_f146only.json", "ft_g08e12_f146only": "rows_full_ft_g08e12_f146only.json",
    "fspl5s_g08_f146only": "rows_full_fspl5s_g08_f146only.json",
}


def main():
    import pyarrow.parquet as pq
    m = pq.read_table(META, columns=["event_id", "t0lens1", "rho", "u0lens1", "Planet_q", "Source_Is_Binary"]).to_pandas().set_index("event_id")
    S = json.load(open(os.path.join(HERE, "rmdc26_schedule.json")))["seasons"]
    season_of = lambda t: next((x["index"] for x in S if x["start_bjd"] <= t <= x["end_bjd"]), -1)
    cols, base = [], None
    for name, f in CHECKPOINTS.items():
        path = os.path.join(CURVES, f)
        if not os.path.exists(path):
            print(f"[skip] {name}: {f} not found"); continue
        rows = {r["event_id"]: r for r in json.load(open(path)) if r.get("dense") and "pred" in r}
        if base is None:
            base = rows
        cols.append((name, {e: r["p_nonpspl"] for e, r in rows.items()}))
    ids = sorted(base)
    out = os.path.join(HERE, "rmdc26_scores.csv.gz")
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["event_id", "sim_label", "season", "tE", "u0", "rho", "planet_q", "source_is_binary", "m_base", "weight"] + [c for c, _ in cols])
        for e in ids:
            r = base[e]; mm = m.loc[e]
            w.writerow([e, r["sim_label"], season_of(float(mm["t0lens1"])), round(r["tE"], 5), repr(float(mm["u0lens1"])),
                        repr(float(mm["rho"])), "" if mm["Planet_q"] != mm["Planet_q"] else f"{float(mm['Planet_q']):.4e}",
                        "" if mm["Source_Is_Binary"] != mm["Source_Is_Binary"] else int(mm["Source_Is_Binary"]),
                        r["m_base"], f"{r['weight']:.6g}"] + [d.get(e, "") for _, d in cols])
    print(f"wrote {out}: {len(ids)} events x {len(cols)} checkpoints ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())

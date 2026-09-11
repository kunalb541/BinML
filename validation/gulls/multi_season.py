"""Peak-in-gap GULLS events: score the ADJACENT seasons and combine -- the single-season contract's cost, measured.

BinML's input is one contiguous season. In RMDC26 the mission observes 685 of 1,717 days in ten
seasons separated by 109-120 d gaps, so 25.5% of the amplitude- and t_E-eligible GULLS events
have their peak inside an inter-season gap and were skipped by every transfer run ("t0 falls in an
inter-season gap"). This script asks what the model does with them WITHOUT retraining: for each
such event, score every DENSE season adjacent to the gap (the season ending before t0, the season
starting after it), take the maximum anomaly probability across them, and report single-lens
false alarms and planetary recall on this population next to the in-season numbers. The truth
policy (detectability_relabel._stats) is applied per season too, so recall can be quoted on
binaries whose anomaly is detectable in at least one adjacent season.

Adjacent low-cadence seasons (65 d, 3 epochs/day) are out of BinML's support and are counted, not
scored. Events with no dense adjacent season are reported as unscorable.

Resumable per block. Usage:
  python validation/gulls/multi_season.py --extract --cap 1000
  python validation/gulls/multi_season.py --reduce
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import gulls_transfer as gt                          # noqa: E402
from detectability_relabel import _stats, L1, L2, L3  # noqa: E402

CURVES = os.path.expanduser("~/Desktop/Research/microlensing/gulls_curve_cache")
OUT_CACHE = os.path.expanduser("~/Desktop/Research/microlensing/gulls_multiseason_cache")
FROZEN = 0.9042405486106873
MODELS = {"fspl5s_g08": "validation/gulls/weights/ft_fspl5s_g08.pt",
          "ft_g08e12": "validation/gulls/weights/ft_g08e12.pt"}
SKIP = "t0 falls in an inter-season gap"


def extract(args):
    import duckdb
    import pyarrow.parquet as pq
    import binml
    os.makedirs(args.out_cache, exist_ok=True)
    ref = json.load(open(os.path.join(CURVES, args.ref_rows)))
    gap_ids = sorted({r["event_id"] for r in ref if r.get("skipped") == SKIP})
    m = pq.read_table(gt._fetch(gt.META, args.meta_cache, "meta"), columns=[
        "event_id", "sim_label", "t0lens1", "tE_ref", "u0lens1", "rho", "fs_F146", "final_weight"]).to_pydict()
    pos = {int(e): i for i, e in enumerate(np.asarray(m["event_id"], np.int64))}
    by = {}
    for e in gap_ids:
        by.setdefault(m["sim_label"][pos[e]], []).append(e)
    picks = []
    for lab in (L2, L3, L1):
        ids = sorted(by.get(lab, []))[:args.cap] if args.cap else sorted(by.get(lab, []))
        print(f"[select] {lab}: {len(ids)} of {len(by.get(lab, []))} peak-in-gap events, ids {ids[0]}-{ids[-1]}", flush=True)
        picks.extend(ids)
    et = pq.read_table(gt._fetch(gt.EPOCH, args.epoch_cache, "epoch"), columns=["epoch_id", "bjd"]).to_pydict()
    ep_id = np.asarray(et["epoch_id"], np.int64); ep_bjd = np.asarray(et["bjd"], float)
    o = np.argsort(ep_bjd); ep_id, ep_bjd = ep_id[o], ep_bjd[o]
    io = np.argsort(ep_id); ids_s, bjd_s = ep_id[io], ep_bjd[io]

    def bjd_lookup(q):
        p = np.searchsorted(ids_s, q); np.clip(p, 0, len(ids_s) - 1, out=p)
        return np.where(ids_s[p] == q, bjd_s[p], np.nan)

    gaps = np.flatnonzero(np.diff(ep_bjd) > gt.SEASON_GAP_D)
    starts = np.r_[ep_bjd[0], ep_bjd[gaps + 1]]; ends = np.r_[ep_bjd[gaps], ep_bjd[-1]]
    n_ep = np.array([((ep_bjd >= a) & (ep_bjd <= b)).sum() for a, b in zip(starts, ends)])
    dense_season = n_ep >= 3 * gt.DENSE_MIN          # all three bands counted here; dense seasons have ~16k epochs
    print(f"[seasons] {len(starts)}; dense: {dense_season.astype(int).tolist()}", flush=True)

    def adjacent(t0):
        before = np.flatnonzero(ends < t0); after = np.flatnonzero(starts > t0)
        out = []
        if before.size: out.append(("before", int(before[-1])))
        if after.size: out.append(("after", int(after[0])))
        return out

    clfs = {k: binml.Classifier(weights=os.path.join(REPO, v)) for k, v in MODELS.items()}
    con = duckdb.connect(); con.execute("INSTALL httpfs; LOAD httpfs;")
    pool = ProcessPoolExecutor(max_workers=args.workers)
    t_start, n_done, n_new = time.time(), 0, 0
    for c0 in range(0, len(picks), args.chunk):
        ids = picks[c0:c0 + args.chunk]
        outf = os.path.join(args.out_cache, f"ms_{ids[0]}_{ids[-1]}_{len(ids)}.json")
        n_done += len(ids)
        if os.path.exists(outf):
            continue
        if args.max_blocks and n_new >= args.max_blocks:
            break
        n_new += 1
        try:
            q = con.execute(
                f"SELECT event_id, epoch_id, filt, flux_uJy, true_flux_uJy, flux_err_uJy FROM read_parquet('{gt.OBS}') "
                f"WHERE event_id BETWEEN {ids[0]} AND {ids[-1]} AND event_id IN ({','.join(map(str, ids))}) "
                f"AND saturation_flag = 0").fetchnumpy()
        except Exception as exc:
            print(f"[query] block {ids[0]}-{ids[-1]} failed: {str(exc)[:160]}", flush=True); time.sleep(20); continue
        qe = np.asarray(q["event_id"], np.int64); srt = np.argsort(qe, kind="stable"); qe = qe[srt]
        q_ep = np.asarray(q["epoch_id"], np.int64)[srt]; q_ft = np.asarray(q["filt"])[srt]
        q_fl = np.asarray(q["flux_uJy"], float)[srt]; q_tf = np.asarray(q["true_flux_uJy"], float)[srt]; q_er = np.asarray(q["flux_err_uJy"], float)[srt]
        lo_i = np.searchsorted(qe, ids, "left"); hi_i = np.searchsorted(qe, ids, "right")
        rows, truth_jobs = [], []
        for k, eid in enumerate(ids):
            j = pos[eid]; t0 = float(m["t0lens1"][j]); tE = float(m["tE_ref"][j])
            a, b_ = int(lo_i[k]), int(hi_i[k])
            row = {"event_id": eid, "sim_label": m["sim_label"][j], "weight": float(m["final_weight"][j]),
                   "tE": tE, "u0": float(m["u0lens1"][j]), "rho": float(m["rho"][j]), "t0": t0, "seasons": {}}
            if a == b_:
                row["skipped"] = "no unsaturated photometry"; rows.append(row); continue
            bjd = bjd_lookup(q_ep[a:b_]); ft = q_ft[a:b_]; fl = q_fl[a:b_]; tf = q_tf[a:b_]; er = q_er[a:b_]
            off_all = np.isfinite(bjd) & (np.abs(bjd - t0) > 5.0 * tE)
            for side, si in adjacent(t0):
                sd = {"side": side, "season_index": si, "dense": bool(dense_season[si]),
                      "gap_days_to_t0": round(float(t0 - ends[si]) if side == "before" else float(starts[si] - t0), 2)}
                if not dense_season[si]:
                    row["seasons"][side] = sd; continue
                lo_t, hi_t = starts[si], min(ends[si], starts[si] + gt.WINDOW_D)
                in_win = np.isfinite(bjd) & (bjd >= lo_t) & (bjd <= hi_t)
                bands, tbands = {}, {}
                for bd in ("F146", "F087", "F213"):
                    sb = in_win & (ft == bd); so = off_all & (ft == bd)
                    mag = gt.flux_to_ab(fl[sb]); g = np.isfinite(mag)
                    if g.sum() >= 10 and (so & np.isfinite(gt.flux_to_ab(fl))).sum() >= (200 if bd == "F146" else 20):
                        ordt = np.argsort(bjd[sb][g]); bands[bd] = ((bjd[sb][g] - lo_t)[ordt], mag[g][ordt])
                    ok = sb & np.isfinite(tf) & (tf > 0) & (er > 0) & (tf / er >= 3)
                    sot = so & np.isfinite(tf) & (tf > 0)
                    if ok.sum() >= 10 and sot.sum() >= (200 if bd == "F146" else 20):
                        ordt = np.argsort(bjd[ok])
                        tbands[bd] = ((bjd[ok] - lo_t)[ordt], gt.flux_to_ab(tf[ok])[ordt], (1.0857 * er[ok] / tf[ok])[ordt],
                                      float(np.median(gt.flux_to_ab(tf[sot]))))
                if "F146" not in bands:
                    sd["skipped"] = "no usable F146 in season"; row["seasons"][side] = sd; continue
                mo = off_all & (ft == "F146"); mall = gt.flux_to_ab(fl[mo]); mb = float(np.median(mall[np.isfinite(mall)]))
                sd["m_base"] = round(mb, 3); sd["n_f146"] = int(bands["F146"][0].size)
                for name, clf in clfs.items():
                    p = clf.predict(bands, m_base_ref=mb, t_start=0.0)
                    sd[name] = {"pred": p.label, "p_nonpspl": round(p.probabilities["NonPSPL"], 4)}
                if "F146" in tbands:
                    truth_jobs.append((eid, side, {"event_id": eid, "sim_label": row["sim_label"], "weight": row["weight"],
                                                   "tE": tE, "u0": row["u0"], "rho": row["rho"], "fs": float(m["fs_F146"][j]),
                                                   "t0_win": t0 - lo_t, "bands": tbands}))
                row["seasons"][side] = sd
            rows.append(row)
        res = list(pool.map(_stats, [x[2] for x in truth_jobs], chunksize=8))
        by_id = {r["event_id"]: r for r in rows}
        for (eid, side, _), st in zip(truth_jobs, res):
            by_id[eid]["seasons"][side]["truth"] = {k: st[k] for k in ("dchi2_event", "max_amp_mag", "dchi2_anomaly", "anomaly_amp_mag", "label_detect")}
        json.dump(rows, open(outf + ".tmp", "w")); os.replace(outf + ".tmp", outf)
        print(f"  {n_done}/{len(picks)}  ({time.time() - t_start:.0f}s)", flush=True)
    pool.shutdown(); print("[extract] done", flush=True)


def reduce(args):
    rows = []
    for f in sorted(os.listdir(args.out_cache)):
        if f.startswith("ms_") and f.endswith(".json"):
            rows += json.load(open(os.path.join(args.out_cache, f)))
    calib = {}
    for name in MODELS:
        f = os.path.join(HERE, f"gapped_threshold_{name}.json")
        if os.path.exists(f):
            calib[name] = float(json.load(open(f))["arms"]["rmdc26_gapped"]["threshold_at_target_purity"])
    out = {"_doc": __doc__.split("\n")[0], "n_events": len(rows), "population": "amplitude- and t_E-eligible events whose t0 falls in an inter-season gap (25.5% of the eligible set)",
           "combiner": "max p_nonpspl over dense adjacent seasons; truth = NonPSPL if any adjacent dense season has a detectable anomaly, PSPL if any is a detectable event, else Flat",
           "by_class": {}, "models": {}}
    def scored(r):
        return [s for s in r["seasons"].values() if s.get("dense") and "fspl5s_g08" in s]
    for L in (L1, L2, L3):
        rs = [r for r in rows if r["sim_label"] == L]
        sc = [r for r in rs if scored(r)]
        both = [r for r in sc if len(scored(r)) == 2]
        det = [r for r in sc if any(s.get("truth", {}).get("label_detect") == "NonPSPL" for s in scored(r))]
        ev = [r for r in sc if any(s.get("truth", {}).get("label_detect") in ("NonPSPL", "PSPL") for s in scored(r))]
        out["by_class"][L] = {"n": len(rs), "n_with_dense_adjacent_season": len(sc), "n_with_two_dense_seasons": len(both),
                              "frac_scorable": round(len(sc) / max(len(rs), 1), 4),
                              "frac_detectable_event_in_adjacent_season": round(len(ev) / max(len(sc), 1), 4),
                              "frac_detectable_anomaly_in_adjacent_season": round(len(det) / max(len(sc), 1), 4),
                              "median_gap_days_to_t0": round(float(np.median([min(s["gap_days_to_t0"] for s in scored(r)) for r in sc])), 1) if sc else None}
    s1 = [r for r in rows if r["sim_label"] == L1 and scored(r)]
    for name in MODELS:
        pm = lambda r: max(s[name]["p_nonpspl"] for s in scored(r))
        p1 = np.array([pm(r) for r in s1])
        blk = {"thresholds": {"frozen": FROZEN, **({"calibrated_gapped": calib[name]} if name in calib else {})}, "at_threshold": {}}
        for tn, thr in blk["thresholds"].items():
            d = {"fa_1S1L": round(float((p1 >= thr).mean()), 4), "n_1S1L": int(p1.size)}
            for L, key in ((L2, "1S2L"), (L3, "2S2L")):
                rs = [r for r in rows if r["sim_label"] == L and scored(r)]
                p = np.array([pm(r) for r in rs]); det = np.array([any(s.get("truth", {}).get("label_detect") == "NonPSPL" for s in scored(r)) for r in rs])
                d[f"recall_{key}_generator"] = round(float((p >= thr).mean()), 4) if p.size else None
                d[f"recall_{key}_detectable"] = round(float((p[det] >= thr).mean()), 4) if det.sum() else None
                d[f"n_{key}"] = int(p.size); d[f"n_{key}_detectable"] = int(det.sum())
            blk["at_threshold"][tn] = d
        # matched budgets on THIS population
        ths = np.unique(np.round(p1, 4))[::-1]; fa = np.array([(p1 >= t).mean() for t in ths])
        rs2 = [r for r in rows if r["sim_label"] == L2 and scored(r)]; p2 = np.array([pm(r) for r in rs2])
        det2 = np.array([any(s.get("truth", {}).get("label_detect") == "NonPSPL" for s in scored(r)) for r in rs2], bool)
        blk["recall_1S2L_at_matched_fa"] = []
        for tgt in ((0.02, 0.052, 0.117) if p1.size and p2.size else ()):
            i = int(np.argmin(np.abs(fa - tgt)))
            blk["recall_1S2L_at_matched_fa"].append({"fa_target": tgt, "fa": round(float(fa[i]), 4), "threshold": float(ths[i]),
                                                     "recall_generator": round(float((p2 >= ths[i]).mean()), 4),
                                                     "recall_detectable": round(float((p2[det2] >= ths[i]).mean()), 4) if det2.sum() else None})
        out["models"][name] = blk
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out["by_class"], indent=1))
    for name, b in out["models"].items():
        for tn, d in b["at_threshold"].items():
            print(f"{name:12s} {tn[:8]:>8} FA {d['fa_1S1L']:.4f} (n={d['n_1S1L']})  1S2L gen {d['recall_1S2L_generator']} det {d['recall_1S2L_detectable']}  2S2L gen {d['recall_2S2L_generator']} det {d['recall_2S2L_detectable']}")
        print("   matched:", [(a['fa'], a['recall_generator'], a['recall_detectable']) for a in b["recall_1S2L_at_matched_fa"]])
    print("->", args.out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--extract", action="store_true"); ap.add_argument("--reduce", action="store_true")
    ap.add_argument("--ref-rows", default="rows_full_fspl5s_g08.json")
    ap.add_argument("--cap", type=int, default=1000, help="first N peak-in-gap ids per class (0 = all)")
    ap.add_argument("--chunk", type=int, default=200); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-blocks", type=int, default=0)
    ap.add_argument("--out-cache", default=OUT_CACHE)
    ap.add_argument("--meta-cache", default="/tmp/rmdc26_meta.parquet"); ap.add_argument("--epoch-cache", default="/tmp/rmdc26_epoch.parquet")
    ap.add_argument("--out", default=os.path.join(HERE, "transfer_multiseason.json"))
    args = ap.parse_args(argv)
    if args.extract: extract(args)
    if args.reduce: reduce(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

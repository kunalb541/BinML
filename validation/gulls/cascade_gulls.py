"""The real-time cascade on an independent simulation: half-day scan of GULLS/RMDC26 seasons.

The manuscript's partial-light-curve numbers (premature-alert rate, within-season detection,
lag) come from our own simulator only. This scan repeats the paper's PRIMARY protocol on
RMDC26 events from the curve cache: reveal each 72-day window in 144 half-day cuts, score every
cut, and record the first threshold crossing. Two things the in-house scan could not give:

  * an alert BURDEN on the dominant class -- single lenses are scanned too, so alerts per event
    per season and per 1,000 events per day are measured on contaminants, and streaming purity
    can be quoted at a stated planetary prevalence (RMDC26's rate weights over-represent planets:
    22% of the rate-weighted eligible set is planetary, so purity is also given at 1% and 5%);
  * an onset from a simulator we did not write: the truth-informed first-detectable onset is
    recomputed on the noise-free `true_flux_uJy` curve with our own label rule at every half-day
    cut (validation/cascade_reduce.py's definition), using GULLS' own errors as sigma.

Eligibility for the timing numbers, as in the paper: generated as a binary lens, anomaly
detectable over the full window by our rule, finite onset. Nothing is selected on the model's
output. Bands: F146-only (the paper's primary protocol) and three-band, both reported.

Usage:
  python validation/gulls/cascade_gulls.py --extract --cap-binary 1500          # true curves (network)
  python validation/gulls/cascade_gulls.py --scan --cap-1s1l 3000 --cap-binary 1500 --workers 6
  python validation/gulls/cascade_gulls.py --reduce
"""
from __future__ import annotations

import argparse
import glob
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
import gulls_transfer as gt                                        # noqa: E402
from detectability_relabel import L1, L2, L3, CURVES, FROZEN, CFG  # noqa: E402
from pipeline.assemble import _pspl_refit_dchi2                    # noqa: E402

CACHE = os.path.expanduser("~/Desktop/Research/microlensing/gulls_cascade_cache")   # truth_*.npz reused; scan_v2_*.npz rescanned
STEP, N_CUTS = 0.5, 144
CKPT = {"fspl5s_g08": os.path.join(REPO, "validation/gulls/weights/ft_fspl5s_g08.pt")}
PREVALENCES = (0.01, 0.05)
# final_weight planetary fraction of the 56,975 scored (dense, in-season) RMDC26 events; recomputed from
# rows_full_fspl5s_g08.json in 2026-09-11's verification (22.07%). Not the eligible set (10.8%) and not the
# 3,000:3,000 scan sample (30.7%, the value the first version of this reducer used).
SCORED_PLANET_FRAC_WEIGHTED = 0.2207
INHOUSE = os.path.join(os.path.dirname(HERE), "cascade_reproduce_result.json")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def sample_ids(ref_rows, cap_1s1l, cap_binary):
    ref = json.load(open(os.path.join(CURVES, ref_rows)))
    by = {}
    for r in ref:
        if r.get("dense") and "pred" in r:
            by.setdefault(r["sim_label"], []).append(r["event_id"])
    out = {}
    for lab, cap in ((L2, cap_binary), (L3, cap_binary), (L1, cap_1s1l)):
        ids = sorted(by.get(lab, []))
        out[lab] = ids[:cap] if cap else ids
    return out


# ------------------------------------------------------------------ curve-cache access
def curve_index(ids):
    """event_id -> chunk file, from the curve cache written by gulls_transfer.py."""
    want = set(ids); idx = {}
    for f in glob.glob(os.path.join(CURVES, "c_*.npz")):
        lo, hi = int(os.path.basename(f).split("_")[1]), int(os.path.basename(f).split("_")[2])
        if not any(lo <= i <= hi for i in (min(want), max(want))) and not any(lo <= i <= hi for i in want):
            continue
        z = np.load(f, allow_pickle=False)
        for k in z.files:
            if k.startswith("mb|"):
                e = int(k.split("|")[1])
                if e in want:
                    idx[e] = f
    return idx


def load_curves(chunk, ids):
    z = np.load(chunk, allow_pickle=False); out = {}
    for e in ids:
        if f"mb|{e}" not in z.files:
            continue
        bands = {}
        for bd in ("F146", "F087", "F213"):
            if f"b|{e}|{bd}|t" in z.files:
                bands[bd] = (np.asarray(z[f"b|{e}|{bd}|t"], float), np.asarray(z[f"b|{e}|{bd}|m"], float))
        out[e] = {"bands": bands, "mb": float(z[f"mb|{e}"][0])}
    return out


# ------------------------------------------------------------------ truth extraction (binaries)
def extract(args):
    import duckdb
    import pyarrow.parquet as pq
    os.makedirs(args.cache, exist_ok=True)
    sel = sample_ids(args.ref_rows, 0, args.cap_binary)
    picks = sorted(sel[L2] + sel[L3])
    m = pq.read_table(gt._fetch(gt.META, args.meta_cache, "meta"), columns=["event_id", "sim_label", "t0lens1", "tE_ref", "u0lens1", "fs_F146"]).to_pydict()
    pos = {int(e): i for i, e in enumerate(np.asarray(m["event_id"], np.int64))}
    et = pq.read_table(gt._fetch(gt.EPOCH, args.epoch_cache, "epoch"), columns=["epoch_id", "bjd"]).to_pydict()
    ep_id = np.asarray(et["epoch_id"], np.int64); ep_bjd = np.asarray(et["bjd"], float)
    o = np.argsort(ep_bjd); ep_id, ep_bjd = ep_id[o], ep_bjd[o]; io = np.argsort(ep_id); ids_s, bjd_s = ep_id[io], ep_bjd[io]
    gaps = np.flatnonzero(np.diff(ep_bjd) > gt.SEASON_GAP_D)
    seasons = list(zip(np.r_[ep_bjd[0], ep_bjd[gaps + 1]], np.r_[ep_bjd[gaps], ep_bjd[-1]]))

    def lookup(q):
        p = np.searchsorted(ids_s, q); np.clip(p, 0, len(ids_s) - 1, out=p); return np.where(ids_s[p] == q, bjd_s[p], np.nan)

    con = duckdb.connect(); con.execute("INSTALL httpfs; LOAD httpfs;")
    t0_, n_done = time.time(), 0
    for c0 in range(0, len(picks), args.chunk):
        ids = picks[c0:c0 + args.chunk]; n_done += len(ids)
        outf = os.path.join(args.cache, f"truth_{ids[0]}_{ids[-1]}_{len(ids)}.npz")
        if os.path.exists(outf):
            continue
        try:
            q = con.execute(f"SELECT event_id, epoch_id, filt, true_flux_uJy, flux_err_uJy FROM read_parquet('{gt.OBS}') "
                            f"WHERE event_id BETWEEN {ids[0]} AND {ids[-1]} AND event_id IN ({','.join(map(str, ids))}) "
                            f"AND saturation_flag = 0 AND filt = 'F146'").fetchnumpy()
        except Exception as exc:
            print(f"[query] failed {str(exc)[:120]}", flush=True); time.sleep(20); continue
        qe = np.asarray(q["event_id"], np.int64); srt = np.argsort(qe, kind="stable"); qe = qe[srt]
        q_ep = np.asarray(q["epoch_id"], np.int64)[srt]; tf = np.asarray(q["true_flux_uJy"], float)[srt]; er = np.asarray(q["flux_err_uJy"], float)[srt]
        lo_i = np.searchsorted(qe, ids, "left"); hi_i = np.searchsorted(qe, ids, "right")
        payload = {}
        for k, e in enumerate(ids):
            j = pos[e]; t0 = float(m["t0lens1"][j]); tE = float(m["tE_ref"][j])
            seas = next(((a, b) for a, b in seasons if a <= t0 <= b), None)
            a, b_ = int(lo_i[k]), int(hi_i[k])
            if seas is None or a == b_:
                continue
            lo_t, hi_t = seas[0], min(seas[1], seas[0] + gt.WINDOW_D)
            bjd = lookup(q_ep[a:b_]); f = tf[a:b_]; s = er[a:b_]
            ok = np.isfinite(bjd) & np.isfinite(f) & (f > 0) & (s > 0) & (f / s >= CFG.snr_threshold)
            win = ok & (bjd >= lo_t) & (bjd <= hi_t); off = ok & (np.abs(bjd - t0) > 5 * tE)
            if win.sum() < 10 or off.sum() < 200:
                continue
            ordt = np.argsort(bjd[win])
            payload[f"t|{e}"] = (bjd[win] - lo_t)[ordt].astype(np.float32)
            payload[f"m|{e}"] = gt.flux_to_ab(f[win])[ordt].astype(np.float32)
            payload[f"s|{e}"] = (1.0857 * s[win] / f[win])[ordt].astype(np.float32)
            payload[f"p|{e}"] = np.array([float(np.median(gt.flux_to_ab(f[off]))), t0 - lo_t, tE, float(m["u0lens1"][j]), float(m["fs_F146"][j])])
        np.savez_compressed(outf + ".tmp.npz", **payload); os.replace(outf + ".tmp.npz", outf)
        print(f"  {n_done}/{len(picks)}  ({time.time() - t0_:.0f}s)", flush=True)
    print("[extract] done", flush=True)


def load_truth(cache):
    out = {}
    for f in glob.glob(os.path.join(cache, "truth_*.npz")):
        z = np.load(f, allow_pickle=False)
        for k in z.files:
            if k.startswith("p|"):
                e = int(k.split("|")[1])
                out[e] = (np.asarray(z[f"t|{e}"], float), np.asarray(z[f"m|{e}"], float), np.asarray(z[f"s|{e}"], float), np.asarray(z[k], float))
    return out


# ------------------------------------------------------------------ scan (worker)
_W = {}


def _init_worker(ckpts):
    import torch
    torch.set_num_threads(1)
    import binml
    _W["clf"] = {k: binml.Classifier(weights=v) for k, v in ckpts.items()}


def _scan_one(job):
    from binml.preprocess import to_tokens, BAND_BINS
    e, curves, truth = job
    cuts = np.arange(1, N_CUTS + 1) * STEP
    out = {"event_id": e}
    for name, clf in _W["clf"].items():
        for variant, bands in (("f146", {"F146": curves["bands"]["F146"]}), ("threeband", curves["bands"])):
            feats = {b: [] for b in BAND_BINS}; fracs = {b: [] for b in BAND_BINS}; valid = []
            for c in cuts:
                rev = {b: (t[t <= c], m[t <= c]) for b, (t, m) in bands.items()}
                if rev["F146"][0].size < 10:                  # as the in-house scan: a cut needs >= 10 F146 points
                    valid.append(False); continue
                tok = to_tokens(rev, m_base_ref=curves["mb"], t_start=0.0)
                for b in BAND_BINS:
                    feats[b].append(tok.feat[b]); fracs[b].append(tok.frac[b])
                valid.append(True)
            p = np.full(N_CUTS, np.nan)
            if any(valid):
                probs = clf._forward({b: np.stack(feats[b]) for b in BAND_BINS}, {b: np.stack(fracs[b]) for b in BAND_BINS})
                p[np.array(valid)] = probs[:, clf.class_names.index("NonPSPL")]
            out[f"p|{name}|{variant}"] = p.astype(np.float64)          # unrounded: compared with a 16-digit threshold
    if truth is not None:
        t, mag, sig, (mb, t0w, tE, u0, fs) = truth
        params = {"t0": t0w, "tE": tE, "u0": u0}
        det = np.zeros(N_CUTS, bool); amp = np.zeros(N_CUTS, np.float32)
        for i, c in enumerate(cuts):
            m = t <= c
            if m.sum() < 10:
                continue
            d, a = _pspl_refit_dchi2(t[m], mag[m], sig[m], mb, fs, params)
            det[i] = d >= CFG.dchi2_anomaly and a >= CFG.min_amplitude_mag; amp[i] = a
        out["onset_detectable"] = det; out["onset_amp"] = amp
    return out


def scan(args):
    os.makedirs(args.cache, exist_ok=True)
    sel = sample_ids(args.ref_rows, args.cap_1s1l, args.cap_binary)
    ids = sorted(sel[L1] + sel[L2] + sel[L3]); lab = {e: L for L, v in sel.items() for e in v}
    truth = load_truth(args.cache)
    idx = curve_index(ids)
    print(f"[scan] {len(ids)} events, {sum(1 for e in ids if e in idx)} in the curve cache, {sum(1 for e in ids if e in truth)} with true curves", flush=True)
    done = set()
    for f in glob.glob(os.path.join(args.cache, "scan_v2_*.npz")):
        done |= {int(k.split("|")[1]) for k in np.load(f, allow_pickle=False).files if k.startswith("id|")}
    todo = [e for e in ids if e in idx and e not in done]
    print(f"[scan] {len(done)} already scanned, {len(todo)} to do", flush=True)
    by_chunk = {}
    for e in todo:
        by_chunk.setdefault(idx[e], []).append(e)
    t0_, n = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(CKPT,)) as ex:
        for chunk, es in by_chunk.items():
            curves = load_curves(chunk, es)
            jobs = [(e, curves[e], truth.get(e)) for e in es if e in curves]
            res = list(ex.map(_scan_one, jobs, chunksize=4))
            payload = {}
            for r in res:
                e = r["event_id"]; payload[f"id|{e}"] = np.array([e]); payload[f"lab|{e}"] = np.array(lab[e])
                for k, v in r.items():
                    if k != "event_id":
                        payload[f"{k}|{e}"] = v
            outf = os.path.join(args.cache, f"scan_v2_{os.path.basename(chunk)[2:-4]}.npz")
            np.savez_compressed(outf + ".tmp.npz", **payload); os.replace(outf + ".tmp.npz", outf)
            n += len(jobs); print(f"  {n}/{len(todo)}  ({time.time() - t0_:.0f}s)", flush=True)
    print("[scan] done", flush=True)


# ------------------------------------------------------------------ reduce
def reduce(args):
    import pyarrow.parquet as pq
    rows = {}
    for f in glob.glob(os.path.join(args.cache, "scan_v2_*.npz")):
        z = np.load(f, allow_pickle=False)
        for k in z.files:
            if k.startswith("id|"):
                e = int(k.split("|")[1]); rows[e] = {kk.rsplit("|", 1)[0]: z[kk] for kk in z.files if kk.endswith(f"|{e}")}
    w = {int(e): float(x) for e, x in zip(*pq.read_table(args.meta_cache, columns=["event_id", "final_weight"]).to_pydict().values())}
    cuts = np.arange(1, N_CUTS + 1) * STEP
    calib = {}
    for name in CKPT:
        f = os.path.join(HERE, f"gapped_threshold_{name}_seasons.json")
        if os.path.exists(f):
            fp = json.load(open(f))["arms"]["rmdc26_gapped"].get("pool", {}).get("full_pool", {})
            if fp.get("achievable"):
                calib[name] = float(fp["threshold"])       # recommended: full pool, measured seasons
    out = {"_doc": __doc__.split("\n")[0], "protocol": {"step_days": STEP, "n_cuts": N_CUTS, "alert": "first cut with P(NonPSPL) >= threshold",
           "onset": "first half-day cut at which our label rule (dchi2 >= 160 and >= 0.02 mag vs the best PSPL on the noise-free F146 curve, GULLS errors as sigma) is met",
           "eligible": "binary lens, anomaly detectable at the full window, finite onset; no selection on model output"},
           "n_scanned": {L: sum(1 for r in rows.values() if str(r["lab"]) == L) for L in (L1, L2, L3)}, "results": {}}
    for name in CKPT:
        thrs = {"frozen": FROZEN, **({"calibrated_seasons_fullpool": calib[name]} if name in calib else {})}
        for variant in ("f146", "threeband"):
            for tn, thr in thrs.items():
                key = f"{name}|{variant}|{tn}"; res = {"threshold": thr}
                # timing on eligible binaries
                elig = [r for r in rows.values() if str(r["lab"]) in (L2, L3) and "onset_detectable" in r and r["onset_detectable"][-1]]
                onset = np.array([cuts[np.argmax(r["onset_detectable"])] for r in elig])
                first = np.array([cuts[np.argmax(r[f"p|{name}|{variant}"] >= thr)] if np.any(r[f"p|{name}|{variant}"] >= thr) else np.nan for r in elig])
                n = len(elig); det = np.isfinite(first); prem = det & (first < onset)
                lag = first[det & ~prem] - onset[det & ~prem]
                res["timing"] = {"n_eligible": n, "detected_frac": round(float(det.mean()), 4) if n else None, "detected_ci95": wilson(int(det.sum()), n),
                                 "premature_frac": round(float(prem.mean()), 4) if n else None, "premature_ci95": wilson(int(prem.sum()), n),
                                 "median_lag_nonpremature_days": round(float(np.median(lag)), 2) if lag.size else None,
                                 "median_onset_day": round(float(np.median(onset)), 2) if n else None}
                # burden per class: alerts per event per season, per 1,000 events per day; rate-weighted
                burden = {}
                for L in (L1, L2, L3):
                    rs = [r for r in rows.values() if str(r["lab"]) == L]
                    if not rs:
                        continue
                    al = np.array([np.any(r[f"p|{name}|{variant}"] >= thr) for r in rs]); ww = np.array([w[int(r["id"][0])] for r in rs])
                    burden[L] = {"n": len(rs), "alert_frac_per_season": round(float(al.mean()), 4), "alert_frac_weighted": round(float(ww[al].sum() / ww.sum()), 4),
                                 "alerts_per_1000_events_per_day": round(float(al.mean() / gt.WINDOW_D * 1000), 3)}
                res["burden"] = burden
                # streaming purity at a stated planetary prevalence: alerts from detectable-anomaly binaries / all alerts.
                # RMDC26 contains no Flat or variable-star contaminants, so this counts single-lens contamination only;
                # a real stream would be less pure. Alert rates are unrounded; the third prevalence is RMDC26's own
                # rate-weighted planetary fraction of the SCORED set (not of this deliberately balanced sample).
                rb = [r for r in rows.values() if str(r["lab"]) in (L2, L3) and "onset_detectable" in r]
                alert = lambda r: bool(np.any(r[f"p|{name}|{variant}"] >= thr))
                a_det = float(np.mean([alert(r) for r in rb if r["onset_detectable"][-1]])) if rb else np.nan
                a_bin = float(np.mean([alert(r) for r in rb])) if rb else np.nan
                f_det = float(np.mean([bool(r["onset_detectable"][-1]) for r in rb])) if rb else np.nan
                s1 = [r for r in rows.values() if str(r["lab"]) == L1]
                a_1 = float(np.mean([alert(r) for r in s1])) if s1 else np.nan
                w1 = np.array([w[int(r["id"][0])] for r in s1]); a_1w = float(w1[[alert(r) for r in s1]].sum() / w1.sum()) if s1 else np.nan
                wb = np.array([w[int(r["id"][0])] for r in rb]); a_binw = float(wb[[alert(r) for r in rb]].sum() / wb.sum()) if rb else np.nan
                rbd = [r for r in rb if r["onset_detectable"][-1]]
                wd = np.array([w[int(r["id"][0])] for r in rbd]); a_detw = float(wd[[alert(r) for r in rbd]].sum() / wd.sum()) if rbd else np.nan
                f_detw = float(wd.sum() / wb.sum()) if rb else np.nan
                res["streaming_purity"] = {}
                for prev, weighted in [(x, False) for x in PREVALENCES] + [(SCORED_PLANET_FRAC_WEIGHTED, True)]:
                    ad, ab, a1, fd = (a_detw, a_binw, a_1w, f_detw) if weighted else (a_det, a_bin, a_1, f_det)
                    tp = prev * fd * ad; allal = prev * ab + (1 - prev) * a1
                    tag = f"planetary_prevalence_{prev}" + ("_rmdc26_scored_rate_weighted" if weighted else "")
                    res["streaming_purity"][tag] = {"purity_detectable_anomaly_alerts": round(tp / allal, 4) if allal > 0 else None,
                                                    "purity_any_binary_alerts": round(prev * ab / allal, 4) if allal > 0 else None,
                                                    "alerts_per_1000_events_per_season": round(allal * 1000, 1),
                                                    "rates": "final_weight-weighted" if weighted else "unweighted"}
                out["results"][key] = res
    if os.path.exists(INHOUSE):
        ih = json.load(open(INHOUSE))
        out["inhouse_reference"] = {"model": ih.get("model"), "n_eligible": ih.get("n_eligible"),
                                    "premature_rate_of_eligible": ih.get("premature_rate_of_eligible"),
                                    "premature_ci_of_eligible": ih.get("premature_ci_of_eligible"),
                                    "detection_fraction": ih.get("detection_fraction"),
                                    "median_lag_non_premature_days": ih.get("median_lag_non_premature_days"),
                                    "by_mass_ratio": ih.get("stratified", {}).get("by_mass_ratio"),
                                    "note": ("in-house = the SHIPPED checkpoint on our simulator, 80% stellar-mass-ratio binaries; "
                                             "RMDC26 anomalies are all planetary, so compare with the giant/neptune strata")}
    out["command"] = " ".join(sys.argv)
    json.dump(out, open(args.out, "w"), indent=1)
    print("scanned", out["n_scanned"])
    for key, res in out["results"].items():
        t = res["timing"]; b = res["burden"]
        print(f"{key:36s} elig {t['n_eligible']}: detected {t['detected_frac']} premature {t['premature_frac']} {t['premature_ci95']} lag {t['median_lag_nonpremature_days']} d | "
              f"1S1L alerts/season {b.get(L1, {}).get('alert_frac_per_season')} | purity@1% {res['streaming_purity']['planetary_prevalence_0.01']['purity_detectable_anomaly_alerts']} @5% {res['streaming_purity']['planetary_prevalence_0.05']['purity_detectable_anomaly_alerts']}")
    print("->", args.out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--extract", action="store_true"); ap.add_argument("--scan", action="store_true"); ap.add_argument("--reduce", action="store_true")
    ap.add_argument("--ref-rows", default="rows_full_fspl5s_g08.json")
    ap.add_argument("--cap-1s1l", type=int, default=3000); ap.add_argument("--cap-binary", type=int, default=1500)
    ap.add_argument("--chunk", type=int, default=200); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--meta-cache", default="/tmp/rmdc26_meta.parquet"); ap.add_argument("--epoch-cache", default="/tmp/rmdc26_epoch.parquet")
    ap.add_argument("--out", default=os.path.join(HERE, "cascade_gulls.json"))
    args = ap.parse_args(argv)
    if args.extract: extract(args)
    if args.scan: scan(args)
    if args.reduce: reduce(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

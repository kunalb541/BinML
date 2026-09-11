"""Detectability-conditioned relabelling of GULLS/RMDC26 with BinML's OWN label policy.

BinML's classes are observational: an event is NonPSPL only if the best-fitting PSPL leaves
delta-chi^2 >= 160 AND a peak deviation >= 0.02 mag on the noise-free curve, and it is an event at
all only if delta-chi^2 >= 500 against a flat baseline with a >= 0.02 mag excursion (pipeline/assemble.py).
GULLS labels by GENERATOR: every 1S2L event is "planetary" whether or not the planet leaves a
trace. The transfer numbers so far therefore count undetectable planets as misses ("recall 0.46").

RMDC26 ships the noise-free curve (`true_flux_uJy`) and the per-epoch error (`flux_err_uJy`), so
the same policy can be applied to GULLS events, with GULLS' own noise model as sigma. Caveats, all
disclosed with the numbers: RMDC26's planetary classes are themselves detection-selected by GULLS
(ObsGroup_0_chi2 >= 60 for every 1S2L/2S2L event); GULLS F146 has ~7,700 epochs per season against
6,912 on the training grid, so a fixed delta-chi^2 cut is ~11% easier here (the reducer reports the
labels with delta-chi^2 rescaled by 6912/n as a sensitivity); the refit is a static point-source
single lens, so parallax and finite-source deviations in GULLS also register as misfit; and the
single-start refit is numerically fragile for rare borderline events (a 1e-14 mag perturbation
flips one of 1,000 in-house binaries).

  1. for each event already scored from the curve cache, fetch true_flux and flux_err for the
     SAME season window BinML saw (same season logic as gulls_transfer.py, saturation_flag = 0);
  2. sigma_mag = 1.0857 * flux_err / true_flux; usable = true_flux/flux_err >= 3 (incident-flux SNR,
     as photometry.observe does); per-band baseline = median true magnitude at |t-t0| > 5 t_E over
     the full mission (the noise-free analogue of the empirical baseline the model was handed);
  3. dchi2_event = sum over bands of ((mag_true - m_base_band)/sigma)^2, max_amp = max |mag_true -
     m_base_band|; anomaly statistics from pipeline.assemble._pspl_refit_dchi2 on F146 (the same
     function that labels the training data), seeded from the catalogue (t0, tE, u0, fs);
  4. label_detect = Flat / PSPL / NonPSPL by the rule in simulate_event: single lenses (1S1L) are
     never NonPSPL (the training rule refits only binaries); pspl_refit_anomaly records what the
     point-source refit says about every event. When the host's peak lies outside the window (the
     seasons adjacent to an inter-season gap, validation/gulls/multi_season.py) the refit is also
     started at the in-window maximum and the best fit is kept.

--extract fills a resumable per-block cache (one JSON per contiguous id block; rerun to resume).
--reduce joins it with any number of rows files and writes transfer_detectability_relabel.json:
recall on detectable-anomaly binaries, the model's verdict on binaries whose planet is NOT
detectable, single-lens false alarms on detectable-event 1S1L, and matched-budget recall under the
relabelled ontology, per checkpoint.

Usage:
  python validation/gulls/detectability_relabel.py --extract --cap-1s1l 1600 --cap-binary 2500
  python validation/gulls/detectability_relabel.py --reduce \
      --models shipped=rows_full_shipped_v2.json ft_g08e12=rows_full_ft_g08e12_v2.json \
              fspl_g08=rows_full_fspl_g08.json fspl5s_g08=rows_full_fspl5s_g08.json
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
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(HERE))
import gulls_transfer as gt                                    # noqa: E402  constants + helpers
from pipeline.assemble import SurveyConfig, _pspl_refit_dchi2  # noqa: E402  THE label policy

CFG = SurveyConfig()
CURVES = os.path.expanduser("~/Desktop/Research/microlensing/gulls_curve_cache")
TRUTH = os.path.expanduser("~/Desktop/Research/microlensing/gulls_truth_cache")
FROZEN = 0.9042405486106873
L1, L2, L3 = "RMDC26_1S1L_ML", "RMDC26_1S2L_ML", "RMDC26_2S2L_ML"
SNR_MIN = CFG.snr_threshold


def _stats(ev):
    """Detectability statistics for one event. ev: dict from the extractor (plain arrays)."""
    out = {"event_id": ev["event_id"], "sim_label": ev["sim_label"], "weight": ev["weight"],
           "tE": ev["tE"], "u0": ev["u0"], "rho": ev["rho"]}
    dchi2_event, max_amp = 0.0, 0.0
    mb = {}
    for bd, (t, mag, sig, mbase) in ev["bands"].items():
        if t.size < 3:
            continue
        mb[bd] = float(mbase)
        r = (mag - mbase) / sig
        dchi2_event += float(np.sum(r ** 2))
        max_amp = max(max_amp, float(np.max(np.abs(mag - mbase))))
    out.update(dchi2_event=float(dchi2_event), max_amp_mag=float(max_amp), m_base_true=mb,
               n_f146=int(ev["bands"]["F146"][0].size) if "F146" in ev["bands"] else 0)
    if "F146" in ev["bands"]:
        # GULLS' own per-epoch noise at the quiescent brightness (window epochs within 0.05 mag of
        # the baseline), for the noise-model comparison against pipeline.photometry.photometric_sigma
        t, mag, sig, mbase = ev["bands"]["F146"]
        q = np.abs(mag - mbase) < 0.05
        out["sigma_f146_quiescent_median"] = round(float(np.median(sig[q])), 5) if q.sum() >= 20 else None
        out["sigma_f146_window_median"] = round(float(np.median(sig)), 5)
    d_an, a_an, multistart = 0.0, 0.0, False
    if "F146" in ev["bands"] and ev["bands"]["F146"][0].size >= 10:
        t, mag, sig, mbase = ev["bands"]["F146"]
        params = {"t0": ev["t0_win"], "tE": ev["tE"], "u0": ev["u0"]}
        d_an, a_an = _pspl_refit_dchi2(t, mag, sig, mbase, ev["fs"], params)
        if not (0.0 <= ev["t0_win"] <= float(t.max()) + 1e-9):
            # The host's peak is OUTSIDE this window (a season adjacent to an inter-season gap).
            # A modeller seeing only this season would also try a single lens centred on what IS in
            # the window; seeding at the catalogue t0 alone leaves an isolated bump labelled as an
            # anomaly (2026-09-11 verification). Keep the best of several starts (lowest residual).
            imax = int(np.argmin(mag))
            for tE_s in (ev["tE"], 0.3, 1.0, 3.0, 10.0):
                for u0_s in (0.05, 0.3, 1.0):
                    d2, a2 = _pspl_refit_dchi2(t, mag, sig, mbase, ev["fs"], {"t0": float(t[imax]), "tE": tE_s, "u0": u0_s})
                    if d2 < d_an:
                        d_an, a_an = d2, a2
            multistart = True
    out.update(dchi2_anomaly=float(d_an), anomaly_amp_mag=float(a_an), refit_multistart=multistart)
    # The training rule (pipeline.assemble.simulate_event) runs the anomaly refit only for binaries:
    # a single lens -- point or finite source -- is PSPL or Flat, never NonPSPL. 1S1L events follow
    # that rule; what the point-source refit says about them is kept as a separate diagnostic.
    if dchi2_event < CFG.dchi2_event or max_amp < CFG.min_amplitude_mag:
        lab = "Flat"
    elif ev["sim_label"] != L1 and d_an >= CFG.dchi2_anomaly and a_an >= CFG.min_amplitude_mag:
        lab = "NonPSPL"
    else:
        lab = "PSPL"
    out["label_detect"] = lab
    out["pspl_refit_anomaly"] = bool(lab != "Flat" and d_an >= CFG.dchi2_anomaly and a_an >= CFG.min_amplitude_mag)
    out["generator_label"] = "PSPL" if ev["sim_label"] == L1 else "NonPSPL"
    return out


def extract(args):
    import duckdb
    import pyarrow.parquet as pq
    os.makedirs(args.truth_cache, exist_ok=True)
    ref = json.load(open(os.path.join(CURVES, args.ref_rows) if not os.path.isabs(args.ref_rows) else args.ref_rows))
    dense = sorted({r["event_id"] for r in ref if r.get("dense") and "pred" in r})
    m = pq.read_table(gt._fetch(gt.META, args.meta_cache, "meta"), columns=[
        "event_id", "sim_label", "t0lens1", "tE_ref", "u0lens1", "rho", "fs_F146", "final_weight"]).to_pydict()
    eid_arr = np.asarray(m["event_id"], np.int64)
    pos = {int(e): i for i, e in enumerate(eid_arr)}
    by = {}
    for e in dense:
        by.setdefault(m["sim_label"][pos[e]], []).append(e)
    # Deterministic, EXTENSIBLE selection: the first N dense ids of each class (event_id does not
    # order events by sightline -- a 250-id block spans 105 of 129 fields -- so a prefix is not a
    # biased sample). Raising a cap later only appends blocks; nothing already cached is redone.
    # Binaries first: they are the payload, and their dense ids are 3x denser in id space than
    # 1S1L's, so their blocks read 3x less of the remote table per event.
    picks = []
    for lab in (L2, L3, L1):
        ids = sorted(by.get(lab, []))
        cap = args.cap_1s1l if lab == L1 else args.cap_binary
        if cap and len(ids) > cap:
            ids = ids[:cap]
        if ids:
            print(f"[select] {lab}: {len(ids)} events, ids {ids[0]}-{ids[-1]}", flush=True)
        picks.extend(ids)

    et = pq.read_table(gt._fetch(gt.EPOCH, args.epoch_cache, "epoch"), columns=["epoch_id", "bjd"]).to_pydict()
    ep_id = np.asarray(et["epoch_id"], np.int64); ep_bjd = np.asarray(et["bjd"], float)
    o = np.argsort(ep_bjd); ep_id, ep_bjd = ep_id[o], ep_bjd[o]
    ids_o = np.argsort(ep_id); _ids_s, _bjd_s = ep_id[ids_o], ep_bjd[ids_o]

    def bjd_lookup(q):
        p = np.searchsorted(_ids_s, q); np.clip(p, 0, len(_ids_s) - 1, out=p)
        return np.where(_ids_s[p] == q, _bjd_s[p], np.nan)

    gaps = np.flatnonzero(np.diff(ep_bjd) > gt.SEASON_GAP_D)
    seasons = list(zip(np.r_[ep_bjd[0], ep_bjd[gaps + 1]], np.r_[ep_bjd[gaps], ep_bjd[-1]]))

    def season_of(t0):
        for a, b in seasons:
            if a <= t0 <= b:
                return a, b
        return None

    con = duckdb.connect(); con.execute("INSTALL httpfs; LOAD httpfs;")
    t_start, n_done, n_new = time.time(), 0, 0
    pool = ProcessPoolExecutor(max_workers=args.workers)
    for c0 in range(0, len(picks), args.chunk):
        ids = picks[c0:c0 + args.chunk]
        outf = os.path.join(args.truth_cache, f"t_{ids[0]}_{ids[-1]}_{len(ids)}.json")
        n_done += len(ids)
        if os.path.exists(outf):
            continue
        if args.max_blocks and n_new >= args.max_blocks:
            break
        n_new += 1
        try:
            q = con.execute(
                f"SELECT event_id, epoch_id, filt, true_flux_uJy, flux_err_uJy FROM read_parquet('{gt.OBS}') "
                f"WHERE event_id BETWEEN {ids[0]} AND {ids[-1]} AND event_id IN ({','.join(map(str, ids))}) "
                f"AND saturation_flag = 0").fetchnumpy()
        except Exception as exc:
            print(f"[query] block {ids[0]}-{ids[-1]} failed: {str(exc)[:160]}", flush=True)
            time.sleep(20)
            continue
        qe = np.asarray(q["event_id"], np.int64); srt = np.argsort(qe, kind="stable"); qe = qe[srt]
        q_ep = np.asarray(q["epoch_id"], np.int64)[srt]; q_ft = np.asarray(q["filt"])[srt]
        q_tf = np.asarray(q["true_flux_uJy"], float)[srt]; q_er = np.asarray(q["flux_err_uJy"], float)[srt]
        lo_i = np.searchsorted(qe, ids, "left"); hi_i = np.searchsorted(qe, ids, "right")
        evs, skipped = [], []
        for k, eid in enumerate(ids):
            j = pos[eid]
            seas = season_of(float(m["t0lens1"][j]))
            a, b_ = int(lo_i[k]), int(hi_i[k])
            if seas is None or a == b_:
                skipped.append({"event_id": eid, "sim_label": m["sim_label"][j], "skipped": "no season / no rows"}); continue
            lo_t, hi_t = seas[0], min(seas[1], seas[0] + gt.WINDOW_D)
            bjd = bjd_lookup(q_ep[a:b_]); ft = q_ft[a:b_]; tf = q_tf[a:b_]; er = q_er[a:b_]
            ok = np.isfinite(bjd) & np.isfinite(tf) & np.isfinite(er) & (tf > 0) & (er > 0) & (tf / er >= SNR_MIN)
            in_win = ok & (bjd >= lo_t) & (bjd <= hi_t)
            off = ok & (np.abs(bjd - float(m["t0lens1"][j])) > 5.0 * float(m["tE_ref"][j]))
            bands = {}
            for bd in ("F146", "F087", "F213"):
                sb = in_win & (ft == bd); so = off & (ft == bd)
                if sb.sum() < 10 or so.sum() < (200 if bd == "F146" else 20):
                    continue
                mag = gt.flux_to_ab(tf[sb]); sig = 1.0857 * er[sb] / tf[sb]
                mbase = float(np.median(gt.flux_to_ab(tf[so])))
                ordt = np.argsort(bjd[sb])
                bands[bd] = ((bjd[sb] - lo_t)[ordt], mag[ordt], sig[ordt], mbase)
            if "F146" not in bands:
                skipped.append({"event_id": eid, "sim_label": m["sim_label"][j], "skipped": "no usable F146 / baseline"}); continue
            evs.append({"event_id": eid, "sim_label": m["sim_label"][j], "weight": float(m["final_weight"][j]),
                        "tE": float(m["tE_ref"][j]), "u0": float(m["u0lens1"][j]), "rho": float(m["rho"][j]),
                        "fs": float(m["fs_F146"][j]), "t0_win": float(m["t0lens1"][j]) - lo_t, "bands": bands})
        res = list(pool.map(_stats, evs, chunksize=8)) + skipped
        json.dump(res, open(outf + ".tmp", "w")); os.replace(outf + ".tmp", outf)
        el = time.time() - t_start
        print(f"  {n_done}/{len(picks)}  ({el:.0f}s, {el / max(n_done - c0, 1):.2f}s/ev this block)", flush=True)
    pool.shutdown()
    print("[extract] done", flush=True)


def _load_truth(d):
    rows = {}
    for f in sorted(os.listdir(d)):
        if f.startswith("t_") and f.endswith(".json"):
            for r in json.load(open(os.path.join(d, f))):
                if "label_detect" in r:
                    rows[r["event_id"]] = r
    return rows


def _rate(p, thr):
    return round(float((p >= thr).mean()), 4) if p.size else None


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    ph = k / n; d = 1 + z * z / n; c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def _kn(p, sel, thr):
    k, n = int((p[sel] >= thr).sum()), int(sel.sum())
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson95": wilson(k, n)}


def selected_ids(ref_rows, cap_1s1l, cap_binary):
    """The documented selection: the first N scored (dense) event ids of each class, in id order."""
    ref = json.load(open(os.path.join(CURVES, ref_rows) if not os.path.isabs(ref_rows) else ref_rows))
    by = {}
    for r in ref:
        if r.get("dense") and "pred" in r:
            by.setdefault(r["sim_label"], []).append(r["event_id"])
    out = {}
    for lab in (L1, L2, L3):
        ids = sorted(by.get(lab, []))
        cap = cap_1s1l if lab == L1 else cap_binary
        out[lab] = ids[:cap] if cap else ids
    return out


def reduce(args):
    truth = _load_truth(args.truth_cache)
    sel = selected_ids(args.ref_rows, args.cap_1s1l, args.cap_binary)
    want = set(x for v in sel.values() for x in v)
    extra = sorted(set(truth) - want)                   # cached by other runs: excluded, and said so
    models, f146 = {}, {}
    for it in args.models:
        k, v = it.split("=", 1)
        v = v if os.path.isabs(v) else os.path.join(CURVES, v)
        models[k] = {r["event_id"]: float(r["p_nonpspl"]) for r in json.load(open(v)) if r.get("dense") and "pred" in r}
    for it in args.f146 or []:
        k, v = it.split("=", 1)
        v = v if os.path.isabs(v) else os.path.join(CURVES, v)
        f146[k] = {r["event_id"]: float(r["p_nonpspl"]) for r in json.load(open(v)) if r.get("dense") and "pred" in r}
    common = sorted(want & set(truth) & set.intersection(*[set(v) for v in list(models.values()) + list(f146.values())]))
    missing = sorted(want - set(truth))
    T = [truth[i] for i in common]
    lab = np.array([t["sim_label"] for t in T]); det = np.array([t["label_detect"] for t in T])
    w = np.array([t["weight"] for t in T])
    d2 = np.array([t["dchi2_anomaly"] for t in T]); amp = np.array([t["anomaly_amp_mag"] for t in T])
    d2e = np.array([t["dchi2_event"] for t in T]); ampe = np.array([t["max_amp_mag"] for t in T])
    nf = np.array([max(t.get("n_f146", 6912), 1) for t in T])
    refit_anom = np.array([bool(t.get("pspl_refit_anomaly", False)) for t in T])
    tE = np.array([t["tE"] for t in T]); u0 = np.abs(np.array([t["u0"] for t in T])); rho = np.array([t["rho"] for t in T])
    s1, s2, s3 = lab == L1, lab == L2, lab == L3
    binr = s2 | s3
    out = {"_doc": __doc__.split("\n")[0], "n_events": len(common),
           "selection": {"rule": f"first {args.cap_1s1l} scored 1S1L and first {args.cap_binary} scored 1S2L / 2S2L event ids "
                                 f"(dense, in id order) of {args.ref_rows}",
                         "ids": {L: [v[0], v[-1]] if v else None for L, v in sel.items()},
                         "n_selected": {L: len(v) for L, v in sel.items()},
                         "n_missing_from_cache": len(missing), "n_cached_but_excluded": len(extra)},
           "policy": {"dchi2_event": CFG.dchi2_event, "dchi2_anomaly": CFG.dchi2_anomaly, "min_amplitude_mag": CFG.min_amplitude_mag,
                      "sigma": "GULLS flux_err_uJy converted to magnitudes at the true flux", "snr_min": SNR_MIN,
                      "baseline": "median true-flux magnitude at |t-t0| > 5 tE over the full mission, per band",
                      "anomaly_statistic": "pipeline.assemble._pspl_refit_dchi2 on F146 (the training-label function)",
                      "single_lenses": "never NonPSPL (training rule); pspl_refit_anomaly kept as a diagnostic"},
           "relabelling": {}, "models": {}}
    for L in (L1, L2, L3):
        m = lab == L
        blk = {"n": int(m.sum()), **{f"frac_{k}": round(float((det[m] == k).mean()), 4) for k in ("Flat", "PSPL", "NonPSPL")},
               **{f"wfrac_{k}": round(float(w[m & (det == k)].sum() / w[m].sum()), 4) for k in ("Flat", "PSPL", "NonPSPL")},
               "median_dchi2_anomaly": round(float(np.median(d2[m])), 1), "median_anomaly_amp_mag": round(float(np.median(amp[m])), 4)}
        if L != L1:
            ev_ok = (d2e >= CFG.dchi2_event) & (ampe >= CFG.min_amplitude_mag)
            blk["decomposition"] = {
                "fail_dchi2_anomaly": round(float((m & ev_ok & (d2 < CFG.dchi2_anomaly)).sum() / m.sum()), 4),
                "floor_vetoed": round(float((m & ev_ok & (d2 >= CFG.dchi2_anomaly) & (amp < CFG.min_amplitude_mag)).sum() / m.sum()), 4),
                "flat_event": round(float((m & ~ev_ok).sum() / m.sum()), 4)}
            resc = (d2 * 6912.0 / nf >= CFG.dchi2_anomaly) & (amp >= CFG.min_amplitude_mag) & ev_ok
            blk["frac_NonPSPL_dchi2_rescaled_to_6912_epochs"] = round(float((m & resc).sum() / m.sum()), 4)
        else:
            blk["pspl_refit_anomaly_frac"] = round(float(refit_anom[m].mean()), 4)
            blk["pspl_refit_anomaly_median_rho_over_u0"] = round(float(np.median((rho / np.maximum(u0, 1e-6))[m & refit_anom])), 3) if (m & refit_anom).any() else None
        out["relabelling"][L] = blk
    pool = {"binaries": binr}
    ev_ok = (d2e >= CFG.dchi2_event) & (ampe >= CFG.min_amplitude_mag)
    floor_vet = binr & ev_ok & (d2 >= CFG.dchi2_anomaly) & (amp < CFG.min_amplitude_mag)
    fail_chi = binr & ev_ok & (d2 < CFG.dchi2_anomaly)
    bin_det = binr & (det == "NonPSPL"); bin_undet = binr & (det != "NonPSPL")
    out["relabelling"]["binaries_pooled"] = {"n": int(binr.sum()), "frac_fail_dchi2_anomaly": round(float(fail_chi.sum() / binr.sum()), 4),
                                             "frac_floor_vetoed": round(float(floor_vet.sum() / binr.sum()), 4),
                                             "frac_detectable": round(float(bin_det.sum() / binr.sum()), 4)}
    calib = {}
    for name in models:
        for suffix, key in (("_seasons", "calibrated_seasons"), ("", "calibrated_legacy")):
            f = os.path.join(HERE, f"gapped_threshold_{name}{suffix}.json")
            if os.path.exists(f):
                calib.setdefault(name, {})[key] = float(json.load(open(f))["arms"]["rmdc26_gapped"]["threshold_at_target_purity"])
    MIX = {L1: 33353, L2: 11388, L3: 12234}                # scored-set class mix, for prevalence-fixed precision
    cw = np.where(s1, MIX[L1] / max(s1.sum(), 1), np.where(s2, MIX[L2] / max(s2.sum(), 1), MIX[L3] / max(s3.sum(), 1)))
    single_subfloor = s1 & ev_ok & (d2 >= CFG.dchi2_anomaly) & (amp < CFG.min_amplitude_mag)   # static-refit misfit, no planet
    for name, d in models.items():
        p = np.array([d[i] for i in common])
        thrs = {"frozen": FROZEN, **calib.get(name, {})}
        blk = {"thresholds": thrs, "at_threshold": {}}
        for tn, thr in thrs.items():
            flag = p >= thr
            blk["at_threshold"][tn] = {
                "fa_1S1L": _kn(p, s1, thr), "recall_1S2L_generator": _kn(p, s2, thr), "recall_2S2L_generator": _kn(p, s3, thr),
                "recall_1S2L_detectable": _kn(p, s2 & bin_det, thr), "recall_2S2L_detectable": _kn(p, s3 & bin_det, thr),
                "flag_binaries_undetectable": _kn(p, bin_undet, thr), "flag_binaries_floor_vetoed": _kn(p, floor_vet, thr),
                "flag_binaries_floor_vetoed_lt_5mmag": _kn(p, floor_vet & (amp < 0.005), thr),
                "flag_binaries_fail_dchi2": _kn(p, fail_chi, thr),
                "flag_1S1L_pspl_refit_anomaly": _kn(p, s1 & refit_anom, thr),
                "flag_1S1L_other": _kn(p, s1 & ~refit_anom, thr),
                "flag_1S1L_significant_subfloor_misfit": _kn(p, single_subfloor, thr),
                "ontology_precision_sample_mix": round(float((flag & (det == "NonPSPL")).sum() / max(flag.sum(), 1)), 4),
                "ontology_precision_scored_mix": round(float((cw * (flag & (det == "NonPSPL"))).sum() / max((cw * flag).sum(), 1e-12)), 4)}
        ths = np.unique(np.round(p, 4))[::-1]
        fa = np.array([(p[s1] >= t).mean() for t in ths])
        blk["recall_at_matched_fa"] = []
        for tgt in (0.02, 0.031, 0.052, 0.117):
            i = int(np.argmin(np.abs(fa - tgt)))
            blk["recall_at_matched_fa"].append({"fa_target": tgt, "fa": round(float(fa[i]), 4), "threshold": float(ths[i]),
                                                "recall_1S2L_generator": round(float((p[s2] >= ths[i]).mean()), 4),
                                                "recall_1S2L_detectable": round(float((p[s2 & bin_det] >= ths[i]).mean()), 4),
                                                "recall_2S2L_generator": round(float((p[s3] >= ths[i]).mean()), 4),
                                                "recall_2S2L_detectable": round(float((p[s3 & bin_det] >= ths[i]).mean()), 4)})
        blk["recall_detectable_by_anomaly_amp_frozen"] = []
        for lo, hi in zip([0.02, 0.05, 0.1, 0.2, 0.5], [0.05, 0.1, 0.2, 0.5, np.inf]):
            m_ = bin_det & (amp >= lo) & (amp < hi)
            blk["recall_detectable_by_anomaly_amp_frozen"].append({"amp_bin": [lo, None if hi == np.inf else hi], **_kn(p, m_, FROZEN),
                                                                   "median_dchi2_anomaly": round(float(np.median(d2[m_])), 1) if m_.any() else None})
        blk["recall_detectable_by_dchi2_frozen"] = []
        for lo, hi in zip([160, 500, 1e3, 1e4, 1e5], [500, 1e3, 1e4, 1e5, np.inf]):
            m_ = bin_det & (d2 >= lo) & (d2 < hi)
            blk["recall_detectable_by_dchi2_frozen"].append({"dchi2_bin": [lo, None if hi == np.inf else hi], **_kn(p, m_, FROZEN)})
        blk["flag_floor_vetoed_by_amp_frozen"] = []
        for lo, hi in zip([0, 0.005, 0.01, 0.02], [0.005, 0.01, 0.02, 0.05]):
            m_ = binr & ev_ok & (d2 >= CFG.dchi2_anomaly) & (amp >= lo) & (amp < hi)
            blk["flag_floor_vetoed_by_amp_frozen"].append({"amp_bin": [lo, hi], **_kn(p, m_, FROZEN)})
        if name in f146:
            q = np.array([f146[name][i] for i in common])
            blk["colour_ablation"] = {}
            for tgt in (0.02, 0.052):
                ia = int(np.argmin(np.abs(fa - tgt)))
                thsb = np.unique(np.round(q, 4))[::-1]; fab = np.array([(q[s1] >= t).mean() for t in thsb]); ib = int(np.argmin(np.abs(fab - tgt)))
                blk["colour_ablation"][f"budget_{tgt}"] = {
                    "three_band": {"detectable": round(float((p[bin_det] >= ths[ia]).mean()), 4), "undetectable": round(float((p[bin_undet] >= ths[ia]).mean()), 4)},
                    "f146_only": {"detectable": round(float((q[bin_det] >= thsb[ib]).mean()), 4), "undetectable": round(float((q[bin_undet] >= thsb[ib]).mean()), 4)},
                    "n_detectable": int(bin_det.sum()), "n_undetectable": int(bin_undet.sum())}
            blk["colour_ablation"]["flag_floor_vetoed_by_amp_frozen_f146only"] = [
                {"amp_bin": [lo, hi], **_kn(q, binr & ev_ok & (d2 >= CFG.dchi2_anomaly) & (amp >= lo) & (amp < hi), FROZEN)}
                for lo, hi in zip([0, 0.005, 0.01, 0.02], [0.005, 0.01, 0.02, 0.05])]
        out["models"][name] = blk
    out["command"] = " ".join(sys.argv)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"n={len(common)} (missing {len(missing)}, cached-but-excluded {len(extra)})")
    for L, b in out["relabelling"].items():
        print(L, {k: v for k, v in b.items() if not isinstance(v, dict)}, b.get("decomposition", ""))
    for name, b in out["models"].items():
        for tn, a in b["at_threshold"].items():
            print(f"{name:12s} {tn[:12]:>12} FA {a['fa_1S1L']['rate']} rec gen {a['recall_1S2L_generator']['rate']}/{a['recall_2S2L_generator']['rate']} "
                  f"det {a['recall_1S2L_detectable']['rate']}/{a['recall_2S2L_detectable']['rate']} floor-vetoed {a['flag_binaries_floor_vetoed']['rate']} "
                  f"1S1L subfloor-misfit {a['flag_1S1L_significant_subfloor_misfit']}")
    print("->", args.out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--extract", action="store_true"); ap.add_argument("--reduce", action="store_true")
    ap.add_argument("--ref-rows", default="rows_full_fspl5s_g08.json", help="rows file defining the scored dense set")
    ap.add_argument("--cap-1s1l", type=int, default=1600, help="first N dense 1S1L ids (0 = all)")
    ap.add_argument("--cap-binary", type=int, default=2500, help="first N dense ids per binary class (0 = all)")
    ap.add_argument("--chunk", type=int, default=200); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-blocks", type=int, default=0, help="stop after this many NEW blocks (smoke test)")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--truth-cache", default=TRUTH)
    ap.add_argument("--meta-cache", default="/tmp/rmdc26_meta.parquet"); ap.add_argument("--epoch-cache", default="/tmp/rmdc26_epoch.parquet")
    ap.add_argument("--models", nargs="*", default=[], help="name=rows.json for --reduce")
    ap.add_argument("--f146", nargs="*", default=[], help="name=rows.json F146-only runs (colour decomposition)")
    ap.add_argument("--out", default=os.path.join(HERE, "transfer_detectability_relabel.json"))
    args = ap.parse_args(argv)
    if args.extract:
        extract(args)
    if args.reduce:
        reduce(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

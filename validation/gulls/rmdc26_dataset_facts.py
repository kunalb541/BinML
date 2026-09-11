"""Every RMDC26 dataset fact the revision quotes, computed from the pinned tables -> rmdc26_dataset_facts.json.

Before 2026-09-11 several of these lived only in code comments or prose (and one, "385,004 events",
was an event id, not a count). This script derives them from:
  * the pinned metadata and epoch tables (RGES-PIT/MachineLearning @ a338d5ba),
  * validation/gulls/rmdc26_schedule.json (seasons, pauses),
  * the transfer rows (selection outcome per event; rows_full_fspl5s_g08.json by default),
  * our training priors (pipeline/priors.py, pipeline/assemble.SurveyConfig) and a natural-prior
    held-out pool (cadence_local_work/work15/mm_eval) for the q comparison.

Usage:  python validation/gulls/rmdc26_dataset_facts.py
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
from pipeline.assemble import SurveyConfig  # noqa: E402
from pipeline.priors import DEFAULT_PRIORS as PR  # noqa: E402

META = "/tmp/rmdc26_meta.parquet"
CURVES = os.path.expanduser("~/Desktop/Research/microlensing/gulls_curve_cache")
POOL = os.path.expanduser("~/Desktop/Research/microlensing/cadence_local_work/work15/mm_eval")
L1, L2, L3 = "RMDC26_1S1L_ML", "RMDC26_1S2L_ML", "RMDC26_2S2L_ML"
FROZEN = 0.9042405486106873


def q(x, qs=(10, 50, 90)):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return {f"p{k}": float(np.percentile(x, k)) for k in qs} if x.size else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", default="rows_full_fspl5s_g08.json")
    ap.add_argument("--out", default=os.path.join(HERE, "rmdc26_dataset_facts.json"))
    args = ap.parse_args(argv)
    import pyarrow.parquet as pq
    from scipy.stats import ks_2samp
    m = pq.read_table(META).to_pandas().set_index("event_id")
    S = json.load(open(os.path.join(HERE, "rmdc26_schedule.json")))["seasons"]
    rows = {r["event_id"]: r for r in json.load(open(os.path.join(CURVES, args.rows)))}
    out = {"_doc": __doc__.split("\n")[0], "dataset_revision": "a338d5bab441b5caf551d2fea9469aadfdc81ec1"}
    lab = m["sim_label"]
    out["catalogue"] = {"n_events": int(len(m)), "n_unique_event_id": int(m.index.nunique()),
                        "by_class": {L: {"n": int((lab == L).sum()), "id_min": int(m.index[lab == L].min()),
                                         "id_max": int(m.index[lab == L].max())} for L in (L1, L2, L3)}}
    # host single-lens peak amplitude, as the transfer selects on
    u0 = m["u0lens1"].abs().values; fs = m["fs_F146"].values
    Amax = (u0 ** 2 + 2) / np.maximum(u0 * np.sqrt(u0 ** 2 + 4), 1e-12)
    amp = np.abs(-2.5 * np.log10(np.maximum(1 + fs * (Amax - 1), 1e-12)))
    tE = m["tE_ref"].values
    ampok = amp >= 0.1; teok = (tE >= 1) & (tE <= 300)
    out["selection_inputs"] = {}
    for L in (L1, L2, L3):
        c = (lab == L).values
        out["selection_inputs"][L] = {"median_host_peak_amp_mag": float(np.median(amp[c])), "frac_host_peak_amp_ge_0.1": round(float(ampok[c].mean()), 4),
                                      "frac_subday_tE_all": round(float((tE[c] < 1).mean()), 4),
                                      "frac_subday_tE_among_amp_ge_0.1": round(float((tE[c & ampok] < 1).mean()), 4),
                                      "n_eligible_amp_and_tE": int((c & ampok & teok).sum())}
    # outcome of the transfer selection (season logic lives in gulls_transfer.py)
    starts = np.array([x["start_bjd"] for x in S]); ends = np.array([x["end_bjd"] for x in S])
    out["selection_outcome"] = {"n_eligible": len(rows), "by_class": {}}
    reasons = {}
    for r in rows.values():
        L = r["sim_label"]
        if "skipped" in r:
            t0 = float(m.loc[r["event_id"], "t0lens1"])
            k = r["skipped"]
            if k.startswith("t0 falls"):
                k = "peak before first season" if t0 < starts[0] else ("peak after last season" if t0 > ends[-1] else "peak between seasons")
            else:
                k = "no usable baseline or F146"
        else:
            k = "scored, high-cadence season" if r.get("dense") else "scored, low-cadence season"
        reasons.setdefault(L, {}).setdefault(k, 0); reasons[L][k] += 1
    for L, d in reasons.items():
        out["selection_outcome"]["by_class"][L] = d
    tot = {}
    for d in reasons.values():
        for k, v in d.items():
            tot[k] = tot.get(k, 0) + v
    out["selection_outcome"]["total"] = tot
    out["selection_outcome"]["fractions_of_eligible"] = {k: round(v / len(rows), 4) for k, v in tot.items()}
    scored = [r for r in rows.values() if r.get("dense") and "pred" in r]
    ids = np.array([r["event_id"] for r in scored]); sl = np.array([r["sim_label"] for r in scored])
    w = np.array([r["weight"] for r in scored])
    out["scored_set"] = {"n": int(ids.size), "by_class": {L: int((sl == L).sum()) for L in (L1, L2, L3)},
                         "rate_weighted_class_mix": {L: round(float(w[sl == L].sum() / w.sum()), 4) for L in (L1, L2, L3)},
                         "rate_weighted_planetary_fraction": round(float(w[sl != L1].sum() / w.sum()), 4)}
    sub = m.loc[ids]
    qp = {L: sub["Planet_q"][sl == L].values for L in (L2, L3)}
    out["planets"] = {"median_Planet_q_scored": {L: float(np.median(qp[L])) for L in (L2, L3)},
                      "median_Planet_q_all": {L: float(m["Planet_q"][lab == L].median()) for L in (L2, L3)},
                      "ks_log_q_1S2L_vs_2S2L_scored": {"D": float(ks_2samp(np.log10(qp[L2]), np.log10(qp[L3])).statistic),
                                                       "p": float(ks_2samp(np.log10(qp[L2]), np.log10(qp[L3])).pvalue)},
                      "frac_2S2L_binary_source": {"scored": round(float((sub["Source_Is_Binary"][sl == L3] > 0).mean()), 4),
                                                  "all": round(float((m["Source_Is_Binary"][lab == L3] > 0).mean()), 4)},
                      "gulls_detection_cut_ObsGroup_0_chi2_min": {L: float(m["ObsGroup_0_chi2"][lab == L].min()) for L in (L1, L2, L3)},
                      "frac_1S1L_ObsGroup_0_chi2_lt_60": round(float((m["ObsGroup_0_chi2"][lab == L1] < 60).mean()), 4)}
    if "Planet_s" in m.columns:
        sv = m["Planet_s"].values
        out["planets"]["host_amplitude_cut_by_s"] = {}
        for L in (L2, L3):
            c = (lab == L).values
            for lo, hi in ((0, 0.5), (0.5, 2), (2, 5), (5, np.inf)):
                cc = c & (sv >= lo) & (sv < hi)
                out["planets"]["host_amplitude_cut_by_s"][f"{L}|s[{lo},{hi})"] = {"frac_of_class": round(float(cc.sum() / c.sum()), 4),
                                                                                  "frac_passing_amp_cut": round(float(ampok[cc].mean()), 4) if cc.any() else None}
    # our priors vs the scored population
    cfg = SurveyConfig()
    mb = np.array([r["m_base"] for r in scored])
    oos = {}
    for L in (L1, L2, L3):
        c = sl == L; ss = sub[c]
        d = {"frac_m_base_lt_prior_min": round(float((mb[c] < cfg.m_base_min).mean()), 4),
             "frac_fs_lt_prior_min": round(float((ss["fs_F146"] < PR.FS_MIN).mean()), 4) if hasattr(PR, "FS_MIN") else None,
             "median_tE": float(ss["tE_ref"].median()), "frac_tE_lt_3d": round(float((ss["tE_ref"] < 3).mean()), 4),
             "median_abs_u0": float(ss["u0lens1"].abs().median())}
        if L != L1:
            d["frac_s_outside_prior"] = round(float(((ss["Planet_s"] < PR.S_MIN) | (ss["Planet_s"] > PR.S_MAX)).mean()), 4)
            d["frac_q_lt_prior_min"] = round(float((ss["Planet_q"] < PR.Q_MIN).mean()), 4)
            d["frac_rho_gt_binary_prior_max"] = round(float((ss["rho"] > PR.RHO_MAX).mean()), 4)
        else:
            d["rho"] = {**q(ss["rho"], (50, 90, 99)), "n_gt_1": int((ss["rho"] > 1).sum()), "n_gt_5": int((ss["rho"] > 5).sum())}
        oos[L] = d
    out["vs_training_priors"] = {"priors": {"m_base_range": [cfg.m_base_min, cfg.m_base_max], "fs_min": getattr(PR, "FS_MIN", None),
                                            "s_range": [PR.S_MIN, PR.S_MAX], "q_range": [PR.Q_MIN, PR.Q_MAX], "binary_rho_max": PR.RHO_MAX,
                                            "tE_range": [PR.TE_MIN_DAYS, PR.TE_MAX_DAYS], "u0_max": PR.U0_MAX},
                                 "by_class": oos}
    p5 = np.array([r["p_nonpspl"] for r in scored]); s1 = sl == L1
    out["residual_fa_fspl5s_by_support"] = {
        "m_base_lt_20": {"n": int((s1 & (mb < 20)).sum()), "fa": round(float((p5[s1 & (mb < 20)] >= FROZEN).mean()), 4)},
        "m_base_20_25": {"n": int((s1 & (mb >= 20) & (mb <= 25)).sum()), "fa": round(float((p5[s1 & (mb >= 20) & (mb <= 25)] >= FROZEN).mean()), 4)},
        "note": f"single-lens false-alarm rate at the frozen threshold for {args.rows}"}
    if os.path.exists(os.path.join(POOL, "params.npy")):
        from pipeline.classes import CLASS_NAMES
        meta = json.load(open(os.path.join(POOL, "meta.json"))); pf = meta["param_fields"]
        par = np.load(os.path.join(POOL, "params.npy")); labp = np.load(os.path.join(POOL, "label.npy"))
        non = labp == CLASS_NAMES.index("NonPSPL"); qq = par[non, pf.index("q")]
        out["our_nonpspl_q"] = {"pool": "cadence_local_work/work15/mm_eval (natural prior, shards 100-103)", "n": int(non.sum()),
                                "median_q": float(np.median(qq)), "frac_q_gt_1e-2": round(float((qq > 1e-2).mean()), 4),
                                "median_q_planetary_lt_1e-2": float(np.median(qq[qq < 1e-2]))}
    dense = [x for x in S if x["dense"]]
    gaps = [round(S[i + 1]["start_bjd"] - S[i]["end_bjd"], 2) for i in range(len(S) - 1)]
    out["survey_geometry"] = {"n_seasons": len(S), "n_high_cadence": len(dense), "n_low_cadence": len(S) - len(dense),
                              "high_cadence_length_days": sorted({x["length_days"] for x in dense}),
                              "low_cadence_length_days": sorted({x["length_days"] for x in S if not x["dense"]}),
                              "inter_season_gaps_days": gaps, "observed_days": round(sum(x["length_days"] for x in S), 1),
                              "mission_span_days": round(S[-1]["end_bjd"] - S[0]["start_bjd"], 1),
                              "f146_pause_hours_per_season": [x["summary"]["pause_hours_total"] for x in dense],
                              "f146_pauses_per_season": [x["summary"]["n_pauses"] for x in dense],
                              "pause_days_common_to_all_seasons": "about 0.99, 35.24 and 69.49 d (validation/gulls/rmdc26_schedule.json)",
                              "pause_start_days_common_to_all_seasons": [
                                  p0["start_day"] for p0 in dense[0]["pauses"]
                                  if all(any(abs(p0["start_day"] - p1["start_day"]) < 0.1 for p1 in x["pauses"]) for x in dense[1:])],
                              "season_shortfall_vs_72d_window_days": round(72.0 - min(x["length_days"] for x in dense), 2),
                              "f146_bins_with_frac_lt_1": [x["summary"]["f146_bins_frac_lt_1"] for x in dense],
                              "high_cadence_fraction_of_span": round(sum(x["length_days"] for x in dense) / (S[-1]["end_bjd"] - S[0]["start_bjd"]), 4)}
    # where the catalogue puts the peaks: RMDC26 concentrates t0 in the high-cadence seasons
    t0c = m["t0lens1"].values
    in_dense = np.zeros(len(t0c), bool)
    for x in dense:
        in_dense |= (t0c >= x["start_bjd"]) & (t0c <= x["end_bjd"])
    out["survey_geometry"]["frac_catalogue_t0_in_high_cadence_seasons"] = {L: round(float(in_dense[(lab == L).values].mean()), 4) for L in (L1, L2, L3)}
    # the catalogue baseline (Source_F146 + 2.5 log10 fs_F146) against the empirical quiescent baseline of the scored events
    sc = [e for e, r in rows.items() if r.get("dense") and "pred" in r and np.isfinite(r.get("m_base", np.nan))]
    cat = (m.loc[sc, "Source_F146"] + 2.5 * np.log10(m.loc[sc, "fs_F146"])).values
    off = np.array([rows[e]["m_base"] for e in sc]) - cat
    out["catalogue_baseline_offset_mag"] = {"median": round(float(np.median(off)), 4), "p5": round(float(np.percentile(off, 5)), 4),
                                            "p95": round(float(np.percentile(off, 95)), 4), "n": int(off.size),
                                            "sign": "empirical minus catalogue; positive = catalogue brighter"}
    out["command"] = " ".join(sys.argv)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k in ("catalogue", "selection_outcome", "scored_set", "survey_geometry")}, indent=1)[:3500])
    print(json.dumps(out["planets"], indent=1)[:1500]); print(json.dumps(out["vs_training_priors"]["by_class"], indent=1)[:1800])
    print(json.dumps(out.get("our_nonpspl_q"), indent=1), json.dumps(out["residual_fa_fspl5s_by_support"]))
    print("->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

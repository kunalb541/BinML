#!/usr/bin/env python3
"""RMDC26 / follow-up artifacts -> gulls_macros.tex + outputs/gulls_{transfer,gap}_table.tex.

Every number in paper/draft_gulls_section.tex is a \\bmlGulls* / \\bmlGap* / \\bmlSidecar* macro defined
here from a committed artifact, rounded ONCE from exact counts where the artifact has them, so the text
cannot drift from the data (the 2026-09-11 verification found typed numbers double-rounded, one table
cell taken from the wrong run, and a count that was an event id). FAIL CLOSED: a missing artifact or
key is fatal. --allow-missing exists only for development while artifacts are being produced; the
build never uses it.   Usage:  python paper/make_gulls_macros.py
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
G = os.path.join(REPO, "validation", "gulls")
L, MISSING = [], []
ALLOW = "--allow-missing" in sys.argv


def load(name):
    p = os.path.join(G, name)
    if not os.path.exists(p):
        if ALLOW:
            MISSING.append(name); return None
        raise SystemExit(f"FATAL: {p} missing")
    return json.load(open(p))


def cmd(name, value):
    assert name.isalpha(), f"macro {name}: TeX control words are letters only"
    if isinstance(value, float) and not math.isfinite(value):
        raise SystemExit(f"FATAL: macro {name} is {value!r}")
    if isinstance(value, str) and value.strip().lower() in ("nan", "inf", "-inf", "none", ""):
        raise SystemExit(f"FATAL: macro {name} formatted as {value!r}")
    L.append(r"\newcommand{\%s}{%s}" % (name, str(value).replace(",", "{,}")))


pct = lambda x: f"{100 * x:.1f}"
pct0 = lambda x: f"{100 * x:.0f}"
two = lambda x: f"{x:.2f}"
three = lambda x: f"{x:.3f}"
ratio = lambda kn: kn[0] / kn[1]
CLS = {"RMDC26_1S1L_ML": "Single", "RMDC26_1S2L_ML": "Planet", "RMDC26_2S2L_ML": "PlanetBin"}


def sci(x):
    e = int(math.floor(math.log10(abs(x)))); m = x / 10 ** e
    return f"{m:.1f}\\times10^{{{e}}}"


# ------------------------------------------------------------------ the paper's frozen threshold
cmd("bmlThreshold", three(json.load(open(os.path.join(HERE, "results", "metrics.json")))["headline"]["threshold"]))
GS = load("gap_sensitivity.json")
if GS:
    sg1 = GS["single_gap_by_length_h"]
    cmd("bmlGapSensNone", two(GS["no_gaps"]["PSPL"]["recall"]))
    assert abs(sg1["0.5"]["PSPL"]["recall"] - GS["no_gaps"]["PSPL"]["recall"]) < 1e-9, "0.5 h gap no longer free"
    cmd("bmlGapSensTwo", two(min(sg1["1.0"]["PSPL"]["recall"], sg1["2.0"]["PSPL"]["recall"])))
    cmd("bmlGapSensN", str(GS["n_per_class"]))

# ------------------------------------------------------------------ dataset facts
F = load("rmdc26_dataset_facts.json")
if F:
    cat = F["catalogue"]
    cmd("bmlGullsNcat", f"{cat['n_events']:,}")
    for k, nm in CLS.items():
        cmd(f"bmlGullsNcat{nm}", f"{cat['by_class'][k]['n']:,}")
    si = F["selection_inputs"]
    cmd("bmlGullsAmpMedSingle", three(si["RMDC26_1S1L_ML"]["median_host_peak_amp_mag"]))
    cmd("bmlGullsAmpFracSingle", pct0(si["RMDC26_1S1L_ML"]["frac_host_peak_amp_ge_0.1"]))
    cmd("bmlGullsSubdaySingle", pct0(si["RMDC26_1S1L_ML"]["frac_subday_tE_all"]))
    cmd("bmlGullsSubdayPlanet", f"{100 * si['RMDC26_1S2L_ML']['frac_subday_tE_all']:.2f}")
    cmd("bmlGullsSubdayPlanetBin", f"{100 * si['RMDC26_2S2L_ML']['frac_subday_tE_all']:.2f}")
    so = F["selection_outcome"]; t = so["total"]; n = so["n_eligible"]
    cmd("bmlGullsEligibleN", f"{n:,}")
    cmd("bmlGullsScoredPct", pct(t["scored, high-cadence season"] / n))
    cmd("bmlGullsBetweenPct", pct(t["peak between seasons"] / n))
    cmd("bmlGullsOutsidePct", pct((t["peak before first season"] + t["peak after last season"]) / n))
    cmd("bmlGullsLowcadPct", pct(t["scored, low-cadence season"] / n))
    cmd("bmlGullsNoBaseN", str(t["no usable baseline or F146"]))
    ss = F["scored_set"]
    cmd("bmlGullsN", f"{ss['n']:,}")
    for k, nm in CLS.items():
        cmd(f"bmlGullsN{nm}", f"{ss['by_class'][k]:,}")
    cmd("bmlGullsPlanetFracW", pct0(ss["rate_weighted_planetary_fraction"]))
    pl = F["planets"]
    cmd("bmlGullsQmedPlanet", sci(pl["median_Planet_q_scored"]["RMDC26_1S2L_ML"]))
    cmd("bmlGullsQmedPlanetBin", sci(pl["median_Planet_q_scored"]["RMDC26_2S2L_ML"]))
    cmd("bmlGullsQksExp", str(int(math.floor(math.log10(pl["ks_log_q_1S2L_vs_2S2L_scored"]["p"])))))
    cmd("bmlGullsBinSrcPct", pct0(pl["frac_2S2L_binary_source"]["scored"]))
    cmd("bmlGullsDetCut", f"{min(pl['gulls_detection_cut_ObsGroup_0_chi2_min'][k] for k in ('RMDC26_1S2L_ML', 'RMDC26_2S2L_ML')):.0f}")
    cut = pl["host_amplitude_cut_by_s"]
    def pass_frac(k, bins):
        num = sum(cut[f"{k}|s[{b}"]["frac_of_class"] * cut[f"{k}|s[{b}"]["frac_passing_amp_cut"] for b in bins)
        return num / sum(cut[f"{k}|s[{b}"]["frac_of_class"] for b in bins)
    cmd("bmlGullsWidePass", pct0(pass_frac("RMDC26_1S2L_ML", ("2,5)", "5,inf)"))))
    cmd("bmlGullsWideFrac", pct0(sum(cut[f"RMDC26_1S2L_ML|s[{b}"]["frac_of_class"] for b in ("2,5)", "5,inf)"))))
    cmd("bmlGullsResonantPass", pct0(cut["RMDC26_1S2L_ML|s[0.5,2)"]["frac_passing_amp_cut"]))
    vt = F["vs_training_priors"]["by_class"]
    for k, nm in CLS.items():
        cmd(f"bmlGullsBright{nm}", pct0(vt[k]["frac_m_base_lt_prior_min"]))
        cmd(f"bmlGullsFsLow{nm}", pct0(vt[k]["frac_fs_lt_prior_min"]))
    cmd("bmlGullsRhoPnn", two(vt["RMDC26_1S1L_ML"]["rho"]["p99"]))
    cmd("bmlGullsNrhoGtOne", str(vt["RMDC26_1S1L_ML"]["rho"]["n_gt_1"]))
    rf = F["residual_fa_fspl5s_by_support"]
    cmd("bmlGullsFaBright", pct(rf["m_base_lt_20"]["fa"])); cmd("bmlGullsFaInside", pct(rf["m_base_20_25"]["fa"]))
    oq = F["our_nonpspl_q"]
    cmd("bmlOursNonpsplQmed", two(oq["median_q"])); cmd("bmlOursNonpsplStellarPct", pct0(oq["frac_q_gt_1e-2"]))
    sg = F["survey_geometry"]
    cmd("bmlGullsSeasonDays", f"{min(sg['high_cadence_length_days']):.1f}")
    cmd("bmlGullsPauseHoursLo", f"{min(sg['f146_pause_hours_per_season']):.0f}")
    cmd("bmlGullsPauseHoursHi", f"{max(sg['f146_pause_hours_per_season']):.0f}")
    cmd("bmlGullsGapLo", f"{min(sg['inter_season_gaps_days']):.0f}"); cmd("bmlGullsGapHi", f"{max(sg['inter_season_gaps_days']):.0f}")
    cmd("bmlGullsObservedDays", f"{sg['observed_days']:.0f}"); cmd("bmlGullsSpanDays", f"{sg['mission_span_days']:,.0f}")
    cmd("bmlGullsPartialBinsPct", pct0(min(sg["f146_bins_with_frac_lt_1"]) / 864))

# ------------------------------------------------------------------ transfer table + text numbers
T = load("transfer_tradeoff_all.json")
ROWS = [("shipped", "shipped"), ("ft_g08e12", "gap augmentation (g08e12)"),
        ("pspl5s_ctrl_g08", "\\quad + 12 epochs, point-source single lenses (control)"),
        ("fspl5s_g08", "\\quad + 12 epochs, finite-source single lenses"),
        ("fspl5s_espl_g08", "\\quad + 12 epochs, finite-source, smooth magnification")]
if T:
    M = T["models"]
    def fa(m):
        return ratio(M[m]["frozen_threshold"]["fa_1S1L_k_n"])
    def at(m, t):
        return next(a for a in M[m]["recall_at_matched_fa"] if abs(a["fa_target"] - t) < 1e-9)
    def rec(m, t, cls="1S2L"):
        return ratio(at(m, t)[f"recall_{cls}_k_n"])
    budgets = T["budgets"]; mid = budgets[2]
    cmd("bmlGullsBudgetLo", pct(budgets[0])); cmd("bmlGullsBudgetMid", pct(mid)); cmd("bmlGullsBudgetHi", pct(budgets[3]))
    for m, nm in (("shipped", "Shipped"), ("ft_g08e12", "Gapaware"), ("fspl5s_g08", "Fspl"), ("fspl_g08", "FsplOne"),
                  ("fspl5_g08", "FsplFive"), ("fspl5s_noisy_g08", "Noisy"), ("fspl5s_v2_g08", "Onset"),
                  ("pspl5s_ctrl_g08", "Ctrl"), ("fspl5s_espl_g08", "Espl")):
        if m not in M:
            if ALLOW:
                MISSING.append(f"tradeoff:{m}"); continue
            raise SystemExit(f"FATAL: {m} missing from transfer_tradeoff_all.json")
        cmd(f"bmlGullsFa{nm}", pct(fa(m)))
        cmd(f"bmlGullsRec{nm}", three(rec(m, mid))); cmd(f"bmlGullsRecBin{nm}", three(rec(m, mid, "2S2L")))
        cmd(f"bmlGullsRecAtRecall{nm}", three(ratio(M[m]["frozen_threshold"]["recall_1S2L_k_n"])))
        bins = M[m]["fa_1S1L_by_rho_over_u0"]
        cmd(f"bmlGullsRhoLo{nm}", three(bins[0]["k"] / bins[0]["n"])); cmd(f"bmlGullsRhoHi{nm}", three(bins[-1]["k"] / bins[-1]["n"]))
    cmd("bmlGullsNcheckpoints", str(len([k for k in M if k != "shipped" and not k.startswith("sched_")])))
    if "sched_rand" in M:            # same recipe as g08e12 on another pool: the closest thing to a seed replicate
        d_ = [abs(rec("sched_rand", t) - rec("ft_g08e12", t)) for t in (budgets[0], mid, budgets[3])]
        cmd("bmlSeedSpreadLo", three(min(d_))); cmd("bmlSeedSpreadHi", three(max(d_)))
    elif not ALLOW:
        raise SystemExit("FATAL: sched_rand missing from transfer_tradeoff_all.json (seed-spread estimate)")
    lines = []
    for m, label in ROWS:
        if m not in M:
            continue
        b = M[m]["fa_1S1L_by_rho_over_u0"]
        cells = [pct(fa(m))] + [three(rec(m, t)) for t in (budgets[0], mid, budgets[3])] + [three(x["k"] / x["n"]) for x in (b[0], b[3], b[4], b[5])]
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
    ach = {t: max(abs(at(m, t)["fa_achieved_exact"] - t) for m, _ in ROWS if m in M) for t in (budgets[0], mid, budgets[3])}
    lines.append(f"% achieved single-lens false-alarm rate within {100 * max(ach.values()):.2f} points of each budget")
    os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
    open(os.path.join(HERE, "outputs", "gulls_transfer_table.tex"), "w").write("\n".join(lines) + "\n")
    cmd("bmlGullsBudgetSlack", f"{100 * max(ach.values()):.2f}")

C = load("transfer_colour_ablation.json")
if C:
    gains = [next(a for a in c["three_band"]["recall_at_matched_fa"] if a["fa_target"] == 0.052)["recall_1S2L"]
             - next(a for a in c["f146_only"]["recall_at_matched_fa"] if a["fa_target"] == 0.052)["recall_1S2L"] for c in C["models"].values()]
    cmd("bmlGullsColourGainLo", three(min(gains))); cmd("bmlGullsColourGainHi", three(max(gains)))

# ------------------------------------------------------------------ relabelling, floor, sub-day, noise
D = load("transfer_detectability_relabel.json")
if D:
    rl = D["relabelling"]
    und = lambda k: 1 - rl[k]["frac_NonPSPL"]
    cmd("bmlGullsTruthN", f"{D['n_events']:,}")
    ns = D["selection"]["n_selected"]
    cmd("bmlGullsTruthNSingle", f"{ns['RMDC26_1S1L_ML']:,}"); cmd("bmlGullsTruthNPlanet", f"{ns['RMDC26_1S2L_ML']:,}")
    if "median_n_f146" in D["relabelling"]["binaries_pooled"]:
        cmd("bmlGullsEpochExcess", pct0(D["relabelling"]["binaries_pooled"]["median_n_f146"] / 6912 - 1))
    elif not ALLOW:
        raise SystemExit("FATAL: relabel artifact lacks median_n_f146 (rerun its reduce)")
    cmd("bmlGullsUndetPlanet", pct0(und("RMDC26_1S2L_ML"))); cmd("bmlGullsUndetPlanetBin", pct0(und("RMDC26_2S2L_ML")))
    cmd("bmlGullsDetPlanet", pct0(rl["RMDC26_1S2L_ML"]["frac_NonPSPL"])); cmd("bmlGullsDetPlanetBin", pct0(rl["RMDC26_2S2L_ML"]["frac_NonPSPL"]))
    cmd("bmlGullsUndetPlanetW", pct0(1 - rl["RMDC26_1S2L_ML"]["wfrac_NonPSPL"]))
    bp = rl["binaries_pooled"]
    cmd("bmlGullsFailChi", pct0(bp["frac_fail_dchi2_anomaly"])); cmd("bmlGullsFloorVeto", pct0(bp["frac_floor_vetoed"]))
    cmd("bmlGullsDetRescaled", pct0(rl["RMDC26_1S2L_ML"]["frac_NonPSPL_dchi2_rescaled_to_6912_epochs"]))
    cmd("bmlGullsSinglePsplRefitPct", pct(rl["RMDC26_1S1L_ML"]["pspl_refit_anomaly_frac"]))
    b = D["models"]["fspl5s_g08"]; a = b["at_threshold"]["frozen"]
    r = lambda d: d["k"] / d["n"]
    cmd("bmlGullsRecDetPlanet", three(r(a["recall_1S2L_detectable"]))); cmd("bmlGullsRecDetPlanetBin", three(r(a["recall_2S2L_detectable"])))
    cmd("bmlGullsRecGenPlanet", three(r(a["recall_1S2L_generator"]))); cmd("bmlGullsRecGenPlanetBin", three(r(a["recall_2S2L_generator"])))
    cmd("bmlGullsFlagUndet", pct0(r(a["flag_binaries_undetectable"])))
    cmd("bmlGullsFlagFloorVeto", pct0(r(a["flag_binaries_floor_vetoed"])))
    sub = a["flag_1S1L_significant_subfloor_misfit"]
    cmd("bmlGullsSingleMisfitK", str(sub["k"])); cmd("bmlGullsSingleMisfitN", str(sub["n"]))
    cmd("bmlGullsSingleMisfitLo", pct0(sub["wilson95"][0])); cmd("bmlGullsSingleMisfitHi", pct0(sub["wilson95"][1]))
    fv = b["flag_floor_vetoed_by_amp_frozen"]; fvf = b["colour_ablation"]["flag_floor_vetoed_by_amp_frozen_f146only"]
    sub3 = [x for x in fv if x["amp_bin"][1] <= 0.02]; subf = [x for x in fvf if x["amp_bin"][1] <= 0.02]
    cmd("bmlGullsFloorFlagThreeLo", pct0(min(x["k"] / x["n"] for x in sub3))); cmd("bmlGullsFloorFlagThreeHi", pct0(max(x["k"] / x["n"] for x in sub3)))
    cmd("bmlGullsFloorFlagOneLo", pct0(min(x["k"] / x["n"] for x in subf))); cmd("bmlGullsFloorFlagOneHi", pct0(max(x["k"] / x["n"] for x in subf)))
    dc = b["recall_detectable_by_dchi2_frozen"]
    cmd("bmlGullsRecChiLo", two(dc[0]["k"] / dc[0]["n"])); cmd("bmlGullsRecChiHi", two(dc[-1]["k"] / dc[-1]["n"]))
    ca = b["colour_ablation"]["budget_0.052"]
    cmd("bmlGullsColDetOne", three(ca["f146_only"]["detectable"])); cmd("bmlGullsColDetThree", three(ca["three_band"]["detectable"]))
    cmd("bmlGullsColUndetOne", three(ca["f146_only"]["undetectable"])); cmd("bmlGullsColUndetThree", three(ca["three_band"]["undetectable"]))
    cmd("bmlGullsPrecSample", three(a["ontology_precision_sample_mix"])); cmd("bmlGullsPrecScored", three(a["ontology_precision_scored_mix"]))
    if "calibrated_seasons_fullpool" in b["at_threshold"]:
        s_ = b["at_threshold"]["calibrated_seasons_fullpool"]
        cmd("bmlSidecarRecDetPlanet", three(r(s_["recall_1S2L_detectable"]))); cmd("bmlSidecarRecDetPlanetBin", three(r(s_["recall_2S2L_detectable"])))
    elif not ALLOW:
        raise SystemExit("FATAL: relabel artifact lacks the recommended operating point (rerun its reduce)")
FS = load("floor_sensitivity.json")
if FS:
    byf = {round(x["floor_mag"], 3): x for x in FS["by_floor"]}
    cmd("bmlGullsFloorDetLo", pct0(byf[0.005]["relabelling"]["RMDC26_1S2L_ML"]["NonPSPL"]))
    cmd("bmlGullsFloorDetHi", pct0(byf[0.05]["relabelling"]["RMDC26_1S2L_ML"]["NonPSPL"]))
SD = load("transfer_subday.json")
if SD:
    m5 = SD["models"]["fspl5s_g08"]
    cmd("bmlGullsSubdayN", f"{SD['n_matched_dense_1S1L_subday']:,}"); cmd("bmlGullsSubdayTe", two(SD["te_days"]["median"]))
    cmd("bmlGullsSubdayFa", pct(m5["fa_at"]["frozen"]["fa"])); cmd("bmlGullsSubdayFaW", pct(m5["fa_at"]["frozen"]["fa_weighted"]))
    pv = [SD["models"][k]["argmax_distribution"].get("PeriodicVar", 0) for k in SD["models"]]
    cmd("bmlGullsSubdayPerLo", pct0(min(pv))); cmd("bmlGullsSubdayPerHi", pct0(max(pv)))
NM = load("gulls_noise_model.json")
if NM:
    bm = NM["by_mag"]
    cmd("bmlGullsNoiseBright", two(bm[0]["ratio_median"])); cmd("bmlGullsNoiseFaint", two(bm[-1]["ratio_median"]))
    cmd("bmlGullsNoiseBkg", f"{NM['best_fit']['bkg_mult']['mult']:.0f}"); cmd("bmlGullsNoiseGlobal", two(NM["best_fit"]["noise_mult"]["mult"]))

# ------------------------------------------------------------------ calibration, occupancy, schedule, seasons, cascade
K = load("gapped_threshold_fspl5s_g08_seasons.json")
if K:
    arm = K["arms"]["rmdc26_gapped"]; pool = arm["pool"]; sl = pool["slice_thresholds"]
    cmd("bmlSidecarThr", three(pool["full_pool"]["threshold"]))
    cmd("bmlSidecarThrLo", three(sl["p16"])); cmd("bmlSidecarThrHi", three(sl["p84"]))
    g = arm["gulls_at_full_pool_threshold"]
    cmd("bmlSidecarFa", pct(g["fa_1S1L_k"] / g["fa_1S1L_n"]))
    cmd("bmlSidecarRecPlanet", two(g["recall_1S2L_k"] / g["recall_1S2L_n"])); cmd("bmlSidecarRecPlanetBin", two(g["recall_2S2L_k"] / g["recall_2S2L_n"]))
    lo, hi = arm["gulls_at_slice_p16_p84"]
    cmd("bmlSidecarFaLo", pct(hi["fa_1S1L_k"] / hi["fa_1S1L_n"])); cmd("bmlSidecarFaHi", pct(lo["fa_1S1L_k"] / lo["fa_1S1L_n"]))
    cmd("bmlSidecarPoolPrev", pct(pool["prevalence_nonpspl_population_weighted"]))
    cmd("bmlSidecarPoolCompl", three(pool["full_pool"]["completeness"]))
O = load("occupancy_sensitivity.json")
if O:
    o = O["models"]["fspl5s_g08"]
    cmd("bmlOccN", f"{o['n']:,}")
    cmd("bmlOccFaObs", pct(o["fa_1S1L"]["as_observed"])); cmd("bmlOccFaOne", pct(o["fa_1S1L"]["occupancy_set_to_1"]))
    cmd("bmlOccRecObs", two(o["recall_1S2L"]["as_observed"])); cmd("bmlOccRecOne", two(o["recall_1S2L"]["occupancy_set_to_1"]))
S = load("schedule_finetune.json")
if S and ALLOW and "n_pool" not in next(iter(S["heldout_eval"].values()))["clean"]:
    MISSING.append("schedule_finetune.json (pre-rerun version)"); S = None
if S:
    H = S["heldout_eval"]
    first = next(iter(H.values()))
    cmd("bmlGapHeldN", f"{first['clean']['n_scored_test']:,}" if first["clean"].get("n_scored_test") else "FATAL")
    cmd("bmlGapPoolN", f"{first['clean']['n_pool']:,}")
    rows = [("shipped", "shipped"), ("ft_g08e12", "gap augmentation, g08e12 (other pool)"),
            ("rand", "random gaps"), ("rand_norelabel", "random gaps, relabel off"),
            ("sched_norelabel", "first-season mask, relabel off"), ("sched_seasons", "measured seasons, relabel off")]
    lines = []
    for k, label in rows:
        e = H[k]; ms = e["rmdc26_seasons"]; rc = ms["argmax_population"]["recall"]
        lines.append(label + " & " + " & ".join([three(e["clean"]["ap"]), three(ms["ap"]), two(rc["PSPL"]), two(rc["Flat"]), two(rc["NonPSPL"]),
                                                 three(ms["argmax_population"]["macro_f1"])]) + " \\\\")
    open(os.path.join(HERE, "outputs", "gulls_gap_table.tex"), "w").write("\n".join(lines) + "\n")
    sh = H["shipped"]
    for k, nm in (("PSPL", "Pspl"), ("Flat", "Flat"), ("NonPSPL", "Nonpspl"), ("Eruptive", "Eruptive"), ("LongPeriodVar", "Lpv")):
        cmd(f"bmlGapShipped{nm}Clean", two(sh["clean"]["argmax_population"]["recall"][k]))
        cmd(f"bmlGapShipped{nm}Sched", two(sh["rmdc26_seasons"]["argmax_population"]["recall"][k]))
    cmd("bmlGapShippedPrecClean", two(sh["clean"]["argmax_population"]["precision"]["NonPSPL"]))
    cmd("bmlGapShippedPrecSched", two(sh["rmdc26_seasons"]["argmax_population"]["precision"]["NonPSPL"]))
    for k, nm in (("shipped", "Shipped"), ("ft_g08e12", "Gapaware"), ("rand", "Rand"), ("rand_norelabel", "RandNorelabel"),
                  ("sched_norelabel", "FirstSeason"), ("sched_seasons", "Seasons")):
        cmd(f"bmlGapAp{nm}", three(H[k]["rmdc26_seasons"]["ap"])); cmd(f"bmlGapApClean{nm}", three(H[k]["clean"]["ap"]))
        cmd(f"bmlGapFone{nm}", three(H[k]["rmdc26_seasons"]["argmax_population"]["macro_f1"]))
        cmd(f"bmlGapFoneClean{nm}", three(H[k]["clean"]["argmax_population"]["macro_f1"]))
    rx = S["legacy_relabel_exposure"]
    cmd("bmlGapRelabelPct", pct(rx["frac_of_nonpspl_labelled"]))
MS = load("transfer_multiseason.json")
if MS and ALLOW and "population_context" not in MS:
    MISSING.append("transfer_multiseason.json (pre-rerun version)"); MS = None
if MS:
    pc = MS["population_context"]["by_class"]; bc = MS["by_class"]
    cmd("bmlMsN", f"{bc['RMDC26_1S2L_ML']['n']:,}")
    for k, nm in CLS.items():
        cmd(f"bmlMsScorable{nm}", pct0(pc[k]["frac_between_with_dense_adjacent_season"]))
    for k, nm in (("RMDC26_1S2L_ML", "Planet"), ("RMDC26_2S2L_ML", "PlanetBin")):
        cmd(f"bmlMsDet{nm}", pct0(bc[k]["frac_detectable_anomaly_of_scorable"]))
        cmd(f"bmlMsDetHost{nm}", pct0(bc[k]["frac_detectable_anomaly_of_host_visible"]))
    mm = MS["models"]["fspl5s_g08"]; fz = mm["at_threshold"]["frozen"]
    cmd("bmlMsFa", pct(fz["fa_1S1L"]["rate"])); cmd("bmlMsFaHi", pct(fz["fa_1S1L"]["wilson95"][1]))
    cmd("bmlMsRecDetPlanet", two(fz["recall_1S2L_detectable"]["rate"])); cmd("bmlMsRecDetPlanetBin", two(fz["recall_2S2L_detectable"]["rate"]))
    cmd("bmlMsRecGenPlanet", two(fz["recall_1S2L_generator"]["rate"]))
    b5 = mm["recall_at_matched_fa"]["0.052"]
    cmd("bmlMsRecBudgetPlanet", two(b5["recall_1S2L_detectable"])); cmd("bmlMsRecBudgetPlanetBin", two(b5["recall_2S2L_detectable"]))
    cmd("bmlMsBudgetThr", two(b5["threshold"])); cmd("bmlMsBudgetExceed", str(b5["n_1S1L_exceedances"]))
    if "calibrated_seasons_fullpool" in mm["at_threshold"]:
        cs = mm["at_threshold"]["calibrated_seasons_fullpool"]
        cmd("bmlMsSidecarFa", pct(cs["fa_1S1L"]["rate"])); cmd("bmlMsSidecarRecDetPlanet", two(cs["recall_1S2L_detectable"]["rate"]))
CG = load("cascade_gulls.json")
if CG and ALLOW and "inhouse_reference" not in CG:
    MISSING.append("cascade_gulls.json (pre-rescan version)"); CG = None
if CG:
    R_ = CG["results"]; pr = R_["fspl5s_g08|f146|frozen"]; t_ = pr["timing"]
    cmd("bmlCgScanN", f"{sum(CG['n_scanned'].values()):,}"); cmd("bmlCgElig", f"{t_['n_eligible']:,}")
    cmd("bmlCgPrem", pct(t_["premature_frac"])); cmd("bmlCgPremLo", pct(t_["premature_ci95"][0])); cmd("bmlCgPremHi", pct(t_["premature_ci95"][1]))
    cmd("bmlCgLag", f"{t_['median_lag_nonpremature_days']:+.1f}"); cmd("bmlCgDet", pct0(t_["detected_frac"]))
    bu = pr["burden"]["RMDC26_1S1L_ML"]
    cmd("bmlCgSingleAlert", pct(bu["alert_frac_per_season"])); cmd("bmlCgSingleAlertDay", two(bu["alerts_per_1000_events_per_day"]))
    sp = pr["streaming_purity"]
    cmd("bmlCgPurOne", pct0(sp["planetary_prevalence_0.01"]["purity_detectable_anomaly_alerts"]))
    cmd("bmlCgPurFive", pct0(sp["planetary_prevalence_0.05"]["purity_detectable_anomaly_alerts"]))
    kw = [k for k in sp if k.endswith("_rmdc26_scored_rate_weighted")][0]
    cmd("bmlCgPurRmdc", pct0(sp[kw]["purity_detectable_anomaly_alerts"]))
    k_cal = "fspl5s_g08|f146|calibrated_seasons_fullpool"
    if k_cal in R_:
        tc_ = R_[k_cal]["timing"]
        cmd("bmlCgPremCal", pct(tc_["premature_frac"])); cmd("bmlCgDetCal", pct0(tc_["detected_frac"])); cmd("bmlCgLagCal", f"{tc_['median_lag_nonpremature_days']:+.1f}")
        cmd("bmlCgSingleAlertCal", pct(R_[k_cal]["burden"]["RMDC26_1S1L_ML"]["alert_frac_per_season"]))
        cmd("bmlCgPurOneCal", pct0(R_[k_cal]["streaming_purity"]["planetary_prevalence_0.01"]["purity_detectable_anomaly_alerts"]))
    ih = CG["inhouse_reference"]["by_mass_ratio"]
    for st, nm in (("giant", "Giant"), ("neptune", "Neptune")):
        cmd(f"bmlCgIn{nm}N", str(ih[st]["n_eligible"])); cmd(f"bmlCgIn{nm}Det", pct0(ih[st]["detection_fraction"]))
        cmd(f"bmlCgIn{nm}Prem", pct(ih[st]["premature_rate_of_eligible"]))
OUT = os.path.join(HERE, "gulls_macros.tex")
open(OUT, "w").write("% AUTO-GENERATED by make_gulls_macros.py -- do not edit.\n" + "\n".join(L) + "\n")
print(f"wrote {OUT} ({len(L)} macros)" + (f"; MISSING (dev only): {MISSING}" if MISSING else ""))

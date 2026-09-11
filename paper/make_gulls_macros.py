#!/usr/bin/env python3
"""RMDC26 / follow-up artifacts -> gulls_macros.tex + outputs/gulls_{transfer,gap}_table.tex.

Every number in paper/draft_gulls_section.tex is a \\bmlGulls* / \\bmlGap* / \\bmlSidecar* macro defined
here from a committed artifact, rounded ONCE from exact counts where the artifact has them, so the text
cannot drift from the data (the 2026-09-11 verification found typed numbers double-rounded, one table
cell taken from the wrong run, and a count that was an event id). FAIL CLOSED: a missing artifact, key
or block is fatal, and nothing is written unless every input passed (the 2026-09-12 re-verification
found tables written before later inputs were checked, and four silent fallbacks). --allow-missing
exists only for development while artifacts are being produced; the build never uses it.

REC names the recommended gap-aware checkpoint: every checkpoint-specific macro that the text attributes
to "the recommended checkpoint" (operating point, labels, floor, sub-day, occupancy, between seasons,
cascade) is read from REC's blocks. R3 is the finite-source round-3 checkpoint used in comparisons.
Usage:  python paper/make_gulls_macros.py
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
G = os.path.join(REPO, "validation", "gulls")
L, MISSING, TABLES = [], [], {}
ALLOW = "--allow-missing" in sys.argv
REC, R3 = "fspl5s_seasons_g08", "fspl5s_g08"


def need(ok, msg):
    """Fail closed on a missing block or key (dev mode records it instead)."""
    if not ok:
        if ALLOW:
            MISSING.append(msg)
        else:
            raise SystemExit(f"FATAL: {msg}")
    return ok


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
    if isinstance(value, str) and value.strip().lower() in ("nan", "inf", "-inf", "none", "", "fatal"):
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
    bp_ = [vt[k]["frac_m_base_lt_prior_min"] for k in ("RMDC26_1S2L_ML", "RMDC26_2S2L_ML")]
    cmd("bmlGullsBrightPlanetLo", pct0(min(bp_))); cmd("bmlGullsBrightPlanetHi", pct0(max(bp_)))
    fl_ = [vt[k]["frac_fs_lt_prior_min"] for k in CLS]
    cmd("bmlGullsFsLowLo", pct0(min(fl_))); cmd("bmlGullsFsLowHi", pct0(max(fl_)))
    bc_ = F["selection_outcome"]["by_class"]
    def share(k, key):
        return bc_[k][key] / sum(v for kk, v in bc_[k].items())
    cmd("bmlGullsBetweenSingle", pct0(share("RMDC26_1S1L_ML", "peak between seasons")))
    bpl = [share(k, "peak between seasons") for k in ("RMDC26_1S2L_ML", "RMDC26_2S2L_ML")]
    cmd("bmlGullsBetweenPlanetLo", pct0(min(bpl))); cmd("bmlGullsBetweenPlanetHi", pct0(max(bpl)))
    lpl = [share(k, "scored, low-cadence season") for k in ("RMDC26_1S2L_ML", "RMDC26_2S2L_ML")]
    cmd("bmlGullsLowcadSingle", pct0(share("RMDC26_1S1L_ML", "scored, low-cadence season")))
    cmd("bmlGullsLowcadPlanetLo", pct0(min(lpl))); cmd("bmlGullsLowcadPlanetHi", pct0(max(lpl)))
    need("catalogue_baseline_offset_mag" in F, "rmdc26_dataset_facts.json lacks catalogue_baseline_offset_mag (rerun it)")
    if "catalogue_baseline_offset_mag" in F:
        cmd("bmlGullsBaselineOffset", two(F["catalogue_baseline_offset_mag"]["median"]))
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
    pb_ = sorted(sg["f146_bins_with_frac_lt_1"]); cmd("bmlGullsPartialBinsPct", pct0(pb_[len(pb_) // 2] / 864))   # median season
    need("pause_start_days_common_to_all_seasons" in sg, "rmdc26_dataset_facts.json lacks the schedule facts (rerun it)")
    if "pause_start_days_common_to_all_seasons" in sg:
        for d_, nm in zip(sg["pause_start_days_common_to_all_seasons"], ("A", "B", "C")):
            cmd(f"bmlGullsCommonPause{nm}", f"{d_:.1f}")
        cmd("bmlGullsSeasonShort", f"{sg['season_shortfall_vs_72d_window_days']:.1f}")
        cmd("bmlGullsDenseSpanPct", pct0(sg["high_cadence_fraction_of_span"]))
        tz = sg["frac_catalogue_t0_in_high_cadence_seasons"].values()
        cmd("bmlGullsTzeroDenseLo", pct0(min(tz))); cmd("bmlGullsTzeroDenseHi", pct0(max(tz)))
# our own prior's share of short events (truncated log-normal tE prior, pipeline/priors.py)
sys.path.insert(0, REPO)
from pipeline.priors import EventPriors as _P  # noqa: E402
_p = _P(); _mu, _sd = math.log10(_p.TE_MEDIAN_DAYS), _p.TE_SIGMA_DEX
_cdf = lambda x: 0.5 * (1 + math.erf((math.log10(x) - _mu) / (_sd * math.sqrt(2))))
cmd("bmlPriorShortPct", f"{100 * (_cdf(3.0) - _cdf(_p.TE_MIN_DAYS)) / (_cdf(_p.TE_MAX_DAYS) - _cdf(_p.TE_MIN_DAYS)):.1f}")

# ------------------------------------------------------------------ preprocessing sensitivity (epoch- vs raw-pooled binning)
RP0, RP1 = load("transfer_full_reduced.json"), load("transfer_full_reduced_rawpool.json")
if RP0 and RP1:
    sh0 = RP0["by_class"]["RMDC26_1S1L_ML"]["per_model"]["shipped"]["over_thr"]["rate"]
    sh1 = RP1["by_class"]["RMDC26_1S1L_ML"]["per_model"]["shipped"]["over_thr"]["rate"]
    cmd("bmlGullsRawpoolFa", pct(sh1)); cmd("bmlGullsRawpoolShift", pct0(sh1 - sh0))

# ------------------------------------------------------------------ transfer table + text numbers
T = load("transfer_tradeoff_all.json")
ROWS = [("shipped", "shipped"), ("ft_g08e12", "gap augmentation (g08e12)"),
        ("pspl5s_ctrl_g08", "\\quad + 12 epochs, point-source single lenses (control)"),
        ("fspl5s_g08", "\\quad + 12 epochs, finite-source single lenses"),
        ("fspl5s_seasons_g08", "\\quad + 12 epochs, finite source, measured-season pauses (recommended)")]
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
                  ("pspl5s_ctrl_g08", "Ctrl"), ("fspl5s_seasons_g08", "Combined"), ("fspl5s_espl_g08", "Espl"),
                  ("sched_rand_norelabel", "RandNorelabel"), ("sched_sched_seasons", "Seasons")):
        if m not in M:
            if ALLOW:
                MISSING.append(f"tradeoff:{m}"); continue
            raise SystemExit(f"FATAL: {m} missing from transfer_tradeoff_all.json")
        cmd(f"bmlGullsFa{nm}", pct(fa(m)))
        cmd(f"bmlGullsRec{nm}", three(rec(m, mid))); cmd(f"bmlGullsRecBin{nm}", three(rec(m, mid, "2S2L")))
        cmd(f"bmlGullsRecAtRecall{nm}", three(ratio(M[m]["frozen_threshold"]["recall_1S2L_k_n"])))
        bins = M[m]["fa_1S1L_by_rho_over_u0"]
        cmd(f"bmlGullsRhoLo{nm}", three(bins[0]["k"] / bins[0]["n"])); cmd(f"bmlGullsRhoHi{nm}", three(bins[-1]["k"] / bins[-1]["n"]))
        cmd(f"bmlGullsRhoMid{nm}", three(bins[-2]["k"] / bins[-2]["n"]))
        need("mean_recall_1S2L_fa_le_0p3_exact" in M[m], f"{m}: no unrounded mean recall (rerun gulls_summary_tables.py)")
        cmd(f"bmlGullsMeanRec{nm}", three(M[m].get("mean_recall_1S2L_fa_le_0p3_exact", M[m]["mean_recall_1S2L_fa_le_0p3"])))
        if need("weighted" in M[m], f"{m}: no weighted block (rerun gulls_summary_tables.py)"):
            W_ = M[m]["weighted"]; wat = next(a for a in W_["recall_at_matched_fa"] if abs(a["fa_target"] - mid) < 1e-9)
            cmd(f"bmlGullsFaW{nm}", pct(W_["frozen_threshold"]["fa_1S1L"]))
            cmd(f"bmlGullsRecW{nm}", three(wat["recall_1S2L"])); cmd(f"bmlGullsRecBinW{nm}", three(wat["recall_2S2L"]))
    # every fine-tuned checkpoint scored on RMDC26 (the multiplicity behind "optimistic")
    cmd("bmlGullsNcheckpoints", str(len([k for k in M if k != "shipped"])))
    need("sample" in T, "transfer_tradeoff_all.json lacks the sample block (rerun gulls_summary_tables.py)")
    if "sample" in T:
        cmd("bmlGullsShortSingle", pct0(T["sample"]["frac_1S1L_tE_lt_3d"]["unweighted"]))
        cmd("bmlGullsShortSingleW", pct0(T["sample"]["frac_1S1L_tE_lt_3d"]["weighted"]))
        cmd("bmlGullsTeMedSingle", f"{T['sample']['median_tE_days']['RMDC26_1S1L_ML']:.1f}")
        cmd("bmlGullsTeMedPlanet", f"{T['sample']['median_tE_days']['RMDC26_1S2L_ML']:.1f}")
        ke = T["sample"]["kish_n_eff"]
        cmd("bmlGullsNeffSingle", f"{ke['RMDC26_1S1L_ML']:,.0f}"); cmd("bmlGullsNeffPlanet", f"{ke['RMDC26_1S2L_ML']:,.0f}")
    for m, nm in ((REC, "Rec"), (R3, "Fspl")):                  # false alarms by timescale, weighted
        te_ = M[m]["weighted"]["fa_1S1L_by_tE"]
        cmd(f"bmlGullsFaTeLo{nm}", pct(te_[0]["fa_unweighted"])); cmd(f"bmlGullsFaTeHi{nm}", pct(te_[-1]["fa_unweighted"]))
    # paired differences on the same events (class-stratified bootstrap over events; not training-seed variance)
    need("paired_differences" in T, "transfer_tradeoff_all.json lacks paired_differences (rerun with --pairs)")
    PD = T.get("paired_differences", {}).get("pairs", {})
    def pdm(pair, nm):
        if not need(pair in PD, f"paired difference {pair} missing"):
            return
        for wt, suf in (("unweighted", ""), ("weighted", "W")):
            for key, knm in (("recall_1S2L_at_0.052", "Rec"), ("recall_2S2L_at_0.052", "RecBin"), ("fa_frozen", "Fa")):
                d_ = PD[pair][wt][key]; f_ = pct if knm == "Fa" else three
                cmd(f"bmlDiff{nm}{knm}{suf}", ("+" if d_["diff"] >= 0 else "") + f_(d_["diff"]))
                cmd(f"bmlDiff{nm}{knm}{suf}Lo", f_(d_["ci95"][0])); cmd(f"bmlDiff{nm}{knm}{suf}Hi", f_(d_["ci95"][1]))
        dd = [PD[pair]["weighted"][f"recall_1S2L_at_{t}"]["diff"] for t in ("0.02", "0.052", "0.117")]
        cmd(f"bmlDiff{nm}RecWMin", three(min(dd))); cmd(f"bmlDiff{nm}RecWMax", three(max(dd)))
        lo_ = [PD[pair]["weighted"][f"recall_1S2L_at_{t}"]["ci95"][0] for t in ("0.02", "0.052", "0.117")]
        cmd(f"bmlDiff{nm}RecWLoMin", three(min(lo_)))
        if nm == "Comb" and not min(lo_) > 0:          # the draft says every weighted interval is above zero
            raise SystemExit(f"FATAL: the combined-arm weighted recall intervals no longer all exclude zero ({lo_}); revise the text")
    for pair, nm in ((f"{REC}-{R3}", "Comb"), ("fspl5s_g08-pspl5s_ctrl_g08", "Phys"), ("pspl5s_ctrl_g08-ft_g08e12", "Train"),
                     ("sched_sched_seasons-sched_rand_norelabel", "Sched"), ("fspl5s_espl_g08-fspl5s_g08", "Espl")):
        pdm(pair, nm)
    if "sched_rand-ft_g08e12" in PD:                  # two runs of one recipe on different pools, weighted
        cmd("bmlPoolSpreadW", three(max(abs(PD["sched_rand-ft_g08e12"]["weighted"][f"recall_1S2L_at_{t}"]["diff"]) for t in ("0.02", "0.052", "0.117"))))
    need("sched_sched_seasons" in M and "by_season" in M["sched_sched_seasons"] and "by_season" in M.get("sched_rand_norelabel", {}),
         "per-season blocks missing (rerun gulls_summary_tables.py --by-season)")
    if "sched_sched_seasons" in M and "sched_rand_norelabel" in M:
        wins = sum(1 for si, b in M["sched_sched_seasons"].get("by_season", {}).items()
                   if next(a["recall_1S2L"] for a in b["recall_at_matched_fa"] if a["fa_target"] == mid)
                   > next(a["recall_1S2L"] for a in M["sched_rand_norelabel"]["by_season"][si]["recall_at_matched_fa"] if a["fa_target"] == mid))
        cmd("bmlGullsSeasonWins", str(wins)); cmd("bmlGullsNseasons", str(len(M["sched_sched_seasons"].get("by_season", {}))))
    if "fspl5s_seasons_g08" in M:    # combined arm vs round 3: largest planetary-recall difference over the three budgets
        cmd("bmlGullsCombDiffMax", three(max(abs(rec("fspl5s_seasons_g08", t) - rec("fspl5s_g08", t)) for t in (budgets[0], mid, budgets[3]))))
    # smooth-magnification-only control vs round 3: recall at every budget, and the rho/|u0| bins
    cmd("bmlGullsEsplDiffMax", three(max(abs(rec("fspl5s_espl_g08", t) - rec("fspl5s_g08", t)) for t in (budgets[0], mid, budgets[3]))))
    cmd("bmlGullsEsplBinDiffMax", three(max(abs(x["k"] / x["n"] - y["k"] / y["n"]) for x, y in
                                            zip(M["fspl5s_espl_g08"]["fa_1S1L_by_rho_over_u0"], M["fspl5s_g08"]["fa_1S1L_by_rho_over_u0"]))))
    if "sched_rand" in M:            # same recipe as g08e12 on another pool: the closest thing to a seed replicate
        d_ = [abs(rec("sched_rand", t) - rec("ft_g08e12", t)) for t in (budgets[0], mid, budgets[3])]
        cmd("bmlSeedSpreadLo", three(min(d_))); cmd("bmlSeedSpreadHi", three(max(d_)))
    elif not ALLOW:
        raise SystemExit("FATAL: sched_rand missing from transfer_tradeoff_all.json (seed-spread estimate)")
    lines = []
    for m, label in ROWS:
        if not need(m in M, f"table row {m} missing"):
            continue
        b = M[m]["fa_1S1L_by_rho_over_u0"]; W_ = M[m]["weighted"]
        wat = next(a for a in W_["recall_at_matched_fa"] if abs(a["fa_target"] - mid) < 1e-9)
        cells = ([pct(fa(m)), pct(W_["frozen_threshold"]["fa_1S1L"])] + [three(rec(m, t)) for t in (budgets[0], mid, budgets[3])]
                 + [three(wat["recall_1S2L"])] + [three(x["k"] / x["n"]) for x in (b[0], b[3], b[4], b[5])])
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
    ach = {t: max(abs(at(m, t)["fa_achieved_exact"] - t) for m, _ in ROWS if m in M) for t in (budgets[0], mid, budgets[3])}
    lines.append(f"% achieved single-lens false-alarm rate within {100 * max(ach.values()):.2f} points of each budget")
    TABLES["gulls_transfer_table.tex"] = "\n".join(lines) + "\n"
    cmd("bmlGullsBudgetSlack", f"{100 * max(ach.values()):.2f}")

C = load("transfer_colour_ablation.json")
if C:
    gains = [next(a for a in c["three_band"]["recall_at_matched_fa"] if a["fa_target"] == 0.052)["recall_1S2L"]
             - next(a for a in c["f146_only"]["recall_at_matched_fa"] if a["fa_target"] == 0.052)["recall_1S2L"] for c in C["models"].values()]
    cmd("bmlGullsColourGainLo", three(min(gains))); cmd("bmlGullsColourGainHi", three(max(gains)))
    cmd("bmlGullsNcolour", str(len(C["models"])))
    need(REC in C["models"], f"colour ablation lacks {REC} (score it with --bands F146)")

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
    need(REC in D["models"], f"relabel artifact lacks {REC} (rerun its reduce with it)")
    b = D["models"][REC] if REC in D["models"] else D["models"][R3]; a = b["at_threshold"]["frozen"]
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
    need("k_undetectable" in ca["f146_only"], "relabel colour block lacks counts (rerun its reduce)")
    kn_ = lambda blk, key, n: blk.get(f"k_{key}", blk[key] * n) / n            # counts where stored: rounded once
    cmd("bmlGullsColDetOne", three(kn_(ca["f146_only"], "detectable", ca["n_detectable"])))
    cmd("bmlGullsColDetThree", three(kn_(ca["three_band"], "detectable", ca["n_detectable"])))
    cmd("bmlGullsColUndetOne", three(kn_(ca["f146_only"], "undetectable", ca["n_undetectable"])))
    cmd("bmlGullsColUndetThree", three(kn_(ca["three_band"], "undetectable", ca["n_undetectable"])))
    cmd("bmlGullsPrecSample", three(a["ontology_precision_sample_mix"])); cmd("bmlGullsPrecScored", three(a["ontology_precision_scored_mix"]))
    bm_ = next(x for x in b["recall_at_matched_fa"] if abs(x["fa_target"] - 0.052) < 1e-9)
    cmd("bmlGullsRecDetBudgetPlanet", two(bm_["recall_1S2L_detectable"])); cmd("bmlGullsRecDetBudgetPlanetBin", two(bm_["recall_2S2L_detectable"]))
    if need("calibrated_seasons_fullpool" in b["at_threshold"], "relabel artifact lacks the recommended operating point (rerun its reduce)"):
        s_ = b["at_threshold"]["calibrated_seasons_fullpool"]
        cmd("bmlSidecarRecDetPlanet", three(r(s_["recall_1S2L_detectable"]))); cmd("bmlSidecarRecDetPlanetBin", three(r(s_["recall_2S2L_detectable"])))
FS = load("floor_sensitivity.json")
if FS:
    byf = {round(x["floor_mag"], 3): x for x in FS["by_floor"]}
    cmd("bmlGullsFloorDetLo", pct0(byf[0.005]["relabelling"]["RMDC26_1S2L_ML"]["NonPSPL"]))
    cmd("bmlGullsFloorDetHi", pct0(byf[0.05]["relabelling"]["RMDC26_1S2L_ML"]["NonPSPL"]))
SD = load("transfer_subday.json")
if SD and need(REC in SD["models"], f"transfer_subday.json lacks {REC}"):
    m5 = SD["models"][REC]
    cmd("bmlGullsSubdayN", f"{SD['n_matched_dense_1S1L_subday']:,}"); cmd("bmlGullsSubdayTe", two(SD["te_days"]["median"]))
    cmd("bmlGullsSubdayFa", pct(m5["fa_at"]["frozen"]["fa"])); cmd("bmlGullsSubdayFaW", pct(m5["fa_at"]["frozen"]["fa_weighted"]))
    need(all("PeriodicVar" in SD["models"][k]["argmax_distribution"] for k in SD["models"]), "sub-day argmax lacks PeriodicVar")
    pv = [SD["models"][k]["argmax_distribution"].get("PeriodicVar", 0) for k in SD["models"]]
    cmd("bmlGullsSubdayPerLo", pct0(min(pv))); cmd("bmlGullsSubdayPerHi", pct0(max(pv)))
    ad_ = m5["argmax_distribution"]; cmd("bmlGullsSubdayMl", pct0(ad_.get("NonPSPL", 0) + ad_.get("PSPL", 0)))
NM = load("gulls_noise_model.json")
if NM:
    bm = NM["by_mag"]
    cmd("bmlGullsNoiseBright", two(bm[0]["ratio_median"])); cmd("bmlGullsNoiseFaint", two(bm[-1]["ratio_median"]))
    cmd("bmlGullsNoiseBkg", f"{NM['best_fit']['bkg_mult']['mult']:.0f}"); cmd("bmlGullsNoiseGlobal", two(NM["best_fit"]["noise_mult"]["mult"]))

# ------------------------------------------------------------------ calibration, occupancy, schedule, seasons, cascade
for tag, nm in (("fspl5s_g08", "Fspl"), ("pspl5s_ctrl_g08", "Ctrl")):
    FT = load(f"fspl_finetune_{tag}.json")
    if FT:
        pr_ = FT["models"][tag]["pspl_recall_by_rho_u0"]
        cmd(f"bmlHeldPsplMid{nm}", two(pr_["[1,3)"]["pspl_recall"])); cmd(f"bmlHeldPsplHi{nm}", two(pr_["[3,inf)"]["pspl_recall"]))
K = load(f"gapped_threshold_{REC}_seasons.json")
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
    cmd("bmlSidecarHeldAp", three(arm["our_heldout"]["ap"])); cmd("bmlSidecarHeldApClean", three(K["arms"]["clean"]["our_heldout"]["ap"]))
    # the paper's own procedure (one 20% validation slice) and why the full pool is used instead
    gs_ = arm["gulls_at_this_threshold"]
    cmd("bmlSidecarSliceThr", three(arm["threshold_at_target_purity"]))
    cmd("bmlSidecarSliceFa", pct(gs_["fa_1S1L_k"] / gs_["fa_1S1L_n"])); cmd("bmlSidecarSliceRec", two(gs_["recall_1S2L_k"] / gs_["recall_1S2L_n"]))
    if T and need("at_own_calibrated_threshold" in T["models"].get(REC, {}), f"{REC}: no weighted operating point (rerun gulls_summary_tables.py)"):
        oc = T["models"][REC]["at_own_calibrated_threshold"]
        assert abs(oc["threshold"] - pool["full_pool"]["threshold"]) < 1e-12, "summary and calibration disagree on the threshold"
        cmd("bmlSidecarFaW", pct(oc["fa_1S1L"]["weighted"])); cmd("bmlSidecarRecPlanetW", two(oc["recall_1S2L"]["weighted"]))
# round 3 (finite source, random gaps), calibrated the same way on the same held-out pool: the comparison row
K3 = load(f"gapped_threshold_{R3}_seasons.json")
if K3:
    a3 = K3["arms"]["rmdc26_gapped"]; g3 = a3["gulls_at_full_pool_threshold"]
    cmd("bmlFsplCalThr", three(a3["pool"]["full_pool"]["threshold"]))
    cmd("bmlFsplCalFa", pct(g3["fa_1S1L_k"] / g3["fa_1S1L_n"])); cmd("bmlFsplCalRec", two(g3["recall_1S2L_k"] / g3["recall_1S2L_n"]))
    cmd("bmlFsplHeldAp", three(a3["our_heldout"]["ap"])); cmd("bmlFsplHeldApClean", three(K3["arms"]["clean"]["our_heldout"]["ap"]))
O = load("occupancy_sensitivity.json")
if O and need(REC in O["models"] and "k_as_observed" in O["models"][REC]["fa_1S1L"], f"occupancy artifact lacks {REC} or counts (rerun it)"):
    o = O["models"][REC]
    cmd("bmlOccN", f"{o['n']:,}")
    kk = lambda blk, key: blk[f"k_{key}"] / blk["n"]
    cmd("bmlOccFaObs", pct(kk(o["fa_1S1L"], "as_observed"))); cmd("bmlOccFaOne", pct(kk(o["fa_1S1L"], "occupancy_set_to_1")))
    cmd("bmlOccRecObs", two(kk(o["recall_1S2L"], "as_observed"))); cmd("bmlOccRecOne", two(kk(o["recall_1S2L"], "occupancy_set_to_1")))
S = load("schedule_finetune.json")
if S and ALLOW and "n_pool" not in next(iter(S["heldout_eval"].values()))["clean"]:
    MISSING.append("schedule_finetune.json (pre-rerun version)"); S = None
if S:
    H = S["heldout_eval"]
    first = next(iter(H.values()))
    need(bool(first["clean"].get("n_scored_test")), "schedule_finetune.json lacks n_scored_test")
    cmd("bmlGapHeldN", f"{first['clean'].get('n_scored_test') or 0:,}")
    cmd("bmlGapPoolN", f"{first['clean']['n_pool']:,}")
    rows = [("shipped", "shipped"), ("ft_g08e12", "gap augmentation, g08e12 (other pool)"),
            ("rand", "random gaps"), ("rand_norelabel", "random gaps, relabel off"),
            ("sched_norelabel", "first-season mask, relabel off"), ("sched_seasons", "measured seasons, relabel off")]
    lines = []
    for k, label in rows:
        e = H[k]; ms = e["rmdc26_seasons"]; rc = ms["argmax_population"]["recall"]
        lines.append(label + " & " + " & ".join([three(e["clean"]["ap"]), three(ms["ap"]), two(rc["PSPL"]), two(rc["Flat"]), two(rc["NonPSPL"]),
                                                 three(ms["argmax_population"]["macro_f1"])]) + " \\\\")
    TABLES["gulls_gap_table.tex"] = "\n".join(lines) + "\n"
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
    need(all("n_detectable_anomaly_host_visible" in bc[k] for k in bc), "transfer_multiseason.json lacks counts (rerun its reduce)")
    for k, nm in (("RMDC26_1S2L_ML", "Planet"), ("RMDC26_2S2L_ML", "PlanetBin")):
        fd = bc[k]["n_detectable_anomaly"] / bc[k]["n_scorable"]                      # from counts: rounded once
        cmd(f"bmlMsDet{nm}", pct0(fd))
        if "n_detectable_anomaly_host_visible" in bc[k]:
            cmd(f"bmlMsDetHost{nm}", pct0(bc[k]["n_detectable_anomaly_host_visible"] / bc[k]["n_host_visible"]))
        if D:
            cmd(f"bmlMsDetLoss{nm}", pct0(D["relabelling"][k]["frac_NonPSPL"] - fd))
    need(REC in MS["models"], f"transfer_multiseason.json lacks {REC} (re-extract with it)")
    mm = MS["models"][REC] if REC in MS["models"] else MS["models"][R3]; fz = mm["at_threshold"]["frozen"]
    cmd("bmlMsFa", pct(fz["fa_1S1L"]["rate"])); cmd("bmlMsFaHi", pct(fz["fa_1S1L"]["wilson95"][1]))
    cmd("bmlMsRecDetPlanet", two(fz["recall_1S2L_detectable"]["rate"])); cmd("bmlMsRecDetPlanetBin", two(fz["recall_2S2L_detectable"]["rate"]))
    cmd("bmlMsRecGenPlanet", pct0(fz["recall_1S2L_generator"]["k"] / fz["recall_1S2L_generator"]["n"]))
    cmd("bmlMsRecGenPlanetBin", pct0(fz["recall_2S2L_generator"]["k"] / fz["recall_2S2L_generator"]["n"]))
    b5 = mm["recall_at_matched_fa"]["0.052"]
    cmd("bmlMsRecBudgetPlanet", two(b5["recall_1S2L_detectable"])); cmd("bmlMsRecBudgetPlanetBin", two(b5["recall_2S2L_detectable"]))
    cmd("bmlMsBudgetThr", two(b5["threshold"])); cmd("bmlMsBudgetExceed", str(b5["n_1S1L_exceedances"]))
    if need("calibrated_seasons_fullpool" in mm["at_threshold"], "between-season block lacks the recommended threshold"):
        cs = mm["at_threshold"]["calibrated_seasons_fullpool"]
        cmd("bmlMsSidecarFa", pct(cs["fa_1S1L"]["k"] / cs["fa_1S1L"]["n"])); cmd("bmlMsSidecarRecDetPlanet", two(cs["recall_1S2L_detectable"]["k"] / cs["recall_1S2L_detectable"]["n"]))
CG = load("cascade_gulls.json")
if CG and ALLOW and "inhouse_reference" not in CG:
    MISSING.append("cascade_gulls.json (pre-rescan version)"); CG = None
if CG:
    R_ = CG["results"]
    need(f"{REC}|f146|frozen" in R_, f"cascade_gulls.json lacks {REC} (scan it with --ckpt, then --reduce)")
    CM = REC if f"{REC}|f146|frozen" in R_ else R3
    pr = R_[f"{CM}|f146|frozen"]; t_ = pr["timing"]
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
    k_cal = f"{CM}|f146|calibrated_seasons_fullpool"
    if need(k_cal in R_, f"cascade_gulls.json lacks {k_cal}"):
        tc_ = R_[k_cal]["timing"]
        cmd("bmlCgPremCal", pct(tc_["premature_frac"])); cmd("bmlCgDetCal", pct0(tc_["detected_frac"])); cmd("bmlCgLagCal", f"{tc_['median_lag_nonpremature_days']:+.1f}")
        cmd("bmlCgSingleAlertCal", pct(R_[k_cal]["burden"]["RMDC26_1S1L_ML"]["alert_frac_per_season"]))
        cmd("bmlCgPurOneCal", pct0(R_[k_cal]["streaming_purity"]["planetary_prevalence_0.01"]["purity_detectable_anomaly_alerts"]))
    ih = CG["inhouse_reference"]["by_mass_ratio"]
    for st, nm in (("giant", "Giant"), ("neptune", "Neptune")):
        cmd(f"bmlCgIn{nm}N", str(ih[st]["n_eligible"])); cmd(f"bmlCgIn{nm}Det", pct0(ih[st]["detection_fraction"]))
        cmd(f"bmlCgIn{nm}Prem", pct(ih[st]["premature_rate_of_eligible"]))
        cmd(f"bmlCgIn{nm}Lag", f"{ih[st]['median_lag_non_premature_days']:+.1f}")
    cmd("bmlCgNsingle", f"{CG['n_scanned']['RMDC26_1S1L_ML']:,}")
    cmd("bmlCgNbinary", f"{CG['n_scanned']['RMDC26_1S2L_ML'] + CG['n_scanned']['RMDC26_2S2L_ML']:,}")
    cmd("bmlCgDetLo", pct0(t_["detected_ci95"][0])); cmd("bmlCgDetHi", pct0(t_["detected_ci95"][1]))
    t3 = R_[f"{CM}|threeband|frozen"]
    cmd("bmlCgPremThree", pct(t3["timing"]["premature_frac"])); cmd("bmlCgDetThree", pct0(t3["timing"]["detected_frac"]))
    cmd("bmlCgLagThree", f"{t3['timing']['median_lag_nonpremature_days']:+.1f}")
    cmd("bmlCgSingleAlertThree", pct(t3["burden"]["RMDC26_1S1L_ML"]["alert_frac_per_season"]))
# ------------------------------------------------------------------ write everything at once, only now
names = [x.split("}")[0].split("\\")[-1] for x in L]
dup = sorted({n for n in names if names.count(n) > 1})
if dup:
    raise SystemExit(f"FATAL: macros defined twice: {dup}")
if MISSING and not ALLOW:
    raise SystemExit(f"FATAL: {MISSING}")
os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
for name, text in TABLES.items():
    open(os.path.join(HERE, "outputs", name), "w").write(text)
OUT = os.path.join(HERE, "gulls_macros.tex")
open(OUT, "w").write("% AUTO-GENERATED by make_gulls_macros.py -- do not edit.\n" + "\n".join(L) + "\n")
print(f"wrote {OUT} ({len(L)} macros) and {sorted(TABLES)}" + (f"; MISSING (dev only): {MISSING}" if MISSING else ""))

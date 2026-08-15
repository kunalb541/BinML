#!/usr/bin/env python3
"""canonical_numbers.json (+ outputs/figures_stats.json) -> paper_macros.tex.

Every number in paper.tex is a \\bml* macro defined here, so the manuscript can never drift
from the evaluation artifact. Run after make_figures.py:  python make_macros.py
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))

cn = json.load(open(os.path.join(HERE, "canonical_numbers.json")))
fs_path = os.path.join(HERE, "outputs", "figures_stats.json")
fs = json.load(open(fs_path)) if os.path.exists(fs_path) else {}

L = []
def cmd(name, value):
    # Thousands separators must be {,} not ",": several macros are used inside $...$, where a bare
    # comma is math punctuation and typesets with a trailing space ("505, 479" in the abstract).
    L.append(r"\newcommand{\%s}{%s}" % (name, str(value).replace(",", "{,}")))

def pct(x):   return f"{100*x:.1f}"
def two(x):   return f"{x:.2f}"
def three(x): return f"{x:.3f}"

h = cn["headline"]
cmd("bmlNtest", f"{cn['test']['n_events']:,}")
cmd("bmlNpool", f"{cn['test']['n_pool']:,}")
cmd("bmlNval", f"{cn['test']['n_val']:,}")
cmd("bmlCompleteness", three(h["completeness_at_purity"]))
cmd("bmlCompletenessPct", pct(h["completeness_at_purity"]))
cmd("bmlCompletenessLo", three(h["completeness_ci_lo"]))
cmd("bmlCompletenessHi", three(h["completeness_ci_hi"]))
cmd("bmlPurity", three(h["purity_achieved"]))
cmd("bmlPurityTarget", two(h["purity_target"]))
cmd("bmlAP", three(h["average_precision"]))
cmd("bmlMacroF", three(h["macro_f1"]))

for c, v in cn["per_class"].items():
    if c == "note":
        continue
    cmd(f"bml{c}R", three(v["recall"]))
    cmd(f"bml{c}P", three(v["precision"]))
    cmd(f"bml{c}F", three(v["f1"]))

cmd("bmlNonpsplPhysP", three(cn["nonpspl_physical_precision"]))
cmd("bmlNonpsplDemoted", pct(cn["nonpspl_demoted_fraction"]))
cmd("bmlSliceStrongAnom", three(cn["nonpspl_recall_strong_anomaly"]))

sup = cn["per_class_support"]
for c in ["Flat", "PSPL", "NonPSPL", "PeriodicVar", "LongPeriodVar", "Eruptive"]:
    cmd(f"bmlSup{c}", f"{sup[c]:,}")

lab = cn["labelling"]
cmd("bmlRelabelled", f"{lab['overall_relabelled_pct']:.1f}")
cmd("bmlNonpsplKept", f"{lab['nonpspl_kept_pct']:.1f}")
cmd("bmlNonpsplToPspl", f"{lab['nonpspl_to_pspl_pct']:.1f}")
cmd("bmlLpvToFlat", f"{lab['lpv_to_flat_pct']:.1f}")

# CASCADE: read straight from the tracked artifact, never from a hand-copied block. If the
# artifact is missing or its schema changed, FAIL rather than emit stale numbers -- prose/artifact
# drift is exactly how the withdrawn 42%->9% claim survived in the manuscript.
_casc_path = os.path.join(os.path.dirname(HERE), "validation", "cascade_reproduce_result.json")
if not os.path.exists(_casc_path):
    raise SystemExit(f"FATAL: {_casc_path} missing; run validation/cascade_trace.py then "
                     f"validation/cascade_reduce.py")
cas = json.load(open(_casc_path))
for _k in ("n_eligible", "detection_fraction", "premature_rate_of_eligible",
           "premature_ci_of_eligible", "premature_rate_of_detected", "median_lag_detected_days",
           # the two medians must BOTH exist: reporting only the all-detections median as if it
           # were the post-onset delay was an arithmetic error in an earlier revision
           "median_lag_non_premature_days", "n_lag_non_premature", "sensitivity"):
    if _k not in cas:
        raise SystemExit(f"FATAL: cascade artifact lacks '{_k}'; regenerate it "
                         f"(schema changed -- do not hand-edit the JSON)")
cmd("bmlCascN", str(cas["n_eligible"]))
cmd("bmlCascDetFrac", pct(cas["detection_fraction"]))
cmd("bmlCascPrematurePct", pct(cas["premature_rate_of_eligible"]))
cmd("bmlCascPrematureLo", pct(cas["premature_ci_of_eligible"][0]))
cmd("bmlCascPrematureHi", pct(cas["premature_ci_of_eligible"][1]))
cmd("bmlCascPrematureOfDet", pct(cas["premature_rate_of_detected"]))
# NAME THE POPULATION IN THE MACRO NAME. \bmlCascMedianLagAll is over ALL detections and mixes in
# the negative lags of premature alerts; \bmlCascMedianLagNonPrem is the post-onset delay.
cmd("bmlCascMedianLagAll", f'{cas["median_lag_detected_days"]:+.1f}')
cmd("bmlCascMedianLagNonPrem", f'{cas["median_lag_non_premature_days"]:+.1f}')
cmd("bmlCascNnonPrem", str(cas["n_lag_non_premature"]))
cmd("bmlCascNdet", str(cas["n_detected"]))
cmd("bmlCascCensored", str(cas["n_censored"]))

_sen = cas["sensitivity"]
# ONSET DEFINITION bracket. The premature rate moves by a factor of ~20 across these, which is far
# more than any modelling choice moves it, so the paper reports the range rather than a number.
_od = _sen["onset_definition"]
cmd("bmlCascPrematurePersistPct", pct(_od["persistent_detectable"]["premature_rate_of_eligible"]))
cmd("bmlCascPrematurePersistLo",
    pct(_od["persistent_detectable"]["premature_ci_of_eligible"][0]))
cmd("bmlCascPrematurePersistHi",
    pct(_od["persistent_detectable"]["premature_ci_of_eligible"][1]))
cmd("bmlCascMedianLagPersist",
    f'{_od["persistent_detectable"]["median_lag_non_premature_days"]:+.1f}')
cmd("bmlCascOnsetShift", f'{_od["median_coarse_minus_first_days"]:.1f}')
cmd("bmlCascNonMonotone", str(_od["n_non_monotone"]))
cmd("bmlCascCoarsePrematurePct", pct(_sen["onset_coarse_7p2d"]["premature_rate_of_eligible"]))
cmd("bmlCascCoarseMedianLagAll",
    f'{_sen["onset_coarse_7p2d"]["median_lag_detected_days"]:+.1f}')
cmd("bmlCascCoarseMedianLagNonPrem",
    f'{_sen["onset_coarse_7p2d"]["median_lag_non_premature_days"]:+.1f}')
_g = _sen["evaluation_grid_days"]
cmd("bmlCascGridHalfPct", pct(_g["0.5"]["premature_rate_of_eligible"]))
cmd("bmlCascGridOnePct", pct(_g["1.0"]["premature_rate_of_eligible"]))
cmd("bmlCascGridTwoPct", pct(_g["2.0"]["premature_rate_of_eligible"]))
cmd("bmlCascGridOneDet", pct(_g["1.0"]["detection_fraction"]))
_p2 = _sen["persistence_consecutive_crossings"]["2"]
cmd("bmlCascPersistTwoPct", pct(_p2["premature_rate_of_eligible"]))
cmd("bmlCascPersistTwoDet", pct(_p2["detection_fraction"]))
_mb = _sen["bands"]["all_three_bands"]
cmd("bmlCascMultibandPct", pct(_mb["premature_rate_of_eligible"]))
cmd("bmlCascMultibandDet", pct(_mb["detection_fraction"]))
cmd("bmlCascMultibandMedianLag", f'{_mb["median_lag_non_premature_days"]:+.1f}')
cmd("bmlCascArgmaxPct", pct(_sen["argmax_rule"]["premature_rate_of_eligible"]))

# The alert-policy table is GENERATED from the same artifact rather than typed into paper.tex.
# Fifteen numbers in a hand-written tabular is fifteen chances to update the analysis and forget
# the table.
_rows = [
    ("Onset: first detectable\\tablenotemark{a}", _od["first_detectable"]),
    ("Onset: persistently detectable", _od["persistent_detectable"]),
    ("Onset: generator grid (7.2\\,d)", _od["coarse_7p2d_generator_grid"]),
    ("Grid: every 1\\,d", _g["1.0"]),
    ("Grid: every 2\\,d", _g["2.0"]),
    ("Rule: 2 consecutive crossings", _sen["persistence_consecutive_crossings"]["2"]),
    ("Rule: 3 consecutive crossings", _sen["persistence_consecutive_crossings"]["3"]),
    ("Rule: \\texttt{NonPSPL} is argmax", _sen["argmax_rule"]),
    ("Bands: all three revealed", _sen["bands"]["all_three_bands"]),
]
# A flat table with self-describing row labels. \cutinhead subheadings are the idiomatic AASTeX
# construct but cannot open a \startdata block, and \multicolumn raises "Misplaced \omit" inside
# deluxetable; neither is worth a fragile workaround for four labels.
os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
_out = []
for _i, (_label, _v) in enumerate(_rows):
    _ci = _v["premature_ci_of_eligible"]
    _md = _v["median_lag_non_premature_days"]
    _row = r"%s & %.1f & $[%.1f,\,%.1f]$ & %.1f & %s" % (
        _label, 100 * _v["premature_rate_of_eligible"], 100 * _ci[0], 100 * _ci[1],
        100 * _v["detection_fraction"],
        ("$%+.1f$" % _md) if _md is not None else r"\nodata")
    _out.append(_row if _i == len(_rows) - 1 else _row + r" \\")
with open(os.path.join(HERE, "outputs", "cascade_policy_table.tex"), "w") as _f:
    _f.write("% AUTO-GENERATED by make_macros.py from validation/cascade_reproduce_result.json\n")
    _f.write("\n".join(_out) + "\n")

# PREVALENCE: thresholds chosen on the reserved validation rows, reported on final test. An
# earlier revision selected them on the final-test PR curve and reported on the same rows.
_prev_path = os.path.join(os.path.dirname(HERE), "validation", "prevalence_result.json")
if not os.path.exists(_prev_path):
    raise SystemExit(f"FATAL: {_prev_path} missing; run validation/prevalence_thresholds.py")
prv = json.load(open(_prev_path))
for _k, _nm in (("synthetic", "Synth"), ("one_pct", "One"), ("tenth_pct", "Tenth")):
    _s = prv["scenarios"][_k]
    cmd(f"bmlPrevComp{_nm}", three(_s["completeness_test"]))
    cmd(f"bmlPrevPur{_nm}", three(_s["purity_test"]))
    cmd(f"bmlPrevThr{_nm}", three(_s["threshold"]))
cmd("bmlPrevPurTarget", two(prv["purity_target"]))

s = cn["stress"]
cmd("bmlStressN", f"{s['n_events_millions']:.1f}")
cmd("bmlStressNatN", f"{s['natural_events_millions']:.1f}")
cmd("bmlStressMacroF", three(s["natural_macro_f1"]))
cmd("bmlSeedTrain", str(s["seed_base_train"]))
cmd("bmlSeedStress", str(s["seed_base_stress"]))

sr = cn["stress_regimes"]
cmd("bmlStressNatNp", three(sr["natural_np_recall"]))
cmd("bmlStressPlanetNp", three(sr["planetary_np_recall"]))
cmd("bmlStressPlanetPrec", three(sr["planetary_np_prec"]))
cmd("bmlStressWidesep", three(sr["widesep_np_recall"]))
cmd("bmlStressLongp", three(sr["longp_per_recall"]))
cmd("bmlStressShortte", three(sr["shortte_pspl_recall"]))
cmd("bmlStressFaintPspl", three(sr["faint_pspl_recall"]))
cmd("bmlStressFaintPrec", three(sr["faint_np_prec"]))

gm = cn["gap_matched"]
cmd("bmlGapN", str(gm["n_events"]))
cmd("bmlGapVisitsA", f"{gm['visits_per_day_a']:.0f}")
cmd("bmlGapUniformA", three(gm["uniform_a"]))
cmd("bmlGapUniformALo", three(gm["uniform_a_ci"][0]))
cmd("bmlGapUniformAHi", three(gm["uniform_a_ci"][1]))
cmd("bmlGapNightlyA", three(gm["nightly_a"]))
cmd("bmlGapNightlyAHi", three(gm["nightly_a_ci"][1]))
cmd("bmlGapVisitsB", f"{gm['visits_per_day_b']:.0f}")
cmd("bmlGapUniformB", three(gm["uniform_b"]))
cmd("bmlGapVisitsC", f"{gm['visits_per_day_c']:.0f}")
cmd("bmlGapUniformC", three(gm["uniform_c"]))

mr = cn["mass_ratio_regimes"]
for key, name in (("stellar", "Stellar"), ("giant", "Giant"),
                  ("neptune", "Neptune"), ("lowmass", "Lowmass")):
    cmd(f"bmlQ{name}Comp", three(mr[f"{key}_comp"]))
    cmd(f"bmlQ{name}N", f"{mr[f'{key}_n']:,}")

ft = cn["massregime_finetune"]
cmd("bmlFtEvalN", f"{ft['eval_n']:,}")
for key, name in (("stellar", "Stellar"), ("giant", "Giant"),
                  ("neptune", "Neptune"), ("lowmass", "Lowmass")):
    cmd(f"bmlFt{name}Before", three(ft[f"{key}_before"]))
    cmd(f"bmlFt{name}After", three(ft[f"{key}_after"]))
    cmd(f"bmlFt{name}N", str(ft[f"{key}_n"]))
cmd("bmlFtWorstRegress", three(abs(min(ft[k] for k in ft if k.startswith("f1_") and k.endswith("_delta")))))

hr = cn["hardregime_finetune"]
for k, name in (("longp_periodic", "LongpPer"), ("widesep_nonpspl", "WidesepNp"),
                ("shortte_pspl", "Shortte"), ("faint_pspl", "FaintPspl")):
    cmd(f"bmlHr{name}Before", three(hr[f"{k}_before"]))
    cmd(f"bmlHr{name}After", three(hr[f"{k}_after"]))
cmd("bmlHrFaintLpvPrecBefore", three(hr["faint_lpv_prec_before"]))
cmd("bmlHrFaintLpvPrecAfter", three(hr["faint_lpv_prec_after"]))
cmd("bmlHrFaintLpvRecBefore", three(hr["faint_lpv_recall_before"]))
cmd("bmlHrFaintLpvRecAfter", three(hr["faint_lpv_recall_after"]))
cmd("bmlHrWidesepN", str(hr["widesep_n"]))
cmd("bmlHrNatNpPrecBefore", three(hr["natural_nonpspl_prec_before"]))
cmd("bmlHrNatNpPrecAfter", three(hr["natural_nonpspl_prec_after"]))

# ABLATIONS: read the artifacts directly, like the cascade block. Fourteen of these numbers used
# to be hand-copied into canonical_numbers.json, which is one transcription step away from the
# prose/artifact drift that produced the withdrawn 42%->9% claim.
#
# The cascade arm comes from the original four-arm run; the labelling arms come from the rerun that
# holds the loss weights fixed across arms (the first run confounded labels with weighting, because
# pipeline.train derives class weights from whichever labels it is handed).
def _load(fname, required):
    path = os.path.join(os.path.dirname(HERE), "validation", fname)
    if not os.path.exists(path):
        raise SystemExit(f"FATAL: {path} missing; run the experiment that produces it")
    obj = json.load(open(path))
    cur = obj
    for key in required:                      # required is a path into the nested schema
        if key not in cur:
            raise SystemExit(f"FATAL: {fname} lacks '{'/'.join(required)}'; regenerate it "
                             f"(schema changed -- do not hand-edit the JSON)")
        cur = cur[key]
    return obj

abl = _load("ablations_result.json", ("cascade_on", "realtime", "premature_rate_thr"))
cmd("bmlAbCascN", str(abl["cascade_on"]["realtime"]["n_eligible"]))
for tag, nm in (("cascade_on", "CascOn"), ("cascade_off", "CascOff")):
    rt, st = abl[tag]["realtime"], abl[tag]["static"]
    cmd(f"bmlAb{nm}PremThr", three(rt["premature_rate_thr"]))
    cmd(f"bmlAb{nm}PremArgmax", three(rt["premature_rate_argmax"]))
    cmd(f"bmlAb{nm}Det", three(rt["detection_fraction_thr"]))
    cmd(f"bmlAb{nm}MacroFOne", three(st["macro_f1"]))

lab2 = _load("labelling_ablation_result.json", ("arms", "labels_observational"))
for tag, nm in (("labels_observational", "LabelObs"), ("labels_generator", "LabelGen"),
                ("labels_generator_ownweights", "LabelGenOwnW")):
    a = lab2["arms"][tag]
    cmd(f"bmlAb{nm}MacroFOne", three(a["vs_observational"]["macro_f1"]))
    cmd(f"bmlAb{nm}MacroFOneGen", three(a["vs_generator"]["macro_f1"]))
    for cls, cnm in (("PSPL", "Pspl"), ("NonPSPL", "Nonpspl")):
        cmd(f"bmlAb{nm}{cnm}FOne", three(a["vs_observational"]["per_class_f1"][cls]))
    cmd(f"bmlAb{nm}Comp", three(a["anomaly_completeness_at_thr"]))
    cmd(f"bmlAb{nm}Pur", three(a["anomaly_purity_at_thr"]))
_gap = lab2["macro_f1_gap_vs_observational"]
cmd("bmlAbLabelGap", three(_gap["point"]))
cmd("bmlAbLabelGapLo", three(_gap["ci95"][0]))
cmd("bmlAbLabelGapHi", three(_gap["ci95"][1]))
cmd("bmlAbLabelNeval", f'{lab2["n_eval"]:,}')
cmd("bmlAbLabelEpochs", str(lab2["config"]["epochs"]))
# MATCHED-DETECTION cascade comparison: the experiment the previous revision named as the paper's
# clearest outstanding weakness. Read fail-closed like the others.
mtc = _load("cascade_matched_result.json", ("matched", "detection_0.60"))
cmd("bmlMatchN", str(mtc["n_eligible"]))
for _t, _nm in (("0.60", "Sixty"), ("0.80", "Eighty"), ("0.90", "Ninety")):
    _r = mtc["matched"][f"detection_{_t}"]
    cmd(f"bmlMatchOn{_nm}", pct(_r["cascade_on"]["premature_rate_of_eligible"]))
    cmd(f"bmlMatchOff{_nm}", pct(_r["cascade_off"]["premature_rate_of_eligible"]))
    cmd(f"bmlMatchP{_nm}", f'{_r["mcnemar"]["p_two_sided_conditional"]:.4g}')
    cmd(f"bmlMatchP{_nm}Adj", f'{_r["mcnemar"]["p_bonferroni_conditional"]:.4g}')
    cmd(f"bmlMatchOn{_nm}Hi", pct(_r["cascade_on"]["premature_ci_of_eligible"][1]))



cx = cn["cadence_experiment"]
# The paper bounds the per-class F1 difference between cadence arms. Compute the bound rather than
# asserting it in prose: a bound that is typed by hand is a bound nobody rechecks.
_cadraw = os.path.join(os.path.dirname(HERE), "validation", "cadence_result.json")
if os.path.exists(_cadraw):
    _cd = json.load(open(_cadraw))
    _f1 = [_a["per_class_f1"] for _a in _cd]
    cmd("bmlCadMaxFOneDelta",
        f'{max(abs(_f1[0][k] - _f1[1][k]) for k in _f1[0]):.3f}')
cmd("bmlCadN", f"{cx['n_events']:,}")
# Per-arm surviving-event counts. canonical_numbers stores only the 15-min arm's value, and the
# paper previously presented it as "matched ... events", which it is not: the arms are matched by
# construction (same shards, seed, epochs) and their surviving counts differ by a few dozen events.
_cadraw2 = os.path.join(os.path.dirname(HERE), "validation", "cadence_result.json")
if os.path.exists(_cadraw2):
    _cd2 = {int(_a["cadence_min"]): _a for _a in json.load(open(_cadraw2))}
    cmd("bmlCadNfifteen", f"{_cd2[15]['n_events']:,}")
    cmd("bmlCadNtwelve", f"{_cd2[12]['n_events']:,}")
cmd("bmlCadEpochs", str(cx["epochs"]))
for k, name in (("completeness", "Completeness"), ("purity", "Purity"),
                ("ap", "Ap"), ("nonpspl_f1", "Nonpspl")):   # no digits: TeX names are letters only
    cmd(f"bmlCadFifteen{name}", three(cx[f"c15_{k}"]))
    cmd(f"bmlCadTwelve{name}", three(cx[f"c12_{k}"]))

bl = cn["baselines"]
cmd("bmlBaseBinml", three(bl["ap_binml"]))
cmd("bmlBaseGbt", three(bl["ap_gbt"]))
cmd("bmlBaseLogistic", three(bl["ap_logistic"]))
cmd("bmlBasePspl", three(bl["ap_fitted_pspl"]))
cmd("bmlBaseNtest", f"{bl['n_test']:,}")

# THROUGHPUT: read from the reproducible benchmark, never hand-entered. The previous value
# (1,224 events/s) had no retained script or hardware record and could not be reproduced.
_inf_path = os.path.join(os.path.dirname(HERE), "validation", "inference_benchmark_result.json")
if not os.path.exists(_inf_path):
    raise SystemExit(f"FATAL: {_inf_path} missing; run validation/inference_benchmark.py")
inf = json.load(open(_inf_path))
cmd("bmlInferEps", f"{inf['events_per_sec']:,}")
cmd("bmlInferMs", two(inf["ms_per_event"]))
cmd("bmlInferCores", str(inf["environment"]["logical_cores"]))
cmd("bmlInferCpu", inf["environment"]["cpu"])
cmd("bmlInferThreads", str(inf["environment"]["torch_threads"]))

for _k, _nm in (("missed_anomaly_rate", "MissedRate"), ("lpv_to_pspl_rate", "LPVtoPSPL")):
    if _k not in fs:
        raise SystemExit(f"FATAL: figures_stats lacks '{_k}'; rerun make_figures.py")
    cmd("bml" + _nm, three(fs[_k]))

ia = cn["infra"]
cmd("bmlSimEvents", f"{ia['sim_events_millions']:.0f}")
cmd("bmlSimInstances", str(ia["sim_instances"]))
cmd("bmlSimVcpus", str(ia["sim_vcpus_each"]))
cmd("bmlSimEps", str(ia["sim_events_per_sec"]))
cmd("bmlSimHours", str(ia["sim_hours"]))
cmd("bmlSimRegion", ia["sim_region"])
cmd("bmlTrainDevice", ia["train_device"])
cmd("bmlTrainEps", f"{ia['train_events_per_sec']:,}")
cmd("bmlTrainBatch", str(ia["train_batch"]))
cmd("bmlTrainHours", str(ia["train_hours"]))
cmd("bmlTrainEpochs", str(ia["train_epochs_effective"]))

sl = cn["slices"]
cmd("bmlSliceDeepPlanet", three(sl["deep_planetary_q_lt_1e-3"]["recall"]))
cmd("bmlSlicePlanet", three(sl["planetary_q_lt_1e-2"]["recall"]))
cmd("bmlSliceSuzuki", three(sl["near_suzuki_break"]["recall"]))
cmd("bmlSliceFaint", three(sl["faint_mbase_gt_22p5"]["recall"]))
cmd("bmlSliceNoBlue", three(sl["no_blue_band"]["recall"]))

m = cn["model"]
cmd("bmlNparams", f"{m['n_params']:,}")
cmd("bmlNtokens", str(m["n_tokens"]))
cmd("bmlDmodel", str(m["d_model"]))
cmd("bmlNlayers", str(m["n_layers"]))
cmd("bmlNheads", str(m["n_heads"]))
cmd("bmlBinsF", str(m["bins_f146"]))
cmd("bmlEpochsF", f"{m['epochs_f146']:,}")
cmd("bmlCadence", str(m["cadence_f146_min"]))
cmd("bmlWindow", str(m["window_days"]))

# derived-in-make_figures numbers
if "ap_oracle_dchi2" in fs:
    cmd("bmlAPoracle", three(fs["ap_oracle_dchi2"]))
if "ap_network" in fs:
    cmd("bmlAPnet", three(fs["ap_network"]))
if "prior_pi0" in fs:
    cmd("bmlPriorPiZero", pct(fs["prior_pi0"]))
    cmd("bmlPurityOnePct", three(fs["purity_pi_1pct"]))
    cmd("bmlPurityTenthPct", three(fs["purity_pi_p1pct"]))
for _k, _nm in (("conf_nonpspl_to_pspl_pct", "ConfNpToPspl"),
                ("conf_pspl_to_nonpspl_pct", "ConfPsplToNp"),
                ("gen_nonpspl_demoted_pct", "GenNpDemoted"),
                ("no_f087_pct", "NoFilterBlue")):
    # Previously bare literals in paper.tex. The F087-loss one disagreed with the artifact by a
    # factor of nearly two, which is exactly what hard-coded prose numbers do.
    if _k not in fs:
        raise SystemExit(f"FATAL: figures_stats lacks '{_k}'; rerun make_figures.py")
    cmd("bml" + _nm, f"{fs[_k]:.1f}")

if "eff_cond_recall_median" in fs:
    cmd("bmlEffMedian", three(fs["eff_cond_recall_median"]))
if "nonpspl_ece_weighted" in fs:
    cmd("bmlECE", f"{fs['nonpspl_ece_weighted']:.3f}")
    cmd("bmlBrier", f"{fs['nonpspl_brier_weighted']:.3f}")

header = "% AUTO-GENERATED by make_macros.py -- do not edit. Source: canonical_numbers.json\n"
with open(os.path.join(HERE, "paper_macros.tex"), "w") as f:
    f.write(header + "\n".join(L) + "\n")
print(f"wrote paper_macros.tex ({len(L)} macros)")

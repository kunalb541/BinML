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
    L.append(r"\newcommand{\%s}{%s}" % (name, value))

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
    raise SystemExit(f"FATAL: {_casc_path} missing; run validation/cascade_reproduce.py")
cas = json.load(open(_casc_path))
for _k in ("n_eligible", "detection_fraction", "premature_rate_of_eligible",
           "premature_ci_of_eligible", "premature_rate_of_detected", "median_lag_detected_days"):
    if _k not in cas:
        raise SystemExit(f"FATAL: cascade artifact lacks '{_k}'; regenerate it "
                         f"(schema changed -- do not hand-edit the JSON)")
cmd("bmlCascN", str(cas["n_eligible"]))
cmd("bmlCascDetFrac", pct(cas["detection_fraction"]))
cmd("bmlCascPrematurePct", pct(cas["premature_rate_of_eligible"]))
cmd("bmlCascPrematureLo", three(cas["premature_ci_of_eligible"][0]))
cmd("bmlCascPrematureHi", three(cas["premature_ci_of_eligible"][1]))
cmd("bmlCascPrematureOfDet", pct(cas["premature_rate_of_detected"]))
cmd("bmlCascMedianLag", f'{cas["median_lag_detected_days"]:.1f}')
cmd("bmlCascCensored", str(cas["n_censored"]))

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

po = cn["prevalence_operating"]
cmd("bmlPrevCompSynth", three(po["comp_at_synthetic"]))
cmd("bmlPrevCompOne", three(po["comp_at_one_pct"]))
cmd("bmlPrevCompTenth", three(po["comp_at_tenth_pct"]))

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

cx = cn["cadence_experiment"]
cmd("bmlCadN", f"{cx['n_events']:,}")
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

inf = cn["inference"]
cmd("bmlInferEps", f"{inf['events_per_sec']:,}")
cmd("bmlInferMs", two(inf["ms_per_event"]))
cmd("bmlInferCores", str(inf["cpu_cores"]))
cmd("bmlDetLag", str(inf["median_detection_lag_days"]))

fp = cn["false_positive"]
cmd("bmlMissedRate", three(fp["missed_planet_rate"]))
cmd("bmlLPVtoPSPL", three(fp["long_period_var_to_pspl"]))

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
if "eff_cond_recall_median" in fs:
    cmd("bmlEffMedian", three(fs["eff_cond_recall_median"]))
if "ece_weighted" in fs:
    cmd("bmlECE", f"{fs['ece_weighted']:.3f}")
    cmd("bmlBrier", f"{fs['brier_weighted']:.3f}")

header = "% AUTO-GENERATED by make_macros.py -- do not edit. Source: canonical_numbers.json\n"
with open(os.path.join(HERE, "paper_macros.tex"), "w") as f:
    f.write(header + "\n".join(L) + "\n")
print(f"wrote paper_macros.tex ({len(L)} macros)")

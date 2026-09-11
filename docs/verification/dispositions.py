"""Dispositions for the 2026-09-11 triple check -> docs/VERIFICATION_2026-09-11.md (the finding table).

2026-09-11_findings.json holds the raw output of the verification workflow (8 independent verifiers,
2 adversarial checkers; 182 findings, none refuted). Each finding is mapped to one issue below; every
issue records what was done. Regenerate the table with:  python docs/verification/dispositions.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ISSUES = {
 "G01": ("RMDC26 event count quoted as 385,004 (an event id)", "Fixed: 377,999 events (299,466 / 35,647 / 42,886), from rmdc26_dataset_facts.json."),
 "G02": ("'argmax of most flagged single lenses is PeriodicVar' (impossible)", "Fixed: flagged events are NonPSPL by construction; argmax split over all single lenses 53% NonPSPL / 44% PeriodicVar / 2.5% PSPL."),
 "G03": ("shipped model under the schedule: 'anomaly and variable classes unaffected'; two experiments quoted interchangeably", "Fixed: Eruptive, LongPeriodVar and NonPSPL degradations and the NonPSPL precision collapse reported; one experiment (schedule_finetune.json) per statement, sources named."),
 "G04": ("schedule described as seven pauses at fixed season phases; 'exact schedule' mask matches one season in six", "Fixed: rmdc26_schedule.json (six measured pause patterns); training --gap-schedule rmdc26_seasons; calibration under measured seasons; experiment 2 rerun with single-factor arms; colour bins blanked only where empty; 'planned exposure and calibration' removed."),
 "G05": ("double rounding (11.7%, 4.9%, 0.312 ...)", "Fixed: exact counts stored; rates rounded once (11.6%, 4.8%, 0.313); achieved false-alarm rate reported at each budget."),
 "G06": ("gap table: one cell from another arm; 'gap-aware' meaning three checkpoints; macro-F1 and clean cost from different checkpoints", "Fixed: table regenerated from schedule_finetune.json with named arms; each sentence quotes one checkpoint on one held-out set."),
 "G07": ("gap table N = pool size, not scored rows", "Fixed: 24,011 scored of a 30,013-event pool (20% chooses the operating point); both stored."),
 "G08": ("'gaps <= 2 h are harmless'", "Fixed: 0.5 h costs nothing; one 1-2 h gap moves 7 of 100 single lenses to NonPSPL."),
 "G09": ("2S2L q 'identical to 1S2L (1.25e-4)'; binary source in '56%'", "Fixed: similar, not identical (1.4e-4 vs 1.2e-4, KS p ~2e-18); binary source flagged in 55% of scored 2S2L; a NonPSPL call on 2S2L may respond to the binary source."),
 "G10": ("selection accounting (eligible defined with the season cut; 25.5% 'between seasons' includes out-of-mission peaks)", "Fixed: eligible = amplitude + t_E cuts; split 56.4% scored / 22.0% between seasons / 3.6% before or after the survey / 17.9% low-cadence / 0.08% no baseline; gaps 109-120 d."),
 "G11": ("numbers with no committed artifact", "Fixed: derived quantities written by the reducers (relabelling decomposition and flag rates, relabel exposure, dataset facts, occupancy, per-event score table)."),
 "G12": ("'no anomaly a survey could claim' / 'recall ceiling 0.56'", "Fixed: 'no anomaly our policy would claim within one season' (policy-, floor- and selection-dependent); 0.56 is what a classifier following the policy exactly would score, not a bound; both label sets reported."),
 "G13": ("'the schedule hides about half of the planets'", "Fixed: rerun on between-season events only, with a multi-start refit; the unobserved gap costs 12 / 20 points (1S2L / 2S2L) of claimable anomalies relative to in-season (44% / 34% vs 56% / 54%)."),
 "G14": ("finite-source gain confounded with 12 extra epochs and a different pool; round-2 penalty understated", "Confound confirmed and separated: the point-source control (pspl5s_ctrl, same pool shape, recipe and warm start) gives FA 6.4% and recall 0.361 at 5.2% (g08e12 11.6% / 0.295; fspl5s 4.8% / 0.390), so the extra training explains most of the gain and the physics the rest, concentrated in the two largest rho/|u0| bins (0.41 / 0.51 -> 0.28 / 0.29); only the physics restores large-source single-lens recall on our own held-out (0.22 / 0.16 -> 0.66 / 0.84). Round-2 penalty quantified (FA 8.7% vs 4.8%)."),
 "G15": ("precision-vs-floor 'peak' (true by construction)", "Fixed: argument removed; precision labelled sample-mix and also given at the scored-set mix."),
 "G16": ("'median q a factor of several below our prior'", "Fixed: RMDC26 anomalies are all planetary (median q ~1.3e-4); 82% of our NonPSPL evaluation events are stellar-mass-ratio binaries (median q 0.12); ratios from rmdc26_dataset_facts.json."),
 "G17": ("table bin labels; colour gain 'for every checkpoint'", "Fixed: lowest rho/|u0| bin added; colour gain given per checkpoint (0.042-0.068)."),
 "G18": ("noise arm 'worse on every measure'; pools mixed in its calibrated comparison", "Fixed: worse at every matched budget and in false alarms, higher frozen-threshold recall; single seed; like-for-like calibration."),
 "G19": ("floor sweep vs relabel artifact disagree at 0.02 (rounding)", "Fixed: full-precision truth cache; the sweep asserts agreement at the adopted floor."),
 "G20": ("sub-day false alarms unweighted only; sample described as 'first dense ids'", "Fixed: 0.7% unweighted, 3.1% rate-weighted; sample is a random contiguous id window."),
 "G21": ("'not one training event has an empty mid-season bin'", "Fixed: essentially none (6 of 89,919)."),
 "G22": ("stale header/status lines, inconsistent signed lags, housekeeping", "Fixed."),
 "G23": ("example notebooks gitignored (CI test fails on a clean checkout)", "Fixed: force-added."),
 "G24": ("ft_g08e12.pt gitignored; GULLS artifacts not manifest-hashed", "ft_g08e12.pt committed. Manifest hashing of the GULLS artifacts is part of the macro integration (open)."),
 "G25": ("binml.gulls combiner differs from the measured one; notebook demo events not between seasons", "Fixed: modes peak/adjacent/all with p_nonpspl_max; low-cadence skipped; notebook uses a real between-season event (315360)."),
 "G26": ("recall and false alarms quoted at different operating points", "Fixed: every rate carries its threshold; the recommended operating point is re-derived under measured seasons (full-pool threshold with slice spread)."),
 "G27": ("Reproduce commands reproduce a 600-per-class sample", "Fixed: full-population commands; artifacts record command and code."),
 "G28": ("assorted ledger / section-1 numbers (Wilson half-widths, |u0|<0.1 range, floor sequence, 0.77 vs 0.76, flip fraction, comments)", "Fixed in paper/REVISION.md and code comments."),
 "G29": ("relabel rate '20% of all binaries on every presentation'", "Fixed: 20.1% of NonPSPL-labelled pool events are eligible (7.4% of generated binaries); it fires on about half of their gapped presentations; stored in schedule_finetune.json."),
 "G30": ("streaming-purity prevalence computed on the balanced scan sample", "Fixed: RMDC26 scored-set rate-weighted planetary fraction (22.07%) with weighted alert rates; unrounded."),
 "G31": ("cascade comparison mixes models; untested causal claim; scan details differ from in-house", "Fixed: in-house numbers named as the shipped model on 80% stellar binaries, planetary strata quoted; causal wording withdrawn. Rescanned with >= 10 F146 points per cut and unrounded P (1,667 eligible binaries): premature 2.2% (1.6-3.0%), lag +6.5 d, detected 57% at the frozen threshold, against 1.4-4.2% premature, lag 4.5 d and 73-89% detected in the in-house planetary strata; single-lens alert burden 5.7% of events per season."),
 "G32": ("quickstart demo never crosses the threshold", "Fixed: planetary example (seed 241) crosses at day 49.5 after a 43.5-day onset."),
 "G33": ("rho <= 5 prior 'to cover GULLS'", "Fixed: scored single lenses have rho p99 0.29 (43 above 1); the high rho/|u0| bins come from tiny |u0|."),
 "G34": ("out-of-support inputs not quantified", "Fixed: out-of-support fractions per class in rmdc26_dataset_facts.json; residual false alarms 11.8% at m_base < 20 vs 4.1% inside the prior."),
 "G35": ("'12.1-min cadence indistinguishable after preprocessing'", "Fixed: partial occupancy disclosed and measured on real inputs (occupancy_sensitivity.json); raw-pooling sensitivity restored."),
 "G36": ("amplitude cut described as neutral; baseline 'as a survey pipeline would have'", "Fixed: host-lens peak amplitude, removing most wide-orbit planets (s > 2: 18% / 6% pass); baseline uses catalogue t0 and t_E."),
 "G37": ("binml.gulls docstring and density definition", "Fixed."),
 "G38": ("wing-season refit seeded at an out-of-window t0", "Fixed: multi-start refit when the peak is outside the window; host-visible subset reported; rerun."),
 "G39": ("'amplitude-limited' recall", "Fixed: significance-limited; recall by delta-chi2 stored."),
 "G40": ("sub-floor flag rate compared with all single lenses", "Added single-lens control: half of single lenses with a significant sub-floor static-refit misfit (mostly parallax) are flagged, like sub-floor planets."),
 "G41": ("single lenses labelled NonPSPL by the relabelling (training rule never does)", "Fixed: single lenses follow the training rule; the point-source-refit result is a separate diagnostic with n and interval."),
 "G42": ("truth sample included 600 events from an aborted run", "Fixed: re-extracted with the documented selection (1,600 / 2,500 / 2,500)."),
 "G43": ("fixed delta-chi2 cuts on ~11% more epochs", "Disclosed; rescaled-cut sensitivity stored (< 1 point)."),
 "G44": ("RMDC26 described as a pure test set", "Disclosed: it guided the diagnosis, the single-lens rho prior and the choice among seven fine-tunes; numbers for the chosen checkpoint are optimistic."),
 "G45": ("'g08e12 cannot reach the purity target at any threshold'", "Fixed: no usable operating point (0.90 purity only on the top few events); calibration-pool prevalence stored (6.0% vs 5.6% on the paper test set)."),
 "G46": ("fine-onset scan off by one at both ends", "Fixed: full-grid first-detectable scan; boundary tests."),
 "G47": ("onset default changed released behaviour", "Fixed: default back to the legacy 7.2-d grid; opt-in --onset-resolution-days; settings recorded in shard attributes."),
 "G48": ("caustic-in-gap relabel is not truth-based even with an exact onset", "Audit finding 9 re-opened; relabel off in the new schedule arm."),
 "G49": ("recorded commits wrong; resumable stamps ignore settings", "Fixed going forward (generation-time git describe --dirty, settings in stamps); validation/gulls/PROVENANCE.md records the past."),
 "G50": ("ESPLMag2 hand-off steps of 4-8 mmag", "Fixed: ESPLMag (matches disc integration); legacy flag. The combined arm (smooth function + measured-season pauses + relabel off) matches round 3 at every matched budget (1S2L recall within 0.008); the smooth-function-only control (fspl5s_espl_g08) is indistinguishable from round 3 (FA 4.83% vs 4.85%, recall within 0.001 at every budget, same recalibrated operating point), so the steps did not shape the result."),
 "G51": ("truncation amplitude uses point-source magnification for finite sources", "Fixed for rho > 0.01 (released pools unaffected)."),
 "G52": ("np.trapezoid needs NumPy 2", "Fixed: fallback."),
 "G53": ("transfer status string wrong for fine-tunes", "Fixed."),
 "G54": ("'exact schedule matches random gaps'; two factors changed at once", "Rerun with one pool and one factor: measured seasons beat random gaps (FA 10.0% vs 12.8%; 1S2L recall 0.352 vs 0.279 at 5.2%; ahead in 6 of 6 seasons; held-out AP under the measured pauses 0.921 vs 0.911). Combined with finite-source single lenses it adds nothing further on RMDC26 at matched budgets."),
 "G55": ("onset-fix twin described as 'within a point or two'", "Fixed: finite-source bins 4 points worse; held-out labels identical."),
 "G56": ("colour ablation read as support for the three-band design", "Fixed: the colour gain comes from binaries without a detectable F146 anomaly; mechanism unidentified."),
 "G57": ("'the remaining transfer gap was a missing piece of physics'", "Fixed: 'a large part of the residual false alarms'; the largest-source bins remain at ~0.28."),
 "G58": ("'nearly doubles' against the shipped baseline at a post-hoc budget", "Fixed: like-for-like 0.295 -> 0.390; the budget's origin stated."),
 "G59": ("missing caveats (approximate relabel, static refit, uniform disc, single seed)", "Added."),
 "G60": ("macro names with digits; undefined \\bmlThreshold", "Fixed in the macro list."),
 "G61": ("REVISION section 1 'Where it goes' stale", "Fixed."),
}
MAP = {
 "G01": ["numbers-draft-1", "numbers-ledger-docs-3", "dataset-facts-1", "rederive-transfer-3", "referee-logic-7"],
 "G02": ["numbers-draft-2", "rederive-transfer-1", "referee-logic-1"],
 "G03": ["numbers-draft-3", "numbers-draft-25", "training-experiments-1", "referee-logic-8", "numbers-ledger-docs-13", "referee-logic-14"],
 "G04": ["numbers-draft-4", "numbers-ledger-docs-1", "dataset-facts-2", "code-regression-1", "referee-logic-2", "code-regression-12", "training-experiments-7", "referee-logic-22"],
 "G05": ["numbers-draft-5", "rederive-transfer-4", "rederive-transfer-12", "numbers-draft-19"],
 "G06": ["numbers-draft-6", "numbers-ledger-docs-4", "training-experiments-2", "numbers-draft-7", "training-experiments-4", "referee-logic-11", "referee-logic-12"],
 "G07": ["numbers-draft-8", "training-experiments-5"],
 "G08": ["numbers-draft-9", "training-experiments-12", "referee-logic-13"],
 "G09": ["numbers-draft-10", "numbers-ledger-docs-14", "dataset-facts-9", "dataset-facts-11", "rederive-transfer-16", "science-relabel-18", "referee-logic-4"],
 "G10": ["numbers-draft-11", "numbers-ledger-docs-8", "dataset-facts-7", "dataset-facts-12", "rederive-transfer-6", "dataset-facts-13", "dataset-facts-14"],
 "G11": ["numbers-draft-12", "science-relabel-10", "referee-logic-27", "numbers-draft-18", "dataset-facts-8", "rederive-transfer-8"],
 "G12": ["numbers-draft-13", "science-relabel-2", "science-relabel-3", "referee-logic-5", "science-relabel-15", "numbers-ledger-docs-11"],
 "G13": ["numbers-draft-14", "numbers-ledger-docs-2", "rederive-transfer-2", "science-relabel-1", "referee-logic-6"],
 "G14": ["numbers-draft-15", "training-experiments-11", "referee-logic-15"],
 "G15": ["numbers-draft-16", "numbers-ledger-docs-9", "science-relabel-9", "code-regression-9", "referee-logic-10"],
 "G16": ["numbers-draft-17", "dataset-facts-3", "referee-logic-24"],
 "G17": ["numbers-draft-20", "numbers-draft-21", "rederive-transfer-15", "referee-logic-34"],
 "G18": ["numbers-draft-22", "training-experiments-18", "referee-logic-16"],
 "G19": ["numbers-draft-23", "science-relabel-12"],
 "G20": ["numbers-draft-24", "rederive-transfer-18"],
 "G21": ["numbers-draft-26", "dataset-facts-19", "training-experiments-17"],
 "G22": ["numbers-draft-27", "training-experiments-19", "referee-logic-35", "numbers-ledger-docs-16"],
 "G23": ["numbers-ledger-docs-5", "code-regression-5"],
 "G24": ["numbers-ledger-docs-6", "code-regression-6"],
 "G25": ["numbers-ledger-docs-7", "code-regression-8", "referee-logic-30", "dataset-facts-17"],
 "G26": ["numbers-ledger-docs-10", "numbers-ledger-docs-12", "rederive-transfer-7", "science-relabel-5", "science-relabel-14", "referee-logic-19", "rederive-transfer-19"],
 "G27": ["numbers-ledger-docs-15", "referee-logic-29"],
 "G28": ["numbers-ledger-docs-17", "numbers-ledger-docs-19", "science-relabel-13", "numbers-ledger-docs-21", "rederive-transfer-13", "rederive-transfer-14", "dataset-facts-21", "dataset-facts-18", "dataset-facts-16", "rederive-transfer-11", "rederive-transfer-21", "numbers-ledger-docs-24"],
 "G29": ["numbers-ledger-docs-18", "training-experiments-3", "training-experiments-16"],
 "G30": ["numbers-ledger-docs-20", "dataset-facts-15", "rederive-transfer-5", "rederive-transfer-17"],
 "G31": ["numbers-ledger-docs-22", "referee-logic-17", "code-regression-15"],
 "G32": ["numbers-ledger-docs-23", "code-regression-16", "referee-logic-33"],
 "G33": ["dataset-facts-4"], "G34": ["dataset-facts-5"], "G35": ["dataset-facts-6", "referee-logic-20"],
 "G36": ["dataset-facts-10", "referee-logic-23"], "G37": ["dataset-facts-20", "code-regression-14"],
 "G38": ["science-relabel-4"], "G39": ["science-relabel-6"], "G40": ["science-relabel-7"],
 "G41": ["science-relabel-8", "science-relabel-16"], "G42": ["science-relabel-11"], "G43": ["science-relabel-17"],
 "G44": ["referee-logic-3", "training-experiments-10"], "G45": ["training-experiments-6", "referee-logic-18"],
 "G46": ["training-experiments-8", "code-regression-2"], "G47": ["training-experiments-9", "code-regression-4", "referee-logic-31"],
 "G48": ["code-regression-3"], "G49": ["code-regression-7", "code-regression-13"], "G50": ["code-regression-10"],
 "G51": ["code-regression-11"], "G52": ["code-regression-17"], "G53": ["rederive-transfer-20"],
 "G54": ["rederive-transfer-10", "training-experiments-13", "training-experiments-14"], "G55": ["training-experiments-15"],
 "G56": ["referee-logic-9"], "G57": ["referee-logic-25"], "G58": ["referee-logic-26"], "G59": ["referee-logic-21"],
 "G60": ["referee-logic-28"], "G61": ["referee-logic-32", "rederive-transfer-9"],
}


def main():
    R = json.load(open(os.path.join(HERE, "2026-09-11_findings.json")))
    ids = {f["id"]: f for f in R["findings"]}
    mapped = {i: g for g, v in MAP.items() for i in v}
    missing = sorted(set(ids) - set(mapped)); extra = sorted(set(mapped) - set(ids))
    assert not missing and not extra, (missing, extra)
    lines = []
    for g, (title, disp) in ISSUES.items():
        fs = [ids[i] for i in MAP[g]]
        sev = "critical" if any(f["severity"] == "critical" for f in fs) else ("major" if any(f["severity"] == "major" for f in fs) else "minor")
        lines.append(f"| {g} | {sev} | {title} | {disp} | {', '.join(MAP[g])} |")
    return lines, R


if __name__ == "__main__":
    lines, R = main()
    print(len(lines), "issues;", len(R["findings"]), "findings mapped")

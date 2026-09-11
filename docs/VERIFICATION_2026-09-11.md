# Verification of the revision work, 2026-09-11

**What was checked.** Everything that will go into the A&C revision from the post-submission work: the
draft manuscript text (`paper/draft_gulls_section.tex`), the experiment ledger and section 1 of
`paper/REVISION.md`, the README / usage / CHANGELOG / audit entries, the two example notebooks, and every
code path changed since the audit fixes (27 files: `binml/gulls.py`, `pipeline/*`, `validation/gulls/*`,
the runners, the tests).

**How.** A workflow of eight independent verifiers, each re-deriving one slice from the raw data without
trusting any stated number (numbers in the draft; numbers in the ledger and docs; dataset facts from the
pinned RMDC26 tables; a fresh re-implementation of the transfer tables; the science of the relabelling;
the training experiments; code review with the full test suite in a separate worktree; a referee-style
logic read), then two adversarial checkers that tried to refute every finding and listed what nobody
had checked. Raw output: `docs/verification/2026-09-11_findings.json`.

**Result.** 182 findings (30 critical, 86 major, 66 minor);
160 upheld as stated, 22 upheld in part, 0 refuted. They reduce to
61 distinct issues, listed below with what was done. Several changed conclusions, not just wording:

* **RMDC26's pause schedule differs between its six high-cadence seasons.** The "exact schedule" mask used in
  the schedule experiment and in the threshold calibration matched only the first season (16% of scored
  events). Experiment 2 was rerun with the measured per-season schedule and single-factor arms, and every
  calibration was redone under the measured seasons on one common held-out pool.
* **Partial bin occupancy.** RMDC26's colour visits leave one of eight F146 epochs empty in about a third of
  the model's bins, a pattern training never contains. It cannot be imposed faithfully on binned data (doing
  so produced a spurious collapse, caught before use); on real RMDC26 inputs its effect was measured directly.
* **The colour-band gain is not anomaly signal.** It comes from planetary events without a detectable F146
  anomaly; its mechanism is not identified.
* **"The schedule hides half the planets" was wrong.** The unobserved gap costs 12 (1S2L) and 20 (2S2L) points of claimable anomalies.
* **Recall against RMDC26's generator labels is not bounded by the detectable fraction**, and "no anomaly a
  survey could claim" was an overstatement of what our 0.02 mag, single-season policy says.
* **The onset fix had two off-by-one errors and changed a released default**; the relabel rule that consumes
  the onset is not truth-based even with an exact onset (audit finding 9 re-opened).
* **The finite-source magnification function had artificial 4-8 mmag steps**; replaced, and the recommended
  checkpoint's recipe rerun with the smooth function.
* **The finite-source gain had no continued-training control**; one was run.
* **RMDC26 served as a development set** (diagnosis, a prior bound, checkpoint choice among the scored fine-tunes: seven at the time, 14 by the second pass); disclosed.
* Plus the count error (385,004 was an event id; the release has 377,999 events), an impossible argmax
  statement, uncommitted notebooks and weights, numbers with no artifact, double rounding, and stale text.

Two statements made in conversation during the work were also wrong and are corrected here: the quantised
onset DOES enter the shipped model's training labels (through the truncation augmentation, as the paper
says), and the between-seasons result does not show the schedule hiding half the planets.

A label-policy fragility older than this work was found along the way (a borderline single-start refit that
flips under 1e-14 mag perturbations, about one event per thousand); it is recorded in
`validation/gulls/PROVENANCE.md` and not changed, because changing it would change the released labels.

## Issues and dispositions

| Issue | Severity | What was wrong | What was done | Findings |
|---|---|---|---|---|
| G01 | critical | RMDC26 event count quoted as 385,004 (an event id) | Fixed: 377,999 events (299,466 / 35,647 / 42,886), from rmdc26_dataset_facts.json. | numbers-draft-1, numbers-ledger-docs-3, dataset-facts-1, rederive-transfer-3, referee-logic-7 |
| G02 | critical | 'argmax of most flagged single lenses is PeriodicVar' (impossible) | Fixed: flagged events are NonPSPL by construction; argmax split over all single lenses 53% NonPSPL / 44% PeriodicVar / 2.5% PSPL. | numbers-draft-2, rederive-transfer-1, referee-logic-1 |
| G03 | critical | shipped model under the schedule: 'anomaly and variable classes unaffected'; two experiments quoted interchangeably | Fixed: Eruptive, LongPeriodVar and NonPSPL degradations and the NonPSPL precision collapse reported; one experiment (schedule_finetune.json) per statement, sources named. | numbers-draft-3, numbers-draft-25, training-experiments-1, referee-logic-8, numbers-ledger-docs-13, referee-logic-14 |
| G04 | critical | schedule described as seven pauses at fixed season phases; 'exact schedule' mask matches one season in six | Fixed: rmdc26_schedule.json (six measured pause patterns); training --gap-schedule rmdc26_seasons; calibration under measured seasons; experiment 2 rerun with single-factor arms; colour bins blanked only where empty; 'planned exposure and calibration' removed. | numbers-draft-4, numbers-ledger-docs-1, dataset-facts-2, code-regression-1, referee-logic-2, code-regression-12, training-experiments-7, referee-logic-22 |
| G05 | major | double rounding (11.7%, 4.9%, 0.312 ...) | Fixed: exact counts stored; rates rounded once (11.6%, 4.8%, 0.313); achieved false-alarm rate reported at each budget. | numbers-draft-5, rederive-transfer-4, rederive-transfer-12, numbers-draft-19 |
| G06 | critical | gap table: one cell from another arm; 'gap-aware' meaning three checkpoints; macro-F1 and clean cost from different checkpoints | Fixed: table regenerated from schedule_finetune.json with named arms; each sentence quotes one checkpoint on one held-out set. | numbers-draft-6, numbers-ledger-docs-4, training-experiments-2, numbers-draft-7, training-experiments-4, referee-logic-11, referee-logic-12 |
| G07 | major | gap table N = pool size, not scored rows | Fixed: 24,011 scored of a 30,013-event pool (20% chooses the operating point); both stored. | numbers-draft-8, training-experiments-5 |
| G08 | major | 'gaps <= 2 h are harmless' | Fixed: 0.5 h costs nothing; one 1-2 h gap moves 7 of 100 single lenses to NonPSPL. | numbers-draft-9, training-experiments-12, referee-logic-13 |
| G09 | critical | 2S2L q 'identical to 1S2L (1.25e-4)'; binary source in '56%' | Fixed: similar, not identical (1.4e-4 vs 1.2e-4, KS p ~2e-18); binary source flagged in 55% of scored 2S2L; a NonPSPL call on 2S2L may respond to the binary source. | numbers-draft-10, numbers-ledger-docs-14, dataset-facts-9, dataset-facts-11, rederive-transfer-16, science-relabel-18, referee-logic-4 |
| G10 | major | selection accounting (eligible defined with the season cut; 25.5% 'between seasons' includes out-of-mission peaks) | Fixed: eligible = amplitude + t_E cuts; split 56.4% scored / 22.0% between seasons / 3.6% before or after the survey / 17.9% low-cadence / 0.08% no baseline; gaps 109-120 d. | numbers-draft-11, numbers-ledger-docs-8, dataset-facts-7, dataset-facts-12, rederive-transfer-6, dataset-facts-13, dataset-facts-14 |
| G11 | major | numbers with no committed artifact | Fixed: derived quantities written by the reducers (relabelling decomposition and flag rates, relabel exposure, dataset facts, occupancy, per-event score table). | numbers-draft-12, science-relabel-10, referee-logic-27, numbers-draft-18, dataset-facts-8, rederive-transfer-8 |
| G12 | critical | 'no anomaly a survey could claim' / 'recall ceiling 0.56' | Fixed: 'no anomaly our policy would claim within one season' (policy-, floor- and selection-dependent); 0.56 is what a classifier following the policy exactly would score, not a bound; both label sets reported. | numbers-draft-13, science-relabel-2, science-relabel-3, referee-logic-5, science-relabel-15, numbers-ledger-docs-11 |
| G13 | critical | 'the schedule hides about half of the planets' | Fixed: rerun on between-season events only, with a multi-start refit; the unobserved gap costs 12 / 20 points (1S2L / 2S2L) of claimable anomalies relative to in-season (44% / 34% vs 56% / 54%). | numbers-draft-14, numbers-ledger-docs-2, rederive-transfer-2, science-relabel-1, referee-logic-6 |
| G14 | major | finite-source gain confounded with 12 extra epochs and a different pool; round-2 penalty understated | Confound confirmed and separated: the point-source control (pspl5s_ctrl, same pool shape, recipe and warm start) gives FA 6.4% and recall 0.361 at 5.2% (g08e12 11.6% / 0.295; fspl5s 4.8% / 0.390), so the extra training explains most of the gain; the physics lowers false alarms further in every rho/|u0| bin, most in rate in the two largest (0.41 / 0.51 -> 0.28 / 0.29), and its recall gain is not resolved under rate weighting (second pass, H02); only the physics restores large-source single-lens recall on our own held-out (0.22 / 0.16 -> 0.66 / 0.84). Round-2 penalty quantified (FA 8.7% vs 4.8%). | numbers-draft-15, training-experiments-11, referee-logic-15 |
| G15 | critical | precision-vs-floor 'peak' (true by construction) | Fixed: argument removed; precision labelled sample-mix and also given at the scored-set mix. | numbers-draft-16, numbers-ledger-docs-9, science-relabel-9, code-regression-9, referee-logic-10 |
| G16 | critical | 'median q a factor of several below our prior' | Fixed: RMDC26 anomalies are all planetary (median q ~1.3e-4); 82% of our NonPSPL evaluation events are stellar-mass-ratio binaries (median q 0.12); ratios from rmdc26_dataset_facts.json. | numbers-draft-17, dataset-facts-3, referee-logic-24 |
| G17 | minor | table bin labels; colour gain 'for every checkpoint' | Fixed: lowest rho/|u0| bin added; colour gain given per checkpoint (0.042-0.068). | numbers-draft-20, numbers-draft-21, rederive-transfer-15, referee-logic-34 |
| G18 | major | noise arm 'worse on every measure'; pools mixed in its calibrated comparison | Fixed: worse at every matched budget and in false alarms, higher frozen-threshold recall; single seed; like-for-like calibration. | numbers-draft-22, training-experiments-18, referee-logic-16 |
| G19 | minor | floor sweep vs relabel artifact disagree at 0.02 (rounding) | Fixed: full-precision truth cache; the sweep asserts agreement at the adopted floor. | numbers-draft-23, science-relabel-12 |
| G20 | minor | sub-day false alarms unweighted only; sample described as 'first dense ids' | Fixed: 0.7% unweighted, 3.1% rate-weighted; sample is a random contiguous id window. | numbers-draft-24, rederive-transfer-18 |
| G21 | minor | 'not one training event has an empty mid-season bin' | Fixed: essentially none (6 of 89,919). | numbers-draft-26, dataset-facts-19, training-experiments-17 |
| G22 | minor | stale header/status lines, inconsistent signed lags, housekeeping | Fixed. | numbers-draft-27, training-experiments-19, referee-logic-35, numbers-ledger-docs-16 |
| G23 | major | example notebooks gitignored (CI test fails on a clean checkout) | Fixed: force-added. | numbers-ledger-docs-5, code-regression-5 |
| G24 | major | ft_g08e12.pt gitignored; GULLS artifacts not manifest-hashed | ft_g08e12.pt committed. Manifest hashing of the GULLS artifacts is part of the macro integration (open). | numbers-ledger-docs-6, code-regression-6 |
| G25 | major | binml.gulls combiner differs from the measured one; notebook demo events not between seasons | Fixed: modes peak/adjacent/all with p_nonpspl_max; low-cadence skipped; notebook uses a real between-season event (315360). | numbers-ledger-docs-7, code-regression-8, referee-logic-30, dataset-facts-17 |
| G26 | major | recall and false alarms quoted at different operating points | Fixed: every rate carries its threshold; the recommended operating point is re-derived under measured seasons (full-pool threshold with slice spread). | numbers-ledger-docs-10, numbers-ledger-docs-12, rederive-transfer-7, science-relabel-5, science-relabel-14, referee-logic-19, rederive-transfer-19 |
| G27 | major | Reproduce commands reproduce a 600-per-class sample | Fixed: full-population commands; artifacts record command and code. | numbers-ledger-docs-15, referee-logic-29 |
| G28 | minor | assorted ledger / section-1 numbers (Wilson half-widths, |u0|<0.1 range, floor sequence, 0.77 vs 0.76, flip fraction, comments) | Fixed in paper/REVISION.md and code comments. | numbers-ledger-docs-17, numbers-ledger-docs-19, science-relabel-13, numbers-ledger-docs-21, rederive-transfer-13, rederive-transfer-14, dataset-facts-21, dataset-facts-18, dataset-facts-16, rederive-transfer-11, rederive-transfer-21, numbers-ledger-docs-24 |
| G29 | critical | relabel rate '20% of all binaries on every presentation' | Fixed: 20.1% of NonPSPL-labelled pool events are eligible (7.4% of generated binaries); it fires on about half of their gapped presentations; stored in schedule_finetune.json. | numbers-ledger-docs-18, training-experiments-3, training-experiments-16 |
| G30 | major | streaming-purity prevalence computed on the balanced scan sample | Fixed: RMDC26 scored-set rate-weighted planetary fraction (22.07%) with weighted alert rates; unrounded. | numbers-ledger-docs-20, dataset-facts-15, rederive-transfer-5, rederive-transfer-17 |
| G31 | major | cascade comparison mixes models; untested causal claim; scan details differ from in-house | Fixed: in-house numbers named as the shipped model on 80% stellar binaries, planetary strata quoted; causal wording withdrawn. Rescanned with >= 10 F146 points per cut and unrounded P (1,667 eligible binaries): premature 2.2% (1.6-3.0%), lag +6.5 d, detected 57% at the frozen threshold, against 1.4-4.2% premature, lag 4.5 d and 73-89% detected in the in-house planetary strata; single-lens alert burden 5.7% of events per season. | numbers-ledger-docs-22, referee-logic-17, code-regression-15 |
| G32 | minor | quickstart demo never crosses the threshold | Fixed: planetary example (seed 241) crosses at day 49.5 after a 43.5-day onset. | numbers-ledger-docs-23, code-regression-16, referee-logic-33 |
| G33 | major | rho <= 5 prior 'to cover GULLS' | Fixed: scored single lenses have rho p99 0.29 (43 above 1); the high rho/|u0| bins come from tiny |u0|. | dataset-facts-4 |
| G34 | major | out-of-support inputs not quantified | Fixed: out-of-support fractions per class in rmdc26_dataset_facts.json; residual false alarms 11.8% at m_base < 20 vs 4.1% inside the prior. | dataset-facts-5 |
| G35 | major | '12.1-min cadence indistinguishable after preprocessing' | Fixed: partial occupancy disclosed and measured on real inputs (occupancy_sensitivity.json); raw-pooling sensitivity restored. | dataset-facts-6, referee-logic-20 |
| G36 | major | amplitude cut described as neutral; baseline 'as a survey pipeline would have' | Fixed: host-lens peak amplitude, removing most wide-orbit planets (s > 2: 18% / 6% pass); baseline uses catalogue t0 and t_E. | dataset-facts-10, referee-logic-23 |
| G37 | minor | binml.gulls docstring and density definition | Fixed. | dataset-facts-20, code-regression-14 |
| G38 | major | wing-season refit seeded at an out-of-window t0 | Fixed: multi-start refit when the peak is outside the window; host-visible subset reported; rerun. | science-relabel-4 |
| G39 | major | 'amplitude-limited' recall | Fixed: significance-limited; recall by delta-chi2 stored. | science-relabel-6 |
| G40 | major | sub-floor flag rate compared with all single lenses | Added single-lens control: half of single lenses with a significant sub-floor static-refit misfit (mostly parallax) are flagged, like sub-floor planets. | science-relabel-7 |
| G41 | major | single lenses labelled NonPSPL by the relabelling (training rule never does) | Fixed: single lenses follow the training rule; the point-source-refit result is a separate diagnostic with n and interval. | science-relabel-8, science-relabel-16 |
| G42 | major | truth sample included 600 events from an aborted run | Fixed: re-extracted with the documented selection (1,600 / 2,500 / 2,500). | science-relabel-11 |
| G43 | minor | fixed delta-chi2 cuts on ~11% more epochs | Disclosed; rescaled-cut sensitivity stored (< 1 point). | science-relabel-17 |
| G44 | critical | RMDC26 described as a pure test set | Disclosed: it guided the diagnosis, the single-lens rho prior and the choice among the fine-tuned checkpoints scored on it (14 by 2026-09-12; the first version of this disposition said seven and the draft macro counted 8-9, second pass H05); numbers for the chosen checkpoint are optimistic. | referee-logic-3, training-experiments-10 |
| G45 | major | 'g08e12 cannot reach the purity target at any threshold' | Fixed: no usable operating point (0.90 purity only on the top few events); calibration-pool prevalence stored (6.0% vs 5.6% on the paper test set). | training-experiments-6, referee-logic-18 |
| G46 | major | fine-onset scan off by one at both ends | Fixed: full-grid first-detectable scan; boundary tests. | training-experiments-8, code-regression-2 |
| G47 | major | onset default changed released behaviour | Fixed: default back to the legacy 7.2-d grid; opt-in --onset-resolution-days; settings recorded in shard attributes. | training-experiments-9, code-regression-4, referee-logic-31 |
| G48 | major | caustic-in-gap relabel is not truth-based even with an exact onset | Audit finding 9 re-opened; relabel off in the new schedule arm. | code-regression-3 |
| G49 | major | recorded commits wrong; resumable stamps ignore settings | Fixed going forward (generation-time git describe --dirty, settings in stamps); validation/gulls/PROVENANCE.md records the past. | code-regression-7, code-regression-13 |
| G50 | minor | ESPLMag2 hand-off steps of 4-8 mmag | Fixed: ESPLMag (matches disc integration); legacy flag. The combined arm (smooth function + measured-season pauses + relabel off) matches round 3 at every matched budget (1S2L recall within 0.008); the smooth-function-only control (fspl5s_espl_g08) is indistinguishable from round 3 (FA 4.83% vs 4.85%, recall within 0.001 at every budget, same recalibrated operating point), so the steps did not shape the result. | code-regression-10 |
| G51 | minor | truncation amplitude uses point-source magnification for finite sources | Fixed for rho > 0.01 (released pools unaffected). | code-regression-11 |
| G52 | minor | np.trapezoid needs NumPy 2 | Fixed: fallback. | code-regression-17 |
| G53 | minor | transfer status string wrong for fine-tunes | Fixed. | rederive-transfer-20 |
| G54 | minor | 'exact schedule matches random gaps'; two factors changed at once | Rerun with one pool and one factor: measured seasons beat random gaps (FA 10.0% vs 12.8%; 1S2L recall 0.352 vs 0.279 at 5.2%; ahead in 6 of 6 seasons; held-out AP under the measured pauses 0.921 vs 0.911). Combined with finite-source single lenses it adds nothing further on RMDC26 at matched budgets. | rederive-transfer-10, training-experiments-13, training-experiments-14 |
| G55 | minor | onset-fix twin described as 'within a point or two' | Fixed: finite-source bins 4 points worse; held-out labels identical. | training-experiments-15 |
| G56 | critical | colour ablation read as support for the three-band design | Fixed: the colour gain comes from binaries without a detectable F146 anomaly; mechanism unidentified. | referee-logic-9 |
| G57 | major | 'the remaining transfer gap was a missing piece of physics' | Fixed: 'a large part of the residual false alarms'; the largest-source bins remain at ~0.28. | referee-logic-25 |
| G58 | major | 'nearly doubles' against the shipped baseline at a post-hoc budget | Fixed: like-for-like 0.295 -> 0.390; the budget's origin stated. | referee-logic-26 |
| G59 | major | missing caveats (approximate relabel, static refit, uniform disc, single seed) | Added. | referee-logic-21 |
| G60 | major | macro names with digits; undefined \bmlThreshold | Fixed in the macro list. | referee-logic-28 |
| G61 | major | REVISION section 1 'Where it goes' stale | Fixed. | referee-logic-32, rederive-transfer-9 |

## What no verifier checked (the adversarial pass's completeness list) and what happened to it

* GULLS was used for design and model selection, and no verifier assessed this. The sidecar was picked from at least 7 checkpoints on the same 56,975 events (REVISION row 8: 'better on the external test'). The single-lens rho prior was widened 'to cover GULLS' single-lens rho' from GULLS metadata. bkg_mult was fitted to GULLS noise. The amplitude and tE cuts were designed after the first run 'returned PeriodicVar almost uniformly' (gulls_transfer.py comments). The fspl5s numbers are therefore best-of-N on the test set. The draft's 'We use it as a test set … no threshold was chosen on it' and 'trained only on our own simulations' need qualifying.
* Training-seed variance was never used to judge between-checkpoint claims; every arm is single-seed. The closest replicate pair (rand vs g08e12, same recipe, different pools) differs by 0.014/0.020/0.023 at the 2/5.2/11.7% budgets. That is the size of the v2-vs-v1 gap (0.015), fspl5s-vs-fspl (0.021) and norelabel-vs-rand (0.013), which the text interprets as real effects ('marginally worse', 'slightly hurt', 'matches').
* The cascade comparison mixes models. The in-house cascade (1.6%, +5.0 d, 89%) is from the shipped checkpoint (cascade_reproduce_result.json model 'shipped (cascade-trained)') on 80% stellar-q binaries, while the GULLS cascade uses fspl5s. 'The timing behaviour transfers' compares different models, and no in-house fspl5s cascade was run.
* The 0.949 calibration's gapped arm (calibrate_gapped_threshold.gap_mask_f146, called with mask=None at l.101) blanks only the seven season-0 pauses. It omits the 1.3 d past the 70.7-d season end, which train.py's rmdc26_schedule_mask and Table gap's mm_heldout_sched (37 bins) include. It also omits the other five seasons' pause patterns and the frac=0.875 colour holes. Nobody tested how 0.949 and the 2.0% FA depend on this.
* Residual false alarms were not attributed to out-of-support inputs. fspl5s FA is 11.8% on the 3,186 scored single lenses with m_base<20 (outside the U(20,25) training range) vs 4.1% within range, and 6.1% vs 4.5% for fs<0.1. The draft attributes the residual FA to finite source alone.
* Held-out/training disjointness was not verified: seeds and shard indices of the fspl*/schedule pools vs their held-outs, and whether the Table-gap held-out (work15 mm_eval, shards 100-103, seed 20260720) is disjoint from the shipped stage-5 and g08e12 Modal training pools.
* validation/cadence_local.py is new since 76a50de and a target, but nobody reproduced its outputs (cadence_result AP 0.827 vs 0.830, in the paper via macros) from cadence_local_work.
* Nobody checked whether the new default onset_resolution_days=0.5 changes any paper figure or number regenerated from simulate_event in paper/build.sh (only the onset-line positions were computed). Nobody checked either whether a clean-archive build or CI passes, given the missing notebooks.
* Nobody quantified the sensitivity of the GULLS FA to the frac-channel mismatch, for example by rescoring with frac forced to 1 in the 0.875 bins, or how per-season FA varies for shipped and fspl5s. fspl5s per-season FA is 0.041-0.063 and shipped 0.29-0.42; the per-season structure was examined only for the schedule arms.
* Not checked: whether RMDC26 flux_err_uJy is computed at true or at measured flux (this affects the relabel σ, the 44-46% and the noise-ratio 2.6x); the meaning of ObsGroup_0_chi2; and whether GULLS includes limb darkening or orbital motion.
* The RGESPIT2026 bib entry is missing from refs.bib ([F] TODO; \citep will break the build). RMDC26 data-use and citation terms were not checked.
* Nobody checked the end-to-end regenerability claimed in [E] ('all per-event rows are regenerable from validation/gulls_transfer.py'), which needs the 172 GB obs table and --per-class 200000. Nor whether the full-mission empirical baseline, which uses post-alert data, is consistent with the cascade's real-time framing.
* Model inference was not re-run: the training-experiments-2 logits re-run, the 37-bin re-evaluation behind training-experiments-7 (only its GULLS consequences at 0.9395 / 0.9436 were checked), and a fresh-worktree pytest for code-regression-5 (verified from .gitignore, git ls-files and the test code instead).
* Curve-cache chunks (c_*.npz) were not read, so these remain unchecked: code-regression-1's per-chunk F146 occupancy (324/665 pause instances) and code-regression-12's 89.8% F087 occupancy of pause-overlapping colour bins.
* The augmentation simulations (b_sim) for training-experiments-3 and -16 were not re-run; analytic probabilities from the _apply_truncation / _apply_gaps code were used instead.
* Not verified: code-regression-3's random-mask numbers (the fixed-mask part reproduced); code-regression-10's pool prevalence (43.8% / 5.7% / 0.16%); code-regression-11's 1.3%; code-regression-14's 3-event density mismatch; science-relabel-1's caustic-timing and near-resonant statistics and the '80% of wing seasons t0 > 0.5 tE outside'; referee-logic-23's t0 concentration (65-71%); training-experiments-11's trained_epoch values; training-experiments-15's paired bootstrap.
* science-relabel-7's property-matched single-lens flag rate depends on binning: the finding gives 13.5%, my tE x |u0| x m_base (x rho/|u0|) cells give 18-21%. Both are far above 4.9%, but the replacement number should be defined explicitly before it goes into the text.
* New observation, not in any finding: frozen cascade seed 304033 now replays as PSPL at HEAD (dchi2_anomaly 4.5; the trace recorded t_anom_coarse 72.0 and fine 67.5). validation/cascade_trace.py would therefore raise RuntimeError at HEAD whatever the onset setting. _pspl_refit_dchi2 is unchanged since the trace commit 9c3389a (which was git_dirty), so the cause is probably environment or numerics in a borderline fit; not diagnosed further.

Dispositions of these gaps: development-set use, seed variance, the cascade model mix, the calibration
mask, out-of-support attribution, held-out disjointness (held-outs use shards 100-103, outside the shipped
model's training shards 0-89 and every fine-tune's shards 0-11), the occupancy mismatch, the notebook/CI
break and the missing bibliography entry are all addressed above or in paper/REVISION.md section 1.5. Not
done: a clean-archive paper build (deferred to the macro integration), and establishing whether RMDC26's
flux_err is computed at the true or the measured flux, whether GULLS includes limb darkening, and what
ObsGroup_0_chi2 is exactly (questions for the data providers; disclosed as unknowns).

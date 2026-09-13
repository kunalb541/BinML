"""Dispositions of the fourth verification (2026-09-13, final state after the third round's fixes).

Read by docs/verification/build_doc_2026_09_12b.py, which appends them to docs/VERIFICATION_2026-09-12b.md as an
addendum and asserts that every finding in 2026-09-13_fourth_findings.json has exactly one disposition."""

DISPO = {
    # stress (the released model's stress numbers)
    "stress-01": "Fixed. run_shard --legacy-t0-pad reproduces the July generator's peak-time draw (tested: same random stream, "
                 "kept t0); the three sweep tiers were regenerated with it (sub-day PSPL fraction 14.8% vs the suite's 14.75%; "
                 "stage-5 PSPL recall 0.532 vs 0.524), the corrected sub-day tier is kept as a second row; 43% quoted for the "
                 "suite's mix; 'bit for bit' removed from code, PROVENANCE, REVISION row 23 and CHANGELOG.",
    "stress-02": "Fixed. The subset is described as a new realisation (unpinned fleet environment; local versions recorded in the "
                 "artifact); the 2.3% anomaly deficit and stage 5's 0.956 vs 0.941 are stated, checkpoints compared within the "
                 "subset (wide s: 0.328 vs 0.281); 'stands in' dropped; guards check the two largest differences and the deficit.",
    "stress-03": "Fixed. The faint sweep's precision is attributed to its class mix (4% binaries by weight vs 68%; detectable "
                 "fraction 8.1% vs 4.9%, guarded); the paper, abstract and docs quote the mix-independent false-anomaly rates "
                 "among microlensing events (3.3% vs 26%; 0.9% vs 12% above the threshold).",
    "stress-04": "Fixed. The RMDC26 sub-day sentence now gives the matched-timescale contrast (released model 4.3% of RMDC26's "
                 "0.25-1 d single lenses vs 35% of our sweep) and says timescale alone does not explain it; guarded.",
    "stress-05": "Fixed. The abstract quotes the operating-threshold rates (35% of sub-day single lenses; 12% of faint microlensing "
                 "events without an anomaly, 0.9% natural).",
    "stress-06": "Fixed. Low-q and faint (23.5-25) pools entered at stage 4; stage 6 added more faint data and pools overlapping the "
                 "wide-s and sub-day sweeps (s 3-8, tE 0.3-10 d); appendix and docs say so.",
    "stress-07": "Fixed ('comparable to or longer than the 72-day season (60-200 d)').",
    "stress-08": "Fixed: 8.7M in 12 targeted regimes (mostly training pools) and 1.7M in 17 sweeps (README, evaluation.md, "
                 "leakage_audit.md, model card; the CHANGELOG notes the earlier figure).",
    # RMDC26 cascade
    "cascade-1": "Fixed. cascade_gulls.py reduces the RMDC26 timing by the in-house mass-ratio strata (offline, from the cached "
                 "scan and the pinned meta parquet; every earlier value unchanged); the paper compares giant with giant and "
                 "Neptune with Neptune (F146 and three bands) and states the 40% at q <= 1e-4 that the in-house scan samples "
                 "with 9 events; REVISION row 10 corrected.",
    "cascade-2": "Fixed by disclosure: Data availability says the scan's traces are not distributed and regenerate only by "
                 "rerunning the scan; the re-derivation claim names this exception.",
    "cascade-3": "Fixed. cascade_reduce.py now reports the in-house three-band strata (cascade_reproduce_result.json regenerated "
                 "through the validated reducer); the paper quotes them.",
    "cascade-4": "Fixed. Guards for 'come later', 'lower', 'no more frequent', the three-band directions, the recalibrated "
                 "directions, 'stay rare' and 'several times' (now >= 3x); perturbation cases added.",
    "cascade-5": "Fixed. Sample described as the first 3,000 scored single lenses and 1,500 of each planetary class; Wilson "
                 "interval for the single-lens alert rate; the sample's lower full-season flag rate (4.8% vs 6.3%) stated as "
                 "a slight optimism (guarded).",
    "cascade-6": "Fixed ('rarely alert for the shipped checkpoint ... not scanned with the recommended one').",
    "cascade-7": "Fixed in the text ('generated single lenses with a PSPL label', also in Sec. cascade, README, model card, "
                 "evaluation.md); the artifact key keeps its name.",
    # ledger
    "ledger-referee-archive-untracked": "Fixed. The archive is committed (gitignore exceptions) and a test checks every hashed "
                                         "file is tracked and matches.",
    "ledger-row7-stale-colour": "Fixed (recommended checkpoint added, 'only' dropped).",
    "ledger-rows6-11-not-updated": "Fixed (rows 6 and 11 carry the recommended checkpoint's values).",
    "ledger-row10-stale-wording": "Fixed (units, weighted figure, per-event labels, stratified conclusion).",
    "ledger-row9-host-visible-rounding": "Fixed (51% / 37%).",
    "ledger-row22-section-label": "Fixed (Sec. cascade, Sec. limits; also AUDIT, CHANGELOG, status table).",
    "ledger-header-counts-stale": "Fixed (row 23 cited, 23 rows, status table refreshed).",
    "ledger-subday-sentence-unlogged": "Fixed (row 6 records the matched-timescale comparison).",
    "ledger-unlogged-paper-changes": "Fixed (row 2 qualifier; the three paper changes listed in the CHANGELOG).",
    "ledger-verification-addendum-missing": "Fixed: this addendum.",
    # text
    "text-stress-subset-not-the-set": "Fixed with stress-01/02.",
    "text-subday-reconciliation-unsupported": "Fixed with stress-04; guards for the median, the in-support comparison and the "
                                              "PeriodicVar majority.",
    "text-regeneration-not-bit-identical": "Fixed. The colour paragraph says 75% of the regenerated events reproduce held-out "
                                           "events and 19% of those are threshold-selection rows (macros, guarded); Data "
                                           "availability says regeneration is statistically equivalent, not event-identical.",
    "text-abstract-subday-and-faint-wording": "Fixed with stress-05.",
    "text-abstract-length": "Flagged to the author (now about 380 words; A&C's limit not checked).",
    "text-finite-size-cannot-matter": "Fixed (physical claim removed).",
    "text-pauses-at-least-as-well": "Fixed ('as well within the seed spread'; 'lead in recall by more than the seed range of "
                                    "Table tab:seeds').",
    "text-seed-table-display-ties": "Fixed ('compared at full precision' in the caption).",
    "text-truncation-label-wording": "Fixed ('usually delays', 212 of 241, guarded; the fixed rule's 2.4% attributed to both "
                                     "sources without a proportion).",
    "text-stress-regime-descriptions": "Fixed.",
    "text-appendix-xref-0p3d": "Fixed (points to Sec. limits; the appendix gives the pool ranges).",
    "text-conclusion-submitted-version": "Fixed ('an earlier version of this work').",
    # code
    "code-1": "Fixed with stress-01.",
    "code-2": "Fixed with ledger-referee-archive-untracked.",
    "code-3": "Fixed. inference_benchmark_result.json hashed; make_macros.py has an input registry and --list-inputs; a test "
              "checks its inputs against the manifest.",
    "code-4": "Fixed. Each tier writes WORK/stamp_<tier>.txt when generated and the artifact records it per tier; an empty "
              "code stamp is fatal.",
    "code-5": "Fixed. PROVENANCE lists which intermediates of referee_round.json were made at which commit; the 12b disposition "
              "I22 reworded.",
    "code-6": "Fixed. Every fail-closed case asserts the guard it is named for; the best-seed guard and the new cascade and "
              "sub-day guards have cases; the make_macros stress guards have value-perturbation checks.",
    # docs
    "docs-01": "Fixed. Peak learning rates of one-cycle schedules from the checkpoints' optimizer states (base 4e-4).",
    "docs-02": "Fixed. The baselines paragraph says up to 1.9M events per stage; the simulation paragraph no longer describes "
               "the earlier 10M-event run (canonical_numbers notes it).",
    "docs-03": "Fixed in all four docs.",
    "docs-04": "Fixed (all methods see all 6,912 F146 epochs; residual AP 0.545).",
    "docs-05": "Fixed with stress-08.",
    "docs-06": "Fixed (stage 6 relabels truncated binaries until the recorded onset).",
    "docs-07": "Fixed (one-cycle schedule, base run's early stop, 'five of them warm-started', fine-tunes' batch 256 and factor 2).",
    "docs-08": "Fixed (stages loaded data in the main process; per-worker seeding added afterwards).",
    "docs-09": "Fixed. The unsourced 958 ev/s benchmark is replaced by the logged per-epoch throughput (376-656 ev/s).",
    "docs-10": "Fixed with text-appendix-xref-0p3d.",
    "docs-11": "Fixed: this addendum.",
    "docs-12": "Fixed (README, model card, REVISION row 14 narrowed to planetary recall).",
    "docs-13": "Fixed (the notebook points to command_from_clone).",
    "docs-14": "Fixed.",
    "docs-15": "Fixed (finding 14's status cites the stage-4 log and the regenerated-subset result; the stage-6 shard indices "
               "remain the open item).",
    "docs-16": "Fixed (training.md gives the stage order and calls the commands illustrative).",
    "docs-17": "Fixed (corrections added to the CHANGELOG entries; REVISION 'stage-6 recipe').",
    "docs-18": "Fixed ('going to NonPSPL', guarded).",
    "docs-19": "Fixed (the unstored count dropped).",
}

GAPS4 = [
    "Confirmed and fixed. make_figures.py now derives where the misses sit from the frozen artifact: 55% of misses are at "
    "s > 2.9 (11% of detectable anomalies), 46% have anomaly dchi2 > 1e4; argmax recall 0.744 vs 0.974, completeness 0.602 vs "
    "0.913; the plane's widest stellar-ratio cells 0.69-0.96. Sec. results rewritten from these macros (guarded).",
    "Fixed: 'across the evaluation pool'; the two false-positive diagnostics now have their own macros (94.4% of NonPSPL false "
    "positives; 95.0% of PSPL-to-NonPSPL errors).",
    "Fixed: argmax recall labelled as such where it sits next to threshold completeness.",
    "Fixed (caption: the periodic-variable example has no F087 epochs).",
    "Checked, fine.",
    "Checked, fine.",
    "Fixed ('about one percent, at most 1.6% at 95% confidence').",
    "Fixed: the RMDC26 comparison uses the final test's detectable anomalies (82%, median q 0.13).",
    "Disclosed: Data availability names the no-blue-band slice as read from metrics.json (per-event band counts not saved).",
    "Flagged to the author: paper/SUBMISSION_FIELDS.txt (untracked) still has the submitted abstract; regenerate it from the "
    "current abstract before resubmitting.",
    "Checked, fine.",
]

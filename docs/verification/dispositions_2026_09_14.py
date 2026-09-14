"""Dispositions of the fifth verification (2026-09-14, the state after the fourth verification's fixes, 9e95776).

Read by docs/verification/build_doc_2026_09_12b.py, which appends them to docs/VERIFICATION_2026-09-12b.md as a second
addendum and asserts that every finding in 2026-09-14_fifth_findings.json has exactly one disposition. The round's
completeness critic stalled (six attempts) and produced nothing; its job was done by the sixth check, recorded after it."""

DISPO = {
    # Sec. results and the figures_stats diagnostics
    "results-01": "Fixed. Sec. results now names two sources of misses, both derived in make_figures.py from the frozen artifact "
                  "and guarded: wide binaries (s > 2.9; 11% of detectable anomalies, 56% of argmax misses, missed 22-30% of the "
                  "time at every evidence strength) and weak anomalies at smaller separations (dchi2 < 2000: 19% of those "
                  "anomalies, 56% of their misses; miss rate 0.116 below 500 falling to 0.004 above 1e5).",
    "results-02": "Fixed. Guards added for every directional word (wide share of NonPSPL-to-PSPL > 50%, wide recall below the plane "
                  "median, the wide cells' support, the minimum cell below the support floor, stellar share and median q of the "
                  "detectable anomalies), with perturbation cases in tests/test_validation_and_invariants.py (two guards lacked "
                  "one until the sixth check added them).",
    "results-03": "Fixed. The plane paragraph quotes the two best-populated widest-separation cells at q >= 0.1 (0.69 and 0.715, "
                  "Nd 1,298 and 2,595) and calls the minimum a low-support cell (Nd 35); both computed in make_figures.py.",
    "results-04": "Fixed. A false negative is 'a detectable binary not classified as an anomaly, almost always called a single lens'; "
                  "the rate keeps the six-class argmax definition.",
    "results-05": "Fixed. Argmax misses are labelled as such, and the threshold view is added: 12% of detectable anomalies are "
                  "missed at the operating threshold, 36% of them wide.",
    "results-06": "Fixed. README and evaluation.md say 94.4% (final test, argmax, weighted, from figures_stats); "
                  "nonpspl_demoted_fraction is retired in canonical_numbers.json and the \\bmlNonpsplDemoted macro removed.",
    "results-07": "Fixed. The labelling block is computed on the final test split in make_figures.py and both passages say so.",
    "results-08": "Fixed. Both passages say 'stored events' and give the weighted fractions (8.2% keep the anomalous label, 79.9% "
                  "become single lenses) with a pointer to the subsampling in Sec. eval.",
    "results-09": "Fixed. The model card and evaluation.md list the in-distribution wide-separation failure; the CHANGELOG records "
                  "the Sec. results rewrite.",
    "results-10": "Fixed. The minimum cell's edges are printed with one decimal (-5.5 < log q < -5).",
    # RMDC26 cascade
    "cascade-optimistic-mixed-protocol": "Fixed. The full-window flag rate (4.8%, F146 only) is compared with the F146-only rate "
                  "over all scored single lenses (5.75%, from transfer_full_fspl5s_seasons_g08_f146only.json, now in MANIFEST "
                  "and --list-inputs); the guard compares like with like; REVISION row 10 corrected.",
    "cascade-stratum-directions-unresolved": "Fixed. The paragraph gives the in-house stratum sizes and states that detection is "
                  "lower in both strata, but that the intermediate-ratio difference rests on 48 in-house events and is not "
                  "resolved; guarded stratum by stratum.",
    "cascade-no-more-frequent-scope": "Fixed. 'About as frequent' is scoped to F146, the three-band clause says RMDC26's premature "
                  "rates are the higher ones there (counts given as macros), and REVISION row 10 is scoped the same way.",
    "cascade-data-availability-exception-incomplete": "Fixed. Data availability names three RMDC26 results (the partial-season "
                  "scan, the sub-day summary, the detectability relabelling) as regenerating only by rerunning on the pinned "
                  "release.",
    "cascade-nplanet-unguarded": "Fixed. make_gulls_macros.py requires the two planetary classes to have the same scanned count.",
    "cascade-strata-omit-stellar-q": "Fixed. The strata paragraph adds the 1% of eligible anomalies with q > 10^-2 "
                  "(\\bmlCgRmStellarFrac).",
    # appendix, Data availability, docs
    "appendix-regeneration-cause": "Fixed. Both passages give the environment as the cause (shards regenerated outside the "
                  "unpinned fleet environment reproduce 75% of events; a single-lens refit that sets a few labels differs, and "
                  "each changed label shifts the random stream) and say regeneration is close, not identical.",
    "appendix-referee-archive-cannot-rederive": "Fixed by disclosure. The referee round's overlap statistics read the work "
                  "directory; Data availability lists that as an exception and the runner's docstring says the archive alone "
                  "does not re-reduce the JSON. A --from-archive path is not implemented (left open).",
    "appendix-throughput-hours-wording": "Fixed. The rate is training events per logged epoch including the validation pass; "
                  "22 h includes one epoch stretched by a system sleep, about 17 h with a typical epoch in its place.",
    "appendix-sim-paragraph-provenance": "Fixed. canonical_numbers.json has v5-fleet fields (region, instance type, vCPUs) from the "
                  "launch scripts and the two macros read them; the text says 'mostly on transient (Spot) instances'.",
    "appendix-cost-xref-stale": "Fixed. The pointer now refers to the cloud-scale generation and local training described in "
                  "the appendix, whose cost was not recorded.",
    "appendix-loss-class-balancing-omitted": "Fixed. 'cross-entropy loss with class-balanced weights (the selection weights of "
                  "Sec. eval, rescaled so each class's mass is equal) and an additional factor 2 on the anomaly class'.",
    "appendix-recipe-numbers-unsourced": "Fixed by scoping. The rebuild statement now covers every data-derived number and names "
                  "the configuration constants, among them the appendix's training settings (from the stage logs and optimizer "
                  "states, outside the release), as typed in the scripts and canonical_numbers.json.",
    "appendix-stale-recipe-records": "Fixed. dispositions_2026_09_12b.py critic #1 and #2 corrected (peak 4e-4 from the optimizer "
                  "state; #1 marked superseded); the CHANGELOG and canonical comment say which values were train.py defaults.",
    "appendix-modelcard-stale-subset": "Fixed (255,481 events; a new realisation of the suite's populations).",
    "appendix-targeted-regimes-all-trained": "Fixed. 'all of them enriched pools also used in training (drawn afresh)' in README, "
                  "evaluation.md and the paper.",
    "appendix-evaluation-trained-on-1p9M": "Fixed. 'trained on up to 1.5 million events per stage (1.9 million in the final "
                  "stage's data including its validation and test splits)'.",
    "appendix-pipeline-md-pointer-missing": "Fixed. docs/pipeline.md points to the every-class scan.",
    "appendix-revision-page-count": "Fixed. REVISION.md gives the page count of the current build (30 pages at 2133a41).",
    "appendix-revision-row10-unscoped": "Fixed (scoped to F146; three-band rates given as the higher ones on RMDC26).",
    "appendix-revision-sec2-single-lens-label": "Fixed ('generated single lenses with a PSPL label').",
    "appendix-noblue-source-mislabelled": "Fixed. make_macros.py checks every hand-copied slice in canonical_numbers.json against "
                  "paper/results/metrics.json (recall and, for the no-blue-band slice, its count), fail-closed, so the "
                  "Data-availability statement that the slice is read from metrics.json holds.",
    # stress
    "stress-faint-precision-attribution": "Fixed. The faint tier's low precision is attributed to faint photometry: at the "
                  "natural population's class mix the same events give 0.152, against 0.726 in the natural tier, while the "
                  "natural tier at the faint sweep's mix gives 0.637 (reweighted_precision in stress_rescore_local.py, guarded). "
                  "The 0.440 prior shift and 'set mostly by the class mix' are gone from the paper and docs.",
    "stress-widesep-population-differs": "Fixed. Sec. limits, the Table tab:stress caption and the appendix state that the "
                  "regenerated wide-separation tier labels 27% fewer detectable anomalies than the set with the same code, and "
                  "that the other regenerated tiers agree within 4.5 SE (guarded).",
    "stress-abstract-detectable-qualifier": "Fixed. The abstract says '35-37% of detectable sub-day single lenses and 12% of "
                  "detectable faint microlensing events without an anomaly (0.85% in distribution)'; README follows.",
    "stress-rmdc26-matched-timescale-range": "Fixed. single_lens_above_frozen_tE_0p25_1 is computed in stress_rescore_local.py; "
                  "the RMDC26 comparison quotes the sweep's 0.25-1 d rate (33%) against RMDC26's 4.3%, and the guard uses it.",
    "stress-median-includes-macro-f1": "Fixed. The median excludes macro-F1 (0.008; maximum 0.060).",
    "stress-legacy-test-does-not-pin-july": "Fixed. tests/test_stress_legacy.py regenerates shard 0 of oor_pspl_shortte through "
                  "run_shard.main with both legacy switches and checks the July generator's label counts (slow-marked).",
    "stress-tier-stamps-backfilled": "Fixed. stamp() refuses to create a stamp for a tier that already has outputs; PROVENANCE and "
                  "the docstring say the four 6016bc6 stamps were reconstructed from the earlier artifact's record.",
    "stress-model-card-stale-subset-size": "Fixed (255,481).",
    "stress-intro-targeted-out-of-range": "Fixed ('an independent same-prior sample, targeted regimes and out-of-range sweeps').",
    "stress-docs-faint-regime-and-arm-labels": "Fixed in evaluation.md, README and CHANGELOG ('out-of-range sweeps'; faint sources "
                  "at m 23.5-25, below the sweep's 25-27.5, were training pools from stage 4).",
    "stress-demoted-share-subset-vs-suite": "Refuted by the adversarial check (the regenerated tier's share matches the suite's); "
                  "text kept, attributed to the tier.",
    # code
    "code-01": "Fixed with results-01 (the same sentence).",
    "code-02": "Fixed with cascade-optimistic-mixed-protocol (F146-only comparison, 4.8% against 5.75%).",
    "code-03": "Fixed. Guards added for the four sentences; 'mostly by the class mix' is gone (stress-faint-precision-attribution), "
                  "so its guard became the faint-photometry guard.",
    "code-04": "Fixed in part. Value cases added for most new directional guards in make_gulls_macros.py and make_macros.py; "
                  "tests/test_macros_fail_closed.py asserts each case's specific guard message (37 cases at 2133a41). The "
                  "sixth check found seven guards still without a case and added them.",
    "code-05": "Fixed with stress-tier-stamps-backfilled (stamps cannot be created for tiers with outputs; back-fill disclosed).",
    "code-06": "Fixed. legacy_oor_mix and legacy_t0_pad added to GEN_ATTRS.",
    "code-07": "Fixed with cascade-stratum-directions-unresolved.",
    "code-08": "Fixed. 'RMDC26's anomalies are planetary (only 1% of the eligible ones have q > 10^-2)'; the guard on the 1% was "
               "added by the sixth check.",
    "code-09": "Fixed. The docstring and text call the rates independent of the flat and variable-star share, not of the mix as "
                  "a whole.",
    "code-10": "Fixed. The per-event stress evaluations are archived (validation/stress_rescore_archive/, 14 files, 3.4 MB) with "
                  "--from-archive re-reduction and a test; Data availability states that the stage-5 weights are not "
                  "distributed.",
    "code-11": "Fixed. cascade_gulls.json records git describe and the reducer's sha256 (regenerated at 358fa7b; numbers unchanged).",
    # whole-paper read
    "read-01": "Fixed with stress-faint-precision-attribution (0.440 replaced by the natural-mix counterfactual 0.152).",
    "read-02": "Fixed with appendix-regeneration-cause and stress-widesep-population-differs (close, not identical; the fleet "
                  "deficits stated).",
    "read-03": "Fixed with results-01.",
    "read-04": "Fixed with cascade-optimistic-mixed-protocol.",
    "read-05": "Fixed ('we simulate millions of events (1.9 million in the final training stage alone)').",
    "read-06": "Fixed with cascade-stratum-directions-unresolved and cascade-no-more-frequent-scope.",
    "read-07": "Fixed with stress-median-includes-macro-f1.",
    "read-08": "Fixed with stress-widesep-population-differs.",
    "read-09": "Fixed. Data availability lists three exceptions and states that the stage-5 weights are not distributed.",
    "read-10": "Refuted by the adversarial check; the abstract gives the range 35-37% over the two sub-day tiers anyway.",
    "read-11": "Fixed with appendix-throughput-hours-wording.",
    "read-12": "Fixed with appendix-sim-paragraph-provenance.",
    "read-13": "Fixed. Sec. gulls:cascade uses mass-ratio names (high planet-like, intermediate) with q ranges.",
    "read-14": "Fixed. The Conclusion adds the binary-source degeneracy and sub-day coverage to the open tests and says 'mostly because'.",
    "read-15": "Fixed. The every-class stream paragraph discloses its threshold-selection rows (14.5%); the Sec. results sentence "
               "is scoped to final-test numbers.",
}

GAPS5 = []   # the critic stalled and returned nothing

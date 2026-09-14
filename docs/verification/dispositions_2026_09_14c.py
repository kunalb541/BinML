"""Dispositions of the seventh check (2026-09-14): paper, code and docs reviewers of the sixth check's fixes
(2133a41..52e2d8c), their adversarial checkers and a narrow completeness critic.

Read by docs/verification/build_doc_2026_09_12b.py, which appends them to docs/VERIFICATION_2026-09-12b.md as a fourth
addendum and asserts that every finding in 2026-09-14_seventh_findings.json has exactly one disposition."""

DISPO = {
    # paper
    "paper7-01": "Fixed. gap_matched_density.py has a full-cadence arm (0.942 on the same events); the text says empty bins "
                 "cause the collapse to zero, that thinning alone costs recall too, and that at the highest density the "
                 "regular grid even trails the random subsample; '(next paragraph)' dropped; guarded.",
    "paper7-02": "Fixed. The ablation paragraph says only what the 7.2 d grid records (undetectable at the previous cut, "
                 "detectable at the grid onset) and that the first-detectable onset can lie earlier still (7% of the in-house "
                 "binaries, a guarded macro from cascade_trace.npz, now hashed in MANIFEST); 'premature under any onset' and "
                 "'where the true onset lies' are gone from the paper, make_macros, the CHANGELOG and the guard message.",
    "paper7-03": "Fixed. The ECE is small because 81% of the weight sits below 0.1; 64% of it comes from 0.1-0.9, where the "
                 "score is over-confident (weighted 0.041 at 0.26; stored 0.16 at 0.29), more so under weighting; guarded.",
    "paper7-04": "Fixed ('nominally resolved at high ratios, p = 0.035, one of six comparisons here'), paper and draft.",
    "paper7-05": "Fixed. 'At the edge of resolution' is guarded by the bootstrap share at or below zero (1-10%); 'at the price "
                 "of later alerts' by the grid lags; both with cases.",
    "paper7-06": "Fixed. Data availability adds the schedule measurement and the dataset facts (rmdc26_schedule.py, "
                 "rmdc26_dataset_facts.py).",
    "paper7-07": "Fixed (\\bmlStressN in Data availability; the inference note in canonical_numbers.json rewritten).",
    # code
    "code7-01": "Fixed with paper7-02.",
    "code7-02": "Fixed with paper7-01.",
    "code7-03": "Fixed. Empty bins are counted with binml.preprocess.bin_band, exactly as the model bins (nightly minimum 59%, "
                "was 60%); the artifact was regenerated at a clean commit; a unit test covers the helper.",
    "code7-04": "Fixed with paper7-05.",
    "code7-05": "Fixed (1,015/s, about 17% below the earlier figure, in the docstring and comment; the canonical note updated).",
    "code7-06": "Fixed. espl_function, min_amplitude_mag and band_set are dated by the commits that first wrote them.",
    "code7-07": "Fixed. make_gulls_macros.py requires cascade_gulls.json's in-house strata to equal "
                "cascade_reproduce_result.json (with a case); unit tests cover alert_times, _vs_inhouse and _empty_bin_frac.",
    # docs
    "docs7-01": "Fixed. paper/SUBMISSION_FIELDS.txt (untracked) gives the current page count (32 with the new per-class figure).",
    "docs7-02": "Fixed (REVISION row 6: 37% over the corrected tier's full range, 35% in the legacy tier; disposition docs6-03 "
                "annotated).",
    "docs7-03": "Fixed (the inference note in canonical_numbers.json rewritten; MANIFEST hash updated).",
    "docs7-04": "Fixed with code7-05.",
    "docs7-05": "Fixed. PROVENANCE drops gap_matched_result.json from the list of artifacts without a checkpoint hash and says "
                "it now records the stage-6 hash and its code.",
    "docs7-06": "Fixed. The fifth-verification CHANGELOG entry is annotated at both lines (generator identity; nine RMDC26 "
                "results need the caches).",
    "docs7-07": "Fixed. README separates the RMDC26 tables (regenerate from a clone via rmdc26_scores.csv.gz) from the results "
                "that need a rerun on the pinned release.",
    "docs7-08": "Fixed. \\bmlInferMs is computed from the unrounded batch time (0.99 ms, not the double-rounded 0.98).",
}

GAPS7 = [
    "Checked fine by the critic (grid rows recomputed from the trace).",
    "Checked fine by the critic (wide-binary miss rates and the share called PSPL).",
    "Fixed: the low-q corner guard also requires the plane's minimum to lie at log q < -4.",
    "Checked fine by the critic (baselines mixture).",
    "Checked fine by the critic (July-shard test tolerance).",
    "Checked fine by the critic on the in-house side; the RMDC26 per-event lags need the GULLS curve cache.",
]

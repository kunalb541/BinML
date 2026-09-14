"""Dispositions of the eighth check (2026-09-15): three readers of the unreviewed commits 52e2d8c..e65a510 (the
seventh-round fixes, the 2026-09-14 length cut, the new abstract and Appendix B), a refuter per slice, and a narrow
critic. Read by docs/verification/build_doc_2026_09_12b.py, which appends them as a fifth addendum.

The main finding: the length cut had removed hedges that earlier rounds established and the data still require, and
the new abstract overclaimed. Every finding below was fixed in the text, the captions, the figure's self-checks or
the records; none was refuted."""

_HEDGE = "Fixed: the hedge the cut had dropped is restored in a few words"
DISPO = {
    # prose
    "prose8-01": "Fixed: 'weighted by event rate the physics resolves nothing' (the weighted false-alarm drop lies within the seed range).",
    "prose8-02": f"{_HEDGE} ('nominally resolved, one of six comparisons here'; 'at the edge of resolution').",
    "prose8-03": "Fixed: the measured pauses lead random gaps in recall in both weightings, in false alarms per event only; Sec. limits says 'in recall'.",
    "prose8-04": "Fixed: Data availability says the stress numbers re-reduce from their archive alone, while the floor, colour and stream results keep archived inputs but reduce from the runner's work directory; 'dataset facts' is back in the rerun list.",
    "prose8-05": "Fixed ('not established limits of the method').",
    "prose8-06": "Fixed: 'for most of them ... tens of points; the flat source stays near 1'; the caption restores the Flat sentence, '(generator grid)' and the PSPL condition; make_data_figures.py's self-checks now follow these sentences.",
    "prose8-07": "Fixed ('Essentially none').",
    "prose8-08": "Fixed: the relabel-off arm's values (bmlGapApRandNorelabel, bmlGapApCleanRandNorelabel), as the sentence describes.",
    "prose8-09": "Fixed: each baseline named with its score.",
    "prose8-10": "Fixed ('events with q < 10^-3').",
    "prose8-11": f"{_HEDGE} ('suggests ... (or to correlated source properties)').",
    "prose8-12": "Fixed ('in the regenerated tier').",
    "prose8-13": "Fixed: thinning costs recall and, at the highest density, the regular grid trails the random subsample; the 12-minute sentence says 'no difference we can resolve with one seed per arm' (the refuter found the original wording defensible).",
    "prose8-14": "Fixed: the floor arms are partly overlapping draws (shares as macros); the colour boundary depends on where training stops.",
    "prose8-15": "Fixed ('their most probable class being mostly PeriodicVar').",
    "prose8-16": "Fixed: the conclusion says F146 for the half-day numbers, three bands for the stream, and gives the F146-alone purity.",
    "prose8-17": "Fixed: 'the likely cause' and 'explains only part of the residual'; the colour loss spans the ablated checkpoints and the flag rates are the recommended one's; the slice scatter is 68% of 200 slices; the wing budget rests on few exceedances; the burden is slightly optimistic.",
    "prose8-18": "Fixed: the CHANGELOG no longer claims every result and macro was kept; REVISION.md notes where the cut moved statements to the draft; the record's fifth addendum lists the restored seventh-round hedges.",
    "prose8-19": "Fixed ('compared at full precision').",
    "prose8-20": "Fixed ('of the final-test split').",
    "prose8-21": "Fixed: the Discussion's between-season share is scoped to eligible planetary events; the appendix gives 22 h of wall-clock, about 17 with a typical epoch, and says the throughput range excludes the stretched epoch.",
    # front
    "front8-01": "Fixed: the abstract says RMDC26 guided the checkpoint choice, so its numbers are optimistic.",
    "front8-02": "Fixed: 'labels are conditioned on an adopted detectability policy, so a binary whose perturbation falls below its thresholds is labelled a single lens'.",
    "front8-03": "Fixed ('mostly because').",
    "front8-04": "Fixed ('Code, weights and artifacts are public').",
    "front8-05": "Fixed: the abstract no longer states a cadence (15 min is the legacy benchmark, 12 min the survey).",
    "front8-06": "Fixed: F146 for the half-day numbers and three bands for the stream, in the abstract and the conclusion; the conclusion's 1.6% is of eligible binaries.",
    "front8-07": "Fixed: Fig. 3's caption says the control acquires no appreciable NonPSPL probability, the events are selected illustrations and not evidence, the scan measures the first crossing, and the axis is days revealed.",
    "front8-08": "Fixed with prose8-06.",
    "front8-09": "Fixed with prose8-02 and prose8-17 (the three cascade hedges).",
    "front8-10": "Fixed: Table 8's caption names both earlier checkpoints and says the budgets are close to their rates.",
    "front8-11": "Fixed: Table 10's caption says the pauses row compares the schedule arms of Table 7 and that bold compares at full precision.",
    "front8-12": "Fixed with prose8-01.",
    "front8-13": "Fixed: Table 9 gives the fraction of single lenses over the threshold.",
    "front8-14": "Fixed: Fig. 9's caption states both blanking rules.",
    "front8-15": "Fixed: Table 6's caption restores the precision and macro-F1 exceptions for N.",
    "front8-16": "Fixed with prose8-08.",
    "front8-17": "Fixed: paper/README.md lists make_gulls_macros.py, render_draft.py and the committed outputs; REVISION.md and the draft header no longer call the draft the manuscript text; the CHANGELOG is corrected.",
    "front8-18": "Fixed: the Appendix B table's bin edges are asserted against the artifact, with a perturbation case.",
    "front8-19": "Fixed: the introduction says the reweighting partially corrects for how the training set was sampled.",
    # records
    "records8-01": "Fixed with prose8-18.",
    "records8-02": "Fixed with prose8-04.",
    "records8-03": "Fixed with prose8-01.",
    "records8-04": "Fixed with front8-01 and front8-03; the helper's abstract regenerated.",
    "records8-05": "Fixed with front8-05.",
    "records8-06": "Fixed with front8-02.",
    "records8-07": "Fixed: REVISION.md's status row and a dated note say which rows' statements the cut condensed (rows 6, 11, 20, 23).",
    "records8-08": "Fixed: Sec. gulls again says a NonPSPL call on a 2S2L event may respond to its second source (binary-source flag share as a macro), and Table 8's caption says which rows trained with the approximate relabel.",
    "records8-09": "Fixed: this addendum records which seventh-round fixes the cut condensed and that the eighth check restored them.",
    "records8-10": "Fixed with front8-11.",
    "records8-11": "Fixed with prose8-08.",
    "records8-12": "Fixed: the draft's transfer table has seven columns and the rho table of its own.",
    "records8-13": "Fixed: README says eight verification rounds; 'dataset facts' restored in the paper.",
    "records8-14": "Fixed with front8-17 ('truth-informed reference'; the committed outputs).",
    "records8-15": "Fixed: the helper's abstract regenerated (247 words), highlight 3 says eligible binaries (79 characters), the data field says close but not identical, and the article-type note no longer calls the exploratory comparison a claim.",
}
GAPS8 = [
    "Confirmed and fixed: the physics lowers false alarms further in the two largest rho/|u0| bins, but in the 1-3 bin the "
    "extra training carries most of the drop; the text no longer credits the physics with 'the gain' there.",
    "Confirmed and fixed: above 0.9 the text gives the weighted frequency (0.90) at the mean score (0.99) instead of saying they "
    "agree; the stored-event comparison (0.16 at 0.29) that the calibration guard checks is back, with the 64% mid-range share.",
    "Checked fine by the critic (ablation window counts and the 7% fine-onset share; trace hash in MANIFEST).",
    "Checked fine by the critic (matched-density artifact, full cadence, model binning); its nit (a guard without a sentence) is "
    "resolved because the regular-grid sentence is restored.",
    "Checked fine by the critic (the figure regenerates byte-identically; not added to CI's byte compare, since the simulation is "
    "not guaranteed bit-identical across platforms).",
    "Checked fine by the critic (clean-export regeneration of every generated output; attribute dates in pipeline/cache.py).",
]

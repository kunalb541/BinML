# BinML paper plan

Two papers: a **methods paper** (primary) framed around a Roman-ready deep-learning
microlensing classifier, and a short **JOSS companion** for the `binml` software.

Status: plan (2026-07-10). The current `paper/paper.md` is a placeholder; this plan drives
its rewrite.

---

## Paper A — Methods paper (primary)

**Target venue:** *Astronomy & Computing* (fits a methods+software contribution) or *AJ /
MNRAS* if positioned more as a survey-readiness study. A&C is the recommended first target.

**Working title (options):**
- *BinML: A Deep-Learning Classifier for Microlensing Light Curves in the Roman Era*
- *A Causal CNN–GRU for Real-Time Three-Class Microlensing Classification with the Nancy
  Grace Roman Space Telescope*

**Thesis / one-sentence pitch.** BinML is a compact, causal CNN–GRU that classifies
microlensing light curves (Flat / PSPL / Binary) at Roman cadence in real time, evaluated
with a detectability-honest methodology and validated on real OGLE events — a ready-to-use
tool for the Roman Galactic Bulge Time-Domain Survey.

### Section outline (what goes where → which result/figure)

1. **Introduction** — microlensing & exoplanet detection; the Roman bulge survey and its data
   volume; the classification/triage problem; prior ML work (MicroLIA random forest; other
   CNN/RNN light-curve classifiers); our contribution (compact causal net + detectability-honest
   evaluation + real-data validation). *No figure.*
2. **Simulations & data** — VBBinaryLensing engine; the three classes; Roman cadence (6912 pts,
   15 min, 72 d); the compact HDF5 format; the `anomaly_dchi2` detectability metric; the 10M
   simulation and detectability-aware subsets. *Fig: example simulated light curves per class;
   Fig: parameter distributions (q, tE, u0, anomaly_dchi2).*
3. **Model** — 2-channel input (magnification, Δt); compaction + lengths; causal DW-separable
   conv; unidirectional GRU; masked attention pooling; hierarchical head (+aux); why it is
   causal; leakage-safe design (train-only norm stats, masked pooling). *Fig: architecture
   schematic (NEEDED).* *Table: hyperparameters (131K params).*
4. **Training** — streaming single-GPU pipeline; hierarchical loss; AMP/cosine/early-stop;
   warm-start fine-tuning (`--init-weights`) targeting the low-q planetary regime; the
   round-1/round-2 lesson (targeted fine-tune > more base training; round-2 collapse). *Fig:
   ft_recall_before_after.png.*
5. **Evaluation methodology** — the detectability-conditioned framework: a binary with no
   detectable anomaly is observationally a PSPL, so report binary recall vs `anomaly_dchi2` +
   the indistinguishable fraction, not the raw population number; the temporal-bias KS test
   (effect size vs p-value at 1e6 events). *Fig: binary_recall_vs_detectability.png (KEY).*
6. **Results** — full 1M held-out evaluation: per-class recall/precision, per-q binary recall,
   detectable-only recall; calibration (ECE); ROC; confusion matrix; probability-evolution /
   early-detection. *Figs: eval_1M_binary_recall_by_q.png, binary_recall_by_preset.png,
   confusion_matrix, roc_curves, calibration, early_detection_curve.*
7. **Real-data validation** — OGLE-IV EWS events run through the model: clean PSPL, caustic-
   crossing binaries (OGLE-2014-BLG-0289, 2013-BLG-0578), the super-Earth
   OGLE-2017-BLG-0482Lb (residual bump); the domain shift; the fine-tuned model recovering
   missed binaries. *Figs: evolution_OGLE-2014-BLG-0289.png, planetary_bump_real_vs_synth.png,
   real-OGLE grid.*
8. **Cadence study** — detection vs characterization as a function of cadence; Roman vs
   LSST-like sampling; why dense cadence is required to characterize (not just detect)
   binaries. *Fig: lsst_cadence_degradation.png.*
9. **Discussion** — implications for Roman triage/alerts; limitations (sim-to-real gap;
   single-band; achromatic assumption); future work (the v5 multi-band, multi-class extension —
   cite the design doc). *No figure.*
10. **Software & reproducibility** — the `binml` package (pip, bundled weights, CLI,
    detectability-conditioned evaluate); code + data release. *No figure.*
11. **Conclusions.**

### Figures — have vs. need
- **Have** (in `docs/figures/` + `results/`): recall-by-detectability, 1M-recall-by-q,
  recall-by-preset, ft before/after, LSST degradation, planetary bump real-vs-synth,
  real-event evolution; plus confusion/ROC/calibration/early-detection from the eval run.
- **Need to make:**
  - **Architecture schematic** (the CNN–GRU pipeline) — essential.
  - **Example simulated light curves** per class (clean, publication-quality).
  - **Parameter-distribution** panel for the simulations.
  - Optionally: a **real-OGLE gallery** figure (curated from the grids we made).

### Gaps to close before submission (honest — reviewers will ask)
1. **Baseline comparison (CRITICAL).** Compare BinML to a simple baseline — a random-forest on
   hand-features (à la MicroLIA) and/or a plain CNN — on the same 1M test set, to show the
   causal CNN–GRU earns its complexity. *We do not have this yet; it is the top priority.*
2. **Ablations.** Attention vs masked-mean pooling; hierarchical vs flat head; with/without the
   Δt channel; d_model/n_layers scaling. A small ablation table.
3. **Statistical rigor.** Bootstrap CIs on the headline metrics (we have some); report them.
4. **Larger real-data test.** Expand from ~a dozen curated OGLE events to a systematic sample
   (e.g. all well-sampled events in a season) for an aggregate real-data number, with the
   caveat that ground truth (published solutions) is only available for a subset.
5. **Reproducibility artifact.** Freeze a data release + weights on **Zenodo** for a DOI;
   pin the exact commit.

---

## Paper B — JOSS companion (software)

Short (~1000 words). Rewrite `paper/paper.md` (the current draft is close but was auto-drafted
and is a placeholder). Sections: Summary; Statement of need (contrast with MicroLIA; the 3-class
framing + detectability-honest evaluation as the software's distinguishing features);
Functionality (Classifier API, CLI, survey loaders, detectability-conditioned `evaluate`);
a minimal usage example; Availability; Acknowledgements. Requires the repo to have: tests (✓),
CI (✓), docs (✓), an example (✓), a LICENSE (✓), and a review-ready README (✓). JOSS wants the
software archived (Zenodo DOI) — shared prerequisite with Paper A.

---

## Shared prerequisites
- **Author list & affiliations** — confirm co-authors / advisor (currently only K. Bhatia,
  Heidelberg). *User decision.*
- **Zenodo archive** of code + weights + a data release → DOI (needed by both papers).
- **Data availability statement** — where the simulated data / trained weights live.
- Decide **which trained model is "the" model** in the paper (base vs fine-tuned; report both,
  ship fine-tuned).

## Execution plan (ordered)

| step | task | owner | blocks |
|---|---|---|---|
| 1 | **Baseline comparison** — random-forest (features) + plain-CNN on the 1M test; results table | me (needs compute) | Results section |
| 2 | **Ablation table** — pooling / head / Δt / size | me (compute) | Method/Results |
| 3 | **Make missing figures** — architecture schematic, example curves, param distributions | me | Figures |
| 4 | **Bootstrap CIs** on headline metrics | me | Results |
| 5 | (optional) **Systematic real-OGLE sample** + aggregate number | me | Real-data section |
| 6 | **Draft methods paper** (LaTeX, `paper/methods/`) section by section, dropping in figures | me + user review | — |
| 7 | **Rewrite JOSS `paper.md`** | me | — |
| 8 | **Zenodo release** + DOI; finalize author list & data statement | user | submission |
| 9 | Internal review → revise → submit | user | — |

**Estimated compute:** steps 1–2 (baselines + ablations) need a few GPU-hours (Modal top-up).
Steps 3–7 are local/free. Figures and both drafts can be produced now; the baseline and
ablation results are the only hard prerequisites that need running before the Results section
is final.

## Thesis feedback (round 2) — must carry into the paper

The MSc thesis examiner's round-2 comments (34 substantive notes) map directly onto paper
actions. The paper must not repeat these weaknesses:

1. **Motivation first.** Open with the *science* (exoplanet demographics, the Roman bulge
   survey) before the method (p5).
2. **Physics correctness** — get these exactly right and cite sources:
   - caustics are in the **source plane**, critical curves on the **lens/image plane**
     (Schneider, Ehlers & Falco); Roman covers the **bulge**, not the whole sky (p16);
   - binary caustic topology + **close / intermediate / wide** regimes (Tsapras 2018) (p25);
   - `α` = angle between source trajectory and the binary axis; high magnification arises
     from the trajectory approaching a caustic (p26);
   - **redo or replace the caustic figure** — the thesis one "does not show caustics"; either
     compute properly (Bozza 2010, Green's theorem/contour integration) or borrow-with-citation
     (p26, p28);
   - re-check the KMTNet-vs-OGLE resolution/precision claim (p34).
3. **Cite thoroughly.** The thesis was under-referenced. Required: Paczyński; Schneider &
   Weiss 1986; Witt 1990; Griest & Safizadeh 1998; Bozza 2010; Tsapras 2016 & 2018;
   Dominik 2007 & 2008 and Tsapras 2009 (alert systems); MACHO & EROS surveys.
4. **No overclaiming.** Drop/soften unsupported claims (e.g. "precision sufficient to
   constrain planetary formation scenarios") unless directly defensible (p27).
5. **tE distribution (ML-relevant!).** The examiner notes the real `tE` distribution peaks
   ~25 d and long-timescale events were missed (Tsapras 2016) — **verify the simulator's `tE`
   sampling matches reality**; if biased, note it as a limitation or re-simulate. This affects
   the training distribution and must be addressed in the Data section.
6. **Reproducibility.** State where training/sims ran (Modal L4; local GPU), compute cost, any
   restrictions, and link the public repo — exactly what the examiner asked for (p52).
7. **Define abbreviations at first use**; consider a glossary/notation table (p17).

## What I can start immediately (no compute, no blockers)
- The **architecture schematic** and **example-light-curve / parameter-distribution** figures.
- A **detailed section-by-section methods-paper skeleton** in LaTeX (`paper/methods/`) with the
  figures and the results we already have slotted in, leaving `TODO` placeholders for the
  baseline/ablation numbers.
- The **JOSS `paper.md`** rewrite.

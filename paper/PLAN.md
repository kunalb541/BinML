# BinML paper plan (1.0 — 6-class, real-time cascade)

Status: 2026-07-28. Drives the rewrite of `paper.md`.

> **Public framing.** Externally this is **BinML 1.0**. Internal development numbering
> (training stages, simulator iterations) is not exposed in the paper, docs, or public code —
> where an ablation is scientifically useful it is framed as **"baseline vs cascade model,"**
> not by internal stage names.

Two outputs: a **methods paper** (primary) and a short **JOSS software note** for the package.

---

## 0. Roadmap (where the project goes next)

1. **Write the methods paper** (this plan) — the science is at a defensible milestone.
2. **Real Roman / OGLE-IV validation** — run BinML on real events; this is what turns
   "works on sims" into "works," and is the trigger for a targeted retrain.
3. **Targeted retrain (conditional)** — only if real data shows the faint-noise false-binary
   weak spot is a practical problem. Otherwise deferred (rare-corner physics limits).

---

## Paper A — Methods paper (primary)

**Target venue:** *Astronomy & Computing* (methods+software) first choice; *AJ*/*MNRAS* if
positioned as a survey-readiness study.

**Working title:** *BinML: A Real-Time, Detectability-Conditioned Deep-Learning Classifier for
the Roman Galactic Bulge Time-Domain Survey.*

**One-sentence pitch.** BinML is a compact multi-band conv-stem+transformer that classifies
Roman light curves into six physically-meaningful classes and — uniquely — only flags a class
once its evidence is *observable*, so it never triggers a false planet alert before the caustic
has been seen.

### Three contributions (the spine of the paper)
1. **Detectability-conditioned labelling** — label by what is observable, not by generator
   intent. Removes the label noise that penalises a model for not seeing what isn't there.
2. **The real-time cascade** — truncation labelling keyed on a per-binary anomaly-onset time,
   yielding Flat→PSPL→NonPSPL as evidence arrives. Premature binary flagging 42%→9%.
3. **Honest evaluation at survey scale** — completeness at fixed purity as the headline;
   selection-aware reweighting; a 12.9M-event unseen-parameter stress test with documented
   failure modes.

### Section outline → result / figure

| § | content | figure / number |
|---|---|---|
| 1 Introduction | microlensing & exoplanets; Roman GBTDS data volume; the real-time triage problem; prior ML (§refs); our 3 contributions | — |
| 2 Simulations | 6 classes, 3 bands, Roman cadence; priors; VBBinaryLensing; **detectability-conditioned labelling** | Fig: example light curves, class distributions |
| 3 Model | conv stem + min/max carry lanes + transformer; 505k params; why compact | schematic — **NEW, to draw** |
| 4 The cascade | anomaly-onset time; truncation relabelling; Flat→PSPL→NonPSPL | Fig: probability evolution, early detection, + the 42%→9% table |
| 5 Evaluation methodology | completeness@purity; selection reweighting; detectability conditioning | Fig: ROC/PR + operating point, calibration |
| 6 Results | per-class; confusion; (log s, log q) efficiency; parameter dependence | Fig: confusion, per-class, param dependence, efficiency planes, slices |
| 7 Generalisation & limits | 12.9M stress test; OOR failure modes; the cascade ablation | Fig: failure analysis + stress-test table + matched-purity table |
| 8 Discussion | operating-point choice for follow-up; comparison to classical Δχ²; real-data outlook | Fig: baseline overlay |
| 9 Conclusion | — | — |

### Figures — inventory & gaps

**Have (14 diagnostic figures + 300 per-event evolution plots, archived in the results set):**
confusion, ROC/PR+operating, calibration, per-class, parameter-dependence, slices,
failure-analysis, efficiency-planes, light-curves, training, class-distributions,
temporal-bias, probability-evolution, early-detection.

**Need to make (NEW):**
- **F1. Model schematic** — bands → conv stem (min/max lanes) → 156 tokens → transformer → 6-way
  head.
- **F2. The cascade hero figure** — 2–3 per-event panels (a clean binary, an ambiguous one, a
  faint) showing Flat→PSPL→NonPSPL with the commit-time marker. The paper's signature figure.
- **F3. Cascade summary** — pre-onset P(NonPSPL) and premature-flag rate vs day observed,
  **baseline vs cascade model** (from the all-class temporal scan). Quantifies contribution #2.
- **T1. Training-curriculum table** — completeness@purity across the fine-tuning curriculum
  (framed as capability added, not stage numbers).
- **T2. Matched-purity table** — NonPSPL completeness at purity {0.90…0.99}, baseline vs
  cascade (already computed; shows the crossover).
- **T3. Stress-test / OOR table** — per-class on 12.9M unseen; OOR failure modes.

### Headline numbers (locked)
- completeness@purity 0.879; AP 0.952; per-class F1 [0.97, 0.96, 0.82, 0.97, 0.91, 0.88] (population/argmax; NonPSPL F1 precision-limited by detectability-floor demotion, recall 0.95)
- cascade: premature NonPSPL flag 42%→9%; pre-onset P(NonPSPL) 0.411→0.033
- NonPSPL→PSPL (missed planet) 0.055→0.048; stress test 12.9M events, macroF1 0.927

---

## Paper B — JOSS software note (short)

The `binml` package + the v1 pipeline. ~1000 words: statement of need (Roman triage),
functionality, the detectability-honest evaluation, links to docs. Cite the methods paper.

---

## References to cite (researched 2026-07-28; grouped by what they justify)

**Roman GBTDS survey & yields** (§1, §2 — motivation, cadence, expected planet counts)
- **Penny et al. 2019, ApJS 241, 3** — "Predictions of the WFIRST Microlensing Survey I: Bound
  Planet Detection Rates" (~1400 bound planets). Survey-design + cadence reference.
  2019ApJS..241....3P
- **Johnson et al. 2020, AJ 160, 123** — Roman Galactic Exoplanet Survey II: free-floating
  planet detection rates. arXiv:2006.10760
- **Terry et al. 2023, ApJS** — transiting exoplanet yields for the Roman GBTDS from pixel-level
  simulations (establishes the survey's cadence/photometry). 10.3847/1538-4365/acf3df

**Event rates & parameter priors** (§2 — justify the simulation priors)
- **Mróz et al. 2019, ApJS 244, 29** — optical depth & event rate from 8 yr OGLE-IV; the tE
  distribution (peak ~10–40 d, ±3 power-law tails). Our tE prior. arXiv:1906.02210
- **Suzuki et al. 2016, ApJ 833, 145** — planet mass-ratio function; break at q≈1.7×10⁻⁴.
  Our q prior and the "planetary" regime boundary.
- **Sumi et al. 2023** (MOA-II 9-yr free-floating planet MF, arXiv:2303.08280) — low-mass /
  short-tE population context.

**Binary-lens modelling** (§2 — the NonPSPL generator)
- **Bozza 2010, MNRAS 408, 2188** — advanced contour-integration algorithm.
- **Bozza et al. 2018, MNRAS 479, 5157** — VBBinaryLensing public package (what we use).
  arXiv:1805.05653

**Prior ML classification of microlensing** (§1 — position our contribution)
- **Godines et al. 2019, A&C 28** — LIA / MicroLIA: Random Forest with ~50 features on OGLE-II,
  tested on ZTF/Palomar. The reference "classical ML" approach.
- **Mróz 2020, Acta Astron. 70, 169** — "Identifying Microlensing Events Using Neural Networks."
  arXiv:2008.11930
- **Classifying High-cadence Microlensing Light Curves I: Defining Features, 2021, AJ**
  (10.3847/1538-3881/abd6cc) — Roman/high-cadence-specific ML; closest prior art on cadence.
- **Early recognition of Microlensing Events from Archival Photometry with ML, 2022**
  (arXiv:2201.12209) — early/real-time detection; closest prior art to our cascade angle.

**Deep learning for astronomical time series** (§3 — architecture context)
- **Donoso-Oliva et al. 2023, A&A (ASTROMER)** — transformer embeddings for single-band
  astronomical time series. Our transformer lineage.
- **Astro-MoE / multi-band transformers, 2024–2025** (arXiv:2507.12611; A&A multi-band
  vision-transformer 2025) — positions our 3-band design.
- A CNN/RNN light-curve classifier (e.g. Naul et al. 2018 RNN; or the general light-curve
  classification framework 10.3847/1538-4365/ad62fd) as the pre-transformer baseline.

**Bulge variable/contaminant populations** (§2 — the non-microlensing classes)
- **Soszyński et al. (OGLE) LPV catalogs** — Mira / SRV / OSARG classification and P–L relations
  (the LongPeriodVar generator + the OSARG small-amplitude impostor).
- **Soszyński et al. (OGLE) RR Lyrae / eclipsing-binary catalogs** — the PeriodicVar generator.
- **Identifying low-amplitude pulsating stars through microlensing observations, 2021**
  (arXiv:2108.08650) — LPV-vs-microlensing confusion; motivates LongPeriodVar as the dangerous
  impostor.

> **Action:** pull exact bibkeys/DOIs into `paper.bib` (several already present); verify each
> arXiv ID and journal ref before submission.

---

## Immediate next actions
1. Rewrite `paper.md` from this outline.
2. Make the 3 NEW figures (schematic, cascade hero, cascade summary) + the 3 tables.
3. Reconcile `paper.bib` with the reference list above.
4. Draft the JOSS note.

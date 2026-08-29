# Revision plan — items to fold into the next submission

**Status 2026-08-29:** Astronomy & Computing desk-rejected without peer review
(ASCOM-D-26-00349), citing author status ("independent researchers without an established
publication record") — not any scientific criticism. The science is unreviewed. This plan now
targets the resubmission venue; everything below folds into that version rather than a
response-to-referees.

Manuscript as submitted to A&C: commit `08337ed`, 2026-08-15. Nothing below
changes a submitted number. Each item is either complete (artifacts committed, ready to write up)
or deferred (needs compute). Paper macros are regenerated from `paper/results/`; new numbers
must enter through `make_macros.py`, never typed.

---

## 1. Gap sensitivity and the first cross-simulator validation — COMPLETE, ready to write up

**Status:** all artifacts committed (`9b921c5`, `ecd5178`). Shipped weights unchanged.

### What to say

The submitted manuscript describes the training schedule as "legacy-like" and notes that the
current GBTDS definition samples F146 every ~12 min with colour visits on a 6-h cycle. It does
not say that the real schedule has *gaps*, and it does not test them. It should, because the
shipped model fails on them.

**The finding.** The RMDC26 release from the Roman Galactic Exoplanet Survey PIT (GULLS
simulator, `huggingface.co/datasets/RGES-PIT/MachineLearning`, revision `a338d5ba`) implements
the GBTDS schedule as planned, in which F146 pauses for ~6.2 h seven times per 70.7-day season.
BinML's training grid (`pipeline/assemble._epochs`) is continuous; the only way an epoch is lost
is SNR < 3 or saturation, and across 1,800 sampled training events in every class none contains
an empty F146 bin. The model's sole prior for an empty mid-season token is the unrevealed future
of a truncated season, so it reads a gap as evidence against a clean single lens.

Inserting RMDC26's seven gaps into in-distribution events (n = 100 per class,
`validation/gulls/gap_sensitivity.json`):

| condition | PSPL | NonPSPL | Flat | PeriodicVar |
|---|---|---|---|---|
| no gaps | 0.930 | 0.980 | 1.000 | 0.990 |
| 1 gap × 1 h | 0.860 | 0.980 | 1.000 | 0.990 |
| 1 gap × 2 h | 0.860 | 0.990 | 1.000 | 0.990 |
| 1 gap × 4 h | 0.650 | 0.990 | 0.600 | 0.990 |
| 1 gap × 6 h | 0.090 | 1.000 | 0.810 | 0.990 |
| RMDC26 schedule (7 × 6.2 h) | 0.110 | 0.970 | 0.080 | 1.000 |

Recall, argmax. Gaps ≤ 2 h are nearly harmless. The lost PSPL and Flat events go to NonPSPL and
PeriodicVar; NonPSPL and PeriodicVar recall are unaffected. This is a property of the input
contract, not of the physics, and it is reproduced with no GULLS data at all.

**The remedy.** `pipeline/train.py --gap-aug` blanks 1–8 contiguous runs of 1–12 h in every
band and relabels on the same rule as the existing truncation augmentation (caustic inside a gap
→ PSPL; nothing detectable left → Flat). The existing `--cadence-aug` thins bins at random,
which is the sparse-ground-survey regime, not this one. A warm-start fine-tune from the shipped
weights on 45k natural-prior events (`validation/modal_gap_finetune.py`, 12 epochs, p = 0.8,
lr 1e-4, checkpoint `validation/gulls/weights/ft_g08e12.pt`), scored on 14,958 held-out events
(`validation/gulls/gap_finetune_g08e12.json`):

| held-out macro-F1 | shipped | fine-tuned |
|---|---|---|
| clean | 0.920 | 0.907 |
| RMDC26 schedule | 0.384 | 0.879 |
| random 1–8 gaps | 0.454 | 0.886 |

Under the RMDC26 schedule PSPL recall goes 0.078 → 0.885 and Flat 0.071 → 0.975. The cost is
1.3 macro-F1 points on clean data, mostly PeriodicVar precision (0.956 → 0.926).

**Cross-simulator transfer.** With the fine-tuned checkpoint BinML can, for the first time, be
scored on an independent simulator. Selection: amplitude ≥ 0.1 mag from the metadata (GULLS
simulates the whole population; its median 1S1L peak is 0.063 mag), t_E in BinML's training
support [1, 300] d (RMDC26 1S1L is 33% sub-day, its binary classes 0.2%, so an uncut comparison
conflates timescale with lens multiplicity), one contiguous id block per class (a 250-id block
spans 105 of 129 fields), baseline measured empirically from the off-event flux. 1,286 dense
events, identical set for both models (`validation/gulls/transfer_*.json`):

| RMDC26 class | truth | n | PSPL | NonPSPL | PeriodicVar | ≥ 0.9042 |
|---|---|---|---|---|---|---|
| 1S1L single lens | PSPL | 279 | 0.03 → 0.66 | 0.69 → 0.31 | 0.27 → 0.00 | 0.52 → 0.07 |
| 1S2L planet | NonPSPL | 527 | 0.00 → 0.31 | 0.77 → 0.69 | 0.23 → 0.00 | 0.58 → 0.37 |
| 2S2L planet + binary source | NonPSPL | 480 | 0.00 → 0.29 | 0.81 → 0.70 | 0.19 → 0.00 | 0.68 → 0.36 |

Shipped → fine-tuned, argmax fractions; last column is the fraction over the frozen threshold.
Single-lens false alarms at the frozen threshold fall from 52% to 7% and the PeriodicVar
contamination disappears. Planetary over-threshold recall of 0.37 is the first meaningful such
number: the shipped 0.58 was obtained while flagging single lenses at 52%. The threshold was
calibrated on gap-free data and is open to re-tuning for this regime.

### What NOT to claim

- 2S2L is **not** the binary-source (1L2S) contaminant. Every 2S2L event carries a planetary
  lens (median q 1.25e-4, same as 1S2L); the binary source is an extra complication in 56%.
  RMDC26 ships no 2S1L class, so the Gaudi (1998) degeneracy raised in §discussion is still
  untested. Say so.
- GULLS planets are harder than ours at matched amplitude (median q 1.25e-4; the amplitude cut
  keeps faint perturbations). Do not read 0.37 against the in-distribution 0.879.
- The catalogue baseline `Source_F146 + 2.5 log10(fs_F146)` is uniformly 0.471 mag brighter
  than the quiescent flux in this release. We did not use it. Mention in a footnote only if a
  referee asks how the baseline was obtained.

### Where it goes

- **§Validation / limitations:** replace the "legacy-like schedule" paragraph with the gap
  table and the statement that the shipped checkpoint requires continuous F146.
- **New short subsection, cross-simulator transfer:** the GULLS table, with the selection
  stated. This directly answers the standing objection that all validation uses our own
  simulator.
- **Abstract:** one sentence. "On an independent simulator (GULLS/RMDC26) a gap-aware
  fine-tune reduces single-lens false alarms at the operating threshold from 52% to 7%."
- **Model card / README:** input contract now states "continuous F146; for Roman's planned
  schedule use the gap-aware checkpoint."
- **Decide:** whether `ft_g08e12.pt` becomes the shipped weights. If yes, every headline number
  is regenerated from it and the clean-data cost (−1.3 macro-F1) is reported. If no, it ships
  alongside as `binml-gapaware.pt`. Either is defensible; the second is less work and keeps the
  submitted numbers exact.

### Reproduce

```
python validation/gulls/gap_sensitivity.py --n 100
modal run --detach validation/modal_gap_finetune.py --epochs 12 --gap-aug 0.8 --lr 1e-4 --tag g08e12
python validation/gulls_transfer.py --per-class 600 --chunk 250
python validation/gulls_transfer.py --per-class 600 --chunk 250 --weights validation/gulls/weights/ft_g08e12.pt
```

The transfer runs locally in ~5 min each; the dataset must not be read from Modal (HF
rate-limits the datacenter IP). Revision is pinned in the script.

---

## 2. Deferred items from the pre-submission referee round — need compute

Scored 7.5/10 Major Revision (likely accept). Each needs re-simulation or retraining, deliberately
not rushed before submission.

- **Sensitivity to the 0.02 mag detectability floor.** Needs re-simulation; anomaly amplitude
  is not stored in the frozen artifact so it cannot be done by reduction. Sweep 0.01 / 0.02 /
  0.05 on one shard, report how the NonPSPL prevalence and headline completeness move.
- **Colour-band calibration ablation.** `ROMAN_BANDS_AUDITED` already holds corrected F087/F213
  zeropoints; retrain on one shard with each and compare. GPU time only.
- **Mixed-class sequential evaluation.** The streaming scan holds eligible binaries only, so it
  measures timing, not streaming purity or alert burden. Run the cascade on a natural-prior mix
  and report alerts per 1,000 events per day alongside the timing numbers.
- **Seed sweep** (3 seeds, stage-5 recipe) for the headline completeness/purity.

## 3. Administrative

- Zenodo DOI to replace the bare GitHub URL (before acceptance).
- arXiv posting once an endorser is found; SSRN preprint (10.2139/ssrn.7295158) stands meanwhile.
- Convert AASTeX → elsarticle only if the editor asks.

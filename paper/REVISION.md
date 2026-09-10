# Revision plan — items to fold into the next submission

**Status 2026-09-09 (evening):** the desk rejection was REVERSED after a formal complaint; A&C will
peer-review the manuscript (third reviewer added) once we resubmit the revised version. The editor
asked that the latest results be incorporated before review ("Please incorporate the latest results
of your models before re-submission", 2026-09-07). No deadline was stated; review begins when we
resubmit. The paper is under active consideration at A&C and may not be submitted elsewhere.

**Where each piece stands — artifact vs manuscript.** "Artifact" means computed, committed and
manifest-hashed; "In paper" means written into `paper/paper.tex`. Nothing in the second column is
done until this table says so.

| Item | Artifact | In paper |
|---|---|---|
| Gap-sensitivity table (7 × 6.2 h schedule; single-gap sweep) | ✅ `validation/gulls/gap_sensitivity.json` | ❌ not yet — replaces the "legacy-like schedule" paragraph |
| Gap-aware fine-tune g08e12 (held-out macro-F1 0.384 → 0.879 under the schedule) | ✅ `validation/gulls/gap_finetune_g08e12.json`, weights `validation/gulls/weights/ft_g08e12.pt` | ❌ not yet |
| GULLS/RMDC26 full-population transfer (56,975 matched dense events; 35.6% → 11.7% false alarms; recall at threshold 0.50 → 0.46) | ✅ `validation/gulls/transfer_full_{shipped,ft_g08e12,reduced}.json` (+ `_rawpool` variants) | ❌ not yet — new cross-simulator subsection |
| Preprocessing-sensitivity note (epoch-pooled vs raw-pooled binning, ~10 points) | ✅ both variants committed | ❌ not yet — one paragraph in the subsection |
| Abstract sentence for the above | draft below | ❌ not yet |
| Model card / README input contract ("continuous F146; gap-aware checkpoint for the planned schedule") | — | ✅ README + model card carry the limitation note; contract line still to sharpen |
| Fitted-PSPL baseline rescored at full cadence (0.261 → 0.545) | ✅ `validation/baselines_result.json`, canonical, manifest | ✅ via `\bmlBasePspl` + prose |
| Cadence experiment re-evaluated held-out (AP 0.827 vs 0.830) | ✅ `validation/cadence_result.json`, canonical, manifest | ✅ paragraph rewritten |
| OOR swept-class support (e.g. widesep NonPSPL n = 438) | ✅ read from `stress_report.json` | ✅ via `\bmlStress*N` |
| F087 saturation physics corrected | ✅ `photometry.py` | ✅ paragraph corrected |
| `t_anom` 7.2-day resolution of training labels | — | ✅ stated in §training |
| McNemar discordant counts as macros | ✅ | ✅ |
| Finite-source fine-tune `ft_fspl_g08.pt` (GULLS false alarms 11.7% → 5.2%; recall at matched budget +6–7 pts; re-calibrated threshold 0.935 → 3.1% FA) | ✅ `validation/gulls/{fspl_finetune,transfer_full_reduced,transfer_tradeoff,gapped_threshold}_fspl_g08.json` | ❌ not yet — one paragraph + matched-budget table |
| Round 2 `ft_fspl5_g08.pt` (rho ≤ 5 + binary rho ≤ 0.1): negative — no gain over round 1, finite-source bins worse | ✅ `fspl_finetune_fspl5_g08.json`, `transfer_tradeoff_all.json` | — (one sentence at most) |
| Round 3 `ft_fspl5s_g08.pt` (single-lens rho ≤ 5, binaries unchanged): best on every GULLS measure — FA 4.9%, recall 0.390 @ 5.2% FA, mean 0.538 | ✅ `fspl_finetune_fspl5s_g08.json`, `transfer_tradeoff_all.json` | ❌ not yet — the matched-budget table |
| Colour ablation on GULLS (both checkpoints): colour adds 4–7 pts planetary recall at matched budget | ✅ `transfer_colour_ablation.json` | ❌ not yet — one sentence |
| Decision: sidecar = `ft_fspl5s_g08.pt` (threshold from `gapped_threshold_fspl5s_g08.json`) | **OPEN — author's call** | — |
| Figures rebuilt through `build.sh` (weighted prevalence line) | ❌ not yet | — |
| Zenodo release + DOI in Data Availability, CITATION.cff, README | ❌ needs one-time GitHub↔Zenodo authorisation by the author | — |
| Resubmit via Editorial Manager (starts review) | — | ❌ after the rows above |

The GULLS work itself is finished: the curve cache (~4 GB at
`~/Desktop/Research/microlensing/gulls_curve_cache`, outside the repo) re-scores the whole population
with any checkpoint in ~6 minutes, so threshold re-tuning or a new checkpoint is cheap. Seven follow-up
experiments were proposed on 2026-09-09; three are done and four are open — see the ledger in §1½.
The Beginner/Experienced challenge tiers were set aside (different format, low information for this paper).

**Audit pass 2026-09-09** (`docs/AUDIT_2026-09-09.md`): two paper numbers changed — fitted-PSPL baseline AP
0.261 → 0.545 (rescored at full cadence) and the cadence experiment re-evaluated on held-out events
(AP 0.827 vs 0.830; the old protocol scored training data). Both are already in the paper via macros.

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

Recall, argmax. Gaps ≤ 2 h are nearly harmless (a 0.5 h gap, not shown, costs nothing). The lost
PSPL and Flat events go to NonPSPL and PeriodicVar; NonPSPL and PeriodicVar recall are unaffected.
Flat is NOT monotonic in gap length (1.00 → 0.60 at 4 h → 0.81 at 6 h); every condition scores
the same 100 events per class (the harness re-seeds per condition), so this is model behaviour,
not sampling noise — do not describe the degradation as monotonic. This is a property of the input
contract, not of the physics, and it is reproduced with no GULLS data at all.

**The remedy.** `pipeline/train.py --gap-aug` blanks 1–8 contiguous runs of 1–12 h in every
band and relabels. **Audit correction (2026-09-09) — describe what g08e12 actually ran with, not
the intended rule:** the "caustic inside a gap → PSPL" test used `t_anom`, which the simulator
resolves only to 7.2-day steps (the END of the first 7.2-d interval in which the anomaly became
detectable), so it tested a 6-hour slot up to 7.2 d after the real caustic — effectively an
arbitrary slot, relabelling ~1–2% of NonPSPL events to PSPL at random rather than the ones whose
caustic was hidden. The "nothing detectable left → Flat" branch compares a NOISY max against the
0.02 mag floor and never fires for m ≳ 21. Neither defect touches any reported metric (both act
on training labels only), and the g08e12 result stands as measured, but the manuscript must say
the fine-tune's relabelling was approximate. Fixing the rule (finer onset resolution; noise-aware
floor) changes `--gap-aug` semantics relative to the released checkpoint, so do it under a
version flag or retrain, not silently. The existing `--cadence-aug` thins bins at random,
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
1.3 macro-F1 points on clean data. Per class the F1 losses are Eruptive −0.025 (precision
0.829 → 0.790), **NonPSPL −0.021 (precision 0.685 → 0.661 — the science class gets less pure)**,
PeriodicVar −0.019 (precision 0.956 → 0.926), PSPL −0.009, Flat −0.004, LongPeriodVar −0.002.
Say this plainly; an earlier draft attributed the cost "mostly" to PeriodicVar, which the
artifact does not support. Recall/precision in gap_finetune_g08e12.json are keep_prob-WEIGHTED
population estimates; the `n` next to them is the raw unweighted support.

**Cross-simulator transfer (full population, 2026-09-09).** With the fine-tuned checkpoint BinML
can, for the first time, be scored on an independent simulator. Selection: amplitude ≥ 0.1 mag
from the metadata (GULLS simulates the whole population; its median 1S1L peak is 0.063 mag), t_E
in BinML's training support [1, 300] d (RMDC26 1S1L is 33% sub-day, its binary classes 0.2%, so
an uncut comparison conflates timescale with lens multiplicity), **every** eligible event
(100,935 requested; 25,871 skipped for t0 in an inter-season gap, no usable F146, or too few
off-event epochs for a baseline; 18,089 fell in low-cadence seasons and are reported separately),
baseline measured empirically from the off-event flux. **56,975 dense events**, identical set
and bit-identical inputs for both models (`validation/gulls/transfer_full_*.json`, reduced by
`validation/gulls/transfer_reduce.py`):

| RMDC26 class | truth | n | PSPL | NonPSPL | PeriodicVar | ≥ 0.9042 | weighted ≥ 0.9042 |
|---|---|---|---|---|---|---|---|
| 1S1L single lens | PSPL | 33,353 | 0.03 → 0.54 | 0.53 → 0.43 | 0.44 → 0.01 | **0.356 → 0.117** | 0.383 → 0.135 |
| 1S2L planet | NonPSPL | 11,388 | 0.00 → 0.17 | 0.64 → 0.83 | 0.36 → 0.00 | 0.504 → 0.457 | 0.474 → 0.512 |
| 2S2L planet + binary source | NonPSPL | 12,234 | 0.00 → 0.16 | 0.71 → 0.83 | 0.29 → 0.00 | 0.582 → 0.469 | 0.505 → 0.523 |

Shipped → fine-tuned, argmax fractions; the two right-hand columns are the fraction over the
frozen threshold, unweighted and GULLS `final_weight`-weighted (Wilson 95% intervals are ±0.5
points or better at these n; they are in the reduced JSON). Single-lens false alarms at the
frozen threshold fall by a factor of three (35.6% → 11.7%) while planetary recall at threshold
is nearly preserved (0.50 → 0.46; 0.58 → 0.47), and the PeriodicVar contamination disappears.
The fine-tune also lifts argmax anomaly recall on genuine binaries from 0.64/0.71 to 0.83. The
threshold was calibrated on gap-free data and is open to re-tuning for this regime.

**Preprocessing sensitivity (state it).** RMDC26 samples F146 every 12.1 min against BinML's
15-min epoch grid, so a quarter of the observations share an epoch. `binml.preprocess` now
reduces to one value per epoch before binning, which reproduces the training-cache
representation exactly (bit-identical on-grid). Binning the same rows by raw observation
instead gives 45.6% → 9.7% on 1S1L and 0.36/0.37 planetary recall at threshold
(`transfer_full_*_rawpool.json`): the operating point moves by ~10 points with the pooling rule,
because the per-bin min/max channels carry the noise footprint. The epoch-pooled numbers are the
ones to report, with this sensitivity disclosed.

### Where the gap-aware model's residual GULLS false alarms come from (2026-09-09, from the reduced rows)

Joining the scored 1S1L rows with the RMDC26 metadata (`rho`, `piE`, `u0lens1`) gives a clean
mechanism for the remaining 11.7%:

| rho / \|u0\| (finite-source strength) | n | false-alarm rate, g08e12 |
|---|---|---|
| < 0.03 | 18,151 | 0.046 |
| 0.03–0.1 | 8,722 | 0.100 |
| 0.1–0.3 | 3,931 | 0.229 |
| 0.3–1 | 1,743 | 0.429 |
| 1–3 | 539 | 0.646 |
| > 3 | 267 | 0.674 |

Monotonic over a factor of 15, and it persists at fixed \|u0\| < 0.1 (0.33 → 0.65 across the same
bins), so it is the source-size effect, not high magnification per se. Parallax is irrelevant:
at \|u0\| ≥ 0.3 the rate is 0.033–0.043 across all \|piE\| bins. **Cause:** `PSPLGen` is
point-source and has no parallax, while `NonPSPLGen` samples rho ∈ [1e-4, 1e-2] — so in the training
set a rounded, flattened peak only ever belonged to a binary, and the model learned "finite-source
rounding ⇒ NonPSPL". GULLS single lenses with rho/\|u0\| ≳ 0.3 (7.6% of the eligible 1S1L
population, concentrated at high magnification and bright baselines) are then flagged. This is a
training-set physics gap, fixable by adding finite-source single lenses (VBBinaryLensing `ESPLMag2`
is available) and fine-tuning; the curve cache re-scores GULLS in ~6 min. For the shipped
checkpoint the gap effect swamps this (0.40 → 0.26 across the same bins, i.e. no finite-source
signal visible).

**Finite-source single lenses — RESULT (2026-09-09 night).** `priors.PSPL_FINITE_SOURCE` (opt-in;
released training set unchanged), `generators.espl_magnification` (VBBinaryLensing ESPLMag2, rho
log-uniform in [1e-3, 1] to cover GULLS' single-lens rho, median 0.012 / p90 0.60), regimes `fspl`
and `fspl_highmag` (U0_MAX 0.2, PSPL-heavy mix with substantial NonPSPL so "small u0 ⇒ PSPL" cannot be
learned as a shortcut), runner `validation/fspl_finetune_local.py`. Warm-start from ft_g08e12, same
gap augmentation, 12 epochs, 129,683 training events (12 `fspl` + 4 `fspl_highmag` shards), ~35 min on
the M5. Checkpoint `validation/gulls/weights/ft_fspl_g08.pt`; results in
`validation/gulls/fspl_finetune_fspl_g08.json`, `transfer_full_{fspl_g08,reduced_fspl_g08}.json`,
`transfer_tradeoff_fspl_g08.json`.

*Did it learn the physics?* Held-out PSPL recall on our own finite-source population (39,864 events,
disjoint seeds), by rho/|u0|: shipped / g08e12 / new = 0.86 / 0.80 / **0.91** at 0.3–1, 0.16 / 0.16 /
**0.62** at 1–3, 0.08 / 0.09 / **0.80** above 3. Macro-F1 0.885 / 0.870 / **0.909**; no other class
lost (Flat 0.972, PeriodicVar 0.975, LPV 0.894, Eruptive 0.898). Caveat: this population is the new
model's own training distribution and out-of-distribution for the other two, so their AP here (~0.53
vs 0.93) is not a regression figure — the like-for-like test is GULLS.

*Did it transfer?* GULLS, identical 56,975 matched events, frozen threshold 0.9042:

| RMDC26 class | n | g08e12 → **fspl_g08** at threshold | weighted |
|---|---|---|---|
| 1S1L single lens (false alarms) | 33,353 | 0.117 → **0.052** | 0.135 → 0.080 |
| 1S2L planet (recall) | 11,388 | 0.457 → 0.370 | 0.512 → 0.442 |
| 2S2L planet + binary source (recall) | 12,234 | 0.469 → 0.400 | 0.523 → 0.459 |

The rho/|u0| false-alarm bins collapsed as predicted: 0.046 → 0.032, 0.100 → 0.036, 0.229 → 0.066,
0.429 → 0.184, 0.646 → 0.327, 0.674 → 0.333 — halved everywhere, though the largest-source bins are
not yet at the floor (GULLS rho reaches 5; the prior stops at 1).

*Threshold shift or real gain?* Recall at **matched** single-lens false-alarm budgets
(`transfer_tradeoff_fspl_g08.json`) — the frozen threshold simply lands the new model at a lower
false-alarm rate:

| 1S1L false-alarm budget | shipped | g08e12 | **fspl_g08** |
|---|---|---|---|
| 2% | 0.122 | 0.185 | **0.249** |
| 5.2% | 0.214 | 0.295 | **0.369** |
| 11.7% | 0.313 | 0.457 | **0.518** |
| 20% | 0.393 | 0.601 | **0.621** |
| mean 1S2L recall over FA ≤ 30% | 0.322 | 0.486 | **0.523** |

So at the operating point the paper actually uses (whatever false-alarm rate one accepts), the
finite-source fine-tune recovers 6–7 more points of planetary recall than g08e12 on an independent
simulator. The frozen 0.9042 threshold was calibrated on gap-free, point-source data and should be
re-calibrated on our own gapped finite-source simulations before either fine-tuned checkpoint is
used as a sidecar; that is a minutes-long job from the curve cache.

*Threshold re-calibration (`validation/gulls/calibrate_gapped_threshold.py`, results
`gapped_threshold_{fspl_g08,ft_g08e12}.json`).* The threshold is chosen the way the paper chooses it
— at the 0.90 purity target on OUR held-out simulations — but with finite-source single lenses in the
population and the RMDC26 seven-gap schedule blanked in; GULLS is only read afterwards to report the
consequence. Two results:

| checkpoint | threshold @ purity 0.90 (gapped held-out) | our held-out compl. @ purity | GULLS FA / recall 1S2L / 2S2L |
|---|---|---|---|
| ft_g08e12 | 0.9992 (saturated) | 0.001 @ 0.42 | 0.000 / 0.000 / 0.000 |
| ft_fspl_g08 | 0.9353 | 0.717 @ 0.882 | 0.031 / 0.296 / 0.329 |
| ft_fspl_g08, clean (no gaps) | 0.9289 | 0.817 @ 0.887 | 0.035 / 0.312 / 0.348 |
| (frozen 0.9042, for reference) | — | — | 0.052 / 0.370 / 0.400 |

For g08e12 the target is **unreachable**: once finite-source single lenses exist in the population its
NonPSPL precision cannot get near 0.90 at any threshold, so the search runs to 0.999 and completeness
collapses. That is the operational meaning of the rho/|u0| table — not "a few extra false alarms" but
"the paper's operating point does not exist". For the finite-source checkpoint the principled
threshold is a little stricter than the frozen one and lands GULLS at ~3% single-lens false alarms
with ~0.30–0.33 planetary recall at threshold; the matched-budget curve above remains the primary
comparison, this row is the operating point one would actually ship.

*Round 2 — GULLS-matched source sizes for both classes (`fspl5`: single-lens rho ≤ 5, binary rho ≤ 0.1;
same recipe, warm-start from ft_g08e12; checkpoint `ft_fspl5_g08.pt`; `fspl_finetune_fspl5_g08.json`,
`transfer_full_reduced_fspl5_g08{,_vs_fspl_g08}.json`, `transfer_tradeoff_all.json`).* A clean NEGATIVE
result. On GULLS at the frozen threshold: FA 0.087, recall 0.446 / 0.471 — between g08e12 and round 1.
The finite-source bins are worse than round 1 (rho/|u0| 0.3–1: 0.184 → 0.355; 1–3: 0.327 → 0.486;
>3: 0.333 → 0.464) though still better than g08e12; at matched false-alarm budgets it trails round 1
at low budgets (FA 2%: 0.216 vs 0.249) and ties it at high ones (FA 11.7%: 0.513 vs 0.518; mean recall
over FA ≤ 30%: 0.528 vs 0.523). Own held-out physics check was as good as round 1 (PSPL recall 0.58 /
0.83 in the 1–3 / >3 bins), so the model learned our extended population; it just transferred less
cleanly. Most plausible reading: widening the BINARY rho to 0.1 smooths caustic features toward the
finite-source single-lens shapes, so the two classes overlap more and the round-1 discrimination is
partly lost — the combined change confounds this with the single-lens extension, so a PSPL-rho-only
run (≈1 h) is the way to settle it. Until then **ft_fspl_g08 remains the sidecar candidate.**
Single seed per arm throughout: the between-round differences in the small high-rho bins (n = 267–539)
are several times the binomial error but seed-to-seed variance of the fine-tune is unmeasured.

*Colour ablation on GULLS (2026-09-10; `gulls_transfer.py --bands F146`, identical 56,975 matched
events).* For ft_fspl_g08, removing F087/F213 lowers single-lens false alarms slightly at the frozen
threshold (0.052 → 0.048) but costs planetary recall everywhere: at a matched 5.2% false-alarm budget
1S2L recall 0.369 → 0.302 and 2S2L 0.399 → 0.334; mean 1S2L recall over FA ≤ 30% 0.523 → 0.481.
Per event the colour bands move P(NonPSPL) by a median 0.11 (p90 0.44) and flip 9.5% of threshold
decisions. So GULLS' colour photometry — with its own blending fractions and zeropoints, none of which
we corrected — still carries usable anomaly signal for a model trained on our colour model. This is the
first cross-simulator evidence for the paper's three-band design; state it as such, with the caveat
that the colour channels were also the ones flagged as mis-calibrated in §limits. (g08e12 half and
the artifact `transfer_colour_ablation.json` to follow.)

*Round 3 — attribution (`fspl5s`: single-lens rho ≤ 5, binaries UNCHANGED; `ft_fspl5s_g08.pt`;
`fspl_finetune_fspl5s_g08.json`, `transfer_full_reduced_fspl5s_g08{,_vs_fspl_g08}.json`,
`transfer_tradeoff_all.json`).* Settled: the single-lens extension helps and round 2's regression was
entirely the binary-rho widening. On the identical 56,975 GULLS events (`gulls_summary_tables.py`):

| checkpoint | FA @ frozen | 1S1L FA by rho/\|u0\| bin (<0.03 … >3) | recall @ 5.2% FA (1S2L) | mean 1S2L recall, FA ≤ 0.3 |
|---|---|---|---|---|
| shipped | 0.356 | 0.40 0.31 0.30 0.26 0.27 0.26 | 0.214 | 0.322 |
| ft_g08e12 | 0.117 | 0.046 0.100 0.229 0.429 0.646 0.674 | 0.295 | 0.486 |
| ft_fspl_g08 (rho ≤ 1) | 0.052 | 0.032 0.036 0.066 0.184 0.327 0.333 | 0.369 | 0.523 |
| ft_fspl5_g08 (rho ≤ 5 + binary rho ≤ 0.1) | 0.087 | 0.040 0.061 0.160 0.355 0.486 0.464 | 0.344 | 0.528 |
| **ft_fspl5s_g08 (rho ≤ 5, binaries unchanged)** | **0.049** | **0.031 0.037 0.060 0.154 0.280 0.288** | **0.390** | **0.538** |

Own held-out physics check for fspl5s: PSPL recall 0.66 / 0.84 in the 1–3 / >3 bins (g08e12: 0.20 /
0.16), macro-F1 0.906, no other class regressed. **ft_fspl5s_g08 is the sidecar candidate.** The
remaining 0.28–0.29 in the largest-source bins is now the floor to chase with something other than the
prior (limb darkening, or the binary-source population, which we do not simulate at all).

*Gapped-threshold calibration for fspl5s (`gapped_threshold_fspl5s_g08.json`).* Purity-0.90 threshold on
our gapped finite-source held-out set: 0.9492 (held-out completeness
0.693 @ purity 0.897); on GULLS that gives
2.0% single-lens false alarms with planetary recall
0.268 / 0.295
(clean-data threshold 0.9236 → 3.7% /
0.335). This is the operating point to ship with the sidecar:
two percent single-lens false alarms on an independent simulator, chosen without looking at it.

*Colour ablation, both checkpoints (`transfer_colour_ablation.json`).* Removing F087/F213 costs
planetary recall at matched false-alarm budget for both: fspl_g08 0.369/0.399 → 0.302/0.334 at 5.2%
FA (mean recall 0.523 → 0.481); g08e12 0.295/0.313 → 0.253/0.271 (0.486 → 0.429). Per event the colour
bands move P(NonPSPL) by a median 0.11 and flip ~9–10% of threshold decisions. GULLS' colour photometry,
with its own blending and zeropoints, carries usable anomaly signal for a model trained on our colour
model: the first cross-simulator evidence for the three-band design.

**How to present it.** One paragraph plus the matched-budget table in the cross-simulator
subsection: the residual false alarms were traced to a missing physical effect in the training set,
the effect was added, the false alarms halved and the recall-at-budget curve moved up. That is the
cleanest possible demonstration of the paper's thesis that the labels and the training population,
not the architecture, are the lever. Do not claim the gap is closed (the >1 bins sit at 0.33), and
state that the fine-tune used the pre-fix `--gap-aug` relabelling semantics like g08e12.

### What NOT to claim

- 2S2L is **not** the binary-source (1L2S) contaminant. Every 2S2L event carries a planetary
  lens (median q 1.25e-4, same as 1S2L); the binary source is an extra complication in 56%.
  RMDC26 ships no 2S1L class, so the Gaudi (1998) degeneracy raised in §discussion is still
  untested. Say so.
- GULLS planets are harder than ours at matched amplitude (median q 1.25e-4; the amplitude cut
  keeps faint perturbations). Do not read 0.46 against the in-distribution 0.879.
- The catalogue baseline `Source_F146 + 2.5 log10(fs_F146)` is uniformly 0.471 mag brighter
  than the quiescent flux in this release. We did not use it. Mention in a footnote only if a
  referee asks how the baseline was obtained.

### Where it goes

- **§Validation / limitations:** replace the "legacy-like schedule" paragraph with the gap
  table and the statement that the shipped checkpoint requires continuous F146.
- **New short subsection, cross-simulator transfer:** the GULLS table, with the selection
  stated. This directly answers the standing objection that all validation uses our own
  simulator.
- **Abstract:** one sentence. "On an independent simulator (GULLS/RMDC26, 56,975 matched
  events) a gap-aware fine-tune cuts single-lens false alarms at the operating threshold from
  36% to 12% while preserving planetary recall at threshold (0.50 → 0.46)."
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

## 1½. Follow-up experiment ledger — the seven proposed 2026-09-09 (kept current)

Seven post-revision experiments were proposed once the full-population GULLS numbers were final.
Each row records the prediction made BEFORE running it, what actually came out, the artifact, and
what remains. Three are done, four are open. Costs are wall-clock on the M5 (10 cores, MPS).

| # | Experiment | Prediction (made first) | Outcome | Artifact | Status |
|---|---|---|---|---|---|
| 1 | Finite-source single lenses in the generator (VBBinaryLensing `ESPLMag2`), PSPL given a ρ prior, warm-start from g08e12 with gap augmentation, re-score GULLS from the cache | 1S1L false alarms 11.7% → ~5–6%; the ρ/\|u₀\| > 0.3 bins collapse toward the 4.6% floor | **Total hit, bins half-right.** FA 11.7% → 5.2% (round 1, ρ ≤ 1) → **4.9%** (round 3 `fspl5s`, single-lens ρ ≤ 5, binaries unchanged). Bins (0.3–1 / 1–3 / >3): 0.43/0.65/0.67 → **0.15/0.28/0.29** — halved to thirded, not at the floor. Planetary recall at the matched 5.2% budget 0.295 → **0.390**; mean recall over FA ≤ 0.3 0.486 → 0.538. Round 2 (also widening the *binary* ρ prior to 0.1) was negative; round 3 attributes the gain cleanly to the single-lens extension. Three rounds took ~2 h each including two disk-stall restarts. | `fspl_finetune_{fspl_g08,fspl5_g08,fspl5s_g08}.json`, `transfer_full_{fspl_g08,fspl5_g08,fspl5s_g08}.json`, `transfer_tradeoff_all.json`, weights `weights/ft_fspl*_g08.pt`; `tests/test_fspl.py` | ✅ done (3 rounds) |
| 2 | Schedule-matched augmentation: fine-tune with the exact seven fixed ~6.2 h pauses at 12.1-min cadence in a 70.7-d season (the `arm12` tree from the cadence rerun has the 12-min config) and compare to g08e12's random 1–12 h gaps | not stated | — | — | ❌ open, ~1 h. Note the *evaluation* side is already schedule-exact: `calibrate_gapped_threshold.py` blanks the seven-gap schedule into held-out memmaps; only the training side still uses random gaps. `gap_sensitivity.json` (held-out macro-F1 0.879 under the exact schedule after random-gap training) suggests random gaps suffice; this experiment would settle it. |
| 3 | Detectability-conditioned relabelling of GULLS itself: apply our Δχ² + 0.02 mag floor policy to the noise-free `true_flux_uJy` of GULLS binaries and score recall against detectability labels instead of generator labels | the 1S2L "recall 0.46" counts undetectable planets as misses; this gives the first like-for-like number | — | — | ❌ open. Needs a true_flux re-extraction (the curve cache holds `flux_uJy` only): same DuckDB/httpfs block reader with the extra column, ~20 min for 5k/class; also needs the per-epoch flux errors (check the obs schema for an error column) to form Δχ². Highest remaining scientific value: the paper currently says reconciling the two label ontologies is out of scope. |
| 4 | Threshold recalibration for the gapped regime on OUR gapped finite-source held-out simulations (never on GULLS) | not stated | **g08e12 cannot reach purity 0.90 on a finite-source population**: its threshold saturates at 0.999 with completeness 0.001 — the gap-aware checkpoint has no valid operating point there. `fspl_g08`: clean 0.929 / gapped 0.935 → GULLS FA 3.5% / 3.1%, recall 0.312 / 0.296 (1S2L). **`fspl5s_g08`: clean 0.924 → FA 3.7%, recall 0.335 / 0.369; gapped 0.949 → FA 2.0%, recall 0.268 / 0.295** (held-out completeness 0.693 at purity 0.897). | `gapped_threshold_{ft_g08e12,fspl_g08,fspl5s_g08}.json` | ✅ done |
| 5 | Noise-model matching: GULLS is 3–4× noisier at the faint end than our photometry; a noise-multiplier augmentation, fine-tune, re-score | not stated | — | — | ❌ open, ~1 h. Requires a new augmentation flag in `pipeline/train.py`; because the augmentation relabelling is under audit (findings 8–10) this should go in behind a version flag, not by editing `--gap-aug`. |
| 6 | Sub-day-t_E single lenses (the FFP-like third of RMDC26 1S1L removed by the t_E ∈ [1, 300] d support cut) as an explicit out-of-support row | not stated | — | — | ❌ open, ~30 min on a sample. Not in the curve cache (the cut was applied at selection), so ~5k events must be re-extracted (~0.05–0.09 s/event) and scored with all five checkpoints. |
| 7 | Colour ablation on GULLS: F146-only vs three-band from the cache | tells whether GULLS' colour bands help or hurt in transfer | **Colour helps for both checkpoints.** Recall at the matched 5.2% budget: `fspl_g08` 0.302 → 0.369 (1S2L), 0.334 → 0.399 (2S2L); g08e12 0.253 → 0.295, 0.271 → 0.313. Mean recall over FA ≤ 0.3: +0.042 / +0.057. About 9.4% of frozen-threshold decisions flip between the two inputs. | `transfer_colour_ablation.json`, `transfer_full_reduced_*_colour_ablation.json` | ✅ done for `fspl_g08` and `ft_g08e12`; `fspl5s_g08` F146-only not yet scored (12 min) |

Open items in recommended order: 3 (changes what "recall" means in the paper), 6 (cheap, Roman will find these), 2, 5. None is needed for the resubmission; each is one paragraph if done.

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

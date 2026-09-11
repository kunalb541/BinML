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
| Gap-sensitivity table (7 × 6.2 h first-season pauses; single-gap sweep; the measured per-season schedule is §1½ rows 2 and 16) | ✅ `validation/gulls/gap_sensitivity.json` | ❌ not yet — replaces the "legacy-like schedule" paragraph |
| Gap-aware fine-tune g08e12 (held-out macro-F1 0.384 → 0.879 under the first-season pauses) | ✅ `validation/gulls/gap_finetune_g08e12.json`, weights `validation/gulls/weights/ft_g08e12.pt` | ❌ not yet |
| GULLS/RMDC26 full-population transfer (56,975 matched dense events; 35.6% → 11.6% false alarms; recall at threshold 0.50 → 0.46) | ✅ `validation/gulls/transfer_full_{shipped,ft_g08e12,reduced}.json` (+ `_rawpool` variants) | ❌ not yet — new cross-simulator subsection |
| Preprocessing-sensitivity note (epoch-pooled vs raw-pooled binning, ~10 points) | ✅ both variants committed | ❌ not yet — one paragraph in the subsection |
| Abstract sentence for the above | draft below | ❌ not yet |
| Model card / README input contract ("continuous F146; gap-aware checkpoint for the planned schedule") | — | ✅ README + model card carry the limitation note; contract line still to sharpen |
| Fitted-PSPL baseline rescored at full cadence (0.261 → 0.545) | ✅ `validation/baselines_result.json`, canonical, manifest | ✅ via `\bmlBasePspl` + prose |
| Cadence experiment re-evaluated held-out (AP 0.827 vs 0.830) | ✅ `validation/cadence_result.json`, canonical, manifest | ✅ paragraph rewritten |
| OOR swept-class support (e.g. widesep NonPSPL n = 438) | ✅ read from `stress_report.json` | ✅ via `\bmlStress*N` |
| F087 saturation physics corrected | ✅ `photometry.py` | ✅ paragraph corrected |
| `t_anom` 7.2-day resolution of training labels | — | ✅ stated in §training |
| McNemar discordant counts as macros | ✅ | ✅ |
| Finite-source fine-tune `ft_fspl_g08.pt` (GULLS false alarms 11.6% → 5.2%; recall at matched budget +6–7 pts; threshold under the measured seasons 0.945 → 2.4% FA; the 0.935 first-season calibration is superseded) | ✅ `validation/gulls/{fspl_finetune,transfer_full_reduced,transfer_tradeoff,gapped_threshold}_fspl_g08.json` | ❌ not yet — one paragraph + matched-budget table |
| Round 2 `ft_fspl5_g08.pt` (rho ≤ 5 + binary rho ≤ 0.1): negative — no gain over round 1, finite-source bins worse | ✅ `fspl_finetune_fspl5_g08.json`, `transfer_tradeoff_all.json` | — (one sentence at most) |
| Round 3 `ft_fspl5s_g08.pt` (single-lens rho ≤ 5, binaries unchanged): best at matched budgets, tied by the combined arm (§1½ row 14) — FA 4.8%, recall 0.390 @ 5.2% FA, mean 0.538 | ✅ `fspl_finetune_fspl5s_g08.json`, `transfer_tradeoff_all.json` | ❌ not yet — the matched-budget table |
| Colour ablation on GULLS (three checkpoints): removing colour costs 4–7 pts planetary recall at matched budget; the reading that colour carries anomaly signal is withdrawn (§1½ row 7) | ✅ `transfer_colour_ablation.json` | ❌ not yet — one sentence |
| Decision: gap-aware checkpoint = `ft_fspl5s_g08.pt` at 0.957 (`gapped_threshold_fspl5s_g08_seasons.json`) | **Recommended** (combined arm tied at matched budgets; the smooth-magnification rerun is indistinguishable, §1½ row 14) — author confirms at release, as `binml-gapaware.pt` | draft only |
| Figures rebuilt through `build.sh` (weighted prevalence line) | ❌ not yet | — |
| Zenodo release + DOI in Data Availability, CITATION.cff, README | ❌ needs one-time GitHub↔Zenodo authorisation by the author | — |
| Resubmit via Editorial Manager (starts review) | — | ❌ after the rows above |

The GULLS work itself is finished: the curve cache (~4 GB at
`~/Desktop/Research/microlensing/gulls_curve_cache`, outside the repo) re-scores the whole population
with any checkpoint in ~6 minutes, so threshold re-tuning or a new checkpoint is cheap. The follow-up
experiments and the 2026-09-11 verification's controls are the seventeen rows of the ledger in §1½.
The Beginner/Experienced challenge tiers were set aside (different format, low information for this paper).

**Audit pass 2026-09-09** (`docs/AUDIT_2026-09-09.md`): two paper numbers changed — fitted-PSPL baseline AP
0.261 → 0.545 (rescored at full cadence) and the cadence experiment re-evaluated on held-out events
(AP 0.827 vs 0.830; the old protocol scored training data). Both are already in the paper via macros.

Manuscript as submitted to A&C: commit `08337ed`, 2026-08-15. Nothing below
changes a submitted number. Each item is either complete (artifacts committed, ready to write up)
or deferred (needs compute). Paper macros are regenerated from `paper/results/`; new numbers
must enter through `make_macros.py`, never typed.

---

## 1. Gap sensitivity and the first cross-simulator validation — historical record (2026-08-23 to 09-10)

**Read this first.** This section records the work as it was done. The 2026-09-11 verification
(`docs/VERIFICATION_2026-09-11.md`: 182 findings, 61 issues) corrected several of its statements;
the corrections are made in place below where they are numbers, and where this section conflicts
with section 1½ or the verification record, those win. In particular: RMDC26's pause schedule
differs between seasons (the "seven pauses" below are the first season's), the calibrations below
were redone under the measured seasons, and the manuscript text lives in `paper/draft_gulls_section.tex`.

**Status:** all artifacts committed (`9b921c5`, `ecd5178`). Shipped weights unchanged.

### What to say

The submitted manuscript describes the training schedule as "legacy-like" and notes that the
current GBTDS definition samples F146 every ~12 min with colour visits on a 6-h cycle. It does
not say that the real schedule has *gaps*, and it does not test them. It should, because the
shipped model fails on them.

**The finding.** The RMDC26 release from the Roman Galactic Exoplanet Survey PIT (GULLS
simulator, `huggingface.co/datasets/RGES-PIT/MachineLearning`, revision `a338d5ba`) implements
its own implementation of the GBTDS schedule, in which F146 pauses for 43-44 h per 70.7-day
season, in six or seven pauses whose phases differ between its six high-cadence seasons
(`validation/gulls/rmdc26_schedule.json`; the seven ~6.2 h pauses used in this section are the first
season's).
BinML's training grid (`pipeline/assemble._epochs`) is continuous; the only way an epoch is lost
is SNR < 3 or saturation, and across 1,800 sampled training events in every class none contains
an empty F146 bin (in a full natural-prior pool, 6 of 89,919). The model's sole prior for an empty mid-season token is the unrevealed future
of a truncated season, so it reads a gap as evidence against a clean single lens.

Inserting the first RMDC26 season's seven pauses into in-distribution events (n = 100 per class,
`validation/gulls/gap_sensitivity.json`):

| condition | PSPL | NonPSPL | Flat | PeriodicVar |
|---|---|---|---|---|
| no gaps | 0.930 | 0.980 | 1.000 | 0.990 |
| 1 gap × 1 h | 0.860 | 0.980 | 1.000 | 0.990 |
| 1 gap × 2 h | 0.860 | 0.990 | 1.000 | 0.990 |
| 1 gap × 4 h | 0.650 | 0.990 | 0.600 | 0.990 |
| 1 gap × 6 h | 0.090 | 1.000 | 0.810 | 0.990 |
| first RMDC26 season's pauses (7 × 6.2 h) | 0.110 | 0.970 | 0.080 | 1.000 |

Recall, argmax. Gaps ≤ 2 h are nearly harmless (a 0.5 h gap, not shown, costs nothing). The lost
PSPL events go to NonPSPL and PeriodicVar, Flat mostly to PeriodicVar; in this four-class test NonPSPL
and PeriodicVar recall are unaffected. (On the 30,013-event six-class held-out, Eruptive,
LongPeriodVar and NonPSPL also degrade and NonPSPL precision collapses; section 1½ row 2.)
Flat is NOT monotonic in gap length (1.00 → 0.60 at 4 h → 0.81 at 6 h); every condition scores
the same 100 events per class (the harness re-seeds per condition), so this is model behaviour,
not sampling noise — do not describe the degradation as monotonic. This is a property of the input
contract, not of the physics, and it is reproduced with no GULLS data at all.

**The remedy.** `pipeline/train.py --gap-aug` blanks 1–8 contiguous runs of 1–12 h in every
band and relabels. **Audit correction (2026-09-09) — describe what g08e12 actually ran with, not
the intended rule:** the "caustic inside a gap → PSPL" test used `t_anom`, which the simulator
resolves only to 7.2-day steps (the END of the first 7.2-d interval in which the anomaly became
detectable), so it tested a 6-hour slot up to 7.2 d after the real caustic — effectively an
arbitrary slot, relabelling about 0.4% of NonPSPL training presentations to PSPL essentially at
random (measured 2026-09-11) rather than the ones whose caustic was hidden. The "nothing detectable left → Flat" branch compares a NOISY max against the
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
in BinML's training support [1, 300] d (33% of all RMDC26 1S1L are sub-day, 0.3% / 0.15% of the planetary classes, so
an uncut comparison conflates timescale with lens multiplicity), **every** eligible event
(100,935 requested; 25,871 skipped: 22,170 with t0 between seasons, 3,617 before the first or after
the last season, 84 without usable F146 or enough off-event epochs for a baseline; 18,089 fell in
low-cadence seasons and are reported separately),
baseline measured empirically from the off-event flux. **56,975 dense events**, identical set
and bit-identical inputs for both models (`validation/gulls/transfer_full_*.json`, reduced by
`validation/gulls/transfer_reduce.py`):

| RMDC26 class | truth | n | PSPL | NonPSPL | PeriodicVar | ≥ 0.9042 | weighted ≥ 0.9042 |
|---|---|---|---|---|---|---|---|
| 1S1L single lens | PSPL | 33,353 | 0.03 → 0.54 | 0.53 → 0.43 | 0.44 → 0.01 | **0.356 → 0.116** | 0.383 → 0.135 |
| 1S2L planet | NonPSPL | 11,388 | 0.00 → 0.17 | 0.64 → 0.83 | 0.36 → 0.00 | 0.504 → 0.457 | 0.474 → 0.512 |
| 2S2L planet + binary source | NonPSPL | 12,234 | 0.00 → 0.16 | 0.71 → 0.83 | 0.29 → 0.00 | 0.582 → 0.469 | 0.505 → 0.523 |

Shipped → fine-tuned, argmax fractions; the two right-hand columns are the fraction over the
frozen threshold, unweighted and GULLS `final_weight`-weighted (Wilson 95% half-widths are
±0.5 points for single lenses and up to ±1.8 points for the weighted planetary rates; they are in
the reduced JSON). Single-lens false alarms at the frozen threshold fall by a factor of three
(35.6% → 11.6%) while planetary recall at that threshold falls (0.50 → 0.46; 0.58 → 0.47; at a
matched false-alarm budget it rises, section 1½), and the PeriodicVar contamination disappears.
The fine-tune also lifts argmax anomaly recall on genuine binaries from 0.64/0.71 to 0.83. The
threshold was calibrated on gap-free data and is open to re-tuning for this regime.

**Preprocessing sensitivity (state it).** RMDC26 samples F146 every 12.1 min against BinML's
15-min epoch grid, so about a fifth of the used epochs hold two observations (a third of the
observations share an epoch), and colour visits leave one of eight epochs empty in about a third of
the 2-h bins (occupancy 7/8, never seen in training; section 1½). `binml.preprocess` now
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

Monotonic over a factor of 15, and it persists at fixed \|u0\| < 0.1 (0.23 → 0.67 across the same
bins), so it is the source-size effect, not high magnification per se. Parallax does not drive it:
at \|u0\| ≥ 0.3 the rate is 0.033–0.049 across \|piE\| quintiles. **Cause:** `PSPLGen` is
point-source and has no parallax, while `NonPSPLGen` samples rho ∈ [1e-4, 1e-2] — so in the training
set a rounded, flattened peak only ever belonged to a binary, and the model learned "finite-source
rounding ⇒ NonPSPL". GULLS single lenses with rho/\|u0\| ≳ 0.3 (7.6% of the scored 1S1L, 6.3% of the eligible; median
\|u0\| 0.05, i.e. high magnification) are then flagged. This is a
training-set physics gap, fixable by adding finite-source single lenses (VBBinaryLensing `ESPLMag2`
is available) and fine-tuning; the curve cache re-scores GULLS in ~6 min. For the shipped
checkpoint the gap effect swamps this (0.40 → 0.26 across the same bins, i.e. no finite-source
signal visible).

**Finite-source single lenses — RESULT (2026-09-09 night).** `priors.PSPL_FINITE_SOURCE` (opt-in;
released training set unchanged), `generators.espl_magnification` (VBBinaryLensing ESPLMag2 — replaced 2026-09-11 by ESPLMag, see section 1½; rho log-uniform in
[1e-3, 1]. The rho statistics then quoted, median 0.012 / p90 0.60, describe the WHOLE 1S1L class
including its sub-day third; the scored single lenses have median 0.0087, p90 0.046, p99 0.29), regimes `fspl`
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
not yet at the floor. (The then-stated reason, "GULLS rho reaches 5", is a whole-class statistic;
only 43 of the 33,353 scored single lenses have rho > 1, and the high-ratio bins come from tiny |u0|.)

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
with ~0.30–0.33 planetary recall at threshold. (SUPERSEDED 2026-09-11: these calibrations used the
first season's pauses only, no season end, colour blanked on any overlap, and a different held-out pool
per checkpoint; redone under the measured seasons on one pool, section 1½ row 4.)

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
decisions. WITHDRAWN READING (2026-09-11): the colour gain comes almost entirely from planetary events WITHOUT
a policy-detectable F146 anomaly (at the 5.2% budget, +0.6 points on detectable, +10 points on
undetectable binaries), so it is not evidence that colour carries anomaly signal across simulators, and
not evidence for the three-band design; its mechanism is unidentified (section 1½ row 7).

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
prior (limb darkening, or the binary-source population, which we do not simulate at all). Caveats found
2026-09-11: its finite-source training curves carry ESPLMag2's artificial 4-8 mmag hand-off steps, and
the gain over g08e12 also includes 12 more epochs on a larger PSPL-heavy pool; both were rerun as
controls (section 1½ rows 13-14).

*Gapped-threshold calibration for fspl5s (`gapped_threshold_fspl5s_g08.json`, SUPERSEDED).* Purity-0.90
threshold on our gapped finite-source held-out set with the first season's pauses: 0.9492 (held-out
completeness 0.693 @ purity 0.897); on GULLS 2.0% single-lens false alarms with planetary recall
0.268 / 0.295 (clean-data threshold 0.9236 → 3.7% / 0.334). Under the MEASURED per-season schedule,
with the threshold chosen on the full gapped pool, the operating point is 0.957 (68% of random
validation slices 0.942–0.962): 1.6% single-lens false alarms, recall 0.244 / 0.269
(`gapped_threshold_fspl5s_g08_seasons.json`; section 1½ row 4).

*Colour ablation, both checkpoints (`transfer_colour_ablation.json`).* Removing F087/F213 costs
planetary recall at matched false-alarm budget for both: fspl_g08 0.369/0.399 → 0.302/0.334 at 5.2%
FA (mean recall 0.523 → 0.481); g08e12 0.295/0.313 → 0.253/0.271 (0.486 → 0.429). Per event the colour
bands move P(NonPSPL) by a median 0.09–0.11 and flip 9.1–9.5% of threshold decisions. The reading
"colour carries anomaly signal across simulators" is withdrawn: see the note above and section 1½ row 7.

**How to present it.** One paragraph plus the matched-budget table in the cross-simulator
subsection: the residual false alarms were traced to a missing physical effect in the training set,
the effect was added, the false alarms halved and the recall-at-budget curve moved up. (The earlier
sentence calling this the cleanest demonstration that "the training population, not the architecture,
is the lever" is withdrawn: the architecture was never varied.) Do not claim the gap is closed (the >1
bins sit at 0.28-0.29), state that the fine-tune used the pre-fix `--gap-aug` relabelling semantics
like g08e12, and state that RMDC26 was used for diagnosis and checkpoint choice (section 1½).

### What NOT to claim

- 2S2L is **not** the binary-source (1L2S) contaminant. Every 2S2L lens has a planetary
  companion (median q 1.2e-4 vs 1.4e-4 for 1S2L on the scored events: similar, not identical, KS
  p ~ 2e-18); the source is flagged binary in 55% of scored 2S2L. RMDC26 ships no 2S1L class, so
  the Gaudi (1998) degeneracy raised in §discussion is still untested, and a NonPSPL call on 2S2L
  may respond to the binary source rather than the planet. Say so.
- RMDC26's anomalous classes are entirely planetary (median q ~1.3e-4) while 82% of our NonPSPL
  evaluation events are stellar-mass-ratio binaries (median q 0.12). Do not compare GULLS recall
  with the in-distribution completeness at purity (0.879). ("Harder at matched amplitude" and "a
  factor of several below our prior" were unsupported and are withdrawn.)
- Do not describe the RMDC26 schedule as seven pauses at fixed phases; do not call the first-season
  mask "the exact schedule".
- Do not say the colour bands carry anomaly signal across simulators, that the schedule hides half
  the planets, or that 0.56 bounds generator-label recall (section 1½).
- The catalogue baseline `Source_F146 + 2.5 log10(fs_F146)` is uniformly 0.471 mag brighter
  than the quiescent flux in this release. We did not use it. Mention in a footnote only if a
  referee asks how the baseline was obtained.

### Where it goes (rewritten 2026-09-11; the manuscript text is `paper/draft_gulls_section.tex`)

- **Abstract [A]:** replace the last two sentences with the draft's; the finite-source checkpoint
  is quoted at a matched false-alarm budget with the development-set caveat, and recall is quoted
  against both label sets.
- **§Limits [B]:** replace the "legacy-like schedule" paragraph with the schedule paragraph and the
  gap table (measured per-season schedule; named arms).
- **New section [C]:** cross-simulator validation (data and selection, results table, the two label
  ontologies, events between seasons, the cascade on RMDC26, the floor on the label side, what the
  test does and does not establish).
- **Discussion [D]** and **model card / data availability [E]** as in the draft.
- **Decided:** the shipped weights stay; the gap-aware finite-source checkpoint ships alongside as
  `binml-gapaware.pt` with its own operating point (which checkpoint and threshold: section 1½,
  rows 4, 13 and 14). Every number enters through `make_macros.py` from the artifacts named in the
  draft's macro list; the GULLS artifacts join the manifest at that step.

### Reproduce (full population; each writes its own artifact and records its command and code version)

```
python validation/gulls/rmdc26_schedule.py                 # measured per-season schedule
python validation/gulls/rmdc26_dataset_facts.py            # every dataset fact quoted
python validation/gulls/gap_sensitivity.py --n 100
modal run --detach validation/modal_gap_finetune.py --epochs 12 --gap-aug 0.8 --lr 1e-4 --tag g08e12
python validation/gulls_transfer.py --per-class 200000 --chunk 250 --curve-cache <cache> --weights <ckpt> --out <summary> --rows-out <rows>
python validation/fspl_finetune_local.py --prefix fspl5s --init validation/gulls/weights/ft_g08e12.pt --tag <tag> --work <dir>
python validation/schedule_finetune_local.py
python validation/gulls/calibrate_gapped_threshold.py --heldout <fspl5s held-out> --ckpt <ckpt> --tag <tag> --rows <rows> --schedule seasons
python validation/gulls/gulls_summary_tables.py --models <name=rows ...> --f146 <name=rows ...> --by-season
python validation/gulls/gulls_summary_tables.py --scores validation/gulls/rmdc26_scores.csv.gz --models <name=column ...> --f146 <name=column ...> --by-season   # from a clone: no caches needed
python validation/gulls/detectability_relabel.py --extract --cap-1s1l 1600 --cap-binary 2500 --truth-cache <dir>
python validation/gulls/detectability_relabel.py --reduce  --cap-1s1l 1600 --cap-binary 2500 --truth-cache <dir> --models ... --f146 ...
python validation/gulls/floor_sensitivity.py --models ...
python validation/gulls/multi_season.py --extract --cap 1000 && python validation/gulls/multi_season.py --reduce
python validation/gulls/cascade_gulls.py --extract --scan --reduce
python validation/gulls/occupancy_sensitivity.py
python validation/gulls/build_scores_table.py
```

The RMDC26 tables are read locally (the Hugging Face datacenter rate limit blocks Modal); the revision
is pinned in `validation/gulls_transfer.py` and `binml/gulls.py`.

---

## 1½. Follow-up experiment ledger (rewritten 2026-09-11 after the triple check)

Rows 1-12 are the follow-ups proposed on 2026-09-09 and after; rows 13-17 were added by the
2026-09-11 verification (`docs/VERIFICATION_2026-09-11.md`, 182 findings in 61 issues, none refuted).
Every number below is in the artifact named in its row; the manuscript draft
(`paper/draft_gulls_section.tex`) takes them only through `paper/make_gulls_macros.py`. RMDC26 was used
for diagnosis, for the upper end of the single-lens rho prior and for choosing among the fine-tuned
checkpoints, so numbers for the chosen checkpoint are optimistic; every fine-tune is a single seed, and
two fine-tunes with one recipe on different pools differ by 0.014-0.023 in matched-budget recall.

| # | Experiment | Outcome (corrected where the verification required) | Artifacts | Status |
|---|---|---|---|---|
| 1 | Finite-source single lenses in the generator, then fine-tune from g08e12 with gap augmentation | Single-lens false alarms at the frozen threshold 11.6% (g08e12) → 5.2% (round 1, rho ≤ 1) → **4.8% (round 3 `fspl5s`, single-lens rho ≤ 5, binaries unchanged)**; 1S2L recall at the matched 5.2% budget 0.295 → 0.390; rho/\|u0\| bins 0.3-1 / 1-3 / >3: 0.43/0.65/0.67 → 0.15/0.28/0.29. Round 2 (binary rho also widened to 0.1) was worse: FA 8.7%, recall 0.344. The high-ratio bins are tiny-\|u0\| events, not large sources (43 of 33,353 scored single lenses have rho > 1; p99 0.29), so "rho up to 5 to cover GULLS" was the wrong justification. Caveats found by the verification: the finite-source curves were built with ESPLMag2, which has artificial 4-8 mmag hand-off steps (row 14 reruns with the smooth function); the gain over g08e12 also includes 12 more epochs on a larger, PSPL-heavy pool (row 13 is the control); out-of-support brightness explains part of the residual (FA 11.8% at m_base < 20 vs 4.1% inside our prior). | `fspl_finetune_{fspl,fspl5,fspl5s}_g08.json`, `transfer_tradeoff_all.json`, `rmdc26_dataset_facts.json` | ✅ (controls in rows 13-14) |
| 2 | Schedule-matched augmentation | **Corrected premise:** RMDC26's pauses differ between its six high-cadence seasons (row 16); the 2026-09-10 "exact schedule" arms used the first season's pauses only (16% of scored events), so their "no transfer" result was confounded (inside season 0 that arm did beat random gaps: 0.300 vs 0.237 at 5.2%). **Rerun with single-factor arms (one pool, one recipe, relabel off in both):** training on a measured RMDC26 season drawn per example (`sched_seasons`) against random 1-12 h gaps (`rand_norelabel`): held-out AP under the measured pauses 0.921 vs 0.911 (clean 0.948 vs 0.952); on RMDC26, single-lens FA 10.0% vs 12.8%, 1S2L recall at the 2 / 5.2 / 11.7% budgets 0.237 / 0.352 / 0.491 vs 0.179 / 0.279 / 0.433, mean recall over FA ≤ 0.3 0.501 vs 0.455, ahead in all six seasons. A fixed first-season mask generalises poorly to the other seasons (held-out AP under the measured pauses 0.568 with the relabel off, 0.401 with it on; shipped 0.169). The relabel makes no measurable difference with random gaps (AP 0.910 on vs 0.911 off) but hurts with a fixed mask: with the legacy 7.2-d onset grid, 20.1% of NonPSPL-labelled pool events (2,108 of 10,471; 7.4% of the 28,419 generated binaries) are eligible. **Schedule-matched augmentation with the measured seasons transfers; the combined finite-source + measured-season checkpoint is row 14.** | `schedule_finetune.json` (held-out on 24,011 scored of a 30,013 pool, three schedules; relabel exposure), `transfer_tradeoff_all.json` (by season), weights `validation/schedule_finetune_local.py` work dir | ✅ positive (rerun) |
| 3 | Detectability-conditioned relabelling of RMDC26 | On the first 2,500 scored 1S2L and 2S2L and 1,600 1S1L, with the training rule (single lenses never NonPSPL) and full-precision statistics: **44% / 46% of 1S2L / 2S2L carry no anomaly our policy would claim within one season** (rate-weighted 41% / 40%); pooled, 25.9% fail dchi2 ≥ 160 and 19.3% pass it but fall under 0.02 mag; rescaling the dchi2 cuts for RMDC26's 11% more F146 epochs moves this under a point. Not a ceiling: `fspl5s` flags 24% of the planetary events without a detectable anomaly. Recall on detectable anomalies: 0.474 / 0.557 at the frozen threshold, 0.326 / 0.398 at the recommended 0.957 (generator labels 0.380 / 0.404 and 0.244 / 0.267). Recall is significance-limited (0.17 at dchi2 160-500 → 0.87 above 1e5), not amplitude-limited. RMDC26's planetary classes are themselves pre-selected by GULLS' detection statistic (≥ 60 for every event). | `transfer_detectability_relabel.json`, `detectability_relabel.py`, truth cache v2 | ✅ |
| 4 | Threshold for the gapped regime, chosen on our own simulations | **Redone** under the measured per-season pauses on one common finite-source held-out pool (population-weighted NonPSPL prevalence 6.0%, vs 5.6% on the paper's test set), threshold on the full pool with the spread over 200 random 20% slices; RMDC26 read only afterwards. Full-pool threshold → RMDC26 single-lens FA / 1S2L / 2S2L recall: `fspl5s` **0.957 → 1.6% / 0.244 / 0.269** (68% slice range 0.942-0.962 → FA 1.3-2.4%); `fspl` 0.945 → 2.4% / 0.270 / 0.302; `fspl5` 0.970 → 1.8% / 0.202 / 0.225; `fspl5s_v2` 0.946 → 2.5% / 0.286 / 0.312; `fspl5s_noisy` 0.951 → 7.0% / 0.349 / 0.370; g08e12: no usable operating point (0.90 purity only on its top few events). The evaluator's own validation slice gives 0.931 for `fspl5s`, outside the 68% slice range. The 2026-09-10 calibrations (first-season pauses only, no season end, a different pool per checkpoint; `fspl5s` 0.949 → 2.0%) are superseded. | `gapped_threshold_<tag>_seasons.json` (legacy files kept) | ✅ redone |
| 5 | Noise-model matching | GULLS' F146 errors equal ours for m_base < 20 and reach 2.6× at 24-25 mag; a ~7× background variance reproduces this, a global factor does not. The noise-matched fine-tune (bkg_mult 7, `fspl5s` recipe) is worse at every matched budget (recall 0.299 vs 0.390 at 5.2%) and in false alarms (12.1% vs 4.8%), though its frozen-threshold recall is higher; single seed, single noise level, mechanism not attributed. | `gulls_noise_model.json`, `fspl_finetune_fspl5s_noisy_g08.json`, `gapped_threshold_fspl5s_noisy_g08{,_cleancal,_seasons}.json` | ✅ negative |
| 6 | Sub-day t_E single lenses (out of support) | 1,385 dense sub-day 1S1L from a random contiguous id window (median t_E 0.13 d): `fspl5s` flags 0.7% (unweighted) / 3.1% (rate-weighted) at the frozen threshold; argmax is PeriodicVar for 60-93% across checkpoints, so these are not read as microlensing. | `transfer_subday*.json` | ✅ |
| 7 | Colour ablation | Removing F087/F213 lowers 1S2L recall at the matched 5.2% budget by 0.042 (g08e12), 0.067 (fspl_g08), 0.068 (fspl5s); 9.1-9.5% of frozen-threshold decisions flip. **Reading withdrawn:** on binaries with a policy-detectable F146 anomaly colour changes recall only 0.534 → 0.539; on those without one it raises the flag rate 0.170 → 0.270. Not evidence that colour carries anomaly signal across simulators; mechanism unidentified. | `transfer_colour_ablation.json`, `transfer_detectability_relabel.json` (colour block) | ✅ |
| 8 | Finer anomaly-onset grid | The first implementation (2026-09-11 morning) skipped the grid points next to each coarse cut (two off-by-one errors) and flipped a released default; fixed in 580e172: the default is the legacy 7.2-d grid again, and `--onset-resolution-days 0.5` is a full-grid first-detectable scan that equals the cascade definition. The relabel rule that consumes the onset is not truth-based even with an exact onset (t_anom is when the anomaly first becomes detectable, not where the caustic is), so **audit finding 9 is re-opened**. The twin trained with the first implementation (`fspl5s_v2`): FA 5.35% vs 4.85%, recall 0.375 vs 0.390 at 5.2%, finite-source bins about 4 points worse; not adopted. | `fspl_finetune_fspl5s_v2_g08.json`, `tests/test_onset_resolution.py` | ✅ |
| 9 | Peak between seasons: per-season combiner | **Corrected twice.** First, "the schedule hides half the planets" was wrong. Second, the first run's own numbers were inflated: it included events peaking before or after the survey and seeded the wing-season refit at a host peak outside the window, which labels isolated bumps as anomalies. **Rerun** on between-season events only (22.0% of eligible; 3.6% peak before or after the survey), with the refit also started at the in-window maximum, first 1,000 per class: a high-cadence season borders the gap for 76% / 91% / 86% of between-season 1S1L / 1S2L / 2S2L (whole population); a claimable anomaly is visible in a wing for 44% of scorable 1S2L and 34% of 2S2L (in-season 56% / 54%), so the gap costs 12 / 20 points; host-visible subset 50% / 37%. `fspl5s` at the frozen threshold: single-lens FA 0.7% (5/739; 95% ≤ 1.6%), recall on wing-claimable anomalies 0.31 / 0.38 (in-season 0.47 / 0.56), generator-label 0.14; at 0.957: FA 0.1%, recall 0.20. At a 5.2% budget set on this population's 739 single lenses (threshold 0.61, 38 exceedances): 0.50 / 0.55, comparable to in-season (0.48 / 0.57). g08e12 is similar or better there (0.55 / 0.62 at 5.2%). Wing seasons need their own operating point. | `transfer_multiseason.json`, `multi_season.py`, cache `gulls_multiseason_cache_v2` | ✅ rerun |
| 10 | Cascade on RMDC26 | **Corrected, then rescanned.** The in-house comparison is the shipped model on our simulator (80% stellar binaries), so compare with its planetary strata; "because the anomalies are smaller" was untested and is withdrawn; the RMDC26-rate prevalence row had used the balanced scan sample (the scored set's rate-weighted planetary fraction is 22%). **Rescan** (half-day cuts, ≥ 10 F146 points per cut, unrounded probabilities, onset = first cut at which our label rule is met on GULLS' noise-free F146 curve with GULLS' errors; 3,000 single lenses and 3,000 binaries; 1,667 eligible binaries, nothing selected on model output). `fspl5s`, F146 only, frozen threshold: premature 2.2% (95% 1.6-3.0%), median lag of non-premature alerts +6.5 d, detected by season end 57% (55-59%). In-house planetary strata (shipped, our simulator): giant 1.4% premature, 89% detected, lag 4.5 d (n = 138); Neptune 4.2%, 73%, 4.5 d (n = 48). Single-lens alert burden, which the in-house scan cannot measure: 5.7% of single lenses alert at some point in the season (0.80 per 1,000 events per day); purity of the alert stream (single-lens contamination only; RMDC26 has no variables or flat sources) 5% at 1% planetary prevalence, 21% at 5%, 51% at RMDC26's own rate-weighted 22%. At the recommended 0.957: premature 0.9%, detected 42%, lag +8.75 d, single-lens alerts 2.2%, purity 10% at 1%. Three bands, frozen: premature 1.4%, detected 53%, lag +11.25 d. **Timing transfers; detection does not (same shortfall as full-season recall); at realistic prevalence the alerts need a second stage.** | `cascade_gulls.json`, `cascade_gulls.py`, cache `gulls_cascade_cache` (scan_v2) | ✅ rescanned |
| 11 | Detectability floor, label side | Fraction of 1S2L with a claimable anomaly: 70 / 64 / 60 / 56 / 49 / 40 / 27% at floors 0.005 / 0.01 / 0.015 / 0.02 / 0.03 / 0.05 / 0.1 mag. The earlier inference "the model behaves as if the floor were 0.005-0.01 mag" is withdrawn: precision of a fixed flagged set is non-increasing in the floor for any classifier. `fspl5s` flags 40% of floor-vetoed planetary events at a similar rate however far below the floor (38-47%), 16-20% with F146 alone; it also flags 10 of 20 single lenses with a significant sub-floor static-refit misfit (95% 30-70%), so the response is not planet-specific. | `floor_sensitivity.json`, `transfer_detectability_relabel.json` | ✅ |
| 12 | Usable by others | `binml.gulls` (modes peak / adjacent / all, low-cadence seasons skipped, `p_nonpspl_max`), two executed notebooks now committed (the first was gitignored, so its CI test failed on a clean checkout), `ft_g08e12.pt` committed, per-event score table `rmdc26_scores.csv.gz`. | `binml/gulls.py`, `examples/*.ipynb`, `tests/test_gulls_helpers.py`, `tests/test_notebooks.py` | ✅ |
| 13 | Continued-training control for row 1 | **The confound was real.** `pspl5s_ctrl` = the `fspl5s` pool shape (same shard indices, same PSPL-heavy high-magnification companion shards) with point-source single lenses, same warm start from g08e12 and recipe. On RMDC26: single-lens FA 6.4% (g08e12 11.6%, `fspl5s` 4.8%); 1S2L recall at 5.2% 0.361 (0.295, 0.390); mean recall over FA ≤ 0.3 0.490 (0.486, 0.538); rho/\|u0\| bins 0.036 / 0.048 / 0.084 / 0.208 / 0.410 / 0.513 (fspl5s 0.031 / 0.037 / 0.060 / 0.154 / 0.280 / 0.288). So the extra training on a high-magnification-rich pool accounts for most of the drop in false alarms and of the recall gain over g08e12; the finite-source physics adds the rest, concentrated in the two largest-ratio bins (0.41 → 0.28, 0.51 → 0.29), and only it restores single-lens recall for large sources on our own finite-source held-out (1-3 / >3 bins: control 0.22 / 0.16, `fspl5s` 0.66 / 0.84). Single seed; the recall-at-budget difference (0.029) is about 1.5× the seed spread, the bin differences several times the binomial error. | `fspl_finetune_pspl5s_ctrl_g08.json`, `transfer_full_pspl5s_ctrl_g08.json`, `transfer_tradeoff_all.json`, weights `ft_pspl5s_ctrl_g08.pt` | ✅ |
| 14 | Smooth finite-source magnification; combined arm | ESPLMag2 has artificial 4-8 mmag steps where it hands off to the point-source formula (and errs by up to 5e-3 there); ESPLMag matches direct disc integration to ~1e-5 outside the disc. Generator switched (legacy flag `PSPL_ESPL_LEGACY` kept). **Combined arm** `fspl5s_seasons_g08`: the `fspl5s` pool regenerated with ESPLMag (same shards and seeds), trained with measured-season pauses and the relabel off, same warm start and recipe. On RMDC26 it ties `fspl5s` at every matched budget (1S2L recall within 0.008 at 2 / 5.2 / 11.7%, below the 0.014-0.023 seed spread; 0.391 vs 0.390 at 5.2%), flags more single lenses at the frozen threshold (6.3% vs 4.8%, mostly in the two largest rho/\|u0\| bins: 0.330 / 0.333 vs 0.280 / 0.288), and is ahead in five of six seasons by at most 0.012. On our own held-out with the measured pauses it is better (AP 0.893 vs 0.874; clean 0.929 vs 0.931); its own recalibrated threshold 0.956 gives RMDC26 FA 2.4%, recall 0.294 / 0.329 (fspl5s at 0.957: 1.6%, 0.244 / 0.269), the same trade-off curve at a different point. **`fspl5s` stays the recommended checkpoint** (tied transfer, fewer false alarms at both thresholds, the between-season and cascade analyses were run on it). The two gains (finite source, measured seasons) do not add on RMDC26. **Smooth-function-only control** `fspl5s_espl_g08` (same ESPLMag pool as the combined arm, random gaps, relabel on, otherwise the `fspl5s` recipe): indistinguishable from `fspl5s` — FA 4.83% vs 4.85%; 1S2L recall within 0.001 at every budget (0.389 vs 0.390 at 5.2%); every rho/\|u0\| bin within 0.001; seasons within 0.005; own held-out PSPL recall in the 1-3 / >3 bins 0.66 / 0.84 as before; recalibrated threshold 0.956 (68% 0.942-0.962) → FA 1.6%, recall 0.244 / 0.269. **ESPLMag2's hand-off steps did not shape the finite-source result**, and the combined arm's extra false alarms come with its pause training (measured seasons with the relabel off), not with the magnification function. | `fspl_finetune_fspl5s_{seasons,espl}_g08.json`, `transfer_full_fspl5s_{seasons,espl}_g08.json`, `gapped_threshold_fspl5s_{seasons,espl}_g08_seasons.json`, `transfer_tradeoff_all.json`, weights `ft_fspl5s_{seasons,espl}_g08.pt`, `tests/test_fspl.py` | ✅ |
| 15 | Partial bin occupancy | RMDC26's colour visits leave one of eight F146 epochs empty in about 35% of the 2-h bins (colour bins at 2/3 in about 45%); training never contains either. On 6,024 real RMDC26 inputs, setting the occupancy to full changes `fspl5s` single-lens FA 4.9% → 3.6% and 1S2L recall 0.35 → 0.26 (g08e12: FA unchanged, recall 0.45 → 0.38). Imposing the pattern on our binned held-out instead is not faithful (the features still come from all epochs) and produced a spurious collapse (AP 0.51), caught before use. | `occupancy_sensitivity.json`, `gapped_threshold_fspl5s_g08_{occupancy,seasonsocc}.json` (diagnostic) | ✅ |
| 16 | The RMDC26 schedule, measured | Six high-cadence seasons of 70.7 d and four low-cadence of 65.1 d; gaps 109-120 d; 685 of 1,717 days observed; F146 paused 43-44 h per season in six or seven pauses; only the pauses near days 1.0, 35.2 and 69.5 recur; three seasons merge two pauses into one ~12 h pause; colour bins almost never empty. | `rmdc26_schedule.json`, `rmdc26_dataset_facts.json` | ✅ |
| 17 | Label-policy fragility (pre-existing) | The single-start static refit is chaotic for rare borderline events: in-house seed 304033 flips PSPL ↔ NonPSPL under a 1e-14 mag perturbation (4 of 60 trials), about one event per thousand. Disclosed, not changed (it would change the released labels). | `validation/gulls/PROVENANCE.md` | ✅ documented |

### Superseded ledger (2026-09-09 to 09-11), kept for the predictions made before each run

Superseded by the table above wherever the two differ; several outcomes below were corrected by the
2026-09-11 verification (`docs/VERIFICATION_2026-09-11.md`).

Seven post-revision experiments were proposed once the full-population GULLS numbers were final.
Each row records the prediction made BEFORE running it, what actually came out, the artifact, and
what remains. Three are done, four are open. Costs are wall-clock on the M5 (10 cores, MPS).

| # | Experiment | Prediction (made first) | Outcome | Artifact | Status |
|---|---|---|---|---|---|
| 1 | Finite-source single lenses in the generator (VBBinaryLensing `ESPLMag2`), PSPL given a ρ prior, warm-start from g08e12 with gap augmentation, re-score GULLS from the cache | 1S1L false alarms 11.7% → ~5–6%; the ρ/\|u₀\| > 0.3 bins collapse toward the 4.6% floor | **Total hit, bins half-right.** FA 11.7% → 5.2% (round 1, ρ ≤ 1) → **4.9%** (round 3 `fspl5s`, single-lens ρ ≤ 5, binaries unchanged). Bins (0.3–1 / 1–3 / >3): 0.43/0.65/0.67 → **0.15/0.28/0.29** — halved to thirded, not at the floor. Planetary recall at the matched 5.2% budget 0.295 → **0.390**; mean recall over FA ≤ 0.3 0.486 → 0.538. Round 2 (also widening the *binary* ρ prior to 0.1) was negative; round 3 attributes the gain cleanly to the single-lens extension. Three rounds took ~2 h each including two disk-stall restarts. | `fspl_finetune_{fspl_g08,fspl5_g08,fspl5s_g08}.json`, `transfer_full_{fspl_g08,fspl5_g08,fspl5s_g08}.json`, `transfer_tradeoff_all.json`, weights `weights/ft_fspl*_g08.pt`; `tests/test_fspl.py` | ✅ done (3 rounds) |
| 2 | Schedule-matched augmentation: fine-tune with the exact seven fixed ~6.2 h pauses at 12.1-min cadence in a 70.7-d season (the `arm12` tree from the cadence rerun has the 12-min config) and compare to g08e12's random 1–12 h gaps | not stated | **In-simulation gain, no GULLS gain — and the experiment exposes the `t_anom` bug as blocking.** Three arms on ONE pool (the cadence rerun's 15-min natural shards 0–11, 89,919 events), warm-start from the shipped weights, identical recipe (12 epochs, lr 1e-4, truncation 0.5, gap probability 0.8): `rand` = g08e12's random 1–12 h runs; `sched` = the exact RMDC26 mask (seven ~6.2 h pauses + the 1.3 d past the 70.7 d season end); `sched_norelabel` = the same mask with the caustic-in-gap relabel off. **Held-out under the exact schedule** (30,013 events): AP rand 0.906 / sched 0.805 / **sched_norelabel 0.924**; clean 0.951 / 0.934 / 0.952 (reference rows on the same held-out: shipped 0.211, g08e12 0.887). `sched` loses NonPSPL recall (0.80 vs 0.91) because `t_anom` is quantised to ten values 7.2 d apart and two of them (21.6 d, 72.0 d) sit inside the fixed mask, so the relabel fires on **20.1% of ALL NonPSPL training events on every presentation** (measured on the pool) — audit finding 9 with a measured cost; random gaps dilute the same error across positions. **GULLS** (56,975 matched): `rand` replicates g08e12 on a different pool (FA 11.8% vs 11.7%; recall at the 5.2% budget 0.275 vs 0.295); `sched` is worse (0.163) with FLAT false alarms across ρ/\|u₀\| (0.15 even on point sources — the moved boundary); `sched_norelabel` matches `rand` at every budget (0.154 / 0.262 / 0.413 vs 0.171 / 0.275 / 0.434 at 2 / 5.2 / 11.7%) but its probabilities shift up wholesale — 37% of single lenses over the frozen threshold, recall 0.77 — because binaries whose caustic really is in the mask now keep the NonPSPL label and teach the model that a gapped single-lens shape can be a binary. Neither relabel setting is right; the correct fix is the un-quantised anomaly onset (regenerate with the true `t_anom`), which is now a prerequisite for any schedule-matched training. The finite-source extension (row 1) remains the only lever that moved GULLS. The 12.1-min cadence is not an arm: `binml.preprocess` pools to one value per 15-min epoch. | `validation/gulls/schedule_finetune.json`, `transfer_full_sched_{rand,sched,sched_norelabel}.json`, weights `ft_sched_{rand,sched,norelabel}.pt`, `validation/schedule_finetune_local.py`, `pipeline/train.py --gap-schedule rmdc26 --gap-relabel-anomaly off`, `tests/test_gap_augmentation.py` | ✅ done 2026-09-11 (in-simulation positive, GULLS neutral; `t_anom` fix now blocking) |
| 3 | Detectability-conditioned relabelling of GULLS itself: apply our Δχ² + 0.02 mag floor policy to the noise-free `true_flux_uJy` of GULLS binaries and score recall against detectability labels instead of generator labels | the 1S2L "recall 0.46" counts undetectable planets as misses; this gives the first like-for-like number | **Done (6,600 events: 2,500 1S2L, 2,500 2S2L, 1,600 1S1L).** Under BinML's own policy **44% of GULLS 1S2L and 46% of 2S2L "planetary" events have no detectable anomaly** (weighted 41% / 40%): 26% fail Δχ² ≥ 160 outright, 19% pass Δχ² but fall under the 0.02 mag floor. So the generator-label recall ceiling is ~0.56, not 1. **Recall on detectable-anomaly binaries** at the frozen threshold: fspl5s 0.474 / 0.557 (1S2L / 2S2L) vs 0.380 / 0.404 on generator labels; g08e12 0.568 / 0.627 vs 0.458 / 0.465; at the matched 5.2% budget fspl5s 0.483 / 0.570. Recall is amplitude-limited: fspl5s catches 0.40 of detectable anomalies of 0.02–0.05 mag, 0.51 at 0.1–0.2 mag, 0.85 above 0.5 mag. **The model flags 40% of the floor-vetoed binaries** (Δχ² ≥ 160 but deviation < 0.02 mag; even 45% of those under 5 mmag) against 13% of the Δχ² < 160 ones and 4.9% of single lenses: it responds to statistically significant sub-floor anomalies that our label policy deliberately declines to claim. Whether those are false alarms is exactly the 0.02 mag floor's justification (REVISION §2, referee item 1). Also: 2.2% of GULLS single lenses are NonPSPL by the policy's point-source refit (median ρ/\|u₀\| 2.0 — strong finite-source); the model flags 22% of those vs 4.6% of the rest, so part of the residual large-ρ false alarms are events our own refit would call anomalous. | `validation/gulls/transfer_detectability_relabel.json`; extractor/reducer `detectability_relabel.py`; truth cache outside the repo (extensible prefix selection) | ✅ done 2026-09-10 | |
| 4 | Threshold recalibration for the gapped regime on OUR gapped finite-source held-out simulations (never on GULLS) | not stated | **g08e12 cannot reach purity 0.90 on a finite-source population**: its threshold saturates at 0.999 with completeness 0.001 — the gap-aware checkpoint has no valid operating point there. `fspl_g08`: clean 0.929 / gapped 0.935 → GULLS FA 3.5% / 3.1%, recall 0.312 / 0.296 (1S2L). **`fspl5s_g08`: clean 0.924 → FA 3.7%, recall 0.335 / 0.369; gapped 0.949 → FA 2.0%, recall 0.268 / 0.295** (held-out completeness 0.693 at purity 0.897). | `gapped_threshold_{ft_g08e12,fspl_g08,fspl5s_g08}.json` | ✅ done |
| 5 | Noise-model matching: GULLS is 3–4× noisier at the faint end than our photometry; a noise-multiplier augmentation, fine-tune, re-score | not stated | **Premise measured (overstated), remedy tried: NEGATIVE.** GULLS/ours F146 error ratio 0.95 at m_base 17–19 rising to 2.64 at 24–25 (5,261 events); a ~7× background variance reproduces it (best fit 7.4, rms 0.062 in log σ), a global multiplier does not (1.41×, rms 0.27). Fine-tune `fspl5s_noisy_g08` = the exact fspl5s recipe (same shard seeds, same warm start from g08e12) with `bkg_mult` 7: on its own noisy finite-source held-out it is fine (completeness 0.78 at purity 0.92, AP 0.93, where g08e12/shipped collapse to AP 0.54/0.56), but **on GULLS it is worse than fspl5s on every measure**: single-lens false alarms 12.1% vs 4.9% at the frozen threshold; recall at the matched 5.2% budget 0.298 vs 0.390; mean recall over FA ≤ 0.3 0.499 vs 0.538; and the finite-source bins revert toward the g08e12 pattern (0.42 / 0.58 / 0.51 vs 0.15 / 0.28 / 0.29). Calibrated on its noisy held-out (threshold 0.961): GULLS FA 5.5%, recall 0.307 / 0.329 — vs fspl5s' 2.0% / 0.268 / 0.295 at its own 0.949. Reading: the extra faint-end scatter teaches the model to discount small deviations, which is exactly the finite-source signal round 3 taught it to read; GULLS' own noise is not what limits transfer, its physics is. Noise matching is not a lever. | `fspl_finetune_fspl5s_noisy_g08.json`, `transfer_full_fspl5s_noisy_g08.json`, `transfer_full_reduced_fspl5s_noisy_g08_vs_fspl5s_g08.json`, `gapped_threshold_fspl5s_noisy_g08{,_cleancal}.json`, `transfer_tradeoff_all.json` (six models), weights `ft_fspl5s_noisy_g08.pt`; `gulls_noise_vs_ours.py` → `gulls_noise_model.json`; knobs `SurveyConfig.noise_mult/bkg_mult` (defaults bit-identical, `tests/test_noise_mult.py`) | ✅ done 2026-09-10 (negative) |
| 6 | Sub-day-t_E single lenses (the FFP-like third of RMDC26 1S1L removed by the t_E ∈ [1, 300] d support cut) as an explicit out-of-support row | not stated | **Done (1,385 sub-day 1S1L, median t_E 0.13 d).** Not anomaly false alarms: at the frozen threshold 1.5% (shipped), 3.0% (g08e12), 1.5% (fspl), 3.3% (fspl5), **0.7% (fspl5s)**; at fspl5s' calibrated 0.949 threshold 0.07%. But they are not recognised as microlensing either: argmax is PeriodicVar for 60–93% of them (PSPL ≤ 5%) — a 3-hour spike reads as a variable star. For t_E 0.5–1 d fspl5s already calls 22% PSPL. Out of support, as the prior says; an FFP-aware model needs sub-day t_E in the prior, not a threshold change. | `transfer_subday.json` + `transfer_subday_{shipped,ft_g08e12,fspl_g08,fspl5_g08,fspl5s_g08}.json`; `subday_summary.py`; separate curve cache | ✅ done 2026-09-10 | |
| 7 | Colour ablation on GULLS: F146-only vs three-band from the cache | tells whether GULLS' colour bands help or hurt in transfer | **Colour helps for all three checkpoints.** Recall at the matched 5.2% budget (1S2L / 2S2L): fspl5s 0.322 → 0.390 / 0.354 → 0.421; fspl_g08 0.302 → 0.369 / 0.334 → 0.399; g08e12 0.253 → 0.295 / 0.271 → 0.313. Mean recall over FA ≤ 0.3: +0.044 / +0.042 / +0.057. About 9.4% of frozen-threshold decisions flip between the two inputs. | `transfer_colour_ablation.json` (three checkpoints), `transfer_full_*_f146only.json` | ✅ done (fspl5s added 2026-09-10) | |

| 8 | **Onset fix** (audit finding 9): resolve `t_anom` on the 0.5 d grid the cascade evaluation uses instead of rounding it up to 7.2 d; regenerate the fspl5s pool and rerun the round-3 recipe as `fspl5s_v2_g08` | rows 2/3 showed the old grid mislabels 20% of binaries under a fixed schedule; the fine-tune labels should improve, GULLS numbers should not get worse | **Fix is correct; transfer unchanged.** Same seeds, same recipe, onset on the 0.5 d grid: held-out completeness 0.845 at purity 0.877, AP 0.934 (v1: 0.831 / 0.882 / 0.931) — a small gain in the labels' own currency. GULLS (56,975): FA 5.35% vs 4.85% at the frozen threshold, recall 0.375 vs 0.390 at the 5.2% budget, mean recall 0.532 vs 0.538, finite-source bins 0.20 / 0.33 / 0.33 vs 0.15 / 0.28 / 0.29 — marginally worse, within a point or two. Expected: with random gaps the old grid's error was diluted, so fixing it buys label correctness, not transfer. Sidecar recommendation stays `ft_fspl5s_g08.pt` (better on the external test); `ft_fspl5s_v2_g08.pt` is its label-correct twin for anyone who needs the fixed generator (any schedule-matched training must start from it). | `fspl_finetune_fspl5s_v2_g08.json`, `transfer_full_fspl5s_v2_g08.json`, `transfer_full_reduced_fspl5s_v2_g08_vs_fspl5s_g08.json`, `gapped_threshold_fspl5s_v2_g08.json`, `transfer_tradeoff_all.json` (seven models), weights `ft_fspl5s_v2_g08.pt`; `SurveyConfig.onset_resolution_days`, `tests/test_onset_resolution.py` | ✅ done 2026-09-11 |
| 9 | **Per-season combiner for peak-in-gap events** (the single-season contract's cost, measured without retraining): score every dense season adjacent to the gap, take the max anomaly probability; truth policy per season | 25.5% of eligible GULLS events (22,581 1S1L, 1,384 1S2L, 1,822 2S2L) have t0 in a 109–120 d inter-season gap and were skipped by every transfer run | **Most of the population is recoverable; the lost half of the planets is lost to the schedule, not the model.** 1,000 per class: a dense adjacent season exists for 77% / 92% / 88% (1S1L / 1S2L / 2S2L; the rest border the four low-cadence seasons); a detectable EVENT is visible in a wing for 60% / 86% / 88%; a detectable ANOMALY for 0.3% / **52% / 44%** (in-season: 56% / 54%) — the peak is unobserved, yet half the planets still leave a policy-detectable trace in a wing. fspl5s with the max-combiner: single-lens false alarms **0.4%** at the frozen threshold (0.1% at 0.949) — a wing without a peak reads as PSPL/Flat; recall on detectable-anomaly binaries 0.28 / 0.32 at the frozen threshold and **0.41 / 0.46 at the 2% / 5.2% budgets on this population** (in-season 0.48 at 5.2%); generator-label recall 0.15 / 0.16. g08e12 is similar (0.44 / 0.50 at matched budgets). Reading: a whole-light-curve model would not recover the anomalies that fall in the gap (no data), and the combiner already scores the rest with in-season-like recall and lower false alarms; what a multi-season model uniquely adds is events spanning seasons and the four low-cadence seasons (18% of eligible events, out of support today). | `validation/gulls/multi_season.py`, `transfer_multiseason.json`; cache `~/Desktop/Research/microlensing/gulls_multiseason_cache` | ✅ done 2026-09-11 |

| 10 | **The cascade on an independent simulation**: the paper's half-day streaming protocol on GULLS seasons from the curve cache, with the truth-informed onset recomputed per half-day cut on the noise-free curve by our own rule; single lenses scanned too, for alert burden and streaming purity at a stated planetary prevalence | the in-house premature rate (1.6%) and lag (+5 d) have never been checked outside our simulator; RMDC26's rate weights over-represent planets (22% of the rate-weighted eligible set), so purity must be quoted at 1% / 5% as well | **Timing transfers; within-season detection does not; the alert burden is the new number.** 6,000 events scanned (3,000 1S1L, 1,500 + 1,500 binaries), fspl5s, 144 half-day cuts. Eligible binaries (anomaly detectable at the full window by our rule on the noise-free curve, finite onset): 1,667 of 3,000; median onset day 35. **Primary protocol (F146 only, frozen threshold): premature alerts 2.2% [1.6, 3.0] (in-house 1.6%), median non-premature lag +6.5 d (in-house +5.0 d), detected within the season 57% [55, 59] (in-house 89%)** — GULLS' eligible anomalies are smaller (down to the 0.02 mag floor at matched selection), so fewer cross within the season, but when they do the timing behaves as in-house. Three bands: premature 1.4%, lag 11 d, detected 53%. At the calibrated 0.949 threshold: premature 1.2% / 0.4% (F146 / three-band), lag 8 / 13.5 d, detected 45% / 40%. **Alert burden on single lenses: 5.7% alert at least once per season at the frozen threshold (0.80 alerts per 1,000 events per day), 2.7% at 0.949 (0.38 per 1,000 per day).** **Streaming purity** (alerts from detectable-anomaly binaries over all alerts) at a stated planetary prevalence: 1% → 5% (frozen) / 8% (0.949); 5% → 21% / 30%; at RMDC26's own rate weights (31% planetary in this sample, an over-representation) 57% / 67%. Reading: at a realistic prevalence the streaming list is a few percent planets, which is the paper's "post-hoc modelling priority, not real-time triggering" conclusion with a number attached; the referee's mixed-class item is answered on external data. | `validation/gulls/cascade_gulls.py` → `cascade_gulls.json`; scan cache `~/Desktop/Research/microlensing/gulls_cascade_cache` (resumable) | ✅ done 2026-09-11 |
| 11 | **Detectability-floor sensitivity, label side** (referee item 1, half): re-derive the GULLS relabelling at floors 0.005–0.1 mag from the stored noise-free statistics; recall on detectable binaries and ontology precision per floor | the floor is the binding choice (row 3); expect the detectable fraction to move a lot and the model's precision to peak below 0.02 | **Detectable planets: 70% at 0.005 mag → 64% (0.01) → 56% (0.02) → 49% (0.03) → 40% (0.05) → 27% (0.1).** fspl5s recall on detectable anomalies rises 0.49 → 0.51 → 0.54 → 0.56 → 0.63 across the same floors while its ontology precision falls 0.84 → 0.78 → 0.70 → 0.64 → 0.54 → 0.41: **the model behaves as if the floor were ~0.005–0.01 mag**, i.e. it detects a class of small but statistically significant anomalies the adopted policy declines to claim. The training side (retraining at another floor) is not covered. | `validation/gulls/floor_sensitivity.py` → `floor_sensitivity.json` | ✅ done 2026-09-11 |
| 12 | **Usable by others**: `binml.gulls` helper (pinned revision, season finding, empirical baseline, remote fetch, per-season combiner), two executed notebooks (offline quickstart run in CI; Roman event classification), README note updated to the sidecar and current numbers | — | Notebook 01 on event 306535 (1S2L, t_E 46 d): P(NonPSPL) 0.996 in its season, 0.070 for a single lens across all six dense seasons; 306534 (t_E 2.7 d) sits at 0.59, under the threshold — a short-timescale planet the model does not commit to. | `binml/gulls.py`, `tests/test_gulls_helpers.py`, `examples/00_quickstart_synthetic.ipynb`, `examples/01_classify_roman_event.ipynb`, `tests/test_notebooks.py`, `docs/usage.md` § Roman | ✅ done 2026-09-11 |

**All twelve rows done (2026-09-11).** Manuscript text for rows 1–9 drafted in `paper/draft_gulls_section.tex`; rows 10–11 add one paragraph each there (cascade on RMDC26; floor sensitivity); row 12 is repository work (helper, notebooks, docs). Manuscript text drafted in `paper/draft_gulls_section.tex` (numbers still to enter through `make_macros.py`). What moved GULLS: only the finite-source single lenses (row 1) and the colour bands (row 7). What did not: schedule-matched augmentation (row 2, gain confined to our own simulations) and noise matching (row 5, negative). What changed how the numbers should be read: the detectability relabelling (row 3) — quote recall on detectable-anomaly binaries next to the generator-label number, and say that 44–46% of GULLS planetary events carry no anomaly our policy would claim. What is out of support and stays so: sub-day t_E (row 6). Two of the seven (rows 2 and 3) turn audit finding 9 (`t_anom` quantised to 7.2 d) from a documented caveat into a measured cost; regenerating with the true onset is now the first item for any further training.

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

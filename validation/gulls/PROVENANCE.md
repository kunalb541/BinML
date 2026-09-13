# Provenance of the RMDC26 / follow-up artifacts (written 2026-09-11 after the triple check)

The 2026-09-11 verification found that several artifacts record the git commit of the invocation that
WROTE them, not of the code that GENERATED their data, and that some data were generated from an
uncommitted working tree. Runners now record `git describe --dirty` at data-generation time
(`provenance.json` in each work dir). For the artifacts made before that, this is what is known.

| Artifact | Recorded commit | What actually generated the data | Settings that matter |
|---|---|---|---|
| `fspl_finetune_fspl_g08.json`, `ft_fspl_g08.pt` | 7ed2cc4 | working tree during the 2026-09-09 night session | ESPLMag2 finite source; legacy 7.2-d onset; random 1-12 h gaps with any-overlap colour blanking |
| `fspl_finetune_fspl5_g08.json`, `ft_fspl5_g08.pt` | 1f8f193 | uncommitted tree: the `fspl5` regimes and the runner's `--prefix` arrived in 067ceb7 | as above; binary rho prior widened to 0.1 |
| `fspl_finetune_fspl5s_g08.json`, `ft_fspl5s_g08.pt` (round 3; recommended until 2026-09-12) | 0822e4e | tree at or just before 0822e4e | ESPLMag2 (artificial 4-8 mmag hand-off steps in its finite-source single lenses); legacy onset; truncation augmentation with the point-source visible amplitude. Regenerate its pool with `--prefix fspl5s_legacy` (regimes `fspl5s_legacy{,_highmag}`, same seeds); a retrain at HEAD also uses the finite-source visible amplitude, which changes about 0.2% of presentations |
| `fspl_finetune_fspl5s_noisy_g08.json` | 958a362 | uncommitted tree: the `fspl5s_noisy` regimes arrived in dfa3ebf | bkg_mult 7; ESPLMag2; legacy onset |
| `fspl_finetune_fspl5s_v2_g08.json`, `ft_fspl5s_v2_g08.pt` | ad8eab2 | uncommitted tree from 08:47:51 (before 2594e27, 08:48:34) | first fine-onset scan, with two off-by-one errors (fixed in 580e172): 8.5% of onsets stayed on the coarse grid |
| `schedule_finetune.json` rows `rand`, `sched`, `sched_norelabel` | none | 2026-09-10 tree (958a362-era) | `sched*` use the FIRST-SEASON pause mask, which matches one RMDC26 season in six |
| `gapped_threshold_<tag>.json` (no suffix) | none | 2026-09-10 tree | first-season pauses only, no season end, colour blanked on any overlap; each tag calibrated on a DIFFERENT held-out pool (ft_g08e12 and fspl_g08 on fspl_local_work; fspl5s on fspl5s_local_work; noisy on its own and on fspl5s; v2 on its own). Superseded by `gapped_threshold_<tag>_seasons.json` (measured seasons, common fspl5s pool) |
| `transfer_detectability_relabel.json`, `floor_sensitivity.json` before 580e172 | — | truth cache v1: 4-d.p. rounded statistics; 600 of the 1,600 single lenses left over from an aborted run with a different selection | regenerated in 580e172 from `gulls_truth_cache_v2` (full precision, documented selection). `floor_sensitivity.json`'s calibrated block at 580e172 used the superseded first-season threshold 0.949 (key `calibrated_gapped`) although the script had moved on; regenerated 2026-09-12 with the recommended threshold stored in the block |
| `transfer_multiseason.json` before 2026-09-11 evening | — | v1 cache: included 3.6% of events peaking before or after the mission; single-start refit seeded at a t0 outside the window | superseded by the v2 cache (between-season events only, multi-start refit) |
| `cascade_gulls.json` before 2026-09-11 evening | — | scan v1: P rounded to 4 d.p.; cuts with 1-9 F146 points scored; prevalence row computed on the balanced sample | superseded by `scan_v2_*` |
| `transfer_full_<tag>.json` of fine-tunes scored before 2026-09-11 (13 files, including `transfer_full_fspl5s_g08.json`) | — | written by `gulls_transfer.py` before its status string was fixed | their `status` field reads 'NOT A TRANSFER MEASUREMENT for the shipped checkpoint'; that string was meant for gap-aware checkpoints in general and does not mean the numbers are invalid. Artifacts written since carry the corrected string |
| `gapped_threshold_fspl5s_g08_seasonsocc.json` | — | 2026-09-11 | the blank + occupancy-cap DIAGNOSTIC (AP 0.743); its `schedule` field said `seasons` and was corrected by hand to `seasons_occ` on 2026-09-12 (note field in the file; numbers unchanged). The evaluation directories under `fspl5s_local_work/calib/` were renamed after evaluation, so their `meta.json` cache paths are stale (the `_seasons` eval points at a non-existent `mm_heldout_gapped_seasonsmask`, the `_seasonsocc` eval at `mm_heldout_gapped_seasons`, which now holds the blank-only data); the saved logits reproduce exactly on the current directories (2026-09-12 check) |

Later artifacts (2026-09-12/13) record their generating code themselves: `validation/referee_round.json` (`code`,
`command`, per-arm `run_shard_args`, fine-tune checkpoint hashes, and an `archive` block hashing the per-event inputs
committed under `validation/referee_round_archive/`; shards 90-91 of the held-out pool regenerated under
`~/Desktop/Research/microlensing/referee_local_work`) and `validation/truth_relabel_impact.json` (`code`,
`gen_settings`; shard 2 regenerated with `--truth-bins` after the 2026-09-13 truth-binning fix, and its 0.5-d-onset twin
`raw05/shard_00002.h5`, whose events are identical). Both were regenerated from a clean commit after the third
verification. The two colour fine-tunes ran on Apple MPS (pipeline.train default device), whose kernels are not
bitwise deterministic; their checkpoints are archived with hashes, so the reported numbers re-derive from them, but
a rerun of the fine-tunes need not reproduce them to the last digit. The seed replicates `fspl_finetune_fspl5s_seasons_g08_s{2,3}.json` use the same pool (one memmap) and
recipe as `fspl5s_seasons_g08`; only `--seed` differs (20260910, 20260911 against 20260909), which also changes the
80/10/10 split and hence the kept epoch (9, 5, 11). Their recorded code is 618aca3-dirty (s1), 4a37b8e-dirty (s2) and
1c4234f-dirty (s3; scored on RMDC26 at fa91bc7-dirty); `pipeline/train.py` differs between those commits only in
comments and truth-gated branches that caches without truth bins never enter.

Which checkpoint scored what (2026-09-13, from the third verification): the 14.9M-event stress suite
(`paper/results/stress_report.json`) was scored with the stage-5 checkpoint (`aws/ud_bineval.sh`,
`v5runs/binml_v5_stage5.pt`, sha256 4e7a5a85...), not the released `binml.pt` (= stage 6, 897a9aec...);
`validation/stress_rescore_local.py` re-scores regenerated shards of the quoted tiers with both
(`validation/stress_rescore_local.json`: 255,527 events; shards generated and scored at 6016bc6, metrics recomputed at
b216d54; the raw shards were deleted after caching and regenerate from the seeds). Stage 5 on the subset reproduces
the suite (macro-F1 0.927 vs 0.927; median difference 0.012 over the quoted numbers, largest 0.095 for the
wide-separation recall on 60 events), so the subset stands in for the suite. The suite's PSPL recalls are per label,
and in `oor_pspl_shortte` 38% of the weighted PSPL-labelled events are demoted binaries of natural timescale: the
sub-day single lenses themselves are classified PSPL 0.215 (stage 5) / 0.274 (released) of the time and called
anomalies 55% / 71% (21% / 37% above the frozen threshold). The frozen
evaluation arrays (`paper/results/`) are stage 6 (`meta.json`). The pre-RMDC26 artifacts that record no checkpoint
hash (`baselines_result.json`, `gap_matched_result.json`, `latency_gaps_result.json`, `prevalence_result.json`,
`cadence_result.json`, `cascade_events.json`) were all committed on or after 2026-08-04, after stage 6 shipped
(2026-07-26), so stage 6 is likely but not recorded; `prevalence_result.json` is computed from the stage-6 frozen
arrays.

Label-policy fragility, measured (2026-09-13, `validation/truth_relabel_impact.json`): rebuilding the 858 binaries of
the truth shard and refitting the full window, 5 give a different anomaly decision when the refit's seed is perturbed
at the 1e-7 level (in one, dchi2 is 72,931 at the generator's seed and 0.7 at a perturbed one, so a single-lens model
fits a curve labelled NonPSPL); 0.5% of truncated presentations are similarly unstable. The released labels carry
this at that rate; it is disclosed, not fixed (changing the refit would change the released training labels).

Unaffected: the shipped weights, every paper artifact under `paper/results/`, and the manifest-hashed
validation results. The ESPLMag2 steps and the first-season mask affect only the post-submission
fine-tunes listed above. The quantised 7.2-d onset is different: it is shared by the shipped model's
training labels (through the truncation augmentation; the paper states the 7.2-day label resolution),
and the post-submission fine-tunes inherit it. Each is disclosed where its numbers are quoted
(paper/REVISION.md section 1.5).

A label-policy fragility found during the check, older than all of this: the single-start PSPL refit
in `pipeline.assemble._pspl_refit_dchi2` is numerically chaotic for rare borderline events. In-house
cascade seed 304033 flips between PSPL (dchi2 4.5) and NonPSPL (dchi2 3,665) under a 1e-14 mag
perturbation of its noise-free curve (4 of 60 trials), so the first event simulated in a fresh process
can replay with a different label. It is about one event per thousand and is disclosed, not fixed:
changing the refit would change the released training labels.

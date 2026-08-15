# Validation and reproducibility artifacts

Two models are exercised in different regimes. **BinML v1** (6-class, 3-band) is a synthetic
specialist for Roman's dense 15-min cadence. The **legacy 3-class single-band model**
(`binml.legacy`) is separately checked on a small set of real, sparse, ground-based light curves;
those checks have no frozen result artifact and are not counted as validation.

The released 15-min, one-season cadence is a legacy Cycle-7-inspired design. Current GBTDS
planning uses approximately 12-min F146 sampling, 66-s exposures, staggered colour visits, and
multiple seasons. The released simulator also has known F087/F213 zeropoint, F087 saturation, and
colour-background discrepancies from the current calibration; the checkpoint has not been
retrained with corrected values.

## Scripts

- **`cadence_robustness.py`** — subsamples dense simulated events across point densities and
  measures anomaly recovery. The script prints an exploratory small-sample sweep; no frozen result
  artifact from that sweep is shipped, so its exact printed rates are not publication results.

- **`real_data_validate.py`** — fetches selected OGLE-IV events and runs BinML v1 in single-band
  mode. It is an exploratory, network-dependent diagnostic. The repository ships neither the
  downloaded inputs nor a frozen output artifact; do not quote its console output as a validated
  performance result.

- **`kmtnet_validate.py`** — fetches selected KMTNet events for another exploratory single-band
  diagnostic. The tracked synthetic matched-visit experiment (`gap_matched_result.json`) shows
  that nightly windowing can be much more damaging than uniform thinning at the same visit count.
  That controlled simulation is consistent with cadence gaps contributing to the real-data
  failure, but it does not isolate cadence from passband, photometric-systematic, or population
  shifts in the live KMTNet check. No frozen KMTNet output is shipped.

- **`microlia_compare.py`** — MicroLIA (Godines+2019) on the same real events. **Health warning:**
  MicroLIA is bit-rotted (PyPI 2.8.1 missing its Mira simulator; GitHub main won't import — dead
  RRLyrae template URL), so this monkeypatches two simulators. No frozen comparison artifact is
  shipped; treat it as a debugging aid rather than a benchmark.

## Cascade / streaming artifacts

The streaming numbers in the paper come from one stored scan of a frozen event sample, reduced
under several policies. The scan reveals F146 in 0.5-day steps. Its onset mask is computed from
the injected, noise-free binary-versus-PSPL deviation, so onset is truth-informed rather than an
observable quantity available to a survey broker. Regenerate in this order:

```
python validation/cascade_reproduce.py --n 1000 --promote   # redraw the frozen sample (rarely)
python validation/cascade_trace.py                          # full probability traces + onset mask
python validation/cascade_reduce.py                         # every reported statistic
```

- **`cascade_events.json`** — the frozen 1,000-event sample (seeds + the originally published
  first crossings). Also the regression fixture: `cascade_reduce.py` refuses to publish a
  reduction whose detection statuses differ from these.
- **`cascade_trace.npz`** — P(NonPSPL) and argmax at every 0.5 d cut, for F146-only and all-three-
  band revealing, plus the anomaly-detectability mask on the same grid. It records useful run
  metadata, including the checkpoint hash and dependency versions, but it was produced from a
  dirty tree and records neither a source hash nor the diff. It is not complete source provenance.
- **`cascade_reproduce_result.json`** — the reduction. Read directly and fail-closed by
  `paper/make_macros.py`.

Two things the reduction makes explicit because they move the headline more than the model does:
the **alert policy** (grid, persistence, band set, threshold vs argmax) and the **onset
definition** (first-detectable vs persistently-detectable vs the generator's 7.2 d grid).
At the frozen F146 operating point, 89.0% of eligible events are detected in-season; among
nonpremature detections the median lag is +5.0 days. This supports partial-season triage, not
demonstrated response during a short planetary perturbation.

## Ablations

- **`modal_labelling_ablation.py`** → `labelling_ablation_result.json` — the label-only ablation
  with the class-balancing weights pinned across arms (`--weight-labels observational`), every arm
  scored against **both** ontologies, and a paired bootstrap on the macro-F1 gap.
- **`modal_ablations.py`** → `ablations_result.json` — the original four-arm run. Its cascade arm is
  still the reported one; its labelling arms are superseded (they confounded labels with weighting).
- **`modal_cascade_matched.py`** → `matched_traces.npz`, reduced by **`cascade_matched_reduce.py`**
  → `cascade_matched_result.json` — an **exploratory finite-sample risk–coverage comparison** on
  400 paired events. Each arm's threshold is selected to attain a detection count on the same
  events used to score premature crossings. Conditional McNemar values describe these 400 events;
  they are not confirmatory population-level p-values. A confirmatory experiment must select
  thresholds on a disjoint calibration set. The stored matched trace also lacks code and
  checkpoint hashes, so the original remote run does not have complete provenance.

## Reproducing the Modal experiments

The current `modal_labelling_ablation.py` keys caches and checkpoints by source/config hashes and
writes a detailed manifest. The published `labelling_ablation_result.json` predates a verified run
of that mechanism: its source hash was repaired after execution rather than emitted by the run.
Do not present the existing artifact as proof that stale-artifact protection was exercised.
Training checkpoints live in the Modal volume `binml-labelling-v2` and are not part of the public
release. A provenance-complete replacement requires rerunning the current script and preserving
its emitted manifest and checkpoint hashes.

## Artifact-integrity checks

`paper/validate_artifacts.py` checks the frozen NumPy arrays, every load-bearing file named in
`paper/results/MANIFEST.json`, the identity of the shipped checkpoint, and byte-exact regeneration
of both deterministic trace reductions. This protects the distributed artifact bundle from silent
editing. It does not repair the disclosed provenance gaps in the original remote executions.

## Real-data status

No frozen, hashed real-data performance result is shipped for either model. The selected OGLE and
KMTNet scripts above are useful transfer diagnostics, but their samples, downloaded inputs, and
outputs are not a reproducible population evaluation. The paper therefore does not count the
previously quoted event totals as validation.

## Conclusion

BinML v1 remains a simulation-evaluated specialist for its legacy dense, single-season schedule.
Real Roman photometry will be necessary for transfer testing, but operational validation also
requires corrected photometric assumptions, the current multi-season design, and a frozen
evaluation protocol with durable event-level artifacts.

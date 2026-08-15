# Legacy — the 3-class (Flat / PSPL / Binary) model

The original BinML was a **3-class** classifier — **Flat** (no event), **PSPL** (single-lens),
**Binary** (planetary/binary lens) — for **single-band** Roman light curves. It is superseded by
the 6-class multi-band model, but is documented here and preserved for reproducibility and
provenance. This is how it was built.

> **Note on paths.** The scripts named below (`pipeline/simulate.py`,
> `pipeline/select_subset.py`, `pipeline/train_modal.py`) belong to the legacy 3-class
> codebase and are **not present in this repository**. They are recorded here for provenance;
> the current 6-class pipeline lives in `pipeline/` under different names
> (`run_shard.py`, `cache.py`, `to_memmap.py`, `train.py`, `evaluate.py`).

## Where the code lives
- **Inference package:** still importable at `binml.legacy` (`from binml.legacy import Classifier`).
  Weights `binml/weights/binml_base.pt` and `binml/weights/binml_finetuned.pt` are bundled.
- **Research pipeline (simulate/train/evaluate):** removed from the working tree in the v1.0
  clean-up; recover it from git at commit **`6a6ccb7`** (or earlier), files
  `pipeline/{simulate,select_subset,train,train_modal,model,evaluate}.py`,
  `pipeline/{analysis,curricula}/`.

## Architecture
A **strictly causal CNN-GRU**: two input channels — magnification `A = 10^(0.4·(m_base − mag))`
(baseline `A=1`) and `delta_t` (days since the previous valid point) — a causal
depthwise-separable convolution stack (left-pad only), a unidirectional GRU, masked attention
pooling, and a hierarchical head. Causality was the point: the early-detection /
probability-evolution analyses were valid because truncated prefixes never saw future timesteps.
(The 6-class model replaced this with a multi-band conv-stem + transformer.)

## Data format (compact HDF5 v4.2.0)
Single-band shards with row-aligned `flux` (magnification A, `0.0` = unobserved), `delta_t`,
`labels`, `m_base`, a packed observation mask, a global `params` struct
(`t0, tE, u0, q, s, rho, alpha, m_base, peak_magnification, snr, anomaly_dchi2, max_anomaly`),
and a shared `time_grid`. The load-bearing quantity was **`anomaly_dchi2`** — the χ² of the true
binary against a matched single-lens fit — which drove subset selection and all
detectability-conditioned evaluation.

## Procedure (how the weights were produced)
1. **Simulate** a large 3-class population on Roman cadence (6912 points at 15-min over 72 days),
   binaries via VBBinaryLensing — `pipeline/simulate.py`.
2. **Select** a detectability-aware training subset (weighted toward detectable anomalies) —
   `pipeline/select_subset.py`.
3. **Train** the base model with a single-GPU streaming trainer — `pipeline/train.py`, orchestrated
   on a Modal L4 GPU via `pipeline/train_modal.py`.
4. **Fine-tune** with a warm start (`train.py --init-weights`) on a bump-rich, low-mass-ratio
   (hard planetary) set. Historical summaries report that a second, more aggressive round collapsed
   and was discarded. The current repository does not ship a provenance-complete matched comparison
   against additional base training, so this is lineage, not a causal recipe claim.

## Historical results (not reproduced by the current release)

The following values were recorded for the legacy model, but the current tree does not contain a
frozen, hashed evaluation artifact from which to regenerate them. Treat them as historical context,
not current validation.

| model | accuracy | Flat | PSPL | Binary (raw pop.) | planetary recall (q∈1e-4..1e-3) |
|---|---|---|---|---|---|
| base | 64.3% | 100% | 80.0% | 52.2% | 41.2% |
| **fine-tuned** (shipped) | 67.2% | 100% | 70.9% | 62.3% | 57.4% |

Historical analysis reported that binary recall rose with anomaly Δχ². Prior ad-hoc OGLE checks
are not counted as validation: no frozen, hashed result artifact or complete protocol is shipped.

## Why it was superseded
The 3-class, single-band model could not use colour (a real discriminator against variables),
had no variable-star classes (contaminants had nowhere to go but the three lensing classes), and
labelled/flagged without the detectability-conditioned partial-season objective. The 6-class multi-band
model addresses all three. See [architecture.md](architecture.md) and [evaluation.md](evaluation.md).

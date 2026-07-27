# Legacy — the 3-class (Flat / PSPL / Binary) model

The original BinML was a **3-class** classifier — **Flat** (no event), **PSPL** (single-lens),
**Binary** (planetary/binary lens) — for **single-band** Roman light curves. It is superseded by
the 6-class multi-band model, but is documented here and preserved for reproducibility and
provenance. This is how it was built.

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
   (hard planetary) set — moved the needle far more than additional base training. A second, more
   aggressive fine-tune round **collapsed** (drove PSPL recall to ~18% by calling nearly everything
   Binary) and was discarded; round 1 shipped. This is why the fine-tune schedule is conservative.

## Results (full 1,000,000-event held-out evaluation)
| model | accuracy | Flat | PSPL | Binary (raw pop.) | planetary recall (q∈1e-4..1e-3) |
|---|---|---|---|---|---|
| base | 64.3% | 100% | 80.0% | 52.2% | 41.2% |
| **fine-tuned** (shipped) | 67.2% | 100% | 70.9% | 62.3% | 57.4% |

Binary recall rose monotonically with anomaly Δχ² from ~11% (no detectable signal) to ~85%
(strong anomaly); at Δχ²≥300, ~32% of binaries were physically indistinguishable from PSPL and
the detectable-only binary recall was ~78%. Validated on real OGLE-IV events (correctly flagged
caustic-crossing binaries OGLE-2014-BLG-0289, OGLE-2013-BLG-0578).

## Why it was superseded
The 3-class, single-band model could not use colour (a real discriminator against variables),
had no variable-star classes (contaminants had nowhere to go but the three lensing classes), and
labelled/flagged without the detectability-conditioned, real-time cascade. The 6-class multi-band
model addresses all three. See [architecture.md](architecture.md) and [evaluation.md](evaluation.md).

# Data-leakage audit — 2026-07-08

Adversarial audit of the model (`model.py`) and training/eval pipeline (`train.py`,
`evaluate.py`) for data leakage. Six dimensions, each finding independently verified
(refute-or-confirm) before counting.

## Verdict: model is clean. One minor, benign pipeline leak found and fixed.

### Confirmed SAFE (no leakage)
- **Attention pooling** (production path): masks padded keys with additive `-inf` before
  softmax (`model.py` `FlashAttentionPooling.forward`). No length/padding leak.
- **Causality**: GRU is unidirectional; `CausalConv1d` is left-pad only; residual is
  causal. The early-detection and probability-evolution analyses in `evaluate.py` are
  therefore valid — truncated prefixes never see future timesteps.
- **Train/val split**: `stream_train_val_split` / `train_test_split` produce a disjoint,
  complete partition; the block-shuffle sampler is a pure permutation of train positions;
  `drop_last` trims only train.
- **Input features**: the model forward consumes ONLY `flux` + `delta_t` (+ `lengths`).
  `m_base` and physical params (`q, s, rho, anomaly_dchi2, ...`) are loaded in
  `evaluate.py` for post-hoc analysis only, never fed to the network. The cadence mask
  (`flux==0` pattern) and `delta_t` gap structure are class-independent by construction
  (`simulate.py`).
- **Batch-level**: the only cross-sample op is `BatchNorm1d`, correctly frozen by
  `model.eval()` at every inference site → not exploitable at eval.
- **Eval stats**: `evaluate.py` reuses the frozen checkpoint stats and never recomputes
  normalization on the eval data.

### Confirmed LEAK (minor) → FIXED
- **Normalization statistics were fit on the FULL dataset (train+val) before the split**,
  in both the streaming production path (`compute_streaming_statistics`, called at the
  `--stream` branch of `main`) and the legacy in-memory path (`compute_robust_statistics`
  in `load_and_split_data`). Validation rows contributed to the 4 global mean/std scalars
  used to standardize training inputs (and reused at eval via the checkpoint).
- **Severity: minor / benign.** The leaked quantity is 4 global scalars aggregated over
  billions of observation points with ZERO per-sample label correlation, so the model
  gains no discriminative shortcut and per-sample val/test metrics are not measurably
  inflated. It is textbook fit-before-split leakage in mechanism, negligible in magnitude.
- **Fix**: both `compute_streaming_statistics` and `compute_robust_statistics` now accept
  `train_idx` and fit on TRAIN rows only; call sites split first, then fit. Verified:
  train-only stats differ from full-data stats, and `train_idx=all` reproduces the legacy
  result (backward compatible).

### Defensive hardening (not leaks in production)
- Unmasked mean-pool fallback now RAISES instead of silently pooling over padding
  (unmasked mean over compacted+normalized sequences would leak valid-length — the
  original "mean pooling is cheating" concern). Masked mean is length-invariant and safe.
- Attention pooling documents that `lengths` must be passed for compacted inputs
  (`lengths=None` is valid only for genuinely unpadded sequences).
- Noted (not fixed, benign): BatchNorm running stats accumulate over padded timesteps
  during training — training-time cross-sample dependence only, frozen and not exploitable
  at eval.

## Impact on existing results
NONE. `best.pt`, the round-1 fine-tuned `ckpt_ft`, and the round-2 fine-tune were trained
with the pre-fix (leaky-but-benign) stats. Because the leak carries no label correlation,
**no retraining is required and no reported number is invalidated.** The train-only stats
apply to future training runs.

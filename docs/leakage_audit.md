# Data-leakage audit

## v5 (current pipeline) — how train/test disjointness is guaranteed

The v5 data (`pipeline/sim_v5/`) makes leakage structurally hard:

- **Disjoint by shard index.** Each shard is its own RNG stream: `seed = seed_base + shard*7919`.
  Training used shards **0–89**, the independent test set shards **90–149** — different streams,
  zero event overlap. (The `cache` and `cache2` S3 prefixes are the *same* events at a given
  index — a re-bin, not a second population — so independence is by **index**, never by prefix.)
- **Unseen-parameter stress test.** The 12.9M-event generalisation set uses seed bases **≥900M**,
  ≥1M apart per regime, versus training's base `20260720` (max seed ~23.4M). Different PCG64
  streams → parameter tuples the model never saw. `--seed-base` exposes this.
- **The network never sees the labels' inputs.** The model forward (`model_v5.py`) consumes only
  the `feat`/`frac` tokens + per-band presence. The `params` table (including `q, s, dchi2_*`,
  and `t_anom`) is used for **labelling and post-hoc analysis only**, never fed to the network.
- **No fit-before-split statistic.** v5 uses a fixed input scale (`MAG_SCALE = 1.0`), so the
  normalization-leak class below (a 3-class-pipeline issue) does not exist in v5.

Evaluation is detectability-conditioned throughout (see [`evaluation.md`](evaluation.md)); the
`keep_prob` reweighting is applied to precision/purity only, never recall.

---

# Legacy 3-class audit — 2026-07-08

*(The audit below concerns the original CNN-GRU package `binml/` — `model.py`, `train.py`,
`evaluate.py`. Retained for the record; superseded by the v5 section above.)*

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

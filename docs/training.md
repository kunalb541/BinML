# Training pipeline

This document describes the end-to-end pipeline used to produce the BinML classifier
weights, from raw simulation to the shipped fine-tuned model. The stages are:

1. **Simulate** a large event population with `pipeline/simulate.py`.
2. **Select** a detectability-aware training subset with `pipeline/select_subset.py`.
3. **Train** the base model with the single-GPU streaming trainer `pipeline/train.py`.
4. **Orchestrate** the whole thing on Modal (L4 GPU) with `pipeline/train_modal.py`.
5. **Fine-tune** the base checkpoint with a warm start (`train.py --init-weights`) on a
   bump-rich, targeted set to recover the hard planetary (low mass-ratio) regime.

The model itself is a causal CNN-GRU with attention pooling and a hierarchical head; see
[`docs/`](.) and `pipeline/model.py` for architecture details. This document covers only how
the weights are produced.

All commands below are shown relative to the `pipeline/` directory unless otherwise noted.

---

## 1. Simulation — `simulate.py`

`simulate.py` generates the three classes — Flat (no event), PSPL (single-lens), and Binary
(planetary/binary lens) — on the Roman cadence (6912 points at 15-minute sampling over a
72-day season) using `VBBinaryLensing` for the binary magnification. Output is a compact HDF5
v4.2.0 shard: row-aligned magnification `flux` (A, with `0.0` marking unobserved points),
`delta_t`, `labels`, `m_base`, a packed observation mask, a global `params` struct
(`t0, tE, u0, q, s, rho, alpha, m_base, peak_magnification, snr, anomaly_dchi2, max_anomaly`),
and the shared `time_grid` stored once.

The physically important quantity written per event is **`anomaly_dchi2`** — the chi-squared
of the true binary against a matched single-lens fit — i.e. the *detectability* of the binary
anomaly. This drives subset selection (Stage 2) and all honest, detectability-conditioned
evaluation downstream.

### Example: one balanced shard

```bash
python simulate.py \
  --n_flat 25000 --n_pspl 25000 --n_binary 25000 \
  --binary_preset baseline \
  --output ../data/shard_000.h5 \
  --seed 42
```

Key arguments:

| Argument | Purpose |
| --- | --- |
| `--n_flat / --n_pspl / --n_binary` | Events per class in this shard. |
| `--binary_preset` | `baseline`, `distinct`, `planetary`, or `stellar` — controls the binary parameter ranges. |
| `--q_min / --q_max` | Override the preset mass-ratio range to target a specific `q` regime (used to build fine-tuning sets). |
| `--force_caustic` | Require a caustic signature, favouring detectable bumps (used for bump-rich fine-tuning shards). |
| `--oversample` | Oversample factor to absorb simulation failures. |
| `--num_workers / --seed` | Parallelism and reproducibility. |

Production runs generate roughly **10M events** across many shards. Each shard is
self-contained, so simulation parallelizes trivially by varying `--seed` and `--output`.

---

## 2. Detectability-aware subset — `select_subset.py`

Training on the full ~10M population is neither necessary nor efficient: most binaries near
the class boundary carry little or no detectable anomaly. `select_subset.py` scans the source
shards' `params` and builds a **~300k** training subset, stratifying binaries by
`anomaly_dchi2` (and `q`) so the training signal is concentrated where the anomaly is actually
learnable, while keeping Flat and PSPL representation balanced.

```bash
python select_subset.py \
  --shards ../data \
  --out ../data/subset \
  --target 300000 \
  --shard-size 25000 \
  --seed 42
```

Key arguments:

| Argument | Default | Purpose |
| --- | --- | --- |
| `--shards` | — | Source shards: directory, glob, or comma-list. |
| `--out` | — | Output directory for the subset shards. |
| `--target` | `300000` | Total events in the subset. |
| `--shard-size` | `25000` | Events per output shard. |
| `--flat-frac / --pspl-frac / --binary-frac` | preset | Class fractions in the subset. |

The output is a set of `subset_*.h5` shards, in the same compact v4.2.0 format, ready for the
streaming trainer.

---

## 3. Base training — `train.py`

`train.py` is a single-GPU streaming trainer built around the subset shards:

- **`StreamingShardDataset`** reads events on demand from the HDF5 shards (no full in-memory
  load), paired with a **block-shuffle sampler** (`--block-shuffle N`) that randomizes within
  blocks to keep reads sequential-ish while still de-correlating batches.
- **Hierarchical loss**: a stage-1 deviation logit (is there any deviation from flat?) and a
  stage-2 binary-vs-PSPL logit, each with its own BCE loss, plus an auxiliary 3-class head.
  The stage/aux weights are exposed as `--stage1-weight`, `--stage2-weight`, `--aux-weight`.
- **AMP** mixed precision (`--use-amp`), **cosine schedule with warmup**
  (`--epochs`, `--warmup-epochs`), **early stopping** (`--early-stop-patience`), **atomic
  checkpoints**, and **`--resume auto`** to pick up the latest checkpoint in the output dir.
- **Leakage-safe normalization**: the four global normalization statistics are fit on the
  **train split only**, then frozen into the checkpoint and reused at evaluation. (See
  `docs/leakage_audit.md`.)

The shipped base model uses `d_model=64`, `n_layers=4` (~130K parameters). Note that the
argparse default for `--d-model` is `128`; the production configuration passes `64`
explicitly, as below.

### Base training command (production configuration)

```bash
python train.py \
  --stream --block-shuffle 256 --resume auto \
  --data ../data/subset --output ../results/ckpt \
  --hierarchical --use-aux-head --attention-pooling --use-amp \
  --d-model 64 --n-layers 4 \
  --batch-size 128 --accumulation-steps 2 \
  --num-workers 6 --prefetch-factor 6 \
  --stage1-weight 0.5 --stage2-weight 2.0 --aux-weight 0.5 \
  --epochs 40 --warmup-epochs 3 \
  --early-stop-patience 8
```

The best checkpoint (by validation loss) is written as `best.pt` in a run subdirectory of
`--output`. This is the artifact promoted to base weights.

**Base model result** (full 1,000,000-event held-out evaluation): overall accuracy
**64.3%**; recall Flat **100%** / PSPL **80.0%** / Binary **52.2%**; binary precision
**81.7%**.

---

## 4. Modal orchestration — `train_modal.py`

`train_modal.py` runs the entire pipeline on Modal against a single **L4 GPU** and a
persistent volume, so simulation output, checkpoints, and evaluation artifacts all live on
one durable volume across steps. Each function is invoked with `modal run`:

```bash
modal run train_modal.py::train
modal run train_modal.py::finetune --ckpt /vol/ckpt/<run>/best.pt
modal run train_modal.py::eval_big --ckpts /vol/ckpt/<run>/best.pt
```

The default local entrypoint (`modal run train_modal.py`) runs `download` then `train`.

| Function | Role |
| --- | --- |
| `download` | Fetch the training data onto the persistent volume. |
| `train` | Copy subset shards to local container disk (random reads off the network volume stall the loader), then run the base `train.py` command above and commit checkpoints to `/vol/ckpt`. |
| `finetune` | Warm-start fine-tune from a checkpoint via `--init-weights` (see Stage 5). Re-chunks the bump-rich shards to `(256, 6912)` for fast block-shuffle reads, then trains on a fresh low-LR schedule into `/vol/ckpt_ft`. |
| `evaluate` | Run `evaluate.py` on a checkpoint over one subset shard (summary JSON, calibration, per-parameter metrics, plots). |
| `eval_val` | Evaluate on the held-out validation split. |
| `eval_big` | Streaming evaluation over the full **1M** held-out set (the source of the headline numbers). |
| `eval_presets` | Evaluate against the per-preset test sets. |
| `eval_lsst` | Zero-shot evaluation under LSST-like resampled cadence. |
| `eval_detectability` | Detectability-conditioned evaluation (binary recall vs `anomaly_dchi2`). |
| `download_1m` | Stage the 1M evaluation set on the volume. |

A performance note baked into `train`: `simulate.py` writes large `(10000, 6912)` HDF5
chunks, so per-batch random reads decompress a quarter-gigabyte at a time. Both `train`
(copy-to-local-disk) and `finetune` (re-chunk to `(256, 6912)`) exist to avoid that I/O
stall — roughly a 40% throughput improvement.

---

## 5. Warm-start fine-tuning — the planetary regime

### Philosophy: targeted warm-start beats more base training

The base model plateaus on the **hard low-mass-ratio (planetary) regime** — binaries with
`q` in roughly `1e-4..1e-3`, whose caustic anomaly is short and easily missed. More base
training does **not** move this: the population is dominated by easier events and the extra
epochs simply re-learn what the model already knows.

The lever that works is a **targeted warm start**. Fine-tuning loads *only the weights* from a
base checkpoint (`train.py --init-weights <ckpt>`, distinct from `--resume`, which would also
restore the optimizer and schedule) and trains on a fresh, low learning-rate schedule over a
**bump-rich, planetary-targeted** dataset — shards simulated with `--force_caustic` and a
narrowed `--q_min/--q_max` so the model spends its fine-tuning budget almost entirely on
detectable planetary anomalies.

### Building the fine-tuning set and running the fine-tune

```bash
# bump-rich, planetary-targeted shards
python simulate.py \
  --n_flat 5000 --n_pspl 5000 --n_binary 40000 \
  --binary_preset planetary --q_min 1e-4 --q_max 1e-3 --force_caustic \
  --output ../data/finetune/ft_000.h5 --seed 100
```

Locally, the fine-tune is a `train.py` run identical to the base command except that
`--resume auto` is replaced by `--init-weights` (pointing at the base `best.pt`), with a fresh
low LR, short warmup, and a separate output directory:

```bash
python train.py \
  --stream --block-shuffle 256 \
  --init-weights ../results/ckpt/<run>/best.pt \
  --data ../data/finetune --output ../results/ckpt_ft \
  --hierarchical --use-aux-head --attention-pooling --use-amp \
  --d-model 64 --n-layers 4 \
  --batch-size 128 --accumulation-steps 2 \
  --num-workers 6 --prefetch-factor 6 \
  --stage1-weight 0.5 --stage2-weight 2.0 --aux-weight 0.5 \
  --lr 2e-4 --epochs 12 --warmup-epochs 1 \
  --early-stop-patience 5
```

On Modal the same step is a single call, which additionally re-chunks the shards for fast
reads:

```bash
modal run train_modal.py::finetune --ckpt /vol/ckpt/<run>/best.pt --lr 2e-4 --epochs 12
```

### Round 1 — shipped

The first fine-tuning round is the **shipped fine-tuned model** (`Classifier()` default).
On the full 1M held-out set it improves overall accuracy to **67.2%** with recall Flat
**100** / PSPL **70.9** / Binary **62.3**. On the targeted planetary regime
(`q` in `1e-4..1e-3`), binary recall rises from **41.2% to 57.4%** — the intended effect.
The trade-off is a modest drop in raw PSPL recall (80.0% → 70.9%), consistent with the model
becoming more willing to call a subtle deviation "binary".

### Round 2 — collapsed, discarded

A second, more aggressive fine-tuning round **collapsed**: PSPL recall fell to ~**18%** and
the model called essentially everything Binary. This round was **discarded**. Round 1 is the
shipped fine-tuned model. The lesson is that fine-tuning on a bump-rich set is a precise lever,
not a monotone one — pushing harder past round 1 destroys the stage-2 balance rather than
improving the planetary regime further.

---

## Reading the results honestly: detectability conditioning

The raw population Binary numbers above are the pessimistic view. A large fraction of
simulated binaries carry no detectable anomaly and are **physically indistinguishable from a
PSPL** — the network cannot, and should not, be expected to separate them.

Binary recall therefore rises monotonically with `anomaly_dchi2`, from ~**11%** (no detectable
signal) to **85%** (strong anomaly). At a detection threshold of **Δχ² ≥ 300**, about
**32%** of binaries are physically PSPL-like, and the detectable-only binary recall is
~**78%** — versus the ~62% raw population number. About **60%** of the binaries "missed" at
that threshold are physically PSPL-like, i.e. not model errors.

**Report binary performance conditioned on detectability, never the raw population number.**
Use `eval_detectability` (Modal) or the detectability-conditioned path in `evaluate.py` to
reproduce these curves.

---

## Summary

| Stage | Script | Output |
| --- | --- | --- |
| Simulate ~10M events | `simulate.py` | compact HDF5 v4.2.0 shards |
| Detectability-aware subset (~300k) | `select_subset.py` | `subset_*.h5` |
| Base train (streaming, single GPU) | `train.py` | base `best.pt` (acc 64.3%) |
| Orchestrate on L4 GPU | `train_modal.py` | checkpoints + eval on volume |
| Warm-start fine-tune (planetary) | `train.py --init-weights` | fine-tuned `best.pt` (acc 67.2%, shipped) |

The shipped weights in the `binml` package are the **round-1 fine-tuned** model
(`Classifier()`), with the balanced **base** model available as `Classifier("base")`.

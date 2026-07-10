# Single-GPU Training + Free-GPU Deployment Plan

Date: 2026-07-05. Produced by a verified multi-agent audit of `code/train.py` / `code/model.py`
(all line refs checked against the actual code) plus current-terms free-GPU research. See also
[claude_audit_response.md](claude_audit_response.md).

## TL;DR

`train.py` was built for a 40-A100 SLURM run. On one T4-class GPU it doesn't just run slow — it
**OOMs before the first training step**, because it loads full `flux`+`delta_t` arrays into RAM and
(on the DDP `local_rank==0` path) also `shutil.copy2`s the whole file into `/dev/shm`, doubling
residency. Fix that one thing and the rest are tuning. Recommended platform: **Kaggle Notebooks**
(P100 16 GB, 200 GB persistent Datasets), training a **detectability-selected subset** (300k → 1M →
3M) so the data fits free-tier disk and only that subset egresses from S3 once.

## 1. Single-GPU training optimizations (adversarially verified)

| # | Change | Expected gain | Risk | 1-GPU-specific |
|---|--------|--------------|------|----------------|
| 1 | Replace `/dev/shm` full-array load with a lazy multi-shard HDF5 `Dataset` (**core blocker**) | OOM-on-load → O(batch) RAM; unblocks data > RAM | med | Yes |
| 2 | Detach loss tensors; stop per-batch `.item()` syncs in `compute_hierarchical_loss` | ~1.2–2× on this sync-bound tiny model | low | Yes |
| 3 | `--num-workers 4 --prefetch-factor 4` (default 0 → synchronous inline loading) | 1.3–2× when data-bound | low | Yes |
| 4 | `--batch-size 256–512` (default 64; model is tiny, d_model=16) | 1.5–3× samples/sec | low | Yes |
| 5 | Streaming/sampled normalization stats, not `f['flux'][:]` | removes 2nd startup OOM | low | Yes |
| 6 | `--use-amp` (fp16+scaler correct on T4) | fits 16 GB; modest | low | Yes |
| 7 | `collate_fn` pads to batch-max length, not fixed 6912 | large on selected subsets (GRU is O(T)) | med | Yes |
| 8 | `--val-every 2–3` gate on validation | ~7–13% wall-clock | low | Partly |
| 9 | Atomic checkpoint write (`tmp`+`os.replace`) | preemption safety | low | matters on preemptible |
| 10 | `amp_dtype = bf16 if supported else fp16` | **0 on T4**; prevents fp16-no-scaler NaN on Ampere+ | low | No |
| 11 | `torch.set_float32_matmul_precision('high')` | **0 on T4** (Turing has no TF32) | low | No |
| 12 | `torch.compile` | **keep OFF for bring-up**; 0–15% on T4, GRU opaque to Dynamo | med | No |
| 13 | Temporal downsampling before GRU | 1.8–3.5× **but risks binary recall** | high | Yes |
| 14 | Tail-flush partial accumulation batch | correctness only, and only if accum>1 | low | Yes |

**Skip/deprioritize for the first run:** #10–#11 are zero-impact on a T4 (Turing has no TF32; AMP
already runs fp16+scaler) — apply #10 only as forward-compat hardening if you borrow an Ampere/L4
box. #12 buys little (the cuDNN GRU at T=6912 is opaque to Dynamo; its default `reduce-overhead`
mode uses CUDA graphs most likely to break on 16 GB). #13 is the only change that can *degrade the
science metric* — defer behind an opt-in `temporal_stride` flag and A/B against `anomaly_dchi2`.
#14 is a no-op at the default `accumulation_steps=1`.

### High-priority items (concrete)

**#1 — Lazy multi-shard HDF5 Dataset (the unblocker).** `_load_arrays_from_file` (train.py:460–466)
does `f['flux'][:]`, `f['delta_t'][:]`, `f['labels'][:]`; `load_data_to_shared_memory` (called
unconditionally by `create_dataloaders`, train.py:767) adds a second full copy into `/dev/shm` —
pure overhead on one GPU, and it hard-raises `RuntimeError` when `/dev/shm` is absent (macOS), so
you can't even smoke-test locally. Minimal fix: a no-shm fast path gated on
`not is_ddp or not Path('/dev/shm').exists()` that reads arrays directly and returns `shm_path=None`
(make `cleanup_shared_memory` a no-op on `None`). **Read-amplification trap:** the on-disk chunk
shape is `(10000, 6912)` (simulate.py:1752), so a single random-row `__getitem__` decompresses the
whole ~276 MB lzf chunk. For 1M→3M, don't ship pure random-access; use a **block/chunk-aligned
sampler** (shuffle 10 000-row blocks) *or* one-time repack the selected subset with
`chunks=(64,6912)`. Add `__getstate__` dropping open h5py handles (not fork-safe) so each worker
opens its own. Bit-exact vs stored arrays; touches no AMP/loss/normalization semantics.

**#2 — Kill per-batch host↔device syncs in the loss (best gain-per-line).**
`compute_hierarchical_loss` returns Python floats via `.item()` (train.py:887, 912–916); the epoch
loop adds them onto GPU accumulators every batch (train.py:959–961). Each `.item()` forces a
`cudaStreamSynchronize` on every batch (~5/step), defeating prefetch overlap on this launch-bound
model. Return detached 0-dim GPU tensors, keep `+=` as tensor accumulation, and gate `.item()`
reads behind the existing `batch_idx % PROGRESS_UPDATE_FREQ` block (train.py:998). Loss math
unchanged.

**#3/#4 — Feed the GPU.** `DEFAULT_NUM_WORKERS=0` (train.py:148) runs decompress+normalize inline;
`DEFAULT_BATCH_SIZE=64` (train.py:145) under-fills 16 GB for a d_model=16 model. Launch flags, not
code: `--num-workers 4 --prefetch-factor 4 --batch-size 256`. `pin_memory`/`non_blocking` H2D are
already correct (train.py:825/945).

**#5 — Stats without a 2nd OOM.** `compute_robust_statistics` (train.py:596–603) does `f['flux'][:]`
on every shard *before* subset selection. Rewrite to block-stream (10 000-row blocks, accumulate
n/sum/sumsq over `flux!=0`) or sample ~100k selected rows; cache to `stats.json`. Correctness-neutral.

**Good news (verified):** GRU gradient checkpointing is already OFF by default (model.py:297) — correct
for a 16 GB GPU with headroom. And a plain `python train.py` (no torchrun) cleanly takes the
single-process else-branch — no DDP wrapping, no `DistributedSampler`, nothing hangs (train.py:1160–1219).

## 2. Recommended free-GPU platform

AWS is out for GPU, but the 292 GB dataset is in S3 (eu-central-1) — so the choice is "where to host
a selected subset + get a stable T4-class GPU with resume."

| Rank | Platform | GPU / VRAM | Session cap | Quota | Persistent data disk | Verdict |
|------|----------|-----------|-------------|-------|----------------------|---------|
| **Primary** | **Kaggle Notebooks** | P100 16 GB *or* 2×T4 16 GB | 12 h | ~30 GPU-h/wk | **200 GB Kaggle Datasets** (read-only mount, persists) | **Best fit** |
| Backup | Modal | serverless A10G/L4/A100 (pick) | none | $30/mo recurring credit | `modal.Volume` (persistent, no ingress fee) | Strong, but serverless not interactive |
| Cheap-not-free | GCP | T4/L4 (~100 T4-h from $300 credit) | none | $300/90d | GCS | **GPU blocked on non-billable trial** — must upgrade to paid billing + often a quota request |
| Marginal | Colab free | T4 16 GB | ~12 h, ~90-min idle kill | soft ~15–30 GPU-h/wk | ephemeral; Drive 15 GB free | prototyping only |

**Primary: Kaggle.** Only free option whose *persistent* storage (200 GB per Dataset, raised from
100 GB in 2026) holds a 1M (~29 GB) or 3M (~87 GB) subset mounted read-only at `/kaggle/input`,
leaving the ~20 GB writable `/kaggle/working` for checkpoints. The 300k→1M→3M ladder fits under
30 GPU-h/week (each rung independent, straddles weekly resets); 12 h session cap is fine given
checkpoint-resume. PyTorch+CUDA preinstalled. Discipline: run rungs in "Save & Run All" (commit)
mode (interactive dies on tab close); a single P100 is simpler than 2×T4 for bring-up.

**Backup: Modal.** $30/mo recurring credit on real serverless GPUs, per-second billing, persistent
Volumes, no ingress fees — good for scripted batch training. Catch: serverless, not SSH; long
interactive dev on big GPUs burns $30 fast. Use if you exceed Kaggle's 30 h/week.

**GCP:** viable *within* the $300 credit (~100 T4-h) but with two friction points — GPUs can't attach
until you upgrade the trial to paid billing, and new accounts usually start at GPU quota 0 (request
required). Fine as a fallback for the 3M rung if you want an uninterrupted box, not the default.

**Skip for real training:** Colab free (no background exec, ~90-min idle kill, non-guaranteed GPU,
30 GB overflows free 15 GB Drive and re-downloads each session). Paperspace Free is a weak M4000 /
6 h cap, being sunset. Oracle Always-Free is CPU-only.

**Uncertain — re-check at run time:** free-tier terms change. Kaggle's 30 h/week can be silently
reduced under load; GPU/internet need phone verification. Colab's weekly budget is undisclosed.

## 3. Data logistics

**Never stream from S3 per-epoch.** s3fs/boto3 in the DataLoader bills egress on every read — 3M
(87 GB) × 50 epochs ≈ 4.35 TB ≈ ~$390 recurring, and FUSE bottlenecks the GPU. Egress **once**, keep
a resident copy.

**Select inside AWS so only the subset egresses.** Spin a tiny `t3.medium` in **eu-central-1**
(same region → zero intra-AWS transfer), read the stored `anomaly_dchi2` per event, select balanced
top-N, write ~1–2 GB shards (**<50 top-level files** for Kaggle) to `s3://…/subset_NM/`. Then one
`aws s3 sync` to local, `kaggle datasets create -p ./subset_NM/` → persistent read-only mount, zero
per-epoch egress.

| Rung | Events | ~Size¹ | S3 egress² | Fits free disk? |
|------|--------|--------|-----------|-----------------|
| Smoke | 300k | ~9 GB | ~$0 (within free egress) | Yes — validate end-to-end first |
| Ramp | 1M | ~29 GB | ~$0–2.6 one-time | Yes (Kaggle 200 GB; overflows Colab Drive) |
| Target | 3M | ~87 GB | ~$8 one-time | Yes on Kaggle |
| (Full) | 10M | ~292 GB | ~$26 one-time | not needed |

¹ Linear from the measured 292 GB / 10M ≈ 29.2 GB per 1M — estimates, not measured.
² S3 egress ≈ $0.09/GB after the free monthly allowance; **verify current pricing**. 300k/1M are
largely within free egress; 3M is a bounded one-time ~$8. (Covered by standing AWS authorization.)

**Gotchas:** don't decompress the subset into the ~20 GB `/kaggle/working` — read shards lazily from
`/kaggle/input`; upload in sharded form and use `kaggle datasets version` to resume flaky uploads;
the current `/dev/shm` load OOMs on 29–87 GB (fixed by §1 #1).

## Status: code changes IMPLEMENTED (2026-07-05)

All five optimizations are landed in `train.py` (v4.2.0) and tested locally (torch 2.11 CPU + real
compact shards), including a full `main() --stream` training run end-to-end:

- **#1 lazy loader** — new `StreamingShardDataset` (lazy, fork-safe via `__getstate__`, multi-shard),
  `make_block_shuffle_indices`, and a streaming branch in `create_dataloaders`. Wired through
  `main()` behind `--stream` (accepts a shard **directory** or comma-list) + `resolve_shard_paths` /
  `stream_train_val_split` helpers. `/dev/shm` bypassed; cleanup guarded on `shm_path is None`.
  *Verified:* item-for-item identical to the RAM path across a shard boundary; fork/spawn-safe under
  `num_workers=2` persistent workers; block-shuffle sampler is a valid permutation.
- **#2 loss syncs** — `compute_hierarchical_loss` now returns detached 0-dim tensors and uses a
  branch-free sum/clamp stage-2. *Verified byte-identical* (0.00e+00 over 200 random batches) and
  NaN-safe on all-flat batches.
- **#3/#4** — `DEFAULT_NUM_WORKERS 0→4`, `DEFAULT_BATCH_SIZE 64→256`, `DEFAULT_PREFETCH_FACTOR 2→4`.
- **#5 streaming stats** — `compute_streaming_statistics` (block-streamed). *Verified* to match the
  full-load `compute_robust_statistics` to <1e-6.

Not yet done (deferred by design): float16 flux storage, temporal downsampling (#13, science-risky),
`torch.compile` (#12, keep off for bring-up). The `--stream` path is verified at component + `main()`
level on CPU; **still needs a first real GPU smoke-test** on the 300k subset.

## 4. Concrete next steps

Code changes #1–#5 are **done** (see Status above). Remaining path to first GPU run:

1. **Verify `anomaly_dchi2` coverage** in the finishing sim; sanity-check class balance.
2. **Select + shard in-region** on a `t3.medium` in eu-central-1: top-300k by `anomaly_dchi2`
   (balanced), `<50` shards → `s3://…/subset_300k/`. Start at 300k to validate cheaply.
3. **Code A — lazy Dataset (§1 #1):** no-shm fast path + block-aligned sampler / small-chunk repack +
   `__getstate__` dropping h5py handles. Also unblocks macOS smoke-testing.
4. **Code B — remove loss syncs (§1 #2):** detached 0-dim tensors; gate `.item()` behind `PROGRESS_UPDATE_FREQ`.
5. **Code C — streaming stats (§1 #5):** block-stream/sample; cache `stats.json`. (Required before 1M.)
6. **Local smoke test** (macOS/CPU) on 300k: no OOM, correct tuple order `(mag, dt, lengths, labels)`, finite loss.
7. **Stage on Kaggle:** `aws s3 sync … ./` then `kaggle datasets create -p ./subset_300k/` → `/kaggle/input/<name>`.
8. **First GPU run (Kaggle P100, commit mode)** — use `--stream` (the lazy loader) pointing at the
   shard **directory**; never torchrun; clear stray DDP env so the safe single-process branch is taken:
   ```bash
   env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u MASTER_ADDR -u MASTER_PORT \
     python train.py --stream --block-shuffle 10000 \
       --data /kaggle/input/<subset-300k>/ \
       --output /kaggle/working/checkpoints \
       --hierarchical --use-aux-head --attention-pooling --use-amp \
       --epochs 40 --warmup-epochs 3
   ```
   (`--batch-size 256 --num-workers 4 --prefetch-factor 4` are now the defaults. Keep `--compile`
   OFF. If 16 GB is tight: `--batch-size 128 --accumulation-steps 2`. `--block-shuffle 10000` matches
   the on-disk chunk size to cut lzf read amplification.)
9. **Validate baseline:** one epoch trains, finite loss, GPU util high (not <30%), and **binary-class
   recall** tracked against `anomaly_dchi2`. Add `--val-every 2` + atomic checkpointing before long runs.
10. **Climb the ladder:** repeat 2/7/8 for 1M then 3M, spreading 3M across sessions via checkpoint-resume.
    Revisit `--compile` (#12) or temporal downsampling (#13, A/B'd against binary recall) only after a
    trusted 3M baseline. Move to Modal if you exceed Kaggle's 30 h/week.

---

**Honesty flags:** line refs verified against actual `train.py`. Subset GB are linear extrapolations
from 292 GB/10M. S3 egress $ and all free-tier limits **change — re-check at run time**. Biggest single
lever is step 3 (lazy loader): without it the job OOMs before training regardless of GPU.

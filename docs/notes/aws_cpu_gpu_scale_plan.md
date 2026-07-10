# Cheap CPU-First Simulation + Single-GPU Training Plan

Date: 2026-07-05

## Goal

Generate a very large bank of Roman microlensing lightcurves cheaply using CPU-heavy workers, then train/evaluate with a much smaller GPU footprint. The target is not to start with a 40-GPU training cluster; it is to exploit cheap/free CPU, keep only useful shards, and use one stable GPU VM for model work.

## Current Code Facts

- Simulator: `code/simulate.py`
- Trainer: `code/train.py`
- Output format: HDF5 with dense arrays:
  - `flux`: `(n_events, 6912)` float32
  - `delta_t`: `(n_events, 6912)` float32
  - `timestamps`: `(n_events, 6912)` float32
  - `labels`: `(n_events,)` int32
  - `m_base` and parameter datasets
- Current simulator already uses multiprocessing and chunked HDF5 writes.
- Current trainer expects full HDF5 arrays and loads `flux`, `delta_t`, and `labels`.

## Critical Scale Reality Check

Each event has 6912 time points. The dense training arrays alone are large:

```text
flux + delta_t + timestamps ~= 3 arrays * 6912 points * 4 bytes
                           ~= 82,944 bytes/event
                           ~= 81 KiB/event before compression
```

Approximate dense storage:

| Events | Dense Raw Size | Practical Note |
|---:|---:|---|
| 700k | ~58 GB | Current HPC-scale dataset territory |
| 10M | ~829 GB | Feasible with sharding + object storage |
| 100M | ~8.3 TB | Expensive but possible if compressed/filtered |
| 1B | ~83 TB | Needs serious data engineering |
| 10B | ~829 TB raw | Not cheap if stored densely |

So "10 billion lightcurves" should not mean "write 10B full HDF5 dense rows and keep them all." The cheap design is:

1. Generate in shards.
2. Keep metadata and compact representations aggressively.
3. Store only selected training/evaluation shards.
4. Use active learning/hard-example mining to decide what deserves GPU training storage.

## Recommended Architecture

### Phase 0: Benchmark One Shard

Create a benchmark shard locally and on one cheap cloud CPU worker.

Suggested shard size:

```bash
python code/simulate.py \
  --n_flat 10000 \
  --n_pspl 10000 \
  --n_binary 10000 \
  --binary_preset baseline \
  --output data/raw/shard_bench_000.h5 \
  --num_workers "$(nproc)" \
  --seed 1000 \
  --oversample 1.3
```

Measure:

- events/sec
- output GB per 100k events
- failure rate by class
- CPU utilization
- wall time

Do not scale until these numbers are recorded.

### Phase 1: CPU Fanout On Cheap AWS

Use CPU EC2 instances, not GPU instances, for simulation. The simulator is CPU-heavy and already multiprocessing-friendly.

Cheapest AWS options:

- `t4g.small` / `t4g.medium`: cheap/free-trial ARM workers, good for testing.
- `c7g.large` / `c7g.xlarge`: Graviton CPU workers, good price/perf.
- `c7i.large` / `c7i.xlarge`: x86 workers if any dependency has ARM pain.
- Spot for batch simulation, because simulation shards are restartable.

Important: `VBBinaryLensing`, `numba`, `h5py`, and conda/pip builds must be verified on ARM before committing to Graviton. If ARM install is annoying, use x86 `c7i` Spot.

### Phase 2: Shard Contract

Use many independent shards. A shard must be small enough to retry cheaply.

Suggested shard sizes:

| Shard Events | Expected Raw Dense Size | Use |
|---:|---:|---|
| 30k | ~2.5 GB | benchmark/debug |
| 100k | ~8.3 GB | routine CPU shard |
| 300k | ~24.9 GB | only after stable |

Each shard should have:

- deterministic seed range
- class counts
- preset
- simulator git commit
- simulator version
- generation host metadata
- start/end timestamps
- success/failure counts

Preferred path layout:

```text
s3://<bucket>/microlensing/raw/baseline/date=YYYY-MM-DD/shard_000001.h5
s3://<bucket>/microlensing/manifests/baseline_manifest.jsonl
s3://<bucket>/microlensing/checkpoints/<experiment>/
s3://<bucket>/microlensing/results/<experiment>/
```

### Phase 3: Do Not Store Timestamps Per Event

Current HDF5 writes `timestamps` for every event even though the timestamp row is identical for all events.

Before scaling past ~1M events, change storage so timestamps are stored once:

```text
timestamps: (6912,)
```

instead of:

```text
timestamps: (n_events, 6912)
```

This removes about one third of dense storage. For 10M events, that saves roughly 276 GB raw.

The trainer currently does not appear to need per-row `timestamps`; it loads `flux`, `delta_t`, and `labels`.

### Phase 4: Compact Or Two-Tier Storage

For huge generation, keep two tiers:

1. Full dense HDF5 shards for selected training/evaluation data.
2. Compact metadata/parameter records for the massive generated universe.

The compact tier should store parameters, label, seed, class, quality flags, and maybe a small set of summary features. It should not store all 6912 points unless that event is selected.

Potential future format:

- Parquet/Arrow for metadata.
- HDF5/Zarr only for selected dense lightcurves.
- Zarr is attractive for cloud/object-store chunked reads, but it requires a training-loader change.

### Phase 5: Active Selection Before GPU

Do not train on all generated curves.

Use a loop:

1. Generate many CPU shards.
2. Train one GPU model on a balanced subset.
3. Score candidate shards.
4. Keep hard examples, rare regimes, and calibration/evaluation sets.
5. Discard or compact easy duplicates.

This is how "10B generated" can become cheap-ish: most of the universe becomes metadata and selection statistics, not dense GPU training input.

## AWS Execution Plan

### Minimal Stable Setup

- S3 bucket for shards/checkpoints/results.
- One EC2 launch template for CPU simulation workers.
- One EC2 GPU VM for training.
- One manifest file tracking shards.
- Budget alarm.

### CPU Worker Lifecycle

1. Launch Spot CPU instance.
2. Install environment or pull prebuilt image.
3. Generate one shard.
4. Upload shard + manifest row to S3.
5. Terminate instance.

Use `systemd`, `tmux`, or a simple bootstrap script. No notebooks.

Example bootstrap shape:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /opt/binml
conda activate microlens

SHARD_ID="${SHARD_ID:?}"
SEED="${SEED:?}"
OUT="data/raw/shard_${SHARD_ID}.h5"

python code/simulate.py \
  --n_flat 33334 \
  --n_pspl 33333 \
  --n_binary 33333 \
  --binary_preset baseline \
  --output "$OUT" \
  --num_workers "$(nproc)" \
  --seed "$SEED" \
  --oversample 1.3

aws s3 cp "$OUT" "s3://<bucket>/microlensing/raw/baseline/shard_${SHARD_ID}.h5"
aws s3 cp "logs/shard_${SHARD_ID}.json" "s3://<bucket>/microlensing/manifests/shard_${SHARD_ID}.json"
sudo shutdown -h now
```

### Free/Cheap AWS Levers

- Use AWS credits for storage and spot compute.
- Use `t4g.small` free-trial capacity only for validation or small shards.
- Use `c7g`/`c7i` Spot for real CPU fanout.
- Keep simulation workers stateless.
- Store only S3 output; terminate everything else.

## GPU Training Plan

Start with one stable GPU VM:

- Cheapest AWS: `g4dn.xlarge` Spot.
- Better price/perf if available: `g6.xlarge` Spot.
- More expensive but stronger: `g5.xlarge`.

Training should use selected shards, not the whole generated universe.

The current trainer loads arrays into RAM/shared storage. For multi-shard training at scale, add an iterable or indexed shard dataset that:

- reads one shard at a time,
- avoids loading huge full corpora into RAM,
- supports deterministic train/val split from manifest,
- can stream from local NVMe/EBS after prefetching from S3.

Until that loader exists, keep GPU training datasets at a size the VM can fit in RAM and disk.

## Cost Model

The previous HPC reference:

```text
700k lightcurves, 40 A100 GPUs, 28 min
```

That is about:

```text
40 * 28 / 60 = 18.7 A100 GPU-hours
```

For CPU generation, the cost must be measured from Phase 0. Use this formula:

```text
cost_per_million =
  (instance_hourly_price * hours_to_generate_shard / shard_events) * 1,000,000
  + storage_cost_per_million
```

Storage cost dominates if every curve is kept densely.

Approximate dense raw storage cost at 81 KiB/event:

```text
1M events  ~= 81 GB raw
10M        ~= 810 GB raw
100M       ~= 8.1 TB raw
1B         ~= 81 TB raw
10B        ~= 810 TB raw
```

S3 Standard for 810 TB is not "free cheap." Therefore the 10B version must be compact/selective.

## Quota Requests

For cheapest CPU fanout:

- Standard EC2 On-Demand vCPU quota: enough for fallback.
- Standard Spot quota: request 64-256 vCPUs depending ambition.

For GPU:

- G and VT Spot: 16-32 vCPUs first.
- G and VT On-Demand: 4-8 vCPUs as backup.

Avoid P-family until there is evidence that cheaper G-family GPUs cannot handle the selected training set.

## Milestones

### Milestone 1: Benchmark

- Generate 30k events locally.
- Generate 30k events on one AWS CPU worker.
- Record events/sec, GB/event, failure rate, and cost estimate.

### Milestone 2: First Cloud Shards

- Generate 10 shards of 100k events.
- Upload to S3.
- Build manifest.
- Verify HDF5 integrity.

### Milestone 3: One-GPU Training

- Train on selected 700k-1M events on one GPU VM.
- Save checkpoints to S3.
- Evaluate and compare to HPC reference.

### Milestone 4: Selection Loop

- Add scoring/hard-example selection.
- Keep compact metadata for broad generation.
- Keep dense curves only for selected training/eval sets.

### Milestone 5: Scale Decision

Only after the above:

- decide whether to scale to 10M, 100M, 1B, or a generated-but-not-stored 10B universe.
- decide whether storage format must change to Zarr/Parquet.
- decide whether multi-GPU is worth the cost.

## Immediate Code Changes Recommended Before Large Runs

1. Add a shard manifest writer to `simulate.py`.
2. Store timestamps once instead of once per event.
3. Add a shard integrity checker.
4. Add a cost/size estimator script.
5. Add a multi-shard training loader before training on more than one dense shard.
6. Add an active-selection stage so GPU training uses informative curves, not every generated curve.

## Bottom Line

The cheap plan is viable if "10B" means "generate/search/sample a massive synthetic universe" rather than "store 10B full dense lightcurves forever."

Use AWS CPU Spot/free-credit workers for stateless shard generation, S3 for durable output, and one stable GPU VM for selected training. Keep the live dense training set in the 700k-10M range until active selection proves that larger storage is worth paying for.

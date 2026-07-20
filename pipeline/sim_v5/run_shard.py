"""
BinML v5 — production shard driver.

One process generates one shard, writes an HDF5 file and uploads it to S3. Shards are
independent and content-addressed by index, so the whole run is resumable: a worker skips any
shard already present in the bucket, and a Spot interruption costs at most one shard.

Byproduct subsampling
---------------------
Only ~6% of generated binaries show a detectable anomaly. Reaching a useful NonPSPL count
therefore generates a very large excess of binaries that observationally ARE single-lens
events. Those are real and belong in the training set -- but not at 15:1. We keep all of
them at probability ``BYPRODUCT_KEEP`` and record that probability per event as
``keep_prob``, so the true population can be recovered by reweighting. Nothing is silently
discarded: the counts of generated-but-dropped events are written to the shard attributes.

Usage:
    python -m pipeline.sim_v5.run_shard --shard 0 --n-shards 400 --bucket BUCKET [--out DIR]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter

import numpy as np

from .assemble import SurveyConfig, simulate_event
from .classes import CLASS_NAMES
from .writer import ShardWriter

# Events GENERATED per shard, by true class. Scaled so one shard is a few minutes of work.
SHARD_MIX = {
    "NonPSPL": 10_800,      # the bottleneck: ~6% yield a detectable anomaly
    "PSPL": 1_000,
    "Flat": 750,
    "PeriodicVar": 875,
    "LongPeriodVar": 1_750,
    "Eruptive": 750,
}
BYPRODUCT_KEEP = 0.15       # keep-rate for NonPSPL that observationally became PSPL/Flat


BATCH = 500     # events held in RAM at once, per worker


def build_shard(shard: int, cfg: SurveyConfig, seed_base: int = 20260720):
    """Yield batches of one shard's events, seeded by shard index for reproducibility.

    This is a GENERATOR on purpose. Accumulating a whole shard first costs ~172 KB/event
    x ~7,500 events = 1.3 GB per worker; with one worker per vCPU that is 10.6 GB of light
    curves alone on a 16 GB instance, before Python overhead and the writer's own staging
    buffers -- an out-of-memory kill part-way through a paid fleet run. Streaming in small
    batches keeps a worker's resident set at a few hundred MB.

    Yields ``(batch, gen_counts, dropped)``; the counters are cumulative and the caller
    should read them after the final batch.
    """
    rng = np.random.default_rng(seed_base + shard * 7919)
    gen_counts, dropped = Counter(), Counter()
    i_pspl = CLASS_NAMES.index("PSPL")
    i_flat = CLASS_NAMES.index("Flat")

    order = []
    for cname, k in SHARD_MIX.items():
        order += [cname] * k
    rng.shuffle(order)

    batch = []
    for cname in order:
        ev = simulate_event(cname, rng, cfg)
        if ev is None:
            dropped["unusable"] += 1
            continue
        gen_counts[cname] += 1
        keep_prob = 1.0
        if cname == "NonPSPL" and ev.label_index in (i_pspl, i_flat):
            keep_prob = BYPRODUCT_KEEP
            if rng.random() >= BYPRODUCT_KEEP:
                dropped["byproduct"] += 1
                continue
        ev.params["_keep_prob"] = keep_prob
        batch.append(ev)
        if len(batch) >= BATCH:
            yield batch, gen_counts, dropped
            batch = []
    yield batch, gen_counts, dropped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--bucket", type=str, default=None)
    ap.add_argument("--prefix", type=str, default="v5/raw")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    # HARD GATE. _binary_magnification falls back to PSPL when VBBinaryLensing is missing,
    # which would silently generate every NonPSPL event as a single lens -- a ruined dataset
    # that looks completely normal until training. Never let a worker run without it.
    from .generators import has_vbb
    if not has_vbb():
        print("FATAL: VBBinaryLensing is not importable. Binary-lens events would silently "
              "degrade to PSPL. Refusing to generate.", file=sys.stderr)
        return 2

    cfg = SurveyConfig()
    shards = [s for s in range(args.n_shards) if s % args.workers == args.worker] \
        if args.workers > 1 else [args.shard]

    for s in shards:
        name = f"shard_{s:05d}.h5"
        key = f"{args.prefix}/{name}"
        if args.bucket:
            probe = subprocess.run(["aws", "s3api", "head-object", "--bucket", args.bucket,
                                    "--key", key], capture_output=True)
            if probe.returncode == 0:
                print(f"[shard {s}] already in S3, skipping", flush=True)
                continue

        t0 = time.time()
        out_dir = args.out or tempfile.gettempdir()
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name)
        lab, n_kept = Counter(), 0
        gen_counts = dropped = Counter()
        with ShardWriter(path, cfg) as w:
            # Streamed: each batch is written and released before the next is generated.
            for batch, gen_counts, dropped in build_shard(s, cfg):
                if batch:
                    w.append(batch)
                    n_kept += len(batch)
                    lab.update(CLASS_NAMES[e.label_index] for e in batch)
            w.set_run_attrs(shard=s, byproduct_keep_prob=BYPRODUCT_KEEP,
                            gen_counts=gen_counts, dropped=dropped)
        gen_s = time.time() - t0

        mb = os.path.getsize(path) / 1e6
        n_gen = sum(gen_counts.values())
        print(f"[shard {s}] {n_kept:,} kept / {n_gen:,} generated "
              f"in {gen_s:.0f}s ({n_gen/max(gen_s,1e-9):.0f} evt/s), {mb:.0f} MB", flush=True)
        print(f"[shard {s}] labels: " + ", ".join(f"{k}={v}" for k, v in sorted(lab.items())),
              flush=True)

        if args.bucket:
            r = subprocess.run(["aws", "s3", "cp", path, f"s3://{args.bucket}/{key}",
                                "--only-show-errors"], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[shard {s}] UPLOAD FAILED: {r.stderr}", flush=True)
                return 1
            os.unlink(path)
            print(f"[shard {s}] uploaded -> s3://{args.bucket}/{key}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

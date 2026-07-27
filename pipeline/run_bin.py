"""
BinML v5 — distributed binning: raw shard (312 MB) -> compact cache shard (~46 MB).

Runs on the same worker/partition pattern as run_shard.py. Binning is I/O bound rather than
CPU bound, so this exists mainly to avoid pulling 125 GB of raw shards down to a laptop:
the reduction happens in-region where S3 reads are fast and free, and only the compact
result is ever downloaded.

Usage:
    python -m pipeline.run_bin --n-shards 400 --worker 0 --workers 40 --bucket B
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from .cache import build_cache


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--raw-prefix", default="v5/raw")
    ap.add_argument("--out-prefix", default="v5/cache")
    ap.add_argument("--work", default="/mnt/work")
    args = ap.parse_args(argv)

    os.makedirs(args.work, exist_ok=True)
    mine = [s for s in range(args.n_shards) if s % args.workers == args.worker]
    for s in mine:
        name = f"shard_{s:05d}.h5"
        key = f"{args.out_prefix}/{name}"
        if subprocess.run(["aws", "s3api", "head-object", "--bucket", args.bucket,
                           "--key", key], capture_output=True).returncode == 0:
            print(f"[{s}] cached already, skip", flush=True)
            continue
        raw = os.path.join(args.work, name)
        t0 = time.time()
        # Probe first, and SKIP a shard that is not there. The raw shard indices are SPARSE
        # whenever generation striped across more workers than there are shards per prefix --
        # e.g. the hard regimes hold indices 2,3,14,15,26,... An earlier version returned 1
        # here, so every binning worker died on its very first missing index and the whole
        # fleet idled for hours having produced almost nothing.
        if subprocess.run(["aws", "s3api", "head-object", "--bucket", args.bucket,
                           "--key", f"{args.raw_prefix}/{name}"],
                          capture_output=True).returncode != 0:
            continue
        r = subprocess.run(["aws", "s3", "cp", f"s3://{args.bucket}/{args.raw_prefix}/{name}",
                            raw, "--only-show-errors"], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[{s}] download failed, skipping: {r.stderr[:120]}", flush=True)
            continue
        out = os.path.join(args.work, f"cache_{s:05d}.h5")
        res = build_cache([raw], out, verbose=False)
        os.unlink(raw)
        up = subprocess.run(["aws", "s3", "cp", out, f"s3://{args.bucket}/{key}",
                             "--only-show-errors"], capture_output=True, text=True)
        if up.returncode != 0:
            print(f"[{s}] upload failed, skipping: {up.stderr[:120]}", flush=True)
            os.unlink(out) if os.path.exists(out) else None
            continue
        os.unlink(out)
        print(f"[{s}] {res['n_events']:,} events -> {res['bytes']/1e6:.0f} MB "
              f"in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""BinML v5 — combined in-region bin + evaluate.

Per raw shard: download -> build_cache -> upload cache -> run the checkpoint -> upload the
compact preds. One raw download per shard; the light curves never leave the region. Missing
shard indices are skipped (raw indices are sparse when generation striped across more workers
than shards-per-prefix), the same fix run_bin carries.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from .cache import build_cache
from .eval_shard import eval_shard


def _exists(bucket, key):
    return subprocess.run(["aws", "s3api", "head-object", "--bucket", bucket, "--key", key],
                          capture_output=True).returncode == 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--raw-prefix", required=True)
    ap.add_argument("--cache-prefix", required=True)
    ap.add_argument("--preds-prefix", required=True)
    ap.add_argument("--ckpt-key", required=True)
    ap.add_argument("--work", default="/mnt/work")
    args = ap.parse_args(argv)

    os.makedirs(args.work, exist_ok=True)
    ckpt = os.path.join(args.work, "model.pt")
    if not os.path.exists(ckpt):
        subprocess.run(["aws", "s3", "cp", f"s3://{args.bucket}/{args.ckpt_key}", ckpt,
                        "--only-show-errors"], check=True)

    mine = [s for s in range(args.n_shards) if s % args.workers == args.worker]
    for s in mine:
        name = f"shard_{s:05d}.h5"
        pkey = f"{args.preds_prefix}/preds_{s:05d}.npz"
        if _exists(args.bucket, pkey):
            print(f"[{s}] preds exist, skip", flush=True)
            continue
        if not _exists(args.bucket, f"{args.raw_prefix}/{name}"):
            continue
        raw = os.path.join(args.work, name)
        r = subprocess.run(["aws", "s3", "cp", f"s3://{args.bucket}/{args.raw_prefix}/{name}",
                            raw, "--only-show-errors"], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[{s}] download failed: {r.stderr[:120]}", flush=True); continue
        t0 = time.time()
        cache = os.path.join(args.work, f"cache_{s:05d}.h5")
        try:
            res = build_cache([raw], cache, verbose=False)
        except Exception as e:
            print(f"[{s}] build_cache failed: {e}", flush=True); os.unlink(raw); continue
        os.unlink(raw)
        ck = f"{args.cache_prefix}/{name}"
        if not _exists(args.bucket, ck):
            subprocess.run(["aws", "s3", "cp", cache, f"s3://{args.bucket}/{ck}",
                            "--only-show-errors"], capture_output=True)
        preds = os.path.join(args.work, f"preds_{s:05d}.npz")
        try:
            info = eval_shard(cache, ckpt, preds, device="cpu")
        except Exception as e:
            print(f"[{s}] eval failed: {e}", flush=True); os.unlink(cache); continue
        os.unlink(cache)
        up = subprocess.run(["aws", "s3", "cp", preds, f"s3://{args.bucket}/{pkey}",
                             "--only-show-errors"], capture_output=True, text=True)
        os.unlink(preds) if os.path.exists(preds) else None
        if up.returncode != 0:
            print(f"[{s}] preds upload failed: {up.stderr[:120]}", flush=True); continue
        print(f"[{s}] {res['n_events']:,} ev, pred_mix={info['pred_mix']} in {time.time()-t0:.0f}s",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

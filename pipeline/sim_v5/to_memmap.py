"""
BinML v5 — convert cache shards into raw fp16 memmaps for training.

Why not just train from the HDF5 cache: ``cache.py`` writes ``feat/F146`` in lzf-compressed
chunks of (512, 864, 3). A shuffled batch of 256 events drawn from 500k touches ~256
*distinct* chunks, so a single batch forces ~256 chunk decompressions -- hundreds of MB
decompressed to deliver ~2 MB of useful rows. That is a milder form of the exact pathology
recorded in writer.py's docstring, where (10000, 6912) chunks drove training to 0.028 it/s.

Raw memmaps have no chunking and no decompression, so a random row costs exactly one page
fault. They are also the politest option on a shared machine: memmapped pages are clean and
file-backed, so under memory pressure the OS can evict them instead of swapping, which an
anonymous in-RAM array would force.

Events are PRE-SHUFFLED on write, so training can read contiguous slices if it wants to.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np

from .model_v5 import BAND_BINS

SCALARS = ("label", "true_class", "keep_prob", "dchi2_event", "dchi2_anomaly",
           "m_base_ref", "a_ks", "n_usable_bands")
# Per-band bookkeeping kept for slicing (colour-band present/absent, blending).
PERBAND = ("f_s", "n_kept")


def convert(cache_paths, out_dir: str, max_events: int = 0, seed: int = 20260720) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(cache_paths)

    counts = []
    for p in paths:
        with h5py.File(p, "r") as f:
            counts.append(int(f.attrs["n_events"]))
    total = sum(counts)
    n_out = min(max_events, total) if max_events else total

    rng = np.random.default_rng(seed)
    # Global shuffle: choose which source rows survive and where each lands, so the output is
    # already in random order and a shard's class composition cannot correlate with position.
    order = rng.permutation(total)[:n_out]
    dest_of = np.empty(total, dtype=np.int64); dest_of.fill(-1)
    dest_of[order] = np.arange(n_out)

    mm = {}
    for b in BAND_BINS:
        mm[f"feat/{b}"] = np.memmap(os.path.join(out_dir, f"feat_{b}.f16"), dtype=np.float16,
                                    mode="w+", shape=(n_out, BAND_BINS[b], 3))
        mm[f"frac/{b}"] = np.memmap(os.path.join(out_dir, f"frac_{b}.f16"), dtype=np.float16,
                                    mode="w+", shape=(n_out, BAND_BINS[b]))
    sc = {k: np.zeros(n_out, dtype=np.float64) for k in SCALARS}
    for b in BAND_BINS:
        for k in PERBAND:
            sc[f"{k}_{b}"] = np.zeros(n_out, dtype=np.float64)
    # params must ride the SAME dest permutation as the light curves; appending them in shard
    # order instead would scramble every slice relative to its own light curve.
    par = None
    param_fields = None

    base = 0
    for p, cnt in zip(paths, counts):
        d = dest_of[base:base + cnt]
        keep = d >= 0
        if keep.any():
            src = np.nonzero(keep)[0]
            dst = d[keep]
            with h5py.File(p, "r") as f:
                for b in BAND_BINS:
                    mm[f"feat/{b}"][dst] = f[f"feat/{b}"][:][src]
                    mm[f"frac/{b}"][dst] = f[f"frac/{b}"][:][src]
                for k in SCALARS:
                    sc[k][dst] = f[k][:][src]
                for b in BAND_BINS:
                    for k in PERBAND:
                        sc[f"{k}_{b}"][dst] = f[f"{k}/{b}"][:][src]
                if "params" in f:
                    if par is None:
                        par = np.full((n_out, f["params"].shape[1]), np.nan, dtype=np.float32)
                        pf = f.attrs.get("param_fields")
                        if pf is not None:
                            param_fields = [x.decode() if isinstance(x, bytes) else str(x)
                                            for x in pf]
                    par[dst] = f["params"][:][src]
        base += cnt

    for v in mm.values():
        v.flush()
    np.save(os.path.join(out_dir, "label.npy"), sc["label"].astype(np.int64))
    np.save(os.path.join(out_dir, "true_class.npy"), sc["true_class"].astype(np.int64))
    for k in ("keep_prob", "dchi2_event", "dchi2_anomaly", "m_base_ref", "a_ks"):
        np.save(os.path.join(out_dir, f"{k}.npy"), sc[k].astype(np.float32))
    if par is not None:
        np.save(os.path.join(out_dir, "params.npy"), par)
    for b in BAND_BINS:
        for k in PERBAND:
            np.save(os.path.join(out_dir, f"{k}_{b}.npy"), sc[f"{k}_{b}"].astype(np.float32))
    meta = {"n_events": int(n_out), "n_source_events": int(total),
            "param_fields": param_fields,
            "bands": {b: BAND_BINS[b] for b in BAND_BINS}, "dtype": "float16",
            "shuffled": True, "seed": seed}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    nbytes = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
    meta["bytes"] = nbytes
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-events", type=int, default=0)
    args = ap.parse_args(argv)
    paths = glob.glob(os.path.join(args.in_dir, "*.h5"))
    if not paths:
        print(f"no cache shards in {args.in_dir}", file=sys.stderr)
        return 1
    m = convert(paths, args.out, args.max_events)
    print(f"{m['n_events']:,} of {m['n_source_events']:,} events -> {args.out} "
          f"({m['bytes']/1e9:.2f} GB, {m['bytes']/max(m['n_events'],1):.0f} B/event)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

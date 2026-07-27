"""
BinML v5 — compact training cache.

Turns raw shards (~41.6 KB/event) into a binned representation (~6.3 KB/event) small enough
that a whole training set is resident in RAM. This is what makes training affordable: the
model is tiny, and essentially all of the cost was moving 6912-point float32 light curves.

Binning uses (mean, min, max) per bin, NOT striding or plain averaging. Measured on real
NonPSPL events, reducing F146 by 8x:

    min/max-pooled   median  0.0 mmag amplitude lost, worst    0.0
    mean-pooled      median  105 mmag                , worst 2723
    strided          median   28 mmag                , worst 2910

Caustic crossings last hours -- tens of epochs at 15-minute cadence -- so striding or plain
averaging aliases away the very feature that defines the NonPSPL class. Carrying the bin
extrema preserves it exactly, which is why each bin costs three numbers rather than one.

The same reasoning already bit this project once, in the anomaly statistic, where a 400-point
subsample stepped over narrow caustics and understated delta-chi^2 by up to ~100x.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, Tuple

import h5py
import numpy as np

__all__ = ["BIN_FACTORS", "bin_curve", "build_cache"]

# Bins per band. F146: 6912 -> 864 (8x). Colour bands: 288 -> 96 (3x); they are already
# sparse, so binning them harder would erase what little colour information exists.
BIN_FACTORS: Dict[str, int] = {"F146": 8, "F087": 3, "F213": 3}
CACHE_VERSION = "5.0.0"


def bin_curve(mag: np.ndarray, factor: int) -> Tuple[np.ndarray, np.ndarray]:
    """Bin ``mag`` (rows x epochs, NaN = no observation) by ``factor``.

    Returns ``(feat, frac)``:
      * ``feat``  (rows, nbins, 3) float32 -- per-bin mean, min, max, NaN where the bin is empty
      * ``frac``  (rows, nbins)    float32 -- fraction of epochs in the bin that were observed

    ``frac`` is kept deliberately: an empty or partly empty bin is a real observational fact
    (the source dropped below the detection limit), not padding, and the model must be able
    to tell "faint here" from "not observed here".
    """
    rows, n = mag.shape
    nb = n // factor
    if nb * factor != n:
        mag = mag[:, :nb * factor]
    r = mag.reshape(rows, nb, factor)
    obs = np.isfinite(r)
    cnt = obs.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, np.nansum(np.where(obs, r, 0.0), axis=2) / np.maximum(cnt, 1), np.nan)
        mn = np.where(cnt > 0, np.nanmin(np.where(obs, r, np.inf), axis=2), np.nan)
        mx = np.where(cnt > 0, np.nanmax(np.where(obs, r, -np.inf), axis=2), np.nan)
    feat = np.stack([mean, mn, mx], axis=-1).astype(np.float32)
    return feat, (cnt / factor).astype(np.float32)


def build_cache(shard_paths, out_path: str, verbose: bool = True) -> dict:
    """Bin every shard into one compact cache file."""
    bands = list(BIN_FACTORS)
    feats: Dict[str, list] = {b: [] for b in bands}
    fracs: Dict[str, list] = {b: [] for b in bands}
    scal: Dict[str, list] = {k: [] for k in
                             ("label", "true_class", "keep_prob", "dchi2_event",
                              "dchi2_anomaly", "m_base_ref", "a_ks", "n_usable_bands")}
    # The generator PARAMETERS (q, s, tE, u0, rho, t0, P, ...) and the per-band bookkeeping
    # are carried through as well. Every scientifically load-bearing evaluation slice keys off
    # them -- the (log s, log q) detection-efficiency plane, the Suzuki-break slice, the
    # short/long-tE slice, peak-in-window -- so dropping them here (as the first version did)
    # makes half the paper's figures uncomputable from the cache alone.
    params: list = []
    perband: Dict[str, list] = {f"{k}/{b}": [] for b in BIN_FACTORS for k in ("f_s", "n_kept")}
    param_fields = None
    n_total = 0
    for i, p in enumerate(sorted(shard_paths)):
        with h5py.File(p, "r") as f:
            n = int(f.attrs["n_events"])
            if n == 0:
                continue
            for b in bands:
                fe, fr = bin_curve(f[f"mag/{b}"][:], BIN_FACTORS[b])
                feats[b].append(fe)
                fracs[b].append(fr)
            for k in scal:
                scal[k].append(f[k][:])
            params.append(f["params"][:])
            if param_fields is None:
                pf = f.attrs.get("param_fields")
                param_fields = [x.decode() if isinstance(x, bytes) else str(x) for x in pf]
            for b in bands:
                for k in ("f_s", "n_kept"):
                    perband[f"{k}/{b}"].append(f[f"{k}/{b}"][:])
            n_total += n
        if verbose and (i + 1) % 20 == 0:
            print(f"  binned {i+1} shards, {n_total:,} events", flush=True)

    m_base = np.concatenate(scal["m_base_ref"]).astype(np.float32)
    with h5py.File(out_path, "w") as o:
        o.attrs["cache_version"] = CACHE_VERSION
        o.attrs["n_events"] = n_total
        for b in bands:
            o.attrs[f"bin_factor_{b}"] = BIN_FACTORS[b]
            arr = np.concatenate(feats[b])
            # Store the DEVIATION FROM BASELINE, not the absolute magnitude. float16 spacing
            # at m ~ 22 is ~0.016 mag, so absolute storage quantises to ~15 mmag -- comparable
            # to the 20 mmag amplitude floor that decides the Flat/event boundary, i.e. it
            # would blur the very distinction the labels encode. Deviations are small
            # (typically <1 mag), where float16 resolves ~5e-4 mag; even a 5-magnitude peak
            # quantises to ~4 mmag, which is negligible against a 5-magnitude signal.
            #
            # NOTE the raw shard makes the OPPOSITE choice (absolute float32) on purpose:
            # that file is the archival product and must not bake in a baseline convention.
            # This is a model input, so relative precision where the signal lives wins.
            arr = arr - m_base[:, None, None]
            o.create_dataset(f"feat/{b}", data=arr.astype(np.float16),
                             chunks=(min(512, len(arr)),) + arr.shape[1:], compression="lzf")
            o.attrs["feat_is_baseline_relative"] = True
            o.create_dataset(f"frac/{b}", data=np.concatenate(fracs[b]).astype(np.float16),
                             compression="lzf")
            feats[b].clear(); fracs[b].clear()
        for k, v in scal.items():
            o.create_dataset(k, data=np.concatenate(v))
        o.create_dataset("params", data=np.concatenate(params).astype(np.float32))
        if param_fields:
            o.attrs["param_fields"] = [x.encode() for x in param_fields]
        for k, v in perband.items():
            o.create_dataset(k, data=np.concatenate(v))
    return {"n_events": n_total, "bytes": os.path.getsize(out_path)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    paths = glob.glob(os.path.join(args.in_dir, "shard_*.h5"))
    if not paths:
        print(f"no shards in {args.in_dir}", file=sys.stderr)
        return 1
    print(f"binning {len(paths)} shards -> {args.out}")
    r = build_cache(paths, args.out)
    print(f"  {r['n_events']:,} events, {r['bytes']/1e9:.2f} GB "
          f"({r['bytes']/max(r['n_events'],1):.0f} bytes/event)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

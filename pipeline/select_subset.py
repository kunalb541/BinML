#!/usr/bin/env python3
"""
select_subset.py -- v4.2.0

Build a training-ready SUBSET from the full microlensing shard corpus using the
per-event detectability metric (`anomaly_dchi2`) written by simulate.py v4.2.0.

Why: the full 10M / ~292GB corpus is too big to host on a free GPU platform. The plan
is to select a balanced, detectability-aware subset (300k -> 1M -> 3M) and stage only
that. This selector:
  1. SCAN  -- reads only the small `params`/`labels` datasets from every shard (cheap)
             to build global per-event (label, anomaly_dchi2, q, snr) arrays.
  2. SELECT-- picks a class-balanced subset that deliberately OVERSAMPLES the hard
             regimes the old models failed on: low-`anomaly_dchi2` (smooth, PSPL-like)
             binaries and low-`q` (planetary) binaries; keeps flat minimal.
  3. WRITE -- copies the selected rows (flux/delta_t/mask/params/m_base) into new
             compact shards, one source shard opened at a time (bounded memory/disk),
             so it can run in-region on a small EC2 reading shards from local/EBS.

Output shards are the SAME v4.2.0 compact format, so train.py --stream reads them
directly and evaluate.py works unchanged.

Usage:
  python select_subset.py --shards <dir|glob|comma-list> --out <dir> \
      --target 300000 --shard-size 25000 [--seed 42]
"""
import argparse
import glob
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import h5py

FLAT, PSPL, BINARY = 0, 1, 2

# Default class budget (fractions of target). Flat is solved -> minimal ballast;
# spend the budget on the PSPL/binary boundary.
DEFAULT_FRACTIONS = {FLAT: 0.05, PSPL: 0.35, BINARY: 0.60}


def resolve_shards(arg: str) -> List[str]:
    p = Path(arg)
    if p.is_dir():
        return sorted(str(x) for x in p.glob('*.h5'))
    if ',' in arg:
        return [s for s in arg.split(',') if s]
    return sorted(glob.glob(arg)) or [arg]


def scan(shards: List[str]) -> Dict[str, np.ndarray]:
    """Read labels + detectability params across all shards. Returns global arrays
    plus the (shard_idx, local_row) mapping. Only tiny datasets are read here."""
    labels, anomaly, qv, snr, shard_ix, local_ix = [], [], [], [], [], []
    for si, path in enumerate(shards):
        with h5py.File(path, 'r') as f:
            lab = f['labels'][:].astype(np.int8)
            n = len(lab)
            # v4.2.0: fail fast on legacy shards. write_subset assumes every source has
            # the compact 'params'+'mask' datasets; a params-less shard would append None
            # rows and crash the compound-dtype write mid-run. Refuse up front.
            if 'params' not in f or 'mask' not in f:
                raise ValueError(
                    f"{path} lacks compact 'params'/'mask' (pre-v4.2.0). Regenerate with "
                    f"simulate.py >= v4.2.0; mixed-format corpora are not supported.")
            if 'params' in f and f.attrs.get('has_global_params', False):
                P = f['params'][:]
                names = P.dtype.names
                a = P['anomaly_dchi2'] if 'anomaly_dchi2' in names else np.full(n, np.nan)
                q = P['q'] if 'q' in names else np.full(n, np.nan)
                s = P['snr'] if 'snr' in names else np.full(n, np.nan)
            else:  # pre-v4.2.0 shard: no detectability -> selection falls back to random
                a = np.full(n, np.nan); q = np.full(n, np.nan); s = np.full(n, np.nan)
            labels.append(lab); anomaly.append(a); qv.append(q); snr.append(s)
            shard_ix.append(np.full(n, si, dtype=np.int32))
            local_ix.append(np.arange(n, dtype=np.int64))
    return {
        'label': np.concatenate(labels),
        'anomaly': np.concatenate(anomaly).astype(np.float64),
        'q': np.concatenate(qv).astype(np.float64),
        'snr': np.concatenate(snr).astype(np.float64),
        'shard': np.concatenate(shard_ix),
        'row': np.concatenate(local_ix),
    }


def _pick_binaries(idx: np.ndarray, anomaly: np.ndarray, q: np.ndarray,
                   n_want: int, rng: np.random.Generator) -> np.ndarray:
    """Select n_want binary indices, oversampling the hard regimes:
       40% lowest-anomaly (smooth/PSPL-like), 30% lowest-q (planetary), 30% random.
    Deduplicated; back-filled from the remainder if strata are thin."""
    if len(idx) <= n_want:
        return idx
    a = anomaly[idx]; qq = q[idx]
    order_a = idx[np.argsort(np.where(np.isnan(a), np.inf, a))]          # smooth first
    order_q = idx[np.argsort(np.where(np.isnan(qq), np.inf, qq))]         # planetary first
    n_smooth = int(0.40 * n_want); n_plan = int(0.30 * n_want)
    chosen = set(order_a[:n_smooth].tolist())
    chosen.update(order_q[:n_plan].tolist())
    remaining = np.array([i for i in idx if i not in chosen])
    n_rand = n_want - len(chosen)
    if n_rand > 0 and len(remaining) > 0:
        chosen.update(rng.choice(remaining, size=min(n_rand, len(remaining)), replace=False).tolist())
    out = np.array(sorted(chosen))
    if len(out) > n_want:  # trim if strata overlap pushed us over
        out = rng.choice(out, size=n_want, replace=False)
    return out


def select(meta: Dict[str, np.ndarray], target: int, fractions: Dict[int, float],
           seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    label = meta['label']
    picks = []
    for cls, frac in fractions.items():
        cls_idx = np.where(label == cls)[0]
        n_want = min(int(round(target * frac)), len(cls_idx))
        if n_want == 0 or len(cls_idx) == 0:
            continue
        if cls == BINARY:
            picks.append(_pick_binaries(cls_idx, meta['anomaly'], meta['q'], n_want, rng))
        else:
            picks.append(rng.choice(cls_idx, size=n_want, replace=False))
    sel = np.concatenate(picks)
    rng.shuffle(sel)
    return sel


def write_subset(shards: List[str], meta: Dict[str, np.ndarray], sel: np.ndarray,
                 out_dir: str, shard_size: int) -> None:
    """Write selected rows into new compact shards, MEMORY-BOUNDED: stream one source
    shard at a time, append its selected rows to the current output buffer, and flush
    whenever the buffer fills. Peak RAM ~= shard_size rows + one source shard's selected
    rows (not the whole subset). Output is grouped by source shard; train.py --stream
    reshuffles at load time so ordering does not matter.
    Preserves flux/delta_t/mask/params/m_base/time_grid + attrs."""
    os.makedirs(out_dir, exist_ok=True)
    sel_shard = meta['shard'][sel]; sel_row = meta['row'][sel]
    by_src: Dict[int, List[int]] = {}
    for k in range(len(sel)):
        by_src.setdefault(int(sel_shard[k]), []).append(int(sel_row[k]))

    with h5py.File(shards[0], 'r') as f0:
        time_grid = f0['time_grid'][:] if 'time_grid' in f0 else None
        param_dtype = f0['params'].dtype if 'params' in f0 else None
        n_mask_bytes = f0['mask'].shape[1] if 'mask' in f0 else None
        n_points = f0['flux'].shape[1]
    comp = {'compression': 'lzf'}

    buf = {'flux': [], 'dt': [], 'lab': [], 'mb': [], 'mask': [], 'par': []}
    state = {'n': 0, 'shard_no': 0}

    def flush(force=False):
        while buf['flux'] and (state['n'] >= shard_size or force):
            take = min(shard_size, state['n'])
            outp = Path(out_dir) / f"subset_{state['shard_no']:04d}.h5"
            with h5py.File(outp, 'w') as g:
                # v4.2.0: chunk by ROWS spanning the full time axis. Without explicit
                # chunks h5py auto-splits the time axis, so one row spans many chunks and
                # even sequential reads thrash the HDF5 cache. (min(256,take),6912) = one
                # row lives in one chunk (~6.9MB), so a per-row read decompresses one chunk.
                rchunk = (min(256, take), n_points)
                g.create_dataset('flux', data=np.stack(buf['flux'][:take]), chunks=rchunk, **comp)
                g.create_dataset('delta_t', data=np.stack(buf['dt'][:take]), chunks=rchunk, **comp)
                g.create_dataset('labels', data=np.array(buf['lab'][:take], np.int32))
                g.create_dataset('m_base', data=np.array(buf['mb'][:take], np.float32), **comp)
                if n_mask_bytes:
                    g.create_dataset('mask', data=np.stack(buf['mask'][:take]), **comp)
                if param_dtype is not None:
                    g.create_dataset('params', data=np.array(buf['par'][:take], dtype=param_dtype), **comp)
                if time_grid is not None:
                    g.create_dataset('time_grid', data=time_grid)
                g.attrs.update({'n_events': take, 'subset': True,
                                'has_global_params': param_dtype is not None,
                                'has_time_grid': time_grid is not None,
                                'timestamps_dropped': True, 'n_points': int(n_points)})
            print(f"  wrote {outp.name}: {take} events")
            for key in buf:
                buf[key] = buf[key][take:]
            state['n'] -= take; state['shard_no'] += 1
            if not force:
                break

    for src, rows in sorted(by_src.items()):
        rows_sorted = np.sort(np.array(rows))
        with h5py.File(shards[src], 'r') as f:
            flux = f['flux'][rows_sorted]; dt = f['delta_t'][rows_sorted]
            lab = f['labels'][rows_sorted]
            mb = f['m_base'][rows_sorted] if 'm_base' in f else np.zeros(len(rows_sorted), np.float32)
            mask = f['mask'][rows_sorted] if ('mask' in f and n_mask_bytes) else None
            par = f['params'][rows_sorted] if ('params' in f and param_dtype is not None) else None
        for j in range(len(rows_sorted)):
            buf['flux'].append(flux[j]); buf['dt'].append(dt[j])
            buf['lab'].append(int(lab[j])); buf['mb'].append(float(mb[j]))
            buf['mask'].append(None if mask is None else mask[j])
            buf['par'].append(None if par is None else par[j])
            state['n'] += 1
            if state['n'] >= shard_size:
                flush()
    flush(force=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', required=True, help='dir | glob | comma-list of source shards')
    ap.add_argument('--out', required=True, help='output directory for subset shards')
    ap.add_argument('--target', type=int, default=300000)
    ap.add_argument('--shard-size', type=int, default=25000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--flat-frac', type=float, default=DEFAULT_FRACTIONS[FLAT])
    ap.add_argument('--pspl-frac', type=float, default=DEFAULT_FRACTIONS[PSPL])
    ap.add_argument('--binary-frac', type=float, default=DEFAULT_FRACTIONS[BINARY])
    args = ap.parse_args()

    shards = resolve_shards(args.shards)
    print(f"scanning {len(shards)} shards...")
    meta = scan(shards)
    n_total = len(meta['label'])
    counts = {c: int((meta['label'] == c).sum()) for c in (FLAT, PSPL, BINARY)}
    print(f"  total events: {n_total:,}  (flat {counts[FLAT]:,} / pspl {counts[PSPL]:,} / binary {counts[BINARY]:,})")

    fractions = {FLAT: args.flat_frac, PSPL: args.pspl_frac, BINARY: args.binary_frac}
    sel = select(meta, args.target, fractions, args.seed)
    sc = {c: int((meta['label'][sel] == c).sum()) for c in (FLAT, PSPL, BINARY)}
    print(f"selected {len(sel):,}  (flat {sc[FLAT]:,} / pspl {sc[PSPL]:,} / binary {sc[BINARY]:,})")
    # report detectability spread of selected binaries
    ba = meta['anomaly'][sel][meta['label'][sel] == BINARY]
    ba = ba[np.isfinite(ba)]
    if len(ba):
        print(f"  binary anomaly_dchi2 selected: p10={np.percentile(ba,10):.1f} "
              f"median={np.median(ba):.1f} p90={np.percentile(ba,90):.1f}")
    print(f"writing subset to {args.out} ...")
    write_subset(shards, meta, sel, args.out, args.shard_size)
    print("done.")


if __name__ == '__main__':
    main()

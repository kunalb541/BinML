#!/usr/bin/env python3
"""EXPERIMENTAL -- CPU inference cost of BinML, to size a deployment (experimental/deployment/README.md).

Measures, on this machine's CPU (never MPS/GPU): (a) preprocessing one 72-day three-band light curve into tokens
(binml.preprocess.to_tokens), (b) the network forward pass per light curve at batch sizes 1, 64 and 512, and
(c) a partial-season scan of 144 half-day prefixes of one event (what the cascade does). Each with one thread
and with all threads. Inputs are real RMDC26 light curves from the local curve cache when it exists (the
recommended checkpoint's own input), else synthetic single-lens curves at the GBTDS cadence.

Usage:  python experimental/deployment/benchmark_inference.py [--n 200] [--out experimental/deployment/benchmark_results.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import resource
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
CACHE = os.path.expanduser("~/Desktop/Research/microlensing/gulls_curve_cache")
WEIGHTS = os.path.join(REPO, "validation/gulls/weights/ft_fspl5s_seasons_g08.pt")


def real_curves(n):
    out = []
    for f in sorted(glob.glob(os.path.join(CACHE, "c_*.npz")))[::40]:
        z = np.load(f, allow_pickle=False)
        for k in z.files:
            if k.startswith("mb|"):
                e = k.split("|")[1]
                bands = {b: (np.asarray(z[f"b|{e}|{b}|t"], float), np.asarray(z[f"b|{e}|{b}|m"], float))
                         for b in ("F146", "F087", "F213") if f"b|{e}|{b}|t" in z.files}
                if "F146" in bands:
                    out.append((bands, float(z[k][0])))
            if len(out) >= n:
                return out
    return out


def synthetic_curves(n, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        t146 = np.arange(0, 70.7, 15 / 1440.0); tc = np.arange(0, 70.7, 7.0 / 24)
        t0, tE, u0, mb = rng.uniform(10, 60), 10 ** rng.uniform(0.3, 1.8), rng.uniform(0.05, 1), rng.uniform(19, 23)
        amp = lambda t: (lambda u: (u * u + 2) / (u * np.sqrt(u * u + 4)))(np.hypot(u0, (t - t0) / tE))
        mk = lambda t, s: mb - 2.5 * np.log10(amp(t)) + rng.normal(0, s, t.size)
        out.append(({"F146": (t146, mk(t146, 0.01)), "F087": (tc, mk(tc, 0.02)), "F213": (tc, mk(tc, 0.02))}, mb))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--scan-events", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(REPO, "experimental/deployment/benchmark_results.json"))
    args = ap.parse_args(argv)
    import torch
    import binml
    from binml.preprocess import BAND_BINS, to_tokens
    curves = real_curves(args.n) if os.path.isdir(CACHE) else []
    source = "RMDC26 curve cache" if len(curves) >= args.n else "synthetic"
    if source == "synthetic":
        curves = synthetic_curves(args.n)
    clf = binml.Classifier(weights=WEIGHTS, device="cpu")
    res = {"_doc": __doc__.split("\n")[0], "machine": {"platform": platform.platform(), "processor": platform.processor(),
           "cpu_count": os.cpu_count(), "python": platform.python_version(), "torch": torch.__version__},
           "inputs": {"source": source, "n_curves": len(curves),
                      "median_f146_points": int(np.median([c[0]["F146"][0].size for c in curves]))}, "threads": {}}
    for nthreads in (1, os.cpu_count()):
        torch.set_num_threads(nthreads); r = {}
        t = time.perf_counter(); toks = [to_tokens(b, m_base_ref=mb, t_start=0.0) for b, mb in curves]
        r["tokens_ms_per_curve"] = 1e3 * (time.perf_counter() - t) / len(curves)
        F = {b: np.stack([k.feat[b] for k in toks]) for b in BAND_BINS}; P = {b: np.stack([k.frac[b] for k in toks]) for b in BAND_BINS}
        clf._forward({b: F[b][:2] for b in BAND_BINS}, {b: P[b][:2] for b in BAND_BINS})          # warm-up
        for bs in (1, 64, 512):
            bs_ = min(bs, len(curves)); reps = max(1, len(curves) // bs_) if bs_ > 1 else min(len(curves), 100)
            t = time.perf_counter()
            for i in range(reps):
                sl = slice((i * bs_) % len(curves), (i * bs_) % len(curves) + bs_)
                clf._forward({b: F[b][sl] for b in BAND_BINS}, {b: P[b][sl] for b in BAND_BINS})
            r[f"forward_ms_per_curve_batch{bs_}"] = 1e3 * (time.perf_counter() - t) / (reps * bs_)
        t = time.perf_counter()
        for b, mb in curves[:args.scan_events]:
            clf.predict_evolution(b, m_base_ref=mb, t_start=0.0, n_steps=144)
        r["season_scan_144_cuts_s_per_event"] = (time.perf_counter() - t) / args.scan_events
        r["full_season_scores_per_core_second"] = 1e3 / (r["tokens_ms_per_curve"] + r["forward_ms_per_curve_batch64"]) / nthreads
        res["threads"][str(nthreads)] = {k: round(v, 3) for k, v in r.items()}
        print(nthreads, "threads:", res["threads"][str(nthreads)], flush=True)
    res["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    res["command"] = " ".join(sys.argv)
    json.dump(res, open(args.out, "w"), indent=1)
    print("->", args.out, "| inputs:", res["inputs"], "| peak RSS MB", res["peak_rss_mb"])


if __name__ == "__main__":
    sys.exit(main())

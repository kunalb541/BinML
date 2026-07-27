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
    python -m pipeline.run_shard --shard 0 --n-shards 400 --bucket BUCKET [--out DIR]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter

from typing import Optional

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


# HARD-REGIME generation. The weak slices are weak because they are RARE: q<1e-3 is 0.69% of
# events, s>3 is 1.23%, and the intersection s>3 AND q<1e-3 is 0.02% -- 71 events in 443k. No
# reweighting can teach a model from 71 examples, so the fix is to generate a population that
# deliberately concentrates there and mix it into training. These override the science priors
# ON PURPOSE and must never be used for a population/rate measurement -- only for training.
HARD_REGIMES = {
    "planetary":  dict(Q_MIN=1e-6, Q_MAX=1e-3),                       # deep planetary
    "wide":       dict(S_MIN=2.0, S_MAX=5.0),                         # wide-separation caustics
    "close":      dict(S_MIN=0.2, S_MAX=0.6),                         # close-topology caustics
    "longte":     dict(TE_MEDIAN_DAYS=70.0, TE_SIGMA_DEX=0.30, TE_MIN_DAYS=40.0),
    "planet_wide": dict(Q_MIN=1e-6, Q_MAX=1e-3, S_MIN=2.0, S_MAX=5.0),  # the 0.02% corner
    "planet_res": dict(Q_MIN=1e-6, Q_MAX=1e-3, S_MIN=0.8, S_MAX=1.25),  # resonant + planetary
    # stage-6 weak-spot coverage (stress test found these under-trained):
    "wider":      dict(S_MIN=3.0, S_MAX=8.0),                         # wide-caustic (recall was 0.22 at s>5)
    "shortte":    dict(TE_MEDIAN_DAYS=2.0, TE_SIGMA_DEX=0.45,
                       TE_MIN_DAYS=0.3, TE_MAX_DAYS=10.0),            # sub-day/short (recall was 0.52 at tE<1)
}

# Regimes that alter the OBSERVING conditions or the contaminant amplitude rather than the
# lens priors. These target the Flat decision boundary and the observational edge cases.
CONFIG_REGIMES = {
    # Variables scaled to straddle the 0.02 mag floor: the model must learn where Flat ends.
    "boundary":   dict(amp_scale=(0.05, 0.60), mix="contaminant"),
    "boundary2":  dict(amp_scale=(0.02, 0.20), mix="contaminant"),
    # Faint sources: photometry is noise-dominated, so "is anything happening" is hardest.
    "faint":      dict(cfg=dict(m_base_min=23.5, m_base_max=25.0)),
    # Heavily reddened: colour bands drop out, model must work from F146 alone.
    "reddened":   dict(cfg=dict(m_base_min=22.5, m_base_max=25.0), a_ks=(0.9, 1.2)),
    # Bright: saturation and very high S/N, the opposite extreme.
    "bright":     dict(cfg=dict(m_base_min=20.0, m_base_max=21.0)),
    # Pure flat stars across the FULL faint/extinction range -- the class that must be perfect.
    "flatrich":   dict(mix="flat"),
    # CONTAMINANT-hard regimes. The lens regimes above are ~68% microlensing by construction,
    # so on their own they would fine-tune the model almost entirely on microlensing and leave
    # the three variable classes with no new coverage. These concentrate on the contaminants
    # that are actually dangerous or actually hard.
    "lpvhard":    dict(mix="lpv", amp_scale=(0.3, 1.5)),      # the PSPL mimic
    "erupthard":  dict(mix="eruptive", amp_scale=(0.1, 0.8)), # weak/slow outbursts, Be bumpers
    "perhard":    dict(mix="periodic", amp_scale=(0.15, 1.0)),# low-amplitude periodics
}
# Alternative class mixes for regimes that target a specific decision.
MIXES = {
    "contaminant": {"PeriodicVar": 5000, "LongPeriodVar": 6000, "Eruptive": 3000,
                    "Flat": 3000, "PSPL": 500, "NonPSPL": 500},
    "flat":        {"Flat": 9000, "PeriodicVar": 2000, "LongPeriodVar": 2500,
                    "Eruptive": 1500, "PSPL": 500, "NonPSPL": 500},
    # LongPeriodVar is the single most dangerous microlensing impostor (a season-long smooth
    # trend), so it gets the largest dedicated pool, mixed against real PSPL to sharpen the
    # exact discrimination that matters.
    "lpv":         {"LongPeriodVar": 9000, "PSPL": 3000, "NonPSPL": 1000,
                    "Flat": 1500, "PeriodicVar": 500, "Eruptive": 500},
    "eruptive":    {"Eruptive": 8000, "PSPL": 2500, "NonPSPL": 1000,
                    "Flat": 2000, "PeriodicVar": 1000, "LongPeriodVar": 1500},
    "periodic":    {"PeriodicVar": 8000, "LongPeriodVar": 2500, "Flat": 2000,
                    "PSPL": 2000, "NonPSPL": 1000, "Eruptive": 500},
}

# ---------------------------------------------------------------------------------
# OUT-OF-RANGE sweeps. Each redraws ONE physical parameter to a range beyond (or at the
# extreme edge of) the training priors, to probe extrapolation for EVERY class. Events are
# still labelled by observability, so an unobservable OOR draw lands in its observable class.
# Format: class it applies to -> {param: (lo, hi, scale)} with scale "log" or "lin".
# A single-class mix concentrates the shard on the swept class.
OOR_REGIMES = {
    # PSPL
    "oor_pspl_longte":   ("PSPL",    {"tE": (300.0, 700.0, "log")}),
    "oor_pspl_shortte":  ("PSPL",    {"tE": (0.2, 1.0, "log")}),
    "oor_pspl_highu0":   ("PSPL",    {"u0": (2.0, 4.0, "lin")}),
    "oor_pspl_lowfs":    ("PSPL",    {"f_s": (0.01, 0.1, "log")}),
    # NonPSPL
    "oor_np_tinyq":      ("NonPSPL", {"q": (1e-8, 1e-6, "log")}),
    "oor_np_widesep":    ("NonPSPL", {"s": (5.0, 12.0, "log")}),
    "oor_np_closesep":   ("NonPSPL", {"s": (0.05, 0.2, "log")}),
    "oor_np_bigrho":     ("NonPSPL", {"rho": (1e-2, 1e-1, "log")}),
    # PeriodicVar
    "oor_per_longp":     ("PeriodicVar", {"P": (60.0, 200.0, "log")}),
    "oor_per_bigamp":    ("PeriodicVar", {"amp_I": (1.5, 3.5, "lin")}),
    "oor_per_tinyamp":   ("PeriodicVar", {"amp_I": (0.005, 0.02, "log")}),
    # LongPeriodVar (periods rebuilt in _apply_oor)
    "oor_lpv_longp":     ("LongPeriodVar", {"P": (700.0, 2000.0, "log")}),
    "oor_lpv_bigamp":    ("LongPeriodVar", {"amp_I": (3.0, 6.0, "lin")}),
    # Eruptive
    "oor_erupt_bigamp":  ("Eruptive", {"amp_I": (7.0, 12.0, "lin")}),
    "oor_erupt_slowrise":("Eruptive", {"rise": (2.0, 8.0, "lin")}),
}
# Config-level OOR (baseline magnitude beyond the training window) -- handled like CONFIG_REGIMES.
OOR_CONFIG = {
    "oor_flat_faint":  dict(cfg=dict(m_base_min=25.0, m_base_max=27.5), mix="flat"),
    "oor_flat_bright": dict(cfg=dict(m_base_min=18.0, m_base_max=20.0), mix="flat"),
}


def _draw(lo, hi, scale, rng):
    import math
    if scale == "log":
        return float(10.0 ** rng.uniform(math.log10(lo), math.log10(hi)))
    return float(rng.uniform(lo, hi))


def _make_oor_override(regime):
    """Return a param_override(true_class, params, rng) for an OOR regime, or None."""
    if regime not in OOR_REGIMES:
        return None
    target_class, spec = OOR_REGIMES[regime]

    def override(true_class, params, rng):
        if true_class != target_class:
            return                       # only the swept class is perturbed
        for name, (lo, hi, scale) in spec.items():
            v = _draw(lo, hi, scale, rng)
            if name == "P" and true_class == "LongPeriodVar":
                params["periods"] = [(v, 1.0)]   # rebuild the consumed structure
                params["P"] = v
            else:
                params[name] = v
    return override



def _priors_for(regime: Optional[str]):
    from .priors import DEFAULT_PRIORS
    if not regime or regime == "none" or regime not in HARD_REGIMES:
        return DEFAULT_PRIORS
    import dataclasses
    return dataclasses.replace(DEFAULT_PRIORS, **HARD_REGIMES[regime])


def _config_for(regime: Optional[str], cfg: SurveyConfig):
    """Apply a CONFIG_REGIMES override to the survey config, and return the class mix."""
    import dataclasses
    from .generators import set_amp_scale
    set_amp_scale(1.0)
    mix = SHARD_MIX
    spec = None
    if regime in CONFIG_REGIMES:
        spec = CONFIG_REGIMES[regime]
    elif regime in OOR_CONFIG:
        spec = OOR_CONFIG[regime]
    elif regime in OOR_REGIMES:
        # concentrate the shard on the swept class (plus a little context) so the OOR draw
        # dominates the statistics
        tgt = OOR_REGIMES[regime][0]
        mix = {tgt: 9000, "Flat": 1500, "PSPL": 1000, "NonPSPL": 500,
               "PeriodicVar": 500, "LongPeriodVar": 500, "Eruptive": 500}
        mix = {k: v for k, v in mix.items()}
    if spec is not None:
        if "cfg" in spec:
            cfg = dataclasses.replace(cfg, **spec["cfg"])
        if "mix" in spec:
            mix = MIXES[spec["mix"]]
    return cfg, mix, (CONFIG_REGIMES.get(regime) or OOR_CONFIG.get(regime) or {})


def build_shard(shard: int, cfg: SurveyConfig, seed_base: int = 20260720,
                regime: Optional[str] = None):
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
    priors = _priors_for(regime)
    oor_override = _make_oor_override(regime)
    cfg, SHARD_MIX_LOCAL, spec = _config_for(regime, cfg)
    amp_rng = spec.get("amp_scale")
    aks_rng = spec.get("a_ks")
    gen_counts, dropped = Counter(), Counter()
    i_pspl = CLASS_NAMES.index("PSPL")
    i_flat = CLASS_NAMES.index("Flat")

    order = []
    for cname, k in SHARD_MIX_LOCAL.items():
        order += [cname] * k
    rng.shuffle(order)

    batch = []
    from .generators import set_amp_scale
    from .photometry import BulgeExtinction
    ext = BulgeExtinction(a_ks_min=aks_rng[0], a_ks_max=aks_rng[1]) if aks_rng else None
    for cname in order:
        if amp_rng is not None:
            # redraw per event so the shard covers the whole boundary band, not one scale
            set_amp_scale(float(10 ** rng.uniform(np.log10(amp_rng[0]), np.log10(amp_rng[1]))))
        ev = simulate_event(cname, rng, cfg, ext=ext, priors=priors, param_override=oor_override)
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
    ap.add_argument("--regime", type=str, default=None,
                    help="regime override: " + ", ".join(list(HARD_REGIMES) + list(CONFIG_REGIMES) + list(OOR_REGIMES) + list(OOR_CONFIG)))
    ap.add_argument("--seed-base", type=int, default=20260720,
                    help="RNG base; seed = seed_base + shard*7919. Training/val/test all used "
                         "20260720 (shards 0-399). Use a far-off base (e.g. 900000000) to "
                         "generate a provably-unseen evaluation set: different base => "
                         "different PCG64 streams => parameter tuples the model never saw.")
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
            for batch, gen_counts, dropped in build_shard(s, cfg, regime=args.regime, seed_base=args.seed_base):
                if batch:
                    w.append(batch)
                    n_kept += len(batch)
                    lab.update(CLASS_NAMES[e.label_index] for e in batch)
            w.set_run_attrs(shard=s, byproduct_keep_prob=BYPRODUCT_KEEP,
                            gen_counts=gen_counts, dropped=dropped)
            w._h5.attrs["regime"] = args.regime or "none"
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

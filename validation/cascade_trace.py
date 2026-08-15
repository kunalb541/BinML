#!/usr/bin/env python3
"""Full anomaly-probability traces for the frozen real-time cascade sample.

WHY THIS EXISTS.  The committed cascade artifact stored one number per event -- the first day
P(NonPSPL) crossed the operating threshold on a 0.5 d grid, using F146 only, compared against an
anomaly onset that was itself quantised to a 7.2 d grid.  Every follow-up question the audit asked
(does a coarser evaluation grid change the premature rate?  does requiring persistence?  does
revealing the colour bands?  is the onset quantisation biasing the answer?  how does latency depend
on mass ratio?) needed a different scan, and each ad-hoc rerun risked a different event sample.

So this script scans ONCE and stores the whole trace: for each event, P(NonPSPL) and the argmax
label at every 0.5 d cut, for two revealing policies (F146-only and all three bands), plus the
anomaly onset recomputed on a 0.5 d grid matched to the evaluation grid.  Every downstream
statistic is then a pure reduction of one artifact over one fixed event sample
(``validation/cascade_reduce.py``), so two cascade numbers in the paper can no longer come from
two different runs.

THE EVENT SAMPLE IS FROZEN BY SEED.  Seeds are read from the committed ``cascade_events.json``
and replayed, so this measures the same 1,000 events as the published headline.  The reducer
asserts that the 0.5 d F146 first crossings recovered here equal the committed ones; if the
simulator or the model ever changes, that assertion fails loudly rather than silently reporting
new numbers under an old protocol.

ONSET QUANTISATION.  ``_anomaly_onset_day`` defaults to a 10-point grid, i.e. the onset is rounded
UP to a multiple of 7.2 d, which inflates the measured premature rate: an alert falling in the
7.2 d window before the recorded onset is counted premature even when the anomaly was already
detectable.  Here the onset is recomputed with ``n_cuts=144`` (0.5 d), matching the cut grid, so
the premature comparison is like-for-like.  The coarse value is retained for continuity.

Usage:  python validation/cascade_trace.py [--procs N] [--limit N] [--out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_OUT = os.path.join(HERE, "cascade_trace.npz")
STEP_DAYS = 0.5
ONSET_CUTS = 144                      # 72 d / 144 = 0.5 d, matched to STEP_DAYS
PARAM_FIELDS = ["t0", "tE", "u0", "s", "q", "alpha", "rho", "_m_base_ref"]


def _worker(seeds):
    """Trace one chunk of events.  Returns per-event arrays; see main() for the layout."""
    import binml
    from binml.preprocess import to_tokens
    from pipeline.assemble import (SurveyConfig, _anomaly_onset_day, _pspl_refit_dchi2,
                                   simulate_event)
    from pipeline.model import BAND_BINS
    import torch

    torch.set_num_threads(1)           # one BLAS thread per process; the pool provides parallelism
    cfg = SurveyConfig()
    clf = binml.Classifier()
    cuts = np.arange(STEP_DAYS, cfg.window_days + 1e-9, STEP_DAYS)
    nc = len(cuts)

    def onset_mask(ref, params):
        """Is the anomaly detectable in the revealed window [0, t_cut], at EVERY cut?

        _anomaly_onset_day returns only the first crossing. The full mask is needed because the
        statistic is not monotone: as more of the season arrives, the best-fitting single-lens
        model can absorb an early deviation and push the residual back below threshold. That
        distinction decides whether an alert counts as premature, so the reduction needs both the
        first day the anomaly was EVER detectable and the day after which it STAYS detectable.
        """
        t, mag, sig, mb_, fs = ref
        m = np.zeros(nc, bool)
        for i, tc in enumerate(cuts):
            sel = t <= tc
            if int(sel.sum()) < 10:
                continue
            d, amp = _pspl_refit_dchi2(t[sel], mag[sel], sig[sel], mb_, fs, params)
            m[i] = bool(d >= cfg.dchi2_anomaly and amp >= cfg.min_amplitude_mag)
        return m

    out = []
    for seed in seeds:
        ev, ref = simulate_event("NonPSPL", np.random.default_rng(int(seed)), cfg,
                                 _return_ref_truth=True)
        if ev is None or ev.label != "NonPSPL":
            raise RuntimeError(f"seed {seed} no longer replays as a labelled NonPSPL event")
        mb = ev.params["_m_base_ref"]
        rec = {"seed": int(seed),
               "t_anom_coarse": float(ev.params["t_anom"]),
               "t_anom_fine": float(_anomaly_onset_day(ref, ev.params, cfg, n_cuts=ONSET_CUTS)),
               "onset_mask": onset_mask(ref, ev.params),
               "params": [float(ev.params.get(k, np.nan)) for k in PARAM_FIELDS]}
        for mode in ("f146", "multi"):
            pn = np.full(nc, np.nan, np.float32)
            am = np.full(nc, -1, np.int8)
            toks, idx = [], []
            for i, cut in enumerate(cuts):
                if mode == "f146":
                    b = ev.bands["F146"]
                    m = b.t <= cut
                    if int(m.sum()) < 10:
                        continue
                    bands = {"F146": (b.t[m], b.mag[m])}
                else:
                    bands = {k: (v.t[v.t <= cut], v.mag[v.t <= cut]) for k, v in ev.bands.items()}
                    if bands["F146"][0].size < 10:
                        continue
                    # a colour band with no epochs yet is absent, not empty -- to_tokens marks the
                    # band not-present, which is exactly the live situation before the first visit
                    bands = {k: v for k, v in bands.items() if k == "F146" or v[0].size > 0}
                toks.append(to_tokens(bands, m_base_ref=mb, t_start=0.0))
                idx.append(i)
            # One forward pass PER CUT, not one batched pass over all cuts, because that is what
            # binml.Classifier.predict does -- the trace records what a user of the released
            # package would actually get.
            #
            # A note on a wrong diagnosis, kept because it cost a rerun. When this reduction first
            # disagreed with the published per-event artifact on 13 of 890 crossings, I blamed the
            # BLAS path taken by batched matmuls and rewrote this loop to batch-1 to fix it. The
            # disagreement did not move. The actual cause was storing the trace as float16 below:
            # half precision has a spacing of ~5e-4 near the 0.904 operating threshold, so 78 of
            # the stored probabilities were within one ulp of it and a handful crossed the wrong
            # way. At float32 the agreement is exact. Batch-1 is retained on its own merits, not
            # because it fixed anything.
            for t, i in zip(toks, idx):
                P = clf._forward({b: t.feat[b][None] for b in BAND_BINS},
                                 {b: t.frac[b][None] for b in BAND_BINS})[0]
                pn[i] = P[2]
                am[i] = int(P.argmax())
            rec[f"p_{mode}"] = pn
            rec[f"a_{mode}"] = am
        out.append(rec)
    return out


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _provenance():
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None
    import binml
    import torch
    from pipeline.assemble import SurveyConfig
    cfg = SurveyConfig()
    ck = os.path.join(REPO, "binml", "weights", "binml.pt")
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
            "checkpoint": "binml/weights/binml.pt",
            "checkpoint_sha256": _sha256(ck),
            "python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
            "binml": getattr(binml, "__version__", "?"),
            "platform": sys.platform,
            "survey_config": {k: getattr(cfg, k) for k in
                              ("window_days", "dchi2_event", "dchi2_anomaly", "min_amplitude_mag")
                              if hasattr(cfg, k)},
            "step_days": STEP_DAYS, "onset_cuts": ONSET_CUTS}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0, help="trace only the first N seeds (smoke test)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    # A limited run is a smoke test, not a replacement publication artifact.  An earlier version
    # wrote both to the same default path, so `--limit 2` could destroy an hour-long 1,000-event
    # trace before the reducer had a chance to reject the shortened seed list.
    if args.limit and os.path.abspath(args.out) == os.path.abspath(DEFAULT_OUT):
        smoke_dir = os.path.join(HERE, "runs")
        os.makedirs(smoke_dir, exist_ok=True)
        args.out = os.path.join(smoke_dir, f"cascade_trace_smoke_{args.limit}.npz")
        print(f"[trace] --limit is a smoke test; publication artifact protected -> {args.out}",
              flush=True)

    rows = json.load(open(os.path.join(HERE, "cascade_events.json")))
    seeds = [r["seed"] for r in rows][: args.limit or None]
    chunks = [list(c) for c in np.array_split(np.array(seeds), max(args.procs, 1)) if len(c)]
    print(f"[trace] {len(seeds)} events over {len(chunks)} processes", flush=True)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(chunks)) as pool:
        recs = [r for chunk in pool.map(_worker, chunks) for r in chunk]
    recs.sort(key=lambda r: seeds.index(r["seed"]))
    print(f"[trace] done in {time.time() - t0:.0f}s", flush=True)

    from pipeline.assemble import SurveyConfig
    cuts = np.arange(STEP_DAYS, SurveyConfig().window_days + 1e-9, STEP_DAYS)
    np.savez_compressed(
        args.out,
        cuts=cuts.astype(np.float32),
        seed=np.array([r["seed"] for r in recs], np.int64),
        t_anom_coarse=np.array([r["t_anom_coarse"] for r in recs], np.float32),
        t_anom_fine=np.array([r["t_anom_fine"] for r in recs], np.float32),
        onset_mask=np.stack([r["onset_mask"] for r in recs]),
        params=np.array([r["params"] for r in recs], np.float64),
        param_fields=np.array(PARAM_FIELDS),
        # float32, NOT float16. Half precision has a spacing of ~5e-4 near the 0.904 operating
        # threshold, so storing the trace at float16 silently flips every crossing whose
        # probability sits within that of the threshold -- the artifact would change the answer it
        # exists to record. The file is ~2x larger and that is the right trade.
        p_f146=np.stack([r["p_f146"] for r in recs]).astype(np.float32),
        a_f146=np.stack([r["a_f146"] for r in recs]),
        p_multi=np.stack([r["p_multi"] for r in recs]).astype(np.float32),
        a_multi=np.stack([r["a_multi"] for r in recs]),
        provenance=np.array(json.dumps(_provenance())))
    print(f"[trace] -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

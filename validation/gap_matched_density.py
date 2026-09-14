#!/usr/bin/env python3
"""Matched-visit-count comparison of uniform vs nightly-gapped sampling (tracked, reproducible).

Audit finding: the manuscript's gap numbers had no tracked artifact -- an earlier Modal run wrote
uniform 0.0 / nightly 0.0 at 1,200 visits (where BOTH arms fail),
while the text quoted 0.59/0.24 from an untracked 17-event check. This script is the reproducible
version: many events, several matched visit counts, and a committed JSON result with binomial CIs.

Both arms use exactly the same events and the same number of visits; only the temporal placement
differs (uniform across the season vs confined to an 8 h nightly window). Nightly pools cap the
matched count at roughly a third of the season's epochs.

A third arm (sixth check, 2026-09-14) places the same number of visits on a gap-free REGULAR grid. The
"uniform" arm is a random subsample, which leaves some of the model's 2 h F146 bins empty (the fraction is
recorded per arm); the regular arm separates empty bins from sparsity. Without it, the uniform arm's collapse at
low density was read as a sparsity floor, whereas the regular grid keeps working there.

Usage:  python validation/gap_matched_density.py [n_events]
"""
from __future__ import annotations
import json, os, sys, warnings
import numpy as np
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import binml
from pipeline.assemble import simulate_event, SurveyConfig

VISIT_COUNTS = [2300, 1800, 1400]          # per 72-day season -> ~32, 25, 19 visits/day
NIGHT_FRACTION = 8.0 / 24.0


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


BIN_DAYS = 2.0 / 24.0                      # the model's F146 bin (864 bins over the 72 d season)


def _empty_bin_frac(t_sel, window_days):
    n_bins = int(round(window_days / BIN_DAYS))
    return 1.0 - len(np.unique(np.floor(t_sel / BIN_DAYS).astype(int))) / n_bins


def main(n_events=120):
    import hashlib, subprocess
    clf = binml.Classifier(); cfg = SurveyConfig()
    evs, s = [], 900000
    while len(evs) < n_events and s < 900000 + 40 * n_events:
        s += 1
        ev = simulate_event("NonPSPL", np.random.default_rng(s), cfg)
        if ev and ev.label == "NonPSPL":
            evs.append(ev)
    print(f"{len(evs)} detectable binaries")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=12"], cwd=root,
                          capture_output=True, text=True).stdout.strip()
    if not code:
        raise SystemExit("FATAL: git describe returned nothing; an artifact must record its code")
    ckpt = getattr(clf, "weights_path", None) or os.path.join(root, "binml", "weights", "binml.pt")
    out = {"n_events": len(evs), "night_fraction": NIGHT_FRACTION, "code": code,
           "checkpoint_sha256": hashlib.sha256(open(ckpt, "rb").read()).hexdigest(),
           "script_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
           "empty_bin_days": BIN_DAYS, "arms": []}
    for V in VISIT_COUNTS:
        u = n = r = tot = 0
        eu, en, er = [], [], []
        for i, ev in enumerate(evs):
            b = ev.bands["F146"]; mb = ev.params["_m_base_ref"]
            rng = np.random.default_rng(10_000 + i)
            pool_n = np.nonzero((b.t % 1.0) < NIGHT_FRACTION)[0]
            if V > len(pool_n) or V > len(b.t):
                continue                      # matched count impossible for this arm
            ui = np.sort(rng.choice(len(b.t), V, replace=False))
            ni = np.sort(rng.choice(pool_n, V, replace=False))
            ri = np.unique(np.round(np.linspace(0, len(b.t) - 1, V)).astype(int))   # gap-free regular grid
            u += clf.predict(b.t[ui], b.mag[ui], m_base_ref=mb, t_start=0.0).label == "NonPSPL"
            n += clf.predict(b.t[ni], b.mag[ni], m_base_ref=mb, t_start=0.0).label == "NonPSPL"
            r += clf.predict(b.t[ri], b.mag[ri], m_base_ref=mb, t_start=0.0).label == "NonPSPL"
            eu.append(_empty_bin_frac(b.t[ui], cfg.window_days)); en.append(_empty_bin_frac(b.t[ni], cfg.window_days))
            er.append(_empty_bin_frac(b.t[ri], cfg.window_days))
            tot += 1
        if tot == 0:
            continue
        rec = {"visits": V, "visits_per_day": round(V / cfg.window_days, 1), "n": tot,
               "uniform": round(u / tot, 3), "uniform_ci": [round(x, 3) for x in wilson(u, tot)],
               "nightly": round(n / tot, 3), "nightly_ci": [round(x, 3) for x in wilson(n, tot)],
               "regular": round(r / tot, 3), "regular_ci": [round(x, 3) for x in wilson(r, tot)],
               "empty_bin_frac": {"uniform": round(float(np.mean(eu)), 3), "nightly": round(float(np.mean(en)), 3),
                                  "regular": round(float(np.mean(er)), 3)}}
        out["arms"].append(rec)
        print(f"  {V:5d} visits ({rec['visits_per_day']:4.1f}/day, n={tot:3d}): "
              f"uniform {rec['uniform']:.3f} {rec['uniform_ci']}  "
              f"nightly {rec['nightly']:.3f} {rec['nightly_ci']}  regular {rec['regular']:.3f} {rec['regular_ci']}  "
              f"empty bins {rec['empty_bin_frac']}")
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gap_matched_result.json")
    json.dump(out, open(p, "w"), indent=2)
    print("wrote", p)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)

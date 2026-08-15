#!/usr/bin/env python3
"""Draw the frozen cascade event sample and record each event's first threshold crossing.

WHERE THIS SITS IN THE PIPELINE.  This script DEFINES the event sample (which seeds, how many, the
eligibility rule).  It is not where the manuscript's cascade numbers come from any more:

    cascade_reproduce.py   ->  cascade_events.json        the frozen sample + published crossings
    cascade_trace.py       ->  cascade_trace.npz          full probability traces for those seeds
    cascade_reduce.py      ->  cascade_reproduce_result.json   every reported statistic

Everything the paper reports is a reduction of the trace, so the evaluation grid, alert rule,
persistence requirement, band set and onset definition can all be varied on ONE event sample
instead of being re-measured on a new draw each time.  cascade_events.json is kept as the
regression fixture: cascade_reduce.py asserts that its reduction reproduces these crossings
exactly, and fails loudly if the simulator or checkpoint has changed.

WHY --promote EXISTS.  The published artifact is a 1,000-event sample.  An earlier version of this
script defaulted to 150 events and wrote straight to the published path, so running the documented
command with no argument silently replaced the headline artifact with a smaller, noisier one.  The
default is now the published protocol, and writing to the published path requires --promote.

Usage:  python validation/cascade_reproduce.py [--n 1000] [--out PATH] [--promote]
"""
from __future__ import annotations
import json, os, sys, warnings
import numpy as np
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import binml
from pipeline.assemble import simulate_event, SurveyConfig

HERE = os.path.dirname(os.path.abspath(__file__))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main(n_events=1000, step_days=0.5, out_dir=None, promote=False):
    """Event-level time-to-first-crossing. For each eligible detectable binary we sweep the
    revealed season and record the FIRST day P(NonPSPL) crosses the operating threshold. An alert
    is PREMATURE if that first crossing precedes t_anom.

    This is the operationally meaningful quantity: a live stream evaluates continuously, so what
    matters is whether an event's first alert fires before its anomaly is observable -- not whether
    a randomly chosen pre-onset snapshot happens to be over threshold. An earlier version of this
    script measured the snapshot rate (2% of random pre-onset windows) and the manuscript wrongly
    equated it with the premature-alert rate; the event-level rate is several times larger.

    NOTE ON t_anom: the onset recorded here is the generator's, quantised to a 7.2 d grid, so the
    'premature' flag in cascade_events.json is the coarse-onset one. cascade_reduce.py recomputes
    the onset on the 0.5 d grid matched to the sweep and reports that as primary; see its
    docstring for why the coarse grid biases the premature rate upward.
    """
    clf = binml.Classifier(); cfg = SurveyConfig()
    thr = json.load(open(os.path.join(os.path.dirname(HERE), "paper", "results",
                                      "metrics.json")))["headline"]["threshold"]
    rows, s = [], 300000
    while len(rows) < n_events and s < 300000 + 60 * n_events:
        s += 1
        ev = simulate_event("NonPSPL", np.random.default_rng(s), cfg)
        if ev is None or ev.label != "NonPSPL":
            continue
        # ELIGIBILITY (stated explicitly per referee request): an event enters the sample iff
        # (a) it was generated as a binary lens, (b) detectability-conditioned labelling kept the
        # NonPSPL label -- i.e. its anomaly is observable at all -- and (c) it has a finite
        # anomaly-onset time. No event is excluded on the basis of the model's output, so the
        # sample is not selected on the outcome being measured.
        ta = ev.params.get("t_anom")
        if ta is None or not np.isfinite(ta):
            continue
        b = ev.bands["F146"]; mb = ev.params["_m_base_ref"]
        first_cross = None
        for cut in np.arange(step_days, cfg.window_days + 1e-9, step_days):
            m = b.t <= cut
            if m.sum() < 10:
                continue
            p = clf.predict(b.t[m], b.mag[m], m_base_ref=mb, t_start=0.0)
            if p.probabilities["NonPSPL"] >= thr:
                first_cross = float(cut); break
        rows.append({"seed": s, "t_anom": round(float(ta), 2),
                     "first_crossing_day": first_cross,
                     "detected": first_cross is not None,
                     "premature": bool(first_cross is not None and first_cross < ta),
                     "lag_days": (round(first_cross - float(ta), 2)
                                  if first_cross is not None else None)})
    n = len(rows)
    det = [r for r in rows if r["detected"]]
    prem = [r for r in rows if r["premature"]]
    lo, hi = wilson(len(prem), n)
    dlo, dhi = wilson(len(det), n)
    lags = np.array([r["lag_days"] for r in det], float)
    out = {"model": "shipped (cascade-trained)", "protocol": "event-level first threshold crossing",
           "threshold": thr, "step_days": step_days,
           "n_eligible": n, "n_detected": len(det), "n_censored": n - len(det),
           "detection_fraction": round(len(det) / max(n, 1), 3),
           "detection_ci": [round(dlo, 3), round(dhi, 3)],
           "premature_rate_of_eligible": round(len(prem) / max(n, 1), 3),
           "premature_ci_of_eligible": [round(lo, 3), round(hi, 3)],
           "premature_rate_of_detected": round(len(prem) / max(len(det), 1), 3),
           "median_lag_detected_days": round(float(np.median(lags)), 2) if lags.size else None}
    out["onset_grid_days"] = 7.2
    out["_note"] = ("summary under the COARSE generator onset grid. The manuscript reports "
                    "cascade_reduce.py's reduction of cascade_trace.npz, which recomputes the "
                    "onset on the 0.5 d sweep grid.")
    dest = out_dir or os.path.join(HERE, "runs", f"n{n_events}")
    if promote:
        dest = HERE
    os.makedirs(dest, exist_ok=True)
    json.dump(rows, open(os.path.join(dest, "cascade_events.json"), "w"), indent=1)
    json.dump(out, open(os.path.join(dest, "cascade_summary.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"per-event rows -> {dest}/cascade_events.json")
    if not promote:
        print("NOT promoted: the published sample in validation/ is untouched. Re-run with "
              "--promote to replace it, then rerun cascade_trace.py and cascade_reduce.py.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=1000, help="events (published protocol: 1000)")
    ap.add_argument("--out", default=None, help="output directory (default validation/runs/nN)")
    ap.add_argument("--promote", action="store_true",
                    help="write to validation/, REPLACING the published frozen sample")
    a = ap.parse_args()
    main(a.n, out_dir=a.out, promote=a.promote)

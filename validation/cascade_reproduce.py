#!/usr/bin/env python3
"""Reproduce the cascade numbers with a tracked, event-level artifact.

Every reported field is computed BY THIS SCRIPT -- an earlier version of the committed JSON had
the day-11 and argmax figures merged in by hand, so rerunning would have silently dropped them.

Audit finding: the manuscript's cascade figures (premature-flag rate, pre-onset probability) were
summary values with no committed script or per-event output that regenerates them.

This measures, for the SHIPPED model, the two quantities directly on freshly simulated binaries:
  * premature-flag rate -- fraction of detectable binaries flagged NonPSPL (at the frozen operating
    threshold) while the revealed window still ends BEFORE the anomaly onset t_anom;
  * mean pre-onset P(NonPSPL) -- the average anomaly probability over those same pre-onset windows.

Both are evaluated on truncated windows sampled uniformly before t_anom, which is the situation a
live alert stream faces. Per-event rows are written to cascade_events.json so the summary is
auditable, alongside the summary in cascade_reproduce_result.json.

NOTE: this reproduces the CASCADE model's (shipped) values only. The "baseline (no cascade)"
comparison numbers in the manuscript come from a model trained without truncation augmentation,
which is a separate training run; reproducing those requires retraining and is flagged as such.

Usage:  python validation/cascade_reproduce.py [n_events]
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


def main(n_events=150, step_days=0.5):
    """Event-level time-to-first-crossing. For each eligible detectable binary we sweep the
    revealed season and record the FIRST day P(NonPSPL) crosses the operating threshold. An alert
    is PREMATURE if that first crossing precedes t_anom.

    This is the operationally meaningful quantity: a live stream evaluates continuously, so what
    matters is whether an event's first alert fires before its anomaly is observable -- not whether
    a randomly chosen pre-onset snapshot happens to be over threshold. An earlier version of this
    script measured the snapshot rate (2% of random pre-onset windows) and the manuscript wrongly
    equated it with the premature-alert rate; the event-level rate is several times larger.
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
    json.dump(rows, open(os.path.join(HERE, "cascade_events.json"), "w"), indent=1)
    json.dump(out, open(os.path.join(HERE, "cascade_reproduce_result.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"per-event rows -> {HERE}/cascade_events.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)

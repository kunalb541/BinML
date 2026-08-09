#!/usr/bin/env python3
"""Reproduce the cascade numbers with a tracked, event-level artifact.

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


def main(n_events=150):
    clf = binml.Classifier(); cfg = SurveyConfig()
    thr = json.load(open(os.path.join(os.path.dirname(HERE), "paper", "results",
                                      "metrics.json")))["headline"]["threshold"]
    rows, s = [], 300000
    while len(rows) < n_events and s < 300000 + 40 * n_events:
        s += 1
        ev = simulate_event("NonPSPL", np.random.default_rng(s), cfg)
        if ev is None or ev.label != "NonPSPL":
            continue
        ta = ev.params.get("t_anom")
        if ta is None or not np.isfinite(ta) or ta < 5.0:
            continue                       # need room for a pre-onset window
        b = ev.bands["F146"]; mb = ev.params["_m_base_ref"]
        rng = np.random.default_rng(50_000 + len(rows))
        cut = float(rng.uniform(3.0, ta))  # a window ending strictly before onset
        m = b.t <= cut
        if m.sum() < 10:
            continue
        p = clf.predict(b.t[m], b.mag[m], m_base_ref=mb, t_start=0.0)
        pn = float(p.probabilities["NonPSPL"])
        rows.append({"seed": s, "t_anom": round(float(ta), 2), "cut_day": round(cut, 2),
                     "p_nonpspl": round(pn, 4), "flagged": bool(pn >= thr)})
    k = sum(r["flagged"] for r in rows); n = len(rows)
    lo, hi = wilson(k, n)
    out = {"model": "shipped (cascade-trained)", "threshold": thr, "n_events": n,
           "premature_flag_rate": round(k / max(n, 1), 3),
           "premature_flag_ci": [round(lo, 3), round(hi, 3)],
           "mean_pre_onset_p": round(float(np.mean([r["p_nonpspl"] for r in rows])), 4),
           "median_pre_onset_p": round(float(np.median([r["p_nonpspl"] for r in rows])), 4)}
    json.dump(rows, open(os.path.join(HERE, "cascade_events.json"), "w"), indent=1)
    json.dump(out, open(os.path.join(HERE, "cascade_reproduce_result.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"per-event rows -> {HERE}/cascade_events.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)

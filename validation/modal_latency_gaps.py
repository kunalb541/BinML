"""P0.7 censor-aware detection latency + P0.8 matched-density gap test, on Modal.

P0.7 -- the earlier latency histogram sampled until it had 80 DETECTED binaries and dropped events
whose onset fell in the last three days, which is survivor bias with hidden right-censoring, at
2-day prediction resolution against a 7.2-day onset proxy. Here we keep EVERY eligible binary,
treat never-detected events as right-censored, and resolve the reveal grid finely (0.5 d).
Output: a Kaplan-Meier-style detection-time curve plus the non-detection fraction.

P0.8 -- the earlier ground-cadence claim compared uniform ~60/day against an 8h nightly window at
~32/day, so gap structure was confounded with total visit count. Here uniform and nightly-gapped
schedules are compared at MATCHED visit counts, over many events, so any difference is attributable
to gap topology alone.

Run:  modal run validation/modal_latency_gaps.py
"""
import os
import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy", "h5py", "VBBinaryLensing")
    .add_local_dir(os.path.join(REPO, "pipeline"), "/repo/pipeline", copy=True)
    .add_local_dir(os.path.join(REPO, "binml"), "/repo/binml", copy=True)
)

app = modal.App("binml-latency-gaps")


@app.function(image=image, cpu=16.0, timeout=5400)
def latency_chunk(seed0: int, n: int, thr: float, step_days: float = 0.5):
    """Censor-aware latency for n binaries: reveal the season on a fine grid and record the first
    day P(NonPSPL) crosses thr, relative to t_anom. Never-crossing events are right-censored."""
    import sys, warnings
    sys.path.insert(0, "/repo"); warnings.simplefilter("ignore")
    import numpy as np, binml
    from pipeline.assemble import simulate_event, SurveyConfig
    clf = binml.Classifier(); CFG = SurveyConfig(); NON = binml.CLASS_NAMES.index("NonPSPL")
    out = []
    s = seed0
    while len(out) < n and s < seed0 + 20 * n:
        s += 1
        ev = simulate_event("NonPSPL", np.random.default_rng(s), CFG)
        if ev is None or ev.label != "NonPSPL":
            continue
        ta = ev.params.get("t_anom")
        if ta is None or not np.isfinite(ta):
            continue
        b = ev.bands["F146"]; mb = ev.params["_m_base_ref"]
        cuts = np.arange(step_days, CFG.window_days + 1e-9, step_days)
        lag, detected = None, False
        for c in cuts:
            m = b.t <= c
            if m.sum() < 10:
                continue
            p = clf.predict(b.t[m], b.mag[m], m_base_ref=mb, t_start=0.0)
            if p.probabilities["NonPSPL"] >= thr:
                lag = float(c - ta); detected = True; break
        # every eligible binary is kept: detected (lag) or right-censored at end of season
        out.append({"detected": detected,
                    "lag": lag if detected else float(CFG.window_days - ta),
                    "t_anom": float(ta), "q": float(ev.params.get("q", np.nan))})
    return out


@app.function(image=image, cpu=16.0, timeout=5400)
def gap_chunk(seed0: int, n: int, n_visits: int):
    """Matched-density comparison: the SAME events sampled uniformly vs in nightly blocks, both
    with exactly n_visits points, so only the gap topology differs."""
    import sys, warnings
    sys.path.insert(0, "/repo"); warnings.simplefilter("ignore")
    import numpy as np, binml
    from pipeline.assemble import simulate_event, SurveyConfig
    clf = binml.Classifier(); CFG = SurveyConfig(); NON = binml.CLASS_NAMES.index("NonPSPL")
    res = {"uniform": [], "nightly": []}
    s = seed0
    got = 0
    while got < n and s < seed0 + 20 * n:
        s += 1
        ev = simulate_event("NonPSPL", np.random.default_rng(s), CFG)
        if ev is None or ev.label != "NonPSPL":
            continue
        b = ev.bands["F146"]; mb = ev.params["_m_base_ref"]; rng = np.random.default_rng(s)
        # uniform: n_visits spread over the whole season
        ui = np.sort(rng.choice(len(b.t), min(n_visits, len(b.t)), replace=False))
        # nightly: same count, but confined to an 8h window each day (diurnal gaps)
        night = (b.t % 1.0) < (8.0 / 24.0)
        ni_pool = np.nonzero(night)[0]
        ni = np.sort(rng.choice(ni_pool, min(n_visits, len(ni_pool)), replace=False))
        for tag, idx in (("uniform", ui), ("nightly", ni)):
            p = clf.predict(b.t[idx], b.mag[idx], m_base_ref=mb, t_start=0.0)
            res[tag].append(p.label == "NonPSPL")
        got += 1
    return {k: (float(np.mean(v)), len(v)) for k, v in res.items()}


@app.local_entrypoint()
def main(n_latency: int = 400, n_gap: int = 300, visits: int = 1200):
    import json, numpy as np
    thr = 0.9042405486106873   # frozen operating threshold from the validation split

    print(f"[P0.7] censor-aware latency on {n_latency} binaries (0.5 d grid)...")
    chunks = list(latency_chunk.starmap([(700000 + i * 5000, n_latency // 4, thr)
                                         for i in range(4)]))
    ev = [e for c in chunks for e in c]
    det = [e for e in ev if e["detected"]]
    frac_det = len(det) / max(len(ev), 1)
    lags = np.array([e["lag"] for e in det])
    lat = {"n_eligible": len(ev), "n_detected": len(det),
           "detection_fraction": round(frac_det, 3),
           "median_lag_detected_days": round(float(np.median(lags)), 2) if len(lags) else None,
           "p90_lag_days": round(float(np.percentile(lags, 90)), 2) if len(lags) else None,
           "pre_onset_flag_fraction": round(float((lags < 0).mean()), 3) if len(lags) else None}

    print(f"[P0.8] matched-density gap test on {n_gap} binaries at {visits} visits...")
    gchunks = list(gap_chunk.starmap([(900000 + i * 5000, n_gap // 3, visits) for i in range(3)]))
    gap = {}
    for tag in ("uniform", "nightly"):
        num = sum(m * k for m, k in [g[tag] for g in gchunks])
        den = sum(k for _, k in [g[tag] for g in gchunks])
        gap[tag] = round(num / max(den, 1), 3)
    gap["n_per_arm"] = sum(k for _, k in [g["uniform"] for g in gchunks])
    gap["visits"] = visits

    out = {"latency": lat, "gap_matched_density": gap}
    print(json.dumps(out, indent=2))
    json.dump(out, open("/tmp/latency_gaps_result.json", "w"), indent=2)

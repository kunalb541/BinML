#!/usr/bin/env python3
"""Reduce cascade_trace.npz to every cascade number the manuscript reports.

ONE SCAN, ONE ARTIFACT, MANY QUESTIONS.  The trace stores P(NonPSPL) at every 0.5 d cut for every
event under two revealing policies, so the evaluation grid, the alert rule, the persistence
requirement, the band set and the onset definition are all knobs turned HERE rather than reasons to
rerun the scan on a different event sample.  Every comparison below is therefore paired: the same
1,000 events under both settings.

DEFINITIONS, fixed once and used throughout.

  eligible event      a simulated binary whose anomaly survives detectability-conditioned
                      labelling and has a finite onset time.  Eligibility never depends on the
                      model's output.
  threshold crossing  a cut at which P(NonPSPL) >= the frozen operating threshold.
  alert               the first crossing that satisfies the persistence rule (default: one
                      crossing).  An event with no alert in the season is right-censored.
  premature alert     an alert whose time precedes the anomaly onset.
  lag                 alert time minus onset time; negative for premature alerts.

WHY THE ONSET DEFINITION MATTERS -- MORE THAN ANYTHING ABOUT THE MODEL.  ``_anomaly_onset_day``
defaults to a 10-point grid, so the recorded onset is rounded UP to a multiple of 7.2 d and any
alert inside that window is counted premature even when the anomaly was already detectable.  Worse,
the anomaly statistic is not monotone in revealed time: a single-lens refit can absorb an early
deviation as more of the season arrives, so an anomaly can be detectable at one cut and not at the
next.  "The onset" is therefore genuinely ambiguous, and the premature rate moves by more than an
order of magnitude across reasonable definitions.  We report all three and let them bracket the
answer rather than choosing one silently:

    first        first cut at which the anomaly is detectable at all  -> lowest premature rate
    persistent   cut after which it stays detectable to season end    -> highest premature rate
    coarse       the generator's 7.2 d grid (the previously published definition)

The primary numbers use `first`, on the same 0.5 d grid as the alerts, because an alert cannot
reasonably be called premature once the evidence has appeared.  The coarse result also serves as
the regression fixture tying this reduction to the previously published artifact.

Usage:  python validation/cascade_reduce.py [--trace PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NON = 2
Q_REGIMES = [("stellar", 1e-2, np.inf), ("giant", 1e-3, 1e-2),
             ("neptune", 1e-4, 1e-3), ("lowmass", 0.0, 1e-4)]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def alert_times(P, cuts, thr, stride=1, persist=1):
    """First alert day per event, or NaN if censored.

    `stride` subsamples the evaluation grid (1 -> 0.5 d, 2 -> 1.0 d, ...).  `persist` requires that
    many CONSECUTIVE crossings on the subsampled grid before an alert fires -- the debouncing a
    real broker would apply.  The alert time is the day the requirement is met, which is when a
    broker could actually send it.
    """
    p = P[:, ::stride]
    c = cuts[::stride]
    over = np.nan_to_num(p, nan=0.0) >= thr
    if persist > 1:
        run = np.zeros_like(over, dtype=np.int16)
        run[:, 0] = over[:, 0]
        for j in range(1, over.shape[1]):
            run[:, j] = np.where(over[:, j], run[:, j - 1] + 1, 0)
        over = run >= persist
    idx = np.argmax(over, axis=1)
    hit = over.any(axis=1)
    return np.where(hit, c[idx], np.nan)


def summarise(alert, onset):
    """Detection / premature / lag statistics for one (alert-rule, onset) combination."""
    n = len(alert)
    det = np.isfinite(alert)
    lag = alert - onset
    prem = det & (lag < 0)
    nd, npm = int(det.sum()), int(prem.sum())
    lo, hi = wilson(npm, n)
    dlo, dhi = wilson(nd, n)
    lag_det = lag[det]
    lag_np = lag[det & ~prem]
    lag_pos = lag[det & (lag > 0)]
    return {
        "n_eligible": n, "n_detected": nd, "n_censored": n - nd,
        "detection_fraction": round(nd / max(n, 1), 3),
        "detection_ci": [round(dlo, 3), round(dhi, 3)],
        "n_premature": npm,
        "premature_rate_of_eligible": round(npm / max(n, 1), 3),
        "premature_ci_of_eligible": [round(lo, 3), round(hi, 3)],
        "premature_rate_of_detected": round(npm / max(nd, 1), 3),
        # THREE populations, never conflated: the median over all detections mixes in the negative
        # lags of premature alerts and is NOT the typical post-onset delay.
        "median_lag_detected_days": round(float(np.median(lag_det)), 2) if lag_det.size else None,
        "n_lag_detected": int(lag_det.size),
        "median_lag_non_premature_days": round(float(np.median(lag_np)), 2) if lag_np.size else None,
        "n_lag_non_premature": int(lag_np.size),
        "median_lag_strictly_positive_days": (round(float(np.median(lag_pos)), 2)
                                              if lag_pos.size else None),
        "n_lag_strictly_positive": int(lag_pos.size),
    }


def paired_delta(a_alert, b_alert, onset, seed=7, n_boot=2000):
    """Paired comparison of two alert rules on the SAME events: difference in premature rate,
    with a bootstrap interval over events and the count of events that change status."""
    pa = np.isfinite(a_alert) & (a_alert - onset < 0)
    pb = np.isfinite(b_alert) & (b_alert - onset < 0)
    rng = np.random.default_rng(seed)
    n = len(onset)
    d = [float(pa[i].mean() - pb[i].mean())
         for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return {"delta_premature_rate": round(float(pa.mean() - pb.mean()), 3),
            "delta_ci95": [round(float(np.percentile(d, 2.5)), 3),
                           round(float(np.percentile(d, 97.5)), 3)],
            "n_changed_status": int((pa != pb).sum())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace", default=os.path.join(HERE, "cascade_trace.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "cascade_reproduce_result.json"))
    args = ap.parse_args(argv)

    if not os.path.exists(args.trace):
        raise SystemExit(f"FATAL: {args.trace} missing; run validation/cascade_trace.py first")
    d = np.load(args.trace, allow_pickle=False)
    cuts = d["cuts"].astype(np.float64)
    p146 = d["p_f146"].astype(np.float64)
    pmul = d["p_multi"].astype(np.float64)
    a146 = d["a_f146"]
    coarse = d["t_anom_coarse"].astype(np.float64)
    fine = d["t_anom_fine"].astype(np.float64)

    # THREE onset definitions, because the premature rate depends on this choice far more than on
    # anything about the model, and picking one silently would be picking a headline. The anomaly
    # detectability statistic is not monotone in revealed time -- a single-lens refit can absorb an
    # early deviation as more data arrives -- so "the onset" is genuinely ambiguous:
    #   coarse      the generator's 7.2 d grid; rounds UP, so it counts on-time alerts as premature
    #   first       the first cut at which the anomaly is detectable at all (most permissive)
    #   persistent  the cut after which it stays detectable for the rest of the season (strictest)
    # first <= persistent by construction, so the premature rate under `first` is a lower bound and
    # under `persistent` an upper bound on any definition in between.
    mask = d["onset_mask"].astype(bool) if "onset_mask" in d.files else None
    if mask is None:
        raise SystemExit("FATAL: trace lacks onset_mask; regenerate with the current "
                         "validation/cascade_trace.py")
    any_ = mask.any(1)
    onset_first = np.where(any_, cuts[np.argmax(mask, 1)], np.inf)
    # last index that is False, +1 -> first index from which the mask is all True
    rev_false = np.argmax(~mask[:, ::-1], axis=1)
    all_true = ~(~mask).any(1)
    idx_pers = np.where(all_true, 0, mask.shape[1] - rev_false)
    onset_pers = np.where(any_ & (idx_pers < mask.shape[1]), cuts[np.clip(idx_pers, 0, len(cuts) - 1)],
                          np.inf)
    bad_onset = int((np.isfinite(fine) & (np.abs(onset_first - fine) > 1e-6)).sum())
    if bad_onset:
        raise SystemExit(f"FATAL: onset mask disagrees with t_anom_fine for {bad_onset} events; "
                         f"the two are computed from the same statistic and must agree")
    seeds = d["seed"]
    params = d["params"]
    fields = list(d["param_fields"])
    prov = json.loads(str(d["provenance"]))
    metrics_path = os.path.join(REPO, "paper", "results", "metrics.json")
    thr = json.load(open(metrics_path))["headline"]["threshold"]

    # ---- regression fixture: this reduction must reproduce the published per-event rows --------
    #
    # WHAT IS REPRODUCIBLE.  Against a float32 trace this reduction reproduces the published
    # per-event artifact EXACTLY: same detection status and same crossing day for all 1,000 events.
    # The residual is recorded in the artifact rather than merely asserted, so a reader can see it
    # is zero instead of taking the assertion's word for it.
    #
    # This is a reduction of frozen float32 values; it performs no model inference or BLAS work.
    # The stored first-crossing status and day must therefore agree exactly with the earlier
    # per-event artifact. Cross-platform variation belongs at trace generation, not here.
    base = alert_times(p146, cuts, thr)
    ev_path = os.path.join(HERE, "cascade_events.json")
    published = json.load(open(ev_path))
    if [r["seed"] for r in published] != list(map(int, seeds)):
        raise SystemExit("FATAL: trace/event seed list differs")
    pub = np.array([np.nan if r["first_crossing_day"] is None else r["first_crossing_day"]
                    for r in published], float)
    status_mismatch = int((np.isfinite(base) != np.isfinite(pub)).sum())
    if status_mismatch:
        raise SystemExit(f"FATAL: {status_mismatch} events changed detection status against the "
                         f"published artifact. The simulator or checkpoint changed; do not "
                         f"publish this reduction under the old protocol.")
    both = np.isfinite(base) & np.isfinite(pub)
    delta = (base - pub)[both]
    n_diff = int((delta != 0).sum())
    max_dev = float(np.abs(delta).max()) if delta.size else 0.0
    if n_diff:
        raise SystemExit(f"FATAL: {n_diff}/{int(both.sum())} crossing days differ from the "
                         f"published artifact (max {max_dev} d). A reduction of the frozen trace "
                         f"must agree exactly.")
    fixture = {"n_events": int(len(pub)), "detection_status_mismatches": status_mismatch,
               "crossings_differing": n_diff, "max_deviation_days": max_dev,
               "note": "Exact residual vs the published per-event artifact; this reduction "
                       "contains no model inference, so any non-zero residual is fatal."}

    out = {
        "_doc": "Reduction of validation/cascade_trace.npz. Regenerate with "
                "validation/cascade_trace.py then validation/cascade_reduce.py; do not hand-edit.",
        "model": "shipped (cascade-trained)",
        "protocol": {
            "rule": "event-level first threshold crossing",
            "threshold": thr,
            "step_days": float(cuts[1] - cuts[0]),
            "bands_revealed": "F146 only (primary); all three bands reported as a sensitivity",
            "onset": "first day the anomaly is detectable, on the 0.5 d grid matched to the "
                     "alert grid (primary). The persistent-detectability onset and the 7.2 d "
                     "generator grid bracket it and are reported as sensitivities.",
            "persistence": "one crossing (primary); 2 and 3 consecutive crossings reported",
        },
        "provenance": prov,
        "reduction_provenance": {
            "input_trace_sha256": sha256_file(args.trace),
            "reducer_sha256": sha256_file(__file__),
            "threshold_source": "paper/results/metrics.json:headline.threshold",
            "threshold_source_sha256": sha256_file(metrics_path),
        },
        "upstream_provenance_limitation": (
            "The trace records a dirty source tree at its generation commit but not a source-tree "
            "hash or patch. Its checkpoint hash is recorded; its exact uncommitted generator "
            "source cannot be reconstructed from the trace alone."),
        "published_artifact_agreement": fixture,
    }

    primary = summarise(base, onset_first)
    out.update(primary)                      # top-level = the primary definition (macros read these)
    out["primary_definition"] = ("F146-only, 0.5 d alert grid, onset = first day the anomaly is "
                                 "detectable on the same grid, alert = single crossing")

    # ---- sensitivities, all on the same 1,000 events ------------------------------------------
    sens = {}
    sens["onset_definition"] = {
        "first_detectable": summarise(base, onset_first),
        "persistent_detectable": summarise(base, onset_pers),
        "coarse_7p2d_generator_grid": summarise(base, coarse),
        "note": "first <= persistent by construction, so these bracket the premature rate. The "
                "coarse grid is the previously published definition; it rounds the onset up and "
                "so counts alerts within 7.2 d of a detectable anomaly as premature.",
        "median_coarse_minus_first_days": round(float(np.median(coarse - onset_first)), 2),
        "median_persistent_minus_first_days": round(float(np.median(
            (onset_pers - onset_first)[np.isfinite(onset_pers)])), 2),
        "n_non_monotone": int(((np.diff(mask.astype(np.int8), axis=1) != 0).sum(1) > 1).sum()),
    }
    sens["onset_coarse_7p2d"] = sens["onset_definition"]["coarse_7p2d_generator_grid"]

    grid = {}
    for stride, label in ((1, "0.5"), (2, "1.0"), (4, "2.0")):
        a = alert_times(p146, cuts, thr, stride=stride)
        grid[label] = summarise(a, onset_first)
        if stride != 1:
            grid[label]["paired_vs_half_day"] = paired_delta(base, a, onset_first)
    sens["evaluation_grid_days"] = grid

    pers = {}
    for k in (1, 2, 3):
        a = alert_times(p146, cuts, thr, persist=k)
        pers[str(k)] = summarise(a, onset_first)
        if k != 1:
            pers[str(k)]["paired_vs_single"] = paired_delta(base, a, onset_first)
    sens["persistence_consecutive_crossings"] = pers

    a_multi = alert_times(pmul, cuts, thr)
    sens["bands"] = {"f146_only": primary, "all_three_bands": summarise(a_multi, onset_first)}
    sens["bands"]["all_three_bands"]["paired_vs_f146"] = paired_delta(base, a_multi, onset_first)

    over_arg = (a146 == NON)
    idx = np.argmax(over_arg, axis=1)
    a_arg = np.where(over_arg.any(axis=1), cuts[idx], np.nan)
    sens["argmax_rule"] = summarise(a_arg, onset_first)
    sens["argmax_rule"]["note"] = "alert when NonPSPL is the argmax class, ignoring the threshold"
    out["sensitivity"] = sens

    # ---- stratification ------------------------------------------------------------------------
    q = params[:, fields.index("q")]
    strat = {"by_mass_ratio": {}, "by_onset_third": {}}
    for name, lo, hi in Q_REGIMES:
        m = (q > lo) & (q <= hi)
        if m.sum() >= 20:
            s = summarise(base[m], onset_first[m])
            strat["by_mass_ratio"][name] = {k: s[k] for k in
                                            ("n_eligible", "detection_fraction",
                                             "premature_rate_of_eligible",
                                             "median_lag_non_premature_days")}
    edges = np.quantile(onset_first[np.isfinite(onset_first)], [0, 1 / 3, 2 / 3, 1])
    for i, name in enumerate(("early", "middle", "late")):
        m = ((onset_first >= edges[i]) & (onset_first <= edges[i + 1]) if i == 2 else
             (onset_first >= edges[i]) & (onset_first < edges[i + 1]))
        if m.sum() >= 20:
            s = summarise(base[m], onset_first[m])
            strat["by_onset_third"][name] = {
                "onset_range_days": [round(float(edges[i]), 1), round(float(edges[i + 1]), 1)],
                **{k: s[k] for k in ("n_eligible", "detection_fraction",
                                     "premature_rate_of_eligible",
                                     "median_lag_non_premature_days")}}
    out["stratified"] = strat

    json.dump(out, open(args.out, "w"), indent=2)
    keep = ("n_eligible", "n_detected", "detection_fraction", "premature_rate_of_eligible",
            "premature_ci_of_eligible", "premature_rate_of_detected",
            "median_lag_detected_days", "median_lag_non_premature_days", "n_lag_non_premature")
    print(json.dumps({k: out[k] for k in keep}, indent=2))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

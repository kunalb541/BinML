#!/usr/bin/env python3
"""Compare the cascade ablation arms at matched within-season detection counts.

WHY.  At a single frozen threshold the truncation-augmented arm alerted prematurely less often
(0.123 vs 0.193) but also detected far less often (0.643 vs 0.993).  Both facts follow from one
banal explanation -- the augmented model says NonPSPL less -- so that comparison cannot separate
"learned a better temporal ordering" from "is more conservative".  A model that never alerts has a
premature rate of zero.

WHAT THIS DOES.  Using the stored per-arm probability traces, choose each arm's threshold by an
order statistic so that both arms alert on exactly the SAME NUMBER of events.  If the augmented arm
is still lower at matched detection, the ordering pattern is not explained by one arm simply
alerting less often; if the curves lie on top of each other, the original difference was a threshold
shift and nothing more.

The whole curve is reported, not one operating point, because the answer can depend on where the
arms are matched.  This is an exploratory finite-sample risk--coverage analysis: the thresholds are
selected and compared on the same 400 events.  The McNemar values are therefore conditional
descriptions of this sample, not confirmatory population-level p-values.  A publishable test must
choose thresholds on a disjoint calibration sample and evaluate them once on paired held-out
events.

Usage:  python validation/cascade_matched_reduce.py [--traces matched_traces.npz]

The scan writes JSON (easier to move off Modal); convert it once to the committed .npz, which is
much smaller for the same content.  Keep the traces at float32: half precision has a spacing of
~5e-4 near the operating threshold, so a float16 round-trip silently moves crossings and changes
this comparison -- it did, once, and the result shifted on a storage-format change alone.

    modal volume get binml-ablation-data matched_traces.json
    python -c "import json,numpy as np; d=json.load(open('matched_traces.json')); \
      np.savez_compressed('validation/matched_traces.npz', cuts=np.array(d['cuts'],np.float32), \
      onset=np.array(d['onset'],np.float32), seeds=np.array(d['seeds'],np.int64), \
      arms=np.array(list(d['traces'])), \
      traces=np.stack([np.array(d['traces'][a],np.float32) for a in d['traces']]), \
      config=np.array(json.dumps(d['config'])), \
      provenance=np.array(json.dumps(d.get('provenance',{}))))"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


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


def premature_mask(P, cuts, onset, thr):
    over = np.nan_to_num(P, nan=0.0) >= thr
    hit = over.any(1)
    alert = np.where(hit, cuts[np.argmax(over, 1)], np.nan)
    return hit & (alert - onset < 0)


def mcnemar(a, b):
    """Conditional exact McNemar calculation on the same events.

    Only discordant pairs carry information.  Because this script also selects the thresholds on
    these events, the returned value is descriptive rather than a confirmatory p-value; see the
    module docstring and the note written into the result artifact.
    """
    from math import comb
    n01 = int((~a & b).sum())      # premature under b only
    n10 = int((a & ~b).sum())      # premature under a only
    n = n01 + n10
    if n == 0:
        return {"n_discordant": 0, "n_a_only": 0, "n_b_only": 0,
                "p_two_sided_conditional": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"n_discordant": n, "n_a_only": n10, "n_b_only": n01,
            "p_two_sided_conditional": float(p)}


def curve(P, cuts, onset, thresholds):
    """(detection rate, premature rate) for each threshold, on one arm."""
    out = []
    Pf = np.nan_to_num(P, nan=0.0)
    for t in thresholds:
        over = Pf >= t
        hit = over.any(1)
        alert = np.where(hit, cuts[np.argmax(over, 1)], np.nan)
        prem = hit & (alert - onset < 0)
        out.append((float(hit.mean()), float(prem.mean()), int(hit.sum()), int(prem.sum())))
    return out


def _scores(P):
    """Maximum within-season NonPSPL probability for each event."""
    s = np.nanmax(np.asarray(P, dtype=np.float64), axis=1)
    if not np.isfinite(s).all():
        raise ValueError("every event must have at least one finite probability")
    return s


def _attainable_counts(scores):
    """Detection counts attainable by thresholding ``scores`` with ``>=``."""
    _values, counts = np.unique(scores, return_counts=True)
    return {0, *np.cumsum(counts[::-1]).astype(int).tolist()}


def _common_count(scores_by_arm, requested):
    """Nearest count that every arm can attain; ties prefer lower alert burden."""
    common = set.intersection(*(_attainable_counts(s) for s in scores_by_arm.values()))
    if not common:
        raise ValueError("arms have no common attainable detection count")
    return min(common, key=lambda k: (abs(k - requested), k))


def _threshold_for_count(scores, count):
    """Largest observed threshold giving exactly ``count`` detections.

    Using the observed order statistic rather than a rounded threshold or an arbitrary sparse grid
    makes the displayed counts and the paired event masks identical.  ``count`` is required to be
    attainable, so a tie cannot straddle the boundary.
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    if count == 0:
        return float(np.nextafter(s[0], np.inf))
    if count == len(s):
        return float(s[-1])
    threshold = float(s[count - 1])
    if int((scores >= threshold).sum()) != count:
        raise ValueError(f"count {count} is not attainable because a score tie crosses it")
    return threshold


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--traces", default=os.path.join(HERE, "matched_traces.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "cascade_matched_result.json"))
    args = ap.parse_args(argv)
    if not os.path.exists(args.traces):
        raise SystemExit(f"FATAL: {args.traces} missing; run validation/modal_cascade_matched.py "
                         f"then convert the JSON as shown in this module's docstring")

    if args.traces.endswith(".npz"):
        z = np.load(args.traces, allow_pickle=False)
        d = {"cuts": z["cuts"], "onset": z["onset"], "config": json.loads(str(z["config"])),
             "seeds": z["seeds"] if "seeds" in z.files else None,
             "provenance": (json.loads(str(z["provenance"]))
                            if "provenance" in z.files else {}),
             "traces": {a: z["traces"][i] for i, a in enumerate(z["arms"])}}
    else:
        d = json.load(open(args.traces))
    cuts = np.array(d["cuts"], float)
    onset = np.array(d["onset"], float)
    n = len(onset)
    thresholds = np.unique(np.concatenate([
        np.linspace(0.01, 0.99, 197), np.array([0.9042405486106873]),
        1 - np.logspace(-4, -2, 20)]))

    arms = {}
    scores = {}
    for a, P in d["traces"].items():
        arr = np.array(P, float)
        arms[a] = curve(arr, cuts, onset, thresholds)
        scores[a] = _scores(arr)

    targets = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)

    res = {"_doc": __doc__.split("\n")[0],
           "n_eligible": n,
           "config": d.get("config", {}),
           "reduction_provenance": {
               "input_trace_sha256": sha256_file(args.traces),
               "reducer_sha256": sha256_file(__file__),
           },
           "trace_provenance": d.get("provenance", {}),
           "upstream_provenance_limitation": (None if d.get("provenance", {}).get("source_sha256")
               and d.get("provenance", {}).get("checkpoint_sha256") else
               "The stored scan contains its protocol configuration and event seeds but no "
               "source-tree or checkpoint hashes. The reduction is tied to the exact trace bytes, "
               "but the trace cannot be independently tied to immutable training/run state."),
           "analysis_status": "exploratory finite-sample risk-coverage comparison",
           "inference_note": ("Thresholds and paired outcomes use the same events. Conditional "
                              "McNemar values describe this finite sample and are not "
                              "confirmatory population-level p-values. Select thresholds on a "
                              "disjoint calibration set before inferential use."),
           "score_definition": "maximum P(NonPSPL) over the 72-day trace",
           "score_dtype": str(np.asarray(next(iter(d["traces"].values()))).dtype),
           "matching_rule": ("nearest detection count attainable by every arm; ties in distance "
                             "prefer the lower alert count; threshold is the full-precision "
                             "kth-largest observed score and the comparison is >= threshold"),
           "n_predeclared_targets": len(targets),
           "matched": {}, "curves": {}}
    for t in targets:
        requested = int(round(t * n))
        achieved = _common_count(scores, requested)
        row = {}
        masks = {}
        for a, P in d["traces"].items():
            P = np.asarray(P, dtype=np.float64)
            threshold = _threshold_for_count(scores[a], achieved)
            hit = scores[a] >= threshold
            prem = premature_mask(P, cuts, onset, threshold)
            if int(hit.sum()) != achieved:
                raise RuntimeError(f"{a}: selected {achieved} detections but obtained {hit.sum()}")
            lo, hi = wilson(int(prem.sum()), n)
            dlo, dhi = wilson(achieved, n)
            row[a] = {"threshold": threshold,
                      "threshold_hex": float(threshold).hex(),
                      "requested_detection_count": requested,
                      "achieved_detection_count": achieved,
                      "detection_fraction": achieved / n,
                      "detection_ci": [dlo, dhi],
                      "premature_rate_of_eligible": float(prem.mean()),
                      "premature_ci_of_eligible": [round(lo, 3), round(hi, 3)],
                      "premature_rate_of_detected": float(prem.sum() / max(achieved, 1)),
                      "n_premature": int(prem.sum())}
            masks[a] = prem
        if len(row) == 2:
            a, b = list(row)
            row["delta_premature_on_minus_off"] = (
                row[a]["premature_rate_of_eligible"] - row[b]["premature_rate_of_eligible"])
            row["mcnemar"] = mcnemar(masks[a], masks[b])
        res["matched"][f"detection_{t:.2f}"] = row

    # Retain a multiplicity-adjusted CONDITIONAL description, without rounding first. These are
    # not confirmatory p-values because threshold selection and testing use the same sample.
    m = len(res["matched"])
    for row in res["matched"].values():
        if "mcnemar" in row:
            p = row["mcnemar"]["p_two_sided_conditional"]
            row["mcnemar"]["p_bonferroni_conditional"] = min(1.0, p * m)
            row["mcnemar"]["n_comparisons"] = m

    for a, pts in arms.items():
        res["curves"][a] = [{"threshold": round(float(t), 4), "detection_fraction": round(p[0], 4),
                             "premature_rate": round(p[1], 4)}
                            for t, p in zip(thresholds, pts)]

    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "curves"}, indent=2))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

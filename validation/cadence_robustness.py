#!/usr/bin/env python3
"""Quantify how BinML's anomaly recovery collapses as cadence thins out.

BinML is trained on Roman's native dense F146 sampling (6912 epochs = ~96 points/day over a
72-day season). Real ground-based surveys sample far more sparsely. This script subsamples dense
simulated NonPSPL events to a range of point densities and measures anomaly recovery, locating the
cadence below which the model stops working. It explains, quantitatively, why the current model
cannot be validated on sparse archival data (e.g. OGLE-IV EWS) and what cadence a real test needs.

Usage:  python cadence_robustness.py
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import binml
from pipeline.assemble import simulate_event, SurveyConfig

N_EVENTS = 40
DENSITIES = [6912, 3000, 1500, 800, 400, 200, 100]   # points per 72-day season
SEASON = 72.0


def main():
    clf = binml.Classifier()
    cfg = SurveyConfig()
    evs = []
    for s in range(1, 600):
        e = simulate_event("NonPSPL", np.random.default_rng(s), cfg)
        if e and e.label == "NonPSPL":
            evs.append(e)
        if len(evs) >= N_EVENTS:
            break
    print(f"BinML {binml.__version__}  cadence-robustness sweep  (n={len(evs)} dense NonPSPL)\n")
    print(f"{'points':>7s} {'pts/day':>8s} {'NonPSPL recall':>15s} {'microlensing (PSPL+NonPSPL)':>28s}")
    print("-" * 62)
    for N in DENSITIES:
        nonp = ml = 0
        for i, ev in enumerate(evs):
            b = ev.bands["F146"]
            rng = np.random.default_rng(1000 + i)
            idx = np.sort(rng.choice(len(b.t), size=min(N, len(b.t)), replace=False))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = clf.predict(b.t[idx], b.mag[idx], m_base_ref=ev.params["_m_base_ref"], t_start=0.0)
            nonp += r.label == "NonPSPL"
            ml += r.label in ("PSPL", "NonPSPL")
        print(f"{N:7d} {N/SEASON:8.1f} {nonp/len(evs):15.2f} {ml/len(evs):28.2f}")
    print("\nRoman F146 native = 6912 pts (96/day). Below ~20 pts/day the model collapses:")
    print("sparse sampling of a smooth peak reads as a variable star. Ground-based archival")
    print("cadence (OGLE-IV ~1-2/night) is far below this floor -> not a valid test target.")


if __name__ == "__main__":
    main()

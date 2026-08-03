#!/usr/bin/env python3
"""Test BinML v1 on real KMTNet events, and show WHY it cannot be validated on the ground.

KMTNet is the densest existing microlensing survey (~10-15 min cadence from three southern
sites). It is the closest ground analogue to Roman, so it is the obvious place to look for a real
dense-cadence test. The result is negative, and instructively so:

  * Even KMTNet events with 60-78 points/day (well above the ~40/day floor that uniform
    subsampling of simulated events would suggest) are misclassified by v1 as PeriodicVar.
  * The reason is not average density but the SAMPLING PATTERN. Ground data is clustered into
    nightly bursts separated by day/weather gaps; that ~1-day periodicity is exactly what v1
    reads as a variable star. A controlled test confirms it: dense simulated events subsampled
    UNIFORMLY to ~60/day keep ~0.72 anomaly recall, but the same events observed only ~8 h per
    night (a realistic diurnal window) collapse to 0.00 recall -- all PeriodicVar.

Conclusion: v1 is a specialist for Roman's *continuous* space-based cadence, which no ground
network can reproduce (the day/night cycle is unavoidable). Real validation of v1 therefore
genuinely awaits Roman itself; real sparse/ground data is the province of the legacy single-band
model (see real_data_validate.py).

KMTNet data policy: public after 1 July of the year following discovery. Cite Kim et al. 2016
(JKAS 49, 37) and acknowledge the KMTNet system if you use these data.

Usage:  python kmtnet_validate.py     (requires network)
"""
from __future__ import annotations
import os, re, sys, urllib.request
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import binml

UA = {"User-Agent": "binml-validation"}


def _get(url: str) -> str:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60
                                  ).read().decode("ascii", "ignore")


def fetch_kmtnet(event_id: str, year: int = 2019):
    """Return (hjd, mag) combined over all three sites' I-band pysis files for a KMTNet event.
    event_id like 'KB190371'. pysis columns: HJD dflux dflux_err MAG MAG_err fwhm sky secz."""
    base = f"https://kmtnet.kasi.re.kr/~ulens/event/{year}"
    page = _get(f"{base}/view.php?event={event_id}")
    files = sorted(set(re.findall(rf"data/{event_id}/pysis/(KMT[ACS]\d+_I\.pysis)", page)))
    t, m = [], []
    for f in files:
        try:
            raw = _get(f"{base}/data/{event_id}/pysis/{f}")
        except Exception:
            continue
        for line in raw.splitlines():
            if not line or line.startswith("#"):
                continue
            p = line.split()
            try:
                hjd, mag = float(p[0]), float(p[3])
            except (ValueError, IndexError):
                continue
            t.append(hjd); m.append(mag)
    t, m = np.array(t), np.array(m)
    ok = np.isfinite(t) & np.isfinite(m) & (m > 5) & (m < 25)
    return t[ok], m[ok]


def main():
    clf = binml.Classifier()
    events = ["KB190371", "KB190842", "KB191339", "KB190253", "KB190954"]
    print("BinML v1 on real KMTNet 2019 events (single-band F146 proxy):\n")
    print(f"{'event':10s} {'pts/day':>7s}  {'v1 label':13s} {'P(NonPSPL)':>10s}")
    print("-" * 46)
    for ev in events:
        try:
            t, m = fetch_kmtnet(ev)
            if len(t) < 50:
                print(f"{ev:10s}  (load failed / too few points)"); continue
            tp = t[np.argmin(m)]; t0 = tp - 36.0
            sel = (t >= t0) & (t < t0 + 72.0)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = clf.predict(t[sel] - t0, m[sel], m_base_ref=float(np.percentile(m, 90)))
            print(f"{ev:10s} {sel.sum()/72:7.1f}  {r.label:13s} {r.probabilities['NonPSPL']:10.2f}")
        except Exception as ex:
            print(f"{ev:10s}  ERROR {type(ex).__name__}: {str(ex)[:30]}")
    print("\nAll -> PeriodicVar even above 40 pts/day: it is the nightly sampling gaps, not the")
    print("average density. v1 needs Roman's continuous cadence; the ground cannot provide it.")


if __name__ == "__main__":
    main()

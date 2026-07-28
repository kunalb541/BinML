#!/usr/bin/env python3
"""Validate BinML on REAL microlensing events (single-band mode).

BinML is trained on three Roman bands at 15-minute cadence. Real archival events are
ground-based and single-band, so this is a *morphology sanity check*, not a purity/completeness
measurement: we feed the real light curve as the F146 channel (colour bands masked) and ask
whether BinML's class matches the event's published type. Two real domain gaps apply and are
expected to cost performance: (i) wavelength -- OGLE I-band is not Roman F146 (fine for the
achromatic microlensing *shape*, which is what the anomaly decision uses); (ii) cadence --
OGLE-IV is ~nightly, far sparser than Roman's dense grid, so many time bins are empty.

Usage:  python real_data_validate.py            # fetch + classify the curated set
Requires network access (downloads OGLE-IV EWS phot.dat, cached locally).
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import binml
from binml.legacy.surveys import fetch_ogle_ews

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ogle_cache")
os.makedirs(CACHE, exist_ok=True)

# Curated OGLE-IV events with published type. "anomalous" = binary/planetary (expect NonPSPL);
# "single" = single-lens (expect PSPL). Types are from the discovery/analysis literature; the
# planetary events are well documented. Verify/extend this list as needed.
EVENTS = [
    # name,                 year, num,   published_type,  note
    ("OGLE-2016-BLG-1195",  2016, 1195,  "anomalous",  "Earth-mass planet (Bond+2017, Shvartzvald+2017)"),
    ("OGLE-2017-BLG-0482",  2017, 482,   "anomalous",  "planet (Han+2018)"),
    ("OGLE-2012-BLG-0026",  2012, 26,    "anomalous",  "two-planet system (Han+2013)"),
    ("OGLE-2013-BLG-0341",  2013, 341,   "anomalous",  "planet in binary (Gould+2014)"),
    ("OGLE-2018-BLG-0677",  2018, 677,   "anomalous",  "super-Earth (Herrera-Martin+2020)"),
    ("OGLE-2015-BLG-0966",  2015, 966,   "anomalous",  "planet (Street+2016)"),
    ("OGLE-2019-BLG-0960",  2019, 960,   "anomalous",  "planet (Yee+2021)"),
    ("OGLE-2017-BLG-1140",  2017, 1140,  "anomalous",  "planet (Calchi Novati+2018)"),
]

WINDOW = 72.0   # BinML season length (days)


def prep_window(t, m, e):
    """Center a 72-day window on the peak (brightest point); return (t_days, mag, m_base)."""
    ok = np.isfinite(t) & np.isfinite(m)
    t, m = t[ok], m[ok]
    t_peak = t[np.argmin(m)]                       # brightest = smallest magnitude
    t0 = t_peak - WINDOW / 2.0
    sel = (t >= t0) & (t < t0 + WINDOW)
    m_base = float(np.percentile(m, 90))            # faint (quiescent) level of the full curve
    return t[sel] - t0, m[sel], m_base, int(sel.sum())


def main():
    clf = binml.Classifier()
    print(f"BinML {binml.__version__}  |  single-band (F146-proxy) real-data check\n")
    header = f"{'event':22s} {'type':10s} {'npts':>5s}  {'pred':13s} {'P(PSPL)':>7s} {'P(NonPSPL)':>10s}  match"
    print(header); print("-" * len(header))
    rows = []
    for name, yr, num, typ, note in EVENTS:
        try:
            t, m, e = fetch_ogle_ews(yr, num, cache_dir=CACHE)
        except Exception as ex:
            print(f"{name:22s} {typ:10s}    --  FETCH FAILED: {type(ex).__name__}")
            continue
        td, mag, mbase, npts = prep_window(t, m, e)
        if npts < 10:
            print(f"{name:22s} {typ:10s} {npts:5d}  too few in-window points")
            continue
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = clf.predict(td, mag, m_base_ref=mbase)   # single band -> F146, colour masked
        p = r.probabilities
        # "match": for anomalous we want NonPSPL; the model is microlensing-correct if it says
        # PSPL or NonPSPL (i.e. recognises microlensing at all)
        is_ml = (r.label in ("PSPL", "NonPSPL"))
        want = "NonPSPL" if typ == "anomalous" else "PSPL"
        match = "yes" if r.label == want else ("ml" if is_ml else "NO")
        print(f"{name:22s} {typ:10s} {npts:5d}  {r.label:13s} {p['PSPL']:7.2f} {p['NonPSPL']:10.2f}  {match}")
        rows.append((name, typ, r.label, p, is_ml, match))
    # summary
    if rows:
        anom = [r for r in rows if r[1] == "anomalous"]
        n_ml = sum(r[4] for r in anom)
        n_nonpspl = sum(r[2] == "NonPSPL" for r in anom)
        print(f"\nOf {len(anom)} known anomalous events: {n_ml} recognised as microlensing "
              f"(PSPL or NonPSPL), {n_nonpspl} flagged NonPSPL (anomaly detected).")
        print("Reminder: sparse ground cadence is out-of-distribution for BinML; this is a "
              "morphology check, not a survey-performance number.")


if __name__ == "__main__":
    main()

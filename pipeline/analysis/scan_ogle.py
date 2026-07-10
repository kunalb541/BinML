#!/usr/bin/env python3
"""
Scan OGLE-IV EWS events: download phot.dat, measure cadence / SNR / completeness,
save promising light curves + a JSON of scored records for one (year, id-range) slice.

Prints a one-line summary. Full records go to <out>/scored_<year>_<start>_<count>.json
and good phot.dat files to <out>/e_<year>_<id>.dat
"""
import sys, os, json, argparse, ssl, time
import urllib.request
import numpy as np

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
BASE = "https://www.astrouw.edu.pl/ogle/ogle4/ews/{year}/blg-{id:04d}/phot.dat"


def fetch(year, eid, timeout=25):
    url = BASE.format(year=year, id=eid)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
            raw = r.read()
    except Exception:
        return None
    if b"DOCTYPE" in raw[:200] or b"html" in raw[:200].lower():
        return None
    return raw


def score(raw):
    """Parse and compute metrics; return dict or None."""
    try:
        a = np.loadtxt(raw.decode("ascii", "ignore").splitlines())
    except Exception:
        return None
    if a.ndim != 2 or a.shape[0] < 60 or a.shape[1] < 3:
        return None
    hjd, mag, err = a[:, 0], a[:, 1], a[:, 2]
    good = np.isfinite(hjd) & np.isfinite(mag) & np.isfinite(err) & (err < 1.0) & (mag > 5) & (mag < 25)
    hjd, mag, err = hjd[good], mag[good], err[good]
    if len(hjd) < 60:
        return None
    o = np.argsort(hjd); hjd, mag, err = hjd[o], mag[o], err[o]
    t0 = hjd[np.argmin(mag)]
    out = np.abs(hjd - t0) > 60
    base = np.median(mag[out]) if out.sum() > 20 else np.percentile(mag, 90)
    win = np.abs(hjd - t0) <= 36.0
    tw, mw, ew = hjd[win], mag[win], err[win]
    n_win = int(win.sum())
    if n_win < 40:
        return None
    # cadence: median consecutive gap in the DENSE region (+-20d of peak), in minutes
    dense = np.abs(tw - t0) <= 20.0
    td = np.sort(tw[dense])
    cad_min = float(np.median(np.diff(td)) * 1440.0) if len(td) > 5 else 1e9
    # amplitude / SNR
    amp = float(base - mw.min())                       # mag of brightening
    peakA = float(10 ** (0.4 * amp))
    scat = float(np.median(ew[dense])) if dense.sum() > 3 else float(np.median(ew))
    snr = float(amp / (scat + 1e-6))
    # completeness: baseline coverage on both wings + covers the peak
    left = (tw < t0 - 20).sum(); right = (tw > t0 + 20).sum()
    complete = bool(left >= 3 and right >= 3 and dense.sum() >= 15)
    return dict(n_win=n_win, cad_min=round(cad_min, 1), amp=round(amp, 2),
                peakA=round(peakA, 1), scatter=round(scat, 3), snr=round(snr, 1),
                complete=complete, t0=float(t0), base=round(float(base), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sleep", type=float, default=0.05)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    recs = []
    for eid in range(args.start, args.start + args.count):
        raw = fetch(args.year, eid)
        time.sleep(args.sleep)
        if raw is None:
            continue
        m = score(raw)
        if m is None:
            continue
        # keep it if it's a decent candidate at all (save disk only for promising)
        promising = (m["n_win"] >= 50 and m["amp"] >= 0.3 and
                     (m["cad_min"] <= 45 or m["snr"] >= 15))
        m.update(event=f"OGLE-{args.year}-BLG-{eid:04d}", year=args.year, id=eid)
        if promising:
            path = os.path.join(args.out, f"e_{args.year}_{eid:04d}.dat")
            with open(path, "wb") as f:
                f.write(raw)
            m["path"] = path
            recs.append(m)
    jf = os.path.join(args.out, f"scored_{args.year}_{args.start}_{args.count}.json")
    with open(jf, "w") as f:
        json.dump(recs, f)
    best_cad = sorted([r for r in recs if r["complete"]], key=lambda r: r["cad_min"])[:3]
    print(f"SLICE {args.year} {args.start}-{args.start+args.count-1}: "
          f"kept {len(recs)} promising; best cad(min)="
          f"{[ (r['id'], r['cad_min']) for r in best_cad ]}")


if __name__ == "__main__":
    main()

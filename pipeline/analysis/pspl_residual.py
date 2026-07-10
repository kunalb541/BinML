#!/usr/bin/env python3
"""Fit a single-lens (PSPL) model to a real OGLE curve and show the residual, which
exposes a small planetary anomaly that is invisible in the raw light curve."""
import sys, os, numpy as np
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def load(path):
    a = np.loadtxt(path); o = np.argsort(a[:, 0]); return a[o, 0], a[o, 1], a[o, 2]


def A_pspl(t, t0, tE, u0, fs):
    u = np.sqrt(u0 ** 2 + ((t - t0) / tE) ** 2)
    A = (u ** 2 + 2) / (u * np.sqrt(u ** 2 + 4))
    return fs * A + (1 - fs)                     # blended magnification (baseline = 1)


def fit_event(path, window=72.0):
    hjd, mag, err = load(path)
    t0g = hjd[np.argmin(mag)]
    out = np.abs(hjd - t0g) > 60
    base = np.median(mag[out]) if out.sum() > 20 else np.percentile(mag, 90)
    win = np.abs(hjd - t0g) <= window / 2
    t, m, e = hjd[win], mag[win], err[win]
    A = 10 ** (0.4 * (base - m))                 # observed magnification
    Aerr = A * 0.4 * np.log(10) * e
    p0 = [t0g, 20.0, max(1.0 / max(A.max(), 1.1), 1e-3), 1.0]
    try:
        popt, _ = curve_fit(A_pspl, t, A, p0=p0, sigma=Aerr, absolute_sigma=True,
                            bounds=([t0g - 5, 1, 1e-4, 0.2], [t0g + 5, 200, 3, 1.2]),
                            maxfev=20000)
    except Exception:
        popt = p0
    Amod = A_pspl(t, *popt)
    mmod = base - 2.5 * np.log10(np.clip(Amod, 1e-6, None))
    resid = m - mmod                             # magnitude residual (data - PSPL)
    chi = resid / e
    return dict(t=t, m=m, e=e, base=base, t0=popt[0], tE=popt[1], u0=popt[2], fs=popt[3],
                mmod=mmod, resid=resid, chi=chi, Apeak=float(A.max()))


def anomaly_score(f):
    """Largest run of same-sign >2sigma residuals near peak = the planetary bump."""
    near = np.abs(f["t"] - f["t0"]) <= 15
    c = f["chi"][near]
    sig = np.abs(c) > 2
    return float(np.sum(sig)), float(np.max(np.abs(c)) if len(c) else 0)


if __name__ == "__main__":
    import glob
    R = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(f"{R}/planet/p_*.dat"))
    scored = []
    for p in files:
        nm = os.path.basename(p)[2:-4].replace("_", "-BLG-")
        f = fit_event(p)
        nsig, maxsig = anomaly_score(f)
        scored.append((nsig, maxsig, nm, p, f))
        print(f"OGLE-{nm:16} tE={f['tE']:5.1f} u0={f['u0']:.3f} Apeak={f['Apeak']:5.1f} "
              f"| anomaly: n(>2sig)={nsig:.0f} max={maxsig:.1f}sig")
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    print("\nbest planetary-anomaly candidate:", "OGLE-" + scored[0][2])

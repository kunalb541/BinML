"""
BinML v5 — classical Delta-chi^2 baseline.

This is the comparison a referee will ask for, and it is the FAIREST possible baseline for
this task: it is the same decision rule the LABELS were defined by (assemble.py classifies on
delta-chi^2 against flat, then against a refit PSPL), only applied to NOISY OBSERVED data
rather than the noise-free truth. Any margin the network shows over it is therefore margin
from reading light-curve morphology that a PSPL fit cannot express -- variable-star structure,
colour information, partial-window shapes -- rather than from having better access to the
labelling rule.

Recovering the uncertainties
----------------------------
The binned cache stores (mean, min, max) of the baseline-relative magnitude and the observed
FRACTION per bin, but not the per-epoch errors. They do not need to be stored: photometric
noise is a deterministic function of magnitude, so

    sigma_bin = photometric_sigma(band, m_base_ref + mag_rel) / sqrt(n_epochs_in_bin)

reconstructs them exactly, with the sqrt(n) accounting for the averaging the binning already
did. n_epochs_in_bin = frac * BIN_FACTOR.

Cost
----
A per-event non-linear PSPL fit is ~10-30 ms, so the full 443k test set would take hours.
``--max-events`` subsamples; a few tens of thousands is far more than enough to place a
baseline PR curve, and the subsample is random so it is unbiased.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Tuple

import numpy as np

from .cache import BIN_FACTORS
from .classes import CLASS_NAMES
from .photometry import ROMAN_BANDS, photometric_sigma

I_NON = CLASS_NAMES.index("NonPSPL")
REF = "F146"


def pspl_mag(t, t0, tE, u0, fs, mbase):
    u = np.sqrt(u0 ** 2 + ((t - t0) / max(tE, 1e-3)) ** 2)
    u = np.maximum(u, 1e-8)
    A = (u ** 2 + 2.0) / (u * np.sqrt(u ** 2 + 4.0))
    return mbase - 2.5 * np.log10(np.maximum(1.0 + fs * (A - 1.0), 1e-8))


def fit_event(mag_rel: np.ndarray, frac: np.ndarray, m_base: float,
              window_days: float = 72.0) -> Tuple[float, float, float]:
    """Return (dchi2_event, dchi2_anomaly, peak_amplitude) from NOISY binned data.

    dchi2_event   = chi2(flat) - chi2(best PSPL)     -- is there an event at all?
    dchi2_anomaly = chi2(best PSPL) - n_dof          -- residual the PSPL cannot absorb
    """
    from scipy.optimize import least_squares

    obs = np.isfinite(mag_rel) & (frac > 0)
    if obs.sum() < 20:
        return 0.0, 0.0, 0.0
    nb = mag_rel.size
    t = np.linspace(0.0, window_days, nb, endpoint=False)[obs]
    m = m_base + mag_rel[obs]
    n_in_bin = np.maximum(frac[obs] * BIN_FACTORS[REF], 1.0)
    sig = photometric_sigma(ROMAN_BANDS[REF], m) / np.sqrt(n_in_bin)
    sig = np.maximum(sig, 1e-4)

    # flat model: weighted mean is the ML baseline
    w = 1.0 / sig ** 2
    flat = float(np.sum(m * w) / np.sum(w))
    chi2_flat = float(np.sum(((m - flat) / sig) ** 2))

    # PSPL fit. Seed t0 at the extremum, tE from the window, u0 from the amplitude.
    i_pk = int(np.argmin(m))                      # brightest bin
    amp = float(flat - m[i_pk])
    A_guess = max(10.0 ** (0.4 * max(amp, 0.0)), 1.0 + 1e-3)
    u0_guess = float(np.sqrt(max(2.0 / np.sqrt(1.0 - A_guess ** -2) - 2.0, 1e-4))) if A_guess > 1.001 else 1.0
    p0 = np.array([t[i_pk], np.log(max(window_days / 4.0, 1.0)),
                   min(max(u0_guess, 1e-3), 3.0), 0.0, flat])

    def resid(p):
        t0, log_tE, u0, logit_fs, mb = p
        tE = float(np.exp(np.clip(log_tE, -5, 8)))
        fs = 1.0 / (1.0 + np.exp(-np.clip(logit_fs, -30, 30)))
        return (m - pspl_mag(t, t0, tE, abs(u0), fs, mb)) / sig

    best = p0
    try:
        f = least_squares(resid, p0, method="lm", max_nfev=300)
        if float(np.sum(f.fun ** 2)) < float(np.sum(resid(p0) ** 2)):
            best = f.x
    except Exception:
        pass
    chi2_pspl = float(np.sum(resid(best) ** 2))
    dof = max(obs.sum() - 5, 1)
    return chi2_flat - chi2_pspl, chi2_pspl - dof, amp


def run(cache: str, out: str, max_events: int = 30000, seed: int = 11,
        window_days: float = 72.0) -> dict:
    meta = json.load(open(os.path.join(cache, "meta.json")))
    n = int(meta["n_events"])
    L = BIN_FACTORS[REF] and 864
    feat = np.memmap(os.path.join(cache, f"feat_{REF}.f16"), dtype=np.float16, mode="r",
                     shape=(n, L, 3))
    fr = np.memmap(os.path.join(cache, f"frac_{REF}.f16"), dtype=np.float16, mode="r",
                   shape=(n, L))
    lab = np.load(os.path.join(cache, "label.npy"))
    mb = np.load(os.path.join(cache, "m_base_ref.npy"))
    kp = np.load(os.path.join(cache, "keep_prob.npy"))

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.permutation(n)[:min(max_events, n)])
    d_ev = np.zeros(len(idx)); d_an = np.zeros(len(idx)); amp = np.zeros(len(idx))
    for i, j in enumerate(idx):
        d_ev[i], d_an[i], amp[i] = fit_event(
            np.asarray(feat[j, :, 0], dtype=np.float64),
            np.asarray(fr[j], dtype=np.float64), float(mb[j]), window_days)
        if (i + 1) % 2000 == 0:
            print(f"  fitted {i+1}/{len(idx)}", flush=True)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, idx=idx, dchi2_event=d_ev, dchi2_anomaly=d_an, amplitude=amp,
             label=lab[idx], keep_prob=kp[idx])
    return {"n": len(idx), "out": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-events", type=int, default=30000)
    args = ap.parse_args(argv)
    r = run(args.cache, args.out, args.max_events)
    print(f"baseline fits: {r['n']:,} events -> {r['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

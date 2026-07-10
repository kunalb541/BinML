"""
Turn a raw light curve (time, magnitude, error) into the exact tensor representation the
BinML network was trained on.

The model consumes two channels:
  * magnification  A = 10 ** (0.4 * (m_base - mag))   (baseline A = 1.0)
  * delta_t        days since the previous observation

Observations are moved to a contiguous prefix (compaction) and the network is told how
many are valid via ``length``; everything past ``length`` is ignored. This matches the
Roman training pipeline exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = ["Preprocessed", "to_magnification", "preprocess"]

DEFAULT_WINDOW_DAYS = 72.0  # Roman observing-season length the model was trained on


@dataclass
class Preprocessed:
    """Model-ready arrays plus the physical context used to build them."""
    flux: np.ndarray        # (1, seq_len) compacted magnification A
    delta_t: np.ndarray     # (1, seq_len) compacted gaps (days)
    length: int             # number of valid observations in the prefix
    t0: float               # peak time used to centre the window
    m_base: float           # baseline magnitude
    peak_magnification: float
    time: np.ndarray        # windowed observation times (days), sorted
    mag: np.ndarray         # windowed magnitudes
    mag_err: np.ndarray     # windowed magnitude errors
    magnification: np.ndarray  # windowed A values (time order)


def to_magnification(mag: np.ndarray, m_base: float) -> np.ndarray:
    """Convert magnitudes to normalized magnification A (baseline = 1.0)."""
    return 10.0 ** (0.4 * (m_base - np.asarray(mag, dtype=float)))


def _estimate_baseline(time: np.ndarray, mag: np.ndarray, t0: float) -> float:
    """Robust out-of-event baseline: median of points far from the peak."""
    out = np.abs(time - t0) > 60.0
    if out.sum() > 20:
        return float(np.median(mag[out]))
    return float(np.percentile(mag, 90))  # fall back to the faint mode


def preprocess(
    time: np.ndarray,
    mag: np.ndarray,
    mag_err: Optional[np.ndarray] = None,
    *,
    seq_len: int = 6912,
    window_days: float = DEFAULT_WINDOW_DAYS,
    t0: Optional[float] = None,
    m_base: Optional[float] = None,
    is_flux: bool = False,
) -> Preprocessed:
    """
    Parameters
    ----------
    time : array
        Observation times in days (e.g. HJD).
    mag : array
        Magnitudes (default), or fluxes if ``is_flux=True``.
    mag_err : array, optional
        Per-point errors (used only for plotting/repr, not by the model).
    seq_len : int
        Model sequence length (read this from the checkpoint; 6912 for the shipped models).
    window_days : float
        Length of the window centred on the peak (72 d = one Roman season).
    t0 : float, optional
        Peak time. If ``None``, the brightest point is used.
    m_base : float, optional
        Baseline magnitude. If ``None``, estimated from out-of-event points.
    is_flux : bool
        If True, ``mag`` is treated as flux; converted to magnification by dividing by the
        out-of-event median flux.
    """
    time = np.asarray(time, dtype=float)
    y = np.asarray(mag, dtype=float)
    err = (np.asarray(mag_err, dtype=float) if mag_err is not None
           else np.full(len(time), 0.01))
    order = np.argsort(time)
    time, y, err = time[order], y[order], err[order]

    if is_flux:
        # convert flux to magnitude-like via an arbitrary zero point, then proceed
        y = -2.5 * np.log10(np.clip(y, 1e-30, None))

    if t0 is None:
        t0 = float(time[np.argmin(y)])
    if m_base is None:
        m_base = _estimate_baseline(time, y, t0)

    inwin = np.abs(time - t0) <= window_days / 2.0
    tw, mw, ew = time[inwin], y[inwin], err[inwin]
    A = to_magnification(mw, m_base)

    n = len(A)
    if n > seq_len:  # keep the earliest seq_len (real curves never approach this)
        tw, mw, ew, A = tw[:seq_len], mw[:seq_len], ew[:seq_len], A[:seq_len]
        n = seq_len

    flux = np.zeros((1, seq_len), np.float32)
    dt = np.zeros((1, seq_len), np.float32)
    flux[0, :n] = A
    if n > 1:
        dt[0, 1:n] = np.diff(tw).astype(np.float32)

    return Preprocessed(
        flux=flux, delta_t=dt, length=int(max(n, 1)),
        t0=float(t0), m_base=float(m_base),
        peak_magnification=float(A.max()) if n else 1.0,
        time=tw, mag=mw, mag_err=ew, magnification=A,
    )

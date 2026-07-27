"""Turn raw multi-band light curves into the model's binned input.

A scientist has ``(time, mag, mag_err)`` arrays per band. The model consumes fixed-length
per-band **token** tensors (per bin: baseline-relative mean/min/max, observed-fraction,
observed-mask). This module bridges the two, reproducing the *exact* binning the model was
trained on (``pipeline.cache.bin_curve``): F146 → 864 bins, F087/F213 → 96 bins over a 72-day
season, keeping bin min/max so caustic spikes survive.

Key inputs a user controls:
  * ``m_base`` — the source's baseline (quiescent) magnitude per band. Microlensing *brightens*
    the source, so the baseline is the faint out-of-event level. Provide it if you know it
    (from a catalogue); otherwise it is estimated from the faint tail, which is only reliable
    for short, well-sampled events.
  * ``t_start`` — the day the 72-day analysis window opens. Defaults to the first observation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from pipeline.model import BAND_BINS

# Epochs pooled per bin -- MUST equal pipeline.cache.BIN_FACTORS (defined here so inference
# needs only torch+numpy, not the h5py/scipy training stack).
BIN_FACTORS = {"F146": 8, "F087": 3, "F213": 3}

WINDOW_DAYS = 72.0
__all__ = ["BAND_BINS", "estimate_baseline", "bin_band", "to_tokens", "Tokens"]


@dataclass
class Tokens:
    """Binned, model-ready tokens for one event."""
    feat: Dict[str, np.ndarray]   # band -> (nbins, 3) baseline-relative mean/min/max, NaN = empty
    frac: Dict[str, np.ndarray]   # band -> (nbins,) observed fraction
    m_base: Dict[str, float]      # baseline magnitude used per band
    t_start: float
    n_points: Dict[str, int] = field(default_factory=dict)


def estimate_baseline(mag: np.ndarray) -> float:
    """Faint-tail baseline estimate. Microlensing brightens the source, so the quiescent
    level is the FAINT side of the distribution; the 80th percentile is a robust proxy for a
    short event where most epochs are at baseline. Prefer passing a catalogue value."""
    mag = np.asarray(mag, float)
    mag = mag[np.isfinite(mag)]
    return float(np.percentile(mag, 80)) if mag.size else 0.0


def bin_band(time: np.ndarray, mag: np.ndarray, band: str, m_base_ref: float,
             t_start: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """Bin one band's observations onto the model's fixed grid.

    Returns ``(feat[nbins,3], frac[nbins], n_used)``. Bin ``i`` covers
    ``[t_start + i·w, t_start + (i+1)·w)`` with ``w = 72/nbins`` days — the same partition the
    training cache produces by pooling epochs, so a Roman-cadence curve bins identically here
    and in training.
    """
    nbins = BAND_BINS[band]
    factor = BIN_FACTORS[band]             # epochs per bin (frac denominator, matches cache.py)
    n_epochs = nbins * factor              # Roman grid: F146 6912, colour 288
    step = WINDOW_DAYS / n_epochs          # epoch spacing in days
    t = np.asarray(time, float); m = np.asarray(mag, float)
    ok = np.isfinite(t) & np.isfinite(m)
    t, m = t[ok], m[ok]
    # Snap to the nearest Roman epoch, then bin by INTEGER epoch index (epoch // factor),
    # exactly as the training cache pools epochs. Float time/width division rounds points on
    # a bin boundary into the wrong bin; integer epoch indexing does not.
    epoch = np.round((t - t_start) / step).astype(int)
    inwin = (epoch >= 0) & (epoch < n_epochs)
    idx = (epoch[inwin] // factor)
    dm = m[inwin] - m_base_ref             # ONE reference baseline for all bands (colour survives)
    feat = np.full((nbins, 3), np.nan, np.float32)
    frac = np.zeros(nbins, np.float32)
    if idx.size:
        for b in np.unique(idx):
            v = dm[idx == b]
            feat[b] = (v.mean(), v.min(), v.max())
            frac[b] = min(v.size / factor, 1.0)
    return feat, frac, int(idx.size)


def to_tokens(bands: Dict[str, Tuple[np.ndarray, np.ndarray]],
              m_base_ref: Optional[float] = None,
              t_start: Optional[float] = None) -> Tokens:
    """Bin a multi-band event.

    ``bands`` maps band name -> ``(time_days, mag)``. **F146 is required**; F087/F213 are
    optional (the model masks absent colour bands). Times are in days, shared zero point.

    ``m_base_ref`` is the SINGLE reference baseline (the F146 quiescent magnitude) subtracted
    from every band, exactly as the training cache does — this is what lets the colour bands
    carry the source colour. If omitted it is estimated from F146.
    """
    if "F146" not in bands:
        raise ValueError("F146 is required (it is the band the model always expects present)")
    if t_start is None:
        t_start = min(float(np.nanmin(np.asarray(t))) for t, _ in bands.values())
    if m_base_ref is None:
        m_base_ref = estimate_baseline(bands["F146"][1])
    feat, frac, npts = {}, {}, {}
    for b in BAND_BINS:
        if b in bands:
            t, m = bands[b]
            feat[b], frac[b], npts[b] = bin_band(t, m, b, m_base_ref, t_start)
        else:                                   # absent band -> all-empty (model masks it out)
            feat[b] = np.full((BAND_BINS[b], 3), np.nan, np.float32)
            frac[b] = np.zeros(BAND_BINS[b], np.float32)
            npts[b] = 0
    return Tokens(feat=feat, frac=frac, m_base={"ref": float(m_base_ref)},
                  t_start=float(t_start), n_points=npts)

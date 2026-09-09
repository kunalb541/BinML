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

import warnings
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
    """Faint-tail baseline estimate (a FALLBACK; pass a catalogue value when you can).

    Microlensing brightens the source, so the quiescent level is the FAINT (large-magnitude)
    side. The 90th percentile is used rather than the median because a blended or long event
    keeps most epochs slightly brightened, so the median is biased bright — and a bright
    baseline error makes the whole (baseline-relative) curve read as slow variability, which
    can flip a real microlensing event to LongPeriodVar.

    The price is the opposite failure on a QUIESCENT source: for pure noise the 90th percentile
    sits 1.28 sigma faint of the true level, so the baseline-relative curve reads as a constant
    brightening of 1.28 sigma. At typical Roman noise (sigma ~ 0.02-0.1 mag) that straddles the
    0.02 mag amplitude floor the classes are defined on, and a flat star can come back PSPL or
    LongPeriodVar. There is no estimator that is unbiased for both cases from one 72-day window;
    provide ``m_base_ref`` (a catalogue or multi-season baseline) for anything that matters."""
    mag = np.asarray(mag, float)
    mag = mag[np.isfinite(mag)]
    return float(np.percentile(mag, 90)) if mag.size else 0.0


def bin_band(time: np.ndarray, mag: np.ndarray, band: str, m_base_ref: float,
             t_start: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """Bin one band's observations onto the model's fixed grid.

    Returns ``(feat[nbins,3], frac[nbins], n_used)``, where ``n_used`` counts distinct Roman
    epochs with an observation. Bin ``i`` covers ``[t_start + i·w, t_start + (i+1)·w)`` with
    ``w = 72/nbins`` days — the same partition the training cache produces by pooling epochs.
    Observations are first reduced to one value per 15-min Roman epoch (mean of any that share an
    epoch), then binned; for one-observation-per-epoch input this is bit-identical to binning the
    raw observations, and for denser input it reproduces the cache's representation instead of
    inflating the min/max and frac channels with the sampling density.
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
    dm = m[inwin] - m_base_ref             # ONE reference baseline for all bands (colour survives)
    epoch = epoch[inwin]
    feat = np.full((nbins, 3), np.nan, np.float32)
    frac = np.zeros(nbins, np.float32)
    if epoch.size == 0:
        return feat, frac, 0
    # ONE VALUE PER ROMAN EPOCH FIRST. The training cache (pipeline.cache.bin_curve) holds exactly
    # one magnitude per epoch and pools at most `factor` of them per bin; its min/max are extrema
    # over <= factor epoch values and its frac is (observed epochs)/factor. Input sampled finer
    # than the grid (RMDC26's 12.1-min F146; any survey with repeat exposures) lands several
    # observations on one epoch index. Pooling those raw observations directly would inflate the
    # min/max channels with the sampling density and push frac past 1 -- the equivalence with the
    # cache holds only for on-grid input. Averaging the observations that share an epoch first
    # restores the cache's representation; for one-observation-per-epoch input (all training and
    # evaluation data) this is bit-identical to pooling the raw observations.
    uep, inv = np.unique(epoch, return_inverse=True)
    epval = np.bincount(inv, weights=dm) / np.bincount(inv)
    idx = uep // factor
    for b in np.unique(idx):
        v = epval[idx == b]
        feat[b] = (v.mean(), v.min(), v.max())
        frac[b] = min(v.size / factor, 1.0)
    return feat, frac, int(uep.size)


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
        raise ValueError(
            f"F146 is required (the model always expects it present); got bands {sorted(bands)}. "
            f"Recognised bands are {sorted(BAND_BINS)}.")
    unknown = [b for b in bands if b not in BAND_BINS]
    if unknown:
        raise ValueError(f"unrecognised band(s) {unknown}; expected a subset of {sorted(BAND_BINS)}")
    for b, v in bands.items():
        ta, ma = np.asarray(v[0], float).ravel(), np.asarray(v[1], float).ravel()
        if ta.size != ma.size:
            raise ValueError(f"{b}: time and magnitude must have equal length, "
                             f"got {ta.size} and {ma.size}")
        # An empty COLOUR band is legitimate -- the model masks absent bands, and a partially
        # revealed season (predict_evolution) can genuinely contain no colour epochs yet. Only
        # F146, which the model always requires, may not be empty.
        if ta.size == 0 and b == "F146":
            raise ValueError("F146: empty light curve (no observations)")
    bands = {b: v for b, v in bands.items()
             if b == "F146" or np.asarray(v[0], float).size > 0}
    t146, m146 = bands["F146"]
    if not np.isfinite(np.asarray(t146, float)).any() or not np.isfinite(np.asarray(m146, float)).any():
        raise ValueError("F146 has no finite (time, magnitude) observations")
    if t_start is None:
        t_start = min(float(np.nanmin(np.asarray(t))) for t, _ in bands.values()
                      if np.isfinite(np.asarray(t, float)).any())
    elif not np.isfinite(t_start):
        raise ValueError(f"t_start must be finite, got {t_start!r}")
    if m_base_ref is None:
        m_base_ref = estimate_baseline(m146)
        warnings.warn(
            "m_base_ref not given; estimating the F146 baseline as the 90th percentile. This is "
            "biased in BOTH directions: a long or blended event biases it bright (microlensing "
            "-> LongPeriodVar), and a quiescent source biases it 1.28 sigma faint (Flat -> "
            "PSPL/LongPeriodVar). Pass the catalogue F146 baseline magnitude.",
            stacklevel=2)
    elif not np.isfinite(m_base_ref):
        raise ValueError(f"m_base_ref must be finite, got {m_base_ref!r}")
    feat, frac, npts = {}, {}, {}
    for b in BAND_BINS:
        if b in bands:
            t, m = bands[b]
            ta = np.asarray(t, float)
            n_out = int(((ta < t_start) | (ta > t_start + WINDOW_DAYS)).sum())
            if n_out and n_out > 0.05 * max(np.isfinite(ta).sum(), 1):
                warnings.warn(f"{b}: {n_out} observations fall outside the 72-day window from "
                              f"t_start={t_start:.1f} and are dropped; set t_start to window them.",
                              stacklevel=2)
            feat[b], frac[b], npts[b] = bin_band(t, m, b, m_base_ref, t_start)
        else:                                   # absent band -> all-empty (model masks it out)
            feat[b] = np.full((BAND_BINS[b], 3), np.nan, np.float32)
            frac[b] = np.zeros(BAND_BINS[b], np.float32)
            npts[b] = 0
    if npts.get("F146", 0) == 0:
        raise ValueError("no F146 observations fell inside the 72-day window from "
                         f"t_start={t_start:.1f}; check units (days) and t_start")
    return Tokens(feat=feat, frac=frac, m_base={"ref": float(m_base_ref)},
                  t_start=float(t_start), n_points=npts)

"""
Light-curve loaders for common microlensing survey formats.

Every loader returns ``(time, mag, mag_err)`` as float arrays, ready for
``Classifier.predict``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

__all__ = ["load_ogle", "load_moa", "load_generic", "read_lightcurve", "fetch_ogle_ews"]


def _txt(path) -> np.ndarray:
    return np.loadtxt(str(path))


def load_ogle(path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OGLE EWS ``phot.dat``: columns = HJD, I-mag, mag_err, seeing, sky."""
    a = _txt(path)
    return a[:, 0], a[:, 1], a[:, 2]


def load_moa(path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MOA-style photometry: HJD, flux, flux_err, ... -> converted to mag.

    MOA publishes difference flux; we convert to an instrumental magnitude. If the file is
    already (HJD, mag, err) it is passed through.
    """
    a = _txt(path)
    hjd, c1, c2 = a[:, 0], a[:, 1], a[:, 2]
    if np.nanmedian(c1) > 5:          # looks like magnitudes already
        return hjd, c1, c2
    flux = c1 - np.nanmin(c1) + 1.0   # shift to positive
    mag = -2.5 * np.log10(np.clip(flux, 1e-6, None)) + 25.0
    err = 2.5 / np.log(10) * (c2 / np.clip(flux, 1e-6, None))
    return hjd, mag, err


def load_generic(path, cols=(0, 1, 2), skiprows=0, delimiter=None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Any whitespace/CSV table with time, mag, err columns (0-indexed ``cols``)."""
    a = np.loadtxt(str(path), skiprows=skiprows, delimiter=delimiter)
    i, j, k = cols
    return a[:, i], a[:, j], a[:, k]


def read_lightcurve(path, fmt="auto", **kw) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch by format. ``fmt`` in {'auto','ogle','moa','generic'}.

    'auto' picks OGLE if the file has >=5 columns, else generic.
    """
    fmt = fmt.lower()
    if fmt == "ogle":
        return load_ogle(path)
    if fmt == "moa":
        return load_moa(path)
    if fmt == "generic":
        return load_generic(path, **kw)
    a = _txt(path)
    if a.ndim == 2 and a.shape[1] >= 5:
        return a[:, 0], a[:, 1], a[:, 2]     # OGLE-like
    return a[:, 0], a[:, 1], (a[:, 2] if a.shape[1] > 2 else np.full(len(a), 0.01))


def fetch_ogle_ews(year: int, event: int, cache_dir=None
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Download an OGLE-IV EWS event by (year, number) and return (time, mag, err).

    e.g. ``fetch_ogle_ews(2017, 482)`` for OGLE-2017-BLG-0482. Requires network access.
    """
    import urllib.request

    url = f"https://www.astrouw.edu.pl/ogle/ogle4/ews/{year}/blg-{event:04d}/phot.dat"
    req = urllib.request.Request(url, headers={"User-Agent": "binml"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    if b"html" in raw[:200].lower():
        raise FileNotFoundError(f"OGLE-{year}-BLG-{event:04d} not found at {url}")
    if cache_dir:
        p = Path(cache_dir) / f"ogle_{year}_{event:04d}.dat"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    a = np.loadtxt(raw.decode("ascii", "ignore").splitlines())
    return a[:, 0], a[:, 1], a[:, 2]

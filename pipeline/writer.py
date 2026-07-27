"""
BinML v5 — HDF5 shard writer.

Storage model
-------------
Every band samples a FIXED grid (``numpy.arange``-derived), identical for every event, so the
time axis is stored ONCE as a small per-band dataset rather than per event. Each event then
contributes one dense row per band, with **NaN marking epochs that failed detectability**.
That keeps rows rectangular (fast random reads for training) while still recording exactly
which epochs the survey actually delivered.

``mag`` is float32 and ``mag_err`` float16. Magnitudes need ~5 significant digits to preserve
a 1 mmag floor at m ~ 25, which float16 cannot hold; uncertainties only need ~0.1% relative
precision, which it can. Storing the magnitude as a float16 offset from baseline was
considered and rejected: precision degrades to ~4 mmag near a 5-magnitude peak, which is
worse than the photometry at the bright end.

Chunking is ``(256, n_epochs)``. This is not incidental -- an earlier dataset in this project
used ``(10000, 6912)`` with lzf, giving 276 MB chunks, so a single random row read pulled and
decompressed the whole chunk and training ran at 0.028 it/s. Keep the chunk row count small.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np

from .assemble import Event, SurveyConfig, _epochs
from .classes import CLASS_NAMES
from .photometry import ROMAN_BANDS

__all__ = ["FORMAT_VERSION", "PARAM_FIELDS", "ShardWriter"]

FORMAT_VERSION = "5.0.0"
CHUNK_ROWS = 256

# Union of parameters across all generators. Any field a given class does not use is NaN,
# so the table stays rectangular and self-describing.
PARAM_FIELDS: List[str] = [
    "t0", "tE", "u0", "q", "s", "alpha", "rho",      # microlensing
    "P", "amp_I", "ratio_k",                          # periodic / general variability
    "t_start", "rise", "decay", "plateau", "recur",   # eruptive
    "t_anom",                                         # day the binary anomaly first becomes observable
]

# Per-event scalars recorded alongside the light curves.
_SCALARS = {
    "label": np.int8, "true_class": np.int8,
    "dchi2_event": np.float64, "dchi2_anomaly": np.float64,
    "m_base_ref": np.float32, "a_ks": np.float32, "n_usable_bands": np.int8,
    # Probability with which this event was retained. Byproduct binaries that look single
    # are subsampled, so training must reweight by 1/keep_prob to recover the true rate.
    "keep_prob": np.float32,
}


def _param_vector(params: dict) -> np.ndarray:
    """Flatten a generator's parameter dict onto the fixed PARAM_FIELDS layout."""
    out = np.full(len(PARAM_FIELDS), np.nan, dtype=np.float32)
    for i, k in enumerate(PARAM_FIELDS):
        v = params.get(k)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):      # e.g. LongPeriodVar "periods"
            continue
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            pass
    # LongPeriodVar carries a list of (period, weight); record the dominant period as P.
    per = params.get("periods")
    if per:
        try:
            out[PARAM_FIELDS.index("P")] = float(max(p for p, _ in per))
        except (TypeError, ValueError):
            pass
    return out


class ShardWriter:
    """Streams events into one HDF5 shard, growing datasets as it goes."""

    def __init__(self, path: str, cfg: SurveyConfig, compression: Optional[str] = "lzf"):
        self.path = path
        self.cfg = cfg
        self.n = 0
        self._h5 = h5py.File(path, "w")
        self._compression = compression

        g = self._h5
        g.attrs["format_version"] = FORMAT_VERSION
        g.attrs["window_days"] = cfg.window_days
        g.attrs["reference_band"] = cfg.reference_band
        g.attrs["class_names"] = [n.encode() for n in CLASS_NAMES]
        g.attrs["param_fields"] = [n.encode() for n in PARAM_FIELDS]
        g.attrs["dchi2_event_threshold"] = cfg.dchi2_event
        g.attrs["dchi2_anomaly_threshold"] = cfg.dchi2_anomaly
        g.attrs["min_amplitude_mag"] = cfg.min_amplitude_mag
        g.attrs["snr_threshold"] = cfg.snr_threshold
        g.attrs["nan_means"] = "epoch failed detectability (undetected or saturated)"

        self._bands = list(ROMAN_BANDS)
        self._nep: Dict[str, int] = {}
        for b in self._bands:
            t = _epochs(b, cfg.window_days)
            self._nep[b] = t.size
            g.create_dataset(f"time/{b}", data=t.astype(np.float32))
            g.create_dataset(f"mag/{b}", shape=(0, t.size), maxshape=(None, t.size),
                             dtype=np.float32, chunks=(CHUNK_ROWS, t.size),
                             compression=compression)
            g.create_dataset(f"mag_err/{b}", shape=(0, t.size), maxshape=(None, t.size),
                             dtype=np.float16, chunks=(CHUNK_ROWS, t.size),
                             compression=compression)
            g.create_dataset(f"f_s/{b}", shape=(0,), maxshape=(None,), dtype=np.float32)
            g.create_dataset(f"n_kept/{b}", shape=(0,), maxshape=(None,), dtype=np.int32)

        for name, dt in _SCALARS.items():
            g.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dt)
        g.create_dataset("params", shape=(0, len(PARAM_FIELDS)),
                         maxshape=(None, len(PARAM_FIELDS)), dtype=np.float32)

    def append(self, events: Sequence[Event]) -> None:
        if not events:
            return
        k = len(events)
        lo, hi = self.n, self.n + k
        g = self._h5

        for b in self._bands:
            nep = self._nep[b]
            mag = np.full((k, nep), np.nan, dtype=np.float32)
            err = np.full((k, nep), np.nan, dtype=np.float32)
            fs = np.zeros(k, dtype=np.float32)
            nk = np.zeros(k, dtype=np.int32)
            step = self.cfg.window_days / nep
            for i, ev in enumerate(events):
                ob = ev.bands[b]
                fs[i] = ob.f_s
                nk[i] = ob.t.size
                if ob.t.size:
                    # map kept epochs back onto the fixed grid by index
                    idx = np.rint(ob.t / step).astype(np.int64)
                    idx = np.clip(idx, 0, nep - 1)
                    mag[i, idx] = ob.mag
                    err[i, idx] = ob.mag_err
            for name, arr in ((f"mag/{b}", mag), (f"mag_err/{b}", err)):
                d = g[name]; d.resize(hi, axis=0); d[lo:hi] = arr
            for name, arr in ((f"f_s/{b}", fs), (f"n_kept/{b}", nk)):
                d = g[name]; d.resize(hi, axis=0); d[lo:hi] = arr

        vals = {
            "label": np.array([e.label_index for e in events], dtype=np.int8),
            "true_class": np.array([CLASS_NAMES.index(e.true_class) for e in events],
                                   dtype=np.int8),
            "dchi2_event": np.array([e.dchi2_event for e in events], dtype=np.float64),
            "dchi2_anomaly": np.array([e.dchi2_anomaly for e in events], dtype=np.float64),
            "m_base_ref": np.array([e.params.get("_m_base_ref", np.nan) for e in events],
                                   dtype=np.float32),
            "a_ks": np.array([e.params.get("_a_ks", np.nan) for e in events], dtype=np.float32),
            "n_usable_bands": np.array([e.n_usable_bands for e in events], dtype=np.int8),
            "keep_prob": np.array([e.params.get("_keep_prob", 1.0) for e in events],
                                  dtype=np.float32),
        }
        for name, arr in vals.items():
            d = g[name]; d.resize(hi, axis=0); d[lo:hi] = arr
        p = np.stack([_param_vector(e.params) for e in events])
        d = g["params"]; d.resize(hi, axis=0); d[lo:hi] = p

        self.n = hi

    def set_run_attrs(self, shard: int, byproduct_keep_prob: float,
                      gen_counts, dropped) -> None:
        """Record how this shard was produced, including what was NOT kept.

        The dropped counts matter: byproduct subsampling means the stored population is not
        the generated one, and without these numbers the true event rates are unrecoverable.
        """
        a = self._h5.attrs
        a["shard"] = shard
        a["byproduct_keep_prob"] = byproduct_keep_prob
        a["n_generated"] = int(sum(gen_counts.values()))
        a["n_dropped_byproduct"] = int(dropped.get("byproduct", 0))
        a["n_dropped_unusable"] = int(dropped.get("unusable", 0))
        for c in CLASS_NAMES:
            a[f"n_generated_{c}"] = int(gen_counts.get(c, 0))

    def close(self) -> None:
        self._h5.attrs["n_events"] = self.n
        self._h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

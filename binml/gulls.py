"""Score RGES-PIT RMDC26 (GULLS) light curves with BinML.

RMDC26 is an independent simulation of Roman's Galactic Bulge Time Domain Survey, released on the
Hugging Face Hub as ``RGES-PIT/MachineLearning``: a 172 GB observations table, a 4 MB epoch table
(epoch_id -> BJD) and a 149 MB metadata table. This module holds the small amount of logic needed
to hand one of its events to :class:`binml.Classifier` under BinML's input contract, factored out
of ``validation/gulls_transfer.py`` so that a notebook can do it in a few lines:

* the dataset **revision is pinned** (RGES-PIT re-uploaded the tables on 2026-08-18; mixing
  metadata from one release with observations from another silently pairs events wrongly);
* one event is fetched **remotely by id** with DuckDB (the table is clustered by event_id; a query
  on the id alone costs a few seconds, a time-window query reads the whole file);
* the input is **one contiguous season** found from the epoch table, not a window centred on the
  peak (that would straddle the 110-day inter-season gaps), and only the six high-cadence seasons
  are inside the model's support;
* the baseline is **measured from the data** (median F146 magnitude more than 5 t_E from the
  peak over the full mission), because the catalogue baseline in this release is offset from the
  observed quiescent flux by ~0.47 mag.

Network access is needed for :func:`fetch_event` and :func:`load_tables`. Everything else is pure
numpy and is unit-tested offline. See ``examples/01_classify_roman_event.ipynb``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

REVISION = "a338d5bab441b5caf551d2fea9469aadfdc81ec1"
BASE = f"https://huggingface.co/datasets/RGES-PIT/MachineLearning/resolve/{REVISION}/"
OBS_URL = BASE + "RMDC26_ML_Data_obs.parquet"
EPOCH_URL = BASE + "RMDC26_ML_Data_epoch.parquet"
META_URL = BASE + "RMDC26_ML_Data_meta.parquet"
WINDOW_DAYS = 72.0
SEASON_GAP_DAYS = 5.0         # inter-season gaps are ~110 d; intra-season gaps are < 1 d
DENSE_MIN_F146 = 1000         # F146 epochs per season below which the season is low-cadence
CLASS_MEANING = {"RMDC26_1S1L_ML": "single lens (-> PSPL)",
                 "RMDC26_1S2L_ML": "planetary lens (-> NonPSPL)",
                 "RMDC26_2S2L_ML": "planetary lens with a binary source (-> NonPSPL)"}

__all__ = ["REVISION", "flux_to_ab", "Season", "seasons_from_epochs", "season_of", "empirical_baseline",
           "to_bands", "load_tables", "fetch_event", "classify_event"]


def flux_to_ab(flux_ujy) -> np.ndarray:
    """microjansky -> AB magnitude; non-positive flux (noise on faint sources) becomes NaN."""
    f = np.asarray(flux_ujy, float)
    out = np.full(f.shape, np.nan)
    ok = f > 0
    out[ok] = -2.5 * np.log10(f[ok]) + 23.9
    return out


@dataclass(frozen=True)
class Season:
    index: int
    start: float          # BJD
    end: float            # BJD
    n_epochs: int         # all bands
    dense: bool           # high-cadence (inside BinML's support)

    @property
    def window(self) -> Tuple[float, float]:
        """The 72-day input window: from the season start, clipped to the season end."""
        return self.start, min(self.end, self.start + WINDOW_DAYS)


def seasons_from_epochs(bjd: np.ndarray, dense_min_epochs: int = 3 * DENSE_MIN_F146) -> List[Season]:
    """Split the mission's epoch times into seasons at gaps longer than SEASON_GAP_DAYS."""
    b = np.sort(np.asarray(bjd, float))
    if b.size == 0:
        return []
    cut = np.flatnonzero(np.diff(b) > SEASON_GAP_DAYS)
    starts = np.r_[b[0], b[cut + 1]]; ends = np.r_[b[cut], b[-1]]
    out = []
    for i, (a, e) in enumerate(zip(starts, ends)):
        n = int(((b >= a) & (b <= e)).sum())
        out.append(Season(i, float(a), float(e), n, n >= dense_min_epochs))
    return out


def season_of(t0: float, seasons: Iterable[Season]) -> Optional[Season]:
    """The season containing t0, or None when the peak falls in an inter-season gap."""
    for s in seasons:
        if s.start <= t0 <= s.end:
            return s
    return None


def empirical_baseline(bjd: np.ndarray, mag: np.ndarray, t0: float, tE: float,
                       n_tE: float = 5.0, min_points: int = 200) -> float:
    """Median magnitude more than ``n_tE`` Einstein times from the peak, over everything given.

    Raises ValueError with fewer than ``min_points`` usable off-event points: a baseline from a
    handful of epochs is worse than none, and BinML is sensitive to the baseline it is handed.
    """
    bjd = np.asarray(bjd, float); mag = np.asarray(mag, float)
    off = (np.abs(bjd - t0) > n_tE * tE) & np.isfinite(mag)
    if off.sum() < min_points:
        raise ValueError(f"only {int(off.sum())} off-event points (need {min_points}) for a baseline")
    return float(np.median(mag[off]))


def to_bands(bjd: np.ndarray, filt: np.ndarray, mag: np.ndarray, window: Tuple[float, float],
             min_points: int = 10) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Per-band (days since window start, magnitude) inside the window, sorted, finite only."""
    bjd = np.asarray(bjd, float); filt = np.asarray(filt); mag = np.asarray(mag, float)
    lo, hi = window
    out = {}
    for band in ("F146", "F087", "F213"):
        m = (filt == band) & (bjd >= lo) & (bjd <= hi) & np.isfinite(mag)
        if m.sum() < min_points:
            continue
        t = bjd[m] - lo; o = np.argsort(t)
        out[band] = (t[o], mag[m][o])
    return out


# ----------------------------------------------------------------------------- network
def load_tables(cache_dir: str = "~/.cache/binml/rmdc26"):
    """Download (once) and load the epoch and metadata tables. Returns (seasons, meta_dict)."""
    import urllib.request
    import pyarrow.parquet as pq
    d = os.path.expanduser(cache_dir); os.makedirs(d, exist_ok=True)
    paths = {}
    for name, url in (("epoch", EPOCH_URL), ("meta", META_URL)):
        p = os.path.join(d, f"rmdc26_{name}_{REVISION[:8]}.parquet")
        if not os.path.exists(p):
            urllib.request.urlretrieve(url, p)
        paths[name] = p
    ep = pq.read_table(paths["epoch"], columns=["epoch_id", "bjd"]).to_pydict()
    epoch_map = dict(zip(ep["epoch_id"], ep["bjd"]))
    meta = pq.read_table(paths["meta"]).to_pandas().set_index("event_id")
    return seasons_from_epochs(np.asarray(ep["bjd"], float)), epoch_map, meta


def fetch_event(event_id: int, epoch_map: Dict[int, float], columns: Tuple[str, ...] = ("flux_uJy",),
                con=None) -> Dict[str, np.ndarray]:
    """All unsaturated observations of one event, with BJD attached. A few seconds per event."""
    import duckdb
    con = con or duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    cols = ", ".join(("epoch_id", "filt") + tuple(columns))
    q = con.execute(f"SELECT {cols} FROM read_parquet('{OBS_URL}') WHERE event_id = {int(event_id)} "
                    f"AND saturation_flag = 0").fetchnumpy()
    out = {k: np.asarray(v) for k, v in q.items()}
    out["bjd"] = np.array([epoch_map.get(int(e), np.nan) for e in out["epoch_id"]], float)
    return out


def classify_event(clf, event_id: int, seasons: List[Season], epoch_map, meta, con=None,
                   all_seasons: bool = False) -> Dict:
    """Fetch one RMDC26 event and classify its season(s) under BinML's input contract.

    Returns the metadata used, the baseline, and per-season predictions. By default only the
    season containing the peak is scored; ``all_seasons=True`` scores every dense season (the
    per-season combiner of paper/REVISION.md §1½ row 9: take the max anomaly probability).
    """
    row = meta.loc[int(event_id)]
    t0, tE = float(row["t0lens1"]), float(row["tE_ref"])
    obs = fetch_event(event_id, epoch_map, con=con)
    mag = flux_to_ab(obs["flux_uJy"])
    f146 = obs["filt"] == "F146"
    m_base = empirical_baseline(obs["bjd"][f146], mag[f146], t0, tE)
    peak = season_of(t0, seasons)
    targets = [s for s in seasons if s.dense] if all_seasons else ([peak] if peak is not None else [])
    out = {"event_id": int(event_id), "sim_label": str(row["sim_label"]), "meaning": CLASS_MEANING.get(str(row["sim_label"])),
           "t0_bjd": t0, "tE_days": tE, "m_base_empirical": round(m_base, 3),
           "m_base_catalogue": round(float(row["Source_F146"] + 2.5 * np.log10(max(float(row["fs_F146"]), 1e-6))), 3),
           "peak_season": None if peak is None else peak.index, "peak_in_dense_season": bool(peak is not None and peak.dense),
           "seasons": []}
    for s in targets:
        bands = to_bands(obs["bjd"], obs["filt"], mag, s.window)
        if "F146" not in bands:
            out["seasons"].append({"season": s.index, "skipped": "no usable F146"}); continue
        p = clf.predict(bands, m_base_ref=m_base, t_start=0.0)
        out["seasons"].append({"season": s.index, "dense": s.dense, "n_f146": int(bands["F146"][0].size),
                               "bands": sorted(bands), "label": p.label,
                               "probabilities": {k: round(v, 4) for k, v in p.probabilities.items()}, "_bands": bands})
    return out

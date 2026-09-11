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
  peak (that would straddle the ~115-day inter-season gaps), and only the six high-cadence seasons
  are inside the model's support (their F146 pause pattern differs from season to season:
  validation/gulls/rmdc26_schedule.json);
* the baseline is **measured from the data** (median F146 magnitude more than 5 t_E from the
  peak over the full mission, using the catalogue t0 and t_E to exclude the event), because the
  catalogue baseline in this release is offset from the observed quiescent flux by ~0.47 mag. A
  real-time pipeline would not have the catalogue t0/t_E or post-event data.

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
           "to_bands", "load_tables", "fetch_event", "classify_observations", "classify_event"]


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
    """Download (once) and load the epoch and metadata tables.

    Returns ``(seasons, epoch_map, meta)``: the list of :class:`Season`, a dict epoch_id -> BJD, and the
    metadata as a pandas DataFrame indexed by event_id.
    """
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


def _adjacent(t0: float, seasons: List[Season]) -> List[Season]:
    """The seasons bordering t0: the season containing it, or the last one before and first one after."""
    inside = season_of(t0, seasons)
    if inside is not None:
        return [inside]
    before = [x for x in seasons if x.end < t0]
    after = [x for x in seasons if x.start > t0]
    return ([before[-1]] if before else []) + ([after[0]] if after else [])


def classify_observations(clf, obs: Dict[str, np.ndarray], row, seasons: List[Season], mode: str = "peak") -> Dict:
    """Classify one RMDC26 event's observations under BinML's input contract (no network).

    ``obs``: arrays ``bjd``, ``filt``, ``flux_uJy`` (as returned by :func:`fetch_event`); ``row``: the
    event's metadata (t0lens1, tE_ref, Source_F146, fs_F146, sim_label). ``mode``:

    * ``"peak"``: the season containing the peak, if it is a high-cadence season (the scored set of
      the transfer tables). A low-cadence peak season is outside BinML's support and is not scored.
    * ``"adjacent"``: the high-cadence season(s) bordering the peak. For a peak between seasons these
      are the seasons before and after the gap -- the per-season combiner measured in
      paper/REVISION.md section 1.5 row 9; for a peak inside a season, that season.
    * ``"all"``: every high-cadence season of the mission (not validated as a combiner).

    ``p_nonpspl_max`` is the maximum anomaly probability over the scored seasons.
    """
    if mode not in ("peak", "adjacent", "all"):
        raise ValueError(f"mode must be 'peak', 'adjacent' or 'all', not {mode!r}")
    t0, tE = float(row["t0lens1"]), float(row["tE_ref"])
    mag = flux_to_ab(obs["flux_uJy"])
    f146 = np.asarray(obs["filt"]) == "F146"
    m_base = empirical_baseline(obs["bjd"][f146], mag[f146], t0, tE)
    peak = season_of(t0, seasons)
    if mode == "peak":
        targets = [peak] if peak is not None else []
    elif mode == "adjacent":
        targets = _adjacent(t0, seasons)
    else:
        targets = [x for x in seasons if x.dense]
    out = {"sim_label": str(row["sim_label"]), "meaning": CLASS_MEANING.get(str(row["sim_label"])),
           "t0_bjd": t0, "tE_days": tE, "m_base_empirical": round(m_base, 3),
           "m_base_catalogue": round(float(row["Source_F146"] + 2.5 * np.log10(max(float(row["fs_F146"]), 1e-6))), 3),
           "peak_season": None if peak is None else peak.index, "peak_in_dense_season": bool(peak is not None and peak.dense),
           "mode": mode, "seasons": [], "p_nonpspl_max": None}
    if not targets:
        out["note"] = "peak falls outside every season; use mode='adjacent' to score the seasons around it"
    for x in targets:
        if not x.dense:
            out["seasons"].append({"season": x.index, "skipped": "low-cadence season: outside BinML's support"}); continue
        bands = to_bands(obs["bjd"], obs["filt"], mag, x.window)
        if "F146" not in bands:
            out["seasons"].append({"season": x.index, "skipped": "no usable F146"}); continue
        p = clf.predict(bands, m_base_ref=m_base, t_start=0.0)
        n146 = int(bands["F146"][0].size)
        out["seasons"].append({"season": x.index, "dense": x.dense, "n_f146": n146, "dense_event": n146 >= DENSE_MIN_F146,
                               "bands": sorted(bands), "label": p.label,
                               "probabilities": {k: round(v, 4) for k, v in p.probabilities.items()}, "_bands": bands})
    scored = [x["probabilities"]["NonPSPL"] for x in out["seasons"] if "probabilities" in x]
    out["p_nonpspl_max"] = max(scored) if scored else None
    return out


def classify_event(clf, event_id: int, seasons: List[Season], epoch_map, meta, con=None,
                   mode: str = "peak", all_seasons: Optional[bool] = None) -> Dict:
    """Fetch one RMDC26 event (network, a few seconds) and classify it; see :func:`classify_observations`.

    ``all_seasons=True`` is the pre-2026-09-11 spelling of ``mode="all"``.
    """
    if all_seasons:
        mode = "all"
    obs = fetch_event(event_id, epoch_map, con=con)
    out = classify_observations(clf, obs, meta.loc[int(event_id)], seasons, mode=mode)
    out["event_id"] = int(event_id)
    return out

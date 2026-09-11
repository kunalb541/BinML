"""Offline tests for binml.gulls: season splitting, window, baseline, band assembly."""
import numpy as np
import pytest

from binml.gulls import (Season, empirical_baseline, flux_to_ab, season_of, seasons_from_epochs, to_bands,
                         WINDOW_DAYS)


def _mission():
    # three 70-day seasons with 110-day gaps; the middle one sparse
    t = np.concatenate([np.arange(0, 70, 0.01), 180 + np.arange(0, 70, 0.5), 360 + np.arange(0, 70, 0.01)])
    return t


def test_seasons_split_at_long_gaps_and_flag_density():
    s = seasons_from_epochs(_mission())
    assert [x.index for x in s] == [0, 1, 2]
    assert [x.dense for x in s] == [True, False, True]
    assert s[0].window == (0.0, pytest.approx(min(s[0].end, WINDOW_DAYS)))
    assert season_of(30.0, s).index == 0 and season_of(100.0, s) is None and season_of(200.0, s).index == 1


def test_baseline_uses_off_event_points_only():
    t = np.arange(0, 400, 0.05)
    mag = np.full_like(t, 21.0); mag[np.abs(t - 200) < 20] -= 1.0        # a 1 mag bump around t0 = 200
    assert empirical_baseline(t, mag, t0=200.0, tE=3.0) == 21.0
    with pytest.raises(ValueError):
        empirical_baseline(t[:50], mag[:50], t0=200.0, tE=3.0)


def test_to_bands_windows_and_sorts():
    t = np.array([5.0, 1.0, 3.0, 80.0, 2.0] + [10.0 + i for i in range(12)])
    filt = np.array(["F146"] * 5 + ["F087"] * 12)
    mag = np.arange(t.size, dtype=float)
    b = to_bands(t, filt, mag, (0.0, 72.0), min_points=3)
    assert set(b) == {"F146", "F087"}
    assert np.all(np.diff(b["F146"][0]) > 0) and b["F146"][0].max() < 72
    assert b["F146"][0].size == 4                                         # t = 80 is outside


def test_flux_to_ab():
    assert flux_to_ab([3631e6])[0] == pytest.approx(0.0, abs=1e-3)   # 23.9 zeropoint vs 3631 Jy
    assert np.isnan(flux_to_ab([0.0, -1.0])).all()


class _FakeClf:
    def predict(self, bands, m_base_ref=None, t_start=None):
        import types
        pn = 0.9 if bands["F146"][0].size > 500 else 0.1
        return types.SimpleNamespace(label="NonPSPL" if pn > 0.5 else "PSPL",
                                     probabilities={"Flat": 0.0, "PSPL": 1 - pn, "NonPSPL": pn,
                                                    "PeriodicVar": 0.0, "LongPeriodVar": 0.0, "Eruptive": 0.0})


def _obs_and_row(t0):
    from binml.gulls import seasons_from_epochs
    t = _mission()
    S = seasons_from_epochs(t)
    obs = {"bjd": t, "filt": np.array(["F146"] * t.size), "flux_uJy": np.full(t.size, 10 ** (-0.4 * (21.0 - 23.9)))}
    row = {"t0lens1": t0, "tE_ref": 3.0, "Source_F146": 21.0, "fs_F146": 1.0, "sim_label": "RMDC26_1S2L_ML"}
    return obs, row, S


def test_classify_observations_modes():
    from binml.gulls import classify_observations
    obs, row, S = _obs_and_row(30.0)                           # peak inside dense season 0
    r = classify_observations(_FakeClf(), obs, row, S, mode="peak")
    assert [x["season"] for x in r["seasons"]] == [0] and r["p_nonpspl_max"] == 0.9
    obs, row, S = _obs_and_row(120.0)                          # peak between season 0 and (sparse) season 1
    r = classify_observations(_FakeClf(), obs, row, S, mode="peak")
    assert r["seasons"] == [] and r["p_nonpspl_max"] is None and "adjacent" in r["note"]
    r = classify_observations(_FakeClf(), obs, row, S, mode="adjacent")
    assert [x["season"] for x in r["seasons"]] == [0, 1]
    assert "low-cadence" in r["seasons"][1]["skipped"] and r["p_nonpspl_max"] == 0.9
    obs, row, S = _obs_and_row(200.0)                          # peak inside the low-cadence season
    r = classify_observations(_FakeClf(), obs, row, S, mode="peak")
    assert "low-cadence" in r["seasons"][0]["skipped"] and r["p_nonpspl_max"] is None
    r = classify_observations(_FakeClf(), obs, row, S, mode="all")
    assert [x["season"] for x in r["seasons"] if "probabilities" in x] == [0, 2]
    with pytest.raises(ValueError):
        classify_observations(_FakeClf(), obs, row, S, mode="everything")

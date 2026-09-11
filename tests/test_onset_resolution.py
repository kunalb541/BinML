"""Anomaly-onset resolution: the default is the legacy 7.2-d grid (released data, bit-for-bit); the
opt-in fine grid returns the first detectable cut on the absolute grid, as the cascade evaluation does.

Audit 2026-09-09 finding 9 (still open for the relabel rule itself): the released generator rounds
t_anom UP to a multiple of 7.2 d. The first attempt at a fine grid (2026-09-11) searched only inside the
first detectable coarse interval and skipped the two grid points next to each coarse cut; these tests
pin the full-grid replacement at exactly those points.
"""
import dataclasses

import numpy as np
import pytest

from pipeline.assemble import SurveyConfig, _anomaly_onset_day, _epochs
from pipeline.generators import pspl_magnification


def _ref_truth_with_step(t_on, amp=0.15):
    """A PSPL (t0 = 40, tE = 15, u0 = 0.3) plus a 0.15 mag step deviation starting at day t_on."""
    t = _epochs("F146", 72.0)
    params = {"t0": 40.0, "tE": 15.0, "u0": 0.3}
    A = pspl_magnification(t, **params)
    mb, fs = 21.0, 0.7
    mag = mb - 2.5 * np.log10(1 + fs * (A - 1))
    mag = mag - amp * (t >= t_on)
    sig = np.full_like(t, 0.005)
    return (t, mag, sig, mb, fs), params


def test_default_is_the_legacy_grid():
    assert SurveyConfig().onset_resolution_days == 7.2
    rt, p = _ref_truth_with_step(30.3)
    assert _anomaly_onset_day(rt, p, SurveyConfig()) == 36.0          # rounded UP to the 7.2-d grid


@pytest.mark.parametrize("t_on,expected", [(6.8, 7.0), (7.3, 7.5), (13.8, 14.0), (21.3, 21.5),
                                           (30.3, 30.5), (43.3, 43.5), (64.9, 65.0), (9.1, 9.5)])
def test_fine_grid_returns_the_first_half_day_cut_after_the_onset(t_on, expected):
    cfg = dataclasses.replace(SurveyConfig(), onset_resolution_days=0.5)
    rt, p = _ref_truth_with_step(t_on)
    onset = _anomaly_onset_day(rt, p, cfg)
    assert onset == pytest.approx(expected), (t_on, onset)


def test_no_anomaly_is_still_inf():
    rt, p = _ref_truth_with_step(30.3, amp=0.0)
    assert _anomaly_onset_day(rt, p, SurveyConfig()) == float("inf")
    assert _anomaly_onset_day(rt, p, SurveyConfig(), resolution_days=0.5) == float("inf")


def test_fine_grid_tests_the_window_end_when_res_does_not_divide_it():
    """res = 0.7 d: 72/0.7 is not an integer, so the last grid cut is 71.4 d. An anomaly detectable only
    after 71.4 d must still get a finite onset at the window end (72 d), as the legacy grid gives."""
    import numpy as np
    from pipeline import assemble
    cfg = assemble.SurveyConfig()
    calls = []

    def fake(t, mag, sig, mb, fs, params):
        tc = t.max(); calls.append(tc)
        return (1e6, 1.0) if tc >= 71.8 else (0.0, 0.0)

    t = np.linspace(0.0, 72.0, 2000); ref = (t, np.zeros_like(t), np.full_like(t, 0.01), 20.0, 1.0)
    orig = assemble._pspl_refit_dchi2
    assemble._pspl_refit_dchi2 = fake
    try:
        assert assemble._anomaly_onset_day(ref, {}, cfg, resolution_days=0.7) == 72.0
        assert assemble._anomaly_onset_day(ref, {}, cfg, resolution_days=0.5) == 72.0
    finally:
        assemble._pspl_refit_dchi2 = orig

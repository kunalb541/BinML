"""The recorded anomaly onset is resolved to 0.5 d by default; 7.2 d reproduces the legacy grid.

Audit 2026-09-09 finding 9: the released generator rounded t_anom UP to a multiple of 7.2 d, which the
gap/cadence augmentations then used as if exact. Under a fixed gap schedule that mislabelled 20% of all
binaries (validation/schedule_finetune_local.py). These tests pin the fix.
"""
import dataclasses

import numpy as np

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


def test_default_resolution_is_half_day_and_brackets_the_true_onset():
    cfg = SurveyConfig()
    assert cfg.onset_resolution_days == 0.5
    for t_on in (30.3, 9.1, 61.7):
        rt, p = _ref_truth_with_step(t_on)
        onset = _anomaly_onset_day(rt, p, cfg)
        assert np.isfinite(onset)
        assert t_on <= onset <= t_on + 0.5 + 1e-9, (t_on, onset)      # first fine cut at/after the step
        assert abs(onset / 0.5 - round(onset / 0.5)) < 1e-9             # on the 0.5 d grid


def test_legacy_grid_is_reproduced_exactly():
    cfg = dataclasses.replace(SurveyConfig(), onset_resolution_days=7.2)
    rt, p = _ref_truth_with_step(30.3)
    onset = _anomaly_onset_day(rt, p, cfg)
    assert onset == 36.0                                                   # rounded UP to the 7.2 d grid
    assert _anomaly_onset_day(rt, p, SurveyConfig(), resolution_days=7.2) == 36.0


def test_no_anomaly_is_still_inf():
    rt, p = _ref_truth_with_step(30.3, amp=0.0)
    assert _anomaly_onset_day(rt, p, SurveyConfig()) == float("inf")

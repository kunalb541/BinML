"""Finite-source single lens (opt-in via priors.PSPL_FINITE_SOURCE): physics and plumbing.

Background: on GULLS/RMDC26 the gap-aware model's single-lens false-alarm rate rises monotonically
with rho/|u0| because the released PSPL generator is point-source while NonPSPL samples rho
(paper/REVISION.md, 2026-09-09). These tests pin the finite-source path added to close that gap.
"""
import dataclasses

import numpy as np
import pytest

pytest.importorskip("VBBinaryLensing")
from pipeline.generators import PSPLGen, espl_magnification, pspl_magnification  # noqa: E402
from pipeline.priors import DEFAULT_PRIORS  # noqa: E402


def test_espl_reduces_to_point_source_for_tiny_rho():
    t = np.linspace(0, 72, 3000)
    a = espl_magnification(t, 36.0, 20.0, 0.05, 1e-4)
    b = pspl_magnification(t, 36.0, 20.0, 0.05)
    assert np.allclose(a, b, rtol=2e-3), np.abs(a / b - 1).max()


def test_finite_source_suppresses_the_peak_and_is_achromatic():
    t = np.linspace(0, 72, 3000)
    point = pspl_magnification(t, 36.0, 20.0, 0.01)
    fs = espl_magnification(t, 36.0, 20.0, 0.01, 0.1)       # rho = 10 x u0: the disc covers the peak
    assert fs.max() < 0.3 * point.max()                     # peak strongly suppressed
    assert np.all(np.isfinite(fs)) and fs.min() >= 1.0 - 1e-9
    # far from the peak (u >> rho) the two agree. At u = 10 rho the uniform-disc correction is still
    # ~1e-3 of the magnification (physical; ESPLMag matches direct disc integration there to 3e-7,
    # whereas the legacy ESPLMag2 had already switched to the point-source value)
    far = np.abs(t - 36.0) > 20.0
    assert np.allclose(fs[far], point[far], rtol=2e-3)


def test_pspl_generator_is_point_source_by_default_and_finite_source_on_request():
    rng = np.random.default_rng(0)
    g = PSPLGen()
    p0 = g.sample(np.random.default_rng(0), 72.0)
    assert "rho" not in p0                                    # released behaviour unchanged
    pri = dataclasses.replace(DEFAULT_PRIORS, PSPL_FINITE_SOURCE=True)
    p1 = g.sample(np.random.default_rng(0), 72.0, priors=pri)
    assert "rho" in p1 and pri.PSPL_RHO_MIN <= p1["rho"] <= pri.PSPL_RHO_MAX
    t = np.linspace(0, 72, 500)
    d0 = g.delta(t, dict(p0), "F146")
    d1 = g.delta(t, dict(p1), "F146")
    assert np.all(np.isfinite(d0)) and np.all(np.isfinite(d1))
    assert d0.min() >= -1e-9 and d1.min() >= -1e-9             # magnification - 1 is non-negative


def test_regimes_are_wired():
    from pipeline.run_shard import HARD_REGIMES, CONFIG_REGIMES, MIXES
    assert HARD_REGIMES["fspl"]["PSPL_FINITE_SOURCE"] is True
    assert HARD_REGIMES["fspl_highmag"]["U0_MAX"] == 0.2
    assert MIXES[CONFIG_REGIMES["fspl_highmag"]["mix"]]["NonPSPL"] >= 3000   # not a PSPL-only shortcut
    assert HARD_REGIMES["fspl5"]["PSPL_RHO_MAX"] == 5.0 and HARD_REGIMES["fspl5"]["RHO_MAX"] == 0.1
    assert CONFIG_REGIMES["fspl5_highmag"]["mix"] == "highmag"
    assert "RHO_MAX" not in HARD_REGIMES["fspl5s"] and HARD_REGIMES["fspl5s"]["PSPL_RHO_MAX"] == 5.0  # binaries unchanged


def test_espl_is_smooth_outside_the_disc_and_legacy_reproduces_the_old_steps():
    t = np.linspace(0, 72, 20000)
    for rho, tE in ((0.1, 20.0), (0.5, 20.0), (2.0, 2.0)):
        new = espl_magnification(t, 36.0, tE, 0.0, rho)
        old = espl_magnification(t, 36.0, tE, 0.0, rho, legacy=True)
        point = pspl_magnification(t, 36.0, tE, 0.0)
        u = np.abs(t - 36.0) / tE
        out = (u > 2.0 * rho) & (u < 15 * rho)
        # the finite-source correction A_fs/A_ps varies smoothly outside the disc; a hand-off to the
        # point-source formula shows up as a jump in it
        jump = lambda A: np.max(np.abs(np.diff((A / point)[out])))
        assert jump(new) < 1e-4, (rho, jump(new))
        if rho <= 0.5:                                       # for rho = 2 ESPLMag2 hands off at 1.8 rho, inside this region
            assert jump(old) > 1e-3, (rho, jump(old))        # its artificial step (4-8 mmag)


def test_legacy_flag_reaches_the_generator():
    pri = dataclasses.replace(DEFAULT_PRIORS, PSPL_FINITE_SOURCE=True, PSPL_ESPL_LEGACY=True)
    p = PSPLGen().sample(np.random.default_rng(1), 72.0, priors=pri)
    assert p.get("_espl_legacy") is True
    p2 = PSPLGen().sample(np.random.default_rng(1), 72.0, priors=dataclasses.replace(pri, PSPL_ESPL_LEGACY=False))
    assert "_espl_legacy" not in p2 and p2["rho"] == p["rho"]


def test_point_source_control_regime_matches_fspl5s_except_the_physics():
    from pipeline.run_shard import HARD_REGIMES, CONFIG_REGIMES
    assert HARD_REGIMES["pspl5s_ctrl"] == {} and HARD_REGIMES["pspl5s_ctrl_highmag"] == {"U0_MAX": 0.2}
    assert CONFIG_REGIMES["pspl5s_ctrl_highmag"]["mix"] == CONFIG_REGIMES["fspl5s_highmag"]["mix"]
    assert HARD_REGIMES["fspl5s_highmag"]["U0_MAX"] == HARD_REGIMES["pspl5s_ctrl_highmag"]["U0_MAX"]

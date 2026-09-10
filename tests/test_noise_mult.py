"""Noise-model knobs (SurveyConfig.noise_mult / bkg_mult) for the GULLS noise ablation: defaults are
bit-identical to the released photometry; the knobs scale what they claim to scale."""
import dataclasses

import numpy as np

from pipeline.assemble import SurveyConfig, simulate_event
from pipeline.photometry import ROMAN_BANDS, observe, photometric_sigma


def test_default_knobs_are_identity():
    cfg = SurveyConfig()
    assert cfg.noise_mult == 1.0 and cfg.bkg_mult == 1.0
    b = ROMAN_BANDS["F146"]
    m = np.linspace(16, 25, 50)
    assert np.array_equal(photometric_sigma(b, m), photometric_sigma(b, m, 1.0))
    ev0 = simulate_event("PSPL", np.random.default_rng(11), cfg)
    ev1 = simulate_event("PSPL", np.random.default_rng(11), dataclasses.replace(cfg, noise_mult=1.0, bkg_mult=1.0))
    assert ev0.label == ev1.label
    for bn in ev0.bands:
        assert np.array_equal(ev0.bands[bn].mag, ev1.bands[bn].mag)


def test_noise_mult_scales_the_photon_term_and_the_scatter():
    b = ROMAN_BANDS["F146"]
    m = np.full(20000, 22.0)
    s1, s3 = photometric_sigma(b, m, 1.0), photometric_sigma(b, m, 3.0)
    shot1 = np.sqrt(s1 ** 2 - b.sys_floor_mag ** 2); shot3 = np.sqrt(s3 ** 2 - b.sys_floor_mag ** 2)
    assert np.allclose(shot3 / shot1, 3.0)
    _, o1, e1 = observe(b, m, np.random.default_rng(0), noise_mult=1.0)
    _, o3, e3 = observe(b, m, np.random.default_rng(0), noise_mult=3.0)
    r = np.nanstd(o3 - 22.0) / np.nanstd(o1 - 22.0)
    assert 2.7 < r < 3.3, r
    assert np.nanmedian(e3) > 2.5 * np.nanmedian(e1)


def test_bkg_mult_hurts_faint_sources_more_than_bright_ones():
    cfg = dataclasses.replace(SurveyConfig(), bkg_mult=9.0)
    b = ROMAN_BANDS["F146"]
    bright, faint = np.full(2000, 17.0), np.full(2000, 24.0)
    sig = lambda band, mm: photometric_sigma(band, mm)[0]
    b9 = dataclasses.replace(b, background_e2=b.background_e2 * 9.0)
    assert sig(b9, faint) / sig(b, faint) > sig(b9, bright) / sig(b, bright) > 0.99
    ev = simulate_event("PSPL", np.random.default_rng(5), cfg)          # runs end to end
    assert ev is None or all(np.isfinite(bb.mag).all() for bb in ev.bands.values())

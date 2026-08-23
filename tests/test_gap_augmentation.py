"""`pipeline.train._apply_gaps`: contiguous Roman-like gaps, consistent across bands, relabelled.

Background: validation/gulls/gap_sensitivity.py.  The shipped model was trained on a continuous
F146 grid and collapses on Roman's real schedule (seven ~6 h pauses per season).  This
augmentation is the remedy; these tests pin its contract.
"""
import numpy as np
import pytest

pytest.importorskip("torch")
from pipeline.train import _apply_gaps, I_FLAT, I_PSPL, I_NON, MAG_SCALE  # noqa: E402
from binml.preprocess import BAND_BINS  # noqa: E402


def _event(amp=0.5, width_bins=40, centre=432):
    out = {}
    for b, n in BAND_BINS.items():
        x = np.zeros((n, 5), np.float32)
        x[:, 3] = 1.0; x[:, 4] = 1.0
        if b == "F146":
            k = np.arange(n)
            x[:, 0] = -amp / MAG_SCALE * np.exp(-0.5 * ((k - centre) / width_bins) ** 2)
            x[:, 1] = x[:, 0]; x[:, 2] = x[:, 0]
        out[b] = x
    return out


def test_gaps_are_contiguous_and_consistent_across_bands():
    out = _event()
    lab = _apply_gaps(out, I_PSPL, np.random.default_rng(0))
    m146 = out["F146"][:, 4] == 0
    assert 1 <= m146.sum() <= 8 * 12 * 12            # at most 8 gaps x 12 h on a 2-h grid
    runs = np.diff(np.r_[0, m146.astype(int), 0])
    assert 1 <= (runs == 1).sum() <= 8                # contiguous runs, not scattered bins
    for b, n in BAND_BINS.items():
        if b == "F146":
            continue
        expect = m146.reshape(n, 864 // n).any(axis=1)
        assert np.array_equal(out[b][:, 4] == 0, expect), b
        assert np.all(out[b][expect, :4] == 0)
    assert lab == I_PSPL                              # a wide PSPL survives any gap


def test_flat_stays_flat():
    for s in range(10):
        assert _apply_gaps(_event(amp=0.0), I_FLAT, np.random.default_rng(s)) == I_FLAT


def test_signal_entirely_inside_gap_becomes_flat():
    out = _event(amp=0.5, width_bins=0.5, centre=432)   # a 1-bin spike
    out["F146"][:, :3] = 0.0
    out["F146"][431:434, :3] = -0.5 / MAG_SCALE
    hit = None
    for s in range(5000):
        o = {b: x.copy() for b, x in out.items()}
        lab = _apply_gaps(o, I_PSPL, np.random.default_rng(s))
        if o["F146"][431:434, 4].sum() == 0:
            hit = lab; break
    assert hit is not None, "no seed blanked the spike"
    assert hit == I_FLAT


def test_binary_with_caustic_inside_gap_becomes_pspl():
    pf = {"t_anom": 0}; params = np.array([36.0])
    seen = 0
    for s in range(3000):
        o = _event()
        lab = _apply_gaps(o, I_NON, np.random.default_rng(s), params, pf)
        ab = int(36.0 / 72.0 * 864)
        if o["F146"][ab - 1:ab + 2, 4].sum() == 0:
            seen += 1
            assert lab == I_PSPL
        else:
            assert lab == I_NON
    assert seen > 0

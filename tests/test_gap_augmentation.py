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
    assert 1 <= m146.sum() <= 8 * 6                   # at most 8 gaps x 12 h = 6 bins each on a 2-h grid
    runs = np.diff(np.r_[0, m146.astype(int), 0])
    assert 1 <= (runs == 1).sum() <= 8                # contiguous runs, not scattered bins
    for b, n in BAND_BINS.items():
        if b == "F146":
            continue
        expect = m146.reshape(n, 864 // n).any(axis=1)
        assert np.array_equal(out[b][:, 4] == 0, expect), b
        assert np.all(out[b][expect, :4] == 0)
    assert lab == I_PSPL                              # a wide PSPL survives any gap


def test_flat_stays_flat_but_is_still_blanked():
    """The Flat early-return must keep the label AND still apply the gaps to every band --
    a Flat that skipped blanking would teach the model that Flat means 'no gaps'."""
    for s in range(10):
        out = _event(amp=0.0)
        assert _apply_gaps(out, I_FLAT, np.random.default_rng(s)) == I_FLAT
        m146 = out["F146"][:, 4] == 0
        assert m146.sum() >= 1, "Flat event was not blanked"
        for b, n in BAND_BINS.items():
            if b != "F146":
                assert np.array_equal(out[b][:, 4] == 0, m146.reshape(n, 864 // n).any(axis=1)), b


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


def test_rmdc26_schedule_mask_blanks_exactly_the_seven_pauses_and_the_season_end():
    from pipeline.train import rmdc26_schedule_mask, RMDC26_GAPS_D, RMDC26_GAP_H, RMDC26_SEASON_D
    m = rmdc26_schedule_mask(864)
    centres = (np.arange(864) + 0.5) * 72.0 / 864
    # every blanked bin is either inside a pause or past the season end, and vice versa
    inside = np.zeros(864, bool)
    for g in RMDC26_GAPS_D:
        inside |= (centres >= g) & (centres < g + RMDC26_GAP_H / 24.0)
    assert np.array_equal(m, inside | (centres > RMDC26_SEASON_D))
    assert 7 * 3 <= inside.sum() <= 7 * 4          # 6.2 h on a 2 h grid: 3 or 4 bin centres per pause
    assert (centres > RMDC26_SEASON_D).sum() == 16  # 1.3 d past season end


def test_apply_gaps_with_schedule_blanks_the_fixed_mask_in_every_band():
    from pipeline.train import rmdc26_schedule_mask
    rng = np.random.default_rng(3)
    out = {b: np.ones((L, 5), np.float32) for b, L in BAND_BINS.items()}
    for x in out.values():
        x[:, :3] = 0.5                                  # a 0.5 mag signal everywhere
    m = rmdc26_schedule_mask(BAND_BINS["F146"])
    lab = _apply_gaps(out, I_PSPL, rng, None, None, schedule=m)
    assert lab == I_PSPL                                # plenty of signal survives
    assert np.array_equal(out["F146"][:, 4] == 0, m)    # reference band: exactly the mask
    for b, L in BAND_BINS.items():
        if L != BAND_BINS["F146"]:
            exp = m.reshape(L, BAND_BINS["F146"] // L).any(axis=1)
            assert np.array_equal(out[b][:, 4] == 0, exp)
    # deterministic: the same mask again, regardless of rng state
    out2 = {b: np.ones((L, 5), np.float32) for b, L in BAND_BINS.items()}
    _apply_gaps(out2, I_PSPL, np.random.default_rng(99), None, None, schedule=m)
    assert np.array_equal(out2["F146"][:, 4], out["F146"][:, 4])


def test_relabel_anomaly_off_keeps_a_binary_whose_anomaly_is_blanked():
    from pipeline.train import rmdc26_schedule_mask
    m = rmdc26_schedule_mask(BAND_BINS["F146"])
    pf_idx = {"t_anom": 0}
    params = np.array([21.6])                             # a quantised t_anom inside the 21.49 d pause
    mk = lambda: {b: np.concatenate([np.full((L, 3), 0.5, np.float32), np.ones((L, 2), np.float32)], 1)
                  for b, L in BAND_BINS.items()}
    on = _apply_gaps(mk(), I_NON, np.random.default_rng(0), params, pf_idx, schedule=m)
    off = _apply_gaps(mk(), I_NON, np.random.default_rng(0), params, pf_idx, schedule=m, relabel_anomaly=False)
    assert on == I_PSPL and off == I_NON
    # and the 20% figure: two of the ten grid values fall inside the mask
    nb = BAND_BINS["F146"]
    inside = [t for t in np.arange(1, 11) * 7.2
              if m[max(0, int(np.clip(t / 72 * nb, 0, nb - 1)) - 1):int(np.clip(t / 72 * nb, 0, nb - 1)) + 2].all()]
    assert sorted(inside) == [21.6, 72.0]


def test_rmdc26_season_templates_match_the_measured_schedule():
    from pipeline.train import load_rmdc26_templates, _apply_schedule_template
    T = load_rmdc26_templates()
    assert len(T) == 6                                              # six high-cadence seasons
    n_empty = [int(t["F146"][0].sum()) for t in T]
    assert all(28 <= n <= 40 for n in n_empty), n_empty            # pauses + season end, 2-h bins
    assert len({tuple(np.flatnonzero(t["F146"][0])) for t in T}) == 6   # every season differs
    for t in T:
        assert t["F087"][0].sum() <= 2 and t["F213"][0].sum() <= 2  # colour bins are almost never empty
        fr = t["F146"][1]
        assert ((fr > 0) & (fr < 1)).sum() > 200                   # colour visits displace F146 epochs
    # applying a template: empty bins blanked in every band, occupancy capped, label kept
    out = {b: np.concatenate([np.full((L, 3), 0.5, np.float32), np.ones((L, 2), np.float32)], 1)
           for b, L in BAND_BINS.items()}
    lab = _apply_schedule_template(out, I_NON, T[1])
    assert lab == I_NON
    for b in BAND_BINS:
        empty, fr = T[1][b]
        assert np.all(out[b][empty, 4] == 0) and np.all(out[b][~empty, 3] == 1.0)   # occupancy untouched by default
    out2 = {b: np.concatenate([np.full((L, 3), 0.5, np.float32), np.ones((L, 2), np.float32)], 1) for b, L in BAND_BINS.items()}
    _apply_schedule_template(out2, I_NON, T[1], cap_occupancy=True)
    for b in BAND_BINS:
        empty, fr = T[1][b]
        assert np.all(out2[b][~empty, 3] <= fr[~empty] + 1e-6)

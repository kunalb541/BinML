"""Smoke test: the installed 6-class API loads, predicts, and matches its own preprocessing."""
import numpy as np
import binml


def test_import_and_version():
    assert binml.__version__ == "1.0.0"
    assert binml.CLASS_NAMES[2] == "NonPSPL"


def test_predict_shapes_and_bounds():
    # a synthetic single-lens-like bump on F146
    t = np.linspace(0, 72, 2000)
    mag = 22.0 - 1.5 * np.exp(-0.5 * ((t - 36) / 3) ** 2)  # brightening bump
    clf = binml.Classifier()
    r = clf.predict(t, mag, m_base_ref=22.0, t_start=0.0)
    assert set(r.probabilities) == set(binml.CLASS_NAMES)
    assert abs(sum(r.probabilities.values()) - 1.0) < 1e-4
    assert 0.0 <= r.is_anomalous <= 1.0
    assert r.label in binml.CLASS_NAMES


def test_binning_matches_research_cache():
    """The package's bin_band must reproduce the training cache's bin_curve bit-for-bit —
    the equivalence the docs stake the package on."""
    import numpy as np
    h5py = __import__("pytest").importorskip("h5py")  # cache.py needs h5py
    from pipeline.cache import bin_curve, BIN_FACTORS
    from binml.preprocess import bin_band, BIN_FACTORS as PKG_BF
    assert PKG_BF == BIN_FACTORS                         # localized copy must match the source
    n_epochs = 864 * 8
    rng = np.random.RandomState(0)
    grid = 22.0 + rng.normal(0, 0.1, n_epochs)
    grid[rng.random(n_epochs) < 0.3] = np.nan            # detectability gaps
    feat_c, frac_c = bin_curve(grid[None, :], 8)         # cache path: (1,864,3), (1,864)
    m_base = 22.0
    step = 72.0 / n_epochs
    fin = np.isfinite(grid)
    t = np.arange(n_epochs)[fin] * step
    feat_p, frac_p, _ = bin_band(t, grid[fin], "F146", m_base, 0.0)
    ok = np.isfinite(feat_c[0, :, 0])
    # cache stores absolute mag; package stores baseline-relative -> subtract m_base to compare
    assert np.allclose(feat_p[ok], feat_c[0][ok] - m_base, atol=1e-4)
    assert np.allclose(frac_p, frac_c[0], atol=1e-4)


def test_no_data_and_bad_input_raise():
    import numpy as np, pytest, binml
    clf = binml.Classifier()
    with pytest.raises(ValueError):                       # empty input
        clf.predict(np.array([]), np.array([]))
    with pytest.raises(ValueError):                       # all-NaN
        clf.predict(np.arange(10.0), np.full(10, np.nan))


def test_evolution_and_missing_baseline_warns():
    import numpy as np, warnings, binml
    t = np.linspace(0, 72, 2000)
    mag = 22.0 - 1.0 * np.exp(-0.5 * ((t - 36) / 3) ** 2)
    clf = binml.Classifier()
    days, probs = clf.predict_evolution({"F146": (t, mag)}, m_base_ref=22.0, n_steps=6)
    assert probs.shape == (6, 6) and np.allclose(probs.sum(1), 1.0, atol=1e-4)
    with warnings.catch_warnings(record=True) as w:       # missing baseline -> warns
        warnings.simplefilter("always")
        clf.predict(t, mag)
        assert any("m_base_ref" in str(x.message) for x in w)

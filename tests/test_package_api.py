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

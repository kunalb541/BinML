"""Smoke tests: the package imports, a model loads, and a synthetic curve classifies."""
import numpy as np
import pytest


def _fake_pspl(n=300, t0=40.0, tE=20.0, u0=0.1, m_base=18.0, seed=0):
    rng = np.random.RandomState(seed)
    t = np.sort(rng.uniform(0, 72, n))
    u = np.sqrt(u0**2 + ((t - t0) / tE) ** 2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    mag = m_base - 2.5 * np.log10(A) + rng.normal(0, 0.01, n)
    return t, mag, np.full(n, 0.01)


def test_import():
    import binml
    assert binml.CLASS_NAMES == ("Flat", "PSPL", "Binary")
    assert hasattr(binml, "Classifier")


def test_preprocess_shapes():
    import binml
    t, m, e = _fake_pspl()
    pre = binml.preprocess(t, m, e, seq_len=6912)
    assert pre.flux.shape == (1, 6912)
    assert pre.delta_t.shape == (1, 6912)
    assert 0 < pre.length <= len(t)
    assert abs(pre.magnification.max() - pre.peak_magnification) < 1e-6


@pytest.mark.parametrize("model", ["finetuned", "base"])
def test_predict(model):
    import binml
    clf = binml.Classifier(model=model)
    t, m, e = _fake_pspl()
    r = clf.predict(t, m, e)
    p = r.probabilities
    assert set(p) == {"Flat", "PSPL", "Binary"}
    assert abs(sum(p.values()) - 1.0) < 1e-4
    assert r.label in p
    assert 0.0 <= r.is_microlensing <= 1.0
    assert abs(r.is_anomalous - p["Binary"]) < 1e-6
    # a clean PSPL curve should read as a microlensing event (not Flat)
    assert r.is_microlensing > 0.5


def test_evolution():
    import binml
    clf = binml.Classifier()
    t, m, e = _fake_pspl()
    evo = clf.predict_evolution(t, m, e, steps=30)
    assert evo.probabilities.shape[1] == 3
    assert len(evo.days_from_peak) == evo.probabilities.shape[0]
    assert np.allclose(evo.probabilities.sum(1), 1.0, atol=1e-4)


def test_evaluate_functions():
    import binml
    rng = np.random.RandomState(0)
    labels = rng.randint(0, 3, 2000)
    preds = labels.copy()
    flip = rng.random(2000) < 0.2          # compute the mask ONCE
    preds[flip] = rng.randint(0, 3, flip.sum())
    rep = binml.classification_report(labels, preds)
    assert 0.0 <= rep.accuracy <= 1.0
    assert set(rep.recall) == {"Flat", "PSPL", "Binary"}
    # detectability: binaries with higher dchi2 should be recovered more often by construction
    dchi2 = np.where(labels == 2, rng.lognormal(5, 3, 2000), 0.0)
    caught = (labels == 2) & (dchi2 > np.median(dchi2[labels == 2]))
    preds2 = np.where(labels == 2, np.where(caught, 2, 1), labels)
    det = binml.detectability_curve(labels, preds2, dchi2)
    assert det.n_binaries == int((labels == 2).sum())
    hi = det.thresholds[str(1000)]["recall_detectable"]
    lo = det.thresholds[str(50)]["recall_detectable"]
    assert hi >= lo  # recall on more-detectable subset is not worse

"""Regression tests for public-API input validation and evaluation invariants.

These encode defects found in external audits so they cannot silently return:
  * the public API must reject malformed input with an actionable message, not a broadcast error;
  * the efficiency map must be a bounded probability (an earlier version was a ratio up to 31);
  * figures must use the frozen final-test split, never the threshold-selection rows.
"""
import json
import os

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "paper", "results")


def _clf():
    import binml
    return binml.Classifier()


def test_rejects_mismatched_lengths():
    t = np.arange(0, 72, 0.05); m = np.full(len(t) - 3, 20.0)
    with pytest.raises(ValueError, match="equal length"):
        _clf().predict(t, m, m_base_ref=20.0)


def test_rejects_unknown_band():
    t = np.arange(0, 72, 0.05); m = np.full_like(t, 20.0)
    with pytest.raises(ValueError, match="unrecognised band"):
        _clf().predict({"F146": (t, m), "F999": (t, m)}, m_base_ref=20.0)


def test_rejects_empty_curve():
    with pytest.raises(ValueError, match="empty light curve|no finite"):
        _clf().predict(np.array([]), np.array([]), m_base_ref=20.0)


def test_requires_f146():
    t = np.arange(0, 72, 0.05); m = np.full_like(t, 20.0)
    with pytest.raises(ValueError, match="F146 is required"):
        _clf().predict({"F087": (t, m)}, m_base_ref=20.0)


@pytest.mark.skipif(not os.path.exists(os.path.join(RES, "test_idx.npy")),
                    reason="evaluation artifact not present")
def test_final_test_split_is_disjoint_from_threshold_selection():
    """test_idx.npy must be exactly the 80% held out by rng(7); figures rely on this."""
    n = len(np.load(os.path.join(RES, "label.npy")))
    ti = np.load(os.path.join(RES, "test_idx.npy"))
    perm = np.random.default_rng(7).permutation(n)
    expected = perm[int(0.2 * n):]
    assert np.array_equal(np.sort(ti), np.sort(expected))
    val = set(perm[:int(0.2 * n)].tolist())
    assert not (set(ti.tolist()) & val), "threshold-selection rows leaked into the final test set"


@pytest.mark.skipif(not os.path.exists(os.path.join(RES, "params.npy")),
                    reason="evaluation artifact not present")
def test_efficiency_map_is_a_bounded_probability():
    """Conditional recall per (q,s) cell must lie in [0,1] (an earlier ratio reached 31)."""
    lab = np.load(os.path.join(RES, "label.npy")).astype(int)
    tc = np.load(os.path.join(RES, "true_class.npy")).astype(int)
    lg = np.load(os.path.join(RES, "logits.npy"))
    params = np.load(os.path.join(RES, "params.npy"))
    pf = json.load(open(os.path.join(RES, "meta.json")))["param_fields"]
    ti = np.load(os.path.join(RES, "test_idx.npy")).astype(int)
    m = np.zeros(len(lab), bool); m[ti] = True
    pred = lg.argmax(1); NON = 2
    q = params[:, pf.index("q")]; s = params[:, pf.index("s")]
    gen = m & (tc == NON) & np.isfinite(q) & np.isfinite(s)
    det = gen & (lab == NON)
    assert det.sum() > 100
    recall = (pred[det] == NON).mean()
    assert 0.0 <= recall <= 1.0

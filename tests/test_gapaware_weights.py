"""The released gap-aware checkpoint is the recommended one, and its threshold is the calibrated one."""
import hashlib
import json
import os

import pytest

pytest.importorskip("torch")
import binml  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_release_file_is_the_recommended_checkpoint():
    assert _sha(binml.GAPAWARE_WEIGHTS) == _sha(os.path.join(REPO, "validation/gulls/weights/ft_fspl5s_seasons_g08.pt"))


def test_threshold_is_the_calibrated_full_pool_threshold():
    cal = json.load(open(os.path.join(REPO, "validation/gulls/gapped_threshold_fspl5s_seasons_g08_seasons.json")))
    assert binml.GAPAWARE_THRESHOLD == cal["arms"]["rmdc26_gapped"]["pool"]["full_pool"]["threshold"]


def test_named_weights_load():
    a = binml.Classifier(weights="gapaware"); b = binml.Classifier(weights=binml.GAPAWARE_WEIGHTS)
    s = binml.Classifier(weights="shipped"); d = binml.Classifier()
    ka, kb = a._net.state_dict(), b._net.state_dict()
    assert all((ka[k] == kb[k]).all() for k in ka)
    ks, kd = s._net.state_dict(), d._net.state_dict()
    assert all((ks[k] == kd[k]).all() for k in ks)
    assert any((ka[k] != ks[k]).any() for k in ka)

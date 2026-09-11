"""Generation settings travel from raw shard attributes through the cache into the memmap's meta.json.

The 2026-09-12 re-verification found that run_shard's onset / noise / regime / magnification attributes
stopped at the raw shards, which the fine-tune runners delete once cached, so no artifact in use recorded
how its events were generated."""
import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
from pipeline.cache import BIN_FACTORS, build_cache  # noqa: E402
from pipeline.to_memmap import convert  # noqa: E402

N_EPOCHS = {"F146": 864 * 8, "F087": 96 * 3, "F213": 96 * 3}


def _fake_shard(path, n, espl):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.attrs["n_events"] = n
        f.attrs["param_fields"] = [b"tE", b"u0"]
        for a, v in (("regime", "fspl5s"), ("onset_resolution_days", 7.2), ("noise_mult", 1.0), ("bkg_mult", 1.0),
                     ("regime_priors", "{}"), ("espl_function", espl)):
            f.attrs[a] = v
        for b in BIN_FACTORS:
            f.create_dataset(f"mag/{b}", data=(20 + 0.01 * rng.standard_normal((n, N_EPOCHS[b]))).astype(np.float32))
            f.create_dataset(f"f_s/{b}", data=np.ones(n, np.float32)); f.create_dataset(f"n_kept/{b}", data=np.full(n, 10, np.float32))
        for k in ("label", "true_class", "n_usable_bands"):
            f.create_dataset(k, data=np.zeros(n, np.int64))
        for k in ("keep_prob", "dchi2_event", "dchi2_anomaly", "a_ks"):
            f.create_dataset(k, data=np.zeros(n, np.float32))
        f.create_dataset("m_base_ref", data=np.full(n, 20.0, np.float32))
        f.create_dataset("params", data=np.zeros((n, 2), np.float32))


def test_gen_settings_reach_the_memmap(tmp_path):
    _fake_shard(tmp_path / "a.h5", 4, "ESPLMag"); _fake_shard(tmp_path / "b.h5", 3, "ESPLMag2 (legacy)")
    build_cache([str(tmp_path / "a.h5"), str(tmp_path / "b.h5")], str(tmp_path / "c.h5"), verbose=False)
    with h5py.File(tmp_path / "c.h5", "r") as f:
        g = json.loads(f.attrs["gen_settings"])
    assert g["espl_function"] == ["ESPLMag", "ESPLMag2 (legacy)"] and g["onset_resolution_days"] == ["7.2"]
    convert([str(tmp_path / "c.h5")], str(tmp_path / "mm"))
    meta = json.load(open(tmp_path / "mm" / "meta.json"))
    assert meta["gen_settings"]["espl_function"] == ["ESPLMag", "ESPLMag2 (legacy)"]
    assert meta["n_events"] == 7

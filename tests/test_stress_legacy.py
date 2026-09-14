"""--legacy-t0-pad reproduces the July 2026 generator that made the stress suite (fourth and fifth verifications).

Before ef6b0bd the out-of-range re-pad was `params.setdefault("t0", rng.uniform(...))`: the draw was made but
discarded. The legacy switch must keep the pre-override t0 AND consume the same draw, so the rest of the random
stream is unchanged; the default must re-pad t0 to the overridden timescale. The shard-level test pins the July
generator's output: shard 0 of oor_pspl_shortte (seed base 931000000) regenerated with e25e4d6's pipeline/sim_v5 has
these label counts (checked by the fifth verification, 2026-09-13), and run_shard with both legacy switches must
reproduce them through its own argument parsing.

The counts are exact on the platform that recorded them (macOS arm64). Elsewhere the single-lens refit that sets a
few labels can differ in the last digits (a 1e-9 relative nudge of its starting point moves one or two labels), so
other platforms, including CI's Linux runners, check them within 3 per class. That still pins both switches: without
--legacy-t0-pad the shard has 159 fewer Flat and 137 more PSPL labels, without --legacy-oor-mix thousands more of each
(sixth check, 2026-09-14)."""
import platform
import h5py
import numpy as np
import pytest

from pipeline import assemble, run_shard
from pipeline.assemble import SurveyConfig, simulate_event
from pipeline.classes import CLASS_NAMES
from pipeline.run_shard import _make_oor_override

JULY_SHORTTE_SHARD0 = {"Eruptive": 479, "Flat": 2437, "LongPeriodVar": 168, "NonPSPL": 45, "PSPL": 617, "PeriodicVar": 370}


def _event(legacy, seed=12345):
    old = assemble.LEGACY_T0_PAD
    assemble.LEGACY_T0_PAD = legacy
    try:
        rng = np.random.default_rng(seed)
        ev = simulate_event("PSPL", rng, SurveyConfig(), param_override=_make_oor_override("oor_pspl_shortte"))
        return ev.params, rng.random()
    finally:
        assemble.LEGACY_T0_PAD = old


def test_legacy_t0_pad_keeps_t0_and_the_stream():
    for seed in (1, 2, 12345):
        (p_new, next_new), (p_old, next_old) = _event(False, seed), _event(True, seed)
        assert p_new["tE"] == p_old["tE"] and 0.2 <= p_new["tE"] <= 1.0      # the sweep redrew tE in both
        assert next_new == next_old                                           # same number of draws
        assert p_new["t0"] != p_old["t0"]                                     # only the kept t0 differs


def test_default_is_the_fixed_generator():
    assert assemble.LEGACY_T0_PAD is False


@pytest.mark.slow
def test_legacy_switches_reproduce_the_july_shard(tmp_path):
    old = (run_shard.LEGACY_OOR_MIX, assemble.LEGACY_T0_PAD)
    try:
        rc = run_shard.main(["--shard", "0", "--n-shards", "600", "--out", str(tmp_path), "--seed-base", "931000000",
                             "--regime", "oor_pspl_shortte", "--legacy-oor-mix", "--legacy-t0-pad"])
    finally:
        run_shard.LEGACY_OOR_MIX, assemble.LEGACY_T0_PAD = old
    assert not rc
    with h5py.File(tmp_path / "shard_00000.h5", "r") as f:
        assert bool(f.attrs["legacy_oor_mix"]) and bool(f.attrs["legacy_t0_pad"])
        lab = f["label"][:] if "label" in f else f["meta/label"][:]
    counts = {CLASS_NAMES[c]: int((lab == c).sum()) for c in range(len(CLASS_NAMES))}
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        assert counts == JULY_SHORTTE_SHARD0, counts
    else:
        assert all(abs(counts[k] - v) <= 3 for k, v in JULY_SHORTTE_SHARD0.items()), counts

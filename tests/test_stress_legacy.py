"""--legacy-t0-pad reproduces the July 2026 generator that made the stress suite (fourth verification, 2026-09-13).

Before ef6b0bd the out-of-range re-pad was `params.setdefault("t0", rng.uniform(...))`: the draw was made but
discarded. The legacy switch must keep the pre-override t0 AND consume the same draw, so the rest of the random
stream is unchanged; the default must re-pad t0 to the overridden timescale."""
import numpy as np

from pipeline import assemble
from pipeline.assemble import SurveyConfig, simulate_event
from pipeline.run_shard import _make_oor_override


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

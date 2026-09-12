"""Truth-based relabelling (audit findings 8-10, 2026-09-12).

Generation can store, per 2-hour reference-band bin, the noise-free signal deviation from baseline and, for
binaries, the anomaly residual against the best static PSPL (amplitude and chi^2). Training augmentations then
relabel by the SAME rule the labels were made with, restricted to what the augmentation left observed, instead
of a noisy min/max (the dead Flat branch), a range-traversed proxy for periodic variables, and a 7.2-day onset."""
import dataclasses

import numpy as np
import pytest

pytest.importorskip("VBBinaryLensing")
from pipeline.assemble import SurveyConfig, simulate_event  # noqa: E402

CFG = dataclasses.replace(SurveyConfig(), store_truth_bins=True)


def _first(true_class, want_label, seeds=range(200)):
    for s in seeds:
        ev = simulate_event(true_class, np.random.default_rng(s), CFG)
        if ev is not None and ev.label == want_label:
            return ev
    pytest.skip(f"no {true_class} -> {want_label} event in the seeds tried")


def test_truth_bins_reproduce_the_label_statistics():
    ev = _first("NonPSPL", "NonPSPL")
    t = ev.truth
    assert set(t) == {"vis_amp", "anom_amp", "anom_chi2"} and all(v.shape == (864,) for v in t.values())
    assert np.isclose(t["anom_chi2"].sum(), ev.dchi2_anomaly, rtol=1e-3)          # same residuals as the label rule
    assert t["anom_amp"].max() >= 0.02 and t["vis_amp"].max() >= 0.02
    ps = _first("PSPL", "PSPL")
    assert ps.truth["anom_chi2"].sum() == 0 and ps.truth["anom_amp"].max() == 0     # no anomaly for single lenses
    off = simulate_event("PSPL", np.random.default_rng(3), SurveyConfig())
    assert off is None or off.truth is None                                        # off by default


def _out(nb=864):
    from binml.preprocess import BAND_BINS
    return {b: np.concatenate([np.full((L, 3), 0.1, np.float32), np.ones((L, 2), np.float32)], 1)
            for b, L in BAND_BINS.items()}


def test_truth_relabel_rules():
    from pipeline.train import I_FLAT, I_NON, I_PSPL, _truth_relabel
    nb = 864
    vis = np.zeros(nb, np.float32); vis[400:420] = 0.5
    aa = np.zeros(nb, np.float32); aa[410:412] = 0.05
    ac = np.zeros(nb, np.float32); ac[410:412] = 500.0
    tr = (vis, aa, ac); allb = np.ones(nb, bool)
    assert _truth_relabel(I_NON, allb, tr) == I_NON
    hide = allb.copy(); hide[410:412] = False
    assert _truth_relabel(I_NON, hide, tr) == I_PSPL                    # the anomaly itself is hidden
    early = np.zeros(nb, bool); early[:405] = True
    assert _truth_relabel(I_NON, early, tr) == I_PSPL                   # revealed before the anomaly
    none = np.zeros(nb, bool); none[:300] = True
    assert _truth_relabel(I_NON, none, tr) == I_FLAT                    # nothing above the floor yet
    assert _truth_relabel(3, none, tr) == I_FLAT                        # any class, periodic included


def test_gaps_and_truncation_use_truth():
    from pipeline.train import I_NON, I_PSPL, _apply_gaps, _apply_truncation
    nb = 864
    vis = np.full(nb, 0.3, np.float32); aa = np.zeros(nb, np.float32); ac = np.zeros(nb, np.float32)
    aa[600:603] = 0.05; ac[600:603] = 400.0
    sched = np.zeros(nb, bool); sched[598:606] = True
    assert _apply_gaps(_out(), I_NON, np.random.default_rng(0), schedule=sched, truth=(vis, aa, ac)) == I_PSPL
    assert _apply_gaps(_out(), I_NON, np.random.default_rng(0), schedule=sched, truth=(vis, aa, ac),
                       relabel_anomaly=False) == I_NON
    sched2 = np.zeros(nb, bool); sched2[100:108] = True
    assert _apply_gaps(_out(), I_NON, np.random.default_rng(0), schedule=sched2, truth=(vis, aa, ac)) == I_NON
    # truncation: find a draw that cuts before bin 600 and one after
    labs = {}
    for s in range(200):
        o = _out(); lab = _apply_truncation(o, I_NON, np.random.default_rng(s), truth=(vis, aa, ac))
        cut = int((o["F146"][:, 4] > 0).sum())
        labs[cut > 603] = lab
        if len(labs) == 2:
            break
    assert labs == {False: I_PSPL, True: I_NON}


def test_truth_survives_writer_cache_and_memmap(tmp_path):
    import json
    from collections import Counter
    from pipeline.cache import build_cache
    from pipeline.to_memmap import convert
    from pipeline.writer import ShardWriter
    evs = [e for e in (simulate_event(c, np.random.default_rng(s), CFG)
                       for c, s in (("NonPSPL", 1), ("PSPL", 2), ("PeriodicVar", 3), ("NonPSPL", 4), ("Flat", 5)))
           if e is not None]
    raw = str(tmp_path / "shard.h5")
    with ShardWriter(raw, CFG) as w:
        w.append(evs); w.set_run_attrs(shard=0, byproduct_keep_prob=1.0, gen_counts=Counter(), dropped=Counter())
    build_cache([raw], str(tmp_path / "c.h5"), verbose=False)
    convert([str(tmp_path / "c.h5")], str(tmp_path / "mm"))
    meta = json.load(open(tmp_path / "mm" / "meta.json"))
    assert set(meta["truth"]) == {"vis_amp", "anom_amp", "anom_chi2"}
    n = meta["n_events"]
    chi = np.memmap(tmp_path / "mm" / "truth_anom_chi2.f32", dtype="float32", mode="r", shape=(n, 864))
    d = np.load(tmp_path / "mm" / "dchi2_anomaly.npy")
    for i in range(n):                                        # rows travel together through the shuffle
        if d[i] > 0:
            assert np.isclose(chi[i].sum(), d[i], rtol=1e-3)

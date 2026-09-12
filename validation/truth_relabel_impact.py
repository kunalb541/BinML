#!/usr/bin/env python3
"""How wrong were the legacy augmentation labels? -> validation/truth_relabel_impact.json (audit findings 8-10).

One natural-prior training shard generated with per-bin noise-free truth (run_shard --truth-bins) and the released
default onset grid (7.2 d), so the legacy rules are measured as the released training used them. Each event is
presented many times under each training augmentation (truncation, random gaps, the measured RMDC26 seasons); for
each presentation the SAME random draw is relabelled twice, by the legacy proxies (noisy surviving max for Flat, a
range-traversed amplitude for periodic classes, the 7.2-/0.5-day onset for binaries) and by the truth
(pipeline.train._truth_relabel). Reports the share of presentations whose label differs, by class and direction.
The gap and season augmentations are measured twice, with the legacy caustic-in-gap relabel on (the g08e12 and
round-3 recipes) and off (the recommended recipe and the *_norelabel arms); the truth side is the same draw.

Usage:  python validation/truth_relabel_impact.py --raw <shard.h5> [--reps 4]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", required=True); ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "truth_relabel_impact.json"))
    ap.add_argument("--onset-ref", default=None, help="the same shard generated with --onset-resolution-days 0.5: its "
                    "prefix-refit onsets are the reference for truncated binaries (see the truncation_vs_prefix_rule block)")
    args = ap.parse_args(argv)
    from binml.preprocess import BAND_BINS
    from pipeline.cache import build_cache
    from pipeline.classes import CLASS_NAMES
    from pipeline.to_memmap import convert
    from pipeline.train import (MAG_SCALE, _apply_gaps, _apply_schedule_template, _apply_truncation,
                                load_rmdc26_templates)
    work = tempfile.mkdtemp(prefix="truthimpact-")
    build_cache([args.raw], os.path.join(work, "c.h5"), verbose=False)
    mm = os.path.join(work, "mm"); convert([os.path.join(work, "c.h5")], mm)
    meta = json.load(open(os.path.join(mm, "meta.json"))); n = meta["n_events"]
    assert {"vis_amp", "anom_amp", "anom_chi2", "event_chi2"} <= set(meta.get("truth", {})), "shard lacks truth bins (--truth-bins)"
    feat = {b: np.memmap(os.path.join(mm, f"feat_{b}.f16"), dtype=np.float16, mode="r", shape=(n, L, 3)) for b, L in BAND_BINS.items()}
    frac = {b: np.memmap(os.path.join(mm, f"frac_{b}.f16"), dtype=np.float16, mode="r", shape=(n, L)) for b, L in BAND_BINS.items()}
    tr = {k: np.memmap(os.path.join(mm, f"truth_{k}.{'f16' if dt == 'float16' else 'f32'}"), dtype=dt, mode="r", shape=(n, nb))
          for k, (dt, nb) in meta["truth"].items()}
    lab = np.load(os.path.join(mm, "label.npy")); params = np.load(os.path.join(mm, "params.npy"))
    pf_idx = {k: i for i, k in enumerate(meta["param_fields"])}; fs = np.load(os.path.join(mm, "f_s_F146.npy"))
    tmpl = load_rmdc26_templates()

    def event(j):
        out = {}
        for b in BAND_BINS:
            f = np.asarray(feat[b][j], np.float32); obs = np.isfinite(f[:, 0]).astype(np.float32)
            out[b] = np.concatenate([np.nan_to_num(f) / MAG_SCALE, np.asarray(frac[b][j], np.float32)[:, None], obs[:, None]], 1)
        return out
    # self-consistency: with every bin observed, the truth rule must reproduce the stored label of every event
    from pipeline.train import _truth_relabel
    full = np.ones(meta["truth"]["vis_amp"][1], bool)
    incons = Counter()
    for j in range(n):
        tj = tuple(np.asarray(tr[k][j], np.float32) for k in ("vis_amp", "anom_amp", "anom_chi2", "event_chi2"))
        got = _truth_relabel(int(lab[j]), full & np.isfinite(np.asarray(feat["F146"][j][:, 0], np.float32)), tj)
        if got != int(lab[j]):
            incons[(CLASS_NAMES[int(lab[j])], CLASS_NAMES[got])] += 1
    print("full-window inconsistencies:", dict(incons), flush=True)
    res = {"full_window_inconsistent": {f"{a} -> {b}": v for (a, b), v in incons.items()}}
    for aug in ("truncation", "random_gaps", "measured_seasons", "random_gaps_relabel_off", "measured_seasons_relabel_off"):
        off = aug.endswith("_relabel_off")                  # legacy side with the caustic-in-gap relabel off
        cnt = Counter(); tot = Counter()
        for j in range(n):
            tj = tuple(np.asarray(tr[k][j], np.float32) for k in ("vis_amp", "anom_amp", "anom_chi2", "event_chi2"))
            for r in range(args.reps):
                seed = 1000003 * j + r
                if aug == "truncation":
                    a = _apply_truncation(event(j), int(lab[j]), np.random.default_rng(seed), params[j], pf_idx, float(fs[j]))
                    b = _apply_truncation(event(j), int(lab[j]), np.random.default_rng(seed), params[j], pf_idx, float(fs[j]), truth=tj)
                elif aug.startswith("random_gaps"):
                    a = _apply_gaps(event(j), int(lab[j]), np.random.default_rng(seed), params[j], pf_idx, relabel_anomaly=not off)
                    b = _apply_gaps(event(j), int(lab[j]), np.random.default_rng(seed), params[j], pf_idx, truth=tj)
                else:
                    t = tmpl[(j + r) % len(tmpl)]
                    a = _apply_schedule_template(event(j), int(lab[j]), t, params[j], pf_idx, relabel_anomaly=not off)
                    b = _apply_schedule_template(event(j), int(lab[j]), t, params[j], pf_idx, relabel_anomaly=True, truth=tj)
                c = CLASS_NAMES[int(lab[j])]; tot[c] += 1
                if a != b:
                    cnt[(c, CLASS_NAMES[a], CLASS_NAMES[b])] += 1
        res[aug] = {"presentations": dict(tot),
                    "disagree_frac_by_class": {c: round(sum(v for k, v in cnt.items() if k[0] == c) / tot[c], 4) for c in tot},
                    "transitions_legacy_to_truth": {f"{k[0]}: legacy {k[1]} / truth {k[2]}": v for k, v in cnt.most_common(12)}}
        print(aug, json.dumps(res[aug]["disagree_frac_by_class"]), flush=True)
    if args.onset_ref:
        res["truncation_vs_prefix_rule"] = _prefix_rule_check(args, n, lab, params, pf_idx, fs, tr, event, CLASS_NAMES)
    out = {"_doc": __doc__.split("\n")[0], "shard": os.path.basename(args.raw), "n_events": int(n), "reps": args.reps,
           "label_mix": {CLASS_NAMES[k]: int(v) for k, v in zip(*np.unique(lab, return_counts=True))},
           "gen_settings": meta.get("gen_settings"), "results": res,
           "code": subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=12"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
           "command": " ".join(sys.argv)}
    json.dump(out, open(args.out, "w"), indent=1); print("->", args.out)


def _prefix_rule_check(args, n, lab, params, pf_idx, fs, tr, event, CLASS_NAMES):
    """Truncated BINARIES against the label rule as the generator applies it to a prefix: the anomaly is detectable
    in [0, t_cut] iff the single-lens model REFIT ON THE PREFIX leaves dchi2 >= 160 and the floor, i.e. t_cut >= the
    onset recorded at 0.5-d resolution (assemble._anomaly_onset_day, full-grid scan). The truth bins cannot do that
    refit: they hold residuals against the FULL-SEASON fit, which a prefix refit can partly absorb. Reference label
    per presentation: the truth-bin Flat test (exact for the floor, no fit involved), then PSPL before the 0.5-d
    onset and NonPSPL from it. Compared with the legacy label (7.2-d onset) and the truth-bin label."""
    import h5py
    from pipeline.train import I_NON, I_PSPL, I_FLAT, _apply_truncation, _truth_relabel
    with h5py.File(args.onset_ref, "r") as f:
        pf = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["param_fields"]]
        p05 = f["params"][:]; res05 = float(f.attrs.get("onset_resolution_days", float("nan")))
    ti = pf.index("t_anom"); cols = [i for i in range(len(pf)) if i != ti]
    assert res05 == 0.5, f"--onset-ref must be generated with --onset-resolution-days 0.5 (got {res05})"
    # the cache/memmap reorders events: match each memmap row to its raw row by the injected parameters (t_anom excluded)
    key = lambda row: tuple(np.round(np.nan_to_num(row.astype(np.float64), nan=-999.0), 5))
    ref = {}
    for r_, row in enumerate(p05[:, cols]):
        ref.setdefault(key(row), []).append(r_)
    mcols = [pf_idx[pf[i]] for i in cols]
    t05 = np.full(n, np.nan)
    for j in np.flatnonzero(lab == I_NON):                   # binaries have distinct parameters; flat sources need none
        x = ref.get(key(params[j, mcols]), [])
        assert len(x) == 1, f"--onset-ref: memmap row {j} matches {len(x)} raw rows (not the same events)"
        t05[j] = p05[x[0], ti]
    assert p05.shape[0] == n
    names = ("legacy", "full_season_residuals", "floors_plus_onset_7p2", "floors_plus_onset_0p5")
    k = dict.fromkeys(names, 0); tot = 0; tr_ = Counter()
    for j in np.flatnonzero(lab == I_NON):
        tj = tuple(np.asarray(tr[kk][j], np.float32) for kk in ("vis_amp", "anom_amp", "anom_chi2", "event_chi2"))
        p05j = params[j].copy(); p05j[pf_idx["t_anom"]] = t05[j]
        for r in range(args.reps):
            seed = 1000003 * j + r
            f = float(np.random.default_rng(seed).uniform(0.03, 1.0))            # _apply_truncation's first draw
            ev = event(j)
            got = {"legacy": _apply_truncation(ev, I_NON, np.random.default_rng(seed), params[j], pf_idx, float(fs[j]))}
            surv = ev["F146"][:, 4] > 0
            got["full_season_residuals"] = _truth_relabel(I_NON, surv, tj)      # the first truth rule (2026-09-12 morning)
            got["floors_plus_onset_7p2"] = _apply_truncation(event(j), I_NON, np.random.default_rng(seed), params[j], pf_idx,
                                                             float(fs[j]), truth=tj)
            got["floors_plus_onset_0p5"] = _apply_truncation(event(j), I_NON, np.random.default_rng(seed), p05j, pf_idx,
                                                             float(fs[j]), truth=tj)
            ref = _truth_relabel(I_PSPL, surv, tj)                              # Flat test only (PSPL is never demoted further)
            if ref != I_FLAT:
                ref = I_NON if (np.isfinite(t05[j]) and f * 72.0 >= t05[j]) else I_PSPL
            tot += 1
            for m in names:
                if got[m] != ref:
                    k[m] += 1; tr_[f"{m} {CLASS_NAMES[got[m]]} / rule {CLASS_NAMES[ref]}"] += 1
    assert k["floors_plus_onset_0p5"] == 0, "the fixed truncation rule with the 0.5-d onset should reproduce the prefix rule"
    out = {"presentations": tot, "disagree_with_prefix_rule": {m: v / tot for m, v in k.items()},
           "disagree_counts": k, "transitions": dict(tr_.most_common(12)), "onset_ref": os.path.basename(args.onset_ref),
           "note": "legacy = released proxies + 7.2-d onset; full_season_residuals = the truth-bin anomaly test (withdrawn for "
                   "truncation); floors_plus_onset_* = pipeline.train._apply_truncation with truth (floors from the truth bins, "
                   "anomaly from the recorded onset) at the two onset resolutions"}
    print("truncation vs prefix rule:", json.dumps(out["disagree_with_prefix_rule"]), flush=True)
    return out


if __name__ == "__main__":
    sys.exit(main())

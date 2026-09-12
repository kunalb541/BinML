#!/usr/bin/env python3
"""How wrong were the legacy augmentation labels? -> validation/truth_relabel_impact.json (audit findings 8-10).

One natural-prior training shard generated with per-bin noise-free truth (run_shard --truth-bins) and the released
default onset grid (7.2 d), so the legacy rules are measured as the released training used them. Each event is
presented many times under each training augmentation (truncation, random gaps, the measured RMDC26 seasons); for
each presentation the SAME random draw is relabelled twice, by the legacy proxies (noisy surviving max for Flat, a
range-traversed amplitude for periodic classes, the 7.2-/0.5-day onset for binaries) and by the truth
(pipeline.train._truth_relabel). Reports the share of presentations whose label differs, by class and direction.

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
    for aug in ("truncation", "random_gaps", "measured_seasons"):
        cnt = Counter(); tot = Counter()
        for j in range(n):
            tj = tuple(np.asarray(tr[k][j], np.float32) for k in ("vis_amp", "anom_amp", "anom_chi2", "event_chi2"))
            for r in range(args.reps):
                seed = 1000003 * j + r
                if aug == "truncation":
                    a = _apply_truncation(event(j), int(lab[j]), np.random.default_rng(seed), params[j], pf_idx, float(fs[j]))
                    b = _apply_truncation(event(j), int(lab[j]), np.random.default_rng(seed), params[j], pf_idx, float(fs[j]), truth=tj)
                elif aug == "random_gaps":
                    a = _apply_gaps(event(j), int(lab[j]), np.random.default_rng(seed), params[j], pf_idx)
                    b = _apply_gaps(event(j), int(lab[j]), np.random.default_rng(seed), params[j], pf_idx, truth=tj)
                else:
                    t = tmpl[(j + r) % len(tmpl)]
                    a = _apply_schedule_template(event(j), int(lab[j]), t, params[j], pf_idx, relabel_anomaly=True)
                    b = _apply_schedule_template(event(j), int(lab[j]), t, params[j], pf_idx, relabel_anomaly=True, truth=tj)
                c = CLASS_NAMES[int(lab[j])]; tot[c] += 1
                if a != b:
                    cnt[(c, CLASS_NAMES[a], CLASS_NAMES[b])] += 1
        res[aug] = {"presentations": dict(tot),
                    "disagree_frac_by_class": {c: round(sum(v for k, v in cnt.items() if k[0] == c) / tot[c], 4) for c in tot},
                    "transitions_legacy_to_truth": {f"{k[0]}: legacy {k[1]} / truth {k[2]}": v for k, v in cnt.most_common(12)}}
        print(aug, json.dumps(res[aug]["disagree_frac_by_class"]), flush=True)
    out = {"_doc": __doc__.split("\n")[0], "shard": os.path.basename(args.raw), "n_events": int(n), "reps": args.reps,
           "label_mix": {CLASS_NAMES[k]: int(v) for k, v in zip(*np.unique(lab, return_counts=True))},
           "gen_settings": meta.get("gen_settings"), "results": res,
           "code": subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=12"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
           "command": " ".join(sys.argv)}
    json.dump(out, open(args.out, "w"), indent=1); print("->", args.out)


if __name__ == "__main__":
    sys.exit(main())

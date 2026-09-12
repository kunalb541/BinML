#!/usr/bin/env python3
"""How wrong were the legacy augmentation labels? -> validation/truth_relabel_impact.json (audit findings 8-10).

One natural-prior training shard generated with per-bin noise-free truth (run_shard --truth-bins) and the released
default onset grid (7.2 d), so the legacy rules are measured as the released training used them; --onset-ref is
the same shard generated with --onset-resolution-days 0.5 (same events). Each event is presented --reps times under
each training augmentation (truncation, random gaps, the measured RMDC26 seasons, cadence thinning), and the SAME
random draw is labelled by several rules. Two comparisons:

* legacy vs truth bins, every class (results.<augmentation>): the legacy proxies (noisy surviving max for Flat, a
  range-traversed amplitude for periodic classes, the recorded onset for truncated binaries, the caustic-in-gap
  relabel for gaps and pauses) against pipeline.train's truth relabel. For non-binary classes the truth relabel is
  the label rule itself (its floors need no fit), so these are the legacy errors. Counts are stored (k of n).
* refit reference, binaries only (results.refit_reference): the generator's own anomaly rule at the epochs an
  augmentation leaves -- each binary's noise-free F146 curve rebuilt from its stored parameters, a static PSPL
  refit on the surviving usable epochs (pipeline.assemble._pspl_refit_dchi2), dchi2 >= 160 and the amplitude floor
  -- after the truth-bin floors. Compared with it: the legacy label; the truth-bin anomaly test (residuals of the
  FULL-season fit on the surviving bins); and pipeline.train's truncation rule (truth floors + recorded onset) with
  the 7.2-d and the 0.5-d onset. The LM refit is chaotic for rare borderline events, so each reference decision is
  repeated with --perturb seeds perturbed at the 1e-7 level and a presentation whose decision changes is counted
  as unstable, not as a disagreement.

Also a two-sided self-check: from the GENERATOR class, the truth rule with every usable bin observed must reproduce
every stored label.

Usage:  python validation/truth_relabel_impact.py --raw <shard.h5> --onset-ref <shard_0p5.h5> [--reps 4] [--perturb 5]
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


CODE_AT_START = None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", required=True); ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "truth_relabel_impact.json"))
    ap.add_argument("--onset-ref", required=True, help="the same shard generated with --onset-resolution-days 0.5")
    ap.add_argument("--perturb", type=int, default=5, help="perturbed refit seeds per reference decision")
    args = ap.parse_args(argv)
    global CODE_AT_START
    CODE_AT_START = subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=12"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    import h5py
    from binml.preprocess import BAND_BINS
    from pipeline.cache import build_cache
    from pipeline.classes import CLASS_NAMES
    from pipeline.to_memmap import convert
    from pipeline.train import (MAG_SCALE, _apply_cadence, _apply_gaps, _apply_schedule_template, _apply_truncation,
                                _truth_relabel, load_rmdc26_templates)
    work = tempfile.mkdtemp(prefix="truthimpact-")
    build_cache([args.raw], os.path.join(work, "c.h5"), verbose=False)
    mm = os.path.join(work, "mm"); convert([os.path.join(work, "c.h5")], mm)
    meta = json.load(open(os.path.join(mm, "meta.json"))); n = meta["n_events"]
    assert {"vis_amp", "anom_amp", "anom_chi2", "event_chi2"} <= set(meta.get("truth", {})), "shard lacks truth bins (--truth-bins)"
    feat = {b: np.memmap(os.path.join(mm, f"feat_{b}.f16"), dtype=np.float16, mode="r", shape=(n, L, 3)) for b, L in BAND_BINS.items()}
    frac = {b: np.memmap(os.path.join(mm, f"frac_{b}.f16"), dtype=np.float16, mode="r", shape=(n, L)) for b, L in BAND_BINS.items()}
    tr = {k: np.memmap(os.path.join(mm, f"truth_{k}.{'f16' if dt == 'float16' else 'f32'}"), dtype=dt, mode="r", shape=(n, nb))
          for k, (dt, nb) in meta["truth"].items()}
    lab = np.load(os.path.join(mm, "label.npy")); tcl = np.load(os.path.join(mm, "true_class.npy"))
    params = np.load(os.path.join(mm, "params.npy"))
    pf_idx = {k: i for i, k in enumerate(meta["param_fields"])}; fs = np.load(os.path.join(mm, "f_s_F146.npy"))
    tmpl = load_rmdc26_templates()
    TJ = lambda j: tuple(np.asarray(tr[k][j], np.float32) for k in ("vis_amp", "anom_amp", "anom_chi2", "event_chi2"))

    def event(j):
        out = {}
        for b in BAND_BINS:
            f = np.asarray(feat[b][j], np.float32); obs = np.isfinite(f[:, 0]).astype(np.float32)
            out[b] = np.concatenate([np.nan_to_num(f) / MAG_SCALE, np.asarray(frac[b][j], np.float32)[:, None], obs[:, None]], 1)
        return out

    # two-sided self-check: from the GENERATOR class, with every usable bin observed, the truth rule must give the
    # stored label (starting from the stored label could not catch a rule that fails to demote)
    incons = Counter()
    for j in range(n):
        got = _truth_relabel(int(tcl[j]), np.isfinite(np.asarray(feat["F146"][j][:, 0], np.float32)), TJ(j))
        if got != int(lab[j]):
            incons[(CLASS_NAMES[int(tcl[j])], CLASS_NAMES[int(lab[j])], CLASS_NAMES[got])] += 1
    print("full-window inconsistencies (generator class, stored, truth rule):", dict(incons), flush=True)
    res = {"full_window_inconsistent": {f"{a} -> stored {b} / truth {c}": v for (a, b, c), v in incons.items()}}

    AUGS = ("truncation", "random_gaps", "measured_seasons", "random_gaps_relabel_off", "measured_seasons_relabel_off")
    for aug in AUGS:
        off = aug.endswith("_relabel_off")                  # legacy side with the caustic-in-gap relabel off
        cnt = Counter(); tot = Counter()
        for j in range(n):
            tj = TJ(j)
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
                    "disagree_k_by_class": {c: sum(v for k, v in cnt.items() if k[0] == c) for c in tot},
                    "disagree_frac_by_class": {c: sum(v for k, v in cnt.items() if k[0] == c) / tot[c] for c in tot},
                    "transitions_legacy_to_truth": {f"{k[0]}: legacy {k[1]} / truth {k[2]}": v for k, v in cnt.most_common()}}
        print(aug, json.dumps({c: round(v, 4) for c, v in res[aug]["disagree_frac_by_class"].items()}), flush=True)
    res["refit_reference"] = _refit_reference(args, n, lab, params, pf_idx, fs, TJ, event, CLASS_NAMES, tmpl, meta, mm)
    out = {"_doc": __doc__.split("\n")[0], "shard": os.path.basename(args.raw), "onset_ref": os.path.basename(args.onset_ref),
           "n_events": int(n), "reps": args.reps, "perturb": args.perturb,
           "label_mix": {CLASS_NAMES[k]: int(v) for k, v in zip(*np.unique(lab, return_counts=True))},
           "gen_settings": meta.get("gen_settings"), "results": res,
           "code": CODE_AT_START,
           "command": " ".join(sys.argv)}
    json.dump(out, open(args.out, "w"), indent=1); print("->", args.out)


def _refit_reference(args, n, lab, params, pf_idx, fs, TJ, event, CLASS_NAMES, tmpl, meta, mm):
    """Binaries: every augmentation's labels against the generator's own anomaly rule at the surviving epochs."""
    import h5py
    from pipeline.assemble import SurveyConfig, _epochs, _pspl_refit_dchi2
    from pipeline.generators import _binary_magnification
    from pipeline.photometry import ROMAN_BANDS, photometric_sigma
    from pipeline.train import (I_FLAT, I_NON, I_PSPL, _apply_cadence, _apply_gaps, _apply_schedule_template,
                                _apply_truncation, _truth_relabel)
    CFG = SurveyConfig(); T = _epochs("F146", CFG.window_days); BAND = ROMAN_BANDS["F146"]
    nb = meta["truth"]["vis_amp"][1]; BIN = np.arange(T.size) * nb // T.size          # epoch -> reference bin
    mbase = np.load(os.path.join(mm, "m_base_ref.npy"))
    with h5py.File(args.raw, "r") as f:
        pf = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["param_fields"]]
        praw = f["params"][:]; usable_raw = np.isfinite(f["mag/F146"][:])
    with h5py.File(args.onset_ref, "r") as f:
        pf5 = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["param_fields"]]
        p05 = f["params"][:]; res05 = float(f.attrs.get("onset_resolution_days", float("nan")))
    assert res05 == 0.5, f"--onset-ref must be generated with --onset-resolution-days 0.5 (got {res05})"
    ti = pf.index("t_anom"); cols = [i for i in range(len(pf)) if i != ti]
    assert pf5 == pf and np.allclose(np.nan_to_num(p05[:, cols], nan=-9), np.nan_to_num(praw[:, cols], nan=-9)), \
        "--onset-ref is not the same events in the same order"
    key = lambda row: np.nan_to_num(row.astype(np.float32), nan=-999.0).tobytes()
    rawpos = {}
    for r_, row in enumerate(praw[:, cols]):
        rawpos.setdefault(key(row), []).append(r_)
    mcols = [pf_idx[pf[i]] for i in cols]
    binm = np.flatnonzero(lab == I_NON)
    raw_of = {}
    for j in binm:                                   # the memmap reorders events; binaries have distinct parameters
        x = rawpos.get(key(params[j, mcols]), [])
        assert len(x) == 1, f"memmap row {j} matches {len(x)} raw rows"
        raw_of[j] = x[0]
    prng = np.random.default_rng(20260913)
    curves = {}

    def curve(j):
        if j not in curves:
            P = {k: float(params[j, pf_idx[k]]) for k in ("t0", "tE", "u0", "s", "q", "alpha", "rho")}
            A = _binary_magnification(T, P["t0"], P["tE"], P["u0"], P["s"], P["q"], P["alpha"], P["rho"])
            mb, f_ = float(mbase[j]), float(fs[j])
            mag = mb - 2.5 * np.log10(np.maximum(1.0 + f_ * (A - 1.0), 1e-8))
            curves[j] = (P, mag, photometric_sigma(BAND, mag, CFG.noise_mult), usable_raw[raw_of[j]], mb, f_)
        return curves[j]

    def rule(j, surv, tj):
        """Truth-bin floors, then a refit on the surviving usable epochs; 'unstable' if perturbed seeds disagree."""
        if _truth_relabel(I_PSPL, surv, tj) == I_FLAT:
            return I_FLAT
        P, mag, sig, u, mb, f_ = curve(j); m = u & surv[BIN]
        if int(m.sum()) < 10:
            return I_PSPL                              # the generator needs 10 reference epochs for an anomaly
        dec = set()
        for k in range(1 + args.perturb):
            Q = dict(P)
            if k:
                for q in ("t0", "tE", "u0"):
                    Q[q] = P[q] * (1.0 + prng.normal(0.0, 1e-7))
            d, amp = _pspl_refit_dchi2(T[m], mag[m], sig[m], mb, f_, Q)
            dec.add(I_NON if (d >= CFG.dchi2_anomaly and amp >= CFG.min_amplitude_mag) else I_PSPL)
        return dec.pop() if len(dec) == 1 else "unstable"

    # the full-window reference must reproduce the stored NonPSPL labels (checks the rebuild)
    full = Counter(str(rule(j, event(j)["F146"][:, 4] > 0, TJ(j))) for j in binm)
    full_mismatch = sum(v for k, v in full.items() if k not in (str(I_NON), "unstable"))
    print("full-window refit reference on stored binaries:", dict(full), flush=True)
    C = {a: Counter() for a in ("truncation", "measured_seasons", "random_gaps", "cadence")}; tot = Counter()
    for j in binm:
        tj = TJ(j); p05j = params[j].copy(); p05j[pf_idx["t_anom"]] = p05[raw_of[j], ti]
        for r in range(args.reps):
            seed = 1000003 * j + r
            got = {}
            ev = event(j); got["legacy"] = _apply_truncation(ev, I_NON, np.random.default_rng(seed), params[j], pf_idx, float(fs[j]))
            surv = ev["F146"][:, 4] > 0
            got["full_season_residuals"] = _truth_relabel(I_NON, surv, tj)
            got["floors_onset_7p2"] = _apply_truncation(event(j), I_NON, np.random.default_rng(seed), params[j], pf_idx, float(fs[j]), truth=tj)
            got["floors_onset_0p5"] = _apply_truncation(event(j), I_NON, np.random.default_rng(seed), p05j, pf_idx, float(fs[j]), truth=tj)
            _tally(C["truncation"], got, rule(j, surv, tj), CLASS_NAMES); tot["truncation"] += 1
            t = tmpl[(j + r) % len(tmpl)]
            ev = event(j); got = {"legacy_relabel_on": _apply_schedule_template(ev, I_NON, t, params[j], pf_idx, relabel_anomaly=True)}
            surv = ev["F146"][:, 4] > 0
            got["legacy_relabel_off"] = _apply_schedule_template(event(j), I_NON, t, params[j], pf_idx, relabel_anomaly=False)
            got["truth_bins"] = _apply_schedule_template(event(j), I_NON, t, params[j], pf_idx, relabel_anomaly=True, truth=tj)
            _tally(C["measured_seasons"], got, rule(j, surv, tj), CLASS_NAMES); tot["measured_seasons"] += 1
            ev = event(j); got = {"legacy_relabel_on": _apply_gaps(ev, I_NON, np.random.default_rng(seed), params[j], pf_idx, relabel_anomaly=True)}
            surv = ev["F146"][:, 4] > 0
            got["legacy_relabel_off"] = _apply_gaps(event(j), I_NON, np.random.default_rng(seed), params[j], pf_idx, relabel_anomaly=False)
            got["truth_bins"] = _apply_gaps(event(j), I_NON, np.random.default_rng(seed), params[j], pf_idx, truth=tj)
            _tally(C["random_gaps"], got, rule(j, surv, tj), CLASS_NAMES); tot["random_gaps"] += 1
            ev = event(j); got = {"legacy": _apply_cadence(ev, I_NON, np.random.default_rng(seed), params[j], pf_idx)}
            surv = ev["F146"][:, 4] > 0
            got["truth_bins"] = _apply_cadence(event(j), I_NON, np.random.default_rng(seed), params[j], pf_idx, truth=tj)
            _tally(C["cadence"], got, rule(j, surv, tj), CLASS_NAMES, suffix=" [keep<0.2]" if surv.mean() < 0.2 else ""); tot["cadence"] += 1
    out = {"n_binaries": int(binm.size), "full_window_reference_not_binary": int(full_mismatch),
           "full_window_reference_unstable": int(full.get("unstable", 0)),
           "note": "counts per presentation; '<method> X / refit Y' = the method's label X where the refit reference gives Y; "
                   "'unstable' = the reference decision changes under 1e-7 perturbations of the refit seed"}
    for a, c in C.items():
        out[a] = {"presentations": tot[a], "counts": dict(sorted(c.items(), key=lambda x: -x[1]))}
    print("refit reference:", json.dumps({a: {k: v for k, v in list(out[a]["counts"].items())[:6]} for a in C}), flush=True)
    return out


def _tally(counter, got, ref, CLASS_NAMES, suffix=""):
    for m, v in got.items():
        if ref == "unstable":
            counter[f"{m}: reference unstable{suffix}"] += 1
        elif v != ref:
            counter[f"{m} {CLASS_NAMES[v]} / refit {CLASS_NAMES[ref]}{suffix}"] += 1


if __name__ == "__main__":
    sys.exit(main())

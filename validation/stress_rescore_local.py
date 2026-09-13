#!/usr/bin/env python3
"""The stress-test numbers for the RELEASED checkpoint -> validation/stress_rescore_local.json.

The 14.9-million-event stress suite (paper/results/stress_report.json; aws/controller.sh) was scored in July 2026
with the stage-5 checkpoint (v5runs/binml_v5_stage5.pt, sha256 4e7a5a85...), the predecessor of the released
binml.pt (= stage 6). The paper had quoted those numbers as the released model's. The full suite is on S3 and is not
re-scored here; instead this regenerates the first shards of the tiers the paper quotes, with the suite's seed bases
and regimes, its out-of-range class mix (run_shard --legacy-oor-mix) and, for the out-of-range sweeps, its
generator's t0 draw (run_shard --legacy-t0-pad; the July generator left ~20% of sub-day single lenses peaking outside
the season), and scores the same events with both checkpoints:

* stage 5 on the regenerated subset against the suite's full-population numbers shows how close the subset is to the
  suite. It is a new realisation, not the suite's events: the suite's cloud fleet ran an unpinned software
  environment, and locally the same code labels ~2% fewer detectable anomalies in the natural tier, so comparisons
  between the two checkpoints are made within the subset;
* the released checkpoint on the same events gives the numbers the paper quotes for it;
* `oor_pspl_shortte_current` is the sub-day tier with the current (fixed) t0 draw, the corrected population.

Metrics are pipeline.agg_stress's: argmax classes, per-class recall / precision / F1 with population weights
(1/keep_prob) over every event of a tier; macro-F1 over the six classes for the natural tier.

The suite's PSPL recalls are by LABEL. An out-of-range sweep perturbs only its swept generator class, so in
oor_pspl_shortte the PSPL-labelled events are sub-day single lenses plus binaries of natural timescale whose anomaly
fails the detectability policy (demoted to PSPL; population weight 1/keep_prob), and in oor_flat_faint faint single
lenses plus faint demoted binaries. `pspl_label_by_generator_class` splits them, and `single_lens_recall` gives the
number the paper quotes for single lenses (not available for the suite, whose per-event predictions are on S3).
The faint sweep's class mix is mostly flat sources (MIXES["flat"]), so its anomaly prevalence -- and hence its
anomaly precision -- is set mostly by that mix; `pspl_label_false_anomaly_w` and `pspl_label_above_frozen_w` are the
mix-independent false-anomaly rates among microlensing events without a detectable anomaly.

Each tier's shards, caches and evaluations are stamped with the code state that made them (WORK/stamp_<tier>.txt,
written when the tier's first shard is generated) and reused when present.

Usage:  python validation/stress_rescore_local.py [--workers 4]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
WORK = os.path.expanduser("~/Desktop/Research/microlensing/stress_local_work")
CKPTS = {"stage5": os.path.expanduser("~/Desktop/Research/microlensing/v5runs/binml_v5_stage5.pt"),
         "released": os.path.join(REPO, "binml", "weights", "binml.pt")}
# tier: (seed base as in aws/controller.sh and aws/launch_stress.sh, regime, shards regenerated here, run_shard flags)
LEGACY = ["--legacy-oor-mix", "--legacy-t0-pad"]
TIERS = {"natural": (900000000, None, 16, []), "planetary": (910000000, "planetary", 8, []),
         "oor_np_widesep": (935000000, "oor_np_widesep", 4, LEGACY), "oor_per_longp": (938000000, "oor_per_longp", 4, LEGACY),
         "oor_pspl_shortte": (931000000, "oor_pspl_shortte", 4, LEGACY), "oor_flat_faint": (945000000, "oor_flat_faint", 4, ["--legacy-oor-mix"]),   # a config regime: no t0 re-pad
         "oor_pspl_shortte_current": (931000000, "oor_pspl_shortte", 4, ["--legacy-oor-mix"])}
FROZEN = 0.9042405486106873     # the released model's complete-season operating threshold (paper/results/metrics.json)
# the numbers the paper quotes: (tier, class, metric)
QUOTED = {"natural_np_recall": ("natural", "NonPSPL", "recall"), "natural_np_prec": ("natural", "NonPSPL", "precision"),
          "planetary_np_recall": ("planetary", "NonPSPL", "recall"), "planetary_np_prec": ("planetary", "NonPSPL", "precision"),
          "widesep_np_recall": ("oor_np_widesep", "NonPSPL", "recall"), "longp_per_recall": ("oor_per_longp", "PeriodicVar", "recall"),
          "shortte_pspl_recall": ("oor_pspl_shortte", "PSPL", "recall"), "faint_pspl_recall": ("oor_flat_faint", "PSPL", "recall"),
          "faint_np_prec": ("oor_flat_faint", "NonPSPL", "precision")}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r


def stamp(tier, code):
    """The code state that generated and scored a tier: written when its first shard is generated, read back after."""
    p = os.path.join(WORK, f"stamp_{tier}.txt")
    if not os.path.exists(p):
        open(p, "w").write(code + "\n")
    return open(p).read().strip()


def gen(tier, shard, code):
    seed, regime, _, flags = TIERS[tier]
    d = os.path.join(WORK, f"raw_{tier}"); os.makedirs(d, exist_ok=True)
    out = os.path.join(d, f"shard_{shard:05d}.h5")
    if not os.path.exists(out):
        stamp(tier, code)
        cmd = [sys.executable, "-m", "pipeline.run_shard", "--shard", str(shard), "--n-shards", "600", "--out", d, "--seed-base", str(seed)]
        if regime:
            cmd += ["--regime", regime]
        cmd += flags
        t0 = time.time(); run(cmd); log(f"  generated {tier} shard {shard} in {(time.time() - t0) / 60:.1f} min")
    return out


def memmap(tier):
    mm = os.path.join(WORK, f"mm_{tier}")
    if os.path.exists(os.path.join(mm, "meta.json")):
        return mm
    cdir = os.path.join(WORK, f"cache_{tier}"); os.makedirs(cdir, exist_ok=True)
    for s in range(TIERS[tier][2]):
        c = os.path.join(cdir, f"{tier}_{s:05d}.h5")
        if not os.path.exists(c):
            run([sys.executable, "-c", "import sys; from pipeline.cache import build_cache; build_cache([sys.argv[1]], sys.argv[2])",
                 os.path.join(WORK, f"raw_{tier}", f"shard_{s:05d}.h5"), c])
    run([sys.executable, "-m", "pipeline.to_memmap", "--in-dir", cdir, "--out", mm])
    return mm


def evaluate(name, tier):
    ev = os.path.join(WORK, f"eval_{name}_{tier}")
    if not os.path.exists(os.path.join(ev, "metrics.json")):
        run([sys.executable, "-m", "pipeline.evaluate", "--ckpt", CKPTS[name], "--cache", memmap(tier), "--out", ev, "--device", "cpu"])
    return ev


def metrics(ev):
    """pipeline.agg_stress's definitions on every event of the tier."""
    from pipeline.agg_stress import prf
    from pipeline.classes import CLASS_NAMES
    y = np.load(os.path.join(ev, "label.npy")).astype(int); pred = np.load(os.path.join(ev, "logits.npy")).argmax(1)
    score = np.load(os.path.join(ev, "score_nonpspl.npy")).astype(np.float64)
    w = 1.0 / np.clip(np.load(os.path.join(ev, "keep_prob.npy")).astype(np.float64), 1e-3, 1.0)
    cls = {}
    for c, name in enumerate(CLASS_NAMES):
        if (y == c).sum():
            r, p, f = prf(y, pred, w, c)
            cls[name] = {"recall": float(r), "precision": float(p), "f1": float(f), "n": int((y == c).sum())}
    out = {"n": int(y.size), "per_class": cls,
           "label_fractions": {n: float((y == c).mean()) for c, n in enumerate(CLASS_NAMES)},
           "label_fractions_w": {n: float(w[y == c].sum() / w.sum()) for c, n in enumerate(CLASS_NAMES)}}
    if len(cls) == len(CLASS_NAMES):
        out["macro_f1"] = float(np.mean([cls[n]["f1"] for n in CLASS_NAMES]))
    # PSPL-labelled events split by generator class (see the module docstring)
    tc = np.load(os.path.join(ev, "true_class.npy")).astype(int)
    ip, inp = CLASS_NAMES.index("PSPL"), CLASS_NAMES.index("NonPSPL")
    lab = y == ip
    by_gen = {}
    for gname, gi in (("single_lenses", ip), ("demoted_binaries", inp)):
        m = lab & (tc == gi)
        if m.any():
            by_gen[gname] = {"n": int(m.sum()), "recall": float((w[m] * (pred[m] == ip)).sum() / w[m].sum()),
                             "weighted_share_of_label": float(w[m].sum() / w[lab].sum()),
                             "argmax_fractions": {n: float(w[m & (pred == c)].sum() / w[m].sum()) for c, n in enumerate(CLASS_NAMES)},
                             "frac_above_frozen_threshold": float(w[m & (score >= FROZEN)].sum() / w[m].sum())}
    out["pspl_label_by_generator_class"] = by_gen
    # mix-independent false anomalies: microlensing events without a detectable anomaly (label PSPL) called anomalies
    out["pspl_label_false_anomaly_w"] = float((w[lab] * (pred[lab] == inp)).sum() / w[lab].sum()) if lab.any() else None
    out["pspl_label_above_frozen_w"] = float((w[lab] * (score[lab] >= FROZEN)).sum() / w[lab].sum()) if lab.any() else None
    tcls = np.load(os.path.join(ev, "true_class.npy")).astype(int)
    gb = tcls == inp
    out["binary_detectable_fraction_w"] = float((w[gb] * (y[gb] == inp)).sum() / w[gb].sum()) if gb.any() else None
    # anomaly-call rates, weighted: precision depends on the tier's anomaly prevalence, so keep the pieces
    inon = CLASS_NAMES.index("NonPSPL")
    pos, flag = y == inon, pred == inon
    out["nonpspl_rates"] = {"prevalence_w": float(w[pos].sum() / w.sum()),
                            "tpr_w": float(w[pos & flag].sum() / max(w[pos].sum(), 1e-12)),
                            "fpr_w": float(w[~pos & flag].sum() / w[~pos].sum()),
                            "n_flagged": int(flag.sum()), "n_false_flags": int((flag & ~pos).sum())}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "stress_rescore_local.json"))
    args = ap.parse_args(argv)
    code = subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=12"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    if not code:
        raise SystemExit("FATAL: git describe returned nothing (disk stall?); rerun -- an artifact must record its code")
    os.makedirs(WORK, exist_ok=True)
    jobs = [(t, s) for t in TIERS for s in range(TIERS[t][2])]
    log(f"generating {len(jobs)} shards with {args.workers} workers")
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(lambda x: gen(*x, code), jobs))
    report = json.load(open(os.path.join(REPO, "paper", "results", "stress_report.json")))
    res = {"_doc": __doc__.split("\n")[0], "code": code, "command": " ".join(sys.argv),
           "checkpoints": {k: {"path": os.path.relpath(v, REPO) if v.startswith(REPO) else v,
                               "sha256": hashlib.sha256(open(v, "rb").read()).hexdigest()} for k, v in CKPTS.items()},
           "tiers": {t: {"seed_base": TIERS[t][0], "regime": TIERS[t][1], "shards": list(range(TIERS[t][2])),
                         "run_shard_flags": TIERS[t][3], "code_generation_and_scoring": stamp(t, code),
                         "suite_n": report["regimes"][TIERS[t][1] or "natural"]["n"]} for t in TIERS},
           "subset": {}, "quoted": {}}
    for t in TIERS:
        res["subset"][t] = {name: metrics(evaluate(name, t)) for name in CKPTS}
        log(f"{t}: " + json.dumps({k: {c: round(v['per_class'][c]['recall'], 3) for c in v['per_class']} for k, v in res['subset'][t].items()}))
    suite_macro = report["natural_population"]["macro_f1"]
    res["quoted"]["natural_macro_f1"] = {"suite_stage5": suite_macro, "subset_stage5": res["subset"]["natural"]["stage5"]["macro_f1"],
                                         "subset_released": res["subset"]["natural"]["released"]["macro_f1"]}
    for key, (t, c, m) in QUOTED.items():
        suite = (report["natural_population"]["per_class"][c][m] if t == "natural" else report["regimes"][t]["per_class"][c][m])
        res["quoted"][key] = {"suite_stage5": suite, "subset_stage5": res["subset"][t]["stage5"]["per_class"][c][m],
                              "subset_released": res["subset"][t]["released"]["per_class"][c][m],
                              "n_subset": res["subset"][t]["released"]["per_class"][c]["n"]}
    res["single_lens_recall"] = {t: {name: res["subset"][t][name]["pspl_label_by_generator_class"]["single_lenses"]
                                     for name in CKPTS} for t in ("oor_pspl_shortte", "oor_pspl_shortte_current", "oor_flat_faint", "natural")}
    # the same tier's rates at the natural population's anomaly prevalence (prior shift)
    pi = res["subset"]["natural"]["released"]["nonpspl_rates"]["prevalence_w"]
    res["precision_at_natural_prevalence"] = {"natural_prevalence_w": pi}
    for t in ("planetary", "oor_np_widesep", "oor_flat_faint"):
        res["precision_at_natural_prevalence"][t] = {}
        for name in CKPTS:
            r = res["subset"][t][name]["nonpspl_rates"]
            res["precision_at_natural_prevalence"][t][name] = r["tpr_w"] * pi / (r["tpr_w"] * pi + r["fpr_w"] * (1 - pi))
    res["suite_label_fractions"] = {t: {c: v["n"] / report["regimes"][TIERS[t][1] or "natural"]["n"]
                                        for c, v in report["regimes"][TIERS[t][1] or "natural"]["per_class"].items()}
                                    for t in TIERS}
    json.dump(res, open(args.out, "w"), indent=1)
    log(f"wrote {args.out}")
    print(json.dumps(res["quoted"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

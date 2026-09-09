"""Cadence comparison (15-min legacy vs 12-min GBTDS rate), rerun LOCALLY with a HELD-OUT evaluation.

WHY THIS EXISTS.  validation/modal_cadence.py trained each arm on a memmap and then ran
pipeline.evaluate on the SAME memmap.  train.py splits that cache 80/10/10 with its own seed and
evaluate.py re-splits it 20/80 with a different seed, so ~80% of the rows evaluate.py scored had
been trained on (docs/AUDIT_2026-09-09.md, finding 54).  The 15-vs-12-minute CONTRAST survived --
both arms carried the identical defect -- but every absolute number in the paper's cadence
paragraph was optimistic.  The Modal checkpoints lived in container /tmp and are gone, so the
experiment is rerun here with the SAME training recipe (same shard indices -> same seeds -> the
same training events as the original run; 12 epochs; seed 20260720; truncate-aug 0.5) and a
disjoint held-out set: shards 100-103, whose seeds (seed_base + shard*7919) never appear in
training.  For the record the runner also scores each arm on its own training pool, so the
inflation the leak produced is measured rather than asserted.

ISOLATION.  The 12-min arm patches four source files (cadence, F146 bin factor 8->10, epoch count
6912->8640).  Each arm runs from its OWN copy of pipeline/ and binml/ under --work, and every step
is a subprocess with PYTHONPATH pointing at that copy, so the repo tree is never modified and the
two cadences cannot mix -- the same guarantee the Modal containers gave.

RESUMABLE.  Every step skips when its output exists; training is gated on a .done marker rather
than the checkpoint file (train.py rewrites the checkpoint at every new best epoch).

Usage:
  python validation/cadence_local.py --work ~/Desktop/Research/microlensing/cadence_local_work \
      --workers 8
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

CAD_PATCH = [
    ("pipeline/photometry.py", "cadence_minutes=15.0", "cadence_minutes=12.0"),
    ("pipeline/cache.py", '"F146": 8, "F087": 3, "F213": 3', '"F146": 10, "F087": 3, "F213": 3'),
    ("binml/preprocess.py", '"F146": 8, "F087": 3, "F213": 3', '"F146": 10, "F087": 3, "F213": 3'),
    ("pipeline/assemble.py", "6912", "8640"),
]
SEED = 20260720
N_SHARDS_TOTAL = 200          # --n-shards passed to run_shard, as in the Modal recipe
TRAIN_SHARDS = list(range(12))            # identical to modal_cadence.py -> identical events
EVAL_SHARDS = [100, 101, 102, 103]        # disjoint seeds, never trained on


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def arm_dir(work, cad):
    return os.path.join(work, f"arm{cad}")


def setup_arm(work, cad):
    """Copy pipeline/ and binml/ into an isolated tree; patch the 12-min arm."""
    d = arm_dir(work, cad)
    if os.path.exists(os.path.join(d, ".setup.done")):
        return d
    os.makedirs(d, exist_ok=True)
    for pkg in ("pipeline", "binml"):
        dst = os.path.join(d, pkg)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(os.path.join(REPO, pkg), dst,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "weights"))
    if cad == 12:
        for f, a, b in CAD_PATCH:
            p = os.path.join(d, f)
            s = open(p).read()
            assert a in s, f"pattern not found in {f}: {a}"
            open(p, "w").write(s.replace(a, b))
    open(os.path.join(d, ".setup.done"), "w").write(time.strftime("%F %T") + "\n")
    return d


def run(cmd, cwd, extra_env=None):
    env = dict(os.environ, PYTHONPATH=cwd)
    if extra_env:
        env.update(extra_env)
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def gen_one(arm, raw_dir, shard):
    out = os.path.join(raw_dir, f"shard_{shard:05d}.h5")
    if os.path.exists(out):
        return shard, 0.0
    t0 = time.time()
    run([sys.executable, "-m", "pipeline.run_shard", "--shard", str(shard),
         "--n-shards", str(N_SHARDS_TOTAL), "--out", raw_dir, "--seed-base", str(SEED)], cwd=arm,
        extra_env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    assert os.path.exists(out), f"run_shard produced no {out}"
    return shard, time.time() - t0


def cache_one(arm, raw, out):
    if os.path.exists(out):
        return
    run([sys.executable, "-c",
         "import sys; from pipeline.cache import build_cache; build_cache([sys.argv[1]], sys.argv[2])",
         raw, out], cwd=arm)


def build_split(arm, work_arm, split, shards, workers):
    raw_dir = os.path.join(work_arm, f"raw_{split}")
    cache_dir = os.path.join(work_arm, f"cache_{split}")
    mm = os.path.join(work_arm, f"mm_{split}")
    os.makedirs(raw_dir, exist_ok=True); os.makedirs(cache_dir, exist_ok=True)
    todo = [s for s in shards if not os.path.exists(os.path.join(raw_dir, f"shard_{s:05d}.h5"))]
    log(f"  {split}: {len(shards) - len(todo)} shards cached, generating {len(todo)} with {workers} workers")
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for shard, dt in ex.map(gen_one, [arm] * len(todo), [raw_dir] * len(todo), todo):
            log(f"    shard {shard:3d} done in {dt/60:.1f} min")
    for s in shards:
        cache_one(arm, os.path.join(raw_dir, f"shard_{s:05d}.h5"),
                  os.path.join(cache_dir, f"shard_{s:05d}.h5"))
    if not os.path.exists(os.path.join(mm, "meta.json")):
        run([sys.executable, "-m", "pipeline.to_memmap", "--in-dir", cache_dir, "--out", mm], cwd=arm)
    n = json.load(open(os.path.join(mm, "meta.json")))["n_events"]
    log(f"  {split}: memmap ready, {n:,} events")
    return mm, n


def metrics_block(eval_dir):
    m = json.load(open(os.path.join(eval_dir, "metrics.json")))
    return {"n_events": m.get("n_events"),
            "completeness_at_purity": round(m["headline"]["completeness_at_fixed_purity"], 3),
            "purity": round(m["headline"]["purity_achieved"], 3),
            "threshold": m["headline"]["threshold"],
            "ap": round(m["average_precision_population"], 3),
            "per_class_f1": {k: round(v, 3) for k, v in m["argmax_population"]["f1"].items()}}


def run_arm(work, cad, epochs, workers, device):
    arm = setup_arm(work, cad)
    W = os.path.join(work, f"work{cad}")
    os.makedirs(W, exist_ok=True)
    log(f"=== arm {cad} min ===")
    mm_tr, n_tr = build_split(arm, W, "train", TRAIN_SHARDS, workers)
    mm_ev, n_ev = build_split(arm, W, "eval", EVAL_SHARDS, workers)
    ckpt = os.path.join(W, "model.pt")
    if not os.path.exists(ckpt + ".done"):
        for stale in (ckpt, ckpt + ".last"):
            if os.path.exists(stale):
                os.remove(stale)
        t0 = time.time()
        run([sys.executable, "-m", "pipeline.train", "--cache", mm_tr, "--out", ckpt,
             "--epochs", str(epochs), "--truncate-aug", "0.5", "--seed", str(SEED),
             "--device", device], cwd=arm)
        open(ckpt + ".done", "w").write(f"epochs={epochs} seed={SEED} truncate_aug=0.5 device={device}\n")
        log(f"  trained in {(time.time()-t0)/60:.1f} min")
    res = {"cadence_min": cad, "n_train_events": n_tr, "n_eval_events": n_ev, "epochs": epochs,
           "seed": SEED, "train_shards": TRAIN_SHARDS, "eval_shards": EVAL_SHARDS}
    for tag, mm in (("heldout", mm_ev), ("trainpool", mm_tr)):
        ev = os.path.join(W, f"eval_{tag}")
        if not os.path.exists(os.path.join(ev, "metrics.json")):
            run([sys.executable, "-m", "pipeline.evaluate", "--ckpt", ckpt, "--cache", mm,
                 "--out", ev, "--device", device], cwd=arm)
        res[tag] = metrics_block(ev)
    # the paper-facing fields, from the HELD-OUT evaluation
    res.update({k: res["heldout"][k] for k in ("completeness_at_purity", "purity", "ap", "per_class_f1")})
    res["n_events"] = n_tr        # surviving training events, the quantity the paper's sentence reports
    log(f"  held-out: AP {res['heldout']['ap']}  completeness {res['heldout']['completeness_at_purity']} "
        f"@ purity {res['heldout']['purity']}   | on training pool: AP {res['trainpool']['ap']}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", default=os.path.expanduser("~/Desktop/Research/microlensing/cadence_local_work"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--arms", default="15,12")
    ap.add_argument("--out", default=os.path.join(HERE, "cadence_local_result.json"))
    args = ap.parse_args(argv)
    os.makedirs(args.work, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    results = []
    for cad in [int(x) for x in args.arms.split(",")]:
        results.append(run_arm(args.work, cad, args.epochs, args.workers, args.device))
        json.dump({"_doc": __doc__.split("\n")[0], "code_commit": commit,
                   "protocol": "train on shards 0-11 (as modal_cadence.py); evaluate on shards "
                               "100-103 (disjoint seeds). 'trainpool' repeats the original, leaky "
                               "protocol for comparison only.",
                   "arms": results}, open(args.out, "w"), indent=2)
        log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

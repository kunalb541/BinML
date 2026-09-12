"""Finite-source single-lens fine-tune, run locally, scored on our own held-out set AND on GULLS.

WHY.  On RMDC26 (GULLS) the gap-aware checkpoint ft_g08e12 still flags 11.7% of single lenses, and
that residual is not noise: the false-alarm rate rises monotonically with rho/|u0| from 0.05 to
0.67 (paper/REVISION.md, 2026-09-09) while parallax has no effect.  The cause is a training-set
physics gap -- PSPLGen was point-source while NonPSPLGen sampled rho, so a rounded peak only ever
belonged to a binary.  This runner adds finite-source single lenses (priors.PSPL_FINITE_SOURCE via
the `fspl` regimes in run_shard.py; magnification from VBBinaryLensing ESPLMag since 2026-09-11, ESPLMag2
before -- rounds 1-3 were generated with ESPLMag2, and `--prefix fspl5s_legacy` regenerates round 3's pool
with it), warm-starts from
ft_g08e12 with the same gap augmentation, and measures three things:

  1. held-out clean performance on our own finite-source population (no regression?),
  2. PSPL recall as a function of rho/|u0| on that held-out set (did it learn the physics?),
  3. the GULLS full-population transfer, re-scored from the curve cache and reduced against
     ft_g08e12 on the identical 56,975 matched events (did it transfer?).

Training pool: 12 shards of regime `fspl` (natural mix, finite-source single lenses) + 4 shards of
`fspl_highmag` (U0_MAX = 0.2, PSPL-heavy, so finite-source single lenses and high-magnification
binaries are seen side by side -- the discrimination that actually matters).  Held-out: `fspl`
shards 100-103 and `fspl_highmag` shard 100 (disjoint seeds).  Shipped weights are untouched.

Usage:  python validation/fspl_finetune_local.py --workers 8
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SEED_GEN = 20260720            # run_shard default: seed = seed_base + shard*7919
SEED_TRAIN = 20260909
N_SHARDS_TOTAL = 200
def specs(prefix):
    """Regime prefix -> training / held-out (regime, shards). `fspl` and `fspl5` share the layout."""
    return ({prefix: list(range(12)), f"{prefix}_highmag": [0, 1, 2, 3]},
            {prefix: [100, 101, 102, 103], f"{prefix}_highmag": [100]})


TRAIN, HELDOUT = specs("fspl")
INIT = os.path.join(REPO, "validation", "gulls", "weights", "ft_g08e12.pt")
CACHE_CURVES = os.path.expanduser("~/Desktop/Research/microlensing/gulls_curve_cache")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, extra_env=None):
    env = dict(os.environ, PYTHONPATH=REPO)
    if extra_env:
        env.update(extra_env)
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


ONSET_RES = None      # set from --onset-resolution-days in main(); None = generator default (legacy 7.2 d)


def gen_one(raw_dir, regime, shard, onset_res=None):
    out = os.path.join(raw_dir, f"shard_{shard:05d}.h5")
    if os.path.exists(out):
        return shard, 0.0
    t0 = time.time()
    cmd = [sys.executable, "-m", "pipeline.run_shard", "--shard", str(shard), "--n-shards",
           str(N_SHARDS_TOTAL), "--out", raw_dir, "--regime", regime, "--seed-base", str(SEED_GEN)]
    if onset_res is not None:
        cmd += ["--onset-resolution-days", str(onset_res)]
    run(cmd, extra_env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    assert os.path.exists(out), out
    return shard, time.time() - t0


def build(work, name, spec, workers):
    """Generate every (regime, shard) in `spec`, cache, memmap. Resumable."""
    cache_dir = os.path.join(work, f"cache_{name}")
    mm = os.path.join(work, f"mm_{name}")
    os.makedirs(cache_dir, exist_ok=True)
    jobs = []
    for regime, shards in spec.items():
        raw_dir = os.path.join(work, f"raw_{regime}")
        os.makedirs(raw_dir, exist_ok=True)
        jobs += [(raw_dir, regime, s) for s in shards]
    # a shard needs generating only if neither its raw file nor its cached file exists (--delete-raw removes
    # the raw files of a finished pool; reusing that pool for another arm must not regenerate them)
    todo = [j for j in jobs if not os.path.exists(os.path.join(j[0], f"shard_{j[2]:05d}.h5"))
            and not os.path.exists(os.path.join(cache_dir, f"{j[1]}_{j[2]:05d}.h5"))]
    log(f"  {name}: {len(jobs) - len(todo)} shards cached, generating {len(todo)} with {workers} workers")
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for (shard, dt), j in zip(ex.map(gen_one, *zip(*todo), [ONSET_RES] * len(todo)) if todo else [], todo):
            log(f"    {j[1]} shard {shard:3d} done in {dt/60:.1f} min")
    for raw_dir, regime, s in jobs:
        raw = os.path.join(raw_dir, f"shard_{s:05d}.h5")
        out = os.path.join(cache_dir, f"{regime}_{s:05d}.h5")
        if not os.path.exists(out):
            run([sys.executable, "-c",
                 "import sys; from pipeline.cache import build_cache; build_cache([sys.argv[1]], sys.argv[2])",
                 raw, out])
    if not os.path.exists(os.path.join(mm, "meta.json")):
        run([sys.executable, "-m", "pipeline.to_memmap", "--in-dir", cache_dir, "--out", mm])
    n = json.load(open(os.path.join(mm, "meta.json")))["n_events"]
    log(f"  {name}: memmap ready, {n:,} events")
    return mm, n


def pspl_recall_by_rho_u0(eval_dir):
    """PSPL recall on labelled-PSPL events, binned by rho/|u0| -- the GULLS diagnostic on our data."""
    lab = np.load(os.path.join(eval_dir, "label.npy")).astype(int)
    tc = np.load(os.path.join(eval_dir, "true_class.npy")).astype(int)
    lg = np.load(os.path.join(eval_dir, "logits.npy"))
    ti = np.load(os.path.join(eval_dir, "test_idx.npy")).astype(int)
    params = np.load(os.path.join(eval_dir, "params.npy"))
    pf = json.load(open(os.path.join(eval_dir, "meta.json")))["param_fields"] \
        if os.path.exists(os.path.join(eval_dir, "meta.json")) else None
    if pf is None:
        from pipeline.writer import PARAM_FIELDS as pf
    from pipeline.classes import CLASS_NAMES
    I_PSPL, I_NON = CLASS_NAMES.index("PSPL"), CLASS_NAMES.index("NonPSPL")
    m = np.zeros(len(lab), bool); m[ti] = True
    sel = m & (lab == I_PSPL) & (tc == I_PSPL)
    rho = params[:, pf.index("rho")]; u0 = np.abs(params[:, pf.index("u0")])
    ratio = rho / np.maximum(u0, 1e-6)
    pred = lg.argmax(1)
    out = {}
    for lo, hi in [(0, 0.03), (0.03, 0.1), (0.1, 0.3), (0.3, 1), (1, 3), (3, 1e9)]:
        s = sel & (ratio >= lo) & (ratio < hi) & np.isfinite(ratio)
        if s.sum() >= 30:
            out[f"[{lo},{hi if hi < 1e9 else 'inf'})"] = {
                "n": int(s.sum()), "pspl_recall": round(float((pred[s] == I_PSPL).mean()), 4),
                "called_nonpspl": round(float((pred[s] == I_NON).mean()), 4)}
    nofs = m & (lab == I_PSPL) & (tc == I_PSPL) & ~np.isfinite(rho)
    if nofs.sum() >= 30:
        out["point_source"] = {"n": int(nofs.sum()), "pspl_recall": round(float((pred[nofs] == I_PSPL).mean()), 4)}
    return out


def evaluate(ckpt, mm, out_dir, device):
    if not os.path.exists(os.path.join(out_dir, "metrics.json")):
        run([sys.executable, "-m", "pipeline.evaluate", "--ckpt", ckpt, "--cache", mm,
             "--out", out_dir, "--device", device])
    m = json.load(open(os.path.join(out_dir, "metrics.json")))
    return {"n_events": m.get("n_events"),
            "completeness_at_purity": round(m["headline"]["completeness_at_fixed_purity"], 3),
            "purity": round(m["headline"]["purity_achieved"], 3),
            "ap": round(m["average_precision_population"], 3),
            "macro_f1": round(m["argmax_population"]["macro_f1"], 4),
            "per_class_f1": {k: round(v, 3) for k, v in m["argmax_population"]["f1"].items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", default=os.path.expanduser("~/Desktop/Research/microlensing/fspl_local_work"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gap-aug", type=float, default=0.8)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tag", default="fspl_g08")
    ap.add_argument("--prefix", default="fspl", help="regime prefix: fspl (rho<=1) or fspl5 (rho<=5, binary rho<=0.1)")
    ap.add_argument("--init", default=INIT, help="warm-start checkpoint")
    ap.add_argument("--skip-gulls", action="store_true")
    ap.add_argument("--onset-resolution-days", type=float, default=None,
                    help="passed to pipeline.run_shard (default: the generator's legacy 7.2 d grid)")
    ap.add_argument("--heldout-mm", default=None,
                    help="evaluate on this existing held-out memmap instead of generating one (e.g. a control "
                         "arm trained on point sources, evaluated on the finite-source held-out of the arm it controls)")
    ap.add_argument("--delete-raw", action="store_true", help="delete raw shards once cached (they are regenerable)")
    ap.add_argument("--seed", type=int, default=SEED_TRAIN, help="training seed (seed replicates of a recipe; default 20260909)")
    ap.add_argument("--train-extra", default="", help="extra pipeline.train arguments, e.g. "
                    "'--gap-schedule rmdc26_seasons --gap-relabel-anomaly off' (recorded in the recipe stamp)")
    args = ap.parse_args(argv)
    global TRAIN, HELDOUT, ONSET_RES
    TRAIN, HELDOUT = specs(args.prefix)
    ONSET_RES = args.onset_resolution_days
    if args.work == os.path.expanduser("~/Desktop/Research/microlensing/fspl_local_work") and args.prefix != "fspl":
        args.work = os.path.expanduser(f"~/Desktop/Research/microlensing/{args.prefix}_local_work")
    W = args.work; os.makedirs(W, exist_ok=True)
    # provenance at DATA-GENERATION time, with the dirty flag (2026-09-11 verification: artifacts recorded
    # the commit of the last invocation, not the code that generated their data)
    commit = subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=12"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    prov = os.path.join(W, "provenance.json")
    if not os.path.exists(prov):
        espl = ("none: point-source single lenses" if args.prefix.startswith("pspl")
                else "ESPLMag2 (legacy regimes)" if "legacy" in args.prefix
                else "ESPLMag (smooth; pools generated before 2026-09-11 used ESPLMag2)")
        json.dump({"code_at_generation": commit, "onset_resolution_days": ONSET_RES, "prefix": args.prefix,
                   "espl_function": espl}, open(prov, "w"), indent=1)

    log("=== data ===")
    mm_tr, n_tr = build(W, "train", TRAIN, args.workers)
    if args.heldout_mm:
        mm_ev = args.heldout_mm; n_ev = json.load(open(os.path.join(mm_ev, "meta.json")))["n_events"]
        log(f"  heldout: using existing memmap {mm_ev} ({n_ev:,} events)")
    else:
        mm_ev, n_ev = build(W, "heldout", HELDOUT, args.workers)
    if args.delete_raw:
        import shutil
        for d in os.listdir(W):
            if d.startswith("raw_"):
                shutil.rmtree(os.path.join(W, d), ignore_errors=True)

    log("=== fine-tune from ft_g08e12 ===")
    ckpt = os.path.join(W, f"{args.tag}.pt")
    stamp = (f"init={os.path.basename(args.init)} prefix={args.prefix} epochs={args.epochs} lr={args.lr} gap_aug={args.gap_aug} "
             f"truncate_aug=0.5 seed={args.seed} onset_res={ONSET_RES} data={json.load(open(prov))['code_at_generation']}"
             + (f" extra={args.train_extra}" if args.train_extra else ""))
    if not (os.path.exists(ckpt + ".done") and open(ckpt + ".done").read().strip() == stamp):
        # a retrain invalidates everything computed from the old checkpoint: its held-out evaluation and its
        # RMDC26 rows/summaries would otherwise be reused and reported for the new weights
        import shutil
        for stale in (ckpt, ckpt + ".last", ckpt + ".done",
                      os.path.join(CACHE_CURVES, f"rows_full_{args.tag}.json"),
                      os.path.join(HERE, "gulls", f"transfer_full_{args.tag}.json"),
                      os.path.join(HERE, "gulls", f"transfer_full_reduced_{args.tag}.json"),
                      os.path.join(HERE, "gulls", f"transfer_full_reduced_{args.tag}_vs_fspl_g08.json")):
            if os.path.exists(stale):
                os.remove(stale)
        shutil.rmtree(os.path.join(W, f"eval_{args.tag}"), ignore_errors=True)
        t0 = time.time()
        run([sys.executable, "-m", "pipeline.train", "--cache", mm_tr, "--out", ckpt,
             "--init-weights", args.init, "--epochs", str(args.epochs), "--lr", str(args.lr),
             "--truncate-aug", "0.5", "--gap-aug", str(args.gap_aug),
             "--seed", str(args.seed), "--device", args.device] + (args.train_extra.split() if args.train_extra else []))
        open(ckpt + ".done", "w").write(stamp + "\n")
        log(f"  trained in {(time.time()-t0)/60:.1f} min")

    log("=== held-out evaluation: new checkpoint vs ft_g08e12 vs shipped ===")
    res = {"_doc": __doc__.split("\n")[0], "code_commit": commit, "provenance": json.load(open(prov)),
           "train": {"n_events": n_tr, "spec": TRAIN},
           "heldout": {"n_events": n_ev, "spec": HELDOUT if not args.heldout_mm else f"existing memmap {args.heldout_mm}"},
           "recipe": stamp, "models": {}}
    for name, path in ((args.tag, ckpt), ("ft_g08e12", INIT),
                       ("shipped", os.path.join(REPO, "binml", "weights", "binml.pt"))):
        ev = os.path.join(W, f"eval_{name}")
        res["models"][name] = evaluate(path, mm_ev, ev, args.device)
        res["models"][name]["pspl_recall_by_rho_u0"] = pspl_recall_by_rho_u0(ev)
        log(f"  {name:10s} macroF1 {res['models'][name]['macro_f1']}  AP {res['models'][name]['ap']}  "
            f"PSPL recall by rho/u0: " + ", ".join(f"{k}:{v['pspl_recall']}" for k, v in res["models"][name]["pspl_recall_by_rho_u0"].items()))
    out = os.path.join(HERE, "gulls", f"fspl_finetune_{args.tag}.json")
    json.dump(res, open(out, "w"), indent=2); log(f"wrote {out}")

    if not args.skip_gulls:
        log("=== GULLS full population from the curve cache ===")
        rows = os.path.join(CACHE_CURVES, f"rows_full_{args.tag}.json")
        summ = os.path.join(HERE, "gulls", f"transfer_full_{args.tag}.json")
        if not os.path.exists(summ):
            run([sys.executable, "validation/gulls_transfer.py", "--per-class", "200000", "--chunk", "250",
                 "--curve-cache", CACHE_CURVES, "--weights", ckpt, "--out", summ, "--rows-out", rows])
        red = os.path.join(HERE, "gulls", f"transfer_full_reduced_{args.tag}.json")
        run([sys.executable, "validation/gulls/transfer_reduce.py",
             "--rows-a", os.path.join(CACHE_CURVES, "rows_full_ft_g08e12_v2.json"), "--rows-b", rows,
             "--label-a", "ft_g08e12", "--label-b", args.tag, "--out", red])
        log(f"wrote {red}")
        prev = os.path.join(CACHE_CURVES, "rows_full_fspl_g08.json")
        if args.tag != "fspl_g08" and os.path.exists(prev):
            red2 = os.path.join(HERE, "gulls", f"transfer_full_reduced_{args.tag}_vs_fspl_g08.json")
            run([sys.executable, "validation/gulls/transfer_reduce.py", "--rows-a", prev, "--rows-b", rows,
                 "--label-a", "fspl_g08", "--label-b", args.tag, "--out", red2])
            log(f"wrote {red2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Schedule-matched vs distribution-matched gap augmentation, on ONE training pool (local, M5).

Question. g08e12 was fine-tuned with `--gap-aug 0.8`: 1-8 random contiguous blanks of 1-12 h per
event. Roman's planned schedule is not random: seven ~6.2 h pauses at fixed season phases plus a
70.7-day season inside BinML's 72-day window (RMDC26/GULLS). Does training on the EXACT schedule
(`--gap-schedule rmdc26`) beat training on a distribution over gaps, on the schedule itself and on
GULLS? The 12.1-min cadence of the real schedule does not survive preprocessing (binml.preprocess
pools to one value per 15-min Roman epoch, so frac is ~1 either way) and is not an arm.

Design. Two arms, everything identical except the augmentation:
  rand   : warm-start binml.pt, 12 epochs, lr 1e-4, --truncate-aug 0.5, --gap-aug 0.8
  sched  : same, plus --gap-schedule rmdc26
Pool: the cadence rerun's 15-min natural-prior shards 0-11 (89,919 events, seed 20260720) at
~/Desktop/Research/microlensing/cadence_local_work/work15/mm_train; held-out: shards 100-103
(30,013 events). NOTE this pool differs from g08e12's (6 shards, seed 20260823, Modal), so the
clean contrast is rand vs sched; g08e12 and the shipped weights are reference rows only.

Evaluation. (1) held-out clean; (2) held-out with the exact schedule blanked in (pauses + season
end, the same mask the sched arm trained on -- the rand arm never saw this mask); (3) GULLS from the
curve cache (56,975 matched events), summarised with gulls_summary_tables.py. Writes
validation/gulls/schedule_finetune.json.

Usage:  python validation/schedule_finetune_local.py [--epochs 12] [--device mps]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(HERE, "gulls"))
from pipeline.train import rmdc26_schedule_mask  # noqa: E402
from calibrate_gapped_threshold import make_gapped  # noqa: E402

R = os.path.expanduser("~/Desktop/Research/microlensing")
WORK = f"{R}/schedule_local_work"
POOL = f"{R}/cadence_local_work/work15/mm_train"
HELD = f"{R}/cadence_local_work/work15/mm_eval"
CURVES = f"{R}/gulls_curve_cache"
# sched_norelabel: the schedule WITHOUT the caustic-in-gap relabel. With a fixed mask the 7.2-d-quantised
# t_anom (two of ten grid values inside the mask) makes that relabel fire on 20% of ALL binaries on every
# presentation -- the sched arm's NonPSPL recall 0.80 vs rand 0.91 under the schedule is that label error,
# not the augmentation. Turning the relabel off costs ~4% genuine label noise (binaries whose anomaly
# really is inside a 37-bin mask) instead of 20% systematic error.
ARMS = {"rand": [], "sched": ["--gap-schedule", "rmdc26"],
        "sched_norelabel": ["--gap-schedule", "rmdc26", "--gap-relabel-anomaly", "off"]}
COMMON = ["--init-weights", "binml/weights/binml.pt", "--epochs", "12", "--lr", "1e-4",
          "--truncate-aug", "0.5", "--gap-aug", "0.8", "--seed", "20260823"]
REF = {"shipped": "binml/weights/binml.pt", "ft_g08e12": "validation/gulls/weights/ft_g08e12.pt"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(cmd):
    log(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO, env=dict(os.environ, PYTHONPATH=REPO))


def metrics(ev):
    m = json.load(open(os.path.join(ev, "metrics.json")))
    h = m["headline"]
    return {"completeness_at_purity": round(h["completeness_at_fixed_purity"], 4), "purity": round(h["purity_achieved"], 4),
            "threshold": round(h["threshold"], 4), "ap": round(m["average_precision_population"], 4),
            "argmax_population": m["argmax_population"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--epochs", default="12"); ap.add_argument("--device", default="mps")
    ap.add_argument("--skip-gulls", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(WORK, exist_ok=True)
    common = list(COMMON); common[common.index("--epochs") + 1] = args.epochs
    ckpts = dict(REF)
    for arm, extra in ARMS.items():
        ck = f"{WORK}/ft_{arm}.pt"
        stamp = " ".join(common + extra)
        if os.path.exists(ck + ".done"):
            assert open(ck + ".done").read().strip() == stamp, f"{ck} trained with different settings"
        else:
            t0 = time.time()
            run([sys.executable, "-m", "pipeline.train", "--cache", POOL, "--out", ck, "--device", args.device,
                 "--resume"] + common + extra)
            open(ck + ".done", "w").write(stamp + "\n")
            log(f"{arm}: trained in {(time.time() - t0) / 60:.1f} min")
        ckpts[arm] = ck
    gapped = f"{WORK}/mm_heldout_sched"
    make_gapped(HELD, gapped, mask=rmdc26_schedule_mask(), note="RMDC26 seven pauses + 70.7 d season end")
    res = {"_doc": __doc__.split("\n")[0], "pool": POOL, "heldout": HELD, "train_args": {a: common + e for a, e in ARMS.items()},
           "note": "rand vs sched is the controlled contrast; shipped/ft_g08e12 are reference rows trained on other pools",
           "heldout_eval": {}}
    for name, ck in ckpts.items():
        res["heldout_eval"][name] = {}
        for tag, mm in (("clean", HELD), ("rmdc26_schedule", gapped)):
            ev = f"{WORK}/eval_{name}_{tag}"
            if not os.path.exists(os.path.join(ev, "metrics.json")):
                run([sys.executable, "-m", "pipeline.evaluate", "--ckpt", ck, "--cache", mm, "--out", ev, "--device", "cpu"])
            res["heldout_eval"][name][tag] = metrics(ev)
        log(f"{name}: clean AP {res['heldout_eval'][name]['clean']['ap']}  schedule AP {res['heldout_eval'][name]['rmdc26_schedule']['ap']}")
    if not args.skip_gulls:
        models = ["shipped=rows_full_shipped_v2.json", "ft_g08e12=rows_full_ft_g08e12_v2.json", "fspl5s_g08=rows_full_fspl5s_g08.json"]
        for arm in ARMS:
            rows = f"{CURVES}/rows_full_sched_{arm}.json"; summ = f"{HERE}/gulls/transfer_full_sched_{arm}.json"
            if not os.path.exists(summ):
                run([sys.executable, "validation/gulls_transfer.py", "--per-class", "200000", "--chunk", "250",
                     "--curve-cache", CURVES, "--weights", ckpts[arm], "--out", summ, "--rows-out", rows])
            models.append(f"sched_{arm}={rows}")
        run([sys.executable, "validation/gulls/gulls_summary_tables.py", "--models"] + models + ["--out-dir", WORK])
        res["gulls"] = json.load(open(f"{WORK}/transfer_tradeoff_all.json"))
    out = f"{HERE}/gulls/schedule_finetune.json"
    json.dump(res, open(out, "w"), indent=1)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

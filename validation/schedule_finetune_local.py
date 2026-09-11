"""Schedule-matched vs distribution-matched gap augmentation, on ONE training pool (local, M5).

Question. g08e12 was fine-tuned with `--gap-aug 0.8`: 1-8 random contiguous blanks of 1-12 h per
event. RMDC26 (GULLS) implements the survey with ~43-44 h of F146 pauses per 70.7-day season, in six
or seven pauses whose phases DIFFER between the six high-cadence seasons (validation/gulls/
rmdc26_schedule.json), and with colour visits that leave one of eight F146 epochs empty in about a
third of the bins. Does training on the measured schedule beat training on a distribution over gaps?

History, because the first answer was wrong. The 2026-09-10 arms used the pauses of the FIRST season
only, described then as the schedule "at fixed season phases". That mask matches one season in six
(16% of scored RMDC26 events), so their GULLS result (the 'exact schedule' arm no better than random
gaps) was confounded: inside the first season the mask-matched arm WAS better. They are kept, renamed
in the text as 'first-season mask', and two arms were added on 2026-09-11 so each contrast changes
one factor:

  rand             --gap-aug 0.8 (random 1-12 h runs), caustic-in-gap relabel on   [2026-09-10]
  sched            first-season mask, relabel on                                     [2026-09-10]
  sched_norelabel  first-season mask, relabel off                                   [2026-09-10]
  rand_norelabel   random runs, relabel off                 (control for the relabel) [2026-09-11]
  sched_seasons    measured per-season templates, relabel off                        [2026-09-11]

The relabel is off for the new schedule arm because the rule it applies (PSPL if the bins around the
recorded onset t_anom are blanked) is not truth-based: t_anom is when the anomaly first becomes
detectable, not where the caustic is (audit finding 9, still open).

Common recipe: warm-start binml.pt, 12 epochs, lr 1e-4, --truncate-aug 0.5, --gap-aug 0.8, seed
20260823, one seed per arm. Pool: the cadence rerun's 15-min natural-prior shards 0-11 (89,919
events, seed 20260720) at cadence_local_work/work15/mm_train; held-out: shards 100-103 (30,013 events;
pipeline.evaluate scores 24,011 of them and uses the other 20% to choose its operating point).
g08e12 (Modal pool) and the shipped weights are reference rows only.

Evaluation. Held-out (1) clean, (2) with the first-season mask + season end, (3) with the measured
per-season templates (event i -> season i mod 6); labels are the unblanked ones. GULLS from the curve
cache (56,975 matched events), overall and by season, via gulls_summary_tables.py --by-season.
Writes validation/gulls/schedule_finetune.json.

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
from calibrate_gapped_threshold import make_gapped, make_gapped_seasons  # noqa: E402

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
        "sched_norelabel": ["--gap-schedule", "rmdc26", "--gap-relabel-anomaly", "off"],
        "rand_norelabel": ["--gap-relabel-anomaly", "off"],
        "sched_seasons": ["--gap-schedule", "rmdc26_seasons", "--gap-relabel-anomaly", "off"]}
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
    import numpy as np
    n_test = int(np.load(os.path.join(ev, "test_idx.npy")).shape[0]) if os.path.exists(os.path.join(ev, "test_idx.npy")) else None
    return {"completeness_at_purity": round(h["completeness_at_fixed_purity"], 4), "purity": round(h["purity_achieved"], 4),
            "threshold": round(h["threshold"], 4), "ap": round(m["average_precision_population"], 4),
            "n_pool": m.get("n_events"), "n_scored_test": n_test, "checkpoint_sha256": m.get("checkpoint_sha256"),
            "argmax_population": m["argmax_population"]}


def git_describe():
    try:
        return subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=12"], cwd=REPO, capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception as exc:                      # provenance must never kill a finished run
        return f"unavailable: {exc}"


def relabel_exposure(pool):
    """Fraction of NonPSPL-labelled pool events whose recorded onset window lies inside the legacy mask."""
    import numpy as np
    from pipeline.classes import CLASS_NAMES
    meta = json.load(open(os.path.join(pool, "meta.json"))); pf = meta["param_fields"]
    lab = np.load(os.path.join(pool, "label.npy")); par = np.load(os.path.join(pool, "params.npy"), mmap_mode="r")
    non = lab == CLASS_NAMES.index("NonPSPL"); ta = np.asarray(par[:, pf.index("t_anom")], float)[non]
    m = rmdc26_schedule_mask(864); fin = np.isfinite(ta)
    b = np.clip((ta[fin] / 72.0 * 864).astype(int), 0, 863)
    hit = np.array([m[max(0, k - 1):min(864, k + 2)].all() for k in b])
    tc = np.load(os.path.join(pool, "true_class.npy"))
    return {"n_nonpspl_labelled": int(non.sum()), "n_true_class_binaries": int((tc == CLASS_NAMES.index("NonPSPL")).sum()),
            "n_eligible_for_legacy_relabel": int(hit.sum()),
            "frac_of_nonpspl_labelled": round(float(hit.sum() / non.sum()), 4),
            "t_anom_values_inside_mask": sorted({float(x) for x in np.round(ta[fin][hit], 2)}),
            "note": ("eligible = all three bins around the recorded onset are blanked by the first-season mask; the relabel "
                     "fires only on presentations that draw the gap augmentation (p=0.8) and still carry NonPSPL after "
                     "the truncation step (p=0.5), so the per-presentation rate is lower than this fraction")}


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
    make_gapped(HELD, gapped, mask=rmdc26_schedule_mask(), note="RMDC26 first-season pauses + 70.7 d season end")
    seasons_mm = f"{WORK}/mm_heldout_seasons"
    make_gapped_seasons(HELD, seasons_mm)
    res = {"_doc": __doc__.split("\n")[0], "code": git_describe(), "pool": POOL, "heldout": HELD,
           "train_args": {a: common + e for a, e in ARMS.items()},
           "note": ("single-factor contrasts: rand vs rand_norelabel (relabel), rand_norelabel vs sched_norelabel (first-season "
                    "mask), rand_norelabel vs sched_seasons (measured seasons); shipped/ft_g08e12 are reference rows trained on "
                    "other pools. rmdc26_schedule = first-season mask + season end; rmdc26_seasons = measured per-season templates."),
           "legacy_relabel_exposure": relabel_exposure(POOL), "heldout_eval": {}}
    for name, ck in ckpts.items():
        res["heldout_eval"][name] = {}
        for tag, mm in (("clean", HELD), ("rmdc26_schedule", gapped), ("rmdc26_seasons", seasons_mm)):
            ev = f"{WORK}/eval_{name}_{tag}"
            if not os.path.exists(os.path.join(ev, "metrics.json")):
                run([sys.executable, "-m", "pipeline.evaluate", "--ckpt", ck, "--cache", mm, "--out", ev, "--device", "cpu"])
            res["heldout_eval"][name][tag] = metrics(ev)
        e = res["heldout_eval"][name]
        log(f"{name}: clean AP {e['clean']['ap']}  first-season mask AP {e['rmdc26_schedule']['ap']}  seasons AP {e['rmdc26_seasons']['ap']}")
    if not args.skip_gulls:
        models = ["shipped=rows_full_shipped_v2.json", "ft_g08e12=rows_full_ft_g08e12_v2.json", "fspl5s_g08=rows_full_fspl5s_g08.json"]
        for arm in ARMS:
            rows = f"{CURVES}/rows_full_sched_{arm}.json"; summ = f"{HERE}/gulls/transfer_full_sched_{arm}.json"
            if not os.path.exists(summ):
                run([sys.executable, "validation/gulls_transfer.py", "--per-class", "200000", "--chunk", "250",
                     "--curve-cache", CURVES, "--weights", ckpts[arm], "--out", summ, "--rows-out", rows])
            models.append(f"sched_{arm}={rows}")
        run([sys.executable, "validation/gulls/gulls_summary_tables.py", "--models"] + models + ["--out-dir", WORK, "--by-season"])
        res["gulls"] = json.load(open(f"{WORK}/transfer_tradeoff_all.json"))
    out = f"{HERE}/gulls/schedule_finetune.json"
    json.dump(res, open(out, "w"), indent=1)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

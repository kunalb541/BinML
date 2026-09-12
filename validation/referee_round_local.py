#!/usr/bin/env python3
"""Referee-round items that need compute, run locally on our own simulator -> validation/referee_round.json.

The pre-submission referee round (paper/REVISION.md section 2) deferred four items. Three are run here, on our
simulator only, with the SHIPPED checkpoint unless stated:

1. Detectability-floor sensitivity by RE-SIMULATION. Test shards 90-91 (disjoint from training) regenerated with
   the label floor at 0.01, 0.02 (adopted) and 0.05 mag (the same seeds, so the same underlying events): the
   NonPSPL prevalence (raw and population-weighted) and the headline completeness at 90% purity, AP and macro-F1
   by the paper's own procedure (pipeline.evaluate). The model is not retrained; this is the label side plus the
   shipped model's response.
2. Colour-band calibration ablation. The same test shards with the audited F087/F213 zeropoints, backgrounds and
   saturation (pipeline.photometry.ROMAN_BANDS_COLOUR_AUDITED; F146 and cadences unchanged): the shipped model on
   both photometries of the same events; then (--finetune) a short fine-tune of the shipped weights on training
   shards 0-1 generated with each calibration, each evaluated on both test photometries.
3. Mixed-class sequential evaluation. Every event of the adopted-floor test shards (all six classes) revealed in
   144 half-day prefixes and scored at the frozen threshold: the share of each class that raises an alert, alerts
   per 1,000 events per day, streaming purity at the simulated population mix (keep_prob weights, as the paper's
   prevalence), and, for NonPSPL events with a finite onset, premature alerts and lag against the 0.5-day
   first-detectable onset recorded at generation.

The fourth item, a three-seed sweep of the shipped model's final training stage, needs the 1.9M-event training
set, which is on S3 and not on this machine; it is not run here.

Usage:  python validation/referee_round_local.py [--workers 6] [--finetune]
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
sys.path.insert(0, REPO)
WORK = os.path.expanduser("~/Desktop/Research/microlensing/referee_local_work")
SHIPPED = os.path.join(REPO, "binml", "weights", "binml.pt")
FROZEN = 0.9042405486106873
SEED_GEN = 20260720
TEST, TRAIN = [90, 91], [0, 1]
ARMS = {                                  # name: (shards, extra run_shard args)
    "test_f001": (TEST, ["--min-amplitude-mag", "0.01"]),
    "test_f002": (TEST, ["--onset-resolution-days", "0.5"]),      # adopted floor; fine onset for the stream test
    "test_f005": (TEST, ["--min-amplitude-mag", "0.05"]),
    "test_colour": (TEST, ["--band-set", "colour_audited"]),
    "train_trained": (TRAIN, []),
    "train_colour": (TRAIN, ["--band-set", "colour_audited"]),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, env=None):
    e = dict(os.environ, PYTHONPATH=REPO, **(env or {}))
    subprocess.run(cmd, cwd=REPO, check=True, env=e)


def gen(arm, shard):
    d = os.path.join(WORK, f"raw_{arm}"); os.makedirs(d, exist_ok=True)
    out = os.path.join(d, f"shard_{shard:05d}.h5")
    if not os.path.exists(out):
        t0 = time.time()
        run([sys.executable, "-m", "pipeline.run_shard", "--shard", str(shard), "--n-shards", "400", "--out", d,
             "--seed-base", str(SEED_GEN)] + ARMS[arm][1], env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        log(f"  generated {arm} shard {shard} in {(time.time() - t0) / 60:.1f} min")
    return out


def memmap(arm):
    mm = os.path.join(WORK, f"mm_{arm}")
    if os.path.exists(os.path.join(mm, "meta.json")):
        return mm
    cdir = os.path.join(WORK, f"cache_{arm}"); os.makedirs(cdir, exist_ok=True)
    for s in ARMS[arm][0]:
        c = os.path.join(cdir, f"{arm}_{s:05d}.h5")
        if not os.path.exists(c):
            run([sys.executable, "-c", "import sys; from pipeline.cache import build_cache; build_cache([sys.argv[1]], sys.argv[2])",
                 os.path.join(WORK, f"raw_{arm}", f"shard_{s:05d}.h5"), c])
    run([sys.executable, "-m", "pipeline.to_memmap", "--in-dir", cdir, "--out", mm])
    return mm


def evaluate(ckpt, mm, name):
    ev = os.path.join(WORK, f"eval_{name}")
    if not os.path.exists(os.path.join(ev, "metrics.json")):
        run([sys.executable, "-m", "pipeline.evaluate", "--ckpt", ckpt, "--cache", mm, "--out", ev, "--device", "cpu"])
    m = json.load(open(os.path.join(ev, "metrics.json")))
    lab = np.load(os.path.join(mm, "label.npy")); kp = np.load(os.path.join(mm, "keep_prob.npy")).astype(np.float64)
    w = 1.0 / np.clip(kp, 1e-3, 1.0)
    m["_prevalence_nonpspl"] = {"raw": float((lab == 2).mean()), "population_weighted": float(w[lab == 2].sum() / w.sum()), "n": int(lab.size)}
    return m


# ------------------------------------------------------------------ mixed-class sequential scan
_C = {}


def _init():
    import torch
    torch.set_num_threads(1)
    import binml
    _C["clf"] = binml.Classifier(weights=SHIPPED, device="cpu")


def _scan_chunk(job):
    """(shard file, index range) -> per-event P(NonPSPL) at 144 half-day cuts (NaN where no F146 yet)."""
    import h5py
    from binml.preprocess import BAND_BINS, to_tokens
    path, lo, hi = job
    clf = _C["clf"]; inon = clf.class_names.index("NonPSPL")
    cuts = np.arange(1, 145) * 0.5
    out = []
    with h5py.File(path, "r") as f:
        times = {b: np.asarray(f[f"time/{b}"][:], float) for b in BAND_BINS if f"time/{b}" in f}
        mags = {b: f[f"mag/{b}"][lo:hi] for b in times}
        mb = f["m_base_ref"][lo:hi]
        for i in range(hi - lo):
            full = {b: (times[b][np.isfinite(mags[b][i])], mags[b][i][np.isfinite(mags[b][i])]) for b in times}
            feats = {b: [] for b in BAND_BINS}; fracs = {b: [] for b in BAND_BINS}; valid = []
            for c in cuts:
                rev = {b: (t[t <= c], m[t <= c]) for b, (t, m) in full.items()}
                if rev["F146"][0].size < 10:
                    valid.append(False); continue
                tok = to_tokens(rev, m_base_ref=float(mb[i]), t_start=0.0)
                for b in BAND_BINS:
                    feats[b].append(tok.feat[b]); fracs[b].append(tok.frac[b])
                valid.append(True)
            p = np.full(144, np.nan, np.float32)
            if any(valid):
                p[np.array(valid)] = clf._forward({b: np.stack(feats[b]) for b in BAND_BINS}, {b: np.stack(fracs[b]) for b in BAND_BINS})[:, inon]
            out.append(p)
    return lo, np.stack(out)


def stream_scan(workers):
    import h5py
    res = {}
    for s in ARMS["test_f002"][0]:
        path = os.path.join(WORK, "raw_test_f002", f"shard_{s:05d}.h5")
        cache = os.path.join(WORK, f"stream_scan_{s:05d}.npz")
        if not os.path.exists(cache):
            with h5py.File(path, "r") as f:
                n = int(f.attrs["n_events"])
            jobs = [(path, lo, min(lo + 100, n)) for lo in range(0, n, 100)]
            P = np.full((n, 144), np.nan, np.float32); t0 = time.time()
            with cf.ProcessPoolExecutor(max_workers=workers, initializer=_init) as ex:
                for k, (lo, p) in enumerate(ex.map(_scan_chunk, jobs)):
                    P[lo:lo + p.shape[0]] = p
                    if k % 10 == 0:
                        log(f"  stream scan shard {s}: {lo + p.shape[0]}/{n} ({time.time() - t0:.0f}s)")
            np.savez_compressed(cache, p=P)
        res[s] = (path, np.load(cache)["p"])
    return res


def stream_reduce(scans):
    import h5py
    from pipeline.classes import CLASS_NAMES
    P, lab, kp, tan = [], [], [], []
    for s, (path, p) in scans.items():
        with h5py.File(path, "r") as f:
            pf = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["param_fields"]]
            P.append(p); lab.append(f["label"][:]); kp.append(f["keep_prob"][:]); tan.append(f["params"][:, pf.index("t_anom")])
    P, lab, kp, tan = np.concatenate(P), np.concatenate(lab), np.concatenate(kp).astype(np.float64), np.concatenate(tan)
    w = 1.0 / np.clip(kp, 1e-3, 1.0)
    cuts = np.arange(1, 145) * 0.5
    alert = np.nan_to_num(P, nan=-1.0) >= FROZEN
    fired = alert.any(1); first = np.where(fired, cuts[np.argmax(alert, 1)], np.nan)
    out = {"n_events": int(lab.size), "threshold": FROZEN, "by_class": {}}
    for k, name in enumerate(CLASS_NAMES):
        sel = lab == k
        if sel.any():
            out["by_class"][name] = {"n": int(sel.sum()), "alert_frac_per_season": float(fired[sel].mean()),
                                     "alert_frac_population_weighted": float(w[sel & fired].sum() / w[sel].sum())}
    wa = w[fired]
    out["alerts_per_1000_events_per_day"] = {"raw": float(1000 * fired.mean() / 72.0),
                                             "population_weighted": float(1000 * wa.sum() / w.sum() / 72.0)}
    out["streaming_purity_nonpspl"] = {"raw": float((lab[fired] == 2).mean()) if fired.any() else None,
                                       "population_weighted": float(w[fired & (lab == 2)].sum() / wa.sum()) if fired.any() else None}
    out["alert_share_by_class_population_weighted"] = {CLASS_NAMES[k]: float(w[fired & (lab == k)].sum() / wa.sum()) for k in range(len(CLASS_NAMES)) if (lab == k).any()}
    el = (lab == 2) & np.isfinite(tan)
    prem = el & fired & (first < tan); lag = (first - tan)[el & fired & ~prem]
    k_, n_ = int(prem.sum()), int(el.sum())
    z = 1.96; ph = k_ / max(n_, 1); d = 1 + z * z / max(n_, 1); c = (ph + z * z / (2 * max(n_, 1))) / d
    h = z * np.sqrt(ph * (1 - ph) / max(n_, 1) + z * z / (4 * max(n_, 1) ** 2)) / d
    out["timing_nonpspl"] = {"n_eligible": n_, "detected_frac": float(fired[el].mean()) if n_ else None,
                             "premature_frac": ph, "premature_ci95": [max(0.0, c - h), min(1.0, c + h)],
                             "median_lag_nonpremature_days": float(np.median(lag)) if lag.size else None,
                             "onset": "t_anom on the 0.5-d first-detectable grid recorded at generation"}
    return out


# ------------------------------------------------------------------ colour fine-tunes (GPU)
def finetune(arm, device):
    ck = os.path.join(WORK, f"ft_{arm}.pt")
    if not os.path.exists(ck + ".done"):
        run([sys.executable, "-m", "pipeline.train", "--cache", memmap(arm), "--out", ck, "--init-weights", SHIPPED,
             "--epochs", "3", "--lr", "5e-5", "--truncate-aug", "0.5", "--seed", "20260912", "--device", device])
        open(ck + ".done", "w").write("epochs=3 lr=5e-5 truncate_aug=0.5 seed=20260912\n")
    return ck


def summary(m):
    keys = ("completeness_at_purity", "purity", "threshold", "ap", "macro_f1")
    flat = {}
    def walk(d, pre=""):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, pre + k + ".")
            elif isinstance(v, (int, float)) and any(k.endswith(x) or k == x for x in keys):
                flat[pre + k] = v
    walk(m)
    flat["prevalence"] = m["_prevalence_nonpspl"]
    return flat


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--device", default="mps"); ap.add_argument("--out", default=os.path.join(HERE, "referee_round.json"))
    args = ap.parse_args(argv)
    os.makedirs(WORK, exist_ok=True)
    arms = [a for a in ARMS if a.startswith("test") or args.finetune]
    log(f"generating {sum(len(ARMS[a][0]) for a in arms)} shards with {args.workers} workers")
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(lambda x: gen(*x), [(a, s) for a in arms for s in ARMS[a][0]]))
    mms = {a: memmap(a) for a in arms}
    res = {"_doc": __doc__.split("\n")[0], "shards": {a: ARMS[a][0] for a in arms}, "run_shard_args": {a: ARMS[a][1] for a in arms},
           "code": subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=12"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
           "floor_sensitivity": {}, "colour_ablation": {"shipped": {}}, "mixed_class_stream": None}
    for a, fl in (("test_f001", 0.01), ("test_f002", 0.02), ("test_f005", 0.05)):
        res["floor_sensitivity"][str(fl)] = summary(evaluate(SHIPPED, mms[a], f"shipped_{a}"))
        log(f"floor {fl}: {res['floor_sensitivity'][str(fl)]}")
    for a in ("test_f002", "test_colour"):
        res["colour_ablation"]["shipped"][a] = summary(evaluate(SHIPPED, mms[a], f"shipped_{a}"))
    if args.finetune:
        for tr in ("train_trained", "train_colour"):
            ck = finetune(tr, args.device)
            res["colour_ablation"][f"finetuned_on_{tr}"] = {a: summary(evaluate(ck, mms[a], f"{tr}_{a}")) for a in ("test_f002", "test_colour")}
    res["mixed_class_stream"] = stream_reduce(stream_scan(args.workers))
    res["command"] = " ".join(sys.argv)
    json.dump(res, open(args.out, "w"), indent=1)
    log(f"wrote {args.out}")
    print(json.dumps(res["mixed_class_stream"], indent=1)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Referee-round items that need compute, run locally on our own simulator -> validation/referee_round.json.

The pre-submission referee round (paper/REVISION.md section 2) deferred four items. Three are run here, on our
simulator only, with the SHIPPED checkpoint unless stated:

1. Detectability-floor sensitivity by RE-SIMULATION. Shards 90-91 of the held-out pool (disjoint from training)
   regenerated with the label floor at 0.01, 0.02 (adopted) and 0.05 mag. The seeds are the same but the events
   are only partly the same: the generator draws a keep/drop number only for a NonPSPL candidate relabelled PSPL
   or Flat, so its random stream shifts at the first event whose label depends on the floor and re-synchronises
   later; the overlap with the adopted-floor arm is measured (floor_overlap: about a fifth of the events recur in
   the 0.01-mag arm, a twentieth in the 0.05-mag arm). Arm-to-arm differences therefore include sampling noise
   (Wilson intervals on completeness are recorded). Per arm: the NonPSPL prevalence (raw and population-weighted)
   and, on EVERY event of the arm, completeness and population-weighted purity at the frozen threshold, average
   precision and the population-weighted per-class and macro F1 (all_events; the evaluator's own values on its
   80% test split are kept alongside). The model is not retrained; this is the label side plus the shipped model's
   response. About 15% of the adopted-floor events are rows of the pool's threshold-selection split
   (threshold_selection_overlap): the shards are disjoint from training, not from the choice of the frozen
   threshold.
2. Colour-band calibration ablation. The same shards with the audited F087/F213 zeropoints, backgrounds and
   saturation (pipeline.photometry.ROMAN_BANDS_COLOUR_AUDITED; F146, exposure and cadences unchanged). Here the
   events ARE the same (no label-dependent draw changes; parameters match row for row, checked below, and only
   the labels whose detectability depends on colour-band noise differ): the shipped model on both photometries;
   then (--finetune) a short fine-tune of the shipped weights on training shards 0-1 generated with each
   calibration (up to three epochs, lr 5e-5, one seed; the checkpoint kept is the best epoch by validation
   NonPSPL F1 on a 10% split of those shards), each evaluated on both test photometries.
3. Mixed-class sequential evaluation. Every event of the adopted-floor shards (all six classes) revealed in 144
   half-day prefixes and scored at the frozen threshold, twice: with ALL THREE BANDS revealed together
   (mixed_class_stream; its comparison is the paper's three-band cascade variant) and with F146 only
   (mixed_class_stream_f146; the paper's primary protocol). For each: the share of each class that raises an
   alert, alerts per 1,000 events per day, streaming purity at the simulated population mix (keep_prob weights,
   as the paper's prevalence), single-lens alerts split into generated single lenses and binaries demoted by the
   detectability policy, and, for NonPSPL events with a finite onset, premature alerts and lag against the
   0.5-day first-detectable onset recorded at generation. Purity and alert rate are also given at 1% and 0.1%
   anomaly prevalence by prior shift (class-conditional alert fractions and the non-anomalous mix held fixed).

The fourth item, a three-seed sweep of the shipped model's final training stage, needs the 1.9M-event training
set, which is on S3 and not on this machine; it is not run here.

With --archive DIR the per-event inputs of every reported number (stream traces, per-arm scores, labels,
weights, logits, the stream set's labels and onsets, and the two fine-tuned checkpoints) are copied there with
their hashes, so the JSON can be re-reduced without this machine's work directory.

Usage:  python validation/referee_round_local.py [--workers 6] [--finetune] [--archive DIR]
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
    # the paper's FROZEN operating threshold on every event: no threshold is re-selected on this small set, so the
    # arms compare like for like (the evaluator's own 20%-slice threshold lands each arm at a different purity)
    sc = np.load(os.path.join(ev, "score_nonpspl.npy")); lab_e = np.load(os.path.join(ev, "label.npy"))
    kp_e = np.load(os.path.join(ev, "keep_prob.npy")).astype(np.float64); w_e = 1.0 / np.clip(kp_e, 1e-3, 1.0)
    flag = sc >= FROZEN; pos = lab_e == 2
    k, n = int((flag & pos).sum()), int(pos.sum())
    m["_frozen"] = {"threshold": FROZEN, "completeness": k / n, "completeness_k_n": [k, n], "completeness_wilson95": _wilson(k, n),
                    "purity_population_weighted": float(w_e[flag & pos].sum() / w_e[flag].sum()) if flag.any() else None,
                    "false_flag_rate_nonnonpspl_population_weighted": float(w_e[flag & ~pos].sum() / w_e[~pos].sum())}
    # the quantities the evaluator reports on its 80% test split, here on EVERY event of the arm, so that
    # completeness, purity, AP and F1 share one denominator
    from pipeline.evaluate import _prf, average_precision, confusion
    lg = np.load(os.path.join(ev, "logits.npy"))
    prf = _prf(confusion(lab_e, lg.argmax(1), w_e))
    m["_all_events"] = {"n": int(lab_e.size), "ap": float(average_precision(sc, lab_e, w_e)),
                        **{k_: prf[k_] for k_ in ("f1", "macro_f1", "recall", "precision", "confusion")}}
    return m


def _wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    ph = k / n; d = 1 + z * z / n; c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def _same_events(a, b):
    """Row-for-row identity of two arms' raw shards: parameters (t_anom excluded: onset resolution may differ) and labels."""
    import h5py
    n = same = lab_same = 0
    for s in ARMS[a][0]:
        with h5py.File(os.path.join(WORK, f"raw_{a}", f"shard_{s:05d}.h5"), "r") as fa, \
                h5py.File(os.path.join(WORK, f"raw_{b}", f"shard_{s:05d}.h5"), "r") as fb:
            pf = [x.decode() if isinstance(x, bytes) else str(x) for x in fa.attrs["param_fields"]]
            pa, pb = fa["params"][:], fb["params"][:]
            if pa.shape != pb.shape:
                return {"identical_params": False, "n_a": int(pa.shape[0]), "n_b": int(pb.shape[0])}
            cols = [i for i in range(pa.shape[1]) if pf[i] != "t_anom"]
            same += int(np.all(np.isclose(pa[:, cols], pb[:, cols], equal_nan=True), axis=1).sum())
            lab_same += int((fa["label"][:] == fb["label"][:]).sum()); n += pa.shape[0]
    return {"identical_params": same == n, "n": n, "label_agreement": lab_same / n, "n_label_changed": n - lab_same}


def _keys(arm):
    """One fingerprint per event of an arm: the 15 injected parameters (t_anom excluded) and the baseline."""
    import h5py
    out = []
    for s in ARMS[arm][0]:
        with h5py.File(os.path.join(WORK, f"raw_{arm}", f"shard_{s:05d}.h5"), "r") as f:
            pf = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["param_fields"]]
            cols = [i for i in range(len(pf)) if pf[i] != "t_anom"]
            p = f["params"][:].astype(np.float32); mb = f["m_base_ref"][:].astype(np.float32); lab = f["label"][:]
            out += [(p[i, cols].tobytes() + mb[i].tobytes(), int(lab[i])) for i in range(len(p))]
    return out


def _floor_overlap(ref="test_f002", others=("test_f001", "test_f005")):
    """How many events of the adopted-floor arm recur identically in the other floor arms (the keep draw shifts
    the random stream, but it re-synchronises, so the arms are partly the same events)."""
    K = _keys(ref); n = len(K); non = [k for k, l in K if l == 2]
    out = {"n_ref": n, "n_ref_nonpspl": len(non)}
    for a in others:
        S = {k for k, _ in _keys(a)}
        out[a] = {"n_shared": sum(k in S for k, _ in K), "n_shared_nonpspl": sum(k in S for k in non)}
        out[a]["frac_shared"] = out[a]["n_shared"] / n
    return out


def _threshold_selection_overlap(arm="test_f002"):
    """Shards 90-91 are held-out-pool shards; the pool was split 20/80 into threshold-selection and final-test rows
    (paper/results). Count the regenerated events that are threshold-selection rows."""
    R = os.path.join(REPO, "paper", "results")
    P = np.load(os.path.join(R, "params.npy")).astype(np.float32); MB = np.load(os.path.join(R, "m_base_ref.npy")).astype(np.float32)
    isval = np.ones(len(P), bool); isval[np.load(os.path.join(R, "test_idx.npy"))] = False
    pool = {}
    for i in range(len(P)):
        pool.setdefault(P[i].tobytes() + MB[i].tobytes(), []).append(i)
    K = _keys(arm); matched = val = dup = 0
    for k, _ in K:
        hit = pool.get(k)
        if hit:
            matched += 1
            if len(hit) > 1:
                dup += 1
            elif isval[hit[0]]:
                val += 1
    return {"n": len(K), "n_matched_to_pool": matched, "n_ambiguous": dup, "n_threshold_selection_rows": val,
            "frac_threshold_selection_rows": val / len(K)}


# ------------------------------------------------------------------ mixed-class sequential scan
_C = {}


def _init(weights=SHIPPED):
    import torch
    torch.set_num_threads(1)
    import binml
    _C["clf"] = binml.Classifier(weights=weights, device="cpu")


def _scan_rows(job):
    """(shard file, row indices, bands) -> P(NonPSPL) at the 144 half-day cuts for those rows (_scan_chunk on a subset)."""
    path, rows, bands = job
    return np.concatenate([_scan_chunk((path, int(r), int(r) + 1, bands))[1] for r in rows])


def _scan_chunk(job):
    """(shard file, index range, bands) -> per-event P(NonPSPL) at 144 half-day cuts (NaN where no F146 yet).
    bands: "all" reveals every band (the three-band variant of the paper's scan); "f146" reveals F146 only (the
    paper's primary protocol)."""
    import h5py
    from binml.preprocess import BAND_BINS, to_tokens
    path, lo, hi, bands = job
    clf = _C["clf"]; inon = clf.class_names.index("NonPSPL")
    cuts = np.arange(1, 145) * 0.5
    out = []
    with h5py.File(path, "r") as f:
        times = {b: np.asarray(f[f"time/{b}"][:], float) for b in BAND_BINS if f"time/{b}" in f and (bands == "all" or b == "F146")}
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


def stream_scan(workers, bands="all"):
    import h5py
    res = {}
    for s in ARMS["test_f002"][0]:
        path = os.path.join(WORK, "raw_test_f002", f"shard_{s:05d}.h5")
        cache = os.path.join(WORK, f"stream_scan_{s:05d}.npz" if bands == "all" else f"stream_scan_{bands}_{s:05d}.npz")
        if not os.path.exists(cache):
            with h5py.File(path, "r") as f:
                n = int(f.attrs["n_events"])
            jobs = [(path, lo, min(lo + 100, n), bands) for lo in range(0, n, 100)]
            P = np.full((n, 144), np.nan, np.float32); t0 = time.time()
            with cf.ProcessPoolExecutor(max_workers=workers, initializer=_init) as ex:
                for k, (lo, p) in enumerate(ex.map(_scan_chunk, jobs)):
                    P[lo:lo + p.shape[0]] = p
                    if k % 10 == 0:
                        log(f"  stream scan ({bands}) shard {s}: {lo + p.shape[0]}/{n} ({time.time() - t0:.0f}s)")
            np.savez_compressed(cache, p=P)
        res[s] = (path, np.load(cache)["p"])
    return res


def single_lens_scan_recommended(workers):
    """Like-for-like with the RMDC26 cascade (which runs the recommended gap-aware checkpoint, F146 only, frozen
    threshold): the generated single lenses of the stream set scanned with that checkpoint, F146 only. Returns the
    alert fraction at the frozen threshold and at the checkpoint's own threshold."""
    import h5py
    from binml.classifier import GAPAWARE_THRESHOLD, GAPAWARE_WEIGHTS
    from pipeline.classes import CLASS_NAMES
    ip = CLASS_NAMES.index("PSPL"); P = []
    for s in ARMS["test_f002"][0]:
        path = os.path.join(WORK, "raw_test_f002", f"shard_{s:05d}.h5")
        cache = os.path.join(WORK, f"stream_scan_gapaware_f146_single_{s:05d}.npz")
        with h5py.File(path, "r") as f:
            rows = np.flatnonzero((f["label"][:] == ip) & (f["true_class"][:] == ip))
        if not os.path.exists(cache):
            chunks = [rows[i:i + 50] for i in range(0, rows.size, 50)]
            with cf.ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(GAPAWARE_WEIGHTS,)) as ex:
                p = np.concatenate(list(ex.map(_scan_rows, [(path, c, "f146") for c in chunks])))
            np.savez_compressed(cache, p=p, rows=rows)
        P.append(np.load(cache)["p"])
    P = np.nan_to_num(np.concatenate(P), nan=-1.0)
    k_frozen = int((P >= FROZEN).any(1).sum()); k_own = int((P >= GAPAWARE_THRESHOLD).any(1).sum())
    return {"checkpoint": "binml-gapaware.pt (recommended)", "bands": "F146", "n_generated_single_lenses": int(P.shape[0]),
            "alerts_at_frozen_threshold": k_frozen, "alerts_at_own_threshold": k_own,
            "alert_frac_frozen": k_frozen / P.shape[0], "alert_frac_own": k_own / P.shape[0], "own_threshold": GAPAWARE_THRESHOLD}


def _final_epoch_ckpt(ck):
    """The last epoch of a fine-tune (its .last training state) as an evaluable checkpoint, next to the kept best one."""
    import torch
    dst = ck.replace(".pt", "_final_epoch.pt")
    if not os.path.exists(dst):
        best = dict(torch.load(ck, map_location="cpu", weights_only=False))
        last = torch.load(ck + ".last", map_location="cpu", weights_only=False)
        best["model"] = last["model"]; best["epoch"] = last["epoch"]
        torch.save(best, dst)
    return dst


def stream_reduce(scans):
    import h5py
    from pipeline.classes import CLASS_NAMES
    P, lab, kp, tan, tcl = [], [], [], [], []
    for s, (path, p) in scans.items():
        with h5py.File(path, "r") as f:
            pf = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["param_fields"]]
            P.append(p); lab.append(f["label"][:]); kp.append(f["keep_prob"][:]); tan.append(f["params"][:, pf.index("t_anom")])
            tcl.append(f["true_class"][:])
    P, lab, kp, tan, tcl = (np.concatenate(P), np.concatenate(lab), np.concatenate(kp).astype(np.float64), np.concatenate(tan),
                            np.concatenate(tcl))
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
    # prior shift, as the paper's prevalence figure: hold the class-conditional alert fractions and the mix of the
    # non-anomalous classes fixed, rescale the anomaly prevalence pi
    pos = lab == 2
    D = float(w[fired & pos].sum() / w[pos].sum()); F = float(w[fired & ~pos].sum() / w[~pos].sum())
    out["alert_frac_nonnonpspl_population_weighted"] = F
    out["at_prevalence"] = {str(pi): {"purity": pi * D / (pi * D + (1 - pi) * F),
                                      "alerts_per_1000_events_per_day": 1000 * (pi * D + (1 - pi) * F) / 72.0}
                            for pi in (0.01, 0.001)}
    out["simulated_prevalence_population_weighted"] = float(w[pos].sum() / w.sum())
    # single-lens alerts that are binaries generated as such whose anomaly fails the detectability policy
    ps = lab == CLASS_NAMES.index("PSPL"); ps_f = fired & ps; dem = ps_f & (tcl == CLASS_NAMES.index("NonPSPL"))
    out["pspl_alerts_from_demoted_binaries"] = {"k": int(dem.sum()), "n": int(ps_f.sum()),
                                                "population_weighted": float(w[dem].sum() / w[ps_f].sum()) if ps_f.any() else None}
    gen = ps & (tcl == CLASS_NAMES.index("PSPL")); dmb = ps & (tcl == CLASS_NAMES.index("NonPSPL"))
    out["single_lens_alerts_by_origin"] = {"generated_single_lens": {"k": int((fired & gen).sum()), "n": int(gen.sum())},
                                           "demoted_binary": {"k": int((fired & dmb).sum()), "n": int(dmb.sum())},
                                           "note": "PSPL-labelled events by generator class; demoted binaries are subsampled at "
                                                   "keep_prob 0.15, generated single lenses kept whole"}
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
    """pipeline.evaluate's headline quantities (the paper's procedure, on its 80% test split), the prevalence, the
    frozen-threshold operating point, and AP / F1 on every event of the arm (all_events)."""
    h = m["headline"]; ci = m.get("headline_ci95", {})
    return {"completeness_at_purity": h["completeness_at_fixed_purity"], "purity_achieved": h["purity_achieved"],
            "threshold": h["threshold"], "completeness_ci95": [ci.get("lo"), ci.get("hi")],
            "ap": m["average_precision_population"], "macro_f1_population": m["argmax_population"]["macro_f1"],
            "f1_population": m["argmax_population"]["f1"], "n_events": m["n_events"],
            "evaluator_split_note": "ap, macro_f1_population and f1_population are pipeline.evaluate's values on its 80% "
                                    "test split; all_events holds them on every event of the arm",
            "prevalence": m["_prevalence_nonpspl"], "at_frozen_threshold": m["_frozen"], "all_events": m["_all_events"]}


def _archive(dest, arms, finetuned):
    """Copy the per-event inputs behind every reported number into dest (about 30 MB), with their sha256."""
    import hashlib
    import shutil
    import h5py
    os.makedirs(dest, exist_ok=True); files = {}
    sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
    for s in ARMS["test_f002"][0]:
        for src, name in ((f"stream_scan_{s:05d}.npz", f"stream_scan_all_{s:05d}.npz"), (f"stream_scan_f146_{s:05d}.npz", f"stream_scan_f146_{s:05d}.npz"),
                          (f"stream_scan_gapaware_f146_single_{s:05d}.npz", f"stream_scan_gapaware_f146_single_{s:05d}.npz")):
            shutil.copy(os.path.join(WORK, src), os.path.join(dest, name)); files[name] = sha(os.path.join(dest, name))
        with h5py.File(os.path.join(WORK, "raw_test_f002", f"shard_{s:05d}.h5"), "r") as f:
            pf = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["param_fields"]]
            name = f"stream_events_{s:05d}.npz"
            np.savez_compressed(os.path.join(dest, name), label=f["label"][:], keep_prob=f["keep_prob"][:],
                                true_class=f["true_class"][:], t_anom=f["params"][:, pf.index("t_anom")])
        files[name] = sha(os.path.join(dest, name))
    evals = [f"shipped_{a}" for a in arms if a.startswith("test")]
    if finetuned:
        evals += [f"{tr}{fe}_{a}" for tr in ("train_trained", "train_colour") for fe in ("", "_final") for a in ("test_f002", "test_colour")]
    for e in evals:
        d = os.path.join(WORK, f"eval_{e}"); name = f"eval_{e}.npz"
        np.savez_compressed(os.path.join(dest, name), **{k: np.load(os.path.join(d, f"{k}.npy"))
                            for k in ("score_nonpspl", "logits", "label", "true_class", "keep_prob", "test_idx")})
        files[name] = sha(os.path.join(dest, name))
    if finetuned:
        for tr in ("train_trained", "train_colour"):
            name = f"ft_{tr}.pt"; shutil.copy(os.path.join(WORK, name), os.path.join(dest, name)); files[name] = sha(os.path.join(dest, name))
    return {"dir": os.path.relpath(dest, REPO), "sha256": files}


CODE_AT_START = None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--device", default="mps"); ap.add_argument("--out", default=os.path.join(HERE, "referee_round.json"))
    ap.add_argument("--archive", default=None, help="copy the per-event inputs of every reported number here")
    args = ap.parse_args(argv)
    global CODE_AT_START
    CODE_AT_START = subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=12"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    os.makedirs(WORK, exist_ok=True)
    arms = [a for a in ARMS if a.startswith("test") or args.finetune]
    log(f"generating {sum(len(ARMS[a][0]) for a in arms)} shards with {args.workers} workers")
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(lambda x: gen(*x), [(a, s) for a in arms for s in ARMS[a][0]]))
    mms = {a: memmap(a) for a in arms}
    res = {"_doc": __doc__.split("\n")[0], "shards": {a: ARMS[a][0] for a in arms}, "run_shard_args": {a: ARMS[a][1] for a in arms},
           "code": CODE_AT_START,
           "floor_sensitivity": {}, "colour_ablation": {"shipped": {}}, "mixed_class_stream": None}
    for a, fl in (("test_f001", 0.01), ("test_f002", 0.02), ("test_f005", 0.05)):
        res["floor_sensitivity"][str(fl)] = summary(evaluate(SHIPPED, mms[a], f"shipped_{a}"))
        log(f"floor {fl}: {res['floor_sensitivity'][str(fl)]}")
    res["floor_overlap"] = _floor_overlap()
    res["threshold_selection_overlap"] = _threshold_selection_overlap()
    for a in ("test_f002", "test_colour"):
        res["colour_ablation"]["shipped"][a] = summary(evaluate(SHIPPED, mms[a], f"shipped_{a}"))
    res["colour_ablation"]["same_events"] = _same_events("test_f002", "test_colour")
    if args.finetune:
        for tr in ("train_trained", "train_colour"):
            ck = finetune(tr, args.device)
            res["colour_ablation"][f"finetuned_on_{tr}"] = {a: summary(evaluate(ck, mms[a], f"{tr}_{a}")) for a in ("test_f002", "test_colour")}
            # the kept checkpoint is the best epoch on 1,503 validation events; the last epoch shows how much rides on that choice
            fin = _final_epoch_ckpt(ck)
            res["colour_ablation"][f"finetuned_on_{tr}_final_epoch"] = {a: summary(evaluate(fin, mms[a], f"{tr}_final_{a}")) for a in ("test_f002", "test_colour")}
    res["mixed_class_stream"] = stream_reduce(stream_scan(args.workers))
    res["mixed_class_stream_f146"] = stream_reduce(stream_scan(args.workers, bands="f146"))
    res["single_lens_stream_recommended_f146"] = single_lens_scan_recommended(args.workers)
    if args.finetune:
        import hashlib
        res["colour_ablation"]["finetune_checkpoints_sha256"] = {
            tr: hashlib.sha256(open(os.path.join(WORK, f"ft_{tr}.pt"), "rb").read()).hexdigest() for tr in ("train_trained", "train_colour")}
    res["command"] = " ".join(sys.argv)
    if args.archive:
        res["archive"] = _archive(args.archive, arms, args.finetune)
    json.dump(res, open(args.out, "w"), indent=1)
    log(f"wrote {args.out}")
    print(json.dumps(res["mixed_class_stream"], indent=1)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

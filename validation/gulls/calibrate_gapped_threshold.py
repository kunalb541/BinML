"""Re-calibrate the anomaly threshold for Roman's gapped schedule WITHOUT touching GULLS.

The frozen threshold 0.9042 was chosen on gap-free, point-source data. For a gap-aware checkpoint
used on the planned schedule the honest operating point is a threshold chosen the same way the
paper chooses it -- on OUR simulated events, at the target purity -- but with the schedule's gaps
and finite-source single lenses present. Calibrating on GULLS itself would turn the transfer test
into a fit; this script never reads GULLS to choose anything, it only REPORTS what the chosen
threshold does there.

Schedules (--schedule):
  seasons (default, 2026-09-11)  each held-out event receives one of the six MEASURED RMDC26
                                 high-cadence seasons (event i -> season i mod 6), from
                                 validation/gulls/rmdc26_schedule.json: that season's empty bins in
                                 every band (pauses and season end). RMDC26's partial occupancy
                                 (F146 bins at 7/8 where a colour visit displaced an epoch) cannot be
                                 imposed faithfully on binned data; 'seasons_occ' / 'occupancy' do it
                                 anyway as a labelled diagnostic.
  legacy                         the 2026-09-10 calibration: the seven FIRST-season pauses only, no
                                 season end, colour bins blanked wherever a pause overlaps them.
                                 It matches one season in six and over-blanks colour; kept only to
                                 reproduce gapped_threshold_<tag>.json.

Steps: copy the held-out memmap and impose the schedule; run pipeline.evaluate on the clean and
gapped copies (threshold at the target purity on evaluate's 20% validation slice); also report the
threshold chosen on the FULL gapped pool and its spread over 200 random 20% slices, the pool's
NonPSPL prevalence (raw and population-weighted; purity depends on it), and the GULLS rates at
each threshold. Labels of the gapped copy are those of the unblanked season: an event whose
anomaly falls entirely inside a pause still counts as NonPSPL (a small, conservative bias).

Usage:
  python validation/gulls/calibrate_gapped_threshold.py --heldout ~/Desktop/Research/microlensing/fspl5s_local_work/mm_heldout \\
      --ckpt validation/gulls/weights/ft_fspl5s_g08.pt --tag fspl5s_g08 --rows ~/Desktop/Research/microlensing/gulls_curve_cache/rows_full_fspl5s_g08.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

RMDC26_GAPS_D = [0.98, 2.48, 21.49, 31.48, 35.23, 62.23, 69.48]   # LEGACY: first season only
RMDC26_GAP_H = 6.2
BAND_BINS = {"F146": 864, "F087": 96, "F213": 96}
FROZEN = 0.9042405486106873


def gap_mask_f146():
    """LEGACY first-season pause mask (no season end)."""
    centres = (np.arange(864) + 0.5) * 72.0 / 864
    m = np.zeros(864, bool)
    for g in RMDC26_GAPS_D:
        m |= (centres >= g) & (centres < g + RMDC26_GAP_H / 24.0)
    return m


def _copy_meta(src, dst):
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith(".npy") or f == "meta.json":
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    return json.load(open(os.path.join(src, "meta.json")))["n_events"]


def make_gapped(src, dst, mask=None, note="RMDC26 seven ~6.2 h pauses (first season)"):
    """LEGACY: copy a memmap and blank one reference-band mask (colour by any-overlap)."""
    if os.path.exists(os.path.join(dst, "meta.json")):
        return
    n = _copy_meta(src, dst)
    m146 = gap_mask_f146() if mask is None else np.asarray(mask, bool)
    for b, L in BAND_BINS.items():
        mask_b = m146 if L == 864 else m146.reshape(L, 864 // L).any(axis=1)
        feat = np.memmap(os.path.join(src, f"feat_{b}.f16"), np.float16, "r", shape=(n, L, 3))
        frac = np.memmap(os.path.join(src, f"frac_{b}.f16"), np.float16, "r", shape=(n, L))
        fo = np.memmap(os.path.join(dst, f"feat_{b}.f16"), np.float16, "w+", shape=(n, L, 3))
        fr = np.memmap(os.path.join(dst, f"frac_{b}.f16"), np.float16, "w+", shape=(n, L))
        for i in range(0, n, 4096):
            x = np.array(feat[i:i + 4096]); y = np.array(frac[i:i + 4096])
            x[:, mask_b, :] = np.nan; y[:, mask_b] = 0.0      # an empty bin, exactly as the cache encodes one
            fo[i:i + 4096] = x; fr[i:i + 4096] = y
        fo.flush(); fr.flush()
    meta = json.load(open(os.path.join(dst, "meta.json")))
    meta["gapped"] = {"schedule": note, "gaps_d": RMDC26_GAPS_D, "f146_bins_blanked": int(m146.sum())}
    json.dump(meta, open(os.path.join(dst, "meta.json"), "w"), indent=1)


def make_gapped_seasons(src, dst, blank=True, cap=False):
    """Impose a measured RMDC26 season on every event (event i -> dense season i mod 6).

    blank: empty the season's empty bins (pauses, season end); cap: cap each bin's occupancy at the
    season's template (F146 7/8 where a colour visit displaced an epoch; colour 2/3 where RMDC26's
    ~6.9 h colour cadence puts two visits in an 18 h bin). Both = the measured schedule; one at a
    time = the diagnostic split between pause positions and occupancy."""
    if os.path.exists(os.path.join(dst, "meta.json")):
        return
    from pipeline.train import load_rmdc26_templates
    T = load_rmdc26_templates()
    n = _copy_meta(src, dst)
    season_of = np.arange(n) % len(T)
    for b, L in BAND_BINS.items():
        feat = np.memmap(os.path.join(src, f"feat_{b}.f16"), np.float16, "r", shape=(n, L, 3))
        frac = np.memmap(os.path.join(src, f"frac_{b}.f16"), np.float16, "r", shape=(n, L))
        fo = np.memmap(os.path.join(dst, f"feat_{b}.f16"), np.float16, "w+", shape=(n, L, 3))
        fr = np.memmap(os.path.join(dst, f"frac_{b}.f16"), np.float16, "w+", shape=(n, L))
        for i in range(0, n, 4096):
            x = np.array(feat[i:i + 4096]); y = np.array(frac[i:i + 4096]).astype(np.float32)
            for k, tmpl in enumerate(T):
                rows = np.flatnonzero(season_of[i:i + 4096] == k)
                if not rows.size:
                    continue
                empty, ftm = tmpl[b]
                xs = x[rows]; ys = y[rows]
                if blank:
                    xs[:, empty, :] = np.nan; ys[:, empty] = 0.0
                if cap:
                    ys[:, ~empty] = np.minimum(ys[:, ~empty], ftm[~empty])
                x[rows] = xs; y[rows] = ys
            fo[i:i + 4096] = x; fr[i:i + 4096] = y.astype(np.float16)
        fo.flush(); fr.flush()
    meta = json.load(open(os.path.join(dst, "meta.json")))
    meta["gapped"] = {"schedule": "RMDC26 measured per-season templates (event i -> dense season i mod 6)",
                      "blank_empty_bins": bool(blank), "cap_occupancy": bool(cap),
                      "source": "validation/gulls/rmdc26_schedule.json",
                      "f146_empty_bins_per_season": [int(t["F146"][0].sum()) for t in T]}
    json.dump(meta, open(os.path.join(dst, "meta.json"), "w"), indent=1)


def gulls_at(rows_path, thr):
    rows = [r for r in json.load(open(rows_path)) if r.get("dense") and "pred" in r]
    out = {}
    for lab, key in (("RMDC26_1S1L_ML", "fa_1S1L"), ("RMDC26_1S2L_ML", "recall_1S2L"),
                     ("RMDC26_2S2L_ML", "recall_2S2L")):
        p = np.array([r["p_nonpspl"] for r in rows if r["sim_label"] == lab])
        k = int((p >= thr).sum())
        out[key] = round(k / p.size, 4); out[key + "_k"] = k; out[key + "_n"] = int(p.size)
    return out


def pool_threshold_stats(ev_dir, target, reps=200, seed=11):
    """Threshold at target purity on the full pool and over random 20% validation slices."""
    from pipeline.evaluate import completeness_at_purity, population_weights
    from pipeline.classes import CLASS_NAMES
    s = np.load(os.path.join(ev_dir, "score_nonpspl.npy")).astype(np.float64)
    y = np.load(os.path.join(ev_dir, "label.npy"))
    w = population_weights(np.load(os.path.join(ev_dir, "keep_prob.npy")))
    inon = CLASS_NAMES.index("NonPSPL")
    full = completeness_at_purity(s, y, w, target)
    g = np.random.default_rng(seed); n = len(s); thr = []
    for _ in range(reps):
        vi = g.permutation(n)[:int(0.2 * n)]
        op = completeness_at_purity(s[vi], y[vi], w[vi], target)
        if op.get("achievable"):
            thr.append(op["threshold"])
    thr = np.array(thr)
    return {"full_pool": full, "prevalence_nonpspl_raw": round(float((y == inon).mean()), 4),
            "prevalence_nonpspl_population_weighted": round(float((w * (y == inon)).sum() / w.sum()), 4),
            "slice_thresholds": {"n_achievable": int(thr.size), "reps": reps,
                                 **({"median": float(np.median(thr)), "p16": float(np.percentile(thr, 16)),
                                     "p84": float(np.percentile(thr, 84)), "p2.5": float(np.percentile(thr, 2.5)),
                                     "p97.5": float(np.percentile(thr, 97.5))} if thr.size else {})}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--heldout", required=True, help="held-out memmap dir (our own simulations)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rows", required=True, help="GULLS rows for this checkpoint (from gulls_transfer.py)")
    ap.add_argument("--schedule", choices=["seasons", "legacy", "seasons_occ", "occupancy"], default="seasons",
                    help="seasons = measured per-season empty bins (faithful on binned data); seasons_occ / occupancy = "
                         "with / only the occupancy cap (DIAGNOSTIC: capping frac without removing an epoch from the "
                         "bin's features is a combination real data never has); legacy = first-season pauses only")
    ap.add_argument("--target-purity", type=float, default=0.90)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--work", default=None)
    args = ap.parse_args(argv)
    work = args.work or os.path.join(os.path.dirname(args.heldout.rstrip("/")), "calib")
    suffix = {"legacy": "", "seasons": "_seasons", "seasons_occ": "_seasonsocc", "occupancy": "_occupancy"}[args.schedule]
    gapped = os.path.join(work, "mm_heldout_gapped" + suffix)
    if args.schedule == "legacy":
        make_gapped(args.heldout, gapped)
    else:
        make_gapped_seasons(args.heldout, gapped, blank=args.schedule != "occupancy", cap=args.schedule in ("seasons_occ", "occupancy"))
    res = {"_doc": __doc__.split("\n")[0], "checkpoint": args.ckpt, "target_purity": args.target_purity,
           "heldout_source": args.heldout, "schedule": args.schedule, "frozen_threshold": FROZEN,
           "label_note": "gapped copy keeps the unblanked labels (an anomaly entirely inside a pause still counts as NonPSPL)",
           "arms": {}}
    for arm, mm in (("clean", args.heldout), ("rmdc26_gapped", gapped)):
        ev = os.path.join(work, f"eval_{args.tag}_{arm}{suffix if arm != 'clean' else ''}")
        if not os.path.exists(os.path.join(ev, "metrics.json")):
            subprocess.run([sys.executable, "-m", "pipeline.evaluate", "--ckpt", args.ckpt, "--cache", mm,
                            "--out", ev, "--device", args.device, "--target-purity", str(args.target_purity)],
                           check=True, cwd=REPO, env=dict(os.environ, PYTHONPATH=REPO))
        m = json.load(open(os.path.join(ev, "metrics.json")))
        thr = float(m["headline"]["threshold"])
        block = {"threshold_at_target_purity": thr,
                 "our_heldout": {"completeness": round(m["headline"]["completeness_at_fixed_purity"], 4),
                                 "purity": round(m["headline"]["purity_achieved"], 4),
                                 "ap": round(m["average_precision_population"], 4)},
                 "gulls_at_this_threshold": gulls_at(args.rows, thr)}
        if os.path.exists(os.path.join(ev, "score_nonpspl.npy")):
            st = pool_threshold_stats(ev, args.target_purity)
            block["pool"] = st
            if st["full_pool"].get("achievable"):
                block["gulls_at_full_pool_threshold"] = gulls_at(args.rows, st["full_pool"]["threshold"])
            sl = st["slice_thresholds"]
            if "p16" in sl:
                block["gulls_at_slice_p16_p84"] = [gulls_at(args.rows, sl["p16"]), gulls_at(args.rows, sl["p84"])]
        res["arms"][arm] = block
    res["gulls_at_frozen_threshold"] = gulls_at(args.rows, FROZEN)
    out = os.path.join(HERE, f"gapped_threshold_{args.tag}{suffix}.json")
    json.dump(res, open(out, "w"), indent=2)
    for arm, b in res["arms"].items():
        extra = ""
        if "pool" in b and b["pool"]["full_pool"].get("achievable"):
            extra = (f" | full-pool thr {b['pool']['full_pool']['threshold']:.4f}; slices 68% "
                     f"[{b['pool']['slice_thresholds'].get('p16', float('nan')):.4f}, {b['pool']['slice_thresholds'].get('p84', float('nan')):.4f}]; "
                     f"prevalence {b['pool']['prevalence_nonpspl_population_weighted']}")
        print(f"{arm:14s} thr {b['threshold_at_target_purity']:.4f} heldout {b['our_heldout']} GULLS "
              f"{ {k: v for k, v in b['gulls_at_this_threshold'].items() if not k.endswith(('_k', '_n'))} }{extra}")
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

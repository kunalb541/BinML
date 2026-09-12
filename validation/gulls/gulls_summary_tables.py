"""GULLS/RMDC26 summary tables across checkpoints, from the per-event rows -- one command, no typing.

Reads the rows files written by validation/gulls_transfer.py (default: the curve-cache directory),
restricts every comparison to the events ALL listed models scored (identical inputs, identical set),
and writes two artifacts:

  transfer_tradeoff_all.json   -- per model: false alarms at the frozen threshold; 1S1L false-alarm
                                  rate by rho/|u0| (finite-source strength, from the RMDC26 metadata);
                                  planetary recall at MATCHED single-lens false-alarm budgets; mean
                                  1S2L recall over FA <= 0.3.  Matched budgets are the comparison that
                                  cannot be gamed by a threshold shift.
  transfer_colour_ablation.json -- for each checkpoint scored with --bands F146, the same quantities
                                  next to its three-band run.

Usage:
  python validation/gulls/gulls_summary_tables.py \\
      --models shipped=rows_full_shipped_v2.json ft_g08e12=rows_full_ft_g08e12_v2.json \\
               fspl_g08=rows_full_fspl_g08.json fspl5_g08=rows_full_fspl5_g08.json fspl5s_g08=rows_full_fspl5s_g08.json \\
      --f146 fspl_g08=rows_full_fspl_g08_f146only.json ft_g08e12=rows_full_ft_g08e12_f146only.json

From a clone (no curve cache, no RMDC26 metadata): --scores validation/gulls/rmdc26_scores.csv.gz reads P(NonPSPL),
labels, rho, u0 and season from the committed per-event table; name=value pairs then name its COLUMNS:
  python validation/gulls/gulls_summary_tables.py --scores validation/gulls/rmdc26_scores.csv.gz \\
      --models shipped=shipped ft_g08e12=ft_g08e12 fspl5s_g08=fspl5s_g08 --f146 fspl5s_g08=fspl5s_g08_f146only --by-season
The committed transfer_tradeoff_all.json records the exact invocation that regenerates it from a clone
(command_from_clone: every model, the F146-only columns, the pairs and the bootstrap size).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser("~/Desktop/Research/microlensing/gulls_curve_cache")
META = "/tmp/rmdc26_meta.parquet"
FROZEN = 0.9042405486106873
BINS = [(0, 0.03), (0.03, 0.1), (0.1, 0.3), (0.3, 1), (1, 3), (3, float("inf"))]
BUDGETS = (0.02, 0.031, 0.052, 0.117, 0.20)
L1, L2, L3 = "RMDC26_1S1L_ML", "RMDC26_1S2L_ML", "RMDC26_2S2L_ML"
_trapz = getattr(np, "trapezoid", None) or np.trapz          # NumPy < 2.0 has only trapz
SCHEDULE = os.path.join(HERE, "rmdc26_schedule.json")


def load_rows(path):
    if not os.path.isabs(path):
        path = os.path.join(CACHE, path)
    return {r["event_id"]: (r["sim_label"], float(r["p_nonpspl"]), float(r["weight"]), float(r["tE"]))
            for r in json.load(open(path)) if r.get("dense") and "pred" in r}


def load_scores(path):
    """The committed per-event table (build_scores_table.py): {column: {event_id: (label, p)}}, {event_id: (rho, |u0|, season)}."""
    import csv
    import gzip
    with gzip.open(path, "rt", newline="") as fh:
        rows = list(csv.DictReader(fh))
    fixed = {"event_id", "sim_label", "season", "tE", "u0", "rho", "planet_q", "source_is_binary", "m_base", "weight"}
    cols = {c: {int(r["event_id"]): (r["sim_label"], float(r[c]), float(r["weight"]), float(r["tE"])) for r in rows if r[c] != ""}
            for c in rows[0] if c not in fixed}
    meta = {int(r["event_id"]): (float(r["rho"]), abs(float(r["u0"])), int(r["season"])) for r in rows}
    return cols, meta


def load_meta(ids, with_season=False, meta=None):
    """rho/|u0| (and the high-cadence season index) per event, from the RMDC26 metadata or from `meta`."""
    if meta is not None:
        ratio = np.array([meta[i][0] / max(meta[i][1], 1e-6) for i in ids])
        return (ratio, np.array([meta[i][2] for i in ids])) if with_season else ratio
    import pyarrow.parquet as pq
    m = pq.read_table(META, columns=["event_id", "rho", "u0lens1", "t0lens1"]).to_pydict()
    d = {int(e): (float(r), abs(float(u)), float(t)) for e, r, u, t in zip(m["event_id"], m["rho"], m["u0lens1"], m["t0lens1"])}
    ratio = np.array([d[i][0] / max(d[i][1], 1e-6) for i in ids])
    if not with_season:
        return ratio
    S = json.load(open(SCHEDULE))["seasons"]
    def season_of(t):
        for x in S:
            if x["start_bjd"] <= t <= x["end_bjd"]:
                return x["index"]
        return -1
    return ratio, np.array([season_of(d[i][2]) for i in ids])


def summarise(p, lab, ratio):
    s1, s2, s3 = lab == L1, lab == L2, lab == L3
    ths = np.unique(np.round(p, 4))[::-1]
    fa = np.array([(p[s1] >= t).mean() for t in ths])
    r2 = np.array([(p[s2] >= t).mean() for t in ths])
    r3 = np.array([(p[s3] >= t).mean() for t in ths])
    o = np.argsort(fa); k = fa[o] <= 0.3
    kk = lambda sel: [int((p[sel] >= FROZEN).sum()), int(sel.sum())]
    out = {"frozen_threshold": {"fa_1S1L": round(float((p[s1] >= FROZEN).mean()), 4),
                                "recall_1S2L": round(float((p[s2] >= FROZEN).mean()), 4),
                                "recall_2S2L": round(float((p[s3] >= FROZEN).mean()), 4),
                                "fa_1S1L_k_n": kk(s1), "recall_1S2L_k_n": kk(s2), "recall_2S2L_k_n": kk(s3)},
           "fa_1S1L_by_rho_over_u0": [], "recall_at_matched_fa": [],
           "mean_recall_1S2L_fa_le_0p3": round(float(_trapz(r2[o][k], fa[o][k]) / 0.3), 4),
           "mean_recall_2S2L_fa_le_0p3": round(float(_trapz(r3[o][k], fa[o][k]) / 0.3), 4),
           "mean_recall_1S2L_fa_le_0p3_exact": float(_trapz(r2[o][k], fa[o][k]) / 0.3)}
    for lo, hi in BINS:
        sel = s1 & (ratio >= lo) & (ratio < hi)
        out["fa_1S1L_by_rho_over_u0"].append({"bin": [lo, None if hi == float("inf") else hi], "n": int(sel.sum()),
                                              "k": int((p[sel] >= FROZEN).sum()),
                                              "fa": round(float((p[sel] >= FROZEN).mean()), 4) if sel.sum() else None})
    for tgt in BUDGETS:
        i = int(np.argmin(np.abs(fa - tgt)))
        out["recall_at_matched_fa"].append({"fa_target": tgt, "threshold": float(ths[i]), "fa": round(float(fa[i]), 4),
                                            "fa_achieved_exact": float(fa[i]),
                                            "recall_1S2L": round(float(r2[i]), 4), "recall_2S2L": round(float(r3[i]), 4),
                                            "recall_1S2L_k_n": [int((p[s2] >= ths[i]).sum()), int(s2.sum())],
                                            "recall_2S2L_k_n": [int((p[s3] >= ths[i]).sum()), int(s3.sum())]})
    return out


TE_BINS = [(1, 3), (3, 10), (10, 30), (30, 300)]


def _curve(p, sel, w, ths):
    """Weighted exceedance fraction of p[sel] at each threshold in ths (descending), by sorting once."""
    q = np.sort(p[sel])[::-1]; ww = w[sel][np.argsort(p[sel])[::-1]]
    cw = np.r_[0.0, np.cumsum(ww)] / ww.sum()
    return cw[np.searchsorted(-q, -ths, side="right")]


def summarise_weighted(p, lab, ratio, w, te):
    """The unweighted summary's quantities with RMDC26's event-rate weights (final_weight): budgets are
    weighted single-lens false-alarm rates, recall is weighted recall. Unrounded."""
    s1, s2, s3 = lab == L1, lab == L2, lab == L3
    ths = np.unique(np.round(p, 4))[::-1]
    fa, r2, r3 = _curve(p, s1, w, ths), _curve(p, s2, w, ths), _curve(p, s3, w, ths)
    o = np.argsort(fa); k = fa[o] <= 0.3
    wf = lambda sel, t: float(w[sel][p[sel] >= t].sum() / w[sel].sum()) if sel.any() else None
    out = {"frozen_threshold": {"fa_1S1L": wf(s1, FROZEN), "recall_1S2L": wf(s2, FROZEN), "recall_2S2L": wf(s3, FROZEN)},
           "fa_1S1L_by_rho_over_u0": [{"bin": [lo, None if hi == float("inf") else hi],
                                       "fa": wf(s1 & (ratio >= lo) & (ratio < hi), FROZEN),
                                       "kish_n_eff": _kish(w[s1 & (ratio >= lo) & (ratio < hi)])} for lo, hi in BINS],
           "recall_at_matched_fa": [], "mean_recall_1S2L_fa_le_0p3": float(_trapz(r2[o][k], fa[o][k]) / 0.3),
           "mean_recall_2S2L_fa_le_0p3": float(_trapz(r3[o][k], fa[o][k]) / 0.3)}
    for tgt in BUDGETS:
        i = int(np.argmin(np.abs(fa - tgt)))
        out["recall_at_matched_fa"].append({"fa_target": tgt, "threshold": float(ths[i]), "fa": float(fa[i]),
                                            "recall_1S2L": float(r2[i]), "recall_2S2L": float(r3[i])})
    out["fa_1S1L_by_tE"] = []
    for lo, hi in TE_BINS:
        sel = s1 & (te >= lo) & (te < hi)
        out["fa_1S1L_by_tE"].append({"tE_days": [lo, hi], "n": int(sel.sum()), "fa_weighted": wf(sel, FROZEN),
                                     "fa_unweighted": float((p[sel] >= FROZEN).mean()) if sel.any() else None})
    return out


def _kish(ww):
    """Kish effective sample size of a weighted subsample (sum w)^2 / sum w^2."""
    return float(ww.sum() ** 2 / (ww ** 2).sum()) if ww.size else 0.0


def _point(p, lab, w, budgets=(0.02, 0.052, 0.117)):
    """FA at the frozen threshold and 1S2L/2S2L recall at matched single-lens budgets (grid argmin, as summarise)."""
    s1, s2, s3 = lab == L1, lab == L2, lab == L3
    ths = np.unique(np.round(p, 4))[::-1]
    fa, r2, r3 = _curve(p, s1, w, ths), _curve(p, s2, w, ths), _curve(p, s3, w, ths)
    out = [float(w[s1][p[s1] >= FROZEN].sum() / w[s1].sum())]
    for tgt in budgets:
        i = int(np.argmin(np.abs(fa - tgt))); out += [float(r2[i]), float(r3[i])]
    return np.array(out)


def paired_bootstrap(P, lab, w, pairs, reps, seed):
    """Differences a - b on the SAME events, unweighted and rate-weighted, with 95% percentile intervals from a
    class-stratified bootstrap over events. Evaluation-sample noise only: training-seed variance is not in it."""
    rng = np.random.default_rng(seed)
    idx_by = [np.flatnonzero(lab == L) for L in (L1, L2, L3)]
    names = ["fa_frozen"] + [f"recall_{c}_at_{t}" for t in (0.02, 0.052, 0.117) for c in ("1S2L", "2S2L")]
    out = {"_doc": paired_bootstrap.__doc__, "reps": reps, "seed": seed, "pairs": {}}
    for wt_name, ww in (("unweighted", np.ones_like(w)), ("weighted", w)):
        pts = {(a, b): _point(P[a], lab, ww) - _point(P[b], lab, ww) for a, b in pairs}
        draws = {pr: [] for pr in pairs}
        for _ in range(reps):
            ix = np.concatenate([rng.choice(v, v.size) for v in idx_by])
            cache = {}
            for a, b in pairs:
                for m in (a, b):
                    if m not in cache:
                        cache[m] = _point(P[m][ix], lab[ix], ww[ix])
                draws[(a, b)].append(cache[a] - cache[b])
        for pr in pairs:
            d = np.array(draws[pr]); lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
            out["pairs"].setdefault(f"{pr[0]}-{pr[1]}", {})[wt_name] = {
                n: {"diff": float(pts[pr][i]), "ci95": [float(lo[i]), float(hi[i])]} for i, n in enumerate(names)}
    return out


def parse_pairs(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1); out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", required=True, help="name=rows.json (three-band runs)")
    ap.add_argument("--f146", nargs="*", default=[], help="name=rows.json (F146-only runs of the same checkpoints)")
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--by-season", action="store_true", help="also summarise each high-cadence season separately "
                    "(RMDC26 pause phases differ between seasons; validation/gulls/rmdc26_schedule.json)")
    ap.add_argument("--scores", default=None, help="read everything from the committed rmdc26_scores.csv.gz; "
                    "name=value pairs of --models/--f146 then name its columns")
    ap.add_argument("--pairs", nargs="*", default=[], help="a:b model pairs for paired bootstrap differences")
    ap.add_argument("--boot", type=int, default=400, help="bootstrap replicates for --pairs")
    args = ap.parse_args(argv)
    if args.scores:
        cols, meta = load_scores(args.scores)
        models = {k: cols[v] for k, v in parse_pairs(args.models).items()}
        f146 = {k: cols[v] for k, v in parse_pairs(args.f146).items()}
    else:
        meta = None
        models = {k: load_rows(v) for k, v in parse_pairs(args.models).items()}
        f146 = {k: load_rows(v) for k, v in parse_pairs(args.f146).items()}
    common = sorted(set.intersection(*[set(v) for v in list(models.values()) + list(f146.values())]))
    lab = np.array([next(iter(models.values()))[i][0] for i in common])
    ratio, season = load_meta(common, with_season=True, meta=meta)
    ref = next(iter(models.values()))
    w = np.array([ref[i][2] for i in common]); te = np.array([ref[i][3] for i in common])
    s1 = lab == L1
    trade = {"_doc": __doc__.split("\n")[0], "n_matched_dense": len(common), "frozen_threshold": FROZEN,
             "budgets": list(BUDGETS), "models": {},
             "sample": {"weights": "RMDC26 final_weight (event rate); 'weighted' blocks use it, every other number is per simulated event",
                        "kish_n_eff": {L: float(w[lab == L].sum() ** 2 / (w[lab == L] ** 2).sum()) for L in (L1, L2, L3)},
                        "n": {L: int((lab == L).sum()) for L in (L1, L2, L3)},
                        "frac_1S1L_tE_lt_3d": {"unweighted": float((te[s1] < 3).mean()), "weighted": float(w[s1][te[s1] < 3].sum() / w[s1].sum())},
                        "median_tE_days": {L: float(np.median(te[lab == L])) for L in (L1, L2, L3)}}}
    for name, d in models.items():
        p = np.array([d[i][1] for i in common])
        trade["models"][name] = summarise(p, lab, ratio)
        trade["models"][name]["weighted"] = summarise_weighted(p, lab, ratio, w, te)
        cal = os.path.join(HERE, f"gapped_threshold_{name}_seasons.json")
        if os.path.exists(cal):                     # each checkpoint at ITS OWN threshold (chosen on our simulations)
            fp = json.load(open(cal))["arms"]["rmdc26_gapped"]["pool"]["full_pool"]
            if fp.get("achievable"):
                t = float(fp["threshold"]); blk = {"threshold": t, "source": os.path.basename(cal)}
                for L, key in ((L1, "fa_1S1L"), (L2, "recall_1S2L"), (L3, "recall_2S2L")):
                    sel = lab == L
                    blk[key] = {"k": int((p[sel] >= t).sum()), "n": int(sel.sum()),
                                "weighted": float(w[sel][p[sel] >= t].sum() / w[sel].sum())}
                trade["models"][name]["at_own_calibrated_threshold"] = blk
        if args.by_season:
            trade["models"][name]["by_season"] = {}
            for si in sorted(set(season.tolist())):
                sel = season == si
                if (lab[sel] == L1).sum() >= 200 and (lab[sel] == L2).sum() >= 100:
                    trade["models"][name]["by_season"][str(si)] = {"n": int(sel.sum()), **summarise(p[sel], lab[sel], ratio[sel])}
    if args.pairs:
        P = {name: np.array([d[i][1] for i in common]) for name, d in models.items()}
        trade["paired_differences"] = paired_bootstrap(P, lab, w, [tuple(x.split(":", 1)) for x in args.pairs], args.boot, 20260912)
    trade["command"] = " ".join(sys.argv)
    # the same reduction from the committed per-event table alone (no curve cache, no RMDC26 metadata): every
    # checkpoint is a column of rmdc26_scores.csv.gz named as here, its F146-only run as <name>_f146only
    trade["command_from_clone"] = ("validation/gulls/gulls_summary_tables.py --scores validation/gulls/rmdc26_scores.csv.gz --models "
                                   + " ".join(f"{n}={n}" for n in models) + (" --f146 " + " ".join(f"{n}={n}_f146only" for n in f146) if f146 else "")
                                   + (" --by-season" if args.by_season else "") + (" --pairs " + " ".join(args.pairs) if args.pairs else "")
                                   + f" --boot {args.boot} --out-dir validation/gulls")
    json.dump(trade, open(os.path.join(args.out_dir, "transfer_tradeoff_all.json"), "w"), indent=1)
    print(f"matched dense events: {len(common)}")
    print(f"{'model':12s} {'FA@frozen':>9} {'rec 1S2L':>9} {'rec 2S2L':>9} {'rec@5.2%FA':>11} {'mean rec<=0.3':>13}   FA by rho/|u0| bins")
    for name, m in trade["models"].items():
        f = m["frozen_threshold"]; at = next(a for a in m["recall_at_matched_fa"] if a["fa_target"] == 0.052)
        bins = " ".join(f"{b['fa']:.3f}" if b["fa"] is not None else "  -  " for b in m["fa_1S1L_by_rho_over_u0"])
        print(f"{name:12s} {f['fa_1S1L']:9.4f} {f['recall_1S2L']:9.4f} {f['recall_2S2L']:9.4f} {at['recall_1S2L']:11.3f} {m['mean_recall_1S2L_fa_le_0p3']:13.3f}   {bins}")
        for si, b in m.get("by_season", {}).items():
            a5 = next(a for a in b["recall_at_matched_fa"] if a["fa_target"] == 0.052)
            print(f"   season {si}: n={b['n']:6d} FA@frozen {b['frozen_threshold']['fa_1S1L']:.4f} rec@5.2%FA {a5['recall_1S2L']:.3f} (achieved FA {a5['fa']:.4f})")
    if f146:
        col = {"_doc": "Colour ablation on GULLS: three-band vs F146-only inputs for the same checkpoint on identical events.",
               "n_matched_dense": len(common), "models": {}}
        for name, d in f146.items():
            if name not in models:
                continue
            pa = np.array([models[name][i][1] for i in common]); pb = np.array([d[i][1] for i in common])
            col["models"][name] = {"three_band": summarise(pa, lab, ratio), "f146_only": summarise(pb, lab, ratio),
                                   "per_event_abs_dP_median": round(float(np.median(np.abs(pa - pb))), 4),
                                   "per_event_abs_dP_p90": round(float(np.percentile(np.abs(pa - pb), 90)), 4),
                                   "frac_decisions_flipped_at_frozen": round(float(((pa >= FROZEN) != (pb >= FROZEN)).mean()), 4)}
        col["command"] = " ".join(sys.argv)
        json.dump(col, open(os.path.join(args.out_dir, "transfer_colour_ablation.json"), "w"), indent=1)
        print("\ncolour ablation (recall at 5.2% FA, 1S2L/2S2L; mean recall FA<=0.3):")
        for name, c in col["models"].items():
            for tag in ("three_band", "f146_only"):
                at = next(a for a in c[tag]["recall_at_matched_fa"] if a["fa_target"] == 0.052)
                print(f"  {name:12s} {tag:10s} {at['recall_1S2L']:.3f}/{at['recall_2S2L']:.3f}   {c[tag]['mean_recall_1S2L_fa_le_0p3']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

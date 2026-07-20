"""
BinML v5 — standalone evaluation.

Produces the numbers a paper can defend, not the numbers that flatter the model.

Four decisions encoded here, each of which is easy to get wrong in the optimistic direction:

1. **The headline is completeness at a FIXED purity, not F1 and not recall.**
   Accuracy is meaningless (Flat+PSPL are ~61% of events). Bare recall is degenerate --
   predicting NonPSPL for everything scores 1.000, and an early smoke run of this project did
   exactly that. F1 is non-degenerate but sits at an arbitrary operating point nobody chose,
   which a survey team cannot act on. Completeness at a stated purity is what a
   follow-up-triggering pipeline is actually specified against. The threshold is fixed on a
   validation split and applied FROZEN to test.

2. **keep_prob reweighting is ASYMMETRIC, and applying it uniformly is wrong in both
   directions.** Every ``label == NonPSPL`` row has keep_prob = 1.0 by construction (demoted
   binaries are relabelled PSPL/Flat), so reweighting RECALL is a no-op. But the PSPL/Flat rows
   that supply the false positives were byproduct-subsampled 6.67x, and those rows are
   precisely the genuine binaries whose anomaly fell below threshold -- the population most
   likely to be predicted NonPSPL. So PRECISION and PURITY must be reweighted, or they are
   systematically optimistic by an amount that grows as the model improves. Both are reported,
   explicitly named ``_sample`` and ``_population``.

3. **Detectability conditioning.** Recall is reported binned by ``dchi2_anomaly`` and across
   the (log s, log q) plane, because a raw number mixes anomalies that are trivially visible
   with ones that are physically at the noise floor.

4. **Zero-support classes yield None, not 0.0.** ``diag / max(support, 1)`` silently returns a
   hard zero for an absent class, which then drags macro-F1 down as if the model had failed at
   something it was never asked to do.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .classes import CLASS_NAMES, N_CLASSES
from .model_v5 import BAND_BINS, BinMLv5, ModelConfigV5

I_NON = CLASS_NAMES.index("NonPSPL")
MAG_SCALE = 1.0


# ---------------------------------------------------------------------------------
# Data: a vectorised slab reader. Evaluation order is fixed, so there is no need for a
# Dataset/DataLoader with per-item numpy work -- read contiguous memmap slabs instead.
# ---------------------------------------------------------------------------------
def load_arrays(cache: str) -> Tuple[dict, dict, int]:
    meta = json.load(open(os.path.join(cache, "meta.json")))
    n = int(meta["n_events"])
    a = {}
    for b, L in BAND_BINS.items():
        a[f"feat/{b}"] = np.memmap(os.path.join(cache, f"feat_{b}.f16"), dtype=np.float16,
                                   mode="r", shape=(n, L, 3))
        a[f"frac/{b}"] = np.memmap(os.path.join(cache, f"frac_{b}.f16"), dtype=np.float16,
                                   mode="r", shape=(n, L))
    cols = {}
    for name in ("label", "true_class", "keep_prob", "dchi2_event", "dchi2_anomaly",
                 "m_base_ref", "a_ks"):
        p = os.path.join(cache, f"{name}.npy")
        if os.path.exists(p):
            cols[name] = np.load(p)
    pp = os.path.join(cache, "params.npy")
    if os.path.exists(pp):
        cols["params"] = np.load(pp)
        cols["param_fields"] = meta.get("param_fields")
    for b in BAND_BINS:
        p = os.path.join(cache, f"n_kept_{b}.npy")
        if os.path.exists(p):
            cols[f"n_kept_{b}"] = np.load(p)
    return a, cols, n


def _slab(a: dict, lo: int, hi: int) -> Dict[str, torch.Tensor]:
    """Build the (B, L, 5) input for each band directly from contiguous memmap rows."""
    out = {}
    for b, L in BAND_BINS.items():
        f = np.asarray(a[f"feat/{b}"][lo:hi], dtype=np.float32)      # (B, L, 3)
        fr = np.asarray(a[f"frac/{b}"][lo:hi], dtype=np.float32)     # (B, L)
        obs = np.isfinite(f[:, :, 0]).astype(np.float32)
        np.nan_to_num(f, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.concatenate([f / MAG_SCALE, fr[:, :, None], obs[:, :, None]], axis=2)
        # .clone() via torch.tensor: never hand mapped pages to an async device copy
        out[b] = torch.tensor(x, dtype=torch.float32)
    return out


@torch.inference_mode()
def predict(model, a: dict, idx: np.ndarray, device, batch: int = 1024) -> np.ndarray:
    """Return (len(idx), 6) float32 logits. Exactly one host transfer per batch."""
    model.eval()
    order = np.sort(idx)                       # contiguous reads
    logits = np.empty((len(order), N_CLASSES), dtype=np.float32)
    for s in range(0, len(order), batch):
        sel = order[s:s + batch]
        lo, hi = int(sel[0]), int(sel[-1]) + 1
        if hi - lo == len(sel):                # fully contiguous: slab it
            feats = _slab(a, lo, hi)
        else:                                  # gather (rare)
            feats = {}
            for b, L in BAND_BINS.items():
                f = np.asarray(a[f"feat/{b}"][sel], dtype=np.float32)
                fr = np.asarray(a[f"frac/{b}"][sel], dtype=np.float32)
                obs = np.isfinite(f[:, :, 0]).astype(np.float32)
                np.nan_to_num(f, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                feats[b] = torch.tensor(
                    np.concatenate([f / MAG_SCALE, fr[:, :, None], obs[:, :, None]], 2),
                    dtype=torch.float32)
        feats = {k: v.to(device) for k, v in feats.items()}
        present = {b: (feats[b][..., 4].sum(dim=1) > 0) for b in BAND_BINS}
        logits[s:s + len(sel)] = model(feats, present).float().cpu().numpy()
    # restore caller's ordering
    inv = np.empty(len(order), dtype=np.int64)
    inv[np.argsort(idx)] = np.arange(len(order))
    return logits[inv]


# ---------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------
def population_weights(keep_prob: np.ndarray) -> np.ndarray:
    """Weights that undo byproduct subsampling. NOT the training weights.

    ``train_v5.compute_weights`` deliberately mixes in class balancing, which is a training
    device; using it here would report a class-balanced fiction rather than the population.
    """
    # float64 is not optional. keep_prob is stored float32, and np.cumsum over ~4e5 float32
    # weights accumulates ~6e-4 relative error while np.sum uses pairwise summation and does
    # not -- so cumsum(w)/w.sum() overshoots 1. That produced a false-positive rate of
    # 1.000641 and an impossible AUC of 1.0005.
    return 1.0 / np.clip(keep_prob.astype(np.float64), 1e-3, 1.0)


def _prf(conf: np.ndarray) -> dict:
    support = conf.sum(1)
    predicted = conf.sum(0)
    diag = np.diag(conf)
    rec = [float(diag[i] / support[i]) if support[i] > 0 else None for i in range(N_CLASSES)]
    pre = [float(diag[i] / predicted[i]) if predicted[i] > 0 else None for i in range(N_CLASSES)]
    f1 = [None if (rec[i] is None or pre[i] is None or (rec[i] + pre[i]) == 0)
          else 2 * rec[i] * pre[i] / (rec[i] + pre[i]) for i in range(N_CLASSES)]
    got = [x for x in f1 if x is not None]
    return {
        "recall": {CLASS_NAMES[i]: rec[i] for i in range(N_CLASSES)},
        "precision": {CLASS_NAMES[i]: pre[i] for i in range(N_CLASSES)},
        "f1": {CLASS_NAMES[i]: f1[i] for i in range(N_CLASSES)},
        # macro-F1 over classes that are actually PRESENT; a class with no support would
        # otherwise contribute a hard 0.0 and look like a failure.
        "macro_f1": float(np.mean(got)) if got else None,
        "support": {CLASS_NAMES[i]: float(support[i]) for i in range(N_CLASSES)},
        "confusion": conf.tolist(),
    }


def confusion(y: np.ndarray, pred: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    k = y * N_CLASSES + pred
    if w is None:
        return np.bincount(k, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES).astype(np.float64)
    return np.bincount(k, weights=w, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)


def nonpspl_score(logits: np.ndarray) -> np.ndarray:
    """P(NonPSPL) under a softmax -- the score the operating point is chosen on."""
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e[:, I_NON] / e.sum(1)


def completeness_at_purity(score: np.ndarray, y: np.ndarray, w: np.ndarray,
                           target_purity: float = 0.90) -> dict:
    """Threshold giving >= target purity, and the completeness there.

    Purity uses POPULATION weights (false positives come from subsampled rows); completeness
    does not need them (all true NonPSPL have weight 1) but the same weights are applied for
    consistency and are a no-op on that subset.
    """
    is_pos = (y == I_NON)
    order = np.argsort(-score)
    s, p, ww = score[order], is_pos[order], w[order]
    tp = np.cumsum(p * ww)
    fp = np.cumsum((~p) * ww)
    purity = tp / np.maximum(tp + fp, 1e-12)
    total_pos = float((is_pos * w).sum())
    completeness = tp / max(total_pos, 1e-12)
    ok = np.nonzero(purity >= target_purity)[0]
    if len(ok) == 0:
        return {"target_purity": target_purity, "achievable": False,
                "best_purity": float(purity.max()) if len(purity) else None}
    i = ok[-1]                                   # deepest point still meeting purity
    return {"target_purity": target_purity, "achievable": True,
            "threshold": float(s[i]), "completeness": float(completeness[i]),
            "purity": float(purity[i]), "n_selected": int(i + 1)}


def average_precision(score: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    is_pos = (y == I_NON)
    order = np.argsort(-score)
    p, ww = is_pos[order], w[order]
    tp = np.cumsum(p * ww)
    fp = np.cumsum((~p) * ww)
    prec = tp / np.maximum(tp + fp, 1e-12)
    total = float((is_pos * w).sum())
    if total <= 0:
        return float("nan")
    rec = tp / total
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def recall_by_bins(mask_true: np.ndarray, correct: np.ndarray, x: np.ndarray,
                   edges) -> dict:
    out = {}
    for lo, hi in edges:
        m = mask_true & (x >= lo) & (x < hi)
        n = int(m.sum())
        out[f"{lo:g}-{hi:g}"] = {"recall": float(correct[m].mean()) if n else None, "n": n}
    return out


def efficiency_plane(params, pf, y, pred, tc, w, n_s=6, n_q=12) -> Optional[dict]:
    """Three-panel detection efficiency over (log s, log q).

    A: survey detectability  -- of GENERATED NonPSPL, the fraction the survey could see
       (i.e. that kept the NonPSPL label). Population-weighted, because the demoted rows are
       subsampled.
    B: end-to-end recovery   -- of GENERATED NonPSPL, the fraction predicted NonPSPL.
    C: classifier efficiency -- B/A, conditional on detectability. This is the panel that
       isolates "our model missed it" from "the survey physically could not see it".
    """
    if params is None or pf is None or "q" not in pf:
        return None
    iq, isx = pf.index("q"), pf.index("s")
    gen = (tc == I_NON) & np.isfinite(params[:, iq]) & np.isfinite(params[:, isx])
    if not gen.any():
        return None
    lq = np.log10(params[gen, iq]); ls = np.log10(params[gen, isx])
    det = (y[gen] == I_NON).astype(float)          # survey kept the anomaly label
    rec = (pred[gen] == I_NON).astype(float)       # model called it NonPSPL
    ww = w[gen]
    qe = np.linspace(-6, 0, n_q + 1); se = np.linspace(np.log10(0.2), np.log10(5.0), n_s + 1)
    A = np.full((n_q, n_s), np.nan); B = np.full((n_q, n_s), np.nan)
    C = np.full((n_q, n_s), np.nan); N = np.zeros((n_q, n_s))
    qi = np.clip(np.digitize(lq, qe) - 1, 0, n_q - 1)
    si = np.clip(np.digitize(ls, se) - 1, 0, n_s - 1)
    for i in range(n_q):
        for j in range(n_s):
            m = (qi == i) & (si == j)
            neff = ww[m].sum()
            N[i, j] = neff
            if neff < 30:                          # too few to quote
                continue
            A[i, j] = float((det[m] * ww[m]).sum() / neff)
            B[i, j] = float((rec[m] * ww[m]).sum() / neff)
            if A[i, j] > 0:
                C[i, j] = B[i, j] / A[i, j]
    return {"log_q_edges": qe.tolist(), "log_s_edges": se.tolist(),
            "survey_detectability": A.tolist(), "end_to_end_recovery": B.tolist(),
            "classifier_efficiency": C.tolist(), "n_eff": N.tolist()}


def bootstrap_ci(fn, n: int, reps: int = 200, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        s = rng.integers(0, n, n)
        v = fn(s)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return None
    return {"lo": float(np.percentile(vals, 100 * alpha / 2)),
            "hi": float(np.percentile(vals, 100 * (1 - alpha / 2))),
            "reps": len(vals)}


# ---------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------
DCHI2_EDGES = [(-np.inf, 160), (160, 2e3), (2e3, 1e4), (1e4, 1e5), (1e5, np.inf)]


def evaluate_checkpoint(ckpt_path: str, cache: str, out_dir: str, device: str = "mps",
                        batch: int = 1024, val_frac: float = 0.2, seed: int = 7,
                        target_purity: float = 0.90, bootstrap: int = 200,
                        max_events: int = 0) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg_fields = ModelConfigV5.__dataclass_fields__
    cfg = ModelConfigV5(**{k: v for k, v in ck["config"].items() if k in cfg_fields})
    model = BinMLv5(cfg).to(device)
    model.load_state_dict(ck["model"])

    a, cols, n = load_arrays(cache)
    y = cols["label"].astype(np.int64)
    tc = cols["true_class"].astype(np.int64)
    kp = cols["keep_prob"].astype(np.float64)
    w = population_weights(kp)
    params = cols.get("params")
    pf = cols.get("param_fields")

    idx = np.arange(n)
    if max_events and max_events < n:            # smoke-test / quick-look path
        idx = np.sort(np.random.default_rng(seed).permutation(n)[:max_events])
        y = y[idx]; tc = tc[idx]; kp = kp[idx]; w = w[idx]
        if params is not None:
            params = params[idx]
        for k in ("dchi2_anomaly", "dchi2_event", "m_base_ref", "a_ks"):
            if k in cols:
                cols[k] = cols[k][idx]
        for b in BAND_BINS:
            if f"n_kept_{b}" in cols:
                cols[f"n_kept_{b}"] = cols[f"n_kept_{b}"][idx]
        n = len(idx)
    logits = predict(model, a, idx, torch.device(device), batch=batch)
    pred_argmax = logits.argmax(1)
    score = nonpspl_score(logits)

    # Operating point is chosen on a VALIDATION slice of this cache and applied frozen to the
    # rest. Choosing it on the same rows it is scored on inflates completeness by the width of
    # the threshold's sampling noise.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(val_frac * n)
    vi, ti = perm[:n_val], perm[n_val:]
    op = completeness_at_purity(score[vi], y[vi], w[vi], target_purity)

    res: Dict[str, object] = {
        "checkpoint": os.path.abspath(ckpt_path),
        "checkpoint_sha256": hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()[:16],
        "cache": os.path.abspath(cache), "n_events": int(n),
        "trained_epoch": ck.get("epoch"), "val_f1_at_train": ck.get("val_nonpspl_f1"),
        "operating_point_from_val": op,
    }

    # --- argmax metrics, both conventions -------------------------------------
    res["argmax_sample"] = _prf(confusion(y[ti], pred_argmax[ti]))
    res["argmax_population"] = _prf(confusion(y[ti], pred_argmax[ti], w[ti]))

    # --- headline: completeness at fixed purity, threshold frozen from val ----
    if op.get("achievable"):
        thr = op["threshold"]
        sel = score[ti] >= thr
        pos = (y[ti] == I_NON)
        tp_w = float((w[ti] * (sel & pos)).sum())
        fp_w = float((w[ti] * (sel & ~pos)).sum())
        tot_w = float((w[ti] * pos).sum())
        res["headline"] = {
            "completeness_at_fixed_purity": tp_w / max(tot_w, 1e-12),
            "purity_achieved": tp_w / max(tp_w + fp_w, 1e-12),
            "target_purity": target_purity, "threshold": thr,
        }
    res["average_precision_population"] = average_precision(score[ti], y[ti], w[ti])

    # --- detectability conditioning -------------------------------------------
    mt = (y == I_NON)
    corr = (pred_argmax == I_NON)
    res["nonpspl_recall_by_dchi2"] = recall_by_bins(mt[ti], corr[ti],
                                                    cols["dchi2_anomaly"][ti], DCHI2_EDGES)
    # false positives per bin: where do spurious anomaly calls come from?
    fpmask = (~mt) & corr
    res["false_anomaly_by_true_class"] = {
        CLASS_NAMES[c]: {"n": int((fpmask[ti] & (y[ti] == c)).sum()),
                         "weighted": float((w[ti] * (fpmask[ti] & (y[ti] == c))).sum())}
        for c in range(N_CLASSES) if c != I_NON}

    # --- the (log s, log q) planes --------------------------------------------
    plane = efficiency_plane(params, pf, y, pred_argmax, tc, w)
    if plane is not None:
        res["efficiency_plane"] = plane

    # --- stratified slices ------------------------------------------------------
    slices = {}
    if params is not None and pf and "q" in pf:
        q = params[:, pf.index("q")]
        slices["planetary_q_lt_1e-2"] = np.isfinite(q) & (q < 1e-2)
        slices["deep_planetary_q_lt_1e-3"] = np.isfinite(q) & (q < 1e-3)
        slices["near_suzuki_break"] = np.isfinite(q) & (q > 1e-5) & (q < 1e-3)
    slices["faint_mbase_gt_22.5"] = cols["m_base_ref"] > 22.5
    slices["high_extinction_aks_gt_0.7"] = cols["a_ks"] > 0.7
    if "n_kept_F087" in cols:
        slices["no_blue_band"] = cols["n_kept_F087"] <= 0
    res["slices"] = {}
    for name, m in slices.items():
        mm = m[ti] & mt[ti]
        res["slices"][name] = {"n_true_nonpspl": int(mm.sum()),
                               "recall": float(corr[ti][mm].mean()) if mm.any() else None}

    # --- bootstrap CI on the headline -----------------------------------------
    if bootstrap and op.get("achievable"):
        st, yt, wt = score[ti], y[ti], w[ti]
        thr = op["threshold"]
        def f(s):
            sel = st[s] >= thr; pos = (yt[s] == I_NON)
            tw = float((wt[s] * (sel & pos)).sum()); tot = float((wt[s] * pos).sum())
            return tw / tot if tot > 0 else None
        res["headline_ci95"] = bootstrap_ci(f, len(ti), reps=bootstrap)

    # --- reusable per-event artifact ------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "logits.npy"), logits)
    np.save(os.path.join(out_dir, "score_nonpspl.npy"), score.astype(np.float32))
    for k in ("label", "true_class", "keep_prob", "dchi2_event", "dchi2_anomaly",
              "m_base_ref", "a_ks"):
        if k in cols:
            np.save(os.path.join(out_dir, f"{k}.npy"), cols[k])
    if params is not None:
        np.save(os.path.join(out_dir, "params.npy"), params)
    np.save(os.path.join(out_dir, "test_idx.npy"), ti)
    json.dump({"param_fields": pf, "checkpoint": res["checkpoint"],
               "checkpoint_sha256": res["checkpoint_sha256"], "cache": res["cache"]},
              open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    json.dump(res, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2, default=str)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--target-purity", type=float, default=0.90)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--max-events", type=int, default=0)
    args = ap.parse_args(argv)
    r = evaluate_checkpoint(args.ckpt, args.cache, args.out, args.device, args.batch,
                            target_purity=args.target_purity, bootstrap=args.bootstrap,
                            max_events=args.max_events)
    print(json.dumps({k: v for k, v in r.items()
                      if k not in ("efficiency_plane",)}, indent=2, default=str)[:3000])
    print(f"\nartifact -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

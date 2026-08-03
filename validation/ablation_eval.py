#!/usr/bin/env python3
"""Evaluate the detectability-conditioned-labelling ablation.

Two models were trained on identical data, differing only in the label source:
  A = observational (detectability-conditioned) labels   -- what BinML ships with
  B = generator-intent labels (true_class)               -- the ablation
Both are here evaluated on the SAME held-out test split against the OBSERVATIONAL labels (the
honest ground truth: an undetectable anomaly is observationally a single lens). The split is
reproduced from the training seed, so it is identical to what training held out.

Key question: does labelling by generator intent teach the model to assert anomalies with no
observable evidence? If so, model B should flag far more non-anomalous events as NonPSPL
(a higher false-anomaly rate / lower NonPSPL precision) than model A.

Usage:  python ablation_eval.py <cache_dir> <ckpt_A> <ckpt_B> [seed]
"""
from __future__ import annotations
import sys, os, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.model import BAND_BINS, BinMLv5, ModelConfigV5
CLASSES = ["Flat", "PSPL", "NonPSPL", "PeriodicVar", "LongPeriodVar", "Eruptive"]
NON = CLASSES.index("NonPSPL")
MAG_SCALE = 1.0


def load_model(ckpt):
    ck = torch.load(ckpt, map_location="cpu")
    cfg = ModelConfigV5(**{k: v for k, v in ck["config"].items()
                           if k in ModelConfigV5.__dataclass_fields__})
    net = BinMLv5(cfg); net.load_state_dict(ck["model"]); net.eval()
    return net


def test_indices(n, seed):
    perm = np.random.default_rng(seed).permutation(n)
    n_tr, n_va = int(0.8 * n), int(0.1 * n)
    return perm[n_tr + n_va:]


def predict(net, arrays, idx, bs=2048):
    preds = np.zeros(len(idx), int)
    for s in range(0, len(idx), bs):
        j = idx[s:s + bs]
        feats, present = {}, {}
        for b, L in BAND_BINS.items():
            x = np.asarray(arrays[f"feat/{b}"][j], np.float32)          # (B,L,3)
            fr = np.asarray(arrays[f"frac/{b}"][j], np.float32)
            obs = np.isfinite(x[:, :, 0]).astype(np.float32)
            x = np.nan_to_num(x, nan=0.0) / MAG_SCALE
            t = np.concatenate([x, fr[:, :, None], obs[:, :, None]], axis=2)
            feats[b] = torch.tensor(t, dtype=torch.float32)
            present[b] = feats[b][..., 4].sum(1) > 0
        with torch.inference_mode():
            lg = net(feats, present).float().numpy()
        preds[s:s + bs] = lg.argmax(1)
    return preds


def metrics(pred, y, w):
    out = {}
    for c in range(6):
        tp = (w * ((pred == c) & (y == c))).sum()
        fp = (w * ((pred == c) & (y != c))).sum()
        fn = (w * ((pred != c) & (y == c))).sum()
        r = tp / max(tp + fn, 1e-9); p = tp / max(tp + fp, 1e-9)
        out[CLASSES[c]] = dict(recall=round(float(r), 3), precision=round(float(p), 3),
                               f1=round(float(2 * r * p / max(r + p, 1e-9)), 3))
    # false-anomaly rate: of truly non-anomalous events, fraction predicted NonPSPL (weighted)
    nonanom = y != NON
    far = (w * ((pred == NON) & nonanom)).sum() / max((w * nonanom).sum(), 1e-9)
    out["_false_anomaly_rate"] = round(float(far), 4)
    return out


def main():
    cache, ckA, ckB = sys.argv[1], sys.argv[2], sys.argv[3]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 20260720
    meta = json.load(open(os.path.join(cache, "meta.json")))
    n = meta["n_events"]
    arrays = {}
    for b, L in BAND_BINS.items():
        arrays[f"feat/{b}"] = np.memmap(os.path.join(cache, f"feat_{b}.f16"), np.float16, "r", shape=(n, L, 3))
        arrays[f"frac/{b}"] = np.memmap(os.path.join(cache, f"frac_{b}.f16"), np.float16, "r", shape=(n, L))
    y = np.load(os.path.join(cache, "label.npy"))            # OBSERVATIONAL truth for both
    kp = np.load(os.path.join(cache, "keep_prob.npy")).astype(np.float64)
    w = 1.0 / np.clip(kp, 1e-3, 1.0)
    te = test_indices(n, seed)
    print(f"test split: {len(te)} events (seed {seed})\n")
    for name, ck in [("A  observational (shipped)", ckA), ("B  generator-intent (ablation)", ckB)]:
        net = load_model(ck)
        pred = predict(net, arrays, te)
        m = metrics(pred, y[te], w[te])
        print(f"=== model {name} ===")
        print(f"  NonPSPL: recall {m['NonPSPL']['recall']}  precision {m['NonPSPL']['precision']}  f1 {m['NonPSPL']['f1']}")
        print(f"  false-anomaly rate (non-anomalous flagged NonPSPL): {m['_false_anomaly_rate']}")
        macro = np.mean([m[c]['f1'] for c in CLASSES])
        print(f"  macro-F1: {macro:.3f}\n")


if __name__ == "__main__":
    main()

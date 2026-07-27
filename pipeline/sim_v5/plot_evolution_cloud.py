"""Standalone per-event probability-evolution plotter (cloud worker).

Picks N random events of ONE class from a binned cache shard, runs the checkpoint over
progressively-revealed seasons, and writes the 3-panel figure (light curve / class
probabilities / max-prob confidence) per event. Self-contained: needs only the cache h5 and
the checkpoint, no eval artifact.
"""
from __future__ import annotations
import argparse, os
import numpy as np, torch, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .model_v5 import BAND_BINS, BinMLv5, ModelConfigV5
from .classes import CLASS_NAMES, N_CLASSES

COL = ["#7f7f7f", "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
MAG_SCALE = 1.0

def _softmax(z): e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

def _load(ckpt, dev):
    ck = torch.load(ckpt, map_location="cpu")
    cfg = ModelConfigV5(**{k: v for k, v in ck["config"].items() if k in ModelConfigV5.__dataclass_fields__})
    m = BinMLv5(cfg).to(dev); m.load_state_dict(ck["model"]); m.eval(); return m

@torch.inference_mode()
def _scan(m, feat, frac, row, dev, n_steps=36):
    fr = np.linspace(1.0/n_steps, 1.0, n_steps); out = np.zeros((n_steps, N_CLASSES), np.float32)
    for k, f in enumerate(fr):
        feats, pres = {}, {}
        for b, L in BAND_BINS.items():
            cut = int(round(f*L))
            x = np.asarray(feat[b][row:row+1], np.float32).copy()
            fc = np.asarray(frac[b][row:row+1], np.float32).copy()
            x[:, cut:, :] = 0.0; fc[:, cut:] = 0.0
            obs = np.isfinite(x[:, :, 0]).astype(np.float32); obs[:, cut:] = 0.0
            np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            t = np.concatenate([x/MAG_SCALE, fc[:, :, None], obs[:, :, None]], 2)
            feats[b] = torch.tensor(t, dtype=torch.float32, device=dev); pres[b] = feats[b][..., 4].sum(1) > 0
        out[k] = _softmax(m(feats, pres).float().cpu().numpy())[0]
    return fr*72.0, out

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--class-idx", type=int, required=True); ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    m = _load(a.ckpt, a.device)
    f = h5py.File(a.cache, "r"); n = int(f.attrs["n_events"])
    y = np.asarray(f["label"]).astype(int); mb = np.asarray(f["m_base_ref"])
    feat = {b: f[f"feat/{b}"] for b in BAND_BINS}; frac = {b: f[f"frac/{b}"] for b in BAND_BINS}
    rng = np.random.default_rng(a.seed)
    idx = np.nonzero(y == a.class_idx)[0]
    pick = rng.choice(idx, min(a.n, len(idx)), replace=False)
    cols = {"F146": "#1f77b4", "F087": "#2ca02c", "F213": "#d62728"}
    made = 0
    for row in pick:
        row = int(row)
        days, probs = _scan(m, feat, frac, row, a.device)
        pred = int(probs[-1].argmax())
        fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9, 7.6), sharex=True)
        for b in BAND_BINS:
            x = np.asarray(feat[b][row, :, 0], np.float32); t = np.linspace(0, 72, len(x), endpoint=False)
            v = np.isfinite(x)
            if v.any(): a1.plot(t[v], mb[row]+x[v], ".", ms=2.0, color=cols[b], label=b)
        a1.invert_yaxis(); a1.set_ylabel("AB magnitude"); a1.legend(fontsize=7, markerscale=4, ncol=3)
        ok = pred == a.class_idx
        a1.set_title(f"{CLASS_NAMES[a.class_idx]} -> {CLASS_NAMES[pred]} (p={probs[-1,pred]:.2f}) "
                     f"— {'correct' if ok else 'MISCLASSIFIED'}", color="green" if ok else "red")
        for c in range(N_CLASSES): a2.plot(days, probs[:, c], "-", lw=1.6, color=COL[c], label=CLASS_NAMES[c])
        a2.axhline(1.0/N_CLASSES, color="gray", ls="--", lw=0.8, alpha=0.6)
        a2.set_ylabel("class probability"); a2.set_ylim(0, 1.05); a2.legend(fontsize=7, ncol=3)
        conf = probs.max(1); a3.plot(days, conf, "-", color="black", lw=1.6)
        a3.fill_between(days, 0, conf, alpha=0.25, color="gray")
        fin = probs[-1].argmax(); agree = probs.argmax(1) == fin; st = len(agree)-1
        while st > 0 and agree[st-1]: st -= 1
        a3.axvline(days[st], color="#d62728", ls=":", lw=1.2, label=f"commits to {CLASS_NAMES[fin]} at day {days[st]:.0f}")
        a3.set_ylabel("max probability"); a3.set_xlabel("days of season observed"); a3.set_ylim(0, 1.05); a3.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"evolution_{CLASS_NAMES[a.class_idx]}_{row}.png"), dpi=100); plt.close(fig)
        made += 1
    print(f"made {made} plots for {CLASS_NAMES[a.class_idx]}", flush=True)

if __name__ == "__main__": main()

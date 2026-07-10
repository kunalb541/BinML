#!/usr/bin/env python3
"""Local per-class recall on a compact preset test .h5, for a given checkpoint.
Replicates evaluate.load_and_prepare_data compaction + normalization exactly."""
import sys, argparse, numpy as np, torch, h5py
sys.path.insert(0, "/Users/kunalbhatia/Desktop/Research/microlensing/binml/code")
from model import ModelConfig, RomanMicrolensingClassifier
EPS = 1e-8; CN = ["Flat", "PSPL", "Binary"]


def load_model(p, device="cpu"):
    ck = torch.load(p, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    m = RomanMicrolensingClassifier(cfg).to(device)
    sd = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in ck["model_state_dict"].items()}
    m.load_state_dict(sd, strict=True); m.eval()
    s = ck["stats"]
    fm = s.get("flux_mean", s.get("magnification_mean")); fs = s.get("flux_std", s.get("magnification_std"))
    return m, (fm, fs, s["delta_t_mean"], s["delta_t_std"])


def eval_file(model, stats, path, device="cpu", bs=512):
    fm, fs, dm, ds = stats
    with h5py.File(path, "r") as f:
        flux = f["flux"][:]; dt = f["delta_t"][:]; lab = f["labels"][:]
    n, seq = flux.shape
    fc = np.zeros_like(flux); dc = np.zeros_like(dt); vl = np.zeros(n, np.int64)
    for i in range(n):
        vm = flux[i] != 0.0; k = int(vm.sum())
        if k == 0:
            fc[i, 0] = fm; dc[i, 0] = 0.0; vl[i] = 1
        else:
            fc[i, :k] = flux[i, vm]; dc[i, :k] = dt[i, vm]; vl[i] = k
    fn = (fc - fm) / (fs + EPS); dn = (dc - dm) / (ds + EPS)
    preds = np.zeros(n, np.int64)
    with torch.no_grad():
        for s0 in range(0, n, bs):
            e = min(s0 + bs, n)
            xf = torch.from_numpy(fn[s0:e]).float().to(device)
            xd = torch.from_numpy(dn[s0:e]).float().to(device)
            xl = torch.from_numpy(vl[s0:e]).to(device)
            out = model(xf, xd, xl)
            logit = out if torch.is_tensor(out) else out["logits"]
            preds[s0:e] = logit.argmax(-1).cpu().numpy()
    rec = {}
    for c in range(3):
        m = lab == c
        rec[CN[c]] = float((preds[m] == c).mean()) if m.sum() else float("nan")
    acc = float((preds == lab).mean())
    return acc, rec, lab, preds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    a = ap.parse_args()
    model, stats = load_model(a.ckpt)
    for path in a.files:
        name = path.split("/")[-1].replace("test_", "").replace(".h5", "")
        acc, rec, _, _ = eval_file(model, stats, path)
        print(f"{name:10} acc={acc*100:5.1f}%  Flat={rec['Flat']*100:5.1f}  "
              f"PSPL={rec['PSPL']*100:5.1f}  Binary={rec['Binary']*100:5.1f}")

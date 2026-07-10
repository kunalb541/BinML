#!/usr/bin/env python3
"""
Run the trained Roman classifier on REAL OGLE ground-based photometry.

Faithful mapping to the model's training representation (simulate.py v4.2.0):
  flux = normalized magnification A = 10**(0.4*(m_base - mag)), baseline A=1.0
  delta_t = days since previous valid observation (first = 0)
  The model consumes COMPACTED sequences (valid obs moved to a contiguous prefix),
  so we build the time-ordered (A, delta_t) sequence directly, then normalize with
  the checkpoint's own stats and forward through the net.

HONEST CAVEAT: the model was trained on Roman space cadence (dense 15-min sampling,
space photon noise). OGLE is ground-based (irregular, night/season gaps, seeing).
This is a deliberate domain-shift test.
"""
import sys, os, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = "/Users/kunalbhatia/Desktop/Research/microlensing/binml/code"
sys.path.insert(0, CODE)
from model import ModelConfig, RomanMicrolensingClassifier

CLASS_NAMES = ["Flat", "PSPL", "Binary"]
EPS = 1e-8
WINDOW_DAYS = 72.0   # match the Roman season window length used in training


def load_ogle(path):
    """OGLE EWS phot.dat: HJD  mag  mag_err  seeing  sky."""
    a = np.loadtxt(path)
    hjd, mag, err = a[:, 0], a[:, 1], a[:, 2]
    o = np.argsort(hjd)
    return hjd[o], mag[o], err[o]


def preprocess(hjd, mag, err, window=WINDOW_DAYS, t0=None):
    """Return compacted (flux, delta_t) [1, seq], valid_length, and the windowed
    raw arrays for plotting. m_base is the robust out-of-event baseline."""
    # 1) locate the event: brightest point (min mag)
    if t0 is None:
        t0 = hjd[np.argmin(mag)]
    # 2) baseline from OUT-of-event points (exclude +-60 d around t0)
    out = np.abs(hjd - t0) > 60.0
    m_base = np.median(mag[out]) if out.sum() > 20 else np.percentile(mag, 90)
    # 3) window to +-window/2 around t0
    inwin = np.abs(hjd - t0) <= window / 2.0
    tw, mw, ew = hjd[inwin], mag[inwin], err[inwin]
    order = np.argsort(tw)
    tw, mw, ew = tw[order], mw[order], ew[order]
    # 4) magnification A (baseline = 1.0); clip A>=~0 (mag brighter than base -> A>1)
    A = 10.0 ** (0.4 * (m_base - mw))
    n = len(A)
    seq = 6912
    if n > seq:
        # keep the densest central seq points (shouldn't happen for OGLE windows)
        A, tw, mw, ew = A[:seq], tw[:seq], mw[:seq], ew[:seq]
        n = seq
    flux = np.zeros((1, seq), np.float32)
    dt = np.zeros((1, seq), np.float32)
    flux[0, :n] = A
    dt[0, 0] = 0.0
    dt[0, 1:n] = np.diff(tw).astype(np.float32)
    return flux, dt, n, dict(t0=float(t0), m_base=float(m_base),
                             tw=tw, mw=mw, ew=ew, A=A, window=window)


def load_model(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["model_config"]) if not isinstance(ck["model_config"], ModelConfig) else ck["model_config"]
    model = RomanMicrolensingClassifier(cfg).to(device)
    sd = ck["model_state_dict"]
    sd = { (k[10:] if k.startswith("_orig_mod.") else k): v for k, v in sd.items() }
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, ck["stats"], ck.get("epoch")


def infer(model, stats, flux, dt, valid_len, device="cpu"):
    fmean = stats.get("flux_mean", stats.get("magnification_mean"))
    fstd = stats.get("flux_std", stats.get("magnification_std"))
    fn = (flux - fmean) / (fstd + EPS)
    dn = (dt - stats["delta_t_mean"]) / (stats["delta_t_std"] + EPS)
    x_flux = torch.from_numpy(fn).float().to(device)
    x_dt = torch.from_numpy(dn).float().to(device)
    vl = torch.tensor([valid_len], dtype=torch.long, device=device)
    with torch.no_grad():
        # try common forward signatures
        try:
            out = model(x_flux, x_dt, vl)
        except TypeError:
            x = torch.stack([x_flux, x_dt], dim=1)  # [B,2,seq]
            out = model(x, vl)
    logits = out["logits"] if isinstance(out, dict) and "logits" in out else (
             out["stage_logits"] if isinstance(out, dict) and "stage_logits" in out else out)
    if isinstance(out, dict):
        for k in ("logits", "class_logits", "final_logits", "output"):
            if k in out:
                logits = out[k]; break
    probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()[0]
    return probs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phot", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", default="event")
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    hjd, mag, err = load_ogle(args.phot)
    flux, dt, n, meta = preprocess(hjd, mag, err, t0=args.t0)
    model, stats, epoch = load_model(args.ckpt)
    probs = infer(model, stats, flux, dt, n)
    pred = int(np.argmax(probs))
    print(json.dumps({
        "event": args.name, "ckpt_epoch": epoch,
        "n_points_in_window": int(n), "t0_HJD": meta["t0"], "m_base": meta["m_base"],
        "peak_A": float(meta["A"].max()),
        "median_dt_days": float(np.median(np.diff(meta["tw"]))),
        "probs": {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(3)},
        "prediction": CLASS_NAMES[pred],
    }, indent=2))

    # plot: light curve (mag) + prediction bar
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 1]})
    a1.errorbar(meta["tw"] - meta["t0"], meta["mw"], yerr=meta["ew"], fmt=".", ms=3,
                elinewidth=0.5, alpha=0.6, color="#1f77b4")
    a1.axhline(meta["m_base"], ls="--", color="gray", lw=1, label=f"baseline {meta['m_base']:.2f}")
    a1.invert_yaxis()
    a1.set_xlabel("days from peak"); a1.set_ylabel("OGLE I mag")
    a1.set_title(f"{args.name}  (real OGLE data, {n} pts in {int(meta['window'])}d window)  "
                 f"-> pred: {CLASS_NAMES[pred]}")
    a1.legend()
    colors = ["#9aa0a6", "#edae49", "#2e8b57"]
    a2.barh(CLASS_NAMES, probs, color=colors, edgecolor="black")
    a2.set_xlim(0, 1); a2.set_xlabel("model probability")
    for i, p in enumerate(probs):
        a2.text(min(p + 0.02, 0.9), i, f"{p:.2f}", va="center", fontweight="bold")
    plt.tight_layout()
    op = os.path.join(args.out, f"real_{args.name}.png")
    plt.savefig(op, dpi=130, bbox_inches="tight")
    print("saved", op)

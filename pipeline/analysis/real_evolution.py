#!/usr/bin/env python3
"""Probability-evolution on REAL OGLE curves: feed progressively longer prefixes of the
event (first k observations, length=k) and record class probabilities. The model is
causal, so a prefix prediction only ever sees data up to that time (verified in audit)."""
import sys, os, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/Users/kunalbhatia/Desktop/Research/microlensing/binml/code")
import importlib.util
spec = importlib.util.spec_from_file_location("infer_real",
    "/private/tmp/claude-501/-Users-kunalbhatia-Desktop-Research-microlensing/27fcbf76-7077-4777-82da-d78e9a275394/scratchpad/realdata/infer_real.py")
IR = importlib.util.module_from_spec(spec); spec.loader.exec_module(IR)

CN = ["Flat", "PSPL", "Binary"]; COL = ["#9aa0a6", "#edae49", "#2e8b57"]


def evolve(model, stats, flux, dt, n, step=1, kmin=3):
    ks = list(range(kmin, n + 1, step))
    if ks[-1] != n:
        ks.append(n)
    P = np.zeros((len(ks), 3))
    for i, k in enumerate(ks):
        fk = np.zeros_like(flux); dk = np.zeros_like(dt)
        fk[0, :k] = flux[0, :k]; dk[0, :k] = dt[0, :k]
        P[i] = IR.infer(model, stats, fk, dk, k)
    return np.array(ks), P


def plot_event(model, stats, phot, name, out, t0=None, step=1):
    hjd, mag, err = IR.load_ogle(phot)
    flux, dt, n, meta = IR.preprocess(hjd, mag, err, t0=t0)
    ks, P = evolve(model, stats, flux, dt, n, step=max(1, n // 200))
    tw, t0v = meta["tw"], meta["t0"]
    xt = tw[ks - 1] - t0v                       # time (days from peak) at the k-th obs
    pred = int(np.argmax(P[-1]))
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 2, 1.4]})
    ax[0].errorbar(tw - t0v, meta["mw"], yerr=meta["ew"], fmt=".", ms=3, elinewidth=0.4,
                   alpha=0.6, color="#1f77b4")
    ax[0].axhline(meta["m_base"], ls="--", color="gray", lw=0.8)
    ax[0].invert_yaxis(); ax[0].set_ylabel("OGLE I mag")
    ax[0].set_title(f"{name}  (real OGLE, {n} pts)  ->  final pred: {CN[pred]} {P[-1,pred]:.2f}")
    for c in range(3):
        ax[1].plot(xt, P[:, c], color=COL[c], lw=2, label=CN[c])
    ax[1].axhline(1/3, ls=":", color="gray", lw=0.8)
    ax[1].set_ylabel("class probability"); ax[1].set_ylim(-0.02, 1.02)
    ax[1].legend(loc="center left", fontsize=9, ncol=3)
    ax[2].fill_between(xt, P.max(1), color="black", alpha=0.12)
    ax[2].plot(xt, P.max(1), color="black", lw=1.5)
    ax[2].set_ylabel("confidence"); ax[2].set_ylim(0.3, 1.02)
    ax[2].set_xlabel("days from peak (time of latest observation fed)")
    plt.tight_layout()
    op = os.path.join(out, f"evo_real_{name}.png")
    plt.savefig(op, dpi=130, bbox_inches="tight"); plt.close()
    print(f"{name:22} n={n:4d}  ->  {CN[pred]} {P[-1,pred]:.2f}   saved {os.path.basename(op)}")
    return op


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model, stats, epoch = IR.load_model(a.ckpt)
    R = "/private/tmp/claude-501/-Users-kunalbhatia-Desktop-Research-microlensing/27fcbf76-7077-4777-82da-d78e9a275394/scratchpad/realdata"
    events = [
        (f"{R}/bin/b_2014_0289.dat", "OGLE-2014-BLG-0289_binary"),
        (f"{R}/bin/b_2013_0578.dat", "OGLE-2013-BLG-0578_binary"),
        (f"{R}/blg0060.dat",         "OGLE-2015-BLG-0060_binary"),
        (f"{R}/cand_0033.dat",       "OGLE-2015-BLG-0033_PSPL"),
    ]
    for phot, name in events:
        plot_event(model, stats, phot, name, a.out)

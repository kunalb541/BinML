#!/usr/bin/env python3
"""Show a small PLANETARY bump in a real OGLE event and a synthetic one, side by side:
raw light curve + best-fit single-lens (PSPL) model + residual (data - PSPL) that exposes
the planetary anomaly. Plus the model's class probabilities."""
import sys, os, numpy as np, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/Users/kunalbhatia/Desktop/Research/microlensing/binml/code")
import importlib.util
R = "/private/tmp/claude-501/-Users-kunalbhatia-Desktop-Research-microlensing/27fcbf76-7077-4777-82da-d78e9a275394/scratchpad/realdata"
spec = importlib.util.spec_from_file_location("infer_real", f"{R}/infer_real.py")
IR = importlib.util.module_from_spec(spec); spec.loader.exec_module(IR)
PR = importlib.util.module_from_spec(importlib.util.spec_from_file_location("pr", f"{R}/pspl_residual.py"))
importlib.util.spec_from_file_location("pr", f"{R}/pspl_residual.py").loader.exec_module(PR)
CN = ["Flat", "PSPL", "Binary"]


def load_synth_planet(h5, want_q=(1e-4, 2e-3), want_ma=(0.12, 0.45), seed=0):
    """Pick one synthetic planetary binary with a small but VISIBLE bump (max_anomaly in a
    range that stands above per-point noise), low q. Returns t,mag,err + truth meta."""
    f = h5py.File(h5, "r")
    lb = f["labels"][:]; P = f["params"]
    q = P["q"][:]; an = P["anomaly_dchi2"][:]; ma = P["max_anomaly"][:]
    cand = np.where((lb == 2) & (q >= want_q[0]) & (q < want_q[1]) &
                    (ma >= want_ma[0]) & (ma < want_ma[1]))[0]
    rng = np.random.RandomState(seed); i = int(cand[rng.randint(len(cand))])
    tg = f["time_grid"][:]; A = f["flux"][i]; mb = float(f["m_base"][i])
    valid = A != 0
    t = tg[valid]; Av = A[valid]
    mag = mb - 2.5 * np.log10(np.clip(Av, 1e-6, None))
    err = np.full_like(mag, 0.01)  # synthetic: nominal small error for the fit weighting
    meta = dict(q=float(q[i]), anomaly_dchi2=float(an[i]), max_anomaly=float(ma[i]),
                t0=float(P["t0"][i]), tE=float(P["tE"][i]), u0=float(P["u0"][i]),
                s=float(P["s"][i]), row=i, m_base=mb, flux=A[None, :], dt=f["delta_t"][i][None, :])
    f.close()
    return t, mag, err, meta


def fit_and_resid(t, m, e, base=None, window=72.0):
    from scipy.optimize import curve_fit
    t0g = t[np.argmin(m)]
    if base is None:
        out = np.abs(t - t0g) > 40
        base = np.median(m[out]) if out.sum() > 20 else np.percentile(m, 90)
    A = 10 ** (0.4 * (base - m)); Aerr = A * 0.4 * np.log(10) * e
    p0 = [t0g, 25.0, max(1.0 / max(A.max(), 1.1), 1e-3), 1.0]
    try:
        popt, _ = curve_fit(PR.A_pspl, t, A, p0=p0, sigma=Aerr, absolute_sigma=True,
                            bounds=([t0g - 5, 1, 1e-4, 0.2], [t0g + 5, 300, 3, 1.2]), maxfev=30000)
    except Exception:
        popt = p0
    Amod = PR.A_pspl(t, *popt)
    mmod = base - 2.5 * np.log10(np.clip(Amod, 1e-6, None))
    return base, popt, mmod, (m - mmod), (m - mmod) / e


def _bin(x, y, nb=160):
    o = np.argsort(x); x, y = x[o], y[o]
    edges = np.linspace(x.min(), x.max(), nb + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nb - 1)
    bx, by = [], []
    for k in range(nb):
        sel = idx == k
        if sel.sum():
            bx.append(x[sel].mean()); by.append(np.median(y[sel]))
    return np.array(bx), np.array(by)


def panel(ax_lc, ax_res, t, m, e, mmod, resid, chi, t0, title, probs, extra="", dense=False):
    x = t - t0
    o = np.argsort(x)
    if dense:                                   # bin the raw curve so the bump shows above noise
        bx, bm = _bin(x, m); ax_lc.plot(bx, bm, ".", ms=4, color="#1f77b4", alpha=0.8)
    else:
        ax_lc.errorbar(x, m, yerr=e, fmt=".", ms=3, elinewidth=0.4, alpha=0.55, color="#1f77b4", zorder=1)
    ax_lc.plot(x[o], mmod[o], "-", color="crimson", lw=1.6, label="best-fit single-lens (PSPL)", zorder=3)
    ax_lc.invert_yaxis(); ax_lc.set_ylabel("mag")
    pred = int(np.argmax(probs))
    ax_lc.set_title(f"{title}\nmodel: {CN[pred]} {probs[pred]:.2f}  (F{probs[0]:.2f}/P{probs[1]:.2f}/B{probs[2]:.2f}){extra}", fontsize=10)
    ax_lc.legend(fontsize=8, loc="upper right")
    ax_res.axhline(0, color="gray", lw=0.8)
    if dense:                                   # binned residual reveals the planetary bump cleanly
        ax_res.plot(x, resid, ".", ms=1, color="#bbb", alpha=0.3)
        bx, br = _bin(x, resid)
        ax_res.plot(bx, br, "-", color="crimson", lw=1.8, label="planetary anomaly (binned residual)")
        ax_res.legend(fontsize=8, loc="upper right")
    else:
        anom = (np.abs(chi) > 2) & (np.abs(x) <= 20)
        ax_res.errorbar(x, resid, yerr=e, fmt=".", ms=3, elinewidth=0.4, alpha=0.5, color="#555")
        ax_res.plot(x[anom], resid[anom], "o", ms=5, mfc="none", mec="crimson", mew=1.3,
                    label=f"planetary anomaly ({anom.sum()} pts >2σ)")
        if anom.sum():
            ax_res.legend(fontsize=8, loc="upper right")
    ax_res.set_ylabel("data − PSPL"); ax_res.set_xlabel("days from peak")
    ax_res.invert_yaxis()


if __name__ == "__main__":
    model, stats, epoch = IR.load_model(f"{R}/best.pt")
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), gridspec_kw={"height_ratios": [3, 1.4]})

    # ---- REAL: OGLE-2017-BLG-0482 (super-Earth) ----
    t, m, e = PR.load(f"{R}/planet/p_2017_0482.dat")
    win = np.abs(t - t[np.argmin(m)]) <= 36
    t, m, e = t[win], m[win], e[win]
    base, popt, mmod, resid, chi = fit_and_resid(t, m, e)
    fx, dtx, n, meta = IR.preprocess(*PR.load(f"{R}/planet/p_2017_0482.dat"))
    probs = IR.infer(model, stats, fx, dtx, n)
    panel(axes[0, 0], axes[1, 0], t, m, e, mmod, resid, chi, popt[0],
          f"REAL  OGLE-2017-BLG-0482Lb (super-Earth)  tE={popt[1]:.0f}d u0={popt[2]:.3f}",
          probs)

    # ---- SYNTHETIC: matched planetary event ----
    T = "/private/tmp/claude-501/-Users-kunalbhatia-Desktop-Research-microlensing/27fcbf76-7077-4777-82da-d78e9a275394/scratchpad/testset/test_planetary.h5"
    ts, ms, es, sm = load_synth_planet(T, seed=3)
    base2, popt2, mmod2, resid2, chi2 = fit_and_resid(ts, ms, es, base=sm["m_base"])
    ps = IR.infer(model, stats, sm["flux"].astype(np.float32), sm["dt"].astype(np.float32),
                  int((sm["flux"][0] != 0).sum()))
    panel(axes[0, 1], axes[1, 1], ts, ms, es, mmod2, resid2, chi2, sm["t0"],
          f"SYNTHETIC planetary (Roman cadence)  q={sm['q']:.1e} s={sm['s']:.2f} tE={sm['tE']:.0f}d",
          ps, extra=f"  max bump={sm['max_anomaly']*100:.0f}%", dense=True)

    plt.tight_layout()
    out = "/Users/kunalbhatia/Desktop/microlensing_results/planetary_bump_real_vs_synth.png"
    plt.savefig(out, dpi=135, bbox_inches="tight")
    print("REAL OGLE-2017-BLG-0482 probs:", {CN[i]: round(float(probs[i]),3) for i in range(3)})
    print("SYNTH planetary probs:", {CN[i]: round(float(ps[i]),3) for i in range(3)},
          "q=%.1e anomaly_dchi2=%.0f max_anom=%.3f" % (sm["q"], sm["anomaly_dchi2"], sm["max_anomaly"]))
    print("saved", out)

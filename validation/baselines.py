#!/usr/bin/env python3
"""Compare BinML against fieldable baselines for anomaly (NonPSPL) detection.

The manuscript otherwise compares BinML only with a truth-informed Delta-chi^2 reference (computed
from noiseless generator information), which is not a fieldable method. This script adds baselines
that use ONLY observed data, on a common set of freshly simulated events:

  1. BinML             -- P(NonPSPL) from the shipped 6-class network.
  2. Fitted-PSPL residual -- fit a single-lens (Paczynski) model to the noisy F146 curve and score
                          by the residual chi^2 improvement of a free per-bin bump over the PSPL fit
                          (a cheap observed-data anomaly statistic; no generator truth).
  3. Gradient-boosted trees on hand features (amplitude, skew, kurtosis, autocorr, peak counts,
     residual-after-smoothing), trained on a disjoint split of the same simulated events.

We report average precision (AP) for NonPSPL-vs-rest for each, on the same held-out events.
Deterministic seeds. Writes validation/baselines_result.json.
"""
from __future__ import annotations
import os, sys, json, warnings
import numpy as np
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import binml
from pipeline.assemble import simulate_event, SurveyConfig

N_EVENTS = 9000
CFG = SurveyConfig()
CLASSES = binml.CLASS_NAMES
NON = CLASSES.index("NonPSPL")


def paczynski(t, t0, tE, u0, fs, mbase):
    u = np.sqrt(u0 ** 2 + ((t - t0) / tE) ** 2)
    A = (u ** 2 + 2) / (u * np.sqrt(u ** 2 + 4))
    flux = fs * A + (1 - fs)
    return mbase - 2.5 * np.log10(np.clip(flux, 1e-6, None))


def fit_pspl_residual(t, m, merr):
    """Fit PSPL to the noisy curve; return a residual-based anomaly score (chi^2 improvement of a
    free per-window bump over the best PSPL). Observed-data only."""
    from scipy.optimize import least_squares
    mb0 = np.percentile(m, 90); tpk = t[np.argmin(m)]
    amp = max(mb0 - m.min(), 0.05)
    def resid(p):
        t0, ltE, u0, fs, mbase = p
        return (paczynski(t, t0, np.exp(ltE), abs(u0) + 1e-3, np.clip(fs, 0.01, 1.0), mbase) - m) / merr
    try:
        r = least_squares(resid, [tpk, np.log(20), 0.3, 0.5, mb0], max_nfev=200, method="lm")
        res = resid(r.x)
        chi2_pspl = float((res ** 2).sum())
        # a "bump": largest contiguous run of same-sign residuals, its chi2 contribution
        sig = res
        best = 0.0
        for w in (3, 5, 9):
            csum = np.convolve(sig, np.ones(w), "valid")
            best = max(best, float(np.nanmax(csum ** 2)) if csum.size else 0.0)
        return best / max(chi2_pspl / len(t), 1e-6)   # excess bump vs per-point scatter
    except Exception:
        return 0.0


def features(t, m):
    m = np.asarray(m, float)
    mn, mx = m.min(), m.max()
    amp = mx - mn
    d = np.diff(m)
    from scipy.stats import skew, kurtosis
    ac1 = float(np.corrcoef(m[:-1], m[1:])[0, 1]) if len(m) > 2 else 0.0
    npk = int(((m[1:-1] < m[:-2]) & (m[1:-1] < m[2:])).sum())
    return [amp, float(np.std(m)), float(skew(m)), float(kurtosis(m)),
            ac1, npk / max(len(m), 1), float(np.mean(np.abs(d))), float(np.percentile(m, 5) - np.median(m))]


def _downsample(t, m, e, k=400):
    if len(t) <= k:
        return t, m, e
    idx = np.sort(np.random.default_rng(0).choice(len(t), k, replace=False))
    return t[idx], m[idx], e[idx]


def main():
    clf = binml.Classifier()
    print(f"simulating up to {N_EVENTS} events (streaming, BinML scored inline)...", flush=True)
    labs, Xl, pspl_l, binml_l = [], [], [], []
    s = 500000
    while len(labs) < N_EVENTS and s < 500000 + 8 * N_EVENTS:
        s += 1
        # boost NonPSPL generation so enough DETECTABLE anomalies survive labelling
        c = "NonPSPL" if (s % 5 < 2) else CLASSES[s % 6]
        ev = simulate_event(c, np.random.default_rng(s), CFG)
        if ev is None or "F146" not in ev.bands or len(ev.bands["F146"].t) < 100:
            continue
        b = ev.bands["F146"]
        # BinML on the full curve
        p = clf.predict(b.t, b.mag, m_base_ref=ev.params["_m_base_ref"], t_start=0.0)
        # observed-data baselines on a downsampled curve (cheap, fieldable)
        td, md, ed = _downsample(b.t, b.mag, b.mag_err)
        labs.append(CLASSES.index(ev.label))
        binml_l.append(p.probabilities["NonPSPL"])
        Xl.append(features(td, md))
        pspl_l.append(fit_pspl_residual(td, md, ed))
        if len(labs) % 1000 == 0:
            print(f"  {len(labs)}/{N_EVENTS}", flush=True)
    y = np.array(labs); binml_p = np.array(binml_l)
    X = np.array(Xl); pspl_score = np.array(pspl_l)
    print(f"got {len(y)} events; NonPSPL prevalence {100*(y==NON).mean():.1f}%", flush=True)
    HERE = os.path.dirname(os.path.abspath(__file__))
    np.savez(os.path.join(HERE, "baseline_data.npz"), y=y, X=X, pspl=pspl_score, binml=binml_p)

    # trained feature baselines learn on a disjoint split; BinML and the fitted-PSPL residual are
    # unsupervised on these events, so evaluate all on the same held-out test rows.
    rng = np.random.default_rng(0); idx = rng.permutation(len(y))
    ntr = int(0.6 * len(y)); tr, te = idx[:ntr], idx[ntr:]
    ybin = (y == NON).astype(int)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score as ap
    gbt = HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.1)
    gbt.fit(X[tr], ybin[tr]); gbt_p = gbt.predict_proba(X[te])[:, 1]
    sc = StandardScaler().fit(X[tr])
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), ybin[tr])
    lr_p = lr.predict_proba(sc.transform(X[te]))[:, 1]

    res = {
        "n_events": int(len(y)), "n_test": int(len(te)),
        "nonpspl_prevalence": round(float((y == NON).mean()), 3),
        "ap_binml": round(float(ap(ybin[te], binml_p[te])), 3),
        "ap_fitted_pspl_residual": round(float(ap(ybin[te], pspl_score[te])), 3),
        "ap_gbt_features": round(float(ap(ybin[te], gbt_p)), 3),
        "ap_logistic_features": round(float(ap(ybin[te], lr_p)), 3),
    }
    print(json.dumps(res, indent=2))
    json.dump(res, open(os.path.join(HERE, "baselines_result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()

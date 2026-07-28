"""BinML quick start — classify a Roman light curve with the 6-class model.

Runs with only ``pip install binml`` (torch + numpy). It builds a synthetic single-lens
(Paczynski) microlensing event on the Roman F146 cadence, classifies it, and shows the
real-time cascade. Swap in your own ``(time_days, magnitude)`` arrays to classify real data.
"""
import numpy as np
import binml


def paczynski_lightcurve(t0=36.0, tE=25.0, u0=0.5, m_base=20.5,
                         f_s=0.5, window=72.0, cadence_min=15.0, noise=0.03, seed=0):
    """A single-lens (PSPL) event on Roman F146 cadence, in magnitudes.

    NOTE: BinML is trained on *realistic* Roman photometry — with blending and per-epoch
    noise/detectability. An idealised, perfectly-sampled, near-noiseless curve is
    out-of-distribution and can be misread. So this includes a blend fraction ``f_s`` (the
    magnified source is only part of the flux) and realistic scatter. For best results, feed
    real observed light curves, not toy curves.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, window, cadence_min / (60 * 24))      # 15-min sampling over 72 days
    u = np.sqrt(u0**2 + ((t - t0) / tE) ** 2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))                  # magnification
    A_obs = 1.0 + f_s * (A - 1.0)                             # blending dilutes the amplitude
    mag = m_base - 2.5 * np.log10(A_obs) + rng.normal(0, noise, t.size)
    return t, mag, m_base


def main():
    t, mag, m_base = paczynski_lightcurve()

    clf = binml.Classifier()                                 # BinML 1.0, CPU, weights bundled

    # --- classify (single band = F146) --------------------------------------------------
    r = clf.predict(t, mag, m_base_ref=m_base)               # give the F146 baseline magnitude
    print(r)                                                 # <BinML PSPL 0.9x | microlensing .. anomalous ..>
    print("  is this microlensing? P =", round(r.is_microlensing, 3))
    print("  is it anomalous (binary/planet)? P =", round(r.is_anomalous, 3))
    print("  full class probabilities:")
    for cls, p in sorted(r.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"    {cls:14s} {p:.3f}")

    # --- multi-band (if you have colour bands) ------------------------------------------
    #   r = clf.predict({"F146": (t, mag), "F087": (t087, mag087), "F213": (t213, mag213)},
    #                   m_base_ref=m_base)

    # --- the real-time cascade: probabilities as the season is revealed ------------------
    days, probs = clf.predict_evolution({"F146": (t, mag)}, m_base_ref=m_base, n_steps=8)
    i_nonpspl = binml.CLASS_NAMES.index("NonPSPL")
    print("\n  P(NonPSPL) as the season is revealed (should stay low for a plain PSPL):")
    for d, p in zip(days, probs[:, i_nonpspl]):
        print(f"    day {d:5.1f}:  {p:.3f}")


if __name__ == "__main__":
    main()

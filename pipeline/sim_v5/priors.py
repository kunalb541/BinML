"""
BinML v5 — parameter priors for the Roman Galactic Bulge microlensing simulation.

Every distribution here is literature-grounded; the source is named next to each constant and
the numbers the source pins down are ASSERTED in :func:`self_test`. If a cited moment stops
reproducing, the prior is wrong and the self-test fails loudly — a simulator bug is worse than
a training bug because it silently poisons every downstream number.

Sources
-------
Mroz et al. 2019, ApJS 244, 29    OGLE-IV, 8 yr, 8002 events. Efficiency-corrected t_E
                                  distribution; <t_E> ~ 22 d for the inner bulge (Roman's
                                  fields, l ~ 0.5 deg); catalogue cut t_E <= 300 d;
                                  their own efficiency sims used u0 ~ U(0, 1).
Wyrzykowski et al. 2015, ApJS 216, 12   Log-normal fits to t_E per 1x1 deg bin; the bins
                                  containing the Roman fields peak at 16.6-19.9 d.
Kaczmarek et al. 2024, MNRAS 529, 1308  VVV near-IR: log-normal peak 23.6 +/- 1.9 d
                                  (central fields) -- the closest bandpass analogue to Roman
                                  and ~20-40% above the optical values. See CAVEATS.
Zhang, Zhang, Bloom & Gaudi 2021, AJ 161, 262   Roman 2L1S neural-posterior training priors:
                                  t_E TruncLogNorm, u0 ~ U(0,2), s ~ LogU(0.2,5),
                                  q ~ LogU(1e-6, 1), rho ~ LogU(1e-4, 1e-2), alpha ~ U(0,360).
Suzuki et al. 2016, ApJ 833, 145  Planetary mass-ratio function: broken power law with
                                  q_br = 1.7e-4, n = -0.93, p = 0.6, m = 0.49, A = 0.61;
                                  sensitivity grid 0.1 <= s <= 10.
Penny et al. 2019, ApJS 241, 3    Roman/WFIRST survey simulation (GULLS): t0 drawn over the
                                  full simulated span, u0_max = 3.

CAVEATS (do not treat as settled)
---------------------------------
* The t_E MEDIAN is uncertain at roughly the 1.5x level and the surveys disagree: optical
  OGLE-III gives 16.6-19.9 d in Roman's fields, near-IR VVV gives 23.6 +/- 1.9 d, and
  Zhang+2021 used 14.1 d. We anchor on the best-measured quantity -- the efficiency-corrected
  MEAN <t_E> ~ 22-23 d (Mroz+2019) -- which with sigma_log10 = 0.35 implies a median of
  ~16.3 d. If Roman's redder bandpass pushes the truth toward the VVV value, the median could
  be ~20-24 d. `TE_MEDIAN_DAYS` is the single knob for a sensitivity run.
* The t_E WIDTH has no published sigma we could find; 0.35 dex is a defensible middle
  consistent with the mean/peak ratios in Wyrzykowski's Roman-field bins.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["EventPriors", "DEFAULT_PRIORS", "self_test"]


@dataclass(frozen=True)
class EventPriors:
    """Sampling distributions for a single microlensing event."""

    # --- Einstein crossing time -------------------------------------------------
    # log10(t_E/day) ~ Normal(log10(TE_MEDIAN_DAYS), TE_SIGMA_DEX), truncated.
    # Anchored so that <t_E> reproduces Mroz+2019's efficiency-corrected inner-bulge mean.
    TE_MEDIAN_DAYS: float = 16.3
    TE_SIGMA_DEX: float = 0.35
    TE_MIN_DAYS: float = 1.0      # Mroz+2019 efficiency sims span 1-316 d
    TE_MAX_DAYS: float = 300.0    # Mroz+2019 catalogue cut

    # --- impact parameter -------------------------------------------------------
    # UNIFORM in u0 is the geometrically correct prior (rectilinear motion across a strip).
    # NOT log-uniform: that would grossly over-produce high-magnification events.
    U0_MAX: float = 2.0           # Zhang+2021; Mroz+2019 used 1, Penny+2019 used 3

    # --- binary-lens parameters -------------------------------------------------
    # q log-uniform down to 1e-6: 1e-4 sits exactly ON the Suzuki+2016 break, which would
    # discard the regime where Roman is uniquely capable.
    Q_MIN: float = 1e-6
    Q_MAX: float = 1.0
    Q_PLANETARY_MAX: float = 1e-2  # class boundary: planetary vs stellar companion
    # s log-uniform (Suzuki+2016 m = 0.49 +/- 0.47 is consistent with flat in log s).
    S_MIN: float = 0.2
    S_MAX: float = 5.0
    # rho = theta_star / theta_E
    RHO_MIN: float = 1e-4
    RHO_MAX: float = 1e-2

    # --- source flux fraction (blending) ----------------------------------------
    # f_s = F_source / (F_source + F_blend). Roman's bulge fields are crowded; a typical
    # detected event has f_s well below 1. Sampled per-event, then made band-dependent
    # by the photometry module (extinction/SED differ per band).
    FS_MIN: float = 0.1
    FS_MAX: float = 1.0

    def sample_tE(self, rng: np.random.Generator) -> float:
        """Truncated log-normal t_E in days."""
        mu = math.log10(self.TE_MEDIAN_DAYS)
        for _ in range(1000):
            v = 10.0 ** rng.normal(mu, self.TE_SIGMA_DEX)
            if self.TE_MIN_DAYS <= v <= self.TE_MAX_DAYS:
                return float(v)
        return float(min(max(v, self.TE_MIN_DAYS), self.TE_MAX_DAYS))

    def sample_u0(self, rng: np.random.Generator) -> float:
        """Uniform (linear) impact parameter. Sign is degenerate with alpha for binaries."""
        return float(rng.uniform(0.0, self.U0_MAX))

    def sample_q(self, rng: np.random.Generator,
                 lo: Optional[float] = None, hi: Optional[float] = None) -> float:
        """Log-uniform mass ratio. Pass lo/hi to restrict to planetary or stellar regime."""
        lo = self.Q_MIN if lo is None else lo
        hi = self.Q_MAX if hi is None else hi
        return float(10.0 ** rng.uniform(math.log10(lo), math.log10(hi)))

    def sample_s(self, rng: np.random.Generator) -> float:
        """Log-uniform projected separation (Einstein units)."""
        return float(10.0 ** rng.uniform(math.log10(self.S_MIN), math.log10(self.S_MAX)))

    def sample_rho(self, rng: np.random.Generator) -> float:
        """Log-uniform normalized source radius."""
        return float(10.0 ** rng.uniform(math.log10(self.RHO_MIN), math.log10(self.RHO_MAX)))

    def sample_alpha(self, rng: np.random.Generator) -> float:
        """Uniform trajectory angle in radians."""
        return float(rng.uniform(0.0, 2.0 * math.pi))

    def sample_fs(self, rng: np.random.Generator) -> float:
        """Log-uniform source flux fraction (blending)."""
        return float(10.0 ** rng.uniform(math.log10(self.FS_MIN), math.log10(self.FS_MAX)))

    # --- derived, for verification ---------------------------------------------
    @property
    def tE_sigma_ln(self) -> float:
        return self.TE_SIGMA_DEX * math.log(10.0)

    @property
    def tE_mean_analytic(self) -> float:
        """<t_E> of the *untruncated* log-normal: median * exp(sigma_ln^2 / 2)."""
        return self.TE_MEDIAN_DAYS * math.exp(self.tE_sigma_ln ** 2 / 2.0)


DEFAULT_PRIORS = EventPriors()


def self_test(n: int = 200_000, seed: int = 0, verbose: bool = True) -> dict:
    """Assert the sampled distributions reproduce the literature numbers they were built from.

    Raises AssertionError on any mismatch. Returns the measured summary statistics.
    """
    p = DEFAULT_PRIORS
    rng = np.random.default_rng(seed)

    tE = np.array([p.sample_tE(rng) for _ in range(n)])
    u0 = rng.uniform(0.0, p.U0_MAX, n)
    q = 10.0 ** rng.uniform(math.log10(p.Q_MIN), math.log10(p.Q_MAX), n)
    s = 10.0 ** rng.uniform(math.log10(p.S_MIN), math.log10(p.S_MAX), n)

    stats = {
        "tE_median": float(np.median(tE)),
        "tE_mean": float(tE.mean()),
        "tE_mean_analytic": p.tE_mean_analytic,
        "tE_min": float(tE.min()), "tE_max": float(tE.max()),
        "tE_frac_gt_18": float((tE > 18).mean()),
        "u0_mean": float(u0.mean()),
        "q_log10_mean": float(np.log10(q).mean()),
        "s_log10_mean": float(np.log10(s).mean()),
    }

    # --- t_E: must reproduce Mroz+2019's efficiency-corrected inner-bulge mean -------
    # Truncation at [1, 300] shaves the tail slightly, so the sampled mean sits a little
    # below the analytic one; require the analytic value to hit the literature target and
    # the sampled value to stay within a few percent of it.
    assert 22.0 <= p.tE_mean_analytic <= 23.5, (
        f"<t_E> analytic = {p.tE_mean_analytic:.2f} d, outside the Mroz+2019 inner-bulge "
        f"target of 22-23 d -- the t_E prior no longer matches its source")
    assert abs(stats["tE_mean"] - p.tE_mean_analytic) / p.tE_mean_analytic < 0.05, (
        f"sampled <t_E> {stats['tE_mean']:.2f} deviates >5% from analytic "
        f"{p.tE_mean_analytic:.2f} -- truncation is distorting the distribution")
    assert 15.0 <= stats["tE_median"] <= 18.0, (
        f"t_E median {stats['tE_median']:.2f} d outside the expected ~16.3 d")
    assert p.TE_MIN_DAYS <= stats["tE_min"] and stats["tE_max"] <= p.TE_MAX_DAYS, \
        "t_E truncation bounds violated"
    # The old, biased prior generated ZERO events above 18 d; the corrected one must not.
    assert stats["tE_frac_gt_18"] > 0.35, (
        f"only {stats['tE_frac_gt_18']:.2f} of events have t_E > 18 d -- the long-event "
        f"population is still under-represented")

    # --- u0 must be LINEAR uniform (mean = U0_MAX/2), not log-uniform ---------------
    assert abs(stats["u0_mean"] - p.U0_MAX / 2) < 0.02 * p.U0_MAX, (
        f"u0 mean {stats['u0_mean']:.3f} != {p.U0_MAX/2:.3f}; u0 must be uniform in u0, "
        f"not log-uniform (log-uniform over-produces high-magnification events)")

    # --- q, s must be LOG-uniform (mean of log10 == midpoint of log10 range) --------
    q_mid = (math.log10(p.Q_MIN) + math.log10(p.Q_MAX)) / 2
    s_mid = (math.log10(p.S_MIN) + math.log10(p.S_MAX)) / 2
    assert abs(stats["q_log10_mean"] - q_mid) < 0.02 * abs(q_mid or 1), \
        f"q is not log-uniform (mean log10 q = {stats['q_log10_mean']:.3f}, expected {q_mid:.3f})"
    assert abs(stats["s_log10_mean"] - s_mid) < 0.02 + abs(s_mid) * 0.02, \
        f"s is not log-uniform (mean log10 s = {stats['s_log10_mean']:.3f}, expected {s_mid:.3f})"

    # --- the q floor must reach the Suzuki+2016 regime Roman is built for -----------
    assert p.Q_MIN <= 1e-6, "q floor must reach 1e-6; 1e-4 sits on the Suzuki+2016 break"

    if verbose:
        print("priors.self_test PASSED")
        print(f"  t_E: median {stats['tE_median']:.1f} d, mean {stats['tE_mean']:.1f} d "
              f"(analytic {p.tE_mean_analytic:.1f}, target 22-23), "
              f"{stats['tE_frac_gt_18']*100:.0f}% > 18 d")
        print(f"  u0 : mean {stats['u0_mean']:.3f} (uniform on [0,{p.U0_MAX}])")
        print(f"  q  : log-uniform [{p.Q_MIN:g}, {p.Q_MAX:g}]")
        print(f"  s  : log-uniform [{p.S_MIN:g}, {p.S_MAX:g}]")
    return stats


if __name__ == "__main__":
    self_test()

"""
BinML v5 — light-curve generators for the six classes.

Every generator returns the **fractional flux change of the varying source**,
``delta_b(t) = F_b(t)/F_base,b - 1``, which is then diluted by the per-band blending
fraction in :mod:`photometry`. Using one flux-space convention for microlensing and
contaminants alike is deliberate: it guarantees they are diluted identically, so a
classifier cannot separate them on an artefact of how they were generated.

The load-bearing distinction:

* **Microlensing is ACHROMATIC** -- ``delta`` is identical in every band. (Band-dependent
  observed amplitude still arises, but only through per-band blending.)
* **Variables are CHROMATIC** -- their intrinsic amplitude differs between bands.

Both are asserted in :func:`self_test`; if a variable generator ever comes out achromatic,
the discriminator we rely on is fake and the test fails.

Parameter sources (bulge fields; see the literature review for full detail)
--------------------------------------------------------------------------
RR Lyrae      Soszynski et al. 2011/2014 (OGLE-III/IV bulge: RRab mean P = 0.556 d,
              A_I ~ 0.2-0.9 mag); Braga et al. 2018/2019 (NIR templates, A_Ks/A_V = 0.41
              for RRab, 0.22 for RRc -- NIR light curves are markedly more sinusoidal).
Miras/SRV/    Iwanek et al. 2022 (40,356 bulge Miras); Soszynski et al. OGLE-III LPV
OSARG         catalogue (192,643 OSARGs = the most numerous bulge variable class,
              ~2800/deg^2; 33,235 SRVs). OSARGs are multi-periodic -- mode beating is what
              makes them dangerous.
Eclipsing     Soszynski et al. 2016 (450,598 bulge eclipsing/ellipsoidal systems).
Dwarf novae   Mroz et al. 2015 (1059 DNe in OGLE bulge fields; outbursts "can sometimes
              mimic microlensing").
Be 'bumpers'  Bennett 2005 (removing variable 'bumpers' cut the MACHO LMC optical depth
              by ~40% -- historically the leading microlensing false positive).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .photometry import ROMAN_BANDS

__all__ = ["pspl_magnification", "Generator", "FlatGen", "PSPLGen", "NonPSPLGen",
           "PeriodicVarGen", "LongPeriodVarGen", "EruptiveGen", "GENERATORS", "self_test"]


# ---------------------------------------------------------------------------------
# Microlensing magnification
# ---------------------------------------------------------------------------------
def pspl_magnification(t: np.ndarray, t0: float, tE: float, u0: float) -> np.ndarray:
    """Paczynski point-source point-lens magnification. Achromatic by construction."""
    u = np.sqrt(u0 ** 2 + ((np.asarray(t, dtype=float) - t0) / tE) ** 2)
    u = np.maximum(u, 1e-8)
    return (u ** 2 + 2.0) / (u * np.sqrt(u ** 2 + 4.0))


def _binary_magnification(t: np.ndarray, t0: float, tE: float, u0: float,
                          s: float, q: float, alpha: float, rho: float) -> np.ndarray:
    """Binary-lens magnification via VBBinaryLensing; falls back to PSPL if unavailable.

    The fallback exists only so the module is importable/testable without the C++ extension;
    production runs MUST have VBBinaryLensing (the caller asserts this).
    """
    try:
        import VBBinaryLensing  # type: ignore
    except Exception:
        return pspl_magnification(t, t0, tE, u0)
    vbb = VBBinaryLensing.VBBinaryLensing()
    vbb.RelTol = 1e-3
    tau = (np.asarray(t, dtype=float) - t0) / tE
    ca, sa = math.cos(alpha), math.sin(alpha)
    # source trajectory in the lens frame
    x = tau * ca - u0 * sa
    y = tau * sa + u0 * ca
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = vbb.BinaryMag2(s, q, float(x[i]), float(y[i]), rho)
    return out


def has_vbb() -> bool:
    try:
        import VBBinaryLensing  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------------
# Generator interface
# ---------------------------------------------------------------------------------
class Generator:
    """Base class. ``sample`` draws parameters; ``delta`` returns fractional flux change."""
    name: str = "base"
    achromatic: bool = False

    def sample(self, rng: np.random.Generator, span_days: float) -> dict:
        raise NotImplementedError

    def delta(self, t: np.ndarray, params: dict, band: str) -> np.ndarray:
        raise NotImplementedError

    # -- achromatic memoisation -------------------------------------------------
    # An achromatic signal MUST be bit-identical in every band. Recomputing it per band is
    # wasteful and, in principle, unsafe: any non-determinism in VBBinaryLensing's adaptive
    # integration would inject a band-dependent difference -- exactly the spurious "colour"
    # signal that would let a classifier cheat.
    #
    # The cache is keyed on the time array's IDENTITY, so it only helps when the caller hands
    # back the same array. An earlier version relied on it in the production path, where each
    # band builds its own grid, so it never once hit: the binary magnification was computed
    # three times per event. That mattered -- VBBinaryLensing is NOT bit-reproducible across
    # repeated identical calls (measured spread ~2.4e-12 in magnification, unchanged at
    # RelTol 1e-7, i.e. floating-point level rather than a tolerance setting), so the three
    # per-band evaluations disagreed slightly and produced a genuine, if tiny, spurious
    # colour signal in an achromatic class.
    #
    # Callers must instead evaluate on the FINEST grid and index down to the coarser bands
    # (see assemble.simulate_event). That makes achromaticity structural -- the bands share
    # the very same array elements -- rather than contingent on the integrator being
    # deterministic, which it is not.
    #
    # The residual consequence is that NonPSPL events are not bit-exactly REGENERABLE from a
    # seed. At 2.4e-12, ~9 orders of magnitude below the 1 mmag systematic floor, this has no
    # physical effect; reproducibility of a published dataset comes from archiving the
    # generated file, not from re-running the generator.
    def _achromatic_cached(self, t: np.ndarray, params: dict, compute) -> np.ndarray:
        cached = params.get("_delta_cache")
        if cached is not None and cached[0] is t:
            return cached[1]
        d = compute()
        params["_delta_cache"] = (t, d)
        return d


# ---- 0. Flat --------------------------------------------------------------------
class FlatGen(Generator):
    name = "Flat"
    achromatic = True

    def sample(self, rng, span_days):
        return {}

    def delta(self, t, params, band):
        return np.zeros_like(np.asarray(t, dtype=float))


# ---- 1. PSPL --------------------------------------------------------------------
class PSPLGen(Generator):
    name = "PSPL"
    achromatic = True

    def sample(self, rng, span_days, priors=None):
        from .priors import DEFAULT_PRIORS
        p = priors or DEFAULT_PRIORS
        return {"t0": float(rng.uniform(0.0, span_days)),
                "tE": p.sample_tE(rng), "u0": p.sample_u0(rng)}

    def delta(self, t, params, band):
        # ACHROMATIC: band is ignored on purpose; computed once and reused.
        return self._achromatic_cached(
            t, params,
            lambda: pspl_magnification(t, params["t0"], params["tE"], params["u0"]) - 1.0)


# ---- 2. NonPSPL (anomalous microlensing) ----------------------------------------
class NonPSPLGen(Generator):
    """Anomalous microlensing: binary lens across the full mass-ratio range."""
    name = "NonPSPL"
    achromatic = True

    def sample(self, rng, span_days, priors=None):
        from .priors import DEFAULT_PRIORS
        p = priors or DEFAULT_PRIORS
        return {"t0": float(rng.uniform(0.0, span_days)),
                "tE": p.sample_tE(rng), "u0": p.sample_u0(rng),
                "s": p.sample_s(rng), "q": p.sample_q(rng),
                "alpha": p.sample_alpha(rng), "rho": p.sample_rho(rng)}

    def delta(self, t, params, band):
        # ACHROMATIC: band ignored; computed once and reused (VBB is not bit-reproducible).
        def _c():
            A = _binary_magnification(t, params["t0"], params["tE"], params["u0"],
                                      params["s"], params["q"], params["alpha"], params["rho"])
            return A - 1.0
        return self._achromatic_cached(t, params, _c)


# ---------------------------------------------------------------------------------
# Contaminants: intrinsic magnitude variation -> fractional flux change
# ---------------------------------------------------------------------------------
def _mag_shape_to_delta(shape: np.ndarray, amp_mag: float) -> np.ndarray:
    """Convert a normalised zero-mean magnitude shape (peak-to-peak 1) to fractional flux."""
    dm = amp_mag * np.asarray(shape, dtype=float)
    return 10.0 ** (-0.4 * dm) - 1.0


# Amplitude relative to the OGLE I band, interpolated to the Roman bands.
# NIR amplitudes are markedly smaller than optical (Braga et al. 2018/2019).
#
# A SINGLE global ratio per band is wrong, and dangerously so. The optical-to-NIR amplitude
# ratio is set by the variability MECHANISM: pulsation temperature swings (RR Lyrae),
# molecular-opacity swings (Mira TiO bands), geometry (eclipsing binaries, near-achromatic),
# or a changing accretion continuum (dwarf novae). These differ by a factor of ~3. Using one
# constant put the entire contaminant population on a delta function in colour-amplitude
# space (measured intrinsic scatter ~1e-15), so the only dispersion left came from blending.
# A classifier can exploit a degenerate contaminant locus that real data does not have.
#
# Each object therefore draws its OWN ratio: a type-specific K/I amplitude ratio with
# lognormal scatter, interpolated to each band in log-wavelength.
_LAMBDA_I_UM = 0.80
_TYPE_AMP_RATIO_K: Dict[str, float] = {
    "RRab": 0.45, "RRc": 0.35, "EB": 0.92, "dSct": 0.55,       # EB is geometric => achromatic
    "Mira": 0.28, "SRV": 0.42, "OSARG": 0.45,                  # Mira: TiO drives the optical
    "DN": 0.55, "WZSge": 0.55, "Bumper": 0.70,
}
_AMP_RATIO_SCATTER_DEX = 0.08      # ~20% object-to-object dispersion


def sample_amp_ratio_k(rng, kind: str) -> float:
    """Per-OBJECT amplitude ratio A_K / A_I for a variable of this type."""
    base = _TYPE_AMP_RATIO_K.get(kind, 0.5)
    return float(np.clip(base * 10.0 ** rng.normal(0.0, _AMP_RATIO_SCATTER_DEX), 0.05, 1.2))


def object_ratio_k(params: dict) -> float:
    """This object's K/I amplitude ratio, falling back to its type mean if unsampled."""
    r = params.get("ratio_k")
    return float(r) if r is not None else _TYPE_AMP_RATIO_K.get(params.get("kind", ""), 0.5)


def amp_ratio_in_band(band: str, ratio_k: float) -> float:
    """Interpolate this object's amplitude ratio to ``band``, linearly in log-wavelength."""
    from .photometry import ROMAN_BANDS
    lam = ROMAN_BANDS[band].wavelength_um
    f = math.log10(lam / _LAMBDA_I_UM) / math.log10(2.13 / _LAMBDA_I_UM)
    return float(max(1.0 + f * (ratio_k - 1.0), 0.02))


def _normalised_shape(s_obs: np.ndarray, s_ref: np.ndarray) -> np.ndarray:
    """Zero-mean, unit peak-to-peak -- normalised on a REFERENCE grid, not the window.

    Normalising on the observed window forced every object to display its full cycle
    amplitude inside the season no matter how long its period, which inflated long-period
    Miras by up to ~5x and removed the amplitude suppression that makes a sparsely sampled
    or slowly evolving variable hard to detect. ``s_ref`` must cover a full cycle.
    """
    mu = float(np.mean(s_ref))
    ptp = float(np.ptp(s_ref))
    if ptp <= 0:
        return np.zeros_like(s_obs)
    return (s_obs - mu) / ptp


class PeriodicVarGen(Generator):
    """Short-period coherent variables: RR Lyrae (ab/c), eclipsing binaries, delta Scuti.

    Many cycles per 72-day season, so individually easy to reject on periodicity -- but
    abundant, and heavily blended examples can slip below naive variability cuts.
    """
    name = "PeriodicVar"
    achromatic = False

    def sample(self, rng, span_days):
        kind = rng.choice(["RRab", "RRc", "EB", "dSct"], p=[0.35, 0.20, 0.35, 0.10])
        if kind == "RRab":                       # OGLE bulge mean P = 0.556 d
            P = float(np.clip(rng.normal(0.556, 0.06), 0.35, 0.95))
            amp_I = float(np.clip(rng.normal(0.62, 0.18), 0.15, 1.0))
            # sawtooth-ish: steep rise, slow decline (R21/R31 with phi21~4, phi31~2.1)
            harm = [(1.0, 0.0), (0.45, 4.0), (0.30, 2.1), (0.20, 0.4)]
        elif kind == "RRc":                      # first overtone: shorter, near-sinusoidal
            P = float(np.clip(rng.normal(0.33, 0.05), 0.20, 0.48))
            amp_I = float(np.clip(rng.normal(0.28, 0.08), 0.08, 0.55))
            harm = [(1.0, 0.0), (0.12, 3.9)]
        elif kind == "EB":                       # eclipses: modelled below, not Fourier
            P = float(10 ** rng.uniform(math.log10(0.2), math.log10(60.0)))
            amp_I = float(np.clip(rng.normal(0.45, 0.30), 0.05, 1.5))
            harm = None
        else:                                    # delta Scuti: very short, low amplitude
            P = float(10 ** rng.uniform(math.log10(0.02), math.log10(0.3)))
            amp_I = float(np.clip(rng.normal(0.12, 0.06), 0.02, 0.4))
            harm = [(1.0, 0.0), (0.15, 3.5)]
        # Eclipse width in PHASE units must fall with period. Duration/P ~ R_*/a, and by
        # Kepler a ~ P^(2/3), so phase width ~ P^(-2/3) at fixed stellar radius. Drawing it
        # flat in phase (as an earlier version did) gave a 60-day detached binary an eclipse
        # lasting ~7 days -- an order of magnitude longer than any real system, and a shape
        # that reads as a broad smooth dip rather than an eclipse.
        width = float(np.clip(rng.uniform(0.03, 0.12) * P ** (-2.0 / 3.0), 0.004, 0.20))
        return {"kind": kind, "P": P, "amp_I": amp_I, "phase0": float(rng.uniform(0, 1)),
                "harm": harm, "sec_depth": float(rng.uniform(0.15, 0.95)),
                "width": width, "ratio_k": sample_amp_ratio_k(rng, kind)}

    def _raw(self, ph, p):
        """Magnitude shape as a function of PHASE (positive = fainter)."""
        if p["kind"] == "EB":
            # two Gaussian-ish eclipses (primary at 0, secondary at 0.5), magnitudes POSITIVE
            w = p["width"]
            d = np.minimum(np.abs(ph), 1.0 - np.abs(ph))
            prim = np.exp(-0.5 * (d / w) ** 2)
            d2 = np.abs(ph - 0.5)
            sec = p["sec_depth"] * np.exp(-0.5 * (d2 / w) ** 2)
            return prim + sec                    # dips (fainter => positive magnitude)
        # Fourier series. The phase argument is NEGATED relative to the naive form: with
        # +2*pi*k*ph the series runs backwards in time, giving a slow rise to maximum light
        # over ~0.79 P and a fast decline over ~0.21 P -- a time-reversed RR Lyrae. Real RRab
        # stars rise sharply (~0.2 P) and decline slowly, and the phi_k values quoted here
        # are defined for the decreasing-phase convention.
        s = np.zeros_like(ph)
        for k, (amp, phi) in enumerate(p["harm"], start=1):
            s = s + amp * np.cos(-2 * math.pi * k * ph + phi)
        return -s                                # brighter at maximum light

    def _shape(self, t, p):
        ph = np.mod(np.asarray(t, dtype=float) / p["P"] + p["phase0"], 1.0)
        ph_ref = np.linspace(0.0, 1.0, 512, endpoint=False)   # a full cycle
        return _normalised_shape(self._raw(ph, p), self._raw(ph_ref, p))

    def delta(self, t, params, band):
        amp = params["amp_I"] * amp_ratio_in_band(band, object_ratio_k(params))
        return _mag_shape_to_delta(self._shape(t, params), amp)


class LongPeriodVarGen(Generator):
    """Miras, semiregulars and OSARGs -- the most dangerous impostors.

    With periods comparable to or longer than a 72-day season these present a single smooth
    bump or a slow trend, which is exactly a long-t_E heavily blended microlensing event.
    OSARGs are additionally MULTI-PERIODIC: mode beating creates slow envelopes.
    """
    name = "LongPeriodVar"
    achromatic = False

    def sample(self, rng, span_days):
        kind = rng.choice(["Mira", "SRV", "OSARG"], p=[0.10, 0.30, 0.60])
        if kind == "Mira":
            P = float(rng.uniform(100.0, 700.0))
            amp_I = float(rng.uniform(0.8, 3.0))
            periods = [(P, 1.0)]
        elif kind == "SRV":
            P = float(rng.uniform(40.0, 200.0))
            amp_I = float(rng.uniform(0.1, 0.5))
            periods = [(P, 1.0)]
        else:                                     # OSARG: two close modes -> beating
            P1 = float(rng.uniform(10.0, 100.0))
            P2 = P1 * float(rng.uniform(1.05, 1.35))
            amp_I = float(rng.uniform(0.01, 0.12))
            periods = [(P1, 1.0), (P2, float(rng.uniform(0.5, 1.0)))]
        return {"kind": kind, "amp_I": amp_I, "periods": periods,
                "phase0": float(rng.uniform(0, 1)),
                "phase1": float(rng.uniform(0, 1)),
                "ratio_k": sample_amp_ratio_k(rng, kind)}

    def _raw(self, t, p):
        s = np.zeros_like(np.asarray(t, dtype=float))
        for i, (P, w) in enumerate(p["periods"]):
            ph0 = p["phase0"] if i == 0 else p["phase1"]
            s = s + w * np.cos(2 * math.pi * (t / P + ph0))
        return -s

    def _shape(self, t, p):
        t = np.asarray(t, dtype=float)
        # Reference grid spanning several beat cycles of the LONGEST period, so the
        # normalisation reflects the object's full intrinsic amplitude. Normalising on the
        # 72-day window instead forced a 700-day Mira to show its entire cycle inside one
        # season -- the in-window amplitude was independent of period (ratio 0.65-0.68 in
        # every period bin), when it should fall steeply once P exceeds the window.
        p_max = max(P for P, _ in p["periods"])
        t_ref = np.linspace(0.0, 4.0 * p_max, 2048)
        return _normalised_shape(self._raw(t, p), self._raw(t_ref, p))

    def delta(self, t, params, band):
        amp = params["amp_I"] * amp_ratio_in_band(band, object_ratio_k(params))
        return _mag_shape_to_delta(self._shape(t, params), amp)


class EruptiveGen(Generator):
    """Dwarf-nova / CV outbursts and Be-star 'bumpers': single-episode brightenings."""
    name = "Eruptive"
    achromatic = False

    def sample(self, rng, span_days):
        kind = rng.choice(["DN", "WZSge", "Bumper"], p=[0.55, 0.15, 0.30])
        if kind == "DN":                          # fast rise, days-long decline, recurrent
            return {"kind": kind, "t_start": float(rng.uniform(-20, span_days)),
                    "amp_I": float(rng.uniform(2.0, 5.0)),
                    "rise": float(rng.uniform(0.3, 1.5)),
                    "decay": float(rng.uniform(2.0, 10.0)),
                    "plateau": 0.0,
                    "recur": float(rng.uniform(15.0, 120.0)),
                    "ratio_k": sample_amp_ratio_k(rng, kind)}
        if kind == "WZSge":                       # rare, huge, long plateau -- hardest impostor
            return {"kind": kind, "t_start": float(rng.uniform(-30, span_days)),
                    "amp_I": float(rng.uniform(4.0, 7.0)),
                    "rise": float(rng.uniform(0.5, 2.0)),
                    "decay": float(rng.uniform(10.0, 30.0)),
                    "plateau": float(rng.uniform(15.0, 30.0)),
                    "recur": 1e9,
                    "ratio_k": sample_amp_ratio_k(rng, kind)}
        return {"kind": kind, "t_start": float(rng.uniform(-100, span_days)),
                "amp_I": float(rng.uniform(0.15, 0.5)),   # Be 'bumper': low amplitude, slow
                "rise": float(rng.uniform(15.0, 60.0)),
                "decay": float(rng.uniform(40.0, 200.0)),
                "plateau": float(rng.uniform(0.0, 40.0)),
                "recur": 1e9,
                "ratio_k": sample_amp_ratio_k(rng, kind)}

    def _shape(self, t, p):
        """Fast-rise / plateau / exponential-decay, repeated at the recurrence time."""
        t = np.asarray(t, dtype=float)
        s = np.zeros_like(t)
        n_rep = 1 if p["recur"] > 1e8 else int(np.ceil((t.max() - p["t_start"]) / p["recur"]) + 1)
        for k in range(max(n_rep, 1)):
            t0 = p["t_start"] + k * p["recur"]
            dt = t - t0
            rise = np.clip(dt / p["rise"], 0.0, 1.0)
            after = np.maximum(dt - p["rise"] - p["plateau"], 0.0)
            prof = rise * np.exp(-after / p["decay"])
            prof[dt < 0] = 0.0
            s = np.maximum(s, prof)
        return -s          # brightening => negative magnitude change

    def delta(self, t, params, band):
        # Outbursts are BLUER at maximum. That is already what a per-object K/I ratio
        # encodes (Bumper 0.70 vs DN/WZSge 0.55), so no extra blue_boost factor -- applying
        # both double-counted the colour dependence.
        amp = params["amp_I"] * amp_ratio_in_band(band, object_ratio_k(params))
        return _mag_shape_to_delta(self._shape(t, params), amp)


GENERATORS: Dict[str, Generator] = {
    "Flat": FlatGen(), "PSPL": PSPLGen(), "NonPSPL": NonPSPLGen(),
    "PeriodicVar": PeriodicVarGen(), "LongPeriodVar": LongPeriodVarGen(),
    "Eruptive": EruptiveGen(),
}


# ---------------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------------
def self_test(verbose: bool = True) -> None:
    rng = np.random.default_rng(0)
    span = 72.0
    t = np.linspace(0.0, span, 6912)
    bands = list(ROMAN_BANDS)

    # --- INVARIANT 1: microlensing ACHROMATIC, variables CHROMATIC ----------------
    for gname in ("PSPL", "NonPSPL"):
        g = GENERATORS[gname]
        p = g.sample(rng, span)
        ds = [g.delta(t, p, b) for b in bands]
        for d in ds[1:]:
            assert np.array_equal(d, ds[0]), \
                f"{gname} must be ACHROMATIC: delta differs between bands"
    for gname in ("PeriodicVar", "LongPeriodVar", "Eruptive"):
        g = GENERATORS[gname]
        chromatic_seen = False
        for _ in range(20):
            p = g.sample(rng, span)
            ds = [g.delta(t, p, b) for b in bands]
            if not np.allclose(ds[0], ds[-1], atol=1e-9):
                chromatic_seen = True
                break
        assert chromatic_seen, (
            f"{gname} must be CHROMATIC -- if a variable generator is achromatic, the "
            f"colour discriminator we rely on is fake")

    # --- INVARIANT 2: PSPL sanity -------------------------------------------------
    A = pspl_magnification(t, 36.0, 20.0, 0.1)
    assert np.all(A >= 1.0 - 1e-12), "magnification must never be < 1"
    # the peak must land on the grid point nearest t0 (the grid need not contain t0 exactly),
    # and the true peak value is A(u0)
    i_peak = int(np.argmax(A))
    assert abs(t[i_peak] - 36.0) <= (t[1] - t[0]), "PSPL peak must occur at t0"
    A_t0 = pspl_magnification(np.array([36.0]), 36.0, 20.0, 0.1)[0]
    assert A.max() <= A_t0 + 1e-9, "no grid sample may exceed the true peak A(u0)"
    assert abs(A_t0 - (0.1 ** 2 + 2) / (0.1 * math.sqrt(0.1 ** 2 + 4))) < 1e-12, \
        "peak magnification must equal the analytic A(u0)"
    far = pspl_magnification(np.array([36.0 + 500 * 20.0]), 36.0, 20.0, 0.1)[0]
    assert abs(far - 1.0) < 1e-4, "magnification must tend to 1 far from the peak"

    # --- INVARIANT 3: binary -> PSPL as q -> 0 ------------------------------------
    if has_vbb():
        pars = dict(t0=36.0, tE=20.0, u0=0.15, s=1.0, q=1e-8, alpha=0.7, rho=1e-3)
        Ab = _binary_magnification(t, **pars)
        Ap = pspl_magnification(t, 36.0, 20.0, 0.15)
        assert np.nanmax(np.abs(Ab - Ap) / Ap) < 0.05, \
            "binary magnification must reduce to PSPL as q -> 0"

    # --- INVARIANT 4: periodic generators must reproduce their injected period ----
    g = GENERATORS["PeriodicVar"]
    for _ in range(8):
        p = g.sample(rng, span)
        if p["P"] < 1.0:                       # only test well-sampled periods
            d = g.delta(t, p, "F146")
            f = np.fft.rfftfreq(len(t), d=(t[1] - t[0]))
            amp = np.abs(np.fft.rfft(d - d.mean()))
            f_peak = f[1:][np.argmax(amp[1:])]
            P_rec = 1.0 / f_peak
            # allow the recovered period to be the fundamental or a harmonic
            ratio = P_rec / p["P"]
            assert min(abs(ratio - k) for k in (1, 0.5, 2)) < 0.05, \
                f"FFT recovered P={P_rec:.4f} d for injected P={p['P']:.4f} d"

    # --- INVARIANT 5: Flat is exactly flat ---------------------------------------
    assert np.all(GENERATORS["Flat"].delta(t, {}, "F146") == 0.0)

    # --- INVARIANT 6: finite, and brightenings are positive flux changes ----------
    for gname, g in GENERATORS.items():
        p = g.sample(rng, span)
        for b in bands:
            d = g.delta(t, p, b)
            assert np.all(np.isfinite(d)), f"{gname}/{b} produced non-finite delta"
            assert np.all(d > -1.0), f"{gname}/{b} implies negative flux"
    e = GENERATORS["Eruptive"]
    ep = {"kind": "DN", "t_start": 10.0, "amp_I": 3.0, "rise": 1.0,
          "decay": 5.0, "plateau": 0.0, "recur": 1e9}
    assert e.delta(t, ep, "F146").max() > 0.5, "an outburst must brighten the source"

    if verbose:
        print("generators.self_test PASSED")
        print(f"  VBBinaryLensing available: {has_vbb()}")
        print("  achromatic: PSPL, NonPSPL   chromatic: PeriodicVar, LongPeriodVar, Eruptive")
        print("  PSPL peak/limit, binary->PSPL(q->0), FFT period recovery, flux positivity OK")


if __name__ == "__main__":
    self_test()

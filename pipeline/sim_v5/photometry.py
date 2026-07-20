"""
BinML v5 — per-band photometry: SED, bulge extinction, blending, noise, detectability.

This module is what makes the per-band light curves genuinely DIFFERENT rather than scaled
copies. Three separate effects:

1. **Band-dependent extinction.** Bulge extinction is severe and strongly wavelength
   dependent, so a reddened source can be bright in F146/F213 and *below the limit* in F087.
2. **Different depth and cadence per band.** F146 is the deep, high-cadence band; F087/F213
   are shallower and sparse.
3. **Detectability is TIME-DEPENDENT.** Magnification brightens the source during the event,
   so a band that shows nothing at baseline can produce a handful of near-peak detections.
   Bright sources also SATURATE, which clips rather than adds noise.

The physical invariant this module must never break: **microlensing magnification A(t) is
achromatic.** All per-band differences come from (source SED x extinction), blending, noise,
and detectability -- never from a band-dependent A(t).

Sources
-------
Penny et al. 2019, ApJS 241, 3   Roman/WFIRST Cycle-7 photometry: W149 zeropoint 27.615
    (AB, e-/s reference), t_exp = 46.8 s, 3x3-pixel aperture background variance
    ~3218 e-^2, sigma_mag = sqrt((1.0857*sqrt(F+bkg)/F)^2 + floor^2), 1 mmag systematic
    floor; W149 saturation ~14.8 AB, Z087 ~13.9 AB; source catalogue to ~25 AB.
Nishiyama et al. 2009, ApJ 696, 1407   Bulge near-IR extinction law: A_lambda ~ lambda^-alpha
    with alpha ~ 2.0 (A_J/A_Ks = 3.02, A_H/A_Ks = 1.73) -- steeper than the standard ISM law.

VERIFY-BEFORE-TRUSTING (flagged for the cross-check phase)
----------------------------------------------------------
* Detailed photometric parameters (zeropoint, background, exposure) are published in detail
  only for the Cycle-7 W149 design. F087/F213 values here are SCALED estimates, not published
  numbers -- they must be checked before the production run.
* The extinction power-law index and the A_Ks distribution toward Roman's fields are
  approximations; the qualitative conclusion (F087 suffers ~6x the extinction of F213) is
  robust, the exact magnitudes are not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["Band", "ROMAN_BANDS", "ExtinctionLaw", "BulgeExtinction",
           "photometric_sigma", "apply_detectability", "self_test"]


# ---------------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Band:
    """A photometric band with its cadence, depth and noise parameters."""
    name: str
    wavelength_um: float        # effective wavelength (sets extinction)
    cadence_minutes: float      # sampling within a season
    zeropoint: float            # AB mag giving 1 e-/s (Penny+2019 convention)
    exposure_s: float
    background_e2: float        # background variance in the photometric aperture (e-^2)
    saturation_ab: float        # brighter than this saturates
    sys_floor_mag: float = 0.001  # systematic error floor

    def flux_e(self, mag_ab: float | np.ndarray) -> np.ndarray:
        """Detected electrons for a source of the given AB magnitude in one exposure."""
        return 10.0 ** (-0.4 * (np.asarray(mag_ab, dtype=float) - self.zeropoint)) * self.exposure_s


# Primary high-cadence band is F146 (a.k.a. W146/W149). F087/F213 give colour at ~12 hr.
# NOTE: F087/F213 zeropoint/background are SCALED ESTIMATES -- see VERIFY note above.
ROMAN_BANDS: Dict[str, Band] = {
    "F146": Band("F146", 1.46, cadence_minutes=15.0, zeropoint=27.615,
                 exposure_s=46.8, background_e2=3218.0, saturation_ab=14.8),
    "F087": Band("F087", 0.87, cadence_minutes=720.0, zeropoint=26.4,
                 exposure_s=286.0, background_e2=1200.0, saturation_ab=13.9),
    "F213": Band("F213", 2.13, cadence_minutes=720.0, zeropoint=26.0,
                 exposure_s=286.0, background_e2=4000.0, saturation_ab=14.5),
}


# ---------------------------------------------------------------------------------
# Extinction
# ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class ExtinctionLaw:
    """Near-IR power-law extinction, A_lambda ∝ lambda^-alpha (Nishiyama et al. 2009)."""
    alpha: float = 2.0
    reference_um: float = 2.13   # normalise to K-band (A_Ks)

    def relative(self, wavelength_um: float) -> float:
        """A_lambda / A_reference."""
        return (self.reference_um / wavelength_um) ** self.alpha


@dataclass
class BulgeExtinction:
    """Per-event line-of-sight extinction toward the bulge, normalised at A_Ks."""
    law: ExtinctionLaw = field(default_factory=ExtinctionLaw)
    a_ks_min: float = 0.20       # lightly reddened window
    a_ks_max: float = 1.20       # heavily reddened bulge field

    def sample_a_ks(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.a_ks_min, self.a_ks_max))

    def extinction_mag(self, band: Band, a_ks: float) -> float:
        """Extinction in magnitudes for this band given A_Ks."""
        return a_ks * self.law.relative(band.wavelength_um)


# ---------------------------------------------------------------------------------
# Noise and detectability
# ---------------------------------------------------------------------------------
def photometric_sigma(band: Band, mag_ab: np.ndarray) -> np.ndarray:
    """Per-epoch photometric uncertainty (mag), Penny et al. 2019 form.

    sigma = sqrt( (1.0857 * sqrt(F + background) / F)^2 + floor^2 )
    """
    F = band.flux_e(mag_ab)
    F = np.maximum(F, 1e-6)                       # guard: never divide by zero
    shot = 1.0857 * np.sqrt(F + band.background_e2) / F
    return np.sqrt(shot ** 2 + band.sys_floor_mag ** 2)


def apply_detectability(band: Band, mag_ab: np.ndarray, snr_threshold: float = 3.0
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Decide which epochs are usable in this band.

    Returns ``(detected, saturated)`` boolean masks. An epoch is *usable* when it is
    detected and not saturated. Because ``mag_ab`` varies during the event, this is
    genuinely time-dependent: a band can be blank at baseline and detected near peak.
    """
    mag_ab = np.asarray(mag_ab, dtype=float)
    sigma = photometric_sigma(band, mag_ab)
    snr = 1.0857 / np.maximum(sigma, 1e-12)       # SNR ~ 1.0857 / sigma_mag
    detected = snr >= snr_threshold
    saturated = mag_ab < band.saturation_ab
    return detected, saturated


def blend_fraction_in_band(f_s_ref: float, band: Band, ref_band: Band,
                           blend_minus_source_colour: float,
                           ext: BulgeExtinction, a_ks: float) -> float:
    """Per-band source flux fraction ``f_s,b``.

    **This is the single most important piece of the multi-band model.** The source and the
    blend are different stars with different colours, so the blending fraction is
    BAND-DEPENDENT. The consequence is subtle and load-bearing:

        even perfectly ACHROMATIC magnification produces BAND-DEPENDENT OBSERVED AMPLITUDES,
        because f_s,b differs between F087 / F146 / F213.

    So "the amplitude differs between bands" does NOT imply "not microlensing" — a classifier
    must not be allowed to learn that shortcut. Contaminants use this identical machinery.

    ``blend_minus_source_colour`` is the (blend - source) colour excess per unit
    log-wavelength baseline: positive means the blend is redder than the source.
    """
    if f_s_ref >= 1.0:
        return 1.0
    # differential colour term between this band and the reference, plus differential
    # extinction (the blend may sit at a different depth, but to first order they share
    # the line of sight, so the extinction difference cancels except via the colour term)
    dlog_lam = math.log10(band.wavelength_um / ref_band.wavelength_um)
    # Sign convention: C > 0 means the blend is REDDER than the source, i.e. relative to the
    # source the blend gets BRIGHTER (magnitude decreases) toward longer wavelength. So the
    # blend/source FLUX ratio grows with wavelength and the source fraction f_s falls.
    dm_blend_rel_source = -blend_minus_source_colour * dlog_lam     # mag
    ratio_ref = (1.0 - f_s_ref) / f_s_ref              # F_blend / F_source in the ref band
    ratio_b = ratio_ref * 10.0 ** (-0.4 * dm_blend_rel_source)
    return float(1.0 / (1.0 + ratio_b))


def observed_magnitude(band: Band, A: np.ndarray, m_base_intrinsic: float,
                       f_s: float, a_ks: float, ext: BulgeExtinction) -> np.ndarray:
    """Observed AB magnitude in this band for achromatic magnification ``A(t)``.

    ``f_s`` here is the fraction **in this band** (use :func:`blend_fraction_in_band` to get
    it). F_obs = A * F_source + F_blend; magnification multiplies only the SOURCE component,
    so blending dilutes the event.
    """
    A = np.asarray(A, dtype=float)
    m_base = m_base_intrinsic + ext.extinction_mag(band, a_ks)
    # Baseline flux in arbitrary units; only the RATIO matters for the magnitude change.
    f_base = 10.0 ** (-0.4 * m_base)
    f_obs = f_base * (f_s * A + (1.0 - f_s))
    return -2.5 * np.log10(np.maximum(f_obs, 1e-300))


def fractional_flux_change(f_s_band: float, delta: np.ndarray) -> np.ndarray:
    """Shared flux-space machinery for EVERY class (microlensing and contaminants alike).

    ``delta`` is the fractional flux change of the *varying source* (A - 1 for microlensing,
    or the variable's own fractional variation). The observed fractional change is diluted by
    the per-band blending fraction:  F_b(t)/F_base,b = 1 + f_s,b * delta_b(t).

    Using one routine for all classes is deliberate: it guarantees microlensing and
    contaminants are diluted identically, so the classifier cannot separate them on an
    artefact of how they were generated.
    """
    return 1.0 + f_s_band * np.asarray(delta, dtype=float)


# ---------------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------------
def self_test(verbose: bool = True) -> None:
    """Assert the physical invariants this module must never break."""
    ext = BulgeExtinction()
    rng = np.random.default_rng(0)
    A = np.array([1.0, 1.5, 3.0, 10.0])

    # --- INVARIANT 1: magnification is achromatic ---------------------------------
    # The SAME A(t) must be used for every band; only the mapping to magnitude differs.
    # Verify that with NO blending and NO extinction difference, the magnitude CHANGE
    # (relative to baseline) is identical across bands.
    dmags = []
    for b in ROMAN_BANDS.values():
        m = observed_magnitude(b, A, m_base_intrinsic=20.0, f_s=1.0, a_ks=0.0, ext=ext)
        dmags.append(m - m[0])
    for d in dmags[1:]:
        assert np.allclose(d, dmags[0], atol=1e-12), (
            "ACHROMATICITY VIOLATED: magnitude change differs between bands with f_s=1 and "
            "no extinction -- magnification must be identical in every band")

    # --- INVARIANT 2: blending dilutes -------------------------------------------
    b = ROMAN_BANDS["F146"]
    m_full = observed_magnitude(b, A, 20.0, f_s=1.0, a_ks=0.0, ext=ext)
    m_blend = observed_magnitude(b, A, 20.0, f_s=0.2, a_ks=0.0, ext=ext)
    amp_full = m_full[0] - m_full[-1]
    amp_blend = m_blend[0] - m_blend[-1]
    assert amp_blend < amp_full, "blending must REDUCE the observed amplitude"
    assert amp_blend > 0, "a magnified event must still brighten"

    # --- INVARIANT 3: baseline is unchanged by magnification at A=1 ---------------
    for f_s in (0.1, 0.5, 1.0):
        m = observed_magnitude(b, np.array([1.0]), 20.0, f_s=f_s, a_ks=0.0, ext=ext)
        assert abs(m[0] - 20.0) < 1e-9, "A=1 must reproduce the baseline magnitude exactly"

    # --- INVARIANT 4: extinction ordering (the effect the user flagged) -----------
    a_ks = 0.6
    e087 = ext.extinction_mag(ROMAN_BANDS["F087"], a_ks)
    e146 = ext.extinction_mag(ROMAN_BANDS["F146"], a_ks)
    e213 = ext.extinction_mag(ROMAN_BANDS["F213"], a_ks)
    assert e087 > e146 > e213, "extinction must decrease with wavelength (F087 > F146 > F213)"
    ratio = e087 / e213
    assert 4.0 < ratio < 8.0, f"A_F087/A_F213 = {ratio:.1f}, expected ~6 for alpha=2"

    # --- INVARIANT 5: noise and detectability behave monotonically ----------------
    mags = np.array([18.0, 21.0, 24.0, 27.0])
    sig = photometric_sigma(b, mags)
    assert np.all(np.diff(sig) > 0), "photometric error must grow with magnitude"
    det, sat = apply_detectability(b, mags)
    assert det[0] and not det[-1], "bright sources detected, very faint ones not"
    assert not sat[0], "18 mag must not saturate in F146 (limit 14.8)"
    assert apply_detectability(b, np.array([13.0]))[1][0], "13 mag must saturate in F146"

    # --- INVARIANT 6: per-band blending makes ACHROMATIC events band-dependent -----
    # The load-bearing subtlety: magnification is achromatic, yet the OBSERVED amplitude
    # must still differ between bands when the blend has a different colour to the source.
    ref = ROMAN_BANDS["F146"]
    f_ref = 0.4
    # (a) zero colour difference -> f_s identical in every band -> amplitudes identical
    fs_same = [blend_fraction_in_band(f_ref, b, ref, 0.0, ext, 0.0) for b in ROMAN_BANDS.values()]
    assert np.allclose(fs_same, f_ref, atol=1e-12), \
        "with no source-blend colour difference, f_s must be band-independent"
    # (b) redder blend -> f_s,b must ORDER with wavelength, and amplitudes must differ
    fs_col = {n: blend_fraction_in_band(f_ref, b, ref, 1.0, ext, 0.0)
              for n, b in ROMAN_BANDS.items()}
    assert fs_col["F087"] > fs_col["F146"] > fs_col["F213"], (
        f"a redder blend must lower the source fraction at longer wavelengths; got {fs_col}")
    amps = {}
    for n, b in ROMAN_BANDS.items():
        m = observed_magnitude(b, np.array([1.0, 5.0]), 20.0, fs_col[n], 0.0, ext)
        amps[n] = m[0] - m[1]
    assert amps["F087"] > amps["F213"], (
        "ACHROMATIC magnification must still yield band-dependent observed amplitude when "
        f"f_s is band-dependent -- got {amps}. If this fails, a classifier could wrongly "
        "learn 'amplitude differs between bands => not microlensing'.")

    # --- INVARIANT 7: contaminants use the SAME dilution machinery ----------------
    d = np.array([0.0, 0.5, 2.0])
    assert np.allclose(fractional_flux_change(1.0, d), 1.0 + d), "f_s=1 must be undiluted"
    assert np.all(fractional_flux_change(0.3, d) <= fractional_flux_change(1.0, d)), \
        "blending must dilute a variable's fractional change too"

    # --- INVARIANT 8: no NaN/inf anywhere ----------------------------------------
    m = observed_magnitude(b, np.array([1.0, 100.0]), 22.0, 0.3, 0.8, ext)
    assert np.all(np.isfinite(m)) and np.all(np.isfinite(photometric_sigma(b, m))), \
        "NaN/inf produced in photometry"

    if verbose:
        print("photometry.self_test PASSED")
        print(f"  achromaticity: identical dmag across {len(ROMAN_BANDS)} bands (f_s=1, no ext)")
        print(f"  blending: amplitude {amp_full:.2f} -> {amp_blend:.2f} mag at f_s=0.2")
        print(f"  extinction @ A_Ks={a_ks}: F087 {e087:.2f}, F146 {e146:.2f}, F213 {e213:.2f} mag "
              f"(ratio F087/F213 = {ratio:.1f})")


if __name__ == "__main__":
    self_test()

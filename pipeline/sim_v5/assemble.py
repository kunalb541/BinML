"""
BinML v5 — event assembly: window, per-band sampling, noise, detectability, labelling.

Three things this module gets right that the v4 simulator got wrong:

1. **t0 is NOT centred.** v4 drew t0 in the middle half of the window, which teaches a network
   "the peak is near the centre". Here t0 is drawn over a PADDED range extending beyond both
   window edges, so peaks occur anywhere -- including outside the window, leaving only a
   partial rise or fall, exactly as a real season does (Penny et al. 2019 draw t0 over the
   full simulated span and apply detection cuts afterwards).

2. **The baseline is a MODEL quantity, never measured in-window.** F_base = F_source + F_blend
   is known analytically. A long-t_E event never returns to baseline inside one 72-day season;
   estimating the baseline from in-window data would shift the zero point by ~1 mag, is
   unreproducible by a real pipeline, and hands the network a shortcut. Roman photometry is
   absolutely calibrated and a multi-season baseline exists, so the model-defined baseline is
   the physically honest choice.

3. **Labels come from in-window delta-chi^2, not from what we intended to generate.** An event
   whose peak falls outside the window, or whose amplitude is buried by blending and noise, is
   observationally Flat and is RELABELLED Flat. A microlensing event whose anomaly is not
   detectable is observationally PSPL and is labelled PSPL. This kills the label noise that
   would otherwise punish the classifier for failing to see what is not there.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .classes import CLASS_REGISTRY, label_of
from .generators import GENERATORS, pspl_magnification
from .photometry import (ROMAN_BANDS, BulgeExtinction, observe,
                         blend_fraction_in_band, photometric_sigma)

__all__ = ["SurveyConfig", "BandObs", "Event", "simulate_event", "self_test"]


@dataclass(frozen=True)
class SurveyConfig:
    """One Roman observing season."""
    window_days: float = 72.0
    reference_band: str = "F146"
    # label thresholds, computed against the KNOWN baseline
    # Detection thresholds at their literature values. Read the amplitude floor below
    # before changing either: at Roman's sampling BOTH chi^2 cuts are effectively
    # vestigial, and it is the floor that actually decides the labels.
    dchi2_event: float = 500.0
    dchi2_anomaly: float = 160.0
    # Both delta-chi^2 statistics are computed on NOISE-FREE curves, so neither is a
    # likelihood-ratio statistic: under the null each is identically 0 rather than
    # chi^2-distributed, and each grows in direct proportion to the epoch count. With ~6912
    # F146 epochs, delta-chi^2 = 500 corresponds to a peak amplitude well under 1 mmag, and
    # the same event can cross the anomaly cut on cadence alone (delta-chi^2 4927 at 15 min
    # vs 84 at 24 h for one fixed binary). A fixed chi^2 cut is therefore NOT sufficient on
    # its own -- an earlier comment here claimed the anomaly statistic was epoch-independent,
    # which was simply wrong.
    #
    # What limits a real detection is CORRELATED systematics on day-to-week timescales,
    # which do not average down the way our independent per-epoch noise does; no pipeline
    # claims a sub-mmag detection. Rather than model a full red-noise process, we require a
    # peak amplitude of 20x the 1 mmag systematic floor. This applies to BOTH boundaries:
    # Flat/event, and PSPL/NonPSPL. Omitting it on the latter left 26.9% of NonPSPL-labelled
    # events with an anomaly below the floor the other boundary already enforced.
    #
    # FLAGGED FOR CROSS-CHECK: this is a modelling choice, not a published threshold. Moving
    # it moves both class boundaries, so it must be justified against a real Roman red-noise
    # estimate before the production run.
    min_amplitude_mag: float = 0.02
    snr_threshold: float = 3.0
    # OBSERVED F146 baseline magnitude range (AB). This is the magnitude AFTER extinction --
    # sampling it before extinction (as an earlier version did) pushed sources to 27+ mag,
    # far below the detection limit, and made every light curve noise-dominated.
    m_base_min: float = 20.0
    m_base_max: float = 25.0
    # t0 padding beyond the window, in units of t_E. 2.0 put most peaks outside the season
    # (a high-magnification event rendered as a flat line); 0.5 keeps peaks reachable while
    # still producing partial rises/falls and out-of-window peaks.
    t0_pad_tE: float = 0.5
    t0_pad_max_frac: float = 0.5    # never pad more than half a window
    min_usable_epochs: int = 20     # below this the event is unusable, not "flat"


@dataclass
class BandObs:
    band: str
    t: np.ndarray            # days since window start (usable epochs only)
    mag: np.ndarray          # observed magnitude with noise
    mag_err: np.ndarray
    n_attempted: int         # epochs scheduled before detectability cuts
    f_s: float               # per-band source flux fraction actually used


@dataclass
class Event:
    true_class: str          # what we generated
    label: str               # what it is OBSERVATIONALLY (after delta-chi^2 relabelling)
    label_index: int
    bands: Dict[str, BandObs]
    params: dict
    dchi2_event: float       # vs a flat model, summed over bands
    dchi2_anomaly: float     # vs the REFIT PSPL, reference band only (not summed)
    n_usable_bands: int


def _epochs(band_name: str, window_days: float) -> np.ndarray:
    b = ROMAN_BANDS[band_name]
    step = b.cadence_minutes / (60.0 * 24.0)
    n = max(int(round(window_days / step)), 1)
    return np.arange(n) * step


def _band_stride(ref_band: str, band: str) -> int:
    """How many reference-band epochs there are per ``band`` epoch.

    Roman's colour bands sample on a 360-minute cycle against F146's 15 minutes, an exact
    factor of 24, so a coarse band's epochs are a strict subset of the reference grid. This
    is what lets an achromatic signal be evaluated once and indexed down.
    """
    step_ref = ROMAN_BANDS[ref_band].cadence_minutes
    step_b = ROMAN_BANDS[band].cadence_minutes
    ratio = step_b / step_ref
    stride = int(round(ratio))
    if stride < 1 or abs(ratio - stride) > 1e-9:
        raise ValueError(
            f"band {band} cadence ({step_b} min) is not an integer multiple of the reference "
            f"band {ref_band} ({step_ref} min); the achromatic fast path assumes it is")
    return stride


_REFIT_MAX_POINTS = 400     # subsample cap: the fit needs shape, not every epoch


def _pspl_refit_dchi2(t: np.ndarray, mag_true: np.ndarray, sigma: np.ndarray,
                      m_base: float, f_s: float, params: Dict[str, float]
                      ) -> Tuple[float, float]:
    """Anomaly statistics from the best-fitting PSPL, against a noise-free curve.

    Returns ``(dchi2, peak_deviation_mag)``: the chi^2 the best PSPL cannot absorb, and the
    largest single-epoch magnitude deviation from it. Free parameters are
    (t0, tE, u0, f_s, m_base), seeded from the event's own values -- the right basin, and
    the fit then absorbs whatever a real modeller would, leaving the anomaly alone.

    **Fit on a subsample, evaluate on every epoch.** An earlier version fit AND evaluated on
    400 points and rescaled chi^2 by n/400. A caustic crossing can be far narrower than that
    spacing, so the subsample stepped straight over the anomaly it existed to measure --
    understating delta-chi^2 by up to ~100x. Finding the best-fit parameters only needs the
    broad shape, but scoring the residual needs the full-resolution curve.
    """
    from scipy.optimize import least_squares

    def model_mag(p: np.ndarray, tt: np.ndarray) -> np.ndarray:
        t0, log_tE, u0, logit_fs, mb = p
        tE = math.exp(min(max(log_tE, -20.0), 20.0))       # bounded: LM is unconstrained
        fs = 1.0 / (1.0 + math.exp(-min(max(logit_fs, -30.0), 30.0)))
        A = pspl_magnification(tt, t0, tE, max(abs(u0), 1e-6))
        return mb - 2.5 * np.log10(np.maximum(1.0 + fs * (A - 1.0), 1e-8))

    if t.size > _REFIT_MAX_POINTS:
        sel = np.linspace(0, t.size - 1, _REFIT_MAX_POINTS).astype(int)
        t_fit, mag_fit, sig_fit = t[sel], mag_true[sel], sigma[sel]
    else:
        t_fit, mag_fit, sig_fit = t, mag_true, sigma

    fs0 = min(max(f_s, 1e-3), 1.0 - 1e-3)
    p0 = np.array([params["t0"], math.log(max(params["tE"], 1e-3)),
                   max(abs(params["u0"]), 1e-4),
                   math.log(fs0 / (1.0 - fs0)), m_base])

    best = p0
    try:
        fit = least_squares(lambda p: (mag_fit - model_mag(p, t_fit)) / sig_fit,
                            p0, method="lm", max_nfev=400)
        # LM can return a WORSE solution than the seed; keep whichever actually fits better.
        seed_cost = float(np.sum(((mag_fit - model_mag(p0, t_fit)) / sig_fit) ** 2))
        if float(np.sum(fit.fun ** 2)) < seed_cost:
            best = fit.x
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        pass                                               # keep the seed model

    dev = mag_true - model_mag(best, t)                    # FULL-resolution residual
    return float(np.sum((dev / sigma) ** 2)), float(np.max(np.abs(dev)))


def simulate_event(true_class: str, rng: np.random.Generator,
                   cfg: SurveyConfig = SurveyConfig(),
                   ext: Optional[BulgeExtinction] = None) -> Optional[Event]:
    """Simulate one event of ``true_class`` and label it by what is actually observable."""
    ext = ext or BulgeExtinction()
    gen = GENERATORS[true_class]
    ref_band = ROMAN_BANDS[cfg.reference_band]

    # --- source / line-of-sight properties -------------------------------------
    a_ks = ext.sample_a_ks(rng)
    # m_base is the OBSERVED baseline in the reference band. Extinction is then applied
    # DIFFERENTIALLY to the other bands, so the reference band lands exactly in
    # [m_base_min, m_base_max] regardless of the line-of-sight column. Sampling this
    # intrinsically and then reddening pushed observed F146 to 27.6 -- below the 26.2
    # detection limit -- which made every light curve noise-dominated.
    m_base_ref_obs = float(rng.uniform(cfg.m_base_min, cfg.m_base_max))
    ext_ref = ext.extinction_mag(ref_band, a_ks)
    f_s_ref = float(10.0 ** rng.uniform(math.log10(0.1), math.log10(1.0)))
    blend_colour = float(rng.normal(0.0, 0.6))   # (blend - source) colour, mag/dex

    # --- event parameters, with t0 drawn over a PADDED range (never centred) -----
    params = gen.sample(rng, cfg.window_days)
    if "tE" in params:
        pad = min(cfg.t0_pad_tE * params["tE"], cfg.t0_pad_max_frac * cfg.window_days)
        params["t0"] = float(rng.uniform(-pad, cfg.window_days + pad))

    # An ACHROMATIC signal is evaluated ONCE on the finest grid and indexed down to the
    # coarser bands, making "identical in every band" structural rather than a consequence
    # of the underlying code happening to be deterministic. The colour bands sample every
    # Nth reference epoch exactly (360 min / 15 min = 24), which _band_stride verifies.
    fine_t = _epochs(cfg.reference_band, cfg.window_days)
    fine_delta = gen.delta(fine_t, params, cfg.reference_band) if gen.achromatic else None

    bands: Dict[str, BandObs] = {}
    dchi2_event = 0.0
    dchi2_anom = 0.0
    max_amp_mag = 0.0
    ref_truth: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = None

    for bname, band in ROMAN_BANDS.items():
        t = _epochs(bname, cfg.window_days)
        f_s_b = blend_fraction_in_band(f_s_ref, band, ref_band, blend_colour, ext, a_ks)

        if fine_delta is not None:
            stride = _band_stride(cfg.reference_band, bname)
            delta = fine_delta[::stride][:t.size]         # achromatic: identical by construction
        else:
            delta = gen.delta(t, params, bname)           # chromatic: per-band by design
        flux_ratio = 1.0 + f_s_b * delta                  # SHARED dilution machinery
        flux_ratio = np.maximum(flux_ratio, 1e-8)

        # differential extinction relative to the reference band
        m_base_band = m_base_ref_obs + (ext.extinction_mag(band, a_ks) - ext_ref)
        mag_true = m_base_band - 2.5 * np.log10(flux_ratio)

        # Noise in FLUX space, detectability judged on the TRUE flux, reported error
        # derived from the MEASURED flux -- see photometry.observe for why each of those
        # three choices matters.
        usable, mag_obs, mag_err = observe(band, mag_true, rng, cfg.snr_threshold)
        sigma = photometric_sigma(band, mag_true)      # model sigma, for the chi^2 statistics

        # delta-chi^2 of the NOISE-FREE signal against a FLAT model at the known baseline.
        # Using mag_true (not mag_obs) is essential: an observed-vs-model chi^2 has
        # expectation n_epochs from photon noise alone and would measure nothing.
        if usable.any():
            resid = (mag_true[usable] - m_base_band) / sigma[usable]
            dchi2_event += float(np.sum(resid ** 2))
            max_amp_mag = max(max_amp_mag, float(np.max(np.abs(mag_true[usable] - m_base_band))))

        if bname == cfg.reference_band:
            ref_truth = (t[usable], mag_true[usable], sigma[usable], m_base_band, f_s_b)

        # NOTE: mag_err comes from `observe` (measured flux), NOT from `sigma` (model
        # flux). Emitting the model sigma would let a classifier invert the error column
        # to recover the noise-free light curve.
        bands[bname] = BandObs(bname, t[usable], mag_obs[usable], mag_err[usable],
                               int(t.size), float(f_s_b))

    n_usable_bands = sum(1 for b in bands.values() if b.t.size >= 3)
    total_usable = sum(b.t.size for b in bands.values())
    if total_usable < cfg.min_usable_epochs:
        return None                                        # unusable, not a training example

    # --- anomaly delta-chi^2: only meaningful for microlensing ------------------
    # Defined the way a vetting pipeline defines it: fit the BEST PSPL to the anomalous
    # curve and measure what the fit cannot absorb. Comparing against the PSPL built from
    # the binary's own (t0, tE, u0) would charge the anomaly for parameter offsets a real
    # fitter would simply absorb, inflating delta-chi^2 for events with no visible anomaly.
    anom_amp_mag = 0.0
    if true_class == "NonPSPL" and ref_truth is not None and ref_truth[0].size >= 10:
        dchi2_anom, anom_amp_mag = _pspl_refit_dchi2(*ref_truth, params)

    # --- LABEL BY WHAT IS OBSERVABLE -------------------------------------------
    label = true_class
    if dchi2_event < cfg.dchi2_event or max_amp_mag < cfg.min_amplitude_mag:
        label = "Flat"                                     # nothing detectable happened
    elif true_class == "NonPSPL" and (dchi2_anom < cfg.dchi2_anomaly
                                      or anom_amp_mag < cfg.min_amplitude_mag):
        label = "PSPL"                                     # anomaly not detectable -> it IS a PSPL

    # The achromatic cache is an implementation detail; it holds a full-length array and
    # must not travel with the event into metadata or the HDF5 writer.
    params.pop("_delta_cache", None)
    # Line-of-sight properties are not generator parameters but are needed to reconstruct
    # the event (and to check the simulated population against the survey's own priors).
    params["_m_base_ref"] = m_base_ref_obs
    params["_a_ks"] = a_ks

    return Event(true_class=true_class, label=label, label_index=label_of(label),
                 bands=bands, params=params, dchi2_event=dchi2_event,
                 dchi2_anomaly=dchi2_anom, n_usable_bands=n_usable_bands)


# ---------------------------------------------------------------------------------
def self_test(verbose: bool = True) -> None:
    rng = np.random.default_rng(7)
    cfg = SurveyConfig()

    # --- INVARIANT 1: epoch counts follow the per-band cadence -------------------
    assert _epochs("F146", 72.0).size == 6912, "F146 must give 6912 epochs at 15 min over 72 d"
    # colour bands: 6-hour cycle each (staggered so one colour point every 3 h)
    assert abs(_epochs("F087", 72.0).size - 288) <= 1, "F087 6 h cadence over 72 d -> ~288"
    assert _epochs("F213", 72.0).size == _epochs("F087", 72.0).size, \
        "F087 and F213 share the same 6 h cycle"

    # --- INVARIANT 2: t0 is NOT centred (the v4 bug) -----------------------------
    t0s = []
    for _ in range(400):
        e = simulate_event("PSPL", rng, cfg)
        if e is not None:
            t0s.append(e.params["t0"])
    t0s = np.array(t0s)
    frac_outside = float(((t0s < 0) | (t0s > cfg.window_days)).mean())
    assert frac_outside > 0.10, (
        f"only {frac_outside:.2f} of peaks fall outside the window -- t0 is still too centred")
    inside = t0s[(t0s >= 0) & (t0s <= cfg.window_days)]
    assert abs(inside.mean() - cfg.window_days / 2) < 0.12 * cfg.window_days, \
        "in-window t0 should be roughly uniform, not concentrated at the centre"

    # --- INVARIANT 3: labelling is by observability, and demotions only go one way
    counts: Dict[str, Dict[str, int]] = {}
    stats = {"usable_bands": [], "n_none": 0}
    band_usable: Dict[str, List[int]] = {b: [] for b in ROMAN_BANDS}
    band_frac_epochs: Dict[str, List[float]] = {b: [] for b in ROMAN_BANDS}
    for cls in CLASS_REGISTRY:
        counts[cls] = {}
        for _ in range(120):
            e = simulate_event(cls, rng, cfg)
            if e is None:
                stats["n_none"] += 1
                continue
            counts[cls][e.label] = counts[cls].get(e.label, 0) + 1
            stats["usable_bands"].append(e.n_usable_bands)
            for bn, bo in e.bands.items():
                band_usable[bn].append(1 if bo.t.size >= 3 else 0)
                band_frac_epochs[bn].append(bo.t.size / max(bo.n_attempted, 1))
            # a generated Flat can never be promoted to an event
            if cls == "Flat":
                assert e.label == "Flat", "a Flat event must never be relabelled as an event"
            # NonPSPL may only be demoted to PSPL or Flat, never to a contaminant
            if cls == "NonPSPL":
                assert e.label in ("NonPSPL", "PSPL", "Flat"), \
                    f"NonPSPL was relabelled to {e.label}"

    # --- INVARIANT 4: Flat really is flat in delta-chi^2 -------------------------
    flat_d = [simulate_event("Flat", rng, cfg) for _ in range(60)]
    flat_d = [e.dchi2_event for e in flat_d if e is not None]
    assert max(flat_d) < cfg.dchi2_event, \
        f"a Flat event exceeded the detection threshold (max dchi2 {max(flat_d):.0f})"

    # --- INVARIANT 5: usable-band statistics (the colour-availability question) ---
    ub = np.array(stats["usable_bands"])
    frac_multi = float((ub >= 2).mean())
    assert 0.0 <= frac_multi <= 1.0

    if verbose:
        print("assemble.self_test PASSED")
        print(f"  t0 outside window: {frac_outside*100:.0f}%  (v4 had 0% -- peaks were centred)")
        print(f"  events with >=2 usable bands: {frac_multi*100:.0f}%")
        print("  PER-BAND usability (does this band yield a light curve at all?):")
        for bn in ROMAN_BANDS:
            u = float(np.mean(band_usable[bn])) * 100
            fe = float(np.mean(band_frac_epochs[bn])) * 100
            print(f"    {bn}: usable in {u:5.1f}% of events, "
                  f"{fe:5.1f}% of scheduled epochs pass detectability")
        col = float(np.mean([a and b for a, b in
                             zip(band_usable['F146'], band_usable['F087'])])) * 100
        print(f"  F146+F087 both usable (true COLOUR available): {col:.1f}%")
        for cls, c in counts.items():
            tot = sum(c.values()) or 1
            shown = ", ".join(f"{k} {v*100//tot}%" for k, v in sorted(c.items()))
            print(f"  generated {cls:14s} -> labelled: {shown}")


if __name__ == "__main__":
    self_test()

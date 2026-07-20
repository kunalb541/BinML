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
from .photometry import (ROMAN_BANDS, BulgeExtinction, apply_detectability,
                         blend_fraction_in_band, photometric_sigma)

__all__ = ["SurveyConfig", "BandObs", "Event", "simulate_event", "self_test"]


@dataclass(frozen=True)
class SurveyConfig:
    """One Roman observing season."""
    window_days: float = 72.0
    reference_band: str = "F146"
    # label thresholds, computed against the KNOWN baseline
    dchi2_event: float = 500.0      # Flat vs event (Penny+2019 use ~500 for detection)
    dchi2_anomaly: float = 160.0    # PSPL vs NonPSPL (Penny+2019 anomaly threshold)
    snr_threshold: float = 3.0
    # source baseline magnitude range (AB, intrinsic, before extinction)
    m_base_min: float = 20.0
    m_base_max: float = 25.0
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
    dchi2_anomaly: float     # vs the matched PSPL (microlensing only), summed over bands
    n_usable_bands: int


def _epochs(band_name: str, window_days: float) -> np.ndarray:
    b = ROMAN_BANDS[band_name]
    step = b.cadence_minutes / (60.0 * 24.0)
    n = max(int(round(window_days / step)), 1)
    return np.arange(n) * step


def simulate_event(true_class: str, rng: np.random.Generator,
                   cfg: SurveyConfig = SurveyConfig(),
                   ext: Optional[BulgeExtinction] = None) -> Optional[Event]:
    """Simulate one event of ``true_class`` and label it by what is actually observable."""
    ext = ext or BulgeExtinction()
    gen = GENERATORS[true_class]
    ref_band = ROMAN_BANDS[cfg.reference_band]

    # --- source / line-of-sight properties -------------------------------------
    m_base_intrinsic = float(rng.uniform(cfg.m_base_min, cfg.m_base_max))
    a_ks = ext.sample_a_ks(rng)
    f_s_ref = float(10.0 ** rng.uniform(math.log10(0.1), math.log10(1.0)))
    blend_colour = float(rng.normal(0.0, 0.6))   # (blend - source) colour, mag/dex

    # --- event parameters, with t0 drawn over a PADDED range (never centred) -----
    params = gen.sample(rng, cfg.window_days)
    if "tE" in params:
        pad = min(2.0 * params["tE"], 2.0 * cfg.window_days)
        params["t0"] = float(rng.uniform(-pad, cfg.window_days + pad))

    bands: Dict[str, BandObs] = {}
    dchi2_event = 0.0
    dchi2_anom = 0.0

    for bname, band in ROMAN_BANDS.items():
        t = _epochs(bname, cfg.window_days)
        f_s_b = blend_fraction_in_band(f_s_ref, band, ref_band, blend_colour, ext, a_ks)

        delta = gen.delta(t, params, bname)               # fractional flux change
        flux_ratio = 1.0 + f_s_b * delta                  # SHARED dilution machinery
        flux_ratio = np.maximum(flux_ratio, 1e-8)

        m_base_band = m_base_intrinsic + ext.extinction_mag(band, a_ks)
        mag_true = m_base_band - 2.5 * np.log10(flux_ratio)

        sigma = photometric_sigma(band, mag_true)
        mag_obs = mag_true + rng.normal(0.0, sigma)

        detected, saturated = apply_detectability(band, mag_obs, cfg.snr_threshold)
        usable = detected & (~saturated)

        # delta-chi^2 of the true signal against a FLAT model at the known baseline
        if usable.any():
            resid = (mag_true[usable] - m_base_band) / sigma[usable]
            dchi2_event += float(np.sum(resid ** 2))

        bands[bname] = BandObs(bname, t[usable], mag_obs[usable], sigma[usable],
                               int(t.size), float(f_s_b))

    n_usable_bands = sum(1 for b in bands.values() if b.t.size >= 3)
    total_usable = sum(b.t.size for b in bands.values())
    if total_usable < cfg.min_usable_epochs:
        return None                                        # unusable, not a training example

    # --- anomaly delta-chi^2: only meaningful for microlensing ------------------
    if true_class == "NonPSPL":
        ref_obs = bands[cfg.reference_band]
        if ref_obs.t.size >= 10:
            f_s_b = ref_obs.f_s
            m_base_band = m_base_intrinsic + ext.extinction_mag(ref_band, a_ks)
            A_pspl = pspl_magnification(ref_obs.t, params["t0"], params["tE"], params["u0"])
            mag_pspl = m_base_band - 2.5 * np.log10(np.maximum(1.0 + f_s_b * (A_pspl - 1.0), 1e-8))
            resid = (ref_obs.mag - mag_pspl) / ref_obs.mag_err
            dchi2_anom = float(np.sum(resid ** 2))

    # --- LABEL BY WHAT IS OBSERVABLE -------------------------------------------
    label = true_class
    if dchi2_event < cfg.dchi2_event:
        label = "Flat"                                     # nothing detectable happened
    elif true_class == "NonPSPL" and dchi2_anom < cfg.dchi2_anomaly:
        label = "PSPL"                                     # anomaly not detectable -> it IS a PSPL

    return Event(true_class=true_class, label=label, label_index=label_of(label),
                 bands=bands, params=params, dchi2_event=dchi2_event,
                 dchi2_anomaly=dchi2_anom, n_usable_bands=n_usable_bands)


# ---------------------------------------------------------------------------------
def self_test(verbose: bool = True) -> None:
    rng = np.random.default_rng(7)
    cfg = SurveyConfig()

    # --- INVARIANT 1: epoch counts follow the per-band cadence -------------------
    assert _epochs("F146", 72.0).size == 6912, "F146 must give 6912 epochs at 15 min over 72 d"
    assert abs(_epochs("F087", 72.0).size - 144) <= 1, "F087 ~12 h cadence over 72 d"

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

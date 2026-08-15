"""
BinML v5 — the class registry.

Six classes, organised around the questions an observer actually asks:

    1. Is there an event at all?        Flat        vs everything else
    2. Is it microlensing?              {PSPL, NonPSPL} vs the contaminants
    3. Is it anomalous?                 PSPL        vs NonPSPL

The microlensing split is deliberately **PSPL vs NON-PSPL**, not "single vs binary lens".
NonPSPL is defined *operationally* -- by DEVIATION from the best-fitting point-source
point-lens model (delta-chi^2 above a threshold).  The released generator and trained
checkpoint cover static binary-lens (2L1S) events with finite-source magnification only;
they do not cover binary-source (1L2S), parallax, or lens orbital motion.  The broader
operational label could encompass those effects only after they are added to training and
evaluation.

The registry is data, not code: adding a class means adding an entry plus a generator, and
metrics/heads key off the registry so they extend automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

__all__ = ["ClassSpec", "CLASS_REGISTRY", "CLASS_NAMES", "N_CLASSES",
           "MICROLENSING_CLASSES", "CONTAMINANT_CLASSES", "label_of"]


@dataclass(frozen=True)
class ClassSpec:
    index: int
    name: str
    is_microlensing: bool
    description: str
    # Approximate on-sky density in Roman's bulge fields (objects/deg^2), from the
    # OGLE/VVV catalogue counts gathered in the literature review. Used as an importance
    # WEIGHT, not as the sampling rate -- we sample evenly for coverage, because the true
    # rates are wildly imbalanced and would starve the rare-but-critical classes.
    density_per_deg2: float
    sampling_weight: float          # relative number of events to GENERATE
    confusion_risk: str             # how dangerous it is as a microlensing impostor


CLASS_REGISTRY: Dict[str, ClassSpec] = {
    "Flat": ClassSpec(
        0, "Flat", False,
        "Constant source plus photometric noise; no intrinsic variability.",
        density_per_deg2=0.0, sampling_weight=1.0, confusion_risk="none"),

    "PSPL": ClassSpec(
        1, "PSPL", True,
        "Standard point-source point-lens microlensing: a smooth, symmetric, achromatic "
        "magnification with no significant deviation from the 1L1S model.",
        density_per_deg2=0.0, sampling_weight=2.0, confusion_risk="n/a"),

    "NonPSPL": ClassSpec(
        2, "NonPSPL", True,
        "Detectable anomalous microlensing under the released policy. The current training "
        "generator contains static binary lenses (2L1S, planetary through stellar mass ratio) "
        "with finite-source magnification; 1L2S, parallax, and lens orbital motion are omitted. "
        "This is the science-critical class containing detectable binary and planetary lenses.",
        density_per_deg2=0.0, sampling_weight=3.0, confusion_risk="n/a"),

    # --- contaminants: grouped by BEHAVIOUR (what the classifier sees), not taxonomy -----
    "PeriodicVar": ClassSpec(
        3, "PeriodicVar", False,
        "Short-period coherent variables completing many cycles per season: RR Lyrae "
        "(ab/c), Cepheids, eclipsing and contact binaries, delta Scuti. Individually easy "
        "to reject on periodicity, but abundant, and heavily blended examples can slip "
        "below naive variability cuts.",
        density_per_deg2=3000.0, sampling_weight=1.5, confusion_risk="low"),

    "LongPeriodVar": ClassSpec(
        4, "LongPeriodVar", False,
        "Red-giant long-period and semiregular variables: Miras, SRVs and OSARGs. THE most "
        "dangerous impostors: with periods comparable to or longer than one 72-day season "
        "they present a single smooth bump or slow trend that mimics a long-t_E, heavily "
        "blended microlensing event. OSARGs alone are the most numerous variable class in "
        "the bulge (~2800/deg^2).",
        density_per_deg2=3500.0, sampling_weight=3.0, confusion_risk="high"),

    "Eruptive": ClassSpec(
        5, "Eruptive", False,
        "Single-episode brighteners: dwarf-nova / CV outbursts (incl. long WZ Sge-type "
        "superoutbursts) and Be-star 'bumpers'. A lone outburst in one season on a flat "
        "baseline is structurally what a microlensing detector looks for; Be bumpers were "
        "historically the leading microlensing false positive.",
        density_per_deg2=10.0, sampling_weight=2.5, confusion_risk="high"),
}

CLASS_NAMES: List[str] = [c.name for c in sorted(CLASS_REGISTRY.values(), key=lambda c: c.index)]
N_CLASSES: int = len(CLASS_NAMES)
MICROLENSING_CLASSES: List[str] = [n for n, c in CLASS_REGISTRY.items() if c.is_microlensing]
CONTAMINANT_CLASSES: List[str] = [n for n, c in CLASS_REGISTRY.items()
                                  if not c.is_microlensing and n != "Flat"]


def label_of(name: str) -> int:
    return CLASS_REGISTRY[name].index


def self_test(verbose: bool = True) -> None:
    idx = sorted(c.index for c in CLASS_REGISTRY.values())
    assert idx == list(range(len(idx))), f"class indices must be contiguous from 0, got {idx}"
    assert len(CLASS_NAMES) == len(set(CLASS_NAMES)), "duplicate class names"
    assert set(MICROLENSING_CLASSES) == {"PSPL", "NonPSPL"}, \
        "microlensing must be exactly the PSPL / NonPSPL pair"
    assert "Flat" not in CONTAMINANT_CLASSES and len(CONTAMINANT_CLASSES) == 3
    assert all(c.sampling_weight > 0 for c in CLASS_REGISTRY.values())
    # the two high-confusion classes must be sampled at least as heavily as the easy ones
    hi = [c for c in CLASS_REGISTRY.values() if c.confusion_risk == "high"]
    lo = [c for c in CLASS_REGISTRY.values() if c.confusion_risk == "low"]
    assert min(c.sampling_weight for c in hi) >= max(c.sampling_weight for c in lo), \
        "high-confusion contaminants must not be under-sampled relative to easy ones"
    if verbose:
        print("classes.self_test PASSED")
        for c in sorted(CLASS_REGISTRY.values(), key=lambda c: c.index):
            tag = "microlensing" if c.is_microlensing else "contaminant "
            print(f"  {c.index}  {c.name:14s} {tag}  weight={c.sampling_weight:.1f}  "
                  f"confusion={c.confusion_risk}")


if __name__ == "__main__":
    self_test()

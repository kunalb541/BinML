"""BinML — the 6-class inference API.

    import binml
    clf = binml.Classifier()
    r = clf.predict({"F146": (t, mag), "F087": (t2, mag2), "F213": (t3, mag3)})
    print(r)                       # <BinML NonPSPL 0.98 | microlensing 0.99 anomalous 0.98>
    r.probabilities                # {'Flat':.., 'PSPL':.., 'NonPSPL':.., ...}

Single-band (F146 only) is accepted too: ``clf.predict(t, mag)``.
"""
from __future__ import annotations

import os
import operator
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np

from pipeline.model import BAND_BINS, BinMLv5, ModelConfigV5
from .preprocess import Tokens, to_tokens

_HERE = os.path.dirname(__file__)
_DEFAULT_WEIGHTS = os.path.join(_HERE, "weights", "binml.pt")
CLASS_NAMES = ["Flat", "PSPL", "NonPSPL", "PeriodicVar", "LongPeriodVar", "Eruptive"]

BandInput = Dict[str, Tuple[np.ndarray, np.ndarray]]


@dataclass
class Prediction:
    probabilities: Dict[str, float]
    label: str
    n_points: Dict[str, int]

    @property
    def confidence(self) -> float:
        return max(self.probabilities.values())

    @property
    def is_microlensing(self) -> float:
        """P(PSPL) + P(NonPSPL) — 'is this a microlensing event at all'."""
        return self.probabilities["PSPL"] + self.probabilities["NonPSPL"]

    @property
    def is_anomalous(self) -> float:
        """P(NonPSPL) — 'is it binary/planetary rather than a plain single lens'."""
        return self.probabilities["NonPSPL"]

    def __repr__(self) -> str:
        return (f"<BinML {self.label} {self.confidence:.2f} | "
                f"microlensing {self.is_microlensing:.2f} anomalous {self.is_anomalous:.2f}>")


class Classifier:
    """Load BinML and classify Roman light curves into six classes."""

    def __init__(self, weights: Optional[str] = None, device: str = "cpu"):
        import torch
        self._torch = torch
        self.device = device
        ck = torch.load(weights or _DEFAULT_WEIGHTS, map_location="cpu")
        cfg = ModelConfigV5(**{k: v for k, v in ck["config"].items()
                               if k in ModelConfigV5.__dataclass_fields__})
        net = BinMLv5(cfg).to(device)
        net.load_state_dict(ck["model"])
        net.eval()
        self._net = net
        self._mag_scale = float(ck.get("mag_scale", 1.0))
        self.class_names = list(ck.get("class_names", CLASS_NAMES))

    # -- core: tokens -> probabilities (one or many events) -------------------------------
    def _forward(self, feats_np: Dict[str, np.ndarray], present_np: Dict[str, np.ndarray]):
        torch = self._torch
        feats, present = {}, {}
        for b, L in BAND_BINS.items():
            x = feats_np[b].astype(np.float32)                    # (B, L, 3) mean/min/max
            fr = present_np[b].astype(np.float32)                 # (B, L) frac
            obs = np.isfinite(x[:, :, 0]).astype(np.float32)      # mask BEFORE nan->0
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            t = np.concatenate([x / self._mag_scale, fr[:, :, None], obs[:, :, None]], axis=2)
            feats[b] = torch.tensor(t, dtype=torch.float32, device=self.device)
            present[b] = feats[b][..., 4].sum(dim=1) > 0
        with torch.inference_mode():
            logits = self._net(feats, present).float().cpu().numpy()
        z = logits - logits.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def predict_tokens(self, tok: Tokens) -> Prediction:
        feats = {b: tok.feat[b][None] for b in BAND_BINS}     # add batch dim
        frac = {b: tok.frac[b][None] for b in BAND_BINS}
        p = self._forward(feats, frac)[0]
        probs = {c: float(p[i]) for i, c in enumerate(self.class_names)}
        return Prediction(probabilities=probs,
                          label=self.class_names[int(p.argmax())],
                          n_points=tok.n_points)

    def predict(self,
                light_curve: Union[BandInput, np.ndarray],
                mag: Optional[np.ndarray] = None,
                mag_err: Optional[np.ndarray] = None,
                m_base_ref: Optional[float] = None,
                t_start: Optional[float] = None) -> Prediction:
        """Classify one event.

        ``predict({"F146": (t, mag), ...})`` for multi-band, or ``predict(t, mag)`` for
        F146-only. ``m_base_ref`` (the F146 baseline magnitude) is recommended; otherwise
        estimated (with a warning). NOTE: ``mag_err`` is accepted for call-signature convenience
        but IGNORED — the model is error-agnostic (it consumes binned magnitudes only).
        """
        if isinstance(light_curve, dict):
            bands = self._coerce_bands(light_curve)
        else:                                                 # single-band -> F146
            bands = {"F146": (np.asarray(light_curve, float), np.asarray(mag, float))}
        tok = to_tokens(bands, m_base_ref=m_base_ref, t_start=t_start)
        return self.predict_tokens(tok)

    @staticmethod
    def _coerce_bands(light_curve: BandInput) -> BandInput:
        """Normalise public band inputs and turn malformed tuples into useful errors."""
        bands = {}
        for band, values in light_curve.items():
            if not isinstance(values, (tuple, list)) or len(values) != 2:
                raise ValueError(f"{band}: expected a (time, magnitude) pair")
            bands[band] = (np.asarray(values[0], float), np.asarray(values[1], float))
        return bands

    def predict_evolution(self, light_curve: BandInput, m_base_ref=None, t_start=None,
                          n_steps: int = 16):
        """Class probabilities as a 72-day window is progressively revealed.

        Returns ``(days, probs)`` with ``probs`` shape ``(n_steps, 6)``. A row is NaN when
        no usable F146 observation has arrived by that cut; the model has no calibrated
        prior-only output, so an all-empty tensor is not scored.
        """
        try:
            n_steps = operator.index(n_steps)
        except TypeError as exc:
            raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}") from exc
        if n_steps <= 0:
            raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")

        bands = self._coerce_bands(light_curve)
        # Validate every band and both metadata fields once before any inference. Besides
        # providing consistent errors with predict(), this freezes the baseline estimate for
        # every prefix instead of re-estimating it as new points arrive.
        full = to_tokens(bands, m_base_ref=m_base_ref, t_start=t_start)
        t_start = full.t_start
        m_base_ref = full.m_base["ref"]
        fracs = np.linspace(1.0 / n_steps, 1.0, n_steps)
        out = np.full((n_steps, len(self.class_names)), np.nan, np.float32)
        for k, fr in enumerate(fracs):
            cut = t_start + fr * 72.0
            revealed = {b: (t[t <= cut], m[t <= cut]) for b, (t, m) in bands.items()}
            t146, m146 = revealed["F146"]
            if not np.any(np.isfinite(t146) & np.isfinite(m146)):
                continue
            tok = to_tokens(revealed, m_base_ref=m_base_ref, t_start=t_start)
            out[k] = self._forward({b: tok.feat[b][None] for b in BAND_BINS},
                                   {b: tok.frac[b][None] for b in BAND_BINS})[0]
        return fracs * 72.0, out

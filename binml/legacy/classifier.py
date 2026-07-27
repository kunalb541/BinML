"""
High-level BinML inference API.

    import binml
    clf = binml.Classifier()                      # fine-tuned model, CPU
    r = clf.predict(time, mag, mag_err)           # a real light curve
    print(r)                                       # Flat/PSPL/Binary + verdicts
    r.is_microlensing   # P(PSPL)+P(Binary)  -- "is this a microlensing event?"
    r.is_anomalous      # P(Binary)          -- "is it binary/planetary, not plain PSPL?"

BinML is a 3-class microlensing classifier: Flat (no event), PSPL (single-lens
microlensing -- the detection proxy), Binary (planetary/binary lens -- the anomalous
class). A binary is NOT a PSPL; the two questions the model answers are (1) is there a
microlensing event, and (2) is it anomalous.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .preprocess import Preprocessed, preprocess

CLASS_NAMES = ("Flat", "PSPL", "Binary")
_WEIGHTS_DIR = Path(__file__).parent.parent / "weights"  # shared binml/weights/
_BUNDLED = {"finetuned": "binml_finetuned.pt", "base": "binml_base.pt"}
_EPS = 1e-8


@dataclass
class Prediction:
    """Result of classifying one light curve."""
    probabilities: Dict[str, float]
    label: str
    confidence: float
    # -- scientific views (see module docstring) --
    is_microlensing: float   # P(PSPL) + P(Binary)   -> detection
    is_anomalous: float      # P(Binary)             -> characterization (binary != PSPL)
    # -- context --
    n_points: int
    t0: float
    m_base: float
    peak_magnification: float
    _pre: Optional[Preprocessed] = field(default=None, repr=False)

    def __repr__(self) -> str:
        p = self.probabilities
        bar = "  ".join(f"{k} {p[k]:.3f}" for k in CLASS_NAMES)
        return (f"<BinML {self.label} (conf {self.confidence:.2f}) | {bar} | "
                f"microlensing={self.is_microlensing:.2f} anomalous={self.is_anomalous:.2f} "
                f"| n={self.n_points} peakA={self.peak_magnification:.1f}>")


@dataclass
class Evolution:
    """Probability evolution as the light curve is revealed prefix by prefix."""
    days_from_peak: np.ndarray       # time of the latest observation fed
    probabilities: np.ndarray        # (steps, 3) columns = Flat, PSPL, Binary
    n_points: np.ndarray             # prefix length at each step
    final: Prediction


class Classifier:
    """Load a trained BinML model and classify light curves."""

    def __init__(self, model: str = "finetuned", device: str = "cpu"):
        """
        Parameters
        ----------
        model : {'finetuned', 'base'} or path
            'finetuned' (default) = higher planetary/binary sensitivity;
            'base' = balanced 3-class. A filesystem path to a .pt checkpoint also works.
        device : str
            'cpu' (default) or 'cuda'.
        """
        import torch  # deferred so `import binml` is cheap and torch errors are actionable
        from .model import ModelConfig, RomanMicrolensingClassifier

        self._torch = torch
        self.device = device
        path = _BUNDLED.get(model)
        ckpt_path = (_WEIGHTS_DIR / path) if path else Path(model)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {ckpt_path}. Use model in {list(_BUNDLED)} or a valid path.")

        ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        cfg = ck["model_config"]
        self._cfg = ModelConfig(**cfg) if isinstance(cfg, dict) else cfg
        net = RomanMicrolensingClassifier(self._cfg).to(device)
        sd = ck["model_state_dict"]
        sd = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
        net.load_state_dict(sd, strict=True)
        net.eval()
        self._net = net

        s = ck["stats"]
        self._fm = float(s.get("flux_mean", s.get("magnification_mean")))
        self._fs = float(s.get("flux_std", s.get("magnification_std")))
        self._dm = float(s["delta_t_mean"])
        self._ds = float(s["delta_t_std"])
        self.seq_len = int(getattr(self._cfg, "max_length", 6912) or 6912)
        self.model_name = model
        self.epoch = ck.get("epoch")

    # ------------------------------------------------------------------ internals
    def _forward(self, flux: np.ndarray, delta_t: np.ndarray, length: int) -> np.ndarray:
        torch = self._torch
        fn = (flux - self._fm) / (self._fs + _EPS)
        dn = (delta_t - self._dm) / (self._ds + _EPS)
        xf = torch.from_numpy(fn).float().to(self.device)
        xd = torch.from_numpy(dn).float().to(self.device)
        xl = torch.tensor([length], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self._net(xf, xd, xl)
            logits = out if torch.is_tensor(out) else out["logits"]
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()[0]
        return probs

    def predict_arrays(self, flux: np.ndarray, delta_t: np.ndarray, batch: int = 256) -> np.ndarray:
        """Predict directly from stored grid arrays (magnification + delta_t, with 0 for
        unobserved slots), e.g. rows of a compact HDF5 test set. Compaction (moving valid
        observations to the prefix) is done vectorised on-device. Returns (N, 3) probs."""
        torch = self._torch
        flux = np.asarray(flux, dtype=np.float32)
        delta_t = np.asarray(delta_t, dtype=np.float32)
        if flux.ndim == 1:
            flux = flux[None, :]; delta_t = delta_t[None, :]
        n = flux.shape[0]
        out = np.zeros((n, 3), np.float32)
        for s0 in range(0, n, batch):
            e = min(s0 + batch, n)
            fx = torch.from_numpy(flux[s0:e]).to(self.device)
            dx = torch.from_numpy(delta_t[s0:e]).to(self.device)
            valid = fx != 0
            lengths = valid.sum(1).clamp(min=1)
            perm = torch.argsort(valid.int(), dim=1, descending=True, stable=True)
            fc = torch.gather(fx, 1, perm); dc = torch.gather(dx, 1, perm)
            fn = (fc - self._fm) / (self._fs + _EPS)
            dn = (dc - self._dm) / (self._ds + _EPS)
            with torch.no_grad():
                o = self._net(fn, dn, lengths)
                logits = o if torch.is_tensor(o) else o["logits"]
                out[s0:e] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        return out

    def _to_prediction(self, probs: np.ndarray, pre: Preprocessed) -> Prediction:
        pdict = {CLASS_NAMES[i]: float(probs[i]) for i in range(3)}
        k = int(np.argmax(probs))
        return Prediction(
            probabilities=pdict, label=CLASS_NAMES[k], confidence=float(probs[k]),
            is_microlensing=float(probs[1] + probs[2]), is_anomalous=float(probs[2]),
            n_points=pre.length, t0=pre.t0, m_base=pre.m_base,
            peak_magnification=pre.peak_magnification, _pre=pre,
        )

    # ------------------------------------------------------------------ public API
    def predict(self, time, mag, mag_err=None, *, t0=None, m_base=None,
                is_flux=False, window_days=72.0) -> Prediction:
        """Classify one light curve. Returns a :class:`Prediction`."""
        pre = preprocess(time, mag, mag_err, seq_len=self.seq_len, window_days=window_days,
                         t0=t0, m_base=m_base, is_flux=is_flux)
        return self._to_prediction(self._forward(pre.flux, pre.delta_t, pre.length), pre)

    def predict_proba(self, time, mag, mag_err=None, **kw) -> Dict[str, float]:
        """Just the {Flat, PSPL, Binary} probability dict."""
        return self.predict(time, mag, mag_err, **kw).probabilities

    def predict_evolution(self, time, mag, mag_err=None, *, steps=200, **kw) -> Evolution:
        """Feed growing prefixes of the curve and record how the probabilities evolve.

        The network is causal, so a prefix prediction only ever sees data up to that time.
        """
        pre = preprocess(time, mag, mag_err, seq_len=self.seq_len, **kw)
        n = pre.length
        ks = sorted(set(list(range(3, n + 1, max(1, n // steps))) + [n]))
        P = np.zeros((len(ks), 3))
        for i, k in enumerate(ks):
            fk = np.zeros_like(pre.flux); dk = np.zeros_like(pre.delta_t)
            fk[0, :k] = pre.flux[0, :k]; dk[0, :k] = pre.delta_t[0, :k]
            P[i] = self._forward(fk, dk, k)
        ks = np.array(ks)
        xt = pre.time[ks - 1] - pre.t0
        final = self._to_prediction(P[-1], pre)
        return Evolution(days_from_peak=xt, probabilities=P, n_points=ks, final=final)

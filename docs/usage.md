# Using BinML (the 6-class inference API)

`pip install binml` gives a ready-to-use classifier; the weights ship with the package.

## Classify one event

```python
import binml
clf = binml.Classifier()          # device="cpu" by default; pass device="cuda"/"mps" if you have it
```

**Multi-band** — a dict of `{band: (time_days, magnitude)}`. F146 is required; F087/F213 are
optional (the model masks an absent colour band):

```python
r = clf.predict(
    {"F146": (t146, m146), "F087": (t087, m087), "F213": (t213, m213)},
    m_base_ref=22.1,      # F146 baseline (quiescent) magnitude -- recommended
    t_start=None,         # day the 72-day window opens; default = first observation
)
r.probabilities           # {'Flat':.., 'PSPL':.., 'NonPSPL':.., 'PeriodicVar':.., 'LongPeriodVar':.., 'Eruptive':..}
r.label                   # argmax class
r.is_microlensing         # P(PSPL) + P(NonPSPL)
r.is_anomalous            # P(NonPSPL) -- binary/planetary vs plain single lens
r.confidence              # max probability
```

**Single band** (F146 only):

```python
r = clf.predict(t146, m146, m_base_ref=22.1)
```

## Inputs that matter

- **Times are in days**, with any zero point, *shared across bands*.
- **`m_base_ref`** is the single F146 baseline magnitude subtracted from every band (this is
  exactly what the model was trained on, and it lets the colour bands carry the source colour).
  Provide a catalogue value when you have one. If omitted, it is estimated from the faint tail of
  F146 — only reliable for short, well-sampled events.
- **Realistic photometry matters.** The model was trained on Roman photometry *with*
  blending and per-epoch noise/detectability. A perfectly-sampled, near-noiseless toy curve is
  out-of-distribution and may be misread (e.g. a cuspy high-magnification peak with no scatter can
  look like a caustic). Feed real observed light curves; for synthetic tests include a blend
  fraction and realistic noise.
- **Cadence.** The model expects Roman-quality sampling (dense 15-min F146). It bins your points
  onto the Roman epoch grid; sparse data yields low `observed-fraction` bins, which the model
  reads as "poorly observed."

## The real-time cascade

Probabilities as the season is progressively revealed — this is BinML's distinctive capability
(it does not flag a binary before the caustic is on screen):

```python
days, probs = clf.predict_evolution({"F146": (t146, m146)}, m_base_ref=22.1, n_steps=16)
# probs: (n_steps, 6); probs[:, 2] is P(NonPSPL) over time
```

## Command line

```bash
binml classify lc.csv --m-base 22.1        # CSV/whitespace file: time, mag[, mag_err] (F146)
binml --version
```

## Batch / advanced

`clf.predict_tokens(tokens)` accepts pre-binned `binml.Tokens` (from `binml.to_tokens(...)`) if
you want to control binning or batch many events. The underlying model and binning are the exact
research code (`pipeline.model`, `pipeline.cache`) — the package reproduces the training
representation bit-for-bit (verified: package vs research-path predictions agree to <0.001).

## Legacy 3-class model

The original Flat/PSPL/Binary classifier is preserved:

```python
from binml.legacy import Classifier as Legacy3Class
```
See [legacy_3class.md](legacy_3class.md).

# Using BinML (the 6-class inference API)

Install from a checked-out repository with `pip install .` (or `pip install -e .` for development).
The project is not currently published on PyPI; the built wheel includes the trained weights.
The checkpoint is a synthetic benchmark trained on a legacy one-season Roman-like schedule, not
on the current GBTDS cadence and photometric calibration.

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
- **Photometry matters.** The model was trained on synthetic legacy Roman-like photometry with
  blending and per-epoch noise/detectability. A perfectly sampled, near-noiseless toy curve is
  out of distribution. The released colour-band zeropoints, saturation, and backgrounds also have
  known discrepancies from the current Roman calibration, so real photometry is a sim-to-real
  transfer test rather than an in-distribution input.
- **Cadence.** The model expects dense sampling like its legacy 15-min F146 grid. It bins points
  onto that fixed grid; sparse data yields low `observed-fraction` bins. The current multi-season
  GBTDS schedule is not represented by this interface.
- **Uncertainties are not features.** `Classifier.predict(..., mag_err=...)` accepts `mag_err` for
  call-signature compatibility but ignores it. Likewise, a third CSV column is ignored by the
  current CLI; the network consumes binned magnitudes, observed fractions, and masks.

## Partial-season probabilities

`predict_evolution` returns probability traces as a season is progressively revealed; it does not
implement an alert threshold, persistence rule, or broker. Its default `n_steps=16` samples every
4.5 days. The paper's 1.6% premature-crossing result instead used 144 half-day cuts, a frozen
threshold, and a separate event-level reduction. That scan contains eligible binaries only, so it
does not establish streaming purity or false-alert burden on contaminants:

```python
days, probs = clf.predict_evolution({"F146": (t146, m146)}, m_base_ref=22.1, n_steps=144)
# probs: (n_steps, 6); probs[:, 2] is P(NonPSPL) over time
```

## Command line

```bash
binml classify lc.csv --m-base 22.1        # CSV/whitespace: time, mag[, ignored third column]
binml --version
```

## Batch / advanced

`clf.predict_tokens(tokens)` accepts one pre-binned `binml.Tokens` event (from
`binml.to_tokens(...)`) if you want to control binning. Repeated calls can process multiple
events; this public helper is not a batched-event API. The underlying model and binning are the exact
research code (`pipeline.model`, `pipeline.cache`) — the package reproduces the training
representation bit-for-bit (verified: package vs research-path predictions agree to <0.001).

## Legacy 3-class model

The original Flat/PSPL/Binary classifier is preserved:

```python
from binml.legacy import Classifier as Legacy3Class
```
See [legacy_3class.md](legacy_3class.md).

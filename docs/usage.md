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
threshold, and a separate event-level reduction. That scan contains eligible binaries only; the
alert burden over every class is a separate scan (paper §4, `validation/referee_round.json`), on our simulator
and at the complete-season threshold:

```python
days, probs = clf.predict_evolution({"F146": (t146, m146)}, m_base_ref=22.1, n_steps=144)
# probs: (n_steps, 6); probs[:, 2] is P(NonPSPL) over time
```

## Roman (RMDC26 / GULLS) light curves

`binml.gulls` holds the logic needed to hand an event from the RGES-PIT RMDC26 release (an
independent GULLS simulation of the survey, on the Hugging Face Hub) to the classifier under BinML's
input contract: pinned dataset revision, one contiguous season found from the epoch table, baseline
measured from the data (using the catalogue t0 and t_E, which a real-time pipeline would not have),
remote fetch by event id. Install the extras (`pip install -e ".[roman]"`) and see
`examples/01_classify_roman_event.ipynb`.

```python
import json
import binml
from binml import gulls
seasons, epoch_map, meta = gulls.load_tables()                     # one-time 150 MB download
clf = binml.Classifier(weights="gapaware")        # binml/weights/binml-gapaware.pt, the recommended gap-aware checkpoint
threshold = binml.GAPAWARE_THRESHOLD              # 0.956; from validation/gulls/gapped_threshold_fspl5s_seasons_g08_seasons.json
res = gulls.classify_event(clf, 306535, seasons, epoch_map, meta)              # mode="peak"
if res["p_nonpspl_max"] is not None:
    print(res["p_nonpspl_max"] >= threshold)
```

Use the recommended gap-aware checkpoint (finite-source single lenses and the measured RMDC26 pauses in
training) for this data, not the shipped weights: RMDC26 pauses
F146 for 43-44 h per season and the shipped weights were trained on a continuous grid (README, "known
limitations"). `mode="peak"` (default) scores the season containing the peak if it is a high-cadence
season; a low-cadence peak season is outside the model's support and is reported, not scored.
`mode="adjacent"` scores the high-cadence season(s) bordering a peak that falls between seasons and
returns their maximum anomaly probability, the combiner measured in `paper/REVISION.md` §1½ row 9
(seasons holding only a wing of the event need their own operating point);
`mode="all"` scores every high-cadence season (not validated as a combiner). The population-level
numbers are regenerated by the scripts under `validation/gulls/`.

## Command line

```bash
binml classify lc.csv --m-base 22.1        # CSV/whitespace: time, mag[, ignored third column]
# --m-base is required: the classes are defined relative to the quiescent baseline. Pass
# --estimate-baseline only for a quick look; the fallback estimator is biased for every source type.
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

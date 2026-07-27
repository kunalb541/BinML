# Model Architecture (v5)

BinML v5 classifies Roman multi-band light curves into six classes — **Flat, PSPL, NonPSPL
(binary/planetary), PeriodicVar, LongPeriodVar, Eruptive**. The network is a **convolutional
stem feeding a small transformer encoder**, implemented in
[`pipeline/sim_v5/model_v5.py`](../pipeline/sim_v5/model_v5.py). It has **505,479 parameters**.

> The earlier 3-class model (`binml/model.py`) was a causal CNN-GRU. v5 replaced it with a
> multi-band conv-stem + transformer to handle three bands of very different cadence and to
> classify variable-star contaminants alongside microlensing.

---

## 1. Input: three bands, five channels per token

Each event is three separately-sampled light curves:

| band | cadence | epochs | binned tokens |
|---|---|---|---|
| F146 | 15 min | 6912 | 864 → 108 (stride 8) |
| F087 | 6 h | 288 | 96 → 24 (stride 4) |
| F213 | 6 h | 288 | 96 → 24 (stride 4) |

Each band is binned into fixed-length token slots (`cache.py`). Every token carries **five
channels**: `mean, min, max, frac, mask` —

- **mean / min / max** — the baseline-relative magnitude deviation in the bin. Keeping **min and
  max separately** is essential: a caustic crossing is a single narrow spike, and averaging it
  into the bin mean would erase it. Min/max preserve the extrema exactly (0.0 mmag loss vs
  ~2910 mmag for naive striding).
- **frac** — fraction of the bin that was actually observed.
- **mask** — whether the token has any data (bands drop out under extinction).

## 2. Conv stem with non-learned min/max carry lanes

The stem (`ConvStem`) downsamples each band to its token count. The subtlety: a learned
strided convolution would *average away* the very spikes we binned min/max to preserve. So the
stem runs the learned conv **in parallel with two non-learned carry lanes** that pool min and
max through the same downsampling —

```
mx = F.max_pool1d(mx, 2)          # max lane: pooled maximum
mn = -F.max_pool1d(-mn, 2)        # min lane: pooled minimum
```

— and concatenates them back in, so the caustic extremum survives to the transformer regardless
of what the learned filters do. Downsampling is by powers of two so the pooled lanes and the
strided conv stay length-aligned.

## 3. Transformer encoder over 156 tokens

The three bands' tokens are concatenated into one sequence — **108 + 24 + 24 = 156 tokens** —
each projected to `d_model = 96`, plus a learned positional embedding and a per-band embedding.
The encoder is **4 layers, 4 heads**, using PyTorch fused scaled-dot-product attention (SDPA).

**Band dropout is handled by masking, not by architecture.** Each band carries an explicit
presence flag; when a band is absent (extinction, non-detection) its tokens are masked out of
attention. A NaN-guard ensures an all-absent row can't produce a degenerate all-masked
attention.

## 4. Head

Masked mean-pooling over present tokens → a linear **6-way** classifier. A hierarchical head
(microlensing-vs-not, then PSPL-vs-NonPSPL) was implemented and tested but consistently scored
slightly worse than the flat 6-way head, so the flat head ships.

## 5. Why this shape

- **Small (505k params) on purpose.** The task is not data-starved (millions of simulated
  events) but is *inference-heavy* at survey scale. A compact model that reads morphology beats
  a large one that memorises.
- **The conv stem does the compute saving** — attention runs over 156 tokens, not 7488 raw
  epochs, while min/max pooling loses none of the caustic information that separates NonPSPL from
  PSPL.
- **Multi-band by construction** — colour is a discriminator (microlensing is achromatic;
  variables are not), and the per-band presence masking makes "works from F146 alone under heavy
  extinction" a property of the model, not an accident.

See [`docs/pipeline_v5.md`](pipeline_v5.md) for how the model is trained and evaluated, and
[`docs/data_format.md`](data_format.md) for the exact token/channel layout it consumes.

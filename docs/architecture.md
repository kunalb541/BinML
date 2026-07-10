# Model Architecture

BinML classifies gravitational microlensing light curves into three classes:

- **Flat** — no event (constant baseline).
- **PSPL** — single-lens (point-source, point-lens) microlensing. This is the
  "is this microlensing?" detection proxy.
- **Binary** — a binary or planetary lens. This is the distinct anomalous class;
  a binary is **not** a PSPL.

The classifier is a **strictly causal CNN-GRU** network designed for Nancy Grace
Roman Space Telescope cadence. It is implemented in `pipeline/model.py` (research
pipeline) and `binml/model.py` (the installable inference package, with the same
architecture and bundled weights).

This document describes the network end to end: the two-channel input
representation, sequence compaction and the `lengths` mechanism, the causal
depthwise-separable convolution stack, the unidirectional GRU, masked attention
pooling, and the hierarchical classification head. It also explains **why** the
network is causal and **how** padding is masked at every stage so that the
number of valid observations never leaks into the prediction.

---

## 1. Input representation: two channels

Each light curve is presented to the network as a sequence of observations with
exactly **two input channels** — nothing else (no raw fluxes, no absolute times,
no error bars fed as a third channel):

1. **Magnification `A`.**
   Computed from photometry as

   ```
   A = 10 ** (0.4 * (m_base - mag))
   ```

   where `m_base` is the event baseline magnitude and `mag` is the observed
   magnitude at that epoch. At baseline (no magnification) `A = 1.0`. This makes
   the input dimensionless and baseline-referenced: the network sees how far each
   point deviates from the quiescent flux level, independent of the source's
   apparent brightness.

2. **`delta_t`.**
   Days since the previous *valid* observation. This channel encodes the temporal
   spacing of the samples directly, so the network can reason about cadence and
   gaps without needing absolute timestamps. Because only the *elapsed time since
   the last valid point* is provided, the representation is shift-invariant in
   absolute time and depends only on the local sampling geometry.

Presenting magnification (what happened) alongside `delta_t` (when it happened
relative to the previous point) is sufficient for the network to characterize
both the smooth PSPL bump and the short, sharp caustic anomalies that
distinguish binaries.

---

## 2. Compaction and the `lengths` tensor

Roman light curves are sampled on a fixed grid of **6912 points at 15-minute
cadence over a 72-day season**, but any individual light curve has gaps (weather
gaps in real data, observing-window structure in simulation). Rather than carry
masked-out positions scattered throughout the sequence, BinML **compacts** each
light curve:

- All valid observations are moved to a **contiguous prefix** `[0, length)` of
  the sequence.
- A per-sample integer **`lengths`** tensor records how many positions are valid.
- Positions `[length, max_seq_len)` are **padding** and carry no information.

This layout is a hard assumption throughout the model: `lengths` always denotes a
*contiguous valid prefix*, never scattered valid positions. The data-loading
pipeline in `train.py` guarantees this compaction before batches ever reach the
network.

Two consequences follow:

- **Efficiency.** The convolution and attention operate over a dense prefix
  rather than a sparse full-length grid.
- **Masking is simple and exact.** Because valid data is always a prefix, a
  single `lengths` value fully specifies which positions are real. Every
  length-dependent operation in the network derives its mask from `lengths`
  (see §7).

Inputs are additionally **normalized by dataset statistics** before entering the
network. Critically, these normalization statistics are fit on the **training
split only**, so no information from validation or test data leaks into the
model's input scaling (see `docs/leakage_audit.md`).

---

## 3. Linear projection

The two-channel input is first mapped into the model's hidden dimension by a
learned linear projection:

```
(batch, seq, 2)  ->  (batch, seq, d_model)
```

With the shipped model's `d_model = 64`, each timestep's `(A, delta_t)` pair becomes a
64-dimensional feature vector that the downstream convolution and recurrent
layers refine.

---

## 4. Causal depthwise-separable convolution stack

The projected sequence passes through a stack of **causal, depthwise-separable
1-D convolution** blocks (`CausalConv1d` / `DepthwiseSeparableConv1d` in
`model.py`). These extract local temporal features — the shape of the rising
wing, the sharpness of a caustic crossing — before the recurrent layer
integrates them over the full sequence.

Two design choices define this stack:

### 4.1 Causal (left-padding only)

Each convolution is **causal**: it is padded on the **left only**, so the
receptive field for the output at position `t` covers positions `<= t` and never
positions `> t`. There is **no future leakage** — the feature computed at time
`t` depends solely on observations up to and including `t`.

This is what makes the entire network usable on **truncated prefixes**: feeding
the network the first `k` valid observations produces exactly the features it
would have produced for those same `k` observations in the full-length curve.
That property is what enables the probability-evolution and early-detection
analyses (see §6).

### 4.2 Depthwise-separable

Each convolution is factored into a **depthwise** step (one filter per channel,
capturing per-feature temporal structure) followed by a **pointwise** `1x1`
convolution (mixing across channels). This factorization delivers the modeling
capacity of a full convolution at a fraction of the parameters and compute,
which keeps the network small (§8).

Residual connections and normalization are applied around the blocks for
trainability; the residual paths are arranged so they do not introduce any
acausal dependency (features added back in were themselves computed causally).

---

## 5. Unidirectional GRU

The convolutional features feed a **unidirectional (forward-only) GRU** with
`n_layers` recurrent layers (`n_layers = 4` in the shipped model). The GRU integrates the
local convolutional features into a running temporal representation, carrying
information forward along the sequence.

The GRU is **unidirectional by design**. A bidirectional GRU would let the
representation at time `t` depend on observations *after* `t`, which would:

1. break the causal guarantee (prefix predictions would no longer match), and
2. implicitly reveal the sequence length, since the backward pass starts from the
   last valid position.

Keeping the GRU forward-only preserves the property that the state at time `t`
summarizes only observations `<= t`.

The GRU runs strictly over each sample's **valid prefix**; padding positions are
never fed into or read out of the recurrence as real state (see §7).

---

## 6. Why causality matters: valid prefix predictions

Because every stage up to and including the GRU is causal, the network's output
after consuming the first `k` valid observations is a legitimate prediction from
*just those `k` observations*. This is not an approximation — it is exact, by
construction.

This unlocks two capabilities that a non-causal model could not provide honestly:

- **Probability evolution.** `clf.predict_evolution(...)` re-runs the classifier
  over growing prefixes of a light curve and reports how the class probabilities
  evolve as more of the event is observed. Each point on that curve is a
  well-defined prediction from the data available up to that epoch.
- **Early detection.** The same property lets BinML answer "how much of the event
  did the network need to see before it flagged the anomaly?" without any leakage
  of later data into earlier predictions.

A bidirectional or otherwise acausal architecture would contaminate every prefix
prediction with information from the future, making both analyses meaningless.

---

## 7. Masked attention pooling

After the GRU, the variable-length sequence of hidden states must be reduced to a
single fixed-size vector for classification. BinML uses **multi-head attention
pooling** (`FlashAttentionPooling` in `model.py`, default
`num_attention_heads = 4`) rather than simple mean or last-state pooling.

Attention pooling learns *which* timesteps matter — for a binary, the pooled
representation can concentrate on the brief caustic-crossing epochs that carry the
anomaly signal, rather than diluting them across the whole 72-day season.

### Masking padded positions

Pooling is where length leakage is most dangerous, and it is handled explicitly:
**padded positions (indices `>= length`) are masked to `-inf` in the attention
logits before the softmax.** After softmax, those positions receive exactly zero
weight, so padding contributes nothing to the pooled vector.

The mask is derived from the `lengths` tensor (the contiguous-prefix guarantee
from §2 makes this a simple threshold). The additive mask value is chosen per
dtype to avoid numerical overflow under mixed precision (e.g. a large negative
constant for fp32/bf16, a smaller one for fp16). The net effect is the same in
every precision: **padding is invisible to the pooling stage.**

Together with the causal conv stack (§4) and the forward-only GRU (§5), this means
the number of padded positions — i.e. the sequence length — **never enters the
computation**. Two light curves with the same valid prefix but different amounts
of trailing padding produce identical outputs. There is **no length leakage**
anywhere in the network.

---

## 8. Hierarchical classification head

BinML does not classify Flat / PSPL / Binary with a single flat 3-way softmax.
Instead it uses a **hierarchical head** that mirrors the physical decision
structure of the problem, plus an auxiliary 3-class head for gradient stability.

### 8.1 Stage 1 — deviation vs. flat

A single **deviation logit** answers: *is there any deviation from baseline at
all?* Applying a sigmoid gives

```
P(deviation) = sigmoid(stage1_logit)
P(Flat)      = 1 - P(deviation)
```

This is the "is anything happening" gate.

### 8.2 Stage 2 — binary vs. PSPL, given a deviation

A second **binary-vs-PSPL logit** is evaluated *conditional on there being a
deviation*: *given that this is an event, is it a plain single lens or an
anomalous binary?*

```
P(PSPL | deviation)   = sigmoid(stage2_logit)
P(Binary | deviation) = 1 - P(PSPL | deviation)
```

### 8.3 Combining the stages

The final three-class probabilities are formed by the chain rule, so they are
guaranteed to be non-negative and sum to one:

```
P(Flat)   = 1 - P(deviation)
P(PSPL)   = P(deviation) * P(PSPL   | deviation)
P(Binary) = P(deviation) * P(Binary | deviation)
```

This factorization matches how the classification is actually *used*:

- **`is_microlensing` = `P(PSPL) + P(Binary)`** — the Stage-1 detection question,
  "is this microlensing at all?"
- **`is_anomalous` = `P(Binary)`** — the Stage-2 characterization question, "is
  the lens a binary/planet?"

Separating detection (Stage 1) from characterization (Stage 2) is what lets the
network commit to "an event is present" even when it is uncertain whether the
event is single or binary — the harder distinction — instead of forcing a single
softmax to trade the two decisions off against each other.

### 8.4 Losses

The two stages are trained with **separate binary cross-entropy losses** on
`stage1_logit` and `stage2_logit` respectively. In addition, an **auxiliary
direct 3-class head** produces standard 3-way logits trained with cross-entropy.
This auxiliary head stabilizes gradient flow into the shared trunk and prevents
the hierarchical head from collapsing early in training; it is a training aid
layered on top of the hierarchical probabilities, which remain the model's primary
output.

---

## 9. Default hyperparameters

The shipped configuration is:

| Hyperparameter          | Shipped value | Role                                            |
|-------------------------|---------------|-------------------------------------------------|
| `d_model`               | **64**        | Hidden dimension of projection, conv, GRU        |
| `n_layers`              | **4**         | Number of GRU layers                             |
| `num_attention_heads`   | 4             | Heads in the attention-pooling stage             |
| Input channels          | 2             | Magnification `A`, `delta_t`                      |
| Output classes          | 3             | Flat / PSPL / Binary                             |
| Total parameters        | **130,821 (~131K)** | Whole network                             |

> Note: the `ModelConfig` dataclass default is a smaller smoke-test size
> (`d_model = 16`, `n_layers = 2`). The bundled weights — and all training in
> `pipeline/` — use the 64 / 4 configuration above, and a loaded `Classifier` always
> takes its architecture from the checkpoint's stored config.

At roughly **130K parameters** the model is deliberately compact — small enough
to bundle its weights inside the `binml` package and run inference with only
`torch` and `numpy` as dependencies, while retaining the depthwise-separable
convolutions, recurrent integration, and attention pooling needed to resolve the
short-duration binary anomaly.

---

## 10. Summary of the forward pass

```
(A, delta_t)                       # 2 channels, compacted to a contiguous prefix
      |  normalize by train-split statistics
      v
Linear projection                  # (b, seq, 2) -> (b, seq, d_model=64)
      |
Causal depthwise-separable         # left-padding only: output(t) depends on <= t
Conv1d stack                       # no future leakage
      |
Unidirectional GRU (n_layers=4)    # forward-only temporal integration over the valid prefix
      |
Masked multi-head attention        # padded positions -> -inf before softmax -> zero weight
pooling                            # length never leaks
      |
Hierarchical head:
   Stage 1: P(deviation)           # Flat vs. event         (BCE)
   Stage 2: P(PSPL | deviation)    # PSPL vs. Binary        (BCE)
   + auxiliary 3-class head        # gradient stability     (CE)
      |
      v
P(Flat), P(PSPL), P(Binary)        # chain-rule combination, sums to 1
```

Every stage is causal and every length-dependent operation is masked from
`lengths`, so the network produces exact, leakage-free predictions on both full
light curves and truncated prefixes.

---

*See also:* `docs/leakage_audit.md` for the end-to-end audit of train/test
separation and length-masking, and `pipeline/model.py` / `binml/model.py` for the
reference implementation.

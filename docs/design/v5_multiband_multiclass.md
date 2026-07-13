# BinML v5 — Generic N-Band, Extensible Multi-Class — Design

> **Status: design proposal (not yet implemented).** Drafted 2026-07-10 as the plan for
> evolving BinML from a single-band 3-class classifier into a configurable N-band,
> extensible multi-class classifier. Reviewed against the v4 code
> (`binml/model.py`, `binml/preprocess.py`, `pipeline/simulate.py`).

## Design rationale (why these choices)

Four independent architectures were explored and converged on two load-bearing decisions,
which is strong evidence they are right:

1. **One event = a single time-sorted, interleaved, merged stream of per-band
   observations** — no shared time grid. This is the only clean way to handle asynchronous,
   missing, variable-cadence bands (e.g. LSST `ugrizy`). An absent band simply contributes
   zero tokens; no imputation, no alignment.
2. **Warm-start from v4 by expanding `input_proj` and zero-initializing every new
   parameter**, so v5 is *numerically identical to v4* on single-band input. The existing
   trained model is not thrown away — it is the initialization.

On top of that consensus, the design grafts the strongest idea from each exploration:

- the explicit **shared-Â(t) + per-band-blend ĝ_b factorization head**, whose reconstruction
  residual is simultaneously a physics regularizer, the color discriminator, *and* the
  detectability statistic for chromatic contaminants (the single most valuable idea);
- **reusing the v4 backbone verbatim** and treating single-band parity as a hard regression
  gate;
- a reserved **"unknown-band" embedding**, **band-dropout** augmentation, and conditioning
  recall on whether the anomaly window was actually sampled;
- an **open-set prototype layer** for adding a class with a few-shot fit, and serializing the
  band/class **registries** into both the data-file attrs and the checkpoint.

A per-band *weight-shared trunk with union-axis re-fusion* was considered and rejected: it is
heavier and warm-starts less cleanly than a single interleaved stream through the v4 core,
which already lets the recurrent and attention layers read cross-band structure natively.

---

## Overview

BinML v5 generalizes the v4 single-band, 3-class causal classifier to an arbitrary number of
asynchronously-sampled photometric bands and an extensible, registry-defined class taxonomy
spanning microlensing science classes (Flat / PSPL / Planetary / Stellar-binary) and
non-microlensing contaminants (variable stars / CVs / other). The design rests on one
physical fact: the lens magnification `A(t)` is **achromatic**, so every band is a
blend-diluted copy of one shared `A(t)`,

```
a_b(t) = (A(t) + g_b) / (1 + g_b),     g_b = fb_b / fs_b   (blend ratio in band b)
```

v5 makes the model **represent that factorization explicitly** and uses the *failure* of a
shared-`A` fit as the discriminator against chromatic contaminants. It stays strictly causal
(valid-prefix predictions) and keeps detectability-conditioned evaluation, now generalized so
every class is scored against a detectability statistic. It degrades exactly to v4 when
`n_bands = 1`.

## Data representation & format v5

**One event = one merged, time-sorted stream of per-band observations.** Missing bands,
ragged counts, and independent per-band cadence are the default case.

**On-disk (HDF5, CSR/ragged — not padded).** Variable observation counts make dense
`[E, T_max, ·]` wasteful, so store flat arrays indexed by `event_ptr [E+1]`:

```
obs_t, obs_mag, obs_magerr, obs_band   each length Σ N_obs   # raw, re-derivable
m_base       [E, n_bands] f4                                  # per-band baseline (v4 scalar = n_bands=1)
label        [E] i4        class_path [E] S64                 # leaf id + self-describing path
params/      t0,tE,u0,q,s,rho,alpha, fs_b[],fb_b[],g_b[], source_SED, period,...
detect/      anomaly_dchi2 [E]        chromatic_dchi2 [E]     # two detectability handles
file.attrs:  band_registry (JSON), class_registry (JSON), norm_stats, season_days
```

Magnitudes are stored **raw**; normalization happens in the collate transform so features
stay re-derivable and `norm_stats` transfer from v4. The registries live in file attrs **and**
the checkpoint, so data, weights, and code always agree on band/class meaning.

**At load (collate → compacted padded prefix, v4-style).** Each event is compacted so valid
tokens occupy `[0, length)`, producing per batch:

| tensor | shape | meaning |
|---|---|---|
| `A` | `[B,T]` | per-band magnification `a_b = 10^(0.4(m_base_b − mag))` — the exact v4 channel, per band |
| `dt_global` | `[B,T]` | days since previous token, any band (v4's `delta_t`) |
| `dt_band` | `[B,T]` | days since previous token in the **same** band (per-band cadence) |
| `err` | `[B,T]` | photometric error in flux units (heteroscedastic; new — v4 ignored it) |
| `band_id` | `[B,T]` int8 | index into `band_registry` (0 reserved = pad) |
| `lengths` | `[B]` | valid prefix count |

Color is **not** a stored channel — asynchronous sampling means there is rarely a
simultaneous two-band measurement. It is reconstructed inside the model from adjacent
differing-`band_id` tokens and, decisively, from the factorization head below.

## Model architecture

The v4 causal core is reused verbatim; only the front-end tokenizer and two thin heads are
added. `d_model = 64`, `n_layers = 4` (shipped config).

**1. Tokenizer.** Expand `input_proj` from `Linear(2, d)` to `Linear(4, d)` over
`[A, dt_global, dt_band, err]`, then add a band embedding:

```python
tok = input_proj(stack([A, dt_global, dt_band, err], -1))     # [B,T,d]
    + band_emb(band_id)          # Embedding(n_bands+1, d), slot 0 = "unknown band"
```

`band_emb` is zero-init; the `dt_band` and `err` columns of `input_proj` are zero-init;
columns 0–1 copy v4. So at step 0 on single-band data, `tok` equals v4's projection exactly.
The reserved unknown-band slot makes the model genuinely N-agnostic (a filter never seen in
training still embeds).

**2. Causal core (v4 verbatim).** `tok.transpose(1,2)` → `feature_extractor` (depthwise-
separable causal Conv1d, RF≈13, left-pad only, residual added *before* conv per the v4
causality fix) → unidirectional `gru` → `layer_norm`. Output `H [B,T,d]`. Strictly causal:
prefix hidden states are valid streaming predictions.

**3. Causal cross-band fusion (new, 1 layer).** One masked multi-head self-attention layer
where token `i` attends only to `j ≤ i` (additive `−inf` causal mask, reusing the pooling
mask machinery). Because earlier tokens of *other* bands are visible, a token in band `g`
reads the last band-`r` measurement across a gap — the direct "color across time" channel that
sparse ground cadence needs. Its `out_proj` is **zero-init**, so the layer is an identity
residual at step 0 (parity preserved). `H' [B,T,d]`.

**4. Physics factorization head (new — the core).** From `H'`, predict per token a shared
log-magnification `logÂ(t_i)` (`Linear(d,1)`) and per (event, band) a blend `ĝ_b`
(`[B, n_bands]`, from a small head over the pooled state). Reconstruct
`â_b(t_i) = (Â(t_i) + ĝ_b)/(1 + ĝ_b)` and add a `1/σ²`-weighted reconstruction loss
`Σ_i (a_b(t_i) − â_b(t_i))² / err_i²`. This **forces** the latent to hold the physical
`A(t) + g_b` decomposition. The per-(event,band) reconstruction **residual** — how badly one
shared achromatic `A` explains all bands — is the color discriminator: low for achromatic
microlensing, high for chromatic variables/CVs. Its mean/max across bands is `r [B, k]`.

**5. Pooling + classifier input.** `FlashAttentionPooling(H', lengths)` → event embedding
`z [B,d]` (padded tail masked to `−inf`, v4 verbatim). Concatenate the chromatic residual:
`z' = [z ; r]  [B, d+k]`.

Missing bands need no special case (absent tokens); asynchronous cadence is absorbed by
`dt_*` and pooling. New params beyond v4: one `Embedding`, two `input_proj` columns, one
attention layer, two small factorization heads — a few percent over 131K.

## Extensible class taxonomy & head

The head is **registry-driven**, not hard-coded. Three cooperating pieces on `z'`:

**(a) Hierarchical router (carries v4's detectability philosophy).** `class_path` strings
define a tree; each internal node is a `Linear(d+k, 1)` sigmoid gate; `P(leaf) = Π gates`
along its path (v4's `P(PSPL) = P(dev)·P(PSPL|dev)` generalized). Default tree:

```
root         Flat | Deviation                        ← v4 head_stage1
Deviation    Microlensing | Non-ML(chromatic)        ← NEW achromaticity gate, fed by residual r
Microlensing PSPL | Lens-binary                       ← v4 head_stage2
Lens-binary  Planetary (q<~1e-2) | Stellar (q>~1e-2)  ← NEW
Non-ML       VariableStar | CV | Other                ← NEW softmax
```

The Deviation→Microlensing gate is fed primarily by the factorization residual `r`, making
achromaticity the physically-motivated top microlensing split.

**(b) Flat aux head** over all leaves (`Linear(d+k, n_leaves)`, near-zero init) for gradient
stability, exactly as v4's aux 3-class head.

**(c) Open-set prototype layer.** Each leaf `k` has a prototype `p_k ∈ R^{d+k}`; score
`s_k = cos(z', p_k)/τ`. Final leaf logit blends router path-prob and prototype score.
`max_k s_k < threshold ⇒ Other/unknown` (OOD falls out naturally).

**Adding a class = local surgery, no full retrain:** register a generator + a leaf under a
parent node, grow that one node by one output (re-init that node only), append one prototype
row, then **freeze the backbone + existing prototypes + ancestor gates** and few-shot fit the
new prototype/branch with rehearsal of old classes — matching the standing "fine-tune over
base training" preference.

## Simulator changes

`LightCurveGenerator` ABC — `.sample(rng) → (flux_model_or_A, params, class_path)`, registered
in `GENERATORS` keyed by `class_path`; the engine is generator-agnostic.

- **Achromatic core.** Lens generators (PSPL analytic; Planetary/Binary via VBBinaryLensing)
  emit one `A(t)`. Per band `F_b(t) = fs_b·A(t) + fb_b`, with `fs_b` from a sampled **source
  SED**/isochrone and an independent **per-band blend** `g_b` from a crowding model ⇒
  band-dependent amplitude.
- **Chromatic finite source.** Per-band limb-darkening coefficients give band-dependent
  finite-source/`rho` deviations near caustics.
- **Survey object** holds per-band `Cadence` and photometric noise (Roman: dense F146 + sparse
  Z087; LSST: 6 asynchronous `ugrizy` from an OpSim-like schedule with weather/season gaps).
  Sample each band's epochs independently, merge+sort into the token stream; per-band
  photon+sky+read noise ⇒ heteroscedastic `err`.
- **New contaminant generators:** `VariableStar` (RR Lyrae/Cepheid/EB Fourier templates —
  chromatic amplitude+phase, periodic), `CV/DwarfNova` (stochastic, blue-in-outburst),
  `EclipsingBinary`, `Other`. Adding one = subclass + register + registry leaf; no engine
  edits.
- **Detectability labels:** keep `anomaly_dchi2` (binary vs matched single-lens, now a
  multi-band joint fit) and add `chromatic_dchi2` (best achromatic single-source fit vs data —
  the color/periodicity detectability handle for contaminants).

## Training & migration from v4

1. **Load v4 checkpoint into identically-named modules.** `input_proj` (expand cols),
   `feature_extractor`, `gru`, `layer_norm`, `pooling`, `head_shared` load directly. Map
   `head_stage1 → root(Flat/Dev)` gate and `head_stage2 → Microlensing(PSPL/binary)` gate.
   New modules — `band_emb`, fusion `out_proj`, factorization heads, Deviation→Microlensing
   gate (large +bias so `P(ML|dev)≈1`), Planetary/Stellar gate, Non-ML softmax, prototypes,
   enlarged aux — are zero/near-zero init. **Assert single-band parity** against v4 as a
   regression gate.
2. **Stage A — bands (freeze core).** Train front-end (`band_emb`, new `input_proj` cols,
   `err`), fusion, factorization + reconstruction loss, and node heads on multi-band
   *microlensing* data. Reconstruction is self-supervised (needs no new labels); band-dropout
   augmentation hardens against missing filters.
3. **Stage B — taxonomy (unfreeze, low LR).** Add non-ML classes; train the full multi-band
   multi-class mix, class-balanced and **detectability-stratified** on hard regimes (low-`q`
   planetary, low-`chromatic_dchi2` variables).
4. **Stage C — per-new-class.** Freeze all but the new prototype/branch; few-shot with
   rehearsal.

Loss = per-node BCE/CE masked to each node's conditional subset (v4's
`compute_hierarchical_loss` pattern, NLL on product-probs) + aux CE over leaves +
`1/σ²`-weighted reconstruction + prototype contrastive. Reuse v4 `norm_stats`.

## Evaluation (detectability-conditioned, multi-class)

- **Generalized detectability conditioning.** Every class is scored against a statistic `D`:
  Planetary/Stellar recall vs `anomaly_dchi2` (and `q`); Variable/CV recall vs
  `chromatic_dchi2`. A contaminant with no detectable chromatic/periodic signal in-window is
  legitimately PSPL-like — condition on `D`, exactly as v4 conditions binaries on anomaly.
  **Additionally condition on whether any band actually sampled the anomaly window**
  (unsampled anomaly = undetectable, excluded).
- **Single-band parity** must reproduce v4 metrics on v4 data — a hard regression gate.
- **Band-ablation:** drop bands, measure graceful degradation; the single-vs-multi-band recall
  gap quantifies the color contribution.
- **Causal / early detection:** per-node probability evolution along the prefix (when does
  `P(ML)` resolve vs `P(planetary)`), plus a new "epoch of first color-based rejection" for
  contaminants.
- **Open-set AUROC** for Other/unknown; per-leaf + per-node confusion matrices; metrics sliced
  by cadence/depth and **keyed to the registry** so they extend automatically when a class is
  added.
- **Physics sanity check:** report the reconstruction-residual distribution per class (should
  separate achromatic from chromatic).

## Risks & open questions

- **Merged-stream length.** LSST `ugrizy` over multiple seasons can exceed v4's 6912; needs a
  max-length/subsampling policy and receptive-field re-tuning. The `dt_band` signal may be weak
  for rarely-sampled filters.
- **Reconstruction head could dominate or fight classification.** Needs loss-weight
  scheduling; risk that `Â(t)` overfits noise. Predicting a constant `ĝ_b` causally means early
  estimates are noisy (acceptable, but monitor).
- **Prototype few-shot may underperform** a small fine-tune for classes far from the existing
  manifold; keep full fine-tune as a fallback.
- **Sim-to-real color realism** (SED priors, blend distributions, limb-darkening, OpSim
  fidelity) is the dominant generalization risk — more so than architecture.
- **Causal cross-band attention cost** scales with sequence length; may need windowing.
- **Registry/label schema evolution** across checkpoints needs versioning to avoid silent id
  drift.

## Phased implementation plan

| phase | deliverable |
|---|---|
| **P0** | Format + loader: CSR HDF5, registries, collate→compacted prefix, `norm_stats`; round-trip a v4 file as `n_bands=1` |
| **P1** | Warm-start parity: expand `input_proj` + `band_emb`; load v4; **assert bit-parity** metrics on v4 data (regression gate) |
| **P2** | Multi-band sim: SED/blend/limb-darkening, per-band cadence+noise, merged stream; Roman 2-band first |
| **P3** | Fusion + factorization head: causal cross-band attention + shared-`Â`/blend reconstruction; Stage-A training; verify residual separates blends |
| **P4** | Taxonomy: registry router + aux + prototypes; map v4 heads; Stage-B multi-class with contaminant generators; detectability-stratified |
| **P5** | Extensibility + open-set: add-a-class workflow (freeze+prototype), OOD thresholding, band-dropout robustness |
| **P6** | Eval + packaging: generalized detectability suite, band-ablation, causal evolution; bundle weights+registries; ship LSST `ugrizy` config |

**Grounding files:** `binml/model.py` (module names for warm-start, `FlashAttentionPooling`,
`load_checkpoint` hook), `binml/preprocess.py` (2-channel A/delta_t compaction to generalize;
note `mag_err` is currently unused — v5's `err` channel is genuinely new),
`pipeline/simulate.py` (VBBinaryLensing engine + HDF5 schema + generator seams).

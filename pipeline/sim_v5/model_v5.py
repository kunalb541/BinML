"""
BinML v5 — multi-band, multi-class classifier.

Design constraints that drove this architecture:

1. **Attention must not see 6912 points.** Self-attention is quadratic, and the raw F146
   season is 6912 epochs. A strided convolutional stem reduces 864 pre-binned bins to ~108
   tokens before any attention, which is where essentially all of the compute saving comes
   from -- not from the (tiny) parameter count.

2. **The bands have different lengths and are often absent.** F146 always exists; F087 is
   missing entirely in ~38% of events and F213 in ~1%, because a reddened or faint source
   falls below those bands' limits. Missing is a physical fact, not padding, so each band
   carries an explicit presence flag and its tokens are masked out of attention when absent.
   The model must degrade gracefully to F146-only rather than seeing garbage.

3. **Amplitude lives in the extrema, not the mean.** Inputs are (mean, min, max) per bin --
   averaging a caustic crossing away loses up to 2.9 magnitudes of the very signal that
   defines the NonPSPL class.

4. **Inputs are baseline-relative.** The cache stores deviation from the per-event baseline,
   so the network never sees absolute magnitude and cannot key off source brightness. That is
   deliberate: brightness correlates with extinction and hence weakly with class, and we do
   not want that shortcut carrying the prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ModelConfigV5", "BinMLv5"]

# (bins, stem stride product) per band; must match cache.BIN_FACTORS output lengths.
BAND_BINS: Dict[str, int] = {"F146": 864, "F087": 96, "F213": 96}
IN_CH = 5   # mean, min, max, observed-fraction, observed-mask


@dataclass
class ModelConfigV5:
    d_model: int = 96
    n_layers: int = 4
    n_heads: int = 4
    dim_ff: int = 256
    dropout: float = 0.1
    n_classes: int = 6
    stem_channels: int = 64
    f146_stride: int = 8        # 864 -> 108 tokens
    colour_stride: int = 4      # 96  -> 24 tokens each
    bands: List[str] = field(default_factory=lambda: ["F146", "F087", "F213"])

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model {self.d_model} not divisible by n_heads {self.n_heads}")
        # ConvStem realises `stride` as a stack of stride-2 convolutions, so the ACTUAL
        # downsampling is 2**ceil(log2(stride)) while n_tokens computes BAND_BINS // stride.
        # For a non-power-of-two stride those disagree, and `tokens + self.pos` then either
        # raises or -- worse, if two bands' errors cancel in the total -- silently misaligns
        # the positional embedding across a band boundary and trains on scrambled positions.
        for name in ("f146_stride", "colour_stride"):
            v = getattr(self, name)
            if v < 1 or (v & (v - 1)):
                raise ValueError(f"{name} must be a power of two, got {v}")
        if self.stem_channels < 4:
            raise ValueError("stem_channels must exceed the 2 reserved min/max carry lanes")


CH_MIN, CH_MAX = 1, 2       # channel order in the cache: mean, min, max, frac, mask


class ConvStem(nn.Module):
    """Strided conv stack with EXPLICIT extrema carry lanes.

    (B, IN_CH, L) -> (B, d_model, L/stride).

    The subtle point that makes this class necessary: the cache's min/max binning is
    lossless because it is literally a max, and a 2.9-magnitude caustic amplitude survives
    the 8x reduction exactly. But a convolution is NOT a max -- it is a learned weighted
    average followed by GELU and normalisation. A stride-8 stem token spans 8 bins = 64
    epochs = 16 hours, while a caustic crossing occupies only 1-2 bins, so an averaging
    filter would present a 2.9 mag spike to the encoder as roughly 0.36 mag. That is still
    above the 20 mmag detection floor, but it is the same order as ordinary PSPL curvature
    over the same window -- i.e. exactly the discrimination the NonPSPL label encodes.

    The network *could* learn a max-like filter, but nothing forces it to, and normalising
    across a mostly-flat baseline actively discourages high-variance filters. So instead of
    hoping, two channels are reserved as a NON-LEARNED min/max carry lane: the running max
    is max-pooled and the running min is min-pooled at every stage, alongside the learned
    path. Extrema preservation then holds by construction, at the cost of two channels.
    """

    def __init__(self, cfg: ModelConfigV5, stride: int):
        super().__init__()
        c = cfg.stem_channels
        self.stages = nn.ModuleList()
        in_c, remaining = IN_CH, stride
        # Halve the length repeatedly; a stack of stride-2 convs has a much larger receptive
        # field per parameter than one big-stride conv, which matters for narrow features.
        while remaining > 1:
            self.stages.append(nn.Sequential(
                nn.Conv1d(in_c, c - 2, kernel_size=5, stride=2, padding=2),
                nn.GELU(), nn.BatchNorm1d(c - 2)))
            in_c, remaining = c, remaining // 2
        self.out = nn.Conv1d(in_c, cfg.d_model, kernel_size=3, padding=1)

    def forward(self, x):
        mn = x[:, CH_MIN:CH_MIN + 1]
        mx = x[:, CH_MAX:CH_MAX + 1]
        for stage in self.stages:
            y = stage(x)
            # Conv1d(k=5, s=2, p=2) and max_pool1d(2) both map L -> floor(L/2) for even L,
            # which every band length here satisfies (864, 96 and their halves).
            mx = F.max_pool1d(mx, 2)
            mn = -F.max_pool1d(-mn, 2)
            x = torch.cat([y, mn, mx], dim=1)
        return self.out(x)


class EncoderBlock(nn.Module):
    """Pre-norm transformer block built on fused scaled-dot-product attention.

    This replaces ``nn.TransformerEncoderLayer``, which measured 140 ms per forward at
    batch 256 -- 97% of the whole model -- against 4.6 ms for the entire convolutional stem.
    The cost was never FLOPs: at 156 tokens and d_model 96 the attention is trivial. It was
    kernel-launch overhead, and ``norm_first=True`` additionally disables PyTorch's fused
    nested-tensor path (hence the enable_nested_tensor warning it emits).

    So this block minimises the NUMBER of operations rather than their size: one fused QKV
    projection instead of three, and ``F.scaled_dot_product_attention`` instead of an
    unrolled attention implementation.
    """

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float):
        super().__init__()
        self.h = n_heads
        self.dk = d_model // n_heads
        self.n1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(),
                                nn.Linear(dim_ff, d_model))
        self.drop = nn.Dropout(dropout)
        self.p = dropout

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        B, L, D = x.shape
        y = self.n1(x)
        qkv = self.qkv(y).view(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # No dropout_p here on purpose. The attention score tensor is by far the largest in
        # the model (B*h*L*L = 24.9M elements at batch 256, vs 3.8M for every other
        # activation), and a non-zero dropout_p combined with a boolean mask disqualifies the
        # fused SDPA kernel, forcing the `math` decomposition that materialises both the
        # probability tensor and its softmax output for backward -- roughly doubling per-sample
        # activation memory. Regularisation is retained on the residual branches below.
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.drop(self.proj(o.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.ff(self.n2(x)))


class BinMLv5(nn.Module):
    def __init__(self, cfg: Optional[ModelConfigV5] = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfigV5()
        self.stems = nn.ModuleDict({
            b: ConvStem(cfg, cfg.f146_stride if b == "F146" else cfg.colour_stride)
            for b in cfg.bands
        })
        self.n_tokens = {b: BAND_BINS[b] // (cfg.f146_stride if b == "F146" else cfg.colour_stride)
                         for b in cfg.bands}
        total = sum(self.n_tokens.values())
        # Learned positional embedding over the concatenated token sequence.
        #
        # There is deliberately NO separate band embedding: which band a token belongs to is
        # a FIXED function of its position (tokens 0-107 are F146, 108-131 F087, 132-155
        # F213), so `pos` can already represent band identity. Adding a band embedding on top
        # is exactly redundant -- it only ever contributes a per-position constant that `pos`
        # can absorb -- and it cost an embedding lookup plus an add on every forward pass.
        self.pos = nn.Parameter(torch.zeros(1, total, cfg.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.blocks = nn.ModuleList([
            EncoderBlock(cfg.d_model, cfg.n_heads, cfg.dim_ff, cfg.dropout)
            for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.attn_pool = nn.Linear(cfg.d_model, 1)
        self.head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Dropout(cfg.dropout),
                                  nn.Linear(cfg.d_model, cfg.n_classes))

    def forward(self, feats: Dict[str, torch.Tensor],
                present: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        feats[b]   (B, L_b, IN_CH) mean, min, max, observed-fraction, observed-mask --
                   already NaN-free (see loader). Values are baseline-relative magnitudes.
        present[b] (B,)            bool, whether the band yielded any data at all
        """
        toks, keypad = [], []
        for b in self.cfg.bands:
            # feats arrives ALREADY NaN-free with the observed-mask as its 4th channel; the
            # loader does that once per event in numpy rather than the model redoing
            # isfinite + nan_to_num + cat on every band on every step of every epoch.
            t = self.stems[b](feats[b].transpose(1, 2)).transpose(1, 2)   # (B, n_tok, d_model)
            toks.append(t)
            # An absent band contributes no usable tokens: mask them out of attention rather
            # than letting the encoder attend to an all-zero block.
            keypad.append((~present[b]).unsqueeze(1).expand(-1, t.shape[1]))
        h = torch.cat(toks, dim=1) + self.pos
        pad = torch.cat(keypad, dim=1)                              # True = ignore

        # SDPA wants an additive/boolean mask broadcastable to (B, heads, L, L). A
        # key-padding mask becomes a column mask: True = attend, False = ignore.
        # Shape (B, 1, 1, L) and let SDPA broadcast. Expanding to (B, heads, L, L) cost
        # ~50 MB of mask at batch 512 for no benefit.
        attn_mask = (~pad)[:, None, None, :]
        for blk in self.blocks:
            h = blk(h, attn_mask)
        h = self.final_norm(h)

        # Masked attention pooling. Unmasked mean pooling over padded positions is a real bug
        # this project has hit before, so padded tokens are excluded explicitly here.
        # If an event ever had EVERY band absent, the whole row would be masked, softmax over
        # all -inf would give NaN, and that NaN would propagate into the loss and silently
        # poison the entire batch's gradient. F146 is always present by construction, but that
        # invariant lives in the loader, not here -- so enforce it rather than assume it.
        all_masked = pad.all(dim=1, keepdim=True)
        pad = pad & ~all_masked
        w = self.attn_pool(h).squeeze(-1).masked_fill(pad, float("-inf"))
        w = torch.softmax(w, dim=1).unsqueeze(-1)
        pooled = (h * w).sum(dim=1)
        return self.head(pooled)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------------
# Hierarchical head
# ---------------------------------------------------------------------------------
# The class taxonomy already encodes three nested questions (see classes.py):
#     L1  is there an event at all?          Flat        vs everything
#     L2  is it microlensing?                {PSPL, NonPSPL} vs the contaminants
#     L3  is it anomalous?                   PSPL        vs NonPSPL
# A flat 6-way softmax throws that structure away and makes the three compete for one
# output distribution. Measured on 443k test events, 37.6% of ALL errors are the single
# PSPL<->NonPSPL decision, and in the flat view NonPSPL is only 11.86% of events -- whereas
# among MICROLENSING events it is 28.95%, a 2.4x better-balanced problem. Splitting the
# decisions gives that hard call its own head at its own natural balance, instead of forcing
# a class weight (alpha_nonpspl) to compensate -- which is exactly what destabilised the
# first training run.
#
# The trunk is shared, so this costs ~2k extra parameters, and the class probabilities are
# recovered by multiplication rather than by hard routing, so an L1 mistake stays recoverable
# and the model remains trainable end-to-end.
I_FLAT, I_PSPL, I_NONPSPL = 0, 1, 2
CONTAMINANT_IDS = (3, 4, 5)


class HierarchicalHead(nn.Module):
    """Factorised head: P(class) = P(event) * P(kind | event) * P(sub | kind)."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.event = nn.Linear(d_model, 2)     # Flat vs Event
        self.kind = nn.Linear(d_model, 2)      # microlensing vs contaminant
        self.ml = nn.Linear(d_model, 2)        # PSPL vs NonPSPL
        self.cont = nn.Linear(d_model, 3)      # PeriodicVar / LongPeriodVar / Eruptive

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.drop(self.norm(h))
        return {"event": self.event(h), "kind": self.kind(h),
                "ml": self.ml(h), "cont": self.cont(h)}

    @staticmethod
    def to_class_logits(parts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine the factored heads into 6 class log-probabilities.

        Done in LOG space: the products become sums, which keeps the small probabilities of
        the rare classes from underflowing and makes the result a proper distribution that
        cross-entropy can consume directly.
        """
        lp_event = F.log_softmax(parts["event"], -1)   # [:,0] flat   [:,1] event
        lp_kind = F.log_softmax(parts["kind"], -1)     # [:,0] ml     [:,1] contaminant
        lp_ml = F.log_softmax(parts["ml"], -1)         # [:,0] PSPL   [:,1] NonPSPL
        lp_cont = F.log_softmax(parts["cont"], -1)     # periodic / long-period / eruptive
        ev = lp_event[:, 1:2]
        out = [lp_event[:, 0:1],                                   # Flat
               ev + lp_kind[:, 0:1] + lp_ml[:, 0:1],               # PSPL
               ev + lp_kind[:, 0:1] + lp_ml[:, 1:2],               # NonPSPL
               ev + lp_kind[:, 1:2] + lp_cont[:, 0:1],             # PeriodicVar
               ev + lp_kind[:, 1:2] + lp_cont[:, 1:2],             # LongPeriodVar
               ev + lp_kind[:, 1:2] + lp_cont[:, 2:3]]             # Eruptive
        return torch.cat(out, dim=1)

    @staticmethod
    def targets(y: torch.Tensor):
        """Per-level targets and masks. A level is only supervised where it applies.

        Asking the PSPL-vs-NonPSPL head about a Cepheid is meaningless, and back-propagating
        that would teach the head to model something it will never be asked about at
        inference. Hence the masks.
        """
        is_event = (y != I_FLAT).long()
        is_cont = torch.isin(y, torch.tensor(CONTAMINANT_IDS, device=y.device)).long()
        m_kind = y != I_FLAT
        m_ml = (y == I_PSPL) | (y == I_NONPSPL)
        m_cont = is_cont.bool()
        return {"event": (is_event, None),
                "kind": (is_cont, m_kind),                 # 0 = microlensing, 1 = contaminant
                "ml": ((y == I_NONPSPL).long(), m_ml),     # 0 = PSPL, 1 = NonPSPL
                "cont": ((y - 3).clamp(min=0), m_cont)}


def hierarchical_loss(parts: Dict[str, torch.Tensor], y: torch.Tensor,
                      w: torch.Tensor, level_weights: Optional[Dict[str, float]] = None
                      ) -> torch.Tensor:
    """Sum of per-level weighted cross-entropies, each over the rows where it applies."""
    lw = level_weights or {"event": 1.0, "kind": 1.0, "ml": 2.0, "cont": 1.0}
    tg = HierarchicalHead.targets(y)
    total = y.new_zeros((), dtype=torch.float32)
    for name, (t, mask) in tg.items():
        logits = parts[name]
        if mask is not None:
            if mask.sum() == 0:
                continue
            logits, t, ww = logits[mask], t[mask], w[mask]
        else:
            ww = w
        ce = F.cross_entropy(logits, t, reduction="none")
        total = total + lw[name] * (ce * ww).sum() / ww.sum().clamp(min=1e-8)
    return total

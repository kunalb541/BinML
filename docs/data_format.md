# BinML Dataset Format

BinML has **two on-disk representations**: a **raw shard** (full-resolution multi-band light curves,
written by `run_shard.py` via `writer.py`) and a **compact cache shard** (binned tokens the model
consumes, written by `cache.py`). Rows are row-aligned across every dataset within a file:
index `i` is the same event everywhere.

Format version string: `5.0.0` (in each file's `attrs`).

---

## 1. Raw shard (HDF5, ~312 MB)

Full-resolution, one group per band (`F146`, `F087`, `F213`):

| dataset | shape | meaning |
|---|---|---|
| `mag/<band>` | (N, epochs) | observed magnitude; `NaN` = epoch failed detectability |
| `mag_err/<band>` | (N, epochs) | reported error (from the *measured* flux — no label leakage) |
| `time/<band>` | (epochs,) | shared epoch grid, stored once |
| `f_s/<band>` | (N,) | per-band source flux fraction (blending) |
| `n_kept/<band>` | (N,) | usable epochs in the band |

`epochs`: F146 = 6912 (15 min), F087/F213 = 288 (6 h), over a 72-day season.

Per-event scalars (top level): `label`, `true_class`, `keep_prob`, `dchi2_event`,
`dchi2_anomaly`, `m_base_ref`, `a_ks`, `n_usable_bands`, and the `params` table (below).

## 2. Cache shard (HDF5, ~46 MB) — what the model reads

`cache.py` bins each band into fixed token slots, keeping the **min and max** of each bin (not
just the mean), so a narrow caustic spike survives binning:

| dataset | shape | meaning |
|---|---|---|
| `feat/<band>` | (N, L, 3) | per-token `[mean, min, max]` baseline-relative deviation (fp16) |
| `frac/<band>` | (N, L) | fraction of the token's bin actually observed (fp16) |
| `n_kept/<band>`, `f_s/<band>` | (N,) | bookkeeping |

Token counts `L`: F146 = 864, F087 = 96, F213 = 96. The model's conv stem downsamples these to
108 / 24 / 24 = **156 tokens** and forms the 5-channel input `[mean, min, max, frac, mask]` (the
`mask` is derived from finiteness at load time). Plus the same per-event scalars and `params` as
the raw shard.

## 3. The `params` table (`PARAM_FIELDS`)

A fixed (N, 16) float32 array; unused fields for a given class are `NaN`. Order (from
`writer.py`):

```
t0, tE, u0, q, s, alpha, rho,      # microlensing
P, amp_I, ratio_k,                 # periodic / general variability
t_start, rise, decay, plateau, recur,   # eruptive
t_anom                             # truth-informed noise-free anomaly-onset proxy
```

`t_anom` is **new in BinML 1.0** and only finite for events labelled NonPSPL under the adopted
synthetic policy. It is computed from the injected noise-free binary curve and its best-fit PSPL
counterpart, so a live broker cannot observe it directly. It drives the partial-season training
labels (see [`evaluation.md`](evaluation.md) §3): under truncation a binary is labelled PSPL until
`t_anom`, then NonPSPL.

## 4. Labels

`label` is the **observable** class (after detectability relabelling); `true_class` is what was
generated. They differ exactly when the generated event is not observable as itself — an
undetectable microlensing event → Flat, an undetectable binary anomaly → PSPL. Train and
evaluate on `label`; keep `true_class` to audit the label boundary.

```
0 Flat   1 PSPL   2 NonPSPL   3 PeriodicVar   4 LongPeriodVar   5 Eruptive
```

## 5. `keep_prob`

The byproduct-subsampling weight. NonPSPL rows have `keep_prob = 1`; demoted PSPL/Flat byproduct
rows carry the subsampling fraction. **Recall is not reweighted; precision/purity are** — see
[`evaluation.md`](evaluation.md) §1.

## 6. Reading a shard

```python
import h5py, numpy as np
with h5py.File("shard_00000.h5") as f:
    feat = f["feat/F146"][:]          # (N, 864, 3)
    y    = f["label"][:]              # observable class
    par  = f["params"][:]             # (N, 16)
    fields = [x.decode() for x in f.attrs["param_fields"]]
    t_anom = par[:, fields.index("t_anom")]
```

For training you normally go through [`to_memmap.py`](../pipeline/to_memmap.py), which
scatters cache shards into a shuffled fp16 memmap for fast random access.

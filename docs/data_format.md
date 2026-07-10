# BinML Dataset Format — Compact HDF5 v4.2.0

This document specifies the on-disk format of BinML training and evaluation data:
the **compact HDF5 v4.2.0** shard produced by `pipeline/simulate.py`. It describes
every dataset in a shard, the observation cadence, the physical meaning of each
field, and how to read a shard back into arrays.

A *shard* is a single `.h5` file holding a batch of simulated microlensing light
curves. Rows are **row-aligned** across all datasets: index `i` refers to the same
event in `flux`, `delta_t`, `labels`, `m_base`, `mask`, and `params`. Events within
a shard are stored in shuffled order (classes interleaved).

---

## 1. Overview

Each event is a simulated Nancy Grace Roman Space Telescope light curve belonging to
one of three classes:

| Label | Class  | Meaning                                                            |
|:-----:|:-------|:------------------------------------------------------------------|
| `0`   | Flat   | No event (constant baseline plus noise).                          |
| `1`   | PSPL   | Point-source point-lens (single-lens) microlensing.               |
| `2`   | Binary | Binary/planetary lens (the distinct anomalous class).            |

A Binary event is **not** a PSPL event: the Binary class is the anomalous
caustic-bearing regime, distinct from the smooth single-lens PSPL magnification.

The format is *compact*: instead of storing an identical per-event timestamp array,
the shared observation grid is stored **once** (`time_grid`), the per-event
observation pattern is stored as a **packed bitmask** (`mask`), and `delta_t` is
reconstructable exactly from those two. This removes roughly a third of the dense
storage relative to earlier formats that carried a per-event `timestamps` row.

The format version is recorded in the file attribute `version` (`"4.2.0"`).

---

## 2. Cadence: the Roman observation grid

All events share one regular observation grid:

- **6912 points** per event.
- **15-minute** sampling (`cadence_minutes = 15`).
- **72-day** Roman observing season (`season_duration_days = 72`).

That is, `72 days × 24 h × 4 samples/h = 6912` samples. The grid is stored once as
the `time_grid` dataset (see below); every event indexes into the same 6912 time
points. `6912 = 864 × 8` is an exact multiple of 8, which is why the packed
observation mask round-trips with no padding (Section 3.5).

Not every grid point is necessarily observed for a given event — the per-event
`mask` marks which of the 6912 points are valid observations. Unobserved points are
encoded as sentinel values in `flux` and `delta_t` (Sections 3.1, 3.2).

---

## 3. Datasets

A shard is an HDF5 file with the following datasets. `N` is the number of events in
the shard; `T = 6912` is the number of grid points.

| Dataset          | Shape             | dtype               | Meaning                                                  |
|:-----------------|:------------------|:--------------------|:---------------------------------------------------------|
| `flux`           | `(N, 6912)`       | `float32`           | Normalized **magnification** A (0.0 = unobserved).       |
| `delta_t`        | `(N, 6912)`       | `float32`           | Days since previous valid observation.                   |
| `labels`         | `(N,)`            | `int32`             | Class label 0 / 1 / 2 (Flat / PSPL / Binary).            |
| `m_base`         | `(N,)`            | `float32`           | Baseline magnitude of the source.                        |
| `mask`           | `(N, 864)`        | `uint8`             | Packed per-event observation bitmask (`np.packbits`).    |
| `time_grid`      | `(6912,)`         | `float64`           | Observation times in days. Stored **once**, not per-event.|
| `params`         | `(N,)`            | structured `float64`| Row-aligned physical parameters (see Section 4).         |
| `params_flat`    | `(n_flat,)`       | structured `float64`| Per-class parameter table (Flat). Backward compatibility.|
| `params_pspl`    | `(n_pspl,)`       | structured `float64`| Per-class parameter table (PSPL). Backward compatibility.|
| `params_binary`  | `(n_binary,)`     | structured `float64`| Per-class parameter table (Binary). Backward compatibility.|

All 2-D datasets and 1-D physical arrays are LZF-compressed and chunked for
efficient partial reads.

### 3.1 `flux` — normalized magnification A

Despite the historical name `flux`, this dataset holds the **normalized
magnification** A, not a raw flux. Magnification is defined relative to the source
baseline:

```
A = 10 ** (0.4 * (m_base - mag))
```

so the out-of-event baseline is **A = 1.0**. Equivalently, to recover an apparent
magnitude for plotting:

```
mag = m_base - 2.5 * log10(A)
```

- A Flat event sits at A ≈ 1.0 throughout (baseline plus photometric noise).
- A PSPL event shows a single smooth magnification bump.
- A Binary event shows the single-lens bump plus short-lived caustic anomalies.

**Unobserved points are encoded as `0.0`.** Because the physical baseline is
`A = 1.0`, a stored value of exactly `0.0` is unambiguously a masked / unobserved
sample, not a real measurement. Use the `mask` dataset (Section 3.5) as the
authoritative record of which points are valid.

### 3.2 `delta_t` — time since previous observation

`delta_t[i, t]` is the elapsed time, in **days**, since the previous *valid*
observation of event `i` at grid position `t`. The first valid observation of an
event has `delta_t = 0`. Masked / unobserved positions carry `delta_t = 0` as well;
they are identified via the `mask`, never by their `delta_t` value.

`delta_t` is the model's second input channel: it lets the causal network reason
about irregular sampling gaps directly, without an absolute clock.

`delta_t` is fully reconstructable from `time_grid` and the unpacked `mask`
(Section 5); the stored array is bit-exact against that reconstruction.

### 3.3 `labels` — class

Integer class label per event: `0` = Flat, `1` = PSPL, `2` = Binary. This is the
supervised target.

### 3.4 `m_base` — baseline magnitude

The source baseline magnitude for each event, drawn per event during simulation.
`m_base` is the reference used to convert between magnification A and apparent
magnitude (Section 3.1). Its value is also mirrored in the `params` table.

### 3.5 `mask` — packed observation bitmask

`mask` records, for each event, which of the 6912 grid points are valid
observations. It is stored as **packed bits**: the boolean array of length 6912
(`True` = observed) is packed with `np.packbits` along the time axis into
`6912 / 8 = 864` bytes per event. Hence the on-disk shape `(N, 864)`, dtype
`uint8`.

Because `6912` is a multiple of 8, `np.packbits` / `np.unpackbits` round-trip with
no trailing padding. The file attributes `mask_packed = True` and
`mask_n_bytes = 864` document this. To use the mask, unpack it back to a boolean
length-6912 array (Section 5).

### 3.6 `time_grid` — shared observation times (stored once)

`time_grid` is a length-6912 `float64` array giving the observation times in days
across the 72-day season. It is stored **exactly once per shard** and shared by
every event, rather than duplicated as a per-event `timestamps` row. `float64` is
retained (the cost is negligible for a single 6912-element array) so that `delta_t`
reconstructed from `time_grid` is bit-exact against the stored `delta_t`.

The endpoints are also exposed as file attributes `time_grid_start` and
`time_grid_end`.

---

## 4. The `params` struct

`params` is a length-`N` structured array (all fields `float64`), **row-aligned**
with `flux` / `delta_t` / `labels`. This is the provenance-correct parameter table:
unlike the per-class `params_flat` / `params_pspl` / `params_binary` arrays (kept
for backward compatibility), `params` is directly indexable by global event row.

Fields that are **not applicable** to an event's class (for example `q`, `s`, `rho`
for a Flat or PSPL event, or the binary-only detectability metrics) are stored as
**`NaN`**, so they are distinguishable from a genuine zero. The exact field order is
recorded in the file attribute `param_fields`.

| Field                | Applies to     | Meaning                                                                 |
|:---------------------|:---------------|:------------------------------------------------------------------------|
| `t0`                 | PSPL, Binary   | Time of peak / closest approach (days).                                 |
| `tE`                 | PSPL, Binary   | Einstein-radius crossing time (days).                                   |
| `u0`                 | PSPL, Binary   | Impact parameter (minimum lens–source separation, Einstein radii).      |
| `q`                  | Binary         | Lens mass ratio (secondary / primary).                                  |
| `s`                  | Binary         | Projected lens separation (Einstein radii).                             |
| `rho`                | Binary         | Source angular radius (Einstein radii).                                 |
| `alpha`              | Binary         | Source-trajectory angle relative to the binary axis.                    |
| `m_base`             | all            | Baseline magnitude (mirrors the `m_base` dataset).                      |
| `peak_magnification` | PSPL, Binary   | Maximum magnification A reached by the event.                           |
| `snr`                | all            | Signal-to-noise ratio of the event.                                     |
| `anomaly_dchi2`      | Binary         | Detectability of the binary anomaly (see Section 4.1).                  |
| `max_anomaly`        | Binary         | Largest single-point deviation of the binary from the matched single-lens fit. |

The precise numeric field set present in a given shard is authoritatively listed in
`file.attrs["param_fields"]`; treat that attribute as the source of truth for column
order when reading the structured array.

### 4.1 `anomaly_dchi2` — physical detectability of the anomaly

`anomaly_dchi2` is the **physical detectability** of a binary's anomaly, and is the
single most important field for interpreting Binary-class performance. It is the
chi-squared difference between the true binary light curve and a **matched
single-lens (PSPL) fit** to that same event:

```
anomaly_dchi2 = sum( ( (A_binary - A_single_fit) / sigma )^2 )
```

Intuitively: how far, in units of photometric noise, the binary deviates from the
best single-lens explanation of its own data. It answers "could any detector, in
principle, tell this binary apart from a PSPL?" — independent of the model.

- **Low `anomaly_dchi2`** → the binary is physically indistinguishable from a PSPL;
  the caustic anomaly leaves essentially no signal in the sampled photometry.
- **High `anomaly_dchi2`** → a strong, well-sampled anomaly that any competent
  classifier should catch.

`anomaly_dchi2` is defined only for Binary events (`NaN` for Flat and PSPL). It
drives detectability-aware subset selection during training, and it is the axis
along which Binary recall should be reported: Binary performance conditioned on
`anomaly_dchi2` is the honest metric, not the raw population rate.

`max_anomaly` is a companion binary-only metric recording the largest single-point
deviation from the matched single-lens fit.

---

## 5. Reading a shard

Datasets are row-aligned, so a single row index selects one event across all
arrays. The `flux` (magnification) and `delta_t` channels are the model inputs; the
`mask` is the authoritative record of valid observations.

```python
import h5py
import numpy as np

with h5py.File("shard.h5", "r") as f:
    # --- format / cadence metadata ---
    assert f.attrs["version"] == "4.2.0"
    n_points = int(f.attrs["n_points"])          # 6912
    time_grid = f["time_grid"][:]                # (6912,) float64, shared by all events
    param_fields = list(f.attrs["param_fields"]) # column order of the params struct

    # --- one event, row i ---
    i = 0
    A = f["flux"][i]            # (6912,) magnification; 0.0 == unobserved
    dt = f["delta_t"][i]        # (6912,) days since previous valid observation
    label = int(f["labels"][i]) # 0=Flat, 1=PSPL, 2=Binary
    m_base = float(f["m_base"][i])
    params = f["params"][i]     # structured record; NaN where a field doesn't apply

    # --- unpack the packed observation mask (864 bytes -> 6912 bools) ---
    mask = np.unpackbits(f["mask"][i])[:n_points].astype(bool)  # True = observed

    # valid samples only
    obs_times = time_grid[mask]
    obs_A = A[mask]

    # apparent magnitude for plotting (valid points)
    obs_mag = m_base - 2.5 * np.log10(obs_A)

    # detectability of a binary anomaly (NaN unless label == 2)
    dchi2 = float(params["anomaly_dchi2"])
```

### 5.1 Reconstructing `delta_t` from the grid and mask

`delta_t` is stored explicitly, but it can be regenerated exactly from `time_grid`
and the unpacked `mask` — this is the invariant the compact format relies on:

```python
def compute_delta_t(time_grid, mask):
    """Days since previous valid observation; first valid point = 0."""
    dt = np.zeros_like(time_grid, dtype=np.float32)
    prev = -1
    for t in range(len(time_grid)):
        if mask[t]:
            dt[t] = 0.0 if prev < 0 else (time_grid[t] - time_grid[prev])
            prev = t
    return dt

# bit-exact against the stored f["delta_t"][i] on the observed positions
dt_recon = compute_delta_t(time_grid, mask)
```

### 5.2 Notes for loaders

- **Masked sentinels.** In `flux`, `0.0` marks an unobserved point (baseline is
  `A = 1.0`, so real data never equals `0.0`); in `delta_t`, masked and first-valid
  points are `0.0`. Always resolve validity through the unpacked `mask`, not through
  sentinel values.
- **NaN in `params`.** A `NaN` field means "not applicable to this event's class,"
  not a missing measurement. Guard against it before using binary-only fields such
  as `q`, `rho`, or `anomaly_dchi2`.
- **Row alignment.** Index `i` is consistent across `flux`, `delta_t`, `labels`,
  `m_base`, `mask`, and `params`. The per-class `params_{flat,pspl,binary}` arrays
  are **not** indexable by global row and exist only for backward compatibility;
  prefer the row-aligned `params` table.
- **Compaction for the model.** The BinML network consumes the two channels
  (magnification A and `delta_t`) with valid observations compacted to a contiguous
  prefix and a `lengths` count of valid points, then normalized by dataset
  statistics fit on the training split. The mask defines which points are compacted
  in.

---

## 6. File attributes (metadata)

Shard-level provenance and configuration is stored as root HDF5 attributes,
including:

- `version` — format version string (`"4.2.0"`).
- `n_events`, `n_flat`, `n_pspl`, `n_binary` — event counts by class.
- `n_flat_requested`, `n_pspl_requested`, `n_binary_requested` — requested counts.
- `n_points` — samples per event (`6912`).
- `cadence_minutes`, `cadence_days` — sampling interval (`15` min).
- `season_duration_days` — season length (`72` days); `mission_duration_days`.
- `time_grid_start`, `time_grid_end` — first/last observation time (days).
- `mask_packed` (`True`), `mask_n_bytes` (`864`) — packed-mask layout.
- `has_time_grid`, `has_global_params`, `timestamps_dropped` — compact-format flags.
- `param_fields` — ordered field names of the `params` struct.
- `binary_preset`, `require_caustic`, `seed`, `oversample_factor` — simulation
  configuration.

These attributes are sufficient to fully interpret a shard without external
context.

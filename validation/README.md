# Real-data validation

Two models, two regimes. **BinML v1** (6-class, 3-band) is a *specialist* for Roman's dense
15-min cadence. The **legacy 3-class single-band model** (`binml.legacy`) is the tool for real,
sparse, ground-based light curves. These scripts establish that division with real OGLE-IV events.

## Scripts

- **`cadence_robustness.py`** — subsamples dense simulated events across point densities and
  measures anomaly recovery. Recovery is intact at Roman's native ~96 pts/day, ~0.71 at ~40/day,
  and **collapses below ~20 pts/day** (sparse peaks read as variable stars). This is the
  quantitative *scope boundary* of v1, not a bug: Roman will deliver dense cadence.

- **`real_data_validate.py`** — runs BinML v1 (single-band mode) on real OGLE-IV events. Every
  event → PeriodicVar, confirming the sweep: OGLE's ~nightly cadence is an order of magnitude
  below v1's floor. v1 is out of scope for sparse ground data by design.

- **`kmtnet_validate.py`** — BinML v1 on real KMTNet events (the densest ground survey, ~3-site
  10-15 min cadence). Still fails: even events at 60-80 pts/day → PeriodicVar. A controlled test
  shows why — dense simulated events thinned *uniformly* to ~60/day keep ~0.7 anomaly recall, but
  the same events observed through a realistic ~8h *nightly* window collapse to 0.00. It is the
  diurnal sampling gap (read as ~1-day variability), not the average rate. No ground network
  escapes the day/night cycle; v1 needs Roman's continuous space cadence. **Validation of v1
  genuinely awaits Roman.**

- **`microlia_compare.py`** — MicroLIA (Godines+2019) on the same real events. **Health warning:**
  MicroLIA is bit-rotted (PyPI 2.8.1 missing its Mira simulator; GitHub main won't import — dead
  RRLyrae template URL), so this monkeypatches two simulators; numbers are indicative.

## Real-data results (8 known OGLE-IV anomalous events)

| model | what it answers | result |
|---|---|---|
| **legacy 3-class (single-band)** | Flat / PSPL / **Binary** | **4/8 flagged Binary** — every strong caustic-crossing binary correct (2014-BLG-0289, 2013-BLG-0578, 2013-BLG-0341, 2015-BLG-0966); misses are subtle low-amplitude planets |
| **MicroLIA** | microlensing / variable / CV / … | 6/8 detected as microlensing, but **no binary/anomaly class** — cannot flag the planet |
| **BinML v1 (6-class, single-band mode)** | 6 classes | fails: OGLE cadence far below its dense-cadence floor (out of scope) |

## Conclusion
The legacy single-band model validates on real ground data — it correctly flags the strong real
binaries and recognises microlensing. MicroLIA detects microlensing but stops there; distinguishing
the anomaly is exactly what BinML adds. BinML v1 is a Roman-cadence specialist and is validated on
simulations until Roman (or a comparably dense survey) flies; real sparse data is served by the
legacy model.

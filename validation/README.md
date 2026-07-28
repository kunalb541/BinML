# Real-data validation

BinML is trained on Roman's dense three-band cadence. These scripts probe how it behaves on
real, single-band, sparse archival data — and quantify the domain gap.

- **`cadence_robustness.py`** — subsamples dense simulated events to a range of point densities
  and measures anomaly recovery. Result: recovery is intact at Roman's native ~96 pts/day,
  degrades by ~40 pts/day, and **collapses below ~20 pts/day** (sparse peaks read as variable
  stars). This is the quantitative statement of the sparse-cadence limitation.

- **`real_data_validate.py`** — fetches real OGLE-IV EWS events (single I-band) and runs BinML in
  single-band mode (colour masked). Confirms the sweep on real data: at OGLE's ~nightly cadence
  BinML does not recognise known events, because that cadence is far below the ~20 pts/day floor.

## Conclusion
The current model needs ~Roman-like dense cadence. Meaningful real-data validation therefore
requires either (a) dense-cadence real events (KMTNet, or Roman itself), or (b) making the model
cadence-robust via subsampling augmentation during training. Single-band mode itself works — a
dense simulated event classifies correctly with F146 alone; only sparsity breaks it.

# Glossary

Terms used across BinML's docs and API, for readers new to microlensing or to the pipeline.

## Microlensing physics
- **Microlensing** — transient brightening of a background star as a foreground mass (the lens)
  passes near the line of sight and gravitationally focuses its light.
- **PSPL** (point-source point-lens) — a single-lens event; a smooth, symmetric brightening
  described by the Paczyński curve.
- **NonPSPL / anomalous** — under the released training policy, a detectable deviation from a
  single-lens fit produced by a static binary lens (2L1S, planetary through stellar mass ratios),
  with finite-source magnification included. Binary-source (1L2S), parallax, and lens orbital
  motion are not present in the released generator or evaluation.
- **tE** (Einstein timescale, days) — how long the event lasts; the time to cross the Einstein
  radius. Roman-bulge events peak around 10–40 d.
- **u0** (impact parameter) — closest approach of the source to the lens, in Einstein radii.
  Small u0 → high magnification, sharper peak.
- **t0** — time of peak (closest approach).
- **q** (mass ratio) — companion/host mass ratio. Planetary regime q ≲ 10⁻²; the planet
  mass-ratio function breaks near q ≈ 1.7×10⁻⁴ (Suzuki+2016).
- **s** (separation) — projected lens separation in Einstein radii. Sets the caustic topology
  (close / resonant / wide).
- **caustic** — closed curves in the source plane where magnification formally diverges; a source
  crossing one produces the sharp spikes/bumps that mark a binary. Short-lived (hours).
- **ρ** (rho) — source angular size in Einstein radii; finite-source effects smooth sharp caustic
  features.
- **α** (alpha) — angle of the source trajectory relative to the binary axis.
- **blending / f_s** — the source flux fraction: a Roman pixel usually blends the magnified source
  with unmagnified neighbours, so the *observed* amplitude is diluted (`A_obs = 1 + f_s·(A−1)`).
- **achromatic** — microlensing magnifies all wavelengths equally, so the event looks the same in
  every band (a key discriminator against chromatic variable stars).

## Contaminant classes
- **PeriodicVar** — short-period pulsators/eclipsers (RR Lyrae, eclipsing binaries, δ Scuti).
- **LongPeriodVar** — Miras, semiregulars (SRV), OSARGs; periods ≳ the season, so they look like a
  single smooth trend — the most dangerous microlensing impostor.
- **Eruptive** — dwarf novae, WZ Sge, Be-star outbursts; single brightening episodes.

## Survey / data
- **Roman GBTDS** — Nancy Grace Roman Space Telescope Galactic Bulge Time-Domain Survey.
- **F146 / F087 / F213** — Roman filters. The current design samples F146 approximately every
  12 minutes and each colour band on a staggered 6-hour cycle. BinML's released legacy schedule
  instead uses 15-min F146 sampling and non-staggered 6-hour colour grids.
- **season** — a roughly 72-day Roman observing window. BinML classifies one simulated season at a
  time and does not represent the current survey's multi-season structure.
- **m_base** — the source's baseline (quiescent, out-of-event) magnitude. The model input is
  magnitude *relative to* this baseline.
- **detectability-conditioned label** — an event is labelled by what is *observable*: an
  undetectable event → Flat, an undetectable anomaly → PSPL.

## Metrics
- **Δχ² (anomaly)** — χ² improvement of the true binary model over the best single-lens fit; the
  *detectability* of the anomaly. Low Δχ² → observationally indistinguishable from PSPL.
- **completeness @ fixed purity** — the recall achievable at a chosen purity (precision). The
  headline metric, because a follow-up pipeline is specified against a purity target — not accuracy.
- **purity** — precision: fraction of flagged events that are truly of that class.
- **keep_prob** — a per-event byproduct-subsampling weight; NonPSPL rows are 1, subsampled
  Flat/PSPL byproducts are <1. Used to reweight *precision/purity* (never recall) back to the
  true population.
- **the cascade** — the intended training-label progression on a partial synthetic season:
  Flat → PSPL → NonPSPL. The binary-only half-day scan measures timing relative to a
  truth-informed onset proxy; it does not prove that every event follows all three states or that
  no premature alert occurs.
- **t_anom** — a truth-informed, noise-free proxy for the first cut at which the injected binary
  differs sufficiently from its best-fit single-lens curve. It drives truncation labels but is not
  an alert timestamp available from live survey data.

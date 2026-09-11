# Deploying BinML on Roman survey data: an experimental plan

> **Experimental, for fun, 2026-09-12.** Not part of the paper, not validated on real data, and not an official
> Roman, IPAC, STScI or RGES-PIT product or plan. Facts are linked to their sources as of this date; everything
> labelled *estimate* is our arithmetic. BinML has only ever been tested on simulations.

## 1. Where things stand

Roman launched on 30 August 2026 and is commissioning on its way to L2 ([NASA](https://science.nasa.gov/mission/roman-space-telescope/)).
The Galactic Bulge Time Domain Survey (GBTDS) will observe about 1.7 deg² of the bulge at a 12-minute cadence in
high-cadence seasons and a five-day cadence in low-cadence seasons over the five-year mission
([IPAC, Roman observations](https://roman-docs.ipac.caltech.edu/roman-proposals-home/cycle-1-call-for-proposals/roman-observations)).
Pre-launch simulations predicted about 27,000 microlensing events with |u0| < 1 (about 54,000 with |u0| < 3) and
about 1,400 bound planets over the survey, for the design studied then
([Penny et al. 2019, table and abstract](https://arxiv.org/abs/1808.02490)).

## 2. How Roman will release GBTDS data

| What | Fact | Source |
|---|---|---|
| Data rights | No proprietary period for any Roman survey | [STScI, Feb 2026](https://www.stsci.edu/contents/newsletters/2026-volume-43-issue-01/roman-delivering-data-that-unlocks-discovery) |
| Archive | MAST (STScI); cloud copy in the `stpubdata` S3 bucket, AWS us-east-1, anonymous access; ASDF files | [Roman docs, cloud access](https://roman-docs.stsci.edu/data-handbook-home/accessing-wfi-data/the-roman-archive-in-mast/cloud-access) |
| Generic prompt products | L1 immediately, L2 ~2 days, L3 ~5 days, L4 ~7 days after downlink; ~1.4 TB per day overall | [STScI, Feb 2026](https://www.stsci.edu/contents/newsletters/2026-volume-43-issue-01/roman-delivering-data-that-unlocks-discovery) |
| GBTDS processing | IPAC's Microlensing Science Operations System: a **daily** photometry pipeline (PSF fitting and difference imaging) that updates light curves, products **within 48 hours**; an end-of-season Events pipeline (variability and microlensing-event catalogues with model fits), within 6 months; catalogues as Parquet, hosted at MAST; no near-real-time alerts mentioned | [IPAC, GBTDS pipelines](https://roman-docs.ipac.caltech.edu/data-handbook/roman-wfi-data-pipelines/galactic-bulge-survey-pipelines) |
| Data releases | Uniformly reprocessed about six months after a season ends | [Roman docs, data releases](https://roman-docs.stsci.edu/data-handbook/wfi-data-levels-and-products/data-releases) |
| Alerts | RAPID (PI M. Kasliwal): image differencing of every Roman image and a public alert stream of all transient and variable candidates **within 1 hour**, light-curve history **within 24 hours**, for the Core Community and General Astrophysics Surveys, in the format pioneered by ZTF, archived at MAST | [RAPID](https://rapid.ipac.caltech.edu/), [Roman PITs](https://roman.gsfc.nasa.gov/science/Roman_teamlist_pit.html) |
| Transient alerts in general | Treated as user-contributed ("Level 5") products | [STScI, Feb 2026](https://www.stsci.edu/contents/newsletters/2026-volume-43-issue-01/roman-delivering-data-that-unlocks-discovery) |

RAPID's alert packets carry a real/bogus score whose threshold users choose
([Gandhi et al. 2026](https://arxiv.org/html/2606.05103v1)). We found no statement of RAPID's GBTDS alert rate or
transport (ZTF streams over Kafka in Avro, and so will Rubin's; [Fink](https://github.com/astrolabsoftware/fink-broker)).
Rubin's full alert stream goes to seven community brokers, several of which host user classifiers
([Rubin, alerts and brokers](https://rubinobservatory.org/for-scientists/data-products/alerts-and-brokers)); Fink already runs a
microlensing module ([Fink](https://academic.oup.com/mnras/article/501/3/3272/5992334)). Whether RAPID's stream will flow
through those brokers is not stated.

## 3. What BinML would consume, and what changes on real data

BinML scores one season of up to 72 days: F146 plus optional F087/F213, binned into 2-hour tokens, with a baseline
magnitude supplied. The realistic inputs are (a) IPAC's daily light-curve updates for catalogued stars, or
(b) RAPID's alert packets and light-curve histories for difference-image candidates.

- **Use the gap-aware checkpoint** `validation/gulls/weights/ft_fspl5s_seasons_g08.pt` at threshold 0.956. The shipped
  checkpoint collapses on a schedule with pauses.
- **Colour cadence differs.** IPAC describes one exposure in the bluest filter every 12 hours
  ([IPAC GBTDS page](https://vmromanweb1.ipac.caltech.edu/page/galactic-bulge-time-domain-survey.html)); our training
  colour cadence was faster. Start F146-only, which is the cascade's primary mode.
- **Baseline.** A live pipeline must estimate the baseline from earlier seasons or the season's first days (our
  RMDC26 tests used the whole survey, which a live pipeline would not have).
- **Real systematics** (crowding, blending, detector effects, difference-imaging residuals) are in neither
  simulator. Expect calibration drift; plan a shadow period.
- **Alert burden.** On RMDC26, 3.2% of single lenses cross the recommended threshold at some point in a
  half-day scan of the season (`validation/gulls/cascade_gulls.json`), and 2.0% with daily batches (below). At a
  realistic planetary prevalence most alerts would be single lenses, so a second stage (a single-lens refit test, or
  human vetting) is needed before anything is broadcast. RMDC26 has no variable stars, so real contamination is
  worse.

## 4. Getting data in near real time, ranked

1. **IPAC daily light-curve products** (daily, within 48 h, Parquet via MAST/`stpubdata`). Simplest; cadence of
   one batch per day; the product format is not yet public.
2. **RAPID alerts** (within 1 h, ZTF-like packets, archived at MAST). Lowest latency; covers difference-image
   candidates, not every star; transport and GBTDS alert rate not yet documented.
3. **Community brokers**, if RAPID's stream is fed to them: run BinML as a broker module (Fink-style) instead
   of our own ingest.
4. **Pixels** (L1/L2 images) with our own photometry: not sensible for a hobby deployment at ~1.4 TB/day.

## 5. Proposed prototype architecture

```
 MAST stpubdata (us-east-1)            RAPID alerts (MAST archive / broker)
        |  daily Parquet                         |  hourly packets
        v                                        v
 [ingest job] -- append new epochs per object --> [object store: Parquet on S3 or EBS]
        |                                                       |
        v                                                       v
 [rescore objects with new data: BinML ft_fspl5s_seasons_g08, F146, threshold 0.956]
        |
        v
 [second stage: single-lens refit / vetting]  --> [watchlist: JSON/CSV + static page]
        |                                          (optional: TOM Toolkit, broker topic)
        v
 [monitoring: alerts per day, score drift, class mix vs RMDC26 replay]
```

Per-object state is the season's light curve so far plus the first day the object crossed the threshold. Each
batch rescores only objects that received data; the partial-season cascade is exactly this rescoring.

## 6. Cost model (estimates)

Measured on an Apple M5 CPU with real RMDC26 light curves (median 7,701 F146 points), one thread
(`experimental/deployment/benchmark_inference.py` → `benchmark_results.json`): tokenising a season takes 2.6 ms and
a forward pass 1.4–2.0 ms, so **one rescore costs about 4.5 ms of CPU**; the daily replay below measured 3.5 CPU-s
per 1,000 rescored events. Peak memory about 0.5 GB. Cloud vCPUs are slower than an M5 core; the estimates assume
10 ms per rescore to leave room.

| Scenario (*estimate*) | Objects rescored | CPU per day | Fits on |
|---|---|---|---|
| A. Microlensing watchlist, daily IPAC batch | ~10⁴ | 10⁴ × 10 ms ≈ 2 min | anything; Lambda free tier |
| B. All variable candidates, daily | ~10⁶ | 10⁶ × 10 ms ≈ 2.8 h | one small instance |
| C. RAPID-driven, hourly, 10⁵ objects/hour | 2.4 × 10⁶ | ≈ 6.7 h | one 2-vCPU instance |
| D. Every monitored star (~10⁸) daily | 10⁸ | ≈ 280 h (≈ 12 cores busy) | not sensible: filter by variability first |

| Option (us-east-1, on-demand) | Price | Month (730 h) | Notes |
|---|---|---|---|
| t4g.small (2 vCPU Graviton, 2 GiB) | $0.0168/h ([Vantage](https://instances.vantage.sh/aws/ec2/t4g.small)) | ≈ $12 | burstable (sustained load spends CPU credits); fine for A-B; spot ≈ $0.007/h |
| c7g.large (2 vCPU, 4 GiB) | $0.0725/h ([Vantage](https://instances.vantage.sh/aws/ec2/c7g.large)) | ≈ $53 | sustained C |
| m7i-flex.large (2 vCPU, 8 GiB) | $0.096/h ([Vantage](https://instances.vantage.sh/aws/ec2/m7i-flex.large)) | ≈ $70 | Free-Tier eligible for accounts created after 15 July 2025 (6 months or $100-200 of credits; [AWS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-free-tier-usage.html), [AWS Free Tier](https://aws.amazon.com/free/)) |
| Lambda, scenario A or B at 2 GB | $0.0000166667/GB-s, 400,000 GB-s/month free ([AWS](https://aws.amazon.com/lambda/pricing/)) | A ≈ 6,000 GB-s/month: free; B ≈ 600,000 GB-s: ≈ $3 | PyTorch container image; cold starts |
| Storage | S3 Standard $0.023/GB-month; S3→EC2 in-region transfer free; internet egress $0.09/GB after 100 GB ([AWS](https://aws.amazon.com/s3/pricing/)) | 10⁶ light curves ≈ 60 GB ≈ $1.4 | run in us-east-1, next to `stpubdata` |

Arithmetic: B = 10⁶ × 0.01 s = 10⁴ s/day × 2 GB × 30 days = 6 × 10⁵ GB-s, minus 4 × 10⁵ free = 2 × 10⁵ GB-s ×
$0.0000166667 ≈ $3.3/month. Storage: 10⁶ objects × ~7,700 points × 8 bytes ≈ 62 GB. **Conclusion (estimate): the
model is not the cost; one t4g.small or the Lambda free tier covers scenarios A-B, and ingest design dominates.**

## 7. Phased plan

| Phase | What | Done when |
|---|---|---|
| 0 (now) | Replay RMDC26 in daily batches (`replay_rmdc26.py`, below); build the ingest, state store and watchlist against it | the replay runs as a scheduled job on one small instance, no manual steps |
| 1 (first public GBTDS season) | Shadow mode on IPAC daily products or RAPID alerts: score, never broadcast | score distributions and alert rates on real data compared with the replay; thresholds recalibrated on real single lenses |
| 2 | Publish a clearly labelled experimental watchlist; optionally a broker module or TOM Toolkit instance | second-stage vetting in place; community norms for follow-up checked with the RGES/Roman microlensing groups |

**Phase 0 result** (`replay_rmdc26.py` → `replay_results.json`; 300 RMDC26 events per class from the local curve
cache, F146 only, threshold 0.956, one day revealed per batch): 2.0% of single lenses, 33% of 1S2L and 37% of 2S2L
events were on the watchlist by season end, a median 41–48 days into the season for the planetary classes, at
3.5 CPU-s per 1,000 rescored events. These are generator labels; many planetary events carry no anomaly our label
policy would claim (`paper/REVISION.md` §1½ row 3).

## 8. Risks and open questions

- BinML is validated on two simulations only; real systematics may change everything above.
- The single-lens alert burden (above) makes raw alerts a poor follow-up list at realistic prevalence.
- IPAC's daily product schema and RAPID's GBTDS alert rate and transport are not yet published; revisit once
  commissioning data appear.
- Baseline estimation in real time, colour cadence, and the season boundaries all differ from training.
- Data rights are open (no proprietary period), but broadcasting anomaly alerts on others' survey data should
  follow the Roman microlensing community's coordination norms; label everything experimental.
- Nothing here has been deployed; no cloud resources were created for this note.

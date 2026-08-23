"""Cross-simulator transfer of BinML onto RMDC26 (GULLS) light curves.

RESULT (2026-08-23): as shipped, BinML cannot be scored on this data.  RMDC26 implements the
real GBTDS schedule, in which F146 pauses ~6 h seven times per season; BinML was trained on a
continuous grid and reads a gap as evidence against a single lens.  On 1S1L (single lens) the
shipped model returns PSPL 4%, NonPSPL 65%, PeriodicVar 31%.  validation/gulls/gap_sensitivity.py
reproduces that with NO GULLS data by inserting the same gaps into in-distribution events
(PSPL 0.93 -> 0.11, Flat 1.00 -> 0.08, NonPSPL / PeriodicVar unchanged).  The remedy is
`pipeline/train.py --gap-aug`; see validation/modal_gap_finetune.py.  Every other mechanism
(baseline, noise, colour bands, cadence, amplitude, brightness, dataset revision) was tested and
excluded -- the history is in the comments below.

ORIGINAL DESIGN NOTES FOLLOW.

WHY.  Every review round has raised the same objection: the evaluation is synthetic and uses OUR
simulator, so the numbers may describe the simulator rather than the model.  The RGES PIT's
RMDC26 machine-learning release is the first chance to answer that before Roman flies.  It comes
from GULLS -- the community simulator behind the official Roman yield forecasts -- in the same
three bands BinML consumes (F087/F146/F213), and it is independent of our generator, priors,
noise model and cadence.

THE MEASUREMENT THIS TARGETS.  RMDC26 contains a class we cannot simulate: 2S2L, a planetary
lens acting on a BINARY SOURCE.  Binary-source (1L2S) blending is the classic astrophysical
false positive for a binary-lens anomaly, and the paper says plainly that its effect on our
purity is unmeasured.  Running BinML over the 2S2L events turns that disclosed unknown into a
number: how often does the model call a binary-source event an anomaly?

WHAT THIS IS NOT.  Not real data.  A poor score is a domain-shift result, not a refutation.
Three mismatches are left uncorrected and all of them bias the test AGAINST BinML:

  * GULLS labels are GENERATOR labels.  A 1S2L event whose planetary deviation is unobservable
    counts as an anomaly here, though our own detectability-conditioned policy would call it a
    PSPL.  Reconciling the two ontologies is the paper's whole subject; we do not do it here.
  * The GBTDS design has six high-cadence seasons and four low-cadence ones.  A 72 d window may
    land in either.  BinML's input is a dense season, and the paper already documents that it
    misreads a sparse curve as a variable star, so the two regimes are reported separately and
    never pooled -- otherwise this would restate a known cadence failure as a transfer result.
  * Blending, noise and stellar populations are GULLS', not ours.

Data: https://huggingface.co/datasets/RGES-PIT/MachineLearning  (172 GB obs table, queried
remotely with DuckDB/httpfs -- no bulk download).

Usage:  python validation/gulls_transfer.py --per-class 250
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

# PINNED to a dataset revision.  RGES-PIT re-uploaded obs and meta on 2026-08-18; a run that
# cached the metadata from one release and streamed the obs table from another silently paired
# each event_id with a different event's t0 and baseline for two of the three classes (the 1S1L
# rows happened to agree, which hid it).  `resolve/main` is not a reproducible reference.
REVISION = "a338d5bab441b5caf551d2fea9469aadfdc81ec1"
BASE = f"https://huggingface.co/datasets/RGES-PIT/MachineLearning/resolve/{REVISION}/"
OBS = BASE + "RMDC26_ML_Data_obs.parquet"
EPOCH = BASE + "RMDC26_ML_Data_epoch.parquet"
META = BASE + "RMDC26_ML_Data_meta.parquet"
WINDOW_D = 72.0          # BinML's input is one 72-day season
DENSE_MIN = 1000         # F146 epochs below which the season is low-cadence
SEASON_GAP_D = 5.0       # inter-season gaps are ~110 d; intra-season gaps are < 1 d
LABELS = {"RMDC26_1S1L_ML": "single lens (-> PSPL)",
          "RMDC26_1S2L_ML": "planetary lens (-> NonPSPL)",
          "RMDC26_2S2L_ML": "planetary lens AND binary source (-> NonPSPL)"}

# A correction worth recording, because the obvious reading of the class name is wrong.  2S2L is
# NOT the 1L2S contaminant: its Planet_q distribution is the same as 1S2L's (median 1.25e-4), so
# every 2S2L event carries a genuine planetary lens and the binary source is an ADDITIONAL
# complication in 56% of them.  A NonPSPL call on 2S2L is therefore substantially correct, not a
# false positive.  RMDC26 ships no 2S1L class, so the pure binary-source degeneracy this test was
# meant to probe is simply not present in the dataset; the false-positive quantity this test can
# measure is the 1S1L -> NonPSPL rate.


def flux_to_ab(f_ujy):
    """microjansky -> AB magnitude.  Non-positive flux (noise on faint sources) becomes NaN."""
    f = np.asarray(f_ujy, float)
    out = np.full(f.shape, np.nan)
    ok = f > 0
    out[ok] = -2.5 * np.log10(f[ok]) + 23.9
    return out


def _fetch(path, dest, what):
    if not os.path.exists(dest):
        import urllib.request
        print(f"[{what}] downloading {path}", flush=True)
        urllib.request.urlretrieve(path, dest)
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-class", type=int, default=250)
    ap.add_argument("--min-amp", type=float, default=0.1,
                    help="minimum peak observed amplitude in mag; 0 disables the cut")
    ap.add_argument("--min-te", type=float, default=1.0,
                    help="lower tE bound in days; BinML's training prior is truncated to "
                         "[1, 300] d (pipeline/priors.py), so events outside it are outside the "
                         "model's support, not hard cases within it")
    ap.add_argument("--max-te", type=float, default=300.0, help="upper tE bound in days")
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--ids", default=None,
                    help="comma-separated event_ids to run instead of sampling (debugging)")
    ap.add_argument("--chunk", type=int, default=250,
                    help="events per remote query; the obs table is clustered by event_id, so "
                         "one query per contiguous block amortises the row-group seek")
    ap.add_argument("--meta-cache", default="/tmp/rmdc26_meta.parquet")
    ap.add_argument("--epoch-cache", default="/tmp/rmdc26_epoch.parquet")
    ap.add_argument("--out", default=os.path.join(HERE, "gulls_transfer_result.json"))
    ap.add_argument("--rows-out", default=os.path.join(HERE, "gulls_transfer_events.json"))
    args = ap.parse_args(argv)

    import duckdb
    import pyarrow.parquet as pq
    import binml

    m = pq.read_table(_fetch(META, args.meta_cache, "meta"), columns=[
        "event_id", "sim_label", "t0lens1", "tE_ref", "u0lens1", "Source_F146", "fs_F146",
        "Planet_q", "Source_q", "Source_Is_Binary", "final_weight"]).to_pydict()

    # The epoch table is 4 MB and maps epoch_id -> BJD.  Two dead ends are recorded here because
    # each looked right and was not:
    #   1. Joining it REMOTELY against the 172 GB obs table, once per event, was so slow the first
    #      attempt did not finish 25 events in 15 minutes.
    #   2. Resolving the window to an epoch_id RANGE instead looks like it should prune row groups,
    #      but epoch_id is NOT ordered by time -- the first 72 d of the survey span epoch_ids 0 to
    #      98973, i.e. the entire table -- so the range selects everything and silently returns the
    #      full mission rather than one season.
    # What works: query on event_id alone (~3-7 s, the id IS the clustering key) and window
    # locally against this cached map.
    et = pq.read_table(_fetch(EPOCH, args.epoch_cache, "epoch"),
                       columns=["epoch_id", "bjd"]).to_pydict()
    ep_id = np.asarray(et["epoch_id"], np.int64)
    ep_bjd = np.asarray(et["bjd"], float)
    o = np.argsort(ep_bjd)
    ep_id, ep_bjd = ep_id[o], ep_bjd[o]
    # Vectorised epoch_id -> BJD.  A dict lookup per epoch is ~50,000 Python operations per
    # event, which dominated the runtime once the query stopped being the bottleneck.
    _id_sorted = np.argsort(ep_id)
    _ids_s, _bjd_s = ep_id[_id_sorted], ep_bjd[_id_sorted]

    def bjd_lookup(q):
        pos = np.searchsorted(_ids_s, q)
        np.clip(pos, 0, len(_ids_s) - 1, out=pos)
        out = _bjd_s[pos]
        return np.where(_ids_s[pos] == q, out, np.nan)

    # SEASONS, not arbitrary 72 d windows.  Centring the window on t0 straddles season
    # boundaries: one test event returned 5,499 epochs spanning only 50 of 72 days, another 89
    # epochs spanning 0.8 days, because most of the window fell in the inter-season gap.  BinML's
    # input contract is ONE CONTIGUOUS SEASON, so we find the season containing t0 and use that.
    # The RMDC26 baseline holds 10 seasons: six high-cadence (70.7 d, ~231 epochs/day across the
    # three bands) and four low-cadence (65.1 d, 3/day), matching the current GBTDS design.  Only
    # the high-cadence seasons are the regime BinML claims to serve.
    _gaps = np.flatnonzero(np.diff(ep_bjd) > SEASON_GAP_D)
    SEASONS = list(zip(np.r_[ep_bjd[0], ep_bjd[_gaps + 1]], np.r_[ep_bjd[_gaps], ep_bjd[-1]]))
    print(f"[seasons] {len(SEASONS)} in the RMDC26 baseline", flush=True)

    def season_of(t0):
        for a, b in SEASONS:
            if a <= t0 <= b:
                return a, b
        return None

    # DETECTABILITY-MATCHED SELECTION.  GULLS simulates the whole microlensing population,
    # including events no survey would register: the median 1S1L peak amplitude is 0.063 mag and
    # only 41% exceed 0.1 mag.  BinML's classes are detectability-conditioned, so scoring it on
    # the raw draw asks it about events outside anything it was trained on -- the first run
    # returned PeriodicVar almost uniformly, and the cause was population, not a bug (the
    # baseline was verified correct to 0.02 mag against the metadata). We therefore select on a
    # peak amplitude computed from the metadata alone, which is the closest available analogue of
    # our own amplitude floor. This is a selection on the INPUT population, identical across the
    # three classes, not on the model's output.
    u0 = np.abs(np.asarray(m["u0lens1"], float))
    fs = np.asarray(m["fs_F146"], float)
    Amax = (u0 ** 2 + 2) / np.maximum(u0 * np.sqrt(u0 ** 2 + 4), 1e-12)
    peak_amp = np.abs(-2.5 * np.log10(np.maximum(1 + fs * (Amax - 1), 1e-12)))

    rng = np.random.default_rng(args.seed)
    eid_arr = np.asarray(m["event_id"], np.int64)
    lab_arr = np.asarray(m["sim_label"])
    keep = (peak_amp >= args.min_amp) if args.min_amp > 0 else np.ones(len(eid_arr), bool)

    # TIMESCALE SUPPORT.  The first run of this test returned 63% PeriodicVar on the single-lens
    # class and only 1.8% PSPL, which is not a transfer result but a population mismatch: RMDC26's
    # 1S1L class is 33.4% sub-day events (median t_E 3.2 d), an FFP-like population, while its two
    # binary classes are 0.2% sub-day (median t_E 12-15 d).  Comparing them without a timescale cut
    # conflates lens multiplicity with timescale.  BinML's t_E prior is truncated to [1, 300] d, so
    # sub-day events are outside its support entirely -- a t_E of 0.16 d spans two of the 864 F146
    # bins and is a spike, not a profile.  The cut below is applied identically to all classes.
    te_arr = np.asarray(m["tE_ref"], float)
    in_te = (te_arr >= args.min_te) & (te_arr <= args.max_te)
    te_excluded = {}
    for lab in sorted(set(np.asarray(m["sim_label"]).tolist())):
        cl = np.asarray(m["sim_label"]) == lab
        te_excluded[lab] = {
            "n_amp_eligible": int((cl & keep).sum()),
            "n_dropped_outside_te_support": int((cl & keep & ~in_te).sum()),
            "frac_dropped": round(float((cl & keep & ~in_te).sum() /
                                        max(int((cl & keep).sum()), 1)), 4)}
    keep = keep & in_te
    print(f"[tE] support [{args.min_te}, {args.max_te}] d; dropped per class: "
          + ", ".join(f"{k.split('_')[1]} {v['frac_dropped']:.1%}"
                      for k, v in te_excluded.items()), flush=True)

    # A CONTIGUOUS window per class, not a random draw across it.  The obs table is 172 GB in
    # 17,595 row groups clustered on event_id, ~16 groups per id: a scattered sample of 1,000 ids
    # touches every group and reads the whole file, while a contiguous run of the same size
    # touches ~300 and reads 1.5 GB.  The block is not a biased sample -- a 250-id block already
    # spans 105 of the 129 GBTDS fields, so event_id does not order events by sightline -- and
    # the amplitude cut below is applied before the window is cut, identically for all classes.
    picks, windows = [], {}
    for lab in sorted(set(lab_arr.tolist())):
        cls = np.flatnonzero((lab_arr == lab) & keep)
        cls = cls[np.argsort(eid_arr[cls], kind="stable")]
        if len(cls) <= args.per_class:
            sel = cls
        else:
            start = int(rng.integers(0, len(cls) - args.per_class + 1))
            sel = cls[start:start + args.per_class]
        picks.extend(int(j) for j in sel)
        windows[lab] = [int(eid_arr[sel[0]]), int(eid_arr[sel[-1]])]
        print(f"[sample] {lab}: {len(sel)} of {len(cls)} eligible, "
              f"ids {windows[lab][0]}-{windows[lab][1]}", flush=True)
    if args.ids:
        want = {int(x) for x in args.ids.split(",")}
        picks = [int(j) for j in np.flatnonzero(np.isin(eid_arr, list(want)))]
        windows = {"explicit": sorted(want)}
    picks.sort(key=lambda j: int(eid_arr[j]))

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    clf = binml.Classifier()
    CLASSES = clf.class_names
    thr = json.load(open(os.path.join(REPO, "paper", "results",
                                      "metrics.json")))["headline"]["threshold"]

    rows, t_start, n_done = [], time.time(), 0
    for c0 in range(0, len(picks), args.chunk):
        blk = picks[c0:c0 + args.chunk]
        ids = sorted(int(eid_arr[j]) for j in blk)
        j_of = {int(eid_arr[j]): j for j in blk}
        try:
            # ONE query per block.  Per-event queries cost 3-7 s each because each pays the same
            # row-group seek independently; a 250-id block costs 12 s total, i.e. 0.05 s/event.
            q = con.execute(
                f"SELECT event_id, epoch_id, filt, flux_uJy FROM read_parquet('{OBS}') "
                f"WHERE event_id BETWEEN {ids[0]} AND {ids[-1]} "
                f"AND event_id IN ({','.join(map(str, ids))}) AND saturation_flag = 0"
            ).fetchnumpy()
        except Exception as exc:
            for e in ids:
                rows.append({"event_id": e, "error": str(exc)[:200]})
            n_done += len(ids)
            continue
        qe = np.asarray(q["event_id"], np.int64)
        srt = np.argsort(qe, kind="stable")
        qe = qe[srt]
        q_ep = np.asarray(q["epoch_id"], np.int64)[srt]
        q_ft = np.asarray(q["filt"])[srt]
        q_fl = np.asarray(q["flux_uJy"], float)[srt]
        lo_i = np.searchsorted(qe, ids, "left")
        hi_i = np.searchsorted(qe, ids, "right")

        for k, eid in enumerate(ids):
            n_done += 1
            j = j_of[eid]
            seas = season_of(float(m["t0lens1"][j]))
            if seas is None:
                # t0 falls between seasons: the peak is never observed, so there is no season in
                # which this event is the event BinML would be asked about.
                rows.append({"event_id": eid, "sim_label": m["sim_label"][j],
                             "skipped": "t0 falls in an inter-season gap"})
                continue
            a, b_ = int(lo_i[k]), int(hi_i[k])
            if a == b_:
                rows.append({"event_id": eid, "sim_label": m["sim_label"][j],
                             "skipped": "no unsaturated photometry returned"})
                continue
            lo_t, hi_t = seas[0], min(seas[1], seas[0] + WINDOW_D)
            e_ep, e_ft, e_fl = q_ep[a:b_], q_ft[a:b_], q_fl[a:b_]
            q_bjd = bjd_lookup(e_ep)
            in_win = (q_bjd >= lo_t) & (q_bjd <= hi_t)
            bands = {}
            for bd in ("F146", "F087", "F213"):
                sb = in_win & (e_ft == bd)
                if not sb.any():
                    continue
                t = q_bjd[sb] - lo_t
                mag = flux_to_ab(e_fl[sb])
                g = np.isfinite(mag)
                if g.sum() >= 10:
                    ordt = np.argsort(t[g])
                    bands[bd] = (t[g][ordt], mag[g][ordt])
            if "F146" not in bands:
                rows.append({"event_id": eid, "sim_label": m["sim_label"][j],
                             "skipped": "no usable F146 in window"})
                continue
            # BASELINE FROM THE DATA, not the catalogue.  Source_F146 + 2.5 log10(fs) should be
            # the unmagnified total, but in this release it is uniformly 0.471 mag brighter than
            # the observed quiescent flux (measured on both flux_uJy and the noiseless
            # true_flux_uJy, across the whole id range).  BinML is sensitive to the baseline it
            # is handed -- a wrong one turns PSPL into LongPeriodVar -- so we measure it as the
            # median F146 magnitude more than 5 t_E from the peak over the FULL mission, which
            # is what a survey pipeline would have anyway.  The catalogue value is kept for the
            # record.
            m_cat = float(m["Source_F146"][j] + 2.5 * np.log10(max(m["fs_F146"][j], 1e-6)))
            f146_all = (e_ft == "F146")
            mag_all = flux_to_ab(e_fl[f146_all])
            off = (np.abs(q_bjd[f146_all] - float(m["t0lens1"][j])) > 5.0 * float(m["tE_ref"][j])) \
                & np.isfinite(mag_all)
            if off.sum() < 200:
                rows.append({"event_id": eid, "sim_label": m["sim_label"][j],
                             "skipped": "fewer than 200 off-event F146 epochs for a baseline"})
                continue
            m_base = float(np.median(mag_all[off]))
            p = clf.predict(bands, m_base_ref=m_base, t_start=0.0)
            if os.environ.get("GT_DUMP") and str(eid) in os.environ["GT_DUMP"].split(","):
                np.savez(f"/tmp/gt_dump_{eid}.npz", m_base=m_base, pred=p.label,
                         **{f"{b}_t": v[0] for b, v in bands.items()},
                         **{f"{b}_m": v[1] for b, v in bands.items()})
            rows.append({
                "event_id": eid, "sim_label": m["sim_label"][j], "pred": p.label,
                "p_nonpspl": round(p.probabilities["NonPSPL"], 4),
                "probs": {kk: round(v, 4) for kk, v in p.probabilities.items()},
                "n_f146": int(len(bands["F146"][0])), "bands": sorted(bands),
                "dense": bool(len(bands["F146"][0]) >= DENSE_MIN),
                "m_base": round(m_base, 3), "m_base_catalogue": round(m_cat, 3),
                "tE": float(m["tE_ref"][j]),
                "u0": float(m["u0lens1"][j]), "planet_q": m["Planet_q"][j],
                "source_q": m["Source_q"][j], "source_is_binary": m["Source_Is_Binary"][j],
                "weight": float(m["final_weight"][j])})
        print(f"  {n_done}/{len(picks)}  ({time.time() - t_start:.0f}s)", flush=True)

    ok = [r for r in rows if "pred" in r]
    summary = {"_doc": __doc__.split("\n")[0],
               "status": "NOT A TRANSFER MEASUREMENT for the shipped checkpoint: the input contract "
                         "(continuous F146) is violated by Roman's real schedule; see module docstring",
               "dataset": "RGES-PIT/MachineLearning (RMDC26, GULLS simulator)",
               "checkpoint": "binml/weights/binml.pt (shipped)",
               "dataset_revision": REVISION,
               "baseline": "empirical: median F146 mag at |t-t0| > 5 tE over the full mission",
               "window_days": WINDOW_D, "dense_min_f146_epochs": DENSE_MIN, "threshold": thr,
               "min_peak_amplitude_mag": args.min_amp,
               "selection_note": ("events selected on a metadata-derived peak amplitude, applied "
                                  "identically to all three classes; GULLS' raw draw is dominated "
                                  "by sub-0.1 mag events that BinML's detectability-conditioned "
                                  "classes never contain"),
               "id_windows": windows, "te_support_days": [args.min_te, args.max_te],
               "te_support_note": ("BinML's training t_E prior is truncated to "
                   "[1, 300] d; RMDC26 1S1L is 33.4% sub-day and its binary classes "
                   "are 0.2%, so an uncut comparison conflates timescale with lens "
                   "multiplicity"),
               "te_excluded": te_excluded, "n_requested": len(picks), "n_classified": len(ok),
               "n_skipped": len(rows) - len(ok),
               "n_dense": sum(1 for r in ok if r["dense"]),
               "n_sparse_lowcadence": sum(1 for r in ok if not r["dense"]),
               "by_class": {}}
    for lab in sorted(LABELS):
        sub = [r for r in ok if r["sim_label"] == lab and r["dense"]]
        if not sub:
            continue
        pn = np.array([r["p_nonpspl"] for r in sub])
        block = {"description": LABELS[lab], "n_dense": len(sub),
                 "frac_over_threshold": round(float((pn >= thr).mean()), 4),
                 "frac_argmax_nonpspl": round(
                     float(np.mean([r["pred"] == "NonPSPL" for r in sub])), 4),
                 "median_p_nonpspl": round(float(np.median(pn)), 4),
                 "argmax_distribution": {c: round(
                     float(np.mean([r["pred"] == c for r in sub])), 4) for c in CLASSES}}
        # Split 2S2L on whether the source is genuinely binary (56% of the class).  Both halves
        # carry a planetary lens, so this contrasts planet-plus-binary-source against plain
        # planet; it does NOT isolate a binary-source false positive.
        if lab == "RMDC26_2S2L_ML":
            tb = [r for r in sub if (r.get("source_is_binary") or 0) > 0]
            if tb:
                p2 = np.array([r["p_nonpspl"] for r in tb])
                block["true_binary_source_only"] = {
                    "n": len(tb),
                    "frac_over_threshold": round(float((p2 >= thr).mean()), 4),
                    "frac_argmax_nonpspl": round(
                        float(np.mean([r["pred"] == "NonPSPL" for r in tb])), 4),
                    "median_p_nonpspl": round(float(np.median(p2)), 4)}
        sp = [r for r in ok if r["sim_label"] == lab and not r["dense"]]
        if sp:
            block["sparse_lowcadence_window"] = {
                "n": len(sp),
                "frac_argmax_nonpspl": round(
                    float(np.mean([r["pred"] == "NonPSPL" for r in sp])), 4),
                "argmax_distribution": {c: round(
                    float(np.mean([r["pred"] == c for r in sp])), 4) for c in CLASSES},
                "note": "window landed in a low-cadence season; BinML is out of distribution and "
                        "the paper documents the sparse-cadence failure separately"}
        summary["by_class"][lab] = block

    json.dump(summary, open(args.out, "w"), indent=2)
    json.dump(rows, open(args.rows_out, "w"), indent=1)
    print(json.dumps(summary, indent=2))
    print(f"-> {args.out}\n-> {args.rows_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

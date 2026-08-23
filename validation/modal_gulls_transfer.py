"""Cross-simulator transfer test on Modal: BinML vs RMDC26 (GULLS).

WHY THIS SHAPE.  Two earlier designs failed for the same reason, and the fix is worth recording.

  1. Per-event remote queries (locally, concurrency 1): correct but slow -- 4.3 s/event, of which
     the forward pass is 3 ms.  The cost is entirely HTTP round-trips against a 172 GB parquet.
  2. The same per-event queries fanned out over 24 Modal containers: Hugging Face returned
     HTTP 401 on 5,183 of 6,000 events.  Anonymous access is throttled hard from datacenter IPs,
     so the fan-out that justified moving to Modal is precisely what broke it.  Lowering the
     concurrency to 6 with backoff only made it slow instead of failing.

  What works: STAGE ONCE, THEN COMPUTE.  A single container streams the remote parquet in one
  sequential pass, filtering to the sampled events, and writes a small subset to a Modal Volume.
  Every worker then reads local disk.  One big streaming read is what object storage is built to
  serve; thousands of small range requests is what it throttles.

The science -- season windowing, flux->AB conversion, amplitude-matched selection -- is identical
to validation/gulls_transfer.py, which stays the readable reference implementation.

Run:   modal run --detach validation/modal_gulls_transfer.py
       Do NOT wrap in `timeout`: SIGTERM reaches the client and cancels detached containers.
Fetch: modal volume get binml-gulls result.json
"""
import json
import os

import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# duckdb MUST be >= ~1.5: Hugging Face serves files this large through its Xet backend
# (cas-bridge.xethub.hf.co) via a 302, and duckdb 1.1.3's httpfs cannot range-request against it
# -- it returns HTTP 401, which reads exactly like throttling and is not. That misdiagnosis cost
# two full runs: the laptop had 1.5.5 and worked, the pinned image had 1.1.3 and did not.
DEPS = ["torch==2.5.1", "numpy==1.26.4", "duckdb==1.5.5", "pyarrow==18.1.0"]

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install(*DEPS)
         .add_local_dir(os.path.join(REPO, "pipeline"), "/repo/pipeline", copy=True)
         .add_local_dir(os.path.join(REPO, "binml"), "/repo/binml", copy=True))
app = modal.App("binml-gulls")
vol = modal.Volume.from_name("binml-gulls", create_if_missing=True)
VOL = "/data"

BASE = "https://huggingface.co/datasets/RGES-PIT/MachineLearning/resolve/main/"
OBS = BASE + "RMDC26_ML_Data_obs.parquet"
EPOCH = BASE + "RMDC26_ML_Data_epoch.parquet"
META = BASE + "RMDC26_ML_Data_meta.parquet"
WINDOW_D, DENSE_MIN, SEASON_GAP_D = 72.0, 1000, 5.0
MIN_AMP = 0.1
THRESHOLD = 0.9042405486106873
CLASSES = ["Flat", "PSPL", "NonPSPL", "PeriodicVar", "LongPeriodVar", "Eruptive"]


@app.function(image=image, timeout=1800, volumes={VOL: vol})
def prep_epoch() -> dict:
    """Fetch the 4 MB epoch map ONCE into the Volume.

    Twenty containers each calling urlretrieve on it at start-up earned an honest HTTP 429 --
    the big obs reads were never the problem, this small file was. One fetch, then every worker
    reads local disk.
    """
    import time
    import urllib.request
    dest = f"{VOL}/epoch.parquet"
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        return {"cached": True, "bytes": os.path.getsize(dest)}
    # Download STRAIGHT to the Volume. os.replace() from /tmp fails with EXDEV: the Volume is a
    # different device, and the retry loop turned that into an opaque repeated failure.
    last = None
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(EPOCH, dest)
            vol.commit()
            # prove it is readable before declaring success, so a truncated or HTML error page
            # cannot pass as a staged file
            import pyarrow.parquet as pq
            n_rows = pq.ParquetFile(dest).metadata.num_rows
            return {"cached": False, "bytes": os.path.getsize(dest), "rows": n_rows}
        except Exception as exc:
            last = str(exc)
            if os.path.exists(dest):
                try:
                    os.remove(dest)
                except Exception:
                    pass
            time.sleep(5 * 2 ** attempt)
    return {"error": last}


@app.function(image=image, cpu=8.0, timeout=21600, volumes={VOL: vol})
def stage(event_ids: list, tag: str) -> dict:
    """ONE sequential pass over the remote 172 GB table, writing only the sampled events."""
    import time
    import duckdb
    dest = f"{VOL}/subset_{tag}.parquet"
    if os.path.exists(dest):
        return {"cached": True, "path": dest, "bytes": os.path.getsize(dest)}
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET preserve_insertion_order=false;")
    ids = ",".join(map(str, event_ids))
    t0 = time.time()
    try:
        con.execute(f"""COPY (SELECT event_id, epoch_id, filt, flux_uJy
                              FROM read_parquet('{OBS}')
                              WHERE event_id IN ({ids}) AND saturation_flag = 0)
                        TO '{dest}' (FORMAT parquet, COMPRESSION zstd)""")
    except Exception as exc:
        return {"error": str(exc)[:300]}
    # the epoch map is 4 MB; stage it too so workers never touch the network
    ep = f"{VOL}/epoch.parquet"
    if not os.path.exists(ep):
        import urllib.request
        urllib.request.urlretrieve(EPOCH, ep)
    vol.commit()
    import pyarrow.parquet as pq
    md = pq.ParquetFile(dest).metadata
    return {"cached": False, "path": dest, "bytes": os.path.getsize(dest),
            "rows": md.num_rows, "seconds": round(time.time() - t0, 1)}


@app.function(image=image, cpu=2.0, timeout=3600, volumes={VOL: vol}, max_containers=20)
def shard(events: list, tag: str) -> list:
    """Classify one shard, reading the staged subset from the Volume. No network."""
    import sys
    import warnings
    import duckdb
    import numpy as np
    import pyarrow.parquet as pq
    os.chdir("/repo"); sys.path.insert(0, "/repo"); warnings.simplefilter("ignore")
    import binml
    vol.reload()

    # staged by prep_epoch() before the fan-out; no worker fetches it over the network
    ep = pq.read_table(f"{VOL}/epoch.parquet", columns=["epoch_id", "bjd"]).to_pydict()
    ei = np.asarray(ep["epoch_id"], np.int64); eb = np.asarray(ep["bjd"], float)
    o = np.argsort(eb); ei, eb = ei[o], eb[o]
    bjd_of = dict(zip(ei.tolist(), eb.tolist()))
    # epoch_id is NOT ordered by time -- the first 72 d span the whole id range -- so windows are
    # cut on BJD, never on an epoch_id range.
    gaps = np.flatnonzero(np.diff(eb) > SEASON_GAP_D)
    seasons = list(zip(np.r_[eb[0], eb[gaps + 1]], np.r_[eb[gaps], eb[-1]]))

    # Read the STAGED subset from the Volume. Twenty containers querying the remote 172 GB file
    # concurrently earned HTTP 429 -- one big sequential read is what object storage serves well,
    # thousands of concurrent range requests is what it throttles. Staging is not an optimisation
    # here, it is the only shape that works at this fan-out.
    con = duckdb.connect()
    src = f"{VOL}/subset_{tag}.parquet"
    clf = binml.Classifier()
    out = []
    for ev in events:
        eid, t0 = ev["event_id"], ev["t0"]
        seas = next(((a, b) for a, b in seasons if a <= t0 <= b), None)
        if seas is None:
            out.append({**ev, "skipped": "t0 in an inter-season gap"}); continue
        lo_t, hi_t = seas[0], min(seas[1], seas[0] + WINDOW_D)
        q = con.execute(f"SELECT epoch_id, filt, flux_uJy FROM read_parquet('{src}') "
                        f"WHERE event_id = {eid}").fetchnumpy()
        if not len(q["epoch_id"]):
            out.append({**ev, "skipped": "event absent from the staged subset"}); continue
        qb = np.array([bjd_of.get(int(e), np.nan) for e in q["epoch_id"]], float)
        win = (qb >= lo_t) & (qb <= hi_t)
        bands = {}
        for b in ("F146", "F087", "F213"):
            sb = win & (q["filt"] == b)
            if not sb.any():
                continue
            f = q["flux_uJy"][sb].astype(float)
            mag = np.where(f > 0, -2.5 * np.log10(np.maximum(f, 1e-12)) + 23.9, np.nan)
            g = np.isfinite(mag)
            if g.sum() >= 10:
                t = qb[sb][g] - lo_t
                s = np.argsort(t)
                bands[b] = (t[s], mag[g][s])
        if "F146" not in bands:
            out.append({**ev, "skipped": "no usable F146 in the season"}); continue
        p = clf.predict(bands, m_base_ref=ev["m_base"], t_start=0.0)
        out.append({**ev, "pred": p.label,
                    "p_nonpspl": round(p.probabilities["NonPSPL"], 4),
                    "probs": {k: round(v, 4) for k, v in p.probabilities.items()},
                    "n_f146": int(len(bands["F146"][0])), "bands": sorted(bands),
                    "dense": bool(len(bands["F146"][0]) >= DENSE_MIN)})
    return out


@app.local_entrypoint()
def main(per_class: int = 2000, seed: int = 20260817, shard_size: int = 150, tag: str = "v1"):
    import numpy as np
    import pyarrow.parquet as pq
    import urllib.request

    cache = "/tmp/rmdc26_meta.parquet"
    if not os.path.exists(cache):
        print("[meta] downloading", flush=True)
        urllib.request.urlretrieve(META, cache)
    m = pq.read_table(cache, columns=[
        "event_id", "sim_label", "t0lens1", "tE_ref", "u0lens1", "Source_F146", "fs_F146",
        "Planet_q", "Source_q", "Source_Is_Binary", "final_weight"]).to_pydict()

    # Peak observed amplitude from metadata alone. GULLS simulates the whole population,
    # including events no survey would register (median 1S1L amplitude 0.063 mag); BinML's
    # classes are detectability-conditioned. The cut is on the INPUT population, identical
    # across classes, never on the model's output.
    u0 = np.abs(np.asarray(m["u0lens1"], float)); fs = np.asarray(m["fs_F146"], float)
    A = (u0 ** 2 + 2) / np.maximum(u0 * np.sqrt(u0 ** 2 + 4), 1e-12)
    amp = np.abs(-2.5 * np.log10(np.maximum(1 + fs * (A - 1), 1e-12)))

    rng = np.random.default_rng(seed)
    picks = []
    for lab in sorted(set(m["sim_label"])):
        idx = [i for i, L in enumerate(m["sim_label"]) if L == lab and amp[i] >= MIN_AMP]
        sel = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        print(f"[sample] {lab}: {len(sel)} of {len(idx)} above {MIN_AMP} mag", flush=True)
        for j in sel:
            j = int(j)
            picks.append({
                "event_id": int(m["event_id"][j]), "sim_label": m["sim_label"][j],
                "t0": float(m["t0lens1"][j]), "tE": float(m["tE_ref"][j]),
                "u0": float(m["u0lens1"][j]), "peak_amp": round(float(amp[j]), 4),
                "m_base": float(m["Source_F146"][j] + 2.5 * np.log10(max(m["fs_F146"][j], 1e-6))),
                "planet_q": m["Planet_q"][j], "source_q": m["Source_q"][j],
                "source_is_binary": m["Source_Is_Binary"][j],
                "weight": float(m["final_weight"][j])})

    print("[prep] staging the epoch map ...", flush=True)
    st = {"epoch": prep_epoch.remote()}
    print(f"[prep] {st['epoch']}", flush=True)
    if "error" in st["epoch"]:
        raise SystemExit(f"epoch staging failed, refusing to fan out: {st['epoch']['error']}")
    print(f"[stage] one sequential pass for {len(picks)} events ...", flush=True)
    st["obs"] = stage.remote([p["event_id"] for p in picks], tag)
    print(f"[stage] {st['obs']}", flush=True)
    if "error" in st["obs"]:
        raise SystemExit(f"subset staging failed: {st['obs']['error']}")
    shards = [picks[i:i + shard_size] for i in range(0, len(picks), shard_size)]
    print(f"[run] {len(shards)} shards over local Volume reads", flush=True)
    rows = [r for chunk in shard.starmap([(s, tag) for s in shards]) for r in chunk]

    ok = [r for r in rows if "pred" in r]
    summary = {"dataset": "RGES-PIT/MachineLearning (RMDC26, GULLS)",
               "checkpoint": "binml/weights/binml.pt (shipped)",
               "window": "the observing season containing t0, capped at 72 d",
               "min_peak_amplitude_mag": MIN_AMP, "threshold": THRESHOLD,
               "staged": st, "n_requested": len(picks), "n_classified": len(ok),
               "n_skipped": len(rows) - len(ok),
               "n_dense": sum(1 for r in ok if r["dense"]), "by_class": {}}
    for lab in sorted(set(r["sim_label"] for r in ok)):
        sub = [r for r in ok if r["sim_label"] == lab and r["dense"]]
        if not sub:
            continue
        pn = np.array([r["p_nonpspl"] for r in sub])
        blk = {"n_dense": len(sub),
               "frac_over_threshold": round(float((pn >= THRESHOLD).mean()), 4),
               "frac_argmax_nonpspl": round(
                   float(np.mean([r["pred"] == "NonPSPL" for r in sub])), 4),
               "median_p_nonpspl": round(float(np.median(pn)), 4),
               "argmax_distribution": {c: round(
                   float(np.mean([r["pred"] == c for r in sub])), 4) for c in CLASSES}}
        if lab == "RMDC26_2S2L_ML":
            tb = [r for r in sub if (r.get("source_is_binary") or 0) > 0]
            if tb:
                p2 = np.array([r["p_nonpspl"] for r in tb])
                blk["true_binary_source_only"] = {
                    "n": len(tb),
                    "frac_over_threshold": round(float((p2 >= THRESHOLD).mean()), 4),
                    "frac_argmax_nonpspl": round(
                        float(np.mean([r["pred"] == "NonPSPL" for r in tb])), 4)}
        summary["by_class"][lab] = blk

    json.dump(summary, open("/tmp/gulls_result.json", "w"), indent=2)
    json.dump(rows, open("/tmp/gulls_rows.json", "w"), indent=1)
    print(json.dumps(summary, indent=2))
    print("-> /tmp/gulls_result.json  /tmp/gulls_rows.json")

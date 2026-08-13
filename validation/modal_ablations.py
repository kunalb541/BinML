"""The two ablations that substantiate the title's claims (referee Major issue 2).

A paper claiming "detectability-conditioned" and "real-time" must show that each mechanism does
something. Both are cheap by the paper's own accounting, so both are run here:

  A. LABELLING ablation -- identical data, identical recipe, only the training LABELS differ:
     observational (detectability-conditioned, shipped) vs generator-intent (true_class).
     Question: does conditioning labels on observability produce a better classifier, or merely a
     differently-bookkept one?

  B. CASCADE ablation -- identical data, identical recipe, only --truncate-aug differs (0.5 vs 0).
     Question: does the truncation augmentation CAUSE the real-time ordering, or does the model get
     it anyway?

Protocol (addressing the audit's critique of the earlier attempt):
  * every arm trains from scratch on the SAME shards with the SAME hyper-parameters and seed;
  * arms are evaluated on a common held-out set generated from an UNSEEN seed base;
  * both the frozen operating threshold AND true six-class argmax are reported, separately;
  * event-level outputs are returned, not just summaries, with Wilson intervals;
  * full-season cost is measured too (per-class F1 + anomaly completeness), so a cascade gain that
    is paid for in static accuracy cannot hide;
  * checkpoints/configs stay in a Modal Volume rather than /tmp.

A note on what makes the labelling comparison fair. The generator-intent arm is TRAINED on
true_class but is EVALUATED, like every other arm, against the observational label. Scoring each
arm against its own training target would not be a comparison at all -- it would only show that a
model fits whatever it was given. The observational label is the right common yardstick because it
is the operational task: an anomaly nobody can see is, for a follow-up programme, not an anomaly.
The two label sets disagree on 39.5% of events, so this choice is not cosmetic; it is the whole
experiment. If the generator-intent arm wins anyway, detectability conditioning is not earning its
place in the title.

Run:  modal run validation/modal_ablations.py
"""
import os
import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "numpy", "scipy", "h5py", "VBBinaryLensing")
         .add_local_dir(os.path.join(REPO, "pipeline"), "/repo/pipeline", copy=True)
         .add_local_dir(os.path.join(REPO, "binml"), "/repo/binml", copy=True))
app = modal.App("binml-ablations")
vol = modal.Volume.from_name("binml-ablation-data", create_if_missing=True)
VOL = "/data"
CLASSES = ["Flat", "PSPL", "NonPSPL", "PeriodicVar", "LongPeriodVar", "Eruptive"]
SEED = 20260720


@app.function(image=image, cpu=16.0, timeout=5400, volumes={VOL: vol})
def gen(shard: int, seed_base: int, split: str) -> str:
    import subprocess, sys, os, glob, shutil
    os.chdir("/repo"); sys.path.insert(0, "/repo")
    d = f"{VOL}/{split}"; os.makedirs(d, exist_ok=True)
    dest = f"{d}/shard_{shard:05d}.h5"
    if os.path.exists(dest):
        return dest
    W = f"/tmp/g_{split}_{shard}"; os.makedirs(f"{W}/raw", exist_ok=True)
    subprocess.run(["python", "-m", "pipeline.run_shard", "--shard", str(shard),
                    "--n-shards", "200", "--out", f"{W}/raw", "--seed-base", str(seed_base)],
                   check=True, env=dict(os.environ, PYTHONPATH="/repo"), cwd="/repo")
    from pipeline.cache import build_cache
    build_cache([sorted(glob.glob(f"{W}/raw/*.h5"))[0]], f"{W}/c.h5")
    shutil.move(f"{W}/c.h5", dest); vol.commit()
    shutil.rmtree(W, ignore_errors=True)
    return dest


@app.function(image=image, gpu="T4", cpu=8.0, timeout=10800, volumes={VOL: vol})
def run_ablations(epochs: int = 10) -> dict:
    import subprocess, sys, os, json, glob, shutil, warnings
    import numpy as np, torch
    os.chdir("/repo"); sys.path.insert(0, "/repo"); warnings.simplefilter("ignore")
    vol.reload()
    from pipeline.model import BAND_BINS, BinMLv5, ModelConfigV5
    env = dict(os.environ, PYTHONPATH="/repo")

    def mm(split, tag):
        d = f"/tmp/{tag}"; os.makedirs(f"{d}/cache", exist_ok=True)
        for i, p in enumerate(sorted(glob.glob(f"{VOL}/{split}/*.h5"))):
            shutil.copy(p, f"{d}/cache/shard_{i:05d}.h5")
        subprocess.run(["python", "-m", "pipeline.to_memmap", "--in-dir", f"{d}/cache",
                        "--out", f"{d}/mm"], check=True, env=env, cwd="/repo")
        return f"{d}/mm"

    tr, ev = mm("train", "tr"), mm("eval", "ev")

    # ---- four arms: 2 ablations x 2 settings, everything else identical ----
    arms = {
        "labels_observational": ["--label-source", "observational", "--truncate-aug", "0.5"],
        "labels_generator":     ["--label-source", "generator",     "--truncate-aug", "0.5"],
        "cascade_on":           ["--label-source", "observational", "--truncate-aug", "0.5"],
        "cascade_off":          ["--label-source", "observational", "--truncate-aug", "0.0"],
    }
    ckpts = {}
    for tag, extra in arms.items():
        if tag == "cascade_on":                     # identical config to labels_observational
            ckpts[tag] = ckpts["labels_observational"]; continue
        out = f"{VOL}/ckpt_{tag}.pt"
        if not os.path.exists(out):
            subprocess.run(["python", "-m", "pipeline.train", "--cache", tr, "--out", out,
                            "--epochs", str(epochs), "--seed", str(SEED), "--device", "cuda"]
                           + extra, check=True, env=env, cwd="/repo")
            vol.commit()
        ckpts[tag] = out

    # ---- shared evaluation machinery ----
    meta = json.load(open(f"{ev}/meta.json")); n = meta["n_events"]
    arr = {b: (np.memmap(f"{ev}/feat_{b}.f16", np.float16, "r", shape=(n, L, 3)),
               np.memmap(f"{ev}/frac_{b}.f16", np.float16, "r", shape=(n, L)))
           for b, L in BAND_BINS.items()}
    y = np.load(f"{ev}/label.npy"); kp = np.load(f"{ev}/keep_prob.npy").astype(float)
    w = 1.0 / np.clip(kp, 1e-3, 1.0); NON = 2
    thr = 0.9042405486106873

    def load(p):
        ck = torch.load(p, map_location="cpu")
        cfg = ModelConfigV5(**{k: v for k, v in ck["config"].items()
                               if k in ModelConfigV5.__dataclass_fields__})
        net = BinMLv5(cfg).cuda(); net.load_state_dict(ck["model"]); net.eval(); return net

    def probs(net):
        P = np.zeros((n, 6), np.float32)
        for s in range(0, n, 2048):
            sl = slice(s, min(s + 2048, n)); feats, pres = {}, {}
            for b, L in BAND_BINS.items():
                x = np.asarray(arr[b][0][sl], np.float32); fr = np.asarray(arr[b][1][sl], np.float32)
                obs = np.isfinite(x[:, :, 0]).astype(np.float32)
                t = np.concatenate([np.nan_to_num(x, nan=0.0), fr[:, :, None], obs[:, :, None]], 2)
                feats[b] = torch.tensor(t).cuda(); pres[b] = feats[b][..., 4].sum(1) > 0
            with torch.inference_mode():
                lg = net(feats, pres).float().cpu().numpy()
            z = lg - lg.max(1, keepdims=True); e = np.exp(z); P[sl] = e / e.sum(1, keepdims=True)
        return P

    def wilson(k, m, z=1.96):
        if m == 0: return (0.0, 0.0)
        p = k / m; d = 1 + z * z / m
        c = (p + z * z / (2 * m)) / d
        h = z * np.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / d
        return (max(0.0, c - h), min(1.0, c + h))

    def static_metrics(P):
        pred = P.argmax(1); out = {"per_class_f1": {}}
        for c, name in enumerate(CLASSES):
            tp = (w * ((pred == c) & (y == c))).sum(); fp = (w * ((pred == c) & (y != c))).sum()
            fn = (w * ((pred != c) & (y == c))).sum()
            r = tp / max(tp + fn, 1e-9); p_ = tp / max(tp + fp, 1e-9)
            out["per_class_f1"][name] = round(float(2 * r * p_ / max(r + p_, 1e-9)), 3)
        sel = P[:, NON] >= thr; yb = y == NON
        tp = (w * (sel & yb)).sum(); fp = (w * (sel & ~yb)).sum(); fn = (w * (~sel & yb)).sum()
        out["anomaly_completeness_at_thr"] = round(float(tp / max(tp + fn, 1e-9)), 3)
        out["anomaly_purity_at_thr"] = round(float(tp / max(tp + fp, 1e-9)), 3)
        out["macro_f1"] = round(float(np.mean(list(out["per_class_f1"].values()))), 3)
        return out

    # ---- real-time (cascade) evaluation: event-level first crossing, both decision rules ----
    import binml
    from pipeline.assemble import simulate_event, SurveyConfig
    cfg = SurveyConfig()
    evs, s = [], 770000
    while len(evs) < 300 and s < 770000 + 60 * 300:
        s += 1
        e = simulate_event("NonPSPL", np.random.default_rng(s), cfg)
        if e and e.label == "NonPSPL":
            ta = e.params.get("t_anom")
            if ta is not None and np.isfinite(ta):
                evs.append(e)

    def realtime(ckpt_path):
        clf = binml.Classifier(weights=ckpt_path, device="cuda")
        rows = []
        for e in evs:
            ta = float(e.params["t_anom"]); b = e.bands["F146"]; mb = e.params["_m_base_ref"]
            fc_thr = fc_arg = None
            for cut in np.arange(0.5, cfg.window_days + 1e-9, 0.5):
                m = b.t <= cut
                if m.sum() < 10: continue
                p = clf.predict(b.t[m], b.mag[m], m_base_ref=mb, t_start=0.0)
                pn = p.probabilities["NonPSPL"]
                if fc_thr is None and pn >= thr: fc_thr = float(cut)
                if fc_arg is None and p.label == "NonPSPL": fc_arg = float(cut)
                if fc_thr is not None and fc_arg is not None: break
            rows.append({"t_anom": round(ta, 2), "first_thr": fc_thr, "first_argmax": fc_arg,
                         "premature_thr": bool(fc_thr is not None and fc_thr < ta),
                         "premature_argmax": bool(fc_arg is not None and fc_arg < ta)})
        N = len(rows); out = {"n_eligible": N, "events": rows}
        for rule in ("thr", "argmax"):
            k = sum(r[f"premature_{rule}"] for r in rows)
            d = sum(r[f"first_{rule}"] is not None for r in rows)
            lo, hi = wilson(k, N)
            out[f"premature_rate_{rule}"] = round(k / max(N, 1), 3)
            out[f"premature_ci_{rule}"] = [round(lo, 3), round(hi, 3)]
            out[f"detection_fraction_{rule}"] = round(d / max(N, 1), 3)
        return out

    res = {"n_eval_static": int(n), "epochs": epochs, "seed": SEED, "threshold": thr}
    # NOTE: results are written to the VOLUME, not just returned. A detached run (required, since a
    # backgrounded local client disconnects and Modal then stops the app) has no live client to
    # receive the return value.
    for tag, path in ckpts.items():
        res[tag] = {"static": static_metrics(probs(load(path)))}
    # real-time only matters for the cascade ablation
    for tag in ("cascade_on", "cascade_off"):
        res[tag]["realtime"] = realtime(ckpts[tag])
    with open(f"{VOL}/ablations_full.json", "w") as f:
        json.dump(res, f, indent=2)
    vol.commit()
    return res


@app.local_entrypoint()
def main(n_train: int = 16, n_eval: int = 6):
    """Run DETACHED: `modal run --detach validation/modal_ablations.py`. Results land in the
    volume at /data/ablations_full.json; retrieve with
    `modal volume get binml-ablation-data ablations_full.json`."""
    import json
    args = [(i, SEED, "train") for i in range(n_train)] + \
           [(i, 990000, "eval") for i in range(n_eval)]
    print(f"[gen] {len(args)} shards -> volume")
    list(gen.starmap(args))
    print("[run] training 3 arms + evaluating (static and real-time)...")
    res = run_ablations.remote()
    # keep the summary small locally; event rows stay inside the result
    summary = {k: (v if not isinstance(v, dict) else
                   {kk: (vv if kk != "realtime" else
                         {k2: v2 for k2, v2 in vv.items() if k2 != "events"})
                    for kk, vv in v.items()})
               for k, v in res.items()}
    print(json.dumps(summary, indent=2))
    json.dump(res, open("/tmp/ablations_full.json", "w"), indent=2)
    json.dump(summary, open("/tmp/ablations_summary.json", "w"), indent=2)

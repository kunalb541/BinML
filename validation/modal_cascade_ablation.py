"""Train the NO-CASCADE arm so the cascade before/after comparison is reproducible.

Audit finding: the manuscript's cascade numbers (42%->9%) had no tracked artifact. The shipped
(cascade) arm is now reproduced by validation/cascade_reproduce.py. The 'before' arm requires a
model trained WITHOUT truncation augmentation, which is a separate training run -- this is it.

Both arms train on identical data with an identical recipe; only --truncate-aug differs (0 vs 0.5).
Each is then measured with the SAME pre-onset protocol used for the shipped model.

Run:  modal run validation/modal_cascade_ablation.py
"""
import os
import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "numpy", "scipy", "h5py", "VBBinaryLensing")
         .add_local_dir(os.path.join(REPO, "pipeline"), "/repo/pipeline", copy=True)
         .add_local_dir(os.path.join(REPO, "binml"), "/repo/binml", copy=True))
app = modal.App("binml-cascade-ablation")
vol = modal.Volume.from_name("binml-cascade-data", create_if_missing=True)
VOL = "/data"


@app.function(image=image, cpu=16.0, timeout=5400, volumes={VOL: vol})
def gen(shard: int) -> str:
    import subprocess, sys, os, glob, shutil
    os.chdir("/repo"); sys.path.insert(0, "/repo")
    os.makedirs(f"{VOL}/train", exist_ok=True)
    dest = f"{VOL}/train/shard_{shard:05d}.h5"
    if os.path.exists(dest):
        return dest
    W = f"/tmp/g{shard}"; os.makedirs(f"{W}/raw", exist_ok=True)
    subprocess.run(["python", "-m", "pipeline.run_shard", "--shard", str(shard),
                    "--n-shards", "200", "--out", f"{W}/raw"], check=True,
                   env=dict(os.environ, PYTHONPATH="/repo"), cwd="/repo")
    from pipeline.cache import build_cache
    build_cache([sorted(glob.glob(f"{W}/raw/*.h5"))[0]], f"{W}/c.h5")
    shutil.move(f"{W}/c.h5", dest); vol.commit()
    return dest


@app.function(image=image, gpu="T4", cpu=8.0, timeout=7200, volumes={VOL: vol})
def train_both(epochs: int = 10) -> dict:
    """Train cascade and no-cascade arms on identical data, then measure both pre-onset."""
    import subprocess, sys, os, json, glob, shutil, warnings
    import numpy as np, torch
    os.chdir("/repo"); sys.path.insert(0, "/repo"); warnings.simplefilter("ignore")
    vol.reload()
    os.makedirs("/tmp/tr/cache", exist_ok=True)
    for i, p in enumerate(sorted(glob.glob(f"{VOL}/train/*.h5"))):
        shutil.copy(p, f"/tmp/tr/cache/shard_{i:05d}.h5")
    env = dict(os.environ, PYTHONPATH="/repo")
    subprocess.run(["python", "-m", "pipeline.to_memmap", "--in-dir", "/tmp/tr/cache",
                    "--out", "/tmp/tr/mm"], check=True, env=env, cwd="/repo")
    ck = {}
    for tag, aug in (("cascade", "0.5"), ("nocascade", "0.0")):
        out = f"/tmp/{tag}.pt"
        subprocess.run(["python", "-m", "pipeline.train", "--cache", "/tmp/tr/mm", "--out", out,
                        "--epochs", str(epochs), "--truncate-aug", aug, "--seed", "20260720",
                        "--device", "cuda"], check=True, env=env, cwd="/repo")
        ck[tag] = out

    # measure premature flagging on fresh events, identical protocol for both arms
    import binml
    from pipeline.assemble import simulate_event, SurveyConfig
    from pipeline.model import BAND_BINS, BinMLv5, ModelConfigV5
    cfg = SurveyConfig()
    evs, s = [], 400000
    while len(evs) < 150 and s < 400000 + 60 * 150:
        s += 1
        ev = simulate_event("NonPSPL", np.random.default_rng(s), cfg)
        if ev and ev.label == "NonPSPL":
            ta = ev.params.get("t_anom")
            if ta is not None and np.isfinite(ta) and ta > 5:
                evs.append(ev)
    res = {}
    for tag, path in ck.items():
        clf = binml.Classifier(weights=path, device="cuda")
        ps = []
        for i, ev in enumerate(evs):
            ta = ev.params["t_anom"]; b = ev.bands["F146"]
            cut = float(np.random.default_rng(50_000 + i).uniform(3.0, ta))
            m = b.t <= cut
            if m.sum() < 10:
                continue
            p = clf.predict(b.t[m], b.mag[m], m_base_ref=ev.params["_m_base_ref"], t_start=0.0)
            ps.append(p.probabilities["NonPSPL"])
        ps = np.array(ps)
        res[tag] = {"n": len(ps),
                    "premature_flag_rate_argmax_proxy": round(float((ps >= 0.5).mean()), 3),
                    "mean_pre_onset_p": round(float(ps.mean()), 4),
                    "median_pre_onset_p": round(float(np.median(ps)), 4)}
    return res


@app.local_entrypoint()
def main(n_shards: int = 12):
    import json
    list(gen.map(range(n_shards)))
    res = train_both.remote()
    print("\n=== CASCADE ABLATION (identical data/recipe, only truncate-aug differs) ===")
    print(json.dumps(res, indent=2))
    json.dump(res, open("/tmp/cascade_ablation_result.json", "w"), indent=2)

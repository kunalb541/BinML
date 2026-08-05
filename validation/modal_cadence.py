"""P0.5 cadence comparison on Modal — two isolated containers, no local config-mixing.

Each container runs one arm end to end (generate -> cache -> memmap -> train 12 epochs -> eval) at
a fixed cadence: 15 min (production) or 12 min (current GBTDS rate). The 12-min container patches
the cadence config at runtime before importing the pipeline; containers are isolated so the two
cadences cannot mix. Matched recipe (same shard count, epochs, seed) isolates the cadence effect.

Run:  modal run validation/modal_cadence.py
"""
import os
import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy", "h5py", "VBBinaryLensing", "scikit-learn")
    .add_local_dir(os.path.join(REPO, "pipeline"), "/repo/pipeline", copy=True)
    .add_local_dir(os.path.join(REPO, "binml"), "/repo/binml", copy=True)
)

app = modal.App("binml-cadence")


@app.function(image=image, gpu="T4", cpu=8.0, timeout=3600)
def run_arm(cadence_min: int, n_shards: int = 12):
    import subprocess, sys, os, json, glob
    os.chdir("/repo"); sys.path.insert(0, "/repo")
    W = f"/tmp/work_{cadence_min}"
    for d in ("raw", "cache", "mm"):
        os.makedirs(f"{W}/{d}", exist_ok=True)

    if cadence_min == 12:   # patch config to current GBTDS F146 rate (12 min, factor 10)
        for f, a, b in [
            ("pipeline/photometry.py", "cadence_minutes=15.0", "cadence_minutes=12.0"),
            ("pipeline/cache.py", '"F146": 8, "F087": 3, "F213": 3', '"F146": 10, "F087": 3, "F213": 3'),
            ("binml/preprocess.py", '"F146": 8, "F087": 3, "F213": 3', '"F146": 10, "F087": 3, "F213": 3'),
            ("pipeline/assemble.py", "6912", "8640"),
        ]:
            s = open(f).read()
            assert a in s, f"pattern not found in {f}: {a}"
            open(f, "w").write(s.replace(a, b))

    env = dict(os.environ, PYTHONPATH="/repo")
    def run(cmd):
        subprocess.run(cmd, check=True, env=env, cwd="/repo")

    for i in range(n_shards):
        run(["python", "-m", "pipeline.run_shard", "--shard", str(i), "--n-shards", "200", "--out", f"{W}/raw"])
    from pipeline.cache import build_cache
    for h5 in sorted(glob.glob(f"{W}/raw/*.h5")):
        build_cache([h5], f"{W}/cache/" + os.path.basename(h5))
    run(["python", "-m", "pipeline.to_memmap", "--in-dir", f"{W}/cache", "--out", f"{W}/mm"])
    run(["python", "-m", "pipeline.train", "--cache", f"{W}/mm", "--out", f"{W}/model.pt",
         "--epochs", "12", "--truncate-aug", "0.5", "--seed", "20260720", "--device", "cuda"])
    run(["python", "-m", "pipeline.evaluate", "--ckpt", f"{W}/model.pt", "--cache", f"{W}/mm", "--out", f"{W}/eval"])

    m = json.load(open(f"{W}/eval/metrics.json"))
    return {
        "cadence_min": cadence_min,
        "n_events": m.get("n_events"),
        "completeness_at_purity": round(m["headline"]["completeness_at_fixed_purity"], 3),
        "purity": round(m["headline"]["purity_achieved"], 3),
        "ap": round(m["average_precision_population"], 3),
        "per_class_f1": {k: round(v, 3) for k, v in m["argmax_population"]["f1"].items()},
    }


@app.local_entrypoint()
def main():
    import json
    results = list(run_arm.starmap([(15, 12), (12, 12)]))
    print("\n=== CADENCE COMPARISON (matched 12-shard, 12-epoch, seed 20260720) ===")
    print(json.dumps(results, indent=2))

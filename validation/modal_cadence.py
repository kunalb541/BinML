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


CAD_PATCH = [
    ("pipeline/photometry.py", "cadence_minutes=15.0", "cadence_minutes=12.0"),
    ("pipeline/cache.py", '"F146": 8, "F087": 3, "F213": 3', '"F146": 10, "F087": 3, "F213": 3'),
    ("binml/preprocess.py", '"F146": 8, "F087": 3, "F213": 3', '"F146": 10, "F087": 3, "F213": 3'),
    ("pipeline/assemble.py", "6912", "8640"),
]


def _apply_cadence(cadence_min: int):
    """Patch the pipeline config in-container. Isolated per container, so cadences cannot mix."""
    if cadence_min != 12:
        return
    for f, a, b in CAD_PATCH:
        s = open(f).read()
        assert a in s, f"pattern not found in {f}: {a}"
        open(f, "w").write(s.replace(a, b))


@app.function(image=image, cpu=16.0, timeout=5400, max_containers=24)
def gen_shard(cadence_min: int, shard: int) -> bytes:
    """Generate ONE shard at the given cadence and return its binned cache bytes.

    Generation is CPU-bound (per-event scipy PSPL refits), so shards fan out across containers
    rather than looping inside one -- the earlier single-container version timed out here while
    an attached GPU sat idle.
    """
    import subprocess, sys, os, glob
    os.chdir("/repo"); sys.path.insert(0, "/repo")
    _apply_cadence(cadence_min)
    W = f"/tmp/s{cadence_min}_{shard}"
    os.makedirs(f"{W}/raw", exist_ok=True)
    subprocess.run(["python", "-m", "pipeline.run_shard", "--shard", str(shard),
                    "--n-shards", "200", "--out", f"{W}/raw"],
                   check=True, env=dict(os.environ, PYTHONPATH="/repo"), cwd="/repo")
    from pipeline.cache import build_cache
    raw = sorted(glob.glob(f"{W}/raw/*.h5"))[0]
    out = f"{W}/cache.h5"
    build_cache([raw], out)
    return open(out, "rb").read()


@app.function(image=image, gpu="T4", cpu=8.0, timeout=5400)
def train_eval(cadence_min: int, shards: list) -> dict:
    """Assemble the fanned-out cache shards, then memmap -> train -> evaluate on GPU."""
    import subprocess, sys, os, json
    os.chdir("/repo"); sys.path.insert(0, "/repo")
    _apply_cadence(cadence_min)
    W = f"/tmp/work_{cadence_min}"
    os.makedirs(f"{W}/cache", exist_ok=True); os.makedirs(f"{W}/mm", exist_ok=True)
    for i, blob in enumerate(shards):
        open(f"{W}/cache/shard_{i:05d}.h5", "wb").write(blob)
    env = dict(os.environ, PYTHONPATH="/repo")
    def run(cmd): subprocess.run(cmd, check=True, env=env, cwd="/repo")
    run(["python", "-m", "pipeline.to_memmap", "--in-dir", f"{W}/cache", "--out", f"{W}/mm"])
    run(["python", "-m", "pipeline.train", "--cache", f"{W}/mm", "--out", f"{W}/model.pt",
         "--epochs", "12", "--truncate-aug", "0.5", "--seed", "20260720", "--device", "cuda"])
    run(["python", "-m", "pipeline.evaluate", "--ckpt", f"{W}/model.pt", "--cache", f"{W}/mm",
         "--out", f"{W}/eval", "--device", "cuda"])   # evaluate.py defaults to mps (a Mac default)
    m = json.load(open(f"{W}/eval/metrics.json"))
    return {
        "cadence_min": cadence_min,
        "n_events": m.get("n_events"),
        "completeness_at_purity": round(m["headline"]["completeness_at_fixed_purity"], 3),
        "purity": round(m["headline"]["purity_achieved"], 3),
        "ap": round(m["average_precision_population"], 3),
        "per_class_f1": {k: round(v, 3) for k, v in m["argmax_population"]["f1"].items()},
    }


@app.function(image=image, cpu=2.0, timeout=3600)
def _unused_run_arm(cadence_min: int, n_shards: int = 12):
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
def main(n_shards: int = 12):
    """Shards are cached locally so a rerun (e.g. after a downstream bug) skips regeneration,
    which is the expensive CPU-bound step."""
    import json, os, pickle
    results = []
    for cad in (15, 12):
        cache_f = f"/tmp/modal_shards_{cad}_{n_shards}.pkl"
        if os.path.exists(cache_f):
            print(f"[cadence {cad} min] reusing cached shards from {cache_f}")
            shards = pickle.load(open(cache_f, "rb"))
        else:
            print(f"[cadence {cad} min] generating {n_shards} shards in parallel containers...")
            shards = list(gen_shard.starmap([(cad, i) for i in range(n_shards)]))
            pickle.dump(shards, open(cache_f, "wb"))
        print(f"[cadence {cad} min] {len(shards)} shards; training + evaluating on GPU...")
        results.append(train_eval.remote(cad, shards))
    print("\n=== CADENCE COMPARISON (matched shards, 12 epochs, seed 20260720) ===")
    print(json.dumps(results, indent=2))
    with open("/tmp/modal_cadence_result.json", "w") as f:
        json.dump(results, f, indent=2)

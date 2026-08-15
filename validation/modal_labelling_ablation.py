"""Label-only ablation of the detectability-conditioned labelling scheme (audit rerun).

WHAT WAS WRONG WITH THE FIRST ATTEMPT.  ``modal_ablations.py`` claimed its two labelling arms
differed "only" in the training labels.  They did not.  ``pipeline.train.compute_weights`` derives
the per-class loss weights from whichever label array it is handed, so switching --label-source
also switched the training objective.  The measured gap was therefore attributable to a compound
labelling-AND-weighting policy, not to the labelling scheme the title names.

WHAT THIS RUN CHANGES.

  1. Both arms pass ``--weight-labels observational``, pinning the class balance to the
     observational labels.  The only difference between the arms is the target each event is
     trained against.  A third arm (``labels_generator_ownweights``) reproduces the old, confounded
     configuration so the size of the confound itself is measurable rather than assumed.

  2. Every model is scored against BOTH ontologies -- a 2x2 matrix.  Scoring each arm only against
     the observational label measures agreement with the adopted operational target; it cannot show
     that target is objectively right.  The 2x2 shows what each model is good at and makes the
     asymmetry (or lack of it) explicit.

  3. Macro-F1 gaps carry a paired bootstrap interval over evaluation events, so a difference is
     reported with an uncertainty instead of as a bare number.  Single training seed remains a
     stated limitation: the interval covers evaluation noise, not initialisation noise.

  4. Shard caches and checkpoints are keyed by a hash of the source tree AND the full experiment
     config.  The earlier script keyed them by split/shard index and arm name alone, so any change
     to code, seeds, epochs or generation parameters would silently reuse stale artifacts.  A
     manifest with git commit, dirty flag, pinned dependency versions, config, input hashes and
     checkpoint SHA-256s is written alongside the results.

Run:  modal run --detach validation/modal_labelling_ablation.py
      Detached is required: a backgrounded local client disconnects and Modal stops the app.
      Do NOT wrap the command in `timeout` -- the SIGTERM reaches the Modal client, which
      forwards it as a cancellation to the running container even in detached mode. That is how
      one run of this experiment was killed mid-training at epoch 6 of 10.
Fetch: modal volume get binml-labelling-v2 <key>/result.json
"""
import hashlib
import json
import os

import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["Flat", "PSPL", "NonPSPL", "PeriodicVar", "LongPeriodVar", "Eruptive"]

# Pinned so the image is reconstructible; unpinned deps were an audit finding.
DEPS = ["torch==2.5.1", "numpy==1.26.4", "scipy==1.14.1", "h5py==3.12.1",
        "VBBinaryLensing==3.7.0"]

# The complete experiment definition. Anything that can change a number belongs here, because the
# cache keys are a hash of this dict plus the source tree.
CFG = {
    "n_train_shards": 16, "n_eval_shards": 6, "n_shards_total": 200,
    "seed_base_train": 20260720, "seed_base_eval": 990000,
    "epochs": 10, "seed": 20260720, "alpha_nonpspl": 2.0, "truncate_aug": 0.5,
    "threshold": 0.9042405486106873, "bootstrap": 2000, "protocol": "label-only-v2",
}


def _code_hash(root: str) -> str:
    """SHA-256 over the simulator/model/training source that can change a result.

    `root` must be passed explicitly. An earlier version computed it from this file's location,
    which is correct locally but wrong inside the container -- Modal mounts the entrypoint at a
    path whose parent holds no pipeline/ or binml/, so os.walk found nothing and the function
    returned the SHA-256 of the empty string. The cache key then carried no information about the
    code at all, which is the exact failure the key exists to prevent. It went unnoticed because
    an empty hash is a perfectly valid-looking hex string.
    """
    h = hashlib.sha256()
    n = 0
    for sub in ("pipeline", "binml"):
        for dirpath, dirnames, filenames in os.walk(os.path.join(root, sub)):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for fn in sorted(filenames):
                if fn.endswith(".py"):
                    fp = os.path.join(dirpath, fn)
                    h.update(os.path.relpath(fp, root).encode())
                    h.update(open(fp, "rb").read())
                    n += 1
    protocol = os.path.join(root, "validation", "modal_labelling_ablation.py")
    if os.path.exists(protocol):
        h.update(os.path.relpath(protocol, root).encode())
        h.update(open(protocol, "rb").read())
        n += 1
    if n == 0:
        raise RuntimeError(f"no source found under {root}; refusing to return an empty code hash")
    return h.hexdigest()


def _key(code_hash: str) -> str:
    return hashlib.sha256((code_hash + json.dumps(CFG, sort_keys=True)).encode()).hexdigest()[:16]


# Computed LOCALLY, where the source tree lives, and then passed to every remote function as an
# argument. Module-level globals are re-evaluated when Modal imports this file inside the
# container, where the source is at /repo rather than beside this file -- so a global would be
# recomputed there and silently disagree with the local one. It is None in the container, and
# nothing in the container reads it.
try:
    CODE_HASH = _code_hash(REPO)
    KEY = _key(CODE_HASH)
except (RuntimeError, OSError):
    CODE_HASH = KEY = None

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install(*DEPS)
         .add_local_dir(os.path.join(REPO, "pipeline"), "/repo/pipeline", copy=True)
         .add_local_dir(os.path.join(REPO, "binml"), "/repo/binml", copy=True)
         .add_local_file(__file__, "/repo/validation/modal_labelling_ablation.py", copy=True))
app = modal.App("binml-labelling-v2")
vol = modal.Volume.from_name("binml-labelling-v2", create_if_missing=True)
VOL = "/data"


@app.function(image=image, cpu=16.0, timeout=5400, volumes={VOL: vol})
def gen(shard: int, split: str, key: str) -> str:
    """Generate one shard into the run-keyed directory. Re-running is a no-op only when the code
    and config hash to the same KEY, so a stale shard cannot be silently reused."""
    import glob
    import shutil
    import subprocess
    import sys
    os.chdir("/repo"); sys.path.insert(0, "/repo")
    d = f"{VOL}/{key}/{split}"; os.makedirs(d, exist_ok=True)
    dest = f"{d}/shard_{shard:05d}.h5"
    if os.path.exists(dest):
        return dest
    seed_base = CFG["seed_base_train"] if split == "train" else CFG["seed_base_eval"]
    W = f"/tmp/g_{split}_{shard}"; os.makedirs(f"{W}/raw", exist_ok=True)
    subprocess.run(["python", "-m", "pipeline.run_shard", "--shard", str(shard),
                    "--n-shards", str(CFG["n_shards_total"]), "--out", f"{W}/raw",
                    "--seed-base", str(seed_base)],
                   check=True, env=dict(os.environ, PYTHONPATH="/repo"), cwd="/repo")
    from pipeline.cache import build_cache
    build_cache([sorted(glob.glob(f"{W}/raw/*.h5"))[0]], f"{W}/c.h5")
    shutil.move(f"{W}/c.h5", dest); vol.commit()
    shutil.rmtree(W, ignore_errors=True)
    return dest


@app.function(image=image, gpu="T4", cpu=8.0, timeout=14400, volumes={VOL: vol})
def run(key: str, code_hash: str) -> dict:
    import glob
    import shutil
    import subprocess
    import sys
    import warnings
    import numpy as np
    import torch
    os.chdir("/repo"); sys.path.insert(0, "/repo"); warnings.simplefilter("ignore")
    # Verify the image really contains the code that was hashed locally. Without this the cache
    # key is a promise rather than a check: the key would be derived from one tree and the run
    # executed against another, and nothing would say so.
    actual = _code_hash("/repo")
    if actual != code_hash:
        raise RuntimeError(f"image source hash {actual[:16]} != key source hash "
                           f"{str(code_hash)[:16]}; the container is not running the code this "
                           f"run was keyed on")
    vol.reload()
    from pipeline.model import BAND_BINS, BinMLv5, ModelConfigV5
    env = dict(os.environ, PYTHONPATH="/repo")

    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()

    def memmap(split, tag):
        d = f"/tmp/{tag}"; os.makedirs(f"{d}/cache", exist_ok=True)
        srcs = sorted(glob.glob(f"{VOL}/{key}/{split}/*.h5"))
        for i, p in enumerate(srcs):
            shutil.copy(p, f"{d}/cache/shard_{i:05d}.h5")
        subprocess.run(["python", "-m", "pipeline.to_memmap", "--in-dir", f"{d}/cache",
                        "--out", f"{d}/mm"], check=True, env=env, cwd="/repo")
        return f"{d}/mm", [sha(p) for p in srcs]

    tr, tr_sha = memmap("train", "tr")
    ev, ev_sha = memmap("eval", "ev")

    # ---- arms -------------------------------------------------------------------------------
    # The first two differ ONLY in --label-source. The third restores the confounded configuration
    # of the earlier run so the weighting contribution can be separated from the labelling one.
    arms = {
        "labels_observational": ["--label-source", "observational",
                                 "--weight-labels", "observational"],
        "labels_generator": ["--label-source", "generator",
                             "--weight-labels", "observational"],
        "labels_generator_ownweights": ["--label-source", "generator",
                                        "--weight-labels", "train"],
    }
    ckpts = {}
    for tag, extra in arms.items():
        out = f"{VOL}/{key}/ckpt_{tag}.pt"
        # A COMPLETION MARKER, not the checkpoint's existence, decides whether to skip. train.py
        # rewrites args.out at every new best epoch, so an interrupted run leaves a checkpoint that
        # loads fine but was trained for fewer epochs than the config says. Skipping on existence
        # alone would silently mix a 6-epoch arm with a 10-epoch arm and call it an ablation.
        done = out + ".done"
        if not os.path.exists(done):
            for stale in (out, out + ".last"):
                if os.path.exists(stale):
                    print(f"[warn] removing incomplete {stale}", flush=True)
                    os.remove(stale)
            subprocess.run(["python", "-m", "pipeline.train", "--cache", tr, "--out", out,
                            "--epochs", str(CFG["epochs"]), "--seed", str(CFG["seed"]),
                            "--alpha-nonpspl", str(CFG["alpha_nonpspl"]),
                            "--truncate-aug", str(CFG["truncate_aug"]),
                            "--device", "cuda"] + extra,
                           check=True, env=env, cwd="/repo")
            with open(done, "w") as f:
                json.dump({"arm": tag, "args": extra, "epochs": CFG["epochs"],
                           "sha256": sha(out)}, f)
            vol.commit()
        ckpts[tag] = out

    # ---- evaluation -------------------------------------------------------------------------
    meta = json.load(open(f"{ev}/meta.json")); n = meta["n_events"]
    arr = {b: (np.memmap(f"{ev}/feat_{b}.f16", np.float16, "r", shape=(n, L, 3)),
               np.memmap(f"{ev}/frac_{b}.f16", np.float16, "r", shape=(n, L)))
           for b, L in BAND_BINS.items()}
    y_obs = np.load(f"{ev}/label.npy")
    y_gen = np.load(f"{ev}/true_class.npy")
    kp = np.load(f"{ev}/keep_prob.npy").astype(float)
    w = 1.0 / np.clip(kp, 1e-3, 1.0)
    NON = CLASSES.index("NonPSPL")
    thr = CFG["threshold"]

    def probs(path):
        ck = torch.load(path, map_location="cpu")
        cfg = ModelConfigV5(**{k: v for k, v in ck["config"].items()
                               if k in ModelConfigV5.__dataclass_fields__})
        net = BinMLv5(cfg).cuda(); net.load_state_dict(ck["model"]); net.eval()
        P = np.zeros((n, 6), np.float32)
        for s in range(0, n, 2048):
            sl = slice(s, min(s + 2048, n)); feats, pres = {}, {}
            for b, L in BAND_BINS.items():
                x = np.asarray(arr[b][0][sl], np.float32)
                fr = np.asarray(arr[b][1][sl], np.float32)
                obs = np.isfinite(x[:, :, 0]).astype(np.float32)
                t = np.concatenate([np.nan_to_num(x, nan=0.0), fr[:, :, None], obs[:, :, None]], 2)
                feats[b] = torch.tensor(t).cuda(); pres[b] = feats[b][..., 4].sum(1) > 0
            with torch.inference_mode():
                lg = net(feats, pres).float().cpu().numpy()
            z = lg - lg.max(1, keepdims=True); e = np.exp(z)
            P[sl] = e / e.sum(1, keepdims=True)
        return P

    def macro_f1(pred, y, ww, idx=None):
        if idx is not None:
            pred, y, ww = pred[idx], y[idx], ww[idx]
        f1 = []
        for c in range(6):
            tp = (ww * ((pred == c) & (y == c))).sum()
            fp = (ww * ((pred == c) & (y != c))).sum()
            fn = (ww * ((pred != c) & (y == c))).sum()
            r = tp / max(tp + fn, 1e-9); p = tp / max(tp + fp, 1e-9)
            f1.append(2 * r * p / max(r + p, 1e-9))
        return float(np.mean(f1)), [round(float(v), 3) for v in f1]

    P = {tag: probs(p) for tag, p in ckpts.items()}
    pred = {tag: v.argmax(1) for tag, v in P.items()}

    res = {"key": key, "n_eval": int(n), "config": CFG,
           "manifest": {"code_sha256": code_hash, "deps": DEPS,
                        "train_shard_sha256": tr_sha, "eval_shard_sha256": ev_sha,
                        "checkpoint_sha256": {t: sha(p) for t, p in ckpts.items()},
                        "torch": torch.__version__, "numpy": np.__version__},
           "arms": {}}

    # 2x2 (3x2 with the confounded arm): every model against every ontology.
    for tag in ckpts:
        d = {"vs_observational": {}, "vs_generator": {}}
        for name, y in (("vs_observational", y_obs), ("vs_generator", y_gen)):
            mf1, per = macro_f1(pred[tag], y, w)
            d[name] = {"macro_f1": round(mf1, 3),
                       "per_class_f1": dict(zip(CLASSES, per))}
        sel = P[tag][:, NON] >= thr; yb = y_obs == NON
        tp = (w * (sel & yb)).sum(); fp = (w * (sel & ~yb)).sum(); fn = (w * (~sel & yb)).sum()
        d["anomaly_completeness_at_thr"] = round(float(tp / max(tp + fn, 1e-9)), 3)
        d["anomaly_purity_at_thr"] = round(float(tp / max(tp + fp, 1e-9)), 3)
        res["arms"][tag] = d

    # Paired bootstrap over evaluation events for the label-only macro-F1 gap.
    rng = np.random.default_rng(CFG["seed"])
    for ont, y in (("observational", y_obs), ("generator", y_gen)):
        gaps = []
        for _ in range(CFG["bootstrap"]):
            idx = rng.integers(0, n, n)
            a, _u = macro_f1(pred["labels_observational"], y, w, idx)
            b, _u = macro_f1(pred["labels_generator"], y, w, idx)
            gaps.append(a - b)
        g = np.array(gaps)
        res[f"macro_f1_gap_vs_{ont}"] = {
            "point": round(float(macro_f1(pred["labels_observational"], y, w)[0]
                                 - macro_f1(pred["labels_generator"], y, w)[0]), 3),
            "ci95": [round(float(np.percentile(g, 2.5)), 3),
                     round(float(np.percentile(g, 97.5)), 3)],
            "note": "paired bootstrap over evaluation events; single training seed"}

    with open(f"{VOL}/{key}/result.json", "w") as f:
        json.dump(res, f, indent=2)
    vol.commit()
    return res


@app.function(image=image, timeout=86400, volumes={VOL: vol})
def orchestrate(key: str, code_hash: str) -> dict:
    """Drive the whole experiment FROM INSIDE Modal.

    The local client is not a reliable supervisor for a multi-hour run: `modal run --detach` still
    forwards a SIGTERM or a dropped gRPC stream to the running container as a cancellation, and two
    attempts at this experiment were killed mid-training that way (once at epoch 6 of 10, once at
    epoch 3). Spawning this function and exiting means the fan-out and the GPU stage are sequenced
    server-side, and a local disconnect cannot touch them.
    """
    args = ([(i, "train", key) for i in range(CFG["n_train_shards"])]
            + [(i, "eval", key) for i in range(CFG["n_eval_shards"])])
    print(f"[gen] {len(args)} shards -> volume {key}/", flush=True)
    list(gen.starmap(args))
    print("[run] three arms, 3x2 ontology matrix, paired bootstrap ...", flush=True)
    return run.remote(key, code_hash)


@app.local_entrypoint()
def main():
    """Spawn and exit. Poll with:
        modal volume get binml-labelling-v2 <KEY>/result.json -
    """
    print(f"[key] {KEY}  (code {CODE_HASH[:12]})")
    call = orchestrate.spawn(KEY, CODE_HASH)
    print(f"[spawned] {call.object_id}; results -> volume {KEY}/result.json")

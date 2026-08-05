"""Does fine-tuning on low-mass-ratio (planetary) regimes improve BinML where it is weakest?

Motivation: completeness at the fixed operating threshold falls with lens mass ratio -- 0.887 for
stellar binaries but 0.588 for q < 1e-4 (Table: mass-ratio regimes). The natural question is whether
targeted fine-tuning recovers that deficit, and at what cost elsewhere.

Design (all in isolated Modal containers):
  1. Generate a PLANETARY-enriched training set (--regime planetary: q in [1e-6, 1e-3]).
  2. Generate a NATURAL-mix evaluation set (never trained on) for the regression check.
  3. Warm-start from the shipped checkpoint and fine-tune at a low learning rate.
  4. Evaluate BOTH the shipped model and the fine-tuned model on the SAME natural eval set,
     reporting completeness by mass-ratio regime and per-class F1.

The regression check is the point: a fine-tune that lifts low-q recall while wrecking PSPL purity
or the contaminant classes is not an improvement. Reports both sides.

Run:  modal run validation/modal_massregime_finetune.py
"""
import os
import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy", "h5py", "VBBinaryLensing")
    .add_local_dir(os.path.join(REPO, "pipeline"), "/repo/pipeline", copy=True)
    .add_local_dir(os.path.join(REPO, "binml"), "/repo/binml", copy=True)
)

app = modal.App("binml-massregime")


@app.function(image=image, cpu=16.0, timeout=5400)
def gen_shard(shard: int, regime: str, seed_base: int) -> bytes:
    """Generate one shard (optionally in a hard regime) and return its binned cache."""
    import subprocess, sys, os, glob
    os.chdir("/repo"); sys.path.insert(0, "/repo")
    W = f"/tmp/g_{regime}_{shard}"
    os.makedirs(f"{W}/raw", exist_ok=True)
    cmd = ["python", "-m", "pipeline.run_shard", "--shard", str(shard), "--n-shards", "200",
           "--out", f"{W}/raw", "--seed-base", str(seed_base)]
    if regime != "none":
        cmd += ["--regime", regime]
    subprocess.run(cmd, check=True, env=dict(os.environ, PYTHONPATH="/repo"), cwd="/repo")
    from pipeline.cache import build_cache
    raw = sorted(glob.glob(f"{W}/raw/*.h5"))[0]
    out = f"{W}/cache.h5"
    build_cache([raw], out)
    return open(out, "rb").read()


@app.function(image=image, gpu="T4", cpu=8.0, timeout=5400)
def finetune_and_compare(train_shards: list, eval_shards: list, epochs: int = 6) -> dict:
    import subprocess, sys, os, json, warnings
    import numpy as np
    os.chdir("/repo"); sys.path.insert(0, "/repo"); warnings.simplefilter("ignore")

    def write(shards, d):
        os.makedirs(d, exist_ok=True)
        for i, b in enumerate(shards):
            open(f"{d}/shard_{i:05d}.h5", "wb").write(b)

    write(train_shards, "/tmp/tr/cache"); write(eval_shards, "/tmp/ev/cache")
    env = dict(os.environ, PYTHONPATH="/repo")
    def run(c): subprocess.run(c, check=True, env=env, cwd="/repo")
    run(["python", "-m", "pipeline.to_memmap", "--in-dir", "/tmp/tr/cache", "--out", "/tmp/tr/mm"])
    run(["python", "-m", "pipeline.to_memmap", "--in-dir", "/tmp/ev/cache", "--out", "/tmp/ev/mm"])

    base = "/repo/binml/weights/binml.pt"
    run(["python", "-m", "pipeline.train", "--cache", "/tmp/tr/mm", "--out", "/tmp/ft.pt",
         "--init-weights", base, "--epochs", str(epochs), "--lr", "5e-5",
         "--truncate-aug", "0.5", "--seed", "20260720", "--device", "cuda"])

    # score both models on the SAME natural eval set
    import torch
    from pipeline.model import BAND_BINS, BinMLv5, ModelConfigV5
    meta = json.load(open("/tmp/ev/mm/meta.json")); n = meta["n_events"]
    pf = meta.get("param_fields") or []
    arrays = {}
    for b, L in BAND_BINS.items():
        arrays[f"feat/{b}"] = np.memmap(f"/tmp/ev/mm/feat_{b}.f16", np.float16, "r", shape=(n, L, 3))
        arrays[f"frac/{b}"] = np.memmap(f"/tmp/ev/mm/frac_{b}.f16", np.float16, "r", shape=(n, L))
    y = np.load("/tmp/ev/mm/label.npy"); kp = np.load("/tmp/ev/mm/keep_prob.npy").astype(float)
    w = 1.0 / np.clip(kp, 1e-3, 1.0)
    params = np.load("/tmp/ev/mm/params.npy") if os.path.exists("/tmp/ev/mm/params.npy") else None
    q = params[:, pf.index("q")] if (params is not None and "q" in pf) else np.full(n, np.nan)
    NON = 2

    def scores(ckpt):
        ck = torch.load(ckpt, map_location="cpu")
        cfg = ModelConfigV5(**{k: v for k, v in ck["config"].items()
                               if k in ModelConfigV5.__dataclass_fields__})
        net = BinMLv5(cfg).cuda(); net.load_state_dict(ck["model"]); net.eval()
        P = np.zeros((n, 6), np.float32)
        for s in range(0, n, 2048):
            sl = slice(s, min(s + 2048, n))
            feats, present = {}, {}
            for b, L in BAND_BINS.items():
                x = np.asarray(arrays[f"feat/{b}"][sl], np.float32)
                fr = np.asarray(arrays[f"frac/{b}"][sl], np.float32)
                obs = np.isfinite(x[:, :, 0]).astype(np.float32)
                x = np.nan_to_num(x, nan=0.0)
                t = np.concatenate([x, fr[:, :, None], obs[:, :, None]], axis=2)
                feats[b] = torch.tensor(t).cuda(); present[b] = feats[b][..., 4].sum(1) > 0
            with torch.inference_mode():
                lg = net(feats, present).float().cpu().numpy()
            z = lg - lg.max(1, keepdims=True); e = np.exp(z); P[sl] = e / e.sum(1, keepdims=True)
        return P

    thr = 0.9042405486106873
    out = {}
    for tag, ck in (("shipped", base), ("finetuned", "/tmp/ft.pt")):
        P = scores(ck); pred = P.argmax(1)
        d = {"per_class_f1": {}}
        for c, name in enumerate(["Flat", "PSPL", "NonPSPL", "PeriodicVar", "LongPeriodVar", "Eruptive"]):
            tp = (w * ((pred == c) & (y == c))).sum(); fp = (w * ((pred == c) & (y != c))).sum()
            fn = (w * ((pred != c) & (y == c))).sum()
            r = tp / max(tp + fn, 1e-9); p = tp / max(tp + fp, 1e-9)
            d["per_class_f1"][name] = round(float(2 * r * p / max(r + p, 1e-9)), 3)
        for nm, sel in (("stellar", q > 1e-2), ("giant", (q > 1e-3) & (q <= 1e-2)),
                        ("neptune", (q > 1e-4) & (q <= 1e-3)), ("lowmass", (q <= 1e-4) & (q > 0))):
            m = (y == NON) & sel
            d[f"comp_{nm}"] = (round(float((w[m] * (P[m, NON] >= thr)).sum() / w[m].sum()), 3)
                               if m.sum() >= 20 else None)
            d[f"n_{nm}"] = int(m.sum())
        out[tag] = d
    out["eval_n"] = int(n)
    return out


@app.local_entrypoint()
def main(n_train: int = 10, n_eval: int = 6):
    import json
    print(f"[gen] {n_train} planetary-regime training shards + {n_eval} natural eval shards...")
    tr = list(gen_shard.starmap([(i, "planetary", 20260720) for i in range(n_train)]))
    ev = list(gen_shard.starmap([(i, "none", 880000) for i in range(n_eval)]))  # unseen seed base
    print("[train] fine-tuning from shipped weights, then comparing on the natural eval set...")
    res = finetune_and_compare.remote(tr, ev)
    print("\n=== MASS-REGIME FINE-TUNE: shipped vs fine-tuned (same natural eval set) ===")
    print(json.dumps(res, indent=2))
    json.dump(res, open("/tmp/massregime_result.json", "w"), indent=2)

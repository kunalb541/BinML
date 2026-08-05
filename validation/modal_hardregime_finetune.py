"""Multi-regime hard fine-tune: can targeted training close ALL the documented weak spots?

The planetary-only fine-tune recovered most of the low-mass-ratio deficit at no cost
(validation/modal_massregime_finetune.py). This extends that to every regime the stress test flags,
and evaluates ALL SIX CLASSES rather than only the anomaly channel:

  weakness (stress-test value)                     targeted training regime
  ------------------------------------------------ ------------------------
  NonPSPL recall 0.222 at wide separation           wide, wider, planet_wide
  PeriodicVar recall 0.054 for long periods         perhard
  PSPL recall 0.524 sub-day / 0.715 long timescale  shortte, longte
  PSPL recall 0.491 + anomaly precision 0.033 faint faint, flatrich
  deep-planetary deficit                            planetary, planet_res
  LongPeriodVar confusion                           lpvhard

Both the shipped and the fine-tuned model are scored on the SAME evaluation sets, generated from an
unseen seed base: a natural mixture (regression check) plus per-regime out-of-range sets so the
weak spots are measured where they actually occur.

Run:  modal run validation/modal_hardregime_finetune.py
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

app = modal.App("binml-hardregime")

# Shards live in a Modal Volume rather than being returned to the caller. Returning them as bytes
# routes every shard (~300 MB) through the local client, which OOM-kills it at this scale.
vol = modal.Volume.from_name("binml-hardregime-data", create_if_missing=True)
VOL = "/data"

# training mix: two shards each of the regimes covering the documented weaknesses
TRAIN_MIX = ["planetary", "planet_res", "wide", "wider", "planet_wide",
             "shortte", "longte", "faint", "flatrich", "perhard", "lpvhard"]
# evaluation regimes: where the failures were measured (plus natural for regression)
EVAL_REGIMES = ["none", "oor_np_widesep", "oor_per_longp", "oor_pspl_shortte", "oor_flat_faint"]

CLASSES = ["Flat", "PSPL", "NonPSPL", "PeriodicVar", "LongPeriodVar", "Eruptive"]


@app.function(image=image, cpu=16.0, timeout=5400, volumes={VOL: vol})
def gen(shard: int, regime: str, seed_base: int, split: str) -> str:
    """Generate one shard, bin it, and write the cache INTO THE VOLUME. Returns only its path."""
    import subprocess, sys, os, glob, shutil
    os.chdir("/repo"); sys.path.insert(0, "/repo")
    dest_dir = f"{VOL}/{split}/{regime}"
    os.makedirs(dest_dir, exist_ok=True)
    dest = f"{dest_dir}/shard_{shard:05d}.h5"
    if os.path.exists(dest):
        return dest
    W = f"/tmp/g_{regime}_{shard}_{seed_base}"
    os.makedirs(f"{W}/raw", exist_ok=True)
    cmd = ["python", "-m", "pipeline.run_shard", "--shard", str(shard), "--n-shards", "200",
           "--out", f"{W}/raw", "--seed-base", str(seed_base)]
    if regime != "none":
        cmd += ["--regime", regime]
    subprocess.run(cmd, check=True, env=dict(os.environ, PYTHONPATH="/repo"), cwd="/repo")
    from pipeline.cache import build_cache
    tmp_out = f"{W}/c.h5"
    build_cache([sorted(glob.glob(f"{W}/raw/*.h5"))[0]], tmp_out)
    shutil.move(tmp_out, dest)
    vol.commit()
    shutil.rmtree(W, ignore_errors=True)
    return dest


@app.function(image=image, gpu="T4", cpu=8.0, timeout=7200, volumes={VOL: vol})
def finetune_and_eval(eval_regimes: list, epochs: int = 8) -> dict:
    """Reads all shards from the Volume (nothing large crosses the client)."""
    import subprocess, sys, os, json, glob, shutil, warnings
    import numpy as np, torch
    os.chdir("/repo"); sys.path.insert(0, "/repo"); warnings.simplefilter("ignore")
    vol.reload()
    from pipeline.model import BAND_BINS, BinMLv5, ModelConfigV5

    def build_mm(src_glob, tag):
        d = f"/tmp/{tag}"
        os.makedirs(f"{d}/cache", exist_ok=True)
        for i, p in enumerate(sorted(glob.glob(src_glob))):
            shutil.copy(p, f"{d}/cache/shard_{i:05d}.h5")
        subprocess.run(["python", "-m", "pipeline.to_memmap", "--in-dir", f"{d}/cache",
                        "--out", f"{d}/mm"], check=True,
                       env=dict(os.environ, PYTHONPATH="/repo"), cwd="/repo")
        return f"{d}/mm"

    tr_mm = build_mm(f"{VOL}/train/*/*.h5", "tr")
    ev_mm = {r: build_mm(f"{VOL}/eval/{r}/*.h5", f"ev_{r}") for r in eval_regimes}

    base = "/repo/binml/weights/binml.pt"
    subprocess.run(["python", "-m", "pipeline.train", "--cache", tr_mm, "--out", "/tmp/ft.pt",
                    "--init-weights", base, "--epochs", str(epochs), "--lr", "5e-5",
                    "--truncate-aug", "0.5", "--seed", "20260720", "--device", "cuda"],
                   check=True, env=dict(os.environ, PYTHONPATH="/repo"), cwd="/repo")

    def load(ckpt):
        ck = torch.load(ckpt, map_location="cpu")
        cfg = ModelConfigV5(**{k: v for k, v in ck["config"].items()
                               if k in ModelConfigV5.__dataclass_fields__})
        net = BinMLv5(cfg).cuda(); net.load_state_dict(ck["model"]); net.eval()
        return net

    def score(net, mm):
        meta = json.load(open(f"{mm}/meta.json")); n = meta["n_events"]
        arr = {}
        for b, L in BAND_BINS.items():
            arr[b] = (np.memmap(f"{mm}/feat_{b}.f16", np.float16, "r", shape=(n, L, 3)),
                      np.memmap(f"{mm}/frac_{b}.f16", np.float16, "r", shape=(n, L)))
        P = np.zeros((n, 6), np.float32)
        for s in range(0, n, 2048):
            sl = slice(s, min(s + 2048, n))
            feats, present = {}, {}
            for b, L in BAND_BINS.items():
                x = np.asarray(arr[b][0][sl], np.float32)
                fr = np.asarray(arr[b][1][sl], np.float32)
                obs = np.isfinite(x[:, :, 0]).astype(np.float32)
                t = np.concatenate([np.nan_to_num(x, nan=0.0), fr[:, :, None], obs[:, :, None]], 2)
                feats[b] = torch.tensor(t).cuda(); present[b] = feats[b][..., 4].sum(1) > 0
            with torch.inference_mode():
                lg = net(feats, present).float().cpu().numpy()
            z = lg - lg.max(1, keepdims=True); e = np.exp(z); P[sl] = e / e.sum(1, keepdims=True)
        return P, np.load(f"{mm}/label.npy"), np.load(f"{mm}/keep_prob.npy").astype(float)

    nets = {"shipped": load(base), "finetuned": load("/tmp/ft.pt")}
    out = {}
    for rname, mm in ev_mm.items():
        out[rname] = {}
        for tag, net in nets.items():
            P, y, kp = score(net, mm)
            w = 1.0 / np.clip(kp, 1e-3, 1.0)
            pred = P.argmax(1)
            d = {}
            for c, name in enumerate(CLASSES):
                tp = (w * ((pred == c) & (y == c))).sum(); fn = (w * ((pred != c) & (y == c))).sum()
                fp = (w * ((pred == c) & (y != c))).sum()
                nsup = int((y == c).sum())
                d[name] = {"recall": round(float(tp / max(tp + fn, 1e-9)), 3),
                           "precision": round(float(tp / max(tp + fp, 1e-9)), 3),
                           "n": nsup} if nsup >= 20 else None
            out[rname][tag] = d
        out[rname]["n_events"] = int(len(y))
    return out


@app.local_entrypoint()
def main(shards_per_regime: int = 2, eval_shards: int = 2):
    import json
    args = [(i, r, 20260720, "train") for r in TRAIN_MIX for i in range(shards_per_regime)]
    args += [(i, r, 870000, "eval") for r in EVAL_REGIMES for i in range(eval_shards)]
    print(f"[gen] {len(args)} shards -> Modal Volume (train: {len(TRAIN_MIX)} regimes, "
          f"eval: {len(EVAL_REGIMES)} regimes)")
    paths = list(gen.starmap(args))
    print(f"[gen] wrote {len(paths)} shards to the volume")
    print("[train] fine-tuning on the hard mix, then scoring both models everywhere...")
    res = finetune_and_eval.remote(EVAL_REGIMES)
    print("\n=== HARD-REGIME FINE-TUNE: shipped vs fine-tuned, all classes, all regimes ===")
    print(json.dumps(res, indent=2))
    json.dump(res, open("/tmp/hardregime_result.json", "w"), indent=2)

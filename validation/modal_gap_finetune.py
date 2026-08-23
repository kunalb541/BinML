"""Gap fine-tune on Modal: teach BinML that Roman's F146 schedule has gaps.

Why this exists -- validation/gulls/gap_sensitivity.py.  RMDC26 (GULLS) implements the real GBTDS
schedule, in which F146 pauses ~6 h seven times per season.  BinML's training grid is continuous;
inserting those gaps into in-distribution events drops PSPL recall 0.93 -> 0.12 and Flat
1.00 -> 0.08 (both go to NonPSPL / PeriodicVar) with NonPSPL and PeriodicVar unaffected, which is
the GULLS cross-simulator failure reproduced with no GULLS data.

What this does.  Generates NATURAL-prior shards (no hard-regime override, so the class balance
the shipped model was tuned on is preserved), warm-starts from the shipped checkpoint at a low
learning rate with `--gap-aug 0.5` on top of the original `--truncate-aug 0.5`, and scores
shipped vs fine-tuned on the same held-out shards three ways: clean, with RMDC26's seven-gap
schedule, and with random 1-8 gaps.  The clean column is the regression guard: a fine-tune that
buys gap robustness by losing clean performance is not a fix.

Operational rules, learnt the hard way: run with `modal run --detach`; never wrap in `timeout`
(SIGTERM cancels every container); artefacts carry `.done` markers so a partial checkpoint is
never mistaken for a finished one.

    modal run --detach validation/modal_gap_finetune.py --train-shards 6 --eval-shards 2 --epochs 6
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
app = modal.App("binml-gapfinetune")
vol = modal.Volume.from_name("binml-gap-data", create_if_missing=True)
VOL = "/data"
CLASSES = ["Flat", "PSPL", "NonPSPL", "PeriodicVar", "LongPeriodVar", "Eruptive"]
RMDC26_GAPS_D = [0.98, 2.48, 21.49, 31.48, 35.23, 62.23, 69.48]
RMDC26_GAP_H = 6.2


@app.function(image=image, cpu=16.0, timeout=5400, volumes={VOL: vol})
def gen(shard: int, seed_base: int, split: str) -> str:
    import subprocess, glob, shutil
    dest_dir = f"{VOL}/{split}"
    os.makedirs(dest_dir, exist_ok=True)
    dest = f"{dest_dir}/shard_{shard:05d}.h5"
    if os.path.exists(dest + ".done"):
        return dest
    W = f"/tmp/g_{split}_{shard}_{seed_base}"
    os.makedirs(f"{W}/raw", exist_ok=True)
    subprocess.run(["python", "-m", "pipeline.run_shard", "--shard", str(shard), "--n-shards", "200",
                    "--out", f"{W}/raw", "--seed-base", str(seed_base)],
                   check=True, env=dict(os.environ, PYTHONPATH="/repo"), cwd="/repo")
    from pipeline.cache import build_cache
    build_cache([sorted(glob.glob(f"{W}/raw/*.h5"))[0]], f"{W}/c.h5")
    shutil.move(f"{W}/c.h5", dest)
    open(dest + ".done", "w").close()
    vol.commit()
    shutil.rmtree(W, ignore_errors=True)
    return dest


@app.function(image=image, gpu="T4", cpu=8.0, timeout=10800, volumes={VOL: vol})
def finetune_and_eval(epochs: int, gap_aug: float, lr: float, tag: str) -> dict:
    import subprocess, sys, glob, shutil, json, warnings
    import numpy as np, torch
    os.chdir("/repo"); sys.path.insert(0, "/repo"); warnings.simplefilter("ignore")
    vol.reload()
    from pipeline.model import BAND_BINS, BinMLv5, ModelConfigV5

    def build_mm(src_glob, name):
        d = f"/tmp/{name}"
        os.makedirs(f"{d}/cache", exist_ok=True)
        for i, p in enumerate(sorted(glob.glob(src_glob))):
            shutil.copy(p, f"{d}/cache/shard_{i:05d}.h5")
        subprocess.run(["python", "-m", "pipeline.to_memmap", "--in-dir", f"{d}/cache",
                        "--out", f"{d}/mm"], check=True,
                       env=dict(os.environ, PYTHONPATH="/repo"), cwd="/repo")
        return f"{d}/mm"

    tr_mm = build_mm(f"{VOL}/train/*.h5", "tr")
    ev_mm = build_mm(f"{VOL}/eval/*.h5", "ev")
    base = "/repo/binml/weights/binml.pt"
    ckpt = f"{VOL}/ft_{tag}.pt"
    if not os.path.exists(ckpt + ".done"):
        subprocess.run(["python", "-m", "pipeline.train", "--cache", tr_mm, "--out", "/tmp/ft.pt",
                        "--init-weights", base, "--epochs", str(epochs), "--lr", str(lr),
                        "--truncate-aug", "0.5", "--gap-aug", str(gap_aug),
                        "--seed", "20260823", "--device", "cuda"],
                       check=True, env=dict(os.environ, PYTHONPATH="/repo"), cwd="/repo")
        shutil.copy("/tmp/ft.pt", ckpt)
        open(ckpt + ".done", "w").close()
        vol.commit()

    def load(p):
        ck = torch.load(p, map_location="cpu")
        cfg = ModelConfigV5(**{k: v for k, v in ck["config"].items()
                               if k in ModelConfigV5.__dataclass_fields__})
        net = BinMLv5(cfg).cuda(); net.load_state_dict(ck["model"]); net.eval()
        return net

    meta = json.load(open(f"{ev_mm}/meta.json")); n = meta["n_events"]
    arr = {}
    for b, L in BAND_BINS.items():
        arr[b] = (np.memmap(f"{ev_mm}/feat_{b}.f16", np.float16, "r", shape=(n, L, 3)),
                  np.memmap(f"{ev_mm}/frac_{b}.f16", np.float16, "r", shape=(n, L)))
    y = np.load(f"{ev_mm}/label.npy"); kp = np.load(f"{ev_mm}/keep_prob.npy").astype(float)
    from pipeline.train import MAG_SCALE

    def gap_mask(nb, mode, rng):
        """Bins to blank on the 864-grid.  'rmdc26' = the real schedule; 'random' = 1-8 x 1-12 h."""
        m = np.zeros(nb, bool)
        bin_h = 72.0 * 24.0 / nb
        if mode == "rmdc26":
            for s in RMDC26_GAPS_D:
                a = int(s * 24 / bin_h); m[a:a + int(round(RMDC26_GAP_H / bin_h))] = True
        elif mode == "random":
            for _ in range(int(rng.integers(1, 9))):
                w = max(1, int(round(rng.uniform(1, 12) / bin_h))); a = int(rng.integers(0, nb - w))
                m[a:a + w] = True
        return m

    def score(net, mode):
        rng = np.random.default_rng(1)
        P = np.zeros((n, len(CLASSES)), np.float32)
        bs = 512
        for s in range(0, n, bs):
            e = min(n, s + bs); feats = {}
            m146 = gap_mask(BAND_BINS["F146"], mode, rng) if mode != "clean" else None
            for b, L in BAND_BINS.items():
                f = arr[b][0][s:e].astype(np.float32); fr = arr[b][1][s:e].astype(np.float32)
                obs = np.isfinite(f[:, :, 0]).astype(np.float32)
                f = np.nan_to_num(f) / MAG_SCALE
                if m146 is not None:
                    mb = m146 if L == 864 else m146.reshape(L, 864 // L).any(1)
                    f[:, mb] = 0; fr[:, mb] = 0; obs[:, mb] = 0
                x = np.concatenate([f, fr[:, :, None], obs[:, :, None]], 2)
                feats[b] = torch.tensor(x).cuda()
            present = {b: feats[b][..., 4].sum(1) > 0 for b in BAND_BINS}
            with torch.inference_mode():
                P[s:e] = torch.softmax(net(feats, present).float(), 1).cpu().numpy()
        return P

    nets = {"shipped": load(base), "finetuned": load(ckpt)}
    out = {"n_events": int(n), "epochs": epochs, "gap_aug": gap_aug, "lr": lr}
    for mode in ("clean", "rmdc26", "random"):
        out[mode] = {}
        for name, net in nets.items():
            P = score(net, mode); pred = P.argmax(1); w = 1.0 / np.clip(kp, 1e-3, 1.0)
            d = {}
            for c, cname in enumerate(CLASSES):
                tp = (w * ((pred == c) & (y == c))).sum(); fn = (w * ((pred != c) & (y == c))).sum()
                fp = (w * ((pred == c) & (y != c))).sum(); ns = int((y == c).sum())
                d[cname] = {"recall": round(float(tp / max(tp + fn, 1e-9)), 3),
                            "precision": round(float(tp / max(tp + fp, 1e-9)), 3), "n": ns} if ns >= 20 else None
            f1 = [2 * v["precision"] * v["recall"] / max(v["precision"] + v["recall"], 1e-9)
                  for v in d.values() if v]
            d["macro_f1"] = round(float(np.mean(f1)), 4)
            out[mode][name] = d
    json.dump(out, open(f"{VOL}/result_{tag}.json", "w"), indent=2); vol.commit()
    return out


@app.local_entrypoint()
def main(train_shards: int = 6, eval_shards: int = 2, epochs: int = 6,
         gap_aug: float = 0.5, lr: float = 5e-5, tag: str = "g05"):
    import json
    args = [(i, 20260823, "train") for i in range(train_shards)]
    args += [(100 + i, 880000, "eval") for i in range(eval_shards)]
    print(f"[gen] {len(args)} natural-prior shards -> Volume", flush=True)
    paths = list(gen.starmap(args)); print(f"[gen] {len(paths)} shards ready", flush=True)
    print(f"[train] warm-start fine-tune: epochs={epochs} gap_aug={gap_aug} lr={lr}", flush=True)
    res = finetune_and_eval.remote(epochs, gap_aug, lr, tag)
    print(json.dumps(res, indent=2))
    json.dump(res, open(f"/tmp/gap_finetune_{tag}.json", "w"), indent=2)

"""
Modal training for the Roman microlensing classifier.

Runs the existing train.py on an L4 GPU, with a persistent Modal Volume holding both
the data and the checkpoints -- so --resume auto continues automatically across runs
(no Kaggle-style session/checkpoint dance). train.py only imports model.py (no
VBBinaryLensing), so the image is minimal.

Usage (from this directory, after `modal setup`):
    modal run train_modal.py::download     # one-time: pull the 300k subset into the volume
    modal run train_modal.py::train        # train on L4 (re-run to resume; volume persists)
    modal run train_modal.py               # does download then train
Inspect:
    modal volume ls roman-vol /ckpt
"""
import modal

CODE = "/Users/kunalbhatia/Desktop/Research/microlensing/binml/code"

app = modal.App("roman-microlensing")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "numpy", "scipy", "h5py", "numba", "tqdm", "scikit-learn", "kaggle",
        "matplotlib", "seaborn",
    )
    .add_local_file(f"{CODE}/train.py", "/code/train.py", copy=True)
    .add_local_file(f"{CODE}/model.py", "/code/model.py", copy=True)
    .add_local_file(f"{CODE}/evaluate.py", "/code/evaluate.py", copy=True)
)

vol = modal.Volume.from_name("roman-vol", create_if_missing=True)

# Kaggle access token for the account that owns the dataset (kunalb541). Inline secret
# so no extra setup step; rotate the token afterwards.
kaggle_secret = modal.Secret.from_dict(
    {"KAGGLE_API_TOKEN": "KGAT_2380b1ff6eedbcaae64a887a94abaea7"}
)
DATASET = "kunalb541/roman-microlensing-300k"


@app.function(image=image, volumes={"/vol": vol}, secrets=[kaggle_secret], timeout=3600)
def download():
    import os, glob
    os.makedirs("/vol/data", exist_ok=True)
    existing = glob.glob("/vol/data/subset_*.h5")
    if len(existing) >= 12:
        print("already have", len(existing), "shards; skipping download")
    else:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        print("downloading", DATASET, "...")
        api.dataset_download_files(DATASET, path="/vol/data", unzip=True, quiet=False)
    vol.commit()
    print("shards:", len(glob.glob("/vol/data/subset_*.h5")))


@app.function(image=image, gpu="L4", cpu=8.0, memory=16384,
              volumes={"/vol": vol}, timeout=86400)
def train():
    import subprocess, sys, os, glob, shutil
    # v4.2.0 SPEED: copy shards from the network volume to LOCAL container disk once, then
    # train from there. Per-batch random reads off the Modal volume stall the loader
    # (~1.15 it/s avg vs ~2.0 it/s peak); reading from local disk removes that -> ~40%
    # faster / cheaper. Checkpoints still go to the persistent /vol/ckpt.
    local = "/data_local"; os.makedirs(local, exist_ok=True)
    src = sorted(glob.glob("/vol/data/subset_*.h5"))
    for f in src:
        dst = os.path.join(local, os.path.basename(f))
        if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(f)):
            shutil.copy(f, dst)
    print(f"copied {len(glob.glob(local + '/subset_*.h5'))} shards to local disk")
    cmd = [
        sys.executable, "/code/train.py",
        "--stream", "--block-shuffle", "256", "--resume", "auto",
        "--data", local, "--output", "/vol/ckpt",
        "--hierarchical", "--use-aux-head", "--attention-pooling", "--use-amp",
        "--d-model", "64", "--n-layers", "4",
        "--batch-size", "128", "--accumulation-steps", "2",
        "--num-workers", "6", "--prefetch-factor", "6",
        "--stage1-weight", "0.5", "--stage2-weight", "2.0", "--aux-weight", "0.5",
        "--epochs", "40", "--warmup-epochs", "3",
        "--early-stop-patience", "8",   # stop if val loss plateaus for 8 epochs
    ]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd="/code", check=True)
    vol.commit()
    print("checkpoints committed to volume /ckpt")


@app.function(image=image, gpu="L4", cpu=8.0, memory=16384,
              volumes={"/vol": vol}, timeout=3600)
def evaluate(ckpt: str = "", data_file: str = "", evolution: int = 0,
             early_detection: bool = False, roc_ci: bool = False):
    """Run evaluate.py on the trained model (best.pt) over one subset shard.
    Produces evaluation_summary.json, calibration_metrics.json, metrics_by_parameter.json
    (+ plots) into the checkpoint's exp dir on the volume."""
    import subprocess, sys, glob, os
    if not ckpt:
        cks = glob.glob("/vol/ckpt/*/best.pt")
        if not cks:
            raise SystemExit("no best.pt found on the volume")
        ckpt = sorted(cks, key=os.path.getmtime)[-1]
    if not data_file:
        data_file = sorted(glob.glob("/vol/data/subset_*.h5"))[0]
    cmd = [
        sys.executable, "/code/evaluate.py",
        "--experiment-name", ckpt,          # Case-1: direct .pt path
        "--data", data_file,
        "--device", "cuda", "--batch-size", "256",
        "--n-evolution-per-type", str(evolution),
        "--colorblind-safe", "--verbose",   # all-on (skip --use-latex: no LaTeX on image)
        "--save-formats", "png",
    ]
    if early_detection:
        cmd.append("--early-detection")
    if not roc_ci:
        cmd.append("--no-roc-bootstrap-ci")
    print("EVAL:", " ".join(cmd))
    subprocess.run(cmd, cwd="/code", check=True)
    vol.commit()
    # list what got written
    exp = os.path.dirname(ckpt)
    for root, _dirs, files in os.walk(exp):
        for f in files:
            if f.endswith((".json", ".png", ".txt")):
                print("wrote:", os.path.join(root, f))


@app.function(image=image, gpu="L4", cpu=8.0, memory=16384,
              volumes={"/vol": vol}, timeout=3600)
def eval_val(ckpt: str = ""):
    """Evaluate on the exact HELD-OUT val split (never trained on). Reproduces
    train.py's deterministic SEED=42 / val_fraction=0.1 split, writes those 30k events
    to /vol/val.h5 (compact format), then runs evaluate.py on it."""
    import sys, glob, os, numpy as np, h5py
    sys.path.insert(0, "/code")
    import train as T

    shard_paths = sorted(glob.glob("/vol/data/subset_*.h5"))
    _tr, val_idx, _tl, _al = T.stream_train_val_split(shard_paths, 0.1, T.SEED)
    val_idx = np.sort(np.asarray(val_idx))
    print(f"held-out val events: {len(val_idx)}")

    # cumulative offsets -> map global idx to (shard, local row)
    lens = [h5py.File(p, "r")["labels"].shape[0] for p in shard_paths]
    offs = np.concatenate([[0], np.cumsum(lens)])
    with h5py.File(shard_paths[0], "r") as f0:
        n_pts = f0["flux"].shape[1]; tg = f0["time_grid"][:]
        pdt = f0["params"].dtype; nmb = f0["mask"].shape[1]

    # gather per source shard (bounded reads)
    buf = {k: [] for k in ("flux", "dt", "lab", "mb", "mask", "par")}
    by_src = {}
    for g in val_idx:
        s = int(np.searchsorted(offs, g, side="right") - 1)
        by_src.setdefault(s, []).append(g - int(offs[s]))
    # keep output in val_idx order
    order_map = {}
    for s in sorted(by_src):
        rows = np.sort(np.array(by_src[s]))
        with h5py.File(shard_paths[s], "r") as f:
            fl = f["flux"][rows]; dt = f["delta_t"][rows]; lb = f["labels"][rows]
            mb = f["m_base"][rows]; mk = f["mask"][rows]; pr = f["params"][rows]
        for j, r in enumerate(rows):
            order_map[int(offs[s]) + int(r)] = (fl[j], dt[j], int(lb[j]), float(mb[j]), mk[j], pr[j])
    fl = np.stack([order_map[int(g)][0] for g in val_idx])
    dt = np.stack([order_map[int(g)][1] for g in val_idx])
    lb = np.array([order_map[int(g)][2] for g in val_idx], np.int32)
    mb = np.array([order_map[int(g)][3] for g in val_idx], np.float32)
    mk = np.stack([order_map[int(g)][4] for g in val_idx])
    pr = np.array([order_map[int(g)][5] for g in val_idx], dtype=pdt)

    out = "/vol/val.h5"
    with h5py.File(out, "w") as g:
        c = {"compression": "lzf"}
        g.create_dataset("flux", data=fl, chunks=(min(256, len(fl)), n_pts), **c)
        g.create_dataset("delta_t", data=dt, chunks=(min(256, len(fl)), n_pts), **c)
        g.create_dataset("labels", data=lb)
        g.create_dataset("m_base", data=mb, **c)
        g.create_dataset("mask", data=mk, **c)
        g.create_dataset("params", data=pr, **c)
        g.create_dataset("time_grid", data=tg)
        g.attrs.update({"n_events": len(lb), "has_global_params": True,
                        "has_time_grid": True, "timestamps_dropped": True, "n_points": int(n_pts)})
    print(f"wrote {out}: {len(lb)} held-out events, classes={dict(zip(*np.unique(lb, return_counts=True)))}")
    vol.commit()
    evaluate.local(ckpt=ckpt, data_file=out)


@app.function(image=image, gpu="L4", cpu=8.0, memory=16384,
              volumes={"/vol": vol}, timeout=3600)
def eval_presets(ckpt: str = ""):
    """Evaluate the model separately on each binary preset (baseline/planetary/distinct/
    stellar), so binary recall can be compared by regime. Writes eval_<preset>/ dirs."""
    import subprocess, sys, glob, os
    if not ckpt:
        ckpt = sorted(glob.glob("/vol/ckpt/*/best.pt"), key=os.path.getmtime)[-1]
    for p in ("baseline", "planetary", "distinct", "stellar"):
        df = f"/vol/test_{p}.h5"
        if not os.path.exists(df):
            print("MISSING", df); continue
        print(f"=== evaluating preset {p} ===")
        subprocess.run([
            sys.executable, "/code/evaluate.py",
            "--experiment-name", ckpt, "--data", df,
            "--device", "cuda", "--batch-size", "256",
            "--n-evolution-per-type", "0", "--no-roc-bootstrap-ci",
            "--output-dir", f"/vol/eval_{p}", "--save-formats", "png",
        ], cwd="/code", check=True)
    vol.commit()
    print("=== ALL PRESET EVALS DONE ===")


@app.function(image=image, gpu="L4", cpu=8.0, memory=16384,
              volumes={"/vol": vol}, timeout=86400)
def finetune(ckpt: str = "", lr: float = 2e-4, epochs: int = 12,
             data_dir: str = "/vol/finetune", out_dir: str = "/vol/ckpt_ft",
             patience: int = 5):
    """Warm-start fine-tune from a checkpoint (--init-weights, NOT --resume) on the bump-rich
    shards in data_dir, fresh low-LR schedule. Re-chunks shards to (256,6912) for fast reads.
    ckpt default = latest BASE best.pt (so round-2 from base_best works without an arg)."""
    import subprocess, sys, os, glob, shutil
    if not ckpt:
        ckpt = sorted(glob.glob("/vol/ckpt/*/best.pt"), key=os.path.getmtime)[-1]
    import h5py
    # simulate.py writes chunks=(10000, 6912) -- a 276 MB lzf chunk, so random block-shuffle
    # reads decompress a quarter-GB per batch (catastrophic). Re-chunk to (256, 6912).
    local = "/ft_local_" + os.path.basename(out_dir); os.makedirs(local, exist_ok=True)
    for f in sorted(glob.glob(f"{data_dir}/*.h5")):
        dst = os.path.join(local, os.path.basename(f))
        if os.path.exists(dst):
            continue
        with h5py.File(f, "r") as src, h5py.File(dst + ".tmp", "w") as dstf:
            for k in src.keys():
                d = src[k]
                if d.ndim == 2 and d.shape[0] == src["labels"].shape[0]:
                    dstf.create_dataset(k, data=d[:], chunks=(min(256, d.shape[0]), d.shape[1]),
                                        compression="lzf")
                elif d.ndim == 1 and d.shape[0] == src["labels"].shape[0]:
                    dstf.create_dataset(k, data=d[:], chunks=(min(256, d.shape[0]),),
                                        compression="lzf")
                else:
                    dstf.create_dataset(k, data=d[:])
            for a, v in src.attrs.items():
                dstf.attrs[a] = v
        os.replace(dst + ".tmp", dst)
        print("re-chunked", os.path.basename(f))
    print("ft shards:", len(glob.glob(local + "/*.h5")), "warm-start from", ckpt, "-> out", out_dir)
    cmd = [
        sys.executable, "/code/train.py",
        "--stream", "--block-shuffle", "256",
        "--init-weights", ckpt,
        "--data", local, "--output", out_dir,
        "--hierarchical", "--use-aux-head", "--attention-pooling", "--use-amp",
        "--d-model", "64", "--n-layers", "4",
        "--batch-size", "128", "--accumulation-steps", "2",
        "--num-workers", "6", "--prefetch-factor", "6",
        "--stage1-weight", "0.5", "--stage2-weight", "2.0", "--aux-weight", "0.5",
        "--lr", str(lr), "--epochs", str(epochs), "--warmup-epochs", "1",
        "--early-stop-patience", str(patience),
    ]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd="/code", check=True)
    vol.commit()
    print(f"fine-tuned checkpoints committed to {out_dir}")


@app.function(image=image, gpu="L4", cpu=8.0, memory=16384,
              volumes={"/vol": vol}, timeout=7200)
def eval_lsst(ckpt: str = "", data_file: str = "/vol/test_baseline.h5",
              gaps: str = "0,1,3,5", noise_mag: float = 0.0, batch: int = 256):
    """Domain-shift test: resample dense Roman-cadence synthetic events onto LSST-like
    sparse/irregular cadences and measure the 3-CLASS behaviour. Framing: PSPL is the
    'is this microlensing at all' proxy and Binary is a distinct anomalous class, so we
    report (1) DETECTION = fraction of true PSPL+Binary NOT called Flat, and
    (2) CHARACTERISATION = binary<->PSPL confusion, in addition to per-class recall.
    gaps: comma list of mean inter-visit gaps in days; 0 = full Roman cadence (baseline)."""
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import glob, json, numpy as np, torch, h5py, sys
    sys.path.insert(0, "/code")
    from model import ModelConfig, RomanMicrolensingClassifier
    dev = "cuda"; EPS = 1e-8; CN = ["Flat", "PSPL", "Binary"]
    if not ckpt:
        ckpt = sorted(glob.glob("/vol/ckpt/*/best.pt"), key=os.path.getmtime)[-1]
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    model = RomanMicrolensingClassifier(cfg).to(dev)
    sd = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in ck["model_state_dict"].items()}
    model.load_state_dict(sd, strict=True); model.eval()
    s = ck["stats"]
    fm = float(s.get("flux_mean", s.get("magnification_mean"))); fs = float(s.get("flux_std", s.get("magnification_std")))
    dm = float(s["delta_t_mean"]); ds = float(s["delta_t_std"])

    with h5py.File(data_file, "r") as f:
        flux = f["flux"][:]; lab = f["labels"][:].astype(np.int64); tg = f["time_grid"][:].astype(np.float64)
    N, S = flux.shape
    window = float(tg[-1] - tg[0])
    print(f"LSST test on {os.path.basename(data_file)}: {N} events, window={window:.0f}d, "
          f"classes={dict(zip(*[x.tolist() for x in np.unique(lab, return_counts=True)]))}")

    def resample_lsst(A_row, rng, mean_gap):
        """Return compacted (flux[S], delta_t[S], length) for one event at LSST cadence."""
        valid = A_row != 0
        tv = tg[valid]; Av = A_row[valid].astype(np.float64)
        if len(tv) < 2:
            return None
        if mean_gap <= 0:                          # full Roman cadence (all valid points)
            k = len(Av); fc = np.zeros(S, np.float32); dc = np.zeros(S, np.float32)
            fc[:k] = Av; dc[1:k] = np.diff(tv); return fc, dc, k
        # irregular LSST visit times over the window
        t = rng.uniform(0, mean_gap); ts = []
        while t < window:
            ts.append(t); t += rng.exponential(mean_gap)
        ts = np.array(ts)
        ts = ts[rng.random(len(ts)) > 0.25]        # ~25% weather/other loss
        if len(ts) < 2:
            return None
        j = np.clip(np.searchsorted(tv, ts), 1, len(tv) - 1)
        near = np.where(np.abs(tv[j] - ts) < np.abs(tv[j - 1] - ts), j, j - 1)
        near = np.unique(near)                      # dedupe collisions -> sorted
        A_s = Av[near].copy(); t_s = tv[near]
        if noise_mag > 0:                           # optional extra LSST photometric noise
            A_s *= 10 ** (-0.4 * rng.normal(0, noise_mag, len(A_s)))
        k = len(A_s); fc = np.zeros(S, np.float32); dc = np.zeros(S, np.float32)
        fc[:k] = A_s; dc[1:k] = np.diff(t_s); return fc, dc, k

    results = {}
    for gap in [float(x) for x in gaps.split(",")]:
        rng = np.random.RandomState(1234)
        conf = np.zeros((3, 3), np.int64); nlen = []
        fbuf, dbuf, lbuf, ybuf = [], [], [], []

        def flush():
            if not fbuf:
                return
            fnp = (np.stack(fbuf) - fm) / (fs + EPS); dnp = (np.stack(dbuf) - dm) / (ds + EPS)
            with torch.no_grad():
                out = model(torch.from_numpy(fnp).float().to(dev),
                            torch.from_numpy(dnp).float().to(dev),
                            torch.tensor(lbuf, device=dev))
                logit = out if torch.is_tensor(out) else out["logits"]
                pr = logit.argmax(-1).cpu().numpy()
            for t, p in zip(ybuf, pr):
                conf[t, p] += 1
            fbuf.clear(); dbuf.clear(); lbuf.clear(); ybuf.clear()

        for i in range(N):
            r = resample_lsst(flux[i], rng, gap)
            if r is None:
                continue
            fc, dc, k = r; nlen.append(k)
            fbuf.append(fc); dbuf.append(dc); lbuf.append(k); ybuf.append(int(lab[i]))
            if len(fbuf) >= batch:
                flush()
        flush()
        rec = {CN[i]: float(conf[i, i] / max(conf[i].sum(), 1)) for i in range(3)}
        # DETECTION: true lensing (PSPL or Binary) not misclassified as Flat
        lens = conf[1:, :].sum()
        det = float(conf[1:, 1:].sum() / max(lens, 1))
        # CHARACTERISATION: among true binaries, fraction called binary vs leaked to PSPL
        b = conf[2]; bin_as_pspl = float(b[1] / max(b.sum(), 1)); bin_as_bin = float(b[2] / max(b.sum(), 1))
        # among true PSPL, fraction leaked to binary
        p = conf[1]; pspl_as_bin = float(p[2] / max(p.sum(), 1))
        acc = float(np.trace(conf) / max(conf.sum(), 1))
        results[f"gap_{gap:g}d"] = dict(
            mean_points=float(np.mean(nlen)) if nlen else 0, accuracy=acc, recall=rec,
            detection_rate=det, binary_recall=bin_as_bin, binary_leaked_to_pspl=bin_as_pspl,
            pspl_leaked_to_binary=pspl_as_bin, confusion=conf.tolist())
        tag = "ROMAN full" if gap == 0 else f"LSST ~{gap:g}d gap"
        print(f"\n=== {tag}  (mean {np.mean(nlen):.0f} pts/event) ===")
        print(f"  acc={acc*100:.1f}%  recall Flat={rec['Flat']*100:.0f} PSPL={rec['PSPL']*100:.0f} Binary={rec['Binary']*100:.0f}")
        print(f"  DETECTION (lensing not called Flat) = {det*100:.1f}%")
        print(f"  CHARACTERISATION: binary->binary={bin_as_bin*100:.0f}%  binary->PSPL(missed)={bin_as_pspl*100:.0f}%  PSPL->binary(false)={pspl_as_bin*100:.0f}%")
    out = {"data_file": data_file, "ckpt": ckpt, "noise_mag": noise_mag, "by_cadence": results}
    with open("/vol/eval_lsst_results.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit(); print("\nsaved /vol/eval_lsst_results.json")


@app.function(image=image, gpu="L4", cpu=8.0, memory=16384,
              volumes={"/vol": vol}, timeout=7200)
def eval_detectability(ckpt: str = "", data_dir: str = "/vol/data_1m",
                       max_shards: int = 8, batch: int = 128):
    """Score binary recall CONDITIONED ON DETECTABILITY. A binary whose caustic is not
    sampled/perturbing (low anomaly_dchi2) is observationally a PSPL, so calling it PSPL is
    correct, not a miss. We bin binary recall by anomaly_dchi2 and report the
    'indistinguishable' fraction and the detectable-only binary recall (the fair ceiling)."""
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import glob, json, numpy as np, torch, h5py, sys
    sys.path.insert(0, "/code")
    from model import ModelConfig, RomanMicrolensingClassifier
    dev = "cuda"; EPS = 1e-8
    if not ckpt:
        ckpt = sorted(glob.glob("/vol/ckpt_ft/*/best.pt"), key=os.path.getmtime)[-1]
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    net = RomanMicrolensingClassifier(ModelConfig(**ck["model_config"])).to(dev)
    sd = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in ck["model_state_dict"].items()}
    net.load_state_dict(sd, strict=True); net.eval()
    s = ck["stats"]
    fm = float(s.get("flux_mean", s.get("magnification_mean"))); fs = float(s.get("flux_std", s.get("magnification_std")))
    dm = float(s["delta_t_mean"]); ds = float(s["delta_t_std"])

    shards = sorted(glob.glob(f"{data_dir}/subset_*.h5"))[:max_shards]
    an_all, pred_all, q_all = [], [], []
    for sh in shards:
        with h5py.File(sh, "r") as f:
            N = f["labels"].shape[0]; lab = f["labels"][:]
            bmask = lab == 2
            idx = np.where(bmask)[0]
            an = f["params"]["anomaly_dchi2"][:][idx]; qq = f["params"]["q"][:][idx]
            for s0 in range(0, len(idx), batch):
                rows = idx[s0:s0 + batch]
                flux = torch.from_numpy(f["flux"][rows]).float().to(dev)
                dt = torch.from_numpy(f["delta_t"][rows]).float().to(dev)
                valid = (flux != 0); lengths = valid.sum(1).clamp(min=1)
                perm = torch.argsort(valid.int(), dim=1, descending=True, stable=True)
                fc = torch.gather(flux, 1, perm); dc = torch.gather(dt, 1, perm)
                with torch.no_grad():
                    out = net((fc - fm) / (fs + EPS), (dc - dm) / (ds + EPS), lengths)
                    logit = out if torch.is_tensor(out) else out["logits"]
                    pred_all.append(logit.argmax(-1).cpu().numpy())
            an_all.append(an); q_all.append(qq)
    an = np.concatenate(an_all); pred = np.concatenate(pred_all); q = np.concatenate(q_all)
    caught = pred == 2
    print(f"binaries evaluated: {len(an)} (from {len(shards)} shards), model {os.path.basename(os.path.dirname(ckpt))}")
    # detectability thresholds
    out = {"n_binaries": int(len(an)), "overall_binary_recall": float(caught.mean()), "thresholds": {}}
    for thr in [20, 50, 100, 300, 1000]:
        detectable = an >= thr; indist = ~detectable
        rec_det = float(caught[detectable].mean()) if detectable.sum() else float("nan")
        rec_ind = float(caught[indist].mean()) if indist.sum() else float("nan")
        # of the MISSED binaries, what fraction are indistinguishable?
        missed = ~caught
        frac_missed_indist = float(indist[missed].mean()) if missed.sum() else float("nan")
        out["thresholds"][str(thr)] = dict(
            indistinguishable_frac=float(indist.mean()),
            recall_detectable=rec_det, recall_indistinguishable=rec_ind,
            frac_of_missed_that_are_indistinguishable=frac_missed_indist)
        print(f"  dchi2>={thr:5d}: indistinguishable={indist.mean()*100:4.1f}%  "
              f"recall(detectable)={rec_det*100:5.1f}%  recall(indist)={rec_ind*100:5.1f}%  "
              f"| of missed, {frac_missed_indist*100:4.1f}% are indistinguishable")
    # recall in anomaly_dchi2 bins
    bins = [0, 20, 50, 100, 300, 1000, 1e4, 1e12]
    out["by_anomaly_bin"] = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (an >= lo) & (an < hi)
        if m.sum():
            out["by_anomaly_bin"][f"{lo:g}-{hi:g}"] = dict(n=int(m.sum()), recall=float(caught[m].mean()))
    with open("/vol/eval_detectability.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit(); print("saved /vol/eval_detectability.json")


DATASET_1M = "kunalb541/roman-microlensing-1m"


@app.function(image=image, volumes={"/vol": vol}, secrets=[kaggle_secret], timeout=7200)
def download_1m():
    import os, glob
    os.makedirs("/vol/data_1m", exist_ok=True)
    have = glob.glob("/vol/data_1m/subset_*.h5")
    if len(have) >= 30:
        print("already have", len(have), "1M shards; skipping")
    else:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        print("downloading", DATASET_1M, "...")
        api.dataset_download_files(DATASET_1M, path="/vol/data_1m", unzip=True, quiet=False)
    vol.commit()
    shards = glob.glob("/vol/data_1m/subset_*.h5")
    import h5py
    n = sum(h5py.File(s, "r")["labels"].shape[0] for s in shards)
    print(f"1M shards: {len(shards)}  total events: {n}")


@app.function(image=image, gpu="L4", cpu=8.0, memory=32768,
              volumes={"/vol": vol}, timeout=14400)
def eval_big(ckpts: str = "", data_dir: str = "/vol/data_1m", batch: int = 128):
    """Stream every shard in data_dir through one or more checkpoints on GPU and report
    per-class recall + confusion + binary-recall-binned-by-q, over ALL events. Compaction
    is vectorised on-GPU (argsort of the valid mask) so 1M events run fast."""
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import glob, json, numpy as np, torch, h5py, sys
    sys.path.insert(0, "/code")
    from model import ModelConfig, RomanMicrolensingClassifier
    dev = "cuda"
    EPS = 1e-8; CN = ["Flat", "PSPL", "Binary"]

    if not ckpts:
        base = sorted(glob.glob("/vol/ckpt/*/best.pt"), key=os.path.getmtime)[-1]
        ft = sorted(glob.glob("/vol/ckpt_ft/*/best.pt"), key=os.path.getmtime)[-1]
        ckpts = f"{base},{ft}"
    ckpt_list = ckpts.split(",")

    def load(p):
        ck = torch.load(p, map_location=dev, weights_only=False)
        cfg = ModelConfig(**ck["model_config"])
        m = RomanMicrolensingClassifier(cfg).to(dev)
        sd = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in ck["model_state_dict"].items()}
        m.load_state_dict(sd, strict=True); m.eval()
        s = ck["stats"]
        fm = s.get("flux_mean", s.get("magnification_mean")); fs = s.get("flux_std", s.get("magnification_std"))
        return m, (float(fm), float(fs), float(s["delta_t_mean"]), float(s["delta_t_std"]))

    models = {p: load(p) for p in ckpt_list}
    shards = sorted(glob.glob(f"{data_dir}/subset_*.h5"))
    print(f"evaluating {len(shards)} shards with {len(ckpt_list)} checkpoint(s)")

    conf = {p: np.zeros((3, 3), np.int64) for p in ckpt_list}
    all_lab, all_q, all_t0 = [], [], []     # t0 = peak time, for the temporal-bias KS test
    all_pred = {p: [] for p in ckpt_list}
    seen = 0
    for si, sh in enumerate(shards):
        with h5py.File(sh, "r") as f:
            N = f["labels"].shape[0]
            has_q = "params" in f and "q" in f["params"].dtype.names
            for s0 in range(0, N, batch):
                e = min(s0 + batch, N)
                flux = torch.from_numpy(f["flux"][s0:e]).float().to(dev)
                dt = torch.from_numpy(f["delta_t"][s0:e]).float().to(dev)
                lab = f["labels"][s0:e].astype(np.int64)
                # vectorised compaction: stable-sort valid(=nonzero) to the front
                valid = (flux != 0)
                lengths = valid.sum(1).clamp(min=1)
                perm = torch.argsort(valid.int(), dim=1, descending=True, stable=True)
                fluxc = torch.gather(flux, 1, perm)
                dtc = torch.gather(dt, 1, perm)
                all_lab.append(lab)
                pnames = f["params"].dtype.names if "params" in f else ()
                all_q.append(f["params"]["q"][s0:e].astype(np.float32) if "q" in pnames
                             else np.full(e - s0, np.nan, np.float32))
                all_t0.append(f["params"]["t0"][s0:e].astype(np.float32) if "t0" in pnames
                              else np.full(e - s0, np.nan, np.float32))
                for p, (m, st) in models.items():
                    fn = (fluxc - st[0]) / (st[1] + EPS)
                    dn = (dtc - st[2]) / (st[3] + EPS)
                    with torch.no_grad():
                        out = m(fn, dn, lengths)
                        logit = out if torch.is_tensor(out) else out["logits"]
                        pr = logit.argmax(-1).cpu().numpy()
                    all_pred[p].append(pr)
                    for t, pc in zip(lab, pr):
                        conf[p][t, pc] += 1
                seen += (e - s0)
        print(f"  shard {si+1}/{len(shards)} done, {seen} events")

    from scipy.stats import ks_2samp
    lab = np.concatenate(all_lab); q = np.concatenate(all_q); t0 = np.concatenate(all_t0)
    qbins = [(0, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.0)]
    results = {"n_total": int(seen), "class_counts": {CN[c]: int((lab == c).sum()) for c in range(3)}, "by_ckpt": {}}
    for p in ckpt_list:
        pred = np.concatenate(all_pred[p]); c = conf[p]
        recall = {CN[i]: float(c[i, i] / max(c[i].sum(), 1)) for i in range(3)}
        prec = {CN[i]: float(c[i, i] / max(c[:, i].sum(), 1)) for i in range(3)}
        acc = float((pred == lab).mean())
        # binary recall by q bin
        qb = {}
        bmask = lab == 2
        for lo, hi in qbins:
            m2 = bmask & (q >= lo) & (q < hi)
            if m2.sum() > 0:
                qb[f"{lo:g}-{hi:g}"] = dict(n=int(m2.sum()), recall=float((pred[m2] == 2).mean()))
        # TEMPORAL-BIAS KS TEST (matches evaluate.py plot_temporal_bias_check): over events
        # with a peak (non-Flat), is the t0 distribution of CORRECTLY classified events the
        # same as INCORRECTLY classified? A significant difference (p<0.05) => accuracy is
        # biased by where in the 72-day window the event peaks. p>=0.05 => no temporal bias.
        correct = (pred == lab)
        has_peak = (lab != 0) & np.isfinite(t0)
        t0_ok = t0[has_peak & correct]; t0_bad = t0[has_peak & ~correct]
        if len(t0_ok) >= 2 and len(t0_bad) >= 2:
            r = ks_2samp(t0_ok, t0_bad)
            ks = {"D": float(r.statistic), "pvalue": float(r.pvalue),
                  "n_correct": int(len(t0_ok)), "n_incorrect": int(len(t0_bad)),
                  "verdict": "BIAS DETECTED" if r.pvalue < 0.05 else "NO BIAS"}
        else:
            ks = None
        results["by_ckpt"][p] = dict(accuracy=acc, recall=recall, precision=prec,
                                     confusion=c.tolist(), binary_recall_by_q=qb,
                                     temporal_bias_ks=ks)
        print(f"\n=== {p} ===")
        print(f"  overall acc={acc*100:.2f}%  recall F={recall['Flat']*100:.1f} "
              f"P={recall['PSPL']*100:.1f} B={recall['Binary']*100:.1f}")
        if ks:
            print(f"  temporal-bias KS: D={ks['D']:.4f}  p={ks['pvalue']:.3e}  -> {ks['verdict']} "
                  f"(n_ok={ks['n_correct']}, n_bad={ks['n_incorrect']})")
        for k, v in qb.items():
            print(f"    q {k}: n={v['n']:6d}  binary recall={v['recall']*100:.1f}%")
    with open("/vol/eval_1m_results.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
    print("\nsaved /vol/eval_1m_results.json")


@app.local_entrypoint()
def main():
    download.remote()
    train.remote()

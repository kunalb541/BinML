"""BinML v5 — in-region per-shard inference.

Runs a trained checkpoint over a single BINNED cache shard (h5) and writes only the compact
prediction arrays (logits + the columns the metrics need). Purpose: evaluate a huge simulated
set in-region on the generation/binning fleet, so only ~30 floats/event ever leave S3 instead
of the multi-KB light curves. The heavy metric/plot code (evaluate, plots) then runs on
these small arrays locally, exactly as it already does on the saved artifact.

CPU-only: the model is 505k params over 156 tokens, so a 2-vCPU box classifies a 7,500-event
shard in seconds. No GPU, no memmap -- one h5 read, one batched forward pass.
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import torch

from .model import BAND_BINS, BinMLv5, ModelConfigV5

MAG_SCALE = 1.0


def _load_ckpt(path: str, device: str):
    ck = torch.load(path, map_location="cpu")
    cfg = ModelConfigV5(**{k: v for k, v in ck["config"].items()
                           if k in ModelConfigV5.__dataclass_fields__})
    m = BinMLv5(cfg).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


@torch.inference_mode()
def eval_shard(cache_h5: str, ckpt: str, out_npz: str, device: str = "cpu",
               batch: int = 2048) -> dict:
    model = _load_ckpt(ckpt, device)
    f = h5py.File(cache_h5, "r")
    n = int(f.attrs["n_events"])
    logits = np.empty((n, 6), dtype=np.float32)
    # preload band arrays once (fp16 on disk, promote per batch)
    feat = {b: f[f"feat/{b}"] for b in BAND_BINS}
    frac = {b: f[f"frac/{b}"] for b in BAND_BINS}
    for s in range(0, n, batch):
        e = min(s + batch, n)
        feats = {}
        present = {}
        for b, L in BAND_BINS.items():
            x = np.asarray(feat[b][s:e], dtype=np.float32)          # (B,L,3)
            fr = np.asarray(frac[b][s:e], dtype=np.float32)         # (B,L)
            obs = np.isfinite(x[:, :, 0]).astype(np.float32)
            np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            t = np.concatenate([x / MAG_SCALE, fr[:, :, None], obs[:, :, None]], axis=2)
            feats[b] = torch.tensor(t, dtype=torch.float32, device=device)
            present[b] = feats[b][..., 4].sum(dim=1) > 0
        logits[s:e] = model(feats, present).float().cpu().numpy()
    out = {"logits": logits}
    for k in ("label", "true_class", "keep_prob", "dchi2_event", "dchi2_anomaly",
              "m_base_ref", "a_ks", "params"):
        if k in f:
            out[k] = np.asarray(f[k])
    pf = f.attrs.get("param_fields")
    if pf is not None:
        # store as fixed-width ASCII, not an object array -- object dtype forces
        # allow_pickle on load and emits a "metadata not saved" warning
        out["param_fields"] = np.array([str(x) for x in pf], dtype="S16")
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(out_npz, **out)
    f.close()
    return {"n": n, "pred_mix": np.bincount(logits.argmax(1), minlength=6).tolist()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    r = eval_shard(a.cache, a.ckpt, a.out, a.device)
    print(f"eval {a.cache} -> {a.out}: n={r['n']} pred_mix={r['pred_mix']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

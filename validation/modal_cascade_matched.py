"""Cascade ablation at MATCHED detection rate -- the experiment the paper names as its own
clearest outstanding weakness.

THE PROBLEM WITH THE THRESHOLD-ONLY COMPARISON.  The first cascade ablation compared the
truncation-augmented arm against the un-augmented one at a single frozen threshold, and found the
augmented arm both alerted prematurely less often AND detected far less often (0.643 vs 0.993
within-season). Those two facts have one obvious joint explanation: the augmented model simply says
NonPSPL less. A model that never alerts has a premature rate of zero. So the comparison cannot
distinguish "better temporal ordering" from "more conservative", and the paper withdrew the causal
claim rather than assert it.

WHAT THIS RUN DOES INSTEAD.  It stores the full P(NonPSPL) trace for each arm over the same events,
then sweeps each arm's threshold independently to hit a COMMON within-season detection rate, and
compares premature rates there. At matched detection the conservatism explanation is removed by
construction: if the augmented arm still alerts prematurely less often at the same recall, the
ordering is real; if the curves coincide, it was a threshold shift all along.

Both arms' checkpoints come from the original four-arm run (Modal volume binml-ablation-data,
trained on identical shards with identical hyper-parameters and seed, differing only in
--truncate-aug). Nothing is retrained here, so the comparison inherits that run's controls.

Run:  modal run --detach validation/modal_cascade_matched.py
      Do NOT wrap in `timeout`; the SIGTERM reaches the container as a cancellation.
"""
import hashlib
import json
import os

import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPS = ["torch==2.5.1", "numpy==1.26.4", "scipy==1.14.1", "h5py==3.12.1",
        "VBBinaryLensing==3.7.0"]
CFG = {"n_events": 400, "step_days": 0.5, "seed_base": 770000,
       "arms": ["cascade_on", "cascade_off"], "protocol": "matched-detection-v1"}
# The original run did not write a ckpt_cascade_on.pt: the cascade_on arm was configured
# identically to labels_observational, so it reused that checkpoint rather than training a
# duplicate. Map it explicitly here instead of guessing a filename that never existed.
CKPT = {"cascade_on": "ckpt_labels_observational.pt", "cascade_off": "ckpt_cascade_off.pt"}


def _source_hash(root: str) -> str:
    """Hash every local source file that can change the scan or model inference."""
    h = hashlib.sha256()
    n = 0
    for sub in ("pipeline", "binml"):
        for dirpath, dirnames, filenames in os.walk(os.path.join(root, sub)):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for name in sorted(filenames):
                if name.endswith(".py"):
                    path = os.path.join(dirpath, name)
                    h.update(os.path.relpath(path, root).encode())
                    h.update(open(path, "rb").read())
                    n += 1
    protocol = os.path.join(root, "validation", "modal_cascade_matched.py")
    if os.path.exists(protocol):
        h.update(os.path.relpath(protocol, root).encode())
        h.update(open(protocol, "rb").read())
        n += 1
    if not n:
        raise RuntimeError(f"no source found under {root}; refusing an empty source hash")
    return h.hexdigest()


SOURCE_HASH = _source_hash(REPO)

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install(*DEPS)
         .add_local_dir(os.path.join(REPO, "pipeline"), "/repo/pipeline", copy=True)
         .add_local_dir(os.path.join(REPO, "binml"), "/repo/binml", copy=True)
         .add_local_file(__file__, "/repo/validation/modal_cascade_matched.py", copy=True))
app = modal.App("binml-cascade-matched")
vol = modal.Volume.from_name("binml-ablation-data", create_if_missing=False)
VOL = "/data"


@app.function(image=image, gpu="T4", cpu=8.0, timeout=14400, volumes={VOL: vol})
def scan(source_hash: str) -> dict:
    """Full traces for both arms on one shared event sample."""
    import sys
    import warnings
    import numpy as np
    os.chdir("/repo"); sys.path.insert(0, "/repo"); warnings.simplefilter("ignore")
    actual_source_hash = _source_hash("/repo")
    if actual_source_hash != source_hash:
        raise RuntimeError(f"image source hash {actual_source_hash} != requested {source_hash}")
    vol.reload()
    import binml
    from binml.preprocess import to_tokens
    from pipeline.assemble import (SurveyConfig, _anomaly_onset_day, simulate_event)
    from pipeline.model import BAND_BINS

    cfg = SurveyConfig()
    cuts = np.arange(CFG["step_days"], cfg.window_days + 1e-9, CFG["step_days"])
    nc = len(cuts)

    # ONE event sample, shared by both arms, drawn exactly as the original ablation drew it.
    evs, s = [], CFG["seed_base"]
    while len(evs) < CFG["n_events"] and s < CFG["seed_base"] + 60 * CFG["n_events"]:
        s += 1
        e = simulate_event("NonPSPL", np.random.default_rng(s), cfg, _return_ref_truth=True)
        if not isinstance(e, tuple):
            continue
        ev, ref = e
        if ev is None or ev.label != "NonPSPL":
            continue
        ta = ev.params.get("t_anom")
        if ta is None or not np.isfinite(ta):
            continue
        # onset on the SAME grid as the alerts, as in the production cascade analysis
        evs.append((ev, float(_anomaly_onset_day(ref, ev.params, cfg, n_cuts=nc)), s))
    print(f"[scan] {len(evs)} eligible events", flush=True)

    def trace(ckpt):
        clf = binml.Classifier(weights=ckpt, device="cuda")
        P = np.full((len(evs), nc), np.nan, np.float32)
        for k, (ev, _ta, _s) in enumerate(evs):
            b = ev.bands["F146"]; mb = ev.params["_m_base_ref"]
            toks, idx = [], []
            for i, cut in enumerate(cuts):
                m = b.t <= cut
                if int(m.sum()) < 10:
                    continue
                toks.append(to_tokens({"F146": (b.t[m], b.mag[m])}, m_base_ref=mb, t_start=0.0))
                idx.append(i)
            if toks:
                out = clf._forward(
                    {bb: np.stack([t.feat[bb] for t in toks]) for bb in BAND_BINS},
                    {bb: np.stack([t.frac[bb] for t in toks]) for bb in BAND_BINS})
                P[k, idx] = out[:, 2]
            if (k + 1) % 50 == 0:
                print(f"  {k+1}/{len(evs)}", flush=True)
        return P

    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    ckpt_path = {a: f"{VOL}/{CKPT[a]}" for a in CFG["arms"]}
    traces = {a: trace(ckpt_path[a]).tolist() for a in CFG["arms"]}
    res = {"config": CFG, "cuts": cuts.tolist(),
           "provenance": {"source_sha256": source_hash, "deps": DEPS,
                          "checkpoint_sha256": {a: sha(p) for a, p in ckpt_path.items()}},
           "onset": [ta for _e, ta, _s in evs],
           "seeds": [int(sd) for _e, _ta, sd in evs],
           "traces": traces}
    with open(f"{VOL}/matched_traces.json", "w") as f:
        json.dump(res, f)
    vol.commit()
    print("[scan] wrote /data/matched_traces.json", flush=True)
    return {"n_events": len(evs), "n_cuts": nc}


@app.local_entrypoint()
def main():
    """Spawn and exit -- a local client that stays attached forwards any disconnect to the
    container as a cancellation, even under --detach."""
    print(f"[source] {SOURCE_HASH}")
    call = scan.spawn(SOURCE_HASH)
    print(f"[spawned] {call.object_id}; result -> volume binml-ablation-data/matched_traces.json")

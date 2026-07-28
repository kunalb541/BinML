"""
BinML v5 — training.

Four things here are load-bearing and easy to get wrong; each is a decision, not a default.

1. **There is NO `!= 0.0` validity mask.** v4 treated a zero magnification as "missing" and
   compacted it away. The v5 cache stores BASELINE-RELATIVE magnitudes, so a bin with no
   deviation stores exactly 0.0 -- the single most common value in the Flat class, which is
   ~32% of the dataset. Porting that idiom would delete every genuinely flat bin as
   "missing" and destroy the time axis for precisely the class that defines the Flat/event
   boundary. It would not crash. It would train, converge, and be wrong. Validity here is
   `isfinite` (the cache writes NaN for an empty bin), surfaced as an input channel.

2. **keep_prob and class balancing must not both be applied naively.** Binaries whose anomaly
   was undetectable were retained with probability 0.15, so 1/keep_prob restores the
   generated population. But that restoration does two jobs at once: it fixes the INTER-class
   balance and the INTRA-class mix of byproduct vs native rows. Class balancing deliberately
   overrides the inter-class part -- that is what balancing means -- so only the intra-class
   part must survive. Normalising each class weight by the keep_prob-corrected class mass
   (not the raw count) achieves exactly that, and the double-count disappears by
   construction. Multiplying raw class weights by 1/keep_prob would silently over-weight the
   byproduct-heavy classes.

3. **No fitted normalisation statistics.** This project has already shipped one leakage bug
   where normalisation was fit over all rows instead of train-only. Inputs are already
   baseline-relative magnitudes in physically known units, so fixed constants are used and
   stored in the checkpoint. Nothing is fit from data, so nothing can leak.

4. **Accuracy is the wrong headline metric -- and so is bare recall.** Flat+PSPL are ~61% of
   the data, so a model that never predicts NonPSPL still scores well on accuracy. But
   selecting on NonPSPL RECALL fails just as badly in the opposite direction: predicting
   NonPSPL for everything scores 1.000. The first smoke run of this loop did exactly that and
   was selected, at recall 1.000 / precision 0.139. Selection is therefore on NonPSPL F1,
   which has no degenerate optimum at either extreme. Recall is additionally reported
   CONDITIONED ON DETECTABILITY (binned by anomaly delta-chi^2), because many stored events
   are genuinely undetectable and an unconditioned number flatters the model.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def _seed_worker(worker_id):
    """Re-seed each DataLoader worker's RNG. Without this, forked workers inherit an identical
    numpy RNG state, so the random truncation fraction (and thus the detectability relabelling)
    is correlated across workers -- the augmentation degenerates as num_workers grows."""
    import numpy as _np, torch as _t
    info = _t.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "_rng"):
        info.dataset._rng = _np.random.default_rng(info.seed % (2**32))

from .classes import CLASS_NAMES, N_CLASSES
from .model import BAND_BINS, IN_CH, BinMLv5, ModelConfigV5

# Fixed input scaling. Baseline-relative magnitudes: a 1-magnitude deviation is a big event,
# so dividing by 1 mag puts typical signals in [-1, 1] without fitting anything.
MAG_SCALE = 1.0
I_FLAT = CLASS_NAMES.index("Flat")
I_PSPL = CLASS_NAMES.index("PSPL")
I_NON = CLASS_NAMES.index("NonPSPL")
# Same amplitude floor assemble.SurveyConfig uses to call an event undetectable, so a
# truncated window is judged by exactly the rule that produced the labels in the first place.
# Same amplitude floor SurveyConfig uses to declare an event undetectable, applied to the
# revealed span so a truncated window is judged by the rule that produced the labels.
TRUNC_MIN_AMP_MAG = 0.02


def _visible_amplitude(params: np.ndarray, pf_idx: dict, f_s: float, t_cut: float,
                       window: float = 72.0) -> Optional[float]:
    """Peak model amplitude (mag) inside [0, t_cut], from the TRUE parameters.

    This has to be analytic. Deriving it from the stored curve fails: the noise level must be
    estimated from the data, and for a variable whose signal spans the whole window that
    estimate absorbs the signal -- measured on real curves, LongPeriodVar scored BELOW Flat.
    Returns None when the class has no well-defined onset, in which case the label is kept.
    """
    t0 = params[pf_idx["t0"]]; tE = params[pf_idx["tE"]]; u0 = params[pf_idx["u0"]]
    if np.isfinite(t0) and np.isfinite(tE) and np.isfinite(u0) and tE > 0:
        # microlensing: evaluate the PSPL curve on the revealed span only
        t = np.linspace(0.0, max(t_cut, 1e-3), 128)
        u = np.sqrt(u0 ** 2 + ((t - t0) / tE) ** 2)
        A = (u ** 2 + 2.0) / (np.maximum(u, 1e-8) * np.sqrt(u ** 2 + 4.0))
        return float(np.max(np.abs(2.5 * np.log10(np.maximum(1.0 + f_s * (A - 1.0), 1e-8)))))
    ts = params[pf_idx["t_start"]] if "t_start" in pf_idx else np.nan
    if np.isfinite(ts):
        # eruptive: nothing to see before the outburst begins
        return 0.0 if ts > t_cut else float("inf")
    per = params[pf_idx["P"]] if "P" in pf_idx else np.nan
    amp = params[pf_idx["amp_I"]] if "amp_I" in pf_idx else np.nan
    if np.isfinite(per) and np.isfinite(amp) and per > 0:
        # Periodic / long-period: the signal spans the FULL window, but a TRUNCATED window
        # may traverse only a sliver of the cycle. Returning None here (the previous
        # behaviour) kept the LongPeriodVar label on windows where nothing is visible: at a
        # 4.5-day cut, 76% of LongPeriodVar have a visible amplitude below the 0.02 mag floor
        # -- median 0.0079 mag -- so the model was being taught that a flat line is LPV, and
        # it duly reported P(LPV)~0.3 on genuinely flat stars early in a season.
        # PeriodicVar is unaffected in practice (median period 1 day, so a full cycle fits in
        # even a short cut: 0.1% fall below the floor).
        # Peak-to-peak actually traversed over an arc of length 2*pi*min(t_cut/P, 1/2).
        phase = 2.0 * np.pi * min(max(t_cut, 0.0) / per, 0.5)
        return float(amp * (1.0 - np.cos(phase)))
    return None


def _apply_truncation(out: Dict[str, np.ndarray], label: int, rng: np.random.Generator,
                      params: Optional[np.ndarray] = None,
                      pf_idx: Optional[dict] = None, f_s: float = 0.5,
                      min_frac: float = 0.03) -> int:
    """Reveal a random prefix of the season and RE-LABEL by what is observable in it.

    Returning the original label would be a serious error. A PSPL peaking on day 50, cut at
    day 18, shows a flat line -- labelling that "PSPL" trains the model to invent a class
    from absent evidence, which is precisely the failure the whole v5 labelling scheme exists
    to prevent: ``assemble.py`` already relabels an event Flat when its in-window signal is
    undetectable, and truncating the window is just a smaller window.

    So the same rule is re-applied to the visible portion: if the largest baseline-relative
    deviation among revealed bins is below the amplitude floor, the event IS Flat as observed,
    and that is what the model should be taught to say. This now covers the PERIODIC classes
    too -- previously they were exempted, which is what taught the model that a flat-looking
    short window could be LongPeriodVar.

    min_frac is 0.03 rather than 0.15 so that the first ~11 days of a season are in
    distribution; below 15% the model had never seen a training example at all.
    """
    f = float(rng.uniform(min_frac, 1.0))
    for b, x in out.items():
        cut = int(round(f * x.shape[0]))
        x[cut:, :3] = 0.0      # mean/min/max
        x[cut:, 3] = 0.0       # observed fraction
        x[cut:, 4] = 0.0       # observed mask
    if params is None or pf_idx is None or label == I_FLAT:
        return label
    amp = _visible_amplitude(params, pf_idx, f_s, f * 72.0)
    if amp is None:
        return label           # periodic / long-period: signal spans the window, keep label
    if amp < TRUNC_MIN_AMP_MAG:
        return I_FLAT          # nothing observable yet -> Flat
    # THE CASCADE. A binary whose ANOMALY has not yet appeared in the revealed window is,
    # observationally, a plain PSPL: the light curve so far is a smooth Paczynski rise. t_anom
    # is the day the anomaly first becomes detectable (assemble._anomaly_onset_day). Before it,
    # teach PSPL, not NonPSPL -- this is what stops the model flagging binaries before the
    # caustic is on screen (Flat -> PSPL -> NonPSPL as evidence arrives).
    if label == I_NON and "t_anom" in pf_idx:
        ta = params[pf_idx["t_anom"]]
        if np.isfinite(ta) and (f * 72.0) < ta:
            return I_PSPL
    return label


class CacheDataset(Dataset):
    """Serves pre-binned events from memory-mapped fp16 arrays.

    The arrays stay fp16 and memory-mapped on purpose. A 500k-event subset is 4.22 GB in
    fp16; a single `.astype(np.float32)` anywhere in this path would make it 8.44 GB, over
    half of a 16 GB machine, before a single activation is allocated -- which on its own
    reproduces the memory pressure this design exists to avoid. Conversion to a compute
    dtype happens per batch, after indexing.
    """

    def __init__(self, arrays: Dict[str, np.ndarray], labels: np.ndarray,
                 weights: np.ndarray, dchi2_anom: np.ndarray, idx: np.ndarray,
                 truncate_aug: float = 0.0, seed: int = 0,
                 params: Optional[np.ndarray] = None, pf_idx: Optional[dict] = None,
                 f_s_ref: Optional[np.ndarray] = None):
        self.a = arrays
        self.labels = labels
        self.weights = weights
        self.dchi2_anom = dchi2_anom
        self.idx = idx
        self.truncate_aug = truncate_aug
        self._rng = np.random.default_rng(seed)
        self.params = params
        self.pf_idx = pf_idx
        self.f_s_ref = f_s_ref
        if truncate_aug > 0 and (params is None or pf_idx is None):
            raise ValueError(
                "truncate_aug needs params (t0/tE/u0/t_start) to decide whether the event is "
                "visible in the revealed span. Rebuild the cache with the params-carrying "
                "cache.py -- inferring it from the noisy curve does not work.")
        for b in BAND_BINS:
            assert self.a[f"feat/{b}"].dtype == np.float16, "cache must stay fp16"

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        out = {}
        for b in BAND_BINS:
            feat = np.asarray(self.a[f"feat/{b}"][j], dtype=np.float32)   # (L, 3)
            frac = np.asarray(self.a[f"frac/{b}"][j], dtype=np.float32)   # (L,)
            obs = np.isfinite(feat[:, 0]).astype(np.float32)
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0) / MAG_SCALE
            # channel order must match model.CH_MIN / CH_MAX: mean, min, max, frac, mask
            out[b] = np.concatenate([feat, frac[:, None], obs[:, None]], axis=1)
        lab = int(self.labels[j])
        if self.truncate_aug > 0 and self._rng.random() < self.truncate_aug:
            fs = float(self.f_s_ref[j]) if self.f_s_ref is not None else 0.5
            lab = _apply_truncation(out, lab, self._rng, self.params[j], self.pf_idx, fs)
        return (out, lab, float(self.weights[j]), float(self.dchi2_anom[j]))


def collate(batch):
    # np.ascontiguousarray + copy: torch.from_numpy would otherwise alias the stacked buffer,
    # and these rows come from a memory-mapped file.
    feats = {b: torch.from_numpy(np.ascontiguousarray(
                 np.stack([x[0][b] for x in batch]), dtype=np.float32)).clone()
             for b in BAND_BINS}
    for b, t in feats.items():
        if not torch.isfinite(t).all():
            torch.nan_to_num_(t, nan=0.0, posinf=0.0, neginf=0.0)
    # A band is "present" when it yielded any observed bin at all.
    present = {b: (feats[b][..., 4].sum(dim=1) > 0) for b in BAND_BINS}
    y = torch.tensor([x[1] for x in batch], dtype=torch.long)
    w = torch.tensor([x[2] for x in batch], dtype=torch.float32)
    d = torch.tensor([x[3] for x in batch], dtype=torch.float32)
    return feats, present, y, w, d


def compute_weights(labels: np.ndarray, keep_prob: np.ndarray,
                    alpha_nonpspl: float = 2.0) -> np.ndarray:
    """Per-event training weight; see note 2 in the module docstring.

    Returns weights with mean 1 so the effective learning rate is comparable across configs.
    """
    inv = 1.0 / np.clip(keep_prob, 1e-3, 1.0)              # restore the generated population
    n_eff = np.bincount(labels, weights=inv, minlength=N_CLASSES)   # keep_prob-corrected mass
    target = np.ones(N_CLASSES, dtype=np.float64)
    target[CLASS_NAMES.index("NonPSPL")] = alpha_nonpspl   # up-weight the science-critical class
    target = target / target.sum() * N_CLASSES
    cls_w = target / np.maximum(n_eff, 1.0)
    w = inv * cls_w[labels]
    return (w * (len(w) / w.sum())).astype(np.float32)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    conf = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    tot_loss = tot_w = 0.0
    d_all, correct_non, is_non = [], [], []
    for feats, present, y, w, d in loader:
        feats = {k: v.to(device) for k, v in feats.items()}
        present = {k: v.to(device) for k, v in present.items()}
        y, w = y.to(device), w.to(device)
        logits = model(feats, present)
        loss = (F.cross_entropy(logits, y, reduction="none") * w).sum()
        tot_loss += float(loss); tot_w += float(w.sum())
        pred = logits.argmax(1)
        # Vectorised confusion update. The obvious `for t, p in zip(...)` costs one Python
        # iteration per EVENT per epoch -- ~68k on val, ~68k more on test -- which is pure
        # interpreter overhead on top of a GPU-bound loop. bincount does it in one pass.
        conf += np.bincount((y * N_CLASSES + pred).cpu().numpy(),
                            minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)
        i_non = CLASS_NAMES.index("NonPSPL")
        m = (y == i_non)
        if m.any():
            d_all.append(d[m.cpu()].numpy())
            correct_non.append((pred[m] == i_non).cpu().numpy())
        is_non.append((pred == i_non).cpu().numpy())

    recall = np.diag(conf) / np.maximum(conf.sum(1), 1)
    precision = np.diag(conf) / np.maximum(conf.sum(0), 1)
    f1 = 2 * recall * precision / np.maximum(recall + precision, 1e-9)
    out = {"loss": tot_loss / max(tot_w, 1e-9),
           "recall": {CLASS_NAMES[i]: float(recall[i]) for i in range(N_CLASSES)},
           "precision": {CLASS_NAMES[i]: float(precision[i]) for i in range(N_CLASSES)},
           "f1": {CLASS_NAMES[i]: float(f1[i]) for i in range(N_CLASSES)},
           "macro_f1": float(f1.mean()),
           "confusion": conf.tolist()}

    # Detectability-conditioned NonPSPL recall: a raw number mixes in events whose anomaly is
    # physically undetectable, which flatters the model. Report recall per delta-chi^2 decade.
    if d_all:
        d_cat = np.concatenate(d_all); c_cat = np.concatenate(correct_non)
        bins = [(0, 160), (160, 1e3), (1e3, 1e4), (1e4, 1e5), (1e5, np.inf)]
        out["nonpspl_recall_by_dchi2"] = {
            f"{lo:g}-{hi:g}": (float(c_cat[(d_cat >= lo) & (d_cat < hi)].mean())
                               if ((d_cat >= lo) & (d_cat < hi)).any() else None)
            for lo, hi in bins}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="memmap cache directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--alpha-nonpspl", type=float, default=2.0)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--init-weights", default=None)
    ap.add_argument("--truncate-aug", type=float, default=0.0,
                    help="probability of revealing only a random prefix of the season. The "
                         "base model saw F146 100%% observed in EVERY training curve, so a "
                         "partial season is out-of-distribution: at 25%% revealed it predicts "
                         "Eruptive for 100%% of events at 0.985 confidence, because the "
                         "truncation edge reads as an outburst. This augmentation is what "
                         "makes early/online detection possible at all.")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="dataloader workers; safe with a memmap (workers share file pages, "
                         "so there is no per-worker copy of the dataset)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from <out>.last if it exists (for unattended runs)")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)

    meta = json.load(open(os.path.join(args.cache, "meta.json")))
    n = meta["n_events"]
    arrays = {}
    for b in BAND_BINS:
        arrays[f"feat/{b}"] = np.memmap(os.path.join(args.cache, f"feat_{b}.f16"),
                                        dtype=np.float16, mode="r",
                                        shape=(n, BAND_BINS[b], 3))
        arrays[f"frac/{b}"] = np.memmap(os.path.join(args.cache, f"frac_{b}.f16"),
                                        dtype=np.float16, mode="r", shape=(n, BAND_BINS[b]))
    params = pf_idx = f_s_ref = None
    ppath = os.path.join(args.cache, "params.npy")
    if os.path.exists(ppath):
        params = np.load(ppath)
        pf = meta.get("param_fields") or []
        pf_idx = {k: i for i, k in enumerate(pf)}
    fpath = os.path.join(args.cache, "f_s_F146.npy")
    if os.path.exists(fpath):
        f_s_ref = np.load(fpath)
    labels = np.load(os.path.join(args.cache, "label.npy"))
    keep_prob = np.load(os.path.join(args.cache, "keep_prob.npy"))
    dchi2_anom = np.load(os.path.join(args.cache, "dchi2_anomaly.npy"))

    perm = rng.permutation(n)
    if args.max_events and args.max_events < n:
        perm = perm[:args.max_events]
    n_use = len(perm)
    n_tr, n_va = int(0.8 * n_use), int(0.1 * n_use)
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    # Weights are computed from the TRAIN split only; using all rows would leak the test
    # class composition into the training objective.
    w_all = np.zeros(n, dtype=np.float32)
    w_all[tr] = compute_weights(labels[tr], keep_prob[tr], args.alpha_nonpspl)
    w_all[va] = compute_weights(labels[va], keep_prob[va], args.alpha_nonpspl)
    w_all[te] = 1.0

    mk = lambda idx, shuf: DataLoader(
        CacheDataset(arrays, labels, w_all, dchi2_anom, idx,
                     truncate_aug=args.truncate_aug if shuf else 0.0, seed=args.seed,
                     params=params, pf_idx=pf_idx, f_s_ref=f_s_ref),
        batch_size=args.batch_size, shuffle=shuf, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=False,
        worker_init_fn=_seed_worker,          # else forked workers share one RNG -> correlated truncation
        persistent_workers=args.num_workers > 0)
    dl_tr, dl_va, dl_te = mk(tr, True), mk(va, False), mk(te, False)

    cfg = ModelConfigV5(d_model=args.d_model, n_layers=args.n_layers)
    model = BinMLv5(cfg).to(dev)
    if args.init_weights:
        sd = torch.load(args.init_weights, map_location="cpu")
        sd = sd.get("model", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # strict=False silently tolerates a totally mismatched checkpoint, which has bitten
        # this project before -- so report and refuse if essentially nothing matched.
        loaded = len(sd) - len(unexpected)
        print(f"warm start: {loaded}/{len(list(model.state_dict()))} tensors loaded, "
              f"{len(missing)} missing, {len(unexpected)} unexpected")
        if loaded == 0:
            raise SystemExit("--init-weights matched no parameters; refusing to train")

    decay, no_decay = [], []
    for nme, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or nme == "pos" else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(len(dl_tr), 1), pct_start=0.15)

    print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,} | params {model.n_params():,}",
          flush=True)
    print("train label mix: " + ", ".join(
        f"{CLASS_NAMES[i]}={int((labels[tr]==i).sum()):,}" for i in range(N_CLASSES)), flush=True)

    # An unattended multi-hour run must survive being interrupted, so optimiser and
    # scheduler state are checkpointed every epoch, not just the best weights.
    last_path = args.out + ".last"
    start_ep, best, best_ep, hist = 0, -1.0, -1, []
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_ep = ck["epoch"] + 1; best = ck["best"]; best_ep = ck["best_ep"]
        hist = ck.get("history", [])
        print(f"resumed from epoch {ck['epoch']} (best F1 {best:.3f} @ ep {best_ep})", flush=True)
    n_bad = 0          # non-finite training steps, skipped rather than allowed to poison
    i_non = CLASS_NAMES.index("NonPSPL")
    for ep in range(start_ep, args.epochs):
        model.train(); t0 = time.time(); run = torch.zeros((), device=dev); seen = 0
        for feats, present, y, w, _ in dl_tr:
            feats = {k: v.to(dev) for k, v in feats.items()}
            present = {k: v.to(dev) for k, v in present.items()}
            y, w = y.to(dev), w.to(dev)
            opt.zero_grad(set_to_none=True)
            logits = model(feats, present)
            loss = (F.cross_entropy(logits, y, reduction="none") * w).sum() / w.sum()
            # A single non-finite step would otherwise propagate NaN into every weight via
            # the optimiser and silently destroy the run -- the loss just reads "nan" from
            # then on. Detect, report once with enough context to diagnose, and SKIP the
            # update rather than let it through.
            if not torch.isfinite(loss):
                n_bad += 1
                if n_bad <= 3:
                    with torch.no_grad():
                        fin = {b: bool(torch.isfinite(feats[b]).all()) for b in feats}
                        pres = {b: int(present[b].sum()) for b in present}
                    print(f"  [warn] non-finite loss at epoch {ep} step {seen//max(len(y),1)}: "
                          f"logits_finite={bool(torch.isfinite(logits).all())} "
                          f"w.sum={float(w.sum()):.3e} inputs_finite={fin} present={pres}",
                          flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gnorm):
                n_bad += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.step(); sched.step()
            # accumulate on-device: float(loss) forces an MPS sync every step
            run += loss.detach() * len(y); seen += len(y)
        run = float(run) if torch.is_tensor(run) else run    # single sync per epoch
        if n_bad:
            print(f"  [warn] {n_bad} non-finite step(s) skipped so far", flush=True)
        m = evaluate(model, dl_va, dev)
        # Selection on NonPSPL F1, not accuracy and not bare recall.
        #   * accuracy is wrong because Flat+PSPL are ~61% of the data, so a model that never
        #     predicts NonPSPL still scores well;
        #   * bare RECALL is just as wrong in the opposite direction -- a model that predicts
        #     NonPSPL for everything scores 1.000. That is not hypothetical: the first smoke
        #     run of this loop selected exactly such a model, recall 1.000 at precision 0.139.
        # F1 has no degenerate optimum at either extreme.
        score = m["f1"]["NonPSPL"]
        hist.append({"epoch": ep, "train_loss": run / max(seen, 1), **m})
        print(f"ep {ep:3d} loss {run/max(seen,1):.4f} val {m['loss']:.4f} "
              f"NonPSPL f1 {score:.3f} (r {m['recall']['NonPSPL']:.3f} "
              f"p {m['precision']['NonPSPL']:.3f}) macroF1 {m['macro_f1']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        # makedirs before EVERY save, not just the best one. A run whose output directory
        # disappears mid-flight (moved, unmounted) would otherwise die here at the end of an
        # epoch with a bare FileNotFoundError, losing the whole run.
        # Update the best-tracking BEFORE writing .last, so the resume checkpoint records the
        # up-to-date best/best_ep. Writing .last first stored a stale `best`, so a resumed run
        # could re-evaluate an early worse epoch as "new best" and overwrite args.out.
        if score > best:
            best, best_ep = score, ep
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({"model": model.state_dict(), "config": asdict(cfg),
                        "format": "binml-v5", "mag_scale": MAG_SCALE,
                        "epoch": ep, "val_nonpspl_f1": score,
                        "val_nonpspl_recall": m["recall"]["NonPSPL"],
                        "val_nonpspl_precision": m["precision"]["NonPSPL"],
                        "class_names": CLASS_NAMES, "seed": args.seed}, args.out)
        os.makedirs(os.path.dirname(last_path) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best,
                    "best_ep": best_ep, "history": hist}, last_path)

    print(f"\nbest epoch {best_ep}, val NonPSPL F1 {best:.3f}")
    model.load_state_dict(torch.load(args.out, map_location=dev)["model"])
    final = evaluate(model, dl_te, dev)
    print("TEST:", json.dumps({k: v for k, v in final.items() if k != "confusion"}, indent=2))
    json.dump({"history": hist, "test": final}, open(args.out + ".metrics.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

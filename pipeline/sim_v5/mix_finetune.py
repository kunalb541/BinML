"""BinML v5 -- build the hard/natural fine-tuning mix.

The hard regimes are NOT a uniform sample of anything, so mixing them in uniformly gets two
things wrong at once. Uniformly subsampling the hard pool to hit a target fraction discards
most of its NonPSPL (only 4.3% of hard rows), which is the class the regimes exist to supply.
And taking the hard pool whole would make the training set 62% hard, which distorts the
natural-population calibration the precision numbers depend on.

So selection is stratified per class, with a priority regime per class taken in full:

  NonPSPL       every hard row -- scarce and science-critical, never subsampled
  Flat          boundary, boundary2, flatrich taken whole; these ARE the Flat decision
                boundary, and Flat from the lens regimes is generic filler already
                abundant in the natural pool
  PeriodicVar   perhard whole, topped up from elsewhere
  LongPeriodVar lpvhard whole -- the season-long PSPL impostor
  Eruptive      erupthard whole
  PSPL          capped; hard-regime PSPL is useful (odd tE, heavy blending) but plentiful

The natural pool comes from cache shards 0-89 ONLY. Base training used 676,191 events
(~90 shards) and the independent test set used 443,030 (~60 shards, indices 90-149), so
0-89 is exactly the original training range and cannot contaminate the test set.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import h5py
import numpy as np

from .classes import CLASS_NAMES
from .to_memmap import convert_selected

# class -> (regimes taken in full, overall cap across the hard pool).
# cap semantics: None = take every hard row of this class; 0 = take the priority regimes ONLY
# and no top-up; N = top up from the other regimes until N rows total.
# Flat is cap=0 deliberately. Giving it None pulled in all 339k Flat rows from the LENS
# regimes too, which pushed the set to 51.9% hard with Flat at 51% of all events -- and those
# rows are redundant, since Flat under the planetary/wide/close priors is just natural Flat.
# The Flat rows that actually teach the decision boundary live in boundary/boundary2/flatrich.
PRIORITY = {
    "NonPSPL":       ([], None),
    "Flat":          (["boundary", "boundary2", "flatrich"], 0),
    "PeriodicVar":   (["perhard"], 40_000),
    "LongPeriodVar": (["lpvhard"], 40_000),
    "Eruptive":      (["erupthard"], 45_000),
    "PSPL":          ([], 60_000),
}


def build(hard_dir: str, nat_dir: str, out_dir: str, seed: int = 20260721) -> dict:
    rng = np.random.default_rng(seed)
    hard = sorted(glob.glob(os.path.join(hard_dir, "*", "*.h5")))
    # exclude the natural subdir from the hard pool: with the train6 layout everything
    # (including natural) lives under one cache dir, so glob would otherwise count natural
    # events as "hard" and double them into the mix.
    hard = [h for h in hard if os.path.basename(os.path.dirname(h)) != "natural"]
    nat = sorted(glob.glob(os.path.join(nat_dir, "*.h5")))
    if not hard or not nat:
        raise SystemExit(f"missing inputs: {len(hard)} hard, {len(nat)} natural")

    labels, regime = {}, {}
    for p in hard:
        with h5py.File(p, "r") as f:
            labels[p] = np.asarray(f["label"][:], dtype=int)
        regime[p] = os.path.basename(os.path.dirname(p))

    masks = {p: np.zeros(len(labels[p]), dtype=bool) for p in hard}
    report = {}
    for ci, cname in enumerate(CLASS_NAMES):
        prio, cap = PRIORITY[cname]
        # rows of this class, split into priority-regime and the rest
        pri = [(p, np.nonzero((labels[p] == ci))[0]) for p in hard if regime[p] in prio]
        rest = [(p, np.nonzero((labels[p] == ci))[0]) for p in hard if regime[p] not in prio]
        n_pri = sum(len(i) for _, i in pri)
        for p, idx in pri:
            masks[p][idx] = True
        room = None if cap is None else max(cap - n_pri, 0)
        flat = [(p, j) for p, idx in rest for j in idx]
        if room is None or room >= len(flat):
            take = flat
        else:
            sel = rng.choice(len(flat), size=room, replace=False)
            take = [flat[i] for i in sel]
        for p, j in take:
            masks[p][j] = True
        report[cname] = {"priority_regimes": prio, "from_priority": int(n_pri),
                         "from_rest": int(len(take)), "cap": cap}

    n_hard = int(sum(m.sum() for m in masks.values()))
    for p in nat:                      # natural pool always taken whole
        with h5py.File(p, "r") as f:
            masks[p] = np.ones(int(f.attrs["n_events"]), dtype=bool)
    n_nat = int(sum(masks[p].sum() for p in nat))
    print(f"hard {n_hard:,} + natural {n_nat:,} = {n_hard+n_nat:,} "
          f"({100*n_hard/(n_hard+n_nat):.1f}% hard)")
    for c, r in report.items():
        print(f"  {c:14s} priority={r['from_priority']:>7,} other={r['from_rest']:>7,} "
              f"cap={r['cap']}")

    meta = convert_selected(hard + nat, masks, out_dir, seed=seed)
    meta["mix"] = {"n_hard": n_hard, "n_natural": n_nat,
                   "hard_fraction": n_hard / (n_hard + n_nat), "per_class": report,
                   "natural_shards": "cache2 0-89 (train range; test is 90-149)"}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard-dir", required=True)
    ap.add_argument("--nat-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    m = build(a.hard_dir, a.nat_dir, a.out)
    print(f"-> {a.out}  {m['n_events']:,} events  {m['bytes']/1e9:.2f} GB")
    lab = np.load(os.path.join(a.out, "label.npy"))
    print("label mix:", {CLASS_NAMES[c]: int((lab == c).sum()) for c in range(len(CLASS_NAMES))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

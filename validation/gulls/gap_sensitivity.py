"""Why BinML fails on RMDC26 single-lens events: the season has gaps, training seasons do not.

Reproduces the GULLS cross-simulator failure with NO GULLS data, using the training pipeline's own
events.  RMDC26 (GULLS) implements Roman's real schedule, in which F146 pauses for ~6.2 h seven
times per 70.7-day season (at days 0.98, 2.48, 21.49, 31.48, 35.23, 62.23, 69.48 in season 1).
BinML's training grid (`pipeline.assemble._epochs`) is continuous; the only way an epoch drops
is SNR < 3 or saturation, and across every class 0/1800 sampled training events contain an empty
F146 bin.  The model's sole prior for an empty-bin token is the unrevealed future of a partial
season, so a mid-season gap is read as evidence against a clean single lens.

Inserting those seven gaps into in-distribution events gives PSPL recall 0.12 (-> NonPSPL 0.65,
PeriodicVar 0.18) and Flat recall 0.08 (-> PeriodicVar), while NonPSPL (0.98) and PeriodicVar
(1.00) are unaffected.  The GULLS run itself gave PSPL 0.04 / NonPSPL 0.65 / PeriodicVar 0.31 on
1S1L.  The two agree; the transfer failure is the schedule, not the simulator.

`pipeline/train.py --cadence-aug` thins random bins with relabelling and is the designed remedy;
the shipped checkpoint was trained with it at 0.0.

Usage:  python validation/gulls/gap_sensitivity.py [--n 60] [--out validation/gulls/gap_sensitivity.json]
"""
import argparse, collections, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

RMDC26_GAPS_D = [0.98, 2.48, 21.49, 31.48, 35.23, 62.23, 69.48]
RMDC26_GAP_H = 6.2
CLASSES = ("PSPL", "NonPSPL", "Flat", "PeriodicVar")


def recall_under_gaps(clf, cfg, gaps_d, gap_h, n, seed, classes=CLASSES):
    from pipeline.assemble import simulate_event
    out = {}
    for cls in classes:
        rng = np.random.default_rng(seed)
        r2 = np.random.default_rng(seed + 1)
        ok = 0; done = 0; lc = collections.Counter()
        while done < n:
            ev = simulate_event(cls, rng, cfg)
            if ev is None or "F146" not in ev.bands or ev.label != cls:
                continue
            starts = gaps_d if gaps_d is not None else []
            bands = {}
            for k, b in ev.bands.items():
                if b.t.size < 10:
                    continue
                keep = np.ones(b.t.size, bool)
                for s in starts:
                    keep &= ~((b.t > s) & (b.t < s + gap_h / 24.0))
                bands[k] = (b.t[keep], b.mag[keep])
            p = clf.predict(bands, m_base_ref=float(ev.params["_m_base_ref"]), t_start=0.0)
            lc[p.label] += 1; ok += (p.label == cls); done += 1
        out[cls] = {"recall": round(ok / n, 4), "argmax": dict(lc)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "gap_sensitivity.json"))
    args = ap.parse_args(argv)
    import binml
    from pipeline.assemble import SurveyConfig
    clf = binml.Classifier(); cfg = SurveyConfig()
    res = {"_doc": __doc__.split("\n")[0], "n_per_class": args.n, "seed": args.seed,
           "rmdc26_gaps_d": RMDC26_GAPS_D, "rmdc26_gap_h": RMDC26_GAP_H}
    print("[1/3] no gaps", flush=True)
    res["no_gaps"] = recall_under_gaps(clf, cfg, None, 0.0, args.n, args.seed)
    print("[2/3] single gap, length sweep", flush=True)
    res["single_gap_by_length_h"] = {}
    for L in (0.5, 1.0, 2.0, 4.0, 6.0):
        rng = np.random.default_rng(args.seed + 7)
        res["single_gap_by_length_h"][str(L)] = recall_under_gaps(
            clf, cfg, [float(rng.uniform(3, 67))], L, args.n, args.seed)
    print("[3/3] RMDC26 seven-gap schedule", flush=True)
    res["rmdc26_schedule"] = recall_under_gaps(clf, cfg, RMDC26_GAPS_D, RMDC26_GAP_H, args.n, args.seed)
    json.dump(res, open(args.out, "w"), indent=2)
    for k in ("no_gaps", "rmdc26_schedule"):
        print(k, {c: v["recall"] for c, v in res[k].items()})
    print("->", args.out)


if __name__ == "__main__":
    main()

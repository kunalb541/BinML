"""BinML command-line interface.

    binml classify path/to/lightcurve.dat
    binml classify phot.dat --format ogle --model finetuned
    binml classify event.csv --format generic --cols 0 1 2
    binml evolution phot.dat --format ogle -o evolution.png
    binml ogle 2017 482                       # fetch + classify an OGLE EWS event
"""
from __future__ import annotations

import argparse
import sys


def _load(args):
    from . import surveys
    if getattr(args, "cols", None):
        return surveys.load_generic(args.file, cols=tuple(args.cols))
    return surveys.read_lightcurve(args.file, fmt=args.format)


def _print(pred):
    p = pred.probabilities
    print(f"\n  prediction : {pred.label}   (confidence {pred.confidence:.2f})")
    for c in ("Flat", "PSPL", "Binary"):
        bar = "#" * int(round(p[c] * 30))
        print(f"    {c:7s} {p[c]:.3f}  {bar}")
    print(f"  microlensing (PSPL+Binary): {pred.is_microlensing:.3f}   "
          f"anomalous (Binary): {pred.is_anomalous:.3f}")
    print(f"  n_points={pred.n_points}  peak_magnification={pred.peak_magnification:.1f}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="binml", description="Deep-learning microlensing classifier")
    from . import __version__
    ap.add_argument("--version", action="version", version=f"binml {__version__}")
    ap.add_argument("--model", default="finetuned", help="finetuned | base | path to .pt")
    ap.add_argument("--device", default="cpu")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(s):
        s.add_argument("file")
        s.add_argument("--format", default="auto", help="auto|ogle|moa|generic")
        s.add_argument("--cols", type=int, nargs=3, default=None, help="time mag err column indices")
        s.add_argument("--flux", action="store_true", help="input column is flux, not magnitude")

    c = sub.add_parser("classify", help="classify one light curve")
    add_common(c)

    e = sub.add_parser("evolution", help="probability-evolution plot")
    add_common(e)
    e.add_argument("-o", "--out", default="binml_evolution.png")

    o = sub.add_parser("ogle", help="fetch an OGLE EWS event and classify it")
    o.add_argument("year", type=int); o.add_argument("event", type=int)
    o.add_argument("-o", "--out", default=None, help="also save an evolution plot here")

    ev = sub.add_parser("evaluate", help="evaluate on a compact HDF5 test set "
                                         "(3-class report + detectability-conditioned binary recall)")
    ev.add_argument("h5")
    ev.add_argument("--json", default=None, help="also write the reports to this JSON path")

    args = ap.parse_args(argv)
    from .classifier import Classifier
    clf = Classifier(model=args.model, device=args.device)

    if args.cmd == "ogle":
        from . import surveys
        t, m, err = surveys.fetch_ogle_ews(args.year, args.event)
        name = f"OGLE-{args.year}-BLG-{args.event:04d}"
        pred = clf.predict(t, m, err)
        print(f"\n{name}"); _print(pred)
        if args.out:
            from . import plotting
            plotting.plot_evolution(clf.predict_evolution(t, m, err),
                                    title=name).savefig(args.out, dpi=130, bbox_inches="tight")
            print(f"  saved {args.out}")
        return 0

    if args.cmd == "evaluate":
        from .evaluate import evaluate_dataset
        report, detect = evaluate_dataset(clf, args.h5)
        if report is None:
            print("no labels in file; nothing to evaluate"); return 1
        print("\n3-class classification report:"); print(report)
        if detect is not None:
            print("\n" + detect.summary())
        else:
            print("\n(no anomaly_dchi2 in file -> detectability report skipped)")
        if args.json:
            import json
            from dataclasses import asdict
            out = {"classification": asdict(report),
                   "detectability": asdict(detect) if detect else None}
            with open(args.json, "w") as f:
                json.dump(out, f, indent=2)
            print(f"\n  wrote {args.json}")
        return 0

    t, m, err = _load(args)
    if args.cmd == "classify":
        _print(clf.predict(t, m, err, is_flux=args.flux))
    elif args.cmd == "evolution":
        from . import plotting
        evo = clf.predict_evolution(t, m, err)
        _print(evo.final)
        plotting.plot_evolution(evo).savefig(args.out, dpi=130, bbox_inches="tight")
        print(f"  saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

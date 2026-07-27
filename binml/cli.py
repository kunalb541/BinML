"""BinML command line: classify a light-curve file.

    binml classify lc.csv                 # F146 columns: time,mag[,mag_err]
    binml classify lc.csv --m-base 22.1   # provide the F146 baseline magnitude
"""
from __future__ import annotations
import argparse, sys
import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="binml", description="6-class Roman light-curve classifier")
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("classify", help="classify a CSV/whitespace file (F146: time, mag[, err])")
    c.add_argument("file")
    c.add_argument("--m-base", type=float, default=None, help="F146 baseline magnitude")
    c.add_argument("--t-start", type=float, default=None, help="day the 72-d window opens")
    ap.add_argument("--version", action="store_true")
    a = ap.parse_args(argv)
    if a.version:
        import binml; print(binml.__version__); return 0
    if a.cmd != "classify":
        ap.print_help(); return 1
    d = np.loadtxt(a.file, delimiter="," if a.file.endswith(".csv") else None, ndmin=2)
    t, m = d[:, 0], d[:, 1]
    import binml
    r = binml.Classifier().predict(t, m, m_base_ref=a.m_base, t_start=a.t_start)
    print(r)
    for k, v in sorted(r.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {k:14s} {v:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

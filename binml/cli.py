"""BinML command line: classify a light-curve file.

    binml classify lc.csv                 # F146 columns: time,mag[,mag_err]
    binml classify lc.csv --m-base 22.1   # provide the F146 baseline magnitude
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import re
import sys

import numpy as np


_TIME_COLUMNS = {"time", "t", "time_day", "time_days", "mjd", "jd", "hjd", "bjd"}
_MAG_COLUMNS = {"mag", "magnitude", "f146", "f146_mag", "mag_f146"}


def _finite_float(value: str) -> float:
    """Argparse type that rejects NaN/Inf before they reach model preprocessing."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if not np.isfinite(parsed):
        raise argparse.ArgumentTypeError(f"expected a finite number, got {value!r}")
    return parsed


def _normalise_column_name(name: str) -> str:
    """Normalise a modest set of human-readable CSV column names."""
    name = name.lstrip("\ufeff").strip().lower()
    name = re.sub(r"\s*[\[(].*?[\])]\s*", "", name)
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def _column_index(columns: list[str], aliases: set[str], kind: str) -> int:
    matches = [i for i, name in enumerate(columns) if name in aliases]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"CSV header has more than one {kind} column")
    expected = "time/t/MJD/JD" if kind == "time" else "mag/magnitude/F146"
    raise ValueError(f"CSV header has no recognised {kind} column (expected {expected})")


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load legacy headerless CSVs or headered CSVs with named time/magnitude columns."""
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = [
            (line_number, row)
            for line_number, row in enumerate(csv.reader(fh), start=1)
            if row and any(field.strip() for field in row) and not row[0].lstrip().startswith("#")
        ]

    if not rows:
        raise ValueError(f"light-curve file is empty: {path}")

    _, first_row = rows[0]
    if len(first_row) >= 2 and _is_number(first_row[0].strip()) and _is_number(first_row[1].strip()):
        data = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
        if data.shape[1] < 2:
            raise ValueError("light-curve file must contain at least time and magnitude columns")
        return data[:, 0], data[:, 1]

    columns = [_normalise_column_name(field) for field in first_row]
    time_col = _column_index(columns, _TIME_COLUMNS, "time")
    mag_col = _column_index(columns, _MAG_COLUMNS, "magnitude")
    data_rows = rows[1:]
    if not data_rows:
        raise ValueError(f"CSV contains a header but no light-curve rows: {path}")

    needed = max(time_col, mag_col)
    time, mag = [], []
    for line_number, row in data_rows:
        if len(row) <= needed:
            raise ValueError(f"CSV row {line_number} has too few columns")
        try:
            time.append(float(row[time_col].strip()))
            mag.append(float(row[mag_col].strip()))
        except ValueError as exc:
            raise ValueError(f"CSV row {line_number} has a non-numeric time or magnitude") from exc

    return np.asarray(time, dtype=float), np.asarray(mag, dtype=float)


def _load_light_curve(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Return time and magnitude arrays from the CLI's supported text formats."""
    path = Path(filename)
    if path.suffix.lower() == ".csv":
        return _load_csv(path)

    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] < 2:
        raise ValueError("light-curve file must contain at least time and magnitude columns")
    return data[:, 0], data[:, 1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="binml", description="6-class Roman light-curve classifier")
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("classify", help="classify a CSV/whitespace file (F146: time, mag[, err])")
    c.add_argument("file")
    c.add_argument("--m-base", type=_finite_float, default=None, help="F146 baseline magnitude")
    c.add_argument("--t-start", type=_finite_float, default=None, help="day the 72-d window opens")
    ap.add_argument("--version", action="store_true")
    a = ap.parse_args(argv)
    if a.version:
        import binml; print(binml.__version__); return 0
    if a.cmd != "classify":
        ap.print_help(); return 1
    try:
        t, m = _load_light_curve(a.file)
        import binml
        r = binml.Classifier().predict(t, m, m_base_ref=a.m_base, t_start=a.t_start)
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    print(r)
    for k, v in sorted(r.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {k:14s} {v:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

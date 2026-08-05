#!/usr/bin/env bash
# Reproducible build of the BinML methods paper.
#   1. regenerate figures from the archived evaluation artifact (paper/results/)
#   2. regenerate paper_macros.tex from canonical_numbers.json (+ figures_stats.json)
#   3. compile paper.tex -> paper.pdf  (pdflatex x bibtex x pdflatex x2)
# Every number in the PDF traces to canonical_numbers.json; nothing is hand-typed.
set -euo pipefail
cd "$(dirname "$0")"

# The figure scripts import `pipeline` and `binml` from the repository root, which is the parent of
# this directory. Without this a clean checkout fails with ModuleNotFoundError: pipeline.
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
python3 - <<'PYCHECK' || { echo "ERROR: missing dependencies. Install the environment first:
    conda env create -f ../environment.yml && conda activate binml
or, for the artifact-only figures: pip install numpy matplotlib" >&2; exit 1; }
import importlib, sys
missing = [m for m in ("numpy", "matplotlib") if importlib.util.find_spec(m) is None]
if missing:
    print("missing:", missing, file=sys.stderr); sys.exit(1)
PYCHECK

echo "[1/4] figures from eval artifact"; python3 make_figures.py
echo "[2/4] figures from simulated events (slower)"; python3 make_data_figures.py
echo "[3/4] macros";   python3 make_macros.py
echo "[4/4] latex"
export PATH="/Library/TeX/texbin:$PATH"
pdflatex -interaction=nonstopmode -halt-on-error paper.tex >build.log 2>&1 || { tail -40 build.log; exit 1; }
bibtex paper            >>build.log 2>&1 || { tail -40 build.log; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error paper.tex >>build.log 2>&1 || { tail -40 build.log; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error paper.tex >>build.log 2>&1 || { tail -40 build.log; exit 1; }
echo "done -> paper.pdf ($(python3 -c "import os;print(f'{os.path.getsize(\"paper.pdf\")/1024:.0f} kB')"))"

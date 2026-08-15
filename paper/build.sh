#!/usr/bin/env bash
# Reproducible build of the BinML methods paper.
#   1. validate the archived evaluation artifact against its manifest
#   2. regenerate figures from archived evaluation and simulation artifacts
#   3. regenerate paper_macros.tex from the source artifacts
#   4. compile paper.tex -> paper.pdf  (pdflatex x bibtex x pdflatex x2)
set -euo pipefail
cd "$(dirname "$0")"

# Use the active environment's interpreter. `PYTHON_BIN=/path/to/python ./build.sh` is available
# for machines (including Homebrew + pyenv setups) where `python` and `python3` resolve differently.
PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "$PYTHON_BIN" >/dev/null || {
  echo "ERROR: Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
}

# The figure scripts import `pipeline` and `binml` from the repository root, which is the parent of
# this directory. Without this a clean checkout fails with ModuleNotFoundError: pipeline.
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" - <<'PYCHECK' || { echo "ERROR: missing dependencies. Install the environment first:
    conda env create -f ../environment.yml && conda activate binml
or install the complete paper environment: pip install -e '.[all]'" >&2; exit 1; }
import importlib.util, sys      # importlib.util is NOT auto-imported on Python >= 3.12
required = ("numpy", "matplotlib", "torch", "scipy", "VBBinaryLensing")
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    print("missing:", missing, file=sys.stderr); sys.exit(1)
PYCHECK

echo "[1/5] validate frozen artifacts"; "$PYTHON_BIN" validate_artifacts.py
echo "[2/5] figures from eval artifact"; "$PYTHON_BIN" make_figures.py
echo "[3/5] figures from simulated events (slower)"; "$PYTHON_BIN" make_data_figures.py
echo "[4/5] macros";   "$PYTHON_BIN" make_macros.py
echo "[5/5] latex"
export PATH="/Library/TeX/texbin:$PATH"
pdflatex -interaction=nonstopmode -halt-on-error paper.tex >build.log 2>&1 || { tail -40 build.log; exit 1; }
bibtex paper            >>build.log 2>&1 || { tail -40 build.log; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error paper.tex >>build.log 2>&1 || { tail -40 build.log; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error paper.tex >>build.log 2>&1 || { tail -40 build.log; exit 1; }
echo "done -> paper.pdf ($("$PYTHON_BIN" -c "import os;print(f'{os.path.getsize(\"paper.pdf\")/1024:.0f} kB')"))"

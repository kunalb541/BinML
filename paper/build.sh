#!/usr/bin/env bash
# Reproducible build of the BinML methods paper.
#   1. regenerate figures from the archived evaluation artifact (paper/results/)
#   2. regenerate paper_macros.tex from canonical_numbers.json (+ figures_stats.json)
#   3. compile paper.tex -> paper.pdf  (pdflatex x bibtex x pdflatex x2)
# Every number in the PDF traces to canonical_numbers.json; nothing is hand-typed.
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/3] figures";  python3 make_figures.py
echo "[2/3] macros";   python3 make_macros.py
echo "[3/3] latex"
export PATH="/Library/TeX/texbin:$PATH"
pdflatex -interaction=nonstopmode -halt-on-error paper.tex >build.log 2>&1 || { tail -40 build.log; exit 1; }
bibtex paper            >>build.log 2>&1 || { tail -40 build.log; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error paper.tex >>build.log 2>&1 || { tail -40 build.log; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error paper.tex >>build.log 2>&1 || { tail -40 build.log; exit 1; }
echo "done -> paper.pdf ($(python3 -c "import os;print(f'{os.path.getsize(\"paper.pdf\")/1024:.0f} kB')"))"

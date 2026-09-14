# BinML methods paper

Source for the BinML methods paper: a detectability-conditioned classifier for streaming,
partial-season Roman-like light curves (AASTeX 7.0.1). The manuscript frames the system as
synthetic anomaly triage, not a validated real-time planet trigger.

## Build

```bash
pip install -e ".[all]"
cd paper
./build.sh
```

Produces `paper.pdf`. It requires a TeX Live distribution (`pdflatex`, `bibtex`) and the full
`[all]` Python extras, including PyTorch, SciPy, h5py, and VBBinaryLensing. The build has five
fail-fast stages:

1. **`validate_artifacts.py`** — checks frozen array shape/dtype/payload hashes, hashes every
   load-bearing file listed in `results/MANIFEST.json`, verifies the shipped checkpoint identity,
   and requires byte-exact regeneration of the two deterministic trace reductions before any
   reported output is regenerated.
2. **`make_figures.py`** — figures derivable from the archived evaluation artifact in
   `results/` (confusion matrix, NonPSPL PR vs the truth-informed Δχ² reference, efficiency plane, parameter
   dependence, calibration, risk–coverage, and model schematic). Also writes
   `outputs/figures_stats.json` with derived figure statistics.
3. **`make_data_figures.py`** — figures that need raw light curves: it *simulates*
   representative events (deterministic seeds) with the `pipeline` stack, runs them through the
   shipped model, and renders selected data-dependent figures. This stage refuses to run without
   VBBinaryLensing, preventing a silent single-lens fallback.
4. **`make_macros.py`** — turns `canonical_numbers.json`, `figures_stats.json`, and the
   experiment artifacts under `../validation/` into `paper_macros.tex`, the `\bml*` command set
   the manuscript cites. Reads of the validation artifacts are **fail-closed**: a missing file, or
   a file whose schema lost a key the manuscript uses, aborts the build rather than emitting a
   stale number. **`make_gulls_macros.py`** does the same for the RMDC26 section: `gulls_macros.tex`
   and the tables `outputs/gulls_{transfer,rho,gap,seed}_table.tex`, from `../validation/gulls/`.
5. **`pdflatex`, `bibtex`, then `pdflatex` until cross-references settle** — compiles `paper.tex`.

`render_draft.py` renders `draft_gulls_section.tex` (the long-form record behind the RMDC26 section and
`REVISION.md`, not part of the build) with every macro substituted, to `outputs/draft_gulls_rendered.txt`.

## Why it's reproducible

Reported result numbers reach the prose through generated macros. Macros come from two kinds of
source:

- `canonical_numbers.json`, for the static evaluation (headline, per-class, stress test), itself
  sourced from `results/metrics.json`;
- the experiment artifacts in `../validation/`, read **directly** by `make_macros.py`: cascade,
  matched risk–coverage, prevalence scenarios, and ablations.

The second category is read directly rather than copied into `canonical_numbers.json`. To update a
result, regenerate its artifact and rebuild; there is no intermediate numeric table to maintain.

Reproducibility has three levels, and the build script only does the first: rebuilding the
manuscript from frozen artifacts (minutes), regenerating those artifacts from the model and
simulator (minutes to hours), and reproducing training (cloud-scale). The stored main cascade
trace came from a dirty source tree and lacks a source hash/diff; the matched trace lacks code and
checkpoint hashes. The labelling-ablation source hash was repaired after its run, so the newer
provenance mechanism has not yet produced the artifact used by the manuscript.

## Files

| file | role |
|---|---|
| `paper.tex` | the manuscript |
| `canonical_numbers.json` | static-evaluation numbers (headline, per-class, stress test) |
| `../validation/*_result.json` | experiment artifacts read directly by `make_macros.py` (cascade, prevalence, ablations) — fail-closed, never hand-copied |
| `validate_artifacts.py` | integrity checks for frozen arrays, hashed result/source files, checkpoint identity, and deterministic reducers |
| `make_macros.py` | JSON → `paper_macros.tex` |
| `make_gulls_macros.py` | RMDC26 artifacts → `gulls_macros.tex` + `outputs/gulls_{transfer,rho,gap,seed}_table.tex` (committed; CI checks they regenerate byte for byte) |
| `render_draft.py` | `draft_gulls_section.tex` → `outputs/draft_gulls_rendered.txt` (committed, checked the same way) |
| `make_figures.py` | artifact → `outputs/figures/*.pdf` |
| `make_data_figures.py` | simulated events → light-curve gallery + cascade evolution + per-class probability evolution with noise-realisation bands |
| `refs.bib` | bibliography |
| `build.sh` | one-command reproducible build |
| `results/` | archived evaluation artifact (logits, labels, keep_prob, metrics.json, …) |
| `aastex701.cls`, `aasjournalv7.bst` | AAS journal class + bib style |
| `PLAN.md` | outline, reference list, and roadmap |

Build products (`paper.pdf`, `paper_macros.tex`, `outputs/figures/*.pdf`, `*.aux/.log/.bbl/.blg`) are
regenerated by `build.sh` and are gitignored; `gulls_macros.tex`, the RMDC26 tables, the rendered draft and
`outputs/figures/prob_evolution_confidence.json` are committed.

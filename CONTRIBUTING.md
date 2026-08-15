# Contributing to BinML

Thanks for your interest in BinML — a six-class deep-learning classifier for gravitational
microlensing and variable-star light curves, built for Nancy Grace Roman Space Telescope-like
cadence. Contributions of all kinds are welcome: bug reports, new survey
loaders, documentation fixes, tests, and improvements to the model or research pipeline.

By participating you agree that your contributions will be released under the project's
[MIT License](LICENSE). Please be respectful and constructive in all discussions.

## Repository layout

BinML is two things in one repository. Knowing which part your change belongs in makes
review much faster.

- **`binml/`** — the installable 6-class inference API. It is not currently on PyPI; install it
  from a checkout or Git URL. The package exposes `classifier.py`, `preprocess.py`, and `cli.py`,
  bundles the trained weights, and imports the shared network definition from `pipeline/model.py`.
  The earlier 3-class API and its historical loaders live under **`binml/legacy/`**.
- **`pipeline/`** — the research pipeline used to produce the models. It is installed because
  the public package currently imports shared model/preprocessing components from it, but its
  research interfaces are not part of the stable public API. The current stages are
  `run_shard.py`/`assemble.py`/`generators.py` (simulation), `cache.py`/`to_memmap.py`,
  `train.py`, `evaluate.py`, `plots.py`, and the distributed bin/evaluation helpers. Heavier
  optional dependencies are expected here.
- **`validation/`** — frozen experiment artifacts plus local and Modal experiment scripts. Do
  not hand-edit a reported result JSON; regenerate it with its reducer and update the paper
  manifest.
- **`paper/`** — manuscript source, frozen evaluation artifacts, generated-number tooling, and a
  fail-closed build.

Supporting directories: `docs/`, `paper/`, `examples/`, `tests/`, `data/`, `results/`.

As a rule of thumb: if a change affects how a user runs `binml.Classifier` or the `binml`
CLI, it goes in `binml/`. If it affects how models are simulated, trained, or evaluated at
scale, it goes in `pipeline/`.

## Development setup

BinML targets Python >= 3.9 and PyTorch. We recommend a fresh virtual environment.

```bash
git clone https://github.com/kunalb541/BinML.git
cd BinML
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"     # editable install with plotting + analysis extras
pip install pytest          # for running the test suite
```

The `[all]` extra pulls in `matplotlib`, `h5py`, `scipy`, `scikit-learn`, and
`VBBinaryLensing` (used by simulation, validation, plotting, and evaluation).
For a minimal inference-only environment, `pip install -e .` installs just `torch` and
`numpy`.

## Running tests

The test suite lives in `tests/` and runs with `pytest`:

```bash
python -m pytest tests/ -q
```

The suite covers the public API and CLI, numerical/input invariants, artifact integrity,
deterministic reducers, and clean-build behaviour. Some clean-archive integration tests are marked
`slow`. Please make sure the suite passes before opening a pull request, and add a test when you
fix a bug or add a feature. Continuous integration runs the package tests on Python 3.9, 3.10,
and 3.11, exercises the paper dependency preflight on newer interpreters, and builds/checks the
wheel and source distribution on Python 3.11.

## Code style

- Follow [PEP 8](https://peps.python.org/pep-0008/) and match the style of the surrounding
  code.
- Use type hints on public functions and add a short docstring describing arguments and
  return values.
- Keep `binml/` free of heavy or optional dependencies; import them lazily inside the
  functions that need them if a feature genuinely requires one.
- Prefer clear, self-explanatory names over comments, and keep comments focused on *why*
  rather than *what*.
- Keep pull requests focused: one logical change per PR is easier to review.

## Reporting issues

Please open issues at https://github.com/kunalb541/BinML/issues. A good bug report
includes:

- What you expected to happen and what actually happened.
- A minimal, self-contained example that reproduces the problem (ideally a short snippet
  using `binml.Classifier` or a small input file).
- Your environment: OS, Python version, and `torch` / `numpy` / `binml` versions.

For feature requests, describe the use case and, where relevant, which part of the project
(`binml/` inference vs. `pipeline/` research) it would touch.

## Pull requests

1. Fork the repository and create a branch from `main` with a descriptive name.
2. Make your change, keeping it scoped to a single concern.
3. Add or update tests and documentation as needed.
4. Run `python -m pytest tests/ -q` and confirm it passes.
5. Open a pull request with a clear description of the change and its motivation. Link any
   related issue.

Maintainers may request changes during review; this is a normal part of the process and
not a reflection on your work. Small, well-tested PRs are merged fastest.

## Adding a survey format

The current 6-class API accepts arrays directly, or a dictionary mapping F146/F087/F213 to
`(time, magnitude)` arrays. Its CLI accepts a generic CSV or whitespace file for F146. The OGLE,
MOA, and generic loader helpers belong to the historical 3-class API at
`binml/legacy/surveys.py`; do not present them as current `binml` exports.

If a new format belongs in the current API, add an explicit loader module, document its passband
and time/magnitude conventions, and add tests for malformed rows, missing values, and an end-to-end
classification. Network downloads must remain explicit. Note that the current 6-class model does
not consume per-epoch magnitude uncertainties even if a source format supplies them.

## Questions

If anything is unclear, open an issue and ask — we are happy to help. Thank you for
helping improve BinML.

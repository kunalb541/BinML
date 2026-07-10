# Contributing to BinML

Thanks for your interest in BinML — a deep-learning classifier for gravitational
microlensing light curves (Flat / PSPL / Binary), built for Nancy Grace Roman Space
Telescope cadence. Contributions of all kinds are welcome: bug reports, new survey
loaders, documentation fixes, tests, and improvements to the model or research pipeline.

By participating you agree that your contributions will be released under the project's
[MIT License](LICENSE). Please be respectful and constructive in all discussions.

## Repository layout

BinML is two things in one repository. Knowing which part your change belongs in makes
review much faster.

- **`binml/`** — the installable inference package (`pip install binml`). This is what
  end users import. It ships the trained weights and depends only on `torch` and `numpy`
  for inference. Modules: `classifier.py`, `preprocess.py`, `evaluate.py`, `surveys.py`,
  `plotting.py`, `cli.py`, `model.py`, and bundled `weights/`. Keep this package small,
  dependency-light, and stable — it is the public API.
- **`pipeline/`** — the research pipeline used to produce the models. This is *not*
  installed by the package: `simulate.py` (event simulation with VBBinaryLensing),
  `select_subset.py` (detectability-aware subset selection), `train.py` (single-GPU
  streaming trainer), `evaluate.py`, `model.py`, `train_modal.py` (Modal L4 orchestration),
  `analysis/` (real-data scripts) and `curricula/` (fine-tuning round scripts). Heavier
  dependencies are expected here.

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

The `[all]` extra pulls in `matplotlib` and `scipy` (used by plotting and evaluation).
For a minimal inference-only environment, `pip install -e .` installs just `torch` and
`numpy`.

## Running tests

The test suite lives in `tests/` and runs with `pytest`:

```bash
python -m pytest tests/ -q
```

Tests are smoke-level and fast: they check that the package imports, that both bundled
models load, and that a synthetic light curve classifies with a valid probability
distribution. Please make sure the suite passes before opening a pull request, and add a
test when you fix a bug or add a feature. Continuous integration runs the same command on
Python 3.9, 3.10, and 3.11 for every push and pull request.

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

## Adding a new survey loader

Survey loaders live in `binml/surveys.py`. Every loader takes a file path and returns a
tuple of three float arrays `(time, mag, mag_err)`, ready to hand straight to
`Classifier.predict`. Existing examples include `load_ogle`, `load_moa`, and
`load_generic`.

To add support for a new survey format:

1. **Write the loader.** Add a function to `binml/surveys.py` that reads your format and
   returns `(time, mag, mag_err)` as `numpy` arrays. Convert flux to magnitude if the
   survey publishes flux, and provide a sensible default error column if one is missing.

   ```python
   def load_mysurvey(path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
       """MySurvey photometry: columns = time, mag, mag_err."""
       a = np.loadtxt(str(path))
       return a[:, 0], a[:, 1], a[:, 2]
   ```

2. **Export it.** Add the function name to the module's `__all__` list so it is part of the
   public `binml.surveys` API.

3. **Register it for auto-dispatch (optional).** If you want the format available through
   `read_lightcurve(path, fmt="mysurvey")` and the `binml classify --format` CLI option,
   add a branch for it in `read_lightcurve`.

4. **Test it.** Add a small test in `tests/` — a tiny fixture file or an inline array is
   enough — asserting that the loader returns three equal-length arrays and that a curve
   classifies without error.

Loaders should not download data implicitly; network access (as in `fetch_ogle_ews`)
should be an explicit, clearly documented function.

## Questions

If anything is unclear, open an issue and ask — we are happy to help. Thank you for
helping improve BinML.

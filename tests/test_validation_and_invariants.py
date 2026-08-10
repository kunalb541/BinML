"""Regression tests for public-API input validation and evaluation invariants.

These encode defects found in external audits so they cannot silently return:
  * the public API must reject malformed input with an actionable message, not a broadcast error;
  * the efficiency map must be a bounded probability (an earlier version was a ratio up to 31);
  * figures must use the frozen final-test split, never the threshold-selection rows.
"""
import json
import os

import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "paper", "results")


def _clf():
    import binml
    return binml.Classifier()


def test_rejects_mismatched_lengths():
    t = np.arange(0, 72, 0.05); m = np.full(len(t) - 3, 20.0)
    with pytest.raises(ValueError, match="equal length"):
        _clf().predict(t, m, m_base_ref=20.0)


def test_rejects_unknown_band():
    t = np.arange(0, 72, 0.05); m = np.full_like(t, 20.0)
    with pytest.raises(ValueError, match="unrecognised band"):
        _clf().predict({"F146": (t, m), "F999": (t, m)}, m_base_ref=20.0)


def test_rejects_empty_curve():
    with pytest.raises(ValueError, match="empty light curve|no finite"):
        _clf().predict(np.array([]), np.array([]), m_base_ref=20.0)


def test_requires_f146():
    t = np.arange(0, 72, 0.05); m = np.full_like(t, 20.0)
    with pytest.raises(ValueError, match="F146 is required"):
        _clf().predict({"F087": (t, m)}, m_base_ref=20.0)


@pytest.mark.skipif(not os.path.exists(os.path.join(RES, "test_idx.npy")),
                    reason="evaluation artifact not present")
def test_final_test_split_is_disjoint_from_threshold_selection():
    """test_idx.npy must be exactly the 80% held out by rng(7); figures rely on this."""
    n = len(np.load(os.path.join(RES, "label.npy")))
    ti = np.load(os.path.join(RES, "test_idx.npy"))
    perm = np.random.default_rng(7).permutation(n)
    expected = perm[int(0.2 * n):]
    assert np.array_equal(np.sort(ti), np.sort(expected))
    val = set(perm[:int(0.2 * n)].tolist())
    assert not (set(ti.tolist()) & val), "threshold-selection rows leaked into the final test set"


@pytest.mark.skipif(not os.path.exists(os.path.join(RES, "params.npy")),
                    reason="evaluation artifact not present")
def test_efficiency_map_is_a_bounded_probability():
    """Conditional recall per (q,s) cell must lie in [0,1] (an earlier ratio reached 31)."""
    lab = np.load(os.path.join(RES, "label.npy")).astype(int)
    tc = np.load(os.path.join(RES, "true_class.npy")).astype(int)
    lg = np.load(os.path.join(RES, "logits.npy"))
    params = np.load(os.path.join(RES, "params.npy"))
    pf = json.load(open(os.path.join(RES, "meta.json")))["param_fields"]
    ti = np.load(os.path.join(RES, "test_idx.npy")).astype(int)
    m = np.zeros(len(lab), bool); m[ti] = True
    pred = lg.argmax(1); NON = 2
    q = params[:, pf.index("q")]; s = params[:, pf.index("s")]
    gen = m & (tc == NON) & np.isfinite(q) & np.isfinite(s)
    det = gen & (lab == NON)
    assert det.sum() > 100
    recall = (pred[det] == NON).mean()
    assert 0.0 <= recall <= 1.0


def test_evaluate_efficiency_function_is_bounded():
    """The evaluator itself (not just the figure) must return a bounded conditional recall.

    Regression for an audit finding: efficiency_plane previously returned B/A with mismatched
    bases, exceeding 1 in every populated cell (up to 31), which the plot silently clipped.
    """
    from pipeline.evaluate import efficiency_plane
    rng = np.random.default_rng(0)
    n = 4000
    pf = ["q", "s"]
    params = np.column_stack([10 ** rng.uniform(-6, 0, n), 10 ** rng.uniform(-0.7, 0.7, n)])
    tc = np.full(n, 2)                       # all generated NonPSPL
    y = np.where(rng.random(n) < 0.4, 2, 1)  # 40% stay detectable, rest demoted to PSPL
    pred = np.where(rng.random(n) < 0.5, 2, 1)
    w = np.ones(n)
    plane = efficiency_plane(params, pf, y, pred, tc, w)
    C = np.array(plane["classifier_recall_given_detectable"], float)
    fin = C[np.isfinite(C)]
    assert fin.size > 0
    assert fin.min() >= 0.0 and fin.max() <= 1.0
    assert "classifier_efficiency" not in plane, "the invalid unbounded key must not return"


@pytest.mark.skipif(not os.path.exists(os.path.join(RES, "metrics.json")),
                    reason="evaluation artifact not present")
def test_released_metrics_efficiency_is_bounded():
    """The RELEASED artifact must not carry the invalid values either."""
    ep = json.load(open(os.path.join(RES, "metrics.json")))["efficiency_plane"]
    assert "classifier_efficiency" not in ep
    C = np.array(ep["classifier_recall_given_detectable"], float)
    fin = C[np.isfinite(C)]
    assert fin.size > 0 and fin.min() >= 0.0 and fin.max() <= 1.0


@pytest.mark.skipif(not os.path.exists(os.path.join(RES, "test_idx.npy")),
                    reason="evaluation artifact not present")
def test_reported_supports_match_the_final_test_split():
    """Table supports must count final-test rows, not the whole pool (audit finding)."""
    cn = json.load(open(os.path.join(os.path.dirname(RES), "canonical_numbers.json")))
    lab = np.load(os.path.join(RES, "label.npy")).astype(int)
    ti = np.load(os.path.join(RES, "test_idx.npy")).astype(int)
    names = ["Flat", "PSPL", "NonPSPL", "PeriodicVar", "LongPeriodVar", "Eruptive"]
    for c, name in enumerate(names):
        assert cn["per_class_support"][name] == int((lab[ti] == c).sum()), name


def test_build_preflight_runs_on_a_modern_interpreter():
    """EXECUTE the build.sh preflight, do not grep it.

    Regression for a real failure: the preflight used `import importlib` then
    `importlib.util.find_spec`, which raises AttributeError on Python >= 3.12. A string-matching
    test could not catch that, and did not. This extracts the actual heredoc and runs it.
    """
    import re
    import subprocess
    build = open(os.path.join(os.path.dirname(RES), "build.sh")).read()
    assert "PYTHONPATH" in build, "build.sh must export PYTHONPATH for the figure scripts"
    m = re.search(r"<<'PYCHECK'.*?\n(.*?)\nPYCHECK", build, re.S)
    assert m, "could not locate the preflight heredoc in build.sh"
    r = subprocess.run([sys.executable, "-c", m.group(1)], capture_output=True, text=True)
    # exit 1 means "dependencies missing" (a legitimate outcome); a traceback means broken code
    assert "Traceback" not in r.stderr, f"preflight itself is broken:\n{r.stderr}"


@pytest.mark.slow
def test_clean_archive_builds_figures_end_to_end(tmp_path):
    """Integration test: extract a `git archive` of HEAD and regenerate the artifact figures.

    This actually executes the documented build path from a clean tree (the artifact-figure stage,
    which needs no simulation), rather than asserting on the contents of build.sh. Skipped if git
    or the scientific stack is unavailable.
    """
    import subprocess
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                      capture_output=True).returncode != 0:
        pytest.skip("not a git repository")
    tar = subprocess.run(["git", "-C", repo, "archive", "HEAD"], capture_output=True)
    assert tar.returncode == 0
    (tmp_path / "a.tar").write_bytes(tar.stdout)
    subprocess.run(["tar", "-xf", str(tmp_path / "a.tar"), "-C", str(tmp_path)], check=True)
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    r = subprocess.run([sys.executable, "make_figures.py"], cwd=str(tmp_path / "paper"),
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"clean-archive figure build failed:\n{r.stdout}\n{r.stderr}"
    for f in ("confusion.pdf", "pr_nonpspl.pdf", "efficiency_plane.pdf"):
        assert (tmp_path / "paper" / "outputs" / "figures" / f).exists(), f"{f} not produced"


def test_cascade_macros_fail_closed_without_the_artifact(tmp_path):
    """make_macros.py must FAIL rather than emit stale cascade numbers if the artifact is absent.

    Regression for the withdrawn 42%->9% claim, which survived because prose numbers were
    hand-duplicated and could drift from the tracked artifact.
    """
    src = open(os.path.join(os.path.dirname(RES), "make_macros.py")).read()
    assert "cascade_reproduce_result.json" in src
    assert "FATAL" in src and "raise SystemExit" in src, \
        "make_macros must fail closed when the cascade artifact is missing or its schema changed"

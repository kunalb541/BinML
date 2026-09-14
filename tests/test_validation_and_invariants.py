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
    # A real bound, not a tautology: the pooled conditional recall on detectable binaries must
    # agree with the committed headline completeness to within the two quantities' definitional
    # difference (argmax vs thresholded), i.e. sit in the same neighbourhood, not merely in [0,1].
    headline = json.load(open(os.path.join(RES, "metrics.json")))["headline"]["completeness_at_fixed_purity"]
    assert abs(recall - headline) < 0.15, (recall, headline)


@pytest.mark.skipif(not os.path.exists(os.path.join(RES, "params.npy")),
                    reason="evaluation artifact not present")
def test_efficiency_plane_reproduces_committed_artifact_on_test_rows():
    """Regression for an audit finding: evaluate_checkpoint once passed the FULL 450,589-row pool
    to efficiency_plane while every other block used the test rows, so regenerating metrics.json
    would silently change the plane (+25% n_eff). The committed plane is the test-only
    computation; this pins that the code path reproduces it."""
    from pipeline.evaluate import efficiency_plane, population_weights
    lab = np.load(os.path.join(RES, "label.npy")).astype(int)
    tc = np.load(os.path.join(RES, "true_class.npy")).astype(int)
    lg = np.load(os.path.join(RES, "logits.npy"))
    params = np.load(os.path.join(RES, "params.npy"))
    kp = np.load(os.path.join(RES, "keep_prob.npy"))
    pf = json.load(open(os.path.join(RES, "meta.json")))["param_fields"]
    ti = np.load(os.path.join(RES, "test_idx.npy")).astype(int)
    w = population_weights(kp)
    plane = efficiency_plane(params[ti], pf, lab[ti], lg.argmax(1)[ti], tc[ti], w[ti])
    committed = json.load(open(os.path.join(RES, "metrics.json")))["efficiency_plane"]
    got = np.nansum(np.asarray(plane["n_eff"], float))
    want = np.nansum(np.asarray(committed["n_eff"], float))
    assert abs(got - want) < 1e-6 * max(want, 1.0), (got, want)
    a = np.asarray(plane["survey_detectability"], float)
    b = np.asarray(committed["survey_detectability"], float)
    assert np.allclose(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1), atol=1e-9)


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
    # The heredoc OPENER spans several lines, because the `|| { echo ...; exit 1; }` error handler
    # is a multi-line shell string. Matching to the first newline captured that error message
    # instead of the Python body -- and running a line of prose raises IndentationError, whose
    # output contains no "Traceback", so the old assertion passed while testing nothing. Anchor on
    # the closing brace of the handler, and verify we really got the preflight before running it.
    m = re.search(r"<<'PYCHECK'.*?\}\s*\n(.*?)\nPYCHECK", build, re.S)
    assert m, "could not locate the preflight heredoc in build.sh"
    body = m.group(1)
    assert "importlib.util" in body and "find_spec" in body, (
        "extracted the wrong block from build.sh -- this test must execute the dependency "
        f"preflight, got:\n{body}")
    r = subprocess.run([sys.executable, "-c", body], capture_output=True, text=True)
    # A traceback means the preflight itself is broken.
    assert "Traceback" not in r.stderr, f"preflight itself is broken:\n{r.stderr}"
    # Accepting "any exit 1" was too weak: exit 1 is also what a crash produces. Pin the branch.
    if r.returncode == 0:
        assert r.stderr.strip() == "", f"clean preflight should be silent, got:\n{r.stderr}"
    else:
        assert r.returncode == 1, f"unexpected preflight exit {r.returncode}"
        assert "missing" in (r.stdout + r.stderr).lower(), (
            "exit 1 must be the missing-dependency branch, with the names printed; got:\n"
            f"stdout={r.stdout!r} stderr={r.stderr!r}")

    # And prove the missing-dependency branch actually fires, by running the same code against an
    # import name that cannot exist. Otherwise this test never exercises the failure path at all.
    probe = body.replace('"numpy", "matplotlib"',
                               '"numpy", "matplotlib", "definitely_not_a_real_module_xyz"')
    assert probe != body, "preflight module list changed; update this probe"
    r2 = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert r2.returncode == 1, "preflight must exit 1 when a dependency is missing"
    assert "definitely_not_a_real_module_xyz" in (r2.stdout + r2.stderr), \
        "preflight must name the missing dependency"


@pytest.mark.slow
def test_clean_archive_builds_figures_end_to_end(tmp_path):
    """Integration test: build a prospective clean tree and regenerate artifact figures.

    This actually executes the documented build path from a clean tree (the artifact-figure stage,
    which needs no simulation), rather than asserting on the contents of build.sh. Skipped if git
    or the scientific stack is unavailable.
    """
    import subprocess
    root = _archive_head(tmp_path)
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    r = subprocess.run([sys.executable, "make_figures.py"], cwd=str(root / "paper"),
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"clean-archive figure build failed:\n{r.stdout}\n{r.stderr}"
    for f in ("confusion.pdf", "pr_nonpspl.pdf", "efficiency_plane.pdf"):
        assert (root / "paper" / "outputs" / "figures" / f).exists(), f"{f} not produced"


def _archive_head(tmp_path):
    """Create a prospective clean Git tree from tracked plus untracked working-tree files.

    A literal ``git archive HEAD`` would silently test the last commit while audit fixes are still
    uncommitted.  Copy the exact files that would be tracked, commit them in an isolated temporary
    repository, and let the build exercise its own clean-archive path there.
    """
    import shutil
    import subprocess
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                      capture_output=True).returncode != 0:
        pytest.skip("not a git repository")
    listing = subprocess.run(
        ["git", "-C", repo, "ls-files", "--cached", "--others", "--exclude-standard"],
        check=True, capture_output=True, text=True).stdout.splitlines()
    for name in listing:
        src = os.path.join(repo, name)
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "audit@example.invalid"], cwd=tmp_path,
                   check=True)
    subprocess.run(["git", "config", "user.name", "Audit Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "prospective clean tree"], cwd=tmp_path, check=True)
    return tmp_path


def _worktree_copy(tmp_path):
    """Copy the parts of the WORKING TREE that make_macros.py reads.

    Deliberately not `git archive HEAD`: this test asserts how the current code behaves, and
    archiving HEAD silently exercised the last committed version instead -- so a fix in the working
    tree went untested and a regression in it would go unnoticed until after the commit. The
    clean-archive tests below are the ones that check what is committed.
    """
    import shutil
    root = tmp_path / "tree"
    (root).mkdir()
    for sub in ("paper", "validation"):
        # .npz is NOT excluded: make_figures reads validation/cascade_trace.npz and
        # matched_traces.npz. Excluding them made make_figures fail, which the skip below then
        # swallowed -- a test that never runs anywhere.
        shutil.copytree(os.path.join(REPO, sub), root / sub,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pdf", "runs"))
    return root


def test_cascade_macros_fail_closed_without_the_artifact(tmp_path):
    """make_macros.py must FAIL rather than emit stale cascade numbers if the artifact is absent.

    Regression for the withdrawn 42%->9% claim, which survived because prose numbers were
    hand-duplicated and could drift from the tracked artifact.

    This EXECUTES make_macros.py in a clean archive with the artifact removed, and again with a
    key deleted from it. An earlier version of this test only grepped make_macros.py for the
    strings "FATAL" and "raise SystemExit", which would pass even if the guard were unreachable.
    """
    import json as _json
    import subprocess
    root = _worktree_copy(tmp_path)
    art = root / "validation" / "cascade_reproduce_result.json"
    if not art.exists():
        pytest.skip("cascade artifact not present")
    env = dict(os.environ, PYTHONPATH=str(root))

    # make_macros reads paper/outputs/figures_stats.json, which make_figures GENERATES and which
    # is deliberately gitignored -- it is derived, not source. build.sh therefore runs figures
    # before macros, and this test has to honour that order. An earlier version called
    # make_macros standalone and failed in CI on a missing figures_stats key, which was the test
    # skipping a build step rather than a defect in make_macros.
    gen = subprocess.run([sys.executable, "make_figures.py"], cwd=str(root / "paper"),
                         env=env, capture_output=True, text=True)
    if gen.returncode != 0:
        # Only a genuinely absent plotting stack is a legitimate skip. Anything else is a real
        # failure and must not be hidden -- an earlier version of this skip swallowed a missing
        # input artifact and the test silently stopped running.
        if "ModuleNotFoundError" in gen.stderr or "No module named" in gen.stderr:
            pytest.skip(f"plotting stack unavailable:\n{gen.stderr[-300:]}")
        raise AssertionError(f"make_figures failed for a non-dependency reason:\n{gen.stderr[-800:]}")

    def run():
        return subprocess.run([sys.executable, "make_macros.py"], cwd=str(root / "paper"),
                              env=env, capture_output=True, text=True)

    ok = run()
    assert ok.returncode == 0, f"make_macros must succeed with the artifact present:\n{ok.stderr}"

    # every committed input of make_macros.py is hashed in the manifest (figures_stats.json is generated by
    # make_figures.py and gitignored); --list-inputs writes nothing
    listed = subprocess.run([sys.executable, "make_macros.py", "--list-inputs"], cwd=str(root / "paper"),
                            env=env, capture_output=True, text=True)
    assert listed.returncode == 0, listed.stderr
    manifest = _json.load(open(root / "paper" / "results" / "MANIFEST.json"))["files"]
    inputs = [x for x in listed.stdout.split() if not x.startswith("paper/outputs/")]
    assert "validation/cascade_reproduce_result.json" in inputs
    assert not [x for x in inputs if x not in manifest], [x for x in inputs if x not in manifest]

    orig_text = art.read_text()
    payload = _json.loads(orig_text)
    art.unlink()
    missing = run()
    assert missing.returncode != 0, "make_macros must fail when the cascade artifact is absent"
    assert "FATAL" in (missing.stdout + missing.stderr)

    # Schema drift must fail too, not silently drop a macro the manuscript uses.
    payload.pop("median_lag_non_premature_days", None)
    art.write_text(_json.dumps(payload))
    drifted = run()
    assert drifted.returncode != 0, \
        "make_macros must fail when the cascade artifact loses a required key"
    assert "median_lag_non_premature_days" in (drifted.stdout + drifted.stderr)

    # The released checkpoint's stress numbers (third verification: the suite was scored with stage 5) must not
    # silently fall back to the suite's stage-5 values.
    art.write_text(orig_text)
    stress = root / "validation" / "stress_rescore_local.json"
    stress_text = stress.read_text()
    stress.unlink()
    nostress = run()
    assert nostress.returncode != 0, "make_macros must fail without validation/stress_rescore_local.json"
    assert "stress_rescore_local" in (nostress.stdout + nostress.stderr)

    # values that contradict a stress sentence must fire the guard for that sentence (fourth verification)
    def perturbed(edit, expected):
        d = _json.loads(stress_text); edit(d); stress.write_text(_json.dumps(d))
        r = run(); out = r.stdout + r.stderr
        assert r.returncode != 0 and expected in out, (expected, out[-400:])
    sub = lambda d, t: d["subset"][t]["released"]
    perturbed(lambda d: sub(d, "oor_pspl_shortte")["pspl_label_by_generator_class"]["single_lenses"]["argmax_fractions"].update(NonPSPL=0.3),
              "'most' are called anomalies")
    perturbed(lambda d: sub(d, "oor_flat_faint").update(pspl_label_above_frozen_w=sub(d, "natural")["pspl_label_above_frozen_w"]),
              "rises' well above")
    perturbed(lambda d: d["quoted"]["natural_macro_f1"].update(subset_released=0.8), "reproduces' its held-out macro-F1")
    perturbed(lambda d: d["precision_counterfactual"]["released"].update(faint_at_natural_mix=0.7),
              "faint photometry, not the sweep's class mix")
    perturbed(lambda d: d["suite_label_fractions"]["oor_pspl_shortte"].update(PSPL=0.30), "oor_pspl_shortte tier's PSPL fraction differs")
    perturbed(lambda d: d["precision_at_natural_prevalence"]["planetary"].update(released=0.1), "only through prevalence")
    # sixth check: the guards that had no case, and the tightened agreement bound
    perturbed(lambda d: sub(d, "oor_np_widesep")["label_fractions"].update(NonPSPL=d["suite_label_fractions"]["oor_np_widesep"]["NonPSPL"]),
              "labels clearly fewer detectable anomalies")
    perturbed(lambda d: sub(d, "oor_pspl_shortte_current")["single_lens_above_frozen_tE_0p25_1"].update(frac=0.01),
              "matched-timescale sub-day rate is far above")
    perturbed(lambda d: sub(d, "oor_pspl_shortte")["pspl_label_by_generator_class"]["single_lenses"].update(frac_above_frozen_threshold=0.45),
              "sub-day range runs from the legacy tier")
    def _z3(d):                                     # a planetary-tier NonPSPL fraction 3 standard errors off the suite's
        fs_ = d["suite_label_fractions"]["planetary"]["NonPSPL"]; n_ = sub(d, "planetary")["n"]
        sub(d, "planetary")["label_fractions"]["NonPSPL"] = fs_ - 3 * (fs_ * (1 - fs_) / n_) ** 0.5
    perturbed(_z3, "planetary tier's NonPSPL fraction differs")
    stress.write_text(stress_text)
    assert run().returncode == 0

    # sixth check: guards on other artifacts, each fires on its own sentence
    def art_perturbed(rel, edit, expected):
        p_ = root / rel; t_ = p_.read_text(); d = _json.loads(t_); edit(d); p_.write_text(_json.dumps(d))
        try:
            r = run(); out = r.stdout + r.stderr
            assert r.returncode != 0 and expected in out, (rel, expected, out[-400:])
        finally:
            p_.write_text(t_)
    def _grid_worse(d):
        g = d["sensitivity"]["evaluation_grid_days"]; g["1.0"]["premature_rate_of_eligible"] = g["0.5"]["premature_rate_of_eligible"]
    art_perturbed("validation/cascade_reproduce_result.json", _grid_worse, "coarsening the grid lowers the premature rate")
    art_perturbed("validation/cascade_reproduce_result.json",
                  lambda d: d["sensitivity"]["evaluation_grid_days"]["2.0"].update(detection_fraction=0.83), "almost unchanged detection")
    def _abl_unamb(d):                              # an augmented-arm alert far before its grid onset
        e = next(e for e in d["cascade_on"]["realtime"]["events"] if e["first_thr"] is not None)
        e["first_thr"] = e["t_anom"] - 10.0; e["premature_thr"] = True
    art_perturbed("validation/ablations_result.json", _abl_unamb, "fall inside the grid window")
    def _abl_off_ontime(d):                         # the unaugmented arm's early alerts moved inside the window
        for e in d["cascade_off"]["realtime"]["events"]:
            if e["premature_thr"] and e["first_thr"] <= e["t_anom"] - 7.2:
                e["first_thr"] = e["t_anom"] - 1.0
    art_perturbed("validation/ablations_result.json", _abl_off_ontime, "favour the augmented arm under both rules")
    def _abl_on_early_argmax(d):                    # twenty of the augmented arm's argmax alerts moved before the window
        k = 0
        for e in d["cascade_on"]["realtime"]["events"]:
            if k < 20 and e["premature_argmax"] and e["first_argmax"] > e["t_anom"] - 7.2:
                e["first_argmax"] = e["t_anom"] - 10.0; k += 1
    art_perturbed("validation/ablations_result.json", _abl_on_early_argmax, "'almost all of the augmented arm's premature alerts")
    art_perturbed("validation/ablations_result.json", lambda d: d["cascade_on"]["realtime"].update(premature_rate_argmax=0.2),
                  "opposite directions")
    art_perturbed("validation/labelling_ablation_result.json", lambda d: d["arms"]["labels_generator"].update(anomaly_purity_at_thr=0.5),
                  "many false flags per true one")
    art_perturbed("validation/gap_matched_result.json", lambda d: d["arms"][-1].update(regular=0.0, regular_ci=[0.0, 0.031]),
                  "empty two-hour bins, not sparsity")
    art_perturbed("validation/gap_matched_result.json", lambda d: d["arms"][0].update(nightly=0.5, nightly_ci=[0.4, 0.6]),
                  "the gapped schedule fails where uniform sampling works")
    art_perturbed("validation/baselines_result.json", lambda d: d.update(nonpspl_prevalence=0.05), "canonical baselines no longer match")
    art_perturbed("validation/inference_benchmark_result.json", lambda d: d["seconds_per_batch"].update(max=2 * d["seconds_per_batch"]["min"]),
                  "otherwise idle machine")
    art_perturbed("paper/results/metrics.json", lambda d: d["slices"]["no_blue_band"].update(recall=0.5),
                  "canonical slice no_blue_band")
    # seventh check: the remaining new guards, including the integrity checks
    art_perturbed("validation/cascade_reproduce_result.json",
                  lambda d: d["sensitivity"]["evaluation_grid_days"]["1.0"].update(median_lag_non_premature_days=4.5), "later alerts")
    art_perturbed("validation/ablations_result.json", lambda d: d["cascade_on"]["realtime"]["events"][0].update(t_anom=50.0),
                  "no longer on the 7.2 d generator grid")
    def _abl_flag(d):
        e = next(e for e in d["cascade_on"]["realtime"]["events"] if not e["premature_thr"]); e["premature_thr"] = True
    art_perturbed("validation/ablations_result.json", _abl_flag, "premature flags disagree")
    art_perturbed("validation/inference_benchmark_result.json", lambda d: d.update(load_average_1_5_15_min_before=[9.0, 9.0, 9.0]),
                  "otherwise idle machine")
    art_perturbed("validation/gap_matched_result.json", lambda d: d["arms"][0].pop("regular"), "predates the regular arm")
    art_perturbed("validation/gap_matched_result.json", lambda d: d["arms"].pop(), "three visit counts")
    art_perturbed("validation/gap_matched_result.json", lambda d: d["full_cadence"].update(recall=0.6, ci=[0.5, 0.7]),
                  "thinning alone costs recall too")
    tr = root / "validation" / "cascade_trace.npz"; tr_bytes = tr.read_bytes()
    try:
        z = dict(np.load(tr)); z["t_anom_fine"] = z["t_anom_coarse"] - 10.0; np.savez(tr, **z)
        r = run(); out = r.stdout + r.stderr
        assert r.returncode != 0 and "precedes the previous grid cut" in out, out[-400:]
    finally:
        tr.write_bytes(tr_bytes)

    # the Sec. results guards on where the misses sit (fifth verification): each fires on its own sentence
    fs_path = root / "paper" / "outputs" / "figures_stats.json"
    fs_text = fs_path.read_text()
    def fs_perturbed(key, value, expected):
        d = _json.loads(fs_text); d[key] = value; fs_path.write_text(_json.dumps(d))
        r = run(); out = r.stdout + r.stderr
        assert r.returncode != 0 and expected in out, (key, expected, out[-400:])
    fs_perturbed("np_to_pspl_wide_pct", 45.0, "'most' NonPSPL->PSPL confusions")
    fs_perturbed("wide_miss_rate_by_dchi2_hi", 0.6, "at every evidence strength up to")
    fs_perturbed("wide_miss_rate_above_1e6", 0.3, "strongest wide anomalies")
    fs_perturbed("miss_to_pspl_pct", 80.0, "almost always called a single lens")
    fs_perturbed("eff_cond_recall_lowq_hi", 0.99, "log q < -4 lies below the plane's median")
    fs_perturbed("calib_mid_anomaly_freq_stored", 0.05, "the ECE is small because most weight sits below 0.1")
    fs_perturbed("calib_ece_mid_share_pct", 40.0, "the ECE is small because most weight sits below 0.1")
    fs_perturbed("thr_miss_wide_pct", 60.0, "at the operating threshold a smaller share")
    fs_perturbed("nonpspl_kept_pct_w", 50.0, "stored-event fraction overstates")
    fs_perturbed("notwide_weak_share_of_miss_pct", 20.0, "concentrate at weak anomalies")
    fs_perturbed("recall_wide_smallq", 0.7, "confined to q > 0.1")
    fs_perturbed("eff_cond_recall_wide_bigq_hi", 0.99, "clearly below its median")
    fs_perturbed("eff_nd_min_cell", 1000, "low-support")
    fs_perturbed("det_stellar_pct", 40.0, "mostly stellar-mass-ratio")
    fs_path.write_text(fs_text)
    assert run().returncode == 0


def test_frozen_evaluation_manifest_is_enforced(tmp_path):
    """Reject array/file tampering and a result payload not emitted by its reducer."""
    import hashlib
    import shutil
    import subprocess

    repo = tmp_path
    root = repo / "paper"
    results = root / "results"
    results.mkdir(parents=True)
    shutil.copy2(os.path.join(os.path.dirname(RES), "validate_artifacts.py"), root)
    weights = repo / "binml" / "weights"
    weights.mkdir(parents=True)
    checkpoint = weights / "binml.pt"
    checkpoint.write_bytes(b"test checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for name in ("metrics.json", "meta.json"):
        (results / name).write_text(json.dumps({"checkpoint_sha256": checkpoint_sha}))

    original = np.array([0, 1, 2, 3], dtype=np.int64)
    np.save(results / "label.npy", original)
    digest = hashlib.sha256(np.ascontiguousarray(original).tobytes()).hexdigest()[:16]
    validation = repo / "validation"
    validation.mkdir()
    np.savez(validation / "cascade_trace.npz",
             provenance=np.array(json.dumps({"checkpoint_sha256": checkpoint_sha})))
    (validation / "matched_traces.npz").write_bytes(b"matched trace")
    for stem in ("cascade", "cascade_matched"):
        trace = validation / ("cascade_trace.npz" if stem == "cascade" else "matched_traces.npz")
        reducer = validation / f"{stem}_reduce.py"
        result = validation / ("cascade_reproduce_result.json" if stem == "cascade"
                               else "cascade_matched_result.json")
        option = "trace" if stem == "cascade" else "traces"
        reducer.write_text(
            "import argparse, hashlib, json\n"
            "from pathlib import Path\n"
            "ap=argparse.ArgumentParser()\n"
            f"ap.add_argument('--{option}')\n"
            "ap.add_argument('--out')\n"
            "a=ap.parse_args()\n"
            f"trace=Path(a.{option})\n"
            "sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()\n"
            "obj={'reduction_provenance': {'input_trace_sha256': sha(trace), "
            "'reducer_sha256': sha(__file__)}, 'sentinel': 1}\n"
            "json.dump(obj, open(a.out, 'w'), indent=2)\n")
        result.write_text(json.dumps({"reduction_provenance": {
            "input_trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            "reducer_sha256": hashlib.sha256(reducer.read_bytes()).hexdigest()},
            "sentinel": 1}, indent=2))

    hashed = {}
    for path in (checkpoint, results / "metrics.json", results / "meta.json"):
        rel = str(path.relative_to(repo))
        hashed[rel] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (results / "MANIFEST.json").write_text(json.dumps({
        "arrays": {"label.npy": {"shape": [4], "dtype": "int64", "sha256_16": digest}},
        "files": hashed,
    }))
    script = root / "validate_artifacts.py"
    ok = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr

    path = root / "results" / "label.npy"
    arr = np.load(path)
    arr[0] = (int(arr[0]) + 1) % 6
    np.save(path, arr)
    bad = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert bad.returncode != 0
    assert "label.npy" in (bad.stdout + bad.stderr)
    assert "sha256_16" in (bad.stdout + bad.stderr)

    np.save(path, original)
    result = validation / "cascade_reproduce_result.json"
    payload = json.loads(result.read_text())
    payload["sentinel"] = 999
    result.write_text(json.dumps(payload, indent=2))
    bad_result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert bad_result.returncode != 0
    assert "differs from deterministic reducer output" in (bad_result.stdout + bad_result.stderr)

    payload["sentinel"] = 1
    result.write_text(json.dumps(payload, indent=2))
    metrics = results / "metrics.json"
    metrics.write_text(json.dumps({"checkpoint_sha256": checkpoint_sha, "tampered": True}))
    bad_json = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert bad_json.returncode != 0
    assert "paper/results/metrics.json" in (bad_json.stdout + bad_json.stderr)
    assert "file sha256" in (bad_json.stdout + bad_json.stderr)


@pytest.mark.skipif(not os.path.exists(os.path.join(REPO, "validation", "matched_traces.npz")),
                    reason="matched cascade trace not present")
def test_matched_cascade_uses_exact_counts_and_unrounded_thresholds(tmp_path):
    """Displayed counts, paired masks and tests must all use one full-precision threshold.

    Regression for a reducer that chose the nearest point on a sparse grid (50.5% versus 44.5%
    at the nominal 50% target), rounded the chosen threshold, and then recomputed McNemar on a
    different set of events from the summary table.
    """
    import hashlib
    import subprocess

    trace = os.path.join(REPO, "validation", "matched_traces.npz")
    out = tmp_path / "matched.json"
    r = subprocess.run([sys.executable, os.path.join(REPO, "validation",
                                                     "cascade_matched_reduce.py"),
                        "--traces", trace, "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    result = json.loads(out.read_text())
    assert result["reduction_provenance"]["input_trace_sha256"] == \
        hashlib.sha256(open(trace, "rb").read()).hexdigest()
    assert "not confirmatory" in result["inference_note"]
    z = np.load(trace, allow_pickle=False)
    cuts = z["cuts"].astype(float)
    onset = z["onset"].astype(float)
    arms = [str(a) for a in z["arms"]]

    for row in result["matched"].values():
        masks = {}
        achieved = set()
        for i, arm in enumerate(arms):
            threshold = row[arm]["threshold"]
            assert float.fromhex(row[arm]["threshold_hex"]) == threshold
            over = np.nan_to_num(z["traces"][i].astype(float), nan=0.0) >= threshold
            hit = over.any(1)
            alert = np.where(hit, cuts[np.argmax(over, axis=1)], np.nan)
            premature = hit & (alert < onset)
            achieved.add(int(hit.sum()))
            assert int(hit.sum()) == row[arm]["achieved_detection_count"]
            assert int(premature.sum()) == row[arm]["n_premature"]
            masks[arm] = premature
        assert len(achieved) == 1, "the arms must have exactly matched detection counts"
        a, b = arms
        assert int((masks[a] != masks[b]).sum()) == row["mcnemar"]["n_discordant"]


@pytest.mark.skipif(not os.path.exists(os.path.join(REPO, "validation", "cascade_trace.npz")),
                    reason="cascade trace not present")
def test_cascade_reduction_is_exact_and_trace_linked(tmp_path):
    """Frozen trace reduction must exactly reproduce the per-event fixture and record its input."""
    import hashlib
    import subprocess

    trace = os.path.join(REPO, "validation", "cascade_trace.npz")
    out = tmp_path / "cascade.json"
    r = subprocess.run([sys.executable, os.path.join(REPO, "validation", "cascade_reduce.py"),
                        "--trace", trace, "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    result = json.loads(out.read_text())
    fixture = result["published_artifact_agreement"]
    assert fixture["detection_status_mismatches"] == 0
    assert fixture["crossings_differing"] == 0
    assert fixture["max_deviation_days"] == 0.0
    assert result["reduction_provenance"]["input_trace_sha256"] == \
        hashlib.sha256(open(trace, "rb").read()).hexdigest()


@pytest.mark.slow
def test_clean_archive_full_paper_build(tmp_path):
    """The documented build path, end to end, from a clean `git archive HEAD`.

    The figure-only integration test above does not cover build.sh itself: its preflight, its
    PYTHONPATH export, macro generation, or LaTeX. This runs the whole script and checks the PDF
    exists. Skipped when LaTeX is unavailable, which is why it is marked slow rather than being
    the only reproducibility test.
    """
    import shutil
    import subprocess
    if shutil.which("pdflatex") is None and shutil.which("latexmk") is None:
        pytest.skip("no LaTeX toolchain available")
    root = _archive_head(tmp_path)
    r = subprocess.run(["bash", "build.sh"], cwd=str(root / "paper"),
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, f"clean-archive build.sh failed:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}"
    pdf = root / "paper" / "paper.pdf"
    assert pdf.exists() and pdf.stat().st_size > 100_000, "build.sh produced no usable PDF"


def test_referee_round_archive_is_committed_and_matches_its_hashes():
    """Data availability cites validation/referee_round_archive/; every file referee_round.json hashes must be tracked
    by git and match (fourth verification, 2026-09-13: the directory had been gitignored)."""
    import hashlib
    import subprocess
    rr = json.load(open(os.path.join(REPO, "validation", "referee_round.json")))["archive"]
    tracked = set(subprocess.run(["git", "ls-files", rr["dir"]], cwd=REPO, capture_output=True, text=True).stdout.split())
    if not tracked and not os.path.isdir(os.path.join(REPO, ".git")):
        pytest.skip("not a git checkout")
    for name, sha in rr["sha256"].items():
        path = f"{rr['dir']}/{name}"
        assert path in tracked, f"{path} is not committed"
        assert hashlib.sha256(open(os.path.join(REPO, path), "rb").read()).hexdigest() == sha, path


def test_stress_numbers_rederive_from_the_archive(tmp_path):
    """Every number of validation/stress_rescore_local.json recomputes from validation/stress_rescore_archive/ alone
    (fifth verification: the per-event evaluations had lived only in a local work directory)."""
    import subprocess
    out = tmp_path / "stress.json"
    r = subprocess.run([sys.executable, "validation/stress_rescore_local.py", "--from-archive", "--out", str(out)],
                       cwd=REPO, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=REPO))
    assert r.returncode == 0, r.stdout + r.stderr
    committed = json.load(open(os.path.join(REPO, "validation", "stress_rescore_local.json")))
    rebuilt = json.load(open(out))
    # to floating-point precision: the artifact stores unrounded floats, and numpy's summation order differs between
    # platforms in the last bits (CI's Linux runners, sixth check); integers, strings and structure must match exactly
    import math

    def close(a, b, path):
        if isinstance(a, dict):
            assert isinstance(b, dict) and a.keys() == b.keys(), path
            for key in a:
                close(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, list):
            assert isinstance(b, list) and len(a) == len(b), path
            for i, (x, y) in enumerate(zip(a, b)):
                close(x, y, f"{path}[{i}]")
        elif isinstance(a, float) or isinstance(b, float):
            assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), (path, a, b)
        else:
            assert a == b, (path, a, b)
    for k in ("subset", "quoted", "single_lens_recall", "precision_at_natural_prevalence", "precision_counterfactual",
              "suite_label_fractions"):
        close(rebuilt[k], committed[k], k)

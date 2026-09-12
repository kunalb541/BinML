"""paper/make_gulls_macros.py is fail-closed: a missing artifact or block exits non-zero and writes NOTHING.

The 2026-09-12 re-verification found tables written before later inputs were checked (new tables next to
stale macros) and four silent fallbacks ('FATAL' as a macro value, 'ahead in 0 of the 0 seasons', five
cascade macros silently undefined, a sub-day range of 0). Each case below reproduces one of them on a
copy of the committed inputs."""
import json
import os
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTS = ("paper/gulls_macros.tex", "paper/outputs/gulls_transfer_table.tex", "paper/outputs/gulls_gap_table.tex")


def _tree(tmp):
    for d in ("paper/results", "paper/outputs", "validation/gulls", "pipeline"):
        os.makedirs(os.path.join(tmp, d), exist_ok=True)
    shutil.copy(os.path.join(REPO, "paper/make_gulls_macros.py"), os.path.join(tmp, "paper"))
    shutil.copy(os.path.join(REPO, "paper/results/metrics.json"), os.path.join(tmp, "paper/results"))
    for f in os.listdir(os.path.join(REPO, "validation/gulls")):
        if f.endswith(".json"):
            shutil.copy(os.path.join(REPO, "validation/gulls", f), os.path.join(tmp, "validation/gulls"))
    for f in ("referee_round.json", "truth_relabel_impact.json", "cascade_reproduce_result.json"):
        shutil.copy(os.path.join(REPO, "validation", f), os.path.join(tmp, "validation"))
    for f in ("__init__.py", "priors.py"):
        shutil.copy(os.path.join(REPO, "pipeline", f), os.path.join(tmp, "pipeline"))
    for o in OUTS:
        open(os.path.join(tmp, o), "w").write("SENTINEL\n")


def _run(tmp):
    return subprocess.run([sys.executable, "paper/make_gulls_macros.py"], cwd=tmp, capture_output=True, text=True)


def _edit(tmp, name, fn):
    p = os.path.join(tmp, "validation/gulls", name); d = json.load(open(p)); fn(d); json.dump(d, open(p, "w"))


def _drop_seed(tmp):
    _edit(tmp, "transfer_tradeoff_all.json", lambda d: d["models"].pop("fspl5s_seasons_g08_s3"))


CASES = {
    "missing artifact": lambda t: os.remove(os.path.join(t, "validation/gulls/transfer_colour_ablation.json")),
    "late missing artifact": lambda t: os.remove(os.path.join(t, "validation/gulls/cascade_gulls.json")),
    "n_scored_test null": lambda t: _edit(t, "schedule_finetune.json",
                                          lambda d: [e["clean"].update(n_scored_test=None) for e in d["heldout_eval"].values()]),
    "no per-season block": lambda t: _edit(t, "transfer_tradeoff_all.json", lambda d: d["models"]["sched_sched_seasons"].pop("by_season")),
    "no calibrated cascade": lambda t: _edit(t, "cascade_gulls.json",
                                             lambda d: d["results"].pop("fspl5s_seasons_g08|f146|calibrated_seasons_fullpool")),
    "no PeriodicVar": lambda t: _edit(t, "transfer_subday.json", lambda d: d["models"]["shipped"]["argmax_distribution"].pop("PeriodicVar")),
    "no referee round": lambda t: os.remove(os.path.join(t, "validation/referee_round.json")),
    "no colour fine-tunes": lambda t: _edit(t, "../referee_round.json", lambda d: d["colour_ablation"].pop("finetuned_on_train_colour")),
    "no third seed": _drop_seed,
    "no prefix-rule check": lambda t: _edit(t, "../truth_relabel_impact.json", lambda d: d["results"].pop("truncation_vs_prefix_rule")),
}


def test_complete_inputs_write_everything(tmp_path):
    _tree(str(tmp_path))
    r = _run(str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    for o in OUTS:
        assert "SENTINEL" not in open(os.path.join(tmp_path, o)).read()
    assert open(os.path.join(tmp_path, OUTS[0])).read() == open(os.path.join(REPO, OUTS[0])).read()


@pytest.mark.parametrize("case", sorted(CASES))
def test_any_missing_input_writes_nothing(tmp_path, case):
    _tree(str(tmp_path)); CASES[case](str(tmp_path))
    r = _run(str(tmp_path))
    assert r.returncode != 0 and "FATAL" in (r.stdout + r.stderr), case
    for o in OUTS:
        assert open(os.path.join(tmp_path, o)).read() == "SENTINEL\n", (case, o)

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
OUTS = ("paper/gulls_macros.tex", "paper/outputs/gulls_transfer_table.tex", "paper/outputs/gulls_gap_table.tex",
        "paper/outputs/gulls_seed_table.tex")


def _tree(tmp):
    for d in ("paper/results", "paper/outputs", "validation/gulls", "pipeline"):
        os.makedirs(os.path.join(tmp, d), exist_ok=True)
    shutil.copy(os.path.join(REPO, "paper/make_gulls_macros.py"), os.path.join(tmp, "paper"))
    shutil.copy(os.path.join(REPO, "paper/results/metrics.json"), os.path.join(tmp, "paper/results"))
    for f in os.listdir(os.path.join(REPO, "validation/gulls")):
        if f.endswith(".json"):
            shutil.copy(os.path.join(REPO, "validation/gulls", f), os.path.join(tmp, "validation/gulls"))
    for f in ("referee_round.json", "truth_relabel_impact.json", "cascade_reproduce_result.json", "stress_rescore_local.json"):
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


def _set(tmp, name, fn):
    _edit(tmp, name, fn)


def _seed2_ties_at_2pct(d):
    a = next(a for a in d["models"]["fspl5s_seasons_g08_s2"]["recall_at_matched_fa"] if abs(a["fa_target"] - 0.02) < 1e-9)
    k1 = next(a for a in d["models"]["fspl5s_seasons_g08"]["recall_at_matched_fa"] if abs(a["fa_target"] - 0.02) < 1e-9)["recall_1S2L_k_n"][0]
    a["recall_1S2L_k_n"] = [k1 + 1, a["recall_1S2L_k_n"][1]]


# case -> (perturbation, a fragment of the FATAL message the case must trigger: the guard it is named for)
CASES = {
    "missing artifact": (lambda t: os.remove(os.path.join(t, "validation/gulls/transfer_colour_ablation.json")),
                         "transfer_colour_ablation.json missing"),
    "late missing artifact": (lambda t: os.remove(os.path.join(t, "validation/gulls/cascade_gulls.json")), "cascade_gulls.json missing"),
    "n_scored_test null": (lambda t: _edit(t, "schedule_finetune.json",
                                           lambda d: [e["clean"].update(n_scored_test=None) for e in d["heldout_eval"].values()]),
                           "lacks n_scored_test"),
    "no per-season block": (lambda t: _edit(t, "transfer_tradeoff_all.json", lambda d: d["models"]["sched_sched_seasons"].pop("by_season")),
                            "per-season blocks missing"),
    "no calibrated cascade": (lambda t: _edit(t, "cascade_gulls.json",
                                              lambda d: d["results"].pop("fspl5s_seasons_g08|f146|calibrated_seasons_fullpool")),
                              "lacks fspl5s_seasons_g08|f146|calibrated_seasons_fullpool"),
    "no PeriodicVar": (lambda t: _edit(t, "transfer_subday.json", lambda d: d["models"]["shipped"]["argmax_distribution"].pop("PeriodicVar")),
                       "sub-day argmax lacks PeriodicVar"),
    "no referee round": (lambda t: os.remove(os.path.join(t, "validation/referee_round.json")), "referee_round.json missing"),
    "no colour fine-tunes": (lambda t: _edit(t, "../referee_round.json", lambda d: d["colour_ablation"].pop("finetuned_on_train_colour")),
                             "lacks the colour fine-tunes"),
    "no third seed": (_drop_seed, "fspl5s_seasons_g08_s3 missing"),
    "no refit reference": (lambda t: _edit(t, "../truth_relabel_impact.json", lambda d: d["results"].pop("refit_reference")),
                           "lacks the refit reference"),
    "no seed-3 vs seed-2 pair": (lambda t: _edit(t, "transfer_tradeoff_all.json",
                                                 lambda d: d["paired_differences"]["pairs"].pop("fspl5s_seasons_g08_s3-fspl5s_seasons_g08_s2")),
                                 "fspl5s_seasons_g08_s3-fspl5s_seasons_g08_s2 missing"),
    "no cascade strata": (lambda t: _edit(t, "cascade_gulls.json", lambda d: d["results"]["fspl5s_seasons_g08|f146|frozen"].pop("timing_by_mass_ratio")),
                          "lacks the mass-ratio strata"),
    # values that contradict a sentence (the guards, not only missing keys)
    "floor direction flipped": (lambda t: _edit(t, "../referee_round.json",
                                                 lambda d: d["floor_sensitivity"]["0.01"]["prevalence"].update(population_weighted=0.01)),
                                "floor arms no longer move in the directions"),
    "a variable alerts in the stream": (lambda t: _edit(t, "../referee_round.json",
                                                        lambda d: d["mixed_class_stream"]["by_class"]["PeriodicVar"].update(alert_frac_per_season=0.001)),
                                        "a flat source or variable star alerted"),
    "colour loses recall": (lambda t: _edit(t, "../referee_round.json",
                                            lambda d: d["colour_ablation"]["shipped"]["test_colour"]["all_events"]["recall"].update(PeriodicVar=0.5)),
                            "no longer 'precision, not recall'"),
    "seeds 2-3 lead weighted": (lambda t: _edit(t, "transfer_tradeoff_all.json", lambda d: next(
        a for a in d["models"]["fspl5s_seasons_g08_s2"]["weighted"]["recall_at_matched_fa"] if abs(a["fa_target"] - 0.052) < 1e-9).update(recall_1S2L=0.6)),
                                "seeds 2-3 no longer trail the finite-source run"),
    "seed 2 ties seed 1 at 2%": (lambda t: _edit(t, "transfer_tradeoff_all.json", _seed2_ties_at_2pct),
                                 "the released run is no longer the best seed at every table budget"),
    "physics resolves weighted": (lambda t: _edit(t, "transfer_tradeoff_all.json",
                                                  lambda d: d["models"]["fspl5s_g08"]["weighted"].update(mean_recall_1S2L_fa_le_0p3=0.9)),
                                  "the physics now resolves something weighted"),
    "truncation errors not half late": (lambda t: _edit(t, "../truth_relabel_impact.json",
                                                        lambda d: d["results"]["refit_reference"]["truncation"]["counts"].update({"legacy PSPL / refit NonPSPL": 0})),
                                        "no longer 'about half' late PSPL labels"),
    "RMDC26 Neptune alerts no later than in-house": (lambda t: _edit(t, "cascade_gulls.json", lambda d: d["results"]["fspl5s_seasons_g08|f146|frozen"]
                                                                     ["timing_by_mass_ratio"]["neptune"].update(median_lag_nonpremature_days=4.5)),
                                                     "Neptune alerts come later"),
    "cascade sample not optimistic": (lambda t: _edit(t, "cascade_gulls.json", lambda d: d["results"]["fspl5s_seasons_g08|f146|frozen"]
                                                      ["burden"]["RMDC26_1S1L_ML"].update(full_window_flag_frac=0.07)),
                                      "slightly fewer"),
    "planetary classes scanned unequally": (lambda t: _edit(t, "cascade_gulls.json", lambda d: d["n_scanned"].update(RMDC26_2S2L_ML=900)),
                                            "'the first N of each planetary class'"),
    "detection not lower in a stratum": (lambda t: _edit(t, "cascade_gulls.json", lambda d: d["results"]["fspl5s_seasons_g08|f146|frozen"]
                                                         ["timing_by_mass_ratio"]["neptune"].update(detected_frac=0.83)),
                                         "RMDC26 detection is no longer lower in both strata"),
    "three-band premature not higher": (lambda t: _edit(t, "cascade_gulls.json", lambda d: d["results"]["fspl5s_seasons_g08|threeband|frozen"]
                                                        ["timing_by_mass_ratio"]["giant"].update(premature_frac=0.0)),
                                        "premature rates are no longer the higher ones"),
    # the remaining directional guards added by the fourth and fifth verifications
    "detection does not fall with mass ratio": (lambda t: _edit(t, "cascade_gulls.json", lambda d: d["results"]["fspl5s_seasons_g08|f146|frozen"]
                                                                ["timing_by_mass_ratio"]["lowmass"].update(detected_frac=0.9)),
                                                "falls with mass ratio"),
    "in-house samples low q": (lambda t: _edit(t, "cascade_gulls.json", lambda d: d["inhouse_reference"]["n_by_mass_ratio"].update(lowmass=50)),
                               "'barely samples' q < 1e-4"),
    "premature alerts not rare": (lambda t: _edit(t, "cascade_gulls.json", lambda d: d["results"]["fspl5s_seasons_g08|f146|frozen"]
                                                  ["timing"].update(premature_ci95=[0.01, 0.06])),
                                  "premature alerts 'stay rare'"),
    "three bands raise premature alerts": (lambda t: _edit(t, "cascade_gulls.json", lambda d: d["results"]["fspl5s_seasons_g08|threeband|frozen"]
                                                           ["timing"].update(premature_frac=0.05)),
                                           "three bands 'lower premature alerts"),
    "recalibration raises single-lens alerts": (lambda t: _edit(t, "cascade_gulls.json", lambda d: d["results"]["fspl5s_seasons_g08|f146|calibrated_seasons_fullpool"]
                                                                ["burden"]["RMDC26_1S1L_ML"].update(alert_frac_per_season=0.09)),
                                                "recalibrated single-lens alerts 'fall'"),
    "in-house single lenses alert as often": (lambda t: _edit(t, "../referee_round.json",
                                                              lambda d: d["single_lens_stream_recommended_f146"].update(alert_frac_frozen=0.05)),
                                              "several times as often"),
    "coarse onset advances labels": (lambda t: _edit(t, "../truth_relabel_impact.json", lambda d: d["results"]["refit_reference"]["truncation"]
                                                     ["counts"].update({"legacy NonPSPL / refit PSPL": 80})),
                                     "'usually delays'"),
    "gap losses go to PeriodicVar": (lambda t: _edit(t, "gap_sensitivity.json", lambda d: d["single_gap_by_length_h"]["1.0"]["PSPL"].update(
                                         argmax={"PSPL": 86, "PeriodicVar": 11, "Eruptive": 3})),
                                     "lost single lenses no longer go"),
    "RMDC26 sub-day not shorter": (lambda t: _edit(t, "transfer_subday.json", lambda d: d["te_days"].update(median=0.3)),
                                   "no longer mostly shorter than 0.2 d"),
    "sub-day flagged more than in-support": (lambda t: _edit(t, "transfer_subday.json", lambda d: d["models"]["fspl5s_seasons_g08"]["fa_at"]
                                                             ["frozen"].update(fa=0.2)),
                                             "no longer flagged less often than in-support"),
    "sub-day not mostly PeriodicVar": (lambda t: _edit(t, "transfer_subday.json", lambda d: d["models"]["shipped"]["argmax_distribution"].update(PeriodicVar=0.3)),
                                       "no longer mostly PeriodicVar"),
    "regeneration reproduces every event": (lambda t: _edit(t, "../referee_round.json",
                                                            lambda d: d["threshold_selection_overlap"].update(n_matched_to_pool=15016)),
                                            "reproduce 'most' (not all)"),
    "RMDC26 sub-day like our sweep": (lambda t: _edit(t, "transfer_subday.json", lambda d: [b.update(fa=0.9) for b in d["models"]["shipped"]["fa_frozen_by_te"]]),
                                      "no longer cross the threshold far less often than our sweep's"),
}


def test_complete_inputs_write_everything(tmp_path):
    _tree(str(tmp_path))
    r = _run(str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    for o in OUTS:
        assert "SENTINEL" not in open(os.path.join(tmp_path, o)).read()
    for o in OUTS:                                        # macros AND tables equal the committed ones
        assert open(os.path.join(tmp_path, o)).read() == open(os.path.join(REPO, o)).read(), o


def test_every_generator_input_is_in_the_manifest(tmp_path):
    """--list-inputs names every artifact the generator reads; each must be hashed in paper/results/MANIFEST.json."""
    import hashlib
    _tree(str(tmp_path))
    r = subprocess.run([sys.executable, "paper/make_gulls_macros.py", "--list-inputs"], cwd=str(tmp_path), capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    listed = [x for x in r.stdout.split() if x]
    assert "paper/results/metrics.json" in listed and "validation/cascade_reproduce_result.json" in listed
    files = json.load(open(os.path.join(REPO, "paper/results/MANIFEST.json")))["files"]
    for rel in listed:
        assert rel in files, f"{rel} is read by make_gulls_macros.py but not hashed in the manifest"
        assert hashlib.sha256(open(os.path.join(REPO, rel), "rb").read()).hexdigest() == files[rel]["sha256"], rel
    for o in OUTS:
        assert open(os.path.join(tmp_path, o)).read() == "SENTINEL\n", "--list-inputs must write nothing"


@pytest.mark.parametrize("case", sorted(CASES))
def test_any_missing_input_writes_nothing(tmp_path, case):
    perturb, expected = CASES[case]
    _tree(str(tmp_path)); perturb(str(tmp_path))
    r = _run(str(tmp_path))
    out = r.stdout + r.stderr
    assert r.returncode != 0 and "FATAL" in out, case
    assert expected in out, f"{case}: fired a different guard than the one it is named for:\n{out[-600:]}"
    for o in OUTS:
        assert open(os.path.join(tmp_path, o)).read() == "SENTINEL\n", (case, o)

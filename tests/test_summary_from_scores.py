"""`gulls_summary_tables.py --scores` must reproduce the rows-based tables exactly.

The committed per-event table (validation/gulls/rmdc26_scores.csv.gz, written by build_scores_table.py)
is what lets a clone regenerate transfer_tradeoff_all.json and transfer_colour_ablation.json without the
curve cache or the RMDC26 metadata. This builds a synthetic table with the real builder and checks
that both routes give the same JSON, including the rho/|u0| bins and the per-season breakdown."""
import json
import os
import shutil
import sys

import numpy as np
import pytest

pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "validation", "gulls"))
import build_scores_table as bst          # noqa: E402
import gulls_summary_tables as gst        # noqa: E402


def test_scores_route_matches_rows_route(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    S = [s for s in json.load(open(gst.SCHEDULE))["seasons"] if s["dense"]][:2]
    n = 3000
    ids = np.arange(1, n + 1)
    lab = rng.choice([gst.L1, gst.L2, gst.L3], size=n, p=[0.6, 0.2, 0.2])
    season = rng.integers(0, 2, n)
    t0 = np.array([rng.uniform(S[k]["start_bjd"] + 1, S[k]["end_bjd"] - 1) for k in season])
    rho = 10 ** rng.uniform(-4, 0.5, n); u0 = rng.uniform(-1, 1, n) * 10 ** rng.uniform(-3, 0, n)
    meta = tmp_path / "meta.parquet"
    pq.write_table(pa.table({"event_id": ids, "rho": rho, "u0lens1": u0, "t0lens1": t0,
                             "Planet_q": np.where(lab == gst.L1, np.nan, 1e-3), "Source_Is_Binary": np.zeros(n)}), meta)
    rows = {}
    for name, shift in (("a", 0.0), ("b", 0.3)):
        p = np.clip(rng.beta(2, 5, n) + shift * (lab != gst.L1) * rng.random(n), 0, 1)
        p = np.round(p, 6)
        f = tmp_path / f"rows_{name}.json"
        json.dump([{"event_id": int(e), "sim_label": str(l), "p_nonpspl": float(x), "dense": True, "pred": "NonPSPL",
                    "tE": 10.0, "m_base": 20.0, "weight": 1.0} for e, l, x in zip(ids, lab, p)], open(f, "w"))
        rows[name] = str(f)
    shutil.copy(gst.SCHEDULE, tmp_path / "rmdc26_schedule.json")
    monkeypatch.setattr(bst, "META", str(meta)); monkeypatch.setattr(bst, "CURVES", str(tmp_path))
    monkeypatch.setattr(bst, "HERE", str(tmp_path))
    monkeypatch.setattr(bst, "CHECKPOINTS", {"a": "rows_a.json", "b": "rows_b.json", "b_f146only": "rows_b.json"})
    bst.main()
    monkeypatch.setattr(gst, "META", str(meta))
    (tmp_path / "r").mkdir(); (tmp_path / "s").mkdir()
    gst.main(["--models", f"a={rows['a']}", f"b={rows['b']}", "--f146", f"b={rows['b']}", "--by-season", "--out-dir", str(tmp_path / "r")])
    gst.main(["--scores", str(tmp_path / "rmdc26_scores.csv.gz"), "--models", "a=a", "b=b", "--f146", "b=b_f146only",
              "--by-season", "--out-dir", str(tmp_path / "s")])
    for f in ("transfer_tradeoff_all.json", "transfer_colour_ablation.json"):
        x = json.load(open(tmp_path / "r" / f)); y = json.load(open(tmp_path / "s" / f))
        x.pop("command", None); y.pop("command", None)
        assert x == y, f
    t = json.load(open(tmp_path / "s" / "transfer_tradeoff_all.json"))
    assert set(t["models"]["a"]["by_season"]) == {str(S[0]["index"]), str(S[1]["index"])}

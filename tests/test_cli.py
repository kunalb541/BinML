"""Focused tests for the public ``binml classify`` file interface."""

from types import SimpleNamespace

import numpy as np
import pytest

import binml
from binml.cli import _load_light_curve, main


def test_load_light_curve_preserves_headerless_csv(tmp_path):
    path = tmp_path / "event.csv"
    path.write_text("1.0,22.1,0.03\n2.0,21.8,0.04\n", encoding="utf-8")

    time, mag = _load_light_curve(str(path))

    assert np.array_equal(time, [1.0, 2.0])
    assert np.array_equal(mag, [22.1, 21.8])


def test_cli_accepts_named_csv_columns_in_any_order(tmp_path, monkeypatch, capsys):
    path = tmp_path / "event.CSV"
    path.write_text(
        "mag_err, Magnitude (AB), Time [days]\n"
        "0.03,22.1,1.0\n"
        "0.04,21.8,2.0\n",
        encoding="utf-8",
    )
    seen = {}

    class FakeClassifier:
        def predict(self, time, mag, **kwargs):
            seen["time"] = time
            seen["mag"] = mag
            seen["kwargs"] = kwargs
            return SimpleNamespace(probabilities={"Flat": 1.0})

    monkeypatch.setattr(binml, "Classifier", FakeClassifier)

    assert main(["classify", str(path), "--m-base", "22.0", "--t-start", "1.0"]) == 0
    assert np.array_equal(seen["time"], [1.0, 2.0])
    assert np.array_equal(seen["mag"], [22.1, 21.8])
    assert seen["kwargs"] == {"m_base_ref": 22.0, "t_start": 1.0}
    assert "Flat" in capsys.readouterr().out


def test_headered_csv_requires_recognised_time_and_magnitude(tmp_path):
    path = tmp_path / "event.csv"
    path.write_text("flux,error\n10.0,0.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="recognised time column"):
        _load_light_curve(str(path))


@pytest.mark.parametrize("option", ["--m-base", "--t-start"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_cli_rejects_nonfinite_window_metadata(tmp_path, option, value):
    path = tmp_path / "event.csv"
    path.write_text("0.0,22.0\n1.0,21.9\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(["classify", str(path), option, value])

    assert exc.value.code == 2

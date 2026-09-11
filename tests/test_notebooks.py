"""The example notebooks execute. 00 is offline and always run (when nbclient is installed);
01 needs network access to the RMDC26 tables and runs only with BINML_NETWORK_TESTS=1."""
import os

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("matplotlib")

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(os.path.dirname(HERE), "examples")


def _run(name, timeout=600):
    nb = nbformat.read(os.path.join(EXAMPLES, name), as_version=4)
    nbclient.NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": EXAMPLES}}).execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, errors[0]


def test_quickstart_notebook_runs_offline():
    _run("00_quickstart_synthetic.ipynb")


@pytest.mark.skipif(not os.environ.get("BINML_NETWORK_TESTS"), reason="needs network access to RMDC26")
def test_roman_event_notebook_runs():
    pytest.importorskip("duckdb")
    _run("01_classify_roman_event.ipynb", timeout=900)

#!/usr/bin/env python3
"""Fail closed if a frozen evaluation array differs from its manifest.

The paper figures read ``results/*.npy`` directly.  A valid NumPy file can be silently replaced
while retaining the same name, shape and dtype, so merely loading it is not an integrity check.
The manifest hashes the canonical C-order array payload (not the .npy container header), which is
stable across NumPy container versions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPO = HERE.parent


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_trace_reduction(trace: Path, result: Path, reducer: Path,
                              trace_option: str) -> list[str]:
    """Tie a result to its inputs and reproduce its complete payload deterministically."""
    label = result.name
    for path in (trace, result, reducer):
        if not path.exists():
            return [f"{label}: missing required file {path.relative_to(REPO)}"]
    try:
        data = json.loads(result.read_text())
    except Exception as exc:
        return [f"{label}: cannot parse ({exc})"]
    provenance = data.get("reduction_provenance", {})
    failures = []
    expected_trace = provenance.get("input_trace_sha256")
    expected_reducer = provenance.get("reducer_sha256")
    actual_trace = _file_sha256(trace)
    actual_reducer = _file_sha256(reducer)
    if expected_trace != actual_trace:
        failures.append(f"{label}: input trace sha256 {expected_trace} != {actual_trace}")
    if expected_reducer != actual_reducer:
        failures.append(f"{label}: reducer sha256 {expected_reducer} != {actual_reducer}")
    if failures:
        return failures

    # Hash fields alone authenticate inputs, not the result payload: a hand-edited rate could keep
    # the old hashes and previously passed this validator. Re-run the cheap deterministic reducer
    # and require the archived JSON bytes to be exactly what the code emits.
    with tempfile.TemporaryDirectory(prefix="binml-reduce-") as tmpdir:
        regenerated = Path(tmpdir) / result.name
        run = subprocess.run(
            [sys.executable, str(reducer), trace_option, str(trace), "--out", str(regenerated)],
            cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            timeout=60,
        )
        if run.returncode != 0:
            failures.append(f"{label}: deterministic reducer failed: {run.stderr.strip()}")
        elif regenerated.read_bytes() != result.read_bytes():
            failures.append(
                f"{label}: payload differs from deterministic reducer output; regenerate it "
                f"instead of editing JSON by hand")
    return failures


def _validate_hashed_files(manifest: dict) -> list[str]:
    """Validate every load-bearing JSON/checkpoint and source artifact named in the manifest."""
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        return ["results/MANIFEST.json has no non-empty 'files' mapping"]
    failures = []
    for name, spec in files.items():
        path = REPO / name
        if not path.exists():
            failures.append(f"{name}: missing")
            continue
        expected = spec.get("sha256") if isinstance(spec, dict) else None
        actual = _file_sha256(path)
        if expected != actual:
            failures.append(f"{name}: file sha256 {actual} != {expected}")
    return failures


def _validate_checkpoint_identity() -> list[str]:
    """Require paper metadata and the primary trace to name the checkpoint actually shipped."""
    checkpoint = REPO / "binml" / "weights" / "binml.pt"
    if not checkpoint.exists():
        return ["binml/weights/binml.pt: missing"]
    actual = _file_sha256(checkpoint)
    failures = []
    for name in ("metrics.json", "meta.json"):
        data = json.loads((RESULTS / name).read_text())
        expected = str(data.get("checkpoint_sha256", ""))
        if not expected or not actual.startswith(expected):
            failures.append(f"results/{name}: checkpoint sha256 {expected!r} does not match {actual}")
    trace = np.load(REPO / "validation" / "cascade_trace.npz", allow_pickle=False)
    provenance = json.loads(str(trace["provenance"]))
    expected = provenance.get("checkpoint_sha256")
    if expected != actual:
        failures.append(f"validation/cascade_trace.npz: checkpoint sha256 {expected} != {actual}")
    return failures


def validate() -> int:
    manifest_path = RESULTS / "MANIFEST.json"
    if not manifest_path.exists():
        raise SystemExit(f"FATAL: missing evaluation manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict) or not arrays:
        raise SystemExit("FATAL: results/MANIFEST.json has no non-empty 'arrays' mapping")

    failures = []
    for name, spec in arrays.items():
        path = RESULTS / name
        if not path.exists():
            failures.append(f"{name}: missing")
            continue
        try:
            arr = np.load(path, allow_pickle=False)
        except Exception as exc:  # corrupt or unsafe artifact
            failures.append(f"{name}: cannot load ({exc})")
            continue
        expected_shape = tuple(spec.get("shape", ()))
        expected_dtype = str(spec.get("dtype"))
        expected_hash = str(spec.get("sha256_16"))
        actual_hash = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
        if arr.shape != expected_shape:
            failures.append(f"{name}: shape {arr.shape} != {expected_shape}")
        if str(arr.dtype) != expected_dtype:
            failures.append(f"{name}: dtype {arr.dtype} != {expected_dtype}")
        if actual_hash != expected_hash:
            failures.append(f"{name}: payload sha256_16 {actual_hash} != {expected_hash}")

    failures.extend(_validate_hashed_files(manifest))
    failures.extend(_validate_checkpoint_identity())

    validation = REPO / "validation"
    failures.extend(_validate_trace_reduction(
        validation / "cascade_trace.npz",
        validation / "cascade_reproduce_result.json",
        validation / "cascade_reduce.py", "--trace"))
    failures.extend(_validate_trace_reduction(
        validation / "matched_traces.npz",
        validation / "cascade_matched_result.json",
        validation / "cascade_matched_reduce.py", "--traces"))

    if failures:
        raise SystemExit("FATAL: frozen evaluation artifact failed validation:\n  "
                         + "\n  ".join(failures))
    print(f"validated {len(arrays)} frozen arrays, {len(manifest['files'])} hashed files, "
          "the shipped checkpoint, and 2 deterministic trace reductions")
    return 0


if __name__ == "__main__":
    raise SystemExit(validate())

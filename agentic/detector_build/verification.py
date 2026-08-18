"""Driver-owned verification for generated detector programs."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


SMOKE_SUCCESS_MARKER = "DETECTOR_SMOKE_OK"
DEFAULT_DETECTOR_CHECK_TIMEOUT_S = 30_000


def validate_detector_source(detector_path: Path) -> None:
    """Reject syntactically invalid generated source without importing it."""
    try:
        source = detector_path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(detector_path))
    except (OSError, SyntaxError) as exc:
        raise SystemExit(f"Generated detector {detector_path} does not parse: {exc}") from exc


def _run_check(
    detector_path: Path,
    args: list[str],
    label: str,
    timeout_s: int,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    try:
        result = subprocess.run(
            [sys.executable, str(detector_path), *args],
            cwd=str(detector_path.parent),
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(
            f"Generated detector {label} check could not complete: {exc}"
        ) from exc
    if result.returncode != 0:
        diagnostic = (result.stderr or result.stdout).strip()
        if len(diagnostic) > 2000:
            diagnostic = diagnostic[-2000:]
        raise SystemExit(
            f"Generated detector failed its driver-owned {label} check "
            f"(exit {result.returncode}):\n{diagnostic}"
        )
    return result


def validate_detector_executable_contract(
    detector_path: Path,
    model_config_path: Path,
    *,
    timeout_s: int = DEFAULT_DETECTOR_CHECK_TIMEOUT_S,
) -> None:
    """Run no-data checks that the agent cannot satisfy by assertion alone.

    The generated CLI must expose a synthetic mode that exercises model
    selection without loading a real pkl and prints a fixed success marker only
    after all invariants pass.
    """
    before = {
        path.relative_to(detector_path.parent)
        for path in detector_path.parent.rglob("*")
    }
    help_result = _run_check(detector_path, ["--help"], "--help", timeout_s)
    help_text = f"{help_result.stdout}\n{help_result.stderr}"
    if "--measuretime" not in help_text:
        raise SystemExit(
            "Generated detector --help is missing the required --measuretime mode."
        )
    if "--measuretime-occurrences" not in help_text:
        raise SystemExit(
            "Generated detector --help is missing the required sampled "
            "measuretime occurrence limit."
        )
    if "--hypothesis-selection" not in help_text:
        raise SystemExit(
            "Generated detector --help is missing the required hypothesis-selection mode."
        )
    if "--exclude-hypotheses" not in help_text:
        raise SystemExit(
            "Generated detector --help is missing direct manual hypothesis exclusion."
        )
    result = _run_check(
        detector_path,
        [
            "--synthetic-smoke-test",
            "--model-config",
            str(model_config_path.resolve()),
        ],
        "synthetic smoke",
        timeout_s,
    )
    combined = f"{result.stdout}\n{result.stderr}"
    if SMOKE_SUCCESS_MARKER not in combined:
        raise SystemExit(
            "Generated detector synthetic smoke test exited successfully but did "
            f"not print the required {SMOKE_SUCCESS_MARKER!r} marker."
        )

    # Exercise runtime configuration rejection rather than trusting generated
    # source to say it validates the allowlist. Any failure is acceptable; a
    # successful detector run with an unreviewed family is not.
    try:
        config = json.loads(model_config_path.read_text(encoding="utf-8"))
        candidates = config["candidates"]
        if not isinstance(candidates, list) or not candidates:
            raise KeyError("candidates")
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"Cannot construct detector contract check from {model_config_path}: "
            f"{exc}"
        ) from exc
    invalid_config = json.loads(json.dumps(config))
    invalid_config["candidates"][0]["name"] = "__unreviewed_estimator__"
    with tempfile.TemporaryDirectory(prefix="detector-contract-") as tmpdir:
        invalid_path = Path(tmpdir) / "invalid_model_candidates.json"
        invalid_path.write_text(json.dumps(invalid_config), encoding="utf-8")
        try:
            rejected = subprocess.run(
                [
                    sys.executable,
                    str(detector_path),
                    "--synthetic-smoke-test",
                    "--model-config",
                    str(invalid_path),
                ],
                cwd=str(detector_path.parent),
                env={**os.environ, "MPLBACKEND": "Agg"},
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SystemExit(
                f"Generated detector invalid-config check could not complete: {exc}"
            ) from exc
    if rejected.returncode == 0:
        raise SystemExit(
            "Generated detector accepted an unreviewed estimator family during "
            "its synthetic smoke mode."
        )

    after = {
        path.relative_to(detector_path.parent)
        for path in detector_path.parent.rglob("*")
    }
    created = sorted(str(path) for path in after - before)
    if created:
        raise SystemExit(
            "Generated detector no-data contract checks created output paths: "
            + ", ".join(created)
        )

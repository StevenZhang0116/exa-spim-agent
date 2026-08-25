"""Source-bound node-role requirements for split-detector features.

An agent interprets feature semantics, but the driver owns source identity and
provenance.  The agent therefore writes a small draft keyed only by hypothesis
id and source kind; :func:`compile_applicability` binds those judgments to the
exact source paths and SHA-256 values that a detector build can later verify.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ARTIFACT_SUFFIX = ".split-feature-applicability.json"
DRAFT_SUFFIX = ".split-feature-applicability.draft.json"
NODE_ROLE_REQUIREMENTS = frozenset({
    "requires_both_tips",
    "requires_tip_anchor",
    "no_tip_requirement",
    "unclear",
})


def artifact_path(run_json: Path) -> Path:
    return run_json.with_name(f"{run_json.stem}{ARTIFACT_SUFFIX}")


def draft_path(run_json: Path) -> Path:
    return run_json.with_name(f"{run_json.stem}{DRAFT_SUFFIX}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path, project_root: Path) -> str:
    rel = Path(os.path.relpath(path, project_root))
    return path.as_posix() if rel.parts and rel.parts[0] == ".." else rel.as_posix()


def expected_sources(
    selected_ids: list[int],
    rerun_dir: Path,
    fixed_dir: Path | None,
    project_root: Path,
) -> list[dict]:
    """Return the deterministic source sequence the semantic draft must cover."""
    records: list[dict] = []
    for hypothesis_id in selected_ids:
        rerun_path = rerun_dir / f"hypo_{hypothesis_id}.py"
        if not rerun_path.is_file():
            raise SystemExit(
                f"Cannot label split feature {hypothesis_id}: missing "
                f"{_rel(rerun_path, project_root)}."
            )
        variants = [("rerun", rerun_path)]
        fixed_path = (
            fixed_dir / f"hypo_{hypothesis_id}.py" if fixed_dir is not None else None
        )
        if fixed_path is not None and fixed_path.is_file():
            variants.append(("fixed", fixed_path))
        for source, path in variants:
            records.append({
                "id": hypothesis_id,
                "source": source,
                "source_path": _rel(path, project_root),
                "source_sha256": sha256(path),
            })
    return records


def _load_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label.capitalize()} {path} must be a JSON object.")
    return value


def validate_draft(
    path: Path,
    *,
    selected_ids: list[int],
    rerun_dir: Path,
    fixed_dir: Path | None,
    project_root: Path,
) -> dict:
    draft = _load_object(path, "split feature applicability draft")
    if set(draft) != {"schema_version", "records"}:
        raise SystemExit(
            "Split feature applicability draft must contain exactly "
            "schema_version and records."
        )
    if draft.get("schema_version") != 1:
        raise SystemExit("Split feature applicability draft schema_version must be 1.")
    rows = draft.get("records")
    if not isinstance(rows, list):
        raise SystemExit("Split feature applicability draft records must be a list.")
    expected = expected_sources(selected_ids, rerun_dir, fixed_dir, project_root)
    expected_keys = [(row["id"], row["source"]) for row in expected]
    actual_keys: list[tuple[object, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
            "id", "source", "node_role_requirement", "reason"
        }:
            raise SystemExit(
                f"Split feature applicability draft record {index} must contain "
                "exactly id, source, node_role_requirement, and reason."
            )
        actual_keys.append((row["id"], row["source"]))
        if row["node_role_requirement"] not in NODE_ROLE_REQUIREMENTS:
            raise SystemExit(
                f"Split feature applicability record {index} has invalid "
                f"node_role_requirement {row['node_role_requirement']!r}."
            )
        if not isinstance(row["reason"], str) or not row["reason"].strip():
            raise SystemExit(
                f"Split feature applicability record {index} needs a reason."
            )
    if actual_keys != expected_keys:
        raise SystemExit(
            "Split feature applicability draft must cover every current source "
            f"exactly once in order; expected {expected_keys}, got {actual_keys}."
        )
    return draft


def compile_applicability(
    draft: Path,
    output: Path,
    *,
    run_json: Path,
    selected_ids: list[int],
    rerun_dir: Path,
    fixed_dir: Path | None,
    project_root: Path,
) -> None:
    semantic = validate_draft(
        draft,
        selected_ids=selected_ids,
        rerun_dir=rerun_dir,
        fixed_dir=fixed_dir,
        project_root=project_root,
    )
    sources = expected_sources(selected_ids, rerun_dir, fixed_dir, project_root)
    records = []
    for source, judgment in zip(sources, semantic["records"]):
        records.append({
            **source,
            "node_role_requirement": judgment["node_role_requirement"],
            "reason": judgment["reason"].strip(),
        })
    payload = {
        "schema_version": 1,
        "source_run": _rel(run_json, project_root),
        "source_run_sha256": sha256(run_json),
        "selected_ids": selected_ids,
        "records": records,
    }
    temporary = output.with_name(f".{output.name}.compile.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except OSError as exc:
        raise SystemExit(f"Cannot write split feature applicability {output}: {exc}")


def load_applicability(
    path: Path,
    *,
    run_json: Path,
    selected_ids: list[int],
    rerun_dir: Path,
    fixed_dir: Path | None,
    project_root: Path,
) -> dict[tuple[int, str], dict]:
    """Validate a public artifact and index records by ``(id, source_sha256)``."""
    payload = _load_object(path, "split feature applicability")
    if set(payload) != {
        "schema_version", "source_run", "source_run_sha256", "selected_ids", "records"
    }:
        raise SystemExit(
            "Split feature applicability must contain exactly schema_version, "
            "source_run, source_run_sha256, selected_ids, and records."
        )
    if payload.get("schema_version") != 1:
        raise SystemExit("Split feature applicability schema_version must be 1.")
    if payload.get("source_run") != _rel(run_json, project_root):
        raise SystemExit("Split feature applicability points to the wrong run JSON.")
    if payload.get("source_run_sha256") != sha256(run_json):
        raise SystemExit("Split feature applicability is stale for the run JSON.")
    if payload.get("selected_ids") != selected_ids:
        raise SystemExit(
            "Split feature applicability selected_ids do not match detector inputs."
        )
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise SystemExit("Split feature applicability records must be a list.")
    expected = expected_sources(selected_ids, rerun_dir, fixed_dir, project_root)
    if len(rows) != len(expected):
        raise SystemExit(
            "Split feature applicability does not cover every current source."
        )
    index: dict[tuple[int, str], dict] = {}
    for number, (row, source) in enumerate(zip(rows, expected)):
        required_fields = {
            "id", "source", "source_path", "source_sha256",
            "node_role_requirement", "reason",
        }
        if not isinstance(row, dict) or set(row) != required_fields:
            raise SystemExit(
                f"Split feature applicability record {number} has an invalid schema."
            )
        for field in ("id", "source", "source_path", "source_sha256"):
            if row[field] != source[field]:
                raise SystemExit(
                    f"Split feature applicability record {number} is stale: "
                    f"{field} must be {source[field]!r}."
                )
        requirement = row["node_role_requirement"]
        if requirement not in NODE_ROLE_REQUIREMENTS:
            raise SystemExit(
                f"Split feature applicability record {number} has invalid "
                f"node_role_requirement {requirement!r}."
            )
        if not isinstance(row["reason"], str) or not row["reason"].strip():
            raise SystemExit(
                f"Split feature applicability record {number} needs a reason."
            )
        key = (int(row["id"]), str(row["source_sha256"]))
        if key in index:
            raise SystemExit("Split feature applicability contains a duplicate source.")
        index[key] = row
    return index


"""The frozen candidate rule of a detector build, taken from the benchmark declaration.

Since 2026-10-16 the candidate rule of each kind is declared once in
``proofreader_evolve/candidate_pools.json`` and applied by ``candidate_rules.py`` at the
repository root. A detector build no longer selects a rule (the split advice stage and the
merge sweep floor are gone): the driver freezes the declared rule as
``split_candidate_policy.json`` / ``merge_candidate_policy.json`` together with the SHA-256
of the declaration and of ``candidate_rules.py``. Assembly copies ``candidate_rules.py``
unchanged beside the detector, checks it against that hash and embeds the rule, so the
detector enumerates and labels with the benchmark's own code. Sweeps remain evidence for the
declaration; they are not read here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

from .contracts import (
    CANDIDATE_POLICY_NAME,
    CANDIDATE_RULES_NAME,
    MERGE_CANDIDATE_POLICY_NAME,
    DetectorTarget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DECLARATION_PATH = PROJECT_ROOT / "proofreader_evolve" / "candidate_pools.json"
DEFAULT_RULES_PATH = PROJECT_ROOT / "candidate_rules.py"
DECLARATION_SCHEMA_VERSION = 2
CANDIDATE_POLICY_SCHEMA_VERSION = 2
ARTIFACT_TYPE = "declared_candidate_rule"
APPLICATION_STATUS = "selected_for_detector_runtime"
KIND_OF_TARGET = {DetectorTarget.SPLIT: "split", DetectorTarget.MERGE_SITE: "merge"}
POLICY_NAME_OF_KIND = {"split": CANDIDATE_POLICY_NAME, "merge": MERGE_CANDIDATE_POLICY_NAME}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} {path} must contain a JSON object.")
    return value


def _relative(path: Path, project_root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        return str(Path(path).resolve())


def expected_mcl_from_run_path(run_json: Path) -> int | None:
    match = re.search(r"(?:^|[-_])mcl(\d+)(?:[-_.]|$)", Path(run_json).name.lower())
    return int(match.group(1)) if match else None


def kind_of_target(target: DetectorTarget) -> str:
    try:
        return KIND_OF_TARGET[target]
    except KeyError as exc:
        raise SystemExit(f"{target} detectors take no candidate rule.") from exc


def load_rules_module(path: Path = DEFAULT_RULES_PATH):
    """``(module, sha256)`` of a ``candidate_rules.py``, imported under a content-unique name."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"Missing candidate rule module {path}.")
    digest = _sha256(path)
    name = f"_candidate_rules_{digest[:16]}"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module, digest


def declared_rule(kind: str, declaration_path: Path = DEFAULT_DECLARATION_PATH,
                  rules_path: Path = DEFAULT_RULES_PATH) -> dict:
    """The declared rule of ``kind`` with everything a frozen policy records about it."""
    declaration = _load_json(declaration_path, "candidate-pool declaration")
    if declaration.get("schema_version") != DECLARATION_SCHEMA_VERSION:
        raise SystemExit(f"{declaration_path} must have schema_version {DECLARATION_SCHEMA_VERSION}.")
    pools = declaration.get("pools")
    entry = pools.get(kind) if isinstance(pools, dict) else None
    if not isinstance(entry, dict):
        raise SystemExit(f"{declaration_path} declares no {kind} rule.")
    rules, rules_sha256 = load_rules_module(rules_path)
    fields = rules.FIELDS[kind]
    missing = [field for field in fields if field not in entry]
    if missing:
        raise SystemExit(f"The {kind} entry of {declaration_path} lacks {missing}.")
    try:
        rule = rules.validate_rule(kind, {field: entry[field] for field in fields})
    except ValueError as exc:
        raise SystemExit(f"The {kind} entry of {declaration_path} is invalid: {exc}") from exc
    return {"rule": rule, "entry": entry, "declaration": declaration,
            "declaration_sha256": _sha256(declaration_path), "rules": rules,
            "rules_sha256": rules_sha256}


def _artifact(kind: str, project_root: Path, declaration_path: Path, rules_path: Path) -> dict:
    found = declared_rule(kind, declaration_path, rules_path)
    rules, rule = found["rules"], found["rule"]
    return {
        "schema_version": CANDIDATE_POLICY_SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "application_status": APPLICATION_STATUS,
        "kind": kind,
        "selected_policy": rule,
        "rule_sha256": rules.rule_digest(kind, rule),
        "rule_text": rules.describe_rule(kind, rule),
        "declaration": {"path": _relative(declaration_path, project_root),
                        "sha256": found["declaration_sha256"],
                        "revised": found["declaration"].get("revised")},
        "candidate_rules": {"path": _relative(rules_path, project_root), "file": CANDIDATE_RULES_NAME,
                            "sha256": found["rules_sha256"], "rules_version": rules.RULES_VERSION},
        "evidence": found["entry"].get("evidence"),
    }


def compile_declared_candidate_policy(kind: str, output_path: Path, *, project_root: Path = PROJECT_ROOT,
                                      declaration_path: Path = DEFAULT_DECLARATION_PATH,
                                      rules_path: Path = DEFAULT_RULES_PATH) -> dict:
    """Freeze the declared ``kind`` rule as the build's public candidate-policy artifact."""
    artifact = _artifact(kind, project_root, declaration_path, rules_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return artifact


def validate_frozen_declared_candidate_policy(path: Path, kind: str, *, project_root: Path = PROJECT_ROOT,
                                              declaration_path: Path = DEFAULT_DECLARATION_PATH,
                                              rules_path: Path = DEFAULT_RULES_PATH) -> dict:
    """The frozen artifact, if it still equals what the declaration and module give now."""
    stored = _load_json(path, "frozen candidate policy")
    if stored != _artifact(kind, project_root, declaration_path, rules_path):
        raise SystemExit(f"Frozen candidate policy {path} is stale: the declaration or "
                         f"{CANDIDATE_RULES_NAME} changed since it was frozen.")
    return stored


def load_runtime_policy(path: Path, kind: str) -> tuple[dict, str, str]:
    """``(rule, policy SHA-256, candidate_rules SHA-256)`` a runtime embeds, strictly checked."""
    policy = _load_json(path, "frozen candidate policy")
    if (policy.get("schema_version") != CANDIDATE_POLICY_SCHEMA_VERSION
            or policy.get("artifact_type") != ARTIFACT_TYPE):
        raise SystemExit(f"Candidate policy {path} is not a declared-rule artifact "
                         f"(schema_version {CANDIDATE_POLICY_SCHEMA_VERSION}, {ARTIFACT_TYPE}).")
    if policy.get("application_status") != APPLICATION_STATUS:
        raise SystemExit(f"Candidate policy {path} is not selected for detector runtime.")
    if policy.get("kind") != kind:
        raise SystemExit(f"Candidate policy {path} is a {policy.get('kind')!r} rule, not {kind!r}.")
    rules_sha256 = (policy.get("candidate_rules") or {}).get("sha256")
    if not isinstance(rules_sha256, str) or not _SHA256.match(rules_sha256):
        raise SystemExit(f"Candidate policy {path} lacks the SHA-256 of {CANDIDATE_RULES_NAME}.")
    rules, _ = load_rules_module(DEFAULT_RULES_PATH)
    selected = policy.get("selected_policy")
    try:
        normal = rules.validate_rule(kind, selected)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Candidate policy {path} has an invalid rule: {exc}") from exc
    if normal != selected:
        raise SystemExit(f"Candidate policy {path} stores a rule that is not normalised.")
    return normal, _sha256(path), rules_sha256


def load_runtime_candidate_policy(path: Path) -> tuple[dict, str]:
    """``(split rule, policy SHA-256)``."""
    rule, sha256, _ = load_runtime_policy(path, "split")
    return rule, sha256


def load_runtime_merge_candidate_policy(path: Path) -> tuple[dict, str]:
    """``(merge rule, policy SHA-256)``."""
    rule, sha256, _ = load_runtime_policy(path, "merge")
    return rule, sha256


def vendor_candidate_rules(policy_path: Path, destination_dir: Path, *,
                           rules_path: Path = DEFAULT_RULES_PATH) -> Path:
    """Copy ``candidate_rules.py`` beside the detector, byte-identical to the frozen hash."""
    policy = _load_json(policy_path, "frozen candidate policy")
    expected = (policy.get("candidate_rules") or {}).get("sha256")
    if _sha256(rules_path) != expected:
        raise SystemExit(f"{rules_path} changed since {policy_path} was frozen; rebuild the policy.")
    target = Path(destination_dir) / CANDIDATE_RULES_NAME
    if target.is_file() and _sha256(target) == expected:
        return target
    temporary = target.with_name(f".{target.name}.tmp")
    shutil.copyfile(rules_path, temporary)
    if _sha256(temporary) != expected:
        temporary.unlink()
        raise SystemExit(f"Copy of {rules_path} does not match its frozen SHA-256.")
    temporary.replace(target)
    return target

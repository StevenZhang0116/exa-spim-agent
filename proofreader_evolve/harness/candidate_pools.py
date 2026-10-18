"""Benchmark candidate pools: the enumeration rule of each kind, declared once.

``proofreader_evolve/candidate_pools.json`` declares the rule of each kind. The code that
applies a rule is ``candidate_rules.py`` at the repository root, shared with the
candidate-pool sweeps and copied unchanged into every detector built since 2026-10-16.
Tables and companions enumerate with that module, never with a detector: a detector only
computes features and scores for the declared rows. The rule and the module's SHA-256 are
part of the native-table provenance, so a changed rule or a changed module never matches old
tables. ``pool_sha256`` hashes the rows themselves and changes only when the rows do, so
companions and context caches survive a module edit that keeps the rows. Both kinds must be
declared explicitly (schema version 2, 2026-10-16); the earlier ``null`` ("use the detector's
embedded rule") is gone.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
RULES_PATH = REPO / "candidate_rules.py"
POLICY_PATH = Path(__file__).resolve().parents[1] / "candidate_pools.json"
SCHEMA_VERSION = 2
KINDS = ("merge", "split")
_MODULE_NAME = "_repo_candidate_rules"


def rules():
    """The repository's ``candidate_rules`` module, loaded by file path (no sys.path assumptions)."""
    module = sys.modules.get(_MODULE_NAME)
    if module is None:
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, RULES_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        spec.loader.exec_module(module)
    return module


def rules_sha256() -> str:
    """SHA-256 of ``candidate_rules.py`` as it is on disk."""
    return hashlib.sha256(RULES_PATH.read_bytes()).hexdigest()


def rule_fields(kind, policy) -> dict:
    """The rule fields of one declared entry (evidence and notes dropped)."""
    if kind not in KINDS:
        raise ValueError(f"Unknown candidate kind {kind!r}")
    if not isinstance(policy, dict):
        raise ValueError(f"The {kind} pool entry must be an object with the rule fields")
    fields = rules().FIELDS[kind]
    missing = [field for field in fields if field not in policy]
    if missing:
        raise ValueError(f"The {kind} pool entry lacks {missing}")
    return {field: policy[field] for field in fields}


def validate_policy(kind, policy) -> dict:
    """Normalised rule of one declared entry; raises on anything enumeration cannot apply."""
    return rules().validate_rule(kind, rule_fields(kind, policy))


def load_pool_policies(path=POLICY_PATH) -> dict:
    """``{kind: normalised rule}`` from the declaration file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION or not isinstance(data.get("pools"), dict):
        raise ValueError(f"{path}: expected schema_version {SCHEMA_VERSION} with a pools object")
    pools = data["pools"]
    if set(pools) != set(KINDS):
        raise ValueError(f"{path}: pools must declare exactly {KINDS}")
    for kind in KINDS:
        if pools[kind] is None:
            raise ValueError(f"{path}: the {kind} rule must be declared; a detector no longer supplies it")
    return {kind: validate_policy(kind, pools[kind]) for kind in KINDS}


def policy_digest(kind, policy) -> str:
    """Digest of the rule fields (what enumeration applies), independent of evidence notes."""
    return rules().rule_digest(kind, rule_fields(kind, policy))


def describe(kind, policy) -> str:
    """Plain-text statement of a declared rule and its labels (companion manifests)."""
    return rules().describe_rule(kind, rule_fields(kind, policy))


def build_universe(kind, payload, policy):
    """``(rows, labels)`` of ``kind`` under the declared ``policy``."""
    return rules().build_universe(kind, payload, validate_policy(kind, policy))

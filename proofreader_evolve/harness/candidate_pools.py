"""Benchmark candidate pools: the enumeration rule of each kind, declared once.

The frozen detectors embed the candidate rule they were built with as constants of
their runtime module (``CANDIDATE_PAIRING_RULE``, ``CANDIDATE_MAX_DISTANCE_UM``, ...).
``proofreader_evolve/candidate_pools.json`` declares the rule the benchmark uses per
kind: ``null`` keeps the detector's own rule; an explicit entry is applied to the
runtime module while its universe builder runs (``apply_pool_policy``). The applied
rule is part of the native-table provenance, so a changed rule is a new pool hash
and never matches old tables. Only split rules can be overridden today: the merge
runtime takes a different parameter set (junction NMS and claim radius).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
import re

POLICY_PATH = Path(__file__).resolve().parents[1] / "candidate_pools.json"
KINDS = ("merge", "split")
PAIRING_RULES = ("tip_to_tip", "tip_to_any_node", "any_node_to_any_node")
FIELDS = ("config_id", "mode", "radius_um", "per_anchor_k", "global_cap")
RUNTIME_CONSTANTS = {
    "mode": "CANDIDATE_PAIRING_RULE",
    "radius_um": "CANDIDATE_MAX_DISTANCE_UM",
    "per_anchor_k": "CANDIDATE_PER_ANCHOR_K",
    "global_cap": "CANDIDATE_GLOBAL_CAP",
    "config_id": "CANDIDATE_POLICY_CONFIG_ID",
}
_CONFIG_ID = re.compile(r"^(?P<mode>[a-z_]+)\|r=(?P<radius>[0-9.]+)\|k=(?P<k>all|\d+)$")


def config_id(mode, radius_um, per_anchor_k) -> str:
    quota = "all" if per_anchor_k is None else int(per_anchor_k)
    return f"{mode}|r={float(radius_um):g}|k={quota}"


def parse_config_id(text) -> dict:
    """``mode``, ``radius_um`` and ``per_anchor_k`` of a ``mode|r=..|k=..`` id."""
    match = _CONFIG_ID.match(str(text).strip())
    if match is None:
        raise ValueError(f"Not a candidate-policy config id: {text!r}")
    quota = None if match.group("k") == "all" else int(match.group("k"))
    return {"mode": match.group("mode"), "radius_um": float(match.group("radius")), "per_anchor_k": quota}


def validate_policy(kind, policy) -> dict:
    """Normalise one explicit pool entry; raise on anything the runtimes cannot apply."""
    if kind != "split":
        raise ValueError(f"Pool overrides are supported for split only; {kind} must be null")
    if not isinstance(policy, dict) or any(field not in policy for field in FIELDS):
        raise ValueError(f"Pool policy for {kind} needs the fields {FIELDS}")
    mode = str(policy["mode"])
    if mode not in PAIRING_RULES:
        raise ValueError(f"Unknown pairing rule {mode!r}; choose from {PAIRING_RULES}")
    radius = float(policy["radius_um"])
    if not radius > 0:
        raise ValueError("radius_um must be positive")
    quota = policy["per_anchor_k"]
    if quota is not None and (int(quota) != quota or int(quota) < 1):
        raise ValueError("per_anchor_k must be null or a positive integer")
    if policy["global_cap"] is not None:
        raise ValueError("global_cap must be null (the runtimes enumerate without a global cap)")
    expected = config_id(mode, radius, quota)
    if str(policy["config_id"]) != expected:
        raise ValueError(f"config_id {policy['config_id']!r} does not match its fields ({expected})")
    return {"config_id": expected, "mode": mode, "radius_um": radius,
            "per_anchor_k": None if quota is None else int(quota), "global_cap": None}


def load_pool_policies(path=POLICY_PATH) -> dict:
    """``{kind: normalised policy or None}`` from the declaration file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("pools"), dict):
        raise ValueError(f"{path}: expected schema_version 1 with a pools object")
    pools = data["pools"]
    if set(pools) != set(KINDS):
        raise ValueError(f"{path}: pools must declare exactly {KINDS}")
    return {kind: (None if pools[kind] is None else validate_policy(kind, pools[kind])) for kind in KINDS}


def policy_digest(policy) -> str:
    """Digest of the five rule fields (what the runtime applies), independent of evidence notes."""
    canonical = json.dumps({field: policy[field] for field in FIELDS}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


@contextlib.contextmanager
def apply_pool_policy(module, policy):
    """Run the detector runtime with ``policy`` as its candidate rule, then restore it."""
    missing = [name for name in RUNTIME_CONSTANTS.values() if not hasattr(module, name)]
    if missing or not callable(getattr(module, "build_sample_universe", None)):
        raise ValueError(f"Detector runtime cannot take a pool policy (missing {missing or 'build_sample_universe'})")
    saved = {name: getattr(module, name) for name in RUNTIME_CONSTANTS.values()}
    for field, name in RUNTIME_CONSTANTS.items():
        setattr(module, name, policy[field])
    try:
        yield module
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


def describe_split_rule(config) -> str:
    """The companion manifest's split ``candidate_rule`` text for a config id or policy."""
    fields = parse_config_id(config) if isinstance(config, str) else config
    radius = f"{float(fields['radius_um']):g}"
    quota = fields["per_anchor_k"]
    anchors = {"tip_to_tip": "from every degree-1 fragment tip, the nearest tip of each other segment",
               "tip_to_any_node": "from every degree-1 fragment tip, the nearest node of each other segment",
               "any_node_to_any_node": "from every fragment node, the nearest node of each other segment"}[fields["mode"]]
    kept = ("every partner segment is kept" if quota is None
            else f"the {quota} nearest partner segment{'s are' if quota != 1 else ' is'} kept")
    return (f"Candidates: {anchors} within {radius} um; {kept} "
            f"({fields['mode']}, r={radius}, k={'all' if quota is None else quota}). One row per unordered "
            "segment pair; occurrences list every anchor/partner realisation, the first being the shortest "
            "gap. Positive: the pair is a GT split key (distinct non-zero labels across a GT split edge, or "
            "the two flanks of a zero-label GT stretch on one neuron). Label 0 only where GT can judge: both "
            "segments carry a GT node label; otherwise NaN.")

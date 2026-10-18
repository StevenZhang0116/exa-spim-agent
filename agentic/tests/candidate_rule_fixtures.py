"""Shared test helpers for the declared candidate rules (candidate_rules.py, 2026-10-16).

``adapter_namespace`` executes a target adapter the way an assembled detector would, with
``__file__`` placed at the repository root so the runtime loads the repository's
``candidate_rules.py`` (byte-identical to the copy assembly puts beside a detector).
``write_policy`` freezes a rule into a declared-rule artifact via a temporary declaration.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RULES_PATH = REPO / "candidate_rules.py"
POLICY_SHA256 = "a" * 64


def rules_sha256() -> str:
    return hashlib.sha256(RULES_PATH.read_bytes()).hexdigest()


def split_rule(mode="tip_to_any_node", radius_um=50.0, per_anchor_k=2) -> dict:
    quota = "all" if per_anchor_k is None else int(per_anchor_k)
    return {"config_id": f"{mode}|r={float(radius_um):g}|k={quota}", "mode": mode,
            "radius_um": float(radius_um), "per_anchor_k": per_anchor_k, "global_cap": None}


def merge_rule(nms_um=20.0, claim_radius_um=150.0, positive_label_radius_um=20.0,
               tip_gap_um=None, tip_partner_k=None) -> dict:
    if tip_gap_um is None:
        mode, config = "junction", f"junction|nms={float(nms_um):g}|r={float(claim_radius_um):g}"
    else:
        mode = "junction+tip_to_shaft"
        config = (f"{mode}|nms={float(nms_um):g}|gap={float(tip_gap_um):g}"
                  f"|k={tip_partner_k}|r={float(claim_radius_um):g}")
    return {"config_id": config, "mode": mode, "nms_um": float(nms_um),
            "tip_gap_um": None if tip_gap_um is None else float(tip_gap_um),
            "tip_partner_k": tip_partner_k, "claim_radius_um": float(claim_radius_um),
            "positive_label_radius_um": float(positive_label_radius_um)}


def _comp_to_seg(graph):
    return {int(c): int(str(s).split(".")[0]) for c, s in graph.component_id_to_swc_id.items()}


def adapter_namespace(target, rule=None, *, policy_sha256=POLICY_SHA256) -> dict:
    """Namespace holding an executed target adapter (split / merge-site take ``rule``)."""
    from agentic.detector_build.target_runtime import target_adapter_source

    namespace = {"np": np, "defaultdict": defaultdict, "sys": sys,
                 "build_comp_to_seg": _comp_to_seg,
                 "__file__": str(REPO / "_test_assembled_detector.py")}
    kwargs = {} if rule is None else {"candidate_policy": rule, "candidate_policy_sha256": policy_sha256,
                                      "candidate_rules_sha256": rules_sha256()}
    exec(target_adapter_source(target, **kwargs), namespace)
    return namespace


def write_declaration(path: Path, *, merge: dict | None = None, split: dict | None = None) -> Path:
    path = Path(path)
    path.write_text(json.dumps({"schema_version": 2, "revised": "test", "pools": {
        "merge": merge or merge_rule(tip_gap_um=20.0, tip_partner_k=2),
        "split": split or split_rule(radius_um=80.0)}}))
    return path


def write_policy(path: Path, kind: str, rule: dict) -> Path:
    """Freeze ``rule`` as ``path`` (a declared-rule artifact) through a temporary declaration."""
    from agentic.detector_build.candidate_policy import compile_declared_candidate_policy

    path = Path(path)
    declaration = write_declaration(path.with_name(f".{path.stem}.declaration.json"), **{kind: rule})
    compile_declared_candidate_policy(kind, path, project_root=REPO, declaration_path=declaration,
                                      rules_path=RULES_PATH)
    return path

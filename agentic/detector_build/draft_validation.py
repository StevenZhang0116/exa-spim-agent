"""Narrow, read-only validation CLI for detector-build agent drafts.

The build driver reruns every check independently. This command exists so an
agent can see the exact driver-owned contract before ending its turn.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from .assembly import assemble_detector
from .contracts import DetectorTarget
from .inputs import resolve_run_context
from .inventory import validate_semantic_draft
from .model_policy import (
    ModelPolicyContract,
    compile_model_config,
    validate_model_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIVE_POLICY_VERSION = "exclusion-only-v1"
MODEL_CONFIG_SCHEMA_VERSION = 2


def _path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="kind", required=True)

    semantic = subparsers.add_parser("semantic")
    semantic.add_argument("--draft", required=True)
    semantic.add_argument("--run-json", required=True)

    advice = subparsers.add_parser("model-advice")
    advice.add_argument("--draft", required=True)
    advice.add_argument("--inventory", required=True)
    advice.add_argument("--policy", required=True)

    feature = subparsers.add_parser("feature")
    feature.add_argument("--draft", required=True)
    feature.add_argument("--template", required=True)
    feature.add_argument(
        "--target", required=True,
        choices=(DetectorTarget.MERGE.value, DetectorTarget.SPLIT.value),
    )
    args = parser.parse_args()

    if args.kind == "semantic":
        # This narrow draft check must not enforce the unbacked-verdict policy:
        # the driver already decided abort-vs-reconcile at startup, and an agent
        # running this mid-build (under --reconcile-unbacked-verdicts) would
        # otherwise wedge on an inconsistency unrelated to its draft.
        context = resolve_run_context(
            _path(args.run_json), PROJECT_ROOT, PREDICTIVE_POLICY_VERSION,
            reconcile_unbacked_verdicts=True)
        validate_semantic_draft(
            _path(args.draft),
            selected_ids=list(context.selected_ids),
            project_root=PROJECT_ROOT,
        )
    elif args.kind == "model-advice":
        policy_path = _path(args.policy)
        contract = ModelPolicyContract.from_path(policy_path)
        with tempfile.TemporaryDirectory(prefix="model-advice-validation-") as tmpdir:
            output = Path(tmpdir) / "model_candidates.json"
            compile_model_config(
                _path(args.draft), output, _path(args.inventory), policy_path,
                contract, MODEL_CONFIG_SCHEMA_VERSION, PROJECT_ROOT,
            )
            validate_model_config(
                output, _path(args.inventory), policy_path, contract,
                MODEL_CONFIG_SCHEMA_VERSION, PROJECT_ROOT,
            )
    else:
        with tempfile.TemporaryDirectory(prefix="feature-draft-validation-") as tmpdir:
            assemble_detector(
                _path(args.template), _path(args.draft),
                Path(tmpdir) / "detector.py",
                target=DetectorTarget(args.target),
            )
    print("DRAFT_VALIDATION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

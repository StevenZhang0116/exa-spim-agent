"""
Generate a multi-feature merge or split detector from a finished AutoDiscovery run.

Where ``run_discovery_workflow.py`` DIGESTS one run export into a ranked
``<RUN>.summary.md`` report, this workflow goes one step further downstream: it
takes that finished report, the reproducer's loading-fixed hypothesis scripts
(normally ``autodiscovery/<RUN>.json.predictive.rerun/hypo_<id>.py``), and any
test-fixer scripts under the matching ``.fixed/`` directory, and WRITES A DETECTOR — one
script that computes every eligible selected hypothesis's features in shared passes over
the skeleton graph, evaluates several appropriately regularized tabular models
under nested cross-validation, and fits the selected strategy to score one
segment for merge runs or one candidate segment pair for split runs.

The motivation comes from the reports themselves: each hypothesis contributes
one or more related features, and the recurring caveat is that none is precise enough alone
("high recall, ~8% precision — best used as one component of an ensemble").
This workflow builds the combined feature table and selects a model for it.

THIS WORKFLOW IS CODE GENERATION ONLY, and it takes NO DATASET. Its inputs are
the run export, the report beside it, the predictive-selection manifest when
applicable, the ``.rerun/`` scripts, and optional ``.fixed/`` scripts. Split
builds additionally require the source-bound
``<RUN>.split-feature-applicability.json`` written by discovery. It
cross-checks selection, rerun MANIFEST, inventory paths, and source hashes before
generation. It never loads a cache; it runs only the generated CLI's no-data
``--help`` and synthetic smoke modes. The pkl needs >20 GB RAM and may require
hours of
graph traversal, so the operator runs the finished script themselves on a
compute node with the data.
Hence a login node is fine for this. The origin dataset is recorded in the
report, which is where the README's provenance comes from; the run command
printed at the end guesses the cache path from the run name (``...-794495-mcl100``
-> ``cache/dataset_cache_794495_mcl100_add.pkl``) purely as a convenience.

A persistent Claude session runs the ordered STEPS so later steps share earlier
context. Subagents live in ``.claude/agents/`` and are auto-discovered via
``setting_sources``. Deliverables land in
``autodiscovery-application/<RUN>/`` (one subfolder per originating run):

    merge_site_detector.py or         target-specific model comparison + detector
    split_site_detector.py
    split_candidate_policy.json       split-only, evidence-validated policy choice
    feature_inventory.json            feature definition + defined condition
    model_candidates.json             validated model families + small grids
    README.md                         provenance, how to run, how to read it
    RUN_COMMANDS.md                   commands bound to this detector's hashes
    detector_build_workflow.log.txt   this driver's console output (--log-txt)

The generated detector run additionally writes target-row CSV scores, a JSON
model-selection manifest, a fitted winner joblib, a run log and figures.
After a successful run with figures, invoke
``agentic/run_detector_result_analysis.py PATH`` to validate those artifacts and
ask the ``detector-result-analyst`` subagent to write a grounded bilingual
``RESULT_ANALYSIS.md``: complete English first, followed by a faithful complete
Chinese translation. ``PATH`` may be the application directory or one nested
completed run directory under it.

The driver tees its own stdout+stderr — step progress, subagent tool calls, each
step's final reply, and any fatal diagnostic — into the log as it goes rather
than at exit, so a workflow that dies mid-step still leaves a record of where.

If that folder ALREADY EXISTS, the driver offers to delete it first and waits for
a typed confirmation. The reason is that regenerating into a populated folder
leaves a NEW detector script beside a CSV, a log and figures produced by the OLD
one, with nothing on disk marking the mismatch. Before deleting it lists every
file, split into what this workflow rewrites anyway, what a compute-node rerun of
the detector recreates, and what nothing here can bring back — plus any
uncommitted git paths, since an untracked file is gone for good. ``--yes`` skips
the prompt (required in batch jobs, which have no terminal); ``--keep-existing``
writes into the folder as it stands, the older behaviour.

The steps (split builds first run the additional candidate-policy
step described below; merge builds retain the established four-step sequence):
  0. select-candidate-policy — for split builds only, an agent interprets a
                           completed candidate-pool AI review. The driver checks
                           hashes and structured evidence, independently minimizes
                           candidate count subject to worst-brain recall >= 0.90,
                           and freezes split_candidate_policy.json. Assembly then
                           embeds that exact policy in the detector runtime.
  1. inventory-features  — the agent reads <RUN>.summary.md, each selected
                           .rerun/hypo_<id>.py and matching .fixed/hypo_<id>.py;
                           it records only semantic judgments (source choice,
                           feature math, aggregation and defined condition).
                           The driver joins IDs, evidence, paths and hashes and
                           compiles the public feature_inventory.json. For split,
                           it also joins the exact selected source SHA to its
                           validated node-role requirement.
  2. configure-models    — write a validated model-candidate configuration. A
                           versioned declarative policy defines mandatory models,
                           optional limits and parameter rules. No arbitrary
                           estimator imports.
  3. generate-detector   — the agent writes only the task-specific feature
                           implementation. The driver AST-validates it and injects
                           it into a reviewed runtime template that owns the CLI,
                           model selection, outputs and smoke contract. Split
                           features use applicability guards without removing
                           rows from the candidate universe.
  4. verify-and-document — enrich the factual README skeleton that the driver
                           wrote after detector assembly; check the script against the inventory and
                           the inventoried rerun/fixed sources (every feature computed and fitted,
                           constants unchanged, defined flags consistent, no sandbox
                           pip preamble, parses, --help works), report defects
                           without editing the assembled detector, run
                           driver-owned no-data CLI checks, and add semantic
                           interpretation outside the immutable provenance block.
                           No pkl, no results — README.md points to the driver-owned
                           RUN_COMMANDS.md and says in what order to read the numbers
                           when they arrive.

The audit lives INSIDE the generated script because missing structure is a real
predictor and a real confound. On ``merge-error-794495-mcl100_2026-08-04``, 6396
of 9623 adjudicable segments had no fragment component and none was a merge.
Historical sentinels made that group trivially separable and lifted same-brain
cross-validated ROC-AUC. Generated detectors therefore retain NaN, add explicit
is_defined flags, report coverage by class, and let fold-local preprocessing or
native-NaN models handle the numeric value.

A clean merge build runs four steps and a clean split build runs the split-only
policy step plus those four steps. If compilation rejects a hidden draft,
`--keep-existing` can recover it only after rebuilding and strictly validating
its public artifact, avoiding a repeated semantic agent turn.

Usage (from the ``exa-spim-agent/`` project root; a login node is fine):
    conda activate panda
    python agentic/run_detector_build_workflow.py \
        autodiscovery/<merge-error-or-split-error-run>.json

Then run the generated detector yourself, on a compute node:
    sbatch --mem=80G --wrap="\
        source /shared/utils.x86_64/anaconda3-2024.10/etc/profile.d/conda.sh; \
        conda activate panda; \
        python autodiscovery-application/<RUN>/<merge_or_split>_site_detector.py \
            cache/dataset_cache_<brain>_mcl<N>_add.pkl"

Then analyze that completed result directory:
    python agentic/run_detector_result_analysis.py \
        autodiscovery-application/<RUN>/runs/<COMPLETED-RESULT>
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

try:  # package import (tests and ``python -m agentic...``)
    from agentic.detector_build.agent_session import (
        load_claude_sdk,
        open_agent_session,
    )
    from agentic.detector_build.assembly import assemble_detector
    from agentic.detector_build.candidate_policy import (
        OBJECTIVE_NAME as CANDIDATE_POLICY_OBJECTIVE_NAME,
        candidate_policy_source_paths,
        compile_candidate_policy,
        expected_mcl_from_run_path,
        validate_candidate_policy_sources,
        validate_frozen_candidate_policy,
    )
    from agentic.detector_build.contracts import (
        BUILD_ARTIFACT_NAMES,
        CANDIDATE_POLICY_ADVICE_DRAFT_NAME,
        CANDIDATE_POLICY_NAME,
        DetectorTarget,
        DRIVER_LOG_NAME,
        FEATURE_INVENTORY_NAME,
        FEATURE_IMPLEMENTATION_DRAFT_NAME,
        FEATURE_SEMANTICS_DRAFT_NAME,
        MODEL_ADVICE_DRAFT_NAME,
        MODEL_CONFIG_NAME,
        README_NAME,
        RUN_COMMANDS_NAME,
        RunContext,
        build_artifact_names,
        target_spec,
    )
    from agentic.detector_build.inputs import (
        corrected_status_by_id as _resolved_corrected_status_by_id,
        integer_ids as _resolved_integer_ids,
        load_json_object as _resolved_load_json_object,
        manifest_ids as _resolved_manifest_ids,
        origin_cache_hint as _resolved_origin_cache_hint,
        protected_source_paths,
        rel_to_root as _resolved_rel_to_root,
        report_evidence_by_id as _resolved_report_evidence_by_id,
        resolve_run_context as _resolve_run_context,
        run_stem as _resolved_run_stem,
        sha256 as _resolved_sha256,
        unbacked_correction_verdicts as _resolved_unbacked_correction_verdicts,
    )
    from agentic.detector_build.inventory import (
        SEMANTIC_DRAFT_CONTRACT,
        compile_feature_inventory,
    )
    from agentic.detector_build.documentation import (
        read_driver_generated_block,
        validate_readme_enrichment,
        write_readme_skeleton,
        write_run_commands,
    )
    from agentic.detector_build.model_policy import (
        ModelPolicyContract,
        REQUIRED_GRID_PARAMETERS,
        compile_model_config,
        load_model_policy as _resolved_load_model_policy,
        valid_grid_value as _resolved_valid_grid_value,
        validate_model_config as _resolved_validate_model_config,
    )
    from agentic.detector_build.verification import (
        DEFAULT_DETECTOR_CHECK_TIMEOUT_S,
        SMOKE_SUCCESS_MARKER,
        validate_detector_executable_contract,
        validate_detector_source as _validate_detector_source,
    )
    from agentic.split_feature_applicability import load_applicability
except ModuleNotFoundError as exc:  # direct ``python agentic/run_....py``
    if exc.name != "agentic":
        raise
    from detector_build.agent_session import (  # type: ignore[no-redef]
        load_claude_sdk,
        open_agent_session,
    )
    from detector_build.assembly import assemble_detector  # type: ignore[no-redef]
    from detector_build.candidate_policy import (  # type: ignore[no-redef]
        OBJECTIVE_NAME as CANDIDATE_POLICY_OBJECTIVE_NAME,
        candidate_policy_source_paths,
        compile_candidate_policy,
        expected_mcl_from_run_path,
        validate_candidate_policy_sources,
        validate_frozen_candidate_policy,
    )
    from detector_build.contracts import (  # type: ignore[no-redef]
        BUILD_ARTIFACT_NAMES,
        CANDIDATE_POLICY_ADVICE_DRAFT_NAME,
        CANDIDATE_POLICY_NAME,
        DetectorTarget,
        DRIVER_LOG_NAME,
        FEATURE_INVENTORY_NAME,
        FEATURE_IMPLEMENTATION_DRAFT_NAME,
        FEATURE_SEMANTICS_DRAFT_NAME,
        MODEL_ADVICE_DRAFT_NAME,
        MODEL_CONFIG_NAME,
        README_NAME,
        RUN_COMMANDS_NAME,
        RunContext,
        build_artifact_names,
        target_spec,
    )
    from detector_build.inputs import (  # type: ignore[no-redef]
        corrected_status_by_id as _resolved_corrected_status_by_id,
        integer_ids as _resolved_integer_ids,
        load_json_object as _resolved_load_json_object,
        manifest_ids as _resolved_manifest_ids,
        origin_cache_hint as _resolved_origin_cache_hint,
        protected_source_paths,
        rel_to_root as _resolved_rel_to_root,
        report_evidence_by_id as _resolved_report_evidence_by_id,
        resolve_run_context as _resolve_run_context,
        run_stem as _resolved_run_stem,
        sha256 as _resolved_sha256,
        unbacked_correction_verdicts as _resolved_unbacked_correction_verdicts,
    )
    from detector_build.inventory import (  # type: ignore[no-redef]
        SEMANTIC_DRAFT_CONTRACT,
        compile_feature_inventory,
    )
    from detector_build.documentation import (  # type: ignore[no-redef]
        read_driver_generated_block,
        validate_readme_enrichment,
        write_readme_skeleton,
        write_run_commands,
    )
    from detector_build.model_policy import (  # type: ignore[no-redef]
        ModelPolicyContract,
        REQUIRED_GRID_PARAMETERS,
        compile_model_config,
        load_model_policy as _resolved_load_model_policy,
        valid_grid_value as _resolved_valid_grid_value,
        validate_model_config as _resolved_validate_model_config,
    )
    from detector_build.verification import (  # type: ignore[no-redef]
        DEFAULT_DETECTOR_CHECK_TIMEOUT_S,
        SMOKE_SUCCESS_MARKER,
        validate_detector_executable_contract,
        validate_detector_source as _validate_detector_source,
    )
    from split_feature_applicability import load_applicability  # type: ignore[no-redef]

# Project root = the directory that holds .claude/, agentic/, autodiscovery/.
# agentic/run_detector_build_workflow.py -> parent.parent is the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Applications live one subfolder per originating run.
APPLICATION_DIR_REL = "autodiscovery-application"
DEFAULT_SPLIT_CANDIDATE_REVIEW_REL = (
    "notebooks/split_candidate_pool_sweep_outputs/"
    "mcl100_5f92a29e40d94a90/AI_REVIEW.md"
)
DEFAULT_SPLIT_MIN_WORST_BRAIN_RECALL = 0.90

MODEL_CONFIG_SCHEMA_VERSION = 2
MODEL_POLICY_PATH = Path(__file__).resolve().with_name("detector_model_policy.json")
RUNTIME_TEMPLATE_PATH = (
    Path(__file__).resolve().parent
    / "detector_build"
    / "templates"
    / "detector_runtime.py.tmpl"
)
TARGET_RUNTIME_PATH = (
    Path(__file__).resolve().parent / "detector_build" / "target_runtime.py"
)

def _load_model_policy(path: Path) -> dict:
    """Compatibility wrapper for the reusable policy loader."""
    return _resolved_load_model_policy(path)


MODEL_POLICY_CONTRACT = ModelPolicyContract.from_path(MODEL_POLICY_PATH)
MODEL_POLICY = MODEL_POLICY_CONTRACT.payload
MODEL_FAMILIES = MODEL_POLICY_CONTRACT.families
ALLOWED_MODEL_FAMILIES = MODEL_POLICY_CONTRACT.allowed_families
REQUIRED_MODEL_FAMILIES = MODEL_POLICY_CONTRACT.required_families
OPTIONAL_MODEL_FAMILIES = ALLOWED_MODEL_FAMILIES - REQUIRED_MODEL_FAMILIES
MODEL_PARAMETER_RULES = MODEL_POLICY_CONTRACT.parameter_rules
MODEL_PARAMETER_ALLOWLIST = {
    name: frozenset(rules) for name, rules in MODEL_PARAMETER_RULES.items()
}
MODEL_REQUIRED_PACKAGE = MODEL_POLICY_CONTRACT.required_package
MODEL_NATIVE_NAN = MODEL_POLICY_CONTRACT.native_nan
MAX_OPTIONAL_MODELS = MODEL_POLICY["selection"]["max_optional_models"]
MAX_GRID_COMBINATIONS = MODEL_POLICY["selection"]["max_grid_combinations"]
MODEL_SIMPLICITY_ORDER = tuple(MODEL_POLICY["selection"]["simplicity_order"])
PRIMARY_METRIC = MODEL_POLICY["selection"]["primary_metric"]
SELECTION_RULE = MODEL_POLICY["selection"]["selection_rule"]
OUTER_FOLDS = MODEL_POLICY["selection"]["outer_folds"]
INNER_FOLDS = MODEL_POLICY["selection"]["inner_folds"]
RANDOM_SEED = MODEL_POLICY["selection"]["random_seed"]
DEFAULT_REVIEW_BUDGET = MODEL_POLICY["selection"]["review_budget"]
PREDICTIVE_POLICY_VERSION = "exclusion-only-v1"

AGENT_MODEL = os.environ.get("DETECTOR_BUILD_AGENT_MODEL", "claude-opus-4-8")
AGENT_EFFORT = os.environ.get("DETECTOR_BUILD_AGENT_EFFORT", "xhigh")


def _positive_timeout_from_environment(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        timeout = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc
    if timeout < 1:
        raise RuntimeError(f"{name} must be positive.")
    return timeout


# Detector construction can require one long turn to audit dozens of scripts.
AGENT_STEP_TIMEOUT_S = _positive_timeout_from_environment(
    "DETECTOR_BUILD_AGENT_TIMEOUT_S", 6 * 60 * 60
)
AGENT_CONNECT_TIMEOUT_S = _positive_timeout_from_environment(
    "DETECTOR_BUILD_AGENT_CONNECT_TIMEOUT_S", 60
)
DETECTOR_CHECK_TIMEOUT_S = _positive_timeout_from_environment(
    "DETECTOR_BUILD_CHECK_TIMEOUT_S", DEFAULT_DETECTOR_CHECK_TIMEOUT_S
)
MAX_AGENT_REPAIR_ATTEMPTS = 2


class WorkflowCostSummary:
    """Track the API cost the Claude SDK reports for this run.

    ``ResultMessage.total_cost_usd`` is the *session-cumulative* cost, not the
    price of the turn that just finished, so the run total is the latest
    cumulative value per session — summing the raw reports across turns would
    count every earlier turn once per later turn.
    """

    def __init__(self) -> None:
        self.started_turns = 0
        self.reported_turns = 0
        self.unreported_turns = 0
        self._session_usd: dict[object, float] = {}

    @property
    def total_usd(self) -> float:
        return sum(self._session_usd.values())

    def start_turn(self) -> None:
        self.started_turns += 1

    def add(self, cost_usd: object, session_id: object = None) -> float | None:
        """Record one turn's report; return the increment it represents."""
        if cost_usd is None:
            self.unreported_turns += 1
            return None
        try:
            cost = float(cost_usd)
        except (TypeError, ValueError):
            self.unreported_turns += 1
            return None
        if not math.isfinite(cost) or cost < 0:
            self.unreported_turns += 1
            return None
        previous = self._session_usd.get(session_id, 0.0)
        # Cumulative reports are non-decreasing; max() keeps the total intact
        # if a stray out-of-order report ever arrives.
        self._session_usd[session_id] = max(previous, cost)
        self.reported_turns += 1
        return max(0.0, cost - previous)

    def describe(self) -> str:
        unfinished_turns = max(
            0,
            self.started_turns - self.reported_turns - self.unreported_turns,
        )
        coverage = (
            f"{self.reported_turns} reported turn(s), "
            f"{self.unreported_turns} completed turn(s) without a cost, "
            f"{unfinished_turns} started turn(s) without a ResultMessage"
        )
        if self.reported_turns == 0:
            return f"unavailable ({coverage})"
        return f"${self.total_usd:.4f} ({coverage})"


def log(msg: str) -> None:
    """Timestamped progress line to stderr (kept separate from step output)."""
    print(f"[{datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


class _Tee:
    """Fan writes out to several streams at once (console + log file).

    Stands in for ``sys.stdout`` / ``sys.stderr`` so the whole workflow is
    captured to a txt file while still streaming live to the terminal. Flushes on
    every write: this run takes tens of minutes, so a buffered log that only
    lands at exit would be useless while it is going — and empty if the process
    is killed.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        # Never claim a TTY: a tee is not one, and libraries that probe this to
        # decide on progress bars / ANSI codes would otherwise pollute the log.
        return False


def describe_tool(block) -> str:
    """One-line, human-readable summary of a tool-use block for progress logs."""
    name = getattr(block, "name", "tool")
    args = getattr(block, "input", {}) or {}
    if name == "Task":
        sub = args.get("subagent_type") or args.get("description") or "?"
        return f"Task → subagent '{sub}'"
    if name == "Bash":
        cmd = " ".join(str(args.get("command", "")).split())
        return f"Bash: {cmd[:100]}" + ("…" if len(cmd) > 100 else "")
    if name in ("Write", "Edit", "Read", "Glob"):
        target = args.get("file_path") or args.get("path") or args.get("pattern") or ""
        return f"{name}: {target}"
    return name


def run_stem(run_json: Path) -> str:
    """``autodiscovery/foo.json`` -> ``foo`` — names the application subfolder."""
    return _resolved_run_stem(run_json)


def origin_cache_hint(run_json: Path) -> str | None:
    """Guess the origin cache path from the run name, or None if it doesn't say.

    ``merge-error-794495-mcl100_2026-08-04`` -> ``cache/dataset_cache_794495_
    mcl100_add.pkl``. Only a hint, for the command this workflow prints at the
    end: the authority on which dataset the run was generated on is the report
    itself, and older run exports (``ground-truth-error-annotations-...``) carry
    no brain id in the name at all.
    """
    return _resolved_origin_cache_hint(run_json)


def _rel_to_root(path: Path) -> str:
    """Path relative to PROJECT_ROOT (the session cwd), else absolute.

    These strings go into agent instructions and into the run command printed at
    the end, so they have to be usable as typed. A target outside the project —
    an ``--out-dir`` under /tmp, say — relativizes to a chain of ``..`` that is
    correct but unreadable and breaks the moment it is pasted from elsewhere, so
    fall back to the absolute path there.
    """
    return _resolved_rel_to_root(path, PROJECT_ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Steps
# ─────────────────────────────────────────────────────────────────────────────

def build_steps(
    run_rel: str,
    summary_rel: str,
    rerun_rel: str,
    fixed_rel: str | None,
    selection_rel: str | None,
    selected_ids: list[int],
    out_dir_rel: str,
    corrected_results_rel: str | None = None,
    target: DetectorTarget = DetectorTarget.MERGE,
    candidate_review_rel: str | None = None,
    candidate_policy_minimum_recall: float = (
        DEFAULT_SPLIT_MIN_WORST_BRAIN_RECALL
    ),
    split_applicability_rel: str | None = None,
) -> list[dict]:
    """Build the model-selection detector workflow instructions."""
    spec = target_spec(target)
    detector_rel = f"{out_dir_rel}/{spec.detector_name}"
    feature_impl_rel = f"{out_dir_rel}/{FEATURE_IMPLEMENTATION_DRAFT_NAME}"
    inventory_rel = f"{out_dir_rel}/{FEATURE_INVENTORY_NAME}"
    semantics_rel = f"{out_dir_rel}/{FEATURE_SEMANTICS_DRAFT_NAME}"
    model_config_rel = f"{out_dir_rel}/{MODEL_CONFIG_NAME}"
    model_advice_rel = f"{out_dir_rel}/{MODEL_ADVICE_DRAFT_NAME}"
    readme_rel = f"{out_dir_rel}/{README_NAME}"
    run_commands_rel = f"{out_dir_rel}/{RUN_COMMANDS_NAME}"
    candidate_policy_rel = f"{out_dir_rel}/{CANDIDATE_POLICY_NAME}"
    candidate_advice_rel = (
        f"{out_dir_rel}/{CANDIDATE_POLICY_ADVICE_DRAFT_NAME}"
    )
    feature_policy_validation_arg = (
        f" --candidate-policy {candidate_policy_rel}"
        if spec.target is DetectorTarget.SPLIT else ""
    )
    fixed_note = (
        f"Corrected scripts are available under {fixed_rel}/."
        if fixed_rel is not None
        else "No matching corrected-script directory is present for this run."
    )
    source_note = (
        f"the loading-fixed sources under {rerun_rel}/ and corrected sources under {fixed_rel}/"
        if fixed_rel is not None
        else f"the loading-fixed sources under {rerun_rel}/"
    )
    selection_note = (
        f"The authoritative hypothesis set is selected_ids in {selection_rel}: "
        f"{selected_ids}."
        if selection_rel is not None
        else f"This legacy run has no predictive-selection file; the validated "
             f"rerun MANIFEST is authoritative: {selected_ids}."
    )
    corrected_results_note = (
        f"Read corrected execution statuses from {corrected_results_rel}; its "
        "results[].result_status is authoritative for corrected_result_status."
        if corrected_results_rel is not None
        else "No corrected-results JSON is present; the driver therefore sets every "
             "corrected_result_status to null. Input preflight rejects a selected "
             "report row with a post-correction verdict when that JSON is missing."
    )
    required_models = ", ".join(sorted(REQUIRED_MODEL_FAMILIES))
    optional_models = ", ".join(sorted(OPTIONAL_MODEL_FAMILIES))
    native_nan_models = ", ".join(sorted(
        name for name, enabled in MODEL_NATIVE_NAN.items() if enabled
    ))
    package_contract = ", ".join(
        f'{name}="{package}"' for name, package in sorted(MODEL_REQUIRED_PACKAGE.items())
    ) or "none"
    parameter_contract = "\n".join(
        f"- {name}: {', '.join(MODEL_PARAMETER_RULES[name])}"
        for name in MODEL_SIMPLICITY_ORDER
    )
    policy_rel = _rel_to_root(MODEL_POLICY_PATH)
    runtime_template_rel = _rel_to_root(RUNTIME_TEMPLATE_PATH)
    if spec.target is DetectorTarget.SPLIT:
        if split_applicability_rel is None:
            raise ValueError("Split build steps require feature applicability metadata.")
        target_inventory_note = f"""
This is a split-detection run. The detector row unit is one canonical unordered
candidate segment pair, not one segment and not one GT edge. Read the frozen,
driver-validated candidate policy at {candidate_policy_rel}; its pairing rule,
radius, per-anchor distinct-partner-segment quota, and null global cap define the
exact runtime sample universe. Inventory node, node-pair, component-pair, and
segment-pair quantities with their exact reduction to that row unit. Direct
gt_edge_error==1 pairs and gap pairs flanking connected zero-label GT regions
supply is_split only after candidate construction; labels and GT-only split
metadata must never be features. Record any source detector arm whose feature
semantics are incompatible with the selected universe as a limitation rather
than silently changing either the policy or the feature definition.

Read the source-bound node-role annotations at {split_applicability_rel}. Match
the exact feature source selected below by hypothesis id and SHA-256. An
`unclear` source must be excluded rather than guessed. The driver will copy each
included source's validated requirement into every resulting inventory feature.
The usable values are `requires_both_tips`, `requires_tip_anchor`, and
`no_tip_requirement`.
""".strip()
        extraction_contract = """
- FeatureAccumulator with explicit measured/definedness membership;
- extract_features(payload, verbose=True, timing=None,
  enabled_analysis_keys=None, profile_segment_limit=None, image_workers=1),
  returning canonical candidate-pair records, is_split labels, and the
  accumulator;
- endpoint/pair/component geometry helpers needed by those definitions.

Use the runtime-provided build_sample_universe(payload) as the sole authority for
candidate identity and labels. Do not derive a competing split key or candidate
pool in the feature fragment. Aggregate endpoint-level values to the candidate
segment-pair row exactly as recorded by the inventory. Each sample is a dict with
candidate_id, segment_id_a, segment_id_b, node_id_a, node_id_b, gap_um,
anchor_side, partner_rank, node_role_a, node_role_b, and occurrences.
`occurrences` is the stable closest-first tuple of dictionaries carrying those
node/gap/anchor/rank/role fields for every policy-admitted occurrence of the same
unordered segment pair. Under tip_to_any_node, only the anchor is guaranteed to
be a degree-1 tip; the partner may be a tip, shaft, or branch node. Runtime-only
is_merge_creating and
contains_known_merge_segment fields are also present strictly for output audit
and must be ignored by feature extraction. FeatureAccumulator must preserve this exact sample
order in to_frame(). GT-only fields (is_split,
split_kind, GT neuron membership, merge-risk audit fields) may not enter
FEATURE_REGISTRY or any feature computation.

BLINDNESS IS ENFORCED, NOT ADVISORY. At runtime your extract_features receives a
_BlindPayloadView, not the raw payload: reading any GT key (gt_graph,
gt_node_canonical_label, gt_merge_labels, gt_edge_error) raises
BlindnessViolation and aborts the run; `"gt_..." in payload` reports False; the
sample-universe cache hands you audit-stripped rows and ALL-ZERO labels, and the
runtime re-attaches the true labels after extraction by row identity. So never
read GT keys, never condition on a label value, and never branch on the audit
fields — the assembler additionally rejects the fragment statically if any GT
key literal appears in it. Features must be computable from the fragments graph
and image state alone.

For each inventory feature, obey its `node_role_requirement` using the
runtime-provided compatible_occurrences(sample, requirement):
- requires_both_tips uses only occurrences whose two nodes are tips;
- requires_tip_anchor uses occurrences whose recorded anchor is a tip;
- no_tip_requirement uses all occurrences.
If no occurrence is compatible, retain the candidate row and leave that feature
NaN with is_defined=False. Never filter the candidate universe down to the
feature's applicability subset.
""".strip()
        verification_scope = (
            "candidate-pair-level rather than exact missing-voxel localization"
        )
        candidate_policy_generation_note = (
            f"Read {candidate_policy_rel} as the immutable definition of the "
            "runtime candidate universe. Do not copy or redefine its generator "
            "logic in feature code."
        )
        candidate_policy_verification = f"""
Confirm that the assembled detector embeds the exact config_id, pairing rule,
radius, per-anchor k, null global cap, and SHA-256 from {candidate_policy_rel}.
Confirm sample_universe_audit reports those same values. Treat the policy as
immutable and report any mismatch rather than editing either artifact.
Also verify every inventory `node_role_requirement` matches the exact selected
source annotation in {split_applicability_rel}. Restrictive features must call
compatible_occurrences before their endpoint math; absence of a compatible
occurrence produces NaN/is_defined=False without deleting the candidate row.
""".strip()
        oof_verification = """
Segment-disjoint split folds must never share either candidate segment between
train and validation. Candidate rows whose two segments do not land together in
a validation fold remain explicitly unscored (NaN), while each eligible row is
scored once. Persist `n_scored`, report finite-score coverage, and exclude
unscored rows from OOF queues/workload; recall on that queue is conditional on
the OOF-scored subset, not the complete candidate universe.
""".strip()
        measuretime_scope = f"""
For split measuretime, `profile_segment_limit` means a deterministic sample of
at most that many segment ids drawn from those that actually appear in the
candidate pair universe (first-appearance order). Retain every candidate row in
which EITHER segment of the pair is sampled, and restrict expensive feature
computation to the retained rows' nodes/components. Do NOT require BOTH
endpoints to be sampled: the row unit is a segment pair, so an AND filter
almost always retains zero rows and empties the profile. Do not reinterpret the
limit as a number of GT edges, positive split pairs, endpoint occurrences, or
candidate rows. The reported seconds are observed on that bounded sample, not a
full-run estimate. Draw the sample density-stratified, not first-N: rank the
eligible segments by a cheap local-density proxy (for example fragment-node
count within a fixed radius of the segment's anchor endpoint) and take the
deterministic sample evenly across that ranking, so high-, mid-, and
low-density regions are all represented. A sparse-only sample underestimates
density_scaled and component_scaled primitive costs by orders of magnitude and
makes the ranked cost report untrustworthy.
""".strip()
    else:
        target_inventory_note = """
This is a merge-detection run. The detector row unit remains one adjudicable
non-zero canonical segment; the RUNTIME's build_sample_universe derives that
universe and its is_merge labels from GT (the only sanctioned GT consumer), and
segments with no fragment component remain in the row universe. Feature math is
blind: any hypothesis whose distinctive quantity is measured against GT tracings
(a distance to the GT graph, a GT-node density, a deviation from a GT path) is
NOT blind-computable — exclude it with that reason rather than inventorying it
as a feature.
""".strip()
        extraction_contract = """
- SegmentAccumulator with explicit measured/definedness membership;
- extract_features(payload, verbose=True, timing=None,
  enabled_analysis_keys=None, profile_segment_limit=None, image_workers=1),
  returning adjudicable ids, merge labels, and the accumulator;
- only geometry/data helpers directly needed by those definitions.

Read the segment universe and labels ONLY from the runtime-provided
build_sample_universe(payload) (cached tuple of adjudicable segment ids and
is_merge labels); never derive a competing universe from GT arrays.

BLINDNESS IS ENFORCED, NOT ADVISORY. At runtime your extract_features receives a
_BlindPayloadView, not the raw payload: reading any GT key (gt_graph,
gt_node_canonical_label, gt_merge_labels, gt_edge_error) raises
BlindnessViolation and aborts the run; `"gt_..." in payload` reports False; the
sample-universe cache hands you ALL-ZERO labels, and the runtime re-attaches the
true labels after extraction by row identity. A feature like a distance/density
measured against GT tracings is therefore impossible to compute — and is exactly
the ascertainment artifact this gate exists to stop (GT tracing is required to
DETECT a merge, so GT-proximity encodes label availability, not biology). The
assembler additionally rejects the fragment statically if any GT key literal
appears in it. Features must be computable from the fragments graph and image
state alone.
""".strip()
        verification_scope = "segment-level rather than merge-site localization"
        candidate_policy_generation_note = ""
        candidate_policy_verification = ""
        oof_verification = """
Every outer row receives exactly one prediction per available family and the
selector; merge OOF coverage must be complete.
""".strip()
        measuretime_scope = """
For merge measuretime, `profile_segment_limit` is the existing deterministic
sample of component-bearing segment rows. Segments outside the sample remain in
the row universe with profiled feature cells undefined. The reported seconds are
observed sample costs, not a projection of the complete run. Draw the sample
density-stratified, not first-N: rank eligible segments by a cheap
local-density proxy (for example fragment-node count near the segment) and take
the deterministic sample evenly across that ranking, so high-, mid-, and
low-density regions are all represented; a sparse-only sample underestimates
density_scaled and component_scaled primitive costs by orders of magnitude.
""".strip()

    steps = [
        {
            "name": "inventory-features",
            "expects_file": semantics_rel,
            "instruction": f"""
Use the discovery-detector-builder subagent to inventory the features of the
finished AutoDiscovery run {run_rel}. Read the ranked report {summary_rel}, every
loading-fixed hypothesis script under {rerun_rel}/ as indexed by MANIFEST.json,
any same-id corrected script, and markdowns/labeled_dataset_cache.md.
{fixed_note}
{selection_note}
{corrected_results_note}
Inventory exactly those ids: do not add an unselected script merely because it
is present on disk, and do not omit a selected id.

{target_inventory_note}

For each returned hypothesis, first determine whether it is eligible:
- Exclude reproduction FAILED/UNUSABLE unless a later corrected run produced a
  usable measurement that repairs the relevant failure.
- Exclude an uncorrected CRITICAL statistical verdict, an OVERTURNED corrected
  verdict, or a corrected result whose effect vanished.
- Exclude any hypothesis whose report entry carries `Deployability:
  GT-REFERENCING`, and any hypothesis whose feature quantity you find is
  measured against GT state (distance/density to gt_graph, anything derived
  from gt_node_canonical_label / gt_merge_labels / gt_edge_error, deviation
  from a GT path) even when the report predates that token — such a feature is
  not blind-computable and encodes label availability, not biology. The driver
  independently rejects an inventory that includes a GT-REFERENCING row, and
  the assembler/runtime reject GT access in feature code, so an inclusion here
  only wastes the build.
- Keep DIVERGED only with its divergence recorded explicitly; keep WEAK/MINOR
  because multivariate validation can still find independent signal.

For every same-id pair, first require a matching corrected measurement/verdict
in the report. A fixed file without that evidence is stale_fixed_ignored: record
it, ignore it, and use rerun only if the hypothesis is otherwise eligible. A
report correction whose fixed script is missing is unclear and must be excluded.
A correction whose driver measurement exists but is not USABLE (its
results[].result_status is FAILED/UNUSABLE, e.g. a timeout) backs no verdict —
whatever verdict text the report may carry for it is void: classify that
hypothesis unclear and exclude it rather than guessing which feature definition
is safe.
Otherwise compare both scripts and classify correction_scope as test_only,
feature_semantics_changed, or unclear:
- test_only means feature extraction, sample construction, aggregation,
  constants and undefined/default semantics are unchanged. Use the .rerun script
  as feature_source; the corrected result in the report controls evidence only.
- feature_semantics_changed means the correction changes the quantity supplied
  to the detector (including de-duplication, sample construction, aggregation,
  thresholds or defaults). Use the .fixed script as feature_source only when its
  corrected measurement is usable and not OVERTURNED.
- unclear means you cannot prove semantic equivalence or identify a safe fixed
  feature definition; exclude it rather than guessing.

The scripts define feature math; the report defines evidence and verdicts.
Match records ONLY by the original hypothesis ID printed in each report row's
`**ID:**` field and in each script filename `hypo_<ID>.py`. The report's numbered
`### <entry>.` heading is a display rank, NOT the hypothesis ID. Never transfer
evidence from entry number N to hypothesis ID N unless the explicit `**ID:**` also
equals N.

Evidence transcription, paths, hashes, selection metadata, and final schema
assembly are DRIVER responsibilities. Do not copy reproduction_status,
statistical_verdict, corrected_result_status, post_correction_verdict,
selection_manifest, selected_ids, source paths, or hashes into your artifact.
Use that evidence only to make the semantic eligibility/source decision. This
keeps your reasoning focused on what requires code understanding.

Write the internal semantic draft {semantics_rel}. It must obey this exact
driver-owned contract:

{SEMANTIC_DRAFT_CONTRACT}

The driver will combine this draft with same-ID evidence and source hashes to write the public
{inventory_rel}, then apply the strict inventory-v2 validator for merge or the
strict inventory-v3 validator for split. Undefined values
in the eventual detector are NaN plus an explicit <feature>_is_defined column.

Before returning, run this narrow read-only self-check from the project root:
python -m agentic.detector_build.draft_validation semantic --draft \
  {semantics_rel} --run-json {run_rel}
It must print DRAFT_VALIDATION_OK. If it reports a mismatch, repair the draft and
run the same command again.

This turn is ONLY the inventory-features semantic stage. Write only
{semantics_rel}; do not create or modify {inventory_rel}, the model configuration,
detector, README, source scripts, or any helper file. Return immediately after
the semantic draft is complete. Do not load a pkl or invent results.
""".strip(),
        },
        {
            "name": "configure-models",
            "expects_file": model_advice_rel,
            "instruction": f"""
Use the discovery-detector-builder subagent to read {inventory_rel} and write an
internal semantic model-advice draft at {model_advice_rel}. This is advice, not
executable Python and not the public model configuration.

Read the versioned model policy at {policy_rel}. Choose candidates from feature
semantics, coverage and missingness only. Write exactly schema_version=1,
selection_basis, and candidates. Each candidate contains exactly name, reason,
and grid. The driver will add role, native_nan, requires_package, inventory hash,
policy hash, and public schema version, then write and validate
{model_config_rel}. Do not copy those mechanical fields into the draft.

Always include every role=baseline candidate in the versioned model policy:
{required_models}. You may add zero through {MAX_OPTIONAL_MODELS} role=optional
candidates chosen only from: {optional_models}. Choose an extension
only when the inventory gives a concrete reason (for example smooth nonlinearity,
interactions, or native-NaN behavior). Do not use observed model performance,
outer-fold labels, or held-out data to choose the candidate list.

Each candidate's `reason` is non-empty. `grid` maps allowed parameter names to
small non-empty JSON-scalar lists and has at most {MAX_GRID_COMBINATIONS} Cartesian
combinations. Numeric rates/C values must be finite and in their estimator's
valid range; count/depth values must be positive integers (or documented null
where allowed); max_features is null, sqrt/log2, a positive integer, or a
fraction in (0, 1]. Keep grids conservative for sparse positives. The driver
owns native-NaN declarations ({native_nan_models}) and dependency declarations
({package_contract}).

The exact permitted grid keys are:
{parameter_contract}
For explainable_boosting, interactions is a non-negative integer count. Fitting
cost must inform EVERY grid: nested selection refits every grid point three times
per outer fold across five outer folds plus a final all-rows search (~18x per
grid point), so total wall-clock scales with the SUM of all families' grid
sizes times per-fit cost. Budget the whole candidate set to stay clearly under
the {MAX_GRID_COMBINATIONS}-per-family cap rather than filling it: vary only
1-2 hyperparameters per family with 2-3 well-spread values each (defaults from
the literature are close to optimal for tabular data of this size) and pin the
rest to one sensible value. Cheap linear families may use slightly larger grids
than expensive boosted/ensemble families.
- explainable_boosting is among the most expensive fits (minutes per fit on a
  ~100k-row universe). Keep its grid to AT MOST 4 combinations: pick ONE value
  each for max_bins, learning_rate, max_rounds, and min_samples_leaf (its
  internal early stopping makes a generous max_rounds harmless) and vary only
  the axis your inventory rationale actually names — typically interactions
  (e.g. [0, 5]).
- xgboost must list every required parameter, but keep it to AT MOST 8
  combinations: pin subsample, colsample_bytree, and min_child_weight to one
  value each, keep n_estimators moderate (a few hundred, not thousands), and
  vary only max_depth and/or learning_rate (with reg_lambda as an optional
  second axis).
- random_forest / extra_trees, when chosen, need at most ~200-300 trees; do not
  sweep n_estimators.
Never write
a Python class path, module path, import statement, code string, callable, or
arbitrary estimator name. The driver will reject missing baselines, more than
{MAX_OPTIONAL_MODELS} extensions, unknown families/parameters, stale inventory
hashes, and oversized grids before detector generation begins.

Before returning, run this narrow read-only self-check from the project root:
python -m agentic.detector_build.draft_validation model-advice --draft \
  {model_advice_rel} --inventory {inventory_rel} --policy {policy_rel}
It must print DRAFT_VALIDATION_OK. If it reports a mismatch, repair the draft and
run the same command again.

This turn is ONLY the configure-models advice stage. Treat {inventory_rel} as
read-only, write only {model_advice_rel}, do not write {model_config_rel}, start
detector generation, or document results. Return immediately when the advice is
complete.
""".strip(),
        },
        {
            "name": "generate-detector",
            "expects_file": feature_impl_rel,
            "instruction": f"""
Use the discovery-detector-builder subagent to write ONLY the task-specific
feature implementation to {feature_impl_rel}. Read {inventory_rel} and, for each
included feature, only its recorded feature_source_path after verifying the
recorded SHA-256. Never fall back between rerun and fixed sources.
{candidate_policy_generation_note}

The driver owns the reviewed runtime template plus target adapter and will assemble the unchanged
public {detector_rel}; do not write that file. Your fragment must define:
- FEATURE_REGISTRY in inventory order;
- ANALYSIS_TIMING_GROUPS, a literal list mapping stable timing-group keys to
  positive hypothesis ids, actual traversal phase, and owned feature names;
- COMPUTATION_PLAN, the literal cost-discipline plan described below;
{extraction_contract}

PLAN BEFORE IMPLEMENTING. Before writing any feature code, enumerate the
computational primitives the inventoried features require (for example "local
neighborhood around an endpoint", "shortest-path distances within a bounded
region", "per-component aggregate", "global clustering of all candidates",
"read of a precomputed sample field"), classify each primitive's cost growth,
and identify every case where two or more hypotheses need the same
(primitive, parameters) result so they can share one implementation. Record
the outcome as a top-level literal COMPUTATION_PLAN: a non-empty list of
objects with exactly the keys primitive, cost_class, bound, amortization, and
consumers. `primitive` is a unique short name; `consumers` lists the
ANALYSIS_TIMING_GROUPS keys that use it (every timing group must be consumed by
at least one primitive). `cost_class` uses exactly this closed vocabulary:
- constant: reads precomputed row/sample fields or does O(1) arithmetic;
- bounded_local: fixed-k lookups or walks with a hard step limit;
- density_scaled: work grows with local point/node density (for example a
  radius query feeding a subgraph or graph algorithm);
- component_scaled: work grows with connected-component size;
- global_scan: touches the full node/candidate arrays (full-array conditional
  filtering, sorting, clustering, or reductions such as min/max/sum/
  flatnonzero) — even when the array itself was materialized by a pre-pass.
When unsure between two classes, declare the more expensive one; constraints
only tighten with the class, so under-classifying removes protection.
`bound` and `amortization` are honest declarations the driver checks for
substance:
- density_scaled and component_scaled primitives must declare a concrete bound
  (a traversal/shortest-path cutoff, a neighborhood node cap, or per-component
  memoization when the source semantics require the whole component);
- density_scaled primitives must declare their amortization (the memoization
  key covering all result-determining parameters);
- global_scan primitives must declare a one-time pre-pass in `amortization`;
  full-array scans are forbidden on the per-row path — build the index or
  clustering once at startup and do per-row dict/array lookups only.
These cost-discipline invariants bind the implementation, not just the plan:
- compute-once: when several hypotheses need an identical (primitive,
  parameters) result for the same row, route them through one memoized shared
  helper; never re-execute semantically identical expensive work per consumer.
  Use the runtime-provided `_memoized(cache, key, compute)` for every declared
  memoization: keep one dict per primitive, key it by the declared
  amortization key, and pass the expensive work as the compute callable. The
  assembler mechanically rejects a plan that declares density_scaled or
  memoized amortization when the implementation never calls `_memoized`;
- boundedness: every graph traversal or shortest-path call carries an explicit
  cutoff or node cap matching the declared bound;
- pre-pass: any computation touching full-length arrays or the whole candidate
  set runs once before the row loop and is reused via lookup. The pre-pass must
  materialize the DERIVED result each consumer needs (a scalar, a
  per-component aggregate, an index), not merely the raw array: a per-row
  `np.min(xyz[:, 2])` over a pre-pass-built array is still a per-row global
  scan and belongs in the pre-pass as a stored scalar.
The assembler enforces this last boundary independently. It follows local
helper and bounded-worker callbacks from timing-instrumented row loops and
rejects whole-universe iteration/reduction/index construction, unbounded graph
search, and imports reachable from that path. Wrapping such setup in
`_memoized(cache, row_or_component_key, compute)` does not make it legal. A
constant-key global cache may be read from row work only after the same helper
has been explicitly warmed before row processing starts.
The optimizations reorganize computation only; they must not change feature
numeric semantics. If a cutoff would truncate a quantity the selected source
defines over a whole component, keep the source semantics, classify the
primitive component_scaled, and amortize it with per-component memoization
instead of silently truncating. The deterministic assembler rejects a missing
or non-literal COMPUTATION_PLAN, unknown cost classes, missing bounds or
amortization for the classes above, consumers naming unknown timing-group
keys, and timing groups no primitive covers.

Every name your fragment references must be defined by the fragment itself,
the runtime template, or Python builtins. When you rename or restructure a
helper, update every call site: the assembler statically rejects the assembled
detector if any scope references an undefined name, because such leftovers are
latent NameErrors on code paths the smoke test never executes.

On every control-flow path, `extract_features` must return exactly the
three-item tuple `(row_records, labels, accumulator)`. Never return a DataFrame
or call `accumulator.to_frame()` there; the reviewed runtime owns conversion to
the feature frame. The deterministic assembler rejects any other return shape.

Copy feature math, constants, reductions, and measurable conditions from the
inventoried sources. Share traversal passes, keep intermediate quantities out of
FEATURE_REGISTRY, and emit NaN plus <feature>_is_defined for undefined values.
Never infer missingness from historical numeric sentinels. Preserve the complete
runtime-provided target row universe, leaving features undefined when their
required fragment structure is absent, and remove install/sandbox scaffolding.
Preserve whether graph algorithms are weighted or unweighted; do not invent
distance calculations or edge weights that the selected source does not use.
An induced/copied `networkx.Graph` does not retain custom SkeletonGraph
attributes such as `node_xyz`, `node_radius`, or component mappings. Helpers
called with a plain subgraph must use topology only, receive the original
fragment graph explicitly, or receive the required arrays explicitly.

Make every extraction phase observable, profileable, and removable at the finest
honest computation boundary. `ANALYSIS_TIMING_GROUPS` must cover every
FEATURE_REGISTRY name exactly once and every hypothesis id must belong to exactly
one group. FEATURE_REGISTRY owns only public feature names and inventory order;
do not duplicate phase metadata there. Treat FEATURE_REGISTRY as metadata only:
initialize accumulators with the runtime-derived flat string FEATURE_NAMES,
never with FEATURE_REGISTRY itself. ANALYSIS_TIMING_GROUPS is the sole owner
of actual execution phase and selection-unit membership. Treat each group as a
final-run selection unit. A group may own multiple
hypotheses/features only when their computation is genuinely inseparable; record
that shared group honestly instead of dividing or double-counting its time. Call the runtime-provided
timing recorder around every analysis group in edge, junction, x-crossing, chain,
component, and segment passes, and record wall time for every whole pass. Give
eligible calls stable segment/component context and node/edge counts when
available. The deterministic assembler rejects missing, duplicate, or unknown
timing coverage; the runtime owns the bounded atomic timing artifact.

Implement `_analysis_enabled(key)` from `enabled_analysis_keys`. With the default
None, execute every group exactly as before. With an explicit set, do not execute
the graph algorithms, array construction, clustering, traversal, or reductions
owned by disabled groups; leave their public columns present as NaN with
is_defined=False. Avoid expensive common preparation when none of its consumer
groups is enabled. Shared groups are enabled or disabled as one unit. Selection
must not change the target row universe, feature-column order, labels, return
types, or all-enabled results.

Implement `profile_segment_limit` as a profiling-only deterministic sample of at
most that many component-bearing segments. Restrict every expensive edge, node,
junction, chain, component, and segment computation to that sample; do not merely
stop the timer while continuing full-data work. `None` must preserve the complete
row universe and exact normal-run behavior. The runtime passes a default limit of
3 for `--measuretime` and records the sampled ids and full-data counts.

{measuretime_scope}

Choose image-patch concurrency mechanically from ANALYSIS_TIMING_GROUPS after
the inventory has been implemented. Count included hypothesis ids (not feature
columns): an id is image-derived when its actual phase reads the image and must
use the exact phase name `image_patch_pass`. If image-derived ids are a strict
majority of all included hypothesis ids, make the candidate/segment image pass
thread-aware through `image_workers`:
- `image_workers=1` is the exact serial path; `0` calls the runtime-provided
  `_resolve_image_worker_count`, which caps automatic parallelism at 8; `-1`
  uses every CPU available to the process, honoring `os.sched_getaffinity(0)`
  (for example a Slurm cpuset) with `os.cpu_count()` as a fallback;
- use the runtime-provided `_bounded_thread_map` so at most 2x workers futures
  are queued and results are consumed in completion order, so a slow earlier
  candidate cannot block replenishing the pool. Store results by stable row id
  so the complete row/column/definedness output remains identical to the serial
  path;
- assign each candidate row to exactly one worker and do not read accumulator
  arrays until the bounded thread map is fully drained. Write that exclusively
  owned row directly, or return row-local updates for parent-thread commit;
  never acquire a shared lock for every feature write and never let two workers
  own the same candidate row;
- share only read-only graph/image state. TensorStore patch reads may overlap,
  but patch caches remain candidate-local and bounded. Initialize `patch_cache = {{}}`
  inside the per-row worker, never in `extract_features` scope, and do not retain
  image arrays after that row returns;
- force one image worker whenever `timing is not None`, because
  AnalysisTimingRecorder intentionally attributes one active hypothesis at a
  time; `--measuretime` must therefore remain comparable and deterministic;
- print the resolved worker count in the image-pass start record.
If image-derived ids are half or fewer—including graph-only runs and runs with
only rare image access—keep the existing serial traversal. Still accept the
`image_workers=1` argument for the runtime contract, but do not create a thread
pool. Do not parallelize graph traversal merely because worker helpers exist.

When `verbose=True`, additionally print flushed, machine-searchable start/done
records for long segment/component work so the last durable line identifies the
active feature if a run hangs or is killed. Row-loop progress lines must use a
FIXED absolute cadence of every 100 rows (plus the first row), never a
percentage of the total: on a ~500k-row universe a percentage cadence goes
silent for tens of minutes, which is indistinguishable from a hang. From the
second progress line on, each line must also report monotonic elapsed seconds
since the row loop started, the observed rows/second, and the estimated
minutes remaining computed from that rate, so the operator can project total
wall-clock from any single log line. Every per-group summary line and
per-phase done line must also include that group's or phase's monotonic
elapsed seconds, so hot groups are visible in a plain run's log without
--measuretime.
Instrumentation must use a monotonic
clock and must not change feature math, traversal order, random state,
definedness, rows, or returned values when all groups are enabled. Timing writes
a ranked cost report and editable selection template; it does not make the final
decision automatically. Combine removable cost with semantics-preserving
optimization opportunities, coverage, redundancy, nested-OOF value, and held-out
behavior when choosing exclusions.

Do not define validate_model_config, validate_analysis_timing_groups,
load_hypothesis_selection, AnalysisTimingRecorder,
write_hypothesis_cost_artifacts, build_estimator, run_nested_selection,
fit_final_winner, run_smoke_test, build_arg_parser, run_measuretime,
run_detector, main, a
__main__ block, output writers, plots, logging infrastructure, models, or CLI
parsing. The required extraction progress may print through the runtime's tee but
must not create or manage log files. Do not edit {inventory_rel},
{model_config_rel}, {candidate_policy_rel}, {detector_rel}, README.md, or
{run_commands_rel}. The
driver will AST-validate the fragment, inject it into the reviewed template, and
independently execute the assembled detector's no-data contract checks.

Before returning, run this narrow read-only self-check from the project root:
python -m agentic.detector_build.draft_validation feature --draft \
  {feature_impl_rel} --template {runtime_template_rel} --target {target.value}{feature_policy_validation_arg}
It must print DRAFT_VALIDATION_OK. If it reports a mismatch, repair the fragment
and run the same command again.

This turn is ONLY the feature-implementation stage. Write only
{feature_impl_rel} and return immediately after it parses.
""".strip(),
        },
        {
            "name": "verify-and-document",
            "expects_file": readme_rel,
            "instruction": f"""
Use the discovery-detector-builder subagent to review {detector_rel} and
{run_commands_rel} read-only and write {readme_rel}. The driver has already
written its factual provenance, feature, model, command, and output skeleton
there. Preserve the
`BEGIN/END DRIVER-GENERATED PROVENANCE` block and add semantic interpretation,
source/correction rationale, caveats, and fixes outside it. No real dataset has
been run: do not invent a CSV, winning model, metrics or class counts. Obtain
semantic context from {summary_rel}, {inventory_rel} and {model_config_rel}, plus
{source_note}.
{candidate_policy_verification}

VERIFY READ-ONLY BEFORE DOCUMENTING
1. Every inventory feature has one explicit feature_source_path whose current
   SHA-256 matches the inventory; the rerun-vs-fixed choice follows the recorded
   correction_scope and source_reason. Every included feature is genuinely
   computed once, keeps the selected source's constants,
   reaches the frame and has an extraction-time binary is_defined flag. Undefined
   values are NaN; valid 0/1/180 values are not missing. --exclude-empty uses
   flags rather than sentinels.
   Any plain induced/copied networkx graph is treated as topology-only: code
   must not access SkeletonGraph-only arrays through it, and weighted versus
   unweighted graph algorithms must match the selected source exactly.
   With verbose extraction, component traversal has flushed segment/component
   start/done progress (ordinals, ids, node/edge counts, elapsed time), flushed
   start/done records around every component-pass feature or named shared group,
   and a final per-feature cumulative timing/call-count summary. A costly feature
   emits its start record before doing work, and timing instrumentation does not
   change data traversal, random state, feature semantics, or return values.
   ANALYSIS_TIMING_GROUPS covers every public feature exactly once, assigns every
   hypothesis id to exactly one selectable computation unit, and maps it to its
   actual pass. Shared groups are genuinely computationally inseparable. All
   extraction passes call the runtime timing recorder.
   COMPUTATION_PLAN matches the actual code: each primitive declared shared by
   multiple consumers has exactly one implementation and is memoized with the
   declared key through the runtime `_memoized` helper (not a private cache
   whose key omits a result-determining parameter) rather than re-executed per
   consumer; each declared traversal
   cutoff or node cap actually appears in the corresponding graph calls; each
   global_scan primitive executes only in a one-time pre-pass, never inside the
   per-row loop; and no per-row code performs full-array conditional scans,
   full-array reductions of pre-pass-built arrays (for example a per-row
   min/max over all node coordinates), or unbounded shortest-path searches
   that the plan does not declare. These row-path violations are build-blocking
   and must be repaired before assembly; finding one in an assembled detector
   is a deterministic-validator escape to document, not an accepted performance
   caveat. Post-extraction diagnostics and figures
   must also stay bounded on the full universe: any all-rows pairwise or
   rank statistic must be row-subsampled to a fixed cap or be O(n log n)
   (for example, scipy spearmanr with nan_policy="omit" on NaN-bearing data
   silently routes to a masked-array pairwise path that takes hours at 500k
   rows and must not appear; use pandas .corr(method="spearman") on a
   capped row sample instead). A mismatch
   between plan and code is a defect to report, not to silently fix. If `image_patch_pass`
   owns a strict majority of included hypothesis ids, normal extraction uses
   the runtime's bounded completion-order thread map, honors `--n-jobs` (-1 =
   all available CPUs, 0 = auto capped at 8, 1 = serial), assigns each row to
   one worker without a shared per-feature write lock, preserves stable output
   placement by row id, and logs the worker count;
   timing mode forces the identical serial path. If image ids are half or fewer,
   extraction remains serial and does not create a thread pool.
   `--measuretime` profiles
   only its small deterministic segment sample and writes the
   schema-v2 timing JSON, ranked hypothesis cost report, and editable schema-v1
   selection template, then exits without audit, CV, fitting, held-out scoring,
   CSV, joblib, model-selection JSON, or figures. Per-hypothesis rows distinguish
   exclusive cost from shared-unit cost and never claim that removing one member
   saves a shared traversal. Sample-observed seconds are not a full-run duration
   or projection.
   `extract_features(..., enabled_analysis_keys=None)` runs all groups and matches
   unfiltered extraction. An explicit enabled set skips disabled computation but
   keeps its fixed columns undefined and preserves rows, labels, column order, and
   return types. Both `--hypothesis-selection` and direct
   `--exclude-hypotheses ID [ID ...]` reject unknown/duplicate ids, an empty
   final selection, and partial shared groups; the JSON form additionally rejects
   malformed reasons and inconsistent
   measuretime provenance. When a generated selection supplies a source timing
   path/hash, require a complete schema-v2 measuretime artifact produced by the
   exact detector and current inventory. Train and held-out use the same resolved
   selection, whose path/hash, timing provenance, reasons, and enabled/excluded
   membership are saved in model JSON and joblib. Direct CLI exclusions are also
   saved there with `selection_source=manual_cli`. Timing does not choose exclusions automatically.
2. Statistical imputation, scaling and spline fitting live inside candidate
   pipelines. Remove any whole-dataset fill_mean or preprocessing before CV.
   Families declared native-NaN receive NaNs directly; every other family uses
   fold-local imputation.
3. --model-config defaults beside the script and is revalidated before data
   loading. The registry contains exactly its validated allowlisted candidates,
   including every policy-required baseline and at most
   {MAX_OPTIONAL_MODELS} optional families.
   Configuration cannot supply imports/classes/code. Names, implementations,
   grids, dependency declarations and fixed complexity order agree everywhere.
4. Outer validation labels affect metrics only. Inner folds tune and select. The
   one-standard-error rule is correct. {oof_verification}
   --test-pkl cannot influence the final full-data winner.
5. The policy-selected `{PRIMARY_METRIC}` is primary. Queue and threshold sweep
   use the final winner's nested OOF,
   never in-sample scores. Held-out prediction reuses the fitted winner and cannot
   mutate preprocessing.
6. The model-selection JSON explains why the model won; joblib carries pipeline and
   ordered schema; CSV columns are unambiguous. Nonlinear models do not present
   impurity importance as validation importance.
7. There is no install command, sys.path edit or sys.modules deletion. Each pkl
   loads once, caches release sequentially, Agg precedes pyplot, logging is
   durable, and the required `--synthetic-smoke-test` remains a no-data/no-output
   mode that prints `{SMOKE_SUCCESS_MARKER}` only after its assertions pass. The
   driver will independently run parsing, --help, and that smoke test after this
   turn; do not claim success in lieu of executable checks.

Then document every excluded hypothesis and the per-feature rerun/fixed source
choice, plus provenance, feature mapping and defined conditions; why each model
is included, how to safely edit {MODEL_CONFIG_NAME}, and why arbitrary families
require an explicit code review rather than a JSON class path; nested CV and the
one-standard-error rule in plain language; the
difference among family OOF, selector OOF, winner OOF, in-sample and held-out
scores; output schemas; a concise reference to the compute-node commands in
{run_commands_rel} without duplicating them; optional dependency behavior; and a
reading order starting with coverage and AP, then review-budget precision/recall
and cross-brain transfer, with ROC-AUC last. State that weighted classifier scores
are not automatically calibrated probabilities. Preserve caveats about sparse
positive labels, correlated features, discovery-stage feature-selection bias,
uneven coverage, and {verification_scope}. Report
defects found and anything requiring a real compute-node run.

This turn is ONLY the verify-and-document stage. Treat {inventory_rel},
{model_config_rel}, the driver-assembled {detector_rel}, and the instance-bound
{run_commands_rel} as immutable. Report any defect instead of editing it, and write
{readme_rel}; do not regenerate upstream artifacts or run against a pkl.
""".strip(),
        },
    ]
    if spec.target is DetectorTarget.SPLIT:
        if candidate_review_rel is None:
            raise ValueError(
                "Split detector builds require candidate_review_rel for the "
                "candidate-policy advice stage."
            )
        policy_step = {
            "name": "select-candidate-policy",
            "expects_file": candidate_advice_rel,
            "instruction": f"""
Use the discovery-detector-builder subagent to read the completed split
candidate-pool review {candidate_review_rel}. Also read its sibling
ai_review_evidence.json and tables/cross_dataset_aggregate.csv so your
explanation is tied to the structured sweep evidence.

The driver-owned objective is fixed and must not be changed:
- objective: {CANDIDATE_POLICY_OBJECTIVE_NAME}
- minimum worst-brain candidate recall: {candidate_policy_minimum_recall:.12g}
- tie-breaker: first total_candidates ascending, then config_id ascending

Identify the configuration that satisfies the recall constraint with the fewest
total candidates across the evaluated datasets. Explain the selected operating
point, its recall/workload trade-off, the evidence scope, and the fact that
candidate recall is a pre-scoring ceiling. Do not infer a different objective,
average across away the worst brain, or treat candidate prevalence as classifier
precision.

Write ONLY the internal advice draft {candidate_advice_rel} with exactly this
JSON shape:
{{
  "schema_version": 1,
  "objective": {{
    "name": "{CANDIDATE_POLICY_OBJECTIVE_NAME}",
    "minimum_worst_brain_recall": {candidate_policy_minimum_recall:.12g},
    "tie_breaker": ["total_candidates", "config_id"]
  }},
  "selected_config_id": "<mode|r=...|k=...>",
  "explanation": "<non-empty evidence-grounded explanation>"
}}

The driver, not the agent, is authoritative: it will independently hash-check
the review bundle, scan every row of cross_dataset_aggregate.csv, recompute the
minimum-candidate eligible configuration, reject disagreement, and write the
public {candidate_policy_rel}. Split detector assembly requires this exact
artifact, embeds its SHA-256 and generator fields in the runtime, and refuses to
fall back to a default candidate policy.

Before returning, run this narrow read-only self-check from the project root:
python -m agentic.detector_build.draft_validation candidate-policy --draft \
  {candidate_advice_rel} --review {candidate_review_rel} --run-json {run_rel} \
  --minimum-recall {candidate_policy_minimum_recall:.12g}
It must print DRAFT_VALIDATION_OK. Repair the draft if needed.

This turn is ONLY the select-candidate-policy advice stage. Treat the review,
evidence, aggregate table, manifest, AutoDiscovery inputs, policies, templates,
and existing outputs as read-only. Write only {candidate_advice_rel}; do not
write {candidate_policy_rel}, feature inventory, model configuration, detector,
README, or helper files. Do not load a pkl or run feature scoring.
""".strip(),
        }
        steps.insert(0, policy_step)
    return steps


def build_options():
    """Configure the SDK session for this project.

    ``setting_sources=["project"]`` is what makes the SDK auto-discover the
    filesystem subagents in ``.claude/agents/`` (incl. discovery-detector-builder)
    and project settings relative to ``cwd``.
    """
    return load_claude_sdk().agent_options(
        cwd=str(PROJECT_ROOT),
        setting_sources=["project"],
        # The orchestrator delegates to the builder subagent (Task), which needs
        # filesystem tools to read the report plus rerun/fixed sources and write
        # the detector.
        allowed_tools=["Task", "Bash", "Read", "Write", "Edit", "Glob"],
        permission_mode="bypassPermissions",
        # Defaults remain pinned for reproducibility; task-specific environment
        # variables permit controlled upgrades without editing workflow logic.
        model=AGENT_MODEL,
        # CLAUDE_EFFORT is exported to hooks as status; it is not an input that
        # configures reasoning. Use the SDK's native option.
        effort=AGENT_EFFORT,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent step / compute step
# ─────────────────────────────────────────────────────────────────────────────

async def _run_step_inner(
    client: object,
    step: dict,
    verbose: bool,
    costs: WorkflowCostSummary,
) -> str:
    """Send one workflow step to the session and return its final text."""
    sdk = load_claude_sdk()
    costs.start_turn()
    await client.query(step["instruction"])

    chunks: list[str] = []
    n_tools = 0
    async for message in client.receive_response():
        if isinstance(message, sdk.assistant_message):
            for block in message.content:
                if isinstance(block, sdk.text_block):
                    chunks.append(block.text)
                    if verbose:
                        print(block.text, end="", flush=True)
                elif sdk.tool_use_block and isinstance(block, sdk.tool_use_block):
                    n_tools += 1
                    log(f"  → {describe_tool(block)}")
        elif isinstance(message, sdk.result_message):
            if verbose:
                print()  # newline after the streamed text
            cost = getattr(message, "total_cost_usd", None)
            delta = costs.add(cost, getattr(message, "session_id", None))
            dur_ms = getattr(message, "duration_ms", None)
            parts = [f"{n_tools} tool call(s)"]
            if dur_ms is not None:
                parts.append(f"{dur_ms / 1000:.0f}s")
            if delta is not None:
                parts.append(
                    f"${delta:.4f} this turn"
                    f" (${float(cost):.4f} session cumulative)"
                )
            log(f"  step turn finished — {', '.join(parts)}")
    return "".join(chunks)


async def run_step(
    client: object,
    step: dict,
    verbose: bool,
    costs: WorkflowCostSummary,
) -> str:
    """``_run_step_inner`` with a hard timeout, so a hung step fails loudly."""
    try:
        return await asyncio.wait_for(
            _run_step_inner(client, step, verbose, costs),
            timeout=AGENT_STEP_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise SystemExit(
            f"Agent step '{step['name']}' timed out after {AGENT_STEP_TIMEOUT_S}s."
        )


async def finalize_step_with_repairs(
    client: object,
    step: dict,
    finalize: Callable[[], None],
    guard: Callable[[], None],
    verbose: bool,
    costs: WorkflowCostSummary,
) -> None:
    """Validate/compile one draft, with bounded repairs in the same session."""
    for attempt in range(MAX_AGENT_REPAIR_ATTEMPTS + 1):
        try:
            validate_step_output(step)
        except SystemExit as exc:
            diagnostic = str(exc).strip() or repr(exc)
        else:
            # An out-of-scope write is a safety violation, not an artifact
            # mismatch that the agent should attempt to repair.
            guard()
            try:
                finalize()
                return
            except SystemExit as exc:
                diagnostic = str(exc).strip() or repr(exc)

        if attempt >= MAX_AGENT_REPAIR_ATTEMPTS:
            raise SystemExit(diagnostic)
        repair_number = attempt + 1
        log(
            f"  Driver rejected '{step['name']}' output; requesting repair "
            f"{repair_number}/{MAX_AGENT_REPAIR_ATTEMPTS}: {diagnostic}"
        )
        repair_step = {
            "name": f"{step['name']}-repair-{repair_number}",
            "expects_file": step.get("expects_file"),
            "instruction": f"""
Continue the same detector-build task and repair ONLY
{step.get('expects_file')}. The driver rejected the current artifact with this
exact diagnostic:

{diagnostic}

Understand the underlying contract and make the smallest semantically correct
change. Do not modify any other artifact, source, policy, template, README, or
input. Re-run the narrow DRAFT_VALIDATION self-check from the original task when
one was provided. Return immediately after the repaired artifact passes it; do
not load a pkl or invent results.
""".strip(),
        }
        repair_text = await run_step(client, repair_step, verbose, costs)
        if not verbose:
            print(repair_text.strip())


# Files this workflow writes itself. Deleting one of these costs nothing, because
# the run about to start puts it back.
_REGENERATED = frozenset((
    *BUILD_ARTIFACT_NAMES,
    *build_artifact_names(DetectorTarget.SPLIT),
    CANDIDATE_POLICY_ADVICE_DRAFT_NAME,
    FEATURE_SEMANTICS_DRAFT_NAME,
    MODEL_ADVICE_DRAFT_NAME,
    FEATURE_IMPLEMENTATION_DRAFT_NAME,
))


def _survey_out_dir(out_dir: Path) -> dict:
    """Group what is already in ``out_dir`` by what could bring each file back.

    The three groups carry very different stakes, and a delete prompt that lumped
    them together would be useless: regenerated files are free to lose, run outputs
    cost a compute-node job, and anything else may be unrecoverable. ``__pycache__``
    is ignored — it is noise, and Python rebuilds it.
    """
    groups = {"regenerated": [], "rerunnable": [], "irreplaceable": []}
    for path in sorted(out_dir.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(out_dir).as_posix()
        if "__pycache__" in rel:
            continue
        size = path.stat().st_size
        name = path.name
        if rel in _REGENERATED:
            groups["regenerated"].append((rel, size))
        elif rel.startswith("figures/") or (
                name.startswith(("merge_detector_", "split_detector_"))
                and (name.endswith(".csv") or name.endswith(".log.txt")
                     or name.endswith(".joblib"))) or (
                name.startswith("model_selection_") and name.endswith(".json")) or (
                name.startswith("analysis_timing_") and name.endswith(".json")) or (
                name.startswith("hypothesis_cost_report_") and name.endswith(".md")) or (
                name.startswith("hypothesis_selection_template_")
                and name.endswith(".json")) or (
                name.startswith("measuretime_") and name.endswith(".log.txt")):
            groups["rerunnable"].append((rel, size))
        else:
            groups["irreplaceable"].append((rel, size))
    return groups


def _git_dirty(out_dir: Path) -> list[str]:
    """``git status --porcelain`` lines under ``out_dir``, or [] if git can't say.

    Worth asking before deleting: a tracked-and-modified file can be restored from
    git afterwards, an untracked one cannot. That distinction belongs in front of
    the human, not in a footnote.
    """
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain", "--", str(out_dir)],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if res.returncode != 0:
        return []
    return [ln for ln in res.stdout.splitlines() if ln.strip()]


def confirm_and_clean(out_dir: Path, assume_yes: bool, keep_existing: bool):
    """Delete a pre-existing output folder, with a human in the loop.

    Regenerating into a folder that already holds results is how a NEW detector
    script ends up sitting beside a CSV, a log and figures produced by the OLD one,
    with nothing on disk marking the mismatch. Deleting first makes that
    impossible. But the folder can also hold work this workflow cannot recreate, so
    the deletion is always shown in full and confirmed before it happens.

    Returns a one-line note to log once the driver's tee is up (this runs before
    it, deliberately: the log lives inside the folder being deleted), or None when
    there was nothing to do.
    """
    if not out_dir.exists():
        return None
    groups = _survey_out_dir(out_dir)
    total = sum(len(v) for v in groups.values())
    if total == 0:
        return None                      # empty, or only __pycache__

    nbytes = sum(size for v in groups.values() for _, size in v)
    rel_dir = _rel_to_root(out_dir)

    if keep_existing:
        return (f"--keep-existing: writing into {rel_dir}/ as it is. The {total} "
                "file(s) already there are NOT removed, so any run outputs left "
                "behind will describe the PREVIOUS detector.")

    print("\n" + "=" * 78)
    print(f"{rel_dir}/ already exists — {total} file(s), {nbytes / 1e6:.1f} MB")
    print("=" * 78)
    for key, why in (
        ("regenerated", "this workflow rewrites these anyway"),
        ("rerunnable", "recreated by running the generated detector again "
                       "(compute node, potentially hours)"),
        ("irreplaceable", "NOT produced by this workflow — nothing here "
                          "recreates them"),
    ):
        items = groups[key]
        if not items:
            continue
        print(f"\n  {key.upper()} — {why}:")
        for rel, size in items[:12]:
            print(f"    {size / 1024:>9.0f} KB  {rel}")
        if len(items) > 12:
            print(f"    ... and {len(items) - 12} more")

    dirty = _git_dirty(out_dir)
    if dirty:
        print(f"\n  git reports {len(dirty)} uncommitted path(s) here:")
        for line in dirty[:8]:
            print(f"      {line}")
        if len(dirty) > 8:
            print(f"      ... and {len(dirty) - 8} more")
        print("      '??' means UNTRACKED — git cannot restore those after deletion.")

    print(f"\nDeleting removes ALL of the above, the whole {rel_dir}/ tree.")

    if assume_yes:
        shutil.rmtree(out_dir)
        return (f"--yes: deleted {rel_dir}/ without prompting "
                f"({total} files, {nbytes / 1e6:.1f} MB).")

    if not sys.stdin.isatty():
        raise SystemExit(
            f"{rel_dir}/ already exists and there is no terminal to ask on (batch "
            "job?). Re-run with --yes to delete it, or --keep-existing to write "
            "into it as it stands."
        )

    print("\nType 'delete' to remove it and continue, anything else to abort: ",
          end="", flush=True)
    answer = sys.stdin.readline().strip()
    if answer != "delete":
        raise SystemExit(
            f"Aborted — {rel_dir}/ left untouched (answer was {answer!r}). Use "
            "--keep-existing to write into it without deleting."
        )
    shutil.rmtree(out_dir)
    return (f"deleted {rel_dir}/ after confirmation "
            f"({total} files, {nbytes / 1e6:.1f} MB).")


def validate_step_output(step: dict) -> None:
    """Abort if a step's required artifact is missing or empty."""
    rel = step.get("expects_file")
    if not rel:
        return
    path = PROJECT_ROOT / rel
    if not path.is_file():
        raise SystemExit(f"Step '{step['name']}' did not create required {rel}.")
    if path.stat().st_size == 0:
        raise SystemExit(f"Step '{step['name']}' left required {rel} empty.")
    log(f"  OK: {rel} ({path.stat().st_size} bytes)")


def _load_json_object(path: Path, label: str) -> dict:
    """Compatibility wrapper around deterministic artifact loading."""
    return _resolved_load_json_object(path, label, PROJECT_ROOT)


def _integer_ids(value, label: str, path: Path) -> list[int]:
    """Compatibility wrapper around ordered ID validation."""
    return _resolved_integer_ids(value, label, path, PROJECT_ROOT)


def _manifest_ids(manifest_path: Path) -> list[int]:
    """Compatibility wrapper returning MANIFEST IDs in recorded order."""
    return _resolved_manifest_ids(manifest_path, PROJECT_ROOT)


def _report_evidence_by_id(summary_path: Path) -> dict[int, dict[str, str | None]]:
    """Compatibility wrapper joining report evidence by explicit ID."""
    return _resolved_report_evidence_by_id(summary_path, PROJECT_ROOT)


def _corrected_status_by_id(
    corrected_path: Path,
    rerun_dir: Path,
    fixed_dir: Path | None,
) -> dict[int, str]:
    """Compatibility wrapper for corrected execution-status evidence."""
    return _resolved_corrected_status_by_id(
        corrected_path, rerun_dir, fixed_dir, PROJECT_ROOT
    )


def _sha256(path: Path) -> str:
    """Compatibility wrapper for streaming SHA-256."""
    return _resolved_sha256(path)


def _valid_model_grid_value(rule: str, value) -> bool:
    """Compatibility wrapper for closed policy rules."""
    return _resolved_valid_grid_value(rule, value)


def _validate_model_contract_constants() -> None:
    """Catch drift among the code-owned family metadata tables."""
    if set(MODEL_PARAMETER_ALLOWLIST) != ALLOWED_MODEL_FAMILIES:
        raise RuntimeError("MODEL_PARAMETER_ALLOWLIST is out of sync with model families.")
    if set(MODEL_NATIVE_NAN) != ALLOWED_MODEL_FAMILIES:
        raise RuntimeError("MODEL_NATIVE_NAN is out of sync with model families.")
    # A required baseline (e.g. xgboost) may declare an extra package; the
    # runtime then fails hard instead of silently skipping it when missing.
    if not set(MODEL_REQUIRED_PACKAGE) <= ALLOWED_MODEL_FAMILIES:
        raise RuntimeError("MODEL_REQUIRED_PACKAGE names unknown model families.")
    if set(MODEL_SIMPLICITY_ORDER) != ALLOWED_MODEL_FAMILIES:
        raise RuntimeError("MODEL_SIMPLICITY_ORDER is out of sync with model families.")
    if set(REQUIRED_GRID_PARAMETERS) != ALLOWED_MODEL_FAMILIES:
        raise RuntimeError("REQUIRED_GRID_PARAMETERS is out of sync with model families.")
    for family, required in REQUIRED_GRID_PARAMETERS.items():
        if not required <= MODEL_PARAMETER_ALLOWLIST[family]:
            raise RuntimeError(
                f"REQUIRED_GRID_PARAMETERS for {family} names parameters "
                "outside the policy allowlist."
            )


def validate_model_config(
    config_path: Path,
    inventory_path: Path,
    policy_path: Path = MODEL_POLICY_PATH,
) -> None:
    """Reject stale, executable, or oversized model configuration."""
    _validate_model_contract_constants()
    candidate_count, optional_count = _resolved_validate_model_config(
        config_path,
        inventory_path,
        policy_path,
        MODEL_POLICY_CONTRACT,
        MODEL_CONFIG_SCHEMA_VERSION,
        PROJECT_ROOT,
    )
    log(
        f"  OK: model configuration validated ({candidate_count} candidates, "
        f"{optional_count} optional)."
    )


def validate_inventory(
    inventory_path: Path,
    selected_ids: list[int],
    selection_rel: str | None,
    rerun_dir: Path,
    fixed_dir: Path | None,
    summary_path: Path | None = None,
    corrected_results_path: Path | None = None,
    reconcile_unbacked_verdicts: bool = False,
    split_applicability_path: Path | None = None,
    run_json: Path | None = None,
) -> None:
    """Enforce inventory semantics and per-hypothesis source provenance.

    ``reconcile_unbacked_verdicts`` mirrors the input gate's flag: report
    post-correction verdicts with no USABLE corrected measurement are expected
    to have been transcribed as null by the compiler, so the report-match check
    compares against the same normalized value instead of the raw token.
    """
    inventory = _load_json_object(inventory_path, "feature inventory")
    split_inventory = split_applicability_path is not None
    if split_inventory != (run_json is not None):
        raise SystemExit(
            "Split inventory validation requires both applicability path and run JSON."
        )
    top_level_fields = {
        "schema_version", "selection_manifest", "selected_ids", "hypotheses",
    }
    applicability = None
    if split_inventory:
        top_level_fields.update({
            "split_feature_applicability",
            "split_feature_applicability_sha256",
        })
    if set(inventory) != top_level_fields:
        raise SystemExit(
            "Feature inventory must contain exactly: "
            + ", ".join(sorted(top_level_fields))
        )
    expected_schema = 3 if split_inventory else 2
    if inventory.get("schema_version") != expected_schema:
        raise SystemExit(
            f"Feature inventory must set schema_version to {expected_schema}."
        )
    if split_inventory:
        assert split_applicability_path is not None and run_json is not None
        expected_rel = _rel_to_root(split_applicability_path)
        if inventory.get("split_feature_applicability") != expected_rel:
            raise SystemExit(
                "Feature inventory points to the wrong split applicability artifact."
            )
        if inventory.get("split_feature_applicability_sha256") != _sha256(
            split_applicability_path
        ):
            raise SystemExit("Feature inventory split applicability SHA-256 is stale.")
        applicability = load_applicability(
            split_applicability_path,
            run_json=run_json,
            selected_ids=selected_ids,
            rerun_dir=rerun_dir,
            fixed_dir=fixed_dir,
            project_root=PROJECT_ROOT,
        )
    if inventory.get("selection_manifest") != selection_rel:
        actual_selection = inventory.get("selection_manifest")
        raise SystemExit(
            "Feature inventory selection_manifest must be exactly the workflow's "
            f"path string (or null): expected {selection_rel!r}, got "
            f"{actual_selection!r} ({type(actual_selection).__name__})."
        )
    inventory_ids = _integer_ids(
        inventory.get("selected_ids"), "selected_ids", inventory_path
    )
    if inventory_ids != selected_ids:
        raise SystemExit(
            f"Feature inventory selected_ids {inventory_ids} do not match the "
            f"authoritative ids {selected_ids}."
        )
    rows = inventory.get("hypotheses")
    if not isinstance(rows, list):
        raise SystemExit("Feature inventory hypotheses must be a list.")
    row_ids = [row.get("id") if isinstance(row, dict) else None for row in rows]
    if row_ids != selected_ids:
        raise SystemExit(
            "Feature inventory must contain exactly one hypothesis row per selected "
            "id, in selected_ids order."
        )

    allowed_scopes = {
        "none", "test_only", "feature_semantics_changed",
        "stale_fixed_ignored", "unclear",
    }
    allowed_reproduction_statuses = {
        "REPRODUCED", "DIVERGED", "FAILED", "UNUSABLE",
    }
    allowed_statistical_verdicts = {
        "SOUND", "WEAK", "MINOR", "MAJOR", "CRITICAL",
    }
    allowed_corrected_statuses = {None, "USABLE", "FAILED", "UNUSABLE"}
    allowed_post_correction_verdicts = {
        None, "UPHELD", "WEAKENED", "OVERTURNED",
    }
    provenance_fields = (
        "exclusion_reason", "reproduction_status", "statistical_verdict",
        "corrected_result_status", "post_correction_verdict", "correction_scope", "rerun_path",
        "rerun_sha256", "fixed_path", "fixed_sha256", "feature_source",
        "feature_source_path", "feature_source_sha256", "source_reason", "features",
    )
    row_fields = {"id", "included", *provenance_fields}
    report_evidence = (
        _report_evidence_by_id(summary_path) if summary_path is not None else None
    )
    corrected_statuses = (
        _corrected_status_by_id(corrected_results_path, rerun_dir, fixed_dir)
        if corrected_results_path is not None else None
    )
    if reconcile_unbacked_verdicts and report_evidence is not None:
        for unbacked_id, _verdict, _status in _resolved_unbacked_correction_verdicts(
            report_evidence, corrected_statuses or {}, list(report_evidence)
        ):
            report_evidence[unbacked_id]["post_correction_verdict"] = None
    feature_names: set[str] = set()
    for row in rows:
        hypothesis_id = row["id"]
        if set(row) != row_fields:
            raise SystemExit(
                f"Inventory hypothesis {hypothesis_id} must contain exactly: "
                + ", ".join(sorted(row_fields))
            )
        if not isinstance(row.get("included"), bool):
            raise SystemExit(f"Inventory hypothesis {hypothesis_id} included must be boolean.")
        if row["correction_scope"] not in allowed_scopes:
            raise SystemExit(f"Inventory hypothesis {hypothesis_id} has invalid correction_scope.")
        if not isinstance(row["source_reason"], str) or not row["source_reason"].strip():
            raise SystemExit(f"Inventory hypothesis {hypothesis_id} needs a source_reason.")

        reproduction_status = row["reproduction_status"]
        if reproduction_status not in allowed_reproduction_statuses:
            raise SystemExit(
                f"Inventory hypothesis {hypothesis_id} reproduction_status must be "
                f"one of {sorted(allowed_reproduction_statuses)}, got "
                f"{reproduction_status!r}. Match by explicit hypothesis ID, not "
                "report entry rank."
            )
        statistical_verdict = row["statistical_verdict"]
        if statistical_verdict not in allowed_statistical_verdicts:
            raise SystemExit(
                f"Inventory hypothesis {hypothesis_id} statistical_verdict must be "
                f"one of {sorted(allowed_statistical_verdicts)}, got "
                f"{statistical_verdict!r}. Match by explicit hypothesis ID, not "
                "report entry rank."
            )
        corrected_status = row["corrected_result_status"]
        if corrected_status not in allowed_corrected_statuses:
            raise SystemExit(
                f"Inventory hypothesis {hypothesis_id} corrected_result_status "
                "describes execution/measurement and must be USABLE, FAILED, "
                f"UNUSABLE, or null; got {corrected_status!r}. Put "
                "UPHELD/WEAKENED/OVERTURNED only in post_correction_verdict."
            )
        post_correction_verdict = row["post_correction_verdict"]
        if post_correction_verdict not in allowed_post_correction_verdicts:
            raise SystemExit(
                f"Inventory hypothesis {hypothesis_id} post_correction_verdict "
                "describes the scientific conclusion and must be UPHELD, WEAKENED, "
                f"OVERTURNED, or null; got {post_correction_verdict!r}."
            )
        if report_evidence is not None:
            expected = report_evidence.get(hypothesis_id)
            if expected is None:
                raise SystemExit(
                    f"Finished report has no row with explicit hypothesis ID "
                    f"{hypothesis_id}."
                )
            for field_name in (
                "reproduction_status", "statistical_verdict",
                "post_correction_verdict",
            ):
                if row[field_name] != expected[field_name]:
                    raise SystemExit(
                        f"Inventory hypothesis {hypothesis_id} {field_name} "
                        f"does not match the same-ID report row: expected "
                        f"{expected[field_name]!r}, got {row[field_name]!r}. "
                        "Do not join report display rank to hypothesis ID."
                    )
            # Blindness gate (driver-enforced, not builder judgment): a report
            # row the verifier marked GT-REFERENCING measures its feature
            # against ground-truth state, so it cannot run blind at inference —
            # including it would train on the answer key (the ascertainment
            # artifact that dominated the merge image-only-v2 build). Reports
            # written before the Deployability token carry None here and are
            # not constrained.
            if row["included"] and expected.get("deployability") == "GT-REFERENCING":
                raise SystemExit(
                    f"Inventory hypothesis {hypothesis_id} is included but the "
                    "report's verifier marked it Deployability: GT-REFERENCING "
                    "— its feature is measured against GT tracings and is not "
                    "blind-computable. Exclude it (exclusion_reason: "
                    "GT-referencing feature, not blind-computable)."
                )
        if post_correction_verdict is not None and corrected_status != "USABLE":
            raise SystemExit(
                f"Inventory hypothesis {hypothesis_id} has a post-correction "
                "verdict without a USABLE corrected measurement."
            )
        if corrected_statuses is not None:
            expected_corrected_status = corrected_statuses.get(hypothesis_id)
            if corrected_status != expected_corrected_status:
                raise SystemExit(
                    f"Inventory hypothesis {hypothesis_id} corrected_result_status "
                    f"does not match corrected results: expected "
                    f"{expected_corrected_status!r}, got {corrected_status!r}."
                )

        rerun_path = rerun_dir / f"hypo_{hypothesis_id}.py"
        if not rerun_path.is_file():
            raise SystemExit(
                f"Selected hypothesis {hypothesis_id} lacks "
                f"{_rel_to_root(rerun_path)}."
            )
        if row["rerun_path"] != _rel_to_root(rerun_path):
            raise SystemExit(f"Inventory hypothesis {hypothesis_id} has the wrong rerun_path.")
        if row["rerun_sha256"] != _sha256(rerun_path):
            raise SystemExit(f"Inventory hypothesis {hypothesis_id} rerun SHA-256 is stale.")

        fixed_path = fixed_dir / f"hypo_{hypothesis_id}.py" if fixed_dir else None
        fixed_exists = fixed_path is not None and fixed_path.is_file()
        expected_fixed_path = _rel_to_root(fixed_path) if fixed_exists else None
        expected_fixed_hash = _sha256(fixed_path) if fixed_exists else None
        if row["fixed_path"] != expected_fixed_path or row["fixed_sha256"] != expected_fixed_hash:
            raise SystemExit(f"Inventory hypothesis {hypothesis_id} fixed provenance is stale.")
        if fixed_exists and row["correction_scope"] == "none":
            raise SystemExit(
                f"Inventory hypothesis {hypothesis_id} has a fixed script but no "
                "classified correction scope."
            )
        if row["correction_scope"] in {
            "test_only", "feature_semantics_changed", "stale_fixed_ignored"
        } and not fixed_exists:
            raise SystemExit(
                f"Inventory hypothesis {hypothesis_id} correction_scope requires "
                "a same-id fixed script."
            )
        if row["correction_scope"] in {"none", "stale_fixed_ignored"} and (
            corrected_status is not None or post_correction_verdict is not None
        ):
            raise SystemExit(
                f"Inventory hypothesis {hypothesis_id} correction_scope "
                f"{row['correction_scope']} requires null correction evidence."
            )
        if row["correction_scope"] in {"test_only", "feature_semantics_changed"} \
                and (corrected_status != "USABLE" or post_correction_verdict is None):
            raise SystemExit(
                f"Inventory hypothesis {hypothesis_id} classified correction "
                "requires a USABLE measurement and a post-correction verdict."
            )

        if row["included"]:
            if row["exclusion_reason"] is not None:
                raise SystemExit(
                    f"Included hypothesis {hypothesis_id} must have null exclusion_reason."
                )
            if row["correction_scope"] == "unclear":
                raise SystemExit(f"Unclear hypothesis {hypothesis_id} cannot be included.")
            post_verdict = str(post_correction_verdict).upper()
            if "OVERTURNED" in post_verdict:
                raise SystemExit(f"Overturned hypothesis {hypothesis_id} cannot be included.")
            reproduction_status = str(reproduction_status).upper()
            if any(status in reproduction_status for status in ("FAILED", "UNUSABLE")) \
                    and corrected_status != "USABLE":
                raise SystemExit(
                    f"Failed/unusable hypothesis {hypothesis_id} needs a usable correction."
                )
            statistical_verdict = str(statistical_verdict).upper()
            if "CRITICAL" in statistical_verdict and not (
                corrected_status == "USABLE"
                and any(verdict in post_verdict for verdict in ("UPHELD", "WEAKENED"))
            ):
                raise SystemExit(
                    f"Critical hypothesis {hypothesis_id} needs a usable "
                    "upheld/weakened correction."
                )
            if row["correction_scope"] in {"test_only", "feature_semantics_changed"} \
                    and corrected_status != "USABLE":
                raise SystemExit(
                    f"Corrected hypothesis {hypothesis_id} lacks a usable corrected result."
                )
            source = row["feature_source"]
            if source not in {"rerun", "fixed"}:
                raise SystemExit(
                    f"Included hypothesis {hypothesis_id} needs rerun/fixed "
                    "feature_source."
                )
            required_source = (
                "fixed"
                if row["correction_scope"] == "feature_semantics_changed"
                else "rerun"
            )
            if source != required_source:
                raise SystemExit(
                    f"Inventory hypothesis {hypothesis_id} correction_scope "
                    f"{row['correction_scope']} requires {required_source} source."
                )
            expected_source = rerun_path if source == "rerun" else fixed_path
            if expected_source is None or not expected_source.is_file():
                raise SystemExit(
                    f"Included hypothesis {hypothesis_id} selects a missing "
                    f"{source} source."
                )
            if row["feature_source_path"] != _rel_to_root(expected_source):
                raise SystemExit(f"Included hypothesis {hypothesis_id} has the wrong source path.")
            if row["feature_source_sha256"] != _sha256(expected_source):
                raise SystemExit(f"Included hypothesis {hypothesis_id} source SHA-256 is stale.")
            if not isinstance(row["features"], list) or not row["features"]:
                raise SystemExit(f"Included hypothesis {hypothesis_id} has no feature definitions.")
            required_feature_fields = {
                "name", "quantity", "constants", "aggregation", "reduction",
                "traversal_phase", "measurable_condition",
                "historical_undefined_sentinel",
            }
            expected_requirement = None
            if applicability is not None:
                required_feature_fields.add("node_role_requirement")
                annotation = applicability.get(
                    (hypothesis_id, _sha256(expected_source))
                )
                if annotation is None:
                    raise SystemExit(
                        f"Inventory hypothesis {hypothesis_id} has no applicability "
                        "record for its selected feature source."
                    )
                expected_requirement = annotation["node_role_requirement"]
                if expected_requirement == "unclear":
                    raise SystemExit(
                        f"Inventory hypothesis {hypothesis_id} cannot include an "
                        "unclear node-role requirement."
                    )
            for feature in row["features"]:
                if not isinstance(feature, dict):
                    raise SystemExit(
                        f"Inventory hypothesis {hypothesis_id} has a non-object feature."
                    )
                if set(feature) != required_feature_fields:
                    raise SystemExit(
                        f"Inventory hypothesis {hypothesis_id} feature must contain "
                        "exactly: " + ", ".join(sorted(required_feature_fields))
                    )
                if expected_requirement is not None and feature.get(
                    "node_role_requirement"
                ) != expected_requirement:
                    raise SystemExit(
                        f"Inventory hypothesis {hypothesis_id} feature applicability "
                        "does not match its exact selected source annotation."
                    )
                name = feature["name"]
                if not isinstance(name, str) or not name.strip():
                    raise SystemExit(
                        f"Inventory hypothesis {hypothesis_id} has an empty feature name."
                    )
                if name in feature_names:
                    raise SystemExit(f"Feature inventory duplicates feature name {name!r}.")
                if name.endswith("_is_defined"):
                    raise SystemExit(
                        f"Feature name {name!r} collides with generated defined-flag columns."
                    )
                feature_names.add(name)
                for field_name in (
                    "quantity", "aggregation", "reduction", "traversal_phase",
                    "measurable_condition",
                ):
                    if not isinstance(feature[field_name], str) or not feature[field_name].strip():
                        raise SystemExit(
                            f"Inventory hypothesis {hypothesis_id} feature {name!r} "
                            f"needs a non-empty {field_name}."
                        )
                if not isinstance(feature["constants"], dict):
                    raise SystemExit(
                        f"Inventory hypothesis {hypothesis_id} feature {name!r} "
                        "constants must be an object."
                    )
        else:
            if not isinstance(row["exclusion_reason"], str) or not row["exclusion_reason"].strip():
                raise SystemExit(f"Excluded hypothesis {hypothesis_id} needs an exclusion_reason.")
            if any(row[field] is not None for field in (
                "feature_source", "feature_source_path", "feature_source_sha256"
            )):
                raise SystemExit(
                    f"Excluded hypothesis {hypothesis_id} must have null source "
                    "fields."
                )
            if row["features"] != []:
                raise SystemExit(
                    f"Excluded hypothesis {hypothesis_id} must have an empty "
                    "features list."
                )

    log(
        f"  OK: inventory v{expected_schema} validated for {len(rows)} "
        "selected hypothesis ids."
    )


def validate_detector_source(detector_path: Path) -> None:
    """Compatibility wrapper for deterministic syntax validation."""
    try:
        _validate_detector_source(detector_path)
    except SystemExit as exc:
        # The reusable verifier reports absolute paths. Preserve the driver's
        # concise project-relative diagnostic where possible.
        raise SystemExit(str(exc).replace(str(detector_path), _rel_to_root(detector_path)))
    log(f"  OK: {_rel_to_root(detector_path)} parses as Python.")


def validate_detector_executable(
    detector_path: Path,
    model_config_path: Path,
) -> None:
    """Run the generated CLI's mandatory no-data contract checks."""
    validate_detector_executable_contract(
        detector_path,
        model_config_path,
        timeout_s=DETECTOR_CHECK_TIMEOUT_S,
    )
    log(
        f"  OK: {_rel_to_root(detector_path)} --help, synthetic smoke, "
        f"invalid-config rejection, and no-output checks passed "
        f"({SMOKE_SUCCESS_MARKER})."
    )


def require_unchanged(path: Path, expected_sha256: str, owner_step: str) -> None:
    """Protect a validated upstream artifact from later agent turns."""
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise SystemExit(
            f"A later step modified validated {owner_step} artifact "
            f"{_rel_to_root(path)}. Regenerate from a clean output directory."
        )



# ─────────────────────────────────────────────────────────────────────────────
# Workflow
# ─────────────────────────────────────────────────────────────────────────────

def resolve_run_context(
    run_json: Path, *, reconcile_unbacked_verdicts: bool = False
) -> RunContext:
    """Return the typed, fully checked input contract for one build."""
    return _resolve_run_context(
        run_json, PROJECT_ROOT, PREDICTIVE_POLICY_VERSION,
        reconcile_unbacked_verdicts=reconcile_unbacked_verdicts,
    )


def resolve_inputs(
    run_json: Path, *, reconcile_unbacked_verdicts: bool = False
) -> tuple[Path, Path, Path | None, Path | None, list[int]]:
    """Backward-compatible tuple view of :func:`resolve_run_context`."""
    context = resolve_run_context(
        run_json, reconcile_unbacked_verdicts=reconcile_unbacked_verdicts
    )
    return (
        context.summary_path,
        context.rerun_dir,
        context.fixed_dir,
        context.selection_path,
        list(context.selected_ids),
    )


async def run_workflow(
    run_json: Path,
    out_dir: Path,
    verbose: bool,
    costs: WorkflowCostSummary,
    reconcile_unbacked_verdicts: bool = False,
    candidate_review_path: Path | None = None,
    candidate_policy_minimum_recall: float = (
        DEFAULT_SPLIT_MIN_WORST_BRAIN_RECALL
    ),
) -> None:
    context = resolve_run_context(
        run_json, reconcile_unbacked_verdicts=reconcile_unbacked_verdicts
    )
    if context.unbacked_verdict_ids:
        log(
            "Reconciled unbacked post-correction verdict(s) to null for "
            f"hypothesis id(s) {list(context.unbacked_verdict_ids)}; these "
            "corrections are ignored and the hypotheses judged on their "
            "pre-correction evidence."
        )
    spec = target_spec(context.target)
    expected_candidate_mcl = expected_mcl_from_run_path(run_json)
    if spec.target is DetectorTarget.SPLIT:
        if candidate_review_path is None:
            raise SystemExit(
                "Split detector builds require a candidate-pool AI_REVIEW.md."
            )
        validate_candidate_policy_sources(
            candidate_review_path,
            minimum_recall=candidate_policy_minimum_recall,
            expected_mcl=expected_candidate_mcl,
            project_root=PROJECT_ROOT,
        )
    summary_path = context.summary_path
    rerun_dir = context.rerun_dir
    fixed_dir = context.fixed_dir
    selection_path = context.selection_path
    selected_ids = list(context.selected_ids)
    corrected_results_path = context.corrected_results_path
    split_applicability_path = context.split_feature_applicability_path

    steps = build_steps(
        run_rel=_rel_to_root(run_json),
        summary_rel=_rel_to_root(summary_path),
        rerun_rel=_rel_to_root(rerun_dir),
        fixed_rel=_rel_to_root(fixed_dir) if fixed_dir is not None else None,
        selection_rel=(
            _rel_to_root(selection_path) if selection_path is not None else None
        ),
        selected_ids=selected_ids,
        out_dir_rel=_rel_to_root(out_dir),
        corrected_results_rel=(
            _rel_to_root(corrected_results_path)
            if corrected_results_path is not None else None
        ),
        target=context.target,
        candidate_review_rel=(
            _rel_to_root(candidate_review_path)
            if candidate_review_path is not None else None
        ),
        candidate_policy_minimum_recall=candidate_policy_minimum_recall,
        split_applicability_rel=(
            _rel_to_root(split_applicability_path)
            if split_applicability_path is not None else None
        ),
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    log(
        f"Generating detector code from {_rel_to_root(run_json)} → "
        f"{_rel_to_root(out_dir)}/: {len(steps)} agent step(s) "
        f"[{', '.join(s['name'] for s in steps)}]. Nothing is executed against "
        "the data here. "
        + (
            f"Corrected feature sources: {_rel_to_root(fixed_dir)}."
            if fixed_dir is not None
            else "No corrected feature-source directory was found."
        )
    )
    log(
        f"Agent configuration: model={AGENT_MODEL}, effort={AGENT_EFFORT}, "
        f"connect timeout={AGENT_CONNECT_TIMEOUT_S}s, "
        f"step timeout={AGENT_STEP_TIMEOUT_S}s, "
        f"detector check timeout={DETECTOR_CHECK_TIMEOUT_S}s."
    )
    if spec.target is DetectorTarget.SPLIT:
        log(
            "Split candidate-policy selection: source="
            f"{_rel_to_root(candidate_review_path)}, objective="
            f"{CANDIDATE_POLICY_OBJECTIVE_NAME}, minimum worst-brain recall="
            f"{candidate_policy_minimum_recall:.12g}; the frozen selection is "
            "required by split detector assembly."
        )
    wf_start = time.monotonic()
    policy_sha256 = _sha256(MODEL_POLICY_PATH)
    runtime_template_sha256 = _sha256(RUNTIME_TEMPLATE_PATH)
    target_runtime_sha256 = _sha256(TARGET_RUNTIME_PATH)
    protected_hashes = {
        path: _sha256(path) for path in protected_source_paths(context)
    }
    if spec.target is DetectorTarget.SPLIT:
        for path in candidate_policy_source_paths(candidate_review_path):
            protected_hashes[path] = _sha256(path)
    candidate_policy_sha256: str | None = None
    inventory_sha256: str | None = None
    model_config_sha256: str | None = None
    detector_sha256: str | None = None
    run_commands_sha256: str | None = None
    readme_driver_block: str | None = None
    readme_skeleton_sha256: str | None = None

    def guard_immutable_artifacts() -> None:
        """Reject any agent write outside the artifact owned by its stage."""
        require_unchanged(
            MODEL_POLICY_PATH, policy_sha256, "versioned model-policy")
        require_unchanged(
            RUNTIME_TEMPLATE_PATH,
            runtime_template_sha256,
            "reviewed detector runtime template",
        )
        require_unchanged(
            TARGET_RUNTIME_PATH,
            target_runtime_sha256,
            "reviewed detector target adapter",
        )
        for protected_path, protected_sha256 in protected_hashes.items():
            require_unchanged(protected_path, protected_sha256, "discovery input")
        if candidate_policy_sha256 is not None:
            require_unchanged(
                out_dir / CANDIDATE_POLICY_NAME,
                candidate_policy_sha256,
                "select-candidate-policy",
            )
        if inventory_sha256 is not None:
            require_unchanged(
                out_dir / FEATURE_INVENTORY_NAME,
                inventory_sha256,
                "inventory-features",
            )
        if model_config_sha256 is not None:
            require_unchanged(
                out_dir / MODEL_CONFIG_NAME,
                model_config_sha256,
                "configure-models",
            )
        if detector_sha256 is not None:
            require_unchanged(
                out_dir / spec.detector_name,
                detector_sha256,
                "driver assembly",
            )
        if run_commands_sha256 is not None:
            require_unchanged(
                out_dir / RUN_COMMANDS_NAME,
                run_commands_sha256,
                "driver run-command guide",
            )

    def write_documentation_skeleton(detector_path: Path) -> None:
        """Regenerate driver-owned commands and factual README foundation."""
        nonlocal run_commands_sha256, readme_driver_block, readme_skeleton_sha256
        run_commands_path = out_dir / RUN_COMMANDS_NAME
        write_run_commands(
            run_commands_path,
            project_root=PROJECT_ROOT,
            run_rel=_rel_to_root(run_json),
            detector_path=detector_path,
            detector_rel=_rel_to_root(detector_path),
            inventory_path=out_dir / FEATURE_INVENTORY_NAME,
            inventory_rel=_rel_to_root(out_dir / FEATURE_INVENTORY_NAME),
            model_config_path=out_dir / MODEL_CONFIG_NAME,
            model_config_rel=_rel_to_root(out_dir / MODEL_CONFIG_NAME),
            model_policy_path=MODEL_POLICY_PATH,
            model_policy_rel=_rel_to_root(MODEL_POLICY_PATH),
            runtime_template_path=RUNTIME_TEMPLATE_PATH,
            runtime_template_rel=_rel_to_root(RUNTIME_TEMPLATE_PATH),
            cache_hint=origin_cache_hint(run_json),
            agent_model=AGENT_MODEL,
            agent_effort=AGENT_EFFORT,
            target=context.target,
            candidate_policy_path=(
                out_dir / CANDIDATE_POLICY_NAME
                if spec.target is DetectorTarget.SPLIT else None
            ),
            candidate_policy_rel=(
                _rel_to_root(out_dir / CANDIDATE_POLICY_NAME)
                if spec.target is DetectorTarget.SPLIT else None
            ),
        )
        run_commands_sha256 = _sha256(run_commands_path)
        log(
            f"  Driver wrote instance-bound commands at "
            f"{_rel_to_root(run_commands_path)}."
        )
        readme_path = out_dir / README_NAME
        write_readme_skeleton(
            readme_path,
            run_rel=_rel_to_root(run_json),
            summary_rel=_rel_to_root(summary_path),
            inventory_path=out_dir / FEATURE_INVENTORY_NAME,
            inventory_rel=_rel_to_root(out_dir / FEATURE_INVENTORY_NAME),
            model_config_path=out_dir / MODEL_CONFIG_NAME,
            model_config_rel=_rel_to_root(out_dir / MODEL_CONFIG_NAME),
            detector_rel=_rel_to_root(detector_path),
            run_commands_rel=_rel_to_root(run_commands_path),
            cache_hint=origin_cache_hint(run_json),
            primary_metric=PRIMARY_METRIC,
            target=context.target,
            candidate_policy_path=(
                out_dir / CANDIDATE_POLICY_NAME
                if spec.target is DetectorTarget.SPLIT else None
            ),
            candidate_policy_rel=(
                _rel_to_root(out_dir / CANDIDATE_POLICY_NAME)
                if spec.target is DetectorTarget.SPLIT else None
            ),
        )
        log(
            f"  Driver wrote factual README skeleton at "
            f"{_rel_to_root(readme_path)} for semantic review."
        )
        readme_driver_block = read_driver_generated_block(readme_path)
        readme_skeleton_sha256 = _sha256(readme_path)

    # Candidate-policy advice follows the same transactional pattern as feature
    # and model drafts; the resulting public artifact is mandatory assembly input.
    candidate_advice_path = out_dir / CANDIDATE_POLICY_ADVICE_DRAFT_NAME
    candidate_policy_path = out_dir / CANDIDATE_POLICY_NAME
    if spec.target is DetectorTarget.SPLIT:
        if candidate_policy_path.is_file():
            try:
                validate_frozen_candidate_policy(
                    candidate_policy_path,
                    candidate_review_path,
                    minimum_recall=candidate_policy_minimum_recall,
                    expected_mcl=expected_candidate_mcl,
                    project_root=PROJECT_ROOT,
                )
            except SystemExit as exc:
                log(f"Existing split candidate policy is stale; rebuilding it: {exc}")
            else:
                candidate_policy_sha256 = _sha256(candidate_policy_path)
                if candidate_advice_path.is_file():
                    candidate_advice_path.unlink()
                steps = [
                    step for step in steps
                    if step["name"] != "select-candidate-policy"
                ]
                log(
                    "Recovered validated split_candidate_policy.json; "
                    "skipping candidate-policy agent turn."
                )
        if (
            candidate_policy_sha256 is None
            and candidate_advice_path.is_file()
        ):
            try:
                compile_candidate_policy(
                    candidate_advice_path,
                    candidate_policy_path,
                    candidate_review_path,
                    minimum_recall=candidate_policy_minimum_recall,
                    expected_mcl=expected_candidate_mcl,
                    project_root=PROJECT_ROOT,
                )
                validate_frozen_candidate_policy(
                    candidate_policy_path,
                    candidate_review_path,
                    minimum_recall=candidate_policy_minimum_recall,
                    expected_mcl=expected_candidate_mcl,
                    project_root=PROJECT_ROOT,
                )
            except SystemExit as exc:
                log(
                    "Existing candidate-policy advice is not recoverable "
                    f"deterministically; retaining it for agent repair: {exc}"
                )
            else:
                candidate_policy_sha256 = _sha256(candidate_policy_path)
                candidate_advice_path.unlink()
                steps = [
                    step for step in steps
                    if step["name"] != "select-candidate-policy"
                ]
                log(
                    "Recovered validated candidate-policy advice and froze "
                    "split_candidate_policy.json; skipping repeated agent turn."
                )

    # Drafts are transactional: a failed compile/check leaves the stage-owned
    # hidden draft in place, while successful stages remove it. Revalidate a
    # retained semantic draft on --keep-existing before paying for another
    # agent turn; the same recovery pattern continues below for later stages.
    semantics_path = out_dir / FEATURE_SEMANTICS_DRAFT_NAME
    if semantics_path.is_file():
        inventory_path = out_dir / FEATURE_INVENTORY_NAME
        try:
            canonical_changes = compile_feature_inventory(
                semantics_path,
                inventory_path,
                selected_ids=selected_ids,
                selection_manifest=(
                    _rel_to_root(selection_path) if selection_path else None
                ),
                summary_path=summary_path,
                corrected_results_path=corrected_results_path,
                rerun_dir=rerun_dir,
                fixed_dir=fixed_dir,
                project_root=PROJECT_ROOT,
                split_applicability_path=split_applicability_path,
                run_json=(run_json if split_applicability_path is not None else None),
            )
            for change in canonical_changes:
                log(f"  Canonicalized semantic draft: {change}")
            validate_inventory(
                inventory_path,
                selected_ids,
                _rel_to_root(selection_path) if selection_path else None,
                rerun_dir,
                fixed_dir,
                summary_path,
                corrected_results_path,
                split_applicability_path=split_applicability_path,
                run_json=(run_json if split_applicability_path is not None else None),
            )
        except SystemExit as exc:
            log(
                "Existing semantic draft is not recoverable deterministically; "
                f"retaining it for the inventory agent to repair: {exc}"
            )
        else:
            inventory_sha256 = _sha256(inventory_path)
            semantics_path.unlink()
            steps = [
                step for step in steps if step["name"] != "inventory-features"
            ]
            log(
                "Recovered the validated transient feature-semantics draft; "
                "skipping the repeated inventory agent turn."
            )

    # Public artifacts are also resumable only after their complete driver-owned
    # validation passes against the current inputs/policy. This avoids paying for
    # already successful upstream agent stages after a later-stage interruption.
    inventory_path = out_dir / FEATURE_INVENTORY_NAME
    if (any(step["name"] == "inventory-features" for step in steps)
            and not semantics_path.is_file() and inventory_path.is_file()):
        try:
            validate_inventory(
                inventory_path,
                selected_ids,
                _rel_to_root(selection_path) if selection_path else None,
                rerun_dir,
                fixed_dir,
                summary_path,
                corrected_results_path,
                split_applicability_path=split_applicability_path,
                run_json=(run_json if split_applicability_path is not None else None),
            )
        except SystemExit as exc:
            log(f"Existing feature inventory is stale; rebuilding it: {exc}")
        else:
            inventory_sha256 = _sha256(inventory_path)
            steps = [
                step for step in steps if step["name"] != "inventory-features"
            ]
            log("Recovered validated feature_inventory.json; skipping Step 1.")

    advice_path = out_dir / MODEL_ADVICE_DRAFT_NAME
    config_path = out_dir / MODEL_CONFIG_NAME
    if inventory_sha256 is not None and advice_path.is_file():
        try:
            compile_model_config(
                advice_path,
                config_path,
                inventory_path,
                MODEL_POLICY_PATH,
                MODEL_POLICY_CONTRACT,
                MODEL_CONFIG_SCHEMA_VERSION,
                PROJECT_ROOT,
            )
            validate_model_config(config_path, inventory_path)
        except SystemExit as exc:
            log(
                "Existing model-advice draft needs agent repair; retaining it: "
                f"{exc}"
            )
        else:
            model_config_sha256 = _sha256(config_path)
            advice_path.unlink()
            steps = [
                step for step in steps if step["name"] != "configure-models"
            ]
            log("Recovered validated model-advice draft; skipping Step 2.")
    elif (inventory_sha256 is not None and config_path.is_file()
          and any(step["name"] == "configure-models" for step in steps)):
        try:
            validate_model_config(config_path, inventory_path)
        except SystemExit as exc:
            log(f"Existing model configuration is stale; rebuilding it: {exc}")
        else:
            model_config_sha256 = _sha256(config_path)
            steps = [
                step for step in steps if step["name"] != "configure-models"
            ]
            log("Recovered validated model_candidates.json; skipping Step 2.")

    feature_path = out_dir / FEATURE_IMPLEMENTATION_DRAFT_NAME
    detector_path = out_dir / spec.detector_name
    if model_config_sha256 is not None and feature_path.is_file():
        try:
            assemble_detector(
                RUNTIME_TEMPLATE_PATH,
                feature_path,
                detector_path,
                target=context.target,
                candidate_policy_path=(
                    candidate_policy_path
                    if spec.target is DetectorTarget.SPLIT else None
                ),
            )
            validate_detector_source(detector_path)
            validate_detector_executable(detector_path, config_path)
        except SystemExit as exc:
            log(
                "Existing feature draft needs agent repair; retaining it: "
                f"{exc}"
            )
        else:
            detector_sha256 = _sha256(detector_path)
            feature_path.unlink()
            write_documentation_skeleton(detector_path)
            steps = [
                step for step in steps if step["name"] != "generate-detector"
            ]
            log(
                "Recovered validated feature implementation and detector; "
                "skipping Step 3."
            )

    sdk = load_claude_sdk()
    async with open_agent_session(
        sdk.client,
        options=build_options(),
        connect_timeout_s=AGENT_CONNECT_TIMEOUT_S,
    ) as client:
        log("SDK session opened.")
        for i, step in enumerate(steps, start=1):
            print(f"\n=== Step {i}/{len(steps)}: {step['name']} ===")
            log(f"Step {i}/{len(steps)} '{step['name']}' started.")
            step_start = time.monotonic()
            final_text = await run_step(client, step, verbose, costs)
            log(
                f"Step {i}/{len(steps)} '{step['name']}' done in "
                f"{time.monotonic() - step_start:.0f}s."
            )
            if step["name"] == "select-candidate-policy":
                candidate_advice_path = (
                    out_dir / CANDIDATE_POLICY_ADVICE_DRAFT_NAME
                )
                candidate_policy_path = out_dir / CANDIDATE_POLICY_NAME

                def finalize_candidate_policy() -> None:
                    compile_candidate_policy(
                        candidate_advice_path,
                        candidate_policy_path,
                        candidate_review_path,
                        minimum_recall=candidate_policy_minimum_recall,
                        expected_mcl=expected_candidate_mcl,
                        project_root=PROJECT_ROOT,
                    )
                    validate_frozen_candidate_policy(
                        candidate_policy_path,
                        candidate_review_path,
                        minimum_recall=candidate_policy_minimum_recall,
                        expected_mcl=expected_candidate_mcl,
                        project_root=PROJECT_ROOT,
                    )

                await finalize_step_with_repairs(
                    client,
                    step,
                    finalize_candidate_policy,
                    guard_immutable_artifacts,
                    verbose,
                    costs,
                )
                candidate_policy_sha256 = _sha256(candidate_policy_path)
                candidate_advice_path.unlink()
                log(
                    "  Driver independently validated the candidate sweep and "
                    f"froze {_rel_to_root(candidate_policy_path)}; removed "
                    "transient agent advice. Split assembly will require and "
                    "embed this exact policy."
                )
            elif step["name"] == "inventory-features":
                semantics_path = out_dir / FEATURE_SEMANTICS_DRAFT_NAME
                inventory_path = out_dir / FEATURE_INVENTORY_NAME

                def finalize_inventory() -> None:
                    canonical_changes = compile_feature_inventory(
                        semantics_path,
                        inventory_path,
                        selected_ids=selected_ids,
                        selection_manifest=(
                            _rel_to_root(selection_path) if selection_path else None
                        ),
                        summary_path=summary_path,
                        corrected_results_path=corrected_results_path,
                        rerun_dir=rerun_dir,
                        fixed_dir=fixed_dir,
                        project_root=PROJECT_ROOT,
                        reconcile_unbacked_verdicts=reconcile_unbacked_verdicts,
                        split_applicability_path=split_applicability_path,
                        run_json=(
                            run_json if split_applicability_path is not None else None
                        ),
                    )
                    for change in canonical_changes:
                        log(f"  Canonicalized semantic draft: {change}")
                    validate_inventory(
                        inventory_path,
                        selected_ids,
                        _rel_to_root(selection_path) if selection_path else None,
                        rerun_dir,
                        fixed_dir,
                        summary_path,
                        corrected_results_path,
                        reconcile_unbacked_verdicts=reconcile_unbacked_verdicts,
                        split_applicability_path=split_applicability_path,
                        run_json=(
                            run_json if split_applicability_path is not None else None
                        ),
                    )

                await finalize_step_with_repairs(
                    client, step, finalize_inventory, guard_immutable_artifacts,
                    verbose, costs,
                )
                inventory_sha256 = _sha256(inventory_path)
                semantics_path.unlink()
                log(
                    f"  Compiled agent semantic judgments into "
                    f"{_rel_to_root(inventory_path)}; removed transient draft."
                )
            elif step["name"] == "configure-models":
                advice_path = out_dir / MODEL_ADVICE_DRAFT_NAME
                config_path = out_dir / MODEL_CONFIG_NAME

                def finalize_model_config() -> None:
                    compile_model_config(
                        advice_path,
                        config_path,
                        out_dir / FEATURE_INVENTORY_NAME,
                        MODEL_POLICY_PATH,
                        MODEL_POLICY_CONTRACT,
                        MODEL_CONFIG_SCHEMA_VERSION,
                        PROJECT_ROOT,
                    )
                    validate_model_config(
                        config_path,
                        out_dir / FEATURE_INVENTORY_NAME,
                    )

                await finalize_step_with_repairs(
                    client, step, finalize_model_config,
                    guard_immutable_artifacts, verbose, costs,
                )
                model_config_sha256 = _sha256(config_path)
                advice_path.unlink()
                log(
                    f"  Compiled agent model advice into "
                    f"{_rel_to_root(config_path)}; removed transient draft."
                )
            elif step["name"] in {"generate-detector", "verify-and-document"}:
                detector_path = out_dir / spec.detector_name
                if step["name"] == "generate-detector":
                    feature_path = out_dir / FEATURE_IMPLEMENTATION_DRAFT_NAME

                    def finalize_detector() -> None:
                        assemble_detector(
                            RUNTIME_TEMPLATE_PATH,
                            feature_path,
                            detector_path,
                            target=context.target,
                            candidate_policy_path=(
                                candidate_policy_path
                                if spec.target is DetectorTarget.SPLIT else None
                            ),
                        )
                        validate_detector_source(detector_path)
                        validate_detector_executable(
                            detector_path,
                            out_dir / MODEL_CONFIG_NAME,
                        )

                    await finalize_step_with_repairs(
                        client, step, finalize_detector,
                        guard_immutable_artifacts, verbose, costs,
                    )
                    feature_path.unlink()
                    log(
                        f"  Driver assembled {_rel_to_root(detector_path)} from "
                        "the reviewed runtime template; source and no-data checks "
                        "passed, then the transient feature draft was removed."
                    )
                else:
                    def finalize_documentation() -> None:
                        validate_detector_source(detector_path)
                        validate_detector_executable(
                            detector_path,
                            out_dir / MODEL_CONFIG_NAME,
                        )
                        if (readme_driver_block is None
                                or readme_skeleton_sha256 is None):
                            raise SystemExit(
                                "README skeleton validation state is missing.")
                        validate_readme_enrichment(
                            out_dir / README_NAME,
                            expected_driver_block=readme_driver_block,
                            skeleton_sha256=readme_skeleton_sha256,
                        )

                    await finalize_step_with_repairs(
                        client, step, finalize_documentation,
                        guard_immutable_artifacts, verbose, costs,
                    )
                if step["name"] == "generate-detector":
                    detector_sha256 = _sha256(detector_path)
                    write_documentation_skeleton(detector_path)
            if not verbose:
                print(final_text.strip())

    log(
        f"Workflow complete in {time.monotonic() - wf_start:.0f}s; "
        f"reported API cost: {costs.describe()}."
    )
    print("\n=== Workflow complete ===")
    print(f"Generated code in {_rel_to_root(out_dir)}/")
    if (out_dir / RUN_COMMANDS_NAME).is_file():
        print(
            f"Instance-bound smoke, compute-node, Slurm, and reproducibility "
            f"commands: {_rel_to_root(out_dir / RUN_COMMANDS_NAME)}"
        )
    detector_rel = f"{_rel_to_root(out_dir)}/{spec.detector_name}"
    if (out_dir / spec.detector_name).is_file():
        # End on the command, not on a summary: the detector is the deliverable
        # and running it — on a compute node, with the data — is the next action.
        # The cache is a guess from the run name (this workflow was never told
        # one), so it is worth checking against the report before submitting.
        cache = origin_cache_hint(run_json)
        print("\nRun it next, on a compute node (>20 GB RAM):")
        print("    conda activate panda")
        print(f"    python {detector_rel} \\")
        print(f"        {cache or 'cache/dataset_cache_<brain>_mcl<N>_add.pkl'} \\")
        print(f"        --model-config {_rel_to_root(out_dir / MODEL_CONFIG_NAME)}"
              + ("" if cache else "   # the origin cache, per the report"))
        print("\nThat writes the per-model OOF CSV, model-selection JSON, fitted "
              "winner joblib, txt log and figures next to the script.\nRead the "
              "undefined-coverage audit and selection manifest before trusting "
              "the selected model's scores.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_json",
        type=Path,
        help=(
            "The AutoDiscovery run export whose findings to build on, e.g. "
            "autodiscovery/merge-error-...json or split-error-...json. Its "
            "<RUN>.summary.md report, predictive-selection manifest, and matching "
            "predictive rerun scripts must already exist and agree; a matching "
            "fixed directory is consumed when present (run "
            "run_discovery_workflow.py first)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Where to write the deliverables. Default: "
            f"{APPLICATION_DIR_REL}/<RUN>/ (one subfolder per originating run)."
        ),
    )
    parser.add_argument(
        "--split-candidate-review",
        type=Path,
        default=Path(DEFAULT_SPLIT_CANDIDATE_REVIEW_REL),
        metavar="AI_REVIEW.md",
        help=(
            "Completed candidate-pool AI review used only for split builds. "
            "Its sibling evidence, manifest, and aggregate table are verified "
            "before output cleanup. Default: "
            f"{DEFAULT_SPLIT_CANDIDATE_REVIEW_REL}."
        ),
    )
    parser.add_argument(
        "--split-min-worst-brain-recall",
        type=float,
        default=DEFAULT_SPLIT_MIN_WORST_BRAIN_RECALL,
        metavar="RECALL",
        help=(
            "Driver-owned split candidate-policy constraint in (0, 1]. The "
            "selected policy minimizes total candidate count subject to every "
            "evaluated brain meeting this recall. Default: 0.90."
        ),
    )
    parser.add_argument(
        "--log-txt",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path for this driver's captured console log. Default: "
            f"<out-dir>/{DRIVER_LOG_NAME}. Everything printed — step progress, "
            "each subagent tool call, every step's final reply, and any fatal "
            "diagnostic — is tee'd there as well as to the terminal, flushed as "
            "it goes, so a workflow that dies mid-step still leaves a record. "
            "The footer summarizes the SDK-reported API cost across all completed "
            "step turns. "
            "Pass /dev/null to skip. The detector writes its own log when you "
            "run it later."
        ),
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help=(
            "Delete a pre-existing output folder WITHOUT asking. The folder is "
            "still listed first, so the log records what went. Needed in batch "
            "jobs, where there is no terminal to prompt on."
        ),
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help=(
            "Do not delete a pre-existing output folder. Driver-generated build "
            "stages are reused only after validation against current inputs, "
            "policy, and hashes; retained invalid drafts are repaired by the "
            "stage agent. Detector run outputs are preserved unchanged and may "
            "therefore still belong to an older generated detector."
        ),
    )
    parser.add_argument(
        "--reconcile-unbacked-verdicts",
        action="store_true",
        help=(
            "Downgrade one input inconsistency from abort to warning: a report "
            "post-correction verdict with no USABLE corrected measurement "
            "behind it (e.g. the discovery fold step recorded a verdict for a "
            "corrected script that timed out). Each such verdict is treated as "
            "null — the correction is ignored and the hypothesis judged on its "
            "pre-correction evidence — and every affected id is logged. The "
            "default remains a hard abort, because an unbacked "
            "UPHELD/WEAKENED/OVERTURNED can also mean the corrected results "
            "JSON was regenerated after the report was folded; regenerating "
            "the discovery outputs is the thorough fix."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream every assistant text block as it arrives.",
    )
    args = parser.parse_args()

    if args.yes and args.keep_existing:
        parser.error("--yes and --keep-existing contradict each other: one deletes "
                     "the existing folder, the other keeps it.")
    if (
        not math.isfinite(args.split_min_worst_brain_recall)
        or not 0 < args.split_min_worst_brain_recall <= 1
    ):
        parser.error("--split-min-worst-brain-recall must be finite and in (0, 1].")

    def _resolve_existing(p: Path, what: str) -> Path:
        rp = p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
        if not rp.exists():
            parser.error(f"No such {what}: {rp}")
        return rp

    run_json = _resolve_existing(args.run_json, "run export .json")

    out_dir = args.out_dir or Path(APPLICATION_DIR_REL) / run_stem(run_json)
    out_dir = out_dir if out_dir.is_absolute() else (PROJECT_ROOT / out_dir).resolve()

    # Validate the inputs BEFORE anything can be deleted: aborting on a missing
    # report AFTER wiping the output folder would be the worst of both outcomes.
    context = resolve_run_context(
        run_json,
        reconcile_unbacked_verdicts=args.reconcile_unbacked_verdicts,
    )
    candidate_review_path: Path | None = None
    if target_spec(context.target).target is DetectorTarget.SPLIT:
        candidate_review_path = _resolve_existing(
            args.split_candidate_review, "split candidate-pool AI_REVIEW.md"
        )
        validate_candidate_policy_sources(
            candidate_review_path,
            minimum_recall=args.split_min_worst_brain_recall,
            expected_mcl=expected_mcl_from_run_path(run_json),
            project_root=PROJECT_ROOT,
        )

    # Then the deletion, and note it runs before the tee is installed on purpose —
    # the driver log lives inside the folder being removed, so capturing this into
    # it would mean writing to a file that is about to be unlinked.
    clean_note = confirm_and_clean(out_dir, args.yes, args.keep_existing)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        log("WARNING: ANTHROPIC_API_KEY is not set — the SDK session will fail.")

    log_txt = args.log_txt or (out_dir / DRIVER_LOG_NAME)
    log_txt = log_txt if log_txt.is_absolute() else (PROJECT_ROOT / log_txt).resolve()
    log_txt.parent.mkdir(parents=True, exist_ok=True)

    orig_out, orig_err = sys.stdout, sys.stderr
    started = time.monotonic()
    costs = WorkflowCostSummary()
    with log_txt.open("w", encoding="utf-8") as log_fh:
        # Header first: without the argv and the timestamp, a log of step output
        # and numbers cannot be tied back to the invocation that produced it.
        log_fh.write(f"# {' '.join(sys.argv)}\n")
        log_fh.write(f"# started {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        log_fh.write(f"# host {platform.node()}  python {platform.python_version()}\n\n")
        log_fh.flush()

        sys.stdout = _Tee(orig_out, log_fh)
        sys.stderr = _Tee(orig_err, log_fh)
        outcome = "FAILED"
        # Record the deletion now that the log exists — it happened before the tee
        # was up, and a folder wiped with no trace of it in the log is exactly the
        # kind of gap this workflow's logging is meant to close.
        if clean_note:
            log(clean_note)
        try:
            asyncio.run(run_workflow(
                run_json, out_dir, args.verbose, costs,
                reconcile_unbacked_verdicts=args.reconcile_unbacked_verdicts,
                candidate_review_path=candidate_review_path,
                candidate_policy_minimum_recall=(
                    args.split_min_worst_brain_recall
                ),
            ))
            outcome = "OK"
        except SystemExit as exc:
            # Step failures raise SystemExit carrying the diagnostic as its
            # message. Python prints that to the real stderr on the way out — but
            # only after the finally below has restored it, so the log would end
            # with no hint of WHY. Write it straight to the file rather than
            # through the tee, which would duplicate it on the console.
            if exc.code not in (0, None):
                log_fh.write(f"FATAL: {exc.code}\n")
                log_fh.flush()
            raise
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = orig_out, orig_err
            # Footer last, and only here: a log whose final line is missing was
            # TRUNCATED — the process died without unwinding (an OOM kill) rather
            # than finishing.
            log_fh.write(
                f"\n# {outcome} after {time.monotonic() - started:.0f}s "
                f"(ended {datetime.now():%Y-%m-%d %H:%M:%S})\n"
                f"# reported API cost: {costs.describe()}\n"
            )
            log_fh.flush()

    log(f"Driver console log → {_rel_to_root(log_txt)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

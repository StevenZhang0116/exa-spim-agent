"""Evolve a scorer on immutable AutoDiscovery detector-native candidate pools.

No graph edits, candidate expansion, original detector retraining or table preparation.
Default TRAIN: 794495. Validation: all other eligible, prepared local datasets.
TRAIN branches are ranked by the selection protocol: grouped out-of-fold precision on
selection brains the detectors were not fitted on (auxiliary TRAIN brains only supply
fitting rows), or the historical in-sample protocol for attribution runs.
Promote on mean validation Precision@K; individual brain regressions are allowed.
Validation is repeatedly queried development data, never an untouched final test.
"""

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime
from functools import partial
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import tempfile
import time

from . import precompute_error_scores as pc
from ..harness import fixed_pool_scoring as scoring
from ..harness.native_pool import cache_path, ensure_native_tables
from ..harness.reviser_session import DEFAULT_MODEL, build_options, bind_session_options
from ..harness.trajectory import Trajectory, log_metrics, save_diffs
from ..harness.evolution_report import update_report
from ..harness.train_feedback import write_train_feedback, render_compact_json, metrics_only
from ..harness.selection_protocol import SelectionProtocol, load_selection, PROTOCOL_VERSION
from ..harness.feature_ablation import CLASSIFIER_UNITS, FORMULA_UNITS
from ..harness.context_cache import ContextCache
from ..harness.descriptor_runs import DescriptorRuns, CONTEXT_COLUMN, SCOPES as DESCRIPTOR_SCOPES
from ..harness.descriptor_contract import descriptor_spec, referenced_columns, COLUMN_PREFIX
from ..harness.scorer_components import components
from ..harness.train_experiments import ExperimentMemory, TrainingExperiments
from ..harness.candidate_pool import CandidatePool
from ..harness.classifier_contract import frozen_model, MODEL_VERSION, copy_artifacts
from ..harness.training_diagnostics import describe, feature_statistics
from ..harness.local_context import attach_local_context, CONTEXT_VERSION
from ..harness.local_features import augmented_features
from ..harness.image_context import attach_image_context, DEFAULT_ALIGNMENT
from ..harness.image_contract import IMAGE_VERSION
from ..harness.image_evidence import generation_image_evidence
from ..harness.research_evidence import registered_image_available
from ..harness.volume_analysis import VOLUME_ANALYSIS_VERSION, MAX_ANALYSES, MAX_CASES, MAX_RESULT_BYTES


HERE = Path(__file__).resolve().parents[1]
CONTRACT = (HERE / "artifacts" / "scorer_rules.md").read_text()
DEFAULT_REVISER_MAX_TURNS = 24


async def revise(run_dir, policy_path, rules_path, report_path, model, *,
                 max_turns=DEFAULT_REVISER_MAX_TURNS, experiments=None):
    """A fresh SDK session with the existing per-file access guard; no subagents."""
    from claude_agent_sdk import ResultMessage, ClaudeSDKClient
    trace = Trajectory(policy_path.parent)
    options, _ = build_options(model=model, run_dir=run_dir,
                               system_prompt=CONTRACT, max_turns=max_turns)
    readable = []
    if experiments is not None:
        readable = [policy_path.parent / name for name in (
            'feature_statistics.json', 'search_plan.json', 'candidate_pool.json', 'classifier_guide.md',
            'model_environment.json', 'local_context_guide.md', 'failure_cases.json',
            'hypothesis_memory.json', 'research_status.json', 'feature_discovery_guide.md')]
        readable.extend(policy_path.parent / name for name in (
            'image_context_guide.md', 'volume_analysis_guide.md', 'image_scoring_plan.json',
            'descriptor_guide.md'))
    options = bind_session_options(
        options, policy_path, rules_path, report_path,
        readable_paths=readable,
        training_server=experiments.mcp_server() if experiments else None)
    result = None
    prompt = (
        f"Read TRAIN feedback at {report_path}. Improve {policy_path} and update {rules_path}. "
        "Follow the fixed-pool scorer contract. Do not change the candidate pool, budgets or evaluator. "
        f"You have at most {max_turns} SDK turns. The compact feedback is at most 48 KB; read it once. "
        "Feature vectors align with example labels/scores by index. Groups are diagnostic samples, not a dataset. "
        "Reserve turns to write scorer.py and rules.md and return your summary; avoid repeated tiny reads. "
        "Briefly describe the planned change before editing. "
        "At the end, summarize what changed, the TRAIN evidence motivating it, "
        "and the expected effect. Do not claim evaluation results you have not observed."
    )
    if experiments is not None:
        prompt += (
            f" This generation changes ONLY the {experiments.target_kind} component; the harness freezes the other kind. "
            f"Whole-TRAIN feature statistics are at {policy_path.parent / 'feature_statistics.json'}. "
            "Use mcp__training__search_memory to look up related previous strategies before repeating them. "
            "Read failure_cases.json, hypothesis_memory.json and feature_discovery_guide.md once. "
            "Use direct 3D analysis to investigate image evidence on a few TRAIN candidates when no pool-scale "
            "descriptor tools are offered; with a context cache attached, prefer compute_descriptors and keep "
            "run_volume_analysis for debugging describe. "
            "Read volume_analysis_guide.md, edit analysis.py analyze(context) and analysis_request.json, "
            "then call run_volume_analysis({}). Your code receives full original 3D pixel arrays, "
            "valid masks, zyx voxel anchors/spacing and local fragment nodes in the same voxel frame. "
            "Choose your own analysis algorithm and return compact JSON findings; no fitting or "
            "2D viewing is required first. Select 1..4 TRAIN candidate handles per call; eight analysis "
            "executions per generation have a separate budget. Repeated calls run the current code anew. "
            "The request is initially populated with one matched failure pair when available. "
            "These chosen cases need not be in the LOCAL_IMAGE scoring subset. A successful analysis "
            "is exploratory evidence, not a measured ranking improvement or an automatically added feature. "
            "Read image_context_guide.md to convert useful analyses into scoring inputs. LOCAL_IMAGE with "
            "extract_image_features(context) adds image_* predictors; a training program can instead "
            "request raw_patches and receive an images keyword argument in fit/predict. Choose any "
            "installed CPU model within the budget. The host reads and registers images; no network "
            "access or cloud credentials enter agent code. Image extraction uses a bounded subset of "
            "candidates, with missing images explicitly represented. For raw-patch ablation, include "
            "image_available in research.feature_columns to remove the patch input as well. "
            "Bridge small investigations to ranking with LOCAL_IMAGE selection_mode='top_k_boundary', "
            "a literal selection_k equal to feedback.budgets for this kind, and an explicit max_candidates. "
            "Call plan_image_scoring({}) to preview label-free TRAIN coverage without IO. Start with a "
            "bounded boundary pilot, then expand coverage up to 4096 rows if the evidence and IO budget justify it. "
            "Numeric extraction is batched by rows/bytes under one shared timeout; raw-patch model inputs "
            "still have a 1 GiB staging limit. Keep the same frozen selector on all brains. "
            "Inspect image_context.selection_coverage and scoring_coverage: selected rows, rows inside/outside "
            "the base-predictor K boundary, and actual final Top-K image coverage are different quantities. "
            "At the final coverage, ablate exactly the image_* features plus image_available to isolate "
            "image contribution. A small-case analysis or pilot ablation does not establish gain at larger coverage. "
            "Optional inspect_candidate_image/inspect_failure_images tools return 2D previews only, "
            "with a separate eight-request allowance. They are not prerequisites for 3D analysis. "
            "No GT graph or validation inspection is exposed through either route. "
            "In explore mode, start from a matched failure pair and an explicit, falsifiable distinction. "
            "If local geometry can test it, call inspect_failure_cases({}) once for the paired neighborhoods, "
            "rather than repeatedly opening individual examples. These cases are diagnostic, not a dataset. "
            "Use hypothesis_advice in search_plan.json to avoid spending another round on a stagnant direction. "
            "A new parameter value or family name alone is not new evidence. Consider a new information "
            "source, feature mechanism or complementary failure slice when the existing direction stalls. "
            f"For local morphology, read {policy_path.parent / 'local_context_guide.md'}. "
            "Use mcp__training__inspect_candidate with a candidate_ref from TRAIN feedback, "
            "radius_um=50 and occurrence_index=0 to examine location, fragments and nearby branches. "
            "You have eight neighborhood inspections per generation. In explore mode you may add "
            "LOCAL_CONTEXT and extract_local_features(context) to scorer.py or training.py. "
            "The host computes your declared local_* columns in isolation for TRAIN and frozen validation inference. "
            "Only TRAIN can be inspected; GT graphs and absolute IDs never enter feature extraction. "
            f"Read {policy_path.parent / 'search_plan.json'} and {policy_path.parent / 'candidate_pool.json'}. "
            "The TRAIN pool keeps the overall champion plus specialists that recover different positives "
            "or lead a geometry slice, even outside the configured near-best precision tolerance. "
            "A separate reference_branch is protected outside that pool. A newly supplied reference gets "
            "the next same-kind round for one fresh follow-up, with at most one retry; otherwise it receives "
            "every third scheduled round of this kind. Its supplied measurements are TRAIN-only. "
            "When search_plan.branch_role is reference, develop the assigned reference using its TRAIN "
            "diagnostics; do not abandon it solely because another archived method has a higher selection score. "
            "Read search_plan.investigation and research_status.json. These are host-enforced assignments: "
            "when followup_required is true, perform a new measurement from the assigned workspace branch. "
            "Restoring an unrelated historical candidate does not complete that follow-up. After two "
            "same-kind rounds without new measurements, or a performance plateau, an investigation requires "
            "a NEW paired TRAIN feature ablation or numerical 3D analysis on at least two distinct TRAIN "
            "candidates. When required_source is raw_image, compare 3D patches or ablate all image inputs. "
            "Negative findings are valid. Previews, repeated observations, cached restores, changed comments "
            "or deferred plans do not complete the assignment. A score-only parameter tweak is not a paired "
            "comparison. If follow-up and investigation coincide, run that comparison on the assigned branch. "
            "Tool replies include live completion feedback. Reserve budget for the comparison; incomplete "
            "assignments retain the accepted policy and skip outer evaluation. "
            "Inspect specialist_profile, retention, vs_champion and vs_other_retained in candidate_pool.json, "
            "and complementary_candidates in search_plan.json. Slice recall counts positives under the SAME "
            "global Top K, not a separate per-slice budget. In explore mode, use restore_candidate with "
            "these experiment IDs to inspect complementary measured methods and develop a new combination "
            "or improve a weak slice. State which missed-positive group your hypothesis targets. "
            f"This is a {experiments.search_plan['mode']} generation. In tune mode ONLY change numeric values "
            "in the top-level PARAMS dictionary, or refit numeric hyperparameters of the assigned classifier. "
            "The formula or training program and nonnumeric parameters stay fixed. In explore mode "
            "freely redesign the training program, model, preprocessing, loss, ensemble or feature selection. "
            "Put a testable hypothesis, strategy, short family name and parameter_grid in proposal.json. "
            "For a new investigation also include research with a stable hypothesis_id, failure_mode, "
            "information_source (e.g. local_geometry), prediction, and optional feature_columns. "
            "Reuse the ID across numeric variants of the same hypothesis; use a new ID for a different mechanism. "
            "For a claimed feature improvement, call evaluate_feature_ablation({}) with an empty parameter_grid "
            "before final full-TRAIN fitting. The same classifier program and parameters are refit in three "
            "framework-owned TRAIN folds, with and without the declared input feature group. This costs TWO "
            "evaluation units (one per arm). It does not restore or submit a scorer. "
            "Formula ablation also costs TWO units and is descriptive, not a generalization estimate. "
            "Features are constant-masked in both fit/predict to preserve arbitrary model input schemas. "
            "An unavailable or failed diagnostic supplies no scientific conclusion. Treat measured feature "
            "evidence as specific to its exact program, parameters, columns and split. "
            'Example: {"hypothesis":"Geometry helps near K","strategy":"Blend geometry and detector",'
            '"family":"geometry-blend","parameter_grid":{"weight":[0.5,1,2]}}. '
            "A hand-written formula defines PARAMS = {'weight': 1.0} and references PARAMS['weight']. "
            "Call mcp__training__evaluate_train({}) for a single measurement or "
            "mcp__training__search_parameters({}) to evaluate the grid and automatically restore its best candidate. "
            f"For genuine supervised fitting, read {policy_path.parent / 'classifier_guide.md'}, "
            "write training.py with fit(X_train, y_train, artifact_dir, params) and predict(X, artifact_dir, params), "
            "set proposal.json.classifier.parameters, and call mcp__training__train_classifier({}). "
            "Read model_environment.json for installed packages and resource limits. "
            "Choose any CPU model supported by installed packages and the resource budget. "
            "The harness supplies TRAIN labels only, saves your artifacts, and writes measured scorer.py. "
            "Save fitted preprocessing with the model; predict receives read-only artifacts and no labels. "
            "Do not edit the generated manifest. No network, package installation or child processes. "
            f"Branch ranking, grid winners, specialists and stall counters use the official selection score "
            f"described by selection_protocol in the feedback (mode={experiments.protocol.mode}). "
            + ("Under grouped_oof it is grouped out-of-fold precision on the selection brains "
               f"({', '.join(experiments.protocol.selection_brains)}): every classifier configuration is first "
               "refit by the host in three fixed, fragment-purged folds and ranked by out-of-fold precision with "
               "fold budgets summing to K; the full-TRAIN fit is the frozen candidate. Auxiliary TRAIN brains "
               f"({', '.join(experiments.protocol.auxiliary_brains) or 'none'}) only supply fitting rows and are "
               "never held out. Formulas are scored under the same partition. parent_train/candidate_train "
               "in-sample numbers are diagnostics and rank nothing. A model that memorizes TRAIN rows will "
               "score high in-sample and low out-of-fold. "
               "If fit declares a `groups` keyword it receives the label-free source brain of each training "
               "row, so you can weight or block auxiliary rows. "
               if experiments.protocol.requires_fold_fits else
               "In this run the selection score is the historical in-sample TRAIN ranking; it is not an "
               "independent generalization estimate. ")
            + f"Budget: {experiments.max_evaluations} evaluation units total across all evaluation and fitting tools; "
            "one classifier configuration costs one unit (its fold refits plus the full-TRAIN fit and scoring), "
            "including failed fits; a feature ablation costs one unit per arm. "
            "One extra scorer attempt follows the first scorer execution error; failed ablations do not grant it. "
            "Cached repeats and proposal-format errors "
            "do not spend evaluation budget, but tool calls still consume SDK turns. Grid combinations must fit "
            "the remaining budget. Prefer one grid call to hand-editing constants across many calls. "
            "Inspect measured precision, gained/lost positives and top-K overlap before revising again. "
            "The initial scorer is the selected exploration branch. parent_train in feedback and all "
            "comparisons refer to the accepted parent, which can differ from this branch. "
            'Use mcp__training__restore_candidate({"candidate_id":"parent"}) to measure/recover the assigned '
            'branch from cache, or {"candidate_id":"best"} to restore the best measured candidate of this generation. '
            "IDs such as gen002/attempt001 restore earlier measured branches without rewriting their source. "
            "Finish with scorer.py exactly matching a successfully evaluated snapshot. An untested final edit is rejected. "
            "Summarize observed TRAIN evidence and "
            "remaining uncertainty in rules.md. Validation is unavailable to this session."
        )
        if experiments.memory.descriptor_runs is not None:
            budget = experiments.descriptor_budget
            prompt += (
                " POOL-SCALE DESCRIPTORS: the host has cached the raw context (fragment neighbourhood and image "
                "patches) of the top detector-ranked band of every brain. Write descriptor.py with a literal "
                "DESCRIPTOR dict and describe(context) that returns the quantities you want measured; read "
                f"{policy_path.parent / 'descriptor_guide.md'} once. Call plan_descriptor_run({{}}) to time it, then "
                'compute_descriptors({"scope": "pilot" | "boundary" | "all_cached"}) to run it over every cached row '
                f"of the TRAIN brains in {experiments.memory.descriptor_runs.workers} parallel sandboxed workers. "
                f"Budget: {budget['per_call_seconds']:.0f} s per call and {budget['generation_seconds']:.0f} s per "
                "generation of wall time, plus one evaluation unit per uncached call; cached code hashes are free. "
                "The reply returns label-conditional TRAIN quartiles and coverage per column, and registers "
                "bank_agent_<name> predictor columns (NaN outside the band; bank_context_available marks cached rows) "
                "that scorer.py, training.py and evaluate_feature_ablation can use immediately. Prefer this over "
                "four-site run_volume_analysis for any claim about pixels or local geometry; use run_volume_analysis "
                "only to debug describe on a few rows. Examples of earlier descriptors are in the guide."
            )
    (policy_path.parent / "reviser_prompt.txt").write_text(prompt)
    trace.emit("reviser_start", f"Starting {model}; max_turns={max_turns}; prompt saved to reviser_prompt.txt",
               model=model, max_turns=max_turns)
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            trace.sdk_message(message)
            if isinstance(message, ResultMessage):
                result = {"summary": message.result, "usage": message.usage,
                          "cost_usd": message.total_cost_usd, "is_error": message.is_error,
                          "subtype": message.subtype, "num_turns": message.num_turns,
                          "max_turns": max_turns, "session_id": message.session_id,
                          "errors": getattr(message, "errors", None),
                          "stop_reason": getattr(message, "stop_reason", None),
                          "api_error_status": getattr(message, "api_error_status", None)}
                _write(policy_path.parent / "reviser_result.json", result)
                if message.is_error:
                    detail = "; ".join(str(e) for e in (result["errors"] or [])) or message.result or ""
                    raise RuntimeError(
                        f"Scorer reviser SDK error: subtype={message.subtype}; "
                        f"turns={message.num_turns}; max_turns={max_turns}; "
                        f"stop_reason={result['stop_reason']}; api_status={result['api_error_status']}"
                        + (f"; detail={detail[:500]}" if detail else ""))
    if result is None:
        raise RuntimeError("Scorer reviser ended without a result")
    return result


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def _csv_brains(value):
    brains = value.split(",")
    if any(not b.isascii() or not b.isdigit() for b in brains) or len(set(brains)) != len(brains):
        raise argparse.ArgumentTypeError("Use distinct comma-separated numeric brain IDs")
    return brains


def _validation_brains(value):
    return None if value == "auto" else _csv_brains(value)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-brains", type=_csv_brains, default=["794495"])
    p.add_argument("--validation-brains",
                   type=_validation_brains, default=None, metavar="auto|ID,ID,...",
                   help="Other eligible brains with matching caches and prepared tables (default: auto); "
                        "or an explicit comma-separated list. Development validation, not final testing")
    p.add_argument("--selection-brains", type=_csv_brains, default=None, metavar="ID,ID,...",
                   help="TRAIN brains that provide the official grouped out-of-fold selection score "
                        "(default: every TRAIN brain the frozen detectors were not fitted on). Remaining "
                        "TRAIN brains are auxiliary fitting data; their in-sample scores are diagnostics")
    p.add_argument("--selection-protocol", choices=("grouped_oof", "in_sample"), default="grouped_oof",
                   help="grouped_oof ranks TRAIN branches by out-of-fold precision on selection brains "
                        "(default); in_sample keeps the historical full-pool in-sample ranking for attribution")
    p.add_argument("--skip-auxiliary-ablation", action="store_true",
                   help="Skip the one-time baseline diagnostic that refits the seed training template with "
                        "and without auxiliary TRAIN rows under the selection protocol")
    p.add_argument("--bootstrap-draws", type=int, default=200,
                   help="Paired row-resampling draws recorded with each promotion decision (default: 200; "
                        "0 disables). Logging only; the gate is unchanged")
    p.add_argument("--context-cache", type=Path, default=None,
                   help="Directory built by cli.precompute_context_cache; enables pool-scale agent descriptors "
                        "(plan_descriptor_run / compute_descriptors). Omit to run without them")
    p.add_argument("--descriptor-bank", type=Path, default=HERE / "descriptor_bank",
                   help="Persistent cache of computed descriptor values, keyed by descriptor code hash "
                        "(default: proofreader_evolve/descriptor_bank)")
    p.add_argument("--descriptor-workers", default="auto", metavar="auto|N",
                   help="Parallel sandboxed workers for descriptor computation (default: available CPUs minus 2)")
    p.add_argument("--descriptor-wall-seconds", type=float, default=1800.,
                   help="Wall-clock budget for one compute_descriptors call (default: 1800)")
    p.add_argument("--descriptor-generation-wall-seconds", type=float, default=5400.,
                   help="Wall-clock budget for all uncached descriptor computations in one generation (default: 5400)")
    p.add_argument("--mcl", type=int, default=100)
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--merge-k", type=int, default=100)
    p.add_argument("--split-k", type=int, default=100)
    p.add_argument("--precision-margin", type=float, default=0,
                   help="Required absolute gain in equal-weight mean validation Precision@K (default: 0)")
    p.add_argument("--policy-time-budget", type=float, default=120,
                   help="Scoring/feature worker budget and 3D analysis worker timeout, in seconds (default: 120)")
    p.add_argument("--classifier-time-budget", type=float, default=300,
                   help="Wall/CPU fitting budget per classifier configuration, in seconds (default: 300)")
    p.add_argument("--classifier-memory-mb", type=int, default=8192,
                   help="Address-space limit for fitting, fit-time extraction, diagnostics and 3D analysis "
                        "(default: 8192 MiB); ordinary inference uses 8192 MiB")
    p.add_argument("--classifier-threads", type=int, default=1,
                   help="Numerical library CPU threads for fit/predict and 3D analysis (default: 1)")
    p.add_argument("--start-from-artifacts", type=Path,
                   help="Model artifact store for --start-from (default: sibling model_artifacts directory)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--reviser-max-turns", type=int, default=DEFAULT_REVISER_MAX_TURNS,
                   help="Maximum SDK interaction turns per generation (default: 24)")
    p.add_argument("--train-evaluations-per-generation", type=int, default=8,
                   help="Shared TRAIN evaluation units (default: 8); each ablation fold/arm costs one. "
                        "Cache hits are free; the first scorer execution error allows one extra repair")
    p.add_argument("--candidate-pool-size", type=int, default=5,
                   help="Maximum TRAIN exploration branches per kind, plus one protected reference (default: 5)")
    p.add_argument("--candidate-score-tolerance", type=float, default=.02,
                   help="Selection-score gap for near-best fallback branches (default: 0.02); "
                        "complementary-positive and slice specialists are exempt")
    p.add_argument("--plateau-patience", type=int, default=2,
                   help="Measured rounds without selection-score or specialist-coverage progress "
                        "before forced exploration (default: 2)")
    p.add_argument("--exploration-patience", type=int, default=2,
                   help="Unsuccessful measured investigations, incomplete assignments, or consecutive "
                        "execution-blocked rounds before pausing a kind (separate counters; default: 2)")
    p.add_argument("--explore-every", type=int, default=3,
                   help="Periodic exploration every N measured TRAIN-scheduled rounds per kind; "
                        "reserved reference rounds have a separate cadence (default: 3)")
    p.add_argument("--no-early-stop", action="store_true",
                   help="Disable pauses for stagnation, incomplete investigations and execution blocks; "
                        "retain tool budgets, follow-up retry bound and generation limit")
    p.add_argument("--target-kind", choices=("alternate", "merge", "split"), default="alternate",
                   help="Component to revise; alternate changes one kind per generation")
    p.add_argument("--feature-tables-dir", type=Path, default=pc.DEFAULT_OUT)
    p.add_argument('--image-alignment', type=Path, default=DEFAULT_ALIGNMENT,
                   help='Host-owned reviewed image/cache registration receipts for optional image access')
    p.add_argument("--merge-dir", action="append")
    p.add_argument("--split-dir", action="append")
    p.add_argument("--start-from", type=Path, default=HERE / "artifacts" / "scorer.py")
    p.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    args = p.parse_args(argv)
    if set(args.train_brains) & set(args.validation_brains or []):
        p.error("Train and validation brains must be disjoint")
    if args.selection_brains is not None and set(args.selection_brains) - set(args.train_brains):
        p.error("Selection brains must be a subset of the TRAIN brains")
    if args.bootstrap_draws < 0:
        p.error("bootstrap-draws must be >= 0")
    if args.descriptor_workers != "auto":
        if not str(args.descriptor_workers).isdigit() or int(args.descriptor_workers) < 1:
            p.error("descriptor-workers must be 'auto' or a positive integer")
        args.descriptor_workers = int(args.descriptor_workers)
    if (not math.isfinite(args.descriptor_wall_seconds) or args.descriptor_wall_seconds <= 0
            or not math.isfinite(args.descriptor_generation_wall_seconds) or args.descriptor_generation_wall_seconds <= 0):
        p.error("Descriptor wall budgets must be positive and finite")
    if args.generations < 0 or args.mcl < 0 or min(args.merge_k, args.split_k) < 1:
        p.error("generations/MCL must be >= 0; K must be >= 1")
    if args.reviser_max_turns < 1:
        p.error("reviser-max-turns must be >= 1")
    if args.train_evaluations_per_generation < 1:
        p.error("train-evaluations-per-generation must be >= 1")
    if min(args.candidate_pool_size, args.plateau_patience, args.exploration_patience, args.explore_every) < 1:
        p.error("Candidate-pool size and scheduling patience/intervals must be >= 1")
    if not math.isfinite(args.candidate_score_tolerance) or not 0 <= args.candidate_score_tolerance <= 1:
        p.error("candidate-score-tolerance must be in [0, 1]")
    if (not math.isfinite(args.precision_margin) or args.precision_margin < 0
            or not math.isfinite(args.policy_time_budget) or args.policy_time_budget <= 0):
        p.error("Invalid precision margin or policy time budget")
    if not math.isfinite(args.classifier_time_budget) or args.classifier_time_budget <= 0 or args.classifier_memory_mb < 512:
        p.error("Classifier time budget must be positive and finite; classifier-memory-mb must be >= 512")
    if args.classifier_threads < 1:
        p.error("classifier-threads must be >= 1")
    return args


def _load_evaluation_banks(args, selected, trace):
    """Freeze the split before scoring; auto mode skips only unavailable inputs."""
    banks = {}

    def load(brain):
        trace.emit("table_load", f"Loading and validating tables for brain {brain}")
        bank = ensure_native_tables(brain, cache_path(brain, args.mcl), selected,
                                    args.feature_tables_dir, prepare=False, mcl=args.mcl)
        if not bank.tables or any(len(table.truth) == 0 for table in bank.tables.values()):
            raise ValueError(f"Empty candidate pool for brain {brain}; repair its source cache/tables")
        return bank

    for brain in args.train_brains:
        banks[brain] = load(brain)
    fitted_brains = {str(table.meta["training_brain"])
                     for bank in banks.values() for table in bank.tables.values()
                     if table.meta.get("training_brain") is not None}
    if args.selection_protocol == "grouped_oof":
        for brain in args.train_brains:
            if any(table.meta.get("training_brain") is None for table in banks[brain].tables.values()):
                raise ValueError("Detector training-brain provenance is required for the grouped_oof "
                                 "selection protocol; rebuild the tables or use --selection-protocol in_sample")
        selection_brains = (list(args.selection_brains) if args.selection_brains
                            else [b for b in args.train_brains if b not in fitted_brains])
        conflict = sorted(set(selection_brains) & fitted_brains)
        if conflict:
            raise ValueError(f"Selection brains must not be detector training brains: {conflict}")
        if not selection_brains:
            raise ValueError("No eligible selection brain: every TRAIN brain is a detector-fitted brain. "
                             "Add a non-fitted TRAIN brain (e.g. --train-brains 794495,802449) or use "
                             "--selection-protocol in_sample")
    else:
        selection_brains = list(args.train_brains)
    args.selection_brains = selection_brains
    auxiliary_brains = [b for b in args.train_brains if b not in selection_brains]
    auto = args.validation_brains is None
    selection = {"mode": "auto" if auto else "explicit", "train_brains": list(args.train_brains),
                 "detector_training_brains": sorted(fitted_brains), "excluded": {},
                 "selection_protocol": args.selection_protocol, "selection_brains": selection_brains,
                 "auxiliary_train_brains": auxiliary_brains,
                 "roles": {**{b: "selection TRAIN (official out-of-fold score)" if args.selection_protocol == "grouped_oof"
                              else "TRAIN (in-sample selection)" for b in selection_brains},
                           **{b: "auxiliary TRAIN (fitting rows only; in-sample diagnostics)" for b in auxiliary_brains}}}
    if auto:
        directory = cache_path(args.train_brains[0], args.mcl).parent
        prefix, suffix = "dataset_cache_", f"_mcl{args.mcl}_add.pkl"
        candidates = []
        for path in directory.glob(f"{prefix}*{suffix}"):
            brain = path.name[len(prefix):-len(suffix)]
            if path.is_file() and brain.isascii() and brain.isdigit():
                candidates.append(brain)
        candidates = sorted(set(candidates))
    else:
        candidates = list(args.validation_brains)
    selection["candidates"] = candidates
    validation_brains = []
    for brain in candidates:
        reason = None
        if brain in args.train_brains:
            reason = "TRAIN brain"
        elif brain in fitted_brains:
            reason = "Detector-fitted brain cannot be development validation"
        if reason is not None:
            if not auto:
                raise ValueError(reason)
            selection["excluded"][brain] = reason
            trace.emit("validation_excluded", f"Validation excludes {brain}: {reason}")
            continue
        try:
            bank = load(brain)
        except FileNotFoundError as exc:
            if not auto:
                raise
            selection["excluded"][brain] = str(exc)
            trace.emit("validation_excluded", f"Validation excludes {brain}: {exc}")
            continue
        for table in bank.tables.values():
            fitted = table.meta.get("training_brain")
            if fitted is None:
                raise ValueError("Detector training-brain provenance is required for independent validation")
            if str(fitted) == brain:
                raise ValueError("Detector-fitted brain cannot be development validation")
        banks[brain] = bank
        validation_brains.append(brain)
    if not validation_brains:
        raise ValueError("No eligible prepared validation datasets. Prepare feature tables for at least "
                         "one non-TRAIN, non-detector-fitted brain, or supply --validation-brains explicitly.")
    selection["validation_brains"] = validation_brains
    selection["roles"].update({b: "development validation (promotion gate)" for b in validation_brains})
    # All later generations use these already loaded banks, never rediscovering datasets.
    args.validation_brains = validation_brains
    trace.emit("dataset_split", f"TRAIN: {', '.join(args.train_brains)} "
               f"(selection: {', '.join(selection_brains)}; auxiliary: {', '.join(auxiliary_brains) or 'none'}; "
               f"protocol={args.selection_protocol}); validation ({len(validation_brains)}): "
               f"{', '.join(validation_brains)}; split fixed for this run", **selection)
    return banks, selection


def _restore_descriptors(args, banks, kinds, parent_sources, descriptor_runs, budgets, trace):
    """Recompute descriptors a resumed scorer references from the registry saved beside it."""
    required = set()
    for kind in kinds:
        model = None
        try:
            model = frozen_model(parent_sources[kind])
        except (SyntaxError, ValueError):
            model = None
        if model is not None and model.get('columns') is not None:
            required.update(c for c in model['columns'] if c.startswith(COLUMN_PREFIX))
        else:
            required.update(referenced_columns((model or {}).get('program') if model else parent_sources[kind]))
    if not required:
        return None
    registry_path = args.start_from.parent / 'descriptors.json'
    if not registry_path.is_file():
        raise ValueError(f"The starting scorer references {sorted(required)[:3]} but {registry_path} is missing")
    registry = json.loads(registry_path.read_text())
    missing = sorted(required - set(registry.get('columns', {})))
    if missing:
        raise ValueError(f'descriptors.json lacks registered columns {missing[:3]}')
    programs = {}
    for code, stored in registry['programs'].items():
        program, spec = descriptor_spec(stored['program'])
        if spec['code_sha256'] != code:
            raise ValueError('descriptors.json program hash mismatch')
        programs[code] = {'program': program, 'spec': spec}
    columns = {c: registry['columns'][c] for c in required}
    for code in {record['code_sha256'] for record in columns.values()}:
        program, spec = programs[code]['program'], programs[code]['spec']
        scope = next(record['scope'] for record in columns.values() if record['code_sha256'] == code)
        for brain, bank in banks.items():
            table = bank.tables[spec['kind']]
            result = descriptor_runs.compute(table, program, spec, scope, k=budgets[spec['kind']],
                                             wall_budget=args.descriptor_wall_seconds,
                                             log_dir=args.runs_dir / '.descriptor_restore' / brain / code[:12])
            descriptor_runs.register(table, spec['columns'], result['rows'], result['values'])
    trace.emit('descriptors_restored', f'Restored {len(columns)} descriptor columns for the starting scorer',
               columns=sorted(columns), programs=sorted(programs))
    return {'columns': columns, 'programs': programs}


def _auxiliary_ablation(args, protocol, kinds, run_dir, fit_limits, trace):
    """One-time host diagnostic: does auxiliary TRAIN data help the seed template out of fold?

    Refits artifacts/training.py per fold on selection rows only and on selection
    plus auxiliary rows, reporting both official selection scores. It ranks no
    branch, spends no agent budget and never touches validation brains.
    """
    if (not protocol.requires_fold_fits or not protocol.auxiliary_brains
            or getattr(args, "skip_auxiliary_ablation", False)):
        return None
    template = (HERE / "artifacts" / "training.py").read_text()
    results = {}
    for kind in kinds:
        arms = {}
        for arm, include in (("selection_only", False), ("selection_plus_auxiliary", True)):
            directory = run_dir / "auxiliary_ablation" / kind / arm
            try:
                report, _ = protocol.measure_classifier(kind, template, {"parameters": {}}, directory,
                                                        include_auxiliary=include, **fit_limits)
                arms[arm] = {"status": "measured", **metrics_only(report),
                             "fitting_brains": report["fitting_brains"], "fold_fits": report["fold_fits"]}
            except Exception as exc:
                arms[arm] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        measured = all(v["status"] == "measured" for v in arms.values())
        results[kind] = {"program": "artifacts/training.py", "config": {"parameters": {}},
                         "metric": PROTOCOL_VERSION, "arms": arms,
                         "delta_with_auxiliary": (arms["selection_plus_auxiliary"]["macro_precision"]
                                                  - arms["selection_only"]["macro_precision"]) if measured else None,
                         "scope": "Host-only baseline diagnostic of the seed training template under the "
                                  "selection protocol; informs whether auxiliary rows help, ranks no branch"}
        trace.emit("auxiliary_ablation", f"{kind}: seed template grouped_cv_precision selection-only="
                   f"{arms['selection_only'].get('macro_precision')} with-auxiliary="
                   f"{arms['selection_plus_auxiliary'].get('macro_precision')}", **results[kind])
    _write(run_dir / "auxiliary_ablation.json", results)
    return results


async def run(args, revise_fn=None):
    """Budgeted TRAIN search followed by mean-validation-only promotion."""
    revise_fn = partial(revise, max_turns=args.reviser_max_turns) if revise_fn is None else revise_fn
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="precision_" + datetime.now().strftime("%Y%m%d_%H%M%S_"),
                                   dir=args.runs_dir)).resolve()
    trace = Trajectory(run_dir)
    trace.emit("run_start", f"Run directory: {run_dir}",
               config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    report = update_report(run_dir, trace, phase='initializing')
    if report is not None:
        trace.emit('report_ready', f'Offline exploration report: {report}; refresh after each generation')
    try:
        return await _run(args, revise_fn, run_dir, trace)
    except BaseException as exc:
        trace.emit("run_failed", f"{type(exc).__name__}: {exc}")
        update_report(run_dir, trace,
                      status='interrupted' if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else 'failed',
                      reason=f'{type(exc).__name__}: {exc}')
        raise


async def _run(args, revise_fn, run_dir, trace):
    selected = pc.resolve_detector_runs(args.merge_dir, args.split_dir)
    banks, selection = _load_evaluation_banks(args, selected, trace)
    attach_local_context(banks, run_dir / 'local_feature_cache', trace)
    image_store = attach_image_context(banks, run_dir / 'image_cache', args.image_alignment, trace)
    _write(run_dir / 'image_alignment.json', {'brains': image_store.reviews})
    _write(run_dir / "dataset_split.json", selection)
    trace.emit("promotion_rule", "Promotion uses equal-weight mean validation Precision@K only; "
               "TRAIN improvement is diagnostic, and individual validation regressions are allowed",
               gate_version=scoring.VALIDATION_GATE_VERSION, aggregation=scoring.AGGREGATION)
    trace.emit("metric_definitions", "discoverable_pool_positives = GT-positive candidate rows in the fixed pool; "
               "top_k_hits = GT-positive rows selected in Top K; Precision@K = hits / effective_k; "
               "Recall@K = hits / pool positives (N/A when zero). Merge counts are candidate sites; "
               "split counts are segment pairs. These are not total GT error counts or reachable-GT counts. "
               "Evaluation datasets are logged as validation.")
    train = {b: banks[b] for b in args.train_brains}
    validation = {b: banks[b] for b in args.validation_brains}
    budgets = {"merge": args.merge_k, "split": args.split_k}
    artifact_store = run_dir / "model_artifacts"
    source = args.start_from.read_text()
    parent_sources = components(source)
    available = set.intersection(*(set(bank.tables) for bank in banks.values()))
    kinds = [k for k in ("merge", "split") if k in available]
    if args.target_kind != "alternate":
        if args.target_kind not in available:
            raise ValueError("Target kind must exist in every train and validation brain")
        kinds = [args.target_kind]
    if not kinds:
        raise ValueError("No common scoring component to evolve")
    protocol = SelectionProtocol(train, budgets, kinds=kinds, selection_brains=args.selection_brains,
                                 mode=args.selection_protocol)
    fit_limits = {"fit_timeout": args.classifier_time_budget, "score_timeout": args.policy_time_budget,
                  "memory_mb": args.classifier_memory_mb, "threads": args.classifier_threads}
    trace.emit("selection_protocol", f"TRAIN branch selection: {protocol.mode}; selection brains "
               f"{', '.join(protocol.selection_brains)}; auxiliary {', '.join(protocol.auxiliary_brains) or 'none'}",
               **protocol.describe())
    descriptor_runs, restored_descriptors = None, None
    if args.context_cache is not None:
        cache = ContextCache(args.context_cache)
        entries = {}
        for brain, bank in banks.items():
            for kind in kinds:
                entries[f"{brain}/{kind}"] = cache.entry(bank.tables[kind]).summary()
                DescriptorRuns.register_context_column(bank.tables[kind], cache.entry(bank.tables[kind]))
        descriptor_runs = DescriptorRuns(cache, args.descriptor_bank, workers=args.descriptor_workers,
                                         memory_mb=args.classifier_memory_mb, trace=trace)
        trace.emit("context_cache", f"Context cache attached for {len(entries)} brain/kind entries; "
                   f"{descriptor_runs.workers} descriptor workers", entries=entries,
                   workers=descriptor_runs.workers, bank=str(args.descriptor_bank))
        restored_descriptors = _restore_descriptors(args, banks, kinds, parent_sources, descriptor_runs, budgets, trace)
    elif any(referenced_columns(parent_sources[kind]) for kind in kinds):
        raise ValueError("The starting scorer references bank_agent_* descriptor columns; pass --context-cache")
    for kind in available:
        model = frozen_model(parent_sources[kind])
        if model is not None:
            if model['kind'] != kind:
                raise ValueError('Resumed classifier kind does not match its component')
            if set(model['training_brains']) & set(args.validation_brains):
                raise ValueError('A resumed classifier was fitted on a development-validation brain')
            if model['version'] == MODEL_VERSION:
                original_store = args.start_from_artifacts or args.start_from.parent / 'model_artifacts'
                copy_artifacts(model, original_store, artifact_store)
    trace.emit("seed_evaluation", "Evaluating seed scorer on train and validation")
    parent_state = {}
    parent_train = scoring.evaluate(source, train, budgets, args.policy_time_budget, examples=True,
                                    state=parent_state, artifact_store=artifact_store)
    log_metrics(trace, "Seed train", parent_train)
    parent_validation_state = {}
    parent_validation = scoring.evaluate(source, validation, budgets, args.policy_time_budget,
                                         state=parent_validation_state, artifact_store=artifact_store)
    log_metrics(trace, "Seed validation", parent_validation)
    # Record the original frozen-score comparator even when resuming from a custom policy.
    baseline_source = (HERE / "artifacts" / "scorer.py").read_text()
    trace.emit("baseline_evaluation", "Evaluating frozen detector-score baseline")
    baseline = {"train": scoring.evaluate(baseline_source, train, budgets, args.policy_time_budget),
                "validation": scoring.evaluate(baseline_source, validation, budgets, args.policy_time_budget)}
    log_metrics(trace, "Baseline train", baseline["train"])
    log_metrics(trace, "Baseline validation", baseline["validation"])
    baseline["auxiliary_ablation"] = _auxiliary_ablation(args, protocol, kinds, run_dir, fit_limits, trace)
    manifest = {"objective": scoring.SCORING_VERSION, "accounting_semantics": "per_generation", "budgets": budgets,
                "promotion_gate": {"version": scoring.VALIDATION_GATE_VERSION,
                                   "aggregation": scoring.AGGREGATION, "split": "development_validation",
                                   "require_train_improvement": False, "require_no_cell_regression": False,
                                   "precision_margin": args.precision_margin,
                                   "logging": {"paired_bootstrap": scoring.BOOTSTRAP_VERSION,
                                               "draws": args.bootstrap_draws, "affects_decision": False}},
                "dataset_selection": selection,
                "selection_protocol": protocol.describe(),
                "reviser_max_turns": args.reviser_max_turns,
                "feedback_format": "stratified-train-v4",
                "search_version": "agent-descriptor-compute-v14", "target_kind": args.target_kind,
                "descriptor_compute": ({"enabled": True, "context_cache": str(args.context_cache),
                    "descriptor_bank": str(args.descriptor_bank), "workers": descriptor_runs.workers,
                    "per_call_wall_seconds": args.descriptor_wall_seconds,
                    "per_generation_wall_seconds": args.descriptor_generation_wall_seconds,
                    "evaluation_units_per_uncached_call": 1, "scopes": list(DESCRIPTOR_SCOPES),
                    "entries": {f"{b}/{k}": cache.entry(bank.tables[k]).summary() for b, bank in banks.items() for k in kinds},
                    "restored_descriptors": sorted(restored_descriptors["columns"]) if restored_descriptors else []}
                    if descriptor_runs is not None else {"enabled": False}),
                "volume_analysis": {"version": VOLUME_ANALYSIS_VERSION,
                    "primary_image_exploration": descriptor_runs is None,
                    "executions_per_generation": MAX_ANALYSES, "candidates_per_execution": MAX_CASES,
                    "max_result_bytes": MAX_RESULT_BYTES, "worker_timeout_seconds": args.policy_time_budget,
                    "memory_mb": args.classifier_memory_mb, "threads": args.classifier_threads,
                    "uses_scoring_budget": False, "preview_required": False},
                "candidate_images": {"version": IMAGE_VERSION, "inspections_per_generation": 8,
                    "alignment_receipts": "image_alignment.json", "worker_stage_bytes": 1024**3,
                    "cache_bytes": image_store.max_cache_bytes, "raw_patch_models": True,
                    "selection_modes": ["top", "top_k_boundary"], "max_candidates": 4096,
                    "numeric_extraction": "bounded row/byte batches under a shared timeout",
                    "default_batch_rows": 32, "default_batch_mb": 256, "default_total_mb": 4096,
                    "format": "OME-Zarr v2 with explicit axes, units and transforms"},
                "feature_discovery": {"version": "paired-train-feature-ablation-v2",
                    "internal_folds": 3, "split_spatial_block_um": 500,
                    "purge_shared_fragments": True, "classifier_evaluation_units": CLASSIFIER_UNITS,
                    "formula_evaluation_units": FORMULA_UNITS, "affects_promotion_gate": False,
                    "partition": "shared with the selection protocol"},
                "hypothesis_memory": "train-hypothesis-evidence-v1",
                "local_context": {"version": CONTEXT_VERSION, "train_inspections_per_generation": 8,
                                  "feature_time_budget_per_brain": args.policy_time_budget,
                                  "classifier_feature_time_budget_per_brain": args.classifier_time_budget},
                "candidate_selection": "train-champion-and-specialists-v1",
                "classifier_time_budget": args.classifier_time_budget,
                "classifier_memory_mb": args.classifier_memory_mb,
                "classifier_threads": args.classifier_threads,
                "classifier_training_protocol": (
                    "TRAIN-only fit; grouped out-of-fold selection on " + ",".join(protocol.selection_brains)
                    + " (one unit per configuration: fold refits plus full fit); in-sample TRAIN diagnostic; "
                    "frozen validation inference" if protocol.requires_fold_fits else
                    "TRAIN-only fit; in-sample TRAIN ranking; frozen validation inference"),
                "train_evaluations_per_generation": args.train_evaluations_per_generation,
                "repair_attempts": 1,
                "candidate_pool_size": args.candidate_pool_size,
                "candidate_score_tolerance": args.candidate_score_tolerance,
                "plateau_patience": args.plateau_patience, "exploration_patience": args.exploration_patience,
                "explore_every": args.explore_every, "early_stop": not args.no_early_stop,
                "precision_margin": args.precision_margin, "train_brains": args.train_brains,
                "development_validation_brains": args.validation_brains, "selected_detectors": selected,
                "pools": {b: {k: t.meta for k, t in bank.tables.items()} for b, bank in banks.items()},
                "evaluator_sha256": pc._sha256(Path(scoring.__file__)),
                "harness_sha256": {p.name: pc._sha256(p) for p in sorted((HERE / 'harness').glob('*.py'))},
                "driver_sha256": pc._sha256(Path(__file__)),
                "note": "Native-label precision, not proof of repair efficacy; seccomp formula workers and Landlock/seccomp model workers"}
    _write(run_dir / "manifest.json", manifest)
    _write(run_dir / "baseline.json", baseline)
    _write(run_dir / "seed.json", {"train": parent_train, "validation": parent_validation})
    rules = {kind: CONTRACT for kind in kinds}
    memory = ExperimentMemory(run_dir, train=train, protocol=protocol, fit_limits=fit_limits,
                              descriptor_runs=descriptor_runs)
    if restored_descriptors is not None:
        memory.descriptors.update(restored_descriptors['columns'])
        memory.descriptor_programs.update(restored_descriptors['programs'])
        memory.descriptor_generation += 1
    pool = CandidatePool(run_dir / 'candidate_pool.json', kinds, size=args.candidate_pool_size,
                         score_tolerance=args.candidate_score_tolerance, plateau_patience=args.plateau_patience,
                         exploration_patience=args.exploration_patience, explore_every=args.explore_every,
                         early_stop=not args.no_early_stop, coverage=memory.coverage,
                         hypotheses=memory.hypotheses, research=memory.research,
                         image_kinds=[kind for kind in kinds if any(
                             registered_image_available(bank.tables[kind])
                             for bank in train.values())])
    parent_selection = {}
    for kind in kinds:
        entry = memory.seed(kind, parent_sources[kind], rules[kind], parent_train, parent_state)
        parent_selection[kind] = load_selection(entry['snapshot'])
        trace.emit('seed_selection', f"Seed {kind}: {protocol.mode} selection precision "
                   f"{entry['target_precision']:.6f}; in-sample {entry['in_sample_precision']:.6f}",
                   target_kind=kind, selection=entry['selection'], in_sample_precision=entry['in_sample_precision'])
        pool.add(entry)
        pool.set_reference(entry)
    pool.save()
    update_report(run_dir, trace, phase='seed_evaluated')
    trace.emit('candidate_selection', 'TRAIN pool retains the overall champion, complementary-positive/slice '
               'specialists, then near-best alternatives; specialists bypass the precision tolerance. '
               'Each kind also protects its current accepted component outside the TRAIN pool and '
               'follows a newly supplied reference in the next same-kind round, then uses a three-round cadence. '
               'Stalled exploration requires a new measured TRAIN comparison. '
               'All branches share the fixed global K; only TRAIN aggregates are exposed to the reviser.',
               selection=pool.summary()['selection'], coverage=memory.coverage.definitions())
    statistics = {kind: feature_statistics(train, kind) for kind in kinds}
    statistics_version = {kind: memory.descriptor_generation for kind in kinds}
    descriptor_budget = {'per_call_seconds': args.descriptor_wall_seconds,
                         'generation_seconds': args.descriptor_generation_wall_seconds}
    model_environment = {
        'packages': dict(sorted((dist.metadata['Name'], dist.version)
                               for dist in importlib.metadata.distributions() if dist.metadata['Name'])),
        'device': 'cpu', 'network': False, 'child_processes': False,
        'fit_timeout_seconds': args.classifier_time_budget, 'fit_memory_mb': args.classifier_memory_mb,
        'predict_timeout_seconds': args.policy_time_budget, 'predict_memory_mb': 8192,
        'volume_analysis_timeout_seconds': args.policy_time_budget,
        'volume_analysis_memory_mb': args.classifier_memory_mb,
        'numerical_threads': args.classifier_threads, 'max_artifact_bytes': 128 * 1024**2,
        'max_artifact_files': 256, 'sandbox': 'Linux Landlock ABI >= 3 and libseccomp; fail closed',
        'descriptor_compute': ({'enabled': True, 'workers': descriptor_runs.workers,
            'worker_memory_mb': args.classifier_memory_mb, 'threads_per_worker': 1,
            'per_call_wall_seconds': args.descriptor_wall_seconds,
            'per_generation_wall_seconds': args.descriptor_generation_wall_seconds,
            'evaluation_units_per_uncached_call': 1, 'scopes': list(DESCRIPTOR_SCOPES),
            'cached_band': {f'{b}/{k}': cache.entry(bank.tables[k]).summary()
                            for b, bank in train.items() for k in kinds},
            'column_prefix': COLUMN_PREFIX, 'context_column': CONTEXT_COLUMN}
            if descriptor_runs is not None else {'enabled': False}),
    }
    (run_dir / "best_scorer.py").write_text(source)
    completed_generations, stop_reason = 0, 'generation_limit'
    for generation in range(1, args.generations + 1):
        plan = pool.next_plan()
        if plan is None:
            stop_reason = 'all_kinds_stalled'
            trace.emit('search_stopped', 'All target kinds exhausted plateau exploration; retaining accepted scorer')
            break
        generation_started = time.monotonic()
        gen_dir = run_dir / f"gen{generation:03d}"
        gen_dir.mkdir()
        gen_trace = Trajectory(gen_dir)
        target_kind = plan['target_kind']
        branch = plan['search_parent']
        branch_dir = Path(branch['snapshot'])
        policy_path, rules_path = gen_dir / "scorer.py", gen_dir / "rules.md"
        policy_path.write_text((branch_dir / 'scorer.py').read_text())
        rules_path.write_text((branch_dir / 'rules.md').read_text())
        (gen_dir / "parent_scorer.py").write_text(policy_path.read_text())
        (gen_dir / "parent_rules.md").write_text(rules_path.read_text())
        (gen_dir / "accepted_scorer.py").write_text(parent_sources[target_kind])
        (gen_dir / "parent_combined_scorer.py").write_text(source)
        _write(gen_dir / 'search_plan.json', plan)
        _write(gen_dir / 'candidate_pool.json', pool.summary())
        initial_proposal = {'hypothesis': '', 'strategy': '', 'family': branch['family'], 'parameter_grid': {}}
        if 'research' in branch:
            initial_proposal['research'] = deepcopy(branch['research'])
        branch_model = frozen_model(policy_path.read_text())
        program = ((branch_model or {}).get('program') or (HERE / 'artifacts' / 'training.py').read_text())
        (gen_dir / 'training.py').write_text(program)
        (gen_dir / 'parent_training.py').write_text(program)
        if branch_model is not None and branch_model['version'] == MODEL_VERSION:
            initial_proposal['classifier'] = branch_model['config']
        _write(gen_dir / 'proposal.json', initial_proposal)
        _write(gen_dir / 'model_environment.json', model_environment)
        (gen_dir / 'classifier_guide.md').write_text((HERE / 'artifacts' / 'classifier_guide.md').read_text())
        (gen_dir / 'local_context_guide.md').write_text((HERE / 'artifacts' / 'local_context_guide.md').read_text())
        (gen_dir / 'feature_discovery_guide.md').write_text((HERE / 'artifacts' / 'feature_discovery_guide.md').read_text())
        (gen_dir / 'image_context_guide.md').write_text((HERE / 'artifacts' / 'image_context_guide.md').read_text())
        (gen_dir / 'volume_analysis_guide.md').write_text((HERE / 'artifacts' / 'volume_analysis_guide.md').read_text())
        (gen_dir / 'analysis.py').write_text((HERE / 'artifacts' / 'analysis.py').read_text())
        if descriptor_runs is not None:
            (gen_dir / 'descriptor_guide.md').write_text((HERE / 'artifacts' / 'descriptor_guide.md').read_text())
            (gen_dir / 'descriptor.py').write_text((HERE / 'artifacts' / 'descriptor_template.py').read_text()
                                                   .replace("'kind': 'split'", f"'kind': '{target_kind}'"))
        gen_trace.emit("generation_start", f"Generation {generation}/{args.generations}; "
                       f"target={target_kind}; mode={plan['mode']}; reason={plan['reason']}; "
                       f"branch role={plan['branch_role']}; "
                       f"search parent={branch['experiment']}")
        if plan.get('investigation') or plan.get('followup_required'):
            gen_trace.emit('research_assignment',
                f"Required comparison={bool(plan.get('investigation'))}; "
                f"source={(plan.get('investigation') or {}).get('required_source') or 'any'}; "
                f"fresh follow-up from assigned branch={plan.get('followup_required', False)}",
                investigation=plan.get('investigation'), followup_required=plan.get('followup_required'))
        update_report(run_dir, trace, phase='generation', generation=generation)
        log_metrics(gen_trace, "Parent train", parent_train)
        log_metrics(gen_trace, "Parent validation", parent_validation)
        report_path = gen_dir / "train_feedback.json"
        # Examples come from the accepted parent's official selection measurement
        # (out-of-fold on selection brains); in-sample TRAIN metrics stay diagnostic.
        selection_report, selection_state = parent_selection[target_kind]
        feedback_selection = deepcopy(selection_report)
        for cell, values in selection_state.items():
            brain = cell.rsplit('/', 1)[0]
            table = train[brain].tables[target_kind]
            feedback_selection['cells'][cell].update(describe(
                table, values['scores'], values['chosen'], generation, cell,
                features=augmented_features(parent_sources[target_kind], table, args.policy_time_budget)))
        statistics_path = gen_dir / 'feature_statistics.json'
        if statistics_version.get(target_kind) != memory.descriptor_generation:
            # Registered descriptor columns join the whole-pool label-conditional statistics.
            statistics[target_kind] = feature_statistics(train, target_kind)
            statistics_version[target_kind] = memory.descriptor_generation
        statistics_path.write_text(render_compact_json(statistics[target_kind]) + '\n')
        feedback_info = write_train_feedback(report_path, parent_train, [], budgets,
                                             target_kind=target_kind, memory=memory.search(target_kind),
                                             statistics_file=statistics_path, selection=feedback_selection,
                                             protocol=protocol.describe(target_kind))
        gen_trace.emit("feedback_ready", f"Compact TRAIN feedback: {feedback_info['bytes']} bytes; "
                       f"up to {feedback_info['examples_per_group_limit']} examples per group/cell",
                       **feedback_info)
        experiments = TrainingExperiments(
            gen_dir, policy_path, rules_path, target_kind, parent_sources, train, parent_train,
            parent_state, budgets, args.policy_time_budget, args.precision_margin, memory,
            max_evaluations=args.train_evaluations_per_generation, generation=generation, search_plan=plan,
            classifier_timeout=args.classifier_time_budget, classifier_memory_mb=args.classifier_memory_mb,
            classifier_threads=args.classifier_threads, parent_selection=parent_selection[target_kind],
            descriptor_budget=descriptor_budget)
        record = {"generation": generation, "accepted": False, "target_kind": target_kind,
                  "search_mode": plan['mode'], "search_reason": plan['reason'],
                  "search_branch_role": plan['branch_role'],
                  "search_parent": branch['experiment'],
                  "objective": scoring.SCORING_VERSION, "accounting_semantics": "per_generation",
                  "selection_protocol": protocol.mode,
                  "parent_selection_precision": parent_selection[target_kind][0]["macro_precision"],
                  "parent_validation_precision": parent_validation["macro_precision"],
                  "descriptors_referenced": [], "descriptor_inference": [],
                  "parent_policy_sha256": hashlib.sha256(source.encode()).hexdigest(),
                  "candidate_policy_sha256": None, "train": None, "validation": None}
        stage = "revision"
        try:
            record["reviser"] = await revise_fn(run_dir, policy_path, rules_path, report_path, args.model,
                                                experiments=experiments)
            _write(gen_dir / "reviser_result.json", record["reviser"])
            gen_trace.emit("revision_complete", record["reviser"].get("summary") or "Revision completed",
                           usage=record["reviser"].get("usage"), cost_usd=record["reviser"].get("cost_usd"))
            stage = "submission_check"
            submitted, candidate_rules = experiments.submitted()
            candidate_source = submitted['source']
            record["candidate_policy_sha256"] = hashlib.sha256(candidate_source.encode()).hexdigest()
            record['submitted_experiment'] = submitted['entry']['experiment']
            record['candidate_type'] = submitted['entry'].get('candidate_type', 'formula')
            record['classifier_training'] = submitted['entry'].get('training')
            record['selection'] = submitted['entry'].get('selection')
            record['selection_gate'] = submitted['entry'].get('selection_gate')
            record['in_sample_precision'] = submitted['entry'].get('in_sample_precision')
            candidate_train = submitted['report']
            record["train"] = candidate_train
            log_metrics(gen_trace, "Candidate train", candidate_train, parent_train)
            improves_train, reason = scoring.acceptance(parent_train, candidate_train, args.precision_margin)
            record["train_gate"] = {"passed": improves_train, "reason": reason,
                                    "required_for_promotion": False}
            stage = 'research_check'
            research_status = experiments.research_status()
            if not research_status['complete']:
                raise RuntimeError(research_status['requirement'] +
                    '; assigned-branch follow-up complete=' + str(research_status['followup_complete']))
            stage = "validation_descriptors"
            record['descriptors_referenced'] = experiments.required_descriptors(candidate_source)
            if record['descriptors_referenced']:
                record['descriptor_inference'] = experiments.ensure_descriptors(validation, record['descriptors_referenced'])
                gen_trace.emit(stage, f"Computed {len(record['descriptors_referenced'])} registered descriptor columns "
                               f"on {len(validation)} other brains for frozen inference",
                               inference=record['descriptor_inference'])
            stage = "validation_evaluation"
            gen_trace.emit(stage, f"Measured submission; evaluating on all {len(validation)} validation brains "
                           "regardless of TRAIN gain")
            candidate_validation_state = {}
            candidate_validation = scoring.evaluate(candidate_source, validation, budgets, args.policy_time_budget,
                                                    state=candidate_validation_state, artifact_store=artifact_store)
            record["validation"] = candidate_validation
            log_metrics(gen_trace, "Candidate validation", candidate_validation, parent_validation)
            accepted, reason = scoring.acceptance(parent_validation, candidate_validation, args.precision_margin,
                                                  require_no_cell_regression=False)
            bootstrap = None
            if args.bootstrap_draws > 0:
                try:
                    bootstrap = scoring.paired_bootstrap(parent_validation_state, candidate_validation_state,
                                                         validation, budgets, draws=args.bootstrap_draws,
                                                         seed=generation)
                except Exception as exc:  # Logging only; never blocks or changes the decision.
                    bootstrap = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
                gen_trace.emit("promotion_uncertainty", "Paired bootstrap of the validation delta recorded "
                               "(does not affect the gate)", paired_bootstrap=bootstrap)
            record["validation_gate"] = {"version": scoring.VALIDATION_GATE_VERSION,
                                         "passed": accepted, "reason": reason,
                                         "aggregation": scoring.AGGREGATION,
                                         "brains": list(args.validation_brains),
                                         "parent_mean": parent_validation["macro_precision"],
                                         "candidate_mean": candidate_validation["macro_precision"],
                                         "regressed_cells": [cell for cell, before in parent_validation["cells"].items()
                                             if candidate_validation["cells"][cell]["precision"] < before["precision"]],
                                         "paired_bootstrap": bootstrap}
            record.update(accepted=accepted, reason="VALIDATION: " + reason)
            if accepted:
                source, rules[target_kind] = candidate_source, candidate_rules
                parent_sources[target_kind] = submitted['component']
                parent_state = submitted['state']
                parent_selection[target_kind] = (submitted['selection_report'], submitted['selection_state'])
                parent_validation_state = candidate_validation_state
                parent_train, parent_validation = candidate_train, candidate_validation
                (run_dir / "best_scorer.py").write_text(source)
                if record['descriptors_referenced']:
                    _write(run_dir / "descriptors.json", experiments.descriptor_registry(record['descriptors_referenced']))
                elif (run_dir / "descriptors.json").exists():
                    (run_dir / "descriptors.json").unlink()
        except Exception as exc:
            # A failed/timeout policy is never substituted with a baseline measurement.
            record["reason"] = f"Failed: {type(exc).__name__}: {exc}"
            record["failure_stage"] = stage
            if stage == 'research_check':
                record['reason'] = f'Research assignment incomplete: {exc}'
            elif stage in {'revision', 'submission_check'}:
                experiments._record_research(stage, 'execution', 'failure',
                    {'error': f'{type(exc).__name__}: {exc}'}, False,
                    gen_dir / 'trajectory.jsonl', outcome=f'{type(exc).__name__}: {exc}')
            gen_trace.emit("generation_error", record["reason"], stage=stage)
            # Preserve usage/cost even when the SDK's final result reports failure.
            result_path = gen_dir / "reviser_result.json"
            if "reviser" not in record and result_path.exists():
                record["reviser"] = json.loads(result_path.read_text())
        except BaseException as exc:
            gen_trace.emit("generation_interrupted", f"{type(exc).__name__}: {exc}", stage=stage)
            raise
        finally:
            save_diffs(gen_dir)
        record['experiments'] = [attempt['entry'] for attempt in experiments.attempts]
        record['feature_diagnostics'] = experiments.feature_diagnostics
        record['image_evidence'] = generation_image_evidence(gen_dir, record)
        _write(gen_dir / 'image_evidence.json', record['image_evidence'])
        record['train_evaluations_used'] = experiments.evaluations_used
        record['descriptor_runs'] = experiments.descriptor_log
        record['descriptor_seconds_used'] = experiments.descriptor_seconds_used
        record['research_status'] = experiments.research_status()
        receipt = record['research_status']
        gen_trace.emit('research_result', f"Research {receipt['status']}; "
                       f"new measurements={receipt['new_measurements']}; reused={receipt['reused_results']}; "
                       f"execution failures={receipt['execution_failures']}; "
                       f"comparison complete={receipt['investigation_complete']}; "
                       f"branch follow-up complete={receipt['followup_complete']}", **receipt)
        memory.hypotheses.finish_generation(generation, record['experiments'])
        experiments._save_hypothesis_feedback()
        record['search_progress'] = pool.finish(
            plan, record['experiments'],
            reference_entry=submitted['entry'] if record['accepted'] else None,
            research_status=record['research_status'])
        if record['accepted']:
            gen_trace.emit('reference_updated', f"Protected {target_kind} reference: "
                           f"{submitted['entry']['experiment']}; next same-kind round receives priority",
                           target_kind=target_kind, experiment=submitted['entry']['experiment'])
        _write(gen_dir / 'candidate_pool_after.json', pool.summary())
        completed_generations += 1
        progress = record['search_progress']
        gen_trace.emit('search_progress', f"{target_kind}: {len(pool.members[target_kind])} retained branches; "
                       f"new TRAIN positives covered={progress['new_positive_hits']}; "
                       f"improved TRAIN slices={len(progress['improved_slices'])}; "
                       f"stalled rounds={progress['stalled_rounds']}; "
                       f"rounds without new evidence={progress['no_evidence_rounds']}; "
                       f"research={record['research_status']['status']}; "
                       f"stalled explorations={progress['stalled_explorations']}; paused={progress['paused']}",
                       **progress)
        record["wall_seconds"] = time.monotonic() - generation_started
        _write(gen_dir / "evaluation.json", record)
        with (run_dir / "ledger.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        outcome = ('incomplete' if record.get('failure_stage') == 'research_check' else
                   'failed' if record.get('failure_stage') else 'accepted' if record['accepted'] else 'rejected')
        decision = (f"Generation {generation}: {outcome}; "
                    f"{record['reason']}; {record['wall_seconds']:.1f}s; "
                    f"{'new parent saved' if record['accepted'] else 'parent retained'}; "
                    f"exploration pool={len(pool.members[target_kind])}, "
                    f"new TRAIN positives={progress['new_positive_hits']}")
        gen_trace.emit("decision", decision, accepted=record["accepted"],
                       wall_seconds=record["wall_seconds"], reason=record["reason"])
        trace.emit("generation_complete", decision, trajectory=str(gen_dir / "trajectory.txt"))
        update_report(run_dir, trace, phase='generation_complete', generation=generation)
    _write(run_dir / "final.json", {"train": parent_train, "validation": parent_validation})
    log_metrics(trace, "Final accepted train", parent_train)
    log_metrics(trace, "Final accepted validation", parent_validation)
    if all(p['paused'] for p in pool.progress.values()):
        stop_reason = ('all_kinds_stalled' if all(p['pause_reason'] == 'measured_stagnation'
                       for p in pool.progress.values()) else 'all_kinds_paused')
    _write(run_dir / 'search_summary.json', {'stop_reason': stop_reason,
                                           'completed_generations': completed_generations, **pool.summary()})
    trace.emit("run_complete", f"Fixed-pool scorer run: {run_dir}")
    update_report(run_dir, trace, status='completed', stop_reason=stop_reason,
                  completed_generations=completed_generations)
    return run_dir


def main(argv=None):
    asyncio.run(run(parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

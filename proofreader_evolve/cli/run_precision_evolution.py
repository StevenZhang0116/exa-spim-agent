"""Evolve a scorer on immutable AutoDiscovery detector-native candidate pools.

No graph edits, candidate expansion, original detector retraining or table preparation.
Default TRAIN: 794495. Validation: all other eligible, prepared local datasets.
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
from ..harness.train_feedback import write_train_feedback, render_compact_json
from ..harness.scorer_components import components
from ..harness.train_experiments import ExperimentMemory, TrainingExperiments
from ..harness.candidate_pool import CandidatePool
from ..harness.classifier_contract import frozen_model, MODEL_VERSION, copy_artifacts
from ..harness.training_diagnostics import describe, feature_statistics


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
            'model_environment.json')]
    options = bind_session_options(
        options, policy_path, rules_path, report_path,
        readable_paths=readable,
        training_server=experiments.mcp_server() if experiments else None)
    result = None
    prompt = (
        f"Read TRAIN feedback at {report_path}. Improve {policy_path} and update {rules_path}. "
        "Follow the fixed-pool scorer contract. Do not change the candidate pool, budgets or evaluator. "
        f"You have at most {max_turns} SDK turns. The compact feedback is at most 24 KB; read it once. "
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
            f"Read {policy_path.parent / 'search_plan.json'} and {policy_path.parent / 'candidate_pool.json'}. "
            f"This is a {experiments.search_plan['mode']} generation. In tune mode ONLY change numeric values "
            "in the top-level PARAMS dictionary, or refit numeric hyperparameters of the assigned classifier. "
            "The formula or training program and nonnumeric parameters stay fixed. In explore mode "
            "freely redesign the training program, model, preprocessing, loss, ensemble or feature selection. "
            "Put a testable hypothesis, strategy, short family name and parameter_grid in proposal.json. "
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
            "TRAIN classifier feedback is IN-SAMPLE resubstitution, not an independent generalization estimate. "
            f"Budget: {experiments.max_evaluations} NEW configurations total across all evaluation and fitting tools; "
            "one fit plus its TRAIN scoring costs one configuration, including failed fits. "
            "one extra attempt follows the first execution error. Cached repeats and proposal-format errors "
            "do not spend evaluation budget, but tool calls still consume SDK turns. Grid combinations must fit "
            "the remaining budget. Prefer one grid call to hand-editing constants across many calls. "
            "Inspect measured precision, gained/lost positives and top-K overlap before revising again. "
            "The initial scorer is the selected TRAIN exploration branch. parent_train in feedback and all "
            "comparisons refer to the accepted parent, which can differ from this branch. "
            'Use mcp__training__restore_candidate({"candidate_id":"parent"}) to measure/recover the assigned '
            'branch from cache, or {"candidate_id":"best"} to restore the best measured candidate of this generation. '
            "IDs such as gen002/attempt001 restore earlier measured branches without rewriting their source. "
            "Finish with scorer.py exactly matching a successfully evaluated snapshot. An untested final edit is rejected. "
            "Summarize observed TRAIN evidence and "
            "remaining uncertainty in rules.md. Validation is unavailable to this session."
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
    p.add_argument("--mcl", type=int, default=100)
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--merge-k", type=int, default=100)
    p.add_argument("--split-k", type=int, default=100)
    p.add_argument("--precision-margin", type=float, default=0,
                   help="Required absolute gain in equal-weight mean validation Precision@K (default: 0)")
    p.add_argument("--policy-time-budget", type=float, default=120)
    p.add_argument("--classifier-time-budget", type=float, default=300,
                   help="Wall/CPU fitting budget per classifier configuration, in seconds (default: 300)")
    p.add_argument("--classifier-memory-mb", type=int, default=8192,
                   help="Address-space limit of the isolated model fitting worker (default: 8192 MiB)")
    p.add_argument("--classifier-threads", type=int, default=1,
                   help="Numerical library CPU thread setting for fit/predict (default: 1)")
    p.add_argument("--start-from-artifacts", type=Path,
                   help="Model artifact store for --start-from (default: sibling model_artifacts directory)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--reviser-max-turns", type=int, default=DEFAULT_REVISER_MAX_TURNS,
                   help="Maximum SDK interaction turns per generation (default: 24)")
    p.add_argument("--train-evaluations-per-generation", type=int, default=8,
                   help="New TRAIN configurations, plus one execution-error repair (default: 8); cache hits are free")
    p.add_argument("--candidate-pool-size", type=int, default=5,
                   help="Maximum retained exploration branches per kind (default: 5)")
    p.add_argument("--candidate-score-tolerance", type=float, default=.02,
                   help="Maximum absolute TRAIN precision gap below the pool champion (default: 0.02)")
    p.add_argument("--plateau-patience", type=int, default=2,
                   help="Measured rounds without TRAIN improvement before forced exploration (default: 2)")
    p.add_argument("--exploration-patience", type=int, default=2,
                   help="Unsuccessful forced explorations before pausing a kind (default: 2)")
    p.add_argument("--explore-every", type=int, default=3,
                   help="Explore at least every N measured rounds of a kind (default: 3)")
    p.add_argument("--no-early-stop", action="store_true",
                   help="Keep exploring stalled kinds until the generation limit")
    p.add_argument("--target-kind", choices=("alternate", "merge", "split"), default="alternate",
                   help="Component to revise; alternate changes one kind per generation")
    p.add_argument("--feature-tables-dir", type=Path, default=pc.DEFAULT_OUT)
    p.add_argument("--merge-dir", action="append")
    p.add_argument("--split-dir", action="append")
    p.add_argument("--start-from", type=Path, default=HERE / "artifacts" / "scorer.py")
    p.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    args = p.parse_args(argv)
    if set(args.train_brains) & set(args.validation_brains or []):
        p.error("Train and validation brains must be disjoint")
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
    auto = args.validation_brains is None
    selection = {"mode": "auto" if auto else "explicit", "train_brains": list(args.train_brains),
                 "detector_training_brains": sorted(fitted_brains), "excluded": {}}
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
    # All later generations use these already loaded banks, never rediscovering datasets.
    args.validation_brains = validation_brains
    trace.emit("dataset_split", f"TRAIN: {', '.join(args.train_brains)}; validation ({len(validation_brains)}): "
               f"{', '.join(validation_brains)}; split fixed for this run", **selection)
    return banks, selection


async def run(args, revise_fn=None):
    """Budgeted TRAIN search followed by mean-validation-only promotion."""
    revise_fn = partial(revise, max_turns=args.reviser_max_turns) if revise_fn is None else revise_fn
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="precision_" + datetime.now().strftime("%Y%m%d_%H%M%S_"),
                                   dir=args.runs_dir)).resolve()
    trace = Trajectory(run_dir)
    trace.emit("run_start", f"Run directory: {run_dir}",
               config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    try:
        return await _run(args, revise_fn, run_dir, trace)
    except BaseException as exc:
        trace.emit("run_failed", f"{type(exc).__name__}: {exc}")
        raise


async def _run(args, revise_fn, run_dir, trace):
    selected = pc.resolve_detector_runs(args.merge_dir, args.split_dir)
    banks, selection = _load_evaluation_banks(args, selected, trace)
    _write(run_dir / "dataset_split.json", selection)
    trace.emit("promotion_rule", "Promotion uses equal-weight mean validation Precision@K only; "
               "TRAIN improvement is diagnostic, and individual validation regressions are allowed",
               gate_version=scoring.VALIDATION_GATE_VERSION, aggregation=scoring.AGGREGATION)
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
    parent_validation = scoring.evaluate(source, validation, budgets, args.policy_time_budget, artifact_store=artifact_store)
    log_metrics(trace, "Seed train", parent_train)
    log_metrics(trace, "Seed validation", parent_validation)
    # Record the original frozen-score comparator even when resuming from a custom policy.
    baseline_source = (HERE / "artifacts" / "scorer.py").read_text()
    trace.emit("baseline_evaluation", "Evaluating frozen detector-score baseline")
    baseline = {"train": scoring.evaluate(baseline_source, train, budgets, args.policy_time_budget),
                "validation": scoring.evaluate(baseline_source, validation, budgets, args.policy_time_budget)}
    manifest = {"objective": scoring.SCORING_VERSION, "accounting_semantics": "per_generation", "budgets": budgets,
                "promotion_gate": {"version": scoring.VALIDATION_GATE_VERSION,
                                   "aggregation": scoring.AGGREGATION, "split": "development_validation",
                                   "require_train_improvement": False, "require_no_cell_regression": False,
                                   "precision_margin": args.precision_margin},
                "dataset_selection": selection,
                "reviser_max_turns": args.reviser_max_turns, "feedback_format": "stratified-train-v2",
                "search_version": "agent-model-artifacts-pool-v4", "target_kind": args.target_kind,
                "classifier_time_budget": args.classifier_time_budget,
                "classifier_memory_mb": args.classifier_memory_mb,
                "classifier_threads": args.classifier_threads,
                "classifier_training_protocol": "TRAIN-only fit; in-sample TRAIN ranking; frozen validation inference",
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
    memory = ExperimentMemory(run_dir)
    pool = CandidatePool(run_dir / 'candidate_pool.json', kinds, size=args.candidate_pool_size,
                         score_tolerance=args.candidate_score_tolerance, plateau_patience=args.plateau_patience,
                         exploration_patience=args.exploration_patience, explore_every=args.explore_every,
                         early_stop=not args.no_early_stop)
    for kind in kinds:
        pool.add(memory.seed(kind, parent_sources[kind], rules[kind], parent_train, parent_state))
    pool.save()
    statistics = {kind: feature_statistics(train, kind) for kind in kinds}
    model_environment = {
        'packages': dict(sorted((dist.metadata['Name'], dist.version)
                               for dist in importlib.metadata.distributions() if dist.metadata['Name'])),
        'device': 'cpu', 'network': False, 'child_processes': False,
        'fit_timeout_seconds': args.classifier_time_budget, 'fit_memory_mb': args.classifier_memory_mb,
        'predict_timeout_seconds': args.policy_time_budget, 'predict_memory_mb': 8192,
        'numerical_threads': args.classifier_threads, 'max_artifact_bytes': 128 * 1024**2,
        'max_artifact_files': 256, 'sandbox': 'Linux Landlock ABI >= 3 and libseccomp; fail closed',
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
        branch_model = frozen_model(policy_path.read_text())
        program = ((branch_model or {}).get('program') or (HERE / 'artifacts' / 'training.py').read_text())
        (gen_dir / 'training.py').write_text(program)
        (gen_dir / 'parent_training.py').write_text(program)
        if branch_model is not None and branch_model['version'] == MODEL_VERSION:
            initial_proposal['classifier'] = branch_model['config']
        _write(gen_dir / 'proposal.json', initial_proposal)
        _write(gen_dir / 'model_environment.json', model_environment)
        (gen_dir / 'classifier_guide.md').write_text((HERE / 'artifacts' / 'classifier_guide.md').read_text())
        gen_trace.emit("generation_start", f"Generation {generation}/{args.generations}; "
                       f"target={target_kind}; mode={plan['mode']}; reason={plan['reason']}; "
                       f"search parent={branch['experiment']}")
        log_metrics(gen_trace, "Parent train", parent_train)
        log_metrics(gen_trace, "Parent validation", parent_validation)
        report_path = gen_dir / "train_feedback.json"
        feedback_train = deepcopy(parent_train)
        for brain, bank in train.items():
            for kind, table in bank.tables.items():
                cell = f'{brain}/{kind}'
                feedback_train['cells'][cell].update(describe(
                    table, parent_state[cell]['scores'], parent_state[cell]['chosen'], generation, cell))
        statistics_path = gen_dir / 'feature_statistics.json'
        statistics_path.write_text(render_compact_json(statistics[target_kind]) + '\n')
        feedback_info = write_train_feedback(report_path, feedback_train, [], budgets,
                                             target_kind=target_kind, memory=memory.search(target_kind),
                                             statistics_file=statistics_path)
        gen_trace.emit("feedback_ready", f"Compact TRAIN feedback: {feedback_info['bytes']} bytes; "
                       f"up to {feedback_info['examples_per_group_limit']} examples per group/cell",
                       **feedback_info)
        experiments = TrainingExperiments(
            gen_dir, policy_path, rules_path, target_kind, parent_sources, train, parent_train,
            parent_state, budgets, args.policy_time_budget, args.precision_margin, memory,
            max_evaluations=args.train_evaluations_per_generation, generation=generation, search_plan=plan,
            classifier_timeout=args.classifier_time_budget, classifier_memory_mb=args.classifier_memory_mb,
            classifier_threads=args.classifier_threads)
        record = {"generation": generation, "accepted": False, "target_kind": target_kind,
                  "search_mode": plan['mode'], "search_reason": plan['reason'],
                  "search_parent": branch['experiment'],
                  "objective": scoring.SCORING_VERSION, "accounting_semantics": "per_generation",
                  "parent_validation_precision": parent_validation["macro_precision"],
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
            candidate_train = submitted['report']
            record["train"] = candidate_train
            log_metrics(gen_trace, "Candidate train", candidate_train, parent_train)
            improves_train, reason = scoring.acceptance(parent_train, candidate_train, args.precision_margin)
            record["train_gate"] = {"passed": improves_train, "reason": reason,
                                    "required_for_promotion": False}
            stage = "validation_evaluation"
            gen_trace.emit(stage, f"Measured submission; evaluating on all {len(validation)} validation brains "
                           "regardless of TRAIN gain")
            candidate_validation = scoring.evaluate(candidate_source, validation, budgets, args.policy_time_budget,
                                                    artifact_store=artifact_store)
            record["validation"] = candidate_validation
            log_metrics(gen_trace, "Candidate validation", candidate_validation, parent_validation)
            accepted, reason = scoring.acceptance(parent_validation, candidate_validation, args.precision_margin,
                                                  require_no_cell_regression=False)
            record["validation_gate"] = {"version": scoring.VALIDATION_GATE_VERSION,
                                         "passed": accepted, "reason": reason,
                                         "aggregation": scoring.AGGREGATION,
                                         "brains": list(args.validation_brains),
                                         "parent_mean": parent_validation["macro_precision"],
                                         "candidate_mean": candidate_validation["macro_precision"],
                                         "regressed_cells": [cell for cell, before in parent_validation["cells"].items()
                                             if candidate_validation["cells"][cell]["precision"] < before["precision"]]}
            record.update(accepted=accepted, reason="VALIDATION: " + reason)
            if accepted:
                source, rules[target_kind] = candidate_source, candidate_rules
                parent_sources[target_kind] = submitted['component']
                parent_state = submitted['state']
                parent_train, parent_validation = candidate_train, candidate_validation
                (run_dir / "best_scorer.py").write_text(source)
        except Exception as exc:
            # A failed/timeout policy is never substituted with a baseline measurement.
            record["reason"] = f"Failed: {type(exc).__name__}: {exc}"
            record["failure_stage"] = stage
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
        record['train_evaluations_used'] = experiments.evaluations_used
        record['search_progress'] = pool.finish(plan, record['experiments'])
        completed_generations += 1
        progress = record['search_progress']
        gen_trace.emit('search_progress', f"{target_kind}: {len(pool.members[target_kind])} retained branches; "
                       f"stalled rounds={progress['stalled_rounds']}; "
                       f"stalled explorations={progress['stalled_explorations']}; paused={progress['paused']}",
                       **progress)
        record["wall_seconds"] = time.monotonic() - generation_started
        _write(gen_dir / "evaluation.json", record)
        with (run_dir / "ledger.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        decision = (f"Generation {generation}: {'accepted' if record['accepted'] else 'rejected'}; "
                    f"{record['reason']}; {record['wall_seconds']:.1f}s; "
                    f"{'new parent saved' if record['accepted'] else 'parent retained'}")
        gen_trace.emit("decision", decision, accepted=record["accepted"],
                       wall_seconds=record["wall_seconds"], reason=record["reason"])
        trace.emit("generation_complete", decision, trajectory=str(gen_dir / "trajectory.txt"))
    _write(run_dir / "final.json", {"train": parent_train, "validation": parent_validation})
    if all(p['paused'] for p in pool.progress.values()):
        stop_reason = 'all_kinds_stalled'
    _write(run_dir / 'search_summary.json', {'stop_reason': stop_reason,
                                           'completed_generations': completed_generations, **pool.summary()})
    trace.emit("run_complete", f"Fixed-pool scorer run: {run_dir}")
    return run_dir


def main(argv=None):
    asyncio.run(run(parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

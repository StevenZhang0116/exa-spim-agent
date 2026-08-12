"""
Generate a multi-feature merge detector from a finished AutoDiscovery run.

Where ``run_discovery_workflow.py`` DIGESTS one run export into a ranked
``<RUN>.summary.md`` report, this workflow goes one step further downstream: it
takes that finished report, the reproducer's loading-fixed hypothesis scripts
(normally ``autodiscovery/<RUN>.json.predictive.rerun/hypo_<id>.py``), and any
test-fixer scripts under the matching ``.fixed/`` directory, and WRITES A DETECTOR — one
script that computes every eligible selected hypothesis's features in shared passes over
the skeleton graph, evaluates several appropriately regularized tabular models
under nested cross-validation, and fits the selected strategy to produce one
score per segment.

The motivation comes from the reports themselves: each hypothesis is a single
feature, and the recurring caveat is that none is precise enough alone
("high recall, ~8% precision — best used as one component of an ensemble").
This workflow builds the combined feature table and selects a model for it.

THIS WORKFLOW IS CODE GENERATION ONLY, and it takes NO DATASET. Its inputs are
the run export, the report beside it, the predictive-selection manifest when
applicable, the ``.rerun/`` scripts, and optional ``.fixed/`` scripts. It
cross-checks selection, rerun MANIFEST, inventory paths, and source hashes before
generation. It never loads a cache and never runs
what it writes: the pkl needs >20 GB RAM and minutes of graph traversal, so the
operator runs the finished script themselves, on a compute node, with the data.
Hence a login node is fine for this. The origin dataset is recorded in the
report, which is where the README's provenance comes from; the run command
printed at the end guesses the cache path from the run name (``...-794495-mcl100``
-> ``cache/dataset_cache_794495_mcl100_add.pkl``) purely as a convenience.

A persistent Claude session runs the ordered STEPS so later steps share earlier
context. Subagents live in ``.claude/agents/`` and are auto-discovered via
``setting_sources``. Deliverables land in
``autodiscovery-application/<RUN>/`` (one subfolder per originating run):

    merge_site_detector.py            model comparison + selected detector
    feature_inventory.json            feature definition + defined condition
    model_candidates.json             validated model families + small grids
    README.md                         provenance, how to run, how to read it
    detector_build_workflow.log.txt   this driver's console output (--log-txt)

The generated detector run additionally writes per-segment CSV scores, a JSON
model-selection manifest, a fitted winner joblib, a run log and figures.

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

The steps:
  1. inventory-features  — read <RUN>.summary.md, each selected .rerun/hypo_<id>.py and
                           matching .fixed/hypo_<id>.py when present; choose and
                           record the authoritative feature source per id,
                           and record for each returned hypothesis its feature
                           name, the exact quantity computed, its aggregation
                           unit, exact defined condition, and historical value
                           used when undefined. Those conditions drive the audit.
  2. configure-models    — write a validated model-candidate configuration. Three
                           conservative baselines are mandatory; at most two
                           inventory-justified extensions may be selected from a
                           code-owned allowlist. No arbitrary estimator imports.
  3. generate-detector   — write ONE consolidated CLI script: load the pkl once,
                           group the features by the traversal each needs
                           (edge / chain / junction / component / segment), keep
                           every constant verbatim from the inventoried source, compare
                           the configured allowlisted strategies with
                           nested stratified CV, fit the selected strategy, and
                           emit a CSV, selection manifest, log and key figures.
  4. verify-and-document — check the script STATICALLY against the inventory and
                           the inventoried rerun/fixed sources (every feature computed and fitted,
                           constants unchanged, defined flags consistent, no sandbox
                           pip preamble, parses, --help works), fix what is wrong,
                           and write README.md. No pkl, no results — the README
                           says how to run it and in what order to read the
                           numbers when they arrive.

The audit lives INSIDE the generated script because missing structure is a real
predictor and a real confound. On ``merge-error-794495-mcl100_2026-08-04``, 6396
of 9623 adjudicable segments had no fragment component and none was a merge.
Historical sentinels made that group trivially separable and lifted same-brain
cross-validated ROC-AUC. Generated detectors therefore retain NaN, add explicit
is_defined flags, report coverage by class, and let fold-local preprocessing or
native-NaN models handle the numeric value.

All four steps always run, and re-running the workflow simply regenerates the
folder: the steps are minutes of one SDK session, not the hours the compute used
to take, so partial-resume flags cost more in surface area than they save.

Usage (from the ``exa-spim-agent/`` project root; a login node is fine):
    conda activate panda
    python agentic/run_detector_build_workflow.py \
        autodiscovery/merge-error-794495-mcl100_2026-08-04.json

Then run the generated detector yourself, on a compute node:
    sbatch --mem=80G --wrap="\
        source /shared/utils.x86_64/anaconda3-2024.10/etc/profile.d/conda.sh; \
        conda activate panda; \
        python autodiscovery-application/<RUN>/merge_site_detector.py \
            cache/dataset_cache_<brain>_mcl<N>_add.pkl"
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)

# ToolUseBlock is what gives us live progress (which tool/subagent is running).
# Import defensively so a minor SDK version mismatch doesn't break the script.
try:
    from claude_agent_sdk import ToolUseBlock
except ImportError:  # pragma: no cover - depends on installed SDK version
    ToolUseBlock = ()  # type: ignore[assignment]


# Project root = the directory that holds .claude/, agentic/, autodiscovery/.
# agentic/run_detector_build_workflow.py -> parent.parent is the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Applications live one subfolder per originating run.
APPLICATION_DIR_REL = "autodiscovery-application"

# The detector script the generate step writes — the deliverable of this workflow.
DETECTOR_NAME = "merge_site_detector.py"
MODEL_CONFIG_NAME = "model_candidates.json"

# The agent may choose a bounded subset, but it cannot introduce executable
# imports/class paths. These names and parameter surfaces are the security and
# reproducibility boundary shared by the driver and generated detector.
REQUIRED_MODEL_FAMILIES = frozenset({
    "logistic_l2", "logistic_elasticnet", "hist_gradient_boosting",
})
OPTIONAL_MODEL_FAMILIES = frozenset({
    "spline_logistic", "explainable_boosting", "extra_trees",
    "random_forest", "xgboost",
})
ALLOWED_MODEL_FAMILIES = REQUIRED_MODEL_FAMILIES | OPTIONAL_MODEL_FAMILIES
MODEL_PARAMETER_ALLOWLIST = {
    "logistic_l2": frozenset({"C"}),
    "logistic_elasticnet": frozenset({"C", "l1_ratio"}),
    "hist_gradient_boosting": frozenset({
        "learning_rate", "max_leaf_nodes", "min_samples_leaf",
        "l2_regularization",
    }),
    "spline_logistic": frozenset({"n_knots", "degree", "C"}),
    "explainable_boosting": frozenset({
        "max_bins", "learning_rate", "max_rounds", "min_samples_leaf",
    }),
    "extra_trees": frozenset({
        "n_estimators", "max_depth", "min_samples_leaf", "max_features",
    }),
    "random_forest": frozenset({
        "n_estimators", "max_depth", "min_samples_leaf", "max_features",
    }),
    "xgboost": frozenset({
        "n_estimators", "max_depth", "learning_rate", "min_child_weight",
        "subsample", "colsample_bytree", "reg_lambda",
    }),
}
MODEL_REQUIRED_PACKAGE = {
    "explainable_boosting": "interpret",
    "xgboost": "xgboost",
}
MODEL_NATIVE_NAN = {
    "logistic_l2": False,
    "logistic_elasticnet": False,
    "hist_gradient_boosting": True,
    "spline_logistic": False,
    "explainable_boosting": True,
    "extra_trees": False,
    "random_forest": False,
    "xgboost": True,
}
MAX_OPTIONAL_MODELS = 2
MAX_GRID_COMBINATIONS = 24

# A step that hangs should fail loudly rather than stall the workflow forever.
AGENT_STEP_TIMEOUT_S: int = 1800   # 30 min — the generate step writes a lot of code


DRIVER_LOG_NAME = "detector_build_workflow.log.txt"


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
    return run_json.name[: -len(".json")] if run_json.name.endswith(".json") else run_json.stem


def origin_cache_hint(run_json: Path) -> str | None:
    """Guess the origin cache path from the run name, or None if it doesn't say.

    ``merge-error-794495-mcl100_2026-08-04`` -> ``cache/dataset_cache_794495_
    mcl100_add.pkl``. Only a hint, for the command this workflow prints at the
    end: the authority on which dataset the run was generated on is the report
    itself, and older run exports (``ground-truth-error-annotations-...``) carry
    no brain id in the name at all.
    """
    m = re.search(r"(\d{5,7})[-_]mcl(\d+)", run_stem(run_json))
    if not m:
        return None
    return f"cache/dataset_cache_{m.group(1)}_mcl{m.group(2)}_add.pkl"


def _rel_to_root(path: Path) -> str:
    """Path relative to PROJECT_ROOT (the session cwd), else absolute.

    These strings go into agent instructions and into the run command printed at
    the end, so they have to be usable as typed. A target outside the project —
    an ``--out-dir`` under /tmp, say — relativizes to a chain of ``..`` that is
    correct but unreadable and breaks the moment it is pasted from elsewhere, so
    fall back to the absolute path there.
    """
    rel = Path(os.path.relpath(path, PROJECT_ROOT))
    return path.as_posix() if rel.parts and rel.parts[0] == ".." else rel.as_posix()


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
) -> list[dict]:
    """Build the model-selection detector workflow instructions."""
    detector_rel = f"{out_dir_rel}/{DETECTOR_NAME}"
    inventory_rel = f"{out_dir_rel}/feature_inventory.json"
    model_config_rel = f"{out_dir_rel}/{MODEL_CONFIG_NAME}"
    readme_rel = f"{out_dir_rel}/README.md"
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

    return [
        {
            "name": "inventory-features",
            "expects_file": inventory_rel,
            "instruction": f"""
Use the discovery-detector-builder subagent to inventory the features of the
finished AutoDiscovery run {run_rel}. Read the ranked report {summary_rel}, every
loading-fixed hypothesis script under {rerun_rel}/ as indexed by MANIFEST.json,
any same-id corrected script, and markdowns/labeled_dataset_cache.md.
{fixed_note}
{selection_note}
Inventory exactly those ids: do not add an unselected script merely because it
is present on disk, and do not omit a selected id.

For each returned hypothesis, first determine whether it is eligible:
- Exclude reproduction FAILED/UNUSABLE unless a later corrected run produced a
  usable measurement that repairs the relevant failure.
- Exclude an uncorrected CRITICAL statistical verdict, an OVERTURNED corrected
  verdict, or a corrected result whose effect vanished.
- Keep DIVERGED only with its divergence recorded explicitly; keep WEAK/MINOR
  because multivariate validation can still find independent signal.

For every same-id pair, first require a matching corrected measurement/verdict
in the report. A fixed file without that evidence is stale_fixed_ignored: record
it, ignore it, and use rerun only if the hypothesis is otherwise eligible. A
report correction whose fixed script is missing is unclear and must be excluded.
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

Write {inventory_rel} with this exact versioned shape: a top-level object with
schema_version=2, selection_manifest (path or null), selected_ids, and hypotheses
(one object per selected id). Each hypothesis object records id, included,
exclusion_reason, reproduction_status, statistical_verdict,
corrected_result_status, post_correction_verdict, correction_scope (none|test_only|
feature_semantics_changed|stale_fixed_ignored|unclear), rerun_path,
rerun_sha256, fixed_path/fixed_sha256 (null when absent), feature_source
(rerun|fixed|null), feature_source_path/feature_source_sha256 (null when
excluded), source_reason, and features. `features` is a non-empty list for an
included hypothesis and may hold multiple feature-column definitions; each item
uses exactly these keys: name, quantity, constants, aggregation, reduction,
traversal_phase, measurable_condition, historical_undefined_sentinel.
Excluded hypotheses have null feature-source fields and an empty features list. Every included hypothesis
has exactly one authoritative source path and matching SHA-256. The generated
detector represents undefined as NaN plus an explicit <feature>_is_defined
column. Report all exclusions and reasons. Do not load a pkl or invent results.
""".strip(),
        },
        {
            "name": "configure-models",
            "expects_file": model_config_rel,
            "instruction": f"""
Use the discovery-detector-builder subagent to read {inventory_rel} and write
{model_config_rel}. This is configuration, not executable Python.

Write a JSON object with schema_version=1, feature_inventory_sha256 equal to the
current SHA-256 of {inventory_rel}, a non-empty selection_basis explaining the
choice from feature semantics/coverage/missingness only, and candidates.

Always include these three role=baseline candidates: logistic_l2,
logistic_elasticnet, and hist_gradient_boosting. You may add zero, one, or two
role=optional candidates chosen only from: spline_logistic,
explainable_boosting, extra_trees, random_forest, xgboost. Choose an extension
only when the inventory gives a concrete reason (for example smooth nonlinearity,
interactions, or native-NaN behavior). Do not use observed model performance,
outer-fold labels, or held-out data to choose the candidate list.

Each candidate has exactly name, role, reason, grid, native_nan, and
requires_package. `reason` is non-empty. `grid` maps allowed parameter names to
small non-empty JSON-scalar lists and has at most {MAX_GRID_COMBINATIONS} Cartesian
combinations. Numeric rates/C values must be finite and in their estimator's
valid range; count/depth values must be positive integers (or documented null
where allowed); max_features is null, sqrt/log2, a positive integer, or a
fraction in (0, 1]. Keep grids conservative for sparse positives. native_nan is
true only for hist_gradient_boosting, explainable_boosting, and xgboost.
requires_package is null except explainable_boosting="interpret" and
xgboost="xgboost". Never write
a Python class path, module path, import statement, code string, callable, or
arbitrary estimator name. The driver will reject missing baselines, more than
{MAX_OPTIONAL_MODELS} extensions, unknown families/parameters, stale inventory
hashes, and oversized grids before detector generation begins.
""".strip(),
        },
        {
            "name": "generate-detector",
            "expects_file": detector_rel,
            "instruction": f"""
Use the discovery-detector-builder subagent to generate {detector_rel} from
{inventory_rel}, the already driver-validated {model_config_rel}, and
{source_note}. For each included feature, read ONLY the
inventory's feature_source_path and verify its SHA-256 before copying semantics.
Never silently fall back from a missing/mismatched fixed source to rerun, or vice
versa. Produce ONE self-contained
CLI that extracts features once, compares several model strategies fairly, fits
the selected strategy, and scores segments. Do not generate one script per model.

DATA AND EXTRACTION CONTRACT
- Accept a training _add.pkl positional argument and optional --test-pkl for a
  different held-out brain.
- Produce one row per adjudicable segment: non-zero canonical labels define the
  universe; component_id_to_swc_id maps components to segment ids; is_merge is
  membership in gt_merge_labels. Segments with no fragment component still get a
  row.
- Load each pkl once. Share edge, chain, junction, component and segment passes
  across all features. Keep intermediate-only quantities out of MODEL_COLS.
- Copy feature definitions and constants verbatim from each inventory-selected
  feature source. Remove
  their install and sandbox scaffolding. Prefer the real proofreader import and
  use the mock SkeletonGraph unpickler only on ImportError.
- Dictionary/set membership during extraction is the source of truth for whether
  each feature was measurable. Emit an undefined numeric value as NaN and emit a
  binary <feature>_is_defined column. A legitimate measured value equal to an old
  sentinel such as 0, 1 or 180 remains is_defined=1. Never infer new missingness
  by comparing numeric values with sentinels.
- --exclude-empty drops only rows whose is_defined flags are all zero, and applies
  identically to train and held-out data. Audit undefined shares overall and by
  class, the all-undefined group, and single-feature AUC on defined rows.

CANDIDATE MODEL FACTORY
- Accept --model-config PATH, defaulting to {MODEL_CONFIG_NAME} beside the
  script. Validate schema version, inventory hash, required baselines, optional
  count, family names, parameter names, parameter value types/ranges, native-NaN
  declarations, dependency declarations, and the
  {MAX_GRID_COMBINATIONS}-combination cap again at runtime.
- Build only configured allowlisted families through explicit name branches.
  Never eval configuration, dynamically import a configured path, or accept a
  class/module path. The safe family implementations and complexity order live
  in code; the JSON controls only allowlisted family inclusion and grid values.
- Baseline implementations are balanced L2 logistic with fold-local mean
  imputation/scaling; balanced elastic-net logistic with the same fold-local
  preprocessing; and shallow regularized histogram gradient boosting receiving
  raw NaNs plus flags. Implement configured extensions conservatively: spline
  logistic, ExplainableBoostingClassifier, ExtraTreesClassifier,
  RandomForestClassifier, or XGBClassifier according to the allowlisted name.
- `interpret` and `xgboost` are optional dependencies and are never installed.
  If a configured optional package is unavailable, record the candidate as
  skipped and continue. Pin every seed. Add --n-jobs with a conservative default.
  Record failures with tracebacks; never substitute a different estimator under
  the same name.

NESTED MODEL SELECTION — NO TEST-SET SELECTION
- Average precision is the PRIMARY selection score. ROC-AUC, F1, precision@N and
  recall@N are reports, not alternative objectives. Add --review-budget with a
  default of 100 for the @N metrics.
- Use fixed outer StratifiedKFold(5, shuffle=True, random_state=42). Within each
  outer-training partition, use inner stratified CV to tune every family. Every
  imputer, scaler, spline transform, class weight and hyperparameter choice must
  be fitted from inner-training data only.
- For each family, retain nested outer-fold predictions in
  merge_probability_oof_<family>. Also execute the complete selection policy in
  every outer fold: compare inner-CV AP and apply a one-standard-error rule,
  choosing the simplest eligible configured family in this fixed order:
  logistic_l2, logistic_elasticnet, spline_logistic, explainable_boosting,
  hist_gradient_boosting, extra_trees, random_forest, xgboost. Store
  those predictions as merge_probability_oof_selector and report family selection
  frequencies. This evaluates the selection procedure rather than a result chosen
  after looking at outer labels.
- After nested evaluation, repeat the same inner search on ALL training rows,
  apply the same one-standard-error rule, record the winning family and params,
  and fit it on all training rows. --test-pkl must not influence family choice,
  parameters, preprocessing, calibration or threshold.
- Expose merge_probability_oof as the stored family OOF column corresponding to
  the final full-data winner, for an aligned queue and threshold sweep. Explain
  separately that merge_probability_oof_selector estimates the adaptive selection
  policy. Use only OOF predictions for threshold choice and training top-N queues;
  the final winner's in-sample score is reference-only.

OUTPUTS AND AUDIT TRAIL
- Write merge_detector_<brain>.csv with ids, label, raw features, all is_defined
  flags, every family OOF column, selector OOF, selected-winner OOF and final
  in-sample score.
- Write model_selection_<brain>.json containing the model-config path and
  SHA-256, registry and grids,
  skipped/failed candidates, dependency versions, seeds, per-fold inner choices,
  per-family outer metrics, selector metrics, selection frequencies, final winner
  and params, feature order and scope.
- Save the fitted winner plus ordered schema as merge_detector_<brain>.joblib.
  Loading it must not need the test brain or recompute preprocessing statistics.
- Print prevalence, undefined audit, candidate comparison, selector performance,
  final selection rationale, OOF threshold sweep and top-20 queue. Do not call
  class-weighted outputs calibrated probabilities; call them scores unless an
  independent calibration stage is genuinely implemented.
- Produce headless figures: per-family OOF PR/ROC comparison; selected-winner
  threshold/workload curves; score separation; raw feature ECDF/correlation;
  undefined share by class; linear coefficients when applicable; and
  model-agnostic permutation importance on validation or held-out data, never
  tree impurity importance presented as generalization importance.
- Tee stdout and stderr to merge_detector_<brain>.log.txt with argv, timestamp,
  host, versions, success/failure and elapsed footer.

HELD-OUT BRAIN
Load it only after releasing the training payload. Pass it through the already
fitted winning pipeline; do not recompute test imputation or use test labels for
selection. Apply the same scope and report AP, ROC-AUC, precision@N, recall@N,
fixed-threshold performance and transfer gap. Warn on mcl mismatch. Clearly tag
figures heldout-from-<train>.

DO NOT RUN A REAL PKL: it needs >20 GB and belongs on a compute node. Statically
parse the script, run --help, and run a small synthetic-frame smoke test of the
selection code proving fold-local preprocessing handles NaNs, optional
dependency-backed candidates can be skipped, every expected OOF column is
filled exactly once, and held-out
prediction does not mutate fitted preprocessing. Report the generated path and
exact compute-node command.
""".strip(),
        },
        {
            "name": "verify-and-document",
            "expects_file": readme_rel,
            "instruction": f"""
Use the discovery-detector-builder subagent to verify {detector_rel} and write
{readme_rel}. No real dataset has been run: do not invent a CSV, winning model,
metrics or class counts. Obtain provenance from {summary_rel}, {inventory_rel}
and {model_config_rel}, plus {source_note}.

VERIFY AND FIX BEFORE DOCUMENTING
1. Every inventory feature has one explicit feature_source_path whose current
   SHA-256 matches the inventory; the rerun-vs-fixed choice follows the recorded
   correction_scope and source_reason. Every included feature is genuinely
   computed once, keeps the selected source's constants,
   reaches the frame and has an extraction-time binary is_defined flag. Undefined
   values are NaN; valid 0/1/180 values are not missing. --exclude-empty uses
   flags rather than sentinels.
2. Statistical imputation, scaling and spline fitting live inside candidate
   pipelines. Remove any whole-dataset fill_mean or preprocessing before CV.
   Families declared native-NaN receive NaNs directly; every other family uses
   fold-local imputation.
3. --model-config defaults beside the script and is revalidated before data
   loading. The registry contains exactly its validated allowlisted candidates,
   including all three mandatory baselines and at most two optional families.
   Configuration cannot supply imports/classes/code. Names, implementations,
   grids, dependency declarations and fixed complexity order agree everywhere.
4. Outer validation labels affect metrics only. Inner folds tune and select. The
   one-standard-error rule is correct, each outer row gets exactly one prediction
   per available family and selector, and --test-pkl cannot influence the final
   full-data winner.
5. AP is primary. Queue and threshold sweep use the final winner's nested OOF,
   never in-sample scores. Held-out prediction reuses the fitted winner and cannot
   mutate preprocessing.
6. The selection JSON explains why the model won; joblib carries pipeline and
   ordered schema; CSV columns are unambiguous. Nonlinear models do not present
   impurity importance as validation importance.
7. There is no install command, sys.path edit or sys.modules deletion. Each pkl
   loads once, caches release sequentially, Agg precedes pyplot, logging is
   durable, parsing and --help work, and the synthetic smoke test passes.

Then document every excluded hypothesis and the per-feature rerun/fixed source
choice, plus provenance, feature mapping and defined conditions; why each model
is included, how to safely edit {MODEL_CONFIG_NAME}, and why arbitrary families
require an explicit code review rather than a JSON class path; nested CV and the
one-standard-error rule in plain language; the
difference among family OOF, selector OOF, winner OOF, in-sample and held-out
scores; output schemas; compute-node commands; optional dependency behavior; and a
reading order starting with coverage and AP, then review-budget precision/recall
and cross-brain transfer, with ROC-AUC last. State that weighted classifier scores
are not automatically calibrated probabilities. Preserve caveats about sparse
positive labels, correlated features, discovery-stage feature-selection bias,
uneven coverage, and segment-level rather than merge-site localization. Report
fixes made and anything requiring a real compute-node run.
""".strip(),
        },
    ]


def build_options() -> ClaudeAgentOptions:
    """Configure the SDK session for this project.

    ``setting_sources=["project"]`` is what makes the SDK auto-discover the
    filesystem subagents in ``.claude/agents/`` (incl. discovery-detector-builder)
    and project settings relative to ``cwd``.
    """
    return ClaudeAgentOptions(
        cwd=str(PROJECT_ROOT),
        setting_sources=["project"],
        # The orchestrator delegates to the builder subagent (Task), which needs
        # filesystem tools to read the report plus rerun/fixed sources and write
        # the detector.
        allowed_tools=["Task", "Bash", "Read", "Write", "Edit", "Glob"],
        permission_mode="bypassPermissions",
        # Pin Opus 4.8 explicitly so the model is not left to the ambient session
        # default. The subagent is `model: inherit`, so it follows this too.
        model="claude-opus-4-8",
        # Lifting feature code across ten scripts without changing its semantics,
        # then catching the imputation confounds, needs maximum reasoning effort.
        env={**os.environ, "CLAUDE_EFFORT": "xhigh"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent step / compute step
# ─────────────────────────────────────────────────────────────────────────────

async def _run_step_inner(client: ClaudeSDKClient, step: dict, verbose: bool) -> str:
    """Send one workflow step to the session and return its final text."""
    await client.query(step["instruction"])

    chunks: list[str] = []
    n_tools = 0
    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
                    if verbose:
                        print(block.text, end="", flush=True)
                elif ToolUseBlock and isinstance(block, ToolUseBlock):
                    n_tools += 1
                    log(f"  → {describe_tool(block)}")
        elif isinstance(message, ResultMessage):
            if verbose:
                print()  # newline after the streamed text
            cost = getattr(message, "total_cost_usd", None)
            dur_ms = getattr(message, "duration_ms", None)
            parts = [f"{n_tools} tool call(s)"]
            if dur_ms is not None:
                parts.append(f"{dur_ms / 1000:.0f}s")
            if cost is not None:
                parts.append(f"${cost:.4f}")
            log(f"  step turn finished — {', '.join(parts)}")
    return "".join(chunks)


async def run_step(client: ClaudeSDKClient, step: dict, verbose: bool) -> str:
    """``_run_step_inner`` with a hard timeout, so a hung step fails loudly."""
    try:
        return await asyncio.wait_for(
            _run_step_inner(client, step, verbose),
            timeout=AGENT_STEP_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise SystemExit(
            f"Agent step '{step['name']}' timed out after {AGENT_STEP_TIMEOUT_S}s."
        )


# Files this workflow writes itself. Deleting one of these costs nothing, because
# the run about to start puts it back.
_REGENERATED = frozenset({
    "feature_inventory.json", MODEL_CONFIG_NAME, DETECTOR_NAME, "README.md",
    DRIVER_LOG_NAME,
})


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
                name.startswith("merge_detector_")
                and (name.endswith(".csv") or name.endswith(".log.txt")
                     or name.endswith(".joblib"))) or (
                name.startswith("model_selection_") and name.endswith(".json")):
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
                       "(compute node, minutes)"),
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

    print(f"\nType 'delete' to remove it and continue, anything else to abort: ",
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
    """Load a JSON object or fail with a path-specific diagnostic."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read {label} {_rel_to_root(path)}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label.capitalize()} {_rel_to_root(path)} must be a JSON object.")
    return value


def _integer_ids(value, label: str, path: Path) -> list[int]:
    """Validate an ordered JSON list of unique integer hypothesis ids."""
    if not isinstance(value, list) or not value:
        raise SystemExit(f"{label} in {_rel_to_root(path)} must be a non-empty list.")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise SystemExit(f"{label} in {_rel_to_root(path)} must contain only integers.")
    if len(value) != len(set(value)):
        raise SystemExit(f"{label} in {_rel_to_root(path)} contains duplicate ids.")
    return value


def _manifest_ids(manifest_path: Path) -> list[int]:
    """Return the exact id order recorded by a rerun MANIFEST.json."""
    manifest = _load_json_object(manifest_path, "rerun manifest")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise SystemExit(
            f"Rerun manifest {_rel_to_root(manifest_path)} has no non-empty records list."
        )
    ids = [record.get("id") if isinstance(record, dict) else None for record in records]
    return _integer_ids(ids, "records[].id", manifest_path)


def _sha256(path: Path) -> str:
    """Hash a source script without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_model_grid_value(parameter: str, value) -> bool:
    """Return whether one JSON scalar is meaningful for an allowed parameter."""
    is_int = isinstance(value, int) and not isinstance(value, bool)
    is_number = (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (not isinstance(value, float) or math.isfinite(value))
    )
    if parameter in {"C", "learning_rate"}:
        return is_number and value > 0
    if parameter in {"l1_ratio", "subsample", "colsample_bytree"}:
        return is_number and 0 <= value <= 1 and (
            parameter == "l1_ratio" or value > 0
        )
    if parameter in {"l2_regularization", "min_child_weight", "reg_lambda"}:
        return is_number and value >= 0
    if parameter in {
        "max_leaf_nodes", "min_samples_leaf", "n_knots", "degree", "max_bins",
        "max_rounds", "n_estimators",
    }:
        minimum = 2 if parameter in {"max_leaf_nodes", "n_knots", "max_bins"} else 1
        return is_int and value >= minimum
    if parameter == "max_depth":
        return value is None or (is_int and value >= 1)
    if parameter == "max_features":
        return (
            value is None
            or value in {"sqrt", "log2"}
            or (is_int and value >= 1)
            or (is_number and not is_int and 0 < value <= 1)
        )
    return False


def _validate_model_contract_constants() -> None:
    """Catch drift among the code-owned family metadata tables."""
    if set(MODEL_PARAMETER_ALLOWLIST) != ALLOWED_MODEL_FAMILIES:
        raise RuntimeError("MODEL_PARAMETER_ALLOWLIST is out of sync with model families.")
    if set(MODEL_NATIVE_NAN) != ALLOWED_MODEL_FAMILIES:
        raise RuntimeError("MODEL_NATIVE_NAN is out of sync with model families.")
    if not set(MODEL_REQUIRED_PACKAGE) <= OPTIONAL_MODEL_FAMILIES:
        raise RuntimeError("Only optional model families may require extra packages.")


def validate_model_config(config_path: Path, inventory_path: Path) -> None:
    """Reject stale or executable/oversized model configuration.

    The generated script repeats this contract at runtime. Keeping the first
    check in the orchestrator prevents a creative agent response from becoming
    detector source in the following step.
    """
    _validate_model_contract_constants()
    config = _load_json_object(config_path, "model candidate configuration")
    top_level_fields = {
        "schema_version", "feature_inventory_sha256", "selection_basis",
        "candidates",
    }
    if set(config) != top_level_fields:
        raise SystemExit(
            "Model candidate configuration must contain exactly: "
            + ", ".join(sorted(top_level_fields))
        )
    if config.get("schema_version") != 1:
        raise SystemExit("Model candidate configuration must set schema_version to 1.")
    if config.get("feature_inventory_sha256") != _sha256(inventory_path):
        raise SystemExit(
            "Model candidate configuration feature_inventory_sha256 is stale."
        )
    if not isinstance(config.get("selection_basis"), str) \
            or not config["selection_basis"].strip():
        raise SystemExit("Model candidate configuration needs a selection_basis.")
    candidates = config.get("candidates")
    if not isinstance(candidates, list):
        raise SystemExit("Model candidate configuration candidates must be a list.")

    required_fields = {
        "name", "role", "reason", "grid", "native_nan", "requires_package",
    }
    names: list[str] = []
    optional_count = 0
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or set(candidate) != required_fields:
            raise SystemExit(
                f"Model candidate {index} must contain exactly: "
                + ", ".join(sorted(required_fields))
            )
        name = candidate["name"]
        if not isinstance(name, str) or name not in ALLOWED_MODEL_FAMILIES:
            raise SystemExit(f"Model candidate {index} has disallowed family {name!r}.")
        if name in names:
            raise SystemExit(f"Model candidate configuration duplicates {name}.")
        names.append(name)

        expected_role = (
            "baseline" if name in REQUIRED_MODEL_FAMILIES else "optional"
        )
        if candidate["role"] != expected_role:
            raise SystemExit(f"Model candidate {name} must have role={expected_role}.")
        optional_count += expected_role == "optional"
        if not isinstance(candidate["reason"], str) or not candidate["reason"].strip():
            raise SystemExit(f"Model candidate {name} needs a non-empty reason.")
        expected_native_nan = MODEL_NATIVE_NAN[name]
        if candidate["native_nan"] is not expected_native_nan:
            raise SystemExit(
                f"Model candidate {name} native_nan must be "
                f"{expected_native_nan}."
            )
        expected_package = MODEL_REQUIRED_PACKAGE.get(name)
        if candidate["requires_package"] != expected_package:
            raise SystemExit(
                f"Model candidate {name} requires_package must be "
                f"{expected_package!r}."
            )

        grid = candidate["grid"]
        if not isinstance(grid, dict) or not grid:
            raise SystemExit(f"Model candidate {name} grid must be a non-empty object.")
        unknown = set(grid) - MODEL_PARAMETER_ALLOWLIST[name]
        if unknown:
            raise SystemExit(
                f"Model candidate {name} has disallowed parameters: "
                + ", ".join(sorted(unknown))
            )
        combinations = 1
        for parameter, values in grid.items():
            if not isinstance(values, list) or not values:
                raise SystemExit(
                    f"Model candidate {name} parameter {parameter} needs a "
                    "non-empty list."
                )
            if any(isinstance(value, (dict, list)) for value in values):
                raise SystemExit(
                    f"Model candidate {name} parameter {parameter} values must "
                    "be JSON scalars."
                )
            invalid_values = [
                value for value in values
                if not _valid_model_grid_value(parameter, value)
            ]
            if invalid_values:
                raise SystemExit(
                    f"Model candidate {name} parameter {parameter} has invalid "
                    f"values: {invalid_values!r}."
                )
            combinations *= len(values)
        if combinations > MAX_GRID_COMBINATIONS:
            raise SystemExit(
                f"Model candidate {name} grid has {combinations} combinations; "
                f"maximum is {MAX_GRID_COMBINATIONS}."
            )

    missing = REQUIRED_MODEL_FAMILIES - set(names)
    if missing:
        raise SystemExit(
            "Model candidate configuration is missing required baselines: "
            + ", ".join(sorted(missing))
        )
    if optional_count > MAX_OPTIONAL_MODELS:
        raise SystemExit(
            f"Model candidate configuration has {optional_count} optional models; "
            f"maximum is {MAX_OPTIONAL_MODELS}."
        )
    log(
        f"  OK: model configuration validated ({len(names)} candidates, "
        f"{optional_count} optional)."
    )


def validate_inventory(
    inventory_path: Path,
    selected_ids: list[int],
    selection_rel: str | None,
    rerun_dir: Path,
    fixed_dir: Path | None,
) -> None:
    """Enforce the v2 inventory and its per-hypothesis source provenance."""
    inventory = _load_json_object(inventory_path, "feature inventory")
    if inventory.get("schema_version") != 2:
        raise SystemExit("Feature inventory must set schema_version to 2.")
    if inventory.get("selection_manifest") != selection_rel:
        raise SystemExit(
            "Feature inventory selection_manifest does not match the workflow input."
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
    provenance_fields = (
        "exclusion_reason", "reproduction_status", "statistical_verdict",
        "corrected_result_status", "post_correction_verdict", "correction_scope", "rerun_path",
        "rerun_sha256", "fixed_path", "fixed_sha256", "feature_source",
        "feature_source_path", "feature_source_sha256", "source_reason", "features",
    )
    for row in rows:
        hypothesis_id = row["id"]
        missing = [field for field in provenance_fields if field not in row]
        if missing:
            raise SystemExit(
                f"Inventory hypothesis {hypothesis_id} is missing fields: "
                + ", ".join(missing)
            )
        if not isinstance(row.get("included"), bool):
            raise SystemExit(f"Inventory hypothesis {hypothesis_id} included must be boolean.")
        if row["correction_scope"] not in allowed_scopes:
            raise SystemExit(f"Inventory hypothesis {hypothesis_id} has invalid correction_scope.")
        if not isinstance(row["source_reason"], str) or not row["source_reason"].strip():
            raise SystemExit(f"Inventory hypothesis {hypothesis_id} needs a source_reason.")

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

        if row["included"]:
            if row["exclusion_reason"] is not None:
                raise SystemExit(
                    f"Included hypothesis {hypothesis_id} must have null exclusion_reason."
                )
            if row["correction_scope"] == "unclear":
                raise SystemExit(f"Unclear hypothesis {hypothesis_id} cannot be included.")
            post_verdict = str(row["post_correction_verdict"]).upper()
            corrected_status = str(row["corrected_result_status"]).upper()
            if "OVERTURNED" in post_verdict:
                raise SystemExit(f"Overturned hypothesis {hypothesis_id} cannot be included.")
            reproduction_status = str(row["reproduction_status"]).upper()
            if any(status in reproduction_status for status in ("FAILED", "UNUSABLE")) \
                    and corrected_status != "USABLE":
                raise SystemExit(
                    f"Failed/unusable hypothesis {hypothesis_id} needs a usable correction."
                )
            statistical_verdict = str(row["statistical_verdict"]).upper()
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
            for feature in row["features"]:
                if not isinstance(feature, dict):
                    raise SystemExit(
                        f"Inventory hypothesis {hypothesis_id} has a non-object feature."
                    )
                missing_feature_fields = required_feature_fields - feature.keys()
                if missing_feature_fields:
                    raise SystemExit(
                        f"Inventory hypothesis {hypothesis_id} feature is missing: "
                        + ", ".join(sorted(missing_feature_fields))
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

    log(f"  OK: inventory v2 validated for {len(rows)} selected hypothesis ids.")



# ─────────────────────────────────────────────────────────────────────────────
# Workflow
# ─────────────────────────────────────────────────────────────────────────────

def resolve_inputs(
    run_json: Path,
) -> tuple[Path, Path, Path | None, Path | None, list[int]]:
    """Locate and cross-check report, selection, rerun, and corrected sources.

    Pure — it only stats and raises. Called from ``main`` BEFORE the output folder
    can be deleted, so a missing report aborts while the folder is still intact;
    ``run_workflow`` calls it again so it stays usable on its own.
    """
    stem = run_stem(run_json)
    summary_path = run_json.with_name(f"{stem}.summary.md")
    predictive_rerun_dir = run_json.with_name(
        f"{run_json.name}.predictive.rerun"
    )
    legacy_rerun_dir = run_json.with_name(f"{run_json.name}.rerun")
    predictive_fixed_dir = run_json.with_name(
        f"{run_json.name}.predictive.fixed"
    )
    legacy_fixed_dir = run_json.with_name(f"{run_json.name}.fixed")
    selection_path = run_json.with_name(
        f"{run_stem(run_json)}.predictive-selection.json"
    )
    predictive_ready = predictive_rerun_dir.is_dir() and any(
        predictive_rerun_dir.glob("hypo_*.py")
    )
    legacy_ready = legacy_rerun_dir.is_dir() and any(
        legacy_rerun_dir.glob("hypo_*.py")
    )
    rerun_dir = predictive_rerun_dir if predictive_ready else legacy_rerun_dir
    candidate_fixed_dir = (
        predictive_fixed_dir if predictive_ready else legacy_fixed_dir
    )
    fixed_dir = (
        candidate_fixed_dir
        if candidate_fixed_dir.is_dir()
        and any(candidate_fixed_dir.glob("hypo_*.py"))
        else None
    )

    if not summary_path.is_file():
        raise SystemExit(
            f"No finished report at {_rel_to_root(summary_path)}. Run "
            f"`python agentic/run_discovery_workflow.py {_rel_to_root(run_json)} "
            f"--pkl {origin_cache_hint(run_json) or '<ORIGIN>_add.pkl'} "
            "--direction predictive` first — this workflow builds on its output."
        )
    if not (predictive_ready or legacy_ready):
        raise SystemExit(
            "No loading-fixed scripts found. Expected predictive artifacts at "
            f"{_rel_to_root(predictive_rerun_dir)} (or legacy artifacts at "
            f"{_rel_to_root(legacy_rerun_dir)}). Run the discovery workflow with "
            "--direction predictive first; detector feature definitions come from "
            "those scripts, not from the report."
        )
    rerun_manifest_path = rerun_dir / "MANIFEST.json"
    if not rerun_manifest_path.is_file():
        raise SystemExit(
            f"No rerun manifest at {_rel_to_root(rerun_manifest_path)}; refuse to "
            "infer hypothesis membership from possibly stale loose scripts."
        )
    rerun_ids = _manifest_ids(rerun_manifest_path)

    authoritative_selection = selection_path if predictive_ready else None
    if predictive_ready:
        if not selection_path.is_file():
            raise SystemExit(
                f"Predictive rerun sources require {_rel_to_root(selection_path)} "
                "as the authoritative hypothesis selection."
            )
        selection = _load_json_object(selection_path, "predictive selection")
        if selection.get("criterion") != "predictive":
            raise SystemExit(
                f"Predictive selection {_rel_to_root(selection_path)} has the "
                "wrong criterion."
            )
        selected_ids = _integer_ids(
            selection.get("selected_ids"), "selected_ids", selection_path
        )
        if selection.get("source_sha256") != _sha256(run_json):
            raise SystemExit(
                f"Predictive selection {_rel_to_root(selection_path)} is stale for "
                f"{_rel_to_root(run_json)}; regenerate the discovery outputs."
            )
        selected_set, rerun_set = set(selected_ids), set(rerun_ids)
        if selected_set != rerun_set:
            raise SystemExit(
                "Predictive selection and rerun MANIFEST disagree; refuse stale "
                f"sources. Missing from MANIFEST: {sorted(selected_set - rerun_set)}; "
                f"not selected: {sorted(rerun_set - selected_set)}. Regenerate the "
                "predictive rerun artifacts before building a detector."
            )
    else:
        selected_ids = rerun_ids

    missing_scripts = [
        hypothesis_id for hypothesis_id in selected_ids
        if not (rerun_dir / f"hypo_{hypothesis_id}.py").is_file()
    ]
    if missing_scripts:
        raise SystemExit(
            f"Rerun MANIFEST selects ids with no script: {missing_scripts}."
        )
    return (
        summary_path, rerun_dir, fixed_dir, authoritative_selection, selected_ids
    )


async def run_workflow(
    run_json: Path,
    out_dir: Path,
    verbose: bool,
) -> None:
    summary_path, rerun_dir, fixed_dir, selection_path, selected_ids = resolve_inputs(
        run_json
    )

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
    wf_start = time.monotonic()

    async with ClaudeSDKClient(options=build_options()) as client:
        log("SDK session opened.")
        for i, step in enumerate(steps, start=1):
            print(f"\n=== Step {i}/{len(steps)}: {step['name']} ===")
            log(f"Step {i}/{len(steps)} '{step['name']}' started.")
            step_start = time.monotonic()
            final_text = await run_step(client, step, verbose)
            log(
                f"Step {i}/{len(steps)} '{step['name']}' done in "
                f"{time.monotonic() - step_start:.0f}s."
            )
            validate_step_output(step)
            if step["name"] == "inventory-features":
                validate_inventory(
                    out_dir / "feature_inventory.json",
                    selected_ids,
                    _rel_to_root(selection_path) if selection_path else None,
                    rerun_dir,
                    fixed_dir,
                )
            elif step["name"] == "configure-models":
                validate_model_config(
                    out_dir / MODEL_CONFIG_NAME,
                    out_dir / "feature_inventory.json",
                )
            if not verbose:
                print(final_text.strip())

    log(f"Workflow complete in {time.monotonic() - wf_start:.0f}s.")
    print("\n=== Workflow complete ===")
    print(f"Generated code in {_rel_to_root(out_dir)}/")
    detector_rel = f"{_rel_to_root(out_dir)}/{DETECTOR_NAME}"
    if (out_dir / DETECTOR_NAME).is_file():
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
            "autodiscovery/merge-error-794495-mcl100_2026-08-04.json. Its "
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
            "Do not delete a pre-existing output folder — write into it as it "
            "stands, the behaviour before deletion was added. Anything already "
            "there survives, so run outputs from the previous detector will be "
            "left beside the new script with nothing marking the mismatch."
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
    resolve_inputs(run_json)

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
            asyncio.run(run_workflow(run_json, out_dir, args.verbose))
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
            )
            log_fh.flush()

    log(f"Driver console log → {_rel_to_root(log_txt)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

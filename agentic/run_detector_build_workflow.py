"""
Generate a multi-feature merge detector from a finished AutoDiscovery run.

Where ``run_discovery_workflow.py`` DIGESTS one run export into a ranked
``<RUN>.summary.md`` report, this workflow goes one step further downstream: it
takes that finished report plus the reproducer's loading-fixed hypothesis
scripts (normally ``autodiscovery/<RUN>.json.predictive.rerun/hypo_<id>.py``)
and WRITES A DETECTOR — one
script that computes every confirmed hypothesis's feature in a single pass over
the skeleton graph, evaluates several appropriately regularized tabular models
under nested cross-validation, and fits the selected strategy to produce one
score per segment.

The motivation comes from the reports themselves: each hypothesis is a single
feature, and the recurring caveat is that none is precise enough alone
("high recall, ~8% precision — best used as one component of an ensemble").
This workflow builds the combined feature table and selects a model for it.

THIS WORKFLOW IS CODE GENERATION ONLY, and it takes NO DATASET. Its inputs are
the run export, the report beside it and the ``.rerun/`` scripts — that is
everything needed to write the detector. It never loads a cache and never runs
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
  1. inventory-features  — read <RUN>.summary.md and every .rerun/hypo_<id>.py,
                           and record for each returned hypothesis its feature
                           name, the exact quantity computed, its aggregation
                           unit, exact defined condition, and historical value
                           used when undefined. Those conditions drive the audit.
  2. generate-detector   — write ONE consolidated CLI script: load the pkl once,
                           group the features by the traversal each needs
                           (edge / chain / junction / component / segment), keep
                           every constant verbatim from the .rerun source, compare
                           linear, spline and tree-based strategies with
                           nested stratified CV, fit the selected strategy, and
                           emit a CSV, selection manifest, log and key figures.
  3. verify-and-document — check the script STATICALLY against the inventory and
                           the .rerun sources (every feature computed and fitted,
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

All three steps always run, and re-running the workflow simply regenerates the
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

def _build_steps_legacy(
    run_rel: str,
    summary_rel: str,
    rerun_rel: str,
    out_dir_rel: str,
) -> list[dict]:
    """Previous single-logistic workflow, retained only as design history.

    Each dict carries ``name``, ``instruction``, and optionally ``expects_file``
    (a path that must exist and be non-empty after the step) — a cheap structural
    check so a step that produced nothing is reported at once instead of
    surfacing as a confusing failure two steps later.
    """
    detector_rel = f"{out_dir_rel}/{DETECTOR_NAME}"
    inventory_rel = f"{out_dir_rel}/feature_inventory.json"
    readme_rel = f"{out_dir_rel}/README.md"

    return [
        {
            "name": "inventory-features",
            "expects_file": inventory_rel,
            "instruction": (
                "Use the discovery-detector-builder subagent to inventory the "
                f"features of the finished AutoDiscovery run {run_rel}. Read the "
                f"ranked report {summary_rel} to see which hypotheses were "
                "returned and each one's id, title, verdict, ROC-AUC, "
                "generalization call and caveats. Then read EVERY loading-fixed "
                f"hypothesis script under {rerun_rel}/ (hypo_<id>.py, indexed by "
                "MANIFEST.json) — those scripts, not the report's prose, define "
                "what each feature actually computes. Also read "
                "markdowns/labeled_dataset_cache.md for the _add.pkl schema. "
                "For each returned hypothesis record: the hypothesis id, a "
                "snake_case feature column name, the verdict + reported ROC-AUC, "
                "the quantity computed, every constant it depends on (reach "
                "distances, window widths, angle cutoffs, seeds), the unit of "
                "aggregation (per junction / per chain / per component / per "
                "segment) and how it is reduced to one value per segment, which "
                "traversal phase it belongs to (edge / chain / junction / "
                "component / segment), and — most important — THE VALUE THE "
                "FEATURE TAKES WHEN IT IS UNDEFINED for a segment (no junction, "
                "no component, too few nodes), read off the script's "
                "dict.get(seg_id, <default>) or explicit imputation. Include "
                "hypotheses graded WEAK or MINOR: a weak single feature can still "
                "carry independent signal in a combination, and the regression "
                "coefficient is what should decide that. Exclude one only if the "
                "test-fixer OVERTURNED it or its corrected test shows the effect "
                f"vanishes. Write the inventory as JSON to {inventory_rel} and "
                "report the feature count, any hypothesis you excluded with the "
                "reason, and the list of defaults (those are what the later audit "
                "step tests for AUC inflation)."
            ),
        },
        {
            "name": "generate-detector",
            "expects_file": detector_rel,
            "instruction": (
                "Use the discovery-detector-builder subagent to generate the "
                f"detector script {detector_rel} from the inventory you just "
                f"wrote ({inventory_rel}) and the .rerun sources under "
                f"{rerun_rel}/. Produce ONE self-contained CLI script that takes "
                "an _add.pkl path as a positional argument. ONE ROW PER "
                "ADJUDICABLE SEGMENT — this choice fixes every number the script "
                "reports, so take it from markdowns/labeled_dataset_cache.md and "
                "the .rerun sources instead of inventing one: the adjudicable "
                "universe is the set of non-zero canonical labels in "
                "gt_node_canonical_label; components map to segments through "
                "component_id_to_swc_id (the part of the swc id before the '.'); "
                "is_merge is membership in gt_merge_labels; and a segment with NO "
                "fragment component at all STILL GETS A ROW, with every feature at "
                "its default. Those rows are what audit item (b) below counts — "
                "one row per component instead, or skipping the segments that have "
                "no component, silently deletes the confound rather than measuring "
                "it and changes every reported metric. It must load the pkl "
                "ONCE and group the feature computations by the traversal each "
                "needs, so a single pass serves every feature that needs it: a "
                "vectorised numpy pass over the whole edge array; one chain "
                "decomposition (maximal degree-2 chains) scored by every "
                "chain-based feature; one pass over degree>=3 nodes computing all "
                "junction features per junction; one pass over components for the "
                "features needing a whole connected subgraph; then per-segment "
                "aggregation. A phase may also accumulate an INTERMEDIATE "
                "per-segment quantity that is not itself a feature but that a "
                "later phase needs — segment cable length, summed over intra-"
                "segment edges during the edge pass and consumed by a density or "
                "fill-factor feature at aggregation time, is the usual case. Keep "
                "those out of the feature list and out of the CSV's feature "
                "columns. Copy every constant VERBATIM from each feature's "
                ".rerun script — a feature whose numbers no longer match the "
                "report's per-feature AUC is a bug, not a refinement. Load by "
                "preferring `import agentic_neuron_proofreader` and falling back "
                "to the mock-SkeletonGraph unpickler only on ImportError. STRIP "
                "the pip-install preamble the .rerun scripts carry: the host "
                "environment already provides numpy, pandas, scipy, statsmodels, "
                "sklearn, networkx and matplotlib, so emit NO pip/conda install, "
                "no sys.modules deletion, and no sys.path edits. Fit "
                "StandardScaler -> LogisticRegression(class_weight='balanced', "
                "C=0.1) evaluated with StratifiedKFold(5, shuffle=True, "
                "random_state=42) — pin the seed, or the CSV's out-of-fold column "
                "shifts between runs and stops being comparable. Report ROC-AUC AND "
                "average precision (AUC is near-blind to the class imbalance, AP "
                "is not). Also compute POOLED OUT-OF-FOLD probabilities with "
                "cross_val_predict on the same cv object, and draw every curve "
                "and operating point from those, not from the in-sample fit: a "
                "threshold chosen on in-sample scores promises a precision the "
                "detector will not deliver. Print per-class feature means, the "
                "cross-validated scores, the pooled out-of-fold ROC-AUC and AP "
                "next to the prevalence a random ranker would score, z-scored "
                "coefficients sorted by |coef|, an out-of-fold threshold "
                "sweep of recall / precision / count-flagged, "
                "and the top 20 "
                "segments by out-of-fold probability with their is_merge label and "
                "their most telling feature values — that ranking is the "
                "proofreading queue the whole script exists to produce, and the one "
                "output a reader can sanity-check by eye. Write every "
                "per-segment row — all features, is_merge, the in-sample "
                "probability AND the out-of-fold one — to CSV, defaulting to "
                "merge_detector_<brain>.csv next to the script with <brain> "
                "parsed from the pkl filename. BUILD THE IMPUTATION AUDIT INTO "
                "THE SCRIPT, because nothing downstream will do it: keep the "
                "per-feature undefined-value defaults from the inventory in ONE "
                "module-level dict that both the feature assembly and the audit "
                "read, and derive the at-default mask through ONE shared helper so "
                "the audit, the figures and --exclude-empty cannot drift onto "
                "different definitions of the same group. A feature with no "
                "meaningful in-graph default carries NaN in that dict and is filled "
                "statistically at fit time (column mean, or 0.0): test those columns "
                "with isna(), NOT with a closeness test against NaN, which is always "
                "False and would report the statistically-filled features as never "
                "imputed — backwards, and on the features most likely to be. Before "
                "the statistical fill, measure and print (a) the "
                "share of rows sitting at each feature's default, split by class, "
                "(b) how many segments are at the default for EVERY feature and "
                "how many of those are merges, and (c) each feature's "
                "single-feature ROC-AUC over the full set beside its AUC over the "
                "subset where it is genuinely defined — that subset number is the "
                "one that should match the origin report's per-hypothesis AUC, and "
                "a much higher full-set number is inflation contributed by the "
                "imputed segments. Add an --exclude-empty flag that refits "
                "without the all-default group, and leave it OFF by default so "
                "the shipped numbers stay reproducible. GENERATE THE KEY FIGURES "
                "as PNGs into a figures/ directory (--fig-dir, --no-figures), "
                "selecting matplotlib's Agg backend BEFORE importing pyplot (the "
                "compute node is headless) and guarding the import so a missing "
                "matplotlib costs the figures and not the run: out-of-fold ROC "
                "beside out-of-fold precision-recall with the prevalence drawn in; "
                "the threshold sweep as TWO stacked panels sharing the x-axis — "
                "recall, precision and F1 on top with the best-F1 threshold marked, "
                "and the raw counts below on a log axis: segments flagged (the "
                "review load) beside merges detected (true positives), with a "
                "reference line at the total number of merges, because the vertical "
                "gap between those two curves IS the cost per merge found. Mask "
                "zero counts to NaN rather than clamping them to 1, or a log axis "
                "draws 'nothing detected' as if one had been; the "
                "z-scored coefficients; out-of-fold P(merge) by class normalised "
                "within class (raw counts hide the rare class); a per-feature ECDF "
                "grid by class; the Spearman correlation between features; and the "
                "imputed-share-by-class bar chart from the audit above. Prefix "
                "every filename with <brain>_<scope>_NN_ so an --exclude-empty run "
                "does not overwrite the default run's figures. ADD A CROSS-DATASET "
                "HOLDOUT via a --test-pkl flag naming a DIFFERENT brain's _add.pkl: "
                "5-fold CV within one brain says nothing about transfer, so train on "
                "the positional pkl, keep --test-pkl entirely out of the fit, and "
                "score it with the fitted pipeline. Three things make it honest and "
                "are easy to get wrong: (i) LOAD THE TWO CACHES ONE AT A TIME and "
                "del + gc.collect() the first payload before loading the second — "
                "each expands to >20 GB and holding both OOMs even an 80 GB node, "
                "while the extracted feature frames are a few MB so keeping both of "
                "those is free; (ii) impute the test set with the TRAINING set's "
                "fill constant, never one recomputed on the test brain, which would "
                "leak test statistics into its own features; (iii) apply the same "
                "--exclude-empty scope to both sides, or the transfer gap is "
                "confounded with a scope change. Print the held-out ROC-AUC and AP "
                "beside the training brain's out-of-fold numbers as an explicit "
                "transfer gap, plus the test brain's own all-default share (coverage "
                "differences move the gap by themselves), and warn when the two "
                "caches' mcl levels differ. For the held-out brain draw ONLY the "
                "three PERFORMANCE figures — ROC/PR, the threshold sweep, and the "
                "score-separation histogram — and withhold the rest: the "
                "coefficients belong to the training fit and the remaining panels "
                "describe the training data's own feature distributions. Include "
                "the score-separation one specifically because it is where a "
                "transfer gap becomes legible (merged scores sliding down versus "
                "the clean tail creeping up), which the aggregate AUC and AP cannot "
                "distinguish. Tag them "
                "<test-brain>_<scope>_heldout-from-<train-brain>_NN_, and title them "
                "'held-out', NEVER 'out-of-fold'. CAPTURE THE RUN TO A TXT LOG: "
                "tee sys.stdout AND sys.stderr through a small flush-on-every-"
                "write class whose isatty() is False, behind a --log-txt PATH "
                "flag defaulting to <script dir>/merge_detector_<brain>.log.txt, "
                "writing an argv + timestamp + host header first and a "
                "success-or-failure + elapsed footer last, with the work wrapped "
                "in try/except BaseException that calls traceback.print_exc() "
                "before re-raising and restores the real streams in a finally. "
                "A run whose numbers only reached a terminal has produced no "
                "evidence, and the footer is what distinguishes a complete log "
                "from one truncated by an OOM kill. DO NOT RUN THE SCRIPT. The "
                "pkl needs >20 GB RAM and the operator runs it themselves on a "
                "compute node, with the data, after this workflow finishes — an "
                "attempt here just gets OOM-killed on the login node. Verify it "
                "STATICALLY instead: python -c 'import ast; "
                "ast.parse(open(...).read())' and python <script> --help. Then "
                "report the path, the phase structure, the feature count, and the "
                "exact command the operator should run."
            ),
        },
        {
            "name": "verify-and-document",
            "expects_file": readme_rel,
            "instruction": (
                "Use the discovery-detector-builder subagent to VERIFY the "
                f"generated script {detector_rel} statically, and then document "
                "it. NOTHING HAS BEEN RUN: there is no CSV, no log and no "
                "results, and this step must not invent any. No dataset was given "
                "to this workflow either — for the README's provenance take the "
                f"origin brain and cache path from {summary_rel}, which records "
                "the dataset the run was generated on, rather than guessing one "
                "from a filename. VERIFY, without loading a pkl: (a) every feature in "
                f"{inventory_rel} is genuinely computed in the script AND reaches "
                "the feature list the model fits on — a feature that exists only "
                "as a column name or a comment is the failure mode to look for; "
                "(b) each feature's constants match its "
                f"{rerun_rel}/hypo_<id>.py source, compared side by side, because "
                "a changed reach distance or window width silently redefines the "
                "feature and detaches it from the AUC the report recorded for it; "
                "(c) each feature's undefined-value default matches the inventory, "
                "and the audit reads the SAME dict the feature assembly imputes "
                "from — two copies of one default is how the fit and the audit "
                "come to disagree; (d) no pip/conda install, no sys.path edit and "
                "no sys.modules deletion survives from the .rerun sandbox "
                "scaffolding; (e) the pkl is loaded exactly once and traversals "
                "are shared, not repeated per feature; (f) the txt log, the "
                "figures and --exclude-empty are wired as specified, with "
                "matplotlib's Agg backend selected BEFORE pyplot is imported; (g) "
                "the script parses and --help works. Fix what you find, then "
                f"re-verify. THEN write {readme_rel} covering: provenance (origin "
                "run JSON, report, origin brain and pkl, the ranking flags that "
                "selected these hypotheses, the run's own verdict counts); the "
                "feature <-> hypothesis mapping table with each default and "
                "traversal phase; the model choices justified against the actual "
                "class counts; the exact command to run it — conda activate panda, "
                "the compute-node requirement, an sbatch template; and what each "
                "output file and each figure contains. In place of results, write "
                "a 'How to read the results' section giving the order to read them "
                "in when the run lands: the all-default group size and the imputed "
                "share by class FIRST, then each feature's full-set vs "
                "defined-subset AUC gap, then the out-of-fold AP against the "
                "prevalence, and the ROC-AUC last — at ~1% prevalence it is the "
                "least informative of the four. State plainly at the top that the "
                "script has NOT been run and that the README therefore contains NO "
                "measured numbers: a reader must never mistake a described metric "
                "for a reported one. Keep the standing caveats: precision measured "
                "against sparse ground truth is a FLOOR (an unlabeled flagged "
                "segment whose geometry matches the confirmed merges may be a real "
                "merge outside the traced neurons, not a false positive); "
                "coefficients are in-sample and split credit between correlated "
                "features; feature coverage is uneven; and the output is "
                "segment-level, not site-level — it says which segment is merged, "
                "not where. Report what you fixed during verification, and "
                "anything you could not check without running the script."
            ),
        },
    ]


def build_steps(
    run_rel: str,
    summary_rel: str,
    rerun_rel: str,
    out_dir_rel: str,
) -> list[dict]:
    """Build the model-selection detector workflow instructions."""
    detector_rel = f"{out_dir_rel}/{DETECTOR_NAME}"
    inventory_rel = f"{out_dir_rel}/feature_inventory.json"
    readme_rel = f"{out_dir_rel}/README.md"

    return [
        {
            "name": "inventory-features",
            "expects_file": inventory_rel,
            "instruction": f"""
Use the discovery-detector-builder subagent to inventory the features of the
finished AutoDiscovery run {run_rel}. Read the ranked report {summary_rel}, every
loading-fixed hypothesis script under {rerun_rel}/ as indexed by MANIFEST.json,
and markdowns/labeled_dataset_cache.md. The scripts, not report prose, define the
features.

For every returned hypothesis record in {inventory_rel}: id, snake_case feature
name, verdict and reported ROC-AUC, exact mathematical quantity, every constant,
aggregation unit and reduction, traversal phase, the exact condition under which
the value is genuinely measurable, and the historical sentinel/default used by
the hypothesis script when it is not measurable. The measurable-condition field
is mandatory: the generated detector represents undefined as NaN plus an
explicit <feature>_is_defined binary column rather than treating a sentinel as a
measurement. Include WEAK and MINOR hypotheses because multivariate validation,
not a single-feature verdict, decides whether they help. Exclude only hypotheses
the test-fixer overturned or whose corrected effect vanished. Report exclusions
and reasons. Do not load a pkl or invent measured results.
""".strip(),
        },
        {
            "name": "generate-detector",
            "expects_file": detector_rel,
            "instruction": f"""
Use the discovery-detector-builder subagent to generate {detector_rel} from
{inventory_rel} and the sources under {rerun_rel}/. Produce ONE self-contained
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
- Copy feature definitions and constants verbatim from the .rerun scripts. Remove
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
Define candidates and their small explicit grids in one module-level registry:
1. logistic_l2: fold-local SimpleImputer(mean, keep_empty_features=True),
   StandardScaler, balanced L2 LogisticRegression; tune C on a small log grid.
2. logistic_elasticnet: the same preprocessing with solver=saga and
   class_weight=balanced; tune C and l1_ratio on a small grid.
3. spline_logistic: a ColumnTransformer that mean-imputes continuous geometry
   inside the fold, applies a low-capacity SplineTransformer (approximately 3-5
   knots, degree 2), passes the binary is_defined columns, scales as needed, and
   ends in balanced regularized LogisticRegression. Keep its grid small because
   merge positives are scarce.
4. hist_gradient_boosting: shallow HistGradientBoostingClassifier using raw NaNs
   and binary flags, class_weight=balanced, low learning rate, small
   max_leaf_nodes, substantial min_samples_leaf and L2 regularization. Do not put
   an imputer before this native-NaN model.
5. xgboost, OPTIONAL only when importable: shallow XGBClassifier with native NaN
   handling, depth 2-3, conservative min_child_weight/subsampling and strong L2.
   Compute scale_pos_weight from each fitting partition, never an outer validation
   or held-out set. Never install XGBoost. If unavailable, record it as skipped
   and continue successfully.
Pin every seed. Add --n-jobs with a conservative default. Record a candidate
failure with its traceback and exclude it only if other candidates remain; never
silently substitute a different estimator under the same name.

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
  choosing the simplest eligible family in this order: logistic_l2,
  logistic_elasticnet, spline_logistic, hist_gradient_boosting, xgboost. Store
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
- Write model_selection_<brain>.json containing the registry and grids,
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
selection code proving fold-local preprocessing handles NaNs, optional XGBoost
can be skipped, every expected OOF column is filled exactly once, and held-out
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
and the sources under {rerun_rel}/.

VERIFY AND FIX BEFORE DOCUMENTING
1. Every inventory feature is genuinely computed once, keeps original constants,
   reaches the frame and has an extraction-time binary is_defined flag. Undefined
   values are NaN; valid 0/1/180 values are not missing. --exclude-empty uses
   flags rather than sentinels.
2. Statistical imputation, scaling and spline fitting live inside candidate
   pipelines. Remove any whole-dataset fill_mean or preprocessing before CV.
   Native-NaN trees receive NaNs directly.
3. The registry contains logistic_l2, logistic_elasticnet, spline_logistic and
   hist_gradient_boosting, plus optional xgboost without installation. Names,
   implementations, grids and complexity order agree everywhere.
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

Then document provenance, feature mapping and defined conditions; why each model
is included; nested CV and the one-standard-error rule in plain language; the
difference among family OOF, selector OOF, winner OOF, in-sample and held-out
scores; output schemas; compute-node commands; optional XGBoost behavior; and a
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
        # filesystem tools to read the .rerun sources and write the detector.
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
_REGENERATED = frozenset({"feature_inventory.json", DETECTOR_NAME, "README.md",
                          DRIVER_LOG_NAME})


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
    """Warn (don't abort) if a step's expected artifact is missing or empty."""
    rel = step.get("expects_file")
    if not rel:
        return
    path = PROJECT_ROOT / rel
    if not path.is_file():
        log(f"  WARNING: step '{step['name']}' did not create {rel}.")
    elif path.stat().st_size == 0:
        log(f"  WARNING: step '{step['name']}' left {rel} empty.")
    else:
        log(f"  OK: {rel} ({path.stat().st_size} bytes)")



# ─────────────────────────────────────────────────────────────────────────────
# Workflow
# ─────────────────────────────────────────────────────────────────────────────

def resolve_inputs(run_json: Path) -> tuple[Path, Path]:
    """Locate the report and loading-fixed scripts for one discovery run.

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
    predictive_ready = predictive_rerun_dir.is_dir() and any(
        predictive_rerun_dir.glob("hypo_*.py")
    )
    legacy_ready = legacy_rerun_dir.is_dir() and any(
        legacy_rerun_dir.glob("hypo_*.py")
    )
    rerun_dir = predictive_rerun_dir if predictive_ready else legacy_rerun_dir

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
    return summary_path, rerun_dir


async def run_workflow(
    run_json: Path,
    out_dir: Path,
    verbose: bool,
) -> None:
    summary_path, rerun_dir = resolve_inputs(run_json)

    steps = build_steps(
        run_rel=_rel_to_root(run_json),
        summary_rel=_rel_to_root(summary_path),
        rerun_rel=_rel_to_root(rerun_dir),
        out_dir_rel=_rel_to_root(out_dir),
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    log(
        f"Generating detector code from {_rel_to_root(run_json)} → "
        f"{_rel_to_root(out_dir)}/: {len(steps)} agent step(s) "
        f"[{', '.join(s['name'] for s in steps)}]. Nothing is executed against "
        "the data here."
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
        print(f"        {cache or 'cache/dataset_cache_<brain>_mcl<N>_add.pkl'}"
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
            "<RUN>.summary.md report and predictive rerun scripts must already "
            "exist (run run_discovery_workflow.py first)."
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

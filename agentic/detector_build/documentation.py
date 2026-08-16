"""Deterministic README provenance skeleton for detector applications."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

from .inputs import sha256


DRIVER_BLOCK_START = "<!-- BEGIN DRIVER-GENERATED PROVENANCE — preserve this block -->"
DRIVER_BLOCK_END = "<!-- END DRIVER-GENERATED PROVENANCE -->"


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_readme_skeleton(
    readme_path: Path,
    *,
    run_rel: str,
    summary_rel: str,
    inventory_path: Path,
    inventory_rel: str,
    model_config_path: Path,
    model_config_rel: str,
    detector_rel: str,
    run_commands_rel: str,
    cache_hint: str | None,
    primary_metric: str,
) -> None:
    """Write factual mechanical documentation before semantic agent review."""
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    rows = inventory["hypotheses"]
    candidates = model_config["candidates"]

    feature_lines = [
        "| ID | Included | Source | Feature columns | Defined when / exclusion |",
        "|---:|:---:|---|---|---|",
    ]
    for row in rows:
        features = row.get("features") or []
        names = ", ".join(str(feature.get("name")) for feature in features) or "—"
        conditions = "; ".join(
            str(feature.get("measurable_condition")) for feature in features
        ) or row.get("exclusion_reason") or "—"
        feature_lines.append(
            f"| {row['id']} | {'yes' if row['included'] else 'no'} | "
            f"{_cell(row.get('feature_source_path') or '—')} | {_cell(names)} | "
            f"{_cell(conditions)} |"
        )

    model_lines = [
        "| Family | Role | Native NaN | Optional package | Reason |",
        "|---|---|:---:|---|---|",
    ]
    for candidate in candidates:
        model_lines.append(
            f"| {_cell(candidate['name'])} | {_cell(candidate['role'])} | "
            f"{'yes' if candidate['native_nan'] else 'no'} | "
            f"{_cell(candidate.get('requires_package') or '—')} | "
            f"{_cell(candidate['reason'])} |"
        )

    cache = cache_hint or "cache/dataset_cache_<brain>_mcl<N>_add.pkl"
    content = f"""# Merge detector — `{Path(run_rel).stem}`

> **Build status:** the detector code has been generated and checked without a
> real dataset. It has **not** been run on a `_add.pkl`; this README contains no
> measured model winner, performance metric, class count, or detector result.

{DRIVER_BLOCK_START}

## Provenance

- Origin run: `{run_rel}`
- Finished report: `{summary_rel}`
- Feature inventory: `{inventory_rel}`
- Model configuration: `{model_config_rel}`
- Generated detector: `{detector_rel}`
- Instance-bound command guide: `{run_commands_rel}`
- Selection manifest: `{inventory.get('selection_manifest')}`

## Feature and source mapping

{chr(10).join(feature_lines)}

Every numeric feature is represented as its raw value plus a corresponding
`<feature>_is_defined` flag. Undefined values remain NaN; historical sentinel
values are provenance, not the new missingness representation.

## Candidate models

{chr(10).join(model_lines)}

Primary metric: `{primary_metric}`. Candidate-set rationale:
{model_config.get('selection_basis')}

Actual family selection happens only through nested cross-validation in the
generated detector. Weighted classifier outputs are scores and are not
automatically calibrated probabilities.

## Run on a compute node

See `{run_commands_rel}` for hash checks, smoke tests, isolated output
directories, n169/Slurm examples, held-out transfer, and environment capture.

```bash
conda activate panda
python {detector_rel} \\
    {cache} \\
    --model-config {model_config_rel}
```

The pkl requires more than 20 GB RAM. Use `--test-pkl` for a different held-out
brain; it must not influence model or preprocessing selection.

## Expected outputs

- `merge_detector_<brain>.csv`
- `model_selection_<brain>.json`
- `merge_detector_<brain>.joblib`
- `merge_detector_<brain>.log.txt`
- `figures/`
- `analysis_timing_<brain>.json` only for the profiling-only `--measuretime` run
- `hypothesis_cost_report_<brain>.md` and
  `hypothesis_selection_template_<brain>.json` from the same profiling run

Read undefined coverage first, then average precision and review-budget
precision/recall, then held-out transfer; read ROC-AUC last. Training thresholds
and queues must use nested out-of-fold scores, never the final in-sample score.

{DRIVER_BLOCK_END}

## Semantic review

The final builder stage should add feature-specific interpretation, exclusions,
correction/source rationale, and any limitations found while reviewing the
generated implementation. It must not invent results from a real dataset.
"""
    temporary = readme_path.with_name(f".{readme_path.name}.skeleton.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(readme_path)


def read_driver_generated_block(readme_path: Path) -> str:
    """Return the immutable provenance block or reject a damaged README."""
    text = readme_path.read_text(encoding="utf-8")
    start = text.find(DRIVER_BLOCK_START)
    end = text.find(DRIVER_BLOCK_END)
    if start < 0 or end < start:
        raise SystemExit(
            f"README {readme_path} is missing its driver-generated provenance block."
        )
    end += len(DRIVER_BLOCK_END)
    return text[start:end]


def validate_readme_enrichment(
    readme_path: Path,
    *,
    expected_driver_block: str,
    skeleton_sha256: str,
) -> None:
    """Require semantic enrichment while preserving driver-owned provenance."""
    if read_driver_generated_block(readme_path) != expected_driver_block:
        raise SystemExit(
            "The verify-and-document stage modified the immutable "
            "driver-generated README provenance block."
        )
    if sha256(readme_path) == skeleton_sha256:
        raise SystemExit(
            "The verify-and-document stage left the driver README skeleton "
            "unchanged; semantic review was not written."
        )


def write_run_commands(
    commands_path: Path,
    *,
    project_root: Path,
    run_rel: str,
    detector_path: Path,
    detector_rel: str,
    inventory_path: Path,
    inventory_rel: str,
    model_config_path: Path,
    model_config_rel: str,
    model_policy_path: Path,
    model_policy_rel: str,
    runtime_template_path: Path,
    runtime_template_rel: str,
    cache_hint: str | None,
    agent_model: str,
    agent_effort: str,
) -> None:
    """Write a deterministic runbook bound to one assembled detector instance.

    Agent generation is intentionally not described as reproducible.  Instead,
    this file records the exact artifact hashes and commands needed to validate
    and run the already-generated instance.
    """
    cache = cache_hint or "cache/dataset_cache_<brain>_mcl<N>_add.pkl"
    brain_match = re.search(r"dataset_cache_(\d+)_", cache)
    brain = brain_match.group(1) if brain_match else "<brain>"
    app_dir = Path(detector_rel).parent.as_posix()

    detector_sha = sha256(detector_path)
    inventory_sha = sha256(inventory_path)
    model_config_sha = sha256(model_config_path)
    model_policy_sha = sha256(model_policy_path)
    runtime_template_sha = sha256(runtime_template_path)

    q_root = shlex.quote(project_root.as_posix())
    q_app = shlex.quote(app_dir)
    q_cache = shlex.quote(cache)
    q_brain = shlex.quote(brain)
    content = f"""# Commands for `{Path(run_rel).stem}`

This runbook is generated deterministically by the detector-build driver and is
bound to the exact assembled artifacts below. The Claude stages that selected
feature semantics, proposed model candidates, and wrote feature code are **not
purely deterministic**; rerunning the build may produce a different valid
instance. Use the hashes here when reproducing or comparing this instance.

## Bound instance

| Artifact | Path | SHA-256 |
|---|---|---|
| Detector | `{detector_rel}` | `{detector_sha}` |
| Feature inventory | `{inventory_rel}` | `{inventory_sha}` |
| Model configuration | `{model_config_rel}` | `{model_config_sha}` |
| Model policy used at build time | `{model_policy_rel}` | `{model_policy_sha}` |
| Runtime template used at build time | `{runtime_template_rel}` | `{runtime_template_sha}` |

- Origin run: `{run_rel}`
- Agent model: `{agent_model}`
- Agent effort: `{agent_effort}`
- Suggested origin cache: `{cache}`

`RUN_COMMANDS.md` does not hash itself because that would be self-referential.

## Set paths

Run commands from the project root:

```bash
cd {q_root}
conda activate panda

APP_DIR={q_app}
DETECTOR="$APP_DIR/merge_site_detector.py"
INVENTORY="$APP_DIR/feature_inventory.json"
MODEL_CONFIG="$APP_DIR/model_candidates.json"
DATA_PKL={q_cache}
BRAIN={q_brain}
```

Check `DATA_PKL` against the origin dataset recorded in the finished report
before submitting a compute job.

## Verify this exact instance

```bash
printf '%s  %s\\n' '{detector_sha}' "$DETECTOR" | sha256sum --check
printf '%s  %s\\n' '{inventory_sha}' "$INVENTORY" | sha256sum --check
printf '%s  %s\\n' '{model_config_sha}' "$MODEL_CONFIG" | sha256sum --check
```

Keep the detector, inventory, and model configuration together. The runtime
cross-checks the inventory hash recorded in the model configuration.

## No-data checks

```bash
python "$DETECTOR" --help

python "$DETECTOR" \\
    --synthetic-smoke-test \\
    --model-config "$MODEL_CONFIG" \\
    --inventory "$INVENTORY"
```

The smoke command must print `DETECTOR_SMOKE_OK` and must not create detector
outputs.

## Profile hypothesis computational cost

This loads the real cache and runs every candidate hypothesis on only a small,
deterministic sample of component-bearing segments (3 by default), then exits before the
undefined audit, nested CV, model fitting, held-out scoring, CSV, joblib,
model-selection JSON, or figures. Its purpose is to estimate which hypothesis
computations can be removed from the final run. Timing does not make that decision
automatically; inspect expensive units for optimization and scientific value. The
reported seconds are observed sample costs, not full-run projections.

```bash
RUN_DIR="$APP_DIR/runs/${{BRAIN}}-measuretime"
mkdir -p "$RUN_DIR"

python "$DETECTOR" "$DATA_PKL" \\
    --measuretime \\
    --measuretime-occurrences 3 \\
    --inventory "$INVENTORY" \\
    --out-dir "$RUN_DIR" \\
    --log-txt "$RUN_DIR/measuretime_${{BRAIN}}.log.txt"
```

The profiling directory contains:

- `analysis_timing_${{BRAIN}}.json`: schema-v2 machine-readable timings;
- `hypothesis_cost_report_${{BRAIN}}.md`: selection units ranked by observed
  sample seconds;
- `hypothesis_selection_template_${{BRAIN}}.json`: editable final-run selection;
- the requested text log.

The JSON is atomically checkpointed with `running`, `complete`, or `failed`
status. Per-hypothesis rows distinguish exclusive cost from shared selection-unit
cost. A row containing multiple hypothesis ids saves its cost only when the whole
row is excluded. `estimated_removable_seconds` covers only the sampled timed
analysis body, not phase-level shared overhead or projected full-run savings, so
use it only as a rough relative-cost ranking.

## Choose hypotheses and run the reduced final detector

After reading the cost report, the shortest route is to pass IDs directly:

```bash
# Add one or more IDs from this run's cost report.
EXCLUDE_IDS=()
if ((${{#EXCLUDE_IDS[@]}} == 0)); then
    echo "Set EXCLUDE_IDS from the cost report before running." >&2
    exit 2
fi

RUN_DIR="$APP_DIR/runs/${{BRAIN}}-manual-exclusions"
mkdir -p "$RUN_DIR"

python "$DETECTOR" "$DATA_PKL" \\
    --exclude-hypotheses "${{EXCLUDE_IDS[@]}}" \\
    --model-config "$MODEL_CONFIG" \\
    --inventory "$INVENTORY" \\
    --out-dir "$RUN_DIR" \\
    --out-csv "$RUN_DIR/merge_detector_${{BRAIN}}.csv" \\
    --log-txt "$RUN_DIR/merge_detector_${{BRAIN}}.log.txt" \\
    --n-jobs 2 \\
    --no-figures
```

For a shared selection unit, list every hypothesis id in the row or none;
partial shared exclusions are rejected. Unknown IDs, duplicates, and excluding
every hypothesis are also rejected. Direct exclusions are recorded as
`selection_source=manual_cli` in model JSON and joblib.

For an auditable selection with per-ID reasons and timing SHA provenance, copy
the generated template, edit `excluded_hypothesis_ids`, and run:

```bash
SELECTION="$APP_DIR/hypothesis_selection_${{BRAIN}}.json"
cp "$APP_DIR/runs/${{BRAIN}}-measuretime/hypothesis_selection_template_${{BRAIN}}.json" \\
   "$SELECTION"
# Edit $SELECTION after reviewing hypothesis_cost_report_${{BRAIN}}.md.

RUN_DIR="$APP_DIR/runs/${{BRAIN}}-selected"
mkdir -p "$RUN_DIR"

python "$DETECTOR" "$DATA_PKL" \\
    --hypothesis-selection "$SELECTION" \\
    --model-config "$MODEL_CONFIG" \\
    --inventory "$INVENTORY" \\
    --out-dir "$RUN_DIR" \\
    --out-csv "$RUN_DIR/merge_detector_${{BRAIN}}.csv" \\
    --log-txt "$RUN_DIR/merge_detector_${{BRAIN}}.log.txt" \\
    --n-jobs 2 \\
    --no-figures
```

The selected run preserves the normal row and CSV column schema. Excluded
features remain present as NaN with `<feature>_is_defined=False`, and their owned
computations are skipped. The same resolved selection is applied to `--test-pkl`
and saved, together with validated timing provenance and documented reasons, in
model-selection JSON and joblib provenance. A generated selection is accepted
only when its timing artifact is complete and matches this exact detector and
inventory. Omitting both source-timing fields remains valid for a deliberately
hand-authored selection.
Omitting both `--hypothesis-selection` and `--exclude-hypotheses` preserves the
original all-hypothesis behavior.

## First real run: extraction and modelling without figures

Real ExaSPIM caches can require tens of GB and component traversal can take
hours. Use a compute node, request at least 80 GB RAM unless a measured profile
justifies less, and allow several hours.

```bash
RUN_DIR="$APP_DIR/runs/${{BRAIN}}-default"
mkdir -p "$RUN_DIR"

python "$DETECTOR" "$DATA_PKL" \\
    --model-config "$MODEL_CONFIG" \\
    --inventory "$INVENTORY" \\
    --out-dir "$RUN_DIR" \\
    --out-csv "$RUN_DIR/merge_detector_${{BRAIN}}.csv" \\
    --log-txt "$RUN_DIR/merge_detector_${{BRAIN}}.log.txt" \\
    --n-jobs 2 \\
    --no-figures
```

## Monitor component extraction

In another shell, follow the durable log and inspect component timing records:

```bash
tail -f "$RUN_DIR/merge_detector_${{BRAIN}}.log.txt"

rg '\\[extract\\]\\[component\\]' "$RUN_DIR/merge_detector_${{BRAIN}}.log.txt"
```

Each segment and component has start/done progress. Each component-pass feature
has a flushed start record before its work and a done record with elapsed time.
If the process hangs or is killed, the final unmatched `feature:start` line names
the active feature; successful completion prints per-feature `calls` and `total_s`.

## Full run with figures

Use a separate directory so it cannot overwrite the first run's evidence:

```bash
RUN_DIR="$APP_DIR/runs/${{BRAIN}}-with-figures"
mkdir -p "$RUN_DIR"

python "$DETECTOR" "$DATA_PKL" \\
    --model-config "$MODEL_CONFIG" \\
    --inventory "$INVENTORY" \\
    --out-dir "$RUN_DIR" \\
    --out-csv "$RUN_DIR/merge_detector_${{BRAIN}}.csv" \\
    --log-txt "$RUN_DIR/merge_detector_${{BRAIN}}.log.txt" \\
    --fig-dir "$RUN_DIR/figures" \\
    --n-jobs 2
```

## `--exclude-empty` sensitivity run

This changes the evaluated row scope, so keep it separate from the default run:

```bash
RUN_DIR="$APP_DIR/runs/${{BRAIN}}-exclude-empty"
mkdir -p "$RUN_DIR"

python "$DETECTOR" "$DATA_PKL" \\
    --model-config "$MODEL_CONFIG" \\
    --inventory "$INVENTORY" \\
    --out-dir "$RUN_DIR" \\
    --out-csv "$RUN_DIR/merge_detector_${{BRAIN}}_exclude_empty.csv" \\
    --log-txt "$RUN_DIR/merge_detector_${{BRAIN}}_exclude_empty.log.txt" \\
    --exclude-empty \\
    --n-jobs 2 \\
    --no-figures
```

## Held-out brain transfer

Replace the held-out path; it must not be the training cache:

```bash
TEST_PKL='cache/dataset_cache_<heldout-brain>_mcl<N>_add.pkl'
RUN_DIR="$APP_DIR/runs/${{BRAIN}}-heldout-transfer"
mkdir -p "$RUN_DIR"

python "$DETECTOR" "$DATA_PKL" \\
    --test-pkl "$TEST_PKL" \\
    --model-config "$MODEL_CONFIG" \\
    --inventory "$INVENTORY" \\
    --out-dir "$RUN_DIR" \\
    --out-csv "$RUN_DIR/merge_detector_${{BRAIN}}.csv" \\
    --log-txt "$RUN_DIR/merge_detector_${{BRAIN}}.log.txt" \\
    --fig-dir "$RUN_DIR/figures" \\
    --n-jobs 2
```

## Site-specific Slurm example for `n169` (edit for another cluster)

```bash
RUN_DIR="$APP_DIR/runs/${{BRAIN}}-n169"
mkdir -p "$RUN_DIR"

sbatch \\
    --partition=aibs_debug \\
    --nodelist=n169 \\
    --mem=80G \\
    --cpus-per-task=2 \\
    --time=04:00:00 \\
    --job-name=merge-detector \\
    --output="$RUN_DIR/slurm-%j.out" \\
    --chdir={q_root} \\
    --wrap="source /shared/utils.x86_64/anaconda3-2024.10/etc/profile.d/conda.sh && conda activate panda && python \\\"$DETECTOR\\\" \\\"$DATA_PKL\\\" --model-config \\\"$MODEL_CONFIG\\\" --inventory \\\"$INVENTORY\\\" --out-dir \\\"$RUN_DIR\\\" --out-csv \\\"$RUN_DIR/merge_detector_${{BRAIN}}.csv\\\" --log-txt \\\"$RUN_DIR/merge_detector_${{BRAIN}}.log.txt\\\" --n-jobs 2 --no-figures"
```

Check status and follow the logs with:

```bash
squeue -u "$USER"
tail -f "$RUN_DIR"/slurm-*.out
tail -f "$RUN_DIR/merge_detector_${{BRAIN}}.log.txt"
```

## Expected outputs in each run directory

- `merge_detector_{brain}.csv`
- `model_selection_{brain}.json`
- `merge_detector_{brain}.joblib`
- `merge_detector_{brain}.log.txt`
- `figures/` unless `--no-figures` is used

The separate `--measuretime` directory contains the timing JSON, ranked cost
report, editable selection template, and requested log; it contains no model
output.

Read undefined coverage first, then average precision and review-budget
precision/recall, then held-out transfer, and ROC-AUC last.

## Reproducibility record

The detector uses fixed CV splits and seed 42, but bitwise equality is not
guaranteed across dependency versions, optional-family availability, thread
counts, or unordered graph traversal in generated feature code. Capture the
environment beside the run:

```bash
conda list --explicit > "$RUN_DIR/conda-explicit.txt"
python --version > "$RUN_DIR/python-version.txt"
sha256sum "$DATA_PKL" > "$RUN_DIR/input-pkl.sha256"
```
"""
    temporary = commands_path.with_name(f".{commands_path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(commands_path)

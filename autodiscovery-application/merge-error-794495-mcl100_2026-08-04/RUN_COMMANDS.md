# Commands for `merge-error-794495-mcl100_2026-08-04`

This runbook is generated deterministically by the detector-build driver and is
bound to the exact assembled artifacts below. The Claude stages that selected
feature semantics, proposed model candidates, and wrote feature code are **not
purely deterministic**; rerunning the build may produce a different valid
instance. Use the hashes here when reproducing or comparing this instance.

## Bound instance

| Artifact | Path | SHA-256 |
|---|---|---|
| Detector | `autodiscovery-application/merge-error-794495-mcl100_2026-08-04/merge_site_detector.py` | `fed266227b73474f8e7de30c5dbfe10d34155c55b4019a162906ddfcd29664ac` |
| Feature inventory | `autodiscovery-application/merge-error-794495-mcl100_2026-08-04/feature_inventory.json` | `061ac2cadc48b47d043ad9820d565d4152537af6b2e94bc624bc87ddbd68bbb1` |
| Model configuration | `autodiscovery-application/merge-error-794495-mcl100_2026-08-04/model_candidates.json` | `030bf1e48aef83190eb1fb4101acc9f299712ee52a995000e7aa4b83b14ea941` |
| Model policy used at build time | `agentic/detector_model_policy.json` | `4aa9c7a9411f73438faeb74cb6459db084db4aabd82cfa9861328e32c9b22f3c` |
| Current runtime template (synced post-build) | `agentic/detector_build/templates/detector_runtime.py.tmpl` | `417d5644927403df68e1a72e614173966e7613b15e6fe5c6c96ff2c4897e09b0` |

- Origin run: `autodiscovery/merge-error-794495-mcl100_2026-08-04.json`
- Agent model: `claude-opus-4-8`
- Agent effort: `xhigh`
- Suggested origin cache: `cache/dataset_cache_794495_mcl100_add.pkl`

`RUN_COMMANDS.md` does not hash itself because that would be self-referential.
The detector hash includes the post-build hypothesis-40 repair, component
progress instrumentation, hypothesis-cost profiling, explicit final-run
selection, provenance validation, stale-metadata cleanup, the sampled
measuretime limit, and direct validated `--exclude-hypotheses` input.
It also includes winner-aware figure 06: signed coefficients for ordinary
logistic winners, EBM term importance plus top shape functions, native unsigned
tree importance when available, and a cached permutation fallback otherwise.

## Set paths

Run commands from the project root:

```bash
cd /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent
conda activate panda

APP_DIR=autodiscovery-application/merge-error-794495-mcl100_2026-08-04
DETECTOR="$APP_DIR/merge_site_detector.py"
INVENTORY="$APP_DIR/feature_inventory.json"
MODEL_CONFIG="$APP_DIR/model_candidates.json"
DATA_PKL=cache/dataset_cache_794495_mcl100_add.pkl
BRAIN=794495
```

Check `DATA_PKL` against the origin dataset recorded in the finished report
before submitting a compute job.

## Verify this exact instance

```bash
printf '%s  %s\n' 'fed266227b73474f8e7de30c5dbfe10d34155c55b4019a162906ddfcd29664ac' "$DETECTOR" | sha256sum --check
printf '%s  %s\n' '061ac2cadc48b47d043ad9820d565d4152537af6b2e94bc624bc87ddbd68bbb1' "$INVENTORY" | sha256sum --check
printf '%s  %s\n' '030bf1e48aef83190eb1fb4101acc9f299712ee52a995000e7aa4b83b14ea941' "$MODEL_CONFIG" | sha256sum --check
```

Keep the detector, inventory, and model configuration together. The runtime
cross-checks the inventory hash recorded in the model configuration.

## No-data checks

```bash
python "$DETECTOR" --help

python "$DETECTOR" \
    --synthetic-smoke-test \
    --model-config "$MODEL_CONFIG" \
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
RUN_DIR="$APP_DIR/runs/${BRAIN}-measuretime"
mkdir -p "$RUN_DIR"

python "$DETECTOR" "$DATA_PKL" \
    --measuretime \
    --measuretime-occurrences 3 \
    --inventory "$INVENTORY" \
    --out-dir "$RUN_DIR" \
    --log-txt "$RUN_DIR/measuretime_${BRAIN}.log.txt"
```

The profiling directory contains:

- `analysis_timing_${BRAIN}.json`: schema-v2 machine-readable timings;
- `hypothesis_cost_report_${BRAIN}.md`: selection units ranked by observed
  sample seconds;
- `hypothesis_selection_template_${BRAIN}.json`: editable final-run selection;
- the requested text log.

The pre-existing `analysis_timing_794495.json` directly under `APP_DIR` is an
incomplete diagnostic artifact (`status=running`) from an older detector hash.
Do not use it for selection; rerun the command above into the dedicated
measuretime directory with the current detector.

The JSON is atomically checkpointed with `running`, `complete`, or `failed`
status. Per-hypothesis rows distinguish exclusive cost from shared selection-unit
cost. A row containing multiple hypothesis ids saves its cost only when the whole
row is excluded. `estimated_removable_seconds` covers only the sampled timed
analysis body, not phase-level shared overhead or projected full-run savings, so
use it only as a rough relative-cost ranking.

## Choose hypotheses and run the reduced final detector

For this detector, hypotheses 39 and 6 are both independent selection units, so
they can be excluded directly. This is the requested ready-to-run example; the
existing app-root timing artifact is incomplete, so confirm the choice against a
new complete sampled cost report before treating it as a final scientific decision.

```bash
RUN_DIR="$APP_DIR/runs/${BRAIN}-exclude-39-6"
mkdir -p "$RUN_DIR"

python "$DETECTOR" "$DATA_PKL" \
    --exclude-hypotheses 39 6 \
    --model-config "$MODEL_CONFIG" \
    --inventory "$INVENTORY" \
    --out-dir "$RUN_DIR" \
    --out-csv "$RUN_DIR/merge_detector_${BRAIN}.csv" \
    --log-txt "$RUN_DIR/merge_detector_${BRAIN}.log.txt" \
    --n-jobs 2 \
    --no-figures
```

The detector rejects unknown IDs, duplicates, excluding every hypothesis, and a
partial shared unit. Direct exclusions are saved as
`selection_source=manual_cli` in model-selection JSON and joblib.

For a longer-lived, auditable selection with per-ID reasons and timing SHA
provenance, copy the generated template, edit `excluded_hypothesis_ids`, and run:

```bash
SELECTION="$APP_DIR/hypothesis_selection_${BRAIN}.json"
cp "$APP_DIR/runs/${BRAIN}-measuretime/hypothesis_selection_template_${BRAIN}.json" \
   "$SELECTION"
# Edit $SELECTION after reviewing hypothesis_cost_report_${BRAIN}.md.

RUN_DIR="$APP_DIR/runs/${BRAIN}-selected"
mkdir -p "$RUN_DIR"

python "$DETECTOR" "$DATA_PKL" \
    --hypothesis-selection "$SELECTION" \
    --model-config "$MODEL_CONFIG" \
    --inventory "$INVENTORY" \
    --out-dir "$RUN_DIR" \
    --out-csv "$RUN_DIR/merge_detector_${BRAIN}.csv" \
    --log-txt "$RUN_DIR/merge_detector_${BRAIN}.log.txt" \
    --n-jobs 2 \
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
RUN_DIR="$APP_DIR/runs/${BRAIN}-default"
mkdir -p "$RUN_DIR"

python "$DETECTOR" "$DATA_PKL" \
    --model-config "$MODEL_CONFIG" \
    --inventory "$INVENTORY" \
    --out-dir "$RUN_DIR" \
    --out-csv "$RUN_DIR/merge_detector_${BRAIN}.csv" \
    --log-txt "$RUN_DIR/merge_detector_${BRAIN}.log.txt" \
    --n-jobs 2 \
    --no-figures
```

## Monitor component extraction

In another shell, follow the durable log and inspect component timing records:

```bash
tail -f "$RUN_DIR/merge_detector_${BRAIN}.log.txt"

rg '\[extract\]\[component\]' "$RUN_DIR/merge_detector_${BRAIN}.log.txt"
```

Each segment and component has start/done progress. Each component-pass feature
has a flushed start record before its work and a done record with elapsed time.
If the process hangs or is killed, the final unmatched `feature:start` line names
the active feature; successful completion prints per-feature `calls` and `total_s`.

## Full run with figures

Use a separate directory so it cannot overwrite the first run's evidence:

```bash
RUN_DIR="$APP_DIR/runs/${BRAIN}-with-figures"
mkdir -p "$RUN_DIR"

python "$DETECTOR" "$DATA_PKL" \
    --model-config "$MODEL_CONFIG" \
    --inventory "$INVENTORY" \
    --out-dir "$RUN_DIR" \
    --out-csv "$RUN_DIR/merge_detector_${BRAIN}.csv" \
    --log-txt "$RUN_DIR/merge_detector_${BRAIN}.log.txt" \
    --fig-dir "$RUN_DIR/figures" \
    --n-jobs 2
```

## `--exclude-empty` sensitivity run

This changes the evaluated row scope, so keep it separate from the default run:

```bash
RUN_DIR="$APP_DIR/runs/${BRAIN}-exclude-empty"
mkdir -p "$RUN_DIR"

python "$DETECTOR" "$DATA_PKL" \
    --model-config "$MODEL_CONFIG" \
    --inventory "$INVENTORY" \
    --out-dir "$RUN_DIR" \
    --out-csv "$RUN_DIR/merge_detector_${BRAIN}_exclude_empty.csv" \
    --log-txt "$RUN_DIR/merge_detector_${BRAIN}_exclude_empty.log.txt" \
    --exclude-empty \
    --n-jobs 2 \
    --no-figures
```

## Held-out brain transfer

Replace the held-out path; it must not be the training cache:

```bash
TEST_PKL='cache/dataset_cache_<heldout-brain>_mcl<N>_add.pkl'
RUN_DIR="$APP_DIR/runs/${BRAIN}-heldout-transfer"
mkdir -p "$RUN_DIR"

python "$DETECTOR" "$DATA_PKL" \
    --test-pkl "$TEST_PKL" \
    --model-config "$MODEL_CONFIG" \
    --inventory "$INVENTORY" \
    --out-dir "$RUN_DIR" \
    --out-csv "$RUN_DIR/merge_detector_${BRAIN}.csv" \
    --log-txt "$RUN_DIR/merge_detector_${BRAIN}.log.txt" \
    --fig-dir "$RUN_DIR/figures" \
    --n-jobs 2
```

## Site-specific Slurm example for `n169` (edit for another cluster)

```bash
RUN_DIR="$APP_DIR/runs/${BRAIN}-n169"
mkdir -p "$RUN_DIR"

sbatch \
    --partition=aibs_debug \
    --nodelist=n169 \
    --mem=80G \
    --cpus-per-task=2 \
    --time=04:00:00 \
    --job-name=merge-detector \
    --output="$RUN_DIR/slurm-%j.out" \
    --chdir=/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent \
    --wrap="source /shared/utils.x86_64/anaconda3-2024.10/etc/profile.d/conda.sh && conda activate panda && python \"$DETECTOR\" \"$DATA_PKL\" --model-config \"$MODEL_CONFIG\" --inventory \"$INVENTORY\" --out-dir \"$RUN_DIR\" --out-csv \"$RUN_DIR/merge_detector_${BRAIN}.csv\" --log-txt \"$RUN_DIR/merge_detector_${BRAIN}.log.txt\" --n-jobs 2 --no-figures"
```

Check status and follow the logs with:

```bash
squeue -u "$USER"
tail -f "$RUN_DIR"/slurm-*.out
tail -f "$RUN_DIR/merge_detector_${BRAIN}.log.txt"
```

## Expected outputs in each run directory

- `merge_detector_794495.csv`
- `model_selection_794495.json`
- `merge_detector_794495.joblib`
- `merge_detector_794495.log.txt`
- `figures/` unless `--no-figures` is used

The separate `--measuretime` directory contains the timing JSON, ranked cost
report, editable selection template, and requested log; it contains no model
output.

Read undefined coverage first, then average precision and review-budget
precision/recall, then held-out transfer, and ROC-AUC last.

## Generate the bilingual result-analysis report

The completed 794495 result is currently stored directly in `APP_DIR` and has
all training figures 01–07, so its path input is `"$APP_DIR"`:

```bash
RUN_DIR="$APP_DIR"
python agentic/run_detector_result_analysis.py "$RUN_DIR"
```

The analysis driver checks the successful log terminator, JSON/CSV counts,
joblib presence, training provenance hashes, and every figure before invoking
`detector-result-analyst`. It does not reload the pkl, load the fitted joblib, or
rerun feature extraction/model fitting. It writes:

- `result_analysis_evidence.json` (deterministic evidence and artifact hashes);
- `RESULT_ANALYSIS.md` (full English report followed by its full Chinese
  translation, with immutable numeric evidence);
- `result_analysis_workflow.log.txt` (agent/validation log).

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

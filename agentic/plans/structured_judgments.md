# Structured judgments: output diff, canonical statistics, verdicts JSON

Status: implemented 2026-10-20 (`agentic/output_diff.py`, `agentic/canonical_stats.py`,
`agentic/rerun_experiments.py`, `agentic/run_discovery_workflow.py`,
`agentic/detector_build/inputs.py`, five agent specs, tests). Not yet run on a full
workflow.

## Why

The workflow's purpose is a report whose verdicts the detector build can trust. On the
new-merge-1 run of 2026-10-19 three of its judgments depended on agents reading prose:

1. The reproduce fold compared 91 pairs of outputs by eye, and the results JSON truncated
   them (6,000 characters), so it dug the full text out of the checkpoint.
2. The verifier found increments measured against five different "baselines" (logistic
   fits of degree at 0.0258 and 0.0286, a boosted model at 0.0185, non-tie-aware APs at
   0.0319 and 0.0410) where the raw tie-aware degree ranking gives 0.0278, and corrected
   scripts kept their statistics comparable only by hand-copying one batch's block.
3. Verdict tokens were parsed out of markdown by four regex sets (driver, detector
   build, heatmap, collector); a token outside the ladder, or a MINOR without the
   `Concrete test fault` phrase, silently dropped a hypothesis.

## 1. Output diff (`agentic/output_diff.py`)

- The runner's `--outputs-dir` writes every executed record's full stdout and stderr
  (`hypo_<id>[.<pkl stem>].stdout.txt`) and records the path in the result; the driver
  passes one dir per phase (`.reproduce.json.outputs/`, `.extrapolate.json.outputs/`,
  `.corrected.json.outputs/`).
- `output_diff.py RUN.json RESULTS.json --outputs-dir DIR` strips AutoDiscovery's
  `codeOutput` wrapper, drops noise lines (durations, throughput, paths, clock times,
  progress, pip notices, glog, memory), pairs lines by template (numbers replaced by
  `#`, repeated templates in order), and classes each paired value exact, tiny (rel
  1e-3 / abs 1e-4) or material. Per id: `hint` (identical / numeric-differences /
  structural-differences / not-usable), counts, material mismatches with both values,
  lines on one side only.
- New compute step `reproduce-diff` right before `reproduce-fold`; the fold instruction
  and the reproducer spec tell the agent to start from it.

## 2. Canonical statistics (`agentic/canonical_stats.py`)

- The specs' scoring code verbatim (`candidate_keys`, `precision_at_fraction`, tie-aware
  `average_precision`, `labeled_share_at_fraction`), plus `baseline` (degree or `-gap_um`,
  the ranking itself), `fit_rows` / `held_rows` (purged masks), `finite` (NaN last),
  `paired_bootstrap_delta_ap` (within-fold resampling, same draws), `evaluate` and
  `report`, which prints a `CANONICAL ...` block with fixed labels.
- `card` subcommand: the baseline card of a companion. On 794495 it reproduces the specs'
  figures exactly (merge 0.0278, 7/97 at 1%; split 0.8255, 56/61 at 1%).
- The driver computes the cards at startup (`.baseline.json`) and prints them into the
  verify, extrapolate-fold and fix-tests-author instructions (`baseline_rule`).
- The runner links the module beside any script that names it and puts its digest in the
  work key; corrected scripts must import it by the bare name and print `report()`.

## 3. Verdicts JSON

- Each fold step's instruction carries `verdicts_rule(phase, file, ...)`: the agent also
  writes `<RUN>.<phase>-verdicts.json` with `{"phase", "entries": {id: {...}}}`.
  Verify entries carry `verdict`, `test_fault`, `deployability` and a `headline`
  (p, delta_ap, ci95, baseline); fix_tests entries allow a null verdict.
- `_check_phase_verdicts`: file present, phase matches, tokens in `VERDICT_VOCAB`, exactly
  the ids the phase's results JSON measured (`corrected_only` for fix-tests), agreement with
  the report's bullet for every token the report carries, and the bullet's presence.
  Problems get one reconciliation turn; a second failure aborts with the step to resume.
- `_merge_phase_verdicts` folds the file into `<RUN>.verdicts.json`
  (`schema_version` 1, `entries[id][phase]`, `phases[phase].updated`).
- `fix_tests_selection` reads the merged file first; `detector_build.inputs.report_evidence_by_id`
  prefers `<stem>.json[.predictive].verdicts.json` beside the report. Both fall back to the
  bullets for older reports. `summary_heatmap.py` and `collect_summaries.py` still parse
  markdown.

## Tests

- `test_canonical_stats.py`: AP equals sklearn without ties and collapses ties; key-order
  precision counts; raw baselines; NaN last; bootstrap zero / positive / seeded; the
  CANONICAL block; purged rows; the card from an npz and its text.
- `test_output_diff.py`: wrapper stripping; identical with fit noise; a material change
  paired in order; one-sided lines; value-count change; the CLI with an outputs dir.
- `test_rerun_experiments.py`: the module is linked on both paths; its digest keys the
  work item only when used; `--outputs-dir` files and paths.
- `test_discovery_smoke.py`: diff step placement and argv; outputs dirs on every run step;
  verdicts specs on every fold; the merge, the named problems, verify and fix-tests
  checks, selection from the merged file; baseline cards in the instructions.
- `test_blindness_gates.py`: the detector build prefers the merged JSON.

## Rollout

Run new-merge-1 from scratch. Expected: `[baseline] ... degree AP 0.0278` at startup;
`reproduce-diff` writing the diff; `[verdicts] merged N ... entr(ies)` after every fold;
corrected scripts printing `CANONICAL` lines; `<RUN>.predictive.verdicts.json` complete
at the end. A reconciliation turn in the log points at a fold whose JSON and report
disagreed.

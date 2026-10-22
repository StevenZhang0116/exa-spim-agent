# Detector build revision: three-valued labels, purged block folds, batched inventory

Status: planned 2026-10-20 after reviewing `agentic/run_detector_build_workflow.py`,
`agentic/detector_build/` and the runtime template against the finished
`new-split-1_2026-10-20` report, then checked against how Stage 3
(`proofreader_evolve/cli/precompute_error_scores.py`, `harness/native_pool.py`) consumes a
detector (section "Stage 3 contract"). Implemented 2026-10-21: items 1 (three-valued
labels through `candidate_rules.label_availability` / `three_valued_labels`, applied by
the adapters; the runtime fits, selects and evaluates on judgeable rows and scores every
row), 2 (`candidate_rules.fold_groups` / `fold_partitions`, the harness's purged group
folds with brain-prefixed keys so the held sets equal the companions'; outer folds 3;
the embedded policy copy in the template updated with it), 3 (names from the target
spec; Stage 3 resolver accepts `runs/<brain>-*` directories bound by
`data_provenance.detector_sha256`), 4 (`feature_effect` in the fix-tests verdicts and the
`--keep-overturned` gate), 7 (one executable check), 8 (`img_path` whitelisted and the
image-location rule) and the spec refresh of item 9 except the batched-inventory
contract. Items 5 and 6 (batched inventory with the `h<id>_` prefix, recovery of an
assembled detector) are still to do; neither blocks a build.

## Why

Measured on the 794495 split companion (689,677 rows, 6,076 judgeable, 2,497 positive)
with the runtime's own fold rule and seed:

| fold rule as implemented | rows that get an OOF score | judgeable OOF rows | OOF positives | fit rows per fold |
|---|---|---|---|---|
| outer 5 folds | 20.0% | 1,232 | 516 | 64% |
| inner 3 folds | 33.3% | 2,070 | 867 | 44% |

- `_cv_splits` assigns each segment a fold at random and keeps a pair in the held fold
  only when both segments land there; every other pair is dropped from that fold. Each
  inner fit therefore sees about 28% of the rows, and the detector's AP, threshold sweep
  and review queue rest on 516 positives. The README says coverage "can be below 100%"
  without saying it is one fifth.
- The detector trains on every row with a 0/1 label: `candidate_rules.build_universe`
  returns binary labels (0 for every non-truth pair), and the runtime never applies the
  judgeability rule that the same module's `describe_rule` states and that Stage 3 applies
  through `label_availability` sidecars (0 only where GT can judge, otherwise NaN).
  683,601 rows (99.1%) are pseudo-negatives, nested CV runs on all of them (about 285
  fits of 689k x 150 for 15 grid points, hours for the boosted families), and the AP it
  reports is on a 0.36% prevalence population, incomparable with the 0.8255 card.
- `write_run_commands` substitutes output names for split only; a merge-site build (the
  default for merge runs) gets a runbook that writes `merge_detector_<brain>.csv` while
  the runtime names its joblib `merge_junction_detector_<brain>.joblib` and
  `run_detector_result_analysis.py` looks for `merge_junction_detector_<brain>.*`.
- `OVERTURNED` is a hard exclusion in `validate_inventory`, but the token describes the
  record's narrative, not the feature: on new-split-1 ids 9 and 49 were overturned
  because the record's claim pointed the wrong way, and the clutter penalty (id 9) is the
  one feature the corrections showed to gain on both brains.
- The inventory is one agent turn over 93 rerun and 45 fixed scripts, and the feature
  code one turn for about 85 hypotheses with two repairs; a finished detector cannot be
  recovered on `--keep-existing` once its hidden draft is gone; the no-data checks run
  three times per build; the builder spec still describes inventory v2/v3 and a verdict
  set without CRITICAL.

## Stage 3 contract (what the revision must keep)

Stage 3 never reads the detector's CSV, OOF scores or its own labels. It uses:

- `precompute_error_scores._resolve_detector_dir`: exactly one `split_site_detector.py`
  or `merge_junction_detector.py` and exactly one `split_detector_*.joblib` or
  `merge_junction_detector_*.joblib` **in the same directory**, globbed non-recursively;
  `native_pool.training_provenance` reads `model_selection_<brain>.json` from that
  directory too (`train_brain`, `train_mcl`, `scope`, `feature_order`) and requires the
  joblib name to end in `_<brain>.joblib`.
- The joblib bundle: `pipeline` (scores every row; Stage 3 requires finite scores for
  all rows), `feature_order` (must equal the model-selection JSON's).
- The module: `DETECTOR_TARGET`, `ANALYSIS_TIMING_GROUPS` (owner of each feature and
  its `_is_defined` column), `sample_display`, `CANDIDATE_POLICY_CONFIG_ID`,
  `CANDIDATE_POLICY_SHA256`, `CANDIDATE_RULES_SHA256` (recorded as
  `detector_rule_matches`, not required), and
  `_extract_features_runtime(working, verbose=True, enabled_analysis_keys=...)` called
  with `working = {"fragments_graph", "min_cable_length",
  "__detector_sample_universe_cache__": (blind rows, int64 zeros)}` and no `gt_*` keys.
- `candidate_rules.build_universe(kind, payload, rule)`: rows plus **binary** labels
  (`native_pool` asserts `isin([0, 1])` and stores int8); the three-valued labels come
  afterwards from the `label_availability` sidecars. The companion export (`--pool-only`)
  uses the same function with no detector.
- `pool_sha256` hashes the blind rows only (`GEOMETRY_FIELDS`, `OCCURRENCE_FIELDS`);
  the table provenance also hashes `candidate_rules.py`.

Three parts of the first draft of this plan would have broken that contract and are
rewritten below: NaN labels returned by `build_universe` (item 1), fold groups computed
inside the extraction entry point (item 2), and the output layout (item 3). Two
pre-existing gaps are added to the plan because the new detectors will hit them: the
runbook's `runs/<brain>-*` layout versus Stage 3's one-directory rule, and image
features in a Stage 3 extraction that supplies no image location.

## 1. Three-valued labels in the detector runtime

`candidate_rules.py` (shared by the benchmark tables and the detector):

- `build_split_universe` / `build_merge_universe` keep returning binary labels
  (Stage 3 contract). New `label_availability(kind, payload, rows)` returns the boolean
  judgeability mask with the definitions `proofreader_evolve/harness/label_availability.py`
  applies to the tables: split `segment_on_gt` (both segments in
  `gt_segments(gt_node_canonical_label)`); merge `near_gt_150` (segment on GT and the
  candidate within 150 um of a GT node, `payload["gt_graph"].node_xyz`). The two
  definitions live in one place; `precompute_availability` can switch to calling it later
  (optional, no behaviour change).
- `split_audit` / `merge_audit` add `n_judgeable`, `n_unjudgeable` and the positive share
  among judgeable rows.

Runtime template (`detector_runtime.py.tmpl`), target adapters:

- `build_sample_universe` returns `(samples, labels)` with float labels: 1, 0 where the
  mask holds, NaN otherwise, raising if a positive row is unavailable (the
  `three_valued` invariant). It still returns a supplied cache unchanged, so Stage 3's
  int64 zeros pass through; `_extract_features_runtime` stores labels as float and keeps
  its signature.
- `y` is float with NaN; `judged = np.isfinite(y)`. `run_nested_selection`,
  `fit_final_winner`, `undefined_audit`, `ascertainment_audit`, `_report_thresholds`,
  `make_figures` and `_run_heldout` fit and score on `judged` rows. Prevalence, AP and
  precision@k are over judgeable rows.
- Every row is still scored: the final model's score covers all rows (this is what
  Stage 3 recomputes from the joblib). OOF columns stay NaN on unjudgeable rows and the
  queue states how many scored rows are unjudgeable.
- `sample_output_frame` writes the label column as float (NaN preserved).
  `_input_provenance` digests the three-valued labels with the companions' convention
  (`labels_digest`: NaN as -1) so detector and companion digests agree. The
  model-selection JSON keeps `train_brain`, `train_mcl`, `scope`, `feature_order`
  unchanged and adds `label_rule` (definition name and counts).
- `run_smoke_test` builds its synthetic frame with NaN labels and asserts that unjudgeable
  rows are scored but excluded from metrics.

Documentation: README skeleton and RUN_COMMANDS replace "precision is a floor" with
"metrics are over GT-judgeable rows; unjudgeable rows are scored, never evaluated".

Side effects: `candidate_rules.py` changes its SHA-256, so the frozen policies and the
table provenance change for new builds, and a detector built before the edit would show
`detector_rule_matches=false` in the tables (a note, not an error): build the detectors
after the edit. `pool_sha256` hashes rows only and the companions take their labels from
the tables, so neither changes. The tables are stale already (pending rebuild after the
relabel and pool change).

## 2. Folds: purged spatial blocks, every judgeable row held once

Port the harness rule (`grouped_folds` in `proofreader_evolve/harness/`, the folds the
companions carry as `fold` and `fit_mask_fold<f>`) into `candidate_rules.py` so the
detector stays self-contained:

- `fold_groups(kind, samples, node_xyz)`: split rows group by the 500 um block of the
  midpoint of the first occurrence's two anchors; merge-site rows group by segment.
- `fold_partitions(kind, samples, groups, n_folds)`: greedy balancing of groups, largest
  first with a SHA-256 tie-break (label-free). Held rows of fold f are every row of its
  groups; fit rows are the rows outside f that share no segment with a held row (the
  purge). Returns `(fit, held)` index arrays per fold. Held sets partition the rows, so
  OOF coverage is every judgeable row.
- `run_detector` computes `groups` right after extraction, while the payload is alive,
  and passes `groups` and the per-row segment ids to `_cv_splits`; inner folds re-run the
  same partitioning on the outer fit rows. Nothing changes in
  `_extract_features_runtime` or `build_sample_universe`, so Stage 3's extraction path
  is untouched. The random per-segment assignment and the "boundary pairs omitted"
  semantics go away, for split and for merge-site alike.
- Policy: `outer_folds` 3, the companions' fold count, so the detector's OOF held sets
  match the card's and discovery's. Inner folds stay 3. (Decided 2026-10-21.)
- README/RUN_COMMANDS: `n_scored` equals the judgeable count; the split caveat about
  unscored cross-fold pairs is removed.

## 3. Output names and layout shared with Stage 3

- `write_run_commands` interpolates `spec.output_prefix` (and a job name derived from
  it) instead of hard-coding `merge_detector_` and patching split by string replacement.
  Script names and prefixes stay exactly `split_site_detector.py` /
  `split_detector_*` and `merge_junction_detector.py` / `merge_junction_detector_*`,
  which Stage 3 globs.
- Layout: RUN_COMMANDS writes every run into `$APP_DIR/runs/<brain>-<purpose>/`, while
  `_resolve_detector_dir` and `training_provenance` expect script, joblib and
  model-selection JSON in one directory. Fix on the Stage 3 side so run directories stay
  clean: `_resolve_detector_dir` accepts either a deliverable directory or a run
  directory; for a run directory the script is the matching file in the parent and must
  hash to `data_provenance.detector_sha256` recorded in that run's model-selection JSON
  (the runtime writes it today). The detector record then carries `script_dir` and
  `run_dir` separately; `training_provenance` and `_assert_detector_unchanged` read
  from the right one. `DEFAULT_MERGE_DIR` / `DEFAULT_SPLIT_DIR` and the rebuild plan's
  step 4 point at the chosen run directories.
- `_survey_out_dir` recognises the three output prefixes, the `runs/` subtree,
  `slurm-*.out`, `conda-explicit.txt`, `python-version.txt`, `input-pkl.sha256` and
  `result_analysis_evidence.json` as rerunnable.
- Tests: for each target, every file name RUN_COMMANDS writes equals what
  `run_detector_result_analysis.py` and `_resolve_detector_dir` expect; a fixture with a
  deliverable directory plus one run directory resolves through `ensure_native_tables`'
  provenance checks.

## 4. Feature-level signal next to the record verdict

Discovery (`run_discovery_workflow.py`, test-result-folder spec):

- `verdicts_rule("fix_tests")` adds `"feature_effect": "gain|null|loss|artifact"`: what
  the corrected test says about the feature itself against the raw card, independent of
  the record's claim direction (`artifact` = the measured quantity was a stand-in or a
  self-inclusive count), plus `"effect_delta_ap"` (the isolating model's delta or null).
  `VERDICT_VOCAB`, `_check_phase_verdicts` and the report bullet
  `**Feature effect:**` carry it; `_merge_phase_verdicts` merges it. Reports without the
  field read as null.

Build:

- `verdicts_evidence_by_id` reads `feature_effect`. `validate_inventory` and the
  inventory instruction exclude an `OVERTURNED` hypothesis only when `feature_effect` is
  `artifact`, or is absent (old reports keep today's rule); `gain`, `null` and `loss` may
  be included like WEAK/MINOR findings, with the record's reversal in `source_reason`.
- For the two finished runs, whose verdicts predate the field: `--keep-overturned ID
  [ID ...] --keep-overturned-reason TEXT` on the build driver; the ids and reason go into
  the inventory rows and the README provenance block. No discovery step is rerun.

Stage 3 is unaffected: it reads feature names from the joblib, never the verdicts.

## 5. Batch the inventory stage

- `inventory-features` runs per batch of `INVENTORY_BATCH_SIZE` (12) selected ids in
  selected order; each turn writes `.feature_semantics.part<k>.json` with exactly its
  rows, and the draft self-check CLI takes `--ids` to check one part. A ledger
  (`.inventory-ledger.json`, part hash per batch) lets a resumed run redo only open
  parts. The driver concatenates the parts into `.feature_semantics.json`, then compiles
  and checks the whole draft as today; a duplicate feature name across parts is reported
  to one reconciliation turn naming both ids.
- Feature names are required to start with `h<id>_`, which makes cross-batch clashes
  impossible and ties every column to its hypothesis in the CSV and in the Stage 3
  tables. `detector_score` stays reserved. (Decided 2026-10-21; the inventory compiler
  and `validate_inventory` enforce the prefix for every included feature.)
- `configure-models` and `verify-and-document` stay single turns. `generate-detector`
  stays one turn because shared traversal passes are its purpose; its repair budget rises
  to 3 and a step timeout leaves the draft in place for `--keep-existing` (already the
  case for a rejected draft).

## 6. Recover an assembled detector on `--keep-existing`

- Assembly embeds `RUNTIME_TEMPLATE_SHA256` and `FEATURE_INVENTORY_SHA256` constants in
  the target adapter (module attributes only; Stage 3 ignores them).
- New recovery branch, after the model-config branch: when no feature draft exists but
  the detector file does, accept it if its source parses, the executable contract passes,
  the embedded template, inventory and candidate-policy hashes equal the current ones and
  `candidate_rules.py` beside it matches the frozen hash; then skip step 3 and write the
  documentation skeleton. A failed verify-and-document no longer costs a second
  generate-detector turn.

## 7. One executable check per assembled detector

- `finalize_documentation` keeps `validate_detector_source` and the hash guard (the
  detector is immutable after step 3) and drops the repeated `--help`/smoke/invalid-config
  run; the recovery path keeps it.

## 8. Image location for Stage 3 extraction

- Discovery scripts read the image from the companion manifest (`image.uri`, levels);
  a detector fragment has no companion and reads the cache's `img_path` through the
  blind view. Stage 3's `inference_metadata` whitelists only `min_cable_length`, so an
  image feature would raise or go undefined on every table row.
- `inference_metadata` whitelists `img_path` (GT-free), and the builder contract states
  that image features take the volume from `payload["img_path"]` and are undefined
  (NaN, `is_defined` 0) when the key is absent, never a crash. The build instruction
  already requires the level geometry to come from the fragment's own constants.

## 9. Builder spec refresh

`.claude/agents/discovery-detector-builder.md`: inventory v4 (merge-site) and v5 (split),
the CRITICAL token, AP vocabulary, the three row units, three-valued labels and the
block-purge folds, the batched inventory contract, the `feature_effect` gate and the
image-location rule; drop the segment-level wording that no longer describes the default
merge target.

## Tests

- `agentic/tests/test_detector_build_workflow.py`: RUN_COMMANDS names per target; the
  assembled-detector recovery branch; batched inventory merge and ledger resume;
  `--keep-overturned`; no second executable check; `feature_effect` gate.
- `test_detector_target_runtime.py`, `test_detector_label_provenance.py`: NaN labels
  through `build_sample_universe` and `_extract_features_runtime`; a supplied int64
  zero cache passes through unchanged (the Stage 3 call); audit counts; the digest
  convention.
- New `test_candidate_rules_folds.py`: every judgeable row held exactly once, purge
  removes every fit row sharing a segment with a held row, deterministic and
  label-independent; the split rule equals the companion's `fold` array on a fixture;
  `label_availability` equals `harness.label_availability.compute` on a fixture.
- `proofreader_evolve/tests/test_native_pool.py`: `_resolve_detector_dir` with a run
  directory bound by `detector_sha256`; `inference_metadata` with `img_path`.
- `test_discovery_smoke.py` StructuredVerdictTests: the `feature_effect` vocabulary.
- Template smoke with NaN labels. Suites: `python -m unittest discover -s agentic/tests`
  and `python -m unittest discover -s proofreader_evolve/tests -t .` (the rule module is
  shared).

## Order

1. Items 3, 7, 8, 9: small, no behaviour risk; item 3 touches Stage 3's resolver.
2. Item 4: schema and gate, plus the build flag for the finished runs.
3. Items 1 and 2: the rule module, the template, the docs; both suites; then one split
   build and a `--measuretime` run on 794495 to confirm coverage and cost.
4. Items 5 and 6.

Then build split from `new-split-1_2026-10-20`, and merge from `new-merge-1_2026-10-19`
once its workflow has finished; run both detectors into `runs/<brain>-default/`, point
`DEFAULT_*_DIR` at those run directories, and continue with
`proofreader_evolve/plans/rebuild_after_relabel_and_split_pool.md`.

## Addendum 2026-10-21 (evening): image pass always threaded

The 2026-10-19/20 builds ran their image pass serially (3.3 rows/s on 794495):
the builder rule made the pass thread-aware only when image hypotheses were a
strict majority (20 of 73, 22 of 88). Revised and done:

- any image-derived hypothesis makes the pass thread-aware (`assembly.py`
  `image_pass`; driver generate/verify text; builder agent text; runbook text);
- the runtime template provides `_image_row_order` (64 um block order so
  neighbouring rows reuse zarr chunks) and `_image_store_context(workers)`
  (4 GiB cache pool, S3 request limit max(32, 2 x workers)); the assembler
  requires both whenever an image pass exists, and `RUNTIME_OWNED_SYMBOLS`
  reserves them;
- generated runbooks use `--n-jobs 32`; `precompute_error_scores.py` gains
  `--image-workers` (threaded through `ensure_native_tables` and
  `build_native_table`) so Stage 3 scoring pays the same throughput;
- the two existing builds re-run Step 3 and 4 with `--keep-existing` (the
  inventory and model candidates are recovered; the hidden draft is gone, so
  the feature code is re-authored under the new contract).

### Template defects found by the 2026-10-21 verify steps (fixed the same night)

Both re-built detectors' verify-and-document steps reported one build-blocking
defect in the runtime template and three gaps: `_print_queue` crashed on a
GT-unjudgeable (NaN) label in the every-row queue (a real run would die after
the final fit, before any output); `_BlindPayloadView` did not block
`gt_label_policy`; `--test-pkl` equal to the training cache was accepted; the
inventory hash check was skipped silently when the file was missing and the
module docstring overstated what is checked. All four are fixed in the template
(`_print_queue` prints `-`; every `gt_*` key is blocked; same-cache held-out is
rejected; training requires `--inventory`; docstring corrected) with template
tests. Both detectors were re-assembled from their embedded feature fragments
(extracted byte-for-byte, proven by re-assembly with the build-time template),
passed the driver's source and executable checks, and were installed with
regenerated `RUN_COMMANDS.md` and an annotated README; the previous files are
kept under `../tmp_reassemble_20261021/`. Lesson: the driver's post-assembly
checks never reach `run_detector`; a synthetic end-to-end run with NaN labels
belongs in the template tests.

### 2026-10-22: result-analysis driver deleted

`agentic/run_detector_result_analysis.py`, its test and `detector-result-analyst.md` were
deleted unused (no `RESULT_ANALYSIS.md` was ever generated). The per-run report is now
`agentic/run_detector_importance_report.py` (`importance_evidence.json`, short English
`IMPORTANCE_REPORT.md`); the result-directory helpers moved to
`agentic/detector_importance.py`, and the runbook, README skeleton and the rerunnable
output names follow. Mentions of the old driver above are historical.

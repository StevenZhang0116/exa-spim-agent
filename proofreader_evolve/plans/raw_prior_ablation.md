# Plan: ablation from raw candidates (`--detector-prior raw`)

Status (2026-10-27): implemented the same day (`harness/detector_prior.py`, driver, `context_cache.contexts`,
`tests/test_raw_prior.py`; design record section "Raw-candidate ablation" in `WORKFLOW_REVISION.md`).
Step 3 (preflight mode column) was dropped: the preflight does not take the mode. Decisions taken with the
user the same day:
(1) raw columns as listed below, no coordinates, occurrence aggregates included; (2) constant
seed; (3) descriptor generation wall budget 7200 s for raw runs; (4) runs are sequential, one
setup per run; (5) the candidate-pool definition is part of the benchmark and stays.
Builds on `plans/no_detector_score_ablation.md` (`--detector-prior no-score`, implemented
2026-10-13, never run end to end) and on objective v5 with the eleven-brain partition
(`WORKFLOW_REVISION.md`, "Uniform K = GT positives (2026-10-27, objective v5)").

## Question

How much of evolution's final validation Precision@K comes from the whole AutoDiscovery plus
detector pipeline? `no-score` withholds only the fitted ranking (`detector_score`) and keeps
the 293 `h*` feature columns that the AutoDiscovery runs discovered and the detector build
computed (the 802449 merge table holds 294 columns: 293 `h*` plus `detector_score`; the
label-blind candidate attributes are not in the feature frame at all). `raw` withholds the
`h*` columns too. The agent then starts from the candidate pool, the label-blind candidate
attributes, the context cache (fragment neighbourhood of 50 um / 256 nodes and image patches
at level 1 and level 0 for every cached row) and its own descriptors.

The three arms decompose the final number: raw to no-score is the contribution of the
discovered features, no-score to frozen the contribution of the fitted detector, frozen to
the evolved scorer the contribution of evolution. Starting points differ a lot (frozen split
seed about 0.85, raw split seed near the labeled base rate), so the comparison is between
final values at identical pools, K, brains, budgets and gate, not between per-generation
gains.

## What stays fixed (benchmark definition, not detector knowledge)

- Candidate pools: `candidate_rules` schema 2 (merge `junction+tip_to_shaft|nms=20|gap=20|k=2|r=150`,
  split `tip_to_any_node|r=80|k=2`). Decision 5: the pool is the task definition.
- Labels and availability sidecars, the v5 K rule (K = GT positives, no caps), the
  positive-weighted aggregation, the bootstrap gate, the eleven-brain partition (TRAIN
  794495 auxiliary, 802449 grouped folds, 747807 / 754610 / 754612 / 754613 whole-brain
  holdouts; validation 789202, 794493, 794491, 750318, 751473), generations and evaluation
  budgets.
- The context cache and its coverage rule from `no-score`: the band is the detector's top
  50,000 rows plus the labeled extension; every GT-labeled row is cached on all eleven brains
  (eleven-brain preflight, 2026-10-27), so "descriptor is finite" never marks the detector band
  among ranked rows. The band stays hidden (`expose_band=False`, no `bank_context_available`,
  descriptor scope `all_cached` only).

## What the agent gets per arm

| | frozen | no-score | raw |
|---|---|---|---|
| `detector_score` | yes, and the seed | no | no |
| 293 `h*` columns | yes | yes | no |
| raw candidate columns (below) | implicit in `h*` | implicit in `h*` | the only base columns |
| seed scorer | `artifacts/scorer.py` (the score) | `artifacts/scorer_no_score.py` (constant) | the same constant seed |
| `bank_context_available` / `row_in_band` | yes | no | no |
| descriptor scopes | pilot, boundary, all_cached | all_cached | all_cached |
| descriptor bank | `descriptor_bank/` | `descriptor_bank_no_score/` | `descriptor_bank_raw/` |
| context cache, 3D analysis, image previews | same | same | same |

## Raw columns (decision 1)

Built in memory from `table.candidates`, the label-blind rows of `candidate_rules.GEOMETRY_FIELDS`
and `OCCURRENCE_FIELDS` (`native_pool.SAFE_KEYS` / `OCCURRENCE_KEYS`); the table files and their
digests are not touched. Coordinates (`x_um`, `y_um`, `z_um`) and every identifier are excluded:
position is not a property of the error, and the merge folds are grouped by fragment, so
location could be memorised within 802449.

| kind | column | source | notes |
|---|---|---|---|
| merge | `raw_degree` | `degree` | junction arity |
| merge | `raw_candidate_family` | `candidate_family` | junction = 0, tip_to_shaft = 1 |
| merge | `raw_tip_gap_um` | `tip_gap_um` | NaN on junction rows (no tip) |
| split | `raw_gap_um` | `gap_um` | anchor pair gap |
| split | `raw_partner_rank` | `partner_rank` | rank of the partner among the anchor's neighbours |
| split | `raw_anchor_side` | `anchor_side` | a = 0, b = 1 |
| split | `raw_node_role_a`, `raw_node_role_b` | `node_role_a/b` | tip = 1, shaft = 0 |
| split | `raw_occurrence_count` | `len(occurrences)` | how many times the pair was enumerated |
| split | `raw_occurrence_gap_min_um`, `raw_occurrence_gap_max_um` | `occurrences[*].gap_um` | aggregates over the occurrences |

The `raw_` prefix keeps the columns apart from `h*` and `bank_agent_*` in feedback, feature
statistics and failure cases. Every column is numeric; NaN is allowed where the attribute does
not exist (`raw_tip_gap_um` on junction rows).

## Where the discovered features reach the agent today, and the treatment in `raw`

| Path | Location | Treatment in `raw` |
|---|---|---|
| Feature frame | `native_pool.build_native_table` keeps the detector's `feature_order` (`h*`) plus `detector_score` | After the frozen baseline is measured: drop `detector_score` (as `no-score`), drop every `h*` column, insert the raw columns (`detector_prior.strip_to_raw`), on TRAIN and validation tables alike |
| Host copies | `strip` keeps `table.host_detector_score` | Also `table.host_detector_features` (the dropped frame) for host-only diagnostics; never passed to scorers, fits, descriptors or feedback |
| Descriptor contexts | `context_cache.contexts()` passes `base_columns` intersected with `table.features` | Covered by the drop: contexts carry the raw columns only |
| Local / image row selection | default `selection_feature='detector_score'` | As `no-score`: no default; the agent names a raw column; the notice says which exist |
| Feature statistics, failure cases, feedback examples | computed over `table.features` | Raw columns only, nothing to change |
| Classifier fits | `fit_classifier` passes `table.features` plus registered descriptors | Raw columns plus `bank_agent_*`; a fit with the raw columns alone is valid |
| Contract and guides | `scorer_rules.md`, `classifier_guide.md`, local/image guides | A `raw` notice appended to the system prompt: the columns that exist, that every other feature must be built with `compute_descriptors`, that `h*` names do not exist in this run |
| Descriptor bank | descriptor programs of other arms may read `h*` or `detector_score` from their contexts | Own bank `descriptor_bank_raw/`; `check_source` refuses sources that reference `detector_score`, `bank_context_available`, `row_in_band` or any `h<digits>_` name |
| Starting scorer | `--start-from` default | `artifacts/scorer_no_score.py` (constant; decision 2), checked with the extended `check_source` |
| Resumed descriptors (`--start-from` with `descriptors.json`) | driver restores registered columns | Same scope and source checks as `no-score`, plus the `h*` name check |
| Preflight | reports table columns per brain | Add the mode to the preflight row (frozen / no-score / raw) so a run can be matched to its arm |

## Seed and baselines (decision 2)

- Seed: the constant scorer. Ties resolve by candidate identity, a label-independent order, so
  the seed is an arbitrary ranking of the labeled rows and the seed Precision@K is near the
  labeled base rate (802449 merge 1,038 / 27,934 = 0.037; split 6,535 / 15,786 = 0.41).
- Host-only comparators recorded in `baseline.json` before the strip: the frozen detector
  (as `no-score`) and two raw heuristics measured with `scoring.evaluate` on the raw columns:
  split `-raw_gap_um` (the classic gap baseline) and merge `raw_degree` with
  `raw_tip_gap_um` as a tie-breaker. They are references for the paper, never the seed and
  never shown to the agent.

## Budgets (decision 3)

`--descriptor-generation-wall-seconds 7200` (default 5400) for raw runs, passed on the command
line; the per-call wall (1800 s), the evaluation units per generation and the API turn limits
stay. If the first raw run shows descriptor computation as the binding limit, raise the
per-call wall for raw runs in a follow-up, not in this change.

## Steps

1. `harness/detector_prior.py`: `MODES = ('frozen', 'no-score', 'raw')`; `RAW_COLUMNS` per kind
   and `raw_columns(kind, rows) -> pd.DataFrame`; `strip_to_raw(banks)` (drop `detector_score`
   and `h*`, keep host copies, insert raw columns, return the stripped cells); `HIDDEN_PATTERN`
   adding `h<digits>_` names to `check_source` under `raw`; `describe` for the new mode;
   `NOTICE_RAW` text; `enabled(args)` true for both ablation modes plus `raw_enabled(args)`.
2. `cli/run_precision_evolution.py`: argument choices; defaults for `raw` (`scorer_no_score.py`,
   `descriptor_bank_raw/`, refuse the frozen or the no-score bank); the frozen baseline and the
   raw-heuristic baselines before the strip; `strip_to_raw` instead of `strip`; the raw notice in
   the contract; descriptor restore checks; the `detector_prior` manifest entry records the raw
   column names.
3. `cli/preflight.py`: mode column in the per-brain row (small).
4. `harness/local_feature_contract.py` / `image_selection.py`: error messages name the raw
   columns under `raw` (the no-default rule is already there).
5. Tests (`tests/test_detector_prior.py`, extend): raw columns from the fixture rows (values,
   NaN on junction rows, occurrence aggregates); `strip_to_raw` leaves raw columns only and keeps
   host copies once per shared table; `check_source` refuses `h3_caliber_ratio_15um`; a tiny
   driver run under `raw` records the frozen and heuristic baselines, withholds the columns on
   TRAIN and validation, scores a candidate that uses a raw column, and refuses the frozen and
   no-score banks; the context-coverage check still applies.
6. Docs: `WORKFLOW_REVISION.md` section "Raw-candidate ablation (`--detector-prior raw`)" with
   the arm table above and the provenance / evidence rows; `CLAUDE.md` and
   `docs/stage3_evolution.md` flag lines; `artifacts/descriptor_guide.md` one sentence on the
   raw arm; memory note.
7. Seed baseline on the eleven-brain partition (`--generations 0`) for the three arms, in that
   order (frozen, no-score, raw), on one node; record the three seed validation means in the
   evidence log before any evolution run.

Estimated effort: code half a day, tests and docs half a day, seed baselines under an hour.

## Running the comparison (decision 4: sequential, one setup per run)

```bash
# common flags (objective v5, eleven-brain partition)
COMMON="--train-brains 794495,802449,747807,754610,754612,754613 \
  --selection-brains 802449,747807,754610,754612,754613 \
  --selection-holdout-brains 747807,754610,754612,754613 \
  --validation-brains 789202,794493,794491,750318,751473 \
  --context-cache proofreader_evolve/context_cache --classifier-memory-mb 98304 \
  --classifier-time-budget 3600 --policy-time-budget 2400"
# --classifier-threads 8 and --descriptor-workers 24 are the defaults since 2026-10-27.
# 1. frozen (main result)
python -u -m proofreader_evolve.cli.run_evolution $COMMON --generations 20
# 2. no-score (fitted ranking withheld)
python -u -m proofreader_evolve.cli.run_evolution $COMMON --detector-prior no-score --generations 20
# 3. raw (discovered features withheld too)
python -u -m proofreader_evolve.cli.run_evolution $COMMON --detector-prior raw \
  --descriptor-generation-wall-seconds 7200 --generations 20
```

Each run is preceded by its own `--generations 0` seed baseline. Runs share nothing but the
tables and the context cache: each arm has its own descriptor bank. Stopping a run launched
with `nohup ... &` needs `kill -TERM -- -<driver pid>` (SIGINT is ignored in that setting).

## Not in scope

- Changing the candidate pool definition (decision 5).
- Rebuilding tables or context caches: everything happens in memory after table load.
- A fourth arm without the context cache (descriptors disabled): possible with
  `--detector-prior raw` and no `--context-cache`, but then the agent has only the raw columns
  and the 3D analysis tool; not planned.
- Making 794495 a selection cell through the detector's out-of-fold probabilities, and an
  untouched final test brain (both listed as follow-ups in the v5 section).

## Risks

- Early generations will rediscover gap and caliber features that AutoDiscovery already found;
  the run costs more API turns and descriptor compute than frozen. Compare final values.
- Neither `no-score` nor `raw` has completed a real run; the first generation of each must be
  watched (descriptor coverage, fits with few columns, notice wording).
- The context cache neighbourhood (50 um, 256 nodes) bounds every descriptor; long-range
  quantities (whole-fragment cable length) are out of reach for all three arms alike.
- The band membership of unlabeled rows is detector-chosen in every arm; it is hidden and
  cannot influence the ranking of labeled rows, but descriptors computed on unlabeled band rows
  are not a random sample of the pool.

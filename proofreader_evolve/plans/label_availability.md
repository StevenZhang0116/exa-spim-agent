# Plan: three-valued labels via label-availability sidecars

Status (2026-10-09): steps 1–2 implemented, then superseded the same day by full three-valued labels
(WORKFLOW_REVISION "Three-valued labels throughout"); the report-only flags of step 2 were removed.

## Goal

The detector-native labels are binary, but GT covers only the traced neurons, so most
label-0 rows mean "no GT here". Mark, per candidate, whether GT can judge its label.
Binary label + availability = three-valued label (1 error / 0 no error / unavailable).
Do not rebuild feature tables or context caches; do not re-evaluate past runs; do not
change promotion in the first step.

## Step 1 — sidecars (done)

- Definitions (`gt-available-v1`): split = both segments carry a GT node label; merge
  `segment_on_gt` = the candidate segment does; merge `near_gt_150` = and the junction is
  within 150 µm of a GT node (default for merge).
- `harness/label_availability.py`: compute / write / load / attach.
- `cli/precompute_availability.py`: one subprocess per brain, reads `_add.pkl` and the
  validated tables, writes `feature_tables/<brain>/<table_digest>.availability/`.
- Consistency, checked on write and on load: row count; candidate keys hash and
  `pool_sha256`; `labels_sha256`; source-cache identity equal to the table provenance;
  file checksum; every positive available (for every definition). Mismatch raises;
  missing sidecar = absent.

## Step 2 — report-only metrics (done)

- `run_precision_evolution --label-availability {report,off}` (default report),
  `--availability-merge-definition {near_gt_150,segment_on_gt}`.
- Sidecars attached to validation tables only; `fixed_pool_scoring.evaluate(...,
  report_availability=True)` for seed, baseline and candidate validation adds
  `precision_available`, `available_in_top_k`, `n_available`, `available_fraction`,
  and `macro_precision_available` (only when every cell has them).
- Manifest records mode/version/definitions/sidecars; `report.html` shows the columns.
- Unchanged: native P@K, acceptance, bootstrap, TRAIN/selection evaluation, reviser
  inputs. The mask never reaches scorers, models, features or descriptors.

## Step 3 — deferred

Training use (`label_available` to `fit`, TRAIN only, `y` stays binary), aggregate
reviser feedback and guide text, opt-in available-row promotion gate designed together
with the merge-gate proposals, and definition refinements. Tracked in `TODO.md`.

## Commands

```bash
conda activate panda   # compute node
python -u -m proofreader_evolve.cli.precompute_availability \
    --brains 794495 802449 789202 794493 794491 747807 750318 751473 754610 754612 754613 --mcl 100
```

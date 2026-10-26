# Plan: stop re-measuring what was already measured (repeated submissions)

Status (2026-10-26): implemented the same day (tests `tests/test_repeated_submission.py`; design record
section "Repeated submissions and earlier-generation access" in `WORKFLOW_REVISION.md`). One naming
change against decision 4 below: the agent-visible field is `outer_measurement`, not `validation`,
because agent-visible plan and pool files must not contain the word "validation"
(`test_kind_schedule` guard); its content is as planned. Written after run
`precision_20261025_193120_5to8nx13` (20 generations, objective v3) and the objective v4
change (split K = GT positives).

## Problem

Nine of the twenty generations of `5to8nx13` re-measured a candidate that had already been
validated. The ledger shows how each repeat came about:

| Generation | Kind | Submitted | Identical to | How |
|---|---|---|---|---|
| 8 | split | gen008/attempt005 | gen004/attempt003 (validated gen 4) | cached repeat, same component bytes |
| 10 | merge | gen010/attempt004 | gen009/attempt001 (validated gen 9) | cached repeat, same component bytes |
| 13, 14, 18 | split | gen0xx/attempt00x | the accepted split component (gen012/attempt002) | agent restored the reference after failed ideas; caught as `unchanged` |
| 15 | merge | gen015/attempt002 | gen011/attempt005 (validated gen 11) | different component bytes, identical merge score vectors on every TRAIN and selection cell |
| 16, 19 | merge | gen016/attempt008, gen019/attempt008 | gen011/attempt005 via gen015/attempt001 | cached repeats of the champion (control arm of a tuning grid), same bytes |

Every repeat gave the validation gate exactly the hits it had already seen (17 / 4 / 44 for
the champion, four times). Two things let this happen:

1. The host compares a submission only with the accepted component (`unchanged_submission`
   stage in `cli/run_precision_evolution.py`), never with earlier validated submissions, and
   compares bytes only, so a cosmetic rewrite (gen 15) passes as new.
2. Nothing tells the agent that its search parent was already validated and not promoted.
   `search_plan.search_parent` carries `status: evaluated` and the selection score; the
   training tool reports a cached repeat (`cached_from`) but not that the repeated experiment
   was submitted; `hypothesis_memory.handoff.host.promoted` exists only for generations that
   left a handoff. In `tune` rounds the agent includes the champion as a control arm, the
   control ties for best on the selection score, and the agent submits it.

A third, smaller cause: the agent cannot read its own earlier generations' code.
`tool_audit.jsonl` holds six denied reads of `gen0xx/experiments/attempt00x/training.py`,
`gen006/descriptor.py` and `gen003/artifacts/descriptors` from later generations. It then
rebuilds the same program from the hypothesis memory's prose.

## Changes

### A. Repeat detection in the host (no validation of a known ranking)

- Identity of a submission for its kind: `component_sha256` (bytes) or `scores_sha256`, a
  SHA-256 over the score vectors of that kind's cells in the submitted experiment's
  `selection_state.npz` and `train_state.npz` (exact float equality, no top-K proxy). Both
  digests are written to the ledger row (`scores_sha256` next to `candidate_policy_sha256`).
- At the `unchanged_submission` stage the host compares both digests with the accepted
  component and with every earlier ledger row of the same kind that reached validation (also
  rows of a resumed run). A match ends the generation as `repeated_submission`: no descriptor
  inference on validation brains, no validation evaluation, `record["submission_repeat"] =
  {"generation": n, "experiment": ..., "by": "component" | "scores", "promoted": bool}`,
  `validation_gate.skipped = "repeated_submission"`, reason `VALIDATION: Repeated submission:
  ... identical to gen011/attempt005 (validated in generation 11, not promoted)`. Report and
  trajectory status `repeated` (vocabulary next to `unchanged`); `generation_status` and
  `evolution_report` learn the new status; `run_records` readers accept it.
- `TrainingExperiments.restore('best')`, which the submission fallback uses, prefers the best
  measured candidate whose digests are novel; it falls back to the old choice when every
  measured candidate is a repeat. The agent's explicit choice is never replaced by the host.
- `CandidatePool.finish` treats a repeated submission like an unchanged one for stall
  accounting (the generation measured nothing new on validation); `hypothesis_memory` records
  the host status `repeated_submission` in the handoff host facts.

### B. Tell the agent what was already validated (status only, no numbers)

- Pool entries that were submitted carry `validation: {"generation": n, "promoted": bool}`
  (set by the driver after each generation; `REFERENCE_FIELDS` extended). `search_plan.json`
  shows it on `search_parent` and on `complementary_candidates`; `candidate_pool.json` shows
  it on every member. No validation metric, delta or gate reason is exposed, consistent with
  the existing `handoff.host.promoted` flag.
- Training tool results: a cached repeat whose source experiment was validated says
  `"already_validated": {"generation": n, "promoted": false}` and a one-line note that
  submitting it again ends the generation without validation. The `submit`-time check gives
  the same note when the final files match a validated experiment.
- The generation prompt and `artifacts/scorer_rules.md` state the rule: a submission whose
  bytes or kind scores equal an already validated candidate (including the accepted
  component) ends the generation as `repeated`; in a tuning round the control arm is for
  comparison, not for submission.

### C. Read access to earlier generations' own TRAIN artefacts

- `readable_dirs` gains, for every earlier generation directory of this run, the same three
  TRAIN artefact trees the current generation already exposes: `experiments/`,
  `classifier_fits/`, `descriptor_runs/` (attempt snapshots hold `scorer.py`, `training.py`,
  diffs, `proposal.json`, `selection_evaluation.json`, `train_evaluation.json`,
  `*_state.npz`, `result.json`: TRAIN and selection material only).
- `readable_paths` gains, per earlier generation, the agent-authored and TRAIN-only files:
  `scorer.py`, `training.py`, `descriptor.py`, `analysis.py`, `rules.md`, `proposal.json`,
  `search_plan.json`, `hypothesis_memory.json`, `train_feedback.json`,
  `feature_statistics.json`, `failure_cases.json`.
- Never readable: `evaluation.json` (validation cells), `reviser_result.json`,
  `reviser_prompt.txt`, `trajectory*`, `image_evidence.json`, `research_status.json` of other
  generations, and anything outside `runs/<run>/gen*/`. The guard keeps checking resolved
  paths, so symlinks cannot widen this.
- The prompt names the new scope ("earlier generations' experiments/, classifier_fits/,
  descriptor_runs/ and their scorer.py, training.py, descriptor.py, proposal.json are
  readable; use them before rewriting a program from memory").

### D. Tests

- `tests/test_repeated_submission.py` (driver level, tiny fixtures, no API): a submission
  with new bytes but identical kind scores to generation 1's validated candidate ends
  generation 2 as `repeated` with no outer evaluation and the right ledger / gate / status
  fields; a cached repeat of a validated experiment too; a candidate with different scores
  is validated; the fallback picks the best novel candidate; the accepted component still
  ends as `unchanged` (existing test unchanged); the report vocabulary test learns `repeated`.
- `tests/test_reviser_limits.py` (or a new access test): reads of
  `gen001/experiments/attempt001/training.py` and `gen001/descriptor.py` from generation 2
  are allowed, `gen001/evaluation.json` and `gen001/reviser_result.json` are denied and
  audited.
- `tests/test_train_search.py` / pool tests: `validation` status appears on submitted pool
  entries and in `search_plan.json`; no numeric validation field appears in any agent-visible
  file (extend the existing "no validation text in feedback" assertions).
- Training tool: a cached repeat of a validated experiment carries `already_validated`.

### E. Documentation

- `WORKFLOW_REVISION.md`: a dated section (motivation table above, mechanism, what the agent
  sees), an evidence-log row, the regression-test paragraph; no literature row (project-specific).
- `artifacts/scorer_rules.md`, `cli/run_precision_evolution.py` prompt text, `CLAUDE.md`
  (the "unchanged" sentence), `docs/stage3_evolution.md` (one sentence), `TODO.md` (close the
  item if one exists).

## Decisions to confirm

1. Identity = exact kind score vectors (plus bytes). Top-K membership equality
   (`top_k_sha256`) is reported in the ledger for information but does not trigger a skip.
2. A repeat ends the generation without validation; the host does not substitute another
   candidate for the agent's explicit choice (only the existing fallback path prefers a novel
   candidate).
3. Earlier generations' TRAIN artefacts become readable as listed in C; validation-bearing
   files stay closed.
4. `validation: {generation, promoted}` is exposed on pool entries and in the search plan
   (booleans and generation numbers only).

## Out of scope here

Multi-seed selection (the agent's measured +/- 0.025 OOF noise), the validation-brain set
(750318 / 751473 after the rebuild), scheduler changes beyond stall accounting, and any
change to models, features or the objective.

## Expected effect

On `5to8nx13` this would have ended generations 8, 10, 15, 16 and 19 as `repeated` (and
13, 14, 18 remain `unchanged`), and with B the agent would have known in generations 15, 16
and 19 that the champion was already validated. It does not create better candidates; it
stops spending generations on known results and gives the agent its own code back.

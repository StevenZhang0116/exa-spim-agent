# Failure-driven feature discovery and hypothesis evidence

Use this workflow to test whether a feature adds useful information beyond the
current representation. Choose your own formula, estimator, preprocessing,
training objective and feature algorithm. The harness controls data boundaries,
paired comparisons, accounting and evidence records; it does not select a model
family for you.

## Diagnose a specific failure

Read `failure_cases.json`: up to four pairs match a missed TRAIN positive to a
selected label-0 row using detector score and gap (split) or degree (merge).
Distances are scaled by whole-TRAIN interquartile ranges; poor matches are still
reported with their distances. Examples include both near-boundary and farther
missed positives. These diagnostic cases are not a representative evaluation set,
and native label 0 does not establish biological absence of an error.

Call `inspect_failure_cases({})` to inspect the panel in one tool call. It shares
the existing eight-neighborhood budget with `inspect_candidate`. The same local
geometry bounds and TRAIN-only access rules apply. The cases describe the search
branch you start from (`failure_cases.branch.experiment`), measured under the
official selection protocol; when that branch is not the accepted scorer,
`branch_vs_accepted` lists the positives it newly finds (keep them), the positives
it loses, and the positives both miss, which is the open gap worth a hypothesis.
`train_feedback.json` describes the same branch (`search_branch`) and its
`ranking_delta` is the branch minus the accepted scorer; `parent_selection` and
`parent_train` remain the accepted scorer's numbers.

Write a falsifiable prediction before adding features. For example, test whether
endpoint tangent consistency across path lengths distinguishes otherwise similar
split candidates. This is an example hypothesis, not an established result.
Expose new geometry measurements using `LOCAL_CONTEXT` and
`extract_local_features`; see `local_context_guide.md`.
For new image information, write `analysis.py` and `analysis_request.json` and
call `run_volume_analysis({})`. Analyze up to 16 TRAIN 3D patches per call (compare
groups, e.g. missed positives vs selected label-0 rows),
using aligned fragment nodes and original pixels; see `volume_analysis_guide.md`.
Its eight-execution allowance is independent of scoring and optional 2D previews.
The initial request selects a matched failure pair when available. Compare
3D measurements and revise the investigation before building a scorer.
Then use `LOCAL_IMAGE` for stateless image features or raw patches to a learned
model, without a prescribed architecture; see `image_context_guide.md`.
Analysis selections do not change scoring coverage: check that the policy's
image selection actually reaches the candidates implicated by the hypothesis.
Exploratory findings are not measured ranking gains or automatic feature admission.

## Declare the investigation

Extend the normal `proposal.json` with:

```json
{
  "hypothesis": "Multiscale endpoint continuity separates selected label-0 cases from missed positives.",
  "strategy": "Extract tangent consistency and measure its incremental contribution.",
  "family": "multiscale-continuity",
  "parameter_grid": {},
  "classifier": {"parameters": {"seed": 0}},
  "research": {
    "hypothesis_id": "multiscale-continuity",
    "failure_mode": "Similar detector scores and gaps produce different native labels.",
    "information_source": "local_geometry",
    "prediction": "Adding tangent consistency recovers missed positives without replacing more existing positives.",
    "feature_columns": ["local_tangent_consistency", "local_context_available"]
  }
}
```

This example is not executable until you implement the named feature and your
training program. `information_source` is a free short identifier, not a model
whitelist. Reuse `hypothesis_id` for numeric variants of the same mechanism; use
a new ID when the mechanism changes. The declaration is stored as an agent claim,
separately from host-measured evidence. Legacy proposals without `research` still
work but receive text-derived IDs with weaker cross-variant grouping.

## Measure incremental value

Call `evaluate_feature_ablation({})` with a single fixed parameter configuration
and an empty grid. `research.feature_columns` names input columns visible to
fit/predict or score_candidates. If omitted/empty, all declared local and image
columns, including their availability columns, are selected. Internal features created only inside
your learner are not automatically identifiable by this interface.

The framework compares full inputs with those same columns replaced by a constant
zero, including formerly missing entries. Column names and dimensions remain
unchanged so arbitrary training programs can run. This tests the information in
the declared inputs; it is not literal column deletion. Include the availability
column when testing the entire local-context representation. A partial feature
group can leave other informative local columns in the control arm.
For raw-patch models, include `image_available` to make every patch list empty in
the control arm, and include any `image_*` predictors to remove those too. The
model must handle absent images or the diagnostic is unavailable. The host still
fits the same program in both arms; it does not switch to a different learner.

For a classifier, both arms refit the exact same `training.py` and parameters in
three fixed TRAIN-only folds. Merge rows group by fragment. Split rows group by
500-um spatial blocks at the native representative anchor midpoint. In each fold,
fitting rows that share any fragment with a held row are removed. Assignment is
label-independent and reused across comparisons. The host sends only fitting
features/labels and declared fitting-row patches to each isolated fit worker;
prediction receives held predictors, declared held-row patches and frozen
artifacts. Python/NumPy seeds match between arms. Seed other libraries
or explicit generators in your program if used.

If grouping/purging leaves too few groups or an unusable fitting class, the
diagnostic is `unavailable`. There is no random-row fallback and no zero score
substituted for failure. Every original candidate receives one OOF prediction;
the host computes Top-K on the complete per-brain pool with the original K.
Different fold models can have different calibration. These adaptively reused
TRAIN diagnostics do not establish independent cross-brain performance.
Frozen detector-derived base predictors are reused as supplied; that detector
may already have been fitted on this TRAIN brain. Only the new model and its
learned preprocessing are refitted within these diagnostic folds.

For a hand-written formula, the same two input arms are scored on full TRAIN.
The result is explicitly `TRAIN_formula_descriptive`, not cross-validation of
the agent's formula-design process. Formula failures from the constant control
produce no feature conclusion. The framework does not silently replace a failed
control or switch the model.

A classifier comparison costs **two evaluation units**: one per arm, each arm
being three fold refits plus prediction on the selection brains. A formula
comparison also costs **two units**. Failures cost the units actually started.
Exact successful repeats use the diagnostic cache without new evaluation units.
The folds are the selection protocol's partition; auxiliary TRAIN brains are
fitting-only and are not scored. Failed diagnostics do not grant additional
repair units.

The tool does not change the current scorer or register fold models as candidates.
After interpreting the comparison, call `train_classifier({})` or
`evaluate_train({})` normally and submit an exactly measured snapshot. The outer
equal-weight validation promotion gate is unchanged, and no other brain is made
visible to this tool. Feature diagnostics are advisory, not a new promotion veto.

## Reuse evidence without repeating an entire conversation

`hypothesis_memory.json` starts with `handoff`: up to three records left by earlier
sessions, each with the agent's `open_questions`, `evidence` verdicts
(`supports`, `contradicts`, `mixed`, with conditions) and `next_experiment`, placed
next to host-measured facts (selection score, whether that submission was
promoted). Records related to your assigned branch come first. Treat the agent
fields as claims to check, the host fields as facts. Before you finish, add
`proposal.json.handoff` with the same three fields (each at most three items,
300 characters per text) so the next session continues your line of work instead
of restarting from keyword search.

`hypothesis_memory.json` is a bounded host-written view refreshed after measurements.
It records declarations, fresh measurements, repeated rankings, execution failures,
feature ablations and stagnation. Read the relevant evidence before repeating a
direction; use `search_memory` to retrieve more related experiments and code diffs.
Feature evidence binds to exact program/parameter/feature/split identities.

Two measured generations without improving a hypothesis's TRAIN best, or two
recent successful grouped ablations without an increment, lower that hypothesis's
scheduling priority. The scheduler can switch from tuning to exploration and
prefer a less-stagnant alternative. Protected-reference rounds remain reserved.
New rankings do not prove new information; similar TRAIN rankings do not prove
identical behavior on other brains. Infrastructure failures and cached repeats do
not count as new negative scientific evidence. Failure of one configuration does
not refute a mechanism generally: retain its conditions and uncertainty.

### Required investigations and priority follow-ups

`search_plan.json` now distinguishes soft hypothesis advice from a required
`investigation`. The host maintains `research_evidence.jsonl` from actual tools,
independently of proposal names and declared information sources. Only TRAIN
events contribute. `observed_sources` in the plan reflects actual access,
measurement and comparison; an information source declaration proves none of these.

After two same-kind rounds without new measurements, or a performance plateau,
complete a new paired feature ablation or numerical 3D comparison. If registered
images are available and have never been compared, `required_source` is `raw_image`.
For 3D comparisons, return numerical results on at least two distinct candidate
handles with nonzero valid-pixel coverage. Choose the cases and algorithm; a
matched missed-positive/selected-label0 pair is a useful starting point. Empty
results, duplicated handles, previews and promises of a later analysis do not
fulfill the comparison. Alternatively, ablate all declared image columns and
`image_available` to compare image inputs. A joint image/geometry ablation counts
as an executed comparison, but does not isolate image contribution.

A fresh negative result completes the investigation; it need not improve ranking.
Repeating the same inputs/method is reuse. Comments, docstrings and formatting
are removed from method fingerprints; new parameters or candidate inputs can
constitute a new measurement. This is execution verification, not proof of new
scientific information. Read the live `research_status.json` or tool completion
receipt before submitting. Preserve code, request and measured results; a source
edit that is never executed cannot supply evidence.

Newly supplied references get the next same-kind round for one fresh follow-up,
ahead of normal TRAIN champion selection. A follow-up tracks the observed edit/
restore checkpoint, not inferred conceptual ancestry. Restoring an unrelated
method and measuring it does not fulfill the assignment. If an investigation is
also required, run the comparison on the assigned branch. At most one extra
same-kind follow-up round is reserved; seeds and identical references are exempt.

Incomplete required work skips outer evaluation and retains the accepted policy.
Default stopping distinguishes two incomplete investigations from two consecutive
execution-blocked rounds and from measured performance stagnation. IO/worker
failures are not negative scientific results. `--no-early-stop` disables per-kind
pauses, but the generation limit, tool budgets and two-slot follow-up bound remain.

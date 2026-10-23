# Stage 2. Build: from a validated report to frozen detectors

*Part of the [exa-spim-agent](../README.md) documentation.*

A detector is one script that enumerates GT-blind candidates over a fragment
graph, computes the inventoried features in shared passes, selects a model under
nested CV on the GT-judgeable candidates and scores every candidate. The build
workflow writes that script from a finished report; it never loads a dataset.
Fitting happens afterwards on a compute node.

## 2.1 Candidate rules, sweeps and the frozen candidate policy

Before any feature is scored, a cheap enumeration rule fixes which candidates
exist at all. The rule of each kind is declared once in
[`proofreader_evolve/candidate_pools.json`](../proofreader_evolve/candidate_pools.json)
and applied by [`candidate_rules.py`](../candidate_rules.py), the only implementation
(2026-10-16): split `tip_to_any_node|r=80|k=2`; merge junctions (degree >= 3
fragment nodes, segment-scoped non-maximum suppression within 20 um) plus
tip-to-shaft candidates (shaft nodes within 20 um of a tip of one of the two
nearest other segments). Two sweeps measure the enumeration ceiling across all
labeled brains and serve as evidence for the declaration:

- [`notebooks/split_candidate_pool_sweep.py`](../notebooks/split_candidate_pool_sweep.py)
  enumerates segment pairs by pairing mode, maximum gap and per-anchor partner
  quota, and reports candidate count against the fraction of GT split pairs
  reachable in the pool. `agentic/run_split_candidate_result_analysis.py` turns a
  finished bundle into a grounded `AI_REVIEW.md`.
- [`notebooks/merge_candidate_pool_sweep.py`](../notebooks/merge_candidate_pool_sweep.py)
  enumerates junction rules and, with `--tip-gaps`, tip-to-shaft additions through
  `candidate_rules.py`, reports site recall at each geodesic claim radius, and lists
  the declared rule per brain in `RESULTS.md`.

The build driver freezes the declared rule as `split_candidate_policy.json` or
`merge_candidate_policy.json` (with the SHA-256 of the declaration and of
`candidate_rules.py`), copies `candidate_rules.py` beside the detector and embeds
the rule and both hashes. The benchmark tables enumerate with the same module, so
the detector and the benchmark share one definition. Candidate enumeration is
therefore fixed before training and cannot be tuned against labels later.

## 2.2 Build the detector

```bash
conda activate panda
python agentic/run_detector_build_workflow.py autodiscovery/<RUN>.json
```

One persistent Claude session runs the ordered steps so later steps share
earlier context; the driver validates every artifact between steps:

| Step | Agent proposes | Driver validates and writes |
|---|---|---|
| 0. candidate rule | no agent turn | freezes the rule declared in `candidate_pools.json` as `*_candidate_policy.json` with the declaration and module hashes; step 3 copies `candidate_rules.py` beside the detector |
| 1. feature inventory | source choice, feature math, aggregation and defined-condition per hypothesis | joins IDs, evidence, paths and hashes; rejects GT-referencing rows; writes `feature_inventory.json` |
| 2. model setup | optional model families with a feature-based rationale | enforces `agentic/detector_model_policy.json` (L2 logistic, histogram gradient boosting and XGBoost always included, at most two extensions, bounded grids); writes `model_candidates.json` |
| 3. detector assembly | the task-specific feature implementation only | AST-validates it, injects it into the reviewed runtime template, runs no-data smoke tests; writes `merge_junction_detector.py` or `split_site_detector.py` |
| 4. verify and document | semantic review of the assembled script against the inventory | README skeleton with an immutable provenance block and `RUN_COMMANDS.md` bound to the artifact hashes |

Deliverables land in `autodiscovery-application/<RUN>/`. Rebuilding into an
existing folder asks for confirmation because it would leave a new script beside
an old run's outputs; `--yes` skips the prompt in batch jobs and `--keep-existing`
reuses validated artifacts. `--merge-row-unit segment` restores the older
segment-level merge detector instead of junction sites. `--keep-overturned ID ...
--keep-overturned-reason TEXT` lets the inventory include OVERTURNED hypotheses
whose report entries predate the `Feature effect` bullet (reports since
2026-10-21 carry it, and the build then excludes only `ARTIFACT`); the ids and
reason are recorded in the README's provenance block. Feature contracts are
documented in [`docs/merge_site_feature_scope.md`](merge_site_feature_scope.md),
[`docs/split_feature_scope.md`](split_feature_scope.md) and
[`docs/detector_cost_validation.md`](detector_cost_validation.md).

## 2.3 Run the detector on a compute node

`RUN_COMMANDS.md` in each deliverable folder is the authoritative, hash-bound
runbook. The shape of a run is:

```bash
sbatch --mem=80G --wrap="\
    source /shared/utils.x86_64/anaconda3-2024.10/etc/profile.d/conda.sh; \
    conda activate panda; \
    python autodiscovery-application/<RUN>/merge_junction_detector.py \
        cache/dataset_cache_<brain>_mcl<N>_add.pkl"
```

Any detector with an image-derived hypothesis runs its image pass through a
bounded thread pool (`--n-jobs N`). Since 2026-10-21 this holds for any image
hypothesis; before that only when image hypotheses were a strict majority,
which left the 2026-10-19/20 builds serial at about 3 rows/s. The pass waits
on S3, so `N` may exceed the CPU count; rows are submitted in a chunk-friendly
spatial order (`_image_row_order`) and the volume is opened with a cache pool
and S3 request limit sized for the pool (`_image_store_context`). Stage 3
scores every brain with the same extraction code, so this throughput is what
Stage 3 pays.

The runbook first profiles feature cost on a small sample (`--measuretime`;
`agentic/run_feature_cost_analysis.py` adds the complementary code-structure
view), then runs the chosen hypotheses on the full brain with average precision
as the model-selection objective. Labels are three-valued, as everywhere else in
the repository since 2026-10-09: 1 (error), 0 (GT-confirmed no error) and NaN
(GT cannot judge the row: a split pair whose segments do not both carry a GT node
label, a merge candidate whose segment carries none or that lies beyond the claim
radius of a GT node; `candidate_rules.label_availability`, the same rule as the
Stage 3 availability sidecars). The model is fitted, selected and evaluated on the
judgeable rows only and scores every row. Folds are the benchmark's own internal
folds (`candidate_rules.fold_groups` / `fold_partitions`): split rows group by the
500 um block of their anchor midpoint, merge-site rows by segment; every judgeable
row is held out exactly once and the fit side of each fold is purged of every
segment a held row touches. With the brain id as the group prefix the held sets
are the companions' `fold` sets, so the detector's out-of-fold AP is measured on
the rows the baseline card and the discovery scripts used. (Before 2026-10-21 the
runtime assigned segments to folds at random and dropped every pair whose
segments landed in different folds, which left 80% of split rows without an
out-of-fold score, and it trained on all rows with 0 for every unjudgeable pair.)
A completed run writes the candidate scores (`*_detector_<brain>.csv`; the
out-of-fold columns are NaN on unjudgeable rows, the fitted model's score covers
all rows), the model-selection manifest (`model_selection_<brain>.json`, which
records the training brain, feature order, label rule and fold rule that Stage 3
relies on), the fitted winner (`*_<brain>.joblib`), a log and figures. Stage 3
takes the run directory itself (`--merge-dir` / `--split-dir`), finding the
script in the deliverable directory above it and checking the run's recorded
detector SHA-256 against it. `agentic/run_detector_importance_report.py` then writes the
feature-importance evidence and short report from the saved numbers. The GT-site tools in
[the GT merge-site tools](gt_merge_site_tools.md) compare a detector's
positives with the cached GT sites.

## 2.4 Detectors currently in use

Since 2026-10-23 Stage 3 resolves these two runs by default (`DEFAULT_MERGE_DIR` /
`DEFAULT_SPLIT_DIR` in `proofreader_evolve/cli/precompute_error_scores.py`). Both were fitted
on 794495 with the labels of the 2026-10-15 relabel and the schema-2 pools of
`proofreader_evolve/candidate_pools.json`; the AP figures are on the GT-judgeable rows:

| kind | run directory | winner | nested OOF AP | raw baseline | held-out 802449 AP |
|---|---|---|---:|---:|---:|
| merge | `autodiscovery-application/new-merge-1_2026-10-19/runs/794495-with-figures` | `logistic_l2` (C = 0.001), 293 features | 0.0619 | degree 0.0278 | 0.0746 (degree 0.058) |
| split | `autodiscovery-application/new-split-1_2026-10-20/runs/794495-with-figures` | `hist_gradient_boosting`, 352 features | 0.8793 | `gap_um` 0.8255 | 0.9139 (`gap_um` 0.8905) |

Each run directory holds the fitted joblib, `model_selection_794495.json` (whose
`data_provenance.detector_sha256` binds the run to the script two levels up), the scored CSV
with its provenance sidecar, the figures and `IMPORTANCE_REPORT.md`. The 2026-08 detectors
(merge `junction|nms=20|r=150`, split `tip_to_any_node|r=50|k=2`) and all AutoDiscovery results
of that generation were deleted on 2026-10-17: they were trained on the labels before the
relabel and on the old candidate pools. A detector's fitting brain can never serve as a
validation brain in Stage 3.

### Feature-importance short report (2026-10-22)

`agentic/run_detector_importance_report.py <run dir>` writes `importance_evidence.json`
(the winner's standardized coefficients or native importances, in-sample permutation
drops on the judgeable rows, univariate AP of every feature against the raw baseline,
and the same views summed per hypothesis with the discovery verdicts and hypothesis
text) and a short English `IMPORTANCE_REPORT.md` written by the
`detector-importance-summarizer` subagent from that evidence alone; `--no-agent`
writes the evidence and the tables only. The driver checks the report (immutable block,
headings, placeholders) and caps the summary at 600 words. Figures are not required, so
it also serves runs made with `--no-figures`. The earlier full bilingual report
(`run_detector_result_analysis.py`, `RESULT_ANALYSIS.md`) was deleted on 2026-10-22 without
ever having been run; its result-directory helpers live in `agentic/detector_importance.py`.

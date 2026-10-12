# Plan: an AutoDiscovery specification aligned with proofreader_evolve

Status (2026-10-11): partly implemented. Done:
- the specifications, one per error type and submitted separately (2026-10-12):
  `markdowns/merge_features_for_proofreader_evolve.md` and
  `markdowns/split_features_for_proofreader_evolve.md` (`describe(context)` deliverable);
- the companion export `proofreader_evolve/cli/export_rerank_companion.py`, with
  the 794495 companions in `markdowns/` next to the specification (gitignored).

Deferred by decision: everything involving 802449 (audit companion, confirmation
step), `--companion` support in the discovery workflow, and the verify-step checks.

Decisions taken the same day:
1. Keep the existing workflow input (`_add.pkl`) plus a small companion file.
2. Dataset only: AutoDiscovery discovers new features from the dataset alone. It gets
   nothing from earlier AutoDiscovery runs, detectors or evolve runs: no
   `detector_score`, no frozen feature table, no feature catalogue, no past
   hypotheses.
3. AutoDiscovery gets the evaluation principle and the scoring code, but not evolve's
   exact K and not the confirmation data.
4. AutoDiscovery sees one brain only: **794495**. 802449 is the confirmation brain
   in our audit workflow; the development-validation brains are never used.

## Why

The task specifications in `markdowns/` were written in August for discovering
features and building detectors from scratch. A review on 2026-10-11 found five gaps
relative to `proofreader_evolve`:

| Gap | Current specs | proofreader_evolve |
|---|---|---|
| Candidate set | Free choice: random controls; `tip_to_tip` plus `tip_to_any_node` with free r/k | Fixed pools: merge `junction\|nms=20\|r=150`, split `tip_to_any_node\|r=50\|k=2` |
| Negatives | Every non-positive control counts as 0 (merge specs) | 1 / 0 / NaN; only GT-judgeable rows are 0 |
| Metric | Set recall/precision, AUC, AP | Precision@K over GT-labeled rows (merge K=300, split K=500) |
| Firewall | Absolute coordinates allowed as inputs (image specs, split endpoint xyz) | No coordinates or IDs in scorer inputs |
| Image path | "Open the cached `img_path`" plus a gs:// branch | Reviewed URI in `configs/image_alignment.json` (the cached 794491 path is deleted) |

Findings from such runs do not transfer: features are validated on a different
candidate set, against wrong negatives, under a different objective.

## Current workflow this plan must fit

1. AutoDiscovery receives one `_add.pkl`, one specification from `markdowns/`, and
   for image access the GCS credential JSON. It exports one run JSON.
2. `agentic/run_discovery_workflow.py <RUN>.json --pkl <same _add.pkl>` audits it:
   - selection;
   - reproduce (re-runs the recorded code against `--pkl` via
     `agentic/rerun_experiments.py`);
   - generalization on `--extra-pkl` brains;
   - statistical verification;
   - test correction;
   - `summary.md`.
3. The rerun helper owns the inputs. It resolves any `*.pkl` literal to the given
   pkl, and it stages the repo GCS credential next to the scripts (symlink into the
   working directory, `$RERUN_GCS_TOKEN`; `rerun_experiments.py` around lines
   133-145 and 600-625).

The new specification keeps all of this. AutoDiscovery still gets an `_add.pkl`; the
alignment comes from one extra file and from the specification text.

## Principles

**Dataset only.** AutoDiscovery uses only the `_add.pkl` (fragment graph, GT graph,
labels) and the raw image. It gets two things that are pure functions of that
dataset, fixed so its findings transfer to evolve:
- the **candidate rule**: where a feature is evaluated. It is geometric and does not
  depend on any trained model;
- the **label rule**: 1 / 0 / NaN, derived from GT in the same pkl.

Combining new features with `detector_score` is evolve's job, not AutoDiscovery's.

**Objective without overfitting to evolve's score.**
- AutoDiscovery is told how features will be judged: GT-labeled rows only, and
  precision at the top of the ranking.
- It reports a curve, not a single number. It does not get evolve's K (300 / 500
  came from a sweep on the frozen detector, an earlier artifact) and has no access
  to the confirmation brain.
- The exact evolve metric is computed only in our audit, on 802449.

**Brain roles.**

| Brain | Role in this workflow | Labels visible to AutoDiscovery |
|---|---|---|
| 794495 | Discovery (characterization, feature design, out-of-fold checks) | Yes |
| 802449 | Confirmation in our audit (`--extra-pkl`), exact evolve metric | No |
| 789202, 794493, 794491, older-microscope brains | Not used (evolve development validation) | No |

- 794495 is the brain the frozen detectors were fitted on. That is irrelevant here,
  because AutoDiscovery never sees the detectors. In evolve, 794495 stays an
  auxiliary TRAIN brain.
- 802449 is also evolve's selection brain. Every confirmation measurement spends a
  little of that selection signal, so the audit records which features were
  confirmed on it and how often. Only features that pass the 794495 out-of-fold
  checks are confirmed.

## Design

- **Input given to AutoDiscovery**: `dataset_cache_794495_mcl100_add.pkl` +
  specification + credential JSON + companion files
  `rerank_companion_794495_merge.npz` and `rerank_companion_794495_split.npz`.
- **Candidate and label rules in the specification text**, so the task is
  self-contained:
  - merge: every fragment node of degree >= 3, non-maximum suppression 20 um within
    the same segment;
  - split: from each degree-1 fragment tip, the nearest node on each other segment
    within 50 um, keeping the 2 nearest partner segments (`tip_to_any_node`);
  - labels: positives from the GT labels in the pkl (merge: within 20 um geodesic
    of a merge site snapped to its own segment; split: the segment pair is a split
    key); 0 only where GT can judge (split: both segments carry a GT node label;
    merge: segment on GT and junction within 150 um of a GT node); otherwise NaN.
- **Companion contents** (dataset-derived only, generated by the existing
  enumeration and label code, so AutoDiscovery does not re-implement them and the
  workflow can check the match):
  - candidate identities in pool row order (merge: segment id, node id, x/y/z um;
    split: segment ids a/b, node ids);
  - the three-valued label (1 / 0 / NaN);
  - fold ids for out-of-fold checks on 794495: label-free spatial and fragment
    grouping from `internal_validation.grouped_folds`, with 794495 as the grouped
    brain;
  - a manifest: brain, kind, `pool_sha256`, labels digest, sidecar version, rule
    text, reviewed image URI.
  - Not included: `detector_score`, frozen detector features, any model output,
    evolve's K.
- **The 802449 companion** has the same format, with labels. It is exported for the
  audit workflow only and never uploaded to AutoDiscovery.
- **Identity columns** only locate candidates in `fragments_graph` and the image;
  the specification forbids them as feature inputs.
- **Context**: AutoDiscovery computes local geometry and image patches itself from
  the pkl and the reviewed image URI, as today. No context arrays are shipped.

## Remaining decisions

1. **Deliverable.**
   - (a) Recommended: each feature as a `describe(context)`-style function over
     relative geometry and patch arrays (the contract in
     `proofreader_evolve/artifacts/descriptor_guide.md`), plus the code that builds
     that context from the pkl. Accepted findings port to evolve descriptors with
     minimal rewriting.
   - (b) Free-form per-candidate code over the pkl: easier for AutoDiscovery, but
     each finding must be re-implemented.
2. **Who enumerates candidates.**
   - Recommended: the companion supplies them; the specification also states the
     rule.
   - Alternative: AutoDiscovery enumerates from the rule itself. A small
     implementation difference would change the pool and break transfer.

## Steps

1. **Companion export** `proofreader_evolve/cli/export_rerank_companion.py`
   (compute node).
   - For 794495 (AutoDiscovery) and 802449 (audit only): loads tables with
     `ensure_native_tables`, applies `label_availability`, computes the grouped
     folds, and writes one companion per kind with the manifest.
   - No new label, fold or pool logic.
2. **Scoring code inside the specification** (ranking among labeled rows only,
   candidate-key tie-break, per-fold out-of-fold evaluation):
   - Precision at K = 1%, 2.5%, 5% and 10% of the labeled rows;
   - average precision over labeled rows;
   - a paired bootstrap of the difference to a baseline.
   - Check on our side (not shipped) that the same code reproduces evolve's
     `rank_metrics` / `partitioned_rank_metrics` exactly when given evolve's K.
3. **Rerun support** in `agentic/rerun_experiments.py` and
   `agentic/run_discovery_workflow.py`.
   - A `--companion` option per pkl (origin and each `--extra-pkl`), staged like the
     credential: symlinked into the script working directory and exported as
     `$RERUN_COMPANION`, so recorded code that opens the 794495 companion by name
     resolves to the right brain's file.
   - Bound to the pkl by brain id and kind in the manifest.
4. **Confirmation step** in the audit workflow.
   - Run each feature that passed on 794495 against 802449 (`--extra-pkl` plus its
     companion), reporting evolve's exact metric: Precision@K over labeled rows,
     merge K=300, split K=500.
   - Single features are ranked directly. A model of the new features is fitted on
     794495 and scored on 802449, never refitted on 802449.
   - Log every confirmation (feature code hash, date) in the summary, so later
     evolve runs know which features have already been measured on the selection
     brain.
5. **Specification text** (now split into `markdowns/merge_features_for_proofreader_evolve.md` and
   `markdowns/split_features_for_proofreader_evolve.md`, each with Dataset context / Intent / Dataset schema):
   - §1 Context: the candidate rule, the three-valued labels and what NaN means.
     Phase 1 (characterization) may use GT; Phase 2 features may not.
   - §2 Schema: the `_add.pkl` fields needed (pointer to
     `labeled_dataset_cache.md`), the companion schema, how to map a companion row
     to `fragments_graph` nodes, and the reviewed image URI.
   - §3 Evaluation:
     - the judging principle (labeled rows only, top-of-ranking precision) and the
       scoring code;
     - out-of-fold on 794495 with the given folds, each feature alone and a model
       of the new features;
     - baselines computed inside the task: random ranking (equals the labeled base
       rate) and one simple geometric quantity per kind (split: gap distance;
       merge: junction degree);
     - the P@K curve, AP, bootstrap intervals and per-fold results.
     - The text states that features are confirmed later on data the run cannot
       see. It does not name the brain or the K values.
   - §4 Firewall:
     - no GT, labels, coordinates, IDs or pool membership inside a feature;
     - no availability proxies: every ranked row is GT-judgeable, so separating
       "near GT" from "far from GT" cannot help;
     - images only from the manifest URI.
   - §5 Open discovery: no feature catalogue, no example features and no
     references to earlier runs anywhere in the text. Hypotheses must come from the
     data.
   - §6 Deliverable: feature code plus a short evidence record (hypothesis,
     out-of-fold curve, bootstrap interval, failure cases).
6. **Verification checks** in the discovery workflow.
   - Extend the verify step to check that Phase 2 features do not read identity
     columns, coordinates, labels or GT.
   - The GT firewall check already exists; add coordinates and IDs to it.
7. **Dry run on our side**, before any AutoDiscovery run (this test material is never
   given to AutoDiscovery):
   - export the 794495 and 802449 companions;
   - write one simple feature as a recorded-style script (for example local
     intensity continuity across the split gap);
   - run reproduce on 794495 and the confirmation step on 802449;
   - confirm a deliberately forbidden input (coordinates) is flagged.
8. **Hand-back path.** Confirmed features go to
   `proofreader_evolve/artifacts/descriptors/` as examples with their measured
   outcomes on 794495 and 802449. They are not auto-registered; evolve still measures
   them on its own gate.
9. **Docs.**
   - `markdowns/README.md`: the new spec becomes the current re-ranking spec; v2/v3
     are labeled "detector discovery, not aligned with proofreader_evolve".
   - `WORKFLOW_REVISION.md`: dated entry plus a provenance row (project-specific
     design).
   - README Stage 1: the companion files, `--companion`, and the confirmation step.

## Related issue (recorded in proofreader_evolve/TODO.md, 2026-10-11)

The context cache holds only the top 50k detector-ranked rows per brain and kind, so
in evolve many GT-labeled rows have NaN descriptors (794491 split: 41% of positives
in the band). This does not affect this plan, because AutoDiscovery computes context
itself. It does affect how much a ported feature can help inside evolve, until the
cache is extended to all labeled rows.

## Not in scope

- Combining new features with `detector_score`: evolve does that when a finding is
  ported.
- Changing the pools or K.
- Retraining the detectors.
- Changing how evolve registers or gates descriptors.

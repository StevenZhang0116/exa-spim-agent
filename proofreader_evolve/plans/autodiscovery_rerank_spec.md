# Plan: AutoDiscovery specifications aligned with proofreader_evolve

Status (2026-10-12): implemented except the items marked deferred under Steps.

## Pipeline

AutoDiscovery results reach proofreader_evolve through the existing stages:

1. AutoDiscovery receives one submission per error type and exports one run JSON.
2. `agentic/run_discovery_workflow.py <RUN>.json --pkl <_add.pkl> --companion <npz>`
   audits it: selection, reproduce (`agentic/rerun_experiments.py` re-runs the
   recorded code), extrapolate on `--extra-pkl`, verify, fix-tests, `summary.md`.
3. `agentic/run_detector_build_workflow.py <RUN>.json` turns the validated features
   into a detector (inventory, model setup, assembly).
4. The detector runs on a compute node; its out-of-fold score becomes
   `detector_score` (p_0) in proofreader_evolve.

## Why

The task specifications written in August (now in `markdowns/archive/`) were for
discovering features and building detectors from scratch. A review on 2026-10-11
found five gaps relative to proofreader_evolve:

| Gap | August specs | proofreader_evolve |
|---|---|---|
| Candidate set | Free choice: random controls; `tip_to_tip` plus `tip_to_any_node` with free r/k | Fixed pools: merge `junction\|nms=20\|r=150`, split `tip_to_any_node\|r=50\|k=2` |
| Negatives | Every non-positive control counts as 0 (merge specs) | 1 / 0 / NaN; only GT-judgeable rows are 0 |
| Metric | Set recall/precision, AUC, AP | Precision@K over GT-labeled rows |
| Firewall | Absolute coordinates allowed as inputs (image specs, split endpoint xyz) | No coordinates or IDs as feature values |
| Image path | "Open the cached `img_path`" plus a gs:// branch | Reviewed URI in `configs/image_alignment.json` |

Findings from such runs do not transfer: features are validated on a different
candidate set, against wrong negatives, under a different objective.

## Decisions

1. **One submission per error type.** `markdowns/merge_features_for_proofreader_evolve.md`
   and `markdowns/split_features_for_proofreader_evolve.md`, each with Dataset
   context / Intent / Dataset schema. A submission is the spec, the 794495 `_add.pkl`
   and that kind's companion `.npz` + `.json`. No credential: the image is public S3.
2. **Dataset only.** No `detector_score`, frozen feature table, feature catalogue,
   past hypotheses or evolve K. AutoDiscovery gets the candidate rule, the label rule,
   the evaluation principle and the scoring code.
3. **Fixed candidates from the companion.** AutoDiscovery does not re-enumerate; a
   small implementation difference would change the pool and break transfer.
4. **Deliverable: per-candidate features as re-runnable experiments.** Each feature
   is one value per companion row, recorded with hypothesis, code, result and a
   precise definition (split: anchor and occurrence reduction; merge: candidate or
   segment scope; formula and constants), which is what the Stage 2 inventory needs.
5. **Allowed inputs:** the whole fragment graph, the label-free candidate columns
   (split: gap, anchor side, partner rank, node roles, occurrences; merge: degree)
   and the raw image from the manifest URI. Excluded: GT, labels, folds and fit
   masks, ids or coordinates as values, availability proxies, dense segmentation.
   Stage 1 verify flags and Stage 2 rejects GT-referencing features.
6. **Helpers, not a contract.** `neighbourhood()`, `image_patch()` and
   `make_context()` are optional; they reproduce evolve's context-cache geometry and
   image patches exactly (checked by `scripts/check_rerank_spec_code.py`).
7. **Objective without overfitting to evolve.** Out-of-fold precision at 1 / 2.5 / 5
   / 10% of labeled rows, AP and a paired bootstrap against in-task baselines
   (random; split gap; merge degree and segment cable length). The spec names
   neither evolve's K nor the confirmation brain.

## Brain roles

| Brain | Role | Labels visible to AutoDiscovery |
|---|---|---|
| 794495 | Discovery (characterization, feature design, out-of-fold checks) | Yes |
| 802449 | Confirmation in our audit (`--extra-pkl` with its companion) | No |
| 789202, 794493, 794491, older-microscope brains | Not used (evolve development validation) | No |

- 794495 is the brain the current frozen detectors were fitted on; a new detector
  is fitted there too, so it stays out of evolve validation.
- 802449 is also evolve's selection brain. Every confirmation spends a little of
  that selection signal, so the audit should record which features were confirmed
  on it and how often.

## Companion files

`proofreader_evolve/cli/export_rerank_companion.py --brain <brain>` writes
`markdowns/rerank_companion_<brain>_{merge,split}.{npz,json}` (gitignored), from the
existing enumeration, label-availability and grouped-fold code (no new logic):
- candidate identities in pool row order, the three-valued label, label-free folds
  and per-fold fit masks, the label-free candidate columns;
- a manifest with rule text, counts, pool / label / npz digests, fold description,
  the source pkl name and the reviewed image URI with level geometry;
- not included: detector scores, frozen features, model output, evolve K.

## Steps

1. Done: companion export; 794495 companions in `markdowns/`.
2. Done: the two specifications, with scoring code and helpers; end-to-end check in
   a fresh environment (install line, pkl load, helpers against evolve's context
   cache, baselines) by `scripts/check_rerank_spec_code.py`.
3. Done: `--companion` in `agentic/rerun_experiments.py` and
   `agentic/run_discovery_workflow.py`. Each companion is bound to the pkl its
   manifest names (`source_cache.file`), checked against `npz_sha256`, linked into
   the script cwd under every companion name the code uses, and part of the
   checkpoint key.
4. Deferred: export the 802449 companion and confirm features on it (Stage 1
   `--extra-pkl`), logging each confirmation (feature code hash, date).
5. Deferred: extend the verify step to flag ids and coordinates used as feature
   values (GT access is already flagged).
6. Deferred: a dry run with one simple recorded-style feature through Stage 1 and
   Stage 2 before the first real submission.

## Related issue (proofreader_evolve/TODO.md, 2026-10-11)

The context cache held only the top 50k detector-ranked rows per brain and kind, so
many GT-labeled rows had NaN descriptors in evolve. A new detector from this
pipeline scores every candidate, so its p_0 is not affected. Since 2026-10-12
`precompute_context_cache --labeled-extension` also caches the GT-labeled rows outside
the band (code done; the builds are pending), which closes that gap for evolve's own
descriptors.

## Not in scope

- Changing the pools or evolve's K.
- Changing how evolve registers or gates descriptors.

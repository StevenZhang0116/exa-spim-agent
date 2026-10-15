# markdowns/

Self-contained documents for agents: the cache schemas and the task specifications
handed to AutoDiscovery together with one labeled cache. `proofreader_evolve` does not
read these files.

## Current documents

| File | Purpose |
|---|---|
| `labeled_dataset_cache.md` | Schema of the labeled `_add.pkl` caches: fragment and GT graphs plus the baked-in merge/split/omit labels. Loads offline. The reference cited by README, CLAUDE.md, the detector-builder agent and `agentic/run_detector_build_workflow.py`. |
| `dataset_cache.md` | Schema of the plain `dataset_cache_<brain>_mcl<N>.pkl` (GT and U-Net skeleton graphs only, no labels), for trials that receive a collection of plain caches. |
| `merge_features_for_proofreader_evolve.md` | **Merge feature discovery for `proofreader_evolve`** (2026-10-12). Submitted to AutoDiscovery with `dataset_cache_794495_mcl100_add.pkl` and `rerank_companion_794495_merge.{npz,json}`. Dataset only: discover GT-blind per-candidate features (whole fragment graph, label-free candidate columns, raw image) on the fixed merge junctions, judged by out-of-fold top-of-ranking precision over GT-labeled rows (1 / 0; NaN excluded). No detector scores, earlier features or feature catalogue are provided. Results go through `agentic/run_discovery_workflow.py` and `agentic/run_detector_build_workflow.py` to a new frozen detector, whose score is `detector_score` (p_0) in `proofreader_evolve`. |
| `split_features_for_proofreader_evolve.md` | **Split feature discovery for `proofreader_evolve`**, the same task for the fixed split segment pairs. Submitted with the same pkl and `rerank_companion_794495_split.{npz,json}`. |
| `rerank_companion_<brain>_{merge,split}.{npz,json}` | Companion files of the two specifications: candidates, three-valued labels, folds, manifest. Gitignored; regenerate on a compute node with `python -m proofreader_evolve.cli.export_rerank_companion --brain 794495`. Pass them to `agentic/run_discovery_workflow.py` with `--companion` when auditing the run. Present for 794495 (the submission, regenerated 2026-10-13 from the rebuilt cache) and 802449 (audit confirmation, exported 2026-10-14). |

Each error type is one AutoDiscovery submission: its specification, the pkl and its
two companion files. Plan: `proofreader_evolve/plans/autodiscovery_rerank_spec.md`.

The earlier detector-discovery specifications (`merge_detection_from_fragmentsv2.md`,
`split_detection_from_fragmentsv3.md`, `merge_detection_from_imagev2.md`,
`split_detection_from_imagev3.md`) are in `archive/` since 2026-10-12. They are not
aligned with `proofreader_evolve`: they use free candidate sets, count every
non-positive as negative, report AUC/AP or set metrics, and allow coordinates as
inputs.

All task specifications share one rule: a GT firewall. Phase 2 code (detector or feature) may read
only the fragment reconstruction (and, where allowed, the raw image), never GT fields.
The specification names the forbidden cache fields.

## Lineage (dates are when the file was added to git)

| Family | Versions, oldest to newest | Current |
|---|---|---|
| Merge, fragments | `merge_detection_from_fragments.md` (07-28) → `_orig` (08-07, earlier "blind detection" framing saved as a copy) → `v2` (08-24) | archived |
| Split, fragments | `split_detection_from_fragments.md` and `_orig` (08-07) → `_simple` (08-11, simpler computation only) → `v2` (08-24, adds open exploration) → `v3` (08-24, two candidate views) | archived |
| Merge, image | `merge_detection_from_image.md` (08-11) → `v2` (08-20) | archived |
| Split, image | `split_detection_from_image.md` (08-11) → `v2` (08-20, disciplined discovery) → `v3` (08-24) | archived |
| Feature discovery for `proofreader_evolve` | combined draft (10-11, not committed) → one specification per error type (10-12) → both revised (10-13: substitute-data ban, measured baseline figures on the companions regenerated from the rebuilt 794495 cache, increment-over-baseline rule; the 10-12 texts are `archive/{merge,split}_features_for_proofreader_evolve_20261012.md`) | `merge_features_for_proofreader_evolve.md`, `split_features_for_proofreader_evolve.md` |
| Cache intent | `labeled_dataset_cache_section3_discovery.md` (06-29): an alternative "§3 Intent", open discovery of what splits, merges and omits are associated with; superseded by the task specifications | — |

Superseded versions are kept in `archive/` unchanged, so past AutoDiscovery runs can
still be matched to the text they were given. The repository does not record which
specification each AutoDiscovery run used; check that run's export.

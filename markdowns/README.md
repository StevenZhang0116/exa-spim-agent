# markdowns/

Self-contained documents for agents: the cache schemas and the task specifications
handed to AutoDiscovery together with one labeled cache. `proofreader_evolve` does not
read these files.

## Current documents

| File | Purpose |
|---|---|
| `labeled_dataset_cache.md` | Schema of the labeled `_add.pkl` caches: fragment and GT graphs plus the baked-in merge/split/omit labels. Loads offline. The reference cited by README, CLAUDE.md, the detector-builder agent and `agentic/run_detector_build_workflow.py`. |
| `dataset_cache.md` | Schema of the plain `dataset_cache_<brain>_mcl<N>.pkl` (GT and U-Net skeleton graphs only, no labels), for trials that receive a collection of plain caches. |
| `merge_features_for_proofreader_evolve.md` | **Merge feature discovery for `proofreader_evolve`** (2026-10-12, revised 10-13 and 10-15). Submitted to AutoDiscovery with `dataset_cache_794495_mcl100_add.pkl` and `rerank_companion_794495_merge.{npz,json}`. Dataset only: discover GT-blind per-candidate features (whole fragment graph, label-free candidate columns, raw image) on the fixed merge junctions, judged by out-of-fold top-of-ranking precision over GT-labeled rows (1 / 0; NaN excluded). No detector scores, earlier features or feature catalogue are provided. Results go through `agentic/run_discovery_workflow.py` and `agentic/run_detector_build_workflow.py` to a new frozen detector, whose score is `detector_score` (p_0) in `proofreader_evolve`. |
| `split_features_for_proofreader_evolve.md` | **Split feature discovery for `proofreader_evolve`**, the same task for the fixed split segment pairs. Submitted with the same pkl and `rerank_companion_794495_split.{npz,json}`. |
| `rerank_companion_<brain>_{merge,split}.{npz,json}` | Companion files of the two specifications: candidates, three-valued labels, folds, manifest. Gitignored; regenerate on a compute node with `python -m proofreader_evolve.cli.export_rerank_companion --brain 794495 --kinds split --pool-only` (rows and labels straight from the cache with the pool declared in `proofreader_evolve/candidate_pools.json`, minutes; without `--pool-only` it needs prepared tables and sidecars). Pass them to `agentic/run_discovery_workflow.py` with `--companion` when auditing a run, always the files that run was given. Present for 794495 (the submission) and 802449 (audit confirmation). Split: regenerated 2026-10-15 for the `tip_to_any_node|r=80|k=2` pool; the r=50 files of 2026-10-13/14, which split runs 1 and 2 were given, are in `archive/companions_20261014/` under their original names. Merge: still the 2026-10-13/14 exports (pre-relabel labels), to be regenerated. |

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
| Feature discovery for `proofreader_evolve` | combined draft (10-11, not committed) → one specification per error type (10-12) → both revised (10-13: substitute-data ban, measured baseline figures on the companions regenerated from the rebuilt 794495 cache, increment-over-baseline rule; the 10-12 texts are `archive/{merge,split}_features_for_proofreader_evolve_20261012.md`) → both revised again (10-15: tie-aware average precision with corrected baseline figures, AP-difference and subgroup reporting rules, the `INPUTS READ` line, fewer constraints on what to explore; the 10-13 texts are `archive/{merge,split}_features_for_proofreader_evolve_20261013.md`) → split revised a second time the same day (10-15: candidate pool `tip_to_any_node|r=80|k=2` instead of r=50, row and label counts and gap baseline from the regenerated companion; the text before it is `archive/split_features_for_proofreader_evolve_20261015.md`) | `merge_features_for_proofreader_evolve.md`, `split_features_for_proofreader_evolve.md` |
| Cache intent | `labeled_dataset_cache_section3_discovery.md` (06-29): an alternative "§3 Intent", open discovery of what splits, merges and omits are associated with; superseded by the task specifications | — |

Superseded versions are kept in `archive/` unchanged, so past AutoDiscovery runs can
still be matched to the text they were given. The repository does not record which
specification each AutoDiscovery run used; check that run's export.

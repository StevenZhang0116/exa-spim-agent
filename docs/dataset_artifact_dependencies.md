# What depends on a dataset cache, and what to rerun when it changes

Current dataset reservation (2026-11-04): 794495 is excluded from new development
training, selection, tuning, testing and evolution promotion. The 794495 rebuild
measurements below are historical evidence, not instructions to rebuild or reuse
that brain. Use the explicit [current split](stage3_evolution.md#32-prepare-tables-and-launch-a-run)
for new runs; the independent validation protocol remains to be designed.

Segmentation source update (2026-11-05): the Google "denoised" segmentations replace the
sources of 789202 (`whole_brain_denoised/denoised_v2.unet_tri_denoised_2370000_move_09_segment_09_it2`)
and 802449 (`whole_brain_denoised/denoised_v2.unet_tri_denoised_2370000_move_09_segment_09`), and
841260 (`whole_brain_denoised/denoised.unet_tri_denoised_2370000_move_09_segment_09_it2`, 13 GT
neurons, image `exaSPIM_841260_2026-07-07_15-13-51_processed_2026-07-26_09-25-11`) is a new brain.
All three are explicit entries in `configs/segmentation_paths.json` (fragments in `<id>/swcs`; the
formats are unchanged, 748 nm sharded precomputed and zip-sharded voxel SWCs). 820994 was not
added: the shared folder did not exist and the brain has no GT tracings. For 789202 and 802449
this is the graph reload of section 3 (`--rebuild-base`, tables, the 802449 companions, alignment
receipts, preflight, then a new seed baseline; `metrics_out/<brain>/` gets a new segmentation-id
directory; old runs are deleted, not archived). 841260 needs the full build,
its image prefix (added) and an alignment review. All of this was done on 2026-11-06/07 (caches,
raw pools, 802449 companions, receipts from the `figs/candidate_image_alignment_20261106` overlays,
preflight); the old 789202/802449 lines were removed from `configs/segmentation_datasets.rtf` on
2026-11-07 because `configs/segmentation_paths.json` takes precedence. The cache file name carries no
segmentation id, so the old and new sources of one brain cannot coexist in `cache/`.

Written 2026-10-14 after the 794495 rebuild. Every derived artifact of a brain descends from
its labeled dataset cache `cache/dataset_cache_<brain>_mcl<N>_add.pkl`. This page lists those
artifacts, what each one is bound to (the check that fails when the cache changes), the order
in which to rebuild them, and the things that are *not* invalidated automatically but stop
being comparable. Times are what the 794495 rebuild of 2026-10-13 took on n268 (16 CPUs).

## 1. Upstream of the cache

| Input | Where it is defined | Used by |
|---|---|---|
| GT tracings (one SWC per neuron, `voxel/` copy) | `gs://allen-nd-goog/ground_truth_tracings/<brain>/voxel` (`labeled_cache/io.py`, `GT_PATH_TEMPLATE`) | `gt_graph` and every GT-derived label in the cache |
| Predicted segmentation (`label_mask`) and fragment SWCs | `configs/segmentation_paths.json` (new layout, takes precedence) or `configs/segmentation_datasets.rtf` + legacy template, resolved by `scripts/dataset_config.py` | `fragments_graph`; `label_gt_from_segmentation`. The two must be one segmentation run: `python -m labeled_cache.source_check --brains <b>` reads sampled fragment node voxels back from the segmentation and reports the id agreement (`docs/fragment_segmentation_match.md`; all eleven brains 97.9-100% on 2026-10-29) |
| Fused image prefix | `configs/exaspim_image_prefixes.json` | `img_path` in the cache (written by the base stage; `python -m labeled_cache --brains <b> --refresh-image-path` rewrites it in both pkl files after a volume moves). Image reads use the URI reviewed in `configs/image_alignment.json` first: on-demand descriptors and companions through `image_context`, the plotting scripts through `scripts/dataset_config.get_reviewed_img_path` / `get_cache_img_path` |
| Benchmark candidate pools | `proofreader_evolve/candidate_pools.json` (schema 2: split `tip_to_any_node|r=80|k=2` since 2026-10-15; merge `junction+tip_to_shaft|nms=20|gap=20|k=2|r=150` since 2026-10-16), applied by `candidate_rules.py` at the repository root (also copied beside every detector built since 2026-10-16) | the candidate rows of every native table and companion of that kind. The rule and the module's SHA-256 are in the table provenance, so editing either gives new table directories: items 2 and 3 below are redone for that kind on every brain. `pool_sha256` hashes the rows only; when the rows change (the usual case for an edited entry), items 4, 6 and 7 are redone too, while a module edit that keeps the rows keeps them valid (check with a pool-only companion export, as on 2026-10-17 for split). The cache itself is untouched |

`python -m labeled_cache` (2026-10-15; before it, `notebooks/generate_add_caches.py` and
`scripts/relabel_cache.py`) builds `dataset_cache_<brain>_mcl<N>.pkl` (graphs loaded), then
`_add.pkl` (labels baked in), and applies the merge-threshold policy to an existing `_add.pkl`
in place. Before reusing either pkl it compares the cached GT neurons with the current GCS
tracings (node counts and coordinates after the same resampling) and refuses a stale cache
unless `--rebuild-base` is given. The old generator reused a base cache without that check,
which is how the stale 794495 cache of 2026-09-23 came about
(`cache/ground_truth_inventory_20261013.md` has the full account).

Two kinds of change have different blast radii:

- **Graph reload** (new or revised GT, new segmentation, rebuilt base cache): node ids are
  renumbered because `SkeletonGraph.load` assigns component ids in load order, even when the
  fragments are unchanged. Everything keyed by node ids or by the pool hash changes.
- **Label-only refresh** (the `policy` stage of `python -m labeled_cache`, for example
  `--policy length_v2`): graphs and node ids stay, only `gt_*` label arrays,
  `gt_merge_sites` and `gt_label_policy` change. Candidate pools and folds stay identical;
  labels, sidecars, companions and anything carrying labels must still be redone.

## 2. Derived artifacts

| # | Artifact | Path | Bound to (the check that fails) | Rerun after graph reload | Rerun after label-only refresh | Command | Time |
|---|---|---|---|---|---|---|---|
| 1 | Image-alignment receipt | `configs/image_alignment.json` (`brains.<brain>`) | cache identity (path, size, mtime, inode) and image identity; `image_context.source` raises `Image/cache identity changed since alignment review` | yes | yes (the pkl file changes) | `python -u scripts/check_image_alignment_from_cache.py --brains <b> --mcl <N> --out figs/candidate_image_alignment_<date>` (samples fragment nodes from the cache; no tables, no detector), inspect the six overlays, then the same command with `--record-reviewed --review-note "..."` (or `--review-notes <json>` for several brains; no cache load). `python -m proofreader_evolve.cli.check_image_alignment` writes the same receipt from table rows once tables and frozen detectors exist | about 1 min per brain (cache load) plus the visual review |
| 2 | Candidate pool tables: raw pools (`--detector-prior raw`, since 2026-11-06) or detector-native tables (`frozen` / `no-score`) | `proofreader_evolve/feature_tables/<brain>/<table_digest>/` (digest of the table provenance: cache identity, declared rule, implementation hashes including `candidate_rules.py`, plus the detector for detector-native tables or the raw column version for raw pools; `candidates.pkl`, `features.pkl`, `evaluator_labels.npy`, `meta.json`; the two kinds of table share `pool_sha256`, which hashes the rows only) | `meta.provenance.source_cache` (cache identity), `pool_sha256` (candidate identities), detector run ids | yes (new pool hash) | yes (labels inside) | raw pools: `python -u proofreader_evolve/prepare_feature_tables.py --brains <b> --mcl <N> --threads all --detector-prior raw` (`cli/precompute_raw_pools.py` per brain, then step 3 and a raw preflight); detector-native tables: the same command without `--detector-prior raw` and with `--image-workers -1` (also writes step 3 before its closing preflight, since 2026-10-23) | raw pools: one cache load plus the enumeration, minutes per brain, no detector; detector-native tables: about 16 h for 794495 with the 2026-10 detectors, dominated by their image pass (merge 68 min at 47 rows/s with 40 workers; split 15 h for 689,677 rows on 2026-10-23/24 while three nodes read S3); 2 h 52 min with the 2026-08 detectors, which had no image features |
| 3 | GT-availability sidecars | `feature_tables/<brain>/<table_digest>.availability/` | table rows, `pool_sha256`, candidate-key digest, labels digest, `source_cache` (`AvailabilityMismatch`) | yes | yes | written by step 2; alone: `python -u -m proofreader_evolve.cli.precompute_availability --brains <b> --mcl <N> --detector-prior raw` for raw pools, without the flag for detector-native tables (valid sidecars are reused) | about 1.5 min |
| 4 | AutoDiscovery companions | `markdowns/rerank_companion_<brain>_{merge,split}.{npz,json}` (gitignored); stale companions are deleted when they are regenerated (2026-10-17) | manifest `pool_sha256`, `labels_sha256`, `npz_sha256`, `source_cache` (file, size); `agentic/rerun_experiments.bind_companions` refuses a companion whose `source_cache.file` is not one of the run's pkls or whose npz differs from its digest | yes | yes | `python -u -m proofreader_evolve.cli.export_rerank_companion --brain <b> --mcl <N> --kinds <kind> --pool-only` (cache only; same rows, labels and `pool_sha256` as the tables built later), or without `--pool-only` from prepared tables and sidecars (needs 2 and 3) | a few minutes (cache load) |
| 5 | Feature specifications | `markdowns/{merge,split}_features_for_proofreader_evolve.md` | label counts and baseline figures quoted from the 794495 companion | yes (figures) | yes (figures) | recompute with the specification's own scoring code (`agentic/canonical_stats.py card` on the companion) and edit the Baselines sections; the previous text stays in git history | minutes |
| 6 | Descriptor contexts | Read on demand from the matching fragment graph and reviewed image source | Source identity, geometry settings, image request and implementation hashes | regenerated on demand | regenerated on demand | no precompute step or saved context directory | charged to descriptor wall budget |
| 7 | Descriptor bank | `proofreader_evolve/descriptor_bank/<brain>/<kind>/<pool_sha256>/<code_sha256>/<scope>/` | pool, feature/source identities, geometry/image configuration, descriptor code and implementation hashes | recomputed on use | recomputed on use when identity changes | nothing to run; evolution reuses only matching numeric results | lazy; cold source IO can dominate |
| 8 | Canonical metrics | `metrics_out/<brain>/<segmentation_id>/` (`results.csv`, `merge_sites.csv`) | nothing automatic; reads GT and segmentation from GCS itself, so it must simply be the same GT version as the cache | yes when GT changed (move the old run aside: the comparison scripts want exactly one run directory) | no | `python -u notebooks/evaluate_skeleton_metrics.py --brains <b>` | about 30 min for 794495 (2026-10-14); GCS credentials |
| 9 | Alignment overlays and error-context figures | `figs/candidate_image_alignment_<date>/<brain>/`, `figs/error_context_*_<brain>_*.png` | produced from labeled graphs and on-demand image crops | regenerate when wanted (`plotting/plot_error_contexts.py --brain <b>`) | no | optional | minutes |

Preflight checks items 2 and 3; source loading and reviewed image access for item 6 are exercised when descriptors run; item 4 is checked by the
AutoDiscovery audit (`bind_companions`); items 1 and 8 are the ones that surprised us.

## 3. Rebuild order after a graph reload

```bash
conda activate panda            # compute node; one brain at a time; with srun always add -n1
rm -rf proofreader_evolve/feature_tables/<b> proofreader_evolve/descriptor_bank*/<b>
python -u -m labeled_cache --brains <b> --mcl 100 --rebuild-base                      # deletes both pkl files, GCS load; 58 min (794495)
python -u proofreader_evolve/prepare_feature_tables.py --brains <b> --mcl 100 --threads all --detector-prior raw   # raw pools + sidecars, minutes (raw runs, since 2026-11-06)
# frozen / no-score runs only: the same command without --detector-prior raw and with --image-workers -1 (detector image pass, about 16 h for 794495)
python -u -m proofreader_evolve.cli.export_rerank_companion --brain <b> --mcl 100 --pool-only   # minutes; without --pool-only it needs detector-native tables
python -u scripts/check_image_alignment_from_cache.py --brains <b> --mcl 100 --out figs/candidate_image_alignment_<date>
python -u scripts/check_image_alignment_from_cache.py --brains <b> --mcl 100 --out figs/candidate_image_alignment_<date> \
    --record-reviewed --review-note "<what you saw in the six overlays>"
python -m proofreader_evolve.cli.preflight --brains <development brains> --train-brains <TRAIN brains> --selection-brains all --detector-prior raw --mcl 100 --min-positives 10
rm -rf metrics_out/<b>; python -u notebooks/evaluate_skeleton_metrics.py --brains <b>   # only if GT or the segmentation changed; old runs are deleted, not archived
```

After a label-only refresh keep the pkl files: run
`python -m labeled_cache --brains <b> --mcl 100 --policy length_v2 --check-reproduce` instead (the `policy` stage, no
cloud reads apart from the GT check), then steps 2 to 4, then re-record the receipt. Descriptor inputs are read on demand; there is no context-cache rebuild.

As done on 2026-11-11 for the `length_v2` refresh of the twelve caches: relabel (minutes per brain, four groups on two
nodes); delete `proofreader_evolve/feature_tables/<b>` and `proofreader_evolve/descriptor_bank_raw/<b>`; `bash
proofreader_evolve/rebuild_brains.sh <brains>` (one call per node; 25 to 35 min for a large brain because of the split
pool); regenerate the 802449 and 794495 companions with `--pool-only`; run the alignment audit and compare the six
overlays per brain with the previously reviewed set (same seed, same nodes: pixel-identical overlays mean the earlier
review stands, `figs/candidate_image_alignment_20261111/overlay_comparison.json`) before `--record-reviewed`; the
eleven-brain preflight; `notebooks/merge_canonical_gap.py --measure` and `--plot`. Every `pool_sha256` and every split
label were unchanged; only merge labels moved.

The notebooks that read the cache's merge sites were rerun the next day, and their order matters: the site census
(`notebooks/merge_topology_diagnosis.py --mcl 100`, then `notebooks/merge_diagnosis_followup.py --mcl 100`) writes
`merge_diagnosis/mcl100/followup/bridge_sites.csv`, which `merge_shaft_candidate_reach.py`, `merge_site_gt_tip.py` and
`plot_merge_blind_sites.py` read; `merge_site_gt_tip.py` and `plot_merge_blind_sites.py` also read the reach JSONs; and
`merge_shaft_candidate_reach.py` checks its junction family against `log_merge_site_counts.py`'s summary. So: site
counts, census, follow-up, reach (the brains can be split over processes, then `--summarize`), GT tip, blind sites.
Independent of these: `label_counts.py` (then `plot_label_availability.py`), `merge_candidate_pool_sweep.py`
(about 1 h 45 min for twelve caches), `verify_add_cache_metrics.py`, `merge_canonical_gap.py`,
`plot_errors_per_neuron.py`, and `agentic/canonical_stats.py card` for the merge feature specification. A census
older than the cache makes `plot_merge_blind_sites.py` stop with "the census predates this cache".

## 4. Not invalidated automatically, but no longer comparable

- **Frozen detectors** (`autodiscovery-application/<run>/*.joblib`, selected by
  `precompute_error_scores.resolve_detector_runs`): fitted on the cache of their time; a
  rebuilt cache changes their training labels only if they are refitted. Decide explicitly.
  The 2026-08 detectors and all AutoDiscovery results were deleted on 2026-10-17 (fitted
  before the relabel, on the old pools); since 2026-10-23 the defaults point at the 2026-10
  builds (`new-merge-1_2026-10-19`, `new-split-1_2026-10-20`, run directory
  `runs/794495-with-figures`). Companions and the candidate-pool sweeps do not need a
  detector. Since 2026-11-06
  raw runs need no detector either: they load raw pools. The detector-native table directories
  (32.6 GiB of `features.pkl` across nine brains) were deleted on 2026-11-07, so `feature_tables/`
  holds only raw pools; a `frozen` / `no-score` run first re-prepares its tables (item 2 without
  `--detector-prior raw`, hours per brain) with the detector deliverables, whose run directories
  under `autodiscovery-application/` were deleted on 2026-11-06 and restored on 2026-11-08 from
  the `mindscope_daily_2026-11-05_19:00` filesystem snapshot.
- **Evolution runs** (`proofreader_evolve/runs/<run>/`): `best_scorer.py` and
  `descriptors.json` stay executable, but Precision@K on a rebuilt brain is against new labels
  and a new pool hash; the seed baseline of the next run is the reference.
- **AutoDiscovery exports** (`autodiscovery/<run>.json`): their recorded code names the
  companion files, so the audit (`run_discovery_workflow.py --pkl ... --companion ...`) re-runs
  on whatever companion matches the pkl. Row indices in hypotheses stay valid only if the pool's
  row order survived: after the 794495 reload the split rows kept their order (same segment
  pairs, node ids renumbered), the merge rows did not (NMS sorts by node id).
- **Notebook outputs**: `notebooks/merge_label_shadow/`, `notebooks/merge_canonical_gap/`,
  `notebooks/merge_diagnosis/`, `notebooks/merge_candidate_pool_sweep_outputs*/` record the
  cache identity they used (`cache_identity` in the per-brain meta), and
  `notebooks/split_fragment_length/` and `notebooks/gt_neuron_length/` record it as
  `source_cache` in the per-brain JSON (`gt_neuron_length` also reads the GT SWCs from GCS, so
  a GT edit on GCS changes it too); results for a rebuilt brain must be recomputed, not
  merged with old ones. `notebooks/errors_per_neuron/` (`plot_errors_per_neuron.py`, 2026-11-07)
  reads only item 8 and the two canonical-gap outputs and checks their segmentation ids, so it is
  rerun after `evaluate_skeleton_metrics.py` or either `*_canonical_gap.py --measure` (seconds).
- **Documents quoting numbers**: `proofreader_evolve/TODO.md` (pool counts in its data section),
  `notebooks/merge_label_shadow_findings.md`, the feature specifications (item 5).

## 5. Verification after a rebuild

Before the build of a new brain, or after its `configs/segmentation_paths.json` entry changes,
`python -m labeled_cache.source_check --brains <b> --record` confirms that the fragment SWCs and
the dense segmentation are one segmentation run (close to 100% of sampled fragment node voxels
carry the fragment's own segment id). After the build:

1. `preflight` passes for every brain (tables and sidecars).
2. GT matches GCS: `python -m labeled_cache --brains <brain> --mcl 100` compares a reused cache's GT
   with the current GCS tracings (same resampling) before any stage runs and rebuilds a stale cache;
   a current cache reports no changed or added neurons.
3. On-demand contexts: probe a few TRAIN candidates with the chosen geometry/image descriptor
   on a compute node; verify source access, alignment and finite outputs before a full call.
4. Pool identity: compare the new companion with the previous one (same segment pairs or
   junction coordinates; which labels changed) as done for 794495 in
   `cache/ground_truth_inventory_20261013.md` section 8.
5. Canonical agreement: `notebooks/merge_canonical_gap.py --measure --brains <b>` then
   `--plot`; cache precision against canonical near 90% means the two sides use the same GT.
   For splits, `notebooks/split_canonical_gap.py --measure --brains <b>` (labels the raw GT with
   the segmentation itself, about 5 to 10 min per brain): cache labels should hold about 97%
   of the canonical pairs whose segments both exist in the fragments graph.

## 6. Lessons recorded this week

- The base cache is reused silently; a GT update needs both pkl files removed.
- The alignment receipt binds the cache file identity, so every cache rebuild needs the
  review redone before any image read.
- A graph reload renumbers node ids and therefore pool hashes, even with identical fragments.
- S3 image volumes can be deleted: 794491's 2025-10-26 volume was, and its cached `img_path`
  failed with `NOT_FOUND ... .zarray` (2026-10-16). Point the prefix and the alignment review at
  the surviving volume, then run `--refresh-image-path`; `BrainDataset.load_from_cache` opens
  `img_path` eagerly, so the base cache is unusable until then. The rewrite changes the file
  identity, so the alignment receipt is re-recorded afterwards.
  On 2026-11-08 AIND deleted the raw `fused.zarr` of 754613 and 751473 (levels and root metadata
  removed over the day); both prefixes keep a `fused-denoised.zarr` with the same shape, spacing and
  chunking, and the only other fusions of these brains are separate runs on a grid about eight times coarser. The prefixes, the cache
  `img_path` (`--refresh-image-path`, 522 s for both brains), the raw pools and sidecars (new cache
  identity) and the receipts (`figs/candidate_image_alignment_20261108`) were redone the same day;
  On 2026-11-09 the raw `fused.zarr` of 747807, 750318 and 754610 were found deleted as well (read
  the evening before) and the same steps were repeated for them (`--refresh-image-path` 911 s for the
  three brains, receipts `figs/candidate_image_alignment_20261109`); before the switch their denoised
  volumes were checked against the fragments and, on the same voxel windows, against the dense
  segmentation. The same evening 754612 and 794493, whose raw volumes were still present, were switched
  to their `fused-denoised.zarr` after the same checks (raw and denoised patches coincide at zero shift),
  so that a later deletion cannot interrupt a run. These seven brains, 841260 and the reserved 794495
  (whose `fused.zarr` AIND denoised at the tile level before fusion) read denoised pixels; 802449, 789202
  and 794491 read unprocessed fusions and have no denoised volume on the same grid
  (`cache/dataset_source_status_20261108.md`).
- `prepare_feature_tables.py` exits 1 when its final preflight finds no sidecars; the tables
  are complete nonetheless.
- Inside the user's Slurm job use `srun --jobid=<job> --overlap -n1 ...`; without `-n1` the
  command runs once per job task and large steps are OOM-killed.

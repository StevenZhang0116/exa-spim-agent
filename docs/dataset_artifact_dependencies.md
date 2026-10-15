# What depends on a dataset cache, and what to rerun when it changes

Written 2026-10-14 after the 794495 rebuild. Every derived artifact of a brain descends from
its labeled dataset cache `cache/dataset_cache_<brain>_mcl<N>_add.pkl`. This page lists those
artifacts, what each one is bound to (the check that fails when the cache changes), the order
in which to rebuild them, and the things that are *not* invalidated automatically but stop
being comparable. Times are what the 794495 rebuild of 2026-10-13 took on n268 (16 CPUs).

## 1. Upstream of the cache

| Input | Where it is defined | Used by |
|---|---|---|
| GT tracings (one SWC per neuron, `voxel/` copy) | `gs://allen-nd-goog/ground_truth_tracings/<brain>/voxel`, hard-coded in `notebooks/generate_add_caches.py` | `gt_graph` and every GT-derived label in the cache |
| Predicted segmentation (`label_mask`) and fragment SWCs | `configs/segmentation_paths.json` (new layout, takes precedence) or `configs/segmentation_datasets.rtf` + legacy template, resolved by `scripts/dataset_config.py` | `fragments_graph`; `label_gt_from_segmentation` |
| Fused image prefix | `configs/exaspim_image_prefixes.json` | `img_path` in the cache; image reads of the context cache |

`generate_add_caches.py` builds `dataset_cache_<brain>_mcl<N>.pkl` (graphs loaded) and then
`_add.pkl` (labels baked in). **It reuses an existing base cache without checking GCS**, so
GT updates are picked up only if both pkl files are deleted first. Check
`gs://.../voxel` object `updated` times against the base-cache mtime before trusting a cache
(`../claude_scratch/gt_gcs_inventory_20261013.json` holds the 2026-10-13 snapshot;
`cache/ground_truth_inventory_20261013.md` has the full account).

Two kinds of change have different blast radii:

- **Graph reload** (new or revised GT, new segmentation, rebuilt base cache): node ids are
  renumbered because `SkeletonGraph.load` assigns component ids in load order, even when the
  fragments are unchanged. Everything keyed by node ids or by the pool hash changes.
- **Label-only refresh** (`scripts/relabel_cache.py --refresh-merge-sites`, or a future
  length-based merge threshold): graphs and node ids stay, only `gt_*` label arrays and
  `gt_merge_sites` change. Candidate pools and folds stay identical; labels, sidecars,
  companions and anything carrying labels must still be redone.

## 2. Derived artifacts

| # | Artifact | Path | Bound to (the check that fails) | Rerun after graph reload | Rerun after label-only refresh | Command | Time |
|---|---|---|---|---|---|---|---|
| 1 | Image-alignment receipt | `configs/image_alignment.json` (`brains.<brain>`) | cache identity (path, size, mtime, inode) and image identity; `image_context.source` raises `Image/cache identity changed since alignment review` | yes | yes (the pkl file changes) | `python -m proofreader_evolve.cli.check_image_alignment --brains <b> --mcl <N> --out figs/candidate_image_alignment_<date>`, inspect the six overlays, then the same command with `--record-reviewed --review-note "..."` | minutes plus the visual review |
| 2 | Native feature tables | `proofreader_evolve/feature_tables/<brain>/<pool_sha256>/` (`candidates.pkl`, `features.pkl`, `evaluator_labels.npy`, `meta.json`) | `meta.provenance.source_cache` (cache identity), `pool_sha256` (candidate identities), detector run ids | yes (new pool hash) | yes (labels inside) | `python -u proofreader_evolve/prepare_feature_tables.py --brains <b> --mcl <N> --threads all` | 2 h 52 min for 794495; the script's final preflight reports `blocked` and exit code 1 until step 3 exists, with complete tables |
| 3 | GT-availability sidecars | `feature_tables/<brain>/<pool_sha256>.availability/` | table rows, `pool_sha256`, candidate-key digest, labels digest, `source_cache` (`AvailabilityMismatch`) | yes | yes | `python -u -m proofreader_evolve.cli.precompute_availability --brains <b> --mcl <N>` | about 1.5 min |
| 4 | AutoDiscovery companions | `markdowns/rerank_companion_<brain>_{merge,split}.{npz,json}` (gitignored) | `pool_sha256`, `labels_sha256`, `npz_sha256`, `source_cache.file` + size; `agentic/rerun_experiments.bind_companions` refuses a mismatch | yes | yes | `python -u -m proofreader_evolve.cli.export_rerank_companion --brain <b> --mcl <N>` (needs 2 and 3; loads the graph for split folds) | about 1 min |
| 5 | Feature specifications | `markdowns/{merge,split}_features_for_proofreader_evolve.md` | label counts and baseline figures quoted from the 794495 companion | yes (figures) | yes (figures) | recompute with the specification's own scoring code (`../claude_scratch/merge_baselines_20261013.py` for merge) and edit the Baselines sections; archive the previous text under `markdowns/archive/` | minutes |
| 6 | Context cache | `proofreader_evolve/context_cache/<brain>/<kind>/<pool_sha256>/` | `table_identity`: brain, kind, `pool_sha256`, `features_sha256`, `source_cache`; `ContextCache.entry` raises on mismatch; a changed pool hash is simply a new directory | yes | probably: `features.pkl` is rewritten by step 2, check whether `features_sha256` changed | `python -u -m proofreader_evolve.cli.precompute_context_cache --brains <b> --mcl <N> --readers 16 --band-rows 50000 --labeled-extension` (needs 1 and 3) | 2 h 7 min for 794495 (fresh 50,000-row band per kind plus the labeled extension) |
| 7 | Descriptor bank | `proofreader_evolve/descriptor_bank/<brain>/<kind>/<pool_sha256>/<code_sha256>/<scope>/` | directory keyed by pool hash; manifest identity includes `ContextCache.identity_key` | orphaned; delete | recomputed on first use if the context-cache identity changed | nothing to run; the next evolution run recomputes what it needs (about 5 to 10 min per descriptor and brain) | lazy |
| 8 | Canonical metrics | `metrics_out/<brain>/<segmentation_id>/` (`results.csv`, `merge_sites.csv`) | nothing automatic; reads GT and segmentation from GCS itself, so it must simply be the same GT version as the cache | yes when GT changed (move the old run aside: the comparison scripts want exactly one run directory) | no | `python -u notebooks/evaluate_skeleton_metrics.py --brains <b>` | about 30 min for 794495 (2026-10-14); GCS credentials |
| 9 | Alignment overlays and error-context figures | `figs/candidate_image_alignment_<date>/<brain>/`, `figs/error_context_*_<brain>_*.png` | produced from the cache and the context cache | regenerate when wanted (`plotting/plot_error_contexts.py --brain <b>`) | no | optional | minutes |

Items 2, 3 and 6 are what `preflight` and the evolution driver check; item 4 is checked by the
AutoDiscovery audit (`bind_companions`); items 1 and 8 are the ones that surprised us.

## 3. Rebuild order after a graph reload

```bash
conda activate panda            # compute node; one brain at a time; with srun always add -n1
rm cache/dataset_cache_<b>_mcl100.pkl cache/dataset_cache_<b>_mcl100_add.pkl          # force a fresh GT load
rm -rf proofreader_evolve/feature_tables/<b> proofreader_evolve/context_cache/<b> proofreader_evolve/descriptor_bank/<b>
python -u notebooks/generate_add_caches.py --brains <b>                                 # 58 min (794495)
python -u proofreader_evolve/prepare_feature_tables.py --brains <b> --mcl 100 --threads all   # 2 h 52 min
python -u -m proofreader_evolve.cli.precompute_availability --brains <b> --mcl 100     # 1.5 min
python -u -m proofreader_evolve.cli.export_rerank_companion --brain <b> --mcl 100     # 1 min
python -m proofreader_evolve.cli.check_image_alignment --brains <b> --mcl 100 --out figs/candidate_image_alignment_<date>
python -m proofreader_evolve.cli.check_image_alignment --brains <b> --mcl 100 --out figs/candidate_image_alignment_<date> \
    --record-reviewed --review-note "<what you saw in the six overlays>"
python -u -m proofreader_evolve.cli.precompute_context_cache --brains <b> --mcl 100 --readers 16 --band-rows 50000 --labeled-extension   # 2 h
python -m proofreader_evolve.cli.preflight --brains <all brains> --train-brains 794495 802449 --mcl 100 --context-cache proofreader_evolve/context_cache
mv metrics_out/<b> metrics_out/archive/<b>_<olddate>; python -u notebooks/evaluate_skeleton_metrics.py --brains <b>   # only if GT changed
```

After a label-only refresh skip the first two lines and `generate_add_caches.py`; run
`relabel_cache.py` instead, then steps 2 to 4, then re-record the receipt, then check the
context-cache identity before rebuilding it.

## 4. Not invalidated automatically, but no longer comparable

- **Frozen detectors** (`autodiscovery-application/<run>/*.joblib`, selected by
  `precompute_error_scores.resolve_detector_runs`): fitted on the 794495 cache of their time;
  a rebuilt cache changes their training labels only if they are refitted. Decide explicitly.
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
  cache identity they used (`cache_identity` in the per-brain meta); results for a rebuilt brain
  must be recomputed, not merged with old ones.
- **Documents quoting numbers**: `proofreader_evolve/TODO.md` coverage tables,
  `notebooks/merge_label_shadow_findings.md`, the feature specifications (item 5).

## 5. Verification after a rebuild

1. `preflight --context-cache` passes for every brain (tables, sidecars, context cache).
2. GT matches GCS: `../claude_scratch/compare_gt_794495_20261013.py <brain>` loads the cache
   and the current GCS tracings with the same resampling and prints per-neuron differences
   (0 changed, 0 added expected).
3. Context cache: `../claude_scratch/verify_labeled_extension_20261013.py <brain>` (extension
   equals labeled rows minus band, 100% of labeled rows cached, sampled contexts load).
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
- `prepare_feature_tables.py` exits 1 when its final preflight finds no sidecars; the tables
  are complete nonetheless.
- Inside the user's Slurm job use `srun --jobid=<job> --overlap -n1 ...`; without `-n1` the
  command runs once per job task and large steps are OOM-killed.

# Labeled caches: data preparation

*Part of the [exa-spim-agent](../README.md) documentation.*

The caches consumed by every stage are built once per brain and minimum cable
length (MCL) and checked against the official
[`segmentation-skeleton-metrics`](https://github.com/AllenNeuralDynamics/segmentation-skeleton-metrics)
results:

```text
same source brain + segmentation
        |
        +--> python -m labeled_cache: base .pkl --> labels --> policy --> _add.pkl -+
        |                                                                         |
        +--> evaluate_skeleton_metrics.py ----> official results.csv ------------+
                                                                                  |
                                                                                  v
                                                             verify_add_cache_metrics.py
```

Run the pipeline in the `panda` conda environment. Cloud-reading steps require
GCS credentials in `configs/allen-nd-goog-f5d46dbfa2cd.json`.
Run every Python command on a compute node. Use explicit development-brain lists:
794495 is reserved for a future independent validation process, and legacy
all-cache defaults do not enforce that reservation. See the
[current split](stage3_evolution.md#32-prepare-tables-and-launch-a-run).

## A1. Generate the base cache

Run [`notebooks/load_skeletons.py`](../notebooks/load_skeletons.py), selecting the
brain and minimum cable length with command-line arguments, for example:

```bash
python notebooks/load_skeletons.py --brain-id 802449 --min-cable-length 10
```

It reads the GT and UNet-fragment SWCs, constructs two `SkeletonGraph` objects,
and writes:

```text
cache/dataset_cache_<brain>_mcl<N>.pkl
```

This base cache contains skeletons but no baked-in error labels.

## A2. Generate the official reference metrics

Run
[`notebooks/evaluate_skeleton_metrics.py`](../notebooks/evaluate_skeleton_metrics.py)
for the same brain (panda env, compute node):

```bash
python -u notebooks/evaluate_skeleton_metrics.py --brains 802449
```

It runs the official `segmentation-skeleton-metrics` package and writes:

```text
metrics_out/<brain>/<segmentation_id>/results.csv
metrics_out/<brain>/<segmentation_id>/merge_sites.csv
```

This step is independent of the cache. It produces the canonical answers used
to validate the labeled cache later.

## A3. Generate the labeled `_add.pkl`

Run from the repository root (panda, compute node):

```bash
python -m labeled_cache --brains 802449 --mcl 100 --policy length_v2
```

One command covers every state: no cache yet (skeletons from GCS, then labels), a
base cache without labels (labels only), or a labeled cache whose merge-threshold
policy differs from the request (relabeled in place from its stored GT node labels).
A reused cache has its GT compared with the current GCS tracings first. Several
brains can be listed; they are processed one after the other. `--dry-run` prints
the stages, `--help` the options.

The builder reads the dense segmentation, labels GT nodes and edges, identifies
merge labels and sites, and writes the labeled file next to the base cache:

```text
cache/dataset_cache_<brain>_mcl<N>_add.pkl
```

The added fields are:

- `gt_node_canonical_label`
- `gt_edge_error`
- `gt_merge_labels`
- `gt_merge_sites`
- `gt_label_policy` (the merge-threshold policy the labels were computed with)

See [`markdowns/labeled_dataset_cache.md`](../markdowns/labeled_dataset_cache.md)
for the complete schema.

### Check that the two sources are one segmentation run

The base stage loads the fragment SWCs (`fragments_path`) and the label stage reads the
dense segmentation (`segmentation_path`); nothing in the build compares them, and a wrong
pair (a re-exported segmentation with older skeletons, a mistyped folder in
`configs/segmentation_paths.json`) would only surface later as near-zero GT label
coverage. Before building a new brain, or after either entry changes, run (compute node,
seconds, GCS credentials):

```bash
python -m labeled_cache.source_check --brains 751473 --record
```

On 2026-11-05 the sources of 789202 and 802449 were replaced by the denoised Google
segmentations and 841260 was added, all as explicit entries in `configs/segmentation_paths.json`;
their receipts are the 2026-11-05 section of `docs/fragment_segmentation_match.md`. Their caches
were rebuilt on 2026-11-06 (n252; `--rebuild-base` for the two replaced brains, 3 h for the pair,
3.5 h for 841260; logs `notebooks/log/labeled_cache_20261106_*.log`). The GT graphs are unchanged;
the denoised segmentations carry far fewer merges and more omissions than the sources they
replace: merged GT edges 802449 36.7% -> 2.3% and 789202 17.7% -> 0.3%, omitted 2.8% -> 17.3% and
3.2% -> 11.9%, split 0.6% -> 0.7% and 0.5% -> 3.5%; merge sites 1143 -> 121 and 84 -> 17; fragment
nodes 36.9 M -> 12.2 M and 4.49 M -> 2.81 M. 841260: 13 GT neurons, split 2.6%, omit 9.5%, merged
3.4%, 136 merge sites. Merge positives therefore shrink on both replaced brains, and earlier merge
baselines on them are not comparable.

It samples interior fragment nodes, converts them to voxels the way the builder does,
reads the segmentation at those voxels and reports how many carry the fragment's own id.
Matched sources give close to 100% (exact or within one voxel; the eleven current brains gave
97.9-100%, the older-microscope brains at the low end because more of their resampled nodes
sit one voxel off the centre line); an unrelated segmentation gives close to 0%, and the exit
code is 1 below `--min-agreement` (95%). `--record` appends the table to
[`docs/fragment_segmentation_match.md`](fragment_segmentation_match.md), where the eleven
current brains were recorded on 2026-10-29.

## A4. Verify the labeled cache

Run [notebooks/verify_add_cache_metrics.py](../notebooks/verify_add_cache_metrics.py)
from the repository root on a compute node. Its legacy default scans every brain
with an `_add.pkl` at the requested MCL; pass `--brain` explicitly to keep reserved
794495 out of routine development checks:

```bash
srun --partition=aibs_debug --constraint=cpu \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=100G --time=01:00:00 \
  --chdir="$PWD" "$HOME/.conda/envs/panda/bin/python" -u \
  notebooks/verify_add_cache_metrics.py --mcl 100 --brain 802449
```

Use `--mcl 10` for mcl10, or `--brain 802449 794493` to select several development brains.
`--cache-dir`, `--metrics-dir` and `--output-dir` override the repository defaults.
Memory and time requests may need increasing for larger caches or more brains.
Brains missing a canonical `results.csv` are reported as skipped without loading
their caches. Multiple reference runs for one brain are rejected rather than
silently choosing one. Other brains continue after a failure; errors and hard
consistency failures produce exit code 1, while metric warnings alone do not.

The script reconstructs per-neuron split, omit, merge, and edge-accuracy
metrics using only the stored labels, then compares them with the official
`results.csv` from step A2. It opens no image readers, requires no cloud credentials,
and never modifies the caches or reference results. It overwrites matching
`<brain>_mcl<N>_per_neuron.csv`, `_summary.csv`, and `_scatter.png` files under:

```text
notebooks/verify_stats/
```

Canonical `# Merges` comparisons use geometric-walk sites only. Supplemental
two-GT-only, shared-evidence and combined site counts are exported separately;
combined counts are not counts of independent biological merge events. Missing
site data remains unavailable, not zero. Existing columns remain compatible with
[notebooks/compare_add_cache_metrics_across_datasets.py](../notebooks/compare_add_cache_metrics_across_datasets.py).

Expect close agreement, not byte-for-byte equality. Small differences remain
because the cached graphs are resampled and the merge-site implementations have
minor snapping and deduplication differences. For a detailed merge-site check,
run [`notebooks/verify_add_cache_merge_info.py`](../notebooks/verify_add_cache_merge_info.py).

## A5. Compare brains and MCLs

Once verification CSVs are available, run the lightweight comparison script on
a compute node. It does not load skeleton caches, access cloud data, or rewrite
verification CSVs:

```bash
srun --partition=aibs_debug --constraint=cpu \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=3G --time=00:05:00 \
  --chdir="$PWD" "$HOME/.conda/envs/panda/bin/python" -u \
  notebooks/compare_add_cache_metrics_across_datasets.py
```

All generated figures are saved in `notebooks/verify_stats/`:

- `compare_mcl100_across_brains.png`
- `compare_mcl10_across_brains.png`
- `compare_<brain>_mcl100_vs_mcl10.png` for each brain with both MCLs

Missing MCLs are reported and skipped. Use `--stats-dir` for another input
directory; figures go there by default, or to `--output-dir` if provided.
`--dpi` defaults to 160. Re-running overwrites matching comparison PNGs but
leaves the verifier's `<brain>_mcl<N>_scatter.png` files and CSVs untouched.
Legacy tables remain supported via their MCL column or matching summary; MCL-named
tables take precedence over duplicate legacy rows. The plots reflect the saved
CSVs, so rerun verification first if the underlying caches have changed.

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

## A1. Generate the base cache

Run [`notebooks/load_skeletons.py`](../notebooks/load_skeletons.py), selecting the
brain and minimum cable length with command-line arguments, for example:

```bash
python notebooks/load_skeletons.py --brain-id 794495 --min-cable-length 10
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
python -u notebooks/evaluate_skeleton_metrics.py --brains 794495
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
python -m labeled_cache --brains 794495 --mcl 100 --policy length_v1
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

## A4. Verify the labeled cache

Run [notebooks/verify_add_cache_metrics.py](../notebooks/verify_add_cache_metrics.py)
from the repository root on a compute node. By default it processes every brain
with an `_add.pkl` at the requested MCL, one cache at a time:

```bash
srun --partition=aibs_debug --constraint=cpu \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=100G --time=01:00:00 \
  --chdir="$PWD" "$HOME/.conda/envs/panda/bin/python" -u \
  notebooks/verify_add_cache_metrics.py --mcl 100
```

Use `--mcl 10` for mcl10, or add `--brain 794495 794493` to select brains.
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

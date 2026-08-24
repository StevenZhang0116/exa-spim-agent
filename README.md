# exa-spim-agent

This repository builds labeled ExaSPIM skeleton caches and checks them against
the official
[`segmentation-skeleton-metrics`](https://github.com/AllenNeuralDynamics/segmentation-skeleton-metrics)
results.

## Pipeline

```text
same source brain + segmentation
        |
        +--> load_skeletons.py --> base .pkl --> relabel_cache.py --> _add.pkl -----+
        |                                                                         |
        +--> evaluate_skeleton_metrics.ipynb --> official results.csv ------------+
                                                                                  |
                                                                                  v
                                                             verify_add_cache_metrics.ipynb
```

Run the pipeline in the `panda` conda environment. Cloud-reading steps require
GCS credentials in `configs/zihan_gcs_token.json`.

### 1. Generate the base cache

Run [`notebooks/load_skeletons.py`](notebooks/load_skeletons.py), selecting the
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

### 2. Generate the official reference metrics

Open and run
[`notebooks/evaluate_skeleton_metrics.ipynb`](notebooks/evaluate_skeleton_metrics.ipynb)
for the same brain and segmentation.

It runs the official `segmentation-skeleton-metrics` package and writes:

```text
metrics_out/<brain>/<segmentation_id>/results.csv
metrics_out/<brain>/<segmentation_id>/merge_sites.csv
```

This step is independent of the cache. It produces the canonical answers used
to validate the labeled cache later.

### 3. Generate the labeled `_add.pkl`

Run from the repository root:

```bash
python scripts/relabel_cache.py \
  --cache-dir cache \
  --brain 794495 \
  --mcl 100
```

To label every matching base cache, omit `--brain` and `--mcl`:

```bash
python scripts/relabel_cache.py --cache-dir cache
```

The script reads the dense segmentation, labels GT nodes and edges, identifies
merge labels and sites, and writes a new file without modifying the base cache:

```text
cache/dataset_cache_<brain>_mcl<N>_add.pkl
```

The added fields are:

- `gt_node_canonical_label`
- `gt_edge_error`
- `gt_merge_labels`
- `gt_merge_sites`

See [`markdowns/labeled_dataset_cache.md`](markdowns/labeled_dataset_cache.md)
for the complete schema.

### 4. Verify the labeled cache

Open and run
[`notebooks/verify_add_cache_metrics.ipynb`](notebooks/verify_add_cache_metrics.ipynb).
Set `BRAIN_ID` and `MINLEN` to match the `_add.pkl` generated in step 3.

The notebook reconstructs per-neuron split, omit, merge, and edge-accuracy
metrics using only the stored labels, then compares them with the official
`results.csv` from step 2. It writes comparison tables and plots under:

```text
notebooks/verify_stats/
```

Expect close agreement, not byte-for-byte equality. Small differences remain
because the cached graphs are resampled and the merge-site implementations have
minor snapping and deduplication differences. For a detailed merge-site check,
run [`notebooks/verify_add_cache_merge_info.py`](notebooks/verify_add_cache_merge_info.py).

## Main outputs

```text
cache/                 base and labeled pickle files
metrics_out/           official reference metrics
notebooks/verify_stats cache-versus-reference comparisons
```

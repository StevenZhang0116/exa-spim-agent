# Prerequisites: environment and labeled caches

*Part of the [exa-spim-agent](../README.md) documentation.*

## Environment

Every Python command in this repository runs in the `panda` conda environment,
which provides the `agentic_neuron_proofreader` package that defines
`SkeletonGraph` (source: [AllenInstitute/neuron-proofreader](https://github.com/AllenInstitute/neuron-proofreader)):

```bash
conda activate panda
```

Loading an `_add.pkl` cache needs more than 20 GB of RAM, so every step that
touches a dataset runs on a compute node:

```bash
#!/bin/bash
#SBATCH --mem=80G
source /shared/utils.x86_64/anaconda3-2024.10/etc/profile.d/conda.sh
conda activate panda
cd /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent
python your_script.py
```

Agent steps need `ANTHROPIC_API_KEY`. Steps that read raw image patches from
public S3 need `AWS_EC2_METADATA_DISABLED=true`; building caches needs GCS
credentials at `configs/allen-nd-goog-f5d46dbfa2cd.json`. The caches themselves load fully
offline.

## Data: labeled caches

Every stage reads `cache/dataset_cache_<brain>_mcl<N>_add.pkl`, a pickle holding
the UNet fragment reconstruction (`fragments_graph`), the human ground-truth
tracings (`gt_graph`) and baked-in error labels (`gt_node_canonical_label`,
`gt_edge_error`, `gt_merge_labels`, `gt_merge_sites`).
[The labeled-cache guide](labeled_caches.md) explains how the
caches are built and verified;
[`markdowns/labeled_dataset_cache.md`](../markdowns/labeled_dataset_cache.md) holds
the schema. Quick load:

```python
import pickle
import agentic_neuron_proofreader  # noqa — registers SkeletonGraph

with open("cache/dataset_cache_794495_mcl100_add.pkl", "rb") as f:
    payload = pickle.load(f)
frag, gt = payload["fragments_graph"], payload["gt_graph"]
```

The GT graphs cover tens of neurons per brain, so GT-derived labels are sparse
annotations rather than exhaustive truth: label 0 means "no recorded error
here", not a verified clean location. Every downstream metric inherits this.
`proofreader_evolve` handles it with availability sidecars: rows GT cannot judge get
label NaN and are excluded from ranking and fitting (three-valued labels, Stage 3).

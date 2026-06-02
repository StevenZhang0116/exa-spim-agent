# exa-spim-agent

This code is based on [AllenInstitute/agentic-neuron-proofreader](https://github.com/AllenInstitute/agentic-neuron-proofreader) and [AllenNeuralDynamics/segmentation-skeleton-metrics](https://github.com/AllenNeuralDynamics/segmentation-skeleton-metrics).

## Project Structure

```
exa-spim-agent/
├── notebooks/                     # Jupyter notebooks (run in order)
│   ├── load_skeletons.ipynb       # 1. Load SWCs from GCS, build graphs, save cache
│   ├── evaluate_skeleton_metrics.ipynb  # 2. Run full skeleton metrics evaluation
│   ├── load_skeletons_from_cache.ipynb  # 3. Reload cache, visualize, compute metrics
│   └── run_neuron_proofreader.ipynb     # 4. Run split correction pipeline
├── scripts/                       # Standalone Python scripts
│   └── visualize_napari.py        # Interactive 3D viewer with Napari
├── configs/                       # Configuration files
│   └── exaspim_image_prefixes.json
├── figs/                          # Generated figures (gitignored)
├── metrics_out/                   # Evaluation outputs (gitignored)
├── dataset_cache_*.pkl            # Cached BrainDataset (gitignored)
└── zihan_gcs_token.json           # GCS credentials (gitignored)
```

## Notebooks (run in order)

1. **`notebooks/load_skeletons.ipynb`** — Reads GT and UNet fragment SWC files from GCS, builds graph structures, and saves a `dataset_cache_*.pkl` for fast reloading. Run this first (~15 min).

2. **`notebooks/evaluate_skeleton_metrics.ipynb`** — Runs the full `segmentation-skeleton-metrics` evaluation pipeline on the whole brain. Outputs `metrics_out/` with per-skeleton CSV results. Run after step 1.

3. **`notebooks/load_skeletons_from_cache.ipynb`** — Reloads the cached dataset, visualizes patches (image, segmentation, skeletons), computes patch-local skeleton metrics, and displays whole-brain metrics for skeletons in the patch. Run after steps 1 and 2.

4. **`notebooks/run_neuron_proofreader.ipynb`** — Demonstrates the full split correction pipeline: proposal generation, feature extraction, GNN inference, and progressive merging.

## Scripts

- **`scripts/visualize_napari.py`** — Interactive 3D visualization using Napari. Displays raw fluorescence, UNet segmentation, GT skeletons, and fragment skeletons as separate toggleable layers.

  ```bash
  cd scripts
  python visualize_napari.py --patch-size 256
  python visualize_napari.py --patch-size 512 --use-fragments
  ```

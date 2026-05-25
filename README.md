# exa-spim-agent

This code is based on [AllenInstitute/agentic-neuron-proofreader](https://github.com/AllenInstitute/agentic-neuron-proofreader) and [AllenNeuralDynamics/segmentation-skeleton-metrics](https://github.com/AllenNeuralDynamics/segmentation-skeleton-metrics).

## Notebooks (run in order)

1. **`load_skeletons.ipynb`** — Reads GT and UNet fragment SWC files from GCS, builds graph structures, and saves a `dataset_cache_*.pkl` for fast reloading. Run this first (~15 min).

2. **`evaluate_skeleton_metrics.ipynb`** — Runs the full `segmentation-skeleton-metrics` evaluation pipeline on the whole brain. Outputs `metrics_out/` with per-skeleton CSV results. Run after step 1.

3. **`load_skeletons_from_cache.ipynb`** — Reloads the cached dataset, visualizes patches (image, segmentation, skeletons), computes patch-local skeleton metrics, and displays whole-brain metrics for skeletons in the patch. Run after steps 1 and 2.

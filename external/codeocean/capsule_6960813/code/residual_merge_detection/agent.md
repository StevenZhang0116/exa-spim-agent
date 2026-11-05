# Residual merge detection

Scores merge-site predictions made on the **final, error-corrected SWCs** (pipeline
step 4) against the merge sites the pipeline evaluation found in those same SWCs.
"Residual" = merges that survive automated error correction. Output is recall and
false-positive count versus prediction threshold, per segmentation variant.

## Inputs

| What | Where | Notes |
|---|---|---|
| Predictions | `/root/capsule/results/site_predictions_<cfg>.csv` | One row per branch node of the final SWCs, brain-wide. Columns `node, xyz, Segment_ID, Prediction, Degree`; `xyz` is `"(x, y, z)"` in world µm. `<cfg>` is `it2`, `dt5k`, `dt20k`, `dt50k`. |
| Evaluation | `/root/capsule/data/arborist_v2_agg08_eval/arborist_v2_eval_<run>/<result dir>/step4_learned_split_correction/` | `it2` maps to run `baseline`; the others to `arborist_v2_eval_<cfg>`. Uses `*_merge_sites.csv` (ground-truth merge sites, `World` column), `ignored_merges.swc` (reviewed false positives, world µm), and the per-neuron zips in `*_fragments/N*.zip`, whose `N0xx-794495-XX.0.swc` is the ground-truth neuron (node 0 is the soma; columns 2:5 are x, y, z in µm despite the header). |

## Run

```bash
python3 code/residual_merge_detection/evaluate.py      # -> results/residual_merge_detection/
python3 code/residual_merge_detection/make_fig.py      # recall_fp_vs_threshold.png
python3 code/residual_merge_detection/make_slide.py    # slide.html (1280x720, screenshot into PowerPoint)
```

`evaluate.py` finds every `site_predictions_*.csv` in `--predictions-dir`; pass
`--configs it2 dt20k` to restrict. Distances are flags with these defaults:

| Flag | Default | Meaning |
|---|---|---|
| `--gt-filter-um` | 5 | keep predictions within this distance of a ground-truth skeleton node |
| `--ignore-um` | 30 | drop predictions within this distance of a reviewed false-positive site |
| `--site-radius-um` | 10 | a ground-truth merge site is "reachable" if a kept prediction lies within this distance (the detector only scores branch points). Reachable sites define true positives and the headline `recall_reachable`; the `recall` column uses all sites |
| `--match-um` | 10 | a positive prediction within this distance of a reachable site is a true positive; a site with a positive within it is recalled |
| `--soma-um` | 1000 | extra CSV columns only: false positives beyond this straight-line distance from the soma (node 0 of the nearest GT neuron's SWC). Not shown on the slide, table or PNG; Anna asked for it to be dropped from the headline outputs |

`--mark-threshold` (default 0.1) picks the threshold that the slide and PNG mark with a
dashed vertical line, with each run's recall and FP-per-neuron value labelled beside it.

## Outputs (`results/residual_merge_detection/`)

- `pr_curves.csv`: per config and threshold (0.00 to 1.00 by 0.01): positives, TP, FP (total and per neuron), FP far from soma (total and per neuron), precision, recall (all sites) and recall_reachable, site counts.
- `table.md`: FP per neuron and reachable recall at 0.1 steps, ready to paste.
- `gt_merge_sites_<cfg>.csv`: the evaluation's merge sites with distance to the nearest kept prediction and the `kept` flag.
- `predictions_labeled.csv`: kept predictions with `is_tp`, distance to GT skeleton, GT neuron, soma distance, nearest kept merge site.
- `summary.json`, `slide_data.json`, `recall_fp_vs_threshold.png`, `slide.html`.

## Conventions and caveats

- **Headline numbers (table, slide, PNG):** recall is *reachable recall*, over the merge sites that have a branch point within `--site-radius-um` (about 30 of the 75 to 80 per run), and false positives are the total divided by the number of ground-truth neurons (22). Anna asked for exactly this combination; `pr_curves.csv` also carries the raw FP totals and `recall` over all sites.
- Recall over all sites tops out near 0.4 even at threshold 0, since the other sites have no branch point for the detector to score.
- Precision is near 1% at threshold 0 because ~4,000 branch points sit on the GT-matched fragments and only ~30 are merges. Low precision here is expected and is not by itself a filtering bug; the filtering was checked by confirming predictions lie on the evaluated fragments (within ~1.3 µm after a constant one-voxel offset) and that their branches stay on the GT.
- The eval's fragment SWC coordinates are offset ~1 voxel (0.6, 0.6, 1.0 µm) from the prediction coordinates; irrelevant at these radii.
- Ground truth only covers 22 neurons, so a merge with an untraced neuron is scored as a false positive. Treat FP as an upper bound.
- Slide style follows the agg08 slides: light theme only, run colors it2 `#2a78d6`, dt5k `#1baf7a`, dt20k `#eda100`, dt50k `#eb6834`, one terse footnote, no deltas.

## To add a new variant

Drop `site_predictions_<cfg>.csv` into `results/` and make sure
`data/arborist_v2_agg08_eval/arborist_v2_eval_<cfg>/.../step4_learned_split_correction/` exists
(or add the mapping to `RUN_DIR` in `evaluate.py`). Rerun the three commands.

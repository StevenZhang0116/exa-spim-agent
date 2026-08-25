# Split candidate-pool sweep artifacts

This directory is one configuration-scoped, cumulatively checkpointed analysis bundle. Its configuration
signature is `5f92a29e40d94a90`. Candidate recall is measured against reachable
truth pairs before feature scoring.

## Dataset summary

```text
 brain  n_reachable_truth_pairs  best_candidate_recall  fewest_candidates_at_best_recall
789202                     2903               0.967620                             86575
794491                     1827               0.996716                           2667603
794493                     1774               0.997182                           1483054
794495                     2236               0.978533                           2042057
802449                     6853               0.980155                           3563644
```

## Current detector baseline

The baseline is `tip_to_tip`, 30 µm, and unlimited per-anchor partners. All
surviving segment pairs are retained; the sweep has no global candidate cap.

```text
 brain  n_candidate_pairs  n_candidate_truth_pairs  candidate_recall  direct_candidate_recall  gap_candidate_recall
789202              16622                     1982          0.682742                 0.799590              0.567657
794491             313428                      903          0.494253                 0.476715              0.495968
794493             112457                      861          0.485344                 0.475143              0.465185
794495             154989                     1421          0.635510                 0.782403              0.504418
802449             267290                     4113          0.600175                 0.707974              0.464851
```

## Recommended minimal pools

Selection minimizes total candidate count subject to a worst-dataset recall target.

```text
 target_worst_brain_recall       status                config_id  min_recall  mean_recall  total_candidates
                      0.80 meets_target tip_to_any_node|r=20|k=2    0.809034     0.883650           1117332
                      0.90 meets_target tip_to_any_node|r=50|k=2    0.907871     0.949488           2836006
                      0.95 meets_target tip_to_any_node|r=75|k=4    0.950358     0.972241           5423518
```

## Artifact layout

* `per_dataset/<brain>/sweep.csv` — every configuration for one brain.
* `per_dataset/<brain>/metadata.json` — input/cache and truth-universe audit.
* `per_dataset/<brain>/candidate_pool_diagnostics.png` — distance, cost, and truth-type diagnostics.
* `tables/` — combined, aggregate, Pareto, baseline, summary, and recommendation CSVs.
* `figures/cross_dataset_candidate_pool_diagnostics.png` — robust cross-dataset diagnostics.
* `run_summary.json` — machine-readable run configuration and headline results.
* `provenance/split_candidate_pool_sweep.py` — exact analysis-source snapshot.
* `artifact_manifest.json` — SHA-256 and size for every artifact in this bundle.

The figures are ceilings, not classifier performance. After choosing a proposal rule,
the scoring model must be retrained and evaluated on held-out brains.

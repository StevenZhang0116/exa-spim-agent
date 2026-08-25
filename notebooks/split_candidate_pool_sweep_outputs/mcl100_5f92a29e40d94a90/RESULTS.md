# Split candidate-pool sweep artifacts

This directory is one configuration-scoped, cumulatively checkpointed analysis bundle. Its configuration
signature is `5f92a29e40d94a90`. Candidate recall is measured against reachable
truth pairs before feature scoring.

## Dataset summary

```text
 brain  n_reachable_truth_pairs  best_candidate_recall  fewest_candidates_at_best_recall
789202                     2927               0.966860                             84063
794491                     1839               0.997281                           2988906
794493                     1774               0.997182                           1463646
794495                     2226               0.978437                           2019045
802449                     6881               0.980381                           3497037
```

## Current detector baseline

The baseline is `tip_to_tip`, 30 µm, and unlimited per-anchor partners. All
surviving segment pairs are retained; the sweep has no global candidate cap.

```text
 brain  n_candidate_pairs  n_candidate_truth_pairs  candidate_recall  direct_candidate_recall  gap_candidate_recall
789202              16285                     1994          0.681244                 0.809728              0.563407
794491             307979                      912          0.495922                 0.474037              0.500671
794493             110930                      858          0.483653                 0.471405              0.473525
794495             153009                     1419          0.637466                 0.792899              0.506709
802449             261646                     4127          0.599767                 0.725007              0.450417
```

## Recommended minimal pools

Selection minimizes total candidate count subject to a worst-dataset recall target.

```text
 target_worst_brain_recall       status                config_id  min_recall  mean_recall  total_candidates
                      0.80 meets_target tip_to_any_node|r=20|k=2    0.807278     0.880608           1096426
                      0.90 meets_target tip_to_any_node|r=50|k=2    0.906559     0.948127           2793919
                      0.95 meets_target tip_to_any_node|r=80|k=4    0.952381     0.973525           5644623
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

# Merge candidate-pool sweep artifacts

Configuration signature `dcbde4ab2a6f6c44`. Site recall is measured against
ALL recorded ``gt_merge_sites`` (geodesic claim radius from the snapped node to
the nearest kept junction). Junction enumeration is GT-blind at deploy time; GT
enters only this offline recall measurement.

## Dataset summary

```text
 brain  n_merge_sites  n_merge_labels  n_junctions_raw  labels_without_site  labels_without_fragment  sites_unsnappable
789202             67              64            57429                    9                        0                  0
794491             86              98           244820                   30                        0                  0
794493             39              30            84700                    0                        0                  0
794495            105              98            63212                   20                        0                  0
802449            786             674           389422                  119                        2                  0
```

## Sweep aggregate (all configurations)

```text
            config_id  total_candidates  min_site_recall  mean_site_recall  micro_site_recall  min_label_recall
   junction|nms=0|r=5            839583         0.476190          0.571032           0.598338          0.512821
  junction|nms=0|r=10            839583         0.743590          0.798914           0.829178          0.733333
  junction|nms=0|r=15            839583         0.743590          0.819357           0.843029          0.733333
  junction|nms=0|r=20            839583         0.743590          0.827933           0.852262          0.733333
  junction|nms=0|r=30            839583         0.743590          0.836358           0.863343          0.733333
  junction|nms=0|r=50            839583         0.743590          0.851832           0.878116          0.733333
  junction|nms=0|r=75            839583         0.769231          0.866315           0.891967          0.733333
 junction|nms=0|r=100            839583         0.769231          0.875312           0.901200          0.733333
 junction|nms=0|r=150            839583         0.846154          0.895057           0.912281          0.800000
  junction|nms=10|r=5            771335         0.428571          0.516415           0.542936          0.487179
 junction|nms=10|r=10            771335         0.717949          0.779348           0.810711          0.733333
 junction|nms=10|r=15            771335         0.743590          0.814996           0.834718          0.733333
 junction|nms=10|r=20            771335         0.743590          0.827678           0.851339          0.733333
 junction|nms=10|r=30            771335         0.743590          0.836103           0.862419          0.733333
 junction|nms=10|r=50            771335         0.743590          0.851832           0.878116          0.733333
 junction|nms=10|r=75            771335         0.769231          0.866315           0.891967          0.733333
junction|nms=10|r=100            771335         0.769231          0.875312           0.901200          0.733333
junction|nms=10|r=150            771335         0.846154          0.894803           0.911357          0.800000
  junction|nms=20|r=5            679203         0.409524          0.493630           0.512465          0.474359
 junction|nms=20|r=10            679203         0.692308          0.743415           0.766390          0.700000
 junction|nms=20|r=15            679203         0.717949          0.786180           0.804247          0.700000
 junction|nms=20|r=20            679203         0.743590          0.813179           0.835642          0.733333
 junction|nms=20|r=30            679203         0.743590          0.833778           0.861496          0.733333
 junction|nms=20|r=50            679203         0.743590          0.851832           0.878116          0.733333
 junction|nms=20|r=75            679203         0.769231          0.863989           0.891043          0.733333
junction|nms=20|r=100            679203         0.769231          0.875312           0.901200          0.733333
junction|nms=20|r=150            679203         0.846154          0.894803           0.911357          0.800000
```

## Recommended minimal policies

Selection minimizes total candidate count subject to a worst-brain SITE-recall
target. `target_unmet_best_effort` rows report the best achievable point.
The policy at target 0.8 is written to `recommended_policy.json`.

```text
 target_worst_brain_site_recall                   status             config_id  min_site_recall  mean_site_recall  total_candidates
                           0.60             meets_target  junction|nms=20|r=10         0.692308          0.743415            679203
                           0.70             meets_target junction|nms=20|r=100         0.769231          0.875312            679203
                           0.75             meets_target junction|nms=20|r=100         0.769231          0.875312            679203
                           0.80             meets_target junction|nms=20|r=150         0.846154          0.894803            679203
                           0.85 target_unmet_best_effort junction|nms=20|r=150         0.846154          0.894803            679203
```

## Structural ceilings (never coverable by a junction-anchored rule)

Per brain: `labels_without_site` merge labels have no recorded site (invisible
to site-anchored evaluation), `labels_without_fragment` have no fragment
skeleton at all, and `sites_unsnappable` sites have no on-segment node within
50 µm. Bridge-type sites (no junction within the largest
claim radius) cap the recall plateau — see the census
(`notebooks/merge_diagnosis/`) for their anatomy. A second candidate family
(mid-shaft radius-bulge scan) and segment-detector gating were NOT evaluated
here; both are recorded as open in `recommended_policy.json`.

## Artifact layout

* `per_brain/<brain>_sweep.csv` — every configuration for one brain.
* `per_brain/<brain>_site_distances.csv` — per-site distance to nearest kept candidate per NMS variant.
* `per_brain/<brain>_meta.json` — residuals, junction counts, checkpoint signature.
* `tables/` — combined, aggregate, and recommendation CSVs.
* `figures/merge_candidate_pool_frontier.png` — recall-vs-cost frontier.
* `recommended_policy.json` — deterministic policy selection with provenance.
* `run_summary.json` — machine-readable configuration and headline results.
* `provenance/merge_candidate_pool_sweep.py` — exact analysis-source snapshot.
* `artifact_manifest.json` — SHA-256 and size for every artifact in this bundle.

These numbers are enumeration ceilings, not classifier performance. After
adopting a policy, the site-scoring model must be trained and evaluated on
held-out brains.

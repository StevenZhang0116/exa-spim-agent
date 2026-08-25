# Merge detector — `merge-error-794495-mcl100_2026-08-04`

> **Run status (2026-08-16):** the real-data run completed successfully after
> excluding hypotheses 6 and 39 through the validated manual CLI. It produced
> 9,623 rows (98 merges), the model-selection JSON, fitted joblib, CSV, log, and
> training figures 01–07. Build provenance remains below; measured interpretation
> is generated separately in `RESULT_ANALYSIS.md` by the result-analysis agent.

Exact commands and current artifact hashes are in `RUN_COMMANDS.md`.

<!-- BEGIN DRIVER-GENERATED PROVENANCE — preserve this block -->

## Provenance

- Origin run: `autodiscovery/merge-error-794495-mcl100_2026-08-04.json`
- Finished report: `autodiscovery/merge-error-794495-mcl100_2026-08-04.summary.md`
- Feature inventory: `autodiscovery-application/merge-error-794495-mcl100_2026-08-04/feature_inventory.json`
- Model configuration: `autodiscovery-application/merge-error-794495-mcl100_2026-08-04/model_candidates.json`
- Generated detector: `autodiscovery-application/merge-error-794495-mcl100_2026-08-04/merge_site_detector.py`
- Instance-bound command guide: `autodiscovery-application/merge-error-794495-mcl100_2026-08-04/RUN_COMMANDS.md`
- Selection manifest: `autodiscovery/merge-error-794495-mcl100_2026-08-04.predictive-selection.json`

## Feature and source mapping

| ID | Included | Source | Feature columns | Defined when / exclusion |
|---:|:---:|---|---|---|
| 1 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_1.py | max_windowed_path_tortuosity | at least one degree-2 path window with path_length >= 10um exists and euclidean endpoint distance > 0 |
| 2 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_2.py | max_euclidean_geodesic_wraparound_ratio | at least one node pair with euclidean distance < 10um and graph geodesic distance > 150um |
| 3 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_3.py | min_junction_branch_angle_deg | at least one degree>=3 junction with two or more resolvable branch directions |
| 4 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_4.py | max_degree3_radius_asymmetry | at least one degree-3 junction with three positive sampled branch radii |
| 5 | no | — | — | Corrected result OVERTURNED: the effect vanished under the corrected test and the hypothesis does not generalize; per safety gate an OVERTURNED correction is excluded. |
| 6 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_6.py | max_normalized_edge_betweenness_bridge | component has >=50 nodes and at least one bridge edge |
| 8 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_8.py | max_euclidean_edge_jump | segment has at least one component with at least one edge |
| 9 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_9.py | box_counting_fractal_dimension | component has >=50 nodes and coordinate spread >= 50um in all axes so box counting is defined |
| 10 | no | — | — | Corrected result OVERTURNED: the original claim rested on a failed-to-reject (absence-of-evidence) fallacy that the corrected test overturned; per safety gate an OVERTURNED correction is excluded. |
| 11 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_11.py | component_aspect_ratio, component_spatial_density | segment's largest component has >=20 nodes so PCA principal axes are defined; segment's largest component has >=20 nodes and positive bounding-box volume |
| 13 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_13.py | max_junction_tortuosity_variance | at least one degree>=3 junction with >=3 measurable branch tortuosities |
| 14 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_14.py | gmm_bimodality_bic_gain | segment has enough qualifying components (>=50 nodes each) to fit both 1- and 2-component GMMs on centroid distances |
| 15 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_15.py | p95_internode_radius_cv | at least one degree-2 path longer than 10um with positive mean radius exists |
| 16 | no | — | — | Corrected result OVERTURNED: the terminal-branch tortuosity effect vanishes once component size is adjusted for; per safety gate an OVERTURNED correction (effect vanished) is excluded. |
| 17 | no | — | — | Reproduction FAILED (1800s timeout, UNUSABLE) with no later corrected run repairing the failure; generalization INCONCLUSIVE. Per safety gate a FAILED/UNUSABLE reproduction without a corrected repair is excluded. |
| 18 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.fixed/hypo_18.py | max_leaf_cluster_silhouette | component has >=3 leaf nodes and >=2 unique leaf coordinates yielding two distinct clusters, so a real silhouette is computed (not the -1.0 default) |
| 19 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_19.py | max_radius_step_along_pseudo_diameter | component has a pseudo-diameter path with at least two nodes carrying radii |
| 20 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_20.py | pseudo_diameter_tortuosity | component has a resolvable pseudo-diameter path with positive graph length |
| 21 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_21.py | max_edge_betweenness_centrality | component has >=3 nodes so edge betweenness is defined |
| 22 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_22.py | neg_min_component_spatial_density | segment has at least one component with positive bounding-box volume |
| 23 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_23.py | max_high_degree_node_density | segment has at least one component with positive cable length |
| 25 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_25.py | neg_min_uturn_angle_deg | at least one degree-2 node with two resolvable branch directions |
| 26 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_26.py | min_junction_cosine_similarity | at least one degree>=3 junction with two or more resolvable branch directions |
| 27 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_27.py | max_branching_frequency_mismatch | component has >=5 nodes with subtree segments of length >=15um on both sides of a split |
| 29 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_29.py | curvature_spike_density | component has a longest path with enough consecutive edges to compute direction dot products |
| 30 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_30.py | neg_min_radius_assortativity | component has more than 10 edges; when radius variance is zero the historical fill 1.0 was used |
| 31 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_31.py | neg_min_log_convexhull_cable_density | component has >20 nodes forming a non-degenerate convex hull with positive volume |
| 32 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_32.py | max_tapering_reversal_rate | component has at least one edge traversable from the max-radius root |
| 34 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_34.py | neg_min_normalized_leaf_nn_distance | component has >3 leaf nodes and a positive bounding-box diagonal for normalization |
| 35 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_35.py | max_junction_radius_ratio | at least one degree>=4 junction with two or more positive sampled branch radii |
| 36 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_36.py | max_thickness_distance_spearman | component has >20 nodes with non-constant radius and distance ranks so Spearman is defined |
| 37 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_37.py | diameter_to_leaf_ratio | segment has at least one leaf node so the ratio denominator is positive |
| 39 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_39.py | max_bridge_radius_variance_ratio | component has >=100 nodes, spread>=50um, and a bridge that partitions it into two sides each with computable radius variance |
| 40 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_40.py | max_contracted_normalized_edge_betweenness | segment's largest component has >=10 nodes and the contracted graph retains >=10 nodes |
| 41 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_41.py | min_xcrossing_disjoint_pair_dot | at least one junction with two disjoint resolvable branch-direction pairs |
| 44 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_44.py | spatial_bimodality_silhouette | component has >1000 nodes yielding two distinct spatial clusters |
| 45 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_45.py | max_intra_component_tortuosity_variance | component has >=10 nodes and >=2 branches of length>=10um to compute a tortuosity variance |
| 46 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_46.py | min_xcrossing_antiparallel_pair_dot | at least one degree>=4 junction with two anti-parallel branch pairs both below dot -0.8 |
| 47 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_47.py | t_merge_geometry_score | at least one degree-3 junction with resolvable branch directions and radii to evaluate the T-merge geometry |
| 48 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_48.py | max_local_branch_density | segment has at least one degree>=3 branch point with neighbors within 5um |
| 49 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_49.py | max_rall_ratio | at least one degree-3 junction with two positive child branch radii |
| 50 | yes | autodiscovery/merge-error-794495-mcl100_2026-08-04.json.predictive.rerun/hypo_50.py | log_bbox_cable_density | component has cable length >= 500um, >=10 nodes, and positive bounding-box volume |

Every numeric feature is represented as its raw value plus a corresponding
`<feature>_is_defined` flag. Undefined values remain NaN; historical sentinel
values are provenance, not the new missingness representation.

## Candidate models

| Family | Role | Native NaN | Optional package | Reason |
|---|---|:---:|---|---|
| hist_gradient_boosting | baseline | yes | — | Required baseline and the natural strong reference for this inventory: it consumes the heavy structural NaN missingness natively (no fold imputation needed for the many junction/silhouette/bridge features that are undefined on most segments) and learns the thresholded, conjunctive geometry (T-merge, X-crossing, branch-density detectors) that linear models cannot. Shallow depth and higher min_samples_leaf guard against overfitting the sparse positives. |
| logistic_elasticnet | baseline | no | — | Required baseline. Elastic-net regularization is well suited to the many correlated, redundant feature families (angle ids 3/26/41/46/47, density ids 22/31/50, betweenness ids 6/21/40): the L1 component performs selection among near-duplicates while the L2 component stabilizes coefficients under collinearity, giving a sparse interpretable linear reference at low prevalence. |
| logistic_l2 | baseline | no | — | Required baseline. A plain L2-penalized logistic fit is the simplest fair reference in the simplicity order; strong ridge shrinkage keeps coefficients stable across the correlated feature families and the sparse-positive regime, establishing the floor that any nonlinear extension must beat. |
| explainable_boosting | optional | yes | interpret | Optional extension justified by two concrete inventory properties. First, native NaN handling: like hist_gradient_boosting it ingests the structural missingness (junction/silhouette/bridge features undefined on most segments) directly. Second, its explicit pairwise interactions term directly models the conjunctive threshold detectors present in the inventory (id 47 T-merge combining a direction dot with a radius ratio, ids 41/46 X-crossing pairs), and its additive shape functions capture the smooth-but-thresholded nonlinearity of angle/ratio/tortuosity features while staying interpretable for a merge detector. Interactions capped low and rounds kept modest for the sparse-positive regime. |
| xgboost | optional | yes | xgboost | Optional extension justified by native NaN handling of the same heavy structural missingness and by strong performance on imbalanced sparse-positive tabular data with correlated features. Its default-direction learning routes undefined junction/silhouette/bridge values without imputation, column and row subsampling decorrelates the redundant feature families, and reg_lambda plus a nonzero min_child_weight regularize the few positives against overfitting the thresholded geometry. Depth kept shallow for the low-prevalence regime. |

Primary metric: `average_precision`. Candidate-set rationale:
The inventory holds 38 per-segment continuous geometric scores across 37 included hypotheses (junction angles/radius-ratios, tortuosity, betweenness, densities, silhouettes, fractal dimension, correlations). Two properties drive candidate choice, both read only from feature semantics/coverage/missingness and never from any model result. (1) Heavy, structural, feature-specific missingness: most junction features (ids 3,4,13,26,35,41,46,47,48,49) are undefined on segments lacking degree>=3 or degree>=4 junctions; silhouette/bimodality features are gated on very large components (id 44 needs >1000 nodes, id 18 needs a genuinely computable KMeans, id 9 needs >=50 nodes and >=50um spread, id 39 needs >=100-node bridges), so many features are NaN on large fractions of segments and definedness is emitted as explicit is_defined flags with NaN values rather than sentinels. This favors models that ingest NaN natively without a single fold-level imputation blurring the defined/undefined boundary. (2) Non-linear thresholded geometry with correlated redundant families: several features are explicit threshold detectors (id 47 T-merge min_dot<-0.85 AND radius_ratio>0.8, ids 41/46 X-crossing dot<-0.8, id 48 local branch-density) and several families are near-duplicates (angle ids 3/26/41/46/47, density ids 22/31/50, betweenness ids 6/21/40), so conjunctive interactions and monotone thresholds carry signal that a purely linear fit splits or misses. Baselines (all three required linear/tree families) anchor a fair, simple reference; the two optional native-NaN boosting families add interaction- and threshold-aware capacity matched to (1) and (2). Grids are kept small and shallow because this is a merge-error task with likely few positives (~1% prevalence), where deep or wide models overfit sparse positives.

Actual family selection happens only through nested cross-validation in the
generated detector. Weighted classifier outputs are scores and are not
automatically calibrated probabilities.

## Run on a compute node

```bash
conda activate panda
python autodiscovery-application/merge-error-794495-mcl100_2026-08-04/merge_site_detector.py \
    cache/dataset_cache_794495_mcl100_add.pkl \
    --model-config autodiscovery-application/merge-error-794495-mcl100_2026-08-04/model_candidates.json
```

The pkl requires more than 20 GB RAM. Use `--test-pkl` for a different held-out
brain; it must not influence model or preprocessing selection.

## Expected outputs

- `merge_detector_<brain>.csv`
- `model_selection_<brain>.json`
- `merge_detector_<brain>.joblib`
- `merge_detector_<brain>.log.txt`
- `figures/`
- `analysis_timing_<brain>.json` only in profiling-only `--measuretime` mode
- `hypothesis_cost_report_<brain>.md` and
  `hypothesis_selection_template_<brain>.json` from the profiling run

Read undefined coverage first, then average precision and review-budget
precision/recall, then held-out transfer; read ROC-AUC last. Training thresholds
and queues must use nested out-of-fold scores, never the final in-sample score.

<!-- END DRIVER-GENERATED PROVENANCE -->

## Semantic review

This section is the verify-and-document reviewer's addition. Everything above the
`END DRIVER-GENERATED PROVENANCE` marker was written by the driver and is preserved
byte-for-byte. That block records build-time facts and intentionally does not
change after later runs. The 2026-08-16 real-data run is complete; its numeric
evidence and figure interpretation belong in `result_analysis_evidence.json` and
`RESULT_ANALYSIS.md`, rather than being copied into this build-design review.

### What this build contains

- **39 feature columns across 38 included hypotheses**, one row per adjudicable
  segment. Hypothesis 11 alone contributes two columns
  (`component_aspect_ratio` + `component_spatial_density`); all other included
  hypotheses contribute one. (The driver's `selection_basis` prose in
  `model_candidates.json` rounds this to "38 scores across 37 hypotheses" — the
  precise totals are 39 columns / 38 hypotheses. That prose is a driver-owned field
  and was not edited.)
- **4 excluded hypotheses (ids 5, 10, 16, 17)** — see the exclusion table below.
- **5 candidate model families** — 3 required baselines + 2 optional native-NaN
  boosters — compared by nested CV. Family selection is decided only at run time.

### Excluded hypotheses and why

| ID | Report verdict | Correction outcome | Why excluded |
|---:|---|---|---|
| 5 | MAJOR (does-not-generalize) | corrected MWU p=0.33, Cliff's delta CI straddles 0 → **OVERTURNED** | The "inverted bimodality" effect was a parametric/large-tie artifact; under the correct rank test it collapses to non-significance and the direction flips on the second brain. Safety gate: an OVERTURNED correction is excluded. |
| 10 | MAJOR (does-not-generalize) | corrected effect-size CI includes the null → **OVERTURNED** | The original "does NOT distinguish merges" claim rested on a failed-to-reject (absence-of-evidence) fallacy; the corrected test shows "no evidence," not "evidence of no effect," and the second brain shows a genuine positive effect. Safety gate: an OVERTURNED correction is excluded. |
| 16 | MINOR | size-adjusted coef collapses to ≈0, LR p=0.893 → **OVERTURNED** | Terminal-branch tortuosity's apparent effect is entirely a segment-size confound; once size is in the model the independent contribution is null. Safety gate: an OVERTURNED (effect-vanished) correction is excluded. |
| 17 | MAJOR | reproduction **FAILED / UNUSABLE** (1800 s timeout, empty stdout); generalization INCONCLUSIVE; no corrected run | The recorded p could not be re-verified and there is no cross-brain evidence. Safety gate: a FAILED/UNUSABLE reproduction with no corrected repair is excluded. |

Note the related spatial-bimodality hypotheses that were KEPT: id 44
(`spatial_bimodality_silhouette`, >1000-node components, SOUND, reproduced) and
id 18 (`max_leaf_cluster_silhouette`, leaf-cluster silhouette). Id 17 was the only
one of the silhouette family that failed reproduction, so it alone is dropped.

### Per-feature rerun-vs-fixed source choice

Of the 38 included hypotheses, **37 use the loading-fixed `.rerun` script** for the
detector feature math and **one uses the `.fixed` script**:

- **Only hypothesis 18 (`max_leaf_cluster_silhouette`) uses its `.fixed` source**
  (`correction_scope = feature_semantics_changed`). The corrected script restricts
  the aggregation to segments where a silhouette was *actually computed* (KMeans on
  ≥3 leaves with ≥2 unique coordinates yielding two real clusters), removing the
  synthetic −1.0 "un-clusterable" default that had contaminated the clean class. That
  changes the defined/undefined membership and the aggregation input, so it is a
  genuine feature-semantics change, and the corrected report result is USABLE and
  UPHELD (not OVERTURNED). I confirmed in `extract_features` that a segment's
  silhouette column is set ONLY when `seg_measured` is True (a real cluster was
  formed); the −1.0 default is therefore treated as undefined, matching the fixed
  semantics. Its source SHA-256 (`60221a83…`) matches the inventory.
- **Several hypotheses have a `.fixed` script but still use `.rerun`**
  (ids 2, 11, 23, 29, 35, 37, 45): in every case the correction was `test_only`
  (it swapped a mis-specified significance test — a zero-inflation-tie MWU replaced
  by Fisher's exact, a Welch t replaced by a one-sided MWU + Cliff's delta, etc.) but
  left the feature quantity, sample construction, constants, and aggregation
  unchanged. Per the correction-scope rule, `test_only` means the corrected report
  result controls the *evidence* while `.rerun` supplies the *detector math*. I spot-
  checked the recorded corrected measurements in the report and each is UPHELD, so
  none of these hypotheses is disqualified.
- **No included hypothesis is in the `unclear` bucket.** The three `unclear`
  corrections in the run (ids 5, 10, 16) all coincide with OVERTURNED results and are
  excluded above, so no feature was drawn from an ambiguous implementation.

### Feature semantics and defined conditions

Every numeric feature is emitted as its raw value plus an extraction-time
`<feature>_is_defined` boolean set from actual dictionary/set membership in
`SegmentAccumulator.set` — never from a value-vs-sentinel comparison. A real
measurement that happens to equal a historical sentinel (0.0, 1.0, 180.0, −1.0, 2.0)
stays DEFINED. The measurable condition and historical sentinel for each column are
in the driver's feature-mapping table above. Representative examples:

- **Junction geometry** (ids 3, 4, 13, 26, 35, 41, 46, 47, 48, 49): defined only when
  the segment actually has a qualifying degree≥3 or degree≥4 junction with resolvable
  branch directions/radii. Most segments lack these, so these columns are NaN on large
  fractions of the population — this is exactly the structural missingness the
  native-NaN models are chosen for.
- **Silhouette / bimodality** (ids 9, 14, 18, 44): gated on large components (id 44
  needs >1000 nodes; id 9 needs ≥50 nodes and ≥50 µm spread; id 39 needs ≥100-node
  bridges), so they are undefined on small segments.
- **Threshold detectors** (id 47 T-merge = `min_dot < -0.85` AND `radius_ratio > 0.8`;
  ids 41/46 X-crossing `dot < -0.8`; id 48 local branch-density) fire on conjunctive
  geometry that linear models can only approximate.

### Traversal phase structure

Extraction is organized as shared single passes over the fragments graph so that no
feature re-walks structure another feature already visited:

- **edge pass** (1 feature): single-edge Euclidean jump (id 8);
- **junction / chain node pass** (9 features): a single per-node loop keyed on node
  degree covers the shared ids 3, 4, 25, 26, 35, 46, 47 and 49, plus the separate
  KDTree local-branch-density analysis (id 48);
- **x-crossing disjoint-pair pass** (id 41): per-component junction directions;
- **chain / internode pass** (2 chain features): degree-2 path enumeration shared by
  ids 1 and 15;
- **component pass** (22 features in 21 selectable units): everything that walks a whole component
  — pseudo-diameter (ids 19, 20, 37), betweenness (ids 6, 21, 40), densities (ids 22,
  31), bridges (ids 6, 39), and the per-component silhouette/aspect/thickness
  reductions (ids 11, 18, 36);
- **segment pass** (4 features): features aggregating all of a segment's nodes — ids
  9, 14, 44 and 50. The diameter-to-leaf ratio (id 37) is classified in the
  component pass because it also traverses each component.

Feature counts per actual execution phase (declared only in
`ANALYSIS_TIMING_GROUPS`): chain 2, component 22, junction 9, edge 1, segment 4,
x-crossing 1 (= 39). `FEATURE_REGISTRY` owns only public names and column order.

### Why each model family is included

- **`logistic_l2` (required baseline):** the simplest fair reference; strong ridge
  shrinkage stabilizes coefficients across the correlated feature families at ~1 %
  prevalence, establishing the floor any nonlinear model must beat.
- **`logistic_elasticnet` (required baseline):** the L1 term selects among near-
  duplicate families (angle ids 3/26/41/46/47; density ids 22/31/50; betweenness ids
  6/21/40) while the L2 term stabilizes under collinearity.
- **`hist_gradient_boosting` (required baseline):** consumes the heavy structural NaN
  missingness natively and learns the conjunctive/thresholded geometry linear models
  cannot; shallow depth + higher `min_samples_leaf` guard the sparse positives.
- **`explainable_boosting` (optional):** native-NaN, plus an explicit pairwise-
  interactions term that directly models the conjunctive threshold detectors (id 47
  T-merge, ids 41/46 X-crossings) while staying interpretable.
- **`xgboost` (optional):** native-NaN default-direction routing of undefined values,
  column/row subsampling to decorrelate the redundant families, `reg_lambda` +
  `min_child_weight` to regularize the few positives.

The candidate choice was made from feature semantics, likely interactions, and
missingness behavior ONLY — never from any model result, outer label, or held-out
data. This build stays within policy: exactly the three required baselines plus at
most two optional families.

### How to safely edit `model_candidates.json`

The JSON is a *selection surface, not a code surface*. At startup
`validate_model_config` re-validates it against the SHA-pinned embedded policy before
any data is loaded, and `build_estimator` constructs every family through an explicit
named branch — there is **no `eval`, no dynamic import of a configured path, and no
class string** anywhere. Consequently:

- You may drop an optional family, or change grid VALUES, as long as each value
  satisfies the policy rule token (positive counts/depths, rates in (0,1], the
  documented `max_depth`/`max_features` null/string choices) and the grid stays within
  the 24-combination cap.
- You may NOT add a family that is not on the policy allowlist, exceed two optional
  families, drop a required baseline, or introduce a new parameter key — validation
  raises and the run aborts with no output.
- **An "arbitrary" family cannot be added by JSON at all.** Adding a genuinely new
  estimator requires editing the embedded policy AND writing a new named branch in
  `build_estimator` under human code review — precisely so that a JSON edit can never
  smuggle an unreviewed class path or code string into the pipeline.

### Nested CV and the one-standard-error rule, in plain language

The detector uses **5 outer folds × 3 inner folds**. For each outer fold, the inner
3-fold CV tunes each family's hyperparameters on the outer-training rows only, the
tuned family is refit on all outer-training rows, and it scores the held-out outer-
test rows once. Outer-test labels are used **only** to compute metrics — never to
choose hyperparameters or the family. This is what makes the reported out-of-fold
(OOF) numbers honest rather than in-sample-optimistic.

The **one-standard-error rule** applies at two levels: within a family, among grids
whose mean inner AP is within 1 SE of the best, the simpler-scoring choice is taken;
across families, among those whose best inner AP is within 1 SE of the global best,
the **simplest family in the policy simplicity order wins** (logistic before splines
before boosting). The point is to prefer the simplest model that is statistically
indistinguishable from the best, so a marginal AP gain never justifies a more complex,
more overfit-prone model at ~1 % prevalence.

### Which score is which (do not confuse them)

- **Family OOF** — a single family's nested out-of-fold scores. Honest per-family
  comparison.
- **Selector OOF** (`merge_probability_oof_selector`) — for each outer fold, the score
  of whichever family the one-SE policy chose IN that fold; this measures the *whole
  selection procedure*, including its fold-to-fold instability.
- **Winner OOF** (`merge_probability_oof`) — the nested OOF of the single family that
  wins on the full data. This is the number the threshold sweep and review queue use,
  and the one to trust for "how will this rank unseen segments of this brain."
- **In-sample** (`merge_probability_in_sample`) — the final model scoring its own
  training rows. Reference/diagnostic ONLY; it is optimistic and must never be used to
  set a threshold.
- **Held-out** (`--test-pkl`) — the fitted winner applied to a *different* brain, with
  the TRAINING imputation/scaling statistics frozen inside the pipeline. The test
  brain's own statistics never enter, and `--test-pkl` cannot influence the full-data
  winner (the winner is fit before any held-out scoring).

Weighted classifier outputs are **scores, not calibrated probabilities.** The
positive class is up-weighted (`class_weight="balanced"`), so a "0.9" is a ranking
score, not a 90 % merge probability. Rank segments by it; do not read it as a
calibrated likelihood without a separate calibration step.

### The step-4 confound this script is built to expose

The source AutoDiscovery run's own verifier repeatedly caught a single defect: a
feature that imputes a "no-signal" value for segments lacking the required structure
inflates that feature's apparent AUC, because the imputed pile lands almost entirely
in the clean class (see ids 18 and 35 in the report, where a −1.0 / zero pile drove a
misleading separation). **Expect the combined script to reproduce this across every
feature at once**, since 39 features each have their own structural missingness.

The script therefore prints, BEFORE any statistical fill, an
`UNDEFINED / COVERAGE AUDIT` that gives, for every feature: its undefined share
overall and among merges; the size of the **all-undefined group** and how many of
those are merges (a large all-undefined group with ZERO merges is the warning flag it
prints explicitly); and each feature's **full-set AUC beside its defined-subset AUC**.
The **defined-subset AUC is the honest one and should match the AUC the report
recorded** for that hypothesis; a full-set AUC well above it is inflation the
combination introduced. The `--exclude-empty` flag refits without the all-undefined
group and is OFF by default so the shipped numbers match this README's documented
scope.

### How to read the results (in this order)

1. **Coverage / the undefined audit first.** Look at the all-undefined group size and
   the undefined share by class. If a large all-undefined group has no merges, any
   headline metric is partly the model trivially scoring that group clean.
2. **Average precision (AP)**, the policy primary metric, read against the ~1 %
   prevalence baseline. AP just above prevalence is near-useless regardless of ROC-AUC.
3. **Review-budget precision/recall** (precision@k / recall@k at the review budget) —
   the operationally meaningful "how many real merges in the top-k queue" number.
4. **Cross-brain transfer** (`--test-pkl` held-out AP/precision) if you ran it.
5. **ROC-AUC LAST.** At ~1 % prevalence ROC-AUC is the least informative of the four
   and is exactly the metric the discovery-stage inflation confound flatters.

### Output schemas

- **`merge_detector_<brain>.csv`** — one row per adjudicable segment:
  `segment_id`, `is_merge`, then for each feature its raw value and its
  `<feature>_is_defined` flag, then per-family OOF scores
  (`merge_probability_oof_<family>`), the selector OOF
  (`merge_probability_oof_selector`), the winner OOF (`merge_probability_oof`), and
  the in-sample reference (`merge_probability_in_sample`).
- **`model_selection_<brain>.json`** — why the winner won: config/policy/inventory
  SHAs, the configured registry, skipped candidates, per-fold inner choices, per-
  family and selector OOF metrics, selection frequency, final winner + params, feature
  order, the undefined-audit summary, and the held-out report if `--test-pkl` was used.
- **`merge_detector_<brain>.joblib`** — the fitted pipeline, the ordered
  `feature_order`, the winner name/params, and the scope. Load and call on a feature
  matrix in that exact column order.
- **`merge_detector_<brain>.log.txt`** — the full teed console log; a missing
  `# OK … after Ns` footer means the process was killed (OOM / Slurm timeout) rather
  than finishing. During component extraction, `segment:start/done`,
  `component:start/done`, and `feature:start/done` records identify exact ids,
  progress, graph size, and elapsed time; the last `feature:start` identifies the
  active feature if a run hangs or is killed.
- **`analysis_timing_<brain>.json`** — written only by sampled `--measuretime`
  (3 component-bearing segments by default; configurable with
  `--measuretime-occurrences`). Its
  schema-v2 content records
  detector/input/environment provenance, load and extraction wall time, per-pass
  shared/unattributed time, per-analysis considered/eligible calls and aggregate
  durations, per-hypothesis exclusive versus shared-unit observed sample cost,
  and the slowest sampled segment/component contexts. These seconds are a rough
  relative-cost guide, not a full-run projection.
- The existing app-directory `analysis_timing_794495.json` is an incomplete
  diagnostic artifact (`status=running`) tied to an older detector hash. It is
  not a valid selection source; generate a new complete artifact with the current
  detector as shown in `RUN_COMMANDS.md`.
- **`hypothesis_cost_report_<brain>.md`** — ranks executable selection units by
  observed sample seconds. A multi-hypothesis row is indivisible: all member
  ids must be excluded to save that shared cost.
- **`hypothesis_selection_template_<brain>.json`** — edit its
  `excluded_hypothesis_ids`, then pass it with `--hypothesis-selection` to the
  final train/test run. Disabled computations are skipped while their stable CSV
  columns remain NaN with `is_defined=False`; train and held-out use the same
  selection. Selection provenance is stored in model JSON and joblib. Omitting
  the option preserves the all-hypothesis behavior. Profiling does not choose an
  exclusion automatically.
- **Direct manual exclusion** — after reading the cost report, pass IDs without
  creating JSON, for example `--exclude-hypotheses 39 6`. The same shared-unit
  and ID validation applies; model JSON/joblib record
  `selection_source=manual_cli`. This detector's hypotheses 39 and 6 are separate
  units, so they may be excluded independently. Omitting both selection options
  keeps every hypothesis.
- **`figures/`** — per-family PR+ROC, winner workload (precision/recall vs queue
  size), score-by-class histogram, feature ECDFs + Spearman correlation, undefined
  share by class, winner-aware model explanation, and validation permutation
  importance. Figure 06 uses signed standardized coefficients for ordinary
  logistic winners; for `explainable_boosting` it shows unsigned global term
  importance plus the top univariate shape functions; compatible tree winners
  use native unsigned importance, with validation permutation importance as the
  fallback. Figure 07 remains the common AP-scored permutation reference.
  Nonlinear importance has no coefficient-like sign and is not comparable across
  model families.

### Optional-dependency behavior

If `interpret` (explainable_boosting) or `xgboost` is not importable, that family is
**recorded as skipped and the run continues** with the remaining families — the script
never attempts an install. Because both optional families are native-NaN boosters,
their absence removes interaction-aware capacity but leaves all three required
baselines intact, so the detector still produces a fair comparison and a winner.

### Caveats

- **Sparse positive labels.** At ~1 % merge prevalence, PR-based metrics are the
  honest ones; a high ROC-AUC can coexist with a review queue that is mostly false
  positives.
- **Precision measured against sparse ground truth is a FLOOR, not a ceiling.** An
  unlabeled flagged segment whose geometry matches the confirmed merges may be a real
  merge OUTSIDE the traced neurons rather than a false positive, so the true precision
  can only be higher than the number the CSV reports.
- **Correlated features split credit.** Several families are near-duplicates (angle
  ids 3/26/41/46/47; density ids 22/31/50; betweenness ids 6/21/40). Linear
  coefficients and single-feature importances divide the shared signal among them, so
  a small coefficient does not mean a feature is uninformative.
- **Discovery-stage feature-selection bias.** These 38 hypotheses were chosen because
  they looked promising on THIS brain, so multivariate performance on this brain is
  optimistic; the `--test-pkl` transfer number is the closer estimate of generalization.
- **Uneven coverage.** Features have very different definedness rates; a feature
  defined on few segments can still carry independent signal there but contributes
  nothing to the majority, which the audit makes explicit.
- **Segment-level, NOT merge-site localization.** The output says which SEGMENT is a
  merge, not WHERE along it the false fusion occurs. A flagged segment still needs
  human localization of the actual splice.

### Verification performed and what still needs the compute node

The original generated artifacts received a static read-only review. Subsequent
post-build repairs removed hypothesis 40's invalid geometry access and added
observational component progress, hypothesis-cost profiling, explicit final-run
selection, and selection-provenance validation without changing feature math:

1. **Features.** Each of the 39 inventory feature columns has exactly one
   `feature_source_path`; I re-hashed hypo_1 (rerun), hypo_47 (rerun) and hypo_18
   (fixed) and all three SHA-256 values match the inventory. `FEATURE_REGISTRY` lists
   the 39 columns in exact inventory order (verified programmatically). Every feature
   is set through `SegmentAccumulator.set`, which records the value AND flips the
   `is_defined` flag by membership; undefined stays NaN; no value-vs-sentinel test is
   used for missingness; `--exclude-empty` drops rows via the flag columns
   (`_drop_all_undefined`), not sentinel values. hypo_18's `.fixed` measured-only
   semantics are honored (`seg_measured` gate).
2. **Preprocessing inside CV.** All imputation/scaling/spline steps live inside the
   per-family sklearn `Pipeline` built by `build_estimator`; there is no whole-dataset
   `fill_mean` or any preprocessing before `run_nested_selection`. Native-NaN families
   (hist_gradient_boosting, explainable_boosting, xgboost) receive raw NaNs.
3. **Model config.** `--model-config` defaults to the JSON beside the script and is
   revalidated by `validate_model_config` before data loading; the config carries
   exactly the three required baselines + two optional families; SHAs, native-NaN
   flags, dependency names, and the simplicity order agree between the JSON, the
   embedded policy, and `build_estimator`.
4. **Nested CV.** Outer-test labels feed only `average_precision_score`/`roc_auc_score`;
   inner folds tune and select; the one-SE rule is implemented at both the within-
   family and across-family levels with the simplicity order as the across-family
   tiebreak; each family fills each outer-test row exactly once; the final winner is
   fit before any `--test-pkl` scoring.
5. **Metrics.** `average_precision` is primary; the threshold sweep and review queue
   use the winner's nested OOF (`winner_oof`), and in-sample scores are labeled
   reference-only. Held-out prediction reuses the fitted pipeline and freezes its
   preprocessing.
6. **Outputs.** The model-selection JSON records per-fold choices and per-family/selector
   metrics; the joblib carries the pipeline + ordered schema; CSV columns are
   distinct; the coefficient figure is gated to linear winners and the importance
   figure is permutation (not impurity) importance.
7. **Hygiene.** No install command, no `sys.path` edit, no `sys.modules` deletion on
   the happy path (the mock unpickler registers modules only as an ImportError
   fallback and never deletes them); each pkl is loaded once and released with
   `del payload; gc.collect()`; `matplotlib.use("Agg")` precedes `pyplot`; logging is
   line-buffered and teed; `--synthetic-smoke-test` is a no-data/no-output mode that
   prints `DETECTOR_SMOKE_OK` only after its assertions pass.
8. **Component observability.** A targeted three-node extraction completed with
   segment/component start/done records, start/done records for all 21 timed
   component feature/group computations, and the final per-feature call-count and
   cumulative-time summary. The profiling-only path additionally covers all 30
   selectable analysis groups / 39 public features across indexing, edge, junction,
   x-crossing, chain, component, and segment phases, with exact-once registry
   and hypothesis-owner validation. A targeted selection test showed that default
   and explicitly all-enabled extraction agree, while an edge-only selection leaves
   all other feature flags false and does not enter component computations. Python
   compilation and the full no-data detector smoke also pass after this change.
   Generated selection templates bind to a complete timing artifact by path/hash,
   exact detector hash, and inventory hash; validated reasons and provenance flow
   into model-selection JSON and joblib. Minimal hand-authored selections may omit
   both timing-source fields.

The post-build runtime repairs cover the hypothesis-40 plain-graph attribute error,
component-pass observability, hypothesis cost measurement, and explicit final-run
selection. The inventory and model configuration were not changed.

**What still requires a successful end-to-end compute-node run:** full feature
extraction on the real >20 GB pkl, the
undefined-audit numbers, nested-CV selection, and every metric. This README contains
**no measured numbers** for any of these; where a real run's numbers belong, only the
expected content is described.

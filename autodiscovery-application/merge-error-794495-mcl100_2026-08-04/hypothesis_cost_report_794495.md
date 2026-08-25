# Hypothesis computational-cost report: 794495

This is a small-sample profile (3 of 9623 adjudicable segments; limit 3). Seconds are observed sample cost, not projected full-run savings.
Use the ranking only as a rough computational-cost guide. It is not a measure of scientific or predictive value, and phase-level shared overhead is not assigned to individual units.

| selection unit | hypothesis ids | phase | observed sample seconds | eligible / considered | features |
|---|---:|---|---:|---:|---|
| hypo_6 | 6 | component | 3.062250 | 2 / 3 | max_normalized_edge_betweenness_bridge |
| hypo_39 | 39 | component | 3.024400 | 2 / 3 | max_bridge_radius_variance_ratio |
| hypo_18 | 18 | component | 1.199777 | 3 / 3 | max_leaf_cluster_silhouette |
| hypo_21 | 21 | component | 0.097262 | 3 / 3 | max_edge_betweenness_centrality |
| hypo_14 | 14 | segment | 0.089099 | 2 / 3 | gmm_bimodality_bic_gain |
| shared_junction_nodes | 3, 4, 25, 26, 35, 46, 47, 49 | junction | 0.031890 | 1 / 1 | min_junction_branch_angle_deg, max_degree3_radius_asymmetry, neg_min_uturn_angle_deg, min_junction_cosine_similarity, max_junction_radius_ratio, min_xcrossing_antiparallel_pair_dot, t_merge_geometry_score, max_rall_ratio |
| hypo_2 | 2 | component | 0.025435 | 3 / 3 | max_euclidean_geodesic_wraparound_ratio |
| hypo_27 | 27 | component | 0.015835 | 3 / 3 | max_branching_frequency_mismatch |
| hypo_9 | 9 | segment | 0.015711 | 2 / 3 | box_counting_fractal_dimension |
| hypo_20 | 20 | component | 0.010333 | 3 / 3 | pseudo_diameter_tortuosity |
| hypo_40 | 40 | component | 0.006946 | 3 / 3 | max_contracted_normalized_edge_betweenness |
| shared_chain_1_15 | 1, 15 | chain | 0.006669 | 1 / 1 | max_windowed_path_tortuosity, p95_internode_radius_cv |
| hypo_37 | 37 | component | 0.006611 | 3 / 3 | diameter_to_leaf_ratio |
| hypo_30 | 30 | component | 0.004051 | 3 / 3 | neg_min_radius_assortativity |
| hypo_50 | 50 | segment | 0.003794 | 3 / 3 | log_bbox_cable_density |
| hypo_36 | 36 | component | 0.003442 | 3 / 3 | max_thickness_distance_spearman |
| hypo_13 | 13 | component | 0.003299 | 3 / 3 | max_junction_tortuosity_variance |
| hypo_8 | 8 | edge | 0.003133 | 1 / 1 | max_euclidean_edge_jump |
| hypo_32 | 32 | component | 0.002817 | 3 / 3 | max_tapering_reversal_rate |
| hypo_29 | 29 | component | 0.002258 | 3 / 3 | curvature_spike_density |
| hypo_45 | 45 | component | 0.002201 | 3 / 3 | max_intra_component_tortuosity_variance |
| hypo_31 | 31 | component | 0.001190 | 3 / 3 | neg_min_log_convexhull_cable_density |
| hypo_11 | 11 | component | 0.001153 | 3 / 3 | component_aspect_ratio, component_spatial_density |
| hypo_19 | 19 | component | 0.001044 | 3 / 3 | max_radius_step_along_pseudo_diameter |
| hypo_48 | 48 | junction | 0.000374 | 1 / 1 | max_local_branch_density |
| hypo_34 | 34 | component | 0.000372 | 3 / 3 | neg_min_normalized_leaf_nn_distance |
| hypo_41 | 41 | x_crossing | 0.000346 | 1 / 1 | min_xcrossing_disjoint_pair_dot |
| hypo_22 | 22 | component | 0.000277 | 3 / 3 | neg_min_component_spatial_density |
| hypo_23 | 23 | component | 0.000078 | 3 / 3 | max_high_degree_node_density |
| hypo_44 | 44 | segment | 0.000000 | 0 / 3 | spatial_bimodality_silhouette |

A row containing multiple hypothesis ids is indivisible: excluding only one member does not remove the shared computation. Exclude all ids in that row or none.

Edit the generated `hypothesis_selection_template_794495.json`, then pass it to the final run with `--hypothesis-selection`.
For a quick manual choice, pass ids directly as `--exclude-hypotheses <ID1> <ID2>`. The same shared-unit validation applies.

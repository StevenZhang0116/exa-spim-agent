# AI Review — Split Candidate-Pool Sweep

<!-- BEGIN DRIVER-GENERATED AI REVIEW EVIDENCE — preserve this block -->
This block is deterministic evidence, not AI-authored interpretation.
Evidence file: `ai_review_evidence.json` (`sha256=db042a9af16ade8570cc105165f2bff6f5cac3b89d1651337bfe159f373e91c6`)

| Run fact | Value |
|---|---|
| Status | `complete` |
| Configuration signature | `5f92a29e40d94a90` |
| MCL | `100` |
| Completed datasets | 789202, 794491, 794493, 794495, 802449 |
| Evaluated pairing rules | `tip_to_any_node`, `tip_to_tip` |
| `any_node_to_any_node` evaluated | `false` |
| Global-cap values | `none` |
| Global cap swept | `false` |
| Pre-review manifest integrity | `pass` |

| Recall target | Status | Configuration | Worst recall | Mean recall | Micro recall | Total candidates |
|---:|---|---|---:|---:|---:|---:|
| 80.00% | `meets_target` | `tip_to_any_node|r=20|k=2` | 80.73% | 88.06% | 86.78% | 1,096,426 |
| 90.00% | `meets_target` | `tip_to_any_node|r=50|k=2` | 90.66% | 94.81% | 93.95% | 2,793,919 |
| 95.00% | `meets_target` | `tip_to_any_node|r=80|k=4` | 95.24% | 97.35% | 96.99% | 5,644,623 |

| Dataset | Reachable truth pairs | Best candidate recall | Fewest candidates at that recall | Best configuration |
|---|---:|---:|---:|---|
| 789202 | 2,927 | 96.69% | 84,063 | `tip_to_any_node|r=100|k=8` |
| 794491 | 1,839 | 99.73% | 2,988,906 | `tip_to_any_node|r=90|k=8` |
| 794493 | 1,774 | 99.72% | 1,463,646 | `tip_to_any_node|r=100|k=16` |
| 794495 | 2,226 | 97.84% | 2,019,045 | `tip_to_any_node|r=100|k=16` |
| 802449 | 6,881 | 98.04% | 3,497,037 | `tip_to_any_node|r=100|k=16` |

Definitions: candidate recall is a pre-scoring enumeration ceiling; candidate prevalence is not classifier precision; direct and gap truth sets may overlap.
<!-- END DRIVER-GENERATED AI REVIEW EVIDENCE -->

## Executive summary

This sweep measures the pre-feature-scoring candidate-recall ceiling of the split
candidate enumerator across five mcl=100 brains (789202, 794491, 794493, 794495,
802449), not model, validation, or end-to-end recall. The strongest supported
conclusion is that the `tip_to_any_node` pairing rule dominates `tip_to_tip` at every
matched cost: on the cross-dataset Pareto frontier
(`figures/cross_dataset_candidate_pool_diagnostics.png`) every frontier point is a
`tip_to_any_node` configuration, and `tip_to_tip` never reaches the same worst-dataset
recall at any candidate budget. The smallest configuration that meets an 80% worst-brain
target is `tip_to_any_node|r=20|k=2` (worst 80.73%, mean 88.06%, micro 86.78%,
1,096,426 total candidates); reaching a 90% worst-brain target needs
`tip_to_any_node|r=50|k=2` (worst 90.66%, 2,793,919 candidates) and 95% needs
`tip_to_any_node|r=80|k=4` (worst 95.24%, 5,644,623 candidates). All three targets are
recorded as `meets_target`. The recommended operating point is `tip_to_any_node|r=20|k=2`
for a low-cost pool or `tip_to_any_node|r=50|k=2` if 90% worst-brain coverage is required.
The limiting dataset is 794495, which sets the worst-brain recall at both the 80% and 90%
targets and the 95% target. The largest caveat is that this is an enumeration ceiling
only: gap-truth coverage lags direct-truth coverage substantially, `any_node_to_any_node`
was never evaluated, and no downstream feature scoring or classifier was run.

## Experimental scope and integrity

The run is marked `complete` (config signature `5f92a29e40d94a90`, mcl=100) and covers
five brains: 789202, 794491, 794493, 794495, and 802449. The parameter grid sweeps 20
neighborhood radii from 5 to 100 um in 5-um steps, six per-anchor-k values
(1, 2, 4, 8, 16, 32), and two pairing rules (`tip_to_tip` and `tip_to_any_node`), with
unbounded `tip_to_tip` also included. Here per-anchor k counts distinct partner segments
retained per anchor; it is not a node count. Pre-review manifest integrity is `pass`
(25 artifacts checked, no failures), and the total run elapsed 1444.4 s. On old-to-new
naming, the historical `leaf_leaf` mode corresponds to the `tip_to_tip` rule reported
here and historical `leaf_any` corresponds to `tip_to_any_node`; both `modes_original`
and `modes_canonical` are `tip_to_tip` and `tip_to_any_node`, so no renaming ambiguity
remains. The fully symmetric `any_node_to_any_node` rule was not evaluated in this run
(`any_node_to_any_node_evaluated` is `false`). The only global-cap value present is
`none` and `global_cap_swept` is `false`, meaning all surviving pairs within each
neighborhood were retained and no post-enumeration closest-pair truncation was applied
or swept; there is therefore no cap sweep to report.

## Cross-dataset findings

The recorded current-detector baseline is `tip_to_tip|r=30|k=all` (global cap `none`),
which yields a worst-dataset recall of 48.37%, a mean recall of 57.96%, and a micro
recall of 59.50% over 849,849 total candidates (mean 169,969.8 per brain), with mean
candidate prevalence 3.16%. Every recommended `tip_to_any_node` configuration far exceeds
this baseline. The compact 80% target configuration `tip_to_any_node|r=20|k=2` lifts
worst recall to 80.73% (mean 88.06%, micro 86.78%) at 1,096,426 candidates. The robust
90% configuration `tip_to_any_node|r=50|k=2` reaches worst 90.66% (mean 94.81%, micro
93.95%) at 2,793,919 candidates, and the 95% configuration `tip_to_any_node|r=80|k=4`
reaches worst 95.24% (mean 97.35%, micro 96.99%) at 5,644,623 candidates. The best single
configuration observed is `tip_to_any_node|r=100|k=8` (worst 96.69%, mean 98.35%, micro
98.09%, 9,198,085 candidates), which is also the best `tip_to_any_node` result by pairing
rule; the best `tip_to_tip` result, `tip_to_tip|r=100|k=16`, only reaches worst 78.80%
(mean 82.89%, micro 83.72%) at a comparable 8,967,893 candidates. The cross-dataset
Pareto plot in `figures/cross_dataset_candidate_pool_diagnostics.png` visually confirms
this dominance: the red `tip_to_any_node` cloud rides the black Pareto frontier well above
the blue `tip_to_tip` cloud across the entire candidate-count axis, and the recall-vs-gap
panel shows red curves rising steeply and saturating near 0.95-1.0 while blue curves
plateau far lower.

## Dataset-specific observations

Behavior is heterogeneous across the five brains, and 794495 is the limiting brain: it
sets the worst-dataset recall at the 80% target (candidate recall 80.73% under
`tip_to_any_node|r=20|k=2`), the 90% target (90.66% under `tip_to_any_node|r=50|k=2`),
and the 95% target (95.24% under `tip_to_any_node|r=80|k=4`), and its best achievable
candidate recall (97.84%, `tip_to_any_node|r=100|k=16`, 2,019,045 candidates) is the
second lowest ceiling of the group. Its per-dataset figure
(`per_dataset/794495/figures/candidate_pool_diagnostics.png`) shows the red gap-recall
points sitting well below the diagonal in the truth-type panel, confirming gap truth as
its bottleneck. Brain 789202 is the smallest-pool brain: it attains its best candidate
recall of 96.69% with only 84,063 candidates (`tip_to_any_node|r=100|k=8`), by far the
cheapest ceiling, but its figure (`per_dataset/789202/figures/candidate_pool_diagnostics.png`)
shows the largest downward gap-vs-direct spread, and its baseline prevalence (12.24%) is
an order of magnitude above the other brains. Brains 794491 and 794493 are the easiest:
best candidate recalls of 99.73% (`tip_to_any_node|r=90|k=8`, 2,988,906 candidates) and
99.72% (`tip_to_any_node|r=100|k=16`, 1,463,646 candidates), and their figures
(`per_dataset/794491/figures/candidate_pool_diagnostics.png`,
`per_dataset/794493/figures/candidate_pool_diagnostics.png`) show red points hugging the
diagonal, meaning gap and direct coverage are nearly balanced. Brain 802449 is the
largest by reachable truth (6,881 pairs) and reaches a best candidate recall of 98.04%
(`tip_to_any_node|r=100|k=16`, 3,497,037 candidates); its figure
(`per_dataset/802449/figures/candidate_pool_diagnostics.png`) again shows red gap-recall
points falling below the diagonal, so it shares 794495's and 789202's gap-truth deficit,
though less severely at high radius.

## Recall-cost trade-offs

The `tip_to_any_node` radius series at k=2 shows strong diminishing returns in radius.
Worst-dataset recall climbs steeply from 40.46% at r=5 to 80.73% at r=20 and 90.66% at
r=50, then flattens: r=50 to r=100 gains only about 3.5 points of worst recall (90.66% to
94.20%) while total candidates nearly double from 2,793,919 to 4,312,003. The recall-vs-gap
panels in every figure make this saturation visible, with the red curves nearly horizontal
beyond roughly 50-60 um. The k series at the maximum radius r=100 saturates even harder:
worst recall rises from 89.44% at k=1 to 94.20% at k=2, 96.14% at k=4, and 96.69% at k=8,
then stops improving entirely (k=8, k=16, and k=32 all report worst recall 96.69%) even as
total candidates keep growing from 9,198,085 (k=8) to 11,199,248 (k=16) to 12,010,275
(k=32). Thus k>=8 buys no additional worst-dataset recall and only inflates the downstream
feature-scoring workload. Candidate counts here should be read as that downstream workload,
not as a runtime measurement. Candidate prevalence (the truth-pair fraction inside the
pool) is not classifier precision; it falls as pools grow, from 6.13% at r=5,k=2 to 1.11%
at r=100,k=2 and 0.75% at r=100,k=32, reflecting dilution by non-truth pairs as the
neighborhood expands. Because the only global-cap value is `none` and no cap sweep was run,
no cap trade-off can be assessed here.

## Direct versus gap truth

Gap truth, not direct truth, is the coverage-limiting truth type across every recommended
configuration. Under `tip_to_any_node|r=20|k=2` the worst-dataset direct-truth recall is
already 97.65% while the worst-dataset gap-truth recall is only 66.93%; the gap remains
wide at the 90% configuration (`tip_to_any_node|r=50|k=2`: min direct 99.16% versus min gap
83.90%) and persists even at the 95% configuration (`tip_to_any_node|r=80|k=4`: min direct
99.66% versus min gap 91.71%). The "which truth type is missed" panels in all six figures
confirm this: `tip_to_any_node` (red) points cluster near direct-recall 1.0 but sit below
the diagonal on the gap-recall axis, most severely for 794495, 789202, and 802449, and
closest to the diagonal for 794491 and 794493. Direct and gap truth sets may overlap, so
their missed counts are not additive; this review does not sum direct and gap misses, and
the evidence does not state that the two sets are disjoint. The practical implication is
that raising worst-brain coverage above the current ceiling is almost entirely a matter of
capturing more gap-truth pairs, since direct-truth recall is essentially saturated.

## Limitations and next experiment

All numbers here describe a pre-scoring enumeration ceiling: candidate recall is the upper
bound on what any downstream feature scorer or classifier could recover, not a measured
model, validation, or end-to-end recall. No feature scoring, classifier training, or
end-to-end evaluation was performed, and candidate prevalence must not be read as
precision. The sweep also did not test the fully symmetric `any_node_to_any_node` pairing
rule (`any_node_to_any_node_evaluated` is `false`) and applied no global-cap truncation
(cap `none`, not swept), so neither is characterized. Because gap truth on the worst brain
(794495) is the binding constraint and direct-truth recall is already near ceiling, the
smallest informative follow-up is to add the `any_node_to_any_node` pairing rule at a
modest radius/k grid (for example r in {20, 50, 80} and k in {2, 4}) on the three
gap-limited brains 794495, 789202, and 802449, and measure whether interior-node partners
raise worst-dataset gap-truth recall beyond the current 91.71% ceiling at the 95%
configuration. This isolates whether the residual gap-truth misses are reachable at all by
tip-anchored search or require symmetric all-node pairing, before any feature-scoring
investment is made.

# AI Review — Split Candidate-Pool Sweep

<!-- BEGIN DRIVER-GENERATED AI REVIEW EVIDENCE — preserve this block -->
This block is deterministic evidence, not AI-authored interpretation.
Evidence file: `ai_review_evidence.json` (`sha256=b2e1d8364cd0e7cc245166dac4a269e4e5f326ac8f54e48dfdfd4c8819251698`)

| Run fact | Value |
|---|---|
| Status | `complete` |
| Configuration signature | `5f92a29e40d94a90` |
| MCL | `10` |
| Completed datasets | 789202, 794491, 794493, 794495, 802449 |
| Evaluated pairing rules | `tip_to_any_node`, `tip_to_tip` |
| `any_node_to_any_node` evaluated | `false` |
| Global-cap values | `none` |
| Global cap swept | `false` |
| Pre-review manifest integrity | `pass` |

| Recall target | Status | Configuration | Worst recall | Mean recall | Micro recall | Total candidates |
|---:|---|---|---:|---:|---:|---:|
| 80.00% | `meets_target` | `tip_to_any_node|r=20|k=2` | 80.90% | 88.36% | 87.12% | 1,117,332 |
| 90.00% | `meets_target` | `tip_to_any_node|r=50|k=2` | 90.79% | 94.95% | 94.13% | 2,836,006 |
| 95.00% | `meets_target` | `tip_to_any_node|r=75|k=4` | 95.04% | 97.22% | 96.84% | 5,423,518 |

| Dataset | Reachable truth pairs | Best candidate recall | Fewest candidates at that recall | Best configuration |
|---|---:|---:|---:|---|
| 789202 | 2,903 | 96.76% | 86,575 | `tip_to_any_node|r=100|k=8` |
| 794491 | 1,827 | 99.67% | 2,667,603 | `tip_to_any_node|r=80|k=8` |
| 794493 | 1,774 | 99.72% | 1,483,054 | `tip_to_any_node|r=100|k=16` |
| 794495 | 2,236 | 97.85% | 2,042,057 | `tip_to_any_node|r=100|k=16` |
| 802449 | 6,853 | 98.02% | 3,563,644 | `tip_to_any_node|r=100|k=16` |

Definitions: candidate recall is a pre-scoring enumeration ceiling; candidate prevalence is not classifier precision; direct and gap truth sets may overlap.
<!-- END DRIVER-GENERATED AI REVIEW EVIDENCE -->

## Executive summary

This sweep measures candidate recall, the pre-feature-scoring enumeration ceiling
over reachable truth segment pairs. It is not model, validation, classifier, or
end-to-end recall; it caps what any downstream split-detector could later score.
Across all five brains (MCL 10, config signature `5f92a29e40d94a90`), the strongest
supported conclusion is that the `tip_to_any_node` pairing rule dominates
`tip_to_tip` at every matched radius and per-anchor-k, so the historical
tip-only enumeration is the wrong pairing rule if high recall is the objective.
The recommended rule for a first upgrade is `tip_to_any_node|r=20|k=2`, which is
the smallest configuration whose worst-dataset recall clears 80% (worst 80.90%,
mean 88.36%, micro 87.12%, 1,117,332 total candidates across the five brains).
If a higher floor is required, `tip_to_any_node|r=50|k=2` reaches a 90.79%
worst-dataset floor and `tip_to_any_node|r=75|k=4` a 95.04% floor, at
roughly 2.5x and 4.9x the candidate workload respectively. In every one of
these three recommendations the limiting dataset is brain 794495, which sets
the worst-brain recall (80.90%, 90.79%, 95.04% respectively). The largest
caveat is that all recall here is an enumeration ceiling only: no feature is
scored, no classifier is trained, and candidate prevalence (roughly 1-13% truth
pairs inside the pool) is a workload-purity figure, not classifier precision.
Gap truth, not direct truth, is the binding limitation everywhere, so the recall
floor is set by how well the neighborhood search reaches gap-crossing partners.

## Experimental scope and integrity

The run enumerated candidate pools over five brains — 789202, 794491, 794493,
794495, 802449 — at MCL 10. The grid crossed 20 radii (5 to 100 um in 5 um
steps), six per-anchor-k values (1, 2, 4, 8, 16, 32), and two pairing rules,
plus the unbounded `tip_to_tip|k=all` case; `include_unbounded_tip_to_tip` is
true. Per-anchor-k counts the maximum number of distinct partner segments
retained per anchor; it is not a node count. The run is marked `complete` and
finished in about 2,697 s. Pre-review manifest integrity is `pass` with 25
artifacts checked and no failures.

The two pairing rules evaluated are `tip_to_any_node` and `tip_to_tip`. In the
older naming these correspond to `leaf_any` (tip_to_any_node) and `leaf_leaf`
(tip_to_tip). The all-to-all rule `any_node_to_any_node` was NOT tested here
(`any_node_to_any_node_evaluated: false`), so no claim in this report covers
anchoring from interior nodes on both sides.

Only one global-cap value is present, `none`, and `global_cap_swept` is `false`.
That means every surviving neighborhood pair was retained; there was no
post-enumeration closest-pair truncation and no cap sweep to discuss. Where the
historical detector used a global cap, that cap was a post-enumeration
truncation and did not avoid the neighborhood search, but no such cap was
exercised in this bundle.

## Cross-dataset findings

The historical-style baseline is `tip_to_tip|r=30|k=all|cap=none`
(`figures/cross_dataset_candidate_pool_diagnostics.png`, left panel). It yields
a worst-dataset recall of 48.53% (brain 794493), mean recall 57.96%, and micro
recall 59.51% over 864,786 candidates (mean 172,957 per brain). Its worst direct
recall is 47.51% and its worst gap recall is 46.49%, so under the tip-only rule
even the direct partners are barely half-covered. This is the pre-scoring
ceiling the current tip-only pipeline imposes.

Switching the pairing rule to `tip_to_any_node` is the single largest lever. The
compact recommended configuration `tip_to_any_node|r=20|k=2` lifts the
worst-dataset floor from 48.53% to 80.90% (mean 88.36%, micro 87.12%) for
1,117,332 candidates — roughly 1.3x the baseline candidate workload at a similar
radius. A more robust choice, `tip_to_any_node|r=50|k=2`, reaches a 90.79%
worst-dataset floor (mean 94.95%, micro 94.13%) at 2,836,006 candidates, and
`tip_to_any_node|r=75|k=4` reaches 95.04% worst (mean 97.22%, micro 96.84%) at
5,423,518 candidates. All three targets are recorded as `meets_target`.

The ceiling configuration in this sweep is `tip_to_any_node|r=100|k=8`
(`best_by_pairing_rule` and `best_by_global_cap`): worst-dataset recall 96.76%,
mean 98.37%, micro 98.11% over 9,336,028 candidates. For contrast the best
`tip_to_tip` configuration, `tip_to_tip|r=100|k=16`, tops out at only 79.03%
worst-dataset recall (mean 82.86%, micro 83.65%) despite a comparable 9,117,931
candidates — i.e., the tip-only rule cannot reach even the 80% floor at any
tested radius or k. The Pareto frontier in
`figures/cross_dataset_candidate_pool_diagnostics.png` (left panel) shows the
red `tip_to_any_node` points sitting strictly above and left of the blue
`tip_to_tip` cloud: the same recall costs far fewer candidates under
`tip_to_any_node`, and the tip-only cloud never climbs into the top recall band.

## Dataset-specific observations

Behavior is heterogeneous across brains. The `dataset_best` block shows the two
easiest brains reach very high ceilings — 794491 at 99.67%
(`tip_to_any_node|r=80|k=8`) and 794493 at 99.72% (`tip_to_any_node|r=100|k=16`)
— while three brains cap lower: 789202 at 96.76% (`tip_to_any_node|r=100|k=8`),
794495 at 97.85%, and 802449 at 98.02% (both `tip_to_any_node|r=100|k=16`). Note
that the fewest candidates needed to reach the best recall vary by two orders of
magnitude, from 86,575 (789202) to 3,563,644 (802449), reflecting differing
neuron density and truth-pair counts (802449 has 6,853 reachable truth pairs,
more than double any other brain).

Brain 794495 is the limiting dataset across the graded recommendations: it sets
the worst-brain floor at all three targets (80.90% at r=20/k=2, 90.79% at
r=50/k=2, 95.04% at r=75/k=4). Its per-dataset diagnostics
(`per_dataset/794495/figures/candidate_pool_diagnostics.png`) show a slower rise
in the recall-versus-distance panel than the two easy brains, and its
truth-type panel places the worst gap points well below the diagonal, confirming
that gap coverage, not direct coverage, drags 794495 down.

Brain 789202 (`per_dataset/789202/figures/candidate_pool_diagnostics.png`) is
distinctive for its very high candidate prevalence — 13.32% at r=20/k=2 versus
under 2% for the other brains — meaning its pool is far denser in truth pairs;
its `tip_to_any_node` recall saturates near 0.95 by mid-radius but its gap
recall stays visibly under direct recall in the third panel.

Brain 794491 (`per_dataset/794491/figures/candidate_pool_diagnostics.png`) is
the easiest: `tip_to_any_node` recall climbs to nearly 1.0 quickly and its
truth-type panel hugs the diagonal, so direct and gap coverage are nearly
balanced. Its pools are also the largest at high radius (over 2.6M candidates at
its best config), reflecting a dense neighborhood despite few truth pairs
(prevalence 0.43% at r=20/k=2).

Brain 794493 (`per_dataset/794493/figures/candidate_pool_diagnostics.png`)
behaves similarly to 794491 — near-diagonal truth-type balance and a rapid rise
to a 99.72% ceiling — and is one of the two brains where gap and direct recall
converge under `tip_to_any_node`.

Brain 802449 (`per_dataset/802449/figures/candidate_pool_diagnostics.png`) is
the largest-truth brain and shows the widest gap-versus-direct spread of the
five in its third panel: direct recall approaches 1.0 while gap recall trails
markedly, and its recall-versus-cost panel needs the most candidates to reach
its ceiling. It is a secondary limiter behind 794495.

## Recall-cost trade-offs

Radius is the primary recall lever at fixed k. In the `tip_to_any_node` radius
series at k=2, worst-dataset recall climbs from 43.62% at r=5 to 70.30% at r=10,
80.90% at r=20, 90.79% at r=50, and 94.36% at r=100, while total candidates grow
monotonically from 303,913 to 4,370,534. Returns diminish steeply: the first
15 um (r=5 to r=20) buys about 37 recall points, the next 30 um (r=20 to r=50)
about 10 points, and the final 50 um (r=50 to r=100) only about 3.6 points at
roughly 1.5x more candidates. The middle panel of
`figures/cross_dataset_candidate_pool_diagnostics.png` shows this saturation
plateau clearly for the red `tip_to_any_node` curves, which flatten past
~40-50 um, whereas the blue `tip_to_tip` curves keep rising slowly and never
plateau near the top.

Per-anchor-k saturates even faster. In the k series at the maximum radius
r=100, worst-dataset recall goes 89.80% (k=1), 94.36% (k=2), 96.29% (k=4),
96.76% (k=8), and then is unchanged at 96.76% for both k=16 and k=32 — while
total candidates keep growing (9.34M at k=8, 11.38M at k=16, 12.22M at k=32).
So beyond k=8 additional partner segments per anchor add pure workload with no
recall gain; k=8 is the effective saturation point at r=100. This is why the
ceiling configuration uses k=8, and why the recommendations favor small k (k=2
for the 80% and 90% targets, k=4 for the 95% target).

Candidate prevalence — the truth-pair fraction inside the pool — falls as the
pool grows: mean prevalence is 6.02% at r=5/k=2, 3.50% at r=20/k=2, 1.76% at
r=50/k=2, and 0.76% at the r=100/k=8 ceiling. This is a measure of downstream
workload purity, not classifier precision: a larger, higher-recall pool is
proportionally more dilute in true pairs, so each recall gain past the plateau
loads the eventual feature/scoring stage with rapidly more negatives per
positive. Candidate count here is downstream feature/scoring workload, not a
complete runtime measurement. Because the only global-cap value is `none`, all
surviving pairs were retained and there is no cap sweep to trade off.

## Direct versus gap truth

Gap truth is the binding limitation. Under `tip_to_any_node`, direct recall
saturates very high and early while gap recall trails at every configuration.
At the compact r=20/k=2 recommendation the worst direct recall is 97.71%
(794491) but the worst gap recall is only 66.67% (794495); at r=50/k=2 worst
direct is 99.05% versus worst gap 84.02% (794495); and at the r=75/k=4 target
worst direct is 99.43% versus worst gap 91.09% (789202). Even at the r=100/k=8
ceiling the worst direct recall is 99.58% while the worst gap recall is 93.86%.
In every case the worst-brain recall is set by gap coverage, so improving the
overall floor means improving gap-partner reachability, not direct reachability.

This asymmetry is visible in the right-hand truth-type panels of all six
figures. In `figures/cross_dataset_candidate_pool_diagnostics.png` (right panel)
and in each per-dataset third panel
(`per_dataset/789202/figures/candidate_pool_diagnostics.png`,
`per_dataset/794495/figures/candidate_pool_diagnostics.png`,
`per_dataset/802449/figures/candidate_pool_diagnostics.png`), the points sit
below the identity line, i.e., gap recall < direct recall; for the two easy
brains (`per_dataset/794491/figures/candidate_pool_diagnostics.png`,
`per_dataset/794493/figures/candidate_pool_diagnostics.png`) the points sit
much closer to the diagonal, which is why those brains reach the highest
ceilings.

Direct and gap truth sets may overlap, so their missed counts are not additive
and should not be summed into a single "total missed" figure. The evidence does
not state that the two sets are disjoint, so only the per-kind recall values are
reported and no combined miss count is inferred.

## Limitations and next experiment

The overriding limitation is scope: every recall number here is a pre-scoring
enumeration ceiling over reachable truth pairs. No feature is computed, no
classifier is trained, and no end-to-end split-detection performance is
measured. Candidate prevalence describes pool dilution, not precision, and
candidate counts describe downstream feature/scoring workload, not a full
runtime budget. Recall is also bounded by which truth pairs are "reachable" at
all; pairs outside the tested radii or unreachable by the pairing rules are
excluded from the denominator and cannot be recovered by scoring. Results are
specific to MCL 10 and these five brains, with 794495 (and secondarily 802449)
setting the floor.

Two gaps in the grid define the smallest informative follow-ups. First, the
all-to-all pairing rule `any_node_to_any_node` was not evaluated
(`any_node_to_any_node_evaluated: false`); because gap truth is the binding
limitation and interior-to-interior partners are exactly what tip-anchored rules
can miss, the single most informative next experiment is to add
`any_node_to_any_node` at a small radius/k grid (e.g., r in {20, 50, 75} and k
in {2, 4}) on the limiting brains 794495 and 802449, to test whether it raises
the gap-recall floor beyond `tip_to_any_node` at acceptable extra workload.
Second, no global-cap sweep was run (only `none`); once a promising high-recall
configuration is fixed, a post-enumeration cap sweep on those same two brains
would quantify how far the pool can be trimmed before the recall ceiling drops,
directly informing the eventual feature/scoring budget. Both follow-ups can be
scoped to the two limiting brains rather than all five to keep the experiment
minimal.

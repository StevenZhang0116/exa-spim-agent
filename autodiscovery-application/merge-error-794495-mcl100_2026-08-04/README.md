# Merge detector — from run `merge-error-794495-mcl100_2026-08-04`

Logistic-regression merge detector built by combining the **10 confirmed
hypotheses** of one AutoDiscovery run into a single per-segment probability.

## Provenance

| | |
|---|---|
| Origin run | `autodiscovery/merge-error-794495-mcl100_2026-08-04.json` (50 hypotheses ingested) |
| Run report | `autodiscovery/merge-error-794495-mcl100_2026-08-04.summary.md` |
| Origin brain | 794495, `mcl100` — `cache/dataset_cache_794495_mcl100_add.pkl` |
| Hypotheses used | the 10 returned by `--rank-by posterior-surprise --direction positive --top 10` |
| Run's own verdicts | REPRODUCED 10/10 · GENERALIZES 6, PARTIAL 4 (on brain 794493) · SOUND 2, WEAK 6, MINOR 2 |
| Built by | hand (2026-08-03), before `agentic/run_detector_build_workflow.py` existed; that workflow now encodes the same procedure |

Rationale for combining them: the run's cross-cutting caveat was a
**precision problem** — several features have high recall but very low precision
alone (caliber asymmetry: 90 % recall at ~8 % precision), and the report itself
concludes each "is best used as one component of an ensemble rather than as a
standalone threshold." This script is that ensemble.

## Feature ↔ hypothesis mapping

Feature names are the CSV column names. "Default" is the value imputed when the
feature is undefined for a segment (no junction, no component, too small).

| Hypothesis | Verdict | Feature column | Default | Computed in |
|---|---|---|---|---|
| id 1 — windowed path tortuosity | **SOUND** AUC 0.94 | `max_windowed_tortuosity` | 1.0 | phase B |
| id 4 — caliber (radius) asymmetry | **SOUND** AUC 0.91 | `caliber_asymmetry` | 0.0 | phase C |
| id 20 — pseudo-diameter tortuosity | WEAK AUC 0.93 | `pseudo_diam_tortuosity` | 1.0 | phase D |
| id 8 — max inter-node edge jump | MINOR AUC 0.81 | `max_edge_jump` | 0.0 | phase A |
| id 3 — min junction angle | WEAK AUC 0.92 | `min_junction_angle` | 180.0 | phase C |
| id 13 — junction tortuosity variance | WEAK | `junction_tort_var` | 0.0 | phase C |
| id 29 — curvature-spike density | MINOR → UPHELD¹ | `curvature_spike_density` | 0.0 | phase D |
| id 50 — volumetric fill factor | WEAK | `log10_fill_factor` | column mean | phase E |
| id 14 — Sholl bimodality (GMM BIC Δ) | WEAK | `bimodality_score` | 0.0 | phase E |
| id 16 — terminal-branch tortuosity | WEAK | `log10_terminal_tortuosity` | 0.0 | phase D |

¹ id 29's original Welch t-test was wrong (zero-inflated rare-count rate); the
run's test-fixer re-ran it as Mann-Whitney U + negative-binomial GLM and the
finding was **UPHELD** and upgraded PARTIAL → GENERALIZES. The feature is kept.

Traversal phases (the script loads the pkl once and walks each structure once):
**A** vectorised pass over the edge array · **B** maximal degree-2 chain
decomposition · **C** one pass over degree-≥3 junctions · **D** one pass per
connected component · **E** per-segment aggregation.

## Model

`StandardScaler` → `LogisticRegression(class_weight='balanced', C=0.1)`.

- `class_weight='balanced'` — the classes are 97:1 (98 merged / 9 525 clean);
  without it the fit collapses to the majority class.
- `C=0.1` — with only **98 positives**, moderate regularization is what keeps a
  10-feature fit from memorizing them.
- Evaluation: 5-fold `StratifiedKFold` (preserves the merge fraction per fold).

## Running it

```bash
conda activate panda                     # required: unpickles SkeletonGraph
RUN=autodiscovery-application/merge-error-794495-mcl100_2026-08-04
python $RUN/merge_site_logistic_detector.py cache/dataset_cache_794495_mcl100_add.pkl
```

**Must run on a compute node** — the pkl OOMs a login node (exit 137). ~3 min,
80 GB. Measured 2026-08-03: 65 s pkl load + 118 s feature extraction.

Add `--exclude-empty` for the deflated/honest evaluation (see caveat 1).
Point it at `cache/dataset_cache_794493_mcl100_add.pkl` to test cross-brain.

## Results on 794495 (2026-08-03)

`merge_detector_794495.csv` — 9 623 rows × 13 columns (10 features +
`segment_id`, `is_merge`, `merge_probability`).

**Discrimination**

| Scope | n | CV ROC-AUC | CV Avg Precision |
|---|---|---|---|
| Has-component only (`--exclude-empty`) — **the honest number** | 3 227 | **0.8411 ± 0.0326** | 0.2632 |
| All adjudicable (shipped default) — inflated, see caveat 1 | 9 623 | 0.9462 ± 0.0156 | 0.2721 ± 0.0616 |

**Threshold sweep** (full-data, all-adjudicable fit)

| Threshold | Recall | Precision | N flagged |
|---|---|---|---|
| 0.5 | 0.898 (88/98) | 0.079 | 1 113 |
| 0.8 | 0.776 (76/98) | 0.106 | 717 |
| 0.9 | 0.531 (52/98) | 0.207 | 251 |

**Coefficients** (z-scored). The two fits disagree, which is the story:

| All-adjudicable fit | | Has-component fit | |
|---|---|---|---|
| `max_edge_jump` | **+1.727** | `min_junction_angle` | −0.723 |
| `min_junction_angle` | −0.517 | `caliber_asymmetry` | +0.299 |
| `bimodality_score` | +0.207 | `bimodality_score` | +0.221 |
| `max_windowed_tortuosity` | +0.089 | `log10_terminal_tortuosity` | +0.190 |

## Caveats

**1. The headline 0.946 is inflated by +0.105. Prefer 0.841.**
6 396 of the 9 623 adjudicable segments have **no fragment component** at all —
they appear in `gt_node_canonical_label` but were dropped from the skeleton by
the `min_cable_length = 100 µm` filter. Every one of their features sits at its
default, and **none of them is ever a merge**, so they form a trivially
separable class. `max_edge_jump` becomes the single best "does this segment even
exist" indicator (its default is `0.0`, and exactly 6 396 rows have
`max_edge_jump == 0.0`), which is why it dominates the all-adjudicable fit at
+1.727 while dropping out of the top four once the empty segments are removed.

Measured directly: `max_edge_jump` alone scores AUC **0.9385** over all 9 623
segments but **0.8128** over the 3 227 real ones — and 0.8128 is exactly the AUC
the origin run reported for id 8, on "98 merged / 3 129 clean". The original
hypothesis was correctly scoped; expanding it to the full adjudicable universe
is what inflated it.

This is the same defect the run's own verifier flagged on **id 3** (imputing
180° for junction-less segments partly inflated its headline AUC "by the
trivially-separable *has a junction at all* signal"). The script reproduces that
error class at the component level, for every feature at once. `--exclude-empty`
is the fix.

**2. Low precision is partly sparse GT, not only model error.**
At threshold 0.9, 199 of the 251 flagged segments have no merge label. Their
geometry is indistinguishable from the confirmed merges — `max_edge_jump`
8.5–50.5 µm, `min_junction_angle` 17–59°, `caliber_asymmetry` 0.97–1.00. Only
~30 neurons are traced per brain, so a genuine merge outside a traced neuron
*cannot* be labeled. Reported precision is therefore a floor, not an estimate.
Confirming these requires the raw image or a second brain, not this table.

**3. In-sample coefficients, cross-validated scores only.**
The coefficient table and threshold sweep are fitted on all the data; only the
ROC-AUC / AP figures are cross-validated. Nothing here has been tested on a
held-out brain — the origin run's generalization pass covered the individual
features on 794493, not this combined model.

**4. Feature coverage is uneven.** Of 9 623 segments: 3 227 have a component,
1 104 have any junction (so `caliber_asymmetry`, `min_junction_angle`,
`junction_tort_var` are at default for 88 % of rows), 1 672 have a non-zero
bimodality score.

**5. Segment-level, not site-level.** The output flags *which* segment is
merged, not *where*. The per-segment values stored are maxima over the segment,
so the argmax node — the actual candidate merge site — is computed and then
discarded. Localizing it (highest-asymmetry degree-3 node, or the windowed
tortuosity peak) is the natural next step and is **not** implemented here.

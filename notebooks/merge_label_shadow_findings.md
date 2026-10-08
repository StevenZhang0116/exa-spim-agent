# Cache merge labels vs canonical metrics: node-count threshold finding

Date: 2026-10-07. All numbers below were produced on n238 (panda env) from the
`dataset_cache_<brain>_mcl100_add.pkl` caches and the canonical `metrics_out/<brain>/*/`
results. No cache, feature table, context cache or detector was modified.

## Summary

- The cache's merge labels (`gt_merge_sites`, `gt_merge_labels`) recover only **46–89%**
  of the canonical merge sites (site recall @ 50 µm), with precision 0.83–1.00.
  The six older-microscope brains are the worst (0.46–0.74, except 754613).
- The main cause is a **unit mismatch**. `canonical_labeling` ports three merge
  thresholds from `segmentation_skeleton_metrics` as *GT node counts* (50 nodes). The
  canonical package reads raw GT SWCs (≈1.0 µm between nodes), so 50 nodes ≈ 50 µm. The
  cache's GT graph is resampled (≈4.0–4.4 µm between nodes), so the same 50 nodes ≈
  200–220 µm. Short genuine merges are therefore rejected as "pass-throughs".
- Expressing the thresholds as the same cable length (50 µm → 11–13 cache nodes) raises
  site recall to **0.80–0.94** on all 11 brains, with precision 0.83–0.96.
- Downstream impact on the frozen merge detector is **negligible**: positives grow by
  8–24%, but almost all new positives rank far down the detector's list. P@500 / P@2000
  change by ≤ 0.005, and the brain ordering is unchanged.
- **Recommendation:** do not rebuild the caches. Keep the current labels for now so that
  old and new brains share one label definition, and record the bias described here.
  Use the corrected labels if merge candidate generation is redesigned.

## 1. A verifier bug that had to be fixed first

`notebooks/verify_add_cache_merge_info.py` reported a hard FAIL ("cache appears to be in
'World reversed' frame") on every brain, including the old brain 789202. The bug was in
the verifier, not in the caches:

- In `merge_sites.csv`, `Voxel` is (x, y, z). `Voxel * anisotropy` is the (x, y, z) µm
  frame used by the cache.
- The package's `World` column equals `(Voxel_z*0.748, Voxel_y*0.748, Voxel_x*1.0)`. The
  axes are reversed but the anisotropy is applied in the original order, so this column
  is not a physical frame.

The verifier now uses `Voxel * anisotropy` as the reference and keeps the `World` column
only as a diagnostic candidate. With this fix all 11 brains PASS, and the same-segment
placement median is 2.4–4.5 µm.

| brain | cache sites | canonical sites | placement median (µm) | site recall @50µm | site precision @50µm |
|---|---|---|---|---|---|
| 747807 | 102 | 160 | 4.5 | 0.531 | 0.833 |
| 750318 | 109 | 162 | 3.5 | 0.593 | 0.899 |
| 751473 | 108 | 164 | 3.8 | 0.616 | 0.935 |
| 754610 | 14 | 26 | 3.1 | 0.462 | 0.857 |
| 754612 | 18 | 23 | 3.1 | 0.739 | 0.944 |
| 754613 | 9 | 9 | 2.4 | 0.889 | 1.000 |
| 789202 | 71 | 99 | 3.3 | 0.687 | 0.958 |
| 794491 | 104 | 148 | 3.5 | 0.669 | 0.942 |
| 794493 | 39 | 51 | 3.2 | 0.725 | 0.949 |
| 794495 | 134 | 168 | 4.1 | 0.744 | 0.925 |
| 802449 | 945 | 1279 | 3.5 | 0.697 | 0.935 |

Coordinates agree; coverage does not. The merge sweep in
`notebooks/merge_candidate_pool_sweep_outputs_with_oldscope/` does **not** measure this gap,
because its denominator is the cache's own `gt_merge_sites`.

## 2. Diagnosis of the missing canonical sites

Each canonical site with no cache site within 50 µm was classified on five brains:

| brain | canonical | matched | unmatched | segment present, not flagged as merge | flagged, no site here | segment absent (mcl filter) |
|---|---|---|---|---|---|---|
| 789202 | 99 | 68 | 31 | 17 | 11 | 3 |
| 754610 | 26 | 12 | 14 | 13 | 1 | 0 |
| 754612 | 23 | 17 | 6 | 6 | 0 | 0 |
| 750318 | 162 | 96 | 66 | 55 | 7 | 4 |
| 794495 | 168 | 125 | 43 | 22 | 20 | 1 |

What the "present, not flagged" sites look like:

- The merged arm exists in the cache. 98% of these segments have fragment nodes > 50 µm
  from all GT, and the segment passes within ~2 µm of the canonical site.
- The same-label GT component at these sites has a median of only 26–31 cache nodes
  (≈110–135 µm); 70–88% have fewer than 50 nodes. Matched sites have medians of 162–605.
- Measured node spacing: raw GT SWCs ≈ 1.0–1.06 µm (789202, 750318); cache GT graphs
  3.99–4.36 µm (all 11 brains).

The affected thresholds in `agentic_neuron_proofreader/data_modules/canonical_labeling.py`:

| constant | used by | canonical meaning | cache meaning |
|---|---|---|---|
| `MERGE_MIN_NODES = 50` | node-count merge rule (`merge_labels`), `summarize` | > 50 µm per neuron | > ~200 µm per neuron |
| `MERGE_PASSTHRU_MIN_CC = 50` | geometric walk pass-through test | component < 50 µm | component < ~200 µm |
| `min_gt_nodes=MERGE_MIN_NODES` | two-GT junction audit | 50 µm | ~200 µm |

`MERGE_MIN_NODES` is bound as a default argument in `merge_labels` and
`audit_junction_gt_connections`, so changing the module constant alone is not enough.

Smaller contributors, not addressed here:

- The cache walks against the union of all GT neurons; the canonical package walks per
  GT neuron.
- The canonical package snaps sites to a nearby branching node.
- The canonical package dedups using the mislabeled `World` column.
- Fragments shorter than the 100 µm MCL are absent from the cache.

## 3. Shadow relabel (stages 1–2)

`scripts/shadow_merge_relabel.py` recomputes merge labels and sites from each `_add.pkl`.
It reuses the stored GT node labels, makes no cloud reads and never writes the cache.
All three thresholds are set to `round(50 µm / median GT edge length)`. With
`--check-reproduce`, the original thresholds reproduced the stored sites and labels
**exactly** on all 11 brains.

| brain | nodes (scaled) | sites (stored → scaled) | canonical sites | site recall | site precision | canonical merge segments flagged |
|---|---|---|---|---|---|---|
| 747807 | 12 | 102 → 161 | 160 | 0.53 → **0.86** | 0.83 → 0.86 | 0.49 → 0.87 |
| 750318 | 11 | 109 → 171 | 162 | 0.59 → **0.90** | 0.90 → 0.87 | 0.56 → 0.91 |
| 751473 | 12 | 108 → 160 | 164 | 0.62 → **0.87** | 0.94 → 0.89 | 0.64 → 0.91 |
| 754610 | 12 | 14 → 24 | 26 | 0.46 → **0.81** | 0.86 → 0.83 | 0.43 → 0.86 |
| 754612 | 12 | 18 → 23 | 23 | 0.74 → **0.87** | 0.94 → 0.87 | 0.67 → 0.78 |
| 754613 | 12 | 9 → 9 | 9 | 0.89 → 0.89 | 1.00 → 1.00 | 1.00 → 1.00 |
| 789202 | 13 | 71 → 84 | 99 | 0.69 → **0.80** | 0.96 → 0.94 | 0.76 → 0.89 |
| 794491 | 12 | 104 → 132 | 148 | 0.67 → **0.83** | 0.94 → 0.93 | 0.70 → 0.87 |
| 794493 | 12 | 39 → 50 | 51 | 0.73 → **0.94** | 0.95 → 0.96 | 0.69 → 0.93 |
| 794495 | 12 | 134 → 151 | 168 | 0.74 → **0.84** | 0.93 → 0.93 | 0.77 → 0.91 |
| 802449 | 12 | 945 → 1143 | 1279 | 0.70 → **0.83** | 0.94 → 0.92 | 0.73 → 0.92 |

Cache-only `% Merged Edges` rises by only 0.03–1.2 points per brain. That metric is
dominated by long merges, so edge-level metrics are essentially unaffected. Split labels
depend only on GT node segment labels and are not affected by these thresholds (by code
inspection; not separately measured).

The older-microscope brains gain the most, which closes most of their gap to the old
brains. The remaining recall gap (0.80–0.94) comes from the smaller contributors listed in
section 2.

## 4. Downstream impact (stage 3)

`scripts/shadow_merge_impact.py` rebuilds the frozen merge detector's candidate universe
and labels with the detector's own `build_sample_universe`. The detector is
`merge-error-794495-mcl100_2026-08-04` with policy `junction|nms=20|r=150` (positive =
geodesic ≤ 20 µm). It runs once with stored sites and once with scaled sites, and scores
the table's frozen `detector_score` under both. Stored-site labels reproduced each table's
`evaluator_labels.npy` exactly. Only the five old brains have tables.

| brain | candidates | positives (stored → scaled) | flips 0→1 / 1→0 | AP | P@500 | P@2000 | new-positive median rank | new positives in top 2000 |
|---|---|---|---|---|---|---|---|---|
| 789202 | 43,737 | 62 → 71 | 9 / 0 | 0.012 → 0.012 | 0.012 → 0.012 | 0.016 → 0.017 | 5,853 | 2 |
| 794491 | 203,956 | 88 → 111 | 23 / 0 | 0.002 → 0.002 | 0.004 → 0.004 | 0.003 → 0.003 | 62,329 | 0 |
| 794493 | 74,051 | 29 → 33 | 4 / 0 | 0.004 → 0.003 | 0.006 → 0.006 | 0.005 → 0.005 | 10,159 | 0 |
| 794495 | 58,120 | 114 → 123 | 9 / 0 | 0.130 → 0.122 | 0.080 → 0.080 | 0.038 → 0.040 | 27,945 | 2 |
| 802449 | 299,339 | 825 → 973 | 149 / 1 | 0.045 → 0.044 | 0.096 → 0.098 | 0.070 → 0.075 | 75,051 | 10 |

Fraction of sites the frozen policy reaches (claim ≤ 150 µm / positive ≤ 20 µm):

| brain | stored sites | scaled sites |
|---|---|---|
| 789202 | 0.92 / 0.87 | 0.88 / 0.85 |
| 794491 | 0.91 / 0.84 | 0.92 / 0.83 |
| 794493 | 0.85 / 0.74 | 0.74 / 0.66 |
| 794495 | 0.93 / 0.85 | 0.89 / 0.82 |
| 802449 | 0.93 / 0.87 | 0.92 / 0.85 |

Interpretation:

- The recovered merges are mostly short merges away from junctions. The junction-anchored
  policy reaches them less often, and the detector ranks them low.
- Merge Precision@K, AP and the per-brain ordering are therefore essentially unchanged.
- Not measured: whether recent evolution `best_scorer`s change order under the corrected
  labels. Given the size of the P@K changes, this is unlikely.

## 5. Decisions and open items

- **No rebuild.** The cost would be: relabel 11 `_add.pkl`, re-record the image-alignment
  receipts, rebuild feature tables and context caches, and possibly refit the merge
  detector and re-run evolution baselines. This is not justified by the measured impact.
- **Known bias.** Cache merge labels are conservative relative to canonical (site recall
  0.46–0.89). About 10–20% of true merge sites are invisible to the current merge
  pipeline (candidate policy plus labels).
- **New brains.** Build feature tables and context caches from the current caches, so all
  11 brains share one label definition.
- **If the fix is adopted later:** make the thresholds length-based in `canonical_labeling`
  (keeping the old behaviour behind a flag), relabel all brains together, and add a dated
  entry to `proofreader_evolve/WORKFLOW_REVISION.md`.

## Reproduction

```bash
conda activate panda   # compute node, from exa-spim-agent/
python -u notebooks/verify_add_cache_merge_info.py --all --mcl 100
python -u scripts/shadow_merge_relabel.py --all --check-reproduce
python -u scripts/shadow_merge_impact.py --brains 789202 794491 794493 794495 802449
```

Outputs (gitignored):

- `notebooks/merge_label_shadow/<brain>_mcl100.json` and `summary.csv` (stages 1–2)
- `notebooks/merge_label_shadow/impact/` (stage 3)
- logs in `notebooks/log/shadow_merge_relabel_all.log` and `shadow_merge_impact.log`

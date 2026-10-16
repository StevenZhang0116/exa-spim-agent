# Rebuild after the merge relabel and the split pool change (deferred, 2026-10-15)

Two benchmark inputs changed on 2026-10-15. The caches and the code are current; every
derived artifact below is stale. The rebuild is **deferred on purpose**: AutoDiscovery will
be rerun on the revised specifications and new frozen split and merge detectors will be
built from those reports, which changes the feature tables again. Rebuild once, after the
new detectors exist, in the order of section 4.

## 1. What changed

| Change | Where | Effect on the data |
|---|---|---|
| Merge labels recomputed with length-based thresholds (`length_v1`: 50 um of GT cable, 11 to 13 cache nodes instead of 50) | all eleven `cache/dataset_cache_<b>_mcl100_add.pkl`, relabeled in place by `python -m labeled_cache --policy length_v1 --check-reproduce` (logs `notebooks/log/labeled_cache_20261015_*.log`) | merge labels and `gt_merge_sites` grew by 8 to 60% per brain (pooled canonical site recall 0.67 to 0.84); GT node labels, split labels, graphs and node ids unchanged; cache file identity (size, mtime, inode) changed; `gt_label_policy` recorded |
| Split candidate pool `tip_to_any_node|r=50|k=2` to `tip_to_any_node|r=80|k=2` | `proofreader_evolve/candidate_pools.json`, applied by `harness/candidate_pools.apply_pool_policy` in `native_pool.build_native_table`; `POOL_VERSION` detector-native-v2; detector-build default review now the eleven-brain sweep | every split pool gets about 30% more rows and a new pool hash; merge pools keep their rows but their table directory digest changes with the version bump |

Evidence: `notebooks/merge_label_shadow_findings.md` (merge), `notebooks/split_candidate_pool_sweep_outputs_with_oldscope/mcl100_5f92a29e40d94a90` (split; worst brain 754612 at 0.900 for r=80 against 0.866 for r=50).

## 2. Stale now

| Artifact | Brains | Why |
|---|---|---|
| `proofreader_evolve/feature_tables/<b>/<digest>/` (merge and split) | all eleven | merge: labels inside; split: pool rows; both: `source_cache` identity and the v2 provenance digest |
| `feature_tables/<b>/<digest>.availability/` sidecars | all eleven | bound to the tables |
| `markdowns/rerank_companion_{794495,802449}_merge.{npz,json}` | 794495, 802449 | labels and `source_cache.size` (merge). The split companions were regenerated on 2026-10-15 with `--pool-only` for r=80 (section 3); the r=50 ones are in `markdowns/archive/companions_20261014/` |
| `configs/image_alignment.json` receipts | all eleven | bound to the cache file identity |
| `proofreader_evolve/context_cache/<b>/split/` | all eleven | new pool hash, new band rows |
| `proofreader_evolve/context_cache/<b>/merge/` | all eleven, to be checked | identity includes `features_sha256` of the rebuilt merge table; valid only if the rebuilt `features.pkl` is byte-identical |
| `proofreader_evolve/descriptor_bank/<b>/` | all eleven | keyed by pool hash; orphaned, delete |
| `markdowns/merge_features_for_proofreader_evolve.md` §1 counts, §2 baselines | the 794495 merge companion | quoted from the old companion. The split specification was rewritten for r=80 on 2026-10-15 (rule, counts, gap baseline AP 0.826) |
| `notebooks/evaluable_precision_test/`, `notebooks/split_canonical_gap/` (pool column) | all | read tables / the old split pool |
| Frozen detectors `autodiscovery-application/*` | | not invalidated; they are replaced by the new builds |
| Evolution runs before 2026-10-15 | | still executable, no longer comparable: the next run's seed baseline is the reference |

Already regenerated on 2026-10-15 (merge-label dependent notebook outputs): `merge_diagnosis/mcl100` (+ `followup`), `merge_site_counts/mcl100_length_v1`, `verify_stats` (mcl100), `merge_canonical_gap`, both merge candidate-pool sweeps (`notebooks/log/rerun_merge_artifacts_20261015.log`).

## 3. Before the rebuild: the new frozen detectors

1. Rerun AutoDiscovery on the revised specifications with the companions in `markdowns/`. Split: regenerated 2026-10-15 for r=80 with `export_rerank_companion --kinds split --pool-only` (cache only, no tables needed). Merge: regenerate the same way once the merge pool is settled. Companions do not depend on any detector's features or scores, so they are exported now, not in section 4.
2. Audit the runs (`agentic/run_discovery_workflow.py --direction predictive --companion ...`). An audit of a split run recorded against the r=50 companion is judged against that companion; it is internally consistent and does not need the new pool.
3. Build the detectors with `agentic/run_detector_build_workflow.py`: the split policy step now reads the eleven-brain review and selects `tip_to_any_node|r=80|k=2` at the 0.90 floor; the merge-site build reads `notebooks/merge_candidate_pool_sweep_outputs_with_oldscope/mcl100_dcbde4ab2a6f6c44/recommended_policy.json` (eleven brains, regenerated 2026-10-15, `junction|nms=20|r=150`, status `target_unmet_best_effort`: worst brain 794493 at 0.74 against the 0.80 default floor, so pass `--merge-site-min-worst-brain-site-recall 0.74` or decide the second candidate family first; see TODO "Merge labels are conservative", section 6.4 of the findings).
4. Point `proofreader_evolve/cli/precompute_error_scores.py` (`DEFAULT_MERGE_DIR`, `DEFAULT_SPLIT_DIR`) at the new application directories. Keep `candidate_pools.json` explicit for split even when the new detector embeds the same rule: the benchmark rule is declared there, the detector is a scorer.

## 4. Rebuild order (per brain, one brain per process, `srun -n1` on a compute node)

```bash
conda activate panda
rm -rf proofreader_evolve/feature_tables/<b> proofreader_evolve/context_cache/<b>/split proofreader_evolve/descriptor_bank/<b>
python -u proofreader_evolve/prepare_feature_tables.py --brains <b> --mcl 100 --threads all        # about 3 h for 794495; split pool 30% larger
python -u -m proofreader_evolve.cli.precompute_availability --brains <b> --mcl 100                  # 1.5 min
python - <<'PY'                                                                                    # 794495 and 802449: the tables must reproduce the companions
import json, pathlib
for kind in ("merge", "split"):
    comp = json.load(open(f"markdowns/rerank_companion_<b>_{kind}.json"))["pool_sha256"]
    tables = [json.load(open(p))["pool_sha256"] for p in pathlib.Path("proofreader_evolve/feature_tables/<b>").glob("*/meta.json")]
    print(kind, "companion pool matches a table:", comp in tables)
PY
python -m proofreader_evolve.cli.check_image_alignment --brains <b> --mcl 100 --out figs/candidate_image_alignment_<date>
python -m proofreader_evolve.cli.check_image_alignment --brains <b> --mcl 100 --out figs/candidate_image_alignment_<date> --record-reviewed --review-note "<what the six overlays showed>"
python -u -m proofreader_evolve.cli.precompute_context_cache --brains <b> --mcl 100 --readers 16 --band-rows 50000 --labeled-extension   # split: full rebuild, about 2 h; merge: rebuilt only if its identity changed
python -m proofreader_evolve.cli.preflight --brains <all eleven> --train-brains 794495 802449 --mcl 100 --context-cache proofreader_evolve/context_cache
```

Budget: about 3 h x 11 for the tables and 2 h x 11 for the split context caches, serial. Tables can
run two or three brains at once on a 400 GB node (each is its own subprocess); context caches read
S3 with 16 readers, one brain at a time. Allocate several days.

## 5. After the rebuild, once

- `python -u notebooks/split_canonical_gap.py --measure --brains <all>` then `--plot`: coverage of the canonical split pairs by the r=80 pool (r=50 covered 95% of the pairs whose segments both exist).
- Merge specification (split done 2026-10-15): from the new 794495 merge companion recompute the label counts and the degree baseline (tie-aware AP) with the specification's own scoring code; archive the previous text under `markdowns/archive/` and add the lineage row in `markdowns/README.md`.
- `python -u scripts/evaluable_precision_test.py`.
- Label distribution figure: no rebuild needed. `python -u notebooks/label_counts.py` (counts straight from the
  caches and `candidate_pools.json`, `notebooks/label_counts/`), then `python notebooks/plot_label_availability.py`
  (writes `notebooks/label_availability_distribution.svg`). After the rebuild, `--source sidecars` cross-checks
  the same counts from the new sidecars.
- A dated `WORKFLOW_REVISION.md` entry with the first preflight and seed-baseline numbers on the rebuilt tables.

## 6. Until then

- Do not start evolution runs on the current tables: their labels (merge) and pools (split) no longer match the declared benchmark; `ensure_native_tables` will in any case refuse them (provenance digest changed) and ask for `prepare_feature_tables.py`.
- The caches themselves are final; nothing in this plan touches `cache/`.

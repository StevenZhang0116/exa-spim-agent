# Rebuild after the merge relabel and the candidate pool changes (deferred, 2026-10-15; updated 2026-10-16)

Two benchmark inputs changed on 2026-10-15 and the candidate rules changed on 2026-10-16. The caches and the code are current; every
derived artifact below is stale. The rebuild is **deferred on purpose**: AutoDiscovery will
be rerun on the revised specifications and new frozen split and merge detectors will be
built from those reports, which changes the feature tables again. Rebuild once, after the
new detectors exist, in the order of section 4.

## 1. What changed

| Change | Where | Effect on the data |
|---|---|---|
| Merge labels recomputed with length-based thresholds (`length_v1`: 50 um of GT cable, 11 to 13 cache nodes instead of 50) | all eleven `cache/dataset_cache_<b>_mcl100_add.pkl`, relabeled in place by `python -m labeled_cache --policy length_v1 --check-reproduce` (logs `notebooks/log/labeled_cache_20261015_*.log`) | merge labels and `gt_merge_sites` grew by 8 to 60% per brain (pooled canonical site recall 0.67 to 0.84); GT node labels, split labels, graphs and node ids unchanged; cache file identity (size, mtime, inode) changed; `gt_label_policy` recorded |
| Split candidate pool `tip_to_any_node|r=50|k=2` to `tip_to_any_node|r=80|k=2` | `proofreader_evolve/candidate_pools.json`; `POOL_VERSION` detector-native-v2 | every split pool gets about 30% more rows and a new pool hash |
| 2026-10-16: candidate rules defined once in `candidate_rules.py` (repository root); merge pool `junction+tip_to_shaft|nms=20|gap=20|k=2|r=150` | `candidate_pools.json` schema 2 declares both kinds; tables and companions enumerate with the module, detectors only score (`native_pool`, `POOL_VERSION` detector-native-v3); detector builds freeze the declared rule and copy the module beside the detector (no split advice stage, no merge sweep floor, no `--merge-*` / `--split-*` rule flags; `--candidate-pools` names the declaration) | merge pools gain tip-to-shaft rows (about 2.8 times the candidates, 1.9 times the GT-judgeable rows over the eleven brains) with `candidate_family` and `tip_gap_um`; split rows must stay identical (checked in section 3) |

Evidence: `notebooks/merge_label_shadow_findings.md` (merge, section 6.6 for the tip-to-shaft family), `notebooks/merge_shaft_reach/` (per-brain reach of the merge families), `notebooks/split_candidate_pool_sweep_outputs_with_oldscope/mcl100_5f92a29e40d94a90` (split; worst brain 754612 at 0.900 for r=80 against 0.866 for r=50).

## 2. Stale now

| Artifact | Brains | Why |
|---|---|---|
| `proofreader_evolve/feature_tables/<b>/<digest>/` (merge and split) | all eleven | merge: labels inside; split: pool rows; both: `source_cache` identity and the v2 provenance digest |
| `feature_tables/<b>/<digest>.availability/` sidecars | all eleven | bound to the tables |
| `markdowns/rerank_companion_{794495,802449}_merge.{npz,json}` | 794495, 802449 | labels, `source_cache.size` and, since 2026-10-16, the pool itself (tip-to-shaft rows, two new columns). The split companions were regenerated on 2026-10-15 with `--pool-only` for r=80 (section 3); the r=50 ones were deleted on 2026-10-17 |
| `configs/image_alignment.json` receipts | all eleven | done 2026-10-18: re-recorded from the current caches (end of section 3; evidence `figs/candidate_image_alignment_20261018/`). Stale again only if a pkl is rewritten |
| `proofreader_evolve/context_cache/<b>/split/` | all eleven | new pool hash, new band rows |
| `proofreader_evolve/context_cache/<b>/merge/` | all eleven | new merge pool (2026-10-16): full rebuild |
| `proofreader_evolve/descriptor_bank/<b>/` | all eleven | keyed by pool hash; orphaned, delete |
| `markdowns/merge_features_for_proofreader_evolve.md` candidate rule, §1 counts, §2 baselines | the 794495 merge companion | the rule text predates the tip-to-shaft family and the counts the old companion. The split specification was rewritten for r=80 on 2026-10-15 (rule, counts, gap baseline AP 0.826) |
| `notebooks/merge_candidate_pool_sweep_outputs_with_oldscope/mcl100_dcbde4ab2a6f6c44` | all eleven | junction rules only; deleted 2026-10-17, replaced by the `--tip-gaps` rerun `mcl100_96e1b6bbfbf36f5c` (section 3) |
| `notebooks/evaluable_precision_test/`, `notebooks/split_canonical_gap/` (pool column) | all | read tables / the old split pool |
| `notebooks/merge_canonical_gap/` candidate columns | all eleven | done 2026-10-17: the script enumerates the declared rule with `candidate_rules.py` (no table) and was rerun; every brain equals the J+S20 canonical reach of `notebooks/merge_shaft_reach/` (pooled 0.881 at 20 um, 0.951 at 150 um) |
| Frozen detectors `autodiscovery-application/*` and all AutoDiscovery results in `autodiscovery/` | | deleted 2026-10-17 (trained on pre-relabel labels and the old pools); companions and the merge sweep no longer need a detector, and `DEFAULT_MERGE_DIR` / `DEFAULT_SPLIT_DIR` are unset until the new builds exist |
| Evolution runs before 2026-10-15 | | still executable, no longer comparable: the next run's seed baseline is the reference |

Already regenerated on 2026-10-15 (merge-label dependent notebook outputs): `merge_diagnosis/mcl100` (+ `followup`), `merge_site_counts/mcl100_length_v1`, `verify_stats` (mcl100), `merge_canonical_gap`, both merge candidate-pool sweeps (`notebooks/log/rerun_merge_artifacts_20261015.log`).

## 3. Before the rebuild: the new frozen detectors

0. Check `candidate_rules.py` on real data (compute node, panda, `srun -n1`; read-only except the outputs named):
   - split rows unchanged: export pool-only split companions into a scratch directory and compare `pool_sha256` with the 2026-10-15 exports (794495 `32c4d86b...`, 802449 `7efd1e40...`):
     `python -u -m proofreader_evolve.cli.export_rerank_companion --brain 794495 --kinds split --pool-only --out-dir claude_scratch/companion_check` (then 802449) — passed 2026-10-17, both hashes identical (`notebooks/log/split_pool_parity_20261016.log`);
   - merge evidence and parity: `python -u notebooks/merge_candidate_pool_sweep.py --mcl 100 --nms 20 --claim-radii 20 30 150 --tip-gaps 10 20 40 80 --policy-target 0.90 --output-dir notebooks/merge_candidate_pool_sweep_outputs_with_oldscope`; per brain the `junction+tip_to_shaft|nms=20|gap=20|k=2` candidate count must equal `families["J+S20"]["candidates"]` in `notebooks/merge_shaft_reach/<b>_mcl100.json`, and its `RESULTS.md` lists the declared rule per brain. Record the bundle in the merge `evidence` of `candidate_pools.json` — passed 2026-10-17: `mcl100_96e1b6bbfbf36f5c` reproduces J and J+S10/20/40/80 on all eleven brains (J+S20 pooled 4,437,389), recorded in the declaration.
1. Rerun AutoDiscovery on the revised specifications with the companions in `markdowns/`. Split: regenerated 2026-10-15 for r=80 with `export_rerank_companion --kinds split --pool-only` (cache only, no tables needed). Merge: the pool is settled (2026-10-16); the stale merge companions (pre-relabel, junction pool) were deleted on 2026-10-17; regenerated the same way (`--kinds merge --pool-only`, 794495 and 802449; 192,177 and 553,193 rows) and the merge specification rewritten on 2026-10-17 (two candidate families, relabeled counts, degree and cable-length baselines from `../claude_scratch/merge_baselines_20261017.py`, per-family reporting). Discovery and detector fitting stay on 794495. Companions do not depend on any detector's features or scores, so they are exported now, not in section 4.
2. Audit the new runs (`agentic/run_discovery_workflow.py --direction predictive --companion ...`). The 2026-10-15 runs were deleted with the other AutoDiscovery results on 2026-10-17, so there is no older run to audit.
3. Build the detectors with `agentic/run_detector_build_workflow.py`. Both kinds freeze the rule declared in `proofreader_evolve/candidate_pools.json` (`--candidate-pools` to point elsewhere) and copy `candidate_rules.py` beside the detector; there is no recall floor to pass. The merge rule includes degree-2 tip-to-shaft candidates, so junction-local features must be defined or left NaN there (the build prompts say so).
4. Point `proofreader_evolve/cli/precompute_error_scores.py` (`DEFAULT_MERGE_DIR`, `DEFAULT_SPLIT_DIR`) at the new detectors' run directories (`autodiscovery-application/<RUN>/runs/<brain>-default/`, accepted since 2026-10-21 and bound to the script by the run's recorded detector SHA-256) or at the application directories when the run wrote beside the script. Tables record `detector_rule_matches`; it should be true for both new detectors.

Independent of steps 1 to 4 (cache only, no tables, no detector): re-record the image-alignment
receipts of all eleven brains from the current caches. The AutoDiscovery runs do not read the
receipts (they open the image named in the companion manifest), but every image read of
`proofreader_evolve` checks them. The native `check_image_alignment` CLI samples table rows and
needs detectors, so use the cache-based script (about 1 min per brain, serial):

```bash
python -u scripts/check_image_alignment_from_cache.py --mcl 100 --out figs/candidate_image_alignment_<date> \
    --brains 747807 750318 751473 754610 754612 754613 789202 794491 794493 794495 802449
# inspect the 66 overlays; write {brain: note} to figs/candidate_image_alignment_<date>/review_notes.json
python -u scripts/check_image_alignment_from_cache.py --mcl 100 --out figs/candidate_image_alignment_<date> \
    --brains 747807 750318 751473 754610 754612 754613 789202 794491 794493 794495 802449 \
    --record-reviewed --review-notes figs/candidate_image_alignment_<date>/review_notes.json
```

A later rewrite of a pkl (`labeled_cache` relabel, rebuild or `--refresh-image-path`) makes its
receipt stale again.

Done 2026-10-18 (`notebooks/log/image_alignment_20261018.log`): all 66 patches read, image URIs
and image identities unchanged. For the six older-microscope brains the same seed drew the same
nodes, and statistics and overlays were byte-identical to the 2026-10-06 review, which carries
over. The 30 overlays of 789202, 794491, 794493, 794495 and 802449 were inspected: no axis
swap, reflection or global offset; the skeleton signal exceeded every 12 um shifted control in
29 of 30 patches (794491 node 12298945, a bright featureless region, is the exception). The
notes are in `figs/candidate_image_alignment_20261018/review_notes.json`.

## 4. Rebuild order (one brain at a time on the interactive node; updated 2026-10-23)

Step 3.4 is done: since 2026-10-23 `DEFAULT_MERGE_DIR` / `DEFAULT_SPLIT_DIR` point at
`autodiscovery-application/new-merge-1_2026-10-19/runs/794495-with-figures` (logistic_l2) and
`autodiscovery-application/new-split-1_2026-10-20/runs/794495-with-figures` (hist_gradient_boosting),
both fitted on 794495; every Stage 3 entry point resolves them without `--merge-dir` / `--split-dir`.

Everything runs on the interactive node (n280 from 2026-10-23: 40 CPUs, 200 GB, ends
2026-10-31 18:18), in a terminal inside that allocation with panda; there is no Slurm script.
`proofreader_evolve/rebuild_brains.sh` runs the per-brain steps below for one brain after the
other, evolution brains first (794495 802449 789202 794493 794491, then 754613 754612 754610
750318 751473 747807), stops at the first failure and ends with a preflight over the brains of
the call. It prints `# OK brain <b>` per finished brain and `# OK` / `# FAILED` at the end; the
log is `proofreader_evolve/log/rebuild_brains_<timestamp>_<pid>.log.txt`, with each step's own log
beside it. Every step reuses finished work, so rerunning the same command after an interruption
(or on the next allocation) resumes at the first unfinished brain and kind; only the kind that
was cut off is built again from its start.

Once, before the first call, delete the stale artifacts (they never match the new digests, so
this frees disk and avoids confusion; do not repeat it before a resume):

```bash
for b in 747807 750318 751473 754610 754612 754613 789202 794491 794493 794495 802449; do
    rm -rf proofreader_evolve/feature_tables/$b proofreader_evolve/context_cache/$b/split \
           proofreader_evolve/context_cache/$b/merge proofreader_evolve/descriptor_bank/$b
done
```

Then, on the interactive node:

```bash
nohup bash proofreader_evolve/rebuild_brains.sh >/dev/null 2>&1 &   # all eleven brains; or name brains as arguments
tail -f "$(ls -t proofreader_evolve/log/rebuild_brains_*.log.txt | head -1)"
```

`IMAGE_WORKERS` (default -1, one per CPU), `READERS` (32) and `BAND_ROWS` (50000) override the
defaults. The closing preflight is the one this section asks for when the call covers all eleven
brains:

```bash
python -m proofreader_evolve.cli.preflight \
    --brains 747807 750318 751473 754610 754612 754613 789202 794491 794493 794495 802449 \
    --train-brains 794495 802449 --mcl 100 --context-cache proofreader_evolve/context_cache
```

What the script does per brain (the same commands serve a manual rerun of one step, panda):

```bash
python -u proofreader_evolve/prepare_feature_tables.py --brains <b> --mcl 100 --threads all --image-workers -1   # tables of both kinds, sidecars (about 1.5 min), single-brain preflight; -1 = one image worker per CPU
# 794495 and 802449 only: each table's pool_sha256 must equal its companion's
#   (markdowns/rerank_companion_<b>_<kind>.json); the script loads the tables the default detectors resolve to and stops on a mismatch
# image-alignment receipt: re-recorded from the cache at the end of section 3; redo it only if the pkl changed since
python -u -m proofreader_evolve.cli.precompute_context_cache --brains <b> --mcl 100 --readers 32 --band-rows 50000 --labeled-extension --threads all   # both kinds (new pools)
```

Since 2026-10-23 `prepare_feature_tables.py` writes the availability sidecars itself (through
`precompute_availability`) before its closing preflight. Before, that preflight required the
sidecars the next step would write, so a fresh table build always ended `FAILED`
(`log/prepare_feature_tables_20261013_151554_332376_313597.log.txt`) and a stop-at-first-failure
driver would have halted after the first brain's tables.

Budget. The detectors' image pass dominates now (the 2026-08 tables had no image features, so
the earlier "about 3 h for 794495" no longer holds). At the 16-worker rates of the 2026-10 builds
(split 19 rows/s, merge 32 rows/s) the image time per brain is the split pool
(`tip_to_any_node|r=80|k=2`, counts from the r=80 sweep) plus the merge pool (J+S20 counts from
`notebooks/merge_shaft_reach/`):

| brain | split rows | merge rows | image hours at 16 workers |
|---|---:|---:|---:|
| 747807 | 2,211,738 | 1,226,500 | 43 |
| 751473 | 1,992,222 | 835,393 | 36 |
| 794491 | 1,343,936 | 542,686 | 24 |
| 802449 | 1,286,517 | 553,193 | 24 |
| 750318 | 1,237,112 | 491,632 | 22 |
| 794495 | 689,677 | 192,177 | 12 |
| 794493 | 510,914 | 198,751 | 9 |
| 754610 | 401,056 | 191,083 | 7.5 |
| 754612 | 134,265 | 70,950 | 2.6 |
| 754613 | 107,076 | 78,253 | 2.3 |
| 789202 | 54,310 | 56,771 | 1.3 |

About 184 h of image pass in all, plus the graph passes (split 802449 took 1.9 h in the build),
the context caches (about 2 h for a split band of 50,000 rows) and the pkl loads. On one node
the brains run in series, so at the 16-worker rates the whole rebuild needs more than the
roughly 199 h the n280 allocation has left (2026-10-23 10:18 to 2026-10-31 18:18). The pass is
S3-bound, so 40 workers may run faster than the 16-worker rates: read the `rate=` lines of
794495 (first brain, 12 h at 16 workers) and extrapolate before counting on it. The order puts
the five evolution brains first (about 70 h at the 16-worker rates), so evolution can start
before the older-microscope brains finish; whatever is left resumes on the next allocation with
the same command. 200 GB held the 802449 extraction in the detector build; one brain at a time,
never two.

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

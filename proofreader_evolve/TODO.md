# proofreader_evolve TODO

Open items that are discussed but deliberately not implemented yet. Each entry
records the current behaviour, the measured cost of changing it, and what the
change touches, so the decision can be revisited without re-deriving the facts.
Dates are when the item was recorded.

## Context cache: extend the level-0 image tier to the whole band (2026-10-05)

**Status: implemented 2026-10-05** (`max_rows: None`, `ContextCacheBuilder.extend_tier`,
`precompute_context_cache --extend-tier level0`, tier-selective bank identities; build
ran 2026-10-05/06: 10 entries, 0 failures, 2 h 21 min, level-0 tier 1.55 -> 7.77 GB). The
notes below are kept as the design record; the remaining open idea is refreshing the
band with rows the evolving scorer pulls into the Top-K.

**Current behaviour.** `IMAGE_TIERS` in `harness/context_cache.py` caches the
level-1 image patch (30 um radius, 1.496 x 1.496 x 2.0 um voxels, 30 x 41 x 41)
for every row of the 20,000-row detector-ranked band, but the level-0 patch
(16 um radius, 0.748 x 0.748 x 1.0 um voxels, 32 x 43 x 43) only for the first
4,000 rows (`max_rows: 4000`). The cap was a budget choice made when v14 was
built: 4,000 rows is the Top-2000 scoring window plus the next 2,000, the same
boundary convention as the v11 `top_k_boundary` selection.

**Why it may need to change.** A descriptor declared with
`image_tier: 'level0'` is NaN for band ranks 4,001 to 20,000. A reranker can
pull such rows into the Top-2000, and the model then has no fine-resolution
information for exactly those rows. Level-1 voxels (about 1.5 x 1.5 x 2 um)
undersample thin axons, so sub-2-um evidence (continuity, small gaps) is only
measurable at level 0. The gen3 reviser of run
`precision_20261005_114814_zxsbw5pj` chose level 1, trading resolution for
coverage.

**Measured cost (2026-10-05 build, ten brain/kind entries, 0 failed reads).**

| tier | reads | on disk | build time |
|---|---|---|---|
| level1, whole band | 200,000 | 6.07 GB | 123 min |
| level0, first 4,000 | 40,000 | 1.55 GB | 24 min |

Per-read cost is the same for both tiers (25 to 30 reads/s aggregate) because
the level-0 radius was halved to keep the voxel count comparable. Extending
level 0 to the whole band adds about 160,000 reads, roughly +100 min and +6 GB
at that rate.

**What the change touches.**

- `IMAGE_TIERS['level0']['max_rows']` -> `None` in `harness/context_cache.py`.
- `ContextCacheBuilder.build` reuses a complete entry only as a whole and has
  no per-tier increment, so today the change means a `--force` rebuild of
  geometry plus both tiers (about 4.3 h with 16 readers, cache 11 GB -> ~17 GB).
  Adding an "add missing tier to a complete entry" path would avoid re-reading
  level 1.
- Check whether the entry manifest's recorded tier spec should bump
  `CACHE_VERSION` or otherwise invalidate `ContextCache.entry`, and that
  `plan_descriptor_run` / `descriptor_runs` coverage estimates follow the new
  row count for level-0 scopes.
- Do not rebuild a cache root while an evolution run is reading it; build into
  a separate `--context-cache` root or wait for the run to finish.
- Update the cache paragraph in `WORKFLOW_REVISION.md` and the v14 provenance
  row when done.

## Observed in run `precision_20261005_114814_zxsbw5pj`, not yet fixed (2026-10-05)

- Resolved 2026-10-06: `image_evidence.descriptor_image_computations` was always
  empty because the image-evidence summary ran before `record['descriptor_runs']`
  was assigned; the order is now fixed.
- Resolved 2026-10-05 by raising limits (MCP output cap 100,000 tokens, feedback
  budget 96 KB, parked tool results readable): oversized `search_memory`,
  `train_classifier` and `restore_candidate` results and the 48 KB feedback cap.
  Still open if context cost matters later: return file paths plus summaries
  instead of embedding the full feedback (about 39 KB of feature matrix) in
  `train_classifier` / `restore_candidate` results, and index-level
  `search_memory` entries.

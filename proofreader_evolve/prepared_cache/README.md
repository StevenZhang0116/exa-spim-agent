# prepared_cache/

Shared, validated cross-run cache of **prepared brains** (`prepared_<brain>.pkl`),
populated and read by `harness/incremental_scoring.py::get_or_build`.

## What lives here

One pickle per brain: the `PreparedBrain` — the once-loaded, candidate-invariant
state (GT + fragment graphs, raw-label snapshots, the fragment-label universe).
Building it from GCS is the ~30 min step; caching it here means it is paid **once
across all runs**, not once per run.

Each file is stored as `{"meta": {...}, "brain": PreparedBrain}`. The `meta` header
carries `schema_version`, `brain_id`, graph counts, and a content `checksum`.

## Why sharing is safe (unlike other run artifacts)

A `PreparedBrain` is a pure function of the **immutable dataset** and holds **no
run-specific state** — the train/held-out split, seed, GT maps, and policy edits
are all computed *after* loading, per run. It is also **mcl-independent** (built
from the raw, unfiltered GCS graphs; only the separate `../cache/dataset_cache_*`
fragment cache depends on mcl). So reusing it across runs leaks nothing
experiment-relevant.

## How staleness is prevented

`PreparedBrain.load_shared` validates before trusting a file and raises (→ rebuild)
on any of:
- **schema drift** — `schema_version` != `PREPARED_SCHEMA_VERSION` in the current
  code (bump that constant whenever the layout/build logic changes incompatibly);
- **wrong brain** — header `brain_id` != requested;
- **corruption / edit** — recomputed content checksum != the stamped one.

Writes are atomic (temp + `os.replace`), so a crash mid-write can't leave a
truncated pickle for a later run to load.

## Operational notes

- **Safe to delete** anything in here at any time — the next run rebuilds (~30 min)
  and repopulates. Each file is ~1.6 GB.
- To **force a clean rebuild**, delete the brain's file here (and the per-run
  `runs/<run>/prepared_<brain>.pkl`), or call `get_or_build(..., shared_cache=False)`.
- This **shared** cache is the single source of truth and is **preferred over any
  per-run copy**: `get_or_build` reads it first and, on a hit, deletes the redundant
  `runs/<run>/prepared_<brain>.pkl`. A per-run pickle is written only as a fallback
  (a legacy/older-code copy, a resume, or when `shared_cache=False`); when one is
  found it is **promoted into this shared cache and then deleted**, so state converges
  here. The one exception is `get_or_build(..., shared_cache=False)`, which never
  touches this cache and keeps the per-run copy as the only store.

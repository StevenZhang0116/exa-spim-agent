# Local fragment inspection and custom geometry features

Use local morphology when the existing feature table cannot test your hypothesis.
This interface uses the matching, already pruned/filtered fragment skeleton cache.
It does not supply raw images, segmentation volumes, GT skeletons or GT overlays.
Direct 3D fluorescence analysis with aligned fragments is described in
`volume_analysis_guide.md`. To combine geometry and image inputs in scoring,
read `image_context_guide.md`.
The candidate pool, original features and Top K stay fixed.

## Inspect a TRAIN candidate

`train_feedback.json` and evaluation feedback contain `candidate_refs` aligned
with example labels, scores and feature vectors. Call:

```json
{"candidate_ref": "<a reference from feedback>", "radius_um": 50, "occurrence_index": 0}
```

using `mcp__training__inspect_candidate`. A merge has one location. A split row
is a fragment pair and can have multiple native occurrences; inspect another
`occurrence_index` after reading `occurrences_total`. The order is the original
detector's fixed order, not a ranking learned from GT. Up to eight successful
inspections per generation cost no evaluation units, but do consume SDK turns.
Queries resolve only against TRAIN banks, including when a heldout handle is guessed.

Start a failure-driven investigation with `failure_cases.json` and call
`inspect_failure_cases({})` to inspect up to four matched pairs in one operation.
It shares the same eight-query limit; individual queries remain available for
other cases or occurrences. See `feature_discovery_guide.md` for the pairing
limitations and the paired feature comparison before claiming an improvement.

`locations_xyz_um` gives absolute anchor midpoints for TRAIN inspection. The
`geometry` object has the same schema used by feature extraction, except extraction
also includes the row's base `features` (nonfinite values become null). Coordinates
are **xyz in microns**, not zyx voxel coordinates. Each entry in `sites` contains:

- `xyz_um`: node coordinates relative to this site's anchor midpoint.
- `radius_um`, `degree`: node radii (null if unavailable) and full-graph degrees.
- `segment`: locally numbered fragment IDs, assigned in anchor-first order.
- `edges`: pairs of local node indices; only edges with both endpoints retained.
- `anchor_nodes`: one merge node or two split endpoints, indexing the arrays.
- `outside_radius`: marks anchors retained even when outside the query sphere.
- `truncated`: more nearby nodes existed than the node limit permitted.

This is a bounded induced subgraph. A missing edge/path is not proof of biological
disconnection. Nearby fragments can be included. Node/segment numbers are local
to each site and cannot be joined across sites or brains. No persistent IDs,
cache paths, absolute coordinates or labels enter the feature worker. TRAIN labels
remain available in diagnostic feedback and to classifier fitting, as before.

## Add features in explore mode

For a formula, put a literal `LOCAL_CONTEXT` dictionary and
`extract_local_features(context)` in **scorer.py**. For a fitted model, put them
in **training.py** alongside fit/predict. Helpers and installed CPU libraries may
be used inside the extractor. The entire program executes after OS isolation;
it must not load data at module scope. For formulas, place additional imports
inside the extractor: the score worker still allows only its preloaded libraries.

```python
import numpy as np

LOCAL_CONTEXT = {
    "feature_names": ["local_neighbor_segments", "local_branch_fraction"],
    "radius_um": 50,
    "max_nodes": 128,
    "max_occurrences": 2,
    "max_candidates": 5000,
    "selection_feature": "detector_score",
    "selection_largest": True,
}


def extract_local_features(context):
    segments = [len(set(site["segment"])) for site in context["sites"]]
    branches = [np.mean(np.asarray(site["degree"]) >= 3) for site in context["sites"]]
    return {
        "local_neighbor_segments": float(np.mean(segments)),
        "local_branch_fraction": float(np.mean(branches)),
    }
```

This example illustrates the interface; no ranking improvement is implied.
Choose and implement your own geometry, topology or multiscale summaries.
Each call receives one candidate with up to `max_occurrences` sites, starting
at occurrence 0, plus `kind`, `radius_um`, `occurrences_total`, `occurrences_used`,
`occurrences_truncated`, `version`, and base `features`. Return exactly the declared
feature names mapped to scalar numbers or NaN; infinity is rejected. Two calls
on identical contexts must agree. Do not memorize diagnostic examples.

The host appends your columns to the original features for both score_candidates
and fit/predict. It also appends `local_context_available` (1 for extracted rows,
0 otherwise). Handle missing local values explicitly. For example, a formula can
use the base score for unavailable rows and add a bounded geometry adjustment
for rows with context. Preflight also checks an entirely missing predictor row.

Extraction is a deterministic, label-free second stage: the host selects up to
`max_candidates` rows by the chosen **base predictor** before extracting geometry.
Nonfinite predictor values sort last and frozen row order breaks ties. This
selection rule is identical on every brain. All rows remain in the candidate
pool; unselected rows get NaN local features and must still receive finite scores.
If positive cases lie outside this selection, the local features cannot help
those rows. Change the selection feature/direction/budget in explore mode when
that matches your hypothesis; do not equate context availability with a GT label.

Defaults are radius 50 um, 128 nodes, one occurrence and 5000 rows per brain/kind.
Resource bounds are radius 1..200 um, 2..512 nodes, 1..8 occurrences,
1..20000 rows, 1..64 unique `local_*` feature names, and 128 MiB serialized
geometry per extraction. These bounds constrain cost, not the feature formula
or model family. Reducing nodes/occurrences/rows can resolve a budget error.

Geometry preparation plus worker execution has its own time budget per brain:
`--policy-time-budget` for scoring, `--classifier-time-budget` during fitting.
A blocking host cache load is checked after it returns; the isolated worker has
a hard wall timeout. Feature extraction uses one CPU library thread and a 16 GiB
address-space limit (the classifier memory setting applies during fit extraction).
Scoring/fitting then uses its existing, separate budget. No network, GPU, child
processes, model artifacts, labels or raw graph files are accessible to extraction.

## Freeze, evaluate and inspect the trajectory

Use the existing `evaluate_train`, `search_parameters` or `train_classifier` tools.
Extraction is included in the candidate measurement, not another evaluation unit.
The complete extractor source and literal configuration are part of the candidate
identity and snapshots. For classifiers they are embedded in the frozen model
manifest. `tune` cannot change extraction code or LOCAL_CONTEXT; use `explore`.
Classifier parameters are not passed into extraction: put extraction constants
in its source. Formula PARAMS can be referenced and are part of its cache identity.

TRAIN fitting receives augmented TRAIN features and labels. The outer evaluator
repeats the same frozen extraction and prediction on validation, with no fitting
and no GT input. Validation geometry, features and per-brain diagnostics are never
returned to the LLM. This remains adaptive development validation, not a final test.

Run-local `local_feature_cache/` entries bind program, configuration, candidate
pool, base features, source-cache identity and implementation hashes. Only numeric
feature outputs are reused; temporary geometry is removed. Workers cannot access
this cache. Ordinary scoring without LOCAL_CONTEXT does not load fragment graphs.
Existing prepared tables do not need rebuilding for this interface. The optional
split-classifier feature diagnostic also loads the TRAIN fragment graph for its
spatial grouping, even when the program does not declare LOCAL_CONTEXT.

Inspection snapshots are saved under `genNNN/context_inspections/`; trajectory
events record queries, locations in the tool response, truncation and budgets.
Feature events and evaluation cells record selected/pool counts, cache hits,
configuration and timing. Extractor stdout/stderr persists in
`local_feature_cache/worker_logs/`. Inspect these alongside actual TRAIN changes
before claiming the extra context helped.

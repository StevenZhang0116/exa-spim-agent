# Detector computation-cost checks

`run_detector_build_workflow` uses `validate_feature_implementation` before
assembling a detector. Its row-path gate combines the existing graph/import
guards with `detector_build.cost_validation.analyze_row_cost`. Confirmed
violations raise the normal validation error and enter the existing bounded
agent repair loop. The generated detector is never executed by this analysis.

The data-flow analyzer tracks graph-array origins and retained axes through
assignments, NumPy array wrappers, instance fields, multidimensional slices,
local helper arguments/returns and methods. It recognizes the graph's node
arrays and the runtime's candidate-universe API; feature ids and generated
variable names are not part of its rules. A column slice of a node array still
contains all nodes. A scalar node index or statically bounded slice does not.

The split runtime's `FeatureAccumulator.compute(..., evaluate)` callback is
analyzed as row work too. Moving a repeated global reduction inside a split
evaluator does not hide it from the supported data-flow analysis.

Repeated whole-universe reductions, sorting, array expressions/filtering and
index construction on a row path are rejected. Materializing the derived
result before row processing is allowed. Component and radius-query outputs
are variable-size, not assumed constant-cost. A persistent `_memoized` cache
can amortize component/local work only when its key covers the row-dependent
inputs visible to the analyzer. A global cached computation must be warmed
unconditionally before row processing; caching it by candidate or component
does not make it a pre-pass. Row-local image/local caches remain legal but do
not establish reuse across rows.

`CostReport` exposes three outcomes:

- `rejected`: at least one confirmed violation, with line, data source and call
  path. Builder assembly fails and supplies the diagnostic to the repair agent.
- `partial`: no confirmed violation, but unverified variable-size work,
  unresolved calls, recursion or an analysis-budget limit remains. The builder
  emits a bounded warning summary; the complete diagnostics are available on
  `CostReport.unverified`. Build completion is not a full complexity guarantee.
- `checked`: no violation or unresolved path was found in the supported subset.
  This still is not a proof about arbitrary Python execution or wall time.

The interpreter is deliberately bounded. Arbitrary dynamic dispatch, external
library internals, arbitrary manual caches and complex mutation are outside its
proof scope. Numerical equivalence is checked with trusted synthetic fixtures,
not inferred from the static cost check. Never resolve a cost error by changing
the scientific quantity, candidate pool or neighborhood definition.

Regression tests cover rejected and permitted variants, renamed aliases,
column/scalar slices, helper and method paths, cache keys/lifetimes, partial
results, and numerical equivalence with constant global-reduction call counts
as graph and candidate counts increase. Run them on the allocated compute node:

```bash
python -m unittest agentic.tests.test_detector_cost_dataflow agentic.tests.test_detector_build_workflow
```

Existing detector files are not automatically rewritten by a builder change.

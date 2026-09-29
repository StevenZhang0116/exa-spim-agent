# Split candidate feature contracts

New split builds use **inventory v5**. The driver retains the existing
source-SHA-bound `node_role_requirement`, and requires explicit `scope`,
`anchor`, `occurrence_reduction`, `source_feature`, `source_formula`,
`source_aggregation`, and `adaptation` fields. Old v3 inventories and feature
fragments must be rebuilt, including on retained-draft recovery. Previously
generated detectors are not rewritten by changing the builder.

The row unit is still a canonical unordered segment pair. Its policy-admitted
occurrences can have the tip on **either** side: side a is the lower segment ID,
not necessarily the anchor. A-B and A-C are separate candidate rows even though
they share A. All split feature output has `scope=candidate`.

The assembler injects a reviewed accumulator and checks exact feature registry
names/order. Generated fragments cannot define the accumulator or its schema.
They supply formula callbacks, returning a finite scalar or `None` if undefined:

```python
samples, labels = build_sample_universe(payload)
graph = payload['fragments_graph']
acc = FeatureAccumulator(samples, graph)  # after any profiling subset selection
for sample in samples:
    acc.compute('tip_radius', sample['candidate_id'],
                lambda context: graph.node_radius[context['anchor_node_id']])
return samples, labels, acc
```

Only run enabled analyses. Shared preprocessing and cache keys must honor all
dependencies. Do not capture raw sample/occurrence dictionaries in a callback to
override the runtime-selected inputs.

| Declared anchor | Inputs supplied to the formula |
| --- | --- |
| `tip_anchor` | Oriented anchor/partner node IDs and segment IDs, gap, candidate ID |
| `endpoint_pair` | Canonical node-ID pair, segment-ID pair, gap, candidate ID |
| `gap_midpoint` | Endpoint-pair inputs plus runtime-computed `center_um` |
| `component_pair` | Endpoint-pair inputs plus component-ID pair |
| `segment_pair` | Segment-ID pair and candidate ID, evaluated once per pair |

Runtime applicability filtering precedes formula evaluation. `first_compatible`
uses the first policy-ordered compatible occurrence, even if its result is
undefined; it never searches for a later defined result. `min`, `max`, `mean`
and `sum` reduce finite results across compatible occurrences, ignoring `None`.
No finite results means NaN/`is_defined=0`, including an empty sum.
`all_compatible` passes an immutable tuple of contexts to a joint source formula
(e.g. orientation variance). `segment_pair` requires `pair_once` and
`no_tip_requirement`; it composes context from both segments without a broadcast
API. All other anchors require an occurrence-based reduction.

The runtime rejects duplicate candidate IDs/pairs, noncanonical pairs, duplicate
feature evaluation, non-finite outputs and row/occurrence identity changes. Full
extraction must preserve runtime row order. Undefined or incompatible features
never remove candidates. The GT-blind wrapper still hides labels and audit data.

An `identity` definition preserves source math and aggregation. A deliberate
anchor/reduction change uses `candidate_pair` with a distinct name and a written
explanation. Read the source expression, not just a prose summary: `max/min`
and `min/max` are different quantities. A metadata error must be corrected to
match the source; do not invert a correct formula to match an incorrect inventory.
Runtime contracts cannot prove arbitrary callback mathematics, so source review
and independent numerical fixtures remain necessary.

Tests assemble a detector and execute actual candidate enumeration/extraction
on a small graph: A is a shaft, B/C provide side-b tip anchors, and A-B/A-C have
different radii and midpoints. Additional cases cover multiple occurrences,
missing applicability, pair context, row identity, GT changes, disabled groups,
profiling, and a global scan hidden in an evaluator. Run on n244:

```sh
python -m unittest agentic.tests.test_split_feature_scope \
  agentic.tests.test_split_feature_applicability agentic.tests.test_detector_build_workflow
```

Rebuild the inventory and detector, then retrain and recompute feature caches
when correcting anchor semantics. Existing fitted models and feature tables
must not silently be paired with the corrected extractor.

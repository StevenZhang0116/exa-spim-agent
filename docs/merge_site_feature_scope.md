# Merge-site feature scope in detector builds

`run_detector_build_workflow` now produces inventory **v4** for
`merge_site_detection`. Legacy segment-merge inventory remains v2; split now
uses [inventory v5](split_feature_scope.md) with reviewed anchoring and reduction.
Rebuild the merge-site inventory, detector,
and fitted model together; do not reuse an old model with adapted local features.

Each merge-site feature declares `scope` (`candidate` or `segment`),
`source_feature`, `source_aggregation`, and `adaptation` (`identity` or
`candidate_local`). The source path and hash still come from the driver.
An inventory containing only segment context is rejected. Traversal phase does
not determine scope: a junction traversal can still compute a segment statistic.

An identity feature retains its original formula and aggregation. Evaluating a
source local statistic at a particular candidate while removing the source's
across-junction reduction is a **feature-definition change**, represented by a
new name and `candidate_local`. It preserves the local formula and constants,
not the old aggregate value. The original aggregate can be retained separately
with segment scope. This applies to any source feature; no hypothesis IDs or
specific feature names are built into validation.

The assembler embeds scopes from the validated inventory, checks exact registry
names/order, and injects a reviewed `FeatureAccumulator`. Generated merge-site
code cannot define that class or the scope map. The extraction API is:

```python
samples, labels = build_sample_universe(payload)
acc = FeatureAccumulator(samples)  # after selecting a profiling subset
acc.set_candidate("candidate_radius", sample["candidate_id"], local_radius)
acc.set_segment("max_radius", segment_id, segment_maximum)
return samples, labels, acc
```

Candidate values are keyed by candidate ID; segment values alone are broadcast.
Wrong-scope writes, unknown identities, duplicate writes, and non-finite writes
raise. Leave an undefined value unwritten to produce NaN and `is_defined=0`.
The accumulator is safe for distinct concurrent writes. Compute reductions
before writing a scalar, and finish workers before conversion to a DataFrame.
The runtime checks the accumulator type, row identity/order, duplicate rows and
segment membership. Features receive only the existing GT-blind payload.

Both normal builds and retained-draft recovery use the same validators. Legacy
merge-site inventories fail the v4 check and must be rebuilt. The narrow draft
CLI receives `--target` for semantic validation and `--inventory` for assembly.

The regression suite assembles and executes a small feature fragment against a
synthetic graph: two junctions belonging to one segment have local radii 2 and
7, with context maximum 7 at both rows. It also covers missingness, disabled
context extraction, row permutation, scope errors, GT blindness, and inventory
compilation. Run on the designated compute node:

```sh
python -m unittest agentic.tests.test_merge_feature_scope \
  agentic.tests.test_merge_site_target agentic.tests.test_detector_label_provenance
```

These checks enforce storage and declared semantics; they cannot prove arbitrary
generated mathematics equivalent to its source. A generator could still compute
an incorrect value and pass it to `set_candidate`. The verification instructions
therefore require source-formula review and synthetic reference checks for
adaptations, and require reporting which formulas were actually tested. Equal
values for different real candidates are not themselves evidence of a defect.
The existing computation-cost checks continue to reject recognized repeated
global scans; shared preprocessing and enabled-analysis selection still apply.

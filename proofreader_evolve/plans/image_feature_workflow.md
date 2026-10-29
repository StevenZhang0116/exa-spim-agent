# Image access and feature discovery: evidence and revision proposal

2026-10-29. Design proposal; production code has not been changed for this proposal.
Target resource budget: 40 CPUs per evolution run. All executable audit work ran
on n277. Historical run directories were read only; no new evolution trial or
cloud image retrieval was performed.

## Evidence from the latest runs

[Machine-readable audit](../log/image_workflow_audit_20261029.json) records tool
counts, descriptor costs, TRAIN cache sampling and numerical feature checks.

| Run | Completed generations | Image previews | Local volume analyses | Descriptor calls involving images |
|---|---:|---:|---:|---:|
| `precision_20261027_220207_3zf4mz8t` / frozen | 11 | 0 | 1 | 3 successful |
| `precision_20261027_233027_6pgo4846` / no-score | 13 | 0 | 1 | 4 successful, 1 failed |
| `precision_20261028_000623_cun4h9k0` / raw | 19 | 0 | 0 | 8 successful |

Each run also had one generation in progress when the user stopped it. Tool
counts include those partial sessions. The runs precede the objective-unification
revision; future comparisons must use the current per-kind objective throughout.

### 1. Images were computed over, but never visually inspected

There are no `inspect_candidate_image` or `inspect_failure_images` calls and no
preview artifacts in any of the three runs. The two local volume analyses were
primarily context/schema debugging. In frozen gen007, `analysis.py` tried to import
`descriptor.describe`, caught `ModuleNotFoundError`, and returned `describe_error`
for both cases. The outer analysis was still recorded as `status: analyzed` because
the debugging program returned valid JSON. This is a concrete failure to exercise
the intended feature program, not evidence that the feature itself failed.

Sources: [debugging program](../runs/precision_20261027_220207_3zf4mz8t/gen007/volume_analyses/analysis001/analysis.py),
[its result](../runs/precision_20261027_220207_3zf4mz8t/gen007/volume_analyses/analysis001/result.json),
[no-score schema inspection](../runs/precision_20261027_233027_6pgo4846/gen002/volume_analyses/analysis001/analysis.py).
The two calls spent about 110/175 seconds preparing inputs and only 3.2/3.0 seconds
in the worker. Their requested level-0 radius was 40 um, while the cached level-0
descriptor input uses 16 um. Debugging and production therefore also differ in
input extent. The current prompt favors pool-scale descriptors and makes previews
optional; `plan_descriptor_run` is recorded as a raw-image observation when its
descriptor requests images. That observation is not a record of visual inspection.

### 2. The image and geometry support regions differ substantially

The cache fixes geometry at 50 um / at most 256 nodes, level-0 images at a requested
16 um radius, and level-1 images at 30 um. Split patches can expand to include both
anchors. A metadata-only TRAIN audit sampled three random chunks per brain/kind,
then up to 32 rows per chunk, seed 20261029: 576 merge and 555 split candidates
across six TRAIN brains. No pixels were needed for this calculation.

| Kind | Median fraction of geometry nodes inside level 0 | Inside level 1 |
|---|---:|---:|
| Merge | 27.3% | 65.5% |
| Split | 27.8% | 64.3% |

All sampled anchors were inside the images, and no sampled patch had an invalid
domain fraction. This establishes a difference in available spatial support; it
does not establish a registration defect or a precision penalty. The metadata
sample is not an estimate of error prevalence. Across the complete TRAIN cache,
the 256-node geometry cap truncated 1.20% of merge and 1.17% of split neighborhoods;
blindly raising that cap is a lower priority than matching images to feature needs.

Raw gen015 illustrates a support issue worth testing: its level-0 descriptor
samples branches out to 14 um and additional spheres around those nodes, clipping
those spheres to the image boundary. Patch validity alone does not describe the
fraction of each requested measurement region that was available. Its paired
feature ablation was negative, but that result cannot identify cropping as the
cause; model variation and cross-brain transfer are competing explanations.
Source: [descriptor](../runs/precision_20261028_000623_cun4h9k0/gen015/descriptor.py).

### 3. Full computation is expensive and its estimate has the wrong scope

All 20 descriptor calls across the runs used `all_cached`, including four
geometry-only calls. A call processes about 354k merge or 326k split rows across
six TRAIN brains. Successful calls involving images took 800–1781 seconds with
24 single-threaded workers. No-score/raw currently prohibit the detector-ranked
`pilot` and `boundary` scopes, leaving no registered small-sample alternative.

No-score gen010's first image descriptor was estimated at 345 seconds and marked
`fits_budget: true`, then failed after 1806 seconds under the 1800-second call
budget. The plan estimated one brain; execution includes all six. A modified
program subsequently completed in 1781 seconds. The error mentioning 16 seconds
is the remaining shared deadline passed to a late batch, not a separate per-row
limit. Sources: [plan](../runs/precision_20261027_233027_6pgo4846/gen010/descriptor_plans/plan001.json),
[failed call](../runs/precision_20261027_233027_6pgo4846/gen010/descriptor_runs/run001/result.json),
[successful call](../runs/precision_20261027_233027_6pgo4846/gen010/descriptor_runs/run002/result.json).

The current host stages individual patch files and starts a fresh sandboxed
process for each 256-row batch; every row's `describe` runs twice. These are
potential costs to profile, not measured explanations for the entire runtime.

### 4. Feature feedback and image attribution need stronger evidence

No-score gen010's agent summary called `ray_back_traced_frac` and
`ray_back_other_frac` constant zero. The saved TRAIN arrays contradict that claim:
in brain 747807 alone they have 9,362 and 1,719 nonzero rows, respectively.
Quartiles alone do not distinguish a sparse feature from a constant one. The audit
records counts for all six TRAIN brains; it does not use the agent summary as
ground truth.

The image-evidence summaries list zero paired image diagnostics, despite real
`bank_agent_*` ablations. `feature_ablation.py` recognizes `LOCAL_IMAGE` columns
for this flag, but not image-derived descriptor columns. Several descriptors also
mix geometric and image-derived outputs. Existing measurements are useful, but
the summary cannot reliably identify which ones isolate an image contribution.

## Scoped revision plan: inspection, configurable images and small-sample diagnostics

Implement one new tool and a small extension to the descriptor contract. Reuse the
existing image renderer, registered image reader, patch cache, isolated descriptor
worker and full-computation path. Implemented on 2026-10-29 following this scope;
verification is recorded below.

### 1. Add `probe_descriptor` for both selected cases and sampled cases

The tool reads the current `descriptor.py` and runs `describe(context)` directly.
It accepts either explicit TRAIN candidate handles or a seeded uniform sample
from each TRAIN brain's cached candidates. Start sampled probes at 128 rows per
brain, adjustable within the time/byte budget; do not select the highest detector
scores. Sampling may be used in frozen, no-score and raw modes.

Return per-case feature values, errors and compact per-brain diagnostics. When
requested, return a few PNG previews using `image_preview.render_preview`, drawn
from the exact pixels and geometry supplied to the descriptor. Reuse the current
image-preview allowance. Support the existing MIPs, thin slabs and skeleton overlays;
there is no new viewer or custom visualization language.

Probe and full computation must call the same input-assembly helper and isolated
worker. Preserve feature allowlists and no-score/raw restrictions in that helper;
do not expose other registered descriptors or host metadata incidentally. A failed
descriptor is reported as failed, with the affected case, not as successful analysis.
Unregistered drafts may reuse feature names during probing; name-conflict checks
still apply when production features are registered.

Probe results are diagnostic artifacts only: they do not modify feature tables,
the descriptor registry, candidate ranking or evaluation counts. Their complete
preparation/execution time counts against the existing descriptor time budget.
Keep these artifacts under a TRAIN-only directory and retain the current validation
file-tool boundary. The older analysis tools remain available.

### 2. Make one image patch configurable

Add an optional literal `image` field to `DESCRIPTOR`, for example
`'image': {'level': 0, 'radius_um': 40}`. The first version supports radius and native
pyramid level only. Keep the current anchor/midpoint centering, occurrence choice,
geometry neighborhood and single-patch `describe(context)` interface.

Keep existing `image_tier` declarations working. Resolve either representation to
one image specification; reject conflicting explicit declarations. Reuse a cached
level0/level1 patch when the requested specification matches. Otherwise use the
existing registered image reader and coordinate transforms. Apply the same resolved
configuration in probing, full TRAIN computation and frozen validation inference.
Include it and source/registration identity in provenance and cache keys.

Small probes can use the existing patch cache. Full uncached images must be read
and staged batch by batch, then discarded; retain the existing numerical descriptor
cache. The current persistent patch cache has a 20 GiB cap, so a full custom-view
run must not accumulate every patch there. Reuse existing pixels/arrays limits and
add a batch-byte bound so larger patches reduce batch row counts. Avoid loading an
entire brain's custom patches into RAM. Cold image retrieval is charged and timed.

Report actual shape, spacing, valid fraction and the fraction of geometry nodes
inside the patch. These facts describe available inputs; the host will not attempt
to infer arbitrary sampling regions inside user-written feature code.

### 3. Improve diagnostics and connect them to full evaluation

Report per-feature finite/missing counts, nonzero fraction, min/max, standard
deviation and the existing quantiles, separately by TRAIN brain. A sparse column
must not be called constant simply because its quartiles are zero. Return a small
set of candidate handles for exceptional or extreme values so the agent can inspect
those same cases. Labels stay outside descriptor code and the default sampler.

Reuse this diagnostic implementation for full descriptor results. Have
`plan_descriptor_run` estimate the entire requested TRAIN workload, including all
brains and input preparation, with a cold-cache caveat; report frozen inference
cost separately without exposing validation contents. Do not compare one brain's
estimate against the full call's budget.

Update prompts and guides to recommend: inspect/probe a few cases, sample across
brains, revise, compute full descriptors, then fit and run paired ablation. This is
guidance, not a mandatory state machine or another gate on every tuning round.
No-score/raw production calls keep `all_cached`; sample results cannot become sparse
predictors. Record planning, numerical probes and actual preview delivery separately.

Pass existing descriptor provenance into image-evidence reporting so image-consuming
`bank_agent_*` computations and their declared ablations are visible. A descriptor
group that mixes geometry and image outputs is reported as a joint effect, without
automatically asserting an isolated pixel contribution. No new feature dependency
language or automatic causal attribution is included.

### Implementation sequence and scope

1. Implement probes against existing cached inputs, with preview rendering and
   explicit error handling. Wire one new MCP tool into the existing session guard.
2. Add the optional image specification and shared input assembler, including
   bounded temporary staging for full custom-patch computation.
3. Add sampled diagnostics, whole-call estimates, evidence bookkeeping and updated
   instructions. Exercise the complete probe-to-score path on synthetic fixtures.

Main changes stay in `descriptor_contract.py`, `descriptor_runs.py`,
`train_experiments.py`, the existing image-input helpers, SDK tool registration and
the relevant evidence/guide files. Preserve the existing parallel worker model and
determinism checks. Run within the available 40 CPUs and memory limits; no CPU
scheduler rewrite or claim of linear speedup is part of this revision.

Deferred: simultaneous multi-view inputs, arbitrary centers/rotations, branch-aligned
resampling, variable geometry neighborhoods, source-tile cache services, long-lived
workers and a new UI. Multi-scale ideas can first be compared using separate probe
configurations and descriptor programs.

### Acceptance checks

- On the same cases and configuration, probe and full computation produce identical
  descriptor values; preview pixels/geometry match those inputs.
- A descriptor import/runtime error fails the probe clearly. Probing a revised draft
  does not overwrite registered values or consume scoring evaluations.
- Legacy tier behavior is preserved. Changing radius/level changes the actual input
  and cache identity consistently across probe, TRAIN and frozen inference.
- Sample selection is reproducible and not detector-rank based; no-score/raw input
  restrictions hold. Only explicit TRAIN handles can be probed.
- Sparse synthetic columns with zero quartiles and some nonzero values are reported
  correctly. Multi-brain estimates include every requested brain.
- Full custom-patch computation stays within batch/cache limits. Validation image
  data and logs remain inaccessible through current and later agent sessions.

### Implementation and verification (2026-10-29)

The shared input assembler is `harness/descriptor_inputs.py`; TRAIN probe and
planning responses are in `harness/descriptor_probe.py`, connected through
`TrainingExperiments` and the existing MCP session guard. Frozen inference costs
are extrapolated for `all_cached` from host workload sizes and the slowest measured
TRAIN rate; this is an estimate, not heldout execution or a cold-IO guarantee.
The 40-CPU run budget uses the existing configured worker model and single-threaded
workers. This change adds no scheduler or multi-view API.

All executed tests used synthetic fixtures on compute node n277. The image/probe
regression group passed 42 tests, and the final descriptor/cache/permissions group
passed 22 tests, including real isolated workers and MCP tool calls. A further
32-test group passed the end-to-end driver, no-score/raw, MCP and session-boundary
checks. These groups overlap; the counts are not unique-test totals. Coverage includes
probe/full numeric equality, exact preview inputs, old tiers, changed radius/level and
source identities, bounded temporary patches, frozen inference, draft-name reuse,
sparse diagnostics, errors with candidate references, whole-TRAIN cost estimates,
image-group evidence and validation-log denial in current and later generations.
These tests establish implementation behavior; real-data retrieval throughput and
feature usefulness still require the new trials.

Logs: `log/image_descriptor_workflow_20261029_{regression,core,modes}.log`.

### Follow-up consistency audit (2026-10-29)

- Updated runtime prompts, CLI help, image/volume guides and readable-directory
  descriptions for descriptor probes and configurable descriptor crops.
- Production scopes in tool schemas, prompts, the manifest and environment now
  follow the actual mode: no-score/raw expose only `all_cached`. The environment
  no longer advertises an unavailable context indicator in these modes.
- Probe responses distinguish zero newly charged evaluations from cumulative use.
  Geometry-only probes/plans do not report an image tier. Empty sampled caches
  return JSON-safe missing fractions.
- Budget-fit checks retain the whole-TRAIN estimate and separately compare the
  remaining uncached computation with the available allowance. Cache-hit checks
  verify numeric checksums, matching execution. Input identity preparation is
  inside the execution deadline, and later TRAIN brains receive no extra second
  once the shared budget expires.
- Human reports retain image requests and display recorded previews, probes,
  plans, descriptor computations and descriptor-group ablations. Group effects
  remain distinct from isolated image-only comparisons.

Audit validation on n277: 41 descriptor/report/no-score/access tests and seven
cache/execution-budget tests passed, together with a JavaScript syntax check and
an executable report-summary smoke check covering new and older records, cached
and failed computations, and group versus image-only diagnostics. Logs are
`log/image_descriptor_workflow_20261029_audit.log` and
`log/image_descriptor_workflow_20261029_budget.log`. Historical run artifacts were
not regenerated by this audit.

Run executable regression tests on a compute node. After implementation, assess
feature quality with fresh trials under the current per-kind objective, keeping
the data split and comparison budgets fixed. Larger images and more inspection
remain hypotheses for improved precision, not promised outcomes.

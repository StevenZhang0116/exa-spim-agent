## 3) Intent — find what each error type is associated with

The goal is **empirical discovery**: using all the information in the `_add.pkl`
caches, characterize **what splits, merges, and omits are associated with**. Treat
each error as its own response variable and ask, for each: *which feature of the
local geometry, the graph topology, the neuron's morphology, or the acquisition
(brain / annotator) predicts where this error occurs?* The deliverable is a ranked
set of **falsifiable, quantified, generalization-checked relationships**.

The three error types are **different failure modes** with likely **different
drivers** — do not pool them into one "error rate." A split (one neuron broken
into pieces) is a *connectivity* failure; a merge (two neurons fused) is a
*separation* failure; an omit (a stretch with no segment) is a *detection*
failure. Investigate each separately, then ask whether any driver is shared.

### What a finding must look like

Every reported relationship must be between **one error type** and **one (or a
small interaction of) feature(s)**, stated with all of:

- **Claim** — error type **E** varies with feature **F**, with direction (e.g.
  "split rate *increases* with local fragment density").
- **Unit of analysis** — what each data point is (a GT node, a GT edge, a fragment
  endpoint, or a whole GT neuron). The right unit differs by error type.
- **Test + effect size** — a statistic appropriate to the unit, plus an effect
  size (correlation, odds/rate ratio, Cohen's d, …). Lead with the effect size.
- **Generalization** — does it hold in **every brain present**, and survive
  **holding `brain_id` constant**? A relationship carried by one brain, or that
  vanishes once brain is controlled, is a brain artifact, not a finding.
- **Falsifiable prediction** — what to expect (or not) if the claim is true, so it
  can be checked on held-out neurons / another brain.

A bare "E correlates with F (p<0.001)" on pooled data is not a finding: without
direction, effect size, unit, and a generalization check it is the over-powered,
non-independent, possibly-confounded claim this dataset's size makes trivial to
manufacture.

### The feature space

The feature space is **open**: the families below are **starting examples, not a
closed list**. The raw materials are the per-node geometry (`node_xyz` in µm,
`node_radius`), the full GT and fragment **graph structure** (degree, paths,
components, KD-trees), and the stored label arrays — and you are expected to
**engineer your own features from them** whenever a richer descriptor might
explain an error better than the ones listed. Invent and test composite,
nonlinear, multi-scale, or directional features: ratios (caliber mismatch across a
gap), interactions (degree × radius), multi-radius density profiles, curvature /
tangent / co-linearity of the local cable, anisotropy of the local point cloud,
distributional summaries along an arbor, **raw-fluorescence descriptors at an error
site** (family F), or anything else derivable from the graphs, labels, or image. A
novel feature that separates an error class better than the obvious ones **is
itself a finding**. The only requirements are that a feature be (i) **reproducibly
computable** — either cache-only (no network, the cheap default) or from the raw
image via the public-S3 reader the cache documents (`img_util.TensorStoreImage(payload["img_path"])`
with `AWS_EC2_METADATA_DISABLED=true`; no credentials) — and (ii) GT-free in form
where the eventual aim is a deployable signal (features may be *evaluated* against
GT errors, but a feature defined *using* GT-neuron identity is a diagnostic, not a
deployable driver — flag which kind it is).

Use the four families as seeds; pair each error type with features from any of
them or with your own. The interesting findings are usually **local geometry /
topology** (or engineered combinations of them), not the coarse brain/annotator
grouping.

**A. Local geometry** (per GT node / edge), from `gt_graph.node_xyz` (µm) and
`node_radius`:
- **neurite radius / caliber** at the node — thin distal processes vs thick trunks;
- **edge length** (`gt.dist(i, j)`, µm) and local cable density;
- **local fragment density** — # of `fragments_graph` nodes within an *r*-µm ball
  of a GT node (via `fragments_graph.kdtree`): how crowded the neuropil is there;
- **distance to the nearest *other* GT neuron** (cross-neuron proximity via the
  pooled GT KD-tree) — a candidate driver of **merges**.

**B. Graph topology** (per GT node / neuron), from `gt_graph` structure:
- **node degree** — leaf (1), shaft (2), branch point (≥3);
- **geodesic distance to the nearest branch point** and **to the nearest leaf**
  (walk the GT graph) — distal-vs-central position along the arbor;
- **branch order / path distance from soma** (if a root is identifiable) —
  centrifugal position;
- **component structure of the predicted labels** along the neuron (`node_label`
  runs) — how fragmented, how interleaved.

**C. Neuron-level morphology** (per GT neuron), aggregating the above:
- total **cable length** (µm), # branch points, # leaves, mean/percentile
  **radius**, bounding-box extent, tortuosity;
- the neuron's own baseline error rates (so one error type can be tested as a
  predictor of another on the same neuron).

**D. Acquisition / provenance covariates** (per neuron):
- **`brain_id`** — always carried (the grouping factor for generalization);
- **annotator** — the `<initials>` suffix of `N0XX-<brain_id>-<initials>`;
- **`min_cable_length`** (`payload["min_cable_length"]`) — relevant to omit.

**E. Engineered / derived features** (encouraged) — anything built from A–D you
think captures the failure better: cross-gap caliber ratios, local curvature or
tangent agreement, multi-scale density profiles, feature interactions, per-arbor
distributional summaries, learned/clustered descriptors, etc. Define the feature
precisely, justify why it might drive the error, and hold it to the same hypothesis
template and statistical discipline as the listed ones.

**F. Raw-image (fluorescence) features** (optional) — the cache stores only the
image *path*, but the raw fused volume is on **public AIND S3** and reachable with
**no credentials**: `os.environ["AWS_EC2_METADATA_DISABLED"]="true"`, then
`image = img_util.TensorStoreImage(payload["img_path"])` and
`image.read(center_vox, shape)` for a patch centered on a node/error voxel
(`gt.node_voxel(i)` → (z,y,x); a merge site's µm `xyz` → voxel via `anisotropy`).
From a patch you can derive descriptors the skeleton cannot see — local intensity /
contrast / SNR at the error, signal continuity **across a split gap** (is there
bright fluorescence bridging the two fragments?), an intensity **valley between two
merged neurites**, texture / blob-vs-tubule structure, background level, etc. These
are often the most *mechanistic* drivers (they speak to why the U-Net failed), so
they are explicitly in scope. **Cost discipline:** every `read` is a cloud fetch, so
gate image features behind cheap cache-only filters — compute them only for the
candidate sites of interest (the error locations), never brain-wide — and cache
reads. Always carry `brain_id` (image statistics can differ by acquisition, so an
image feature must clear the same within-brain generalization check).

### Candidate drivers per error (hypotheses to test, not conclusions)

- **Splits** (connectivity). Candidates: thin caliber, distal position (far from
  soma / near leaves), high local fragment density (crowding), long GT edges,
  proximity to branch points. Unit: GT edge or fragment-endpoint pair.
- **Merges** (separation). Candidates: small distance to a *different* GT neuron,
  branch-point / crossing geometry, caliber mismatch between the fused neurites,
  dense neuropil. Unit: predicted segment landing on ≥2 neurons (and its merge
  site).
- **Omits** (detection). Candidates: thin caliber, distal / terminal position; and
  — *as a control, not a finding* — the cache's `min_cable_length` filter (a stretch
  reconstructed only by a dropped short fragment reads as omit). Check any apparent
  omit driver against this artifact before claiming it. Unit: GT edge / node.
- **Cross-error structure.** Do splits and omits co-locate along a neuron? Does a
  neuron's split rate predict its omit rate once `brain_id` is held constant?

### Pooling rules

Pool on **physical units (µm)** and **scale-normalized metrics** (percentages,
normalized ERL) so values are comparable across brains; carry **`brain_id`** with
every pooled record as a covariate / grouping factor; load **one brain at a time**,
reduce each to small per-neuron / per-node feature+error records, and let the
multi-GB graph be garbage-collected before the next brain. Every feature here is a
pure lookup + local graph walk over the stored arrays.

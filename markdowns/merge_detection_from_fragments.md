# Blind merge-error detection from fragment features

This document is a companion to [`labeled_dataset_cache.md`](labeled_dataset_cache.md).
It uses the same `_add.pkl` cache, but poses a different task: **detect merge
errors from the predicted reconstruction alone, then check yourself against the
ground truth.** The ground truth is a *grader*, never an *input* to the decision.

> **The one rule (read this first).** Your merge detector may read **only the
> network fragment information** — `fragments_graph` and its geometry / topology
> (plus the scalar build parameters). It must **never** read `gt_graph`,
> `gt_node_canonical_label`, `gt_edge_error`, `gt_merge_labels`, or
> `gt_merge_sites` while deciding where a merge is. Those five are the **answer
> key**: you touch them only afterward, in a separate scoring step, to measure how
> well the blind detector did. Structure the code so this is guaranteed, not just
> intended (see §3).

---

## 1) Dataset context

**Origin — identical to the labeled cache.** Each `_add.pkl` was built once from a
plain `dataset_cache_<brain_id>_mcl<N>.pkl` by reading the brain's dense predicted
**segmentation** volume, labeling every ground-truth node, and running the
canonical split/merge/omit scoring. **You do not run any of this** — the labels are
already baked in. All cloud access happened at build time; the `_add.pkl` loads
with no segmentation and no credentials. See `labeled_dataset_cache.md` §1 for the
full provenance; everything there still holds.

**What this task changes.** The labeled-cache doc treats the stored merge labels as
the *deliverable* — errors already identified, ready for a corrector to consume.
Here they are instead the **held-out answer key**. The deliverable you produce is a
*detector*: a function that looks at the U-Net reconstruction and predicts, without
any ground truth, which predicted segments fuse two neurons and where. This is the
realistic setting — in a fresh brain, or on the vast majority of neurons that were
never traced, there is **no** merge label to read; a proofreading tool has to find
merges from the reconstruction's own geometry and topology.

**What a "merge" is.** Deep-learning segmentation makes two systematic *topological*
errors: **splits** (one true neuron broken into several predicted segments) and
**merges** (one predicted segment fusing two or more distinct neurons). This doc is
about detecting the second. Concretely, a merge is a single predicted **segment id**
whose skeleton bridges what are really two separate neurons — an axon of cell A
touching a dendrite of cell B, two crossing processes assigned one label, a soma
with a foreign process fused onto it, etc.

**Why blindness matters (and why it is the whole point).** The canonical labels
were produced by comparing the segmentation to traced neurons — that is exactly the
information you are forbidden to use at decision time. The scientific question is
whether the merge is recoverable from **intrinsic reconstruction features** alone:
the fragment skeleton's branching structure, the caliber (radius) of its cables, the
angles at its junctions, the presence of multiple somata, and so on. If it is, the
detector generalizes to untraced neurons and to brains with no ground truth at all.
If it is not, that is itself a finding.

---

## 2) The blindness firewall — two disjoint zones of the payload

Every `_add.pkl` is one dict. For this task, partition its keys into two zones and
never let data flow from the right zone into the left:

| Zone | Keys | Role |
|---|---|---|
| **DETECTION INPUT** — the detector may read these | `fragments_graph`; `anisotropy`; `min_cable_length`; `node_spacing`; (optionally `img_path` for raw voxels, see §6) | The U-Net reconstruction and its build parameters. This is *all* the detector sees. |
| **ANSWER KEY** — scoring only, never the detector | `gt_graph`; `gt_node_canonical_label`; `gt_edge_error`; `gt_merge_labels`; `gt_merge_sites` | Ground-truth-derived merge truth, used *after* detection to grade it. |

The two right-zone keys that actually encode merges (recap from
`labeled_dataset_cache.md` §2):

- **`gt_merge_labels`** — `np.ndarray (M,) int64`. Predicted segment ids that fuse
  ≥2 GT neurons. This is the canonical `labels_with_merge` set: the **union** of the
  node-count rule (a segment landing on ≥2 GT neurons with >50 nodes on each) and
  the geometric merge-site walk. This is your **segment-level answer key**.
- **`gt_merge_sites`** — `list[dict]`, one per site:
  `{"segment_id": int, "gt_neuron": str, "xyz": (x, y, z) µm}`. This is your
  **site-level answer key** (where each merge happens, and which traced neuron it
  fuses into).

> **Do not call the answer-key *generators* either.** The functions
> `merge_labels(...)` and `geometric_merge_sites(...)` in
> `agentic_neuron_proofreader.data_modules.canonical_labeling` are how the key was
> *built* — and both read `gt_graph`. They are fine to use in a scoring/verification
> context (they reproduce the stored key), but they are **not** a detector: calling
> them is reading the answer. Your detector reuses none of their GT-dependent logic
> — only their *fragment-side* ideas (leaves, walks, caliber), reframed to need no
> GT.

---

## 3) What a merge looks like in the fragments graph (feature families)

`fragments_graph` is a `SkeletonGraph` (subclass of `networkx.Graph`) whose nodes
carry parallel NumPy arrays. Every feature below is computable from it alone. The
API you have (from `data_modules/graph_classes.py`):

- `frag.node_xyz` — `(N, 3)` float32, **(x, y, z) microns**.
- `frag.node_radius` — `(N,)` float16, skeleton caliber estimate per node.
- `frag.node_component_id` — `(N,)` int; one connected component = one fragment
  skeleton.
- `frag.component_id_to_swc_id` — `dict[int, str]`; the **segment id** is
  `swc_id.split(".")[0]`. Use `frag.node_segment_id(node)` /
  `frag.node_swc_id(node)`.
- Topology via NetworkX: `frag.degree[n]`, `frag.neighbors(n)`,
  `nx.connected_components(frag)`.
- Helpers: `frag.dist(i, j)` (µm), `frag.leaf_nodes()` (degree 1),
  `frag.branching_nodes()` (degree > 2), `frag.nodes_within_distance(root, µm)`,
  `frag.rooted_subgraph(root, µm)`, `frag.cable_length(root=...)`,
  `frag.kdtree` (KD-tree over `node_xyz`), `frag.node_voxel(i)` → (z, y, x) voxel.
- `frag.soma_centroids`, `frag.soma_component_ids` — soma detections (**may be
  empty** in a given cache; treat as a bonus signal, not a dependency).

**Unit of analysis.** The answer key is per **segment id**, but the physical merge
signature lives inside a **connected component** (the actual skeleton). Detect on
components, then aggregate your per-component flags up to segment ids for
segment-level scoring. (A single segment id can span several components when the
skeletonization breaks; `frag.nodes_with_segment_id(seg_id)` unions them.)

```python
from collections import defaultdict

comp_nodes = defaultdict(list)                      # component id -> [node ids]
for n in frag.nodes:
    comp_nodes[int(frag.node_component_id[n])].append(n)

def comp_segment_id(comp_id):
    return int(frag.component_id_to_swc_id[comp_id].split(".")[0])
```

The feature families a blind detector can exploit:

### (a) Topological — junction structure
A neuron's arbor is (topologically) a tree that branches *forward*. A merge splices
two arbors, which shows up at a junction node:

- **Degree-4+ node ("X crossing").** Two cables passing straight through each other
  and assigned one label. Genuine bifurcations are degree 3; a clean degree-4
  crossing is a strong merge cue. Enumerate candidate junctions as
  `[n for n in comp if frag.degree[n] >= 4]`, plus degree-3 nodes that fail the
  bifurcation-geometry test below.
- **"Dumbbell" connectivity.** Two high-cable-length sub-arbors joined by a single
  thin bridge — cut the bridge edge and the component falls into two large pieces of
  comparable size. Betweenness / bridge-edge analysis surfaces these.
- **Cycles.** A true skeleton is acyclic; a loop (`nx.cycle_basis`) often marks two
  processes fused at two points.

### (b) Geometric — angles and straightness at a junction
At a junction node `p`, take a robust outgoing direction per incident branch by
walking a few microns out (the immediate neighbor at ~`node_spacing` = 5 µm is
noisy):

```python
import numpy as np

def branch_direction(g, node, nbr, reach_um=15.0):
    """Unit vector from `node` outward along the branch that starts toward `nbr`."""
    prev, cur, acc = node, nbr, g.dist(node, nbr)
    while acc < reach_um:
        nxt = [k for k in g.neighbors(cur) if k != prev]
        if len(nxt) != 1:          # hit a leaf or another junction
            break
        prev, cur = cur, nxt[0]
        acc += g.dist(prev, cur)
    v = g.node_xyz[cur] - g.node_xyz[node]
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v
```

- **Pass-through pair (≈180°).** If two incident directions are near-antiparallel
  (`dot(d_a, d_b) < -0.8`), the cable goes *straight through* `p` — it does not
  branch there, it crosses. One antiparallel pair at a degree-3 node = a
  T-merge (a third process touching a through-cable); two antiparallel pairs at
  degree-4 = an X-crossing merge. Genuine bifurcations have **no** antiparallel
  pair — the parent and both daughters point into a forward cone (all pairwise
  dots > ~−0.5).
- **Bifurcation-angle outliers.** Real bifurcation angles cluster in a biological
  range; a junction whose angle is wildly outside it (near 0° or near 180°) is
  suspect.

### (c) Caliber — radius continuity
- **No taper across the junction.** At a true bifurcation the daughter cables are
  thinner than the parent (a Rall-ratio-like relationship). At a merge crossing,
  both through-going cables keep roughly constant, similar caliber — measure mean
  `node_radius` a few microns down each incident branch; two comparably **thick**,
  ≈collinear branches meeting is a merge cue.
- **Radius step.** An abrupt caliber discontinuity along an otherwise smooth cable
  can mark where a foreign process was fused on.

### (d) Morphological — soma count (strong but conditional)
Each real neuron has exactly one soma. If `frag.soma_centroids` is populated, a
single connected component (or segment) containing **≥2 soma centroids** is an
almost-certain merge. This is the highest-precision cue available — but it only
catches soma-to-soma or soma-adjacent merges, and **only if the cache actually
stored somata** (the array may be empty). Use it as a high-confidence prior, not the
whole detector.

### (e) Scale — cable and reach
Merged segments tend to be **large and spatially spread**: high total cable length,
a bounding box far larger than a single neurite, or two dense node-clusters far
apart in space bridged by sparse cable. Cheap component-level scalars
(`frag.cable_length(root=nodes[0])`, xyz spread) make an effective **prefilter** to
avoid running the expensive junction analysis on all ~hundreds-of-thousands of
fragments (see §5 performance note).

> **Honesty about difficulty.** None of these features is individually decisive —
> real neurons occasionally cross themselves, some merges are short and geometrically
> bland, and the fragment skeleton is filtered at `min_cable_length` (short fragments
> were dropped, so some bridges are missing entirely). Expect to **combine** cues
> (e.g. crossing-geometry ∧ caliber-continuity, or a soma-count override) and to
> tune thresholds against the answer key in §4. A single hard rule will either
> over-flag self-crossings or miss subtle fusions.

---

## 4) The blind detection interface

### Load — cloud-free, read the input zone only

```python
import pickle
import numpy as np
import agentic_neuron_proofreader  # noqa: F401 — registers SkeletonGraph for unpickling

with open("cache/dataset_cache_794495_mcl100_add.pkl", "rb") as f:
    payload = pickle.load(f)

frag             = payload["fragments_graph"]     # the ONLY graph the detector sees
anisotropy       = tuple(payload["anisotropy"])   # (0.748, 0.748, 1.0) for this batch
min_cable_length = int(payload["min_cable_length"])
```

Use plain `pickle.load` (not `BrainDataset.load_from_cache`, which opens the raw
image on S3). See `labeled_dataset_cache.md` §2 for the environment/memory caveats —
budget >20 GB RAM per cache, load one brain at a time.

### Enforce the firewall in the type system

Make the detector a pure function of the input zone. Its signature **cannot receive
the payload**, so it structurally cannot read the answer key:

```python
def detect_merges(fragments_graph, anisotropy, min_cable_length):
    """
    GT-BLIND. Reads only the fragment reconstruction. Returns predicted merges.

    Returns
    -------
    dict with:
      "segment_ids": set[int]                       # segments flagged as merges
      "sites": list[{"segment_id": int, "xyz": (x, y, z) µm, "score": float}]
    """
    ...  # feature logic from §3 — no gt_* anywhere in this scope
    return {"segment_ids": segment_ids, "sites": sites}


detections = detect_merges(frag, anisotropy, min_cable_length)
# `payload` is deliberately NOT passed in — the detector can't cheat.
```

### A concrete starter detector (illustrative)

Cheap prefilter → junction geometry → per-segment decision. Tune every threshold
against §4 scoring.

```python
import numpy as np

def detect_merges(fragments_graph, anisotropy, min_cable_length,
                  cross_dot=-0.8, thick_um=0.8, reach_um=15.0):
    g = fragments_graph
    comp_nodes = _group_by_component(g)                 # {comp_id: [nodes]}
    sites, seg_ids = [], set()

    for comp_id, nodes in comp_nodes.items():
        # (e) prefilter: skip small/compact components that can't be dumbbell merges
        if g.cable_length(root=nodes[0]) < 5 * min_cable_length:
            continue
        seg_id = int(g.component_id_to_swc_id[comp_id].split(".")[0])

        # (d) soma override, if somata were stored
        if _num_somata_in_component(g, comp_id) >= 2:
            seg_ids.add(seg_id)
            sites.append({"segment_id": seg_id,
                          "xyz": tuple(map(float, g.node_xyz[nodes[0]])),
                          "score": 1.0})
            continue

        # (a)+(b)+(c) junction crossing test
        for p in nodes:
            if g.degree[p] < 3:
                continue
            dirs = [branch_direction(g, p, nb, reach_um) for nb in g.neighbors(p)]
            rads = [_branch_radius(g, p, nb, reach_um) for nb in g.neighbors(p)]
            if _has_thick_passthrough(dirs, rads, cross_dot, thick_um):
                seg_ids.add(seg_id)
                sites.append({"segment_id": seg_id,
                              "xyz": tuple(map(float, g.node_xyz[p])),
                              "score": _crossing_score(dirs, rads)})

    return {"segment_ids": seg_ids, "sites": _dedup(sites, 30.0)}  # 30 µm dedup
```

with `_has_thick_passthrough` flagging any near-antiparallel pair
(`dot < cross_dot`) whose *both* branches exceed `thick_um` mean radius, and
`_dedup` collapsing sites closer than 30 µm (mirroring the canonical
`MERGE_DEDUP_UM` so your sites are comparable to the key's). Fill in the small
helpers from the sketches in §3.

---

## 5) Scoring against the answer key (the check)

Only **now** open the right zone. Detection and scoring are separate calls; the
payload enters here for the first time.

```python
import numpy as np

def score(detections, payload, site_tol_um=30.0):
    key_labels = set(int(x) for x in payload["gt_merge_labels"])
    key_sites  = payload["gt_merge_sites"]

    pred_labels = detections["segment_ids"]

    # --- The evaluable universe (see caveat below) -------------------------------
    # GT can only adjudicate segments that actually land on a traced neuron.
    node_label  = np.asarray(payload["gt_node_canonical_label"])
    adjudicable = set(int(x) for x in np.unique(node_label) if int(x) != 0)

    # --- Segment-level recall / precision / F1 -----------------------------------
    tp_lab = pred_labels & key_labels
    recall_seg = len(tp_lab) / len(key_labels) if key_labels else float("nan")
    # precision only over segments GT can judge (unmatched non-adjudicable
    # detections are ambiguous, NOT necessarily wrong -- see caveat)
    pred_adj = pred_labels & adjudicable
    prec_seg = len(pred_adj & key_labels) / len(pred_adj) if pred_adj else float("nan")

    # --- Site-level localization recall ------------------------------------------
    from scipy.spatial import KDTree
    hit = 0
    if detections["sites"] and key_sites:
        tree = KDTree([s["xyz"] for s in detections["sites"]])
        for ks in key_sites:
            d, _ = tree.query(ks["xyz"])
            hit += int(d <= site_tol_um)
    recall_site = hit / len(key_sites) if key_sites else float("nan")

    return {
        "recall_segment": recall_seg,
        "precision_segment_adjudicable": prec_seg,
        "recall_site": recall_site,
        "n_key_labels": len(key_labels),
        "n_key_sites": len(key_sites),
        "n_pred_labels": len(pred_labels),
        "n_pred_adjudicable": len(pred_adj),
    }
```

### The caveat that governs this whole task: GT is sparse

Only a handful of neurons are traced per brain (`labeled_dataset_cache.md` §1). The
answer key therefore contains **only merges that fuse ≥2 traced neurons**. A blind
detector will also flag merges between **untraced** neurons — those are *real
merges the key simply cannot confirm*. So:

- **Recall is well-defined.** Of the GT-confirmed merges (`gt_merge_labels` /
  `gt_merge_sites`), how many did the blind detector catch? Report this directly.
- **Raw precision is not.** A flagged segment that is not in `gt_merge_labels` may
  be a false positive **or** a true merge of untraced cells. Do **not** count all
  such flags as errors.
- **Fix: restrict precision to the adjudicable universe.** Segments whose id appears
  in `gt_node_canonical_label` land on ≥1 traced neuron, so GT *can* rule on them.
  Compute precision only over `pred_labels ∩ adjudicable`. Detections outside that
  set are reported (`n_pred_labels − n_pred_adjudicable`) but not scored as wrong.
  This is the honest analogue of the labeled-cache doc's "evaluation is relative to
  the traced neurons" framing.

### Relationship to the canonical thresholds

The answer key was built with GT-dependent thresholds: a segment counts as a merge
if it lands on ≥2 GT neurons with **>50 GT nodes on each** (node-count rule) or if
the geometric walk finds a fragment leaf **>50 µm** from GT that walks back to
**within 6 µm** of a *different* GT neuron, sites deduped at **30 µm**
(`canonical_labeling.py`: `MERGE_MIN_NODES`, `MERGE_DIST_AWAY_UM`,
`MERGE_APPROACH_UM`, `MERGE_DEDUP_UM`). Every one of those tests references GT, so
none of them is available to you. Your detector's job is to reproduce the *outcome*
of these tests — the `gt_merge_labels` set — from fragment features that use no GT.
The 30 µm dedup is the one threshold you can borrow directly (it is a property of
the fragment skeleton), which is why the starter detector and the scorer both use
it — it keeps your sites and the key's sites on the same footing.

### What "good" looks like

A useful blind detector achieves **high site-level recall** and **high
adjudicable-segment precision** on each brain, and — when more than one `_add.pkl`
is present — does so **consistently across brains** rather than being carried by
one. Pool per-segment records tagged with `brain_id` (parse it from the filename,
as in `labeled_dataset_cache.md` §2). Because merges are rare, also report absolute
counts (`n_key_labels`, TP, flagged-but-unadjudicable) next to the rates — a recall
of "3/4" means something different from "300/400".

---

## 6) Optional: raw image as a second blind evidence channel

Everything above is fragment-graph-only. If you want *more* evidence than the
skeleton carries — e.g. to confirm that two crossing cables are genuinely two
fluorescent processes and not a tracing artifact — you can read the raw image around
a candidate site. **This is still GT-blind** (the raw fluorescence is not the
answer key), but it goes beyond "network fragment information", so treat it as an
optional secondary channel, not the core detector.

The image is on public S3 (`img_path` in the payload); no credentials, only outbound
network. Set `AWS_EC2_METADATA_DISABLED=true`, open
`img_util.TensorStoreImage(payload["img_path"])`, and read a patch centered on a
candidate site's voxel (`xyz_to_voxel(xyz, anisotropy)`), exactly as in
`labeled_dataset_cache.md` §2 ("Optionally reading the raw image"). Keeping this out
of `detect_merges`'s signature preserves the firewall — if you use it, pass the
opened image reader in explicitly, and still never pass `payload`.

> The dense **segmentation** remains intentionally inaccessible (private GCS,
> provenance only). You do not need it: the merge signal is either in the fragment
> skeleton (§3) or, optionally, in the public raw image (this section).

---

## 7) Intent

The goal is a **ground-truth-blind merge detector** — one that finds fused segments
from the reconstruction's own geometry and topology, so it works on the untraced
neurons that dominate every brain and on brains with no tracing at all. The stored
`gt_merge_labels` / `gt_merge_sites` are a **validation harness**, not an input:
they tell you, after the fact, how close your feature-based decisions came to the
canonical answer. A detector that scores well *only* because it peeked at the labels
is worthless here; a detector that scores well while provably blind is exactly the
component a post-hoc proofreading tool needs to *resolve* merges (cut the fused
segment at the detected site) without a human first tracing the neuron.

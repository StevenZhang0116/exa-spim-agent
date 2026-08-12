# Understanding merge errors in ExaSPIM reconstructions

The scientific question: **what distinguishes a merge-error site from a non-merge
location, across the topology and geometry of the fragment skeleton and the raw
fluorescence image?** The answer lives in three dimensions — topological, geometric,
and image-level — and the goal is to characterize all three.

Work proceeds in two phases that use the ground truth differently:

- **Phase 1 — Characterization (GT-informed).** Use `gt_merge_sites` as positive
  examples and matched control points as negatives. Extract features across all three
  dimensions at every site. Compare distributions to understand what geometry the
  network produces and what it sees in the raw image at a merge vs. a clean location.
  This is the core analysis.
- **Phase 2 — Blind detection (GT-blind).** Build a detector from the features
  identified in Phase 1 that reads only the fragment reconstruction — no GT anywhere
  in scope. Run it, then grade against the GT labels. High performance validates the
  characterization; a gap between characterization and detection reveals which signals
  require GT to locate and which are recoverable from the reconstruction alone.

> **The one rule.** The Phase 2 detector may read **only the network fragment
> information** (`fragments_graph` and its geometry / topology, plus scalars and
> optionally the raw image). It must **never** read `gt_graph`,
> `gt_node_canonical_label`, `gt_edge_error`, `gt_merge_labels`, or `gt_merge_sites`
> while deciding where a merge is. In Phase 1 you are explicitly allowed — and
> expected — to read both zones together.

---

## 1) Dataset context

**Origin.** Each `_add.pkl` was built from a plain skeleton cache by reading the
brain's dense predicted segmentation volume, looking up the predicted segment id at
every ground-truth node's voxel, classifying every ground-truth edge (correct / split
/ omit / merge), running the geometric merge-detection walk, and writing all of that
back into the `_add.pkl`. You do not run any of this — the labels are already baked
in. All cloud access happened at build time; the `_add.pkl` loads with no credentials.
The cache holds two skeleton graphs — the automated **UNet fragment** reconstruction
(`fragments_graph`, hundreds of thousands of fragment components) and the human
**ground-truth** tracings (`gt_graph`, tens of neurons) — plus the baked-in labels.

**What a "merge" is.** Deep-learning segmentation makes two systematic topological
errors: **splits** (one neuron broken into several predicted segments) and **merges**
(one predicted segment fusing two or more distinct neurons). A merge is a single
predicted segment id whose skeleton bridges what are really two separate neurons — an
axon of cell A touching a dendrite of cell B, two crossing processes assigned one
label, a soma with a foreign process fused onto it, etc.

**Why this matters.** In a fresh brain, or on the vast majority of neurons never
traced, there is no merge label to read; a proofreading tool has to find merges from
the reconstruction's own geometry and the raw image. Understanding *why* the network
makes merge errors — in what image conditions and at what geometric configurations —
is the prerequisite for building that tool and for improving the segmentation model
itself.

**GT is sparse.** Only a handful of neurons are traced per brain, so the answer
key sees a merge only where it touches a traced neuron. It is **not** limited to
merges fusing ≥2 traced neurons: the geometric-walk branch also flags a segment
that runs ≥ 50 µm out through untraced space and back onto a single traced
neuron. What stays invisible is narrower — a segment labelling no GT node at all,
an excursion shorter than 50 µm, one returning no closer than 6 µm, or one
landing in a same-label GT component under 50 nodes. Keep this in mind when
sampling control points and interpreting precision.

---

## 2) Dataset schema

### Environment and loading — run this first

All later code blocks assume this block has run. Python 3.10 or newer is required.
Check the package with `python -c "import agentic_neuron_proofreader"`. If that
fails, install the official package and analysis dependencies:

```bash
python -m pip install "git+https://github.com/AllenInstitute/agentic-neuron-proofreader.git" numpy scipy networkx tensorstore matplotlib
```

Run from the delivery directory containing exactly one `_add.pkl`. Loading may use
tens of GB of RAM, so process one cache at a time:

```python
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

__import__("agentic_neuron_proofreader")  # registers SkeletonGraph pickle classes

DELIVERY_DIR = Path.cwd()
pkl_candidates = sorted(DELIVERY_DIR.glob("*_add.pkl"))
if len(pkl_candidates) != 1:
    raise RuntimeError(
        "expected exactly one *_add.pkl in the delivery directory; "
        f"found {len(pkl_candidates)}: {[p.name for p in pkl_candidates]}"
    )
PKL = pkl_candidates[0]
with PKL.open("rb") as f:
    payload = pickle.load(f)

required = {
    "fragments_graph", "gt_graph", "anisotropy", "min_cable_length",
    "gt_node_canonical_label", "gt_edge_error",
    "gt_merge_labels", "gt_merge_sites",
}
missing = required - payload.keys()
if missing:
    raise RuntimeError(f"not a complete _add.pkl; missing {sorted(missing)}")

frag = payload["fragments_graph"]
gt = payload["gt_graph"]
node_label = np.asarray(payload["gt_node_canonical_label"])
edge_error = np.asarray(payload["gt_edge_error"])
merge_labels = {int(x) for x in np.asarray(payload["gt_merge_labels"])}
merge_sites = list(payload["gt_merge_sites"])
anisotropy = tuple(payload["anisotropy"])
min_cable_length = int(payload["min_cable_length"])
node_spacing = payload.get("node_spacing")
gt_edges = list(gt.edges)

if len(node_label) != gt.number_of_nodes():
    raise RuntimeError("gt_node_canonical_label is not parallel to GT nodes")
if len(edge_error) != len(gt_edges):
    raise RuntimeError("gt_edge_error is not parallel to list(gt_graph.edges)")
for site in merge_sites:
    if not {"segment_id", "gt_neuron", "xyz"} <= site.keys():
        raise RuntimeError(f"malformed gt_merge_sites record: {site}")
print("Loaded:", PKL)
```

### The two zones

Every `_add.pkl` is one dict. In Phase 1 you read both zones together for
characterization; the firewall applies only to the Phase 2 detector.

| Zone | Keys | Phase 1 | Phase 2 detector |
|---|---|---|---|
| **RECONSTRUCTION** | `fragments_graph`; `anisotropy`; `min_cable_length`; `node_spacing`; `img_path` | ✓ read | ✓ read |
| **ANSWER KEY** | `gt_graph`; `gt_node_canonical_label`; `gt_edge_error`; `gt_merge_labels`; `gt_merge_sites` | ✓ read (characterization + scoring) | ✗ never |

Full schema:

| Key | Type | Meaning |
|---|---|---|
| `fragments_graph` | `SkeletonGraph` | UNet reconstruction — the detection input. |
| `gt_graph` | `SkeletonGraph` | Human ground-truth tracings. Carries the label arrays below. |
| `anisotropy` | `tuple` | µm/voxel in (x, y, z) — read it, don't hard-code. |
| `min_cable_length` | `int` | µm threshold; shorter fragments were dropped at build time. |
| `node_spacing` | `int` | Target µm spacing between skeleton nodes — per cache (`5` for `mcl100`, `2` for `789202_mcl10`), and only a *target*: each irreducible edge is spline-resampled to `max(int(len/node_spacing), 5)` points, so short edges are denser. Node counts are therefore not proportional to µm. |
| `img_path` | `str` | Raw fused-image path: public `s3://` or authenticated `gs://`. Image features are optional for a fragment-only detector; configure access as shown below before reading them. |
| `segmentation_path` | `str` | Provenance only — private GCS, not readable here. |
| `gt_node_canonical_label` | `np.ndarray (N_gt,) int64` | Predicted segment id at each GT node's voxel; `0` = unlabeled. |
| `gt_edge_error` | `np.ndarray (E_gt,) uint8` | Per-GT-edge class: `0=correct, 1=split, 2=omit, 3=merged`. |
| `gt_merge_labels` | `np.ndarray (M,) int64` | Segment ids flagged as merges, pooled over all GT neurons into one global set (canonical keeps one set *per* neuron). Membership does **not** imply ≥2 fused GT neurons — the geometric-walk branch flags fusions to untraced material off a single traced neuron. Per-neuron attribution lives in `gt_merge_sites[i]["gt_neuron"]`. |
| `gt_merge_sites` | `list[dict]` | One entry per site: `{"segment_id": int, "gt_neuron": str, "xyz": (x, y, z) µm}`. |

The last four are also attached to `gt_graph` as `gt_graph.node_label`,
`gt_graph.edge_error`, `gt_graph.merge_labels`, and `gt_graph.merge_sites`.

### Fragment-graph API

`fragments_graph` is a `SkeletonGraph` (subclass of `networkx.Graph`). Every feature
in Phase 1 and Phase 2 is computable from it (plus optionally the raw image).

- `frag.node_xyz` — `(N, 3)` float32, **(x, y, z) microns**. Note:
  `frag.node_voxel(i)` returns `(z, y, x)` voxel coordinates — a different axis order.
- `frag.node_radius` — `(N,)` float16, skeleton caliber estimate per node.
- `frag.node_component_id` — `(N,)` int; one connected component = one fragment skeleton.
- `frag.component_id_to_swc_id` — `dict[int, str]`; each value has the form
  `"<segment_id>.0"` (the segment id serialised as a float literal, e.g.
  `'32323215387.0'`). The `.0` is an artifact, not a sub-index.
  `swc_id.split(".")[0]` extracts the segment id as a string; wrap in `int(...)` for
  the integer. Convenience wrappers `frag.node_segment_id(node)` → `str` and
  `frag.node_swc_id(node)` → `str` are preferred.
- Topology via NetworkX: `frag.degree[n]`, `frag.neighbors(n)`,
  `nx.connected_components(frag)`.
- Helpers: `frag.dist(i, j)` → float µm (**Euclidean distance between any two nodes**
  by their xyz coords, not restricted to adjacent nodes);
  `frag.leaf_nodes()` (degree 1); `frag.branching_nodes()` (degree > 2);
  `frag.nodes_within_distance(root, µm)`; `frag.rooted_subgraph(root, µm)`;
  `frag.cable_length(root=node)` → total edge-sum cable length (µm) of the entire
  connected component containing `node` (result is the same for any node in that
  component);
  `frag.nodes_with_segment_id(seg_id)` → `set` of all node ids whose segment id
  equals `seg_id`;
  `frag.kdtree` (KD-tree over `node_xyz`); `frag.node_voxel(i)` → (z, y, x) voxel.
- `frag.soma_centroids` — `list` of length K, each element an xyz-µm tuple.
  `frag.soma_component_ids` — `list` of length K (parallel to `soma_centroids`),
  each element the int component id for that centroid. **Both are plain Python lists
  and may be empty** (`K = 0` in the current cache); guard for length zero before
  iterating.

**Grouping nodes by component (use throughout):**

```python
comp_nodes = defaultdict(list)
segment_nodes = defaultdict(list)
for n in frag.nodes:
    n = int(n)
    comp_nodes[int(frag.node_component_id[n])].append(n)
    segment_nodes[int(frag.node_segment_id(n))].append(n)

def comp_segment_id(comp_id):
    return int(frag.component_id_to_swc_id[comp_id].split(".")[0])
```

### The comparison scaffold

The core analysis is a **contrast**: extract features at merge sites (positive class)
and at matched control points (negative class), then compare distributions across
topology, geometry, and image dimensions.

**Positive samples — merge sites.** `gt_merge_sites` gives ground-truth merge
locations directly. Match the stored `segment_id` first and find the nearest node
only within that segment. A global nearest-node query can select an unrelated segment
passing nearby and is not a valid positive association:

```python
def nearest_node_in_segment(segment_id, xyz_um):
    """Return nearest node/distance within the required predicted segment."""
    nodes = segment_nodes.get(int(segment_id), [])
    if not nodes:
        return None, float("nan")
    tree = KDTree(np.asarray(frag.node_xyz[nodes], dtype=float))
    distance, local_index = tree.query(np.asarray(xyz_um, dtype=float))
    return int(nodes[int(local_index)]), float(distance)

positives = []
unmatched_sites = []
for s in merge_sites:
    node, dist = nearest_node_in_segment(s["segment_id"], s["xyz"])
    if node is None:
        unmatched_sites.append(s)
        continue
    comp_id = int(frag.node_component_id[node])
    positives.append({"node": node, "comp_id": comp_id,
                      "xyz": s["xyz"], "label": 1,
                      "segment_id": int(s["segment_id"]),
                      "match_distance_um": dist})

if not positives:
    raise RuntimeError(
        "no GT merge site has a retained fragment skeleton for the same segment"
    )
print("matched / missing merge-site skeletons:",
      len(positives), len(unmatched_sites))
```

**Negative (control) samples.** Sample from the same fragment components that
contain merge sites, but at leaf nodes far from any merge site (≥ 50 µm). Using
the same components controls for the selection bias that large, complex components
are over-represented in the merge set.

```python
def sample_controls(frag, comp_nodes, positives, merge_sites,
                    n_per_site=3, min_dist_um=50.0):
    if not merge_sites:
        raise RuntimeError("cannot construct controls without gt_merge_sites")
    merge_xyzs = np.array([s["xyz"] for s in merge_sites])
    merge_tree = KDTree(merge_xyzs)
    controls = []
    for positive in positives:
        comp_id = int(positive["comp_id"])
        count = 0
        for node in comp_nodes[comp_id]:
            if frag.degree[node] != 1:
                continue
            xyz = tuple(map(float, frag.node_xyz[node]))
            d, _ = merge_tree.query(xyz)
            if d >= min_dist_um:
                controls.append({"node": node, "comp_id": comp_id,
                                  "xyz": xyz, "label": 0})
                count += 1
                if count >= n_per_site:
                    break
    return controls

negatives   = sample_controls(frag, comp_nodes, positives, merge_sites)
all_samples = positives + negatives
```

### Feature dimensions

Extract features at each sample point across three dimensions. Treat these as a
starting catalogue — explore the data for signals this doc did not anticipate.

#### Dimension 1 — Topology

| Feature | How to compute |
|---|---|
| Node degree | `frag.degree[p]` |
| Max degree in component | `max(frag.degree[n] for n in comp_nodes[comp_id])` |
| Any degree-4+ node within 30 µm | query `frag.kdtree`, check degrees |
| Cycle count in component | `len(nx.cycle_basis(frag.subgraph(comp_nodes[comp_id])))` |
| Number of leaves in component | `sum(1 for n in comp_nodes[comp_id] if frag.degree[n] == 1)` |
| Number of branch points | `sum(1 for n in comp_nodes[comp_id] if frag.degree[n] > 2)` |
| Component node count | `len(comp_nodes[comp_id])` |

A genuine bifurcation is degree 3 with no antiparallel pair. A merge junction is
often also degree 3 or 4 but has two branches pointing nearly opposite (a
through-cable). Cycles in the skeleton (which should be acyclic) often mark two
processes fused at two contact points.

#### Dimension 2 — Geometry

| Feature | How to compute |
|---|---|
| Local radius | `float(frag.node_radius[p])` |
| Mean radius within 15 µm | mean `node_radius` over `frag.nodes_within_distance(p, 15)` |
| Radius contrast across junction | ratio of max to min branch radius at `p` |
| Total cable length of component | `frag.cable_length(root=p)` |
| Spatial extent (bbox diagonal) | `np.linalg.norm(xyz[nodes].max(0) - xyz[nodes].min(0))` |
| Crossing score | magnitude of most antiparallel branch-direction dot product (see helpers) |
| Thick-passthrough flag | both antiparallel branches exceed a radius threshold (see helpers) |
| Branch-angle distribution | all pairwise dots among incident branch directions at `p` |

At a true bifurcation, daughters taper relative to the parent. At a merge crossing,
both through-going cables keep similar, thick caliber. Large cable length and wide
spatial extent are cheap prefilter signals — merged segments tend to span two arbors.

#### Dimension 3 — Raw image

The raw fluorescence is what the UNet actually saw. Open it once before the loop:

```python
from agentic_neuron_proofreader.utils import img_util

BUNDLED_GCP_CREDENTIALS = DELIVERY_DIR / "zihan_gcs_token.json"


def configure_image_access(image_path):
    """Configure anonymous S3 or bundled-credential GCS before image open."""
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    if image_path.startswith("s3://"):
        print("Image access: anonymous/public S3")
        return
    if not image_path.startswith("gs://"):
        raise RuntimeError(f"unsupported image path scheme: {image_path}")
    credentials = BUNDLED_GCP_CREDENTIALS.resolve()
    if not credentials.is_file():
        raise RuntimeError(
            "private GCS image requires zihan_gcs_token.json beside the markdown "
            f"and _add.pkl; looked at {credentials}. Fragment-only analysis may "
            "continue, but image-feature claims may not."
        )
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials)
    print(f"Image access: authenticated GCS ({credentials})")


img_path = str(payload["img_path"])
configure_image_access(img_path)
image = img_util.TensorStoreImage(img_path)

def xyz_to_voxel(xyz_um, anisotropy):
    """(x, y, z) µm -> (z, y, x) integer voxel."""
    return tuple(int(c / a) for c, a in zip(xyz_um, anisotropy))[::-1]

def read_patch(xyz_um, anisotropy, half_width=32):
    center = xyz_to_voxel(xyz_um, anisotropy)
    return image.read(center, (2 * half_width,) * 3)
```

| Feature | Interpretation |
|---|---|
| Peak intensity in patch | how bright is the process the network labelled here? |
| Local intensity variance | high variance may indicate two distinct fluorescent structures |
| Intensity histogram modes | one mode = one process; two modes = possible two-process crossing |
| Nearest high-intensity voxel from a *different* component | how close is a foreign labelled process? |
| Mean intensity along the crossing-direction vector | does the through-cable track high fluorescence? |
| Inter-process gap (minimum distance between two local intensity peaks) | how close are the two fusing neurites in the image? |

The most informative signals are proximity (two high-intensity regions very close
with little dark gap), intensity similarity (both processes at similar brightness),
and orientation (processes crossing at a shallow angle are harder to separate).

### Helper functions

Define these once before any analysis or detection code:

```python
def branch_direction(g, node, nbr, reach_um=15.0):
    """Unit vector from `node` outward along the branch that starts toward `nbr`."""
    prev, cur, acc = node, nbr, g.dist(node, nbr)
    while acc < reach_um:
        nxt = [k for k in g.neighbors(cur) if k != prev]
        if len(nxt) != 1:
            break
        prev, cur = cur, nxt[0]
        acc += g.dist(prev, cur)
    v = g.node_xyz[cur] - g.node_xyz[node]
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v


def _group_by_component(g):
    """Return {component_id: [node_ids]}."""
    groups = defaultdict(list)
    for n in g.nodes:
        groups[int(g.node_component_id[n])].append(n)
    return groups


def _branch_radius(g, node, nbr, reach_um=15.0):
    """Mean node_radius (µm) along the branch from node toward nbr, up to reach_um."""
    prev, cur, acc = node, nbr, g.dist(node, nbr)
    radii = [float(g.node_radius[nbr])]
    while acc < reach_um:
        nxt = [k for k in g.neighbors(cur) if k != prev]
        if len(nxt) != 1:
            break
        prev, cur = cur, nxt[0]
        acc += g.dist(prev, cur)
        radii.append(float(g.node_radius[cur]))
    return float(np.mean(radii)) if radii else 0.0


def _has_thick_passthrough(dirs, rads, cross_dot=-0.8, thick_um=0.8):
    """True if any antiparallel branch pair (dot < cross_dot) both exceed thick_um mean radius."""
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            if np.dot(dirs[i], dirs[j]) < cross_dot:
                if rads[i] > thick_um and rads[j] > thick_um:
                    return True
    return False


def _crossing_score(dirs, rads):
    """Merge confidence 0–1: magnitude of the most antiparallel dot product."""
    best = 0.0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            best = max(best, -float(np.dot(dirs[i], dirs[j])))
    return best


def _num_somata_in_component(g, comp_id):
    """Number of detected soma centroids belonging to comp_id."""
    ids = g.soma_component_ids  # plain list, may be empty
    if not ids:
        return 0
    return sum(1 for cid in ids if cid == comp_id)


def _dedup(sites, radius_um=30.0):
    """Suppress sites within radius_um of a higher-scored site; return survivors."""
    if not sites:
        return []
    sites = sorted(sites, key=lambda s: -s["score"])
    kept, suppressed = [], set()
    tree = KDTree([s["xyz"] for s in sites])
    for i, s in enumerate(sites):
        if i in suppressed:
            continue
        kept.append(s)
        for j in tree.query_ball_point(s["xyz"], radius_um):
            if j != i:
                suppressed.add(j)
    return kept
```

### Phase 2 — Blind detection

After characterizing which features separate merge from non-merge sites (Phase 1),
build a detector that uses only those features without reading any GT:

```python
def detect_merges(fragments_graph, anisotropy, min_cable_length,
                  cross_dot=-0.8, thick_um=0.8, reach_um=15.0):
    """
    GT-BLIND. Reads only the fragment reconstruction. Returns predicted merges.

    Returns
    -------
    dict with:
      "segment_ids": set[int]
      "sites": list[{"segment_id": int, "xyz": (x, y, z) µm, "score": float}]
    """
    g = fragments_graph
    comp_nodes = _group_by_component(g)
    sites, seg_ids = [], set()

    for comp_id, nodes in comp_nodes.items():
        if g.cable_length(root=nodes[0]) < 5 * min_cable_length:
            continue
        seg_id = int(g.component_id_to_swc_id[comp_id].split(".")[0])

        if _num_somata_in_component(g, comp_id) >= 2:
            seg_ids.add(seg_id)
            sites.append({"segment_id": seg_id,
                          "xyz": tuple(map(float, g.node_xyz[nodes[0]])),
                          "score": 1.0})
            continue

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

    return {"segment_ids": seg_ids, "sites": _dedup(sites, 30.0)}


detections = detect_merges(frag, anisotropy, min_cable_length)
```

Replace or extend the logic above with whatever Phase 1 identified as the strongest
discriminating features. The starter code above is a geometric baseline.

### Scoring

```python
def score(detections, payload, site_tol_um=30.0):
    key_labels  = set(int(x) for x in payload["gt_merge_labels"])
    key_sites   = payload["gt_merge_sites"]
    pred_labels = detections["segment_ids"]

    node_label  = np.asarray(payload["gt_node_canonical_label"])
    adjudicable = set(int(x) for x in np.unique(node_label) if int(x) != 0)

    tp_lab     = pred_labels & key_labels
    recall_seg = len(tp_lab) / len(key_labels) if key_labels else float("nan")
    pred_adj   = pred_labels & adjudicable
    prec_seg   = len(pred_adj & key_labels) / len(pred_adj) if pred_adj else float("nan")

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

**Precision caveat.** A flagged segment not in `gt_merge_labels` may be a false
positive *or* a true merge of untraced cells — GT cannot distinguish them. Compute
precision only over `pred_labels ∩ adjudicable`. Report
`n_pred_labels − n_pred_adjudicable` separately as "flagged, unadjudicable."

### Loaded-cache scope

The cache was loaded and validated at the start of this section. All Phase 2 code
must receive only `frag`, `anisotropy`, `min_cable_length`, and explicitly permitted
reconstruction inputs; answer-key objects remain scoring-only.

---

## 3) Intent

The goal is to understand *why* the UNet makes merge errors at specific locations.
The Phase 1 characterization — comparing merge sites against control points across
topology, geometry, and raw image — is the primary deliverable. The Phase 2 blind
detector validates that the discovered signals are recoverable without GT; its
performance tells you how much of the merge signature is intrinsic to the
reconstruction and how much only becomes visible in the raw fluorescence.

The feature dimensions above are a floor, not a ceiling. The most useful outcome
may be a merge signature that nobody wrote down here — a fluorescence proximity
pattern, a radius discontinuity ratio, a cycle count, a multi-mode intensity
histogram. Use the answer key freely in Phase 1 to check whether a candidate
feature actually separates the two classes. The one non-negotiable is the Phase 2
firewall: the final detector reads fragment features only, so its performance is a
clean, unbiased measure of what the reconstruction alone reveals about the network's
failure modes.

# Blind merge-error detection from fragment features

This document is self-contained: everything you need is here plus one `_add.pkl`
cache file. The task: **detect merge errors from the predicted reconstruction
alone, then check yourself against the ground truth.** The ground truth is a
*grader*, never an *input* to the decision.

It is organized in three parts: **§1 Dataset context** (what the cache is and why
the task is framed blind — no keys or code), **§2 Dataset schema** (the concrete
payload keys, the fragment-graph API, and the detect→score code you actually run),
and **§3 Intent** (the goal the detector serves).

> **The one rule (read this first).** Your merge detector may read **only the
> network fragment information** — `fragments_graph` and its geometry / topology
> (plus the scalar build parameters). It must **never** read `gt_graph`,
> `gt_node_canonical_label`, `gt_edge_error`, `gt_merge_labels`, or
> `gt_merge_sites` while deciding where a merge is. Those five are the **answer
> key**: you touch them only afterward, in a separate scoring step, to measure how
> well the blind detector did. Structure the code so this is guaranteed, not just
> intended — §2 (*The blind detection interface*) shows how.

---

## 1) Dataset context

*Scope: what this cache is and why the task is framed blind. No payload keys, no
API, no code — those are all in §2.*

**Origin.** Each `_add.pkl` was built once, ahead of time, from a plain skeleton
cache (`dataset_cache_<brain_id>_mcl<N>.pkl`) by reading the brain's dense predicted
**segmentation** volume, looking up the predicted segment id at every ground-truth
node's voxel, classifying every ground-truth edge (correct / split / omit / merge),
running the geometric merge-detection walk, and writing all of that back into the
`_add.pkl`. **You do not run any of this** — the labels are already baked in. All
cloud access happened at build time; the `_add.pkl` you have loads with no
segmentation and no credentials. The cache holds two skeleton graphs — the
automated **UNet fragment** reconstruction (`fragments_graph`, hundreds of thousands
of fragment components) and the human **ground-truth** tracings (`gt_graph`, tens of
neurons) — plus the baked-in canonical labels described in §2.

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

**Explore freely — the feature list is a starting point, not a specification.** The
cues catalogued in §2 are the ones we could name up front; they are almost certainly
*not* the complete or the best set. Treat this cache as a dataset to interrogate,
not a checklist to execute. Look at the data from angles this doc does not mention —
plot distributions of any per-node, per-edge, per-junction, or per-component quantity
you can derive from the fragments; cluster components by shape; ask what actually
separates the merged segments from the clean ones; go looking for structure the
narrative here missed (surprising radius patterns, tortuosity, branch-length
statistics, spatial density, connectivity motifs, soma geometry, whatever the data
suggests). Two disciplined uses of the answer key make this *exploration*, not
cheating: (1) as an **exploratory-data-analysis target** — you may inspect
`gt_merge_labels` / `gt_merge_sites` to *understand* what distinguishes a merge and
to *discover* new fragment-only features; and (2) as a **validation harness** — to
score whatever detector you build. The single invariant is the firewall below: the
deployed `detect_merges` must remain a pure function of fragment features, no matter
how you arrived at those features. Novel, well-motivated signals that beat the
starter cues are the goal, not a deviation from it.

**The blindness firewall, conceptually.** Think of the payload as two disjoint
zones: the **reconstruction** (what the detector sees) and the **answer key** (what
grades it). Data flows one way only — key → scorer, never key → detector. §2 lists
exactly which keys fall in each zone and shows how to make the boundary structural.

---

## 2) Dataset schema

*Scope: the concrete payload keys, the fragment-graph API, and the detect→score
code. This is everything you actually read and call.*

### The two zones of the payload

Every `_add.pkl` is one dict. Partition its keys into two zones and never let data
flow from the right zone into the detector:

| Zone | Keys | Role |
|---|---|---|
| **DETECTION INPUT** — the detector may read these | `fragments_graph`; `anisotropy`; `min_cable_length`; `node_spacing`; (optionally `img_path` for raw voxels — see *Optional: raw image* below) | The U-Net reconstruction and its build parameters. This is *all* the detector sees. |
| **ANSWER KEY** — scoring only, never the detector | `gt_graph`; `gt_node_canonical_label`; `gt_edge_error`; `gt_merge_labels`; `gt_merge_sites` | Ground-truth-derived merge truth, used *after* detection to grade it. |

Full schema of every key in the payload:

| Key | Type | Meaning |
|---|---|---|
| `fragments_graph` | `SkeletonGraph` | Automated UNet reconstruction (the detection input; API in the next subsection). |
| `gt_graph` | `SkeletonGraph` | Human ground-truth tracings. **Answer key** — carries the label arrays below. |
| `anisotropy` | `tuple` | µm/voxel in (x, y, z); stored per dataset — read it, don't hard-code. |
| `min_cable_length` | `int` | µm threshold shorter fragments were dropped at (the `<N>` in the filename). |
| `node_spacing` | `int` | Target µm spacing between skeleton nodes. |
| `img_path` | `str` | Public-S3 path of the raw fused image (optional; see *Optional: raw image*). |
| `segmentation_path` | `str` | Provenance only — private-GCS path of the dense segmentation. Not readable here, not needed. |
| `gt_node_canonical_label` | `np.ndarray (N_gt,) int64` | Predicted segment id at each GT node's voxel; `0` = unlabeled. |
| `gt_edge_error` | `np.ndarray (E_gt,) uint8` | Per-GT-edge class, parallel to `list(gt_graph.edges)`: `0=correct, 1=split, 2=omit, 3=merged`. |
| `gt_merge_labels` | `np.ndarray (M,) int64` | Segment ids that fuse ≥2 GT neurons (see below). |
| `gt_merge_sites` | `list[dict]` | One entry per merge site (see below). |

The last four arrays are also attached to `gt_graph` as `gt_graph.node_label`,
`gt_graph.edge_error`, `gt_graph.merge_labels`, and `gt_graph.merge_sites`, so you
can read them off either the payload dict or the graph.

The two keys that actually encode merges — your answer key:

- **`gt_merge_labels`** — `np.ndarray (M,) int64`. Predicted segment ids that fuse
  ≥2 GT neurons. This is the canonical merge set (`labels_with_merge`): the **union**
  of two rules — (a) a *node-count* rule (a segment landing on ≥2 GT neurons with
  >50 GT nodes on each) and (b) a *geometric walk* (walk a fragment from a leaf far
  from GT inward until it re-approaches a *different* GT neuron). This is your
  **segment-level answer key**.
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

### The fragment-graph API (the detection input)

`fragments_graph` is a `SkeletonGraph` (subclass of `networkx.Graph`) whose nodes
carry parallel NumPy arrays. Every feature below is computable from it alone. The
`SkeletonGraph` class is defined in the `agentic_neuron_proofreader` package (import
it before unpickling — the load snippet does this). The API you have:

- `frag.node_xyz` — `(N, 3)` float32, **(x, y, z) microns**. Note: `frag.node_voxel(i)` returns `(z, y, x)` voxel coordinates — a different axis order.
- `frag.node_radius` — `(N,)` float16, skeleton caliber estimate per node.
- `frag.node_component_id` — `(N,)` int; one connected component = one fragment
  skeleton.
- `frag.component_id_to_swc_id` — `dict[int, str]`; each value is a string of the
  form `"<segment_id>.0"` — the segment id stored as a float literal (e.g.
  `'32323215387.0'`). The `.0` is an artifact of how the id was serialised, not a
  meaningful sub-index. `swc_id.split(".")[0]` extracts the segment id as a string;
  wrap in `int(...)` for the integer. The convenience wrappers
  `frag.node_segment_id(node)` → `str` and `frag.node_swc_id(node)` → `str` do the
  split for you and are preferred.
- Topology via NetworkX: `frag.degree[n]`, `frag.neighbors(n)`,
  `nx.connected_components(frag)`.
- Helpers: `frag.dist(i, j)` → float µm (**Euclidean distance between any two nodes** by their xyz coords, not restricted to adjacent nodes);
  `frag.leaf_nodes()` (degree 1); `frag.branching_nodes()` (degree > 2);
  `frag.nodes_within_distance(root, µm)`; `frag.rooted_subgraph(root, µm)`;
  `frag.cable_length(root=node)` → total edge-sum cable length (µm) of the entire
  connected component containing `node` (the `root=` arg identifies the component,
  not a traversal start — result is the same for any node in that component);
  `frag.nodes_with_segment_id(seg_id)` → `set` of all node ids whose segment
  id equals `seg_id` (unions across components when a segment spans several);
  `frag.kdtree` (KD-tree over `node_xyz`); `frag.node_voxel(i)` → (z, y, x) voxel.
- `frag.soma_centroids` — `list` of length K, each element an xyz-µm tuple of a
  detected soma centroid. `frag.soma_component_ids` — `list` of length K (parallel
  to `soma_centroids`), each element the int component id for that centroid. **Both
  are plain Python lists and may be empty** (`K = 0`, as in the current cache); treat
  as a high-confidence bonus signal, not a dependency. A component whose id appears
  ≥2 times in `soma_component_ids` is an almost-certain merge.

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

### What a merge looks like in the fragments graph (feature families)

The feature families below are a **non-exhaustive** starting catalogue — the signals
we could name in advance. They are meant to seed your own investigation, not to
bound it: combine them, replace them, and add families of your own (see "Explore
freely" in §1). Every attribute in the API above is fair game for a feature you
invent, and the answer key is available to *check* whether a candidate feature
actually separates merges from clean segments. The families a blind detector can
start from:

#### (a) Topological — junction structure
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

#### (b) Geometric — angles and straightness at a junction
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

#### (c) Caliber — radius continuity
- **No taper across the junction.** At a true bifurcation the daughter cables are
  thinner than the parent (a Rall-ratio-like relationship). At a merge crossing,
  both through-going cables keep roughly constant, similar caliber — measure mean
  `node_radius` a few microns down each incident branch; two comparably **thick**,
  ≈collinear branches meeting is a merge cue.
- **Radius step.** An abrupt caliber discontinuity along an otherwise smooth cable
  can mark where a foreign process was fused on.

#### (d) Morphological — soma count (strong but conditional)
Each real neuron has exactly one soma. If `frag.soma_centroids` is populated, a
single connected component (or segment) containing **≥2 soma centroids** is an
almost-certain merge. This is the highest-precision cue available — but it only
catches soma-to-soma or soma-adjacent merges, and **only if the cache actually
stored somata** (the array may be empty). Use it as a high-confidence prior, not the
whole detector.

#### (e) Scale — cable and reach
Merged segments tend to be **large and spatially spread**: high total cable length,
a bounding box far larger than a single neurite, or two dense node-clusters far
apart in space bridged by sparse cable. Cheap component-level scalars
(`frag.cable_length(root=nodes[0])`, xyz spread) make an effective **prefilter** to
avoid running the expensive junction analysis on all ~hundreds-of-thousands of
fragments.

> **Honesty about difficulty.** None of these features is individually decisive —
> real neurons occasionally cross themselves, some merges are short and geometrically
> bland, and the fragment skeleton is filtered at `min_cable_length` (short fragments
> were dropped, so some bridges are missing entirely). Expect to **combine** cues
> (e.g. crossing-geometry ∧ caliber-continuity, or a soma-count override) and to
> tune thresholds against the answer key (*Scoring*, below). A single hard rule will
> either over-flag self-crossings or miss subtle fusions.

### Install the package (one-time setup)

Loading any `_add.pkl` requires `agentic_neuron_proofreader` on the Python path —
the package that defines `SkeletonGraph`. Without it `pickle.load` fails
immediately because the `.pkl` stores `SkeletonGraph` instances that must be
reconstructable.

> **Already installed?** Check with `python -c "import agentic_neuron_proofreader"` and skip the steps below if it succeeds. Inside this repo's environment the package is often already importable.

Clone and install from
[`AllenInstitute/neuron-proofreader`](https://github.com/AllenInstitute/neuron-proofreader):

```bash
git clone https://github.com/AllenInstitute/neuron-proofreader.git
cd neuron-proofreader
pip install -e .          # drop -e for a normal (non-editable) install
```

**Python ≥ 3.9 is required** (the package uses `dict` and `list` type-hint syntax
that is not available in 3.8 or earlier).

Runtime dependencies (`numpy`, `networkx`, `scipy`, `tqdm`) are pulled in
automatically. `tensorstore` is only needed for the optional raw-image section and
is also included in the package.

> **Environment gotcha.** `SkeletonGraph` imports `scipy.spatial.KDTree`, so
> `numpy` and `scipy` must be **binary-compatible** in your interpreter. A mismatch
> raises `ValueError: numpy.dtype size changed, may indicate binary incompatibility`
> on import — fix it by installing numpy and scipy together in a fresh environment.

### Loading — cloud-free, input zone only

> **Conda environment.** On Allen Institute HPC nodes the pre-built environment that
> satisfies all binary constraints is `panda`. Activate it before running:
> `conda activate panda`. If you are building your own environment, install
> `numpy` and `scipy` together to ensure binary compatibility.

Point `ADD_PATH` at *any* `_add.pkl` — the code is dataset-agnostic. Filenames follow
`dataset_cache_<brain_id>_mcl<N>_add.pkl`, so glob a cache directory rather than
hard-coding one brain, and read every dataset-specific quantity (anisotropy, mcl,
node spacing) **from the payload**, never as a literal.

```python
import glob, os, pickle
import numpy as np
import agentic_neuron_proofreader  # noqa: F401 — registers SkeletonGraph for unpickling

CACHE_DIR = os.environ.get("ADD_CACHE_DIR", "cache")     # wherever your caches live
add_paths = sorted(glob.glob(os.path.join(CACHE_DIR, "dataset_cache_*_add.pkl")))
assert add_paths, f"no *_add.pkl under {CACHE_DIR}"

ADD_PATH = add_paths[0]                                  # or pick a brain: ...{brain}_mcl{N}_add.pkl
with open(ADD_PATH, "rb") as f:
    payload = pickle.load(f)

frag             = payload["fragments_graph"]            # the ONLY graph the detector sees
anisotropy       = tuple(payload["anisotropy"])          # µm/voxel — read it, don't hard-code
min_cable_length = int(payload["min_cable_length"])      # the <N> from the filename
node_spacing     = payload.get("node_spacing")           # target µm between skeleton nodes
```

The `brain_id` and `<N>` are recoverable from the filename when you need to tag
pooled results across datasets:

```python
import re
def brain_and_mcl(path):
    m = re.search(r"dataset_cache_(\d+)_mcl(\d+)_add\.pkl$", os.path.basename(path))
    return (m.group(1), int(m.group(2))) if m else (os.path.basename(path), None)
```

Use plain `pickle.load` — it is cloud-free and reconstructs both graphs plus the
label arrays without touching the network. The one environment requirement is that
`numpy` and `scipy` be **binary-compatible** in your interpreter (`SkeletonGraph`
imports `scipy.spatial.KDTree`; a mismatch raises `ValueError: numpy.dtype size
changed` on import — install numpy and scipy together). **Memory:** budget well over
20 GB RAM per cache — the reconstructed graphs are far larger than the on-disk file.
Load one brain at a time; to run over a whole collection, loop `add_paths`, reduce
each to the small per-segment records you pool (tagged with `brain_id`), and let
`payload` be garbage-collected before the next.

### The blind detection interface

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
    ...  # feature logic from the feature families above — no gt_* anywhere in scope
    return {"segment_ids": segment_ids, "sites": sites}


detections = detect_merges(frag, anisotropy, min_cable_length)
# `payload` is deliberately NOT passed in — the detector can't cheat.
```

**Helper functions.** The starter detector below calls seven utilities; define them
once before `detect_merges`:

```python
import numpy as np
from collections import defaultdict
from scipy.spatial import KDTree


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
    """True if any antiparallel branch pair (dot < cross_dot) both have mean radius > thick_um."""
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            if np.dot(dirs[i], dirs[j]) < cross_dot:
                if rads[i] > thick_um and rads[j] > thick_um:
                    return True
    return False


def _crossing_score(dirs, rads):
    """Merge confidence 0–1: magnitude of the most antiparallel dot product.
    Near 1.0 = two branches point almost exactly opposite (a through-cable)."""
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

**A concrete starter detector (illustrative).** Cheap prefilter → junction geometry
→ per-segment decision. Tune every threshold against the scoring step below.

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
`MERGE_DEDUP_UM` so your sites are comparable to the key's).

### Scoring against the answer key (the check)

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

**The caveat that governs this whole task: GT is sparse.** Only a handful of neurons
are traced per brain (tens, in a `gt_graph` sitting next to a `fragments_graph` of
hundreds of thousands of components); there is no dense GT — a region with no traced
neuron is simply unlabeled. The answer key therefore contains **only merges that
fuse ≥2 traced neurons**. A blind detector will also flag merges between **untraced**
neurons — those are *real merges the key simply cannot confirm*. So:

- **Recall is well-defined.** Of the GT-confirmed merges (`gt_merge_labels` /
  `gt_merge_sites`), how many did the blind detector catch? Report this directly.
- **Raw precision is not.** A flagged segment that is not in `gt_merge_labels` may
  be a false positive **or** a true merge of untraced cells. Do **not** count all
  such flags as errors.
- **Fix: restrict precision to the adjudicable universe.** Segments whose id appears
  in `gt_node_canonical_label` land on ≥1 traced neuron, so GT *can* rule on them.
  Compute precision only over `pred_labels ∩ adjudicable`. Detections outside that
  set are reported (`n_pred_labels − n_pred_adjudicable`) but not scored as wrong.
  This keeps evaluation honestly *relative to the traced neurons* — the only thing
  the sparse GT can adjudicate.

**Relationship to the thresholds that built the key.** The answer key was built with
GT-dependent thresholds: a segment counts as a merge if it lands on ≥2 GT neurons
with **>50 GT nodes on each** (node-count rule) or if the geometric walk finds a
fragment leaf **>50 µm** from GT that walks back to **within 6 µm** of a *different*
GT neuron, sites deduped at **30 µm**. Every one of those tests references GT, so
none of them is available to you. Your detector's job is to reproduce the *outcome*
of these tests — the `gt_merge_labels` set — from fragment features that use no GT.
The 30 µm dedup is the one threshold you can borrow directly (it is a property of
the fragment skeleton), which is why the starter detector and the scorer both use
it — it keeps your sites and the key's sites on the same footing.

**What "good" looks like.** A useful blind detector achieves **high site-level
recall** and **high adjudicable-segment precision** on each brain, and — when more
than one `_add.pkl` is present — does so **consistently across brains** rather than
being carried by one. Pool per-segment records tagged with `brain_id` (parse it from
the filename with the `brain_and_mcl` helper in the load section). Because merges are
rare, also report absolute counts (`n_key_labels`, TP, flagged-but-unadjudicable)
next to the rates — a recall of "3/4" means something different from "300/400".

### Optional: raw image as a second blind evidence channel

Everything above is fragment-graph-only. If you want *more* evidence than the
skeleton carries — e.g. to confirm that two crossing cables are genuinely two
fluorescent processes and not a tracing artifact — you can read the raw image around
a candidate site. **This is still GT-blind** (the raw fluorescence is not the
answer key), but it goes beyond "network fragment information", so treat it as an
optional secondary channel, not the core detector.

The image is on the **public** AIND open-data S3 bucket (`img_path` in the payload,
an `s3://aind-open-data/...` path); no credentials, only outbound network. The reader
backend is `tensorstore` (shipped with `agentic_neuron_proofreader`). Set
`AWS_EC2_METADATA_DISABLED=true` before opening it so the S3 client does not stall
probing for instance metadata:

```python
import os
from agentic_neuron_proofreader.utils import img_util

os.environ["AWS_EC2_METADATA_DISABLED"] = "true"        # public S3; no credentials
image = img_util.TensorStoreImage(payload["img_path"])  # raw fused image

def xyz_to_voxel(xyz, anisotropy):
    """(x, y, z) µm -> (z, y, x) integer voxel."""
    return tuple(int(c / a) for c, a in zip(xyz, anisotropy))[::-1]

center = xyz_to_voxel(site["xyz"], anisotropy)          # site from detect_merges
patch  = image.read(center, (128, 128, 128))            # (z, y, x) patch, from S3
```

Keeping this out of `detect_merges`'s signature preserves the firewall — if you use
it, pass the opened image reader in explicitly, and still never pass `payload`.

> The dense **segmentation** remains intentionally inaccessible (private GCS,
> provenance only). You do not need it: the merge signal is either in the fragment
> skeleton (the feature families above) or, optionally, in the public raw image
> (this section).

---

## 3) Intent

*Scope: the goal the detector serves and the bar for success.*

The goal is a **ground-truth-blind merge detector** — one that finds fused segments
from the reconstruction's own geometry and topology, so it works on the untraced
neurons that dominate every brain and on brains with no tracing at all. The stored
`gt_merge_labels` / `gt_merge_sites` are a **validation harness** (and a legitimate
target for *exploratory* feature discovery), not a runtime input: they tell you,
after the fact, how close your feature-based decisions came to the canonical answer.
A detector that scores well *only* because it peeked at the labels at decision time
is worthless here; a detector that scores well while provably blind is exactly the
component a post-hoc proofreading tool needs to *resolve* merges (cut the fused
segment at the detected site) without a human first tracing the neuron.

How you *get* to that detector is wide open. The feature families in §2 are a floor,
not a ceiling — the most useful outcome of this task may well be a merge signature
nobody wrote down here, surfaced by looking at the fragment data from an angle this
doc did not anticipate. Treat the cache as something to explore and be curious
about: characterize the fragments, test hypotheses against the answer key, and let
the data redirect you. The one non-negotiable is the firewall — the final decision
function reads fragment features only — but within it, prize discovery over
compliance.

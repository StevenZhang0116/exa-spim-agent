# Understanding split errors in ExaSPIM reconstructions (V3)

The scientific question: **what distinguishes a split — a place where the network
broke one neuron into two predicted segments — from the tens of thousands of
fragment endings that are genuine, across the fragment reconstruction and the raw
fluorescence image?** Useful evidence may be topological, geometric, image-level, or
something not anticipated by this document; those categories organize historical
examples rather than constrain discovery.

> **V3 discovery scope — two candidate views, no prescribed feature answer.**
> Pair-level discovery must study both `tip_to_tip` and `tip_to_any_node` candidates.
> In both views the anchor is a degree-1 fragment tip; the second view expands the
> receiving side to any node on another valid segment, including non-tip shaft or
> branch nodes. Use one shared GT-blind enumeration implementation and report the
> restricted tip-target view and the expanded any-node-target view separately. Do
> not treat the feature catalogues, formulas, mechanisms, directions, or thresholds
> in this document as a required discovery checklist or expected answer. They are
> non-binding examples and baselines. The discovery process is free to propose,
> reject, or replace features, provided that it covers both candidate views, records
> which view each hypothesis applies to, and validates all claims without leakage.

Work proceeds in two phases that use the ground truth differently:

- **Phase 1 — Characterization (GT-informed).** Derive split sites from
  `gt_node_canonical_label` / `gt_edge_error` (see *Deriving the split answer key*),
  then use them to label fragment endpoints: a leaf at a split site is a
  **truncation** (positive), a leaf where the traced neuron genuinely ends is a
  **terminal** (negative). Freely discover and test defensible measurements at every
  endpoint and across both pair-candidate views; do not fill a predefined feature
  checklist. Compare distributions to understand where the network drops a process
  and what it saw in the raw image at a break vs. at a real neurite terminal. This is
  the core analysis.
- **Phase 2 — Blind detection (GT-blind).** Build a detector from the measurements
  supported in Phase 1 that reads only the fragment reconstruction — no GT anywhere
  in scope. Run it on both candidate views, then grade against the derived key. High
  performance validates the characterization; a gap between characterization and
  detection reveals which signals require GT to locate and which are recoverable from
  the reconstruction alone.

Both phases run at two levels — per endpoint, then per candidate segment pair. See
*Two detection units* for why these units answer complementary questions. Pair-level
work must not assume that the receiving node is also an endpoint.

> **The one rule.** The Phase 2 detector may read **only the network fragment
> information** (`fragments_graph` and its geometry / topology, plus scalars and
> optionally the raw image). It must **never** read `gt_graph`,
> `gt_node_canonical_label`, `gt_edge_error`, `gt_merge_labels`, or `gt_merge_sites`
> while deciding where a split is. GT may shape the detector at **build time** —
> feature choice, thresholds, and the class-balanced fits this document
> recommends are all trained on Phase 1 labels — the rule constrains its
> **runtime inputs** only. In Phase 1 you are explicitly allowed — and
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

**What a "split" is.** Deep-learning segmentation makes two systematic topological
errors: **merges** (one predicted segment fusing two or more distinct neurons) and
**splits** (one neuron broken into several predicted segments). A split is a place
along a single true neuron where the predicted segment id changes: the network
followed the process, lost it, and resumed it under a new label — a dim stretch, a
thin distal branch, an abrupt turn, a crossing vessel, a tile boundary. Every split
is a **join that should have been made and was not**.

**Splits and merges are dual, and the duality is the whole difficulty.** A merge is a
location *inside one* fragment component — you detect a node. A split is a *relation
between two* components — you detect a **pair**, plus the gap between them. That
changes everything downstream: the detection unit, how controls are sampled, and how
performance is scored. It also means the two error types trade against each other:
joining aggressively removes splits and manufactures merges. A split detector that
ignores that trade-off is not measurable (see *Scoring*, and the end-to-end section
that follows it).

**Why this matters.** Splits dominate the error budget of whole-brain
reconstructions: they are what keeps expected run length (ERL) far below total cable
length, and they are the bulk of what a human proofreader spends time re-joining. In
a fresh brain, or on the vast majority of neurons never traced, there is no split
label to read; a proofreading tool has to find the breaks from the reconstruction's
own geometry and the raw image. Understanding *why* the network drops a process — in
what image conditions and at what geometric configurations — is the prerequisite for
building that tool and for improving the segmentation model itself.

**GT is sparse, and for splits it is sparse in a second way.** Only a handful of
neurons are traced per brain, so the answer key sees a split only where it happens on
a traced neuron. But there is an additional, split-specific ceiling:

> **`min_cable_length` removes split partners from the fragments graph entirely.**
> Fragments shorter than `min_cable_length` µm were dropped at build time. A split
> whose far side is a short stub therefore has **no component to join to** — the
> partner does not exist in `fragments_graph` at all, so no detector reading only the
> fragments can ever recover it. This is not a small effect: on `794495` at
> `mcl100`, **6,396 of the 9,623 segments that label GT have no fragment component
> at all**. Before scoring, split the key into *reachable* pairs (both segment ids
> have at least one component in `fragments_graph`) and *unreachable* pairs, and
> report recall against the reachable set with the unreachable count stated
> separately. Reporting one number over the full key silently charges the detector
> for data that was deleted.

What else stays invisible to GT: a split between two segments that never touch a
traced neuron; a split lying entirely inside an omit stretch (both sides unlabeled);
and — in the opposite direction — pairs that GT *can* adjudicate as bad joins,
because the two segments sit on two *different* traced neurons. That last category is
valuable: it is the only unambiguous false-positive class (see *Scoring*).

---

## 2) Dataset schema

### Environment and loading — run this first

All later code blocks assume this block has run. Python 3.10 or newer is required.
Check the package with `python -c "import agentic_neuron_proofreader"`. If that
fails, install the official package and analysis dependencies:

```bash
python -m pip install "git+https://github.com/AllenInstitute/agentic-neuron-proofreader.git" "numpy>=2" scipy networkx tensorstore matplotlib
```

The `_add.pkl` was pickled under NumPy 2.x, so loading it **requires `numpy>=2`**
with a scipy built against it — install them together, as above. Under NumPy 1.x,
`pickle.load` fails with a misleading
`ModuleNotFoundError: No module named 'numpy._core...'`; that error means a
NumPy-version mismatch in your environment, not a corrupt cache.

Run from the delivery directory containing exactly one `_add.pkl`. Loading may use
tens of GB of RAM, so process one cache at a time:

```python
import os
import pickle
from collections import defaultdict
from pathlib import Path

import networkx as nx
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
anisotropy = tuple(payload["anisotropy"])
min_cable_length = int(payload["min_cable_length"])
node_spacing = payload.get("node_spacing")
gt_edges = list(gt.edges)

if len(node_label) != gt.number_of_nodes():
    raise RuntimeError("gt_node_canonical_label is not parallel to GT nodes")
if len(edge_error) != len(gt_edges):
    raise RuntimeError("gt_edge_error is not parallel to list(gt_graph.edges)")
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
| `min_cable_length` | `int` | µm threshold; shorter fragments were dropped at build time. **Read this for splits — it bounds achievable recall.** |
| `node_spacing` | `int` (optional) | Target µm spacing between skeleton nodes — per cache (`5` for `mcl100`, `2` for `789202_mcl10`), and only a *target*: each irreducible edge is spline-resampled to `max(int(len/node_spacing), 5)` points, so short edges are denser. Node counts are therefore not proportional to µm. May be absent in older caches — the loader reads it with `payload.get`. |
| `img_path` | `str` | Raw fused-image path: public `s3://` or authenticated `gs://`. Image features are optional for a fragment-only detector; configure access as shown below before reading them. |
| `segmentation_path` | `str` | Provenance only — private GCS, not readable here. |
| `fragments_path`, `gt_path` | `str` | Provenance only — source GCS paths of the SWCs; not read to use the cache. |
| `gt_node_canonical_label` | `np.ndarray (N_gt,) int64` | Predicted segment id at each GT node's voxel; `0` = unlabeled. **The primary split signal** — a split is a change in this array along a neuron. |
| `gt_edge_error` | `np.ndarray (E_gt,) uint8` | Per-GT-edge class: `0=correct, 1=split, 2=omit, 3=merged`. |
| `gt_merge_labels` | `np.ndarray (M,) int64` | Segment ids flagged as merges. Used here only as a *contra-indication*: a merge segment already spans two neurons, so joining onto it compounds an existing error. |
| `gt_merge_sites` | `list[dict]` | One entry per merge site: `{"segment_id": int, "gt_neuron": str, "xyz": (x, y, z) µm}`. |

The last four are also attached to `gt_graph` as `gt_graph.node_label`,
`gt_graph.edge_error`, `gt_graph.merge_labels`, and `gt_graph.merge_sites`.

> **There is no `gt_split_sites` array.** Merges ship with an explicit site list
> because locating them required a geometric walk over the fragments at build time.
> Splits do not: they are fully determined by `gt_node_canonical_label`, so the cache
> stores the labels and leaves the site derivation to you. *Deriving the split answer
> key* below is that derivation — it is the split analogue of `gt_merge_sites` and
> everything else in this document depends on it.

### How canonical scoring defines a split

From `segmentation_skeleton_metrics/skeleton_metrics.py`, so that the key you derive
matches the pipeline the project is graded against:

| Canonical metric | Definition | Recovery from the `_add` cache |
|---|---|---|
| `# Splits` (`SplitCountMetric`) | `max(len(graph.node_labels()) − 1, 0)` — **distinct nonzero predicted labels on the neuron, minus 1** | group `gt_node_canonical_label` by GT neuron, count distinct nonzero ids, subtract 1 |
| `% Split Edges` (`SplitEdgePercentMetric`) | fraction of GT edges where **both endpoint labels are nonzero and different** | `100 × count(gt_edge_error == 1) / len(gt_edge_error)` |
| `Split Rate` (`SplitRateMetric`) | `segmented_run_length / # Splits` (µm per split), with `segmented_run_length = run_length × (1 − %omit/100 − %split/100)` | derive from the two columns above plus the neuron's cable length |
| `ERL` (`ERLMetric`) | run-length-weighted average of per-label run lengths; a label in `labels_with_merge` contributes **0** | the metric splits punish, and the one a join must improve |

**Three consequences you must design around.**

1. **`# Splits` counts labels, `% Split Edges` counts transitions.** A segment that
   leaves the neuron and comes back (label sequence `A B A`) produces **two** split
   edges but adds only **one** to `# Splits`, because `A` is one distinct label.
   Label flicker at a noisy boundary inflates split edges without inflating
   `# Splits`. Deduplicate transitions into *pairs* before treating them as targets.
2. **`% Split Edges` cannot see a split that spans an omit gap.** The canonical test
   requires *both* endpoints nonzero. Where the reconstruction physically stopped and
   restarted, the intervening GT nodes are unlabeled, so those edges are classified
   `omit`, not `split` — yet `# Splits` still counts both labels. These
   *gap-spanning* splits are real, are often the ones with a wide physical gap, and
   are exactly what a join has to bridge. Derive them from label runs, not from
   `gt_edge_error` alone (the code below does).
3. **Misalignment healing decides which of the two you get.** At build time
   `fix_label_misalignments` fills a run of unlabeled GT nodes with a single label
   *when exactly one nonzero label collides at the run's boundary*. If two different
   labels collide, the run stays unlabeled. So a break flanked by two different
   segments survives as an **omit run**, while a labeling dropout inside one segment
   is silently absorbed. Expect the gap-distance distribution at split sites to be
   **bimodal**: near-zero for directly abutting labels, wider for the healed-out omit
   runs.

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
  **The map is component → segment id, not a bijection**: one segment id can own more
  than one component (its skeleton need not be connected after filtering). Always
  group `segment_id → [component_id, ...]`, never assume 1:1 — a split key is stated
  in segment ids, but the geometry you measure lives on components.
- Topology via NetworkX: `frag.degree[n]`, `frag.neighbors(n)`,
  `nx.connected_components(frag)`.
- Helpers: `frag.dist(i, j)` → float µm (**Euclidean distance between any two nodes**
  by their xyz coords, not restricted to adjacent nodes);
  `frag.leaf_nodes()` (degree 1); `frag.branching_nodes()` (degree > 2);
  `frag.nodes_within_distance(root, µm)` (**geodesic** — walks the skeleton from
  `root`, so it stays inside one component and measures cable distance);
  `frag.rooted_subgraph(root, µm)`;
  `frag.cable_length(root=node)` → total edge-sum cable length (µm) of the entire
  connected component containing `node` (result is the same for any node in that
  component);
  `frag.nodes_with_segment_id(seg_id)` → `set` of all node ids whose segment id
  equals `seg_id`;
  `frag.kdtree` (KD-tree over `node_xyz` — **Euclidean**, and its hits can land on
  other components); `frag.node_voxel(i)` → (z, y, x) voxel.
- `frag.soma_centroids` — `list` of length K, each element an xyz-µm tuple.
  `frag.soma_component_ids` — `list` of length K (parallel to `soma_centroids`),
  each element the int component id for that centroid. **Both are plain Python lists
  and may be empty** (`K = 0` in the current cache); guard for length zero before
  iterating.

**Grouping nodes by component and segment (use throughout):**

```python
comp_nodes = defaultdict(list)
for n in frag.nodes:
    comp_nodes[int(frag.node_component_id[n])].append(n)

def comp_segment_id(comp_id):
    return int(frag.component_id_to_swc_id[comp_id].split(".")[0])

seg_comps = defaultdict(list)          # segment id -> [component ids]  (1 : N)
for comp_id in comp_nodes:
    seg_comps[comp_segment_id(comp_id)].append(comp_id)
```

### Deriving the split answer key

This replaces `gt_merge_sites`. Walk each GT neuron's label sequence and record every
**adjacency between two distinct nonzero labels** — directly across an edge, or
across a run of unlabeled nodes. Each adjacency is one split: a join that should
exist between two predicted segments.

```python
UNLABELED = 0
EDGE_CORRECT, EDGE_SPLIT, EDGE_OMIT, EDGE_MERGED = 0, 1, 2, 3


def derive_split_key(gt, node_label, edge_error):
    """
    ANSWER KEY (Phase 1 / scoring only).

    Returns {(gt_neuron, frozenset({seg_a, seg_b})): {"xyzs": [...], "kind": str}}
    where "kind" is:
      "direct"   -- the two labels abut across a single GT edge (a canonical
                    split edge, gt_edge_error == EDGE_SPLIT),
      "gap"      -- the two labels flank a run of unlabeled GT nodes; canonical
                    counts these edges as OMIT, but `# Splits` still counts both
                    labels, and a join still has to bridge them,
      "both"     -- observed in both forms along the same neuron.

    "xyzs" holds every observed site for that pair (µm): the midpoint of each
    split edge, and the centroid of each unlabeled run.
    """
    node_label = np.asarray(node_label)
    xyz = np.asarray(gt.node_xyz)
    neuron_of = {n: gt.node_segment_id(n) for n in gt.nodes}
    key = {}

    def add(neuron, a, b, site, kind):
        k = (neuron, frozenset((int(a), int(b))))
        rec = key.setdefault(k, {"xyzs": [], "kind": kind})
        rec["xyzs"].append(tuple(map(float, site)))
        if rec["kind"] != kind:
            rec["kind"] = "both"

    # (a) direct adjacency -- exactly the canonical split edges
    for k, (i, j) in enumerate(gt.edges):
        if edge_error[k] != EDGE_SPLIT:
            continue
        add(neuron_of[i], node_label[i], node_label[j],
            0.5 * (xyz[i] + xyz[j]), "direct")

    # (b) adjacency across an unlabeled (omit) run -- invisible to % Split Edges
    zero_nodes = [n for n in gt.nodes if int(node_label[n]) == UNLABELED]
    for cc in nx.connected_components(gt.subgraph(zero_nodes)):
        flank = {int(node_label[m])
                 for n in cc for m in gt.neighbors(n)
                 if int(node_label[m]) != UNLABELED}
        if len(flank) < 2:
            continue                      # a dropout inside one segment, not a split
        centroid = np.mean([xyz[n] for n in cc], axis=0)
        neuron = neuron_of[next(iter(cc))]
        flank = sorted(flank)
        for x in range(len(flank)):
            for y in range(x + 1, len(flank)):
                add(neuron, flank[x], flank[y], centroid, "gap")

    return key


split_key = derive_split_key(gt, node_label, edge_error)
```

**Reachability.** Immediately partition the key by whether the fragments graph can
even express the join:

```python
def partition_reachable(split_key, seg_comps):
    """Split the key into pairs whose BOTH segments have a fragment component
    (a detector can reach them) and pairs it cannot (min_cable_length dropped a
    partner, or the segment has no skeleton). Recall must be reported against
    the reachable set, with the unreachable count stated alongside."""
    reachable, unreachable = {}, {}
    for (neuron, pair), rec in split_key.items():
        a, b = tuple(pair)
        (reachable if (a in seg_comps and b in seg_comps) else unreachable)[
            (neuron, pair)] = rec
    return reachable, unreachable


reachable_key, unreachable_key = partition_reachable(split_key, seg_comps)
print(f"reachable {len(reachable_key)} / unreachable {len(unreachable_key)}")
```

### Two detection units — keep their roles distinct

A merge has one natural detection unit: a location inside a component. A split has
**two**, and they are not equivalent.

| | **A — Endpoint level** | **B — Pair level** |
|---|---|---|
| Question | Is this fragment leaf a **truncation**, or a genuine neurite terminal? | **Which two** segments should be joined? |
| Unit | one tip node | one unordered segment pair, represented by one or more tip-to-node occurrences |
| Mirrors the merge doc | ✓ directly — a per-location classification | ✗ a relation, no merge analogue |
| Needs the partner to exist in `fragments_graph`? | **No** | **Yes** |
| Ceiling from `min_cable_length` | mild | severe (see *Reachability*) |

**Unit A directly characterizes the surviving tip.**
A split site is where the network stopped following a process; that failure is
written into the *surviving* end — flat caliber, a cable pointing into a dim gap —
whether or not the far side was ever reconstructed. Because the answer key for A is
derived from GT labels alone, **a truncation stays adjudicable even when
`min_cable_length` deleted its partner**: the pair is unreachable, but the endpoint is
still visibly broken. That is why A has a materially higher recall ceiling than B, and
why endpoint characterization in Phase 1 should include it. One class escapes it
entirely — the branch-point break, whose surviving trunk has no leaf at the site;
count it with `partition_leaf_visible` below.

**Unit B names the relation a downstream corrector eventually needs.** Naming a split
completely means naming both sides. Study it alongside A: score endpoint truncation,
then compare how well the same GT-blind enumeration recovers and ranks segment pairs
under the `tip_to_tip` and `tip_to_any_node` views. A non-tip receiving node is valid
in the expanded view and must not be replaced by an arbitrary leaf.

Everything below is presented in that order: the endpoint-level scaffold first, then
the pairwise one built on top of it.

**Known invisible class — count it before scoring.** Unit A classifies leaves,
so a split whose surviving side has no leaf near the site — a daughter branch
dropped at a bifurcation, with the trunk passing straight through — is
structurally invisible to it; when `min_cable_length` also deleted the daughter,
it is invisible to unit B as well. Partition the key and report the invisible
count alongside every endpoint-level recall, never silently folded into the
denominator:

```python
def partition_leaf_visible(split_key, frag, tol_um=15.0):
    """Key pairs with at least one fragment LEAF of either segment within
    tol_um of a site are endpoint-visible; the rest unit A cannot express."""
    leaves   = [n for n in frag.nodes if frag.degree[n] == 1]
    leaf_seg = [comp_segment_id(int(frag.node_component_id[n])) for n in leaves]
    tree     = KDTree(np.asarray(frag.node_xyz)[leaves])
    visible, invisible = {}, {}
    for (neuron, pair), rec in split_key.items():
        hit = any(leaf_seg[i] in pair
                  for xyz in rec["xyzs"]
                  for i in tree.query_ball_point(xyz, tol_um))
        (visible if hit else invisible)[(neuron, pair)] = rec
    return visible, invisible


leaf_visible_key, leaf_invisible_key = partition_leaf_visible(split_key, frag)
print(f"endpoint-visible {len(leaf_visible_key)} / invisible {len(leaf_invisible_key)}")
```

*Beyond the catalogue* below sketches how a mid-cable probe could recover part of
the invisible class.

### The comparison scaffold — A, endpoint level

The core analysis is a **contrast between fragment endings**: every degree-1 node in
`fragments_graph` is either a place the network gave up (positive) or a place the
neuron genuinely ends (negative). This is the direct analogue of the merge doc's
merge-sites-vs-control-points contrast.

```python
def derive_truncation_key(gt, frag, split_key,
                          near_gt_um=10.0, site_tol_um=15.0):
    """
    ANSWER KEY, endpoint level (Phase 1 / scoring only).

    Classifies every fragment leaf that sits close enough to a traced neuron to be
    adjudicated at all:
      1    -- TRUNCATION: the leaf sits at a split site involving its own segment.
      0    -- GENUINE TERMINAL: the leaf sits where the traced neuron itself ends
              (the nearest GT node is a degree-1 node of gt_graph).
      None -- AMBIGUOUS: the leaf sits beside mid-cable that is neither a transition
              nor an ending -- usually a skeletonization spur. Exclude from fits;
              do not silently fold into either class.
    Leaves farther than near_gt_um from any GT node are omitted entirely: untraced
    space carries no verdict, exactly as in the merge doc's precision caveat.
    """
    gt_nodes = np.asarray(list(gt.nodes))
    gt_xyz = np.asarray(gt.node_xyz)[gt_nodes]
    gt_tree = KDTree(gt_xyz)

    site_xyz, site_segs = [], []
    for (_, pair), rec in split_key.items():
        for xyz in rec["xyzs"]:
            site_xyz.append(xyz)
            site_segs.append(pair)
    site_tree = KDTree(site_xyz) if site_xyz else None

    def seg_of_node(n):
        comp = int(frag.node_component_id[n])
        return int(frag.component_id_to_swc_id[comp].split(".")[0])

    labels = {}
    for leaf in (n for n in frag.nodes if frag.degree[n] == 1):
        xyz = np.asarray(frag.node_xyz[leaf], float)
        d, k = gt_tree.query(xyz)
        if d > near_gt_um:
            continue                                  # untraced -- unadjudicable
        gt_node = int(gt_nodes[k])
        seg = seg_of_node(leaf)

        at_site = False
        if site_tree is not None:
            for i in site_tree.query_ball_point(xyz, site_tol_um):
                if seg in site_segs[i]:               # this leaf's OWN segment
                    at_site = True                    # is one side of the split
                    break

        if at_site:
            labels[leaf] = 1
        elif gt.degree[gt_node] == 1:
            labels[leaf] = 0
        else:
            labels[leaf] = None
    return labels


trunc_key = derive_truncation_key(gt, frag, split_key)
positives = [n for n, v in trunc_key.items() if v == 1]
negatives = [n for n, v in trunc_key.items() if v == 0]
```

Requiring the leaf's **own** segment id to be one side of the split site is what makes
this key sound: a leaf that merely happens to lie near some other pair's split site is
not itself truncated. Report the three class counts (and the number of leaves dropped
as untraced) before doing anything else — the positive class is small, and any fit
needs class balancing.

**Stratify the contrast by the key's `kind`.** A `direct` transition marks
label-boundary uncertainty; a `gap` run marks evidence that physically ran out.
Their signatures need not agree — a feature that cleanly separates one kind can
dilute into noise when the kinds are pooled — so map each positive leaf to the
`kind` of its matched site and report every feature's separation per kind
alongside the pooled number.

### The comparison scaffold — B, pair level

Built on top of A: a candidate is an unordered pair of valid segment IDs that a
detector might connect, represented geometrically by a tip anchor and a receiving
node on the other segment. Positives are candidates that realise a key pair;
negatives are candidates that do not.

**Candidate generation (GT-blind — reuse it verbatim in Phase 2).** Use the same
enumeration engine for two declared views: `tip_to_tip` restricts receiving nodes to
degree-1 tips, while `tip_to_any_node` allows every node on another valid segment.
The expanded view must keep an explicit `partner_is_tip`/candidate-type audit so its
tip-target and non-tip-target strata can be reported separately. These views define
where discovery must look; they do not prescribe which features it should find.

```python
def candidate_joins(g, radius_um, per_anchor_k, partner_scope="any_node"):
    """
    GT-BLIND. Anchor every search at a degree-1 tip. Rank DISTINCT partner
    segment IDs by their nearest eligible receiving node and retain at most
    per_anchor_k per anchor. Globally deduplicate unordered segment pairs while
    retaining all qualifying occurrences for feature discovery and audit.

    partner_scope="tip"      -> tip_to_tip baseline
    partner_scope="any_node" -> tip_to_any_node expanded view
    """
    xyz = np.asarray(g.node_xyz)
    comp = np.asarray(g.node_component_id)
    tips = np.asarray([int(n) for n in g.nodes if g.degree[n] == 1])
    targets = (tips if partner_scope == "tip"
               else np.asarray([int(n) for n in g.nodes]))
    target_tree = KDTree(xyz[targets])

    def seg_of_node(node):
        component = int(comp[node])
        return int(str(g.component_id_to_swc_id[component]).split(".")[0])

    grouped = defaultdict(list)
    for anchor in tips:
        anchor_seg = seg_of_node(int(anchor))
        if anchor_seg == 0:
            continue
        nearest_by_partner_segment = {}
        for target_index in target_tree.query_ball_point(xyz[anchor], radius_um):
            partner = int(targets[int(target_index)])
            partner_seg = seg_of_node(partner)
            if partner == anchor or partner_seg in {0, anchor_seg}:
                continue
            gap = float(np.linalg.norm(xyz[anchor] - xyz[partner]))
            record = (gap, partner, int(comp[partner]))
            old = nearest_by_partner_segment.get(partner_seg)
            if old is None or record < old:
                nearest_by_partner_segment[partner_seg] = record

        ranked = sorted(
            (gap, partner_seg, partner, partner_comp)
            for partner_seg, (gap, partner, partner_comp)
            in nearest_by_partner_segment.items()
        )[:per_anchor_k]
        for rank, (gap, partner_seg, partner, partner_comp) in enumerate(ranked, 1):
            pair = tuple(sorted((anchor_seg, partner_seg)))
            grouped[pair].append({
                "nodes": (int(anchor), int(partner)),
                "comps": (int(comp[anchor]), int(partner_comp)),
                "gap_um": gap,
                "partner_rank": rank,
                "partner_is_tip": bool(g.degree[partner] == 1),
            })

    candidates = []
    for segment_pair, occurrences in sorted(grouped.items()):
        occurrences = tuple(sorted(
            occurrences,
            key=lambda row: (row["gap_um"], row["nodes"]),
        ))
        representative = dict(occurrences[0])
        representative.update({
            "segment_ids": segment_pair,
            "occurrences": occurrences,
            "has_tip_target": any(row["partner_is_tip"] for row in occurrences),
            "has_non_tip_target": any(not row["partner_is_tip"]
                                      for row in occurrences),
        })
        candidates.append(representative)
    return candidates


tip_to_tip_candidates = candidate_joins(
    frag, radius_um=RADIUS_UM, per_anchor_k=PER_ANCHOR_K,
    partner_scope="tip")
tip_to_any_node_candidates = candidate_joins(
    frag, radius_um=RADIUS_UM, per_anchor_k=PER_ANCHOR_K,
    partner_scope="any_node")
```

**Select `radius_um` and `per_anchor_k` transparently, then freeze them.** Measure the
candidate-recall/workload trade-off for both views on training data only, declare the
selection rule, and freeze the policy before held-out evaluation. For the expanded
view, distance means the eligible tip-to-receiving-node gap, not necessarily a
tip-to-tip gap. A hand-set radius or quota silently caps pair recall; a held-out GT
set must never be used to retune either value.

Before feature discovery, report for each view: total and distinct segment-pair
counts, reachable direct/gap truth coverage, candidate prevalence, and merge-creation
risk. Within `tip_to_any_node`, also report candidates supported by tip targets,
non-tip targets, or both. A hypothesis may apply to either or both strata, but its
applicability and undefined rate must be explicit. Do not require equal hypothesis
counts per stratum and do not prescribe feature families; require only that neither
stratum is silently excluded from discovery, confirmation, or the final report.

**Labelling candidates for Phase 1.** A candidate is positive when its two components'
segment ids form a key pair. Negatives come in two grades, and they are *not*
interchangeable:

```python
from agentic_neuron_proofreader.data_modules import canonical_labeling as cl

key_pairs = {pair for (_, pair) in split_key}
seg_neuron_counts = cl.segment_neuron_node_counts(gt, node_label)   # seg -> {neuron: n}
labeling_segs = set(seg_neuron_counts)

MERGE_MIN_NODES = 50   # canonical threshold for "substantially traced"

def substantial_neurons(seg):
    return {n for n, c in seg_neuron_counts.get(seg, {}).items() if c > MERGE_MIN_NODES}


def label_candidate(cand):
    """1 = true split (join it), 0 = confirmed bad join, None = unadjudicable."""
    a = comp_segment_id(cand["comps"][0])
    b = comp_segment_id(cand["comps"][1])
    if a == b:
        return None                                # same segment, two components
    if frozenset((a, b)) in key_pairs:
        return 1
    na, nb = substantial_neurons(a), substantial_neurons(b)
    if na and nb and not (na & nb):
        return 0                                   # HARD negative: joining fuses
                                                   # two traced neurons -> a merge
    if a in labeling_segs and b in labeling_segs:
        return 0                                   # soft negative: both on GT,
                                                   # not adjacent on any neuron
    return None                                    # untraced space -- no verdict
```

The **hard negatives** are the ones worth over-weighting in any fit: they are the
mistakes that actively destroy ERL, because the canonical `ERLMetric` scores a merged
label's run length as **0**. A missed split costs you the length of one fragment; a
wrong join costs you the *entire* run length of both neurons.

### Historical feature catalogue

The v2 tables are retained as optional, auditable baselines grouped into three
historical dimensions. V3 does not require extracting every dimension—or any listed
row—before proposing something else. *Beyond the catalogue* below discusses signals
these examples may miss.

**Rows marked (B) need both endpoints** and are available only at pair level.
Everything unmarked is **single-sided** — computable from one leaf and its own cable,
so it can feed the endpoint-level classifier of unit A. This notation documents each
example's applicability; it is not an instruction to build the unmarked set first or
to prefer one family. For any retained or newly discovered measurement, declare
whether it is defined for Unit A, `tip_to_tip`, tip-target `tip_to_any_node`, non-tip-
target `tip_to_any_node`, or some combination.

#### Dimension 1 — Topology and endpoint character

| Feature | How to compute |
|---|---|
| Component node count | `len(comp_nodes[c])` for the leaf's own component |
| Component cable length | `g.cable_length(root=n)` — short components are over-represented among split pieces |
| Leaves / branch points in component | counts over `comp_nodes[c]` with `g.degree` |
| Leaf's path distance to the nearest branch point | a stub hanging off a junction behaves differently from a long terminal run |
| Number of cross-component endpoints nearby | `len(tree.query_ball_point(xyz[n], r))` restricted to other components — a truncation usually has *something* on the far side |
| Distance to the nearest cross-component endpoint | the single strongest topological A-feature: a genuine terminal ends in empty space |
| Competitor density | how many of those neighbours there are — high counts mark dense neuropil |
| Endpoint position | `frag.node_xyz[n]` raw (x, y, z) — Phase 2-legal; tiling/illumination-caused splits cluster spatially (see *Beyond the catalogue*) |
| Distance to nearest candidate tile plane | derive the plane spacing in Phase 1 from key-site coordinate histograms, then compute blind |
| Endpoint degrees **(B)** | `(g.degree[na], g.degree[nb])` — `1-1` is a clean break; `1-2`/`1-3` a flank landing |
| Reciprocal-nearest flag **(B)** | is `nb` the nearest cross-component endpoint of `na` *and* vice versa? |
| Partner rank **(B)** | rank of this candidate by gap among `na`'s candidates |
| Would the join create a cycle? **(B)** | union-find over components as joins are accepted |
| Degree after join **(B)** | `g.degree[nb] + 1` — a landing that makes degree 4 is suspect |

A short component with two facing stubs and no competitor is the archetypal split
fragment. High competitor counts mark dense neuropil, where a wrong join is likely.
Reciprocity is a cheap, strong prior: real breaks pair mutually. Note that
*distance to the nearest cross-component endpoint* is single-sided even though it
refers to a neighbour — it asks "is anything over there?", not "is that the partner?",
so it survives when the true partner was filtered out.

#### Dimension 2 — Geometry

| Feature | How to compute |
|---|---|
| **Taper ratio** | `taper_ratio(g, n)` — tip radius ÷ shaft radius. A **true terminal tapers; a cut end does not.** The decisive single-sided feature. |
| Tip radius | `float(g.node_radius[n])` — thin endings are more often genuine |
| Caliber variance along the stub | `np.std(_stub_radii(g, n))` — a clean cut leaves a flat profile |
| Stub straightness | end-to-end distance ÷ cable length over the last ~20 µm |
| Curvature at the tip | change in `leaf_direction` over the last few nodes — a process bending sharply is where the network loses track |
| Distance from the component's bounding-box centre | distal tips vs. interior stubs |
| Gap distance **(B)** | `g.dist(na, nb)` — expect a bimodal key distribution (see *Three consequences*, item 3) |
| Alignment of A toward B **(B)** | `dot(leaf_direction(g, na), unit(xyz[nb] − xyz[na]))` |
| Alignment of B toward A **(B)** | `dot(leaf_direction(g, nb), unit(xyz[na] − xyz[nb]))` |
| Stub antiparallelism **(B)** | `−dot(leaf_direction(g, na), leaf_direction(g, nb))` — collinear cables ≈ +1 |
| Turning angle of the bridged path **(B)** | angle between incoming tangent and the bridge vector |
| Radius match **(B)** | `abs(r_a − r_b) / mean(r_a, r_b)` over the stub tips |
| Gap relative to caliber **(B)** | `gap_um / mean(r_a, r_b)` — a two-radius gap is a break, a fifty-radius gap is not |
| Detour ratio **(B)** | shortest in-graph path between the components (if any) ÷ straight gap |
| Third-party interference **(B)** | is another component's node closer to the bridge midpoint than either endpoint? |

The taper feature is the single most useful topological discriminator and has no merge
analogue: a genuine axon terminal or dendritic tip thins out over its last few
microns, while a segmentation cut leaves the caliber flat right up to the endpoint.
It is also purely single-sided, which is exactly why unit A is viable at all. At pair
level, collinearity plus radius continuity plus a short gap is the geometric signature
of a break; a stub pointing sideways at a thick passing cable is a flank collision,
not a split.

#### Dimension 3 — Raw image

The raw fluorescence is what the UNet actually saw, and for splits it is the decisive
dimension: **a split is, by construction, a place where the network's evidence ran
out.** The single-sided question is *does the fluorescence continue past this
endpoint?* — probe the ray leading outward from the leaf. The pairwise question is
*does it continue all the way to that other endpoint?* — probe the bridge. The first
works with no partner at all and is the primary image feature for unit A.

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


def read_gap_patch(image, anisotropy, xyz_a, xyz_b, pad_um=8.0, min_side=16):
    """ONE patch covering both endpoints plus padding. Reading a separate patch per
    sample point would issue dozens of S3 round-trips per candidate."""
    a, b = np.asarray(xyz_a, float), np.asarray(xyz_b, float)
    center = xyz_to_voxel(0.5 * (a + b), anisotropy)                   # (z, y, x)
    half_um = 0.5 * np.abs(b - a) + pad_um                             # (x, y, z)
    side_xyz = np.ceil(2 * half_um / np.asarray(anisotropy)).astype(int)
    shape = tuple(int(max(s, min_side)) for s in side_xyz[::-1])       # (z, y, x)
    return image.read(center, shape), center, shape


def bridge_profile(patch, center_vox, shape, anisotropy, xyz_a, xyz_b, n=32):
    """Intensity sampled at n points along the straight A->B bridge."""
    origin = np.asarray(center_vox) - np.asarray(shape) // 2           # (z, y, x)
    a, b = np.asarray(xyz_a, float), np.asarray(xyz_b, float)
    out = []
    for t in np.linspace(0.0, 1.0, n):
        v = np.asarray(xyz_to_voxel(a + t * (b - a), anisotropy)) - origin
        v = np.clip(v, 0, np.asarray(patch.shape) - 1).astype(int)
        out.append(float(patch[tuple(v)]))
    return np.asarray(out)


def forward_ray_profile(g, image, anisotropy, leaf, reach_um=20.0, n=24):
    """SINGLE-SIDED (unit A). Intensity sampled along the ray leading OUTWARD from a
    leaf, i.e. where the process would have continued. No partner needed -- this is
    what lets an endpoint be judged truncated even when min_cable_length deleted the
    far side."""
    a = np.asarray(g.node_xyz[leaf], float)
    b = a + reach_um * leaf_direction(g, leaf)
    patch, center, shape = read_gap_patch(image, anisotropy, a, b)
    return bridge_profile(patch, center, shape, anisotropy, a, b, n=n)
```

| Feature | Interpretation |
|---|---|
| **Forward-ray mean ÷ local background** | does signal persist past the endpoint? The decisive single-sided image feature. |
| Forward-ray decay length | how far past the leaf the fluorescence stays above background — a genuine terminal decays within a micron or two |
| Intensity on the leaf's own cable | the reference brightness this process was reconstructed at |
| Local background level | the denominator for every ratio above; estimate from a shell around the leaf |
| Local SNR / noise | dim, noisy neighbourhoods are where the network gives up |
| Off-axis forward maximum | a brighter continuation slightly off the straight ray — the process turned |
| **Minimum intensity along the bridge (B)** | the bottleneck — the darkest point the network had to cross |
| Bridge minimum ÷ mean on the two flanking cables **(B)** | contrast-normalised bottleneck; robust to brightness variation across the brain |
| Mean / median intensity along the bridge **(B)** | does the gap track a continuous process at all? |
| Intensity match between the two stub tips **(B)** | two ends of one process have similar brightness; unrelated neurites often do not |
| Bridge profile shape **(B)** | a monotone dip = one dim stretch; a two-well profile = two separate processes |
| Foreign bright structure crossing the bridge **(B)** | a distractor that likely caused the network's confusion |

The most informative signal is that a true split leaves the fluorescence **continuous
but weak** — past the endpoint, and across the gap. A genuine terminal is followed by
true background; a false pair crosses true background between two unrelated
processes. Two processes that merely pass close by show a bright–dark–bright profile
with the dark band at background level; a split shows a dip that stays above it.

#### Beyond the catalogue — open exploration

**The catalogue is historical context, not a discovery assignment.** It provides
auditable baselines and examples of how to define a measurement, but V3 does not
require, prioritize, or predict the success of any listed feature family. Start from
the data and from falsifiable questions in both candidate views; retain a catalogue
feature only if it earns its place under the same confirmation rules as a newly
discovered one. Absence of a pre-listed feature is not a protocol failure.

The tables above share a bias worth naming: they assume a split is a **clean
break** — two facing leaf stubs, flat taper, collinear cables, a short dim gap.
Many splits are. But the cause list in §1 already names failure modes the
clean-break archetype does not cover, and one whole class sits outside the
machinery (*Known invisible class* above). Treat the catalogue as one archetype,
not the definition, and spend real effort outside it. Families known to be
under-covered:

- **Position and context** — the cause list names dim stretches and tile
  boundaries, yet no catalogue feature reads position. `frag.node_xyz` is
  reconstruction data (Phase 2-legal): raw endpoint coordinates, distance to
  candidate tile planes, imaging depth, local mean brightness. In Phase 1, test
  directly whether key sites cluster spatially — histogram their coordinates per
  axis and look for periodic planes; a systematic-imaging cause would be the
  single most actionable answer to "why".
- **Branch-point breaks** — a daughter dropped at a bifurcation leaves no leaf
  on the surviving trunk; the truncation is written mid-cable where a branch
  should sprout. Probe degree-2 nodes for one-sided radius bulges or direction
  kinks marking an absorbed branch root — this is the only route into the
  endpoint-invisible class counted above.
- **Whatever the data shows you (Phase 1 only).** Look first, featurize second:
  render the skeleton neighbourhood and the raw patch at a few dozen key sites
  of each `kind`, write down what visibly differs from genuine terminals, then
  encode those observations as features. A signature seen by eye and then
  formalized beats a catalogue feature computed blind.
- **Learned features.** The class-balanced fits recommended above are the
  floor, not the ceiling: pooling many cheap descriptors and letting a trained
  model expose which ones matter, or training directly on `forward_ray_profile`
  vectors or raw patches, are legitimate Phase 1 instruments. The Phase 2
  firewall constrains a model's **inputs**, not its origin.

Freedom in generation demands strictness in confirmation. With a large
candidate pool, rank candidates by effect size (or per-feature AUC), correct
for multiple comparisons across *everything you tried* (e.g.
Benjamini–Hochberg), and distrust any signal that cannot survive a held-out
split of endpoints — or better, a second brain's `_add.pkl`.

### Helper functions

Define these once before any analysis or detection code:

```python
def leaf_direction(g, leaf, reach_um=15.0):
    """OUTWARD unit vector at a degree-1 node: points away from its own cable,
    i.e. in the direction the fragment would continue if it had not stopped.

    Only meaningful at a leaf. On a degree>1 node it follows an arbitrary
    neighbour, so guard before calling it on a flank-landing candidate."""
    nbrs = list(g.neighbors(leaf))
    if not nbrs:
        return np.zeros(3)
    prev, cur, acc = leaf, nbrs[0], g.dist(leaf, nbrs[0])
    while acc < reach_um:
        nxt = [k for k in g.neighbors(cur) if k != prev]
        if len(nxt) != 1:
            break
        prev, cur = cur, nxt[0]
        acc += g.dist(prev, cur)
    v = g.node_xyz[leaf] - g.node_xyz[cur]        # inward point -> leaf = outward
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v


def _stub_radii(g, endpoint, reach_um=20.0):
    """Radii along the cable starting at `endpoint` and running inward, ordered
    tip-first. Empty list unless `endpoint` is a leaf -- taper is only defined
    for a free ending."""
    if g.degree[endpoint] != 1:
        return []
    nbrs = list(g.neighbors(endpoint))
    if not nbrs:
        return []
    radii = [float(g.node_radius[endpoint])]
    prev, cur, acc = endpoint, nbrs[0], g.dist(endpoint, nbrs[0])
    radii.append(float(g.node_radius[cur]))
    while acc < reach_um:
        nxt = [k for k in g.neighbors(cur) if k != prev]
        if len(nxt) != 1:
            break
        prev, cur = cur, nxt[0]
        acc += g.dist(prev, cur)
        radii.append(float(g.node_radius[cur]))
    return radii


def taper_ratio(g, endpoint, reach_um=20.0):
    """tip radius / shaft radius. ~1.0 = a flat cut end (split); < 1 = a tapering
    biological terminal that should NOT be joined."""
    radii = _stub_radii(g, endpoint, reach_um)
    if len(radii) < 4:
        return 1.0
    tip = float(np.mean(radii[:2]))
    shaft = float(np.mean(radii[len(radii) // 2:]))
    return tip / shaft if shaft > 0 else 1.0


def alignment_features(g, na, nb):
    """Collinearity of the two stubs and of each stub with the bridge vector.
    All three are ~ +1 for a clean split, and fall toward 0 or negative otherwise."""
    ua, ub = leaf_direction(g, na), leaf_direction(g, nb)
    w = np.asarray(g.node_xyz[nb], float) - np.asarray(g.node_xyz[na], float)
    nrm = np.linalg.norm(w)
    w = w / nrm if nrm > 0 else w
    return {
        "align_a": float(np.dot(ua, w)),      # a points at b
        "align_b": float(np.dot(ub, -w)),     # b points at a
        "antiparallel": float(-np.dot(ua, ub)),
    }


class _UnionFind:
    """Tracks which components a set of accepted joins has already connected --
    used to forbid joins that would close a cycle."""
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False                      # already connected -> would cycle
        self.parent[rx] = ry
        return True


def _match_one_to_one(cands, forbid_cycles=True):
    """Greedy highest-score-first matching. Each endpoint is consumed by at most
    one join -- the split analogue of the merge detector's spatial dedup, and a
    much stronger constraint: a broken cable has exactly one continuation."""
    uf, used, kept = _UnionFind(), set(), []
    for c in sorted(cands, key=lambda c: -c["score"]):
        na, nb = c["nodes"]
        if na in used or nb in used:
            continue
        if forbid_cycles and not uf.union(c["comps"][0], c["comps"][1]):
            continue
        used.add(na)
        used.add(nb)
        kept.append(c)
    return kept
```

### Phase 2 — Blind detection, A: truncated endpoints

The primary detector. Reads only the fragment reconstruction and scores every leaf on
whether it is a break rather than a real ending — no partner required.

```python
def detect_truncations(fragments_graph, min_cable_length,
                       reach_um=20.0, near_um=20.0, min_score=0.5):
    """
    GT-BLIND. Reads only the fragment reconstruction.

    Returns
    -------
    list[{"node": int, "segment_id": int, "xyz": (x, y, z) µm, "score": float}]
        One entry per leaf judged to be a truncation, most confident first.
    """
    g = fragments_graph
    xyz = np.asarray(g.node_xyz)
    comp = np.asarray(g.node_component_id)
    leaves = np.asarray([n for n in g.nodes if g.degree[n] == 1])
    leaf_tree = KDTree(xyz[leaves])

    out = []
    for n in leaves:
        n = int(n)
        # Flat caliber right up to the tip = a cut; a tapering tip = a real ending.
        taper = taper_ratio(g, n, reach_um)

        # Is there anything at all on the far side? Single-sided: we ask whether
        # SOMETHING is over there, not whether it is the correct partner -- so this
        # still fires when min_cable_length deleted the true continuation.
        near = [i for i in leaf_tree.query_ball_point(xyz[n], near_um)
                if comp[int(leaves[i])] != comp[n]]
        d_near = (min(float(g.dist(n, int(leaves[i]))) for i in near)
                  if near else float("inf"))
        proximity = 1.0 - min(d_near / near_um, 1.0)

        # Short components are disproportionately split debris.
        cable = float(g.cable_length(root=n))
        shortness = 1.0 - min(cable / (10.0 * min_cable_length), 1.0)

        score = 0.45 * min(taper, 1.0) + 0.35 * proximity + 0.20 * shortness
        if score < min_score:
            continue
        comp_id = int(comp[n])
        out.append({
            "node": n,
            "segment_id": int(g.component_id_to_swc_id[comp_id].split(".")[0]),
            "xyz": tuple(map(float, xyz[n])),
            "score": float(score),
        })
    return sorted(out, key=lambda d: -d["score"])


truncations = detect_truncations(frag, min_cable_length)
```

As with the merge doc's baseline, this is a geometric starting point — replace the
hand-set weights with a class-balanced fit over the Phase 1 labelled leaves, and add
the `forward_ray_profile` image term to measure what the raw fluorescence contributes
over geometry alone.

### Phase 2 — Blind detection, B: pairing the ends

The extension. Having found truncated ends, ask which of them belong together:

```python
def detect_splits(fragments_graph, anisotropy, min_cable_length,
                  radius_um=20.0, reach_um=15.0,
                  min_align=0.3, max_radius_mismatch=0.6, min_score=0.35,
                  per_anchor_k=5, partner_scope="any_node"):
    """
    GT-BLIND. Reads only the fragment reconstruction. Returns predicted joins.

    Returns
    -------
    dict with:
      "pairs": set[frozenset[int]]      -- segment id pairs to join
      "joins": list[{"segment_ids": (int, int), "nodes": (int, int),
                     "xyz": (x, y, z) µm, "gap_um": float, "score": float}]
    """
    g = fragments_graph
    comp_nodes = defaultdict(list)
    for n in g.nodes:
        comp_nodes[int(g.node_component_id[n])].append(n)

    def seg_of(comp_id):
        return int(g.component_id_to_swc_id[comp_id].split(".")[0])

    scored = []
    for c in candidate_joins(g, radius_um=radius_um,
                             per_anchor_k=per_anchor_k,
                             partner_scope=partner_scope):
        na, nb = c["nodes"]
        ca, cb = c["comps"]
        if seg_of(ca) == seg_of(cb):
            continue                                   # one segment, two pieces

        # Flank landing: nb is mid-cable, so leaf_direction(nb) is undefined
        # (see the helper's docstring) -- gate on the leaf side only and drop
        # the two-sided direction terms from the score.
        flank = g.degree[nb] > 1
        al = alignment_features(g, na, nb)
        aligns = [al["align_a"]] if flank else [al["align_a"], al["align_b"]]
        if min(aligns) < min_align:
            continue                                   # stub does not face the gap

        ra, rb = float(g.node_radius[na]), float(g.node_radius[nb])
        mismatch = abs(ra - rb) / max(0.5 * (ra + rb), 1e-6)
        if mismatch > max_radius_mismatch:
            continue                                   # caliber discontinuity

        # Flat cut ends are splits; tapering tips are real terminals. Taper is
        # only defined at a free ending, so a flank landing uses the leaf side.
        taper = (taper_ratio(g, na, reach_um) if flank else
                 0.5 * (taper_ratio(g, na, reach_um) + taper_ratio(g, nb, reach_um)))

        proximity = 1.0 - min(c["gap_um"] / radius_um, 1.0)
        align_term = al["align_a"] if flank else 0.5 * (al["align_a"] + al["align_b"])
        anti_term = 0.0 if flank else max(al["antiparallel"], 0.0)
        score = (0.40 * proximity
                 + 0.30 * align_term
                 + 0.15 * anti_term
                 + 0.10 * (1.0 - min(mismatch, 1.0))
                 + 0.05 * min(taper, 1.0))
        if score < min_score:
            continue

        mid = 0.5 * (np.asarray(g.node_xyz[na], float)
                     + np.asarray(g.node_xyz[nb], float))
        scored.append({"segment_ids": (seg_of(ca), seg_of(cb)),
                       "nodes": (na, nb), "comps": (ca, cb),
                       "xyz": tuple(map(float, mid)),
                       "gap_um": c["gap_um"], "score": float(score),
                       "flank": flank})

    joins = _match_one_to_one(scored)
    pairs = {frozenset(j["segment_ids"]) for j in joins}
    return {"pairs": pairs, "joins": joins}


detections = detect_splits(frag, anisotropy, min_cable_length)
```

Replace or extend the logic above with whatever Phase 1 identified as the strongest
discriminating features — in particular the raw-image bottleneck, which the geometric
baseline above deliberately omits so that the image dimension's contribution can be
measured as a delta. The hand-set weights are a starting point; a class-balanced
logistic fit over the Phase 1 labelled candidates is the obvious replacement, with
the hard negatives up-weighted.

This starter encodes one clean-break archetype and is not the required feature model.
Replace, retain, or reject its measurements according to free Phase 1 discovery: the
firewall constrains the detector's **runtime inputs**, not its feature content. Run
the same detector/evaluation contract with `partner_scope="tip"` and
`partner_scope="any_node"`; within the expanded view, report tip-target and non-tip-
target candidates separately. Do not interpret a difference between views as a model
gain unless candidate coverage, workload, and merge-creation risk are reported too.

### Scoring — A, endpoint level

The headline number, and the one that answers the scientific question. Graded only on
leaves the key could adjudicate, exactly like the merge doc's adjudicable set:

```python
def score_truncations(truncations, trunc_key):
    """Grade the endpoint detector against the truncation key.

    Ambiguous leaves (trunc_key[n] is None) and untraced leaves (absent from the
    key) are excluded from both numerator and denominator -- they carry no verdict.
    """
    truth = {n: v for n, v in trunc_key.items() if v is not None}
    ambiguous = {n for n, v in trunc_key.items() if v is None}
    pred = {d["node"] for d in truncations}

    pred_adj = pred & set(truth)
    tp = sum(1 for n in pred_adj if truth[n] == 1)
    fp = len(pred_adj) - tp
    n_pos = sum(1 for v in truth.values() if v == 1)

    return {
        "recall_endpoint": tp / n_pos if n_pos else float("nan"),
        "precision_endpoint_adjudicable": tp / len(pred_adj) if pred_adj else float("nan"),
        "n_truth_truncations": n_pos,
        "n_truth_terminals": sum(1 for v in truth.values() if v == 0),
        "n_ambiguous_excluded": sum(1 for v in trunc_key.values() if v is None),
        "n_pred": len(pred),
        "n_pred_adjudicable": len(pred_adj),
        "n_pred_ambiguous": len(pred & ambiguous),
        "n_pred_unadjudicable": len(pred) - len(pred_adj),
        "false_positives_adjudicable": fp,
    }
```

**Precision caveat (endpoint level).** A flagged leaf outside the key may be a false
positive *or* a true truncation in untraced space — GT cannot tell them apart. Compute
precision only over `pred ∩ adjudicable`, and report `n_pred_unadjudicable` separately
as "flagged, unadjudicable." This mirrors the merge doc exactly.

Flagging ambiguous leaves is not free either: skeletonization spurs are the
dominant deployment false-positive risk, and they never enter precision. Report
`n_pred_ambiguous` alongside — a detector whose flags concentrate on ambiguous
leaves has learned to find spurs, not truncations — and characterize the ambiguous
class as a third distribution in Phase 1 rather than discarding it.

Because the key is derived from GT labels and not from the fragments, endpoint recall
is **not** capped by `min_cable_length` the way pair recall is — a truncation whose
partner was deleted is still a positive here, and still findable. Reporting A and B
side by side is what isolates how much of the split problem is the reconstruction's
geometry and how much is data the cache no longer contains.

### Scoring — B, pair level

```python
def score_splits(detections, split_key, reachable_key,
                 seg_neuron_counts, site_tol_um=30.0):
    """Grade a GT-blind split detector against the derived key.

    Recall is reported against REACHABLE key pairs only -- pairs whose partner was
    deleted by min_cable_length cannot be found from the fragments graph, and
    charging them to the detector measures the cache, not the method.
    """
    key_pairs       = {pair for (_, pair) in split_key}
    reachable_pairs = {pair for (_, pair) in reachable_key}
    pred_pairs      = detections["pairs"]
    labeling_segs   = set(seg_neuron_counts)

    def substantial(seg):
        return {n for n, c in seg_neuron_counts.get(seg, {}).items() if c > 50}

    tp_pairs    = pred_pairs & reachable_pairs
    recall_pair = len(tp_pairs) / len(reachable_pairs) if reachable_pairs else float("nan")

    # Precision only over pairs GT can adjudicate: both segments land on a traced
    # neuron. Everything else is untraced space -- neither right nor wrong.
    pred_adj = {p for p in pred_pairs
                if all(s in labeling_segs for s in p)}
    correct  = pred_adj & key_pairs
    prec_pair = len(correct) / len(pred_adj) if pred_adj else float("nan")

    # The costly error class: a join fusing two substantially traced neurons. These
    # manufacture a merge, and canonical ERL scores a merged label's run length as 0.
    bad_joins = set()
    for p in pred_adj - key_pairs:
        a, b = tuple(p)
        na, nb = substantial(a), substantial(b)
        if na and nb and not (na & nb):
            bad_joins.add(p)

    # Site-level: a key site counts as hit only by a join of the SAME segment
    # pair within site_tol_um -- a nearby join between unrelated segments is
    # not a detection of this split.
    joins_by_pair = defaultdict(list)
    for j in detections["joins"]:
        joins_by_pair[frozenset(j["segment_ids"])].append(j["xyz"])
    key_sites = [(pair, xyz) for (_, pair), rec in reachable_key.items()
                 for xyz in rec["xyzs"]]
    hit = 0
    for pair, xyz in key_sites:
        xyzs = joins_by_pair.get(pair)
        if xyzs:
            d, _ = KDTree(xyzs).query(xyz)
            hit += int(d <= site_tol_um)
    recall_site = hit / len(key_sites) if key_sites else float("nan")

    return {
        "recall_pair_reachable": recall_pair,
        "precision_pair_adjudicable": prec_pair,
        "recall_site": recall_site,
        "n_key_pairs": len(key_pairs),
        "n_key_pairs_reachable": len(reachable_pairs),
        "n_key_pairs_unreachable": len(key_pairs) - len(reachable_pairs),
        "n_pred_pairs": len(pred_pairs),
        "n_pred_adjudicable": len(pred_adj),
        "n_merge_creating_joins": len(bad_joins),
    }
```

**Precision caveat.** A predicted join not in the key may be a false positive *or* a
true split between two untraced cells — GT cannot distinguish them. Compute precision
only over `pred_pairs ∩ adjudicable`, and report
`n_pred_pairs − n_pred_adjudicable` separately as "proposed, unadjudicable."

**Report `n_merge_creating_joins` alongside recall, always.** Recall on splits is
trivially maximised by joining everything nearby; the merge-creating count is what
makes the number mean something. A detector that lifts pair recall by 10 points while
adding merge-creating joins has moved error from one column of `results.csv` to a
worse one.

### Optional — end-to-end: what the join actually buys

> **This section goes beyond detection.** The merge doc stops at recall/precision, and
> so does the task defined here: identifying splits from fragment information is the
> deliverable. What follows measures a *corrector* — it belongs to the downstream
> proofreading tool of `labeled_dataset_cache.md` §3, not to Phase 2. Skip it unless
> you are actually applying joins.

Pair recall is a proxy. The metric a corrector is graded on is what happens to
`# Splits` and ERL after the joins are applied. Both are computable from the cache:

```python
def score_after_joins(detections, gt, node_label, seg_neuron_counts):
    """Recompute canonical-style # Splits per neuron after applying the predicted
    joins, and count the merges the joins created.

    # Splits per neuron is (distinct labels on it - 1). Joining segments a and b
    merges their labels into one group, so the post-join count uses GROUPS, not
    raw labels. A group spanning >= 2 substantially traced neurons is a new merge.
    """
    uf = _UnionFind()
    for pair in detections["pairs"]:
        a, b = tuple(pair)
        uf.union(a, b)

    neuron_labels, neuron_groups = defaultdict(set), defaultdict(set)
    for n in gt.nodes:
        lab = int(node_label[n])
        if lab != 0:
            neuron = gt.node_segment_id(n)
            neuron_labels[neuron].add(lab)
            neuron_groups[neuron].add(uf.find(lab))

    before = sum(max(len(s) - 1, 0) for s in neuron_labels.values())
    after  = sum(max(len(s) - 1, 0) for s in neuron_groups.values())

    # A group that now spans two substantially traced neurons is a manufactured merge.
    group_neurons = defaultdict(set)
    for seg, counts in seg_neuron_counts.items():
        for neuron, c in counts.items():
            if c > 50:
                group_neurons[uf.find(seg)].add(neuron)
    new_merges = sum(max(len(v) - 1, 0) for v in group_neurons.values())

    return {"splits_before": before, "splits_after": after,
            "splits_removed": before - after, "merges_created": new_merges}
```

`splits_removed` against `merges_created` is the honest summary of a split corrector.
Canonical `Split Rate` (µm per split) and `ERL` move in opposite directions under a
careless join, and `ERLMetric` zeroes a merged label's contribution entirely — so a
handful of manufactured merges can wipe out the gain from hundreds of correct joins.

### Cache comparability

The cache was loaded and validated at the start of this section. Compare only
like-for-like fragment-filter settings.

> **Compare only like-for-like `mcl`.** `min_cable_length` changes which fragments
> exist, so it changes which split pairs are reachable. An `mcl10` cache and an
> `mcl100` cache of the same brain have different reachable key sets, and recall
> numbers are not comparable across them.

---

## 3) Intent

The goal is to understand *why* the UNet drops a process at specific locations. The
Phase 1 characterization — comparing truncated fragment endings against genuine
neurite terminals across topology, geometry, and raw image — is the primary
deliverable. The Phase 2 blind detector validates that the discovered signals are
recoverable without GT; its performance tells you how much of the split signature is
intrinsic to the reconstruction and how much only becomes visible in the raw
fluorescence. Because a split is, by definition, a failure of image evidence, expect
the image dimension to carry more weight here than it does for merges — and measure
that explicitly, by scoring the geometry-only detector and the geometry-plus-image
detector separately.

**This is a detection task, not a repair task.** The deliverable is *where the splits
are*, in the same sense that the merge document's deliverable is where the merges are.
Pairing truncated ends (unit B) is part of naming a split completely — a split is a
relation, so identifying one means identifying both sides. Actually applying the joins
and re-scoring `# Splits` / ERL is a *corrector*, belongs to the proofreading tool of
`labeled_dataset_cache.md` §3, and is marked optional here for that reason.

The feature dimensions above are optional historical examples, not a floor or a
required checklist. The most useful outcome may
be a split signature that nobody wrote down here — a spatial clustering of breaks on
tile-boundary planes, a mid-cable bulge marking an absorbed branch root, a shape
class of forward-ray profiles, a depth-dependent failure rate. *Beyond the
catalogue* exists precisely because the clean-break archetype is known to be
incomplete. Use the answer key freely in Phase 1 — look at key sites of each `kind`
before featurizing, screen large candidate pools with multiple-comparison
correction, train models on descriptors or raw profiles — and let what the data
shows override what this document anticipated.

Two constraints are non-negotiable. The **Phase 2 firewall**: the final detector's
runtime inputs are fragment features only. Its same-brain score still inherits
optimism from Phase 1 (features and fitted weights were selected on this very GT),
so read it as "the signal is recoverable from the reconstruction alone", and take
the honest generalization number from a brain Phase 1 never saw. And the
**merge-creation accounting**: every pair-level recall number carries its
`n_merge_creating_joins` alongside it, because the two error types are dual and a
detector that trades one for the other has discovered nothing.

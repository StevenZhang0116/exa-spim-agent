"""
Typed proofreading edits + a coordinate-dependent label resolver (EditHandler).

Why this exists
---------------
The original action space was *union-only*: a candidate emitted ``(label_a,
label_b)`` pairs that became a ``LabelHandler(label_pairs=...)``, which can only
**grow** equivalence classes (merge fragment labels → repair splits). That makes
the system monotonic-coarsening: it can never *split* an already-merged segment,
so merge errors are structurally uncorrectable.

This module adds a **typed edit** vocabulary and an ``EditHandler`` whose label
lookup is keyed on ``(raw_label, node_xyz)`` instead of ``raw_label`` alone.
Because one raw segment id can map to *different* pseudo-labels in different
regions of space, the same merged segment can be virtually split — no dense
segmentation volume required, only relabeling at the GT node coordinates the
metrics already read.

Edit types
----------
- ``merge_labels`` — unify two fragment labels (repairs a split). Same effect as
  the old label-pair edit.
- ``split_label``  — partition one raw label into pseudo-labels by seed (repairs a
  merge). Coordinate-dependent: a node of label ``L`` is assigned to one of the
  seeds, yielding ``L#a`` / ``L#b`` / ``L#c`` … Two assignment modes:
    * GRAPH-AWARE (preferred): after ``build_graph_partitions(...)`` is given the
      fragment graphs, each fragment node is assigned to its nearest seed by
      *graph path distance* (multi-source Dijkstra on label ``L``'s subgraph), and
      every query point — GT node or fragment node — inherits the side of its
      nearest fragment node. This is robust to crossings / parallel neurites where
      two sides are spatially close but graph-far.
    * EUCLIDEAN (fallback): if no partition was built for the label, a node is
      assigned to whichever seed it is closest to in straight-line distance.
  Multiple ``split_label`` edits on the SAME label are COMPOSED into one
  multi-seed partition (no last-one-wins): their seeds accumulate with distinct
  suffixes, so a label fused from 3+ neurites can be cut into ``L#a/L#b/L#c/…``.
  Schema: either legacy ``seed_a_xyz`` / ``seed_b_xyz`` (→ suffixes a, b) or a
  ``seeds`` list of ``{"xyz": (x,y,z), "suffix": str?}`` (suffix auto-assigned if
  omitted).
- ``flag_review`` / ``reject_candidate`` — no relabeling effect; recorded so the
  policy can express "leave this alone / mark for a human" and the harness can
  report it. They never change a label.

Resolution order (important): **split first, then merge.** Splitting partitions a
raw label into pseudo-labels by location; merging then unifies labels (raw or
pseudo) into equivalence classes. Applying merge first would erase the raw label
the split keys on. ``flag_review``/``reject_candidate`` are inert.

Backward compatibility
-----------------------
``normalize_edit`` accepts the legacy ``(label_a, label_b)`` tuple and promotes
it to ``{"kind": "merge_labels", ...}``, so existing policies that return tuples
keep working unchanged.
"""

from __future__ import annotations

import numpy as np

# Separator between a raw label and its split pseudo-suffix, e.g. "789#a".
PSEUDO_SEP = "#"

# The recognized edit kinds.
MERGE = "merge_labels"
SPLIT = "split_label"
FLAG = "flag_review"
REJECT = "reject_candidate"
EDIT_KINDS = frozenset({MERGE, SPLIT, FLAG, REJECT})


def normalize_edit(edit) -> dict:
    """Promote a legacy ``(label_a, label_b)`` tuple to a typed merge edit.

    Typed dict edits pass through unchanged (after a light validity check).
    Raises ValueError on anything unrecognized so a malformed policy output
    fails loudly rather than silently doing nothing.
    """
    if isinstance(edit, dict):
        kind = edit.get("kind")
        if kind not in EDIT_KINDS:
            raise ValueError(f"unknown edit kind {kind!r}; expected one of {sorted(EDIT_KINDS)}")
        return edit
    if isinstance(edit, (tuple, list)) and len(edit) == 2:
        a, b = edit
        return {"kind": MERGE, "label_a": str(a), "label_b": str(b)}
    raise ValueError(
        f"cannot interpret edit {edit!r}; expected a (label_a, label_b) pair or a "
        f"typed edit dict with a 'kind' field"
    )


def normalize_edits(edits) -> list[dict]:
    """Normalize a mixed list of legacy tuples and/or typed dicts."""
    return [normalize_edit(e) for e in (edits or [])]


# Suffix alphabet for split pseudo-labels: a, b, …, z, then aa, ab, … (rarely > b).
def _next_suffix(used: set) -> str:
    """Smallest a/b/c/… suffix not already in ``used`` (extends past z if needed)."""
    i = 0
    while True:
        # base-26 -> a, b, ..., z, aa, ab, ...
        n, s = i, ""
        while True:
            s = chr(ord("a") + n % 26) + s
            n = n // 26 - 1
            if n < 0:
                break
        if s not in used:
            return s
        i += 1


def _iter_split_seeds(edit: dict):
    """Yield ``(suffix_or_None, xyz)`` for a split_label edit, both schemas.

    Accepts the legacy two-seed form (``seed_a_xyz`` / ``seed_b_xyz`` → suffixes
    a, b) and the multi-seed ``seeds`` list (``[{"xyz": (...), "suffix": "a"?}]``;
    suffix optional). A ``suffix`` of None tells the caller to auto-assign.
    """
    seeds = edit.get("seeds")
    if seeds:
        for spec in seeds:
            yield spec.get("suffix"), spec["xyz"]
        return
    if "seed_a_xyz" in edit:
        yield "a", edit["seed_a_xyz"]
    if "seed_b_xyz" in edit:
        yield "b", edit["seed_b_xyz"]


def _partition_by_graph_distance(graphs, seeds):
    """Assign each node of ``graphs`` to its nearest seed by GRAPH path distance.

    Builds one networkx graph over the union of ``graphs`` (a label may be split
    across disconnected fragment components), with each edge weighted by physical
    length. Each seed is snapped to its nearest node (by straight-line distance);
    a multi-source Dijkstra then labels every reachable node with the suffix of the
    closest seed. Nodes in a component that contains NO seed fall back to their
    nearest seed by straight-line distance, so every node gets a side.

    Parameters
    ----------
    graphs : list
        FragmentGraph objects sharing one raw label. Need ``.nodes``,
        ``.neighbors(n)``, ``.node_xyz(n)``, ``.physical_dist(i, j)``.
    seeds : list[(suffix, xyz)]
        The label's seeds (suffix already deduped by the caller).

    Returns
    -------
    (assignment, coords, suffixes) or (None, None, None)
        ``assignment`` is an opaque per-node map (unused by the caller beyond
        building the KD-tree); ``coords`` is an (M, 3) array of every node's xyz and
        ``suffixes`` the parallel list of assigned suffixes. ``None`` triple if there
        are no nodes.
    """
    import heapq

    # Collect nodes across all components of this label into one index space.
    # Each entry: (graph, local_node_id). Coordinates drive both the seed snap and
    # the KD-tree the handler queries later.
    entries = []  # [(graph, node)]
    for g in graphs:
        for n in g.nodes:
            entries.append((g, n))
    if not entries:
        return None, None, None

    coords = np.array([g.node_xyz(n) for g, n in entries], dtype=float)
    point_index = {(id(g), n): k for k, (g, n) in enumerate(entries)}

    # Snap each seed to its nearest node (straight-line) as a Dijkstra source.
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    sources = {}  # start point-index -> suffix
    for suffix, sxyz in seeds:
        _, k = tree.query(np.asarray(sxyz, dtype=float))
        sources[int(k)] = suffix

    # Multi-source Dijkstra over the union graph (edges weighted by physical len).
    INF = float("inf")
    dist = [INF] * len(entries)
    assigned = [None] * len(entries)
    heap = []
    for k, suffix in sources.items():
        dist[k] = 0.0
        assigned[k] = suffix
        heapq.heappush(heap, (0.0, k))
    # Adjacency by point-index, lazily from each graph's neighbors.
    while heap:
        d, k = heapq.heappop(heap)
        if d > dist[k]:
            continue
        g, n = entries[k]
        for m in g.neighbors(n):
            kk = point_index[(id(g), m)]
            w = g.physical_dist(n, m)
            nd = d + w
            if nd < dist[kk]:
                dist[kk] = nd
                assigned[kk] = assigned[k]
                heapq.heappush(heap, (nd, kk))

    # Any node unreached by graph distance (seedless component) -> nearest seed by
    # straight line, so the partition is total.
    if any(a is None for a in assigned):
        seed_xyz = np.array([s for _, s in seeds], dtype=float)
        seed_suf = [suf for suf, _ in seeds]
        for k, a in enumerate(assigned):
            if a is None:
                j = int(np.argmin(np.linalg.norm(seed_xyz - coords[k], axis=1)))
                assigned[k] = seed_suf[j]

    return assigned, coords, assigned


class EditHandler:
    """Resolve a node's edited label from ``(raw_label, xyz)`` given typed edits.

    Wraps the merge behavior (equivalence classes, like the old LabelHandler) and
    adds coordinate-dependent split behavior. Use ``get(raw_label, xyz)`` per node.

    Parameters
    ----------
    edits : list
        Typed edits (or legacy tuples; they are normalized). Only ``merge_labels``
        and ``split_label`` have a relabeling effect.
    all_labels : iterable of str, optional
        The fragment-label universe (so merge equivalence classes are stable and
        match the metric package's LabelHandler). Labels not in any merge map to
        themselves.
    max_class_size : int, optional
        Hard guardrail: refuse any merge that would grow an equivalence class
        beyond this many distinct labels. The aggressive "merge everything nearby"
        failure mode chains thousands of pairs into a single brain-spanning class
        (observed: 68 fragments fused), which manufactures merge errors and can
        flukily score well. Capping class size makes that catastrophic merge
        impossible regardless of how bad a proposed policy is. ``None`` = no cap.
        Rejected unions are dropped (the labels stay in separate classes); the
        count of dropped unions is exposed as ``dropped_unions``.
    """

    def __init__(self, edits, all_labels=None, max_class_size=None):
        self.edits = normalize_edits(edits)
        self._all_labels = set(map(str, all_labels)) if all_labels is not None else None
        self.max_class_size = max_class_size
        self.dropped_unions = 0  # # of merges refused by the size guard

        # --- split specs: raw_label -> [(suffix, xyz), ...] ----------------------
        # A node of this raw label is assigned to ONE of the label's seeds, yielding
        # pseudo-label "<raw>#<suffix>". Multiple split_label edits on one label are
        # COMPOSED into a single multi-seed partition (NOT last-one-wins), so a
        # segment fusing 3+ neurites can be cut into L#a / L#b / L#c / … Each edit
        # contributes its seed(s); suffixes are taken from the edit when given, else
        # auto-assigned a, b, c, … in arrival order (deduped per label).
        self._split: dict[str, list] = {}  # raw_label -> [(suffix:str, xyz:np.ndarray)]
        for e in self.edits:
            if e["kind"] != SPLIT:
                continue
            lbl = str(e["label"])
            bucket = self._split.setdefault(lbl, [])
            for suffix, xyz in _iter_split_seeds(e):
                # Keep suffixes unique within a label; if the edit reuses one,
                # fall through to an auto suffix so no seed is silently dropped.
                used = {s for s, _ in bucket}
                if suffix is None or suffix in used:
                    suffix = _next_suffix(used)
                bucket.append((suffix, np.asarray(xyz, dtype=float)))

        # Graph-aware partitions, populated lazily by build_graph_partitions(). Maps
        # raw_label -> {fragment_node_id: suffix} plus the KD-tree needed to map an
        # arbitrary query coordinate to its nearest fragment node's side. Absent =>
        # apply_split falls back to Euclidean nearest-seed (the previous behavior).
        self._graph_partition: dict[str, dict] = {}

        # --- merge equivalence classes over (possibly pseudo-) labels ------------
        # Built lazily on first get(), because split produces pseudo-labels that a
        # merge edit may also reference. We use union-find over the label strings.
        self._merge_pairs = [
            (str(e["label_a"]), str(e["label_b"]))
            for e in self.edits
            if e["kind"] == MERGE
        ]
        self._parent: dict[str, str] = {}
        self._size: dict[str, int] = {}   # root -> # labels in its class
        self._built = False

    # --- union-find for merges ---
    def _find(self, x: str) -> str:
        p = self._parent
        if x not in p:
            p[x] = x
            return x
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:  # path compression
            p[x], x = root, p[x]
        return root

    def _union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return
        # Size guard: refuse the merge if the combined class would exceed the cap.
        if self.max_class_size is not None:
            if self._size.get(ra, 1) + self._size.get(rb, 1) > self.max_class_size:
                self.dropped_unions += 1
                return
        self._parent[rb] = ra
        self._size[ra] = self._size.get(ra, 1) + self._size.get(rb, 1)

    def _ensure_merge_built(self) -> None:
        if self._built:
            return
        self._built = True
        # Seed each endpoint label as a singleton (size 1) before unioning.
        for a, b in self._merge_pairs:
            self._size.setdefault(a, 1)
            self._size.setdefault(b, 1)
        for a, b in self._merge_pairs:
            self._union(a, b)

    # --- core lookup ---
    @property
    def split_labels(self) -> set:
        """Raw labels that carry a split spec.

        The scorer reads this to decide which fragment graphs to virtually
        partition for the ``# Merges`` count (see
        ``incremental_scoring._edited_fragment_graphs``). Empty when no
        ``split_label`` edit was proposed, which is what keeps merge-only /
        baseline scoring byte-identical to ``evaluate()``.
        """
        return set(self._split.keys())

    def build_graph_partitions(self, fragment_graphs) -> None:
        """Precompute a GRAPH-AWARE side for every node of each split label.

        For each raw label ``L`` carrying a split spec, find the fragment graph for
        ``L`` (keyed by label in the metrics package), snap each seed to its nearest
        fragment node, and run a multi-source shortest-path (Dijkstra on physical
        edge length) so every fragment node is assigned to the seed it is closest to
        BY GRAPH DISTANCE — not straight-line. This is what makes the split robust at
        crossings / fasciculations, where two neurites are spatially close but far
        along the skeleton. The resulting ``{node: suffix}`` map plus a KD-tree over
        the fragment nodes lets ``apply_split`` resolve ANY query coordinate (a GT
        node or a fragment node) to a consistent side: it inherits the side of its
        nearest fragment node. Idempotent; safe to call once per candidate.

        ``fragment_graphs`` is the metrics-package ``{label: FragmentGraph}`` mapping
        (or any object exposing ``.label``/``.nodes``/``.neighbors``/``.node_xyz``/
        ``.physical_dist``). Labels with no matching fragment graph keep the
        Euclidean fallback (apply_split handles the absence).
        """
        # Index fragment graphs by their (raw) label. A label may map to multiple
        # disconnected fragment graphs; we union their nodes into one partition
        # problem so seeds reach every component they can.
        by_label: dict[str, list] = {}
        for g in fragment_graphs.values():
            by_label.setdefault(str(g.label), []).append(g)

        for raw_label, seeds in self._split.items():
            graphs = by_label.get(raw_label)
            if not graphs:
                continue  # no fragment graph for this label -> Euclidean fallback
            assignment, coords, suffixes = _partition_by_graph_distance(graphs, seeds)
            if assignment is None:
                continue  # degenerate (e.g. no nodes) -> Euclidean fallback
            from scipy.spatial import cKDTree
            self._graph_partition[raw_label] = {
                "tree": cKDTree(coords),
                "suffix_of_point": suffixes,  # parallel to coords rows
            }

    def apply_split(self, raw_label: str, xyz) -> str:
        """Coordinate-dependent split: raw -> pseudo-label, or raw if not split.

        Uses the graph-aware partition when ``build_graph_partitions`` populated one
        for this label (the query point inherits its nearest fragment node's side);
        otherwise falls back to multi-seed Euclidean nearest-seed. With no split spec
        for the label, returns the raw label unchanged.
        """
        raw_label = str(raw_label)
        seeds = self._split.get(raw_label)
        if seeds is None or xyz is None:
            return raw_label
        xyz = np.asarray(xyz, dtype=float)

        part = self._graph_partition.get(raw_label)
        if part is not None:
            _, idx = part["tree"].query(xyz)
            return f"{raw_label}{PSEUDO_SEP}{part['suffix_of_point'][int(idx)]}"

        # Euclidean fallback: nearest of the label's seeds (handles 2+ seeds).
        seed_xyz = np.array([s for _, s in seeds], dtype=float)
        d = np.linalg.norm(seed_xyz - xyz, axis=1)
        return f"{raw_label}{PSEUDO_SEP}{seeds[int(np.argmin(d))][0]}"

    def apply_merge(self, label: str) -> str:
        """Map a (raw or pseudo) label to its merge-equivalence-class id."""
        if not self._merge_pairs:
            return label
        self._ensure_merge_built()
        return self._find(label) if label in self._parent else label

    def get(self, raw_label, xyz=None) -> str:
        """Edited label for a node: split-by-location first, then merge-union."""
        if raw_label == "0":
            return "0"  # background is never relabeled
        return self.apply_merge(self.apply_split(raw_label, xyz))

    @property
    def inverse_mapping(self) -> dict:
        """Class id -> set of (raw/pseudo) labels fused into it.

        Mirrors ``LabelHandler.inverse_mapping`` so merge-site attribution works
        uniformly for either handler. Only reflects the merge edits (the union
        classes); split pseudo-labels that were not subsequently merged map to
        themselves and need no entry. A pseudo-label like ``"7#a"`` carries its
        raw origin in the string before ``PSEUDO_SEP``.
        """
        self._ensure_merge_built()
        inv: dict[str, set] = {}
        for label in self._parent:
            inv.setdefault(self._find(label), set()).add(label)
        return inv

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
- ``split_label``  — partition one raw label into pseudo-labels by nearest seed
  (repairs a merge). Coordinate-dependent: a node of label ``L`` is assigned to
  whichever seed (``seed_a_xyz`` / ``seed_b_xyz``) it is closest to, yielding
  ``L#a`` / ``L#b``.
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
    """

    def __init__(self, edits, all_labels=None):
        self.edits = normalize_edits(edits)
        self._all_labels = set(map(str, all_labels)) if all_labels is not None else None

        # --- split specs: raw_label -> {"seeds": (xyz_a, xyz_b)} -----------------
        # A node of this raw label is assigned to its nearest seed -> pseudo-label
        # "<raw>#a" or "<raw>#b". Multiple split_label edits on one label are not
        # currently composed (last one wins); the candidate generator emits at
        # most one per label.
        self._split: dict[str, np.ndarray] = {}  # raw_label -> (2,3) seed array
        for e in self.edits:
            if e["kind"] == SPLIT:
                lbl = str(e["label"])
                seeds = np.array([e["seed_a_xyz"], e["seed_b_xyz"]], dtype=float)
                self._split[lbl] = seeds

        # --- merge equivalence classes over (possibly pseudo-) labels ------------
        # Built lazily on first get(), because split produces pseudo-labels that a
        # merge edit may also reference. We use union-find over the label strings.
        self._merge_pairs = [
            (str(e["label_a"]), str(e["label_b"]))
            for e in self.edits
            if e["kind"] == MERGE
        ]
        self._parent: dict[str, str] = {}

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
        if ra != rb:
            self._parent[rb] = ra

    def _ensure_merge_built(self) -> None:
        if self._parent:
            return
        for a, b in self._merge_pairs:
            self._union(a, b)

    # --- core lookup ---
    def apply_split(self, raw_label: str, xyz) -> str:
        """Coordinate-dependent split: raw -> pseudo-label, or raw if not split."""
        raw_label = str(raw_label)
        seeds = self._split.get(raw_label)
        if seeds is None or xyz is None:
            return raw_label
        d = np.linalg.norm(seeds - np.asarray(xyz, dtype=float), axis=1)
        return f"{raw_label}{PSEUDO_SEP}{'a' if d[0] <= d[1] else 'b'}"

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

"""Candidate enumeration rules of the merge and split benchmarks, defined once.

Every place that enumerates merge or split candidates uses this module: the benchmark tables
and companions (``proofreader_evolve``), the candidate-pool sweeps and the frozen detectors.
The parameters of the rule in force are declared in ``proofreader_evolve/candidate_pools.json``.
A detector build copies this file unchanged into the detector directory and records its
SHA-256, so a detector carries this exact code instead of its own version of the rule.

Self-contained on purpose (numpy, scipy, networkx and the standard library only): the copy
inside a detector directory must run without the repository.

Fragment graph: a networkx graph with node ids 0..N-1 and the attributes ``node_xyz``
(N x 3, um), ``node_component_id`` (N) and ``component_id_to_swc_id``. A node's segment is
the integer before the first "." of its component's swc id; only nodes of a positive
segment take part ("valid" nodes). Enumeration reads fragment geometry only; GT enters
afterwards, to attach labels and audit fields.

Split rule ``{config_id, mode, radius_um, per_anchor_k, global_cap}``: from every anchor
node (``tip_to_tip`` / ``tip_to_any_node``: degree-1 tips; ``any_node_to_any_node``: every
valid node), the nearest node of each other segment within ``radius_um`` (Euclidean); the
``per_anchor_k`` nearest partner segments are kept (null keeps all). One row per unordered
segment pair; its occurrences list every anchor/partner realisation, shortest gap first.
Positive: the pair is a GT split key.

Merge rule ``{config_id, mode, nms_um, tip_gap_um, tip_partner_k, claim_radius_um,
positive_label_radius_um}``:
- junction candidates: every valid node of degree >= 3, greedy same-segment non-maximum
  suppression within ``nms_um`` (Euclidean) in order (-degree, node id);
- tip-to-shaft candidates (mode ``junction+tip_to_shaft``): every degree-1 tip pairs with
  the nearest node of each of its ``tip_partner_k`` nearest other segments (the split
  ``tip_to_any_node`` pairing); a partner node that is not a tip and lies within
  ``tip_gap_um`` of the tip is a shaft candidate, with its smallest such gap. Shaft
  candidates within ``nms_um`` of a kept junction of their own segment are dropped, the
  rest pass the same-segment suppression in order (gap, node id).
Positive: a GT merge site, snapped to the nearest node of its own segment within 50 um,
lies within ``positive_label_radius_um`` of fragment cable; cable distances are searched to
``claim_radius_um``, which is also the radius site coverage is reported at.

``build_universe`` returns binary labels (the benchmark tables store them so and attach the
three-valued rule through their availability sidecars). ``label_availability`` is that rule,
written once here for the detectors: GT can judge a split row when both segments carry a GT
node label, and a merge row when its segment does and the candidate lies within
``NEAR_GT_UM`` of a GT node; ``three_valued_labels`` applies it (0 -> NaN where GT cannot
judge). ``fold_groups`` / ``fold_partitions`` are the label-free cross-fold rule the
benchmark's internal folds use (split: 500 um spatial blocks of the anchor midpoint;
merge: the segment), with the fit side purged of every segment a held row touches, so each
row is held out exactly once.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import heapq
import json
import math

import numpy as np

RULES_VERSION = 2
KINDS = ("merge", "split")
# Judgeability (the three-valued label's 0-versus-NaN rule) and the fold grouping, the
# same constants as proofreader_evolve/harness (label_availability.NEAR_GT_UM and the
# internal folds' SPATIAL_BLOCK_UM).
NEAR_GT_UM = 150.0
FOLD_BLOCK_UM = 500.0
AVAILABILITY_DEFINITIONS = {"split": "segment_on_gt", "merge": "near_gt_150"}
SPLIT_MODES = ("tip_to_tip", "tip_to_any_node", "any_node_to_any_node")
MERGE_MODES = ("junction", "junction+tip_to_shaft")
JUNCTION, TIP_TO_SHAFT = "junction", "tip_to_shaft"
FIELDS = {
    "split": ("config_id", "mode", "radius_um", "per_anchor_k", "global_cap"),
    "merge": ("config_id", "mode", "nms_um", "tip_gap_um", "tip_partner_k",
              "claim_radius_um", "positive_label_radius_um"),
}
# Row fields read off fragment geometry only (identity and location); the remaining row
# fields are GT-derived audit values that must never reach a feature or a scorer.
GEOMETRY_FIELDS = {
    "merge": ("candidate_id", "node_id", "segment_id", "degree", "x_um", "y_um", "z_um",
              "candidate_family", "tip_gap_um"),
    "split": ("candidate_id", "segment_id_a", "segment_id_b", "node_id_a", "node_id_b",
              "gap_um", "anchor_side", "partner_rank", "node_role_a", "node_role_b"),
}
OCCURRENCE_FIELDS = ("node_id_a", "node_id_b", "gap_um", "anchor_side", "partner_rank",
                     "node_role_a", "node_role_b")
SITE_SNAP_MAX_UM = 50.0
SITE_SOURCES = ("geometric_walk", "two_gt_junction")
# The tip-to-shaft search radius exceeds the gap by this much so that a partner whose
# Euclidean gap equals the limit is never lost to KD-tree rounding; the gap test is exact.
_QUERY_SLACK_UM = 1e-6
_BATCH = 2000


# ----------------------------------------------------------------------------- rules
def _quota_text(quota) -> str:
    return "all" if quota is None else str(int(quota))


def split_config_id(mode, radius_um, per_anchor_k) -> str:
    return f"{mode}|r={float(radius_um):g}|k={_quota_text(per_anchor_k)}"


def merge_config_id(mode, nms_um, claim_radius_um, tip_gap_um=None, tip_partner_k=None) -> str:
    if mode == "junction":
        return f"junction|nms={float(nms_um):g}|r={float(claim_radius_um):g}"
    return (f"{mode}|nms={float(nms_um):g}|gap={float(tip_gap_um):g}"
            f"|k={_quota_text(tip_partner_k)}|r={float(claim_radius_um):g}")


def _positive_float(value, name, allow_zero=False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'}")
    return number


def _quota(value, name):
    if value is None:
        return None
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be null or a positive integer")
    return int(value)


def validate_rule(kind, rule) -> dict:
    """Normalised copy of one rule; raises on anything the enumeration cannot apply."""
    if kind not in KINDS:
        raise ValueError(f"Unknown candidate kind {kind!r}")
    if not isinstance(rule, dict) or set(rule) != set(FIELDS[kind]):
        raise ValueError(f"A {kind} rule needs exactly the fields {FIELDS[kind]}")
    mode = str(rule["mode"])
    if kind == "split":
        if mode not in SPLIT_MODES:
            raise ValueError(f"Unknown split pairing rule {mode!r}; choose from {SPLIT_MODES}")
        if rule["global_cap"] is not None:
            raise ValueError("global_cap must be null (enumeration has no global cap)")
        radius = _positive_float(rule["radius_um"], "radius_um")
        quota = _quota(rule["per_anchor_k"], "per_anchor_k")
        normal = {"config_id": split_config_id(mode, radius, quota), "mode": mode,
                  "radius_um": radius, "per_anchor_k": quota, "global_cap": None}
    else:
        if mode not in MERGE_MODES:
            raise ValueError(f"Unknown merge mode {mode!r}; choose from {MERGE_MODES}")
        nms = _positive_float(rule["nms_um"], "nms_um", allow_zero=True)
        claim = _positive_float(rule["claim_radius_um"], "claim_radius_um")
        label = _positive_float(rule["positive_label_radius_um"], "positive_label_radius_um")
        if label > claim:
            raise ValueError("positive_label_radius_um must not exceed claim_radius_um")
        if mode == "junction":
            if rule["tip_gap_um"] is not None or rule["tip_partner_k"] is not None:
                raise ValueError("A junction-only merge rule takes null tip_gap_um and tip_partner_k")
            gap, quota = None, None
        else:
            gap = _positive_float(rule["tip_gap_um"], "tip_gap_um")
            quota = _quota(rule["tip_partner_k"], "tip_partner_k")
        normal = {"config_id": merge_config_id(mode, nms, claim, gap, quota), "mode": mode,
                  "nms_um": nms, "tip_gap_um": gap, "tip_partner_k": quota,
                  "claim_radius_um": claim, "positive_label_radius_um": label}
    if str(rule["config_id"]) != normal["config_id"]:
        raise ValueError(f"config_id {rule['config_id']!r} does not match its fields ({normal['config_id']})")
    return normal


def rule_digest(kind, rule) -> str:
    """SHA-256 of the rule fields (what enumeration applies), independent of evidence notes."""
    rule = validate_rule(kind, rule)
    canonical = json.dumps({field: rule[field] for field in FIELDS[kind]}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def describe_rule(kind, rule) -> str:
    """Plain-text statement of a rule and its labels (companion manifests, specifications)."""
    rule = validate_rule(kind, rule)
    if kind == "split":
        radius = f"{rule['radius_um']:g}"
        quota = rule["per_anchor_k"]
        anchors = {"tip_to_tip": "from every degree-1 fragment tip, the nearest tip of each other segment",
                   "tip_to_any_node": "from every degree-1 fragment tip, the nearest node of each other segment",
                   "any_node_to_any_node": "from every fragment node, the nearest node of each other segment"
                   }[rule["mode"]]
        kept = ("every partner segment is kept" if quota is None
                else f"the {quota} nearest partner segment{'s are' if quota != 1 else ' is'} kept")
        return (f"Candidates: {anchors} within {radius} um; {kept} "
                f"({rule['mode']}, r={radius}, k={_quota_text(quota)}). One row per unordered "
                "segment pair; occurrences list every anchor/partner realisation, the first being the shortest "
                "gap. Positive: the pair is a GT split key (distinct non-zero labels across a GT split edge, or "
                "the two flanks of a zero-label GT stretch on one neuron). Label 0 only where GT can judge: both "
                "segments carry a GT node label; otherwise NaN.")
    nms = f"{rule['nms_um']:g}"
    text = ("Candidates: every fragment node of degree >= 3 on a valid segment; greedy non-maximum "
            f"suppression in order (-degree, node id) removes same-segment junctions within {nms} um "
            "(Euclidean).")
    if rule["mode"] == "junction+tip_to_shaft":
        quota = rule["tip_partner_k"]
        partners = ("each other segment" if quota is None
                    else f"each of its {quota} nearest other segment{'s' if quota != 1 else ''}")
        text += (f" Plus tip-to-shaft candidates: every degree-1 tip pairs with the nearest node of {partners}; "
                 f"a partner node that is not a tip and lies within {rule['tip_gap_um']:g} um of the tip is a "
                 "candidate (its smallest such gap is recorded as tip_gap_um). Shaft candidates within "
                 f"{nms} um of a kept junction of the same segment are dropped, the rest pass the same-segment "
                 f"suppression within {nms} um in order (gap, node id). candidate_family names the rule that "
                 "produced a row (junction or tip_to_shaft).")
    return text + (f" Positive: geodesic distance (fragment cable) <= {rule['positive_label_radius_um']:g} um to "
                   f"a GT merge site, after snapping the site to the nearest node of its own segment within "
                   f"{SITE_SNAP_MAX_UM:g} um. Label 0 only where GT can judge: the segment carries a GT node "
                   f"label and the candidate lies within {rule['claim_radius_um']:g} um of a GT node; otherwise NaN.")


# ----------------------------------------------------------------------------- graph
def segment_of_component(fragments_graph) -> dict:
    """component id -> segment id (the part of the swc id before the first '.')."""
    return {int(component): int(str(swc).split(".")[0])
            for component, swc in fragments_graph.component_id_to_swc_id.items()}


def node_arrays(fragments_graph):
    """``(xyz, node_segments, degrees)`` over contiguous node ids; -1 marks invalid nodes."""
    to_segment = segment_of_component(fragments_graph)
    n_nodes = int(fragments_graph.number_of_nodes())
    xyz = np.asarray(fragments_graph.node_xyz, dtype=float)
    components = np.asarray(fragments_graph.node_component_id, dtype=np.int64)
    if xyz.shape[0] != n_nodes or components.shape[0] != n_nodes:
        raise ValueError("candidate enumeration requires node_xyz and node_component_id "
                         "to align with contiguous graph node ids")
    if n_nodes and (not fragments_graph.has_node(0) or not fragments_graph.has_node(n_nodes - 1)):
        raise ValueError("candidate enumeration requires node ids 0..N-1")
    node_segments = np.fromiter((int(to_segment.get(int(c), -1)) for c in components),
                                dtype=np.int64, count=n_nodes)
    degrees = np.fromiter((int(fragments_graph.degree(node)) for node in range(n_nodes)),
                          dtype=np.int64, count=n_nodes)
    return xyz, node_segments, degrees


# ----------------------------------------------------------------------------- split
def pair_occurrences(xyz, node_segments, degrees, mode, radius_um, per_anchor_k):
    """``{(segment_a, segment_b): [occurrence, ...]}`` of the anchor/partner pairing, or None
    when the graph has no anchor or no partner node."""
    tips = np.flatnonzero((degrees == 1) & (node_segments > 0)).astype(np.int64)
    valid = np.flatnonzero(node_segments > 0).astype(np.int64)
    if mode == "tip_to_tip":
        anchors, partners = tips, tips
    elif mode == "tip_to_any_node":
        anchors, partners = tips, valid
    elif mode == "any_node_to_any_node":
        anchors, partners = valid, valid
    else:
        raise ValueError(f"unsupported split pairing rule {mode!r}")
    if not len(anchors) or not len(partners):
        return None
    from scipy.spatial import cKDTree

    tree = cKDTree(xyz[partners], copy_data=False)
    grouped = defaultdict(list)
    for start in range(0, len(anchors), _BATCH):
        batch = anchors[start:start + _BATCH]
        neighbour_lists = tree.query_ball_point(xyz[batch], r=radius_um, workers=1)
        for anchor, positions in zip(batch, neighbour_lists):
            anchor = int(anchor)
            anchor_segment = int(node_segments[anchor])
            best_by_segment = {}
            for position in positions:
                partner = int(partners[int(position)])
                partner_segment = int(node_segments[partner])
                if partner == anchor or partner_segment == anchor_segment:
                    continue
                distance = float(np.linalg.norm(xyz[partner] - xyz[anchor]))
                previous = best_by_segment.get(partner_segment)
                candidate = (distance, partner)
                if previous is None or candidate < previous:
                    best_by_segment[partner_segment] = candidate
            ranked = sorted((distance, segment, partner)
                            for segment, (distance, partner) in best_by_segment.items())
            limit = len(ranked) if per_anchor_k is None else int(per_anchor_k)
            for zero_rank, (distance, partner_segment, partner) in enumerate(ranked[:limit]):
                segment_a, segment_b = sorted((anchor_segment, partner_segment))
                if anchor_segment == segment_a:
                    node_a, node_b, anchor_side = anchor, partner, "a"
                else:
                    node_a, node_b, anchor_side = partner, anchor, "b"
                grouped[(segment_a, segment_b)].append({
                    "node_id_a": int(node_a), "node_id_b": int(node_b), "gap_um": float(distance),
                    "anchor_side": anchor_side, "partner_rank": int(zero_rank + 1),
                    "node_role_a": "tip" if int(degrees[node_a]) == 1 else "non_tip",
                    "node_role_b": "tip" if int(degrees[node_b]) == 1 else "non_tip",
                })
    return grouped


def gt_neuron_membership(gt_graph) -> dict:
    """GT node -> connected-component index (a stable neuron key)."""
    import networkx as nx
    membership = {}
    for index, nodes in enumerate(nx.connected_components(gt_graph)):
        for node in nodes:
            membership[int(node)] = int(index)
    return membership


def split_truth(payload) -> dict:
    """``{(neuron, (segment_a, segment_b)): "direct" | "gap" | "both"}``.

    Direct positives are distinct non-zero canonical labels across an edge with
    ``gt_edge_error == 1``. Gap positives are every distinct non-zero flank pair of a
    connected zero-label GT region. Zero and self pairs are rejected.
    """
    import networkx as nx

    gt = payload["gt_graph"]
    labels = np.asarray(payload["gt_node_canonical_label"])
    edge_error = np.asarray(payload["gt_edge_error"])
    neuron_of = gt_neuron_membership(gt)
    truth = {}

    def add(neuron, a, b, kind):
        a, b = int(a), int(b)
        if a == 0 or b == 0 or a == b:
            return
        key = (int(neuron), tuple(sorted((a, b))))
        old = truth.get(key)
        truth[key] = kind if old in (None, kind) else "both"

    edges = list(gt.edges())
    if len(edge_error) != len(edges):
        raise ValueError("gt_edge_error length %d != gt edge count %d" % (len(edge_error), len(edges)))
    for index, (u, v) in enumerate(edges):
        if int(edge_error[index]) == 1:
            add(neuron_of[int(u)], labels[int(u)], labels[int(v)], "direct")
    zero_nodes = [int(n) for n in gt.nodes if int(labels[int(n)]) == 0]
    for component in nx.connected_components(gt.subgraph(zero_nodes)):
        flanks = sorted({int(labels[int(nb)]) for node in component for nb in gt.neighbors(node)
                         if int(labels[int(nb)]) != 0})
        if len(flanks) < 2:
            continue
        neuron = neuron_of[int(next(iter(component)))]
        for i, a in enumerate(flanks):
            for b in flanks[i + 1:]:
                add(neuron, a, b, "gap")
    return truth


def build_split_universe(payload, rule):
    """``(samples, labels)``: one row per unordered candidate segment pair, sorted by pair."""
    rule = validate_rule("split", rule)
    xyz, node_segments, degrees = node_arrays(payload["fragments_graph"])
    grouped = pair_occurrences(xyz, node_segments, degrees, rule["mode"], rule["radius_um"],
                               rule["per_anchor_k"])
    if grouped is None:
        return [], np.asarray([], dtype=np.int64)
    truth_pairs = {pair for _, pair in split_truth(payload)}
    gt_labels = np.asarray(payload["gt_node_canonical_label"])
    neuron_of = gt_neuron_membership(payload["gt_graph"])
    segment_neurons = defaultdict(set)
    for node in payload["gt_graph"].nodes:
        segment = int(gt_labels[int(node)])
        if segment != 0:
            segment_neurons[segment].add(neuron_of[int(node)])
    known_merge_segments = set(int(x) for x in payload.get("gt_merge_labels", []))
    samples, labels = [], []
    for candidate_index, pair in enumerate(sorted(grouped)):
        occurrences = tuple(sorted(grouped[pair], key=lambda row: (
            row["gap_um"], row["node_id_a"], row["node_id_b"], row["anchor_side"], row["partner_rank"])))
        first = occurrences[0]
        neurons_a = segment_neurons.get(int(pair[0]), set())
        neurons_b = segment_neurons.get(int(pair[1]), set())
        samples.append({
            "candidate_id": int(candidate_index),
            "segment_id_a": int(pair[0]), "segment_id_b": int(pair[1]),
            "node_id_a": int(first["node_id_a"]), "node_id_b": int(first["node_id_b"]),
            "gap_um": float(first["gap_um"]), "anchor_side": first["anchor_side"],
            "partner_rank": int(first["partner_rank"]),
            "node_role_a": first["node_role_a"], "node_role_b": first["node_role_b"],
            "is_merge_creating": int(bool(neurons_a and neurons_b and neurons_a.isdisjoint(neurons_b))),
            "contains_known_merge_segment": int(pair[0] in known_merge_segments
                                                or pair[1] in known_merge_segments),
            "occurrences": occurrences,
        })
        labels.append(int(pair in truth_pairs))
    return samples, np.asarray(labels, dtype=np.int64)


def split_audit(payload, samples, labels) -> dict:
    """Truth coverage counts of a split universe (GT-reading, audit only)."""
    truth = split_truth(payload)
    truth_pairs = {pair for _, pair in truth}
    candidate_pairs = {(int(s["segment_id_a"]), int(s["segment_id_b"])) for s in samples}
    fragment_segments = set(segment_of_component(payload["fragments_graph"]).values())
    reachable = {pair for pair in truth_pairs
                 if pair[0] in fragment_segments and pair[1] in fragment_segments}
    kind_counts = defaultdict(int)
    for kind in truth.values():
        kind_counts[kind] += 1
    labels = np.asarray(labels)
    return {
        "n_rows": int(len(samples)),
        "n_positive": int(np.sum(labels == 1)),
        "n_negative": int(np.sum(labels == 0)),
        **{key: value for key, value in label_counts(labels).items()
           if key in ("n_judgeable", "n_unjudgeable")},
        "n_truth_records": int(len(truth)),
        "n_truth_pairs": int(len(truth_pairs)),
        "n_reachable_truth_pairs": int(len(reachable)),
        "n_candidate_truth_pairs": int(len(candidate_pairs & truth_pairs)),
        "n_merge_creating_candidates": int(sum(int(s["is_merge_creating"]) for s in samples)),
        "n_candidates_containing_known_merge_segment": int(sum(
            int(s["contains_known_merge_segment"]) for s in samples)),
        "candidate_recall_of_reachable": (
            float(len(candidate_pairs & reachable) / len(reachable)) if reachable else None),
        "truth_kind_counts": dict(kind_counts),
    }


# ----------------------------------------------------------------------------- merge
def kept_junctions(xyz, node_segments, degrees, nms_um):
    """``(junctions, kept)``: every valid degree>=3 node, and the mask of those that survive
    same-segment greedy suppression in order (-degree, node id)."""
    junctions = np.flatnonzero((degrees >= 3) & (node_segments > 0))
    kept = np.zeros(len(node_segments), dtype=bool)
    if not len(junctions):
        return junctions, kept
    if nms_um <= 0:
        kept[junctions] = True
        return junctions, kept
    from scipy.spatial import cKDTree

    order = junctions[np.lexsort((junctions, -degrees[junctions]))]
    position_of = {int(node): i for i, node in enumerate(junctions)}
    tree = cKDTree(xyz[junctions], copy_data=False)
    neighbour_lists = tree.query_ball_point(xyz[order], r=nms_um)
    suppressed = np.zeros(len(junctions), dtype=bool)
    for node, neighbours in zip(order, neighbour_lists):
        node = int(node)
        if suppressed[position_of[node]]:
            continue
        kept[node] = True
        own_segment = node_segments[node]
        for j in neighbours:
            if node_segments[junctions[j]] == own_segment:
                suppressed[j] = True
    return junctions, kept


def tip_to_shaft_gaps(xyz, node_segments, degrees, tip_gap_um, tip_partner_k):
    """``(nodes, gaps)`` sorted by node: shaft nodes that a tip of another segment pairs with
    (``tip_to_any_node`` pairing, ``tip_partner_k`` partner segments) within ``tip_gap_um``,
    each with its smallest gap."""
    grouped = pair_occurrences(xyz, node_segments, degrees, "tip_to_any_node",
                               float(tip_gap_um) + _QUERY_SLACK_UM, tip_partner_k)
    best = {}
    for occurrences in (grouped or {}).values():
        for occurrence in occurrences:
            partner = "b" if occurrence["anchor_side"] == "a" else "a"
            if occurrence["node_role_" + partner] != "non_tip":
                continue
            gap = float(occurrence["gap_um"])
            if gap > tip_gap_um:
                continue
            node = int(occurrence["node_id_" + partner])
            if gap < best.get(node, math.inf):
                best[node] = gap
    nodes = np.asarray(sorted(best), dtype=np.int64)
    return nodes, np.asarray([best[int(node)] for node in nodes], dtype=float)


def kept_shafts(xyz, node_segments, nodes, gaps, junction_kept, nms_um):
    """Mask of shaft candidates that survive: not within ``nms_um`` of a kept junction of
    their own segment, then same-segment greedy suppression in order (gap, node id)."""
    kept = np.zeros(len(node_segments), dtype=bool)
    nodes, gaps = np.asarray(nodes, dtype=np.int64), np.asarray(gaps, dtype=float)
    select = node_segments[nodes] > 0
    nodes, gaps = nodes[select], gaps[select]
    if not len(nodes):
        return kept
    if nms_um <= 0:
        kept[nodes[~junction_kept[nodes]]] = True
        return kept
    from scipy.spatial import cKDTree

    junction_nodes = np.flatnonzero(junction_kept)
    if len(junction_nodes):
        near = cKDTree(xyz[junction_nodes]).query_ball_point(xyz[nodes], r=nms_um)
        covered = np.fromiter((any(node_segments[junction_nodes[j]] == node_segments[n] for j in hits)
                               for n, hits in zip(nodes, near)), dtype=bool, count=len(nodes))
        nodes, gaps = nodes[~covered], gaps[~covered]
    if not len(nodes):
        return kept
    nodes = nodes[np.lexsort((nodes, gaps))]
    neighbours = cKDTree(xyz[nodes]).query_ball_point(xyz[nodes], r=nms_um)
    suppressed = np.zeros(len(nodes), dtype=bool)
    for i, node in enumerate(nodes):
        if suppressed[i]:
            continue
        kept[int(node)] = True
        for j in neighbours[i]:
            if node_segments[nodes[j]] == node_segments[node]:
                suppressed[j] = True
    return kept


def merge_candidate_nodes(xyz, node_segments, degrees, rule):
    """``(kept, tip_gap, counts)``: mask of candidate nodes, ``{shaft node: gap}`` and raw counts."""
    rule = validate_rule("merge", rule)
    junctions, kept = kept_junctions(xyz, node_segments, degrees, rule["nms_um"])
    counts = {"n_junctions_raw": int(len(junctions)), "n_junction_candidates": int(kept.sum()),
              "n_tip_to_shaft_raw": 0, "n_tip_to_shaft_candidates": 0}
    tip_gap = {}
    if rule["mode"] == "junction+tip_to_shaft":
        nodes, gaps = tip_to_shaft_gaps(xyz, node_segments, degrees, rule["tip_gap_um"],
                                        rule["tip_partner_k"])
        shafts = kept_shafts(xyz, node_segments, nodes, gaps, kept, rule["nms_um"])
        if (shafts & kept).any():
            raise RuntimeError("A tip-to-shaft candidate coincides with a kept junction")
        gap_of = dict(zip(nodes.tolist(), gaps.tolist()))
        tip_gap = {int(node): float(gap_of[int(node)]) for node in np.flatnonzero(shafts)}
        counts["n_tip_to_shaft_raw"] = int(len(nodes))
        counts["n_tip_to_shaft_candidates"] = int(shafts.sum())
        kept = kept | shafts
    return kept, tip_gap, counts


def merge_site_provenance(payload) -> dict:
    """Snapshot of the label-source records (never feature inputs).

    The digest is order independent but retains duplicate records. It covers segment, XYZ,
    associated neurons and source tags. Legacy untagged sites are geometric; shared
    evidence counts once in total.
    """
    stored = payload.get("gt_merge_sites")
    counts = {"geometric_only": 0, "two_gt_only": 0, "shared": 0, "untagged_legacy": 0}
    records = []
    for site in ([] if stored is None else stored):
        declared = site.get("sources", [])
        if not isinstance(declared, (list, tuple, set)):
            raise ValueError("Merge-site sources must be a collection")
        sources = set(declared)
        if site.get("source") is not None:
            sources.add(site["source"])
        if not sources:
            sources = {"geometric_walk"}
            counts["untagged_legacy"] += 1
        if not sources.issubset(set(SITE_SOURCES)):
            raise ValueError("Unknown merge-site source: %r" % sources)
        category = ("shared" if len(sources) == 2 else
                    "geometric_only" if "geometric_walk" in sources else "two_gt_only")
        counts[category] += 1
        xyz = np.asarray(site["xyz"], dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("Merge-site coordinates must be finite XYZ")
        neurons = site.get("gt_neurons", [])
        if not isinstance(neurons, (list, tuple, set)):
            raise ValueError("Merge-site gt_neurons must be a collection")
        records.append(json.dumps({
            "segment_id": int(site["segment_id"]), "xyz": xyz.tolist(),
            "gt_neuron": str(site.get("gt_neuron", "")),
            "gt_neurons": sorted(set(map(str, neurons))), "sources": sorted(sources),
        }, sort_keys=True, separators=(",", ":"), allow_nan=False))
    digest = hashlib.sha256(json.dumps(sorted(records), separators=(",", ":")).encode()).hexdigest()
    return {
        "schema_version": 1, "sites_available": stored is not None,
        "n_sites": len(records), "source_counts": counts,
        "site_records_sha256": digest if stored is not None else None,
        "site_digest_fields": ["segment_id", "xyz", "gt_neuron", "gt_neurons", "sources"],
        "generation_metadata": copy.deepcopy(payload.get("gt_merge_site_metadata")),
        "negative_label_meaning": "no recorded site within positive-label radius; not verified non-merge",
    }


def site_distances_to_kept(payload, xyz, node_segments, kept, claim_radius_um):
    """GT consumer: cable distances between recorded merge sites and kept candidates.

    Each site snaps to the nearest node of its own segment within ``SITE_SNAP_MAX_UM``;
    a bounded Dijkstra over fragment cable then runs to ``claim_radius_um``. Returns the
    per-candidate minimum distance to any site and site-side audit values.
    """
    from scipy.spatial import cKDTree

    graph = payload["fragments_graph"]
    sites = list(payload.get("gt_merge_sites") or [])
    candidate_distance = {}
    n_snapped = 0
    site_min_distances = []
    if sites:
        tree = cKDTree(xyz, copy_data=False)
        for site in sites:
            segment_id = int(site["segment_id"])
            point = np.asarray(site["xyz"], dtype=float)
            hits = np.asarray(tree.query_ball_point(point, r=SITE_SNAP_MAX_UM), dtype=np.int64)
            own = hits[node_segments[hits] == segment_id] if len(hits) else hits
            if not len(own):
                site_min_distances.append(float("inf"))
                continue
            start = int(own[int(np.argmin(np.linalg.norm(xyz[own] - point, axis=1)))])
            n_snapped += 1
            best = {start: 0.0}
            heap = [(0.0, start)]
            site_best = float("inf")
            while heap:
                distance, node = heapq.heappop(heap)
                if distance > best.get(node, float("inf")):
                    continue
                if kept[node]:
                    site_best = min(site_best, distance)
                    previous = candidate_distance.get(node)
                    if previous is None or distance < previous:
                        candidate_distance[node] = distance
                for neighbour in graph.neighbors(node):
                    neighbour = int(neighbour)
                    step = distance + float(np.linalg.norm(xyz[neighbour] - xyz[node]))
                    if step <= claim_radius_um and step < best.get(neighbour, float("inf")):
                        best[neighbour] = step
                        heapq.heappush(heap, (step, neighbour))
            site_min_distances.append(site_best)
    return candidate_distance, {
        "n_sites": len(sites), "n_sites_snapped": n_snapped,
        "site_min_distances": site_min_distances,
        "site_provenance": merge_site_provenance(payload),
    }


def build_merge_universe(payload, rule):
    """``(samples, labels, audit)``: one row per candidate node, in node-id order."""
    rule = validate_rule("merge", rule)
    xyz, node_segments, degrees = node_arrays(payload["fragments_graph"])
    kept, tip_gap, counts = merge_candidate_nodes(xyz, node_segments, degrees, rule)
    candidate_distance, site_audit = site_distances_to_kept(
        payload, xyz, node_segments, kept, rule["claim_radius_um"])
    samples, labels = [], []
    for candidate_index, node in enumerate(int(n) for n in np.flatnonzero(kept)):
        distance = candidate_distance.get(node, float("inf"))
        positive = distance <= rule["positive_label_radius_um"]
        shaft = node in tip_gap
        samples.append({
            "candidate_id": int(candidate_index), "node_id": int(node),
            "segment_id": int(node_segments[node]), "degree": int(degrees[node]),
            "x_um": float(xyz[node][0]), "y_um": float(xyz[node][1]), "z_um": float(xyz[node][2]),
            "candidate_family": TIP_TO_SHAFT if shaft else JUNCTION,
            "tip_gap_um": tip_gap[node] if shaft else None,
            "distance_to_nearest_gt_site_um": float(distance) if np.isfinite(distance) else float("nan"),
            "in_ambiguous_ring": int(np.isfinite(distance) and not positive),
        })
        labels.append(int(positive))
    return samples, np.asarray(labels, dtype=np.int64), {**site_audit, **counts}


def merge_audit(audit, samples, labels, rule) -> dict:
    """Row and site-coverage counts of a merge universe (GT-reading, audit only)."""
    rule = validate_rule("merge", rule)
    site_min = np.asarray(audit.get("site_min_distances", []), dtype=float)
    n_sites = int(audit.get("n_sites", 0))
    covered = {radius: int(np.sum(site_min <= radius)) if n_sites else 0
               for radius in (rule["positive_label_radius_um"], rule["claim_radius_um"])}
    labels = np.asarray(labels)
    return {
        "n_rows": int(len(samples)),
        "n_positive": int(np.sum(labels == 1)),
        "n_negative": int(np.sum(labels == 0)),
        "n_ambiguous_ring": int(sum(int(sample["in_ambiguous_ring"]) for sample in samples)),
        **{key: int(audit.get(key, 0)) for key in ("n_junctions_raw", "n_junction_candidates",
                                                   "n_tip_to_shaft_raw", "n_tip_to_shaft_candidates")},
        "n_sites": n_sites,
        "site_provenance": audit.get("site_provenance"),
        "n_sites_snapped": int(audit.get("n_sites_snapped", 0)),
        "n_sites_covered_at_label_radius": covered[rule["positive_label_radius_um"]],
        "site_recall_at_label_radius": (
            float(covered[rule["positive_label_radius_um"]] / n_sites) if n_sites else None),
        "n_sites_covered_at_claim_radius": covered[rule["claim_radius_um"]],
        "site_recall_at_claim_radius": float(covered[rule["claim_radius_um"]] / n_sites) if n_sites else None,
    }


def build_universe(kind, payload, rule):
    """``(samples, labels)`` of either kind under ``rule``."""
    if kind == "merge":
        samples, labels, _ = build_merge_universe(payload, rule)
        return samples, labels
    if kind == "split":
        return build_split_universe(payload, rule)
    raise ValueError(f"Unknown candidate kind {kind!r}")


# ----------------------------------------------------------------------------- labels
def gt_segments(gt_node_label) -> set:
    """Segments that carry at least one GT node label."""
    labels = np.asarray(gt_node_label)
    return {int(x) for x in np.unique(labels) if int(x) != 0}


def label_availability(kind, payload, rows):
    """``(mask, definition)``: whether GT can judge each row, and the rule that decided.

    split ``segment_on_gt``: both segments carry a GT node label. merge ``near_gt_150``: the
    segment carries a GT node label and the candidate lies within ``NEAR_GT_UM`` of a GT
    node (``payload["gt_graph"].node_xyz``); without GT coordinates the merge rule falls
    back to ``segment_on_gt`` and says so. A positive row is always judgeable.
    """
    if kind not in KINDS:
        raise ValueError(f"Unknown candidate kind {kind!r}")
    segments = gt_segments(payload["gt_node_canonical_label"])
    if kind == "split":
        mask = np.fromiter((int(r["segment_id_a"]) in segments and int(r["segment_id_b"]) in segments
                            for r in rows), dtype=bool, count=len(rows))
        return mask, "segment_on_gt"
    on_gt = np.fromiter((int(r["segment_id"]) in segments for r in rows), dtype=bool, count=len(rows))
    gt = payload.get("gt_graph")
    gt_xyz = getattr(gt, "node_xyz", None) if gt is not None else None
    if gt_xyz is None or len(gt_xyz) == 0 or not len(rows):
        return on_gt, "segment_on_gt"
    from scipy.spatial import cKDTree
    xyz = np.array([[r["x_um"], r["y_um"], r["z_um"]] for r in rows], dtype=float).reshape(-1, 3)
    distance, _ = cKDTree(np.asarray(gt_xyz, dtype=float)).query(xyz)
    return on_gt & (np.asarray(distance) <= NEAR_GT_UM), "near_gt_150"


def three_valued_labels(kind, payload, rows, labels):
    """``(labels, definition)``: 1 error, 0 GT-confirmed no error, NaN not judgeable."""
    mask, definition = label_availability(kind, payload, rows)
    labels = np.asarray(labels, dtype=float)
    if labels.shape != mask.shape:
        raise ValueError("labels and rows are not aligned")
    if np.any((labels == 1) & ~mask):
        raise ValueError("a positive row is marked unavailable; the inputs contradict each other")
    out = labels.copy()
    out[~mask] = np.nan
    return out, definition


def label_counts(labels) -> dict:
    labels = np.asarray(labels, dtype=float)
    nan = int(np.isnan(labels).sum())
    ones = int(np.sum(labels == 1))
    return {"n_positive": ones, "n_negative": int(len(labels) - nan - ones),
            "n_judgeable": int(len(labels) - nan), "n_unjudgeable": nan}


# ----------------------------------------------------------------------------- folds
def row_segments(kind, row) -> tuple:
    if kind == "merge":
        return (int(row["segment_id"]),)
    return (int(row["segment_id_a"]), int(row["segment_id_b"]))


def fold_groups(kind, rows, node_xyz=None, block_um=FOLD_BLOCK_UM, prefix=None) -> list:
    """Label-free group key per row: the fragment segment (merge) or the ``block_um`` spatial
    block of the midpoint of the row's two anchors (split, ``node_xyz`` of the fragment graph).

    The keys are the benchmark harness's (``<brain>/fragment/<segment>`` and
    ``<brain>/block/<x>/<y>/<z>``) when ``prefix`` is the brain id, so the SHA-256 tie-break
    in ``fold_partitions`` orders groups exactly as the companions' folds do and a detector
    trained on that brain holds out the companion's own held rows.
    """
    head = "" if prefix in (None, "") else f"{prefix}/"
    if kind == "merge":
        return [f"{head}fragment/{int(r['segment_id'])}" for r in rows]
    if kind != "split":
        raise ValueError(f"Unknown candidate kind {kind!r}")
    if not len(rows):
        return []
    if node_xyz is None:
        raise ValueError("split fold groups need the fragment graph's node_xyz")
    xyz = np.asarray(node_xyz, dtype=float)
    a = np.fromiter((int(r["node_id_a"]) for r in rows), dtype=np.int64, count=len(rows))
    b = np.fromiter((int(r["node_id_b"]) for r in rows), dtype=np.int64, count=len(rows))
    if (a < 0).any() or (b < 0).any() or (a >= len(xyz)).any() or (b >= len(xyz)).any():
        raise ValueError("split anchors fall outside node_xyz")
    centers = (xyz[a] + xyz[b]) / 2.0
    if not np.isfinite(centers).all():
        raise ValueError("non-finite anchor coordinates")
    blocks = np.floor(centers / float(block_um)).astype(np.int64)
    return [f"{head}block/{x}/{y}/{z}" for x, y, z in blocks]


def fold_partitions(kind, rows, groups, n_folds):
    """``[(fit, held), ...]`` row-index arrays, one pair per fold.

    Groups are assigned to folds largest first (SHA-256 of the key breaks ties), each to the
    least loaded fold, without reading labels. The held set of a fold is every row of its
    groups, so the held sets partition the rows; the fit set is every other row that shares
    no segment with a held row (the purge). The same rule as the benchmark's internal folds.
    """
    rows = list(rows)
    groups = [str(g) for g in groups]
    if len(groups) != len(rows):
        raise ValueError("groups and rows are not aligned")
    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError("at least two folds are needed")
    counts = defaultdict(int)
    for group in groups:
        counts[group] += 1
    if len(counts) < n_folds:
        raise ValueError(f"{n_folds} folds need at least {n_folds} groups; {len(counts)} found")
    loads, assignment = [0] * n_folds, {}
    for group in sorted(counts, key=lambda g: (-counts[g], hashlib.sha256(g.encode()).hexdigest())):
        fold = min(range(n_folds), key=lambda f: (loads[f], f))
        assignment[group] = fold
        loads[fold] += counts[group]
    fold_of_row = np.fromiter((assignment[g] for g in groups), dtype=np.int64, count=len(rows))
    segments = [row_segments(kind, row) for row in rows]
    partitions = []
    for fold in range(n_folds):
        held = np.flatnonzero(fold_of_row == fold)
        held_segments = {segment for index in held for segment in segments[index]}
        fit = np.asarray([index for index in np.flatnonzero(fold_of_row != fold)
                          if not any(segment in held_segments for segment in segments[index])],
                         dtype=np.int64)
        partitions.append((fit, held))
    return partitions

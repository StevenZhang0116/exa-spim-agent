"""Reviewed target adapters injected into the shared detector runtime.

The feature-writing agent owns feature math only.  Candidate identity and
scientific labels are driver-owned so a generated fragment cannot accidentally
train a split detector on merge labels (or leak GT-only split metadata into X).
"""

from __future__ import annotations

from .contracts import DetectorTarget, target_spec


TARGET_ADAPTER_MARKER = "# __DETECTOR_TARGET_ADAPTER__"


def target_adapter_source(
    target: DetectorTarget,
    *,
    candidate_policy: dict | None = None,
    candidate_policy_sha256: str | None = None,
) -> str:
    spec = target_spec(target)
    common = f'''DETECTOR_TARGET = {spec.target.value!r}
DETECTOR_FILENAME = {spec.detector_name!r}
OUTPUT_PREFIX = {spec.output_prefix!r}
ROW_UNIT = {spec.row_unit!r}
ROW_UNIT_PLURAL = {spec.row_unit_plural!r}
LABEL_NAME = {spec.label_name!r}
POSITIVE_NAME = {spec.positive_name!r}
NEGATIVE_NAME = {spec.negative_name!r}
SCORE_PREFIX = {spec.score_prefix!r}
'''
    if spec.target is DetectorTarget.SPLIT:
        if candidate_policy is None or candidate_policy_sha256 is None:
            raise ValueError(
                "Split target adapters require a validated frozen candidate policy."
            )
        constants = f'''CANDIDATE_POLICY_CONFIG_ID = {candidate_policy["config_id"]!r}
CANDIDATE_PAIRING_RULE = {candidate_policy["mode"]!r}
CANDIDATE_MAX_DISTANCE_UM = {float(candidate_policy["radius_um"])!r}
CANDIDATE_PER_ANCHOR_K = {candidate_policy["per_anchor_k"]!r}
CANDIDATE_GLOBAL_CAP = {candidate_policy["global_cap"]!r}
CANDIDATE_POLICY_SHA256 = {candidate_policy_sha256!r}
'''
        return common + constants + _SPLIT_ADAPTER
    if candidate_policy is not None or candidate_policy_sha256 is not None:
        raise ValueError("Merge target adapters do not accept a split candidate policy.")
    return common + _MERGE_ADAPTER


_MERGE_ADAPTER = r'''
def build_sample_universe(payload):
    """Return one row per non-zero canonical segment and its merge label."""
    cached = payload.get("__detector_sample_universe_cache__")
    if cached is not None:
        return cached
    node_label = np.asarray(payload["gt_node_canonical_label"])
    samples = sorted(int(x) for x in np.unique(node_label) if int(x) != 0)
    positives = set(int(x) for x in payload["gt_merge_labels"])
    labels = np.asarray([int(s in positives) for s in samples], dtype=np.int64)
    payload["__detector_sample_universe_cache__"] = (samples, labels)
    return samples, labels


# Compatibility for already-audited merge feature fragments while new
# generation instructions use the target-neutral name.
build_segment_universe = build_sample_universe


def sample_output_frame(samples, labels):
    import pandas as pd
    return pd.DataFrame({"segment_id": samples, LABEL_NAME: labels})


def sample_display(sample):
    return str(int(sample))


def sample_universe_audit(payload, samples, labels):
    return {
        "row_unit": ROW_UNIT,
        "n_rows": int(len(samples)),
        "n_positive": int(np.sum(np.asarray(labels) == 1)),
        "n_negative": int(np.sum(np.asarray(labels) == 0)),
    }
'''


_SPLIT_ADAPTER = r'''
def _gt_neuron_membership(gt):
    """Stable GT-neuron key without assuming custom methods survived unpickling."""
    import networkx as nx
    membership = {}
    for index, nodes in enumerate(nx.connected_components(gt)):
        for node in nodes:
            membership[int(node)] = int(index)
    return membership


def _derive_split_truth(payload):
    """Return direct/gap positive segment pairs and audit counts.

    Direct positives are distinct non-zero canonical labels across an edge with
    gt_edge_error==1.  Gap positives are every distinct non-zero flank pair of a
    connected zero-label GT region.  This matches the recurring contract in the
    audited split hypothesis sources while rejecting degenerate zero/self pairs.
    """
    import networkx as nx

    gt = payload["gt_graph"]
    labels = np.asarray(payload["gt_node_canonical_label"])
    edge_error = np.asarray(payload["gt_edge_error"])
    neuron_of = _gt_neuron_membership(gt)
    truth = {}

    def add(neuron, a, b, kind):
        a, b = int(a), int(b)
        if a == 0 or b == 0 or a == b:
            return
        pair = tuple(sorted((a, b)))
        key = (int(neuron), pair)
        old = truth.get(key)
        truth[key] = kind if old in (None, kind) else "both"

    edges = list(gt.edges())
    if len(edge_error) != len(edges):
        raise ValueError(
            "gt_edge_error length %d != gt edge count %d"
            % (len(edge_error), len(edges)))
    for index, (u, v) in enumerate(edges):
        if int(edge_error[index]) == 1:
            add(neuron_of[int(u)], labels[int(u)], labels[int(v)], "direct")

    zero_nodes = [int(n) for n in gt.nodes if int(labels[int(n)]) == 0]
    for component in nx.connected_components(gt.subgraph(zero_nodes)):
        flanks = sorted({
            int(labels[int(nb)])
            for node in component
            for nb in gt.neighbors(node)
            if int(labels[int(nb)]) != 0
        })
        if len(flanks) < 2:
            continue
        neuron = neuron_of[int(next(iter(component)))]
        for i, a in enumerate(flanks):
            for b in flanks[i + 1:]:
                add(neuron, a, b, "gap")
    return truth


def compatible_occurrences(sample, node_role_requirement):
    """Return occurrences on which one inventoried feature is defined.

    Applicability affects feature missingness only. It never changes candidate
    identity or removes a row from the runtime sample universe.
    """
    occurrences = tuple(sample.get("occurrences", ()))
    if node_role_requirement == "no_tip_requirement":
        return occurrences
    if node_role_requirement == "requires_both_tips":
        return tuple(
            occurrence for occurrence in occurrences
            if occurrence.get("node_role_a") == "tip"
            and occurrence.get("node_role_b") == "tip"
        )
    if node_role_requirement == "requires_tip_anchor":
        compatible = []
        for occurrence in occurrences:
            side = occurrence.get("anchor_side")
            if side not in {"a", "b"}:
                raise ValueError("split occurrence has invalid anchor_side %r" % side)
            if occurrence.get("node_role_" + side) == "tip":
                compatible.append(occurrence)
        return tuple(compatible)
    raise ValueError(
        "unsupported split feature node_role_requirement %r"
        % node_role_requirement
    )


def build_sample_universe(payload):
    """Build one deterministic row per unordered candidate segment pair.

    Candidate enumeration is completely determined by the frozen build-time
    policy embedded above. Per-anchor k ranks distinct partner segments, not
    partner nodes. Candidate construction uses fragment geometry only; GT is
    consulted only after the universe is frozen to attach is_split.
    """
    from scipy.spatial import cKDTree

    cached = payload.get("__detector_sample_universe_cache__")
    if cached is not None:
        return cached
    frag = payload["fragments_graph"]
    comp_to_seg = build_comp_to_seg(frag)
    n_nodes = int(frag.number_of_nodes())
    xyz = np.asarray(frag.node_xyz, dtype=float)
    node_components = np.asarray(frag.node_component_id, dtype=np.int64)
    if xyz.shape[0] != n_nodes or node_components.shape[0] != n_nodes:
        raise ValueError(
            "split candidate generation requires node_xyz and node_component_id "
            "to align with contiguous graph node ids"
        )
    if n_nodes and (not frag.has_node(0) or not frag.has_node(n_nodes - 1)):
        raise ValueError("split candidate generation requires node ids 0..N-1")
    node_segments = np.fromiter(
        (int(comp_to_seg.get(int(component), -1)) for component in node_components),
        dtype=np.int64,
        count=n_nodes,
    )
    degrees = np.fromiter(
        (int(frag.degree(node)) for node in range(n_nodes)),
        dtype=np.int64,
        count=n_nodes,
    )
    tips = np.flatnonzero((degrees == 1) & (node_segments > 0)).astype(np.int64)
    valid_nodes = np.flatnonzero(node_segments > 0).astype(np.int64)
    if CANDIDATE_PAIRING_RULE == "tip_to_tip":
        anchor_nodes = tips
        partner_nodes = tips
    elif CANDIDATE_PAIRING_RULE == "tip_to_any_node":
        anchor_nodes = tips
        partner_nodes = valid_nodes
    elif CANDIDATE_PAIRING_RULE == "any_node_to_any_node":
        anchor_nodes = valid_nodes
        partner_nodes = valid_nodes
    else:
        raise ValueError(
            "unsupported embedded split candidate pairing rule %r"
            % CANDIDATE_PAIRING_RULE
        )
    if not len(anchor_nodes) or not len(partner_nodes):
        result = ([], np.asarray([], dtype=np.int64))
        payload["__detector_sample_universe_cache__"] = result
        return result

    tree = cKDTree(xyz[partner_nodes], copy_data=False)
    grouped = defaultdict(list)
    batch_size = 2000
    for start in range(0, len(anchor_nodes), batch_size):
        batch = anchor_nodes[start : start + batch_size]
        neighbor_lists = tree.query_ball_point(
            xyz[batch], r=CANDIDATE_MAX_DISTANCE_UM, workers=1
        )
        for anchor_node, neighbor_positions in zip(batch, neighbor_lists):
            anchor_node = int(anchor_node)
            anchor_segment = int(node_segments[anchor_node])
            best_partner_by_segment = {}
            for position in neighbor_positions:
                partner_node = int(partner_nodes[int(position)])
                partner_segment = int(node_segments[partner_node])
                if partner_node == anchor_node or partner_segment == anchor_segment:
                    continue
                distance = float(np.linalg.norm(
                    xyz[partner_node] - xyz[anchor_node]
                ))
                previous = best_partner_by_segment.get(partner_segment)
                candidate = (distance, partner_node)
                if previous is None or candidate < previous:
                    best_partner_by_segment[partner_segment] = candidate

            ranked = sorted(
                (distance, partner_segment, partner_node)
                for partner_segment, (distance, partner_node)
                in best_partner_by_segment.items()
            )
            limit = (
                len(ranked)
                if CANDIDATE_PER_ANCHOR_K is None
                else int(CANDIDATE_PER_ANCHOR_K)
            )
            for zero_rank, (distance, partner_segment, partner_node) in enumerate(
                ranked[:limit]
            ):
                segment_a, segment_b = sorted((anchor_segment, partner_segment))
                if anchor_segment == segment_a:
                    node_a, node_b = anchor_node, partner_node
                    anchor_side = "a"
                else:
                    node_a, node_b = partner_node, anchor_node
                    anchor_side = "b"
                grouped[(segment_a, segment_b)].append({
                    "node_id_a": int(node_a),
                    "node_id_b": int(node_b),
                    "gap_um": float(distance),
                    "anchor_side": anchor_side,
                    "partner_rank": int(zero_rank + 1),
                    "node_role_a": "tip" if int(degrees[node_a]) == 1 else "non_tip",
                    "node_role_b": "tip" if int(degrees[node_b]) == 1 else "non_tip",
                })

    truth_pairs = {pair for _, pair in _derive_split_truth(payload)}
    gt_labels = np.asarray(payload["gt_node_canonical_label"])
    neuron_of = _gt_neuron_membership(payload["gt_graph"])
    segment_neurons = defaultdict(set)
    for node in payload["gt_graph"].nodes:
        segment = int(gt_labels[int(node)])
        if segment != 0:
            segment_neurons[segment].add(neuron_of[int(node)])
    known_merge_segments = set(int(x) for x in payload.get("gt_merge_labels", []))
    samples = []
    labels = []
    for candidate_index, pair in enumerate(sorted(grouped)):
        occurrences = tuple(sorted(
            grouped[pair],
            key=lambda row: (
                row["gap_um"], row["node_id_a"], row["node_id_b"],
                row["anchor_side"], row["partner_rank"],
            ),
        ))
        representative = occurrences[0]
        neurons_a = segment_neurons.get(int(pair[0]), set())
        neurons_b = segment_neurons.get(int(pair[1]), set())
        merge_creating = bool(
            neurons_a and neurons_b and neurons_a.isdisjoint(neurons_b))
        samples.append({
            "candidate_id": int(candidate_index),
            "segment_id_a": int(pair[0]),
            "segment_id_b": int(pair[1]),
            "node_id_a": int(representative["node_id_a"]),
            "node_id_b": int(representative["node_id_b"]),
            "gap_um": float(representative["gap_um"]),
            "anchor_side": representative["anchor_side"],
            "partner_rank": int(representative["partner_rank"]),
            "node_role_a": representative["node_role_a"],
            "node_role_b": representative["node_role_b"],
            "is_merge_creating": int(merge_creating),
            "contains_known_merge_segment": int(
                pair[0] in known_merge_segments or pair[1] in known_merge_segments),
            "occurrences": occurrences,
        })
        labels.append(int(pair in truth_pairs))
    result = (samples, np.asarray(labels, dtype=np.int64))
    payload["__detector_sample_universe_cache__"] = result
    return result


def sample_output_frame(samples, labels):
    import pandas as pd
    public = [
        {key: sample[key] for key in (
            "candidate_id", "segment_id_a", "segment_id_b",
            "node_id_a", "node_id_b", "gap_um", "anchor_side", "partner_rank",
            "node_role_a", "node_role_b",
            "is_merge_creating", "contains_known_merge_segment")}
        for sample in samples
    ]
    frame = pd.DataFrame(public, columns=[
        "candidate_id", "segment_id_a", "segment_id_b",
        "node_id_a", "node_id_b", "gap_um", "anchor_side", "partner_rank",
        "node_role_a", "node_role_b",
        "is_merge_creating", "contains_known_merge_segment"])
    frame[LABEL_NAME] = np.asarray(labels, dtype=np.int64)
    return frame


def sample_display(sample):
    return "%s:%s (nodes %s:%s, gap %.3f um)" % (
        sample["segment_id_a"], sample["segment_id_b"],
        sample["node_id_a"], sample["node_id_b"],
        sample["gap_um"])


def sample_universe_audit(payload, samples, labels):
    truth = _derive_split_truth(payload)
    truth_pairs = {pair for _, pair in truth}
    candidate_pairs = {
        (int(s["segment_id_a"]), int(s["segment_id_b"])) for s in samples
    }
    frag_segments = set(build_comp_to_seg(payload["fragments_graph"]).values())
    reachable = {
        pair for pair in truth_pairs if pair[0] in frag_segments and pair[1] in frag_segments
    }
    kind_counts = defaultdict(int)
    for kind in truth.values():
        kind_counts[kind] += 1
    return {
        "row_unit": ROW_UNIT,
        "candidate_policy_config_id": CANDIDATE_POLICY_CONFIG_ID,
        "candidate_policy_sha256": CANDIDATE_POLICY_SHA256,
        "candidate_pairing_rule": CANDIDATE_PAIRING_RULE,
        "candidate_max_distance_um": CANDIDATE_MAX_DISTANCE_UM,
        "candidate_per_anchor_k": CANDIDATE_PER_ANCHOR_K,
        "candidate_global_cap": CANDIDATE_GLOBAL_CAP,
        "n_rows": int(len(samples)),
        "n_positive": int(np.sum(np.asarray(labels) == 1)),
        "n_negative": int(np.sum(np.asarray(labels) == 0)),
        "n_truth_records": int(len(truth)),
        "n_truth_pairs": int(len(truth_pairs)),
        "n_reachable_truth_pairs": int(len(reachable)),
        "n_candidate_truth_pairs": int(len(candidate_pairs & truth_pairs)),
        "n_merge_creating_candidates": int(sum(
            int(s["is_merge_creating"]) for s in samples)),
        "n_candidates_containing_known_merge_segment": int(sum(
            int(s["contains_known_merge_segment"]) for s in samples)),
        "candidate_recall_of_reachable": (
            float(len(candidate_pairs & reachable) / len(reachable)) if reachable else None),
        "truth_kind_counts": dict(kind_counts),
    }
'''

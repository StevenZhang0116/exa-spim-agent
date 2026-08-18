"""Reviewed target adapters injected into the shared detector runtime.

The feature-writing agent owns feature math only.  Candidate identity and
scientific labels are driver-owned so a generated fragment cannot accidentally
train a split detector on merge labels (or leak GT-only split metadata into X).
"""

from __future__ import annotations

from .contracts import DetectorTarget, target_spec


TARGET_ADAPTER_MARKER = "# __DETECTOR_TARGET_ADAPTER__"


def target_adapter_source(target: DetectorTarget) -> str:
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
        adapter = _SPLIT_ADAPTER.replace(
            "CANDIDATE_MAX_DISTANCE_UM = 30.0",
            f"CANDIDATE_MAX_DISTANCE_UM = {float(spec.candidate_radius_um)!r}",
        )
        return common + adapter
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
CANDIDATE_MAX_DISTANCE_UM = 30.0


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


def build_sample_universe(payload):
    """Build one deterministic row per unordered candidate segment pair.

    All leaf-to-leaf endpoint occurrences within 30 um are retained in the
    internal ``occurrences`` tuple for audited feature reductions.  The closest
    occurrence (distance, then node ids) supplies the stable output location.
    Candidate construction uses fragment geometry only; GT is consulted only
    after the universe is frozen to attach is_split.
    """
    from scipy.spatial import cKDTree

    cached = payload.get("__detector_sample_universe_cache__")
    if cached is not None:
        return cached
    frag = payload["fragments_graph"]
    comp_to_seg = build_comp_to_seg(frag)
    leaves = sorted(int(n) for n, degree in frag.degree() if int(degree) == 1)
    if len(leaves) < 2:
        result = ([], np.asarray([], dtype=np.int64))
        payload["__detector_sample_universe_cache__"] = result
        return result
    xyz = np.asarray(frag.node_xyz, dtype=float)
    tree = cKDTree(xyz[leaves])
    grouped = defaultdict(list)
    for i, j in sorted(tree.query_pairs(CANDIDATE_MAX_DISTANCE_UM)):
        node_a, node_b = leaves[int(i)], leaves[int(j)]
        comp_a = int(frag.node_component_id[node_a])
        comp_b = int(frag.node_component_id[node_b])
        if comp_a == comp_b:
            continue
        seg_a, seg_b = int(comp_to_seg[comp_a]), int(comp_to_seg[comp_b])
        if seg_a == seg_b:
            continue
        if seg_b < seg_a:
            seg_a, seg_b = seg_b, seg_a
            node_a, node_b = node_b, node_a
        distance = float(np.linalg.norm(xyz[node_a] - xyz[node_b]))
        grouped[(seg_a, seg_b)].append((node_a, node_b, distance))

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
        occurrences = tuple(sorted(grouped[pair], key=lambda row: (row[2], row[0], row[1])))
        node_a, node_b, distance = occurrences[0]
        neurons_a = segment_neurons.get(int(pair[0]), set())
        neurons_b = segment_neurons.get(int(pair[1]), set())
        merge_creating = bool(
            neurons_a and neurons_b and neurons_a.isdisjoint(neurons_b))
        samples.append({
            "candidate_id": int(candidate_index),
            "segment_id_a": int(pair[0]),
            "segment_id_b": int(pair[1]),
            "endpoint_node_id_a": int(node_a),
            "endpoint_node_id_b": int(node_b),
            "gap_um": float(distance),
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
            "endpoint_node_id_a", "endpoint_node_id_b", "gap_um",
            "is_merge_creating", "contains_known_merge_segment")}
        for sample in samples
    ]
    frame = pd.DataFrame(public, columns=[
        "candidate_id", "segment_id_a", "segment_id_b",
        "endpoint_node_id_a", "endpoint_node_id_b", "gap_um",
        "is_merge_creating", "contains_known_merge_segment"])
    frame[LABEL_NAME] = np.asarray(labels, dtype=np.int64)
    return frame


def sample_display(sample):
    return "%s:%s (nodes %s:%s, gap %.3f um)" % (
        sample["segment_id_a"], sample["segment_id_b"],
        sample["endpoint_node_id_a"], sample["endpoint_node_id_b"],
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
        "candidate_max_distance_um": CANDIDATE_MAX_DISTANCE_UM,
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

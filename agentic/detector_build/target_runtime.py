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
    if spec.target is DetectorTarget.MERGE_SITE:
        if candidate_policy is None or candidate_policy_sha256 is None:
            raise ValueError(
                "Merge-site target adapters require a validated frozen "
                "candidate policy."
            )
        constants = f'''CANDIDATE_POLICY_CONFIG_ID = {candidate_policy["config_id"]!r}
CANDIDATE_MODE = {candidate_policy["mode"]!r}
CANDIDATE_NMS_UM = {float(candidate_policy["nms_um"])!r}
CANDIDATE_CLAIM_RADIUS_UM = {float(candidate_policy["claim_radius_um"])!r}
CANDIDATE_POSITIVE_LABEL_RADIUS_UM = {float(candidate_policy["positive_label_radius_um"])!r}
CANDIDATE_SITE_SNAP_MAX_UM = 50.0
CANDIDATE_POLICY_SHA256 = {candidate_policy_sha256!r}
'''
        return common + constants + _MERGE_SITE_ADAPTER
    if candidate_policy is not None or candidate_policy_sha256 is not None:
        raise ValueError("Merge target adapters do not accept a candidate policy.")
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


def ascertainment_covariates(payload, samples):
    """AUDIT-ONLY confound covariates per segment row (never model features).

    GT tracing is required to LABEL a merge, so size/GT-coverage covariates
    encode label availability; the ascertainment audit correlates them with
    features and model scores to expose confounded "signal". Runtime-owned and
    GT-reading by design — feature fragments never see these values.
    """
    frag = payload["fragments_graph"]
    comp_to_seg = build_comp_to_seg(frag)
    node_components = np.asarray(frag.node_component_id, dtype=np.int64)
    seg_frag_nodes = {}
    for comp, count in zip(*np.unique(node_components, return_counts=True)):
        seg = int(comp_to_seg.get(int(comp), -1))
        if seg > 0:
            seg_frag_nodes[seg] = seg_frag_nodes.get(seg, 0) + int(count)
    gt_label = np.asarray(payload["gt_node_canonical_label"])
    gt_ids, gt_counts = np.unique(gt_label[gt_label != 0], return_counts=True)
    seg_gt_nodes = {int(s): int(c) for s, c in zip(gt_ids, gt_counts)}
    return {
        "segment_fragment_node_count": np.asarray(
            [seg_frag_nodes.get(int(s), 0) for s in samples], dtype=float),
        "segment_gt_node_count": np.asarray(
            [seg_gt_nodes.get(int(s), 0) for s in samples], dtype=float),
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


def ascertainment_covariates(payload, samples):
    """AUDIT-ONLY confound covariates per candidate-pair row (never features).

    GT tracing is required to LABEL a split, so size/GT-coverage covariates
    encode label availability; the ascertainment audit correlates them with
    features and model scores to expose confounded "signal". Runtime-owned and
    GT-reading by design — feature fragments never see these values.
    """
    frag = payload["fragments_graph"]
    comp_to_seg = build_comp_to_seg(frag)
    node_components = np.asarray(frag.node_component_id, dtype=np.int64)
    seg_frag_nodes = {}
    for comp, count in zip(*np.unique(node_components, return_counts=True)):
        seg = int(comp_to_seg.get(int(comp), -1))
        if seg > 0:
            seg_frag_nodes[seg] = seg_frag_nodes.get(seg, 0) + int(count)
    gt_label = np.asarray(payload["gt_node_canonical_label"])
    gt_ids, gt_counts = np.unique(gt_label[gt_label != 0], return_counts=True)
    seg_gt_nodes = {int(s): int(c) for s, c in zip(gt_ids, gt_counts)}

    def _pair_min(counts, sample):
        return float(min(counts.get(int(sample["segment_id_a"]), 0),
                         counts.get(int(sample["segment_id_b"]), 0)))

    return {
        "gap_um": np.asarray([float(s["gap_um"]) for s in samples], dtype=float),
        "min_segment_fragment_node_count": np.asarray(
            [_pair_min(seg_frag_nodes, s) for s in samples], dtype=float),
        "min_segment_gt_node_count": np.asarray(
            [_pair_min(seg_gt_nodes, s) for s in samples], dtype=float),
    }
'''


_MERGE_SITE_ADAPTER = r'''
def _merge_site_node_arrays(payload):
    """Contiguous per-node arrays candidate enumeration reads (GT-free)."""
    frag = payload["fragments_graph"]
    comp_to_seg = build_comp_to_seg(frag)
    n_nodes = int(frag.number_of_nodes())
    xyz = np.asarray(frag.node_xyz, dtype=float)
    node_components = np.asarray(frag.node_component_id, dtype=np.int64)
    if xyz.shape[0] != n_nodes or node_components.shape[0] != n_nodes:
        raise ValueError(
            "merge-site candidate generation requires node_xyz and "
            "node_component_id to align with contiguous graph node ids"
        )
    if n_nodes and (not frag.has_node(0) or not frag.has_node(n_nodes - 1)):
        raise ValueError("merge-site candidate generation requires node ids 0..N-1")
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
    return xyz, node_segments, degrees


def _nms_kept_junctions(xyz, node_segments, degrees):
    """Segment-scoped greedy NMS over degree>=3 nodes.

    Deterministic order is (-degree, node id); a kept junction suppresses only
    SAME-SEGMENT junctions within CANDIDATE_NMS_UM Euclidean, so a spatially
    passing foreign arbor can never delete another segment's candidate. This
    mirrors nms_keep_mask in notebooks/merge_candidate_pool_sweep.py — the
    sweep that selected the embedded policy — and a parity test holds the two
    implementations together.
    """
    junctions = np.flatnonzero((degrees >= 3) & (node_segments > 0))
    kept = np.zeros(len(node_segments), dtype=bool)
    if not len(junctions):
        return junctions, kept
    if CANDIDATE_NMS_UM <= 0:
        kept[junctions] = True
        return junctions, kept
    from scipy.spatial import cKDTree

    order = junctions[np.lexsort((junctions, -degrees[junctions]))]
    position_of = {int(node): i for i, node in enumerate(junctions)}
    tree = cKDTree(xyz[junctions], copy_data=False)
    neighbor_lists = tree.query_ball_point(xyz[order], r=CANDIDATE_NMS_UM)
    suppressed = np.zeros(len(junctions), dtype=bool)
    for node, neighbors in zip(order, neighbor_lists):
        node = int(node)
        if suppressed[position_of[node]]:
            continue
        kept[node] = True
        own_segment = node_segments[node]
        for j in neighbors:
            if node_segments[junctions[j]] == own_segment:
                suppressed[j] = True
    return junctions, kept


def _merge_site_provenance(payload):
    """Snapshot the actual label-source records, never feature inputs.

    The digest is order independent but retains duplicate records. It covers
    segment, XYZ, associated neurons and source tags, not merely a cached hash.
    Legacy untagged sites are geometric; shared evidence counts once in total.
    """
    import copy
    import hashlib
    import json

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
        if not sources.issubset({"geometric_walk", "two_gt_junction"}):
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


def _site_distances_to_kept(payload, xyz, node_segments, kept):
    """GT consumer: geodesic distances between merge sites and kept candidates.

    Snaps each recorded gt_merge_site to its own segment's nearest node
    (within CANDIDATE_SITE_SNAP_MAX_UM), then walks bounded Dijkstra over
    cable length to CANDIDATE_CLAIM_RADIUS_UM. Returns the per-kept-candidate
    minimum distance to any site plus site-side audit counts. Runs AFTER the
    candidate universe is frozen; feature math never sees these values.
    """
    import heapq
    from scipy.spatial import cKDTree

    frag = payload["fragments_graph"]
    sites = list(payload.get("gt_merge_sites") or [])
    candidate_distance = {}
    n_snapped = 0
    site_min_distances = []
    if sites:
        tree = cKDTree(xyz, copy_data=False)
        for site in sites:
            segment_id = int(site["segment_id"])
            point = np.asarray(site["xyz"], dtype=float)
            neighbor_ids = np.asarray(
                tree.query_ball_point(point, r=CANDIDATE_SITE_SNAP_MAX_UM),
                dtype=np.int64,
            )
            own = neighbor_ids[node_segments[neighbor_ids] == segment_id] \
                if len(neighbor_ids) else neighbor_ids
            if not len(own):
                site_min_distances.append(float("inf"))
                continue
            start = int(own[int(np.argmin(
                np.linalg.norm(xyz[own] - point, axis=1)))])
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
                for neighbor in frag.neighbors(node):
                    neighbor = int(neighbor)
                    step = distance + float(
                        np.linalg.norm(xyz[neighbor] - xyz[node]))
                    if (step <= CANDIDATE_CLAIM_RADIUS_UM
                            and step < best.get(neighbor, float("inf"))):
                        best[neighbor] = step
                        heapq.heappush(heap, (step, neighbor))
            site_min_distances.append(site_best)
    return candidate_distance, {
        "n_sites": len(sites),
        "n_sites_snapped": n_snapped,
        "site_min_distances": site_min_distances,
        "site_provenance": _merge_site_provenance(payload),
    }


def build_sample_universe(payload):
    """One deterministic row per NMS-surviving junction candidate.

    Candidate enumeration is completely determined by the frozen build-time
    policy embedded above and reads fragment geometry only; GT (gt_merge_sites)
    is consulted only after the universe is frozen, to attach is_merge_site
    (geodesic distance <= CANDIDATE_POSITIVE_LABEL_RADIUS_UM) and the
    audit-only distance/ring fields.
    """
    cached = payload.get("__detector_sample_universe_cache__")
    if cached is not None:
        return cached
    xyz, node_segments, degrees = _merge_site_node_arrays(payload)
    junctions, kept = _nms_kept_junctions(xyz, node_segments, degrees)
    kept_nodes = np.flatnonzero(kept)
    candidate_distance, site_audit = _site_distances_to_kept(
        payload, xyz, node_segments, kept)
    samples = []
    labels = []
    for candidate_index, node in enumerate(int(n) for n in kept_nodes):
        distance = candidate_distance.get(node, float("inf"))
        positive = distance <= CANDIDATE_POSITIVE_LABEL_RADIUS_UM
        samples.append({
            "candidate_id": int(candidate_index),
            "node_id": int(node),
            "segment_id": int(node_segments[node]),
            "degree": int(degrees[node]),
            "x_um": float(xyz[node][0]),
            "y_um": float(xyz[node][1]),
            "z_um": float(xyz[node][2]),
            "distance_to_nearest_gt_site_um": (
                float(distance) if np.isfinite(distance) else float("nan")),
            "in_ambiguous_ring": int(np.isfinite(distance) and not positive),
        })
        labels.append(int(positive))
    payload["__detector_universe_site_audit__"] = {
        **site_audit, "n_junctions_raw": int(len(junctions)),
    }
    result = (samples, np.asarray(labels, dtype=np.int64))
    payload["__detector_sample_universe_cache__"] = result
    return result


def sample_output_frame(samples, labels):
    import pandas as pd
    columns = [
        "candidate_id", "node_id", "segment_id", "degree",
        "x_um", "y_um", "z_um",
        "distance_to_nearest_gt_site_um", "in_ambiguous_ring",
    ]
    frame = pd.DataFrame(
        [{key: sample[key] for key in columns} for sample in samples],
        columns=columns,
    )
    frame[LABEL_NAME] = np.asarray(labels, dtype=np.int64)
    return frame


def sample_display(sample):
    return "node %s (segment %s, degree %s)" % (
        sample["node_id"], sample["segment_id"], sample["degree"])


def sample_universe_audit(payload, samples, labels):
    build_sample_universe(payload)
    site_audit = dict(payload.get("__detector_universe_site_audit__") or {})
    site_min = np.asarray(
        site_audit.pop("site_min_distances", []), dtype=float)
    n_sites = int(site_audit.get("n_sites", 0))
    n_covered = int(np.sum(site_min <= CANDIDATE_CLAIM_RADIUS_UM)) if n_sites else 0
    labels_array = np.asarray(labels)
    return {
        "row_unit": ROW_UNIT,
        "candidate_policy_config_id": CANDIDATE_POLICY_CONFIG_ID,
        "candidate_policy_sha256": CANDIDATE_POLICY_SHA256,
        "candidate_mode": CANDIDATE_MODE,
        "candidate_nms_um": CANDIDATE_NMS_UM,
        "candidate_claim_radius_um": CANDIDATE_CLAIM_RADIUS_UM,
        "candidate_positive_label_radius_um": CANDIDATE_POSITIVE_LABEL_RADIUS_UM,
        "candidate_site_snap_max_um": CANDIDATE_SITE_SNAP_MAX_UM,
        "n_rows": int(len(samples)),
        "n_positive": int(np.sum(labels_array == 1)),
        "n_negative": int(np.sum(labels_array == 0)),
        "n_ambiguous_ring": int(sum(
            int(sample["in_ambiguous_ring"]) for sample in samples)),
        "n_junctions_raw": int(site_audit.get("n_junctions_raw", 0)),
        "n_sites": n_sites,
        "site_provenance": site_audit.get("site_provenance"),
        "n_sites_snapped": int(site_audit.get("n_sites_snapped", 0)),
        "n_sites_covered_at_claim_radius": n_covered,
        "site_recall_at_claim_radius": (
            float(n_covered / n_sites) if n_sites else None),
    }


def ascertainment_covariates(payload, samples):
    """AUDIT-ONLY confound covariates per candidate row (never model features).

    GT tracing is required to LABEL a merge site, so size/GT-coverage
    covariates encode label availability; the ascertainment audit correlates
    them with features and model scores to expose confounded "signal".
    Runtime-owned and GT-reading by design — feature fragments never see these.
    """
    frag = payload["fragments_graph"]
    comp_to_seg = build_comp_to_seg(frag)
    node_components = np.asarray(frag.node_component_id, dtype=np.int64)
    seg_frag_nodes = {}
    for comp, count in zip(*np.unique(node_components, return_counts=True)):
        seg = int(comp_to_seg.get(int(comp), -1))
        if seg > 0:
            seg_frag_nodes[seg] = seg_frag_nodes.get(seg, 0) + int(count)
    gt_label = np.asarray(payload["gt_node_canonical_label"])
    gt_ids, gt_counts = np.unique(gt_label[gt_label != 0], return_counts=True)
    seg_gt_nodes = {int(s): int(c) for s, c in zip(gt_ids, gt_counts)}
    return {
        "segment_fragment_node_count": np.asarray(
            [seg_frag_nodes.get(int(s["segment_id"]), 0) for s in samples],
            dtype=float),
        "segment_gt_node_count": np.asarray(
            [seg_gt_nodes.get(int(s["segment_id"]), 0) for s in samples],
            dtype=float),
    }
'''

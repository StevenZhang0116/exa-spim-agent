"""Reviewed target adapters injected into the shared detector runtime.

The feature-writing agent owns feature math only.  Candidate identity and
scientific labels are driver-owned so a generated fragment cannot accidentally
train a split detector on merge labels (or leak GT-only split metadata into X).

Split and merge-site candidates are enumerated and labeled by ``candidate_rules.py``
(2026-10-16): assembly copies that module unchanged beside the detector and embeds the
declared rule and the module's SHA-256. The adapters below only load it and keep the
runtime-owned names that the rest of the runtime and the blindness gates refer to.
"""

from __future__ import annotations

from .contracts import DetectorTarget, target_spec


TARGET_ADAPTER_MARKER = "# __DETECTOR_TARGET_ADAPTER__"


def target_adapter_source(
    target: DetectorTarget,
    *,
    candidate_policy: dict | None = None,
    candidate_policy_sha256: str | None = None,
    candidate_rules_sha256: str | None = None,
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
    if spec.target not in (DetectorTarget.SPLIT, DetectorTarget.MERGE_SITE):
        if (candidate_policy is not None or candidate_policy_sha256 is not None
                or candidate_rules_sha256 is not None):
            raise ValueError("Merge target adapters do not accept a candidate policy.")
        return common + _MERGE_ADAPTER
    if (candidate_policy is None or candidate_policy_sha256 is None
            or candidate_rules_sha256 is None):
        raise ValueError(
            "Split and merge-site target adapters require the frozen declared candidate "
            "rule, its SHA-256 and the SHA-256 of candidate_rules.py."
        )
    rule = dict(candidate_policy)
    shared = f'''CANDIDATE_POLICY_CONFIG_ID = {rule["config_id"]!r}
CANDIDATE_RULE = {rule!r}
CANDIDATE_POLICY_SHA256 = {candidate_policy_sha256!r}
CANDIDATE_RULES_SHA256 = {candidate_rules_sha256!r}
'''
    if spec.target is DetectorTarget.SPLIT:
        constants = f'''CANDIDATE_RULE_KIND = "split"
CANDIDATE_PAIRING_RULE = {rule["mode"]!r}
CANDIDATE_MAX_DISTANCE_UM = {float(rule["radius_um"])!r}
CANDIDATE_PER_ANCHOR_K = {rule["per_anchor_k"]!r}
CANDIDATE_GLOBAL_CAP = {rule["global_cap"]!r}
'''
        return common + shared + constants + _RULES_LOADER + _SPLIT_ADAPTER
    constants = f'''CANDIDATE_RULE_KIND = "merge"
CANDIDATE_MODE = {rule["mode"]!r}
CANDIDATE_NMS_UM = {float(rule["nms_um"])!r}
CANDIDATE_TIP_GAP_UM = {rule["tip_gap_um"]!r}
CANDIDATE_TIP_PARTNER_K = {rule["tip_partner_k"]!r}
CANDIDATE_CLAIM_RADIUS_UM = {float(rule["claim_radius_um"])!r}
CANDIDATE_POSITIVE_LABEL_RADIUS_UM = {float(rule["positive_label_radius_um"])!r}
CANDIDATE_SITE_SNAP_MAX_UM = 50.0
'''
    return common + shared + constants + _RULES_LOADER + _MERGE_SITE_ADAPTER


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


_RULES_LOADER = r'''
CANDIDATE_RULES_FILENAME = "candidate_rules.py"
_CANDIDATE_RULES_MODULE = None


def _candidate_rules():
    """candidate_rules.py beside this detector, checked against CANDIDATE_RULES_SHA256."""
    global _CANDIDATE_RULES_MODULE
    if _CANDIDATE_RULES_MODULE is None:
        import hashlib
        import importlib.util
        import os
        import sys
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            CANDIDATE_RULES_FILENAME)
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        if digest != CANDIDATE_RULES_SHA256:
            raise RuntimeError(
                "%s does not match CANDIDATE_RULES_SHA256 of this detector" % path)
        name = "_detector_candidate_rules_" + digest[:16]
        module = sys.modules.get(name)
        if module is None:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        if module.validate_rule(CANDIDATE_RULE_KIND, CANDIDATE_RULE) != CANDIDATE_RULE:
            raise RuntimeError("The embedded CANDIDATE_RULE is not a normalised rule")
        _CANDIDATE_RULES_MODULE = module
    return _CANDIDATE_RULES_MODULE
'''


_SPLIT_ADAPTER = r'''
def _gt_neuron_membership(gt):
    """Stable GT-neuron key (connected-component index)."""
    return _candidate_rules().gt_neuron_membership(gt)


def _derive_split_truth(payload):
    """Direct and gap positive segment pairs (candidate_rules.split_truth)."""
    return _candidate_rules().split_truth(payload)


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
    """One deterministic row per unordered candidate segment pair.

    Enumeration and is_split labels come from candidate_rules.build_split_universe
    under the embedded declared rule. Candidate construction reads fragment
    geometry only; GT is consulted afterwards to attach is_split.
    """
    cached = payload.get("__detector_sample_universe_cache__")
    if cached is not None:
        return cached
    result = _candidate_rules().build_split_universe(payload, CANDIDATE_RULE)
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
    return {
        "row_unit": ROW_UNIT,
        "candidate_policy_config_id": CANDIDATE_POLICY_CONFIG_ID,
        "candidate_policy_sha256": CANDIDATE_POLICY_SHA256,
        "candidate_rules_sha256": CANDIDATE_RULES_SHA256,
        "candidate_pairing_rule": CANDIDATE_PAIRING_RULE,
        "candidate_max_distance_um": CANDIDATE_MAX_DISTANCE_UM,
        "candidate_per_anchor_k": CANDIDATE_PER_ANCHOR_K,
        "candidate_global_cap": CANDIDATE_GLOBAL_CAP,
        **_candidate_rules().split_audit(payload, samples, labels),
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
    return _candidate_rules().node_arrays(payload["fragments_graph"])


def _nms_kept_junctions(xyz, node_segments, degrees):
    """Segment-scoped greedy NMS over degree>=3 nodes (candidate_rules.kept_junctions)."""
    return _candidate_rules().kept_junctions(xyz, node_segments, degrees, CANDIDATE_NMS_UM)


def _merge_site_provenance(payload):
    """Snapshot of the label-source records (candidate_rules.merge_site_provenance)."""
    return _candidate_rules().merge_site_provenance(payload)


def _site_distances_to_kept(payload, xyz, node_segments, kept):
    """GT consumer: cable distances between merge sites and kept candidates."""
    return _candidate_rules().site_distances_to_kept(
        payload, xyz, node_segments, kept, CANDIDATE_CLAIM_RADIUS_UM)


def build_sample_universe(payload):
    """One deterministic row per candidate node, in node-id order.

    Enumeration (junctions, plus tip-to-shaft candidates under
    junction+tip_to_shaft) and is_merge_site labels come from
    candidate_rules.build_merge_universe under the embedded declared rule. It
    reads fragment geometry only; gt_merge_sites is consulted after the universe
    is frozen, to attach is_merge_site (geodesic distance <=
    CANDIDATE_POSITIVE_LABEL_RADIUS_UM) and the audit-only distance/ring fields.
    """
    cached = payload.get("__detector_sample_universe_cache__")
    if cached is not None:
        return cached
    samples, labels, audit = _candidate_rules().build_merge_universe(payload, CANDIDATE_RULE)
    payload["__detector_universe_site_audit__"] = audit
    result = (samples, labels)
    payload["__detector_sample_universe_cache__"] = result
    return result


def sample_output_frame(samples, labels):
    import pandas as pd
    columns = [
        "candidate_id", "node_id", "segment_id", "degree",
        "x_um", "y_um", "z_um", "candidate_family", "tip_gap_um",
        "distance_to_nearest_gt_site_um", "in_ambiguous_ring",
    ]
    frame = pd.DataFrame(
        [{key: sample[key] for key in columns} for sample in samples],
        columns=columns,
    )
    frame["tip_gap_um"] = pd.to_numeric(frame["tip_gap_um"], errors="coerce")
    frame[LABEL_NAME] = np.asarray(labels, dtype=np.int64)
    return frame


def sample_display(sample):
    return "node %s (segment %s, degree %s)" % (
        sample["node_id"], sample["segment_id"], sample["degree"])


def sample_universe_audit(payload, samples, labels):
    build_sample_universe(payload)
    audit = dict(payload.get("__detector_universe_site_audit__") or {})
    return {
        "row_unit": ROW_UNIT,
        "candidate_policy_config_id": CANDIDATE_POLICY_CONFIG_ID,
        "candidate_policy_sha256": CANDIDATE_POLICY_SHA256,
        "candidate_rules_sha256": CANDIDATE_RULES_SHA256,
        "candidate_mode": CANDIDATE_MODE,
        "candidate_nms_um": CANDIDATE_NMS_UM,
        "candidate_tip_gap_um": CANDIDATE_TIP_GAP_UM,
        "candidate_tip_partner_k": CANDIDATE_TIP_PARTNER_K,
        "candidate_claim_radius_um": CANDIDATE_CLAIM_RADIUS_UM,
        "candidate_positive_label_radius_um": CANDIDATE_POSITIVE_LABEL_RADIUS_UM,
        "candidate_site_snap_max_um": CANDIDATE_SITE_SNAP_MAX_UM,
        **_candidate_rules().merge_audit(audit, samples, labels, CANDIDATE_RULE),
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

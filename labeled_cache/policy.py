"""Merge-threshold policies of the labeled caches.

The GT node-count thresholds of the merge rules live in
``agentic_neuron_proofreader.data_modules.canonical_labeling.MergeThresholds``; a
policy names how they are set for a cache and is recorded in the cache payload as
``gt_label_policy`` (the ``MergeThresholds`` record plus ``reused_geometric_sites``
and ``written``). Caches written before the record existed carry no key and were
built with the module constants, which is the ``original`` policy.
"""

from __future__ import annotations

from agentic_neuron_proofreader.data_modules.canonical_labeling import (
    MERGE_MIN_NODES, MERGE_PASSTHRU_MIN_CC, MergeThresholds,
)

POLICY_ORIGINAL = "original"      # 50 GT nodes everywhere (canonical count, about 200 um in a cache)
POLICY_LENGTH_V1 = "length_v1"    # one cable length (default 50 um) in the cache's own node spacing
POLICIES = (POLICY_ORIGINAL, POLICY_LENGTH_V1)
DEFAULT_LENGTH_UM = 50.0


def thresholds_for(policy, gt_graph=None, length_um=DEFAULT_LENGTH_UM) -> MergeThresholds:
    """The ``MergeThresholds`` a policy means for ``gt_graph`` (needed by ``length_v1``)."""
    if policy == POLICY_ORIGINAL:
        return MergeThresholds.nodes(MERGE_MIN_NODES)
    if policy == POLICY_LENGTH_V1:
        if gt_graph is None:
            raise ValueError("length_v1 needs the GT graph to measure its node spacing")
        return MergeThresholds.from_length(gt_graph, length_um)
    raise ValueError(f"Unknown policy {policy!r}; choose from {POLICIES}")


def policy_matches(record, policy, length_um=DEFAULT_LENGTH_UM) -> bool:
    """Whether a cache's ``gt_label_policy`` record already is the requested policy."""
    if policy == POLICY_ORIGINAL:
        if record is None:
            return True  # written before the record existed: module constants
        return (record.get("policy") == "nodes"
                and int(record.get("min_nodes", -1)) == MERGE_MIN_NODES
                and int(record.get("passthru_min_cc", -1)) == MERGE_PASSTHRU_MIN_CC)
    if policy == POLICY_LENGTH_V1:
        return (record is not None and record.get("policy") == "length"
                and float(record.get("length_um", -1.0)) == float(length_um)
                and not record.get("reused_geometric_sites", False))
    raise ValueError(f"Unknown policy {policy!r}; choose from {POLICIES}")


def describe(policy, length_um=DEFAULT_LENGTH_UM) -> str:
    if policy == POLICY_ORIGINAL:
        return f"{POLICY_ORIGINAL}: {MERGE_MIN_NODES} GT nodes for every merge threshold"
    return f"{POLICY_LENGTH_V1}: {length_um:g} um of GT cable, in each cache's own node spacing"

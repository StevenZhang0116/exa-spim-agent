"""Merge-label policies of the labeled caches.

The GT node-count thresholds of the merge rules live in
``agentic_neuron_proofreader.data_modules.canonical_labeling.MergeThresholds``, and the
geometric merge walk has two forms there (``WALKS``). A policy names both for a cache and
is recorded in the cache payload as ``gt_label_policy`` (the ``MergeThresholds`` record plus
``reused_geometric_sites``, ``walk`` and ``written``). Caches written before the record
existed carry no key and were built with the module constants, which is the ``original``
policy; records written before 2026-11-11 carry no ``walk`` and used the pooled walk.

``length_v2`` (2026-11-11) keeps the thresholds of ``length_v1`` and changes the walk to the
metric's: one GT neuron at a time, the site moved to a branching node. The pooled walk of
``length_v1`` left 90 of 537 canonical merge sites on the six TRAIN brains without a positive
candidate; with the metric's walk 41 remain (``notebooks/merge_junction_study/``). The sites
that segmentation-skeleton-metrics writes to ``metrics_out`` are not a label source (user
decision 2026-11-11): labels come from the cache alone, and those 41 are a known residual.
"""

from __future__ import annotations

from agentic_neuron_proofreader.data_modules.canonical_labeling import (
    MERGE_MIN_NODES, MERGE_PASSTHRU_MIN_CC, WALK_PER_NEURON, WALK_POOLED, MergeThresholds,
)

POLICY_ORIGINAL = "original"      # 50 GT nodes everywhere (canonical count, about 200 um in a cache)
POLICY_LENGTH_V1 = "length_v1"    # one cable length (default 50 um) in the cache's own node spacing
POLICY_LENGTH_V2 = "length_v2"    # the same thresholds, merge walk one GT neuron at a time (the metric's)
POLICIES = (POLICY_ORIGINAL, POLICY_LENGTH_V1, POLICY_LENGTH_V2)
DEFAULT_POLICY = POLICY_LENGTH_V2
DEFAULT_LENGTH_UM = 50.0


def walk_for(policy) -> str:
    """The geometric merge walk a policy means."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy!r}; choose from {POLICIES}")
    return WALK_PER_NEURON if policy == POLICY_LENGTH_V2 else WALK_POOLED


def recorded_walk(record) -> str:
    """The walk a cache was labeled with (records before 2026-11-11 carry no key: pooled)."""
    return WALK_POOLED if record is None else record.get("walk", WALK_POOLED)


def recorded_thresholds(record) -> MergeThresholds:
    """The ``MergeThresholds`` a cache was labeled with (no record: the module constants)."""
    if record is None:
        return MergeThresholds.nodes(MERGE_MIN_NODES)
    return MergeThresholds(min_nodes=int(record["min_nodes"]), passthru_min_cc=int(record["passthru_min_cc"]),
                           policy=record.get("policy", "nodes"), length_um=record.get("length_um"),
                           gt_median_edge_um=record.get("gt_median_edge_um"))


def thresholds_for(policy, gt_graph=None, length_um=DEFAULT_LENGTH_UM) -> MergeThresholds:
    """The ``MergeThresholds`` a policy means for ``gt_graph`` (needed by the length policies)."""
    if policy == POLICY_ORIGINAL:
        return MergeThresholds.nodes(MERGE_MIN_NODES)
    if policy in (POLICY_LENGTH_V1, POLICY_LENGTH_V2):
        if gt_graph is None:
            raise ValueError(f"{policy} needs the GT graph to measure its node spacing")
        return MergeThresholds.from_length(gt_graph, length_um)
    raise ValueError(f"Unknown policy {policy!r}; choose from {POLICIES}")


def policy_matches(record, policy, length_um=DEFAULT_LENGTH_UM) -> bool:
    """Whether a cache's ``gt_label_policy`` record already is the requested policy."""
    if policy == POLICY_ORIGINAL:
        if record is None:
            return True  # written before the record existed: module constants
        return (record.get("policy") == "nodes"
                and int(record.get("min_nodes", -1)) == MERGE_MIN_NODES
                and int(record.get("passthru_min_cc", -1)) == MERGE_PASSTHRU_MIN_CC
                and recorded_walk(record) == WALK_POOLED)
    if policy in (POLICY_LENGTH_V1, POLICY_LENGTH_V2):
        return (record is not None and record.get("policy") == "length"
                and float(record.get("length_um", -1.0)) == float(length_um)
                and not record.get("reused_geometric_sites", False)
                and recorded_walk(record) == walk_for(policy))
    raise ValueError(f"Unknown policy {policy!r}; choose from {POLICIES}")


def describe(policy, length_um=DEFAULT_LENGTH_UM) -> str:
    if policy == POLICY_ORIGINAL:
        return f"{POLICY_ORIGINAL}: {MERGE_MIN_NODES} GT nodes for every merge threshold"
    walk = "merge walk one GT neuron at a time" if policy == POLICY_LENGTH_V2 else "pooled merge walk"
    return f"{policy}: {length_um:g} um of GT cable, in each cache's own node spacing; {walk}"

#!/usr/bin/env python
# %% [markdown]
"""
# Split candidate-pool sweep across relabeled datasets

This is a Jupytext/VS Code-style notebook (``# %%`` cells) and a normal CLI.
It measures the *enumeration ceiling before feature scoring* for a grid of cheap
candidate rules:

* endpoint pairing rule: ``tip_to_tip``, ``tip_to_any_node`` (including
  tip-to-shaft), or opt-in ``any_node_to_any_node`` (including shaft-to-shaft),
* maximum Euclidean gap (default sweep: 5--100 µm in 5 µm increments),
* per-anchor-node quota on distinct partner segment labels.

All distinct segment pairs surviving those rules are retained. There is no
post-enumeration global candidate cap.

The ground-truth and ``reachable`` definitions intentionally match the generated
``split_site_detector.py`` used by the AutoDiscovery run:

* direct truth: distinct non-zero canonical labels across ``gt_edge_error == 1``;
* gap truth: every distinct non-zero flank pair around a connected zero-label GT
  region; and
* reachable: both truth segment ids occur anywhere in ``fragments_graph``.

Outputs are checkpointed per brain, then aggregated into cross-dataset tables and
recall-versus-cost plots. Each scientific configuration gets its own
``notebooks/split_candidate_pool_sweep_outputs/mcl<MCL>_<signature>/`` artifact
bundle with tables, figures, a report, provenance, and SHA-256 manifest. No image
volume or network access is used.

Examples
--------
Run the cheap synthetic test first::

    python notebooks/split_candidate_pool_sweep.py --self-test

Run one brain::

    python notebooks/split_candidate_pool_sweep.py --brains 794495 --mcl 100

Run the expensive all-node rule first at a small maximum radius::

    python notebooks/split_candidate_pool_sweep.py --brains 794495 --mcl 100 \
        --pairing-rules any_node_to_any_node --radii 5 10 --per-anchor-k 1 2 \
        --cpus 8

For ``any_node_to_any_node``, ``--cpus K`` uses K forked processes. Each worker
keeps a local pair map, writes a sorted temporary shard, and the parent performs
a deterministic minimum-gap merge. Temporary shards are deleted automatically;
set ``SPLIT_CANDIDATE_TMPDIR`` to place them on a larger local scratch disk.
For the two tip-anchored rules, the same CPU budget is passed to cKDTree threads.

Run every available ``*_add.pkl`` dataset sequentially::

    python notebooks/split_candidate_pool_sweep.py --mcl 100

The five current mcl100 caches are large (roughly 0.6--4.1 GB each). Run the full
sweep on a compute node with ample RAM. Per-brain CSVs let an interrupted sweep
resume without repeating completed brains.
"""

# %% Imports and configuration
from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import json
import multiprocessing as mp
import os
import pickle
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Headless compute nodes often expose an unwritable user cache. Keep plotting
# caches in /tmp without modifying HOME or any user configuration.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "exaspim-matplotlib-cache")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "exaspim-xdg-cache")
)


THIS_FILE = Path(__file__).resolve()
REPO = THIS_FILE.parents[1]  # .../exa-spim-agent/exa-spim-agent
WORKSPACE = REPO.parent
CACHE_DIR = REPO / "cache"
DEFAULT_OUTPUT_DIR = REPO / "notebooks" / "split_candidate_pool_sweep_outputs"

# A "tip" is a degree-1 graph node (historically called a leaf). "Any node"
# means any sampled graph node belonging to a different valid segment; it does
# not mean arbitrary continuous coordinates or unrestricted node-to-node pairs.
TIP_TO_TIP = "tip_to_tip"
TIP_TO_ANY_NODE = "tip_to_any_node"
ANY_NODE_TO_ANY_NODE = "any_node_to_any_node"
DEFAULT_PAIRING_RULES = (TIP_TO_TIP, TIP_TO_ANY_NODE)
PAIRING_RULES = DEFAULT_PAIRING_RULES + (ANY_NODE_TO_ANY_NODE,)
LEGACY_PAIRING_RULE_ALIASES = {
    "leaf_leaf": TIP_TO_TIP,
    "leaf_any": TIP_TO_ANY_NODE,
}
DEFAULT_RADII_UM = tuple(float(radius) for radius in range(5, 101, 5))

# Register SkeletonGraph classes before pickle.load.
sys.path.insert(0, str(WORKSPACE / "agentic-neuron-proofreader" / "src"))
import agentic_neuron_proofreader  # noqa: F401


@dataclass(frozen=True)
class SweepConfig:
    """Experiment grid shared by every dataset."""

    radii_um: tuple[float, ...] = DEFAULT_RADII_UM
    per_anchor_k: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    modes: tuple[str, ...] = DEFAULT_PAIRING_RULES
    # tip_to_tip also gets an unbounded-per-anchor baseline. This reproduces the
    # detector's current tip-to-tip/all-within-30-um proposal rule.
    include_unbounded_tip_to_tip: bool = True
    anchor_batch_size: int = 2_000
    cpus: int = 1

    def validate(self) -> None:
        if not self.radii_um or min(self.radii_um) <= 0:
            raise ValueError("radii_um must contain positive values")
        if tuple(sorted(set(self.radii_um))) != self.radii_um:
            raise ValueError("radii_um must be sorted and unique")
        if not self.per_anchor_k or min(self.per_anchor_k) < 1:
            raise ValueError("per_anchor_k must contain positive integers")
        if tuple(sorted(set(self.per_anchor_k))) != self.per_anchor_k:
            raise ValueError("per_anchor_k must be sorted and unique")
        bad_modes = set(self.modes) - set(PAIRING_RULES)
        if bad_modes:
            raise ValueError(f"unsupported modes: {sorted(bad_modes)}")
        if self.anchor_batch_size < 1:
            raise ValueError("anchor_batch_size must be positive")
        if self.cpus < 1:
            raise ValueError("cpus must be a positive integer")

    @property
    def max_radius_um(self) -> float:
        return float(max(self.radii_um))

    @property
    def max_k(self) -> int:
        return int(max(self.per_anchor_k))

    def signature(self) -> str:
        # Batch size and CPU count change runtime, not scientific results, so
        # they must not split otherwise-identical artifact bundles/checkpoints.
        scientific_config = asdict(self)
        scientific_config.pop("anchor_batch_size")
        scientific_config.pop("cpus")
        payload = json.dumps(scientific_config, sort_keys=True, default=list)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# %% Cache loading and ground-truth construction
def discover_add_caches(
    cache_dir: Path, mcl: int, brains: Iterable[str] | None = None
) -> list[Path]:
    """Find one relabeled ``*_add.pkl`` cache per requested brain."""

    requested = {str(x) for x in brains} if brains else None
    pattern = re.compile(rf"dataset_cache_(.+)_mcl{int(mcl)}_add\.pkl$")
    found = []
    for path in sorted(cache_dir.glob(f"dataset_cache_*_mcl{int(mcl)}_add.pkl")):
        match = pattern.match(path.name)
        if match and (requested is None or match.group(1) in requested):
            found.append(path)
    if requested:
        seen = {pattern.match(p.name).group(1) for p in found}
        missing = requested - seen
        if missing:
            raise FileNotFoundError(
                f"no mcl{mcl} _add cache for brain(s): {sorted(missing)}"
            )
    if not found:
        raise FileNotFoundError(f"no *_mcl{mcl}_add.pkl caches under {cache_dir}")
    return found


def brain_id_from_cache(path: Path) -> str:
    match = re.match(r"dataset_cache_(.+)_mcl\d+_add\.pkl$", path.name)
    if not match:
        raise ValueError(f"unexpected cache name: {path.name}")
    return match.group(1)


def load_payload(path: Path) -> dict:
    started = time.monotonic()
    print(f"loading {path} ({path.stat().st_size / 2**30:.2f} GiB) ...", flush=True)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    print(f"loaded in {time.monotonic() - started:.1f}s", flush=True)
    return payload


def gt_neuron_membership(gt: nx.Graph) -> dict[int, int]:
    membership = {}
    for index, nodes in enumerate(nx.connected_components(gt)):
        for node in nodes:
            membership[int(node)] = int(index)
    return membership


def derive_split_truth(payload: dict) -> dict[tuple[int, tuple[int, int]], str]:
    """Exact copy of the detector's direct/gap truth contract."""

    gt = payload["gt_graph"]
    labels = np.asarray(payload["gt_node_canonical_label"])
    edge_error = np.asarray(payload["gt_edge_error"])
    neuron_of = gt_neuron_membership(gt)
    truth: dict[tuple[int, tuple[int, int]], str] = {}

    def add(neuron: int, a: int, b: int, kind: str) -> None:
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
            f"gt_edge_error length {len(edge_error)} != gt edge count {len(edges)}"
        )
    for index, (u, v) in enumerate(edges):
        if int(edge_error[index]) == 1:
            add(neuron_of[int(u)], labels[int(u)], labels[int(v)], "direct")

    zero_nodes = [int(n) for n in gt.nodes if int(labels[int(n)]) == 0]
    for component in nx.connected_components(gt.subgraph(zero_nodes)):
        flanks = sorted(
            {
                int(labels[int(neighbor)])
                for node in component
                for neighbor in gt.neighbors(node)
                if int(labels[int(neighbor)]) != 0
            }
        )
        if len(flanks) < 2:
            continue
        neuron = neuron_of[int(next(iter(component)))]
        for i, a in enumerate(flanks):
            for b in flanks[i + 1 :]:
                add(neuron, a, b, "gap")
    return truth


def component_to_segment(frag) -> dict[int, int]:
    return {
        int(component): int(str(swc_id).split(".")[0])
        for component, swc_id in frag.component_id_to_swc_id.items()
    }


def dense_component_segment_map(frag, mapping: dict[int, int]) -> np.ndarray:
    node_components = np.asarray(frag.node_component_id)
    max_component = max(int(node_components.max(initial=0)), max(mapping, default=0))
    # Component ids in these caches are compact. Refuse an unexpectedly sparse id
    # domain instead of silently allocating hundreds of GB.
    if max_component > max(10_000_000, 20 * max(len(mapping), 1)):
        raise MemoryError(
            "component ids are unexpectedly sparse; replace the dense lookup with "
            "a sorted/searchsorted lookup for this cache"
        )
    dense = np.full(max_component + 1, -1, dtype=np.int64)
    for component, segment in mapping.items():
        dense[int(component)] = int(segment)
    return dense


# %% Pair enumeration
def _node_arrays(frag):
    """Return aligned arrays, asserting the cache's contiguous-node invariant."""

    xyz = np.asarray(frag.node_xyz)
    node_components = np.asarray(frag.node_component_id)
    n_nodes = int(frag.number_of_nodes())
    if xyz.shape[0] != n_nodes or node_components.shape[0] != n_nodes:
        raise ValueError(
            "this notebook assumes node ids are contiguous 0..N-1, as required by "
            "the detector's frag.node_xyz[node_id] accesses"
        )
    if n_nodes and (not frag.has_node(0) or not frag.has_node(n_nodes - 1)):
        raise ValueError("fragment node ids do not appear to be contiguous 0..N-1")
    return xyz, node_components


def leaf_nodes(frag) -> np.ndarray:
    started = time.monotonic()
    leaves = np.fromiter(
        (int(node) for node, degree in frag.degree() if int(degree) == 1),
        dtype=np.int64,
    )
    print(f"found {len(leaves):,} degree-1 leaves in {time.monotonic() - started:.1f}s")
    return leaves


@dataclass(frozen=True)
class _EnumerationContext:
    """Read-only state shared by serial and forked anchor-shard adapters."""

    xyz: np.ndarray
    node_components: np.ndarray
    component_segment: np.ndarray
    tips: np.ndarray
    tip_segments: np.ndarray
    partner_nodes: np.ndarray | None
    partner_tree: cKDTree
    k_options: tuple[int | None, ...]
    max_k: int
    unbounded_column: int | None
    max_radius: float
    anchor_batch_size: int
    mode: str


# Linux fork workers inherit this read-only context without pickling the graph,
# coordinate arrays, or KD-tree. Workers write only local dictionaries/shards.
_FORK_ENUMERATION_CONTEXT: _EnumerationContext | None = None
_MAX_ANCHORS_PER_PARALLEL_SHARD = 100_000
_MAX_LOCAL_PAIR_ENTRIES_PER_SHARD = 250_000
_MERGE_FAN_IN = 8


def _anchor_count(context: _EnumerationContext) -> int:
    return (
        len(context.xyz) if context.mode == ANY_NODE_TO_ANY_NODE else len(context.tips)
    )


def _accumulate_anchor_range(
    context: _EnumerationContext,
    start: int,
    stop: int,
    *,
    query_workers: int,
    show_progress: bool,
) -> dict[tuple[int, int], np.ndarray]:
    """Compute one anchor range with no shared mutable state."""

    best_by_pair: dict[tuple[int, int], np.ndarray] = {}
    n_anchors = _anchor_count(context)
    anchor_kind = "nodes" if context.mode == ANY_NODE_TO_ANY_NODE else "tips"
    started = time.monotonic()
    for batch_start in range(start, stop, context.anchor_batch_size):
        batch_stop = min(batch_start + context.anchor_batch_size, stop)
        if context.mode == ANY_NODE_TO_ANY_NODE:
            anchor_nodes = np.arange(batch_start, batch_stop, dtype=np.int64)
            anchor_segments = context.component_segment[
                context.node_components[anchor_nodes]
            ]
            valid_anchor = anchor_segments > 0
            anchor_nodes = anchor_nodes[valid_anchor]
            anchor_segments = anchor_segments[valid_anchor]
        else:
            anchor_nodes = context.tips[batch_start:batch_stop]
            anchor_segments = context.tip_segments[batch_start:batch_stop]
        if not len(anchor_nodes):
            continue

        neighbor_lists = context.partner_tree.query_ball_point(
            context.xyz[anchor_nodes],
            r=context.max_radius,
            workers=query_workers,
        )
        for anchor_node, anchor_segment, neighbor_positions in zip(
            anchor_nodes, anchor_segments, neighbor_lists
        ):
            if not neighbor_positions:
                continue
            neighbor_positions = np.asarray(neighbor_positions, dtype=np.int64)
            neighbor_nodes = (
                neighbor_positions
                if context.partner_nodes is None
                else context.partner_nodes[neighbor_positions]
            )
            target_segments = context.component_segment[
                context.node_components[neighbor_nodes]
            ]
            keep = (neighbor_nodes != anchor_node) & (target_segments > 0)
            keep &= target_segments != int(anchor_segment)
            if not np.any(keep):
                continue
            neighbor_nodes = neighbor_nodes[keep]
            target_segments = target_segments[keep]
            distances = np.linalg.norm(
                context.xyz[neighbor_nodes] - context.xyz[int(anchor_node)], axis=1
            )

            # Reduce duplicate nodes to one closest occurrence per partner
            # segment, then rank the distinct partner segments by distance.
            order = np.lexsort((distances, target_segments))
            sorted_segments = target_segments[order]
            sorted_distances = distances[order]
            first = np.r_[True, sorted_segments[1:] != sorted_segments[:-1]]
            nearest_segments = sorted_segments[first]
            nearest_distances = sorted_distances[first]
            rank_order = np.argsort(nearest_distances, kind="stable")
            nearest_segments = nearest_segments[rank_order]
            nearest_distances = nearest_distances[rank_order]

            limit = (
                len(nearest_segments)
                if context.unbounded_column is not None
                else context.max_k
            )
            for zero_rank, (target_segment, gap) in enumerate(
                zip(nearest_segments[:limit], nearest_distances[:limit])
            ):
                rank = zero_rank + 1
                a, b = sorted((int(anchor_segment), int(target_segment)))
                key = (a, b)
                values = best_by_pair.get(key)
                if values is None:
                    values = np.full(len(context.k_options), np.inf, dtype=np.float32)
                    best_by_pair[key] = values
                for column, quota in enumerate(context.k_options):
                    within_quota = quota is None or rank <= int(quota)
                    if within_quota and gap < values[column]:
                        values[column] = float(gap)

        if show_progress and (
            batch_start == start or batch_stop == stop or batch_stop % 100_000 == 0
        ):
            print(
                f"  {context.mode}: {anchor_kind} {batch_stop:,}/{n_anchors:,}; "
                f"pairs {len(best_by_pair):,}; {time.monotonic() - started:.0f}s",
                flush=True,
            )
    return best_by_pair


def _pair_dict_to_sorted_arrays(
    best_by_pair: dict[tuple[int, int], np.ndarray], k_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one local map to deterministic lexicographic pair order."""

    if not best_by_pair:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, k_count), dtype=np.float32),
        )
    keys = sorted(best_by_pair)
    pairs = np.asarray(keys, dtype=np.int64)
    gaps = np.vstack([best_by_pair[key] for key in keys]).astype(np.float32, copy=False)
    return pairs[:, 0], pairs[:, 1], gaps


def _write_any_node_shard(
    task_id: int, start: int, stop: int, temporary_dir: str
) -> tuple[int, str, str, int, int, float]:
    """Fork-worker entry point; persist a sorted shard instead of pickling it."""

    context = _FORK_ENUMERATION_CONTEXT
    if context is None or context.mode != ANY_NODE_TO_ANY_NODE:
        raise RuntimeError("parallel enumeration context was not inherited")
    started = time.monotonic()
    best = _accumulate_anchor_range(
        context, start, stop, query_workers=1, show_progress=False
    )
    pair_a, pair_b, gaps = _pair_dict_to_sorted_arrays(best, len(context.k_options))
    pairs = np.column_stack((pair_a, pair_b))
    root = Path(temporary_dir)
    pairs_path = root / f"shard-{task_id:06d}-pairs.npy"
    gaps_path = root / f"shard-{task_id:06d}-gaps.npy"
    np.save(pairs_path, pairs, allow_pickle=False)
    np.save(gaps_path, gaps, allow_pickle=False)
    return (
        task_id,
        str(pairs_path),
        str(gaps_path),
        stop - start,
        len(pair_a),
        time.monotonic() - started,
    )


def _merge_sorted_pair_shards(
    shard_results: list[tuple[int, str, str, int, int, float]], k_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-pass external k-way merge with elementwise minimum gap reduction."""

    ordered = sorted(shard_results)
    pair_shards = [np.load(item[1], mmap_mode="r") for item in ordered]
    gap_shards = [np.load(item[2], mmap_mode="r") for item in ordered]

    def initial_heap(
        arrays: list[np.ndarray],
    ) -> tuple[list[tuple[int, int, int]], list[int]]:
        positions = [0] * len(arrays)
        heap = [
            (int(pairs[0, 0]), int(pairs[0, 1]), index)
            for index, pairs in enumerate(arrays)
            if len(pairs)
        ]
        heapq.heapify(heap)
        return heap, positions

    heap, positions = initial_heap(pair_shards)
    unique_count = 0
    previous: tuple[int, int] | None = None
    while heap:
        a, b, shard_index = heapq.heappop(heap)
        key = (a, b)
        if key != previous:
            unique_count += 1
            previous = key
        positions[shard_index] += 1
        position = positions[shard_index]
        if position < len(pair_shards[shard_index]):
            pair = pair_shards[shard_index][position]
            heapq.heappush(heap, (int(pair[0]), int(pair[1]), shard_index))

    pair_a = np.empty(unique_count, dtype=np.int64)
    pair_b = np.empty(unique_count, dtype=np.int64)
    gaps = np.full((unique_count, k_count), np.inf, dtype=np.float32)
    heap, positions = initial_heap(pair_shards)
    output_index = -1
    current_key: tuple[int, int] | None = None
    while heap:
        a, b, shard_index = heapq.heappop(heap)
        key = (a, b)
        position = positions[shard_index]
        if key != current_key:
            output_index += 1
            current_key = key
            pair_a[output_index] = a
            pair_b[output_index] = b
            gaps[output_index] = gap_shards[shard_index][position]
        else:
            np.minimum(
                gaps[output_index],
                gap_shards[shard_index][position],
                out=gaps[output_index],
            )
        positions[shard_index] += 1
        position = positions[shard_index]
        if position < len(pair_shards[shard_index]):
            pair = pair_shards[shard_index][position]
            heapq.heappush(heap, (int(pair[0]), int(pair[1]), shard_index))

    del pair_shards, gap_shards
    gc.collect()
    return pair_a, pair_b, gaps


def _bounded_merge_pair_shards(
    shard_results: list[tuple[int, str, str, int, int, float]],
    k_count: int,
    temporary_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge in bounded-fan-in rounds so open files do not scale with shards."""

    current = sorted(shard_results)
    generation = 0
    root = Path(temporary_dir)
    while len(current) > _MERGE_FAN_IN:
        next_round = []
        for group_index, offset in enumerate(range(0, len(current), _MERGE_FAN_IN)):
            group = current[offset : offset + _MERGE_FAN_IN]
            if len(group) == 1:
                next_round.append(group[0])
                continue
            started = time.monotonic()
            pair_a, pair_b, gaps = _merge_sorted_pair_shards(group, k_count)
            pairs_path = root / (f"merge-g{generation:03d}-{group_index:06d}-pairs.npy")
            gaps_path = root / (f"merge-g{generation:03d}-{group_index:06d}-gaps.npy")
            np.save(
                pairs_path,
                np.column_stack((pair_a, pair_b)),
                allow_pickle=False,
            )
            np.save(gaps_path, gaps, allow_pickle=False)
            next_round.append(
                (
                    group_index,
                    str(pairs_path),
                    str(gaps_path),
                    sum(item[3] for item in group),
                    len(pair_a),
                    time.monotonic() - started,
                )
            )
            del pair_a, pair_b, gaps
            for item in group:
                Path(item[1]).unlink()
                Path(item[2]).unlink()
        current = next_round
        generation += 1
        print(
            f"  bounded merge round {generation}: {len(current)} shards remain",
            flush=True,
        )
    return _merge_sorted_pair_shards(current, k_count)


def _parallel_any_node_to_any_node(
    context: _EnumerationContext, cpus: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map anchor shards across fork workers and deterministically reduce them."""

    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError(
            "any_node_to_any_node multiprocessing requires the Linux/Unix "
            "'fork' start method so large arrays and the KD-tree stay shared"
        )
    n_anchors = _anchor_count(context)
    ranges = _parallel_anchor_ranges(n_anchors, cpus, context.max_k)
    if not ranges:
        return _pair_dict_to_sorted_arrays({}, len(context.k_options))
    max_workers = min(cpus, len(ranges))
    temp_parent = os.environ.get("SPLIT_CANDIDATE_TMPDIR") or None
    print(
        f"  {context.mode}: parallel backend with {max_workers} processes, "
        f"{len(ranges)} anchor shards (worker KD-tree threads=1)",
        flush=True,
    )

    global _FORK_ENUMERATION_CONTEXT
    _FORK_ENUMERATION_CONTEXT = context
    try:
        with tempfile.TemporaryDirectory(
            prefix="split-candidate-shards-", dir=temp_parent
        ) as temporary_dir:
            executor = ProcessPoolExecutor(
                max_workers=max_workers, mp_context=mp.get_context("fork")
            )
            futures = []
            shard_results = []
            completed_anchors = 0
            try:
                for task_id, start, stop in ranges:
                    futures.append(
                        executor.submit(
                            _write_any_node_shard,
                            task_id,
                            start,
                            stop,
                            temporary_dir,
                        )
                    )
                for future in as_completed(futures):
                    result = future.result()
                    shard_results.append(result)
                    completed_anchors += result[3]
                    print(
                        f"  {context.mode}: parallel anchors "
                        f"{completed_anchors:,}/{n_anchors:,}; "
                        f"shard pairs {result[4]:,}; {result[5]:.1f}s",
                        flush=True,
                    )
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
            print(
                f"  {context.mode}: merging {len(shard_results)} sorted shards",
                flush=True,
            )
            return _bounded_merge_pair_shards(
                shard_results,
                len(context.k_options),
                temporary_dir,
            )
    finally:
        _FORK_ENUMERATION_CONTEXT = None


def _parallel_anchor_ranges(
    n_anchors: int, cpus: int, max_k: int
) -> list[tuple[int, int, int]]:
    """Plan process shards independently from the in-worker query batch size."""

    if n_anchors <= 0:
        return []
    max_workers = min(cpus, n_anchors)
    max_anchors_by_pair_budget = max(
        1, _MAX_LOCAL_PAIR_ENTRIES_PER_SHARD // max(1, max_k)
    )
    max_anchors_per_shard = min(
        _MAX_ANCHORS_PER_PARALLEL_SHARD, max_anchors_by_pair_budget
    )
    target_tasks = min(
        n_anchors,
        max(
            max_workers * 4,
            (n_anchors + max_anchors_per_shard - 1) // max_anchors_per_shard,
        ),
    )
    shard_size = max(1, (n_anchors + target_tasks - 1) // target_tasks)
    return [
        (task_id, start, min(start + shard_size, n_anchors))
        for task_id, start in enumerate(range(0, n_anchors, shard_size))
    ]


def enumerate_pair_min_gaps(
    frag,
    leaves: np.ndarray,
    component_segment: np.ndarray,
    config: SweepConfig,
    mode: str,
) -> tuple[list[int | None], np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate once at max radius and retain pair-minimum gaps for each k.

    ``cpus`` changes only the execution adapter. Scientific outputs are sorted
    and identical for serial and parallel ``any_node_to_any_node`` execution.
    """

    if mode not in PAIRING_RULES:
        raise ValueError(mode)
    xyz, node_components = _node_arrays(frag)
    tip_segments = component_segment[node_components[leaves]]
    valid_tip = tip_segments > 0
    tips = leaves[valid_tip]
    tip_segments = tip_segments[valid_tip]
    if mode == TIP_TO_TIP:
        partner_nodes: np.ndarray | None = tips
        partner_coords = xyz[tips]
    else:
        partner_nodes = None
        partner_coords = xyz
    partner_tree = cKDTree(partner_coords, copy_data=False)

    k_options: list[int | None] = list(config.per_anchor_k)
    if mode == TIP_TO_TIP and config.include_unbounded_tip_to_tip:
        k_options.append(None)
    numeric_k = [int(k) for k in k_options if k is not None]
    context = _EnumerationContext(
        xyz=xyz,
        node_components=node_components,
        component_segment=component_segment,
        tips=tips,
        tip_segments=tip_segments,
        partner_nodes=partner_nodes,
        partner_tree=partner_tree,
        k_options=tuple(k_options),
        max_k=max(numeric_k, default=0),
        unbounded_column=(k_options.index(None) if None in k_options else None),
        max_radius=config.max_radius_um,
        anchor_batch_size=config.anchor_batch_size,
        mode=mode,
    )
    started = time.monotonic()
    if mode == ANY_NODE_TO_ANY_NODE and config.cpus > 1:
        pair_a, pair_b, gaps = _parallel_any_node_to_any_node(context, config.cpus)
    else:
        best = _accumulate_anchor_range(
            context,
            0,
            _anchor_count(context),
            query_workers=config.cpus,
            show_progress=True,
        )
        pair_a, pair_b, gaps = _pair_dict_to_sorted_arrays(best, len(k_options))
    print(
        f"{mode}: {len(pair_a):,} distinct pairs at r<={context.max_radius:g} "
        f"using {config.cpus} CPU(s) in {time.monotonic() - started:.1f}s",
        flush=True,
    )
    return k_options, pair_a, pair_b, gaps


# %% Metrics and per-dataset sweep
def metric_rows_for_mode(
    brain: str,
    mode: str,
    k_options: list[int | None],
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    gaps: np.ndarray,
    reachable_truth: set[tuple[int, int]],
    reachable_truth_by_kind: dict[str, set[tuple[int, int]]],
    config: SweepConfig,
) -> list[dict]:
    truth_mask = np.fromiter(
        ((int(a), int(b)) in reachable_truth for a, b in zip(pair_a, pair_b)),
        dtype=bool,
        count=len(pair_a),
    )
    direct_truth = reachable_truth_by_kind.get("direct", set())
    gap_truth = reachable_truth_by_kind.get("gap", set())
    direct_mask = np.fromiter(
        ((int(a), int(b)) in direct_truth for a, b in zip(pair_a, pair_b)),
        dtype=bool,
        count=len(pair_a),
    )
    gap_mask = np.fromiter(
        ((int(a), int(b)) in gap_truth for a, b in zip(pair_a, pair_b)),
        dtype=bool,
        count=len(pair_a),
    )
    denominator = len(reachable_truth)
    rows = []
    for column, quota in enumerate(k_options):
        pair_gap = gaps[:, column]
        for radius in config.radii_um:
            eligible = np.flatnonzero(np.isfinite(pair_gap) & (pair_gap <= radius))
            n_candidates = len(eligible)
            n_truth = int(np.sum(truth_mask[eligible])) if n_candidates else 0
            n_direct = int(np.sum(direct_mask[eligible])) if n_candidates else 0
            n_gap = int(np.sum(gap_mask[eligible])) if n_candidates else 0
            recall = n_truth / denominator if denominator else float("nan")
            prevalence = n_truth / n_candidates if n_candidates else float("nan")
            quota_label = "all" if quota is None else str(int(quota))
            rows.append(
                {
                    "brain": brain,
                    "mode": mode,
                    "radius_um": float(radius),
                    "per_anchor_k": quota_label,
                    "per_anchor_k_value": -1 if quota is None else int(quota),
                    # Retained as a constant compatibility/audit column for
                    # historical bundles and the AI-review collector.
                    "global_cap": "none",
                    "global_cap_value": -1,
                    "config_id": f"{mode}|r={radius:g}|k={quota_label}",
                    "n_reachable_truth_pairs": denominator,
                    "n_candidate_pairs": n_candidates,
                    "n_candidate_truth_pairs": n_truth,
                    "candidate_recall": recall,
                    "n_reachable_direct_truth_pairs": len(direct_truth),
                    "n_candidate_direct_truth_pairs": n_direct,
                    "direct_candidate_recall": (
                        n_direct / len(direct_truth) if direct_truth else float("nan")
                    ),
                    "n_reachable_gap_truth_pairs": len(gap_truth),
                    "n_candidate_gap_truth_pairs": n_gap,
                    "gap_candidate_recall": (
                        n_gap / len(gap_truth) if gap_truth else float("nan")
                    ),
                    "candidate_prevalence": prevalence,
                    "false_candidate_pairs": n_candidates - n_truth,
                }
            )
    return rows


def analyze_dataset(cache_path: Path, config: SweepConfig) -> tuple[pd.DataFrame, dict]:
    brain = brain_id_from_cache(cache_path)
    payload = load_payload(cache_path)
    frag = payload["fragments_graph"]
    truth_records = derive_split_truth(payload)
    truth_pairs = {pair for _, pair in truth_records}
    comp_to_seg = component_to_segment(frag)
    fragment_segments = set(comp_to_seg.values())
    reachable_truth = {
        pair
        for pair in truth_pairs
        if pair[0] in fragment_segments and pair[1] in fragment_segments
    }
    direct_truth = {
        pair
        for (_neuron, pair), kind in truth_records.items()
        if kind in {"direct", "both"} and pair in reachable_truth
    }
    gap_truth = {
        pair
        for (_neuron, pair), kind in truth_records.items()
        if kind in {"gap", "both"} and pair in reachable_truth
    }
    reachable_truth_by_kind = {"direct": direct_truth, "gap": gap_truth}
    component_segment = dense_component_segment_map(frag, comp_to_seg)
    leaves = leaf_nodes(frag)

    kind_counts: dict[str, int] = defaultdict(int)
    for kind in truth_records.values():
        kind_counts[kind] += 1
    meta = {
        "brain": brain,
        "cache": str(cache_path.resolve()),
        "cache_size_bytes": cache_path.stat().st_size,
        "cache_mtime_ns": cache_path.stat().st_mtime_ns,
        "config_signature": config.signature(),
        "config": asdict(config),
        "n_fragment_nodes": int(frag.number_of_nodes()),
        "n_fragment_components": len(comp_to_seg),
        "n_fragment_segments": len(fragment_segments),
        "n_leaves": len(leaves),
        "n_truth_records": len(truth_records),
        "n_truth_pairs": len(truth_pairs),
        "n_reachable_truth_pairs": len(reachable_truth),
        "n_reachable_direct_truth_pairs": len(direct_truth),
        "n_reachable_gap_truth_pairs": len(gap_truth),
        "truth_kind_counts": dict(kind_counts),
    }
    print(
        f"{brain}: truth records={len(truth_records):,}, pairs={len(truth_pairs):,}, "
        f"reachable={len(reachable_truth):,}",
        flush=True,
    )

    # Fork the process pool before running a threaded cKDTree query in the
    # parent process. Output rows are sorted below, so this runtime-only order
    # does not change the artifact contents.
    execution_modes = list(config.modes)
    if ANY_NODE_TO_ANY_NODE in execution_modes and config.cpus > 1:
        execution_modes.remove(ANY_NODE_TO_ANY_NODE)
        execution_modes.insert(0, ANY_NODE_TO_ANY_NODE)

    rows = []
    meta["pairing_rule_execution"] = []
    for mode in execution_modes:
        mode_started = time.monotonic()
        k_options, pair_a, pair_b, gaps = enumerate_pair_min_gaps(
            frag, leaves, component_segment, config, mode
        )
        rows.extend(
            metric_rows_for_mode(
                brain,
                mode,
                k_options,
                pair_a,
                pair_b,
                gaps,
                reachable_truth,
                reachable_truth_by_kind,
                config,
            )
        )
        parallel_ranges = (
            _parallel_anchor_ranges(
                int(frag.number_of_nodes()), config.cpus, config.max_k
            )
            if mode == ANY_NODE_TO_ANY_NODE and config.cpus > 1
            else []
        )
        meta["pairing_rule_execution"].append(
            {
                "mode": mode,
                "backend": (
                    "fork_processes"
                    if mode == ANY_NODE_TO_ANY_NODE and config.cpus > 1
                    else "ckdtree_threads"
                ),
                "requested_cpus": config.cpus,
                "actual_processes": (
                    min(config.cpus, len(parallel_ranges)) if parallel_ranges else None
                ),
                "n_process_shards": len(parallel_ranges) if parallel_ranges else None,
                "n_distinct_pairs_at_max_grid": len(pair_a),
                "elapsed_seconds": time.monotonic() - mode_started,
            }
        )
        del pair_a, pair_b, gaps
        gc.collect()

    frame = pd.DataFrame(rows).sort_values(
        ["mode", "per_anchor_k_value", "radius_um", "global_cap_value"]
    )
    # Current detector baseline: all tip-to-tip label pairs within 30 um and no
    # per-anchor quota.
    baseline = frame[
        (frame["mode"] == TIP_TO_TIP)
        & (frame["radius_um"] == 30.0)
        & (frame["per_anchor_k"] == "all")
        & (frame["global_cap"] == "none")
    ]
    if len(baseline) == 1:
        row = baseline.iloc[0]
        meta["detector_baseline"] = {
            "n_candidate_pairs": int(row["n_candidate_pairs"]),
            "n_candidate_truth_pairs": int(row["n_candidate_truth_pairs"]),
            "candidate_recall": float(row["candidate_recall"]),
        }
        print(f"{brain} detector baseline: {meta['detector_baseline']}")

    del payload, frag, leaves, component_segment
    gc.collect()
    return frame, meta


# %% Cross-dataset aggregation and Pareto frontier
GROUP_COLUMNS = [
    "config_id",
    "mode",
    "radius_um",
    "per_anchor_k",
    "per_anchor_k_value",
    "global_cap",
    "global_cap_value",
]


def aggregate_results(all_results: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        all_results.groupby(GROUP_COLUMNS, dropna=False)
        .agg(
            datasets=("brain", "nunique"),
            total_candidates=("n_candidate_pairs", "sum"),
            mean_candidates=("n_candidate_pairs", "mean"),
            max_candidates=("n_candidate_pairs", "max"),
            total_truth_found=("n_candidate_truth_pairs", "sum"),
            total_reachable_truth=("n_reachable_truth_pairs", "sum"),
            mean_recall=("candidate_recall", "mean"),
            min_recall=("candidate_recall", "min"),
            max_recall=("candidate_recall", "max"),
            mean_direct_recall=("direct_candidate_recall", "mean"),
            min_direct_recall=("direct_candidate_recall", "min"),
            mean_gap_recall=("gap_candidate_recall", "mean"),
            min_gap_recall=("gap_candidate_recall", "min"),
            mean_prevalence=("candidate_prevalence", "mean"),
        )
        .reset_index()
    )
    aggregate["micro_recall"] = (
        aggregate["total_truth_found"] / aggregate["total_reachable_truth"]
    )
    return aggregate.sort_values(
        ["min_recall", "total_candidates"], ascending=[False, True]
    )


def pareto_frontier(aggregate: pd.DataFrame) -> pd.DataFrame:
    """Keep configs not dominated on total candidate count and worst-brain recall."""

    ordered = aggregate.sort_values(
        ["total_candidates", "min_recall"], ascending=[True, False]
    )
    keep = []
    best_recall = -np.inf
    for index, row in ordered.iterrows():
        if float(row["min_recall"]) > best_recall:
            keep.append(index)
            best_recall = float(row["min_recall"])
    return ordered.loc[keep].sort_values("total_candidates").reset_index(drop=True)


def recommend_configs(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in (0.80, 0.90, 0.95):
        eligible = aggregate[aggregate["min_recall"] >= target]
        if len(eligible):
            best = eligible.sort_values(
                ["total_candidates", "mean_recall"], ascending=[True, False]
            ).iloc[0]
            status = "meets_target"
        else:
            best = aggregate.sort_values(
                ["min_recall", "total_candidates"], ascending=[False, True]
            ).iloc[0]
            status = "target_not_reached_best_available"
        rows.append(
            {
                "target_worst_brain_recall": target,
                "status": status,
                **{column: best[column] for column in GROUP_COLUMNS},
                "min_recall": best["min_recall"],
                "mean_recall": best["mean_recall"],
                "micro_recall": best["micro_recall"],
                "total_candidates": int(best["total_candidates"]),
                "mean_prevalence": best["mean_prevalence"],
            }
        )
    return pd.DataFrame(rows)


def _plot_quotas(frame: pd.DataFrame) -> list[str]:
    preferred = ["all", "1", "4", "16", "32"]
    available = set(frame["per_anchor_k"].astype(str))
    return [value for value in preferred if value in available]


SCATTER_STYLE = {
    "s": 64,
    "alpha": 0.72,
    "edgecolors": "#202020",
    "linewidths": 0.5,
    "zorder": 3,
}


def save_per_dataset_plot(frame: pd.DataFrame, output_path: Path) -> None:
    """Save one compact diagnostic figure for a single brain."""

    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    brain = str(frame["brain"].iloc[0])
    markers = {
        TIP_TO_TIP: "o",
        TIP_TO_ANY_NODE: "^",
        ANY_NODE_TO_ANY_NODE: "s",
    }
    colors = {
        TIP_TO_TIP: "#1f77b4",
        TIP_TO_ANY_NODE: "#d62728",
        ANY_NODE_TO_ANY_NODE: "#2ca02c",
    }
    plotted = frame
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    for mode, group in plotted.groupby("mode"):
        for quota in _plot_quotas(group):
            line = group[group["per_anchor_k"].astype(str) == quota].sort_values(
                "radius_um"
            )
            axes[0].plot(
                line["radius_um"],
                line["candidate_recall"],
                marker=markers[mode],
                markersize=4,
                linewidth=1.2,
                color=colors[mode],
                alpha=1.0 if quota in {"all", "4"} else 0.45,
                label=f"{mode}, k={quota}",
            )
    axes[0].set_xlabel("maximum gap (µm)")
    axes[0].set_ylabel("recall of reachable truth pairs")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Recall versus distance")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)

    for mode, group in frame.groupby("mode"):
        axes[1].scatter(
            group["n_candidate_pairs"],
            group["candidate_recall"],
            marker=markers[mode],
            color=colors[mode],
            label=mode,
            **SCATTER_STYLE,
        )
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set_xlabel("candidate pairs (log-like scale)")
    axes[1].set_ylabel("recall of reachable truth pairs")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Recall versus pool cost")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)

    for mode, group in plotted.groupby("mode"):
        axes[2].scatter(
            group["direct_candidate_recall"],
            group["gap_candidate_recall"],
            marker=markers[mode],
            color=colors[mode],
            label=mode,
            **SCATTER_STYLE,
        )
    axes[2].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[2].set_xlabel("direct-truth recall")
    axes[2].set_ylabel("gap-truth recall")
    axes[2].set_xlim(0, 1.02)
    axes[2].set_ylim(0, 1.02)
    axes[2].set_title("Which truth type is missed?")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=8)

    fig.suptitle(f"Split candidate-pool diagnostics — brain {brain}", fontsize=13)
    fig.tight_layout()
    temporary = output_path.with_name(output_path.name + ".tmp")
    fig.savefig(temporary, dpi=180, bbox_inches="tight", format="png")
    temporary.replace(output_path)
    plt.close(fig)


def save_aggregate_plot(
    all_results: pd.DataFrame, aggregate: pd.DataFrame, output_path: Path
) -> None:
    """Save cross-dataset robust-recall and cost diagnostics."""

    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    markers = {
        TIP_TO_TIP: "o",
        TIP_TO_ANY_NODE: "^",
        ANY_NODE_TO_ANY_NODE: "s",
    }
    colors = {
        TIP_TO_TIP: "#1f77b4",
        TIP_TO_ANY_NODE: "#d62728",
        ANY_NODE_TO_ANY_NODE: "#2ca02c",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    frontier = pareto_frontier(aggregate)
    for mode, group in aggregate.groupby("mode"):
        axes[0].scatter(
            group["total_candidates"],
            group["min_recall"],
            marker=markers[mode],
            color=colors[mode],
            label=mode,
            **SCATTER_STYLE,
        )
    axes[0].plot(
        frontier["total_candidates"],
        frontier["min_recall"],
        color="black",
        linewidth=1.5,
        marker=".",
        label="Pareto frontier",
    )
    axes[0].set_xscale("symlog", linthresh=1)
    axes[0].set_xlabel("total candidates across datasets")
    axes[0].set_ylabel("worst-dataset recall")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Robust recall–cost frontier")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    for mode, group in aggregate.groupby("mode"):
        for quota in _plot_quotas(group):
            line = group[group["per_anchor_k"].astype(str) == quota].sort_values(
                "radius_um"
            )
            axes[1].plot(
                line["radius_um"],
                line["min_recall"],
                marker=markers[mode],
                markersize=4,
                linewidth=1.2,
                color=colors[mode],
                alpha=1.0 if quota in {"all", "4"} else 0.45,
                label=f"{mode}, k={quota}",
            )
    axes[1].set_xlabel("maximum gap (µm)")
    axes[1].set_ylabel("worst-dataset recall")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Robust recall versus distance")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)

    for mode, group in aggregate.groupby("mode"):
        axes[2].scatter(
            group["min_direct_recall"],
            group["min_gap_recall"],
            marker=markers[mode],
            color=colors[mode],
            label=mode,
            **SCATTER_STYLE,
        )
    axes[2].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[2].set_xlabel("worst-dataset direct recall")
    axes[2].set_ylabel("worst-dataset gap recall")
    axes[2].set_xlim(0, 1.02)
    axes[2].set_ylim(0, 1.02)
    axes[2].set_title("Robust recall by truth type")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=8)

    fig.suptitle(
        f"Split candidate-pool sweep — {all_results['brain'].nunique()} dataset(s)",
        fontsize=13,
    )
    fig.tight_layout()
    temporary = output_path.with_name(output_path.name + ".tmp")
    fig.savefig(temporary, dpi=180, bbox_inches="tight", format="png")
    temporary.replace(output_path)
    plt.close(fig)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_artifact_manifest(run_dir: Path) -> Path:
    """Hash every artifact except the manifest itself."""

    manifest_path = run_dir / "artifact_manifest.json"
    entries = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        if path == manifest_path:
            continue
        suffix = path.suffix.lower()
        role = (
            "figure"
            if suffix == ".png"
            else "table"
            if suffix == ".csv"
            else "metadata"
        )
        if suffix == ".md":
            role = "report"
        elif suffix == ".py":
            role = "source"
        entries.append(
            {
                "path": str(path.relative_to(run_dir)),
                "role": role,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    atomic_write_text(
        manifest_path, json.dumps({"artifacts": entries}, indent=2) + "\n"
    )
    return manifest_path


def write_results_report(
    run_dir: Path,
    config: SweepConfig,
    dataset_summary: pd.DataFrame,
    baseline: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> Path:
    """Write a compact human-readable index for the artifact bundle."""

    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        if frame.empty:
            return "(not available)"
        return "```text\n" + frame[columns].to_string(index=False) + "\n```"

    report = f"""# Split candidate-pool sweep artifacts

This directory is one configuration-scoped, cumulatively checkpointed analysis bundle. Its configuration
signature is `{config.signature()}`. Candidate recall is measured against reachable
truth pairs before feature scoring.

## Dataset summary

{table(dataset_summary, ["brain", "n_reachable_truth_pairs", "best_candidate_recall", "fewest_candidates_at_best_recall"])}

## Current detector baseline

The baseline is `tip_to_tip`, 30 µm, and unlimited per-anchor partners. All
surviving segment pairs are retained; the sweep has no global candidate cap.

{table(baseline, ["brain", "n_candidate_pairs", "n_candidate_truth_pairs", "candidate_recall", "direct_candidate_recall", "gap_candidate_recall"])}

## Recommended minimal pools

Selection minimizes total candidate count subject to a worst-dataset recall target.

{table(recommendations, ["target_worst_brain_recall", "status", "config_id", "min_recall", "mean_recall", "total_candidates"])}

## Artifact layout

* `per_dataset/<brain>/sweep.csv` — every configuration for one brain.
* `per_dataset/<brain>/metadata.json` — input/cache and truth-universe audit.
* `per_dataset/<brain>/candidate_pool_diagnostics.png` — distance, cost, and truth-type diagnostics.
* `tables/` — combined, aggregate, Pareto, baseline, summary, and recommendation CSVs.
* `figures/cross_dataset_candidate_pool_diagnostics.png` — robust cross-dataset diagnostics.
* `run_summary.json` — machine-readable run configuration and headline results.
* `provenance/split_candidate_pool_sweep.py` — exact analysis-source snapshot.
* `artifact_manifest.json` — SHA-256 and size for every artifact in this bundle.

The figures are ceilings, not classifier performance. After choosing a proposal rule,
the scoring model must be retrained and evaluated on held-out brains.
"""
    path = run_dir / "RESULTS.md"
    atomic_write_text(path, report)
    return path


# %% Synthetic test
def run_synthetic_self_test() -> None:
    """Exercise all endpoint-pairing rules without loading a real cache."""

    default_config = SweepConfig()
    default_config.validate()
    assert default_config.radii_um == tuple(
        float(radius) for radius in range(5, 101, 5)
    )
    try:
        SweepConfig(radii_um=(0.0, 5.0)).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("zero radius should be rejected")

    graph = nx.Graph()
    graph.add_edges_from(
        [
            (0, 1),
            (2, 3),
            (4, 5),
            (5, 6),
            (7, 8),
            (8, 9),
            (10, 11),
            (11, 12),
        ]
    )
    graph.node_xyz = np.asarray(
        [
            [-100.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 2.0, 0.0],  # nearby shaft on segment 30
            [0.0, 104.0, 0.0],
            [-500.0, 50.0, 0.0],
            [50.0, 50.0, 0.0],  # interior node on segment 40
            [-500.0, 60.0, 0.0],
            [500.0, 50.0, 0.0],
            [52.0, 50.0, 0.0],  # interior node on segment 50; shaft-to-shaft
            [500.0, 60.0, 0.0],
        ],
        dtype=float,
    )
    graph.node_component_id = np.asarray(
        [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=np.int64
    )
    graph.component_id_to_swc_id = {
        0: "10.0",
        1: "20.0",
        2: "30.0",
        3: "40.0",
        4: "50.0",
    }
    dense = np.asarray([10, 20, 30, 40, 50], dtype=np.int64)
    leaves = leaf_nodes(graph)
    config = SweepConfig(
        radii_um=(5.0,),
        per_anchor_k=(1, 2, 4),
        modes=PAIRING_RULES,
        anchor_batch_size=2,
        cpus=1,
    )
    truth = {(10, 20), (10, 30), (40, 50)}
    truth_by_kind = {
        "direct": {(10, 20)},
        "gap": {(10, 30), (40, 50)},
    }
    observed = {}
    raw_results = {}
    for mode in config.modes:
        k_options, a, b, gaps = enumerate_pair_min_gaps(
            graph, leaves, dense, config, mode
        )
        raw_results[mode] = (k_options, a.copy(), b.copy(), gaps.copy())
        rows = metric_rows_for_mode(
            "synthetic",
            mode,
            k_options,
            a,
            b,
            gaps,
            truth,
            truth_by_kind,
            config,
        )
        observed[mode] = pd.DataFrame(rows)
    ll = observed[TIP_TO_TIP].query("per_anchor_k == 'all'").iloc[0]
    la = observed[TIP_TO_ANY_NODE].query("per_anchor_k == '4'").iloc[0]
    aa = observed[ANY_NODE_TO_ANY_NODE].query("per_anchor_k == '4'").iloc[0]
    assert int(ll.n_candidate_truth_pairs) == 1, ll
    assert int(la.n_candidate_truth_pairs) == 2, la
    assert int(aa.n_candidate_truth_pairs) == 3, aa

    parallel_config = SweepConfig(
        radii_um=(5.0,),
        per_anchor_k=(1, 2, 4),
        modes=(ANY_NODE_TO_ANY_NODE,),
        anchor_batch_size=2,
        cpus=2,
    )
    serial_any_config = SweepConfig(
        radii_um=(5.0,),
        per_anchor_k=(1, 2, 4),
        modes=(ANY_NODE_TO_ANY_NODE,),
        anchor_batch_size=2,
        cpus=1,
    )
    assert serial_any_config.signature() == parallel_config.signature()
    parallel = enumerate_pair_min_gaps(
        graph, leaves, dense, parallel_config, ANY_NODE_TO_ANY_NODE
    )
    serial = raw_results[ANY_NODE_TO_ANY_NODE]
    assert parallel[0] == serial[0]
    np.testing.assert_array_equal(parallel[1], serial[1])
    np.testing.assert_array_equal(parallel[2], serial[2])
    np.testing.assert_array_equal(parallel[3], serial[3])

    # A larger deterministic case exercises many shards, invalid labels, and
    # duplicate segment pairs that must be reduced across worker boundaries.
    random_graph = nx.Graph()
    nodes_per_component = 6
    n_components = 8
    for component in range(n_components):
        first = component * nodes_per_component
        random_graph.add_edges_from(
            (first + offset, first + offset + 1)
            for offset in range(nodes_per_component - 1)
        )
    rng = np.random.default_rng(794495)
    random_graph.node_xyz = rng.uniform(
        0.0, 15.0, size=(nodes_per_component * n_components, 3)
    )
    random_graph.node_component_id = np.repeat(
        np.arange(n_components, dtype=np.int64), nodes_per_component
    )
    random_dense = np.asarray([101, 102, 103, 104, 105, 106, 0, 108], dtype=np.int64)
    random_graph.component_id_to_swc_id = {
        component: str(segment) for component, segment in enumerate(random_dense)
    }
    random_leaves = leaf_nodes(random_graph)
    random_serial_config = SweepConfig(
        radii_um=(4.0, 7.0),
        per_anchor_k=(1, 2, 4),
        modes=(ANY_NODE_TO_ANY_NODE,),
        anchor_batch_size=3,
        cpus=1,
    )
    random_parallel_config = SweepConfig(
        radii_um=(4.0, 7.0),
        per_anchor_k=(1, 2, 4),
        modes=(ANY_NODE_TO_ANY_NODE,),
        anchor_batch_size=3,
        cpus=3,
    )
    assert random_serial_config.signature() == random_parallel_config.signature()
    random_serial = enumerate_pair_min_gaps(
        random_graph,
        random_leaves,
        random_dense,
        random_serial_config,
        ANY_NODE_TO_ANY_NODE,
    )
    previous_temp_parent = os.environ.get("SPLIT_CANDIDATE_TMPDIR")
    try:
        with tempfile.TemporaryDirectory(
            prefix="split-candidate-shard-parent-"
        ) as temporary_parent:
            os.environ["SPLIT_CANDIDATE_TMPDIR"] = temporary_parent
            random_parallel = enumerate_pair_min_gaps(
                random_graph,
                random_leaves,
                random_dense,
                random_parallel_config,
                ANY_NODE_TO_ANY_NODE,
            )
            assert not list(Path(temporary_parent).iterdir())

            bad_node_components = random_graph.node_component_id.copy()
            bad_node_components[0] = len(random_dense)
            bad_context = _EnumerationContext(
                xyz=random_graph.node_xyz,
                node_components=bad_node_components,
                component_segment=random_dense,
                tips=random_leaves,
                tip_segments=random_dense[
                    random_graph.node_component_id[random_leaves]
                ],
                partner_nodes=None,
                partner_tree=cKDTree(random_graph.node_xyz, copy_data=False),
                k_options=(1, 2, 4),
                max_k=4,
                unbounded_column=None,
                max_radius=7.0,
                anchor_batch_size=3,
                mode=ANY_NODE_TO_ANY_NODE,
            )
            try:
                _parallel_any_node_to_any_node(bad_context, cpus=2)
            except IndexError:
                pass
            else:
                raise AssertionError("injected worker failure did not propagate")
            assert not list(Path(temporary_parent).iterdir())
    finally:
        if previous_temp_parent is None:
            os.environ.pop("SPLIT_CANDIDATE_TMPDIR", None)
        else:
            os.environ["SPLIT_CANDIDATE_TMPDIR"] = previous_temp_parent
    assert random_parallel[0] == random_serial[0]
    for parallel_array, serial_array in zip(random_parallel[1:], random_serial[1:]):
        np.testing.assert_array_equal(parallel_array, serial_array)

    # Validate the complete artifact-writing path in an isolated temporary tree.
    combined = pd.concat(observed.values(), ignore_index=True)
    assert len(combined) == 10, combined
    assert set(combined.global_cap.astype(str)) == {"none"}, combined
    assert not combined.config_id.str.contains("cap=").any(), combined
    aggregate = aggregate_results(combined)
    recommendations = recommend_configs(aggregate)
    dataset_summary = pd.DataFrame(
        [
            {
                "brain": "synthetic",
                "n_reachable_truth_pairs": 3,
                "best_candidate_recall": float(combined.candidate_recall.max()),
                "fewest_candidates_at_best_recall": int(
                    combined.loc[
                        combined.candidate_recall == combined.candidate_recall.max(),
                        "n_candidate_pairs",
                    ].min()
                ),
                "minimal_best_config_id": "synthetic",
            }
        ]
    )
    with tempfile.TemporaryDirectory(prefix="split-candidate-self-test-") as tmp:
        run_dir = Path(tmp)
        save_per_dataset_plot(
            combined, run_dir / "per_dataset" / "synthetic" / "figures" / "diag.png"
        )
        save_aggregate_plot(combined, aggregate, run_dir / "figures" / "cross.png")
        write_results_report(
            run_dir,
            config,
            dataset_summary,
            combined.iloc[0:0],
            recommendations,
        )
        manifest = write_artifact_manifest(run_dir)
        manifest_payload = json.loads(manifest.read_text())
        assert any(item["role"] == "figure" for item in manifest_payload["artifacts"])
        assert any(item["role"] == "report" for item in manifest_payload["artifacts"])
    print(
        "SELF_TEST_OK: tip_to_any_node rescues tip-to-shaft and "
        "any_node_to_any_node rescues shaft-to-shaft; "
        "serial/parallel outputs match exactly; artifact bundle validates"
    )


# %% CLI orchestration
def scheduler_visible_cpu_count() -> int:
    """Return the CPU allocation visible to this process."""

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brains", nargs="*", help="brain ids; default: every cache")
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--radii", nargs="+", type=float, default=None)
    parser.add_argument(
        "--per-anchor-k",
        "--per-tip-k",
        dest="per_anchor_k",
        nargs="+",
        type=int,
        default=None,
        help=(
            "maximum distinct partner segments retained per anchor node; "
            "legacy alias: --per-tip-k"
        ),
    )
    parser.add_argument(
        "--pairing-rules",
        "--modes",
        dest="modes",
        nargs="+",
        choices=PAIRING_RULES + tuple(LEGACY_PAIRING_RULE_ALIASES),
        default=None,
        help=(
            "endpoint pairing rules; default: tip_to_tip tip_to_any_node. "
            "any_node_to_any_node is opt-in and potentially very expensive. "
            "Legacy aliases leaf_leaf and leaf_any are accepted."
        ),
    )
    parser.add_argument(
        "--anchor-batch-size",
        "--tip-batch-size",
        dest="anchor_batch_size",
        type=int,
        default=2_000,
        help="KD-tree query batch size; legacy alias: --tip-batch-size",
    )
    parser.add_argument(
        "--cpus",
        "--workers",
        dest="cpus",
        type=int,
        default=1,
        help=(
            "CPU budget; any_node_to_any_node uses this many processes with "
            "one KD-tree thread each, while tip modes use KD-tree threads. "
            "The value -1 maps to all scheduler-visible CPUs; --workers is a "
            "legacy alias."
        ),
    )
    parser.add_argument("--force", action="store_true", help="recompute checkpoints")
    review_group = parser.add_mutually_exclusive_group()
    review_group.add_argument(
        "--skip-ai-review",
        action="store_true",
        help="do not generate AI_REVIEW.md after deterministic artifacts",
    )
    review_group.add_argument(
        "--require-ai-review",
        action="store_true",
        help=(
            "generate the review even for a partial bundle and fail if it "
            "cannot be completed"
        ),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> SweepConfig:
    defaults = SweepConfig()
    cpus = args.cpus
    visible_cpus = scheduler_visible_cpu_count()
    if cpus == -1:
        cpus = visible_cpus
    if cpus > visible_cpus:
        raise ValueError(
            f"--cpus={cpus} exceeds the {visible_cpus} CPUs visible to this "
            "process/scheduler allocation"
        )
    requested_modes = args.modes if args.modes is not None else defaults.modes
    modes = tuple(
        LEGACY_PAIRING_RULE_ALIASES.get(mode, mode) for mode in requested_modes
    )
    # Preserve user order while avoiding duplicate work when a canonical name
    # and its legacy alias are both supplied.
    modes = tuple(dict.fromkeys(modes))
    config = SweepConfig(
        radii_um=tuple(args.radii) if args.radii is not None else defaults.radii_um,
        per_anchor_k=(
            tuple(args.per_anchor_k)
            if args.per_anchor_k is not None
            else defaults.per_anchor_k
        ),
        modes=modes,
        include_unbounded_tip_to_tip=True,
        anchor_batch_size=args.anchor_batch_size,
        cpus=cpus,
    )
    config.validate()
    return config


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    args = parse_args(argv)
    if args.self_test:
        run_synthetic_self_test()
        return 0

    config = config_from_args(args)
    if ANY_NODE_TO_ANY_NODE in config.modes:
        print(
            "WARNING: any_node_to_any_node queries every valid graph node as an "
            f"anchor up to {config.max_radius_um:g} um. Start with a small maximum "
            "radius and per_anchor_k before attempting the full grid.",
            flush=True,
        )
    output_root = args.output_dir.resolve()
    run_dir = output_root / f"mcl{args.mcl}_{config.signature()}"
    per_dataset_root = run_dir / "per_dataset"
    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    for directory in (per_dataset_root, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.resolve()
    caches = discover_add_caches(cache_dir, args.mcl, args.brains)
    available_brains = {
        brain_id_from_cache(path)
        for path in discover_add_caches(cache_dir, args.mcl, brains=None)
    }
    print(f"config signature: {config.signature()}")
    print(f"artifact bundle: {run_dir}")
    print("datasets:")
    for cache in caches:
        print(f"  {brain_id_from_cache(cache)}: {cache}")

    for cache in caches:
        brain = brain_id_from_cache(cache)
        brain_dir = per_dataset_root / brain
        brain_figure_dir = brain_dir / "figures"
        brain_figure_dir.mkdir(parents=True, exist_ok=True)
        csv_path = brain_dir / "sweep.csv"
        meta_path = brain_dir / "metadata.json"
        reusable = False
        if csv_path.exists() and meta_path.exists() and not args.force:
            old_meta = json.loads(meta_path.read_text())
            cache_stat = cache.stat()
            reusable = (
                old_meta.get("config_signature") == config.signature()
                and old_meta.get("cache_size_bytes") == cache_stat.st_size
                and old_meta.get("cache_mtime_ns") == cache_stat.st_mtime_ns
            )
        if reusable:
            print(f"reusing checkpoint for {brain}: {csv_path}")
            frame = pd.read_csv(
                csv_path, dtype={"per_anchor_k": str, "global_cap": str}
            )
            meta = json.loads(meta_path.read_text())
        else:
            frame, meta = analyze_dataset(cache, config)
            atomic_to_csv(frame, csv_path)
            atomic_write_text(
                meta_path, json.dumps(meta, indent=2, default=list) + "\n"
            )
            print(f"wrote {csv_path}")
            print(f"wrote {meta_path}")
        save_per_dataset_plot(
            frame, brain_figure_dir / "candidate_pool_diagnostics.png"
        )

    # Aggregate every completed checkpoint in this configuration-scoped bundle,
    # not only the brains requested in the current invocation. This makes a
    # sequence of one-brain jobs converge safely to the same all-brain result.
    frames = []
    metadata_records = []
    for brain_dir in sorted(
        path for path in per_dataset_root.iterdir() if path.is_dir()
    ):
        csv_path = brain_dir / "sweep.csv"
        meta_path = brain_dir / "metadata.json"
        if not csv_path.exists() or not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("config_signature") != config.signature():
            continue
        frames.append(
            pd.read_csv(csv_path, dtype={"per_anchor_k": str, "global_cap": str})
        )
        metadata_records.append(meta)
    if not frames:
        raise RuntimeError("no completed per-dataset checkpoints to aggregate")

    all_results = pd.concat(frames, ignore_index=True)
    aggregate = aggregate_results(all_results)
    frontier = pareto_frontier(aggregate)
    recommendations = recommend_configs(aggregate)

    baseline = all_results[
        (all_results["mode"] == TIP_TO_TIP)
        & (all_results["radius_um"] == 30.0)
        & (all_results["per_anchor_k"].astype(str) == "all")
        & (all_results["global_cap"].astype(str) == "none")
    ].copy()
    summary_rows = []
    for brain, group in all_results.groupby("brain"):
        best_recall = float(group["candidate_recall"].max())
        best = (
            group[np.isclose(group["candidate_recall"], best_recall)]
            .sort_values(["n_candidate_pairs", "config_id"])
            .iloc[0]
        )
        summary_rows.append(
            {
                "brain": brain,
                "n_reachable_truth_pairs": int(best["n_reachable_truth_pairs"]),
                "best_candidate_recall": best_recall,
                "fewest_candidates_at_best_recall": int(best["n_candidate_pairs"]),
                "minimal_best_config_id": best["config_id"],
            }
        )
    dataset_summary = pd.DataFrame(summary_rows).sort_values("brain")

    atomic_to_csv(all_results, tables_dir / "all_dataset_configurations.csv")
    atomic_to_csv(aggregate, tables_dir / "cross_dataset_aggregate.csv")
    atomic_to_csv(frontier, tables_dir / "pareto_frontier.csv")
    atomic_to_csv(recommendations, tables_dir / "recommended_minimal_pools.csv")
    atomic_to_csv(baseline, tables_dir / "current_detector_baseline.csv")
    atomic_to_csv(dataset_summary, tables_dir / "dataset_summary.csv")
    save_aggregate_plot(
        all_results,
        aggregate,
        figures_dir / "cross_dataset_candidate_pool_diagnostics.png",
    )

    completed_at = datetime.now(timezone.utc).isoformat()
    run_summary = {
        "result_status": "complete",
        "created_at_utc": completed_at,
        "elapsed_seconds": time.monotonic() - started,
        "mcl": args.mcl,
        "config_signature": config.signature(),
        "config": asdict(config),
        "brains": sorted(str(value) for value in all_results["brain"].unique()),
        "brains_requested_this_invocation": [
            brain_id_from_cache(path) for path in caches
        ],
        "inputs": metadata_records,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "networkx": nx.__version__,
        },
        "recommended_minimal_pools": recommendations.to_dict(orient="records"),
        "artifact_root": str(run_dir),
        "analysis_script": str(THIS_FILE),
        "analysis_script_sha256": sha256_file(THIS_FILE),
        "argv": sys.argv,
    }
    atomic_write_text(
        run_dir / "run_summary.json",
        json.dumps(run_summary, indent=2, default=list) + "\n",
    )
    atomic_write_text(
        run_dir / "provenance" / THIS_FILE.name,
        THIS_FILE.read_text(),
    )
    write_results_report(run_dir, config, dataset_summary, baseline, recommendations)
    # Give the review driver a complete deterministic manifest to verify. The
    # final manifest is regenerated afterward to include the AI review files.
    manifest_path = write_artifact_manifest(run_dir)

    completed_brains = {str(value) for value in all_results["brain"].unique()}
    bundle_covers_all_available = completed_brains == available_brains
    ai_review_status = "skipped"
    ai_review_path = run_dir / "AI_REVIEW.md"
    should_review = not args.skip_ai_review and (
        bundle_covers_all_available or args.require_ai_review
    )
    if not args.skip_ai_review and not should_review:
        ai_review_status = "deferred_incomplete_bundle"
        missing_brains = sorted(available_brains - completed_brains)
        print(
            "\nDeferring automatic AI review until the configuration bundle "
            f"contains every available brain; missing: {missing_brains}",
            flush=True,
        )
    if should_review:
        review_driver = REPO / "agentic" / "run_split_candidate_result_analysis.py"
        review_command = [sys.executable, str(review_driver), str(run_dir)]
        print("\nGenerating automatic English AI review...", flush=True)
        try:
            subprocess.run(review_command, cwd=REPO, check=True)
            ai_review_status = "complete"
        except subprocess.CalledProcessError as exc:
            ai_review_status = "failed"
            print(
                f"WARNING: automatic AI review failed with exit code "
                f"{exc.returncode}; deterministic sweep artifacts are complete.",
                file=sys.stderr,
                flush=True,
            )
            if args.require_ai_review:
                write_artifact_manifest(run_dir)
                raise
    manifest_path = write_artifact_manifest(run_dir)

    # A small pointer at the root makes the most recently completed bundle easy
    # to discover without using a mutable symlink.
    atomic_write_text(
        output_root / "LATEST.json",
        json.dumps(
            {
                "completed_at_utc": completed_at,
                "config_signature": config.signature(),
                "run_dir": str(run_dir),
                "run_summary": str(run_dir / "run_summary.json"),
                "artifact_manifest": str(manifest_path),
                "ai_review_status": ai_review_status,
                "ai_review": str(ai_review_path) if ai_review_path.is_file() else None,
            },
            indent=2,
        )
        + "\n",
    )
    print("\nRecommended minimal pools by worst-dataset recall target:")
    print(recommendations.to_string(index=False))
    print(f"\nall outputs: {run_dir}")
    print(f"artifact manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

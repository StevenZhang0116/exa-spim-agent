#!/usr/bin/env python
"""
Merge candidate-pool sweep: enumeration ceilings of the merge candidate rules + evidence.

The merge-topology census (``merge_topology_diagnosis.py``) measured the
REVERSE distance (GT merge site -> nearest junction). A deployable candidate
policy needs the FORWARD operating characteristics: for each enumeration rule,
how many candidates does a brain produce and what fraction of GT merge sites
does the rule cover? This script measures exactly that, with the same
discipline as ``split_candidate_pool_sweep.py``: per-brain checkpointing, a
configuration-signature-scoped artifact bundle (tables, figures, RESULTS.md,
provenance snapshot, SHA-256 manifest), and a DETERMINISTIC policy selection
written to ``recommended_policy.json``.

Enumeration rules (GT-BLIND at deploy time — candidates are read off the
fragments graph only; GT enters offline, in the recall measurement). Every rule is
applied by ``candidate_rules.py`` at the repository root, the same code the benchmark
tables and the detectors use (2026-10-16):

* junctions: every degree>=3 fragment node, with optional segment-scoped
  non-maximum suppression at ``nms_um`` (greedy, highest degree first, node id
  tie-break; suppression never crosses segments, so a passing foreign arbor
  cannot delete another segment's candidate),
* plus, per ``--tip-gaps`` value, tip-to-shaft candidates: shaft nodes that a
  degree-1 tip of another segment pairs with within the gap (``--tip-partner-k``
  nearest partner segments per tip), dropped near a kept junction of the same
  segment and suppressed within ``nms_um`` in order (gap, node id),
* claim radius ``r``: a GT site counts as covered when its snapped node reaches
  a KEPT candidate within ``r`` µm of GEODESIC cable (bounded Dijkstra — a
  Euclidean ball would credit spatially passing, topologically distant arbors).

The rule the benchmark uses is declared in ``proofreader_evolve/candidate_pools.json``;
this sweep reports evidence for it (``RESULTS.md`` lists the declared rule per brain)
and still writes a floor-based ``recommended_policy.json`` for reference.

Known structural ceilings (measured by the census + follow-up, restated per
brain in RESULTS.md — a junction-anchored rule can NEVER cover these):

* bridge sites with no junction at any useful scale (~9% of sites pooled; tip-to-shaft
  candidates reach part of them),
* merge labels with no recorded site (invisible to site-anchored evaluation),
* merge labels with no fragment skeleton at all.

Usage (panda env, from the project root; needs >20 GB RAM per cache)::

    python notebooks/merge_candidate_pool_sweep.py --self-test
    python notebooks/merge_candidate_pool_sweep.py --mcl 100
    python notebooks/merge_candidate_pool_sweep.py --mcl 100 --brains 794495
    python notebooks/merge_candidate_pool_sweep.py --mcl 100 --policy-target 0.70
    python notebooks/merge_candidate_pool_sweep.py --mcl 100 --nms 20 \
        --claim-radii 20 30 150 --tip-gaps 10 20 40 80 --policy-target 0.90

Artifacts land in
``notebooks/merge_candidate_pool_sweep_outputs/mcl<MCL>_<signature>/``.
Re-running with the same configuration, cache identity and sweep source reuses finished brains (checkpoint);
delete a brain's ``per_brain`` files to force recomputation. The figures are
enumeration ceilings, not classifier performance: after adopting a policy, a
site-scoring model must be trained and evaluated on held-out brains.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import heapq

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import candidate_rules as cr  # noqa: E402

from merge_topology_diagnosis import (
    CACHE_DIR,
    REPO,
    _snap_to_segment,
    atomic_to_csv,
    atomic_write_text,
    brain_id_from_cache,
    build_fragment_index,
    discover_add_caches,
    load_payload,
)

THIS_FILE = Path(__file__).resolve()
DEFAULT_OUTPUT_DIR = THIS_FILE.parent / "merge_candidate_pool_sweep_outputs"
MODE = "junction"
TIP_MODE = "junction+tip_to_shaft"
POOL_DECLARATION = THIS_FILE.parents[1] / "proofreader_evolve" / "candidate_pools.json"


@dataclass(frozen=True)
class SweepConfig:
    """Experiment grid shared by every brain (defines the config signature)."""

    claim_radii_um: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 30.0)
    nms_um: tuple[float, ...] = (0.0, 10.0, 20.0)
    snap_max_um: float = 50.0
    tip_gaps_um: tuple[float, ...] = ()  # empty: junction rules only
    tip_partner_k: int = 2

    def validate(self) -> None:
        if not self.claim_radii_um or min(self.claim_radii_um) <= 0:
            raise ValueError("claim_radii_um must contain positive values")
        if tuple(sorted(set(self.claim_radii_um))) != self.claim_radii_um:
            raise ValueError("claim_radii_um must be sorted and unique")
        if not self.nms_um or min(self.nms_um) < 0:
            raise ValueError("nms_um must contain non-negative values")
        if tuple(sorted(set(self.nms_um))) != self.nms_um:
            raise ValueError("nms_um must be sorted and unique")
        if self.snap_max_um <= 0:
            raise ValueError("snap_max_um must be positive")
        if self.tip_gaps_um and (min(self.tip_gaps_um) <= 0
                                 or tuple(sorted(set(self.tip_gaps_um))) != self.tip_gaps_um):
            raise ValueError("tip_gaps_um must be positive, sorted and unique")
        if int(self.tip_partner_k) != self.tip_partner_k or self.tip_partner_k < 1:
            raise ValueError("tip_partner_k must be a positive integer")

    def signature(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def max_claim_um(self) -> float:
        return max(self.claim_radii_um)


def config_id(nms_um: float, claim_radius_um: float, tip_gap_um: float | None = None,
              tip_partner_k: int | None = None) -> str:
    if tip_gap_um is None:
        return cr.merge_config_id(MODE, nms_um, claim_radius_um)
    return cr.merge_config_id(TIP_MODE, nms_um, claim_radius_um, tip_gap_um, tip_partner_k)


# --------------------------------------------------------------------------- #
# Enumeration: junctions + segment-scoped NMS
# --------------------------------------------------------------------------- #
def enumerate_junctions(index: dict) -> np.ndarray:
    """Sorted node ids of every degree>=3 node on a valid (segment>0) node."""
    return np.flatnonzero((index["degrees"] >= 3) & (index["node_segments"] > 0))


def nms_keep_mask(index: dict, junctions: np.ndarray, nms_um: float) -> np.ndarray:
    """Boolean over ALL nodes: junction survives segment-scoped greedy NMS
    (``candidate_rules.kept_junctions``: order (-degree, node id), same-segment only)."""
    expected = enumerate_junctions(index)
    if not np.array_equal(np.asarray(junctions), expected):
        raise ValueError("nms_keep_mask expects enumerate_junctions(index)")
    _, kept = cr.kept_junctions(index["xyz"], index["node_segments"], index["degrees"], nms_um)
    return kept


def shaft_sources(index: dict, config: SweepConfig) -> dict:
    """Per tip gap: ``(nodes, gaps)`` of the tip-to-shaft pairing (``candidate_rules``)."""
    return {gap: cr.tip_to_shaft_gaps(index["xyz"], index["node_segments"], index["degrees"],
                                      gap, config.tip_partner_k)
            for gap in config.tip_gaps_um}


def geodesic_to_kept(index: dict, start_node: int, max_um: float,
                     kept: np.ndarray) -> float:
    """Cable distance to the nearest KEPT candidate; inf beyond ``max_um``."""
    xyz = index["xyz"]
    graph = index["graph"]
    if kept[start_node]:
        return 0.0
    best: dict[int, float] = {int(start_node): 0.0}
    heap: list[tuple[float, int]] = [(0.0, int(start_node))]
    while heap:
        distance, node = heapq.heappop(heap)
        if distance > best.get(node, float("inf")):
            continue
        if kept[node]:
            return float(distance)
        for neighbor in graph.neighbors(node):
            neighbor = int(neighbor)
            candidate = distance + float(np.linalg.norm(xyz[neighbor] - xyz[node]))
            if candidate <= max_um and candidate < best.get(neighbor, float("inf")):
                best[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor))
    return float("inf")


# --------------------------------------------------------------------------- #
# Per-brain sweep
# --------------------------------------------------------------------------- #
def sweep_payload(payload: dict, brain: str, config: SweepConfig
                  ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Sweep one loaded payload. Returns (sweep rows, site distances, meta)."""
    frag = payload["fragments_graph"]
    merge_sites = list(payload.get("gt_merge_sites") or [])
    merge_labels = set(int(x) for x in np.asarray(
        payload.get("gt_merge_labels", []), dtype=np.int64))
    index = build_fragment_index(frag)

    site_segments = {int(s["segment_id"]) for s in merge_sites}
    labels_without_site = sorted(merge_labels - site_segments)
    labels_without_fragment = sorted(
        s for s in merge_labels if s not in index["fragment_segments"])

    junctions = enumerate_junctions(index)
    started = time.monotonic()
    sources = shaft_sources(index, config)
    variants = {}
    for nms_um in config.nms_um:
        kept = nms_keep_mask(index, junctions, nms_um)
        variants[(nms_um, None)] = {"kept": kept, "n_candidates": int(kept.sum()),
                                    "n_junction_candidates": int(kept.sum()), "n_tip_to_shaft_candidates": 0}
        print(f"{brain}: nms={nms_um:g} keeps {int(kept.sum())} of {len(junctions)} junctions", flush=True)
        for gap, (nodes, gaps) in sources.items():
            shafts = cr.kept_shafts(index["xyz"], index["node_segments"], nodes, gaps, kept, nms_um)
            union = kept | shafts
            variants[(nms_um, gap)] = {"kept": union, "n_candidates": int(union.sum()),
                                       "n_junction_candidates": int(kept.sum()),
                                       "n_tip_to_shaft_candidates": int(shafts.sum())}
            print(f"{brain}: nms={nms_um:g} gap={gap:g} adds {int(shafts.sum())} tip-to-shaft "
                  f"candidates (of {len(nodes)} shaft nodes)", flush=True)

    def column(nms_um, gap):
        return f"dist_um_nms{nms_um:g}" + ("" if gap is None else f"_gap{gap:g}")

    site_rows = []
    for site_index, site in enumerate(merge_sites):
        segment_id = int(site["segment_id"])
        snap_node, snap_um = _snap_to_segment(
            index, segment_id, tuple(site["xyz"]), config.snap_max_um)
        row = {
            "brain": brain, "site_index": site_index, "segment_id": segment_id,
            "snap_node": -1 if snap_node is None else int(snap_node),
            "snap_um": snap_um,
        }
        for (nms_um, gap), variant in variants.items():
            if snap_node is None:
                distance = float("inf")
            else:
                distance = geodesic_to_kept(
                    index, snap_node, config.max_claim_um, variant["kept"])
            row[column(nms_um, gap)] = distance
        site_rows.append(row)
    sites_df = pd.DataFrame(site_rows)

    rows = []
    labels_with_site = sorted(site_segments & merge_labels)
    for (nms_um, gap), variant in variants.items():
        distances = (sites_df[column(nms_um, gap)].to_numpy()
                     if len(sites_df) else np.array([]))
        for radius in config.claim_radii_um:
            covered = distances <= radius
            covered_labels = (
                {int(s) for s, c in zip(sites_df["segment_id"], covered) if c}
                if len(sites_df) else set())
            covered_labels &= set(labels_with_site)
            rows.append({
                "brain": brain,
                "config_id": config_id(nms_um, radius, gap, config.tip_partner_k),
                "mode": MODE if gap is None else TIP_MODE,
                "nms_um": float(nms_um),
                "tip_gap_um": float("nan") if gap is None else float(gap),
                "tip_partner_k": (float("nan") if gap is None else int(config.tip_partner_k)),
                "claim_radius_um": float(radius),
                "n_candidates": variant["n_candidates"],
                "n_junction_candidates": variant["n_junction_candidates"],
                "n_tip_to_shaft_candidates": variant["n_tip_to_shaft_candidates"],
                "n_sites": len(sites_df),
                "n_sites_covered": int(covered.sum()),
                "site_recall": (float(covered.mean())
                                if len(sites_df) else float("nan")),
                "n_labels_with_site": len(labels_with_site),
                "n_labels_covered": len(covered_labels),
                "label_recall": (len(covered_labels) / len(labels_with_site)
                                 if labels_with_site else float("nan")),
            })

    meta = {
        "brain": brain,
        "n_fragment_nodes": index["n_nodes"],
        "n_junctions_raw": int(len(junctions)),
        "n_candidates_by_nms": {
            f"{n:g}": v["n_candidates"] for (n, gap), v in variants.items() if gap is None},
        "n_candidates_by_variant": {
            config_id(n, config.max_claim_um, gap, config.tip_partner_k).rsplit("|r=", 1)[0]: v["n_candidates"]
            for (n, gap), v in variants.items()},
        "n_tip_to_shaft_sources_by_gap": {f"{gap:g}": int(len(nodes)) for gap, (nodes, _) in sources.items()},
        "n_merge_sites": len(merge_sites),
        "n_merge_labels": len(merge_labels),
        "n_labels_with_site": len(labels_with_site),
        "n_labels_without_site": len(labels_without_site),
        "labels_without_site": labels_without_site,
        "n_labels_without_fragment": len(labels_without_fragment),
        "labels_without_fragment": labels_without_fragment,
        "n_sites_unsnappable": int((sites_df["snap_node"] < 0).sum()) if len(sites_df) else 0,
        "elapsed_seconds": time.monotonic() - started,
    }
    frame = pd.DataFrame(rows).sort_values(
        ["nms_um", "tip_gap_um", "claim_radius_um"], na_position="first").reset_index(drop=True)
    del index
    gc.collect()
    return frame, sites_df, meta


def sweep_brain(cache_path: Path, config: SweepConfig, run_dir: Path
                ) -> tuple[pd.DataFrame, dict]:
    """Reuse only checkpoints for the same config, cache file and sweep source."""
    cache_stat = cache_path.stat()
    cache_identity = {"path": str(cache_path.resolve()), "size": cache_stat.st_size,
                      "mtime_ns": cache_stat.st_mtime_ns, "inode": cache_stat.st_ino}
    source_sha256 = hashlib.sha256(THIS_FILE.read_bytes()).hexdigest()
    brain = brain_id_from_cache(cache_path)
    per_brain = run_dir / "per_brain"
    per_brain.mkdir(parents=True, exist_ok=True)
    sweep_path = per_brain / f"{brain}_sweep.csv"
    sites_path = per_brain / f"{brain}_site_distances.csv"
    meta_path = per_brain / f"{brain}_meta.json"
    if sweep_path.exists() and meta_path.exists() and sites_path.exists():
        meta = json.loads(meta_path.read_text())
        if (meta.get("config_signature") == config.signature()
                and meta.get("cache_identity") == cache_identity
                and meta.get("sweep_source_sha256") == source_sha256):
            frame = pd.read_csv(sweep_path)
            if len(frame):
                print(f"{brain}: reusing checkpoint {sweep_path}", flush=True)
                return frame, meta
    payload = load_payload(cache_path)
    frame, sites_df, meta = sweep_payload(payload, brain, config)
    del payload
    gc.collect()
    after = cache_path.stat()
    if (after.st_size, after.st_mtime_ns, after.st_ino) != (
            cache_stat.st_size, cache_stat.st_mtime_ns, cache_stat.st_ino):
        raise RuntimeError(f"Cache changed during sweep: {cache_path}; rerun after refresh completes")
    meta["cache"] = str(cache_path)
    meta["cache_identity"] = cache_identity
    meta["sweep_source_sha256"] = source_sha256
    meta["config_signature"] = config.signature()
    atomic_to_csv(frame, sweep_path)
    atomic_to_csv(sites_df, sites_path)
    atomic_write_text(meta_path, json.dumps(meta, indent=2) + "\n")
    return frame, meta


# --------------------------------------------------------------------------- #
# Aggregation, recommendation, policy
# --------------------------------------------------------------------------- #
def aggregate_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    if "tip_gap_um" not in combined:  # checkpoints of junction-only grids
        combined = combined.assign(tip_gap_um=float("nan"), tip_partner_k=float("nan"))
    grouped = combined.groupby(
        ["config_id", "mode", "nms_um", "tip_gap_um", "tip_partner_k", "claim_radius_um"],
        as_index=False, dropna=False).agg(
        total_candidates=("n_candidates", "sum"),
        min_site_recall=("site_recall", "min"),
        mean_site_recall=("site_recall", "mean"),
        min_label_recall=("label_recall", "min"),
        mean_label_recall=("label_recall", "mean"),
        n_brains=("brain", "nunique"),
    )
    total_sites = combined.groupby("config_id")["n_sites"].sum()
    total_covered = combined.groupby("config_id")["n_sites_covered"].sum()
    grouped["micro_site_recall"] = grouped["config_id"].map(
        (total_covered / total_sites).to_dict())
    return grouped.sort_values(["nms_um", "tip_gap_um", "claim_radius_um"],
                               na_position="first").reset_index(drop=True)


def recommend(aggregate: pd.DataFrame, targets: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    for target in targets:
        eligible = aggregate[aggregate["min_site_recall"] >= target]
        if len(eligible):
            chosen = eligible.sort_values(
                ["total_candidates", "config_id"]).iloc[0]
            status = "meets_target"
        else:
            chosen = aggregate.sort_values(
                ["min_site_recall", "total_candidates", "config_id"],
                ascending=[False, True, True]).iloc[0]
            status = "target_unmet_best_effort"
        rows.append({
            "target_worst_brain_site_recall": target,
            "status": status,
            "config_id": chosen["config_id"],
            "min_site_recall": chosen["min_site_recall"],
            "mean_site_recall": chosen["mean_site_recall"],
            "micro_site_recall": chosen["micro_site_recall"],
            "total_candidates": int(chosen["total_candidates"]),
        })
    return pd.DataFrame(rows)


def repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_policy(run_dir: Path, config: SweepConfig, aggregate: pd.DataFrame,
                 recommendations: pd.DataFrame, policy_target: float,
                 metas: list[dict], mcl: int) -> Path:
    """Deterministic policy selection with provenance — the split-side analog
    of split_candidate_policy.json, produced by the sweep itself."""
    row = recommendations[
        recommendations["target_worst_brain_site_recall"] == policy_target]
    if len(row) != 1:
        raise ValueError(f"policy target {policy_target} missing from targets")
    row = row.iloc[0]
    matched = aggregate[aggregate["config_id"] == row["config_id"]].iloc[0]
    aggregate_path = run_dir / "tables" / "cross_dataset_aggregate.csv"
    policy = {
        "schema_version": 1,
        "artifact_type": "merge_candidate_policy_selection",
        "application_status": "proposed_for_merge_site_detector",
        "objective": {
            "name": "minimum_candidate_count_subject_to_worst_brain_site_recall",
            "minimum_worst_brain_site_recall": policy_target,
            "tie_breaker": ["total_candidates", "config_id"],
        },
        "selection_status": row["status"],
        "selected_policy": {
            "config_id": row["config_id"],
            "mode": str(matched["mode"]),
            "nms_um": float(matched["nms_um"]),
            "tip_gap_um": (None if pd.isna(matched["tip_gap_um"]) else float(matched["tip_gap_um"])),
            "tip_partner_k": (None if pd.isna(matched["tip_partner_k"]) else int(matched["tip_partner_k"])),
            "claim_radius_um": float(matched["claim_radius_um"]),
            "nms_scope": "segment",
            "distance_metric": "geodesic_cable_um",
        },
        "declared_rule_note": ("The benchmark and detector builds use the rule declared in "
                               "proofreader_evolve/candidate_pools.json, not this floor-based selection."),
        "evidence_metrics": {
            "min_site_recall": float(row["min_site_recall"]),
            "mean_site_recall": float(row["mean_site_recall"]),
            "micro_site_recall": float(row["micro_site_recall"]),
            "min_label_recall": float(matched["min_label_recall"]),
            "total_candidates": int(row["total_candidates"]),
        },
        "structural_ceilings": {
            "note": ("Denominator is ALL recorded gt_merge_sites. These labels "
                     "are invisible to any site-anchored rule and NOT counted "
                     "against site recall; report label-level coverage "
                     "separately when evaluating a detector."),
            "per_brain": [{
                "brain": meta["brain"],
                "n_labels_without_site": meta["n_labels_without_site"],
                "n_labels_without_fragment": meta["n_labels_without_fragment"],
                "n_sites_unsnappable": meta["n_sites_unsnappable"],
            } for meta in metas],
        },
        "evidence_scope": {
            "brains": [meta["brain"] for meta in metas],
            "mcl": mcl,
            "config_signature": config.signature(),
            "segment_detector_gating_evaluated": False,
            "bridge_candidate_family_evaluated": bool(config.tip_gaps_um),
            "tip_to_shaft_gaps_um": list(config.tip_gaps_um),
        },
        "provenance": {
            "aggregate_path": repo_relative(aggregate_path),
            "aggregate_sha256": sha256_of(aggregate_path),
            "sweep_script": repo_relative(THIS_FILE),
            "sweep_script_sha256": sha256_of(THIS_FILE),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    path = run_dir / "recommended_policy.json"
    atomic_write_text(path, json.dumps(policy, indent=2, sort_keys=True) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Report, figure, manifest
# --------------------------------------------------------------------------- #
def write_figure(run_dir: Path, combined: pd.DataFrame,
                 aggregate: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for (nms_um, gap), group in aggregate.groupby(["nms_um", "tip_gap_um"], dropna=False):
        group = group.sort_values("total_candidates")
        label = f"nms={nms_um:g} µm" + ("" if pd.isna(gap) else f", tip gap {gap:g} µm")
        axes[0].plot(group["total_candidates"], group["min_site_recall"],
                     marker="o", label=label)
        for _, row in group.iterrows():
            axes[0].annotate(f"r={row['claim_radius_um']:g}",
                             (row["total_candidates"], row["min_site_recall"]),
                             fontsize=7, xytext=(3, 3),
                             textcoords="offset points")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("total candidates (all brains)")
    axes[0].set_ylabel("worst-brain site recall")
    axes[0].set_title("Recall-vs-cost frontier")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    nms_zero = combined[(combined["nms_um"] == combined["nms_um"].min())
                        & combined.get("tip_gap_um", pd.Series(np.nan, index=combined.index)).isna()]
    for brain, group in nms_zero.groupby("brain"):
        group = group.sort_values("claim_radius_um")
        axes[1].plot(group["claim_radius_um"], group["site_recall"],
                     marker="o", label=str(brain))
    axes[1].set_xlabel("claim radius (µm, geodesic)")
    axes[1].set_ylabel("site recall")
    axes[1].set_title(f"Per-brain recall, junctions at nms={combined['nms_um'].min():g} µm")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "merge_candidate_pool_frontier.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def declared_rule_rows(combined: pd.DataFrame, declaration: Path = POOL_DECLARATION) -> pd.DataFrame:
    """Per-brain rows of the merge rule declared in candidate_pools.json (empty if not swept)."""
    try:
        rule = json.loads(declaration.read_text())["pools"]["merge"]
    except (OSError, KeyError, ValueError):
        return pd.DataFrame()
    if combined.empty or rule is None:
        return pd.DataFrame()
    prefix = config_id(rule["nms_um"], rule["claim_radius_um"], rule["tip_gap_um"],
                       rule["tip_partner_k"]).rsplit("|r=", 1)[0] + "|r="
    rows = combined[combined["config_id"].str.startswith(prefix)]
    if rows.empty:
        return rows
    table = rows.pivot_table(index="brain", columns="claim_radius_um", values="site_recall")
    table.columns = [f"site_recall_r{c:g}" for c in table.columns]
    first = rows.drop_duplicates("brain").set_index("brain")
    for name in ("n_sites", "n_candidates", "n_junction_candidates", "n_tip_to_shaft_candidates"):
        if name in first:
            table[name] = first[name]
    table = table.reset_index()
    table.insert(0, "rule", prefix[:-3])
    return table


def write_results_report(run_dir: Path, config: SweepConfig,
                         aggregate: pd.DataFrame,
                         recommendations: pd.DataFrame,
                         metas: list[dict], policy_target: float,
                         combined: pd.DataFrame | None = None) -> Path:
    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        if frame.empty:
            return "(not available)"
        return "```text\n" + frame[columns].to_string(index=False) + "\n```"

    dataset_rows = pd.DataFrame([{
        "brain": meta["brain"],
        "n_merge_sites": meta["n_merge_sites"],
        "n_merge_labels": meta["n_merge_labels"],
        "n_junctions_raw": meta["n_junctions_raw"],
        "labels_without_site": meta["n_labels_without_site"],
        "labels_without_fragment": meta["n_labels_without_fragment"],
        "sites_unsnappable": meta["n_sites_unsnappable"],
    } for meta in metas])
    declared = declared_rule_rows(pd.DataFrame() if combined is None else combined)
    declared_text = ("The declared rule is not in this grid." if declared.empty else
                     "Site recall per claim radius:\n\n" + table(declared, list(declared.columns)))
    report = f"""# Merge candidate-pool sweep artifacts

Configuration signature `{config.signature()}`. Site recall is measured against
ALL recorded ``gt_merge_sites`` (geodesic claim radius from the snapped node to
the nearest kept candidate, junction or tip-to-shaft). Enumeration is GT-blind at
deploy time; GT enters only this offline recall measurement.

## Dataset summary

{table(dataset_rows, list(dataset_rows.columns))}

## Declared rule (proofreader_evolve/candidate_pools.json)

{declared_text}

## Sweep aggregate (all configurations)

{table(aggregate, ["config_id", "total_candidates", "min_site_recall", "mean_site_recall", "micro_site_recall", "min_label_recall"])}

## Recommended minimal policies

Selection minimizes total candidate count subject to a worst-brain SITE-recall
target. `target_unmet_best_effort` rows report the best achievable point.
The policy at target {policy_target:g} is written to `recommended_policy.json`.

{table(recommendations, ["target_worst_brain_site_recall", "status", "config_id", "min_site_recall", "mean_site_recall", "total_candidates"])}

## Structural ceilings (never coverable by a junction-anchored rule)

Per brain: `labels_without_site` merge labels have no recorded site (invisible
to site-anchored evaluation), `labels_without_fragment` have no fragment
skeleton at all, and `sites_unsnappable` sites have no on-segment node within
{config.snap_max_um:g} µm. Bridge-type sites (no junction within the largest
claim radius) cap the recall plateau — see the census
(`notebooks/merge_diagnosis/`) for their anatomy. Tip-to-shaft candidates
(``--tip-gaps``) reach part of them; a path change-point family and
segment-detector gating were NOT evaluated here.

## Artifact layout

* `per_brain/<brain>_sweep.csv` — every configuration for one brain.
* `per_brain/<brain>_site_distances.csv` — per-site distance to nearest kept candidate per NMS (and tip-gap) variant.
* `per_brain/<brain>_meta.json` — residuals, junction counts, checkpoint signature.
* `tables/` — combined, aggregate, and recommendation CSVs.
* `figures/merge_candidate_pool_frontier.png` — recall-vs-cost frontier.
* `recommended_policy.json` — deterministic policy selection with provenance.
* `run_summary.json` — machine-readable configuration and headline results.
* `provenance/merge_candidate_pool_sweep.py` — exact analysis-source snapshot.
* `artifact_manifest.json` — SHA-256 and size for every artifact in this bundle.

These numbers are enumeration ceilings, not classifier performance. After
adopting a policy, the site-scoring model must be trained and evaluated on
held-out brains.
"""
    path = run_dir / "RESULTS.md"
    atomic_write_text(path, report)
    return path


def write_manifest(run_dir: Path) -> Path:
    entries = []
    manifest_path = run_dir / "artifact_manifest.json"
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path != manifest_path:
            entries.append({
                "path": str(path.relative_to(run_dir)),
                "sha256": sha256_of(path),
                "bytes": path.stat().st_size,
            })
    atomic_write_text(
        manifest_path, json.dumps({"artifacts": entries}, indent=2) + "\n")
    return manifest_path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_sweep(cache_paths: list[Path], config: SweepConfig, output_dir: Path,
              targets: tuple[float, ...], policy_target: float, mcl: int) -> Path:
    config.validate()
    if policy_target not in targets:
        targets = tuple(sorted(set(targets) | {policy_target}))
    run_dir = output_dir / f"mcl{mcl}_{config.signature()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    provenance = run_dir / "provenance"
    provenance.mkdir(exist_ok=True)
    atomic_write_text(provenance / THIS_FILE.name, THIS_FILE.read_text())

    frames, metas = [], []
    for cache_path in cache_paths:
        frame, meta = sweep_brain(cache_path, config, run_dir)
        frames.append(frame)
        metas.append(meta)

    combined = pd.concat(frames, ignore_index=True)
    aggregate = aggregate_frames(frames)
    recommendations = recommend(aggregate, targets)
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    atomic_to_csv(combined, tables_dir / "all_dataset_configurations.csv")
    atomic_to_csv(aggregate, tables_dir / "cross_dataset_aggregate.csv")
    atomic_to_csv(recommendations, tables_dir / "recommended_minimal_policies.csv")
    write_figure(run_dir, combined, aggregate)
    policy_path = write_policy(
        run_dir, config, aggregate, recommendations, policy_target, metas, mcl)
    write_results_report(
        run_dir, config, aggregate, recommendations, metas, policy_target, combined)
    summary = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "config_signature": config.signature(),
        "mcl": mcl,
        "python": platform.python_version(),
        "policy_target": policy_target,
        "targets": list(targets),
        "per_brain": metas,
        "recommendations": recommendations.to_dict(orient="records"),
        "policy_path": str(policy_path),
    }
    atomic_write_text(
        run_dir / "run_summary.json", json.dumps(summary, indent=2) + "\n")
    write_manifest(run_dir)
    print(f"bundle complete: {run_dir}", flush=True)
    print(f"policy: {policy_path}", flush=True)
    return run_dir


# --------------------------------------------------------------------------- #
# Synthetic self-test (no cache, no cloud)
# --------------------------------------------------------------------------- #
def _synthetic_payload():
    import networkx as nx

    graph = nx.Graph()
    xyz = np.array([
        [0., 0., 0.], [5., 0., 0.], [10., 0., 0.], [15., 0., 0.],  # 0-3
        [10., 5., 0.],                                             # 4 arm at 2
        [20., 0., 0.],                                             # 5 path end
        [15., 5., 0.],                                             # 6 arm at 3
        [100., 0., 0.], [105., 0., 0.], [110., 0., 0.],            # 7-9 seg 222
        [105., 8., 0.], [105., 30., 0.],                           # 10-11 seg 333
    ])
    edges = [(0, 1), (1, 2), (2, 3), (2, 4), (3, 5), (3, 6), (7, 8), (8, 9), (10, 11)]
    graph.add_nodes_from(range(len(xyz)))
    graph.add_edges_from(edges)
    graph.node_xyz = xyz
    graph.node_component_id = np.array([0] * 7 + [1] * 3 + [2] * 2)
    graph.component_id_to_swc_id = {0: "111.0", 1: "222.0", 2: "333.0"}
    return {
        "fragments_graph": graph,
        "gt_merge_sites": [
            {"segment_id": 111, "xyz": (5., 0., 0.)},     # 5 um from junction 2
            {"segment_id": 111, "xyz": (20., 0., 0.)},    # 5 um from junction 3
            {"segment_id": 222, "xyz": (105., 0., 0.)},   # bridge: no junction; 333's tip is 8 um away
            {"segment_id": 111, "xyz": (500., 500., 500.)},  # unsnappable
        ],
        "gt_merge_labels": np.array([111, 222, 444]),
    }


def run_synthetic_self_test() -> None:
    import tempfile

    config = SweepConfig(tip_gaps_um=(10.0,))
    payload = _synthetic_payload()
    frame, sites_df, meta = sweep_payload(payload, "synthetic", config)
    assert meta["n_junctions_raw"] == 2, meta
    assert meta["n_candidates_by_nms"] == {"0": 2, "10": 1, "20": 1}, meta
    # node 8 (segment 222) is the only shaft reached, by 333's tip at 8 um
    assert meta["n_tip_to_shaft_sources_by_gap"] == {"10": 1}, meta
    assert meta["n_candidates_by_variant"]["junction+tip_to_shaft|nms=20|gap=10|k=2"] == 2, meta
    assert meta["n_labels_without_site"] == 1 and 444 in meta["labels_without_site"]
    assert meta["n_labels_without_fragment"] == 1
    assert meta["n_sites_unsnappable"] == 1

    def recall(nms, r, gap=None):
        row = frame[(frame["nms_um"] == nms) & (frame["claim_radius_um"] == r)
                    & (frame["tip_gap_um"].isna() if gap is None else frame["tip_gap_um"] == gap)]
        assert len(row) == 1, row
        return float(row["site_recall"].iloc[0])

    assert recall(0.0, 5.0) == 0.5, frame       # both seg-111 sites at 5 um
    assert recall(10.0, 5.0) == 0.25, frame     # NMS pushed site 2 to 10 um
    assert recall(10.0, 10.0) == 0.5, frame
    assert recall(0.0, 30.0) == 0.5, frame      # bridge + unsnappable never covered by junctions
    assert recall(0.0, 5.0, gap=10.0) == 0.75, frame  # the bridge site sits on the shaft candidate
    label_row = frame[(frame["nms_um"] == 0.0) & (frame["claim_radius_um"] == 5.0) & frame["tip_gap_um"].isna()]
    assert float(label_row["label_recall"].iloc[0]) == 0.5  # 111 yes, 222 no

    aggregate = aggregate_frames([frame])
    recommendations = recommend(aggregate, (0.5, 0.75, 0.9))
    met = recommendations.iloc[0]
    assert met["status"] == "meets_target" and met["config_id"] == "junction|nms=10|r=10"
    shaft = recommendations.iloc[1]
    assert shaft["status"] == "meets_target", recommendations
    assert shaft["config_id"] == "junction+tip_to_shaft|nms=10|gap=10|k=2|r=10", recommendations
    unmet = recommendations.iloc[2]
    assert unmet["status"] == "target_unmet_best_effort"
    assert unmet["min_site_recall"] == 0.75

    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory) / "bundle"
        run_dir.mkdir()
        (run_dir / "tables").mkdir()
        atomic_to_csv(aggregate, run_dir / "tables" / "cross_dataset_aggregate.csv")
        policy_path = write_policy(
            run_dir, config, aggregate, recommendations, 0.5, [meta], mcl=100)
        policy = json.loads(policy_path.read_text())
        assert policy["selected_policy"]["config_id"] == "junction|nms=10|r=10"
        assert policy["selected_policy"]["tip_gap_um"] is None
        assert policy["selection_status"] == "meets_target"
        assert policy["evidence_scope"]["segment_detector_gating_evaluated"] is False
        assert policy["provenance"]["aggregate_sha256"] == sha256_of(
            run_dir / "tables" / "cross_dataset_aggregate.csv")
        write_results_report(run_dir, config, aggregate, recommendations, [meta], 0.5, frame)
        write_figure(run_dir, frame, aggregate)
        manifest = json.loads(write_manifest(run_dir).read_text())
        names = {entry["path"] for entry in manifest["artifacts"]}
        assert "recommended_policy.json" in names and "RESULTS.md" in names
    print("SELF_TEST_OK: junction and tip-to-shaft enumeration, segment-scoped NMS, "
          "geodesic claim recall, deterministic policy selection, and bundle artifacts "
          "all behave as designed", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--brains", nargs="*", default=None)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--claim-radii", nargs="*", type=float,
                        default=list(SweepConfig.claim_radii_um))
    parser.add_argument("--nms", nargs="*", type=float,
                        default=list(SweepConfig.nms_um))
    parser.add_argument("--snap-max-um", type=float,
                        default=SweepConfig.snap_max_um)
    parser.add_argument("--tip-gaps", nargs="*", type=float, default=list(SweepConfig.tip_gaps_um),
                        help="tip-to-shaft gaps (um) to add to every junction rule; none by default")
    parser.add_argument("--tip-partner-k", type=int, default=SweepConfig.tip_partner_k,
                        help="partner segments per tip for tip-to-shaft candidates (default 2)")
    parser.add_argument("--targets", nargs="*", type=float,
                        default=(0.60, 0.65, 0.70, 0.75, 0.80))
    parser.add_argument("--policy-target", type=float, default=0.70,
                        help="worst-brain site-recall floor baked into "
                             "recommended_policy.json (default 0.70)")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        run_synthetic_self_test()
        return 0
    config = SweepConfig(
        claim_radii_um=tuple(sorted(set(args.claim_radii))),
        nms_um=tuple(sorted(set(args.nms))),
        snap_max_um=args.snap_max_um,
        tip_gaps_um=tuple(sorted(set(args.tip_gaps))),
        tip_partner_k=args.tip_partner_k,
    )
    cache_paths = discover_add_caches(args.cache_dir, args.mcl, args.brains)
    run_sweep(cache_paths, config, args.output_dir,
              tuple(sorted(set(args.targets))), args.policy_target, args.mcl)
    return 0


if __name__ == "__main__":
    sys.exit(main())

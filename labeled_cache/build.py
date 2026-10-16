"""The builder: per brain, run the stages its cache files still need.

Stages (each skipped when its result exists):

- ``base``   build ``dataset_cache_<brain>_mcl<N>.pkl`` from the GCS skeletons;
- ``label``  bake the GT labels from the dense segmentation into ``_add.pkl``
             (reads GCS), under the requested merge-threshold policy;
- ``policy`` an ``_add.pkl`` exists: if its ``gt_label_policy`` differs from the
             request, recompute merge labels, sites and edge errors from the stored
             GT node labels (no cloud reads) and replace the file atomically. Graphs
             and node ids are untouched, so candidate pools keep their hashes.

A reused cache has its GT compared with the current GCS tracings first
(``verify_gt``); a stale cache is refused unless ``rebuild_base`` is set, which
deletes both pkl files and rebuilds from GCS.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
import gc
import pickle
from pathlib import Path
import time

import numpy as np

import agentic_neuron_proofreader  # noqa: F401  registers SkeletonGraph for unpickling
from agentic_neuron_proofreader.data_modules.canonical_labeling import MergeThresholds

from labeled_cache import io
from labeled_cache.policy import DEFAULT_LENGTH_UM, POLICY_LENGTH_V1, describe, policy_matches, thresholds_for

STAGE_BASE, STAGE_LABEL, STAGE_POLICY = "base", "label", "policy"
# Caches of derived detector universes stored by older tooling in the payload; they
# embed labels and must not survive a relabel.
DERIVED_PAYLOAD_KEYS = ("__detector_sample_universe_cache__", "__detector_universe_site_audit__")
LABEL_KEYS = ("gt_edge_error", "gt_merge_labels", "gt_merge_sites", "gt_merge_site_metadata",
              "gt_junction_audit", "gt_label_policy")


class StaleGroundTruth(RuntimeError):
    """The cached GT differs from the current GCS tracings."""


class ReproductionMismatch(RuntimeError):
    """Recomputing with the original thresholds did not reproduce the stored labels."""


@dataclasses.dataclass
class BuildRequest:
    brain: str
    mcl: int = 100
    cache_dir: Path = io.DEFAULT_CACHE_DIR
    policy: str = POLICY_LENGTH_V1
    length_um: float = DEFAULT_LENGTH_UM
    node_spacing: float = 5.0
    anisotropy: tuple = (0.748, 0.748, 1.0)
    verify_gt: bool = True
    gt_workers: int = 4
    rebuild_base: bool = False
    force_relabel: bool = False
    check_reproduce: bool = False
    verbose: bool = True

    @property
    def base_path(self) -> Path:
        return io.base_cache_path(self.cache_dir, self.brain, self.mcl)

    @property
    def labeled_path(self) -> Path:
        return io.labeled_cache_path(self.cache_dir, self.brain, self.mcl)


def plan(request: BuildRequest) -> list[str]:
    """Stages the brain's files call for, before anything is loaded."""
    if request.rebuild_base:
        return [STAGE_BASE, STAGE_LABEL]
    if request.labeled_path.exists():
        return [STAGE_POLICY]
    if request.base_path.exists():
        return [STAGE_LABEL]
    return [STAGE_BASE, STAGE_LABEL]


def run(request: BuildRequest, log=print) -> dict:
    """Run the planned stages for one brain and return a summary dict."""
    started = time.monotonic()
    stages = plan(request)
    log(f"\n=== brain {request.brain} mcl{request.mcl}: stages {stages}; policy {describe(request.policy, request.length_um)}")
    summary = {"brain": request.brain, "mcl": request.mcl, "policy": request.policy,
               "length_um": request.length_um, "stages": stages, "results": {}}
    if stages == [STAGE_POLICY]:
        result = apply_policy(request, log)
        summary["results"][STAGE_POLICY] = result
        if result.get("action") == "rebuild":
            log("GT is stale and --rebuild-base was given: rebuilding from GCS")
            stages = [STAGE_BASE, STAGE_LABEL]
            summary["stages"] = [STAGE_POLICY, *stages]
    if STAGE_BASE in stages or STAGE_LABEL in stages:
        dataset = _base_dataset(request, build=STAGE_BASE in stages, log=log)
        summary["results"][STAGE_LABEL] = _label(request, dataset, log)
        del dataset
        gc.collect()
    summary["elapsed_seconds"] = round(time.monotonic() - started, 1)
    return summary


# --- stages -------------------------------------------------------------------------------
def _base_dataset(request, build, log):
    from agentic_neuron_proofreader.data_modules.datasets import BrainDataset

    io.set_cloud_environment()
    sources = io.dataset_sources(request.brain)
    metadata = dict(fragments_path=sources["fragments_path"], gt_path=sources["gt_path"],
                    img_path=sources["img_path"], anisotropy=tuple(request.anisotropy),
                    min_cable_length=int(request.mcl), node_spacing=request.node_spacing)
    log(f"fragments: {metadata['fragments_path']}\nGT: {metadata['gt_path']}\nsegmentation: {sources['segmentation_path']}")
    base = request.base_path
    if build:
        for path in (base, request.labeled_path):
            if path.exists():
                log(f"removing {path}")
                path.unlink()
        log("loading fragment and GT skeletons from GCS...")
        dataset = BrainDataset(**metadata)
        io.atomic_cache_write(base, dataset.save)
        log(f"saved {base}")
    else:
        log(f"reusing base cache {base}")
        dataset = BrainDataset.load_from_cache(str(base))
        _validate_parameters(dataset, metadata, base)
        dataset.min_cable_length = int(request.mcl)  # legacy caches stored 100.0
        if request.verify_gt:
            _require_current_gt(dataset.gt_graph, metadata, request, log)
    dataset.segmentation_path = sources["segmentation_path"]
    return dataset


def _label(request, dataset, log):
    thresholds = thresholds_for(request.policy, dataset.gt_graph, request.length_um)
    log(f"labeling GT from the segmentation with thresholds {thresholds.as_record()}")
    dataset.label_gt_from_segmentation(dataset.segmentation_path, verbose=request.verbose,
                                       thresholds=thresholds)
    io.atomic_cache_write(request.labeled_path, dataset.save)
    log(f"saved {request.labeled_path}")
    return {"action": "labeled", "thresholds": thresholds.as_record(),
            "merge_labels": int(len(dataset.gt_graph.merge_labels)),
            "merge_sites": int(len(dataset.gt_graph.merge_sites))}


def apply_policy(request, log=print) -> dict:
    """Policy stage on an existing ``_add.pkl``; returns what was done."""
    from agentic_neuron_proofreader.data_modules.datasets import BrainDataset

    path = request.labeled_path
    log(f"loading {path}")
    signature = io.cache_signature(path)
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    gt = payload["gt_graph"]
    if request.verify_gt:
        metadata = {"gt_path": payload["gt_path"], "anisotropy": payload["anisotropy"],
                    "min_cable_length": payload["min_cable_length"], "node_spacing": payload["node_spacing"]}
        try:
            _require_current_gt(gt, metadata, request, log)
        except StaleGroundTruth:
            if request.rebuild_base:
                return {"action": "rebuild"}
            raise
    record = payload.get("gt_label_policy")
    if policy_matches(record, request.policy, request.length_um) and not request.force_relabel:
        log(f"policy already applied ({record}); nothing to do")
        return {"action": "skipped", "policy_record": record}
    if payload.get("gt_node_canonical_label") is None:
        raise ValueError(f"{path} carries no GT node labels; delete it and rerun to label from the segmentation")

    dataset = BrainDataset(payload["fragments_path"], payload["gt_path"], payload["img_path"],
                           anisotropy=payload["anisotropy"], min_cable_length=payload["min_cable_length"],
                           node_spacing=payload["node_spacing"], segmentation_path=payload.get("segmentation_path"),
                           _skip_load=True)
    dataset.fragments_graph, dataset.gt_graph = payload["fragments_graph"], gt
    gt.node_label = np.asarray(payload["gt_node_canonical_label"])
    stored = _label_state(payload["gt_merge_labels"], payload["gt_merge_sites"])
    result = {"action": "relabeled", "previous_policy": record,
              "previous": {"merge_labels": len(stored["labels"]), "merge_sites": len(stored["sites"])}}

    if request.check_reproduce:
        t = time.monotonic()
        dataset.refresh_merge_labels(verbose=request.verbose, thresholds=MergeThresholds.nodes())
        recomputed = _label_state(gt.merge_labels, gt.merge_sites)
        same = recomputed["labels"] == stored["labels"] and recomputed["sites"] == stored["sites"]
        result["reproduction"] = {"labels_equal": recomputed["labels"] == stored["labels"],
                                  "sites_equal": recomputed["sites"] == stored["sites"],
                                  "seconds": round(time.monotonic() - t, 1)}
        log(f"reproduction with the original thresholds: {result['reproduction']}")
        if not same:
            raise ReproductionMismatch(
                f"{path}: the original thresholds do not reproduce the stored labels "
                f"({len(stored['labels'])} stored vs {len(recomputed['labels'])} recomputed merge labels, "
                f"{len(stored['sites'])} vs {len(recomputed['sites'])} sites); nothing written")

    thresholds = thresholds_for(request.policy, gt, request.length_um)
    log(f"relabeling with thresholds {thresholds.as_record()}")
    t = time.monotonic()
    dataset.refresh_merge_labels(verbose=request.verbose, thresholds=thresholds)
    result["seconds"] = round(time.monotonic() - t, 1)
    result["thresholds"] = thresholds.as_record()
    result["current"] = {"merge_labels": int(len(gt.merge_labels)), "merge_sites": int(len(gt.merge_sites))}
    if not set(stored["labels"]) <= set(int(x) for x in gt.merge_labels):
        result["dropped_merge_labels"] = sorted(set(stored["labels"]) - set(int(x) for x in gt.merge_labels))
        log(f"note: {len(result['dropped_merge_labels'])} previous merge labels are not merges under the new thresholds")

    payload.update({
        "gt_edge_error": gt.edge_error,
        "gt_merge_labels": gt.merge_labels,
        "gt_merge_sites": gt.merge_sites,
        "gt_merge_site_metadata": gt.merge_site_metadata,
        "gt_junction_audit": None,
        "gt_label_policy": gt.label_policy,
    })
    for key in DERIVED_PAYLOAD_KEYS:
        payload.pop(key, None)

    def write(destination):
        with open(destination, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    io.atomic_cache_write(path, write, expected_signature=signature)
    log(f"replaced {path}: merge labels {result['previous']['merge_labels']} -> {result['current']['merge_labels']}, "
        f"sites {result['previous']['merge_sites']} -> {result['current']['merge_sites']}")
    return result


# --- helpers ------------------------------------------------------------------------------
def _label_state(merge_labels, merge_sites):
    labels = sorted(int(x) for x in (merge_labels if merge_labels is not None else []))
    sites = {(int(s["segment_id"]), tuple(np.round(np.asarray(s["xyz"], dtype=float), 3).tolist()))
             for s in (merge_sites or [])}
    return {"labels": labels, "sites": sites}


def _validate_parameters(dataset, expected, path):
    """Refuse a base cache built with other sources or skeleton parameters."""
    for name, value in expected.items():
        actual = getattr(dataset, name)
        if name == "anisotropy":
            actual = tuple(actual)
        if actual != value:
            raise ValueError(f"Cannot reuse {path}: {name}={actual!r}, expected {value!r}")


def _require_current_gt(gt_graph, metadata, request, log):
    from labeled_cache.gt_check import compare_gt, format_report

    io.set_cloud_environment()
    log(f"comparing the cached GT with {metadata['gt_path']} ...")
    report = compare_gt(gt_graph, metadata["gt_path"], metadata["anisotropy"], metadata["min_cable_length"],
                        metadata["node_spacing"], workers=request.gt_workers)
    log(f"GT check: {format_report(report)}")
    if report["stale"]:
        raise StaleGroundTruth(
            f"brain {request.brain}: cached GT differs from GCS ({format_report(report)}); "
            "rerun with --rebuild-base to rebuild from the current tracings, or --no-verify-gt to keep it")
    return report


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

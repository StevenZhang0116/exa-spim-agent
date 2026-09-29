#!/usr/bin/env python
"""
Bake canonical GT error labels into existing dataset caches.

For each ``dataset_cache_<brain_id>_mcl<N>.pkl`` this:

  1. loads the cache (reusing the graphs already in it -- no SWC rebuild),
  2. resolves the brain's dense segmentation path from ``brain_id``
     (via ``dataset_config.get_segmentation_id`` + the standard GCS template),
  3. reads the segmentation at every GT node's voxel, stores the canonical
     per-node label and per-edge error class on ``gt_graph``,
    4. combines geometric-walk sites and two-GT junction sites in gt_merge_sites,
    5. writes a labelled cache alongside the original. The output is named
     ``dataset_cache_<brain>_mcl<N>_add.pkl`` -- the original ``.pkl`` is left
     untouched, so this step is safe to re-run and easy to roll back.

Full relabeling is the ONE cloud-dependent step (needs GOOGLE_APPLICATION_CREDENTIALS and
read access to gs://allen-nd-goog/...). Run it once per brain; every later load
is cache-only.

--refresh-merge-sites selects existing _add.pkl caches instead, reuses their
GT node labels and geometric sites, and rebuilds the combined merge sites.
It performs no cloud reads and atomically replaces the selected _add.pkl.
There is one active site list, not parallel label versions. Old model/CSV results
must be regenerated after refreshing labels. Run all Python on a compute node.

Verification: pass ``--results-dir metrics_out`` to compare the freshly computed
cache-only summary against the canonical ``results.csv`` for that brain. The
cache-only numbers will differ from canonical wherever fragment filtering or the
voxel/axis convention diverge -- the comparison is the gate for getting the
labeling right before trusting the stored numbers.
Combined site counts include two-GT junction locations, so they need not equal
the original geometric-only canonical merge-event count.

Usage
-----
    python relabel_cache.py --cache-dir ../cache                 # all caches
    python relabel_cache.py --cache-dir ../cache --brain 794495  # one brain
    python relabel_cache.py --cache-dir ../cache --dry-run       # resolve paths only
    python relabel_cache.py --cache-dir ../cache --results-dir ../metrics_out
    python relabel_cache.py --cache-dir ../cache --brain 794495 --mcl 100 --refresh-merge-sites
"""

import argparse
import glob
import hashlib
import json
import os
import pickle
import re
import stat
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# --- Config / credentials for reference segmentation metrics ----------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_DIR = os.path.abspath(os.path.join(_HERE, "..", "configs"))
os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.join(_CONFIG_DIR, "zihan_gcs_token.json"),
)
os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")

sys.path.insert(0, _HERE)  # for dataset_config
from dataset_config import get_segmentation_id  # noqa: E402

_CONFIG_RTF = os.path.join(_CONFIG_DIR, "segmentation_datasets.rtf")

from agentic_neuron_proofreader.data_modules import canonical_labeling  # noqa: E402
from agentic_neuron_proofreader.data_modules.datasets import BrainDataset  # noqa: E402


def brain_id_from_path(path):
    m = re.search(r"dataset_cache_(\d+)_mcl(\d+)\.pkl$", os.path.basename(path))
    return m.group(1) if m else None


def mcl_from_path(path):
    """min_cable_length (int) parsed from dataset_cache_<brain>_mcl<N>.pkl, or None."""
    m = re.search(r"dataset_cache_\d+_mcl(\d+)\.pkl$", os.path.basename(path))
    return int(m.group(1)) if m else None


def segmentation_path_for(brain_id):
    """brain_id -> dense segmentation volume path (standard GCS template)."""
    segmentation_id = get_segmentation_id(brain_id, rtf_path=_CONFIG_RTF)
    return (
        f"gs://allen-nd-goog/from_google/{brain_id}"
        f"/whole_brain/{segmentation_id}/"
    )


def load_canonical_results(results_dir, brain_id):
    """Best-effort load of metrics_out/<brain_id>/<seg_id>/results.csv."""
    base = os.path.join(results_dir, str(brain_id))
    hits = glob.glob(os.path.join(base, "*", "results.csv"))
    if not hits:
        return None
    return pd.read_csv(sorted(hits)[0], index_col=0)


def cache_signature(path):
    info = Path(path).stat()
    return info.st_size, info.st_mtime_ns, info.st_ino


def atomic_cache_write(path, writer, expected_signature=None):
    """Replace a cache only after a complete write; preserve it on failure."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        writer(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if expected_signature is not None and cache_signature(path) != expected_signature:
            raise RuntimeError("Cache changed during regeneration; refusing to replace it")
        if path.exists():
            os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def refresh_payload_merge_sites(payload, verbose=True):
    """Reuse canonical node labels and geometric positives, rebuilding one site list."""
    required = ("gt_node_canonical_label", "gt_merge_labels", "gt_merge_sites")
    if any(payload.get(key) is None for key in required):
        raise ValueError("Refresh requires an already labelled _add.pkl with merge labels and sites")
    dataset = BrainDataset(None, None, None, _skip_load=True)
    dataset.fragments_graph = payload["fragments_graph"]
    dataset.gt_graph = payload["gt_graph"]
    if dataset.fragments_graph is None or dataset.gt_graph is None:
        raise ValueError("Refresh requires both skeleton graphs")
    dataset.gt_graph.node_label = payload["gt_node_canonical_label"]
    dataset.gt_graph.merge_labels = payload["gt_merge_labels"]
    dataset.gt_graph.merge_sites = payload["gt_merge_sites"]
    previous_site_count = len(payload["gt_merge_sites"])
    old_merge_labels = set(map(int, payload["gt_merge_labels"]))
    dataset.refresh_merge_labels(reuse_geometric_sites=True, verbose=verbose)
    if not old_merge_labels.issubset(set(map(int, dataset.gt_graph.merge_labels))):
        raise RuntimeError("Refresh would remove existing merge positives")
    site_identity = sorted((int(site["segment_id"]), tuple(map(float, site["xyz"])),
                            str(site["gt_neuron"]), site.get("source", "geometric_walk"))
                           for site in dataset.gt_graph.merge_sites)
    metadata = dict(dataset.gt_graph.merge_site_metadata)
    metadata["site_identity_sha256"] = hashlib.sha256(
        json.dumps(site_identity, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    dataset.gt_graph.merge_site_metadata = metadata
    payload.update({
        "gt_edge_error": dataset.gt_graph.edge_error,
        "gt_merge_labels": dataset.gt_graph.merge_labels,
        "gt_merge_sites": dataset.gt_graph.merge_sites,
        "gt_merge_site_metadata": metadata,
        "gt_junction_audit": None,
    })
    payload.pop("__detector_sample_universe_cache__", None)
    payload.pop("__detector_universe_site_audit__", None)
    return {"previous_site_count": previous_site_count, **metadata}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True,
                    help="Directory of dataset_cache_*.pkl files.")
    ap.add_argument("--brain", default=None,
                    help="Relabel only this brain_id (default: all in cache-dir).")
    ap.add_argument("--mcl", type=int, default=None,
                    help="Relabel only caches with this min_cable_length "
                         "(e.g. 10 -> only dataset_cache_*_mcl10.pkl). Default: all "
                         "mcl values found.")
    ap.add_argument("--refresh-merge-sites", action="store_true",
                    help="Refresh existing _add.pkl in place using stored GT labels; no cloud reads")
    ap.add_argument("--results-dir", default=None,
                    help="metrics_out/ root; if given, verify against results.csv.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve segmentation paths and exit (no cloud read).")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress the per-batch progress bar.")
    args = ap.parse_args(argv)

    pattern = os.path.join(args.cache_dir, "dataset_cache_*_add.pkl" if args.refresh_merge_sites
                           else "dataset_cache_*.pkl")
    def base_path(path):
        return path[:-8] + ".pkl" if args.refresh_merge_sites else path

    paths = sorted(p for p in glob.glob(pattern) if brain_id_from_path(base_path(p)) is not None)
    if args.brain:
        paths = [p for p in paths if brain_id_from_path(base_path(p)) == str(args.brain)]
    if args.mcl is not None:
        paths = [p for p in paths if mcl_from_path(base_path(p)) == args.mcl]
    if not paths:
        raise SystemExit(f"No caches matched {pattern}"
                         + (f" for brain {args.brain}" if args.brain else "")
                         + (f" with mcl{args.mcl}" if args.mcl is not None else ""))

    for path in paths:
        brain_id = brain_id_from_path(base_path(path))
        if brain_id is None:
            print(f"[skip] cannot parse brain_id from {os.path.basename(path)}")
            continue
        if args.refresh_merge_sites:
            print(f"\n=== brain {brain_id}: refresh merge sites ===\n  cache: {path}", flush=True)
            if args.dry_run:
                print("  would atomically replace this _add.pkl; no cloud reads", flush=True)
                continue
            signature = cache_signature(path)
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            summary = refresh_payload_merge_sites(payload, verbose=not args.quiet)
            print(json.dumps(summary, indent=2), flush=True)

            def write_payload(destination):
                with open(destination, "wb") as handle:
                    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

            atomic_cache_write(path, write_payload, expected_signature=signature)
            del payload
            print(f"  refreshed {path}; rebuild sweep/CSV/model outputs before using new labels", flush=True)
            continue
        seg_path = segmentation_path_for(brain_id)
        print(f"\n=== brain {brain_id} ===")
        print(f"  cache:        {path}")
        print(f"  segmentation: {seg_path}")
        if args.dry_run:
            continue

        ds = BrainDataset.load_from_cache(path)
        ds.label_gt_from_segmentation(seg_path, verbose=not args.quiet)

        summary = canonical_labeling.summarize(
            ds.gt_graph, ds.gt_graph.node_label, ds.gt_graph.edge_error
        )
        print(f"  -> splits={summary['total_splits']} "
              f"merges={summary['total_merges']} "
              f"%split={summary['pct_split_edges']:.2f} "
              f"%omit={summary['pct_omit_edges']:.2f} "
              f"%merged={summary['pct_merged_edges']:.2f}")

        if args.results_dir:
            canon = load_canonical_results(args.results_dir, brain_id)
            if canon is not None:
                tot_splits = canon["# Splits"].sum()
                tot_merges = canon["# Merges"].sum()
                print(f"  canonical results.csv: splits={tot_splits:.0f} "
                      f"merges={tot_merges:.0f}  (per-neuron rows: {len(canon)})")
            else:
                print(f"  [warn] no results.csv under {args.results_dir}/{brain_id}")

        out_path = path[:-4] + "_add.pkl" if path.endswith(".pkl") else path + "_add"
        atomic_cache_write(out_path, ds.save)
        print(f"  saved canonical labels into {os.path.basename(out_path)} "
              f"(original left untouched)")


if __name__ == "__main__":
    main()

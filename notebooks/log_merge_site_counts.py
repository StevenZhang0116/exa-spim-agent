"""Log GT merge-site and positive junction counts for every dataset at one MCL.

Run in panda on a compute node; only load trusted _add.pkl caches. Reuses the
reviewed detector target adapter to enumerate candidates and attach labels.
No detector CSV, trained model, image reader or label-cache rewrite is needed.
Defaults match the current junction policy: segment-scoped Euclidean NMS 20 um,
positive cable distance <=20 um, claim distance <=150 um, site snap <=50 um.
These are explicit analysis settings, not a newly selected/validated policy.

GT records, positive candidates and GT sites covered within the positive radius
are different counting units. Coincident GT records are retained. Ring candidates
have distance >positive and <=claim radius and remain label 0, not verified
non-merges. An absent GT list is an error; an explicitly empty list is valid.

Examples (on a compute node, from the repository root):
    python notebooks/log_merge_site_counts.py --mcl 100
    python notebooks/log_merge_site_counts.py --mcl 10 --brains 794495 794493

Caches are processed sequentially. A fresh run under notebooks/merge_site_counts/
contains summary.csv, run.json and per_brain/<brain>.json. Errors are logged per
dataset and other datasets continue; any error yields exit status 1. Existing
outputs are never overwritten. No candidate-to-GT association table is generated.
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic.detector_build.contracts import DetectorTarget
from agentic.detector_build.target_runtime import target_adapter_source
from notebooks.merge_topology_diagnosis import brain_id_from_cache, discover_add_caches, load_payload


SUMMARY_COLUMNS = (
    "brain_id", "mcl", "status", "n_gt_merge_sites", "n_is_merge_site_1",
    "n_candidates", "n_in_ambiguous_ring_1", "n_negative", "n_junctions_raw",
    "n_gt_sites_covered_at_positive_radius", "gt_site_recall_at_positive_radius",
    "n_gt_sites_only_within_claim_radius", "n_gt_sites_not_covered_at_claim_radius",
    "n_gt_sites_unsnappable", "nms_um", "positive_radius_um", "claim_radius_um",
    "cache", "error",
)


def make_runtime(nms_um=20., positive_radius_um=20., claim_radius_um=150.):
    values = (nms_um, positive_radius_um, claim_radius_um)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Policy distances must be finite")
    if nms_um < 0 or not 0 < positive_radius_um <= claim_radius_um:
        raise ValueError("Require nms >= 0 and 0 < positive radius <= claim radius")
    policy = {
        "config_id": f"junction|nms={nms_um:g}|r={claim_radius_um:g}", "mode": "junction",
        "nms_um": nms_um, "positive_label_radius_um": positive_radius_um,
        "claim_radius_um": claim_radius_um,
    }
    policy_sha256 = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    source = target_adapter_source(DetectorTarget.MERGE_SITE, candidate_policy=policy,
                                   candidate_policy_sha256=policy_sha256)

    def build_comp_to_seg(graph):
        return {int(component): int(str(swc).split(".")[0])
                for component, swc in graph.component_id_to_swc_id.items()}

    runtime = {"np": np, "build_comp_to_seg": build_comp_to_seg}
    exec(compile(source, "<reviewed merge-site target adapter>", "exec"), runtime)
    provenance = {
        "settings": policy, "settings_sha256": policy_sha256,
        "adapter_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "site_snap_max_um": runtime["CANDIDATE_SITE_SNAP_MAX_UM"],
    }
    return runtime, provenance


def summarize_payload(payload, mcl, runtime):
    """Recompute labels without mutating the payload or trusting cached universes."""
    if float(payload.get("min_cable_length", mcl)) != mcl:
        raise ValueError("Cache min_cable_length disagrees with requested MCL")
    sites = payload.get("gt_merge_sites", getattr(payload.get("gt_graph"), "merge_sites", None))
    if sites is None:
        raise ValueError("Cache has no gt_merge_sites; missing labels are not zero sites")
    working = {key: value for key, value in payload.items()
               if key not in ("__detector_sample_universe_cache__", "__detector_universe_site_audit__")}
    working["gt_merge_sites"] = list(sites)
    runtime["_merge_site_provenance"](working)
    samples, labels = runtime["build_sample_universe"](working)
    audit = runtime["sample_universe_audit"](working, samples, labels)
    distances = np.asarray(working["__detector_universe_site_audit__"]["site_min_distances"], dtype=float)
    positive_radius = runtime["CANDIDATE_POSITIVE_LABEL_RADIUS_UM"]
    n_covered = int(np.sum(distances <= positive_radius))
    n_sites = audit["n_sites"]
    summary = {
        "n_gt_merge_sites": n_sites, "n_is_merge_site_1": audit["n_positive"],
        "n_candidates": audit["n_rows"], "n_in_ambiguous_ring_1": audit["n_ambiguous_ring"],
        "n_negative": audit["n_negative"], "n_junctions_raw": audit["n_junctions_raw"],
        "n_gt_sites_covered_at_positive_radius": n_covered,
        "gt_site_recall_at_positive_radius": n_covered / n_sites if n_sites else None,
        "n_gt_sites_only_within_claim_radius": audit["n_sites_covered_at_claim_radius"] - n_covered,
        "n_gt_sites_not_covered_at_claim_radius": n_sites - audit["n_sites_covered_at_claim_radius"],
        "n_gt_sites_unsnappable": n_sites - audit["n_sites_snapped"],
    }
    site_distances = [
        {"site_index": index, "segment_id": int(site["segment_id"]),
         "distance_to_nearest_candidate_um": float(distance) if np.isfinite(distance) else None,
         "covered_at_positive_radius": bool(distance <= positive_radius)}
        for index, (site, distance) in enumerate(zip(working["gt_merge_sites"], distances))
    ]
    return summary, {"sample_universe_audit": audit, "site_distances": site_distances,
                     "null_distance_meaning": "No kept candidate reachable within claim radius, or site unsnappable",
                     "gt_sites_storage": "gt_merge_sites" if "gt_merge_sites" in payload else "gt_graph.merge_sites"}


def cache_identity(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def inspect_cache(path, mcl, runtime):
    identity = cache_identity(path)
    payload = load_payload(path)
    try:
        summary, detail = summarize_payload(payload, mcl, runtime)
        if cache_identity(path) != identity:
            raise RuntimeError("Cache changed during inspection; retry with stable inputs")
        return summary, {**detail, "cache_identity": identity}
    finally:
        del payload
        gc.collect()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcl", type=int, required=True)
    parser.add_argument("--brains", "--brain", nargs="+", help="Default: all datasets with an _add.pkl at this MCL")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    parser.add_argument("--nms-um", type=float, default=20.)
    parser.add_argument("--positive-radius-um", type=float, default=20.)
    parser.add_argument("--claim-radius-um", type=float, default=150.)
    parser.add_argument("--output-dir", type=Path, help="New or empty output directory")
    args = parser.parse_args(argv)
    if args.mcl < 0:
        parser.error("MCL must be nonnegative")
    if args.brains and any(not brain.isascii() or not brain.isdigit() for brain in args.brains):
        parser.error("Brain IDs must be numeric")
    try:
        runtime, policy = make_runtime(args.nms_um, args.positive_radius_um, args.claim_radius_um)
        caches = discover_add_caches(args.cache_dir, args.mcl, args.brains)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if args.output_dir is not None:
        output = args.output_dir
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            parser.error("--output-dir must be empty; existing outputs will not be overwritten")
        output.mkdir(parents=True, exist_ok=True)
    else:
        parent = ROOT / "notebooks" / "merge_site_counts"
        parent.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=f"mcl{args.mcl}_", dir=parent))
    (output / "per_brain").mkdir()
    run = {"mcl": args.mcl, "started_at": datetime.now(timezone.utc).isoformat(),
           "caches": [str(path.resolve()) for path in caches], "policy": policy,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (output / "run.json").open("x") as handle:
        json.dump(run, handle, indent=2, allow_nan=False)
    print(f"Output directory: {output}", flush=True)
    print(f"Policy: NMS={args.nms_um:g}, positive={args.positive_radius_um:g}, "
          f"claim={args.claim_radius_um:g}, snap={policy['site_snap_max_um']:g} um", flush=True)
    failures = 0
    with (output / "summary.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        handle.flush()
        for path in caches:
            brain = brain_id_from_cache(path)
            row = {"brain_id": brain, "mcl": args.mcl, "cache": str(path.resolve()),
                   "nms_um": args.nms_um, "positive_radius_um": args.positive_radius_um,
                   "claim_radius_um": args.claim_radius_um, "status": "ok", "error": ""}
            try:
                summary, detail = inspect_cache(path, args.mcl, runtime)
                row.update(summary)
                with (output / "per_brain" / f"{brain}.json").open("x") as report:
                    json.dump({**row, **detail}, report, indent=2, allow_nan=False)
                print(f"{brain} mcl{args.mcl}: gt_merge_sites={row['n_gt_merge_sites']}, "
                      f"is_merge_site==1={row['n_is_merge_site_1']}, "
                      f"in_ambiguous_ring==1={row['n_in_ambiguous_ring_1']}, "
                      f"GT covered at positive radius={row['n_gt_sites_covered_at_positive_radius']}", flush=True)
            except Exception as error:
                failures += 1
                row.update(status="error", error=f"{type(error).__name__}: {error}")
                print(f"{brain} mcl{args.mcl}: ERROR {row['error']}", flush=True)
            writer.writerow(row)
            handle.flush()
            gc.collect()
    print(f"Finished: {len(caches) - failures} succeeded, {failures} failed. {output / 'summary.csv'}", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
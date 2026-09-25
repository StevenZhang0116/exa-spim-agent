"""Audit GT branch support for merge junctions without changing labels or scores.

Run in panda on a compute node. Load only trusted pickle caches. The output is
versioned review evidence, not a new training target or a proof of non-merge.
"""

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np

try:
    from scripts.plot_top_merge_predictions import DEFAULT_SCORE, candidate_center, rank_predictions
except ModuleNotFoundError:
    from plot_top_merge_predictions import DEFAULT_SCORE, candidate_center, rank_predictions


def audit_predictions(payload, predictions):
    from agentic_neuron_proofreader.data_modules.canonical_labeling import audit_junction_gt_connections

    labels = payload.get("gt_node_canonical_label")
    if labels is None:
        raise ValueError("Cache must contain canonical GT node labels")
    fragment = payload["fragments_graph"]
    for prediction in predictions:
        candidate_center(fragment, prediction)
    audit = audit_junction_gt_connections(
        fragment, payload["gt_graph"], labels,
        [prediction["node_id"] for prediction in predictions],
    )
    by_node = {record["node_id"]: record for record in audit["junctions"]}
    merge_labels = payload.get("gt_merge_labels")
    merge_segments = None if merge_labels is None else set(map(int, merge_labels))
    sites = payload.get("gt_merge_sites") or []
    records = []
    for prediction in predictions:
        record = dict(by_node[prediction["node_id"]])
        segment = prediction["segment_id"]
        own_sites = [site for site in sites if int(site["segment_id"]) == segment]
        distance = prediction.get("distance_to_nearest_gt_site_um", float("nan"))
        record.update({
            "candidate_id": prediction["candidate_id"],
            "score_rank": prediction.get("score_rank"),
            "score": prediction["score"],
            "original_is_merge_site": prediction["is_merge_site"],
            "original_in_ambiguous_ring": prediction.get("in_ambiguous_ring"),
            "original_gt_site_distance_um": distance if math.isfinite(distance) else None,
            "segment_gt_is_merge": None if merge_segments is None else segment in merge_segments,
            "recorded_segment_site_count": len(own_sites),
            "nearest_recorded_segment_site_euclidean_um": min(
                (float(np.linalg.norm(np.asarray(site["xyz"]) - record["xyz"]))
                for site in own_sites), default=None),
        })
        records.append(record)
    return {**audit, "junctions": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--candidate-id", nargs="+", type=int)
    selection.add_argument("--top-k", type=int)
    parser.add_argument("--score-column", default=DEFAULT_SCORE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; choose a new audit file")
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be positive")
    ranked = [{**row, "score_rank": rank} for rank, row in enumerate(
        rank_predictions(args.csv, None, args.score_column), 1)]
    if args.candidate_id is not None:
        wanted = set(args.candidate_id)
        predictions = [row for row in ranked if row["candidate_id"] in wanted]
        missing = wanted - {row["candidate_id"] for row in predictions}
        if missing:
            parser.error(f"Candidates missing or without finite scores: {sorted(missing)}")
    else:
        predictions = ranked[:args.top_k]
    if not predictions:
        parser.error("No finite scored candidates available")
    print(f"Loading trusted cache: {args.pkl}", flush=True)
    with args.pkl.open("rb") as handle:
        payload = pickle.load(handle)
    report = audit_predictions(payload, predictions)
    report["inputs"] = {
        "cache": str(args.pkl.resolve()), "csv": str(args.csv.resolve()),
        "cache_size_bytes": args.pkl.stat().st_size,
        "cache_mtime_ns": args.pkl.stat().st_mtime_ns,
        "csv_size_bytes": args.csv.stat().st_size,
        "csv_mtime_ns": args.csv.stat().st_mtime_ns,
        "score_column": args.score_column,
        "selection": "candidate_ids" if args.candidate_id is not None else "score_top_k",
    }
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(serialized)
    for record in report["junctions"]:
        print(f"candidate {record['candidate_id']}: {record['status']} "
              f"({record['reason']}); original site label={record['original_is_merge_site']}", flush=True)
    print(f"Audit only; labels and scores unchanged. Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
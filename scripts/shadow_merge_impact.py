#!/usr/bin/env python
"""Stage 3 of the shadow merge relabel: what the corrected merge sites change downstream.

Read-only. For each brain with an existing detector-native merge table it

1. rebuilds the frozen merge detector's candidate universe and labels with its own
   ``build_sample_universe`` (positive = geodesic cable <= 20 um to a merge site),
   once with the stored ``gt_merge_sites`` (must reproduce the table's
   ``evaluator_labels.npy`` exactly) and once with the scaled-threshold sites from
   ``scripts/shadow_merge_relabel.py``;
2. counts label flips;
3. scores the table's frozen ``detector_score`` under both label sets
   (Precision@K, average precision, positive coverage);
4. reports how many sites the frozen candidate policy can reach (claim radius
   150 um and the 20 um positive radius) under both site sets.

Nothing is written outside --out-dir; no table, cache or detector is modified.
Run with panda on a compute node:

    python -u scripts/shadow_merge_impact.py --brains 789202 794491 794493 794495 802449
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent / "agentic-neuron-proofreader" / "src"))
import agentic_neuron_proofreader  # noqa: E402,F401  (registers SkeletonGraph for pickle)

DEFAULT_DETECTOR = PROJECT_ROOT / "autodiscovery-application" / "merge-error-794495-mcl100_2026-08-04"
KS = (100, 500, 1000, 2000)


def load_detector(directory):
    script = directory / "merge_junction_detector.py"
    spec = importlib.util.spec_from_file_location("shadow_merge_detector", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def find_merge_table(tables_dir, brain, detector_dir):
    matches = []
    for meta_path in sorted((tables_dir / brain).glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        prov = meta.get("provenance", {})
        if prov.get("kind") == "merge" and prov.get("detector", {}).get("run_id") == detector_dir.name:
            matches.append(meta_path.parent)
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one merge table for {brain} from {detector_dir.name}, got {len(matches)}")
    return matches[0]


def precision_at(scores, labels, k):
    order = np.argsort(-scores, kind="stable")[:k]
    return float(labels[order].mean()) if len(order) else float("nan")


def average_precision(scores, labels):
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    hits = labels[order]
    cum = np.cumsum(hits)
    return float((cum[hits == 1] / (np.flatnonzero(hits) + 1)).mean())


def score_block(scores, labels):
    out = {"positives": int(labels.sum()), "prevalence": float(labels.mean()),
           "average_precision": average_precision(scores, labels)}
    for k in KS:
        out[f"p@{k}"] = precision_at(scores, labels, k)
    return out


def coverage(audit, module):
    d = np.asarray(audit["site_min_distances"], dtype=float)
    if not len(d):
        return {"n_sites": 0}
    return {"n_sites": int(len(d)),
            "covered_claim_150": float((d <= module.CANDIDATE_CLAIM_RADIUS_UM).mean()),
            "covered_positive_20": float((d <= module.CANDIDATE_POSITIVE_LABEL_RADIUS_UM).mean())}


def run_brain(brain, args, module):
    table_dir = find_merge_table(args.tables_dir, brain, args.detector_dir)
    shadow = json.loads((args.shadow_dir / f"{brain}_mcl{args.mcl}.json").read_text())
    cache = args.cache_dir / f"dataset_cache_{brain}_mcl{args.mcl}_add.pkl"
    if shadow["cache"]["size"] != cache.stat().st_size or shadow["cache"]["mtime_ns"] != cache.stat().st_mtime_ns:
        raise RuntimeError(f"{brain}: cache changed since the shadow relabel")
    print(f"\n=== {brain}: {table_dir.name[:12]} | loading {cache.name}", flush=True)
    with open(cache, "rb") as f:
        payload = pickle.load(f)
    base = {"fragments_graph": payload["fragments_graph"], "gt_graph": payload["gt_graph"],
            "anisotropy": payload["anisotropy"], "gt_merge_site_metadata": None}
    stored_sites = list(payload["gt_merge_sites"] or [])
    del payload
    gc.collect()

    table_candidates = pickle.load(open(table_dir / "candidates.pkl", "rb"))
    table_labels = np.load(table_dir / "evaluator_labels.npy").astype(int)
    scores = pd.read_pickle(table_dir / "features.pkl")["detector_score"].to_numpy(dtype=float)

    results = {}
    for name, sites in (("stored", stored_sites), ("scaled", shadow["variants"]["scaled"]["sites"])):
        # build_sample_universe records its site audit on the dict it is given.
        view = {**base, "gt_merge_sites": sites}
        rows, labels = module.build_sample_universe(view)
        results[name] = {"rows": rows, "labels": np.asarray(labels, dtype=int),
                         "coverage": coverage(view["__detector_universe_site_audit__"], module)}

    same_rows = [r["node_id"] for r in results["stored"]["rows"]] == [c["node_id"] for c in table_candidates]
    reproduced = same_rows and np.array_equal(results["stored"]["labels"], table_labels)
    old, new = results["stored"]["labels"], results["scaled"]["labels"]
    record = {
        "brain": brain, "table": str(table_dir), "rows": int(len(old)),
        "reproduces_table_labels": bool(reproduced),
        "flips_0_to_1": int(((old == 0) & (new == 1)).sum()),
        "flips_1_to_0": int(((old == 1) & (new == 0)).sum()),
        "stored": {**score_block(scores, old), **results["stored"]["coverage"]},
        "scaled": {**score_block(scores, new), **results["scaled"]["coverage"]},
    }
    # Where do the new positives rank under the frozen detector?
    rank = np.empty(len(scores), int)
    rank[np.argsort(-scores, kind="stable")] = np.arange(len(scores))
    flipped = (old == 0) & (new == 1)
    record["new_positive_rank_median"] = float(np.median(rank[flipped])) if flipped.any() else None
    record["new_positive_in_top2000"] = int((rank[flipped] < 2000).sum())
    print(json.dumps({k: v for k, v in record.items() if k != "table"}, indent=1), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brains", nargs="+", required=True)
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--detector-dir", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "cache")
    parser.add_argument("--tables-dir", type=Path, default=PROJECT_ROOT / "proofreader_evolve" / "feature_tables")
    parser.add_argument("--shadow-dir", type=Path, default=PROJECT_ROOT / "notebooks" / "merge_label_shadow")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "notebooks" / "merge_label_shadow" / "impact")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    module = load_detector(args.detector_dir)

    rows, failed = [], False
    for brain in dict.fromkeys(args.brains):
        try:
            rec = run_brain(brain, args, module)
            (args.out_dir / f"{brain}.json").write_text(json.dumps(rec, indent=1))
            for name in ("stored", "scaled"):
                rows.append({"brain": brain, "labels": name, "reproduced": rec["reproduces_table_labels"],
                             "flips+": rec["flips_0_to_1"], "flips-": rec["flips_1_to_0"],
                             **{k: rec[name].get(k) for k in ("positives", "average_precision",
                                "p@100", "p@500", "p@2000", "n_sites", "covered_claim_150", "covered_positive_20")}})
        except Exception as exc:
            failed = True
            print(f"ERROR {brain}: {type(exc).__name__}: {exc}", flush=True)
        gc.collect()
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "summary.csv", index=False)
    print("\nSummary\n" + table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

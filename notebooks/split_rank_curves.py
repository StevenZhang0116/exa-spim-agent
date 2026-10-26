"""Precision and recall of the frozen split detector beyond the old K = 1000 window.

Reads the Stage 3 split feature tables (``features.pkl`` for ``detector_score``,
``evaluator_labels.npy`` and the ``segment_on_gt`` availability sidecar) and ranks the
GT-labeled rows by the frozen detector score, exactly as ``fixed_pool_scoring`` does
(NaN rows never ranked). Per brain it prints Precision@K and Recall@K at K = 1000 and
at multiples of the pool's GT positives, recall at fixed precision floors, the labeled
average precision, precision by labeled rank band, and how many unlabeled rows the
full-pool ranking places above the K-th labeled row. The 2026-10-26 run of this script
(n13, five brains) motivated objective v4: split K = GT positives, no cap
(``proofreader_evolve/WORKFLOW_REVISION.md``, "Split K = GT positives").

Compute node, one brain at a time (the 794491 table is 4.3 GB):

    conda activate panda
    python notebooks/split_rank_curves.py --brains 789202 794493 794491 794495 802449

Writes ``notebooks/split_rank_curves/summary.csv`` (gitignored).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TABLES = os.path.join(REPO, "proofreader_evolve", "feature_tables")
OUT_DIR = os.path.join(HERE, "split_rank_curves")
MULTIPLES = (("K1000", None), ("K=P/2", 0.5), ("K=P", 1.0), ("K=1.5P", 1.5), ("K=2P", 2.0))
FLOORS = (0.95, 0.9, 0.8)
BANDS = (0, 1000, 1500, 2000, 2500, 3000, 4000, 5000)


def split_table(brain):
    for sidecar in glob.glob(os.path.join(TABLES, brain, "*.availability")):
        with open(os.path.join(sidecar, "meta.json")) as stream:
            if json.load(stream)["kind"] == "split":
                return sidecar[:-len(".availability")]
    return None


def curves(brain):
    table = split_table(brain)
    if table is None:
        print(brain, "no split table", file=sys.stderr)
        return None
    features = pd.read_pickle(os.path.join(table, "features.pkl"))
    score = features["detector_score"].to_numpy(dtype=float)
    del features
    truth = np.load(os.path.join(table, "evaluator_labels.npy")).astype(float)
    truth[~np.load(os.path.join(table + ".availability", "availability.npz"))["segment_on_gt"]] = np.nan
    order = np.argsort(-score, kind="stable")
    labeled = ~np.isnan(truth)
    labeled_order = order[labeled[order]]
    y = truth[labeled_order]
    positives, n_labeled = int(y.sum()), len(y)
    cum_tp = np.cumsum(y)
    precision = cum_tp / np.arange(1, n_labeled + 1)
    recall = cum_tp / positives
    rank_in_pool = np.empty(len(score), dtype=np.int64)
    rank_in_pool[order] = np.arange(len(score))
    labeled_ranks = rank_in_pool[labeled_order]
    row = {"brain": brain, "pool": len(score), "labeled": n_labeled, "positives": positives,
           "base_rate": positives / n_labeled, "AP_labeled": float(np.sum(precision * y) / positives)}
    print(f"\n== {brain} split: pool={len(score)} labeled={n_labeled} positives={positives} "
          f"base_rate={row['base_rate']:.3f} AP(labeled)={row['AP_labeled']:.4f}")
    for name, multiple in MULTIPLES:
        k = min(1000 if multiple is None else int(np.ceil(multiple * positives)), n_labeled)
        row[f"{name}_k"], row[f"{name}_P"], row[f"{name}_R"] = k, precision[k - 1], recall[k - 1]
        row[f"{name}_unlabeled_above"] = int(labeled_ranks[k - 1] + 1 - k)
        print(f"  {name:7s} k={k:5d} P@K={precision[k - 1]:.4f} R@K={recall[k - 1]:.4f} "
              f"unlabeled_rows_above_kth_labeled={row[f'{name}_unlabeled_above']}")
    for floor in FLOORS:
        deep = np.where(precision >= floor)[0]
        label = f"P{int(floor * 100)}"
        row[f"R@{label}"], row[f"K@{label}"] = (recall[deep[-1]], int(deep[-1]) + 1) if len(deep) else (0., 0)
        print(f"  R@P>={floor:.2f}: recall={row[f'R@{label}']:.4f} at labeled depth K={row[f'K@{label}']}")
    edges = [edge for edge in BANDS if edge < n_labeled] + [n_labeled]
    print("  band precision:", ", ".join(f"[{edges[i]},{edges[i + 1]}):{y[edges[i]:edges[i + 1]].mean():.3f}"
                                         for i in range(len(edges) - 1)))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--brains", nargs="+", default=["789202", "794493", "794491"])
    args = parser.parse_args()
    rows = [row for row in (curves(brain) for brain in args.brains) if row is not None]
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    print("\nwrote", os.path.join(OUT_DIR, "summary.csv"))


if __name__ == "__main__":
    main()

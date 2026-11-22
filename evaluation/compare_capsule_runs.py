"""Compare two outputs of the capsule's evaluation of the same brain.

Each argument is an ``<out>`` directory of ``run_capsule_original.py`` (it holds
``original/original_results.csv`` and ``original/original_merge_sites.csv``). Printed and
written to ``--out``: the totals, the per-neuron table, and the merge sites the two runs share.
Sites are matched one to one by position within ``--radius`` um.

With ``--cuts`` (the ``*_cuts.csv`` of ``cut_merge_sites.py``, B being the evaluation of the
cut graph): how many merge sites of A have a cut nearby, and how many of those B no longer has.
When the file carries the candidates' labels (a policy's cuts), the same is given apart for the
cuts labeled as merges and for those labeled 0.

Run on a compute node (panda or arborist_eval), from the repository root:

    python evaluation/compare_capsule_runs.py A=<dir> B=<dir> --out <file.txt> [--cuts <csv>]
"""
import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

COUNTS = ("# Merges", "# Splits", "# Truncations")
SHARES = ("% Merged Edges", "% Omit Edges", "% Split Edges", "Edge Accuracy", "ERL", "Normalized ERL")


def load(directory):
    step = Path(directory) / "original"
    results = pd.read_csv(step / "original_results.csv", index_col=0)
    sites_path = step / "original_merge_sites.csv"
    sites = pd.read_csv(sites_path) if sites_path.exists() else pd.DataFrame(columns=["World", "GroundTruth_ID", "Segment_ID"])
    return results, sites


def site_positions(sites):
    if not len(sites):
        return np.zeros((0, 3))
    return np.array([ast.literal_eval(value) for value in sites["World"]], dtype=float)


def match_sites(xyz_a, xyz_b, radius):
    """Pairs (index in A, index in B) of sites matched one to one within ``radius`` um."""
    if not len(xyz_a) or not len(xyz_b):
        return []
    distance = cdist(xyz_a, xyz_b)
    rows, columns = linear_sum_assignment(np.where(distance <= radius, distance, 1e9))
    return [(r, c) for r, c in zip(rows, columns) if distance[r, c] <= radius]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs=2, metavar="NAME=DIR")
    parser.add_argument("--radius", type=float, default=20.0, help="Site matching radius in um (default 20)")
    parser.add_argument("--cuts", type=Path, default=None, help="Cut sites applied between A and B")
    parser.add_argument("--cut-radii", type=float, nargs="+", default=[20.0, 50.0, 100.0],
                        help="Distances in um at which a merge site of A counts as having a cut nearby")
    parser.add_argument("--out", type=Path, default=None)
    options = parser.parse_args()
    (name_a, dir_a), (name_b, dir_b) = (item.split("=", 1) for item in options.runs)
    (res_a, sites_a), (res_b, sites_b) = load(dir_a), load(dir_b)
    if sorted(res_a.index) != sorted(res_b.index):
        sys.exit("the two runs do not hold the same ground-truth neurons")
    res_b = res_b.loc[res_a.index]

    lines = [f"{name_a}: {dir_a}", f"{name_b}: {dir_b}", "", f"{'':24s}{name_a:>14s}{name_b:>14s}"]
    for column in COUNTS:
        if column in res_a and column in res_b:
            lines.append(f"{column + ', total':24s}{int(res_a[column].sum()):>14d}{int(res_b[column].sum()):>14d}")
    for column in SHARES:
        lines.append(f"{column + ', mean':24s}{res_a[column].mean():>14.4f}{res_b[column].mean():>14.4f}")
    lines.append(f"{'SWC run length, total':24s}{res_a['SWC Run Length'].sum():>14.0f}{res_b['SWC Run Length'].sum():>14.0f}")

    table = pd.DataFrame(index=res_a.index)
    for column, short in (("# Merges", "merges"), ("# Splits", "splits"), ("% Merged Edges", "merged%"),
                          ("% Omit Edges", "omit%"), ("Edge Accuracy", "accuracy"), ("ERL", "ERL")):
        table[f"{short}_{name_a}"] = res_a[column].round(2)
        table[f"{short}_{name_b}"] = res_b[column].round(2)
    lines += ["", table.to_string()]

    same = int((res_a["# Merges"] == res_b["# Merges"]).sum())
    lines += ["", f"neurons with the same merge count: {same} of {len(res_a)}"]
    for column in ("% Merged Edges", "% Omit Edges", "Edge Accuracy"):
        difference = res_b[column] - res_a[column]
        lines.append(f"{column}: mean difference {name_b} - {name_a} {difference.mean():+.2f} points, "
                     f"largest |difference| {difference.abs().max():.2f} ({difference.abs().idxmax()})")

    xyz_a, xyz_b = site_positions(sites_a), site_positions(sites_b)
    pairs = match_sites(xyz_a, xyz_b, options.radius)
    lines += ["", f"merge sites: {name_a} {len(xyz_a)}, {name_b} {len(xyz_b)}; within {options.radius:g} um one to one: "
                  f"{len(pairs)} in both, {len(xyz_a) - len(pairs)} only {name_a}, {len(xyz_b) - len(pairs)} only {name_b}"]

    if options.cuts:
        cuts = pd.read_csv(options.cuts)
        cuts = cuts[~cuts["action"].astype(str).str.startswith("skipped")]
        cut_xyz = cuts[["x_um", "y_um", "z_um"]].to_numpy(dtype=float)
        to_cut = cdist(xyz_a, cut_xyz).min(axis=1) if len(xyz_a) and len(cut_xyz) else np.full(len(xyz_a), np.inf)
        to_site = cdist(cut_xyz, xyz_a).min(axis=1) if len(xyz_a) and len(cut_xyz) else np.full(len(cut_xyz), np.inf)
        remaining = np.zeros(len(xyz_a), dtype=bool)
        remaining[[r for r, _ in pairs]] = True
        lines += ["", f"cuts applied: {len(cuts)} ({options.cuts})"]
        for radius in options.cut_radii:
            near = to_cut <= radius
            lines.append(f"within {radius:g} um: {int(near.sum())} of {len(xyz_a)} {name_a} sites have a cut nearby, "
                         f"{int((near & ~remaining).sum())} of them are gone in {name_b}; of the "
                         f"{int((~near).sum())} without one, {int((~near & ~remaining).sum())} are gone; "
                         f"{int((to_site > radius).sum())} cuts have no {name_a} site nearby")
        if "evaluator_label" in cuts and len(xyz_a):
            # A policy's cuts carry the candidate's label: which cuts take a merge site away?
            labeled = cuts["evaluator_label"].to_numpy() == 1
            reach = [cdist(xyz_a, cut_xyz[chosen]).min(axis=1) <= options.radius if chosen.any()
                     else np.zeros(len(xyz_a), dtype=bool) for chosen in (labeled, ~labeled)]
            only_unlabeled = reach[1] & ~reach[0]
            lines.append(f"by the candidates' labels, within {options.radius:g} um: {int(labeled.sum())} cuts labeled "
                         f"as merges, {int(reach[0].sum())} {name_a} sites near one, "
                         f"{int((reach[0] & ~remaining).sum())} of them gone; {int((~labeled).sum())} cuts labeled 0, "
                         f"{int(only_unlabeled.sum())} {name_a} sites near such cuts only, "
                         f"{int((only_unlabeled & ~remaining).sum())} of them gone; "
                         f"{int((to_site[~labeled] > options.radius).sum())} cuts labeled 0 have no {name_a} site nearby")

    text = "\n".join(lines)
    print(text)
    if options.out:
        options.out.parent.mkdir(parents=True, exist_ok=True)
        options.out.write_text(text + "\n")


if __name__ == "__main__":
    main()

"""Figure of the capsule's evaluation after a policy's joins, by the merge cuts made before.

Reads capsule outputs that already exist; nothing is evaluated here. ``--joined`` holds one
directory ``cut<C>_join<J>/`` per graph (``join_split_candidates.py --joins policy`` after the
``C`` highest-ranked merge cuts of a policy, ``C`` = 0 for none; each with ``fragments.json`` and
``capsule/``). The graph before the joins is the unedited one (``--unedited``) for ``C`` = 0 and
``top<C>/capsule`` under ``--cuts`` otherwise.

Written to ``--out`` (default ``evaluation/figure/``, which Git ignores), named after the brain:

* ``<brain>_split_policy_metrics.png``: merges, splits, merged edges, edge accuracy, ERL and the
  merge sites that the joins add, against the number of joins made, one line per cut depth;
* ``<brain>_split_policy_table.png``: the same numbers as a table, one row per graph with the
  unedited graph first, each figure marked better or worse than the unedited graph's;
* ``<brain>_split_policy.csv``: the plotted numbers.

Run on a compute node in panda, from the repository root:

    python evaluation/plot_split_policy_joins.py
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from compare_capsule_runs import load, match_sites, site_positions
from plot_merge_policy_cuts import (AXIS, GRID, INK, INK_SECONDARY, MUTED, RESULTS, SITE_RADIUS_UM, SURFACE, number, style,
                                    totals)

# The first three categorical slots of the dataviz palette, each with its own marker.
SERIES = (("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^"))
# Table cells: light steps of the palette's diverging pair (blue for better, red for worse).
BETTER, WORSE = "#cde2fb", "#f9dbdb"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unedited", type=Path, default=RESULTS / "841260_baseline_alignment_20261118/cache_graph")
    parser.add_argument("--cuts", type=Path, default=RESULTS / "841260_merge_policy_20261121")
    parser.add_argument("--joined", type=Path, default=RESULTS / "841260_split_policy_20261121")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "figure")
    return parser.parse_args()


def rows_of(options):
    found = sorted(tuple(map(int, re.fullmatch(r"cut(\d+)_join(\d+)", path.name).groups()))
                   for path in options.joined.glob("cut*_join*") if (path / "capsule/original").is_dir())
    rows = []
    for depth in sorted({cut for cut, _ in found}):
        before = options.unedited if depth == 0 else options.cuts / f"top{depth}/capsule"
        results, sites = load(before)
        sites_before = site_positions(sites)
        cuts_made = 0 if depth == 0 else json.loads((options.cuts / f"top{depth}/fragments.json").read_text())["sites"]["on_listed_segments"]
        rows.append({"cut_depth": depth, "cuts": cuts_made, "join_depth": 0, "joins": 0, "joins_labeled_as_splits": 0,
                     "joins_on_a_cut_fragment": 0, "rows_dropped_for_a_removed_node": 0, **totals(results),
                     "merge_sites_added_by_the_joins": 0, "merge_sites_gone_after_the_joins": 0})
        for _, join_depth in (item for item in found if item[0] == depth):
            directory = options.joined / f"cut{depth}_join{join_depth}"
            record = json.loads((directory / "fragments.json").read_text())
            results, sites = load(directory / "capsule")
            sites_after = site_positions(sites)
            shared = len(match_sites(sites_before, sites_after, SITE_RADIUS_UM))
            rows.append({"cut_depth": depth, "cuts": cuts_made, "join_depth": join_depth, "joins": record["joins"]["made"],
                         "joins_labeled_as_splits": record["joins"]["labeled_as_splits"],
                         "joins_on_a_cut_fragment": record["joins"]["on_a_cut_fragment"],
                         "rows_dropped_for_a_removed_node": record["candidates"]["dropped"].get("a_node_removed_by_a_cut", 0),
                         **totals(results), "merge_sites_added_by_the_joins": len(sites_after) - shared,
                         "merge_sites_gone_after_the_joins": len(sites_before) - shared})
    return pd.DataFrame(rows)


def figure_of(table, sources, brain, path):
    panels = [("merges", "Merges", "count, lower is better", 0),
              ("splits", "Splits", "count, lower is better", 0),
              ("merged_edges_pct", "Merged edges", "% of ground-truth edges, lower is better", 2),
              ("edge_accuracy", "Edge accuracy", "%, higher is better", 2),
              ("erl_um", "Expected run length", "um, higher is better", 0),
              ("merge_sites_added_by_the_joins", "Merge sites the joins add", "count, against the graph before the joins", 0)]
    depths = sorted(table["cut_depth"].unique())
    names = {depth: "no merge cuts" if depth == 0 else f"{int(table[table['cut_depth'] == depth]['cuts'].iloc[0]):,} merge cuts first"
             for depth in depths}
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), facecolor=SURFACE)
    figure.subplots_adjust(left=0.05, right=0.955, top=0.83, bottom=0.1, hspace=0.5, wspace=0.27)
    for axis, (column, title, note, decimals) in zip(axes.ravel(), panels):
        style(axis)
        ends = []
        for depth, (colour, marker) in zip(depths, SERIES):
            rows = table[table["cut_depth"] == depth]
            axis.plot(rows["joins"], rows[column], color=colour, linewidth=2, marker=marker, markersize=8,
                      markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            ends.append((float(rows[column].iloc[-1]), float(rows["joins"].iloc[-1])))
        axis.margins(x=0.05, y=0.14)
        # Direct labels at the right ends only, pushed apart where two lines end close together.
        low, high = axis.get_ylim()
        height = axis.get_position().height * figure.get_figheight() * 72
        placed = -1e9
        for value, x in sorted(ends):
            position = (value - low) / (high - low) * height
            placed = max(position, placed + 11)
            axis.annotate(number(value, decimals), (x, value), xytext=(9, placed - position), textcoords="offset points",
                          ha="left", va="center", fontsize=8.5, color=INK, annotation_clip=False)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _, d=decimals: number(value, 0 if d == 0 else 1)))
        axis.set_title(title, loc="left", fontsize=11.5, color=INK, pad=20, fontweight="bold")
        axis.text(0.0, 1.035, note, transform=axis.transAxes, fontsize=9, color=INK_SECONDARY, va="bottom")
    for axis in axes[1]:
        axis.set_xlabel("joins made", fontsize=9.5, color=INK_SECONDARY)
    handles = [plt.Line2D([], [], color=colour, linewidth=2, marker=marker, markersize=8, markeredgecolor=SURFACE,
                          markeredgewidth=2, label=names[depth]) for depth, (colour, marker) in zip(depths, SERIES)]
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, 0.925), ncol=3, frameon=False,
                  fontsize=9.5, labelcolor=INK_SECONDARY, handletextpad=0.6, columnspacing=2.2)
    figure.suptitle(f"{brain}: the capsule's figures after a policy's joins, by the merge cuts made before",
                    x=0.05, y=0.985, ha="left", fontsize=14, color=INK, fontweight="bold")
    figure.text(0.05, 0.012, f"{sources} Candidates and scores are those of the unedited graph, both segments under "
                             "ground truth: an upper estimate. Zero joins is the graph before the joins.",
                fontsize=8.5, color=MUTED, ha="left")
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def table_figure(table, sources, brain, path):
    """Every graph as one row of a table, the unedited graph first; a cell of a figure is tinted
    and set bold when it is better than the unedited graph's, tinted the other way when worse."""
    base = table[(table["cut_depth"] == 0) & (table["joins"] == 0)].iloc[0]
    metrics = [("merges", 0, -1), ("splits", 0, -1), ("merged_edges_pct", 2, -1), ("edge_accuracy", 2, 1), ("erl_um", 0, 1)]
    columns = [("Merge cuts first", "", 2.6), ("Joins made", "", 1.4), ("Labeled as splits", "of the joins made", 2.1),
               ("Merges", "lower is better", 1.5), ("Splits", "lower is better", 1.5),
               ("Splits change", "against unedited", 1.9), ("Merged edges, %", "lower is better", 2.1),
               ("Edge accuracy, %", "higher is better", 2.2), ("ERL, um", "higher is better", 1.6),
               ("Better than", "unedited, of 5", 1.8)]
    edges = [0.0]
    for _, _, width in columns:
        edges.append(edges[-1] + width)

    lines, group_starts = [("no edit (the unedited graph)", base, False)], []
    for depth in sorted(table["cut_depth"].unique()):
        rows = table[table["cut_depth"] == depth]
        rows = rows[rows["joins"] > 0] if depth == 0 else rows      # zero joins without cuts is the unedited graph
        group_starts.append(len(lines))
        label = "none" if depth == 0 else f"{int(rows['cuts'].iloc[0]):,}"
        lines += [(label if index == 0 else "", row, True) for index, (_, row) in enumerate(rows.iterrows())]

    header, row_height = 1.7, 1.0
    figure = plt.figure(figsize=(14.0, 0.33 * (len(lines) + header) + 1.75), facecolor=SURFACE)
    axis = figure.add_axes([0.025, 0.065, 0.95, 0.795])
    axis.set_axis_off()
    axis.set_xlim(0, edges[-1])
    axis.set_ylim(header + row_height * len(lines), 0)
    for index, (title, note, _) in enumerate(columns):
        x, align = (edges[0] + 0.12, "left") if index == 0 else (edges[index + 1] - 0.14, "right")
        axis.text(x, 0.55, title, ha=align, va="center", fontsize=9.5, color=INK, fontweight="bold")
        axis.text(x, 1.18, note, ha=align, va="center", fontsize=8, color=MUTED)
    axis.plot([0, edges[-1]], [header, header], color=AXIS, linewidth=1.0)
    for start in group_starts:
        y = header + row_height * start
        axis.plot([0, edges[-1]], [y, y], color=AXIS if start == group_starts[0] else GRID, linewidth=1.0)

    for index, (label, row, edited) in enumerate(lines):
        y = header + row_height * (index + 0.5)
        weight = "bold" if not edited else "normal"
        axis.text(edges[0] + 0.12, y, label, ha="left", va="center", fontsize=9.5, color=INK, fontweight=weight)
        better = 0
        cells = {1: f"{int(row['joins']):,}" if edited else "", 2: f"{int(row['joins_labeled_as_splits']):,}" if row["joins"] else "",
                 5: f"{int(row['splits'] - base['splits']):+,}".replace("-", "−") if edited else ""}
        for position, (column, decimals, direction) in zip((3, 4, 6, 7, 8), metrics):
            gain = direction * (row[column] - base[column])
            better += gain > 0
            if edited and gain != 0:
                axis.add_patch(Rectangle((edges[position] + 0.06, y - 0.42), edges[position + 1] - edges[position] - 0.12, 0.84,
                                         facecolor=BETTER if gain > 0 else WORSE, edgecolor="none"))
            axis.text(edges[position + 1] - 0.14, y, number(row[column], decimals), ha="right", va="center", fontsize=9.5,
                      color=INK, fontweight="bold" if (gain > 0 or not edited) else "normal")
        cells[9] = f"{better} of 5" if edited else ""
        for position, text in cells.items():
            axis.text(edges[position + 1] - 0.14, y, text, ha="right", va="center", fontsize=9.5, color=INK,
                      fontweight="bold" if (position == 9 and better == len(metrics)) else "normal")

    figure.suptitle(f"{brain}: the capsule's figures for each choice of merge cuts and joins", x=0.03, y=0.975, ha="left",
                    fontsize=14, color=INK, fontweight="bold")
    for x, colour, text in ((0.03, BETTER, "bold on blue: better than the unedited graph"), (0.33, WORSE, "on red: worse")):
        figure.add_artist(Rectangle((x, 0.895), 0.014, 0.022, transform=figure.transFigure, facecolor=colour, edgecolor="none"))
        figure.text(x + 0.02, 0.906, text, fontsize=9.5, color=INK_SECONDARY, va="center")
    figure.text(0.03, 0.018, f"{sources} Every row is one run of the capsule on the whole edited graph. Candidates and scores are "
                             "those of the unedited graph,\nboth segments under ground truth, so the rows with joins are an upper "
                             "estimate.", fontsize=8.5, color=MUTED, ha="left", va="bottom", linespacing=1.4)
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def main():
    options = parse_args()
    brain = json.loads((options.unedited / "run_info.json").read_text())["brain"]
    table = rows_of(options)
    if len(table["cut_depth"].unique()) > len(SERIES):
        raise SystemExit(f"more than {len(SERIES)} cut depths: draw them in several figures")
    record = json.loads(next(iter(sorted(options.joined.glob("cut*_join*/fragments.json")))).read_text())
    scores = Path(record["candidates"]["scores"]).parts
    sources = f"Split scores: {scores[-3]}, {Path(scores[-1]).stem}." if len(scores) >= 3 else ""

    options.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(options.out / f"{brain}_split_policy.csv", index=False, float_format="%.4f")
    figure_of(table, sources, brain, options.out / f"{brain}_split_policy_metrics.png")
    table_figure(table, sources, brain, options.out / f"{brain}_split_policy_table.png")
    print(table.to_string(index=False))
    print(f"written to {options.out}")


if __name__ == "__main__":
    main()

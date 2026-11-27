"""Figure of the capsule's evaluation after a policy's joins, by the merge cuts made before.

Reads capsule outputs that already exist; nothing is evaluated here. ``--joined`` holds one
directory ``cut<C>_join<J>/`` per graph (``join_split_candidates.py --joins policy`` after the
``C`` highest-ranked merge cuts of a policy, ``C`` = 0 for none; each with ``fragments.json`` and
``capsule/``). The graph before the joins is the unedited one (``--unedited``) for ``C`` = 0 and
``top<C>/capsule`` under ``--cuts`` otherwise.

The table also shows the upper bound in which ground truth chooses the edits: ``--oracle-cut`` is
the graph after the cuts at every merge candidate labeled as a merge (``cut_merge_sites.py
--sites oracle``), ``--oracle-joined`` the graphs after those cuts and the joins that ground truth
names on the cut graph (``join_split_candidates.py --joins oracle --truth labels``).

Written to ``--out`` (default ``evaluation/figure/``, which Git ignores), named after the brain:

* ``<brain>_split_policy_metrics.png``: merges, splits, merged edges, edge accuracy, ERL and the
  merge sites that the joins add, against the number of joins made, one line per cut depth;
* ``<brain>_split_policy_table.png``: the same numbers as a table, one row per graph with the
  unedited graph first and the upper bound after it, each figure marked better or worse than
  the unedited graph's and the policy's rows that are better on all five framed in red (never
  the upper bound's, whose edits ground truth chose);
* ``<brain>_split_policy.csv``: the numbers of both.

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

# Cut depths are ordered, so the lines take one hue from light to dark (the dataviz palette's blue
# ramp, steps 250 to 700; nothing lighter, to stay visible on the surface), each with its own marker.
RAMP = ("#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")
MARKERS = ("o", "s", "^", "D", "v", "P")
# Table cells: light steps of the palette's diverging pair (blue for better, red for worse).
BETTER, WORSE = "#cde2fb", "#f9dbdb"
# The frame around a policy's row that is better than the unedited graph on every figure (the palette's red).
FRAME = "#e34948"


def series_of(count):
    """Colour and marker of each of ``count`` cut depths, the steps spread over the whole ramp."""
    if count > len(MARKERS):
        raise SystemExit(f"more than {len(MARKERS)} cut depths: draw them in several figures")
    steps = [0] if count == 1 else [round(index * (len(RAMP) - 1) / (count - 1)) for index in range(count)]
    return [(RAMP[step], marker) for step, marker in zip(steps, MARKERS)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unedited", type=Path, default=RESULTS / "841260_baseline_alignment_20261118/cache_graph")
    parser.add_argument("--cuts", type=Path, default=RESULTS / "841260_merge_policy_20261121")
    parser.add_argument("--joined", type=Path, default=RESULTS / "841260_split_policy_20261121")
    parser.add_argument("--oracle-cut", type=Path, default=RESULTS / "841260_merge_oracle_20261118/remove12",
                        help="Upper bound: the graph after the merge cuts that ground truth chooses")
    parser.add_argument("--oracle-joined", type=Path, nargs="*",
                        default=[RESULTS / "841260_split_oracle_20261118/remove12_labels_s50",
                                 RESULTS / "841260_split_oracle_20261118/remove12_labels"],
                        help="Upper bound: graphs after those cuts and the joins that ground truth names")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "figure")
    return parser.parse_args()


def oracle_rows(options):
    """The upper bound: ground truth chooses the merge cuts, then the joins on the cut graph."""
    cut = json.loads((options.oracle_cut / "fragments.json").read_text())
    results, sites = load(options.oracle_cut / "capsule")
    sites_before = site_positions(sites)
    cuts = cut["sites"]["on_listed_segments"]
    rows = [{"label": f"ground truth: {cuts:,} cuts", "cuts": cuts, "joins": 0, "joins_labeled_as_splits": 0,
             "min_support_um": 0.0, **totals(results), "merge_sites_added_by_the_joins": 0,
             "merge_sites_gone_after_the_joins": 0}]
    for directory in options.oracle_joined:
        record = json.loads((directory / "fragments.json").read_text())
        if (record["sites"], record["merge_edit"]) != (cut["sites"], cut["edit"]) or record["truth"]["reading"] != "labels" \
                or record["joins"]["selection"] != "oracle":
            raise SystemExit(f"{directory}: not the joins that ground truth names (labels) after the cuts of {options.oracle_cut}")
        results, sites = load(directory / "capsule")
        sites_after = site_positions(sites)
        shared = len(match_sites(sites_before, sites_after, SITE_RADIUS_UM))
        support = float(record["truth"].get("min_support_um", 0.0))     # records of 2026-11-18 have no key without it
        rows.append({"label": f"+ joins it names ({support:g} um)" if support else "+ every join it names", "cuts": cuts,
                     "joins": record["joins"]["made"], "joins_labeled_as_splits": record["joins"]["made"],
                     "min_support_um": support, **totals(results), "merge_sites_added_by_the_joins": len(sites_after) - shared,
                     "merge_sites_gone_after_the_joins": len(sites_before) - shared})
    return pd.DataFrame(rows).sort_values("joins", kind="stable").reset_index(drop=True)


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
    series = series_of(len(depths))
    names = {depth: "no merge cuts" if depth == 0 else f"{int(table[table['cut_depth'] == depth]['cuts'].iloc[0]):,} merge cuts first"
             for depth in depths}
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), facecolor=SURFACE)
    figure.subplots_adjust(left=0.05, right=0.955, top=0.83, bottom=0.1, hspace=0.5, wspace=0.27)
    for axis, (column, title, note, decimals) in zip(axes.ravel(), panels):
        style(axis)
        ends = []
        for depth, (colour, marker) in zip(depths, series):
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
                          markeredgewidth=2, label=names[depth]) for depth, (colour, marker) in zip(depths, series)]
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, 0.925), ncol=len(handles), frameon=False,
                  fontsize=9.5, labelcolor=INK_SECONDARY, handletextpad=0.6, columnspacing=2.2)
    figure.suptitle(f"{brain}: the capsule's figures after a policy's joins, by the merge cuts made before",
                    x=0.05, y=0.985, ha="left", fontsize=14, color=INK, fontweight="bold")
    figure.text(0.05, 0.012, f"{sources} Candidates and scores are those of the unedited graph, both segments under "
                             "ground truth: an upper estimate. Zero joins is the graph before the joins.",
                fontsize=8.5, color=MUTED, ha="left")
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def table_figure(table, oracle, sources, brain, path):
    """Every graph as one row of a table: the unedited graph, the upper bound (``oracle``), then the
    policy's graphs by cut depth. A cell of a figure is tinted and set bold when it is better than
    the unedited graph's, tinted the other way when worse."""
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

    # Group starts carry a rule: a strong one under the unedited graph and under the upper bound.
    # A line is (label, figures, edited, chosen by the policy).
    lines, group_starts = [("no edit (the unedited graph)", base, False, False)], {1: AXIS}
    lines += [(row["label"], row, True, False) for _, row in oracle.iterrows()]
    for depth in sorted(table["cut_depth"].unique()):
        rows = table[table["cut_depth"] == depth]
        rows = rows[rows["joins"] > 0] if depth == 0 else rows      # zero joins without cuts is the unedited graph
        group_starts.setdefault(len(lines), AXIS if depth == 0 else GRID)
        label = "none" if depth == 0 else f"{int(rows['cuts'].iloc[0]):,}"
        lines += [(label if index == 0 else "", row, True, True) for index, (_, row) in enumerate(rows.iterrows())]

    # Heights in inches, so that title, key and footnote keep their size whatever the number of rows.
    header, row_height = 1.7, 1.0
    top, row_inches, bottom = 1.11, 0.337, 0.51 + 0.17             # the footnote has three lines
    height = top + row_inches * (len(lines) + header) + bottom
    figure = plt.figure(figsize=(14.0, height), facecolor=SURFACE)
    axis = figure.add_axes([0.025, bottom / height, 0.95, row_inches * (len(lines) + header) / height])
    axis.set_axis_off()
    axis.set_xlim(0, edges[-1])
    axis.set_ylim(header + row_height * len(lines), 0)
    for index, (title, note, _) in enumerate(columns):
        x, align = (edges[0] + 0.12, "left") if index == 0 else (edges[index + 1] - 0.14, "right")
        axis.text(x, 0.55, title, ha=align, va="center", fontsize=9.5, color=INK, fontweight="bold")
        axis.text(x, 1.18, note, ha=align, va="center", fontsize=8, color=MUTED)
    axis.plot([0, edges[-1]], [header, header], color=AXIS, linewidth=1.0)
    for start, colour in group_starts.items():
        y = header + row_height * start
        axis.plot([0, edges[-1]], [y, y], color=colour, linewidth=1.0)

    framed, group = 0, ""
    for index, (label, row, edited, by_policy) in enumerate(lines):
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
                      fontweight="bold" if (position == 9 and by_policy and better == len(metrics)) else "normal")
        group = label or group
        # The policy's rows that are better on every figure stand out by a frame. The upper bound's
        # rows never do: ground truth chose their edits, so they are a ceiling, not a result.
        if by_policy and better == len(metrics):
            framed += 1
            if not label:                           # and repeat their group's cut depth, to be read alone
                axis.text(edges[0] + 0.12, y, group, ha="left", va="center", fontsize=9.5, color=INK)
            axis.add_patch(Rectangle((0.03, y - 0.47), edges[-1] - 0.06, 0.94, facecolor="none", edgecolor=FRAME,
                                     linewidth=1.8, clip_on=False, zorder=4))

    figure.suptitle(f"{brain}: the capsule's figures for each choice of merge cuts and joins", x=0.03, y=1 - 0.2 / height,
                    ha="left", fontsize=14, color=INK, fontweight="bold")
    for x, colour, text in ((0.03, BETTER, "bold on blue: better than the unedited graph"), (0.33, WORSE, "on red: worse")):
        figure.add_artist(Rectangle((x, 1 - 0.83 / height), 0.014, 0.174 / height, transform=figure.transFigure,
                                    facecolor=colour, edgecolor="none"))
        figure.text(x + 0.02, 1 - 0.745 / height, text, fontsize=9.5, color=INK_SECONDARY, va="center")
    if framed:
        figure.add_artist(Rectangle((0.47, 1 - 0.83 / height), 0.014, 0.174 / height, transform=figure.transFigure,
                                    facecolor="none", edgecolor=FRAME, linewidth=1.8))
        figure.text(0.49, 1 - 0.745 / height, "red frame: the policy's edits, better than the unedited graph on all five figures",
                    fontsize=9.5, color=INK_SECONDARY, va="center")
    note = (f"{sources} Every row is one run of the capsule on the whole edited graph. A policy's candidates and scores are "
            "those of the unedited graph,\nboth segments under ground truth, so its rows with joins are an upper estimate."
            " Ground-truth rows: cuts at every merge candidate labeled as a merge,\nthen joins at the split candidates "
            "of the cut graph that ground truth names")
    note += " (50 um: only where both pieces label at least 50 um of the neuron)." if (oracle["min_support_um"] == 50).any() else "."
    figure.text(0.03, 0.14 / height, note, fontsize=8.5, color=MUTED, ha="left", va="bottom", linespacing=1.4)
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def main():
    options = parse_args()
    brain = json.loads((options.unedited / "run_info.json").read_text())["brain"]
    table = rows_of(options)
    record = json.loads(next(iter(sorted(options.joined.glob("cut*_join*/fragments.json")))).read_text())
    scores = Path(record["candidates"]["scores"]).parts
    sources = f"Split scores: {scores[-3]}, {Path(scores[-1]).stem}." if len(scores) >= 3 else ""
    oracle = oracle_rows(options)

    options.out.mkdir(parents=True, exist_ok=True)
    # One file for both: the policy's rows, then the upper bound (columns it has no figure for stay empty).
    both = pd.concat([table.assign(chosen_by="policy"), oracle.drop(columns="label").assign(chosen_by="ground truth")],
                     ignore_index=True)
    counts = [column for column in both.columns if column not in ("chosen_by", "min_support_um")
              and (both[column].dropna() % 1 == 0).all()]
    both[counts] = both[counts].astype("Int64")
    both.to_csv(options.out / f"{brain}_split_policy.csv", index=False, float_format="%.4f")
    figure_of(table, sources, brain, options.out / f"{brain}_split_policy_metrics.png")
    table_figure(table, oracle, sources, brain, options.out / f"{brain}_split_policy_table.png")
    print(both.to_string(index=False))
    print(f"written to {options.out}")


if __name__ == "__main__":
    main()

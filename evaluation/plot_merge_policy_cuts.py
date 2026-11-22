"""Figures of the capsule's evaluation before and after a policy's merge cuts.

Reads capsule outputs that already exist (``run_capsule_original.py``) and the cut records of
``cut_merge_sites.py``; nothing is evaluated here. Three kinds of graph are compared, all judged
by the capsule against the same ground truth:

* the unedited fragment graph of the cache (``--unedited``, a capsule output directory);
* the graph cut at a policy's highest-ranked merge candidates, one directory ``top<K>/`` per
  depth under ``--policy`` (each with ``fragments.json``, ``fragments_cuts.csv``, ``capsule/``);
* the graph cut at every candidate that ground truth labels as a merge (``--oracle``, a
  directory of the same form), the upper bound with no policy in it.

Written to ``--out`` (default ``evaluation/figure/``, which Git ignores), named after the brain:

* ``<brain>_merge_policy_metrics.png``: merges, splits, merged edges, edge accuracy, ERL and
  the share of cuts at a merge site of the capsule, against the number of cuts;
* ``<brain>_merge_policy_sites.png``: how many merge sites of the unedited graph each depth
  removes, apart by the label of the candidates cut nearby;
* ``<brain>_merge_policy.csv``: the plotted numbers.

Run on a compute node in panda, from the repository root:

    python evaluation/plot_merge_policy_cuts.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullLocator
from scipy.spatial.distance import cdist

from compare_capsule_runs import load, match_sites, site_positions

RESULTS = Path(__file__).resolve().parent / "results"
SITE_RADIUS_UM = 20.0

# Colours: chart surface, inks and the first two categorical slots of the dataviz palette.
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
POLICY, ORACLE, NEUTRAL = "#2a78d6", "#eb6834", "#c3c2b7"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unedited", type=Path, default=RESULTS / "841260_baseline_alignment_20261118/cache_graph")
    parser.add_argument("--policy", type=Path, default=RESULTS / "841260_merge_policy_20261121")
    parser.add_argument("--oracle", type=Path, default=RESULTS / "841260_merge_oracle_20261118/remove12")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "figure")
    return parser.parse_args()


def totals(results):
    return {"merges": int(results["# Merges"].sum()), "splits": int(results["# Splits"].sum()),
            "merged_edges_pct": float(results["% Merged Edges"].mean()),
            "omit_edges_pct": float(results["% Omit Edges"].mean()),
            "split_edges_pct": float(results["% Split Edges"].mean()),
            "edge_accuracy": float(results["Edge Accuracy"].mean()), "erl_um": float(results["ERL"].mean())}


def edited(name, directory, sites_before):
    """One row for a cut graph: the capsule's totals, and what the cuts did to the merge sites of
    the unedited graph (a site is gone when the cut graph has no site within 20 um, one to one)."""
    record = json.loads((directory / "fragments.json").read_text())
    cuts = pd.read_csv(directory / "fragments_cuts.csv")
    cuts = cuts[~cuts["action"].astype(str).str.startswith("skipped")]
    cut_xyz = cuts[["x_um", "y_um", "z_um"]].to_numpy(dtype=float)
    # The oracle's cuts carry no label column: every one of them is a labeled merge.
    labeled = cuts["evaluator_label"].to_numpy() == 1 if "evaluator_label" in cuts else np.ones(len(cuts), dtype=bool)
    results, sites = load(directory / "capsule")
    gone = np.ones(len(sites_before), dtype=bool)
    gone[[a for a, _ in match_sites(sites_before, site_positions(sites), SITE_RADIUS_UM)]] = False
    near = [cdist(sites_before, cut_xyz[chosen]).min(axis=1) <= SITE_RADIUS_UM if chosen.any()
            else np.zeros(len(sites_before), dtype=bool) for chosen in (labeled, ~labeled)]
    edit = record["edit"]
    return {"graph": name, "selection": record["sites"]["selection"], "scores": record["sites"].get("scores", ""),
            "rule": edit["rule"] + (f" {edit['radius_um']:g} um" if "radius_um" in edit else ""),
            "cuts": len(cuts), "cuts_labeled_as_merges": int(labeled.sum()),
            "cuts_at_a_capsule_site": int((cdist(cut_xyz, sites_before).min(axis=1) <= SITE_RADIUS_UM).sum()),
            **totals(results), "sites_gone": int(gone.sum()),
            "gone_near_a_cut_labeled_as_merge": int((gone & near[0]).sum()),
            "gone_near_cuts_labeled_0_only": int((gone & near[1] & ~near[0]).sum()),
            "gone_without_a_cut_nearby": int((gone & ~near[0] & ~near[1]).sum())}


def style(axis):
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(AXIS)
    axis.tick_params(colors=MUTED, labelsize=9, length=3)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


def number(value, decimals):
    return f"{value:,.{decimals}f}"


def metrics_figure(table, unedited, brain, path):
    policy, oracle = table[table["selection"] == "policy"], table[table["selection"] == "oracle"].iloc[0]
    share = lambda rows: 100.0 * rows["cuts_at_a_capsule_site"] / rows["cuts"]
    panels = [("merges", "Merges", "count, lower is better", 0),
              ("splits", "Splits", "count, lower is better", 0),
              ("merged_edges_pct", "Merged edges", "% of ground-truth edges, lower is better", 2),
              ("edge_accuracy", "Edge accuracy", "%, higher is better", 2),
              ("erl_um", "Expected run length", "um, higher is better", 0),
              (None, "Cuts at a merge site of the capsule", "% of the cuts made, within 20 um", 0)]
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), facecolor=SURFACE)
    for axis, (column, title, note, decimals) in zip(axes.ravel(), panels):
        style(axis)
        values = share(policy) if column is None else policy[column]
        at_oracle = share(oracle) if column is None else oracle[column]
        if column is not None:
            axis.axhline(unedited[column], color=MUTED, linewidth=1.2, zorder=1)
            axis.annotate(f"unedited {number(unedited[column], decimals)}", (1.0, unedited[column]),
                          xycoords=("axes fraction", "data"), xytext=(-2, 4), textcoords="offset points",
                          ha="right", va="bottom", fontsize=8.5, color=INK_SECONDARY)
        axis.plot(policy["cuts"], values, color=POLICY, linewidth=2, marker="o", markersize=8,
                  markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        axis.plot([oracle["cuts"]], [at_oracle], color=ORACLE, linestyle="none", marker="D", markersize=9,
                  markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        # Direct labels on the two ends of the policy's line and on the oracle only; an end label
        # goes on the side away from the unedited line.
        for x, y in ((policy["cuts"].iloc[0], values.iloc[0]), (policy["cuts"].iloc[-1], values.iloc[-1])):
            below = column is not None and y < unedited[column]
            axis.annotate(number(y, decimals), (x, y), xytext=(0, -10 if below else 9), textcoords="offset points",
                          ha="center", va="top" if below else "bottom", fontsize=8.5, color=INK)
        axis.annotate(number(at_oracle, decimals), (oracle["cuts"], at_oracle), xytext=(9, 0), textcoords="offset points",
                      ha="left", va="center", fontsize=8.5, color=INK)
        axis.set_xscale("log")
        axis.set_xticks(list(policy["cuts"]))
        axis.xaxis.set_minor_locator(NullLocator())
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _, d=decimals: number(value, 0 if d == 0 else 1)))
        axis.margins(x=0.08, y=0.18)
        axis.set_title(title, loc="left", fontsize=11.5, color=INK, pad=20, fontweight="bold")
        axis.text(0.0, 1.035, note, transform=axis.transAxes, fontsize=9, color=INK_SECONDARY, va="bottom")
    for axis in axes[1]:
        axis.set_xlabel("merge cuts made (log scale)", fontsize=9.5, color=INK_SECONDARY)
    handles = [plt.Line2D([], [], color=POLICY, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE,
                          markeredgewidth=2, label="policy: its highest-ranked candidates cut"),
               plt.Line2D([], [], color=ORACLE, linestyle="none", marker="D", markersize=9, markeredgecolor=SURFACE,
                          markeredgewidth=2, label=f"oracle: all {oracle['cuts']} candidates labeled as merges cut"),
               plt.Line2D([], [], color=MUTED, linewidth=1.2, label="unedited graph")]
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, 0.925), ncol=3, frameon=False,
                  fontsize=9.5, labelcolor=INK_SECONDARY, handletextpad=0.6, columnspacing=2.2)
    figure.suptitle(f"{brain}: the capsule's figures before and after merge cuts", x=0.05, y=0.985, ha="left",
                    fontsize=14, color=INK, fontweight="bold")
    run = Path(policy["scores"].iloc[0]).parts
    source = f"{run[-3]}, {Path(run[-1]).stem}" if len(run) >= 3 else policy["scores"].iloc[0]
    figure.text(0.05, 0.012, f"Policy scores: {source}. Cut rule: {policy['rule'].iloc[0]}. "
                             "Every point is one run of the capsule on the whole edited graph; no joins are made.",
                fontsize=8.5, color=MUTED, ha="left")
    figure.subplots_adjust(left=0.05, right=0.985, top=0.83, bottom=0.1, hspace=0.5, wspace=0.2)
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def sites_figure(table, total, brain, path):
    policy, oracle = table[table["selection"] == "policy"], table[table["selection"] == "oracle"]
    rows = pd.concat([policy, oracle])
    positions = list(range(len(policy))) + [len(policy) + 0.6]
    parts = [("gone_near_a_cut_labeled_as_merge", POLICY, "gone, a cut labeled as a merge within 20 um"),
             ("gone_near_cuts_labeled_0_only", ORACLE, "gone, only cuts labeled 0 within 20 um"),
             ("gone_without_a_cut_nearby", NEUTRAL, "gone, no cut within 20 um")]
    figure, axis = plt.subplots(figsize=(9.2, 5.6), facecolor=SURFACE)
    style(axis)
    bottom = np.zeros(len(rows))
    for column, colour, label in parts:
        axis.bar(positions, rows[column], bottom=bottom, width=0.56, color=colour, edgecolor=SURFACE, linewidth=1.2, label=label)
        bottom += rows[column].to_numpy()
    for x, height in zip(positions, bottom):
        axis.annotate(f"{int(height)}", (x, height), xytext=(0, 4), textcoords="offset points", ha="center",
                      va="bottom", fontsize=9.5, color=INK)
    axis.axhline(total, color=MUTED, linewidth=1.2)
    axis.annotate(f"all {total} merge sites of the unedited graph", (1.0, total), xycoords=("axes fraction", "data"),
                  xytext=(-2, 4), textcoords="offset points", ha="right", va="bottom", fontsize=9, color=INK_SECONDARY)
    labels = [f"{cuts:,} cuts\n({labeled} labeled\nas merges)" for cuts, labeled in zip(policy["cuts"], policy["cuts_labeled_as_merges"])]
    labels.append(f"{int(oracle['cuts'].iloc[0])} cuts\n(all labeled\nas merges)")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=8.5, color=INK_SECONDARY)
    axis.set_ylim(0, total * 1.12)
    axis.set_ylabel("merge sites of the capsule removed", fontsize=9.5, color=INK_SECONDARY)
    axis.text((positions[0] + positions[len(policy) - 1]) / 2, -0.25, "policy, by depth", transform=axis.get_xaxis_transform(),
              ha="center", va="top", fontsize=9.5, color=INK)
    axis.text(positions[-1], -0.25, "oracle", transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=9.5, color=INK)
    axis.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.01), ncol=3, frameon=False, fontsize=9,
                labelcolor=INK_SECONDARY, columnspacing=1.8, handlelength=1.4)
    figure.suptitle(f"{brain}: which merge sites the cuts remove", x=0.07, y=0.975, ha="left", fontsize=13.5,
                    color=INK, fontweight="bold")
    figure.text(0.07, 0.905, "Merge sites the capsule finds on the unedited graph and no longer finds after the cuts, "
                             "by the label of the candidates cut nearby", fontsize=9, color=INK_SECONDARY, ha="left")
    figure.subplots_adjust(left=0.085, right=0.98, top=0.81, bottom=0.24)
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def main():
    options = parse_args()
    brain = json.loads((options.unedited / "run_info.json").read_text())["brain"]
    results, sites = load(options.unedited)
    sites_before = site_positions(sites)
    unedited = {"graph": "unedited", "selection": "none", **totals(results)}
    depths = sorted(int(path.name[3:]) for path in options.policy.glob("top*") if (path / "capsule/original").is_dir())
    rows = [edited(f"top{depth}", options.policy / f"top{depth}", sites_before) for depth in depths]
    rows.append(edited("oracle", options.oracle, sites_before))
    table = pd.DataFrame(rows)

    options.out.mkdir(parents=True, exist_ok=True)
    frame = pd.concat([pd.DataFrame([unedited]), table])
    counts = [column for column in frame if column.startswith(("cuts", "sites_gone", "gone_"))]
    frame[counts] = frame[counts].astype("Int64")  # the unedited row has no cuts: keep the counts integers
    frame.to_csv(options.out / f"{brain}_merge_policy.csv", index=False, float_format="%.4f")
    metrics_figure(table, unedited, brain, options.out / f"{brain}_merge_policy_metrics.png")
    sites_figure(table, len(sites_before), brain, options.out / f"{brain}_merge_policy_sites.png")
    print(table.drop(columns=["scores"]).to_string(index=False))
    print(f"written to {options.out}")


if __name__ == "__main__":
    main()

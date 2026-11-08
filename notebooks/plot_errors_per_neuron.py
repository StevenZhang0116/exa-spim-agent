#!/usr/bin/env python3
"""Average number of errors per GT neuron, per brain, from the canonical skeleton metrics.

For every brain the one current ``metrics_out/<brain>/<segmentation_id>/results.csv``
(written by ``notebooks/evaluate_skeleton_metrics.py`` with segmentation_skeleton_metrics) has
one row per GT neuron. This script divides each brain's error totals by its number of GT neurons
and plots the result across brains, with the per-neuron values behind each bar:

- ``# Splits`` per neuron: split events of the neuron, ``total # Splits / neurons``;
- ``# Merges`` per neuron: merge sites attributed to the neuron, ``total # Merges / neurons``;
- ``% Omit Edges`` averaged over neurons: omission has no event count in the metrics, so the
  per-neuron percentage of GT edges covered by no fragment is averaged instead.

These are plain means over neurons, every neuron counting once. The "Average Results" block of
``results_overview.txt`` is different: segmentation_skeleton_metrics averages every column with
the neuron's ``SWC Run Length`` as weight (``util.compute_weighted_avg``), so long neurons
dominate it (789202 splits: 4,120 per neuron here, 4,344 weighted). The CSV carries both, the
weighted means as ``*_length_weighted``, and the per-neuron ``% Split Edges`` / ``% Merged Edges``
for reference. Brains are grouped by their current role in the evolution loop (TRAIN/selection,
development validation, reserved; a brain passed with ``--brains`` but absent from ``ROLES`` is
shown as "unlisted"); the grouping is presentation only, the numbers do not depend on it.

The second figure, ``pool_reach_per_neuron.png``, asks how much of that canonical error load the
evolution loop's fixed candidate pools can see, again per GT neuron. It reuses the canonical-gap
outputs (no cache load):

- split: every canonical split pair of ``notebooks/split_canonical_gap/<brain>_mcl<N>.json``
  (adjacent fragment labels along the GT, direct and flank, ``split_canonical_gap.py``) is one of
  *in the split candidate pool* (``tip_to_any_node|r=80|k=2``), *both segments in the fragments
  graph but not in the pool*, or *a segment absent from the fragments graph* (below the minimum
  cable length, so no candidate can ever name it). A pair that touches several neurons counts
  once per neuron (35 of 27,252 pairs on 841260).
- merge: every canonical merge site of ``metrics_out/<brain>/<id>/merge_sites.csv`` (one row per
  site, ``GroundTruth_ID`` is the neuron) is binned by the geodesic distance to the nearest
  declared merge candidate on its segment, stored row-aligned in
  ``notebooks/merge_canonical_gap/<brain>_mcl<N>.json`` (``merge_canonical_gap.py``): within 20 um
  (the positive-label radius), 20-150 um (the claim radius), or none within 150 um (including
  sites that snap to no fragment node).

Both gap files are checked against the brain's current segmentation id and neuron count, and the
per-neuron tallies are checked against the gap files' own pooled recalls before anything is
written, so a stale gap output or a mis-joined neuron fails loudly. Omission has no candidate
kind, so it has no reach panel: its share of the error load is zero by construction.

Run on a compute node (panda), seconds, read-only:
    python -u notebooks/plot_errors_per_neuron.py
    python -u notebooks/plot_errors_per_neuron.py --brains 789202 802449 841260

Outputs under ``notebooks/errors_per_neuron/``: ``errors_per_neuron.png``,
``errors_per_neuron.csv`` (one row per brain: role, segmentation id, neurons, totals, means,
medians, source file), ``per_neuron.csv`` (one row per GT neuron), ``pool_reach_per_neuron.png``,
``pool_reach_per_neuron.csv`` (one row per brain: tier counts per neuron and shares) and
``pool_reach_by_neuron.csv`` (one row per GT neuron). A brain with zero or several current run
directories under ``metrics_out/<brain>/`` is an error: old runs are deleted, not kept beside
the current one (docs/dataset_artifact_dependencies.md).
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "metrics_out"
OUT_DIR = ROOT / "notebooks" / "errors_per_neuron"
GAP_DIRS = {"split": ROOT / "notebooks" / "split_canonical_gap",
            "merge": ROOT / "notebooks" / "merge_canonical_gap"}
MERGE_POSITIVE_RADIUS_UM = 20.0   # merge_canonical_gap.POSITIVE_RADIUS_UM (detector positive label)
MERGE_CLAIM_RADIUS_UM = 150.0     # merge_canonical_gap.CLAIM_RADIUS_UM (candidate claim radius)
# Reach tiers per kind, darkest (reachable by the pool) to lightest (out of reach); one-hue ramp.
REACH_TIERS = {
    "split": (("split_in_pool", "in the split candidate pool"),
              ("split_present_not_in_pool", "both segments in the fragments graph, not in the pool"),
              ("split_segment_absent", "a segment absent from the fragments graph (below the MCL)")),
    "merge": (("merge_within_20um", "merge candidate within 20 um (positive radius)"),
              ("merge_within_150um", "merge candidate within 20-150 um (claim radius)"),
              ("merge_beyond_150um", "no merge candidate within 150 um")),
}
TIER_COLORS = ("#1c5cab", "#5598e7", "#b7d3f6")

# Roles of 2026-11-04 (docs/stage3_evolution.md section 3.2); 841260 joined the development
# validation set on 2026-11-07. The order of this mapping is the plotting order; brains are sorted
# by id inside a role.
ROLES = {
    "TRAIN / selection": ["802449", "747807", "750318", "754610", "754612", "754613"],
    "development validation": ["789202", "794493", "794491", "751473", "841260"],
    "reserved": ["794495"],
}
ALL_BRAINS = [brain for members in ROLES.values() for brain in sorted(members)]
# Reference categorical slots 1-3 (dataviz palette) for the roles that are compared; the reserved
# brain is excluded from development, so it takes the recessive neutral gray.
ROLE_COLORS = {"TRAIN / selection": "#2a78d6", "development validation": "#eb6834",
               "unlisted": "#1baf7a", "reserved": "#c9c8c2"}
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"

METRIC_COLUMNS = ["SWC Run Length", "# Splits", "# Merges", "% Split Edges", "% Omit Edges",
                  "% Merged Edges"]
PANELS = (
    ("# Splits", "splits_per_neuron", "# Splits per GT neuron", "{:.0f}"),
    ("# Merges", "merges_per_neuron", "# Merges per GT neuron", "{:.1f}"),
    ("% Omit Edges", "omit_edges_pct_mean", "% Omit Edges, mean over GT neurons", "{:.1f}"),
)


def role_of(brain: str) -> str:
    for role, members in ROLES.items():
        if brain in members:
            return role
    return "unlisted"


def results_path(brain: str, metrics_dir: Path) -> Path:
    """The brain's one current ``results.csv`` (``archive`` directories are ignored)."""
    paths = sorted(p for p in (metrics_dir / brain).glob("*/results.csv") if "archive" not in p.parts)
    if len(paths) != 1:
        raise FileNotFoundError(f"{metrics_dir / brain}: expected exactly one current run directory "
                                f"with results.csv, found {len(paths)}: {[p.parent.name for p in paths]}")
    return paths[0]


def load_brain(brain: str, metrics_dir: Path) -> pd.DataFrame:
    """One row per GT neuron with the metric columns, the brain and its segmentation id."""
    path = results_path(brain, metrics_dir)
    table = pd.read_csv(path, index_col=0)
    missing = [column for column in METRIC_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if table.index.has_duplicates:
        raise ValueError(f"{path}: duplicate neuron names")
    frame = table[METRIC_COLUMNS].astype(float)
    frame.index.name = "neuron"
    frame.insert(0, "brain", brain)
    frame.insert(1, "role", role_of(brain))
    frame.insert(2, "segmentation_id", path.parent.name)
    frame["results_csv"] = str(path.relative_to(ROOT))
    return frame.reset_index()


def summarize(per_neuron: pd.DataFrame) -> pd.DataFrame:
    """One row per brain: counts divided by the number of GT neurons, plus medians and edge %."""
    rows = []
    for brain, group in per_neuron.groupby("brain", sort=False):
        neurons = len(group)
        weights = group["SWC Run Length"]

        def length_weighted(column):
            # segmentation_skeleton_metrics' "Average Results" definition (util.compute_weighted_avg).
            return float((group[column] * weights).sum() / weights.sum()) if weights.sum() else float("nan")

        rows.append({
            "brain": brain,
            "role": group["role"].iloc[0],
            "segmentation_id": group["segmentation_id"].iloc[0],
            "neurons": neurons,
            "splits_total": int(group["# Splits"].sum()),
            "splits_per_neuron": group["# Splits"].sum() / neurons,
            "splits_median": float(group["# Splits"].median()),
            "merges_total": int(group["# Merges"].sum()),
            "merges_per_neuron": group["# Merges"].sum() / neurons,
            "merges_median": float(group["# Merges"].median()),
            "split_edges_pct_mean": float(group["% Split Edges"].mean()),
            "omit_edges_pct_mean": float(group["% Omit Edges"].mean()),
            "omit_edges_pct_median": float(group["% Omit Edges"].median()),
            "merged_edges_pct_mean": float(group["% Merged Edges"].mean()),
            "splits_length_weighted": length_weighted("# Splits"),
            "merges_length_weighted": length_weighted("# Merges"),
            "omit_edges_pct_length_weighted": length_weighted("% Omit Edges"),
            "results_csv": group["results_csv"].iloc[0],
        })
    return pd.DataFrame(rows)


def _close(a: float, b: float, what: str) -> None:
    if not abs(a - b) <= 1e-9:
        raise ValueError(f"{what}: per-neuron tally {a:.6f} disagrees with the gap file's summary {b:.6f}")


def load_reach(brain: str, metrics_dir: Path, neurons: list, segmentation_id: str, mcl: int,
               gap_dirs: dict = GAP_DIRS) -> tuple[pd.DataFrame, dict]:
    """Per-neuron reach tiers from the canonical-gap outputs of ``split_canonical_gap.py`` and
    ``merge_canonical_gap.py``, checked against the current segmentation and the files' own
    pooled recalls. Returns one row per GT neuron and a dict of file-level facts."""
    tiers = [tier for kind in ("split", "merge") for tier, _ in REACH_TIERS[kind]]
    counts = {neuron: dict.fromkeys(tiers, 0) for neuron in neurons}
    facts = {}

    split_path = gap_dirs["split"] / f"{brain}_mcl{mcl}.json"
    split = json.loads(split_path.read_text())
    split_segmentation = str(split["provenance"]["segmentation"])
    # The legacy layout ends in ".../<segmentation_id>/label_mask", the denoised one in "/<segmentation_id>".
    if f"/{segmentation_id}" not in split_segmentation:
        raise ValueError(f"{split_path}: measured on {split_segmentation}, current is {segmentation_id}; "
                         f"rerun split_canonical_gap.py --measure --brains {brain}")
    if int(split["provenance"]["neurons"]) != len(neurons):
        raise ValueError(f"{split_path}: {split['provenance']['neurons']} GT neurons, results.csv has {len(neurons)}")
    for pair in split["pairs"]:
        if pair["in_pool"]:
            tier = "split_in_pool"
        elif pair["segment_absent_from_fragments"]:
            tier = "split_segment_absent"
        else:
            tier = "split_present_not_in_pool"
        for neuron in pair["neurons"]:
            if neuron not in counts:
                raise ValueError(f"{split_path}: pair {pair['pair']} names unknown neuron {neuron}")
            counts[neuron][tier] += 1
    pairs = split["pairs"]
    _close(sum(p["in_pool"] for p in pairs) / len(pairs), split["summary"]["candidate_recall_canonical_pairs"],
           f"{brain} split in-pool share")
    facts.update({"split_gap_json": str(split_path.relative_to(ROOT)),
                  "split_gap_json_mtime": datetime.fromtimestamp(split_path.stat().st_mtime).isoformat(timespec="minutes"),
                  "split_pairs_total": len(pairs),
                  "split_pairs_multi_neuron": sum(len(p["neurons"]) > 1 for p in pairs),
                  "split_candidate_recall_canonical_pairs": split["summary"]["candidate_recall_canonical_pairs"],
                  # absent from the older (2026-10-14) summary form of 794495
                  "split_present_candidate_recall": split["summary"].get("present_candidate_recall", float("nan"))})

    merge_path = gap_dirs["merge"] / f"{brain}_mcl{mcl}.json"
    merge = json.loads(merge_path.read_text())
    sites_path = metrics_dir / brain / segmentation_id / "merge_sites.csv"
    if Path(merge["canonical_sites_csv"]).resolve() != sites_path.resolve():
        raise ValueError(f"{merge_path}: measured against {merge['canonical_sites_csv']}, current is {sites_path}; "
                         f"rerun merge_canonical_gap.py --measure --brains {brain}")
    sites = pd.read_csv(sites_path)
    distances = merge["canonical_to_candidate_geodesic_um"]
    if not (len(distances) == len(sites) == int(merge["n_canonical"])):
        raise ValueError(f"{merge_path}: {len(distances)} distances for {len(sites)} canonical sites")
    within_20 = within_150 = 0
    for neuron, distance in zip(sites["GroundTruth_ID"].astype(str), distances):
        if neuron not in counts:
            raise ValueError(f"{sites_path}: unknown GT neuron {neuron}")
        if distance is not None and distance <= MERGE_POSITIVE_RADIUS_UM:
            tier, within_20, within_150 = "merge_within_20um", within_20 + 1, within_150 + 1
        elif distance is not None and distance <= MERGE_CLAIM_RADIUS_UM:
            tier, within_150 = "merge_within_150um", within_150 + 1
        else:
            tier = "merge_beyond_150um"
        counts[neuron][tier] += 1
    if len(sites):
        _close(within_20 / len(sites), merge["summary"]["candidate_recall_canonical_at_20"], f"{brain} merge reach at 20 um")
        _close(within_150 / len(sites), merge["summary"]["candidate_recall_canonical_at_150"], f"{brain} merge reach at 150 um")
    facts.update({"merge_gap_json": str(merge_path.relative_to(ROOT)),
                  "merge_gap_json_mtime": datetime.fromtimestamp(merge_path.stat().st_mtime).isoformat(timespec="minutes"),
                  "merge_sites_total": len(sites),
                  "merge_sites_unsnappable": int(merge["canonical_unsnappable"])})

    frame = pd.DataFrame.from_dict(counts, orient="index")
    frame.index.name = "neuron"
    frame.insert(0, "brain", brain)
    frame.insert(1, "role", role_of(brain))
    return frame.reset_index(), facts


def summarize_reach(by_neuron: pd.DataFrame, facts: dict) -> pd.DataFrame:
    """One row per brain: tier counts divided by the number of GT neurons, plus the shares."""
    rows = []
    for brain, group in by_neuron.groupby("brain", sort=False):
        neurons = len(group)
        row = {"brain": brain, "role": group["role"].iloc[0], "neurons": neurons}
        for kind in ("split", "merge"):
            tiers = [tier for tier, _ in REACH_TIERS[kind]]
            total = int(group[tiers].to_numpy().sum())
            row[f"{kind}_total_counted"] = total
            row[f"{kind}_per_neuron"] = total / neurons
            for tier in tiers:
                row[f"{tier}_per_neuron"] = group[tier].sum() / neurons
            reached = int(group[tiers[0]].sum())
            row[f"{kind}_reached_share"] = reached / total if total else float("nan")
            if kind == "merge":
                row["merge_within_150um_share"] = (reached + int(group[tiers[1]].sum())) / total if total else float("nan")
        row.update(facts[brain])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_reach(summary: pd.DataFrame, out_path: Path, dpi: int = 200) -> None:
    fig = Figure(figsize=(11, 9.2))
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 1, sharex=True)
    x = np.arange(len(summary))
    boundaries = [i + 0.5 for i in range(len(summary) - 1)
                  if summary["role"].iloc[i] != summary["role"].iloc[i + 1]]
    tick_labels = [f"{brain}\nn={neurons}" for brain, neurons in zip(summary["brain"], summary["neurons"])]
    role_colors = [ROLE_COLORS.get(role, TEXT_SECONDARY) for role in summary["role"]]
    titles = {"split": "Canonical split pairs per GT neuron, by what the split pool can reach "
                       "(tiers side by side, log scale)",
              "merge": "Canonical merge sites per GT neuron, by distance to the nearest merge candidate (stacked)"}
    for ax, kind in zip(axes, ("split", "merge")):
        tiers = REACH_TIERS[kind]
        if kind == "split":
            # The three tiers differ by two orders of magnitude, so they are drawn side by side on a
            # log axis; stacking them on a log axis would distort the segments.
            offsets = np.linspace(-0.24, 0.24, len(tiers))
            for (tier, label), color, offset in zip(tiers, TIER_COLORS, offsets):
                ax.bar(x + offset, summary[f"{tier}_per_neuron"].to_numpy(), width=0.22, color=color,
                       edgecolor="none", label=label, zorder=2)
            ax.set_yscale("log")
            ax.set_ylim(bottom=1)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            tops = summary[[f"{tier}_per_neuron" for tier, _ in tiers]].max(axis=1).to_numpy()
        else:
            bottom = np.zeros(len(summary))
            for (tier, label), color in zip(tiers, TIER_COLORS):
                values = summary[f"{tier}_per_neuron"].to_numpy()
                ax.bar(x, values, bottom=bottom, width=0.62, color=color, edgecolor="white", linewidth=0.8,
                       label=label, zorder=2)
                bottom = bottom + values
            ax.set_ylim(bottom=0)
            tops = bottom
        for i, (top, share) in enumerate(zip(tops, summary[f"{kind}_reached_share"])):
            text = "n/a" if np.isnan(share) else f"{100 * share:.0f}%"
            ax.annotate(text, (i, top), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.5, color=TEXT_PRIMARY, zorder=5)
        for boundary in boundaries:
            ax.axvline(boundary, color=GRID, lw=0.8, zorder=1)
        strip = blended_transform_factory(ax.transData, ax.transAxes)
        ax.scatter(x, np.full(len(x), -0.035), marker="s", s=38, c=role_colors, transform=strip,
                   clip_on=False, zorder=5, linewidths=0)
        ax.set_title(titles[kind], fontsize=10, color=TEXT_PRIMARY, loc="left", pad=24)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.tick_params(axis="x", labelbottom=True, pad=12)
        ax.set_xlim(-0.6, len(summary) - 0.4)
        # Legends sit between the panel title and the axes, clear of the bars on either scale.
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=7.5,
                  borderaxespad=0, handlelength=1.4, columnspacing=1.2)
        style(ax)
    present = list(dict.fromkeys(summary["role"]))
    handles = [Patch(facecolor=ROLE_COLORS.get(role, TEXT_SECONDARY), label=role) for role in present]
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False, fontsize=8,
               bbox_to_anchor=(0.005, 0.968))
    fig.suptitle("How much of the canonical error load the fixed candidate pools reach, per GT neuron",
                 x=0.01, y=0.99, ha="left", fontsize=12, color=TEXT_PRIMARY)
    fig.text(0.01, 0.006, "bars = canonical count / GT neurons per reach tier (dark = reachable; split tiers side by side on a "
             "log axis, merge tiers stacked); label = reached share; squares under the axis = role. Split pairs: "
             "split_canonical_gap (direct + flank pairs; a multi-neuron pair counts once per neuron); merge sites: "
             "metrics_out merge_sites.csv with merge_canonical_gap geodesic distances. Omission has no candidate kind. "
             f"{date.today().isoformat()}",
             fontsize=6.5, color=TEXT_SECONDARY, wrap=True)
    fig.tight_layout(rect=(0, 0.025, 1, 0.945))
    fig.savefig(out_path, dpi=dpi)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=3)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def plot(summary: pd.DataFrame, per_neuron: pd.DataFrame, out_path: Path, dpi: int = 200) -> None:
    fig = Figure(figsize=(11, 11.5))
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(PANELS), 1, sharex=True)
    x = np.arange(len(summary))
    colors = [ROLE_COLORS.get(role, TEXT_SECONDARY) for role in summary["role"]]
    boundaries = [i + 0.5 for i in range(len(summary) - 1)
                  if summary["role"].iloc[i] != summary["role"].iloc[i + 1]]
    rng = np.random.default_rng(0)
    tick_labels = [f"{brain}\nn={neurons}" for brain, neurons in zip(summary["brain"], summary["neurons"])]
    for ax, (column, mean_column, title, fmt) in zip(axes, PANELS):
        ax.bar(x, summary[mean_column], width=0.62, color=colors, edgecolor="none", zorder=2)
        for i, brain in enumerate(summary["brain"]):
            values = per_neuron.loc[per_neuron["brain"] == brain, column].to_numpy()
            jitter = rng.uniform(-0.17, 0.17, size=len(values))
            ax.scatter(i + jitter, values, s=13, facecolor=TEXT_SECONDARY, edgecolor="white",
                       linewidth=0.7, zorder=3)
        for i, value in enumerate(summary[mean_column]):
            ax.annotate(fmt.format(value), (i, value), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color=TEXT_SECONDARY, zorder=5,
                        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none",
                              "alpha": 0.85})
        for boundary in boundaries:
            ax.axvline(boundary, color=GRID, lw=0.8, zorder=1)
        ax.set_title(title, fontsize=10, color=TEXT_PRIMARY, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_xlim(-0.6, len(summary) - 0.4)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", labelbottom=True)
        style(ax)
    present = list(dict.fromkeys(summary["role"]))
    handles = [Patch(facecolor=ROLE_COLORS.get(role, TEXT_SECONDARY), label=role) for role in present]
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False, fontsize=8,
               bbox_to_anchor=(0.005, 0.968))
    fig.suptitle("Errors per GT neuron by brain (canonical skeleton metrics, metrics_out)",
                 x=0.01, y=0.99, ha="left", fontsize=12, color=TEXT_PRIMARY)
    fig.text(0.01, 0.006, f"bar = brain total / GT neurons (omit: mean of per-neuron %), every neuron weighted once; "
             f"dot = one GT neuron; segmentation ids and the metrics' length-weighted averages in "
             f"errors_per_neuron.csv; {date.today().isoformat()}",
             fontsize=7, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.02, 1, 0.945))
    fig.savefig(out_path, dpi=dpi)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brains", nargs="+", default=ALL_BRAINS)
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--mcl", type=int, default=100, help="MCL of the canonical-gap outputs to read")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)

    brains = [brain for brain in ALL_BRAINS if brain in set(args.brains)]
    brains += [brain for brain in args.brains if brain not in set(ALL_BRAINS)]
    per_neuron = pd.concat([load_brain(brain, args.metrics_dir) for brain in brains], ignore_index=True)
    summary = summarize(per_neuron)

    reach_frames, facts = [], {}
    for brain in brains:
        rows = per_neuron[per_neuron["brain"] == brain]
        frame, brain_facts = load_reach(brain, args.metrics_dir, rows["neuron"].tolist(),
                                        rows["segmentation_id"].iloc[0], args.mcl)
        reach_frames.append(frame)
        facts[brain] = brain_facts
    reach_by_neuron = pd.concat(reach_frames, ignore_index=True)
    reach = summarize_reach(reach_by_neuron, facts)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_neuron.to_csv(args.out_dir / "per_neuron.csv", index=False)
    summary.to_csv(args.out_dir / "errors_per_neuron.csv", index=False)
    plot(summary, per_neuron, args.out_dir / "errors_per_neuron.png", dpi=args.dpi)
    reach_by_neuron.to_csv(args.out_dir / "pool_reach_by_neuron.csv", index=False)
    reach.to_csv(args.out_dir / "pool_reach_per_neuron.csv", index=False)
    plot_reach(reach, args.out_dir / "pool_reach_per_neuron.png", dpi=args.dpi)

    shown = summary[["brain", "role", "neurons", "splits_total", "splits_per_neuron", "merges_total",
                     "merges_per_neuron", "omit_edges_pct_mean"]]
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(shown.to_string(index=False))
        print()
        print(reach[["brain", "neurons", "split_per_neuron", "split_in_pool_per_neuron",
                     "split_present_not_in_pool_per_neuron", "split_segment_absent_per_neuron", "split_reached_share",
                     "merge_per_neuron", "merge_within_20um_per_neuron", "merge_within_150um_per_neuron",
                     "merge_beyond_150um_per_neuron", "merge_reached_share", "merge_within_150um_share"]]
              .to_string(index=False))
    print(f"wrote {args.out_dir / 'errors_per_neuron.png'}, pool_reach_per_neuron.png and four CSVs "
          f"({len(summary)} brains, {len(per_neuron)} neurons)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

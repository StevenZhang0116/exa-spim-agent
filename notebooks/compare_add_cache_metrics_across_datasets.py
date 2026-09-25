"""Compare saved verification metrics and save every figure to verify_stats.

Run in panda on a compute node. Only lightweight *_per_neuron.csv files and
legacy *_summary.csv files are read; no skeleton caches or cloud data are loaded.
The five metrics retain the verifier's definitions. Figures compare canonical
and cache values across brains at mcl100 and mcl10, and across MCLs within each
brain that has both. Missing configurations are reported and skipped.

From the repository root:
    python notebooks/compare_add_cache_metrics_across_datasets.py
    python notebooks/compare_add_cache_metrics_across_datasets.py --stats-dir path/to/verify_stats

Every generated PNG is saved; matching comparison PNGs are overwritten. Existing
verification tables and per-brain verification scatter images are untouched.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


VERIFY_STATS_DIR = Path(__file__).resolve().parent / "verify_stats"
COMPARE_COLS = ["# Splits", "% Split Edges", "% Omit Edges", "% Merged Edges", "Edge Accuracy"]
MCL_ORDER = (100, 10)


def load_per_neuron(path):
    """Load MCL-qualified tables or infer MCL from a legacy matching summary."""
    path = Path(path)
    frame = pd.read_csv(path, dtype={"brain_id": str, "neuron": str})
    required = {"brain_id", "neuron"} | {
        f"{column}_{kind}" for column in COMPARE_COLS for kind in ("cache", "canon")
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path}: empty per-neuron table")
    if frame["brain_id"].isna().any() or not frame["brain_id"].str.fullmatch(r"\d+").all():
        raise ValueError(f"{path}: brain_id must contain numeric IDs")
    if frame["neuron"].isna().any():
        raise ValueError(f"{path}: missing neuron identity")
    mcl_match = re.search(r"_mcl(\d+)_per_neuron\.csv$", path.name)
    if "min_cable_length" not in frame.columns:
        if mcl_match:
            mcl = int(mcl_match[1])
        else:
            brains = frame["brain_id"].unique()
            if len(brains) != 1:
                raise ValueError(f"{path}: cannot infer MCL for multiple brains in a legacy file")
            summary_path = path.parent / f"{brains[0]}_summary.csv"
            if not summary_path.exists():
                raise ValueError(f"{path}: no min_cable_length or matching {summary_path.name}")
            summary = pd.read_csv(summary_path, dtype={"brain_id": str})
            if len(summary) != 1 or "min_cable_length" not in summary:
                raise ValueError(f"{summary_path}: expected one row with min_cable_length")
            if "brain_id" in summary and summary["brain_id"].iloc[0] != brains[0]:
                raise ValueError(f"{summary_path}: brain_id does not match legacy table")
            mcl = summary["min_cable_length"].iloc[0]
        frame.insert(2, "min_cable_length", mcl)
    mcl_values = pd.to_numeric(frame["min_cable_length"], errors="raise")
    if not (np.isfinite(mcl_values) & (mcl_values >= 0) & (mcl_values % 1 == 0)).all():
        raise ValueError(f"{path}: min_cable_length must contain nonnegative integers")
    frame["min_cable_length"] = mcl_values.astype(int)
    if mcl_match and not (frame["min_cable_length"] == int(mcl_match[1])).all():
        raise ValueError(f"{path}: filename and min_cable_length disagree")
    for column in COMPARE_COLS:
        for kind in ("cache", "canon"):
            name = f"{column}_{kind}"
            frame[name] = pd.to_numeric(frame[name], errors="raise")
    frame["_mcl_named"] = bool(mcl_match)
    return frame


def load_saved_results(stats_dir):
    paths = sorted(Path(stats_dir).glob("*_per_neuron.csv"))
    if not paths:
        raise ValueError(f"No *_per_neuron.csv in {stats_dir}; run verify_add_cache_metrics.py "
                         "--mcl 100 (or --mcl 10) on a compute node first")
    combined = pd.concat([load_per_neuron(path) for path in paths], ignore_index=True)
    return (combined.sort_values("_mcl_named", kind="stable")
            .drop_duplicates(["brain_id", "min_cable_length", "neuron"], keep="last")
            .drop(columns="_mcl_named")
            .sort_values(["brain_id", "min_cable_length", "neuron"])
            .reset_index(drop=True))


def plot_metric_grid(data, group_col, group_order, colors, title, legend_title):
    """Construct one headless figure with consistent columns and group colors."""
    present = set(data[group_col].unique())
    groups = [group for group in group_order if group in present]
    if not groups:
        print(f"SKIP: {title} -- no matching saved results", flush=True)
        return None
    figure = Figure(figsize=(4 * len(COMPARE_COLS), 4.2))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, len(COMPARE_COLS))
    plotted_groups = set()
    for axis, column in zip(axes, COMPARE_COLS):
        bounds = []
        for group in groups:
            subset = data[data[group_col] == group]
            canonical = subset[f"{column}_canon"].to_numpy(dtype=float)
            cache = subset[f"{column}_cache"].to_numpy(dtype=float)
            finite = np.isfinite(canonical) & np.isfinite(cache)
            if not finite.any():
                continue
            canonical, cache = canonical[finite], cache[finite]
            axis.scatter(canonical, cache, s=20, alpha=0.7, color=colors[group], label=str(group))
            bounds.extend((canonical.min(), canonical.max(), cache.min(), cache.max()))
            plotted_groups.add(group)
        if bounds:
            lower, upper = min(bounds), max(bounds)
            if lower == upper:
                lower, upper = lower - 0.5, upper + 0.5
            axis.plot([lower, upper], [lower, upper], "--", color="grey", lw=1)
        else:
            axis.text(0.5, 0.5, "No comparable data", ha="center", transform=axis.transAxes)
        axis.set_xlabel("canonical")
        axis.set_ylabel("cache (_add)")
        axis.set_title(column, fontsize=10)
    handles = [Line2D([], [], marker="o", linestyle="none", color=colors[group],
                      markersize=4, alpha=0.7, label=str(group))
               for group in groups if group in plotted_groups]
    if handles:
        figure.legend(handles=handles, title=legend_title, loc="upper right",
                      bbox_to_anchor=(1.0, 0.92), fontsize=8)
    figure.suptitle(title)
    figure.tight_layout(rect=[0, 0, 0.93, 0.93])
    return figure


def save_comparisons(data, output_dir, dpi=160):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    brains = sorted(data["brain_id"].unique())
    configurations = data[["brain_id", "min_cable_length"]].drop_duplicates().sort_values(
        ["brain_id", "min_cable_length"])
    descriptions = [f"{row.brain_id} mcl{row.min_cable_length}" for row in configurations.itertuples(index=False)]
    print(f"{len(configurations)} brain/MCL configurations: {', '.join(descriptions)} "
          f"({len(data)} neurons total)", flush=True)
    palette = colormaps["tab10" if len(brains) <= 10 else "tab20"]
    brain_colors = {brain: palette(index % palette.N) for index, brain in enumerate(brains)}
    written = []

    def save(figure, filename):
        if figure is None:
            return
        path = output_dir / filename
        try:
            figure.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
        finally:
            figure.clear()
        written.append(path)
        print(f"Saved: {path}", flush=True)

    for mcl in MCL_ORDER:
        subset = data[data["min_cable_length"] == mcl]
        figure = plot_metric_grid(subset, "brain_id", brains, brain_colors,
                                  f"Cache _add vs metrics_out across brains - mcl{mcl}", "brain")
        save(figure, f"compare_mcl{mcl}_across_brains.png")
    mcl_colors = {100: "#d97706", 10: "#2563eb"}
    for brain in brains:
        subset = data[data["brain_id"] == brain]
        available = set(subset["min_cable_length"].unique())
        missing = [mcl for mcl in MCL_ORDER if mcl not in available]
        if missing:
            print(f"SKIP {brain}: missing {', '.join(f'mcl{mcl}' for mcl in missing)}", flush=True)
            continue
        figure = plot_metric_grid(subset, "min_cable_length", MCL_ORDER, mcl_colors,
                                  f"Cache _add vs metrics_out - brain {brain} - mcl100 vs mcl10", "MCL (um)")
        save(figure, f"compare_{brain}_mcl100_vs_mcl10.png")
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-dir", type=Path, default=VERIFY_STATS_DIR,
                        help="Directory containing verifier CSVs (default: notebooks/verify_stats)")
    parser.add_argument("--output-dir", type=Path,
                        help="PNG output directory (default: same as --stats-dir)")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args(argv)
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    try:
        data = load_saved_results(args.stats_dir)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    written = save_comparisons(data, args.output_dir or args.stats_dir, dpi=args.dpi)
    print(f"Saved {len(written)} comparison figures. Verification CSVs and cache files are unchanged.", flush=True)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
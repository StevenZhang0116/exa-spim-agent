"""Plot the three-valued label distribution (1 / 0 / NaN) per brain for merge and split.

Reads only the availability sidecars' ``meta.json`` written by
``python -m proofreader_evolve.cli.precompute_availability`` (no tables, caches or
labels are loaded), so it runs in seconds. Labels are three-valued:

    1    error (evaluator label 1; always GT-available)
    0    GT-available, no error
    NaN  not judgeable by GT (evaluator label 0, no GT nearby)

Figure: one row per kind. Left: share of all candidates in each class (NaN dominates).
Right: GT-available candidates only, 1 vs 0 counts, i.e. the rows proofreader_evolve
ranks and fits on (three-valued labels).

Usage (repository root, panda env):
    python notebooks/plot_label_availability.py
    python notebooks/plot_label_availability.py --merge-definition segment_on_gt --out figs/x.png
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "proofreader_evolve" / "feature_tables"
DEFAULT_OUT = ROOT / "figs" / "label_availability_distribution.png"
CURRENT_BRAINS = ("789202", "794491", "794493", "794495", "802449")

# Reference categorical slots (dataviz palette): 1 = slot-2 orange, 0 = slot-1 blue,
# NaN = recessive neutral gray (unknown, not a category to compare).
COLORS = {"1": "#eb6834", "0": "#2a78d6", "nan": "#c9c8c2"}
LABELS = {"1": "1  error", "0": "0  GT-available, no error", "nan": "NaN  not judgeable by GT"}
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_counts(tables_dir=TABLES_DIR, merge_definition="near_gt_150", split_definition="segment_on_gt"):
    """One row per (brain, kind): counts of 1, 0 and NaN from the sidecar metadata."""
    rows = []
    for meta_path in sorted(Path(tables_dir).glob("*/*.availability/meta.json")):
        meta = json.loads(meta_path.read_text())
        kind = meta["kind"]
        definition = merge_definition if kind == "merge" else split_definition
        d = meta["definitions"][definition]
        n_rows, available, positives = meta["rows"], d["available_rows"], d["positives_available"]
        if positives != meta["positives"]:
            raise ValueError(f"{meta_path}: positives outside the available set; regenerate the sidecar")
        rows.append({"brain": meta_path.parent.parent.name, "kind": kind, "definition": definition,
                     "rows": n_rows, "1": positives, "0": available - positives, "nan": n_rows - available})
    if not rows:
        raise FileNotFoundError(f"No availability sidecars under {tables_dir}; run cli.precompute_availability")
    table = pd.DataFrame(rows)
    if table.duplicated(["brain", "kind"]).any():
        raise ValueError("More than one sidecar per brain and kind (stale table digests?)")
    return table


def plot_label_availability(counts, out_path=DEFAULT_OUT):
    """Draw the two-panel figure per kind and save it; returns the output path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [b for b in CURRENT_BRAINS if b in set(counts.brain)] + \
        sorted(set(counts.brain) - set(CURRENT_BRAINS))
    kinds = [k for k in ("merge", "split") if k in set(counts.kind)]
    fig, axes = plt.subplots(len(kinds), 2, figsize=(13, 2.2 + 0.42 * len(order) * len(kinds)),
                             squeeze=False, gridspec_kw={"width_ratios": [1.15, 1]})
    for r, kind in enumerate(kinds):
        sub = counts[counts.kind == kind].set_index("brain").reindex(order).dropna(subset=["rows"])
        y = list(range(len(sub)))[::-1]
        names = [f"{b}{' *' if b not in CURRENT_BRAINS else ''}" for b in sub.index]
        definition = sub["definition"].iloc[0]

        # Left: share of all candidates (stacked 100%).
        ax = axes[r][0]
        left = [0.0] * len(sub)
        for cls in ("1", "0", "nan"):
            share = (sub[cls] / sub["rows"]).tolist()
            ax.barh(y, share, left=left, height=0.62, color=COLORS[cls], edgecolor="white",
                    linewidth=1.5, label=LABELS[cls])
            left = [a + b for a, b in zip(left, share)]
        for yi, (_, row) in zip(y, sub.iterrows()):
            judgeable = (row["1"] + row["0"]) / row["rows"]
            ax.text(1.01, yi, f"{judgeable:.1%} judgeable  ({int(row['rows']):,} rows)",
                    va="center", fontsize=8, color=TEXT_SECONDARY)
        ax.set_xlim(0, 1)
        ax.set_yticks(y, names)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(f"{kind}: share of all candidates ({definition})", loc="left",
                     fontsize=10, color=TEXT_PRIMARY)

        # Right: available rows only, 1 vs 0 counts.
        ax = axes[r][1]
        ax.barh(y, sub["1"], height=0.62, color=COLORS["1"], edgecolor="white", linewidth=1.5)
        ax.barh(y, sub["0"], left=sub["1"], height=0.62, color=COLORS["0"], edgecolor="white", linewidth=1.5)
        for yi, (_, row) in zip(y, sub.iterrows()):
            total = row["1"] + row["0"]
            ax.text(total, yi, f"  {int(row['1']):,} / {int(row['0']):,}  ({row['1'] / total:.0%} errors)"
                    if total else "  none", va="center", fontsize=8, color=TEXT_SECONDARY)
        ax.set_yticks(y, [""] * len(y))
        ax.set_xlim(0, (sub["1"] + sub["0"]).max() * 1.45)
        ax.set_title(f"{kind}: GT-available candidates, 1 / 0", loc="left", fontsize=10, color=TEXT_PRIMARY)

        for ax in axes[r]:
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Three-valued labels per brain (evaluator label + GT availability)", x=0.01, ha="left",
                 fontsize=12, color=TEXT_PRIMARY)
    fig.text(0.01, 0.955, "* older-microscope brain", fontsize=8, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--merge-definition", choices=("near_gt_150", "segment_on_gt"), default="near_gt_150")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    counts = load_counts(args.tables_dir, args.merge_definition)
    print(counts.to_string(index=False))
    print(f"Saved {plot_label_availability(counts, args.out)}")


if __name__ == "__main__":
    main()

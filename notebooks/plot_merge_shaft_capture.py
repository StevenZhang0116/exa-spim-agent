"""Per dataset: merge sites captured by junction candidates and added by tip-to-shaft candidates.

Reads the per-brain results of ``merge_shaft_candidate_reach.py``
(``notebooks/merge_shaft_reach/<brain>_mcl<N>.json``); nothing is hard-coded. A GT merge site
counts as captured when a candidate lies within 20 um of cable (the positive-label radius).

Left panel: share of each dataset's GT merge sites captured by the junction family, added by
tip-to-shaft candidates with a tip gap <= 10 um, added by gaps of 10-20 um, and not captured.
Right panel: the candidates this costs, relative to the junction family of the same dataset
(junction = 1, + tip-to-shaft <= 10 um, + 10-20 um), with the absolute candidate count at the
bar end, so datasets of very different size share one scale. Two panels because sites and candidates are
different measures (no shared axis). Datasets are ordered by their number of GT merge sites;
the pooled row comes first.

    python notebooks/plot_merge_shaft_capture.py [--results-dir DIR] [--mcl 100] [--out FILE.svg]
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "merge_shaft_reach"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"
NOT_CAPTURED = "#dddcd6"
JUNCTION = "#2a78d6"       # categorical slot 1
SHAFT_10 = "#eb6834"       # slot 2
SHAFT_20 = "#1baf7a"       # slot 3 (below 3:1 on the surface: values are labelled at the bar ends)


def load(results_dir: Path, mcl: int) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(results_dir / f"*_mcl{mcl}.json"))):
        result = json.loads(Path(path).read_text())
        f = result["families"]
        rows.append({
            "brain": result["brain"],
            "sites": f["J"]["cache_sites"],
            "junction": f["J"]["cache_covered_20"],
            "add_10": f["J+S10"]["cache_covered_20"] - f["J"]["cache_covered_20"],
            "add_20": f["J+S20"]["cache_covered_20"] - f["J+S10"]["cache_covered_20"],
            "cand_junction": f["J"]["candidates"],
            "cand_10": f["J+S10"]["candidates"] - f["J"]["candidates"],
            "cand_20": f["J+S20"]["candidates"] - f["J+S10"]["candidates"],
        })
    if not rows:
        raise SystemExit(f"no results under {results_dir}; run merge_shaft_candidate_reach.py first")
    rows.sort(key=lambda r: r["sites"])           # bottom to top: smallest ... largest
    pooled = {"brain": f"all {len(rows)}"}
    for key in rows[0]:
        if key != "brain":
            pooled[key] = sum(r[key] for r in rows)
    return rows + [pooled]                         # pooled row drawn on top


def stacked(ax, y, parts, colors, height=0.62):
    left = [0.0] * len(y)
    for values, color in zip(parts, colors):
        ax.barh(y, values, left=left, height=height, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        left = [a + b for a, b in zip(left, values)]
    return left


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, fontweight="semibold", pad=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9.5, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--out", type=Path, default=RESULTS / "merge_shaft_capture.svg")
    args = parser.parse_args(argv)
    rows = load(args.results_dir, args.mcl)

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 0.42 * len(rows) + 2.2), sharey=True,
                                      gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.08})
    y = list(range(len(rows)))
    labels = [r["brain"] for r in rows]

    # left: share of GT merge sites
    share = lambda key: [100.0 * r[key] / r["sites"] for r in rows]  # noqa: E731
    missed = [100.0 * (r["sites"] - r["junction"] - r["add_10"] - r["add_20"]) / r["sites"] for r in rows]
    stacked(left, y, [share("junction"), share("add_10"), share("add_20"), missed],
            [JUNCTION, SHAFT_10, SHAFT_20, NOT_CAPTURED])
    for i, r in enumerate(rows):
        captured = r["junction"] + r["add_10"] + r["add_20"]
        left.text(101.5, i, f"{captured:,} / {r['sites']:,}", va="center", ha="left", fontsize=9, color=INK)
    left.set_xlim(0, 118)
    left.set_xticks([0, 20, 40, 60, 80, 100])
    left.set_yticks(y, labels, fontsize=10, color=INK)
    style(left, "GT merge sites captured within 20 µm (%)")

    # right: candidates relative to the junction family of the same dataset
    totals = stacked(right, y, [[1.0] * len(rows), [r["cand_10"] / r["cand_junction"] for r in rows],
                                [r["cand_20"] / r["cand_junction"] for r in rows]], [JUNCTION, SHAFT_10, SHAFT_20])
    width = max(totals)
    for i, r in enumerate(rows):
        total = r["cand_junction"] + r["cand_10"] + r["cand_20"]
        text = f"{total / 1e6:.2f} M" if total >= 1e6 else f"{total / 1e3:,.0f} k"
        right.text(totals[i] + 0.015 * width, i, text, va="center", ha="left", fontsize=9, color=INK)
    right.set_xlim(0, width * 1.16)
    right.set_xticks(range(0, int(width) + 1))
    right.set_xticklabels([f"×{t}" for t in range(0, int(width) + 1)])
    style(right, "Candidates relative to the junction family")
    right.tick_params(axis="y", length=0)

    # separate the pooled row
    for ax in (left, right):
        ax.axhline(len(rows) - 1.5, color=INK_2, linewidth=0.8, zorder=4)
    left.get_yticklabels()[-1].set_fontweight("semibold")

    handles = [Patch(color=JUNCTION, label="junction (degree ≥ 3)"),
               Patch(color=SHAFT_10, label="+ tip-to-shaft, gap ≤ 10 µm"),
               Patch(color=SHAFT_20, label="+ tip-to-shaft, gap 10–20 µm"),
               Patch(color=NOT_CAPTURED, label="not captured")]
    legend = fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10,
                        bbox_to_anchor=(0.5, 0.0))
    for text in legend.get_texts():
        text.set_color(INK)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.14)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

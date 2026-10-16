"""Merge sites captured within 20 um, by site class: which candidate family reaches which kind of merge.

Reads the per-brain results of ``merge_shaft_candidate_reach.py``
(``notebooks/merge_shaft_reach/<brain>_mcl<N>.json``); nothing is hard-coded. Site classes are
those of the merge topology census (``merge_topology_diagnosis.py`` / ``merge_diagnosis_followup.py``):
bridge sites have no junction within 30 um of cable and are split by the nearest junction
(30-150 um, beyond 150 um, none on the segment); every other site is a branch site. The bridge
counts come from each JSON's ``bridge`` record; the branch row is the total minus the bridge
rows, so no class membership is assumed.

One row per class, pooled over brains: share captured by the junction family, added by
tip-to-shaft candidates (tip gap <= 20 um), and still missed; the bar end reads
"junction + tip-to-shaft / sites". The top row is all sites.

    python notebooks/plot_merge_site_class_capture.py [--results-dir DIR] [--mcl 100] [--out FILE.svg]
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
JUNCTION = "#2a78d6"       # categorical slot 1
SHAFT = "#eb6834"          # slot 2
MISSED = "#dddcd6"
CLASSES = [  # (bridge-record key or None for branch, row label)
    (None, "branch: junction ≤ 30 µm"),
    ("junction_30_150", "bridge: junction 30–150 µm"),
    ("junction_beyond_150", "bridge: junction > 150 µm"),
    ("pure_path", "bridge: unbranched segment"),
]


def load(results_dir: Path, mcl: int) -> list[dict]:
    """Pooled counts per class: sites, captured by junction, added by tip-to-shaft (<= 20 um)."""
    totals = {key: {"sites": 0, "junction": 0, "both": 0} for key, _ in CLASSES}
    paths = sorted(glob.glob(str(results_dir / f"*_mcl{mcl}.json")))
    if not paths:
        raise SystemExit(f"no results under {results_dir}; run merge_shaft_candidate_reach.py first")
    for path in paths:
        result = json.loads(Path(path).read_text())
        f = result["families"]
        bridge = {"sites": 0, "junction": 0, "both": 0}
        for key, _ in CLASSES[1:]:
            record = result["bridge"].get(key, {"n": 0, "covered_20_J": 0, "covered_20_J+S20": 0})
            for name, value in (("sites", record["n"]), ("junction", record["covered_20_J"]),
                                ("both", record["covered_20_J+S20"])):
                totals[key][name] += value
                bridge[name] += value
        totals[None]["sites"] += f["J"]["cache_sites"] - bridge["sites"]
        totals[None]["junction"] += f["J"]["cache_covered_20"] - bridge["junction"]
        totals[None]["both"] += f["J+S20"]["cache_covered_20"] - bridge["both"]
    classes = [{"label": label, **totals[key]} for key, label in CLASSES]
    pooled = {"label": f"all sites ({len(paths)} brains)",
              **{name: sum(r[name] for r in classes) for name in ("sites", "junction", "both")}}
    rows = classes[::-1] + [pooled]   # drawn bottom to top: read top-down as all, branch, bridge ...
    for r in rows:
        r["shaft"] = r["both"] - r["junction"]
        r["missed"] = r["sites"] - r["both"]
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--out", type=Path, default=RESULTS / "merge_site_class_capture.svg")
    args = parser.parse_args(argv)
    rows = load(args.results_dir, args.mcl)

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(rows) + 1.9))
    ax.set_facecolor(SURFACE)
    y = list(range(len(rows)))
    left = [0.0] * len(rows)
    for key, color in (("junction", JUNCTION), ("shaft", SHAFT), ("missed", MISSED)):
        values = [100.0 * r[key] / r["sites"] if r["sites"] else 0.0 for r in rows]
        ax.barh(y, values, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        left = [a + b for a, b in zip(left, values)]
    for i, r in enumerate(rows):
        ax.text(101.5, i, f"{r['junction']:,} + {r['shaft']:,} / {r['sites']:,}", va="center", ha="left",
                fontsize=9.5, color=INK)
    ax.axhline(len(rows) - 1.5, color=INK_2, linewidth=0.8, zorder=4)
    ax.set_yticks(y, [r["label"] for r in rows], fontsize=10, color=INK)
    ax.get_yticklabels()[-1].set_fontweight("semibold")
    ax.set_xlim(0, 122)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9.5, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_title("GT merge sites captured within 20 µm, by site class (%)", loc="left",
                 fontsize=12, color=INK, fontweight="semibold", pad=10)
    handles = [Patch(color=JUNCTION, label="junction (degree ≥ 3)"),
               Patch(color=SHAFT, label="+ tip-to-shaft, gap ≤ 20 µm"),
               Patch(color=MISSED, label="missed")]
    legend = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10,
                        bbox_to_anchor=(0.5, 0.0))
    for text in legend.get_texts():
        text.set_color(INK)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.86, bottom=0.2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")
    for r in rows[::-1]:
        print(f"  {r['label']}: sites {r['sites']}, junction {r['junction']}, tip-to-shaft +{r['shaft']}, missed {r['missed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

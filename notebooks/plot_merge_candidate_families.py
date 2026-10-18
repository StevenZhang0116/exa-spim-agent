"""Schematic of the two merge candidate families (illustration, not data).

Left: junction family (the whole merge pool until 2026-10-16): every degree >= 3 fragment
node; neurites that cross or touch leave a branch point in the merged skeleton.
Right: tip-to-shaft family (measured in ``merge_shaft_candidate_reach.py``, added to the pool
on 2026-10-16 with a 20 um gap, ``candidate_rules.py``): the shaft node nearest a foreign
segment tip; a side-by-side fusion leaves no branch point, but the unfused rest of neuron B
ends next to the merged shaft.

Bottom row: the site classes of the merge topology census (``merge_topology_diagnosis.py`` and
``merge_diagnosis_followup.py``), by the cable distance from the GT merge site to the nearest
junction of its segment: branch (a junction within 30 um) and the three bridge subclasses
(nearest junction 30-150 um away, beyond 150 um, or none on the segment). Distances along the
skeleton are marked by ticks; the drawing is not to scale.

GT neurons are colored bands (A, B); fragment skeletons are dark lines, one continuous line
per segment. Colors: the first three categorical slots of the dataviz reference palette
(validated all-pairs, light mode); candidates carry direct labels.

    python notebooks/plot_merge_candidate_families.py
writes notebooks/merge_candidate_families_illustration.svg.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = Path(__file__).resolve().parent / "merge_candidate_families_illustration"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
SKELETON = "#3a3936"     # fragment skeletons (one line = one segment)
UNTRACED = "#a3a29a"     # a segment with no GT tracing
NEURON_A = "#2a78d6"     # slot 1
NEURON_B = "#eb6834"     # slot 2
CANDIDATE = "#1baf7a"    # slot 3 (below 3:1 on the surface: direct labels)
BAND_ALPHA = 0.38


def densify(points, step=5.0):
    points = np.asarray(points, float)
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.extend(a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:])
    return np.array(out)


def band(ax, points, color, name=None, name_at=None):
    p = np.asarray(points, float)
    ax.plot(p[:, 0], p[:, 1], color=color, alpha=BAND_ALPHA, lw=13, solid_capstyle="round", zorder=1)
    if name:
        ax.text(*name_at, name, color=INK, fontsize=12, fontweight="semibold", ha="center", va="center", zorder=6)


def skeleton(ax, points, color=SKELETON, tips=()):
    p = densify(points)
    ax.plot(p[:, 0], p[:, 1], color=color, lw=1.6, zorder=3)
    ax.scatter(p[:, 0], p[:, 1], s=7, color=color, zorder=4, linewidths=0)
    for tip in tips:
        ax.scatter([tip[0]], [tip[1]], s=30, color=color, edgecolors=SURFACE, linewidths=1.0, zorder=5)


def candidate(ax, xy, label, label_at, ha="center"):
    ax.scatter([xy[0]], [xy[1]], s=280, facecolors="none", edgecolors=CANDIDATE, linewidths=2.6, zorder=6)
    ax.annotate(label, xy=xy, xytext=label_at, fontsize=10, color=INK, ha=ha, va="center", zorder=7,
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8, shrinkA=3, shrinkB=10))


def merge_site(ax, xy):
    ax.scatter([xy[0]], [xy[1]], s=75, marker="X", color=INK, edgecolors=SURFACE, linewidths=0.8, zorder=8)


def frame(ax, title):
    ax.set_xlim(-8, 112)
    ax.set_ylim(-45, 62)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="semibold", pad=6)


def junction_panel(ax):
    a_main, a_branch, b = [(0, 0), (100, 0)], [(82, 0), (96, 34)], [(32, 48), (66, -40)]
    band(ax, a_main, NEURON_A, "A", (-4, 0))
    band(ax, a_branch, NEURON_A)
    band(ax, b, NEURON_B, "B", (28, 54))
    skeleton(ax, a_main, tips=[(0, 0), (100, 0)])          # one merged segment: A and B together
    skeleton(ax, a_branch, tips=[a_branch[-1]])
    skeleton(ax, b, tips=[b[0], b[-1]])
    merge_site(ax, (50.5, 0))
    candidate(ax, (50.5, 0), "merge", (28, -24))
    candidate(ax, (82, 0), "true branch", (94, -22))
    frame(ax, "Junction: degree ≥ 3 node")


def shaft_panel(ax):
    # GT: A runs left to right; B comes down from the upper left, touches A at x=50 and leaves down-right.
    a = [(0, 0), (100, 0)]
    b = [(14, 52), (46, 6), (50, 0), (90, -38)]
    band(ax, a, NEURON_A, "A", (-4, 0))
    band(ax, b, NEURON_B, "B", (10, 57))
    # merged segment: A's left half continues into B's lower half, one unbranched skeleton
    skeleton(ax, [(0, 0), (50, 0), (90, -38)], tips=[(0, 0), (90, -38)])
    # the rest of A and the rest of B are cut off; their tips stop at the junction-free merge
    skeleton(ax, [(100, 0), (60, 0)], tips=[(100, 0), (60, 0)])
    skeleton(ax, [(14, 52), (42, 12)], tips=[(14, 52), (42, 12)])
    # an unrelated, untraced neurite whose tip happens to pass the shaft
    skeleton(ax, [(8, -40), (16, -22), (22, -9)], color=UNTRACED, tips=[(8, -40), (22, -9)])
    ax.text(72, 18, "cut-off tips", fontsize=10, color=INK, ha="left", va="center", zorder=7)
    for tip in ((42, 12), (60, 0)):
        ax.annotate("", xy=tip, xytext=(71, 18),
                    arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8, shrinkA=0, shrinkB=4), zorder=6)
    merge_site(ax, (50, 0))
    candidate(ax, (50, 0), "merge", (38, -28))
    candidate(ax, (25, 0), "unrelated tip", (6, 24), ha="left")
    frame(ax, "Tip-to-shaft: shaft node next to a foreign tip")


SITE_X = 40.0                      # merge site of the class panels
TICKS = {"30": 12.0, "150": 40.0}   # schematic offsets of 30 and 150 um along the skeleton


def class_panel(ax, title, branch_at=None):
    """One site class. Within 30 um of the site, ``branch_at`` is where B joins A (branch);
    otherwise A runs into B through the site unbranched and ``branch_at`` is a true branch of B."""
    if branch_at is not None and branch_at - SITE_X < TICKS["30"]:   # branch: B joins A next to the site
        band(ax, [(-2, 0), (104, 0)], NEURON_A)
        band(ax, [(branch_at, 0), (branch_at + 12, 24)], NEURON_B)
        skeleton(ax, [(-2, 0), (104, 0)], tips=[(-2, 0), (104, 0)])
        skeleton(ax, [(branch_at, 0), (branch_at + 12, 24)], tips=[(branch_at + 12, 24)])
    else:                                                     # bridge: A continues into B, unbranched
        band(ax, [(-2, 0), (SITE_X, 0)], NEURON_A)
        band(ax, [(SITE_X, 0), (104, 0)], NEURON_B)
        skeleton(ax, [(-2, 0), (104, 0)], tips=[(-2, 0), (104, 0)])
        if branch_at is not None:
            band(ax, [(branch_at, 0), (branch_at + 9, 22)], NEURON_B)
            skeleton(ax, [(branch_at, 0), (branch_at + 9, 22)], tips=[(branch_at + 9, 22)])
    if branch_at is not None:
        ax.scatter([branch_at], [0], s=200, facecolors="none", edgecolors=CANDIDATE, linewidths=2.2, zorder=6)
    for name, offset in TICKS.items():
        for x in (SITE_X - offset, SITE_X + offset):
            ax.plot([x, x], [-3.5, 3.5], color=INK_2, lw=1.1, zorder=5)
        ax.text(SITE_X + offset, -8, f"{name} µm", fontsize=8.5, color=INK_2, ha="center", va="top", zorder=7)
    merge_site(ax, (SITE_X, 0))
    ax.set_xlim(-6, 106)
    ax.set_ylim(-16, 30)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="semibold", pad=4)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig = plt.figure(figsize=(14, 8.2))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.42], hspace=0.28, wspace=0.08)
    top = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4])]
    bottom = [fig.add_subplot(grid[1, i]) for i in range(4)]
    for ax in top + bottom:
        ax.set_facecolor(SURFACE)
    junction_panel(top[0])
    shaft_panel(top[1])
    class_panel(bottom[0], "branch: junction ≤ 30 µm", branch_at=SITE_X + 6)
    class_panel(bottom[1], "bridge: junction 30–150 µm", branch_at=SITE_X + 26)
    class_panel(bottom[2], "bridge: junction > 150 µm", branch_at=SITE_X + 52)
    class_panel(bottom[3], "bridge: unbranched segment")
    handles = [
        Line2D([], [], color=NEURON_A, alpha=BAND_ALPHA, lw=10, solid_capstyle="round", label="GT neuron A"),
        Line2D([], [], color=NEURON_B, alpha=BAND_ALPHA, lw=10, solid_capstyle="round", label="GT neuron B"),
        Line2D([], [], color=SKELETON, lw=1.6, marker="o", ms=3, label="fragment segment"),
        Line2D([], [], color=UNTRACED, lw=1.6, marker="o", ms=3, label="untraced segment"),
        Line2D([], [], color="none", marker="o", ms=12, markerfacecolor="none", markeredgecolor=CANDIDATE,
               markeredgewidth=2.4, label="candidate"),
        Line2D([], [], color="none", marker="X", ms=8, markerfacecolor=INK, markeredgecolor=SURFACE,
               label="GT merge site"),
    ]
    legend = fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=10,
                        bbox_to_anchor=(0.5, 0.01), handlelength=2.4, columnspacing=1.8)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.08)
    row = bottom[0].get_position()
    fig.text(0.02, row.y1 + 0.075, "Merge site classes: cable distance from \u00d7 to the nearest junction",
             fontsize=12.5, color=INK, fontweight="semibold", ha="left")
    fig.savefig(OUT.with_suffix(".svg"))
    print(f"wrote {OUT.with_suffix('.svg')}")


if __name__ == "__main__":
    main()

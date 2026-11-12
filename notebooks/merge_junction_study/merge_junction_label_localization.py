#!/usr/bin/env python3
"""Do the merge labels of the candidates agree with the ground truth around them?

Background (``proofreader_evolve/TODO.md`` section 7): inside the ``junction`` family the accepted
merge scorers reach a whole-brain holdout AUROC of only 0.63-0.74, and longer-range fragment
context did not help (``merge_junction_fragment_context.py``). A candidate is positive when it
lies within 20 um of fragment cable of a recorded GT merge site, and 0 when its segment carries
a GT label and a GT node lies within 150 um; so a positive can land on a true branch point next
to the false connection, a false connection without a recorded site is labeled 0, and a 0 can
sit where no tracing runs. This script measures, offline and on TRAIN brains only, how often
each happens, by reading the GT tracing around every GT-labeled merge candidate.

``evidence``
    For every GT-labeled merge candidate, each arm is followed over ``WALK_UM`` of fragment
    cable (through further branch points). Arm nodes farther than ``NEAR_UM`` from the candidate
    are compared with the GT tracing: a node is supported when a GT node lies within ``GT_UM``
    (the distance the cache's own geometric walk uses). An arm is

    * ``covered`` by a GT neuron: at least ``MIN_NODES`` supported nodes and at least half of them
      supported (the neuron is the one that supports most nodes);
    * ``free``: no supported node; ``long`` when it runs the whole ``WALK_UM``;
    * otherwise undecided.

    The candidate then falls in one category:

    =============  ==============================================================================
    ``two_gt``     covered arms belong to different GT neurons (a false connection at this node)
    ``leaves_gt``  a covered arm and a long free arm (the fragment leaves the tracing here: the
                   pattern the geometric walk records; a merge if the tracing is complete)
    ``gt_branch``  every arm covered by the same GT neuron (a true branch point by the tracing;
                   for a ``tip_to_shaft`` shaft node: the shaft runs along one GT neuron)
    ``gt_spur``    covered arms of one neuron, the other arms free and shorter than ``WALK_UM``
    ``mixed``      covered arms of one neuron and undecided arms
    ``no_gt``      no covered arm
    =============  ==============================================================================

    The category is also computed with a wider GT distance (``WIDE_UM``), to see how much of
    ``no_gt`` is a tracing that runs a little farther from the skeleton. To explain a label that
    disagrees with the category, the quantities the cache's site rule
    (``canonical_labeling.geometric_merge_sites``) depends on are stored per row:

    * the cable distance to the nearest recorded site (sites snapped like ``candidate_rules``)
      and its source (``geometric_walk`` or ``two_gt_junction``);
    * the Euclidean distance to the nearest GT node, and the cable distance to the nearest
      fragment node within ``GT_UM`` of a GT node (the "contact");
    * at the contact, whether the GT node carries this segment's label and how many connected
      GT nodes carry it (the walk drops a contact below ``passthru_min_cc`` as a pass-through);
    * whether the segment is in ``gt_merge_labels``;
    * for ``leaves_gt``, the free region behind the long free arms (followed until it comes back
      within ``GT_UM`` of a GT node): cable, fragment leaves, the largest leaf-to-GT distance (a
      walk starts only at a leaf more than 50 um from every GT node) and the number of other
      contacts (one walk records one site).

``examples``
    Per brain, sheets of eight junction candidates (fixed seed) for the ``no_gt`` negatives, the
    ``leaves_gt`` negatives and the ``leaves_gt`` positives. Each candidate gets three panels in
    one view: the image (maximum projection of a 160 x 160 x 48 um box around the candidate,
    read from the brain's reviewed image like ``scripts/plot_top_merge_predictions.py``), the
    image with the fragments in the box (the candidate's segment in cyan) and the image with the
    GT nodes in the box (coloured by neuron; faint where they lie in front of or behind the
    projected slab, up to 80 um from the candidate) and the recorded sites. The first panel is on a
    linear scale up to the 99.9th percentile; the two overlays use a square-root scale so that
    dim neurites beside a bright one stay visible. The fragment within 150 um of cable of the
    candidate is drawn whether or not it lies inside the projected slab. Needs cloud access.

``summarize``
    Counts of label x category per brain and family; positives by site source; why ``leaves_gt``
    negatives have no site; where ``no_gt`` negatives sit; and, with the holdout scores of a
    finished run's accepted merge scorer, the mean within-brain rank of every cell and the
    junction AUROC on all rows against the rows whose label and category agree (positives in
    ``two_gt``/``leaves_gt``, negatives in ``gt_branch``/``gt_spur``).

Run on a compute node in the ``panda`` environment from the repository root::

    python -u notebooks/merge_junction_study/merge_junction_label_localization.py evidence --brains 747807 802449
    python -u notebooks/merge_junction_study/merge_junction_label_localization.py summarize --run proofreader_evolve/runs/<run>
    AWS_EC2_METADATA_DISABLED=true python -u notebooks/merge_junction_study/merge_junction_label_localization.py examples --brains 747807

Outputs under ``notebooks/merge_junction_study/merge_junction_label_localization/`` (gitignored): ``evidence_<brain>.npz``,
``examples/*.png``, ``counts.csv``, ``summary.md``. Read-only for caches, tables and runs.

Limits. TRAIN brains only. The categories are a second reading of the same tracing, not new
truth: ``leaves_gt`` is a merge only where the tracing is complete (an untraced branch of the same
neuron looks identical), and a GT neuron passing within ``GT_UM`` of an arm counts as covering
it. The reasons given for a missing site are read from the stored quantities, not from a rerun
of the walk (its result depends on the order the leaves are visited). Thresholds fixed, not tuned.
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import heapq
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from merge_junction_fragment_context import TRAIN_BRAINS, accepted_merge_scores, log  # noqa: E402

OUT = Path(__file__).resolve().parent / "merge_junction_label_localization"
WALK_UM, NEAR_UM, GT_UM, WIDE_UM, MIN_NODES = 60.0, 10.0, 6.0, 12.0, 3
REACH_UM = 300.0           # how far along the fragment a site or a contact is looked for
WALK_START_UM = 50.0       # canonical_labeling.MERGE_DIST_AWAY_UM
REGION_CAP = 50_000        # nodes of a free region
CATEGORIES = ["two_gt", "leaves_gt", "gt_branch", "gt_spur", "mixed", "no_gt"]
MERGE_LIKE, BRANCH_LIKE = ("two_gt", "leaves_gt"), ("gt_branch", "gt_spur")
EXAMPLES = 8
PATCH_UM, SLAB_UM = 160.0, 48.0   # example sheets: in-plane size and depth of the image box
TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]


def step(xyz, a, b):
    return float(np.linalg.norm(xyz[a] - xyz[b]))


def arm_nodes(graph, xyz, junction, first):
    """Nodes of one arm between ``NEAR_UM`` and ``WALK_UM`` of cable from the junction (the fragment
    graph is a forest, so a depth-first walk never returns), and whether the arm runs past ``WALK_UM``."""
    nodes, long = [], False
    stack = [(first, junction, step(xyz, first, junction))]
    while stack:
        node, previous, travelled = stack.pop()
        if travelled > WALK_UM:
            long = True
            continue
        if travelled >= NEAR_UM:
            nodes.append(node)
        for following in graph.adj[node]:
            if following != previous:
                stack.append((int(following), node, travelled + step(xyz, following, node)))
    return nodes, long


def categorize(arms):
    """``arms``: per arm ``(neuron or None, free, long)``."""
    neurons = {neuron for neuron, _, _ in arms if neuron is not None}
    if not neurons:
        return "no_gt"
    if len(neurons) > 1:
        return "two_gt"
    if any(free and long for _, free, long in arms):
        return "leaves_gt"
    if all(neuron is not None for neuron, _, _ in arms):
        return "gt_branch"
    if all(neuron is not None or free for neuron, free, _ in arms):
        return "gt_spur"
    return "mixed"


def read_arm(nodes, long, gt_dist, gt_index, gt_neuron, radius):
    neuron, supported = None, 0
    if nodes:
        hit = gt_dist[nodes] <= radius
        supported = int(hit.sum())
        if supported >= MIN_NODES and supported >= 0.5 * len(nodes):
            neuron = Counter(gt_neuron[gt_index[np.asarray(nodes)[hit]]].tolist()).most_common(1)[0][0]
    return neuron, supported == 0, long


def nearest_contact(graph, xyz, gt_dist, start):
    """Cable distance from ``start`` to the nearest fragment node within ``GT_UM`` of a GT node."""
    best, heap = {start: 0.0}, [(0.0, start)]
    while heap:
        distance, node = heapq.heappop(heap)
        if distance > best.get(node, np.inf):
            continue
        if gt_dist[node] < GT_UM:
            return distance, node
        for following in graph.adj[node]:
            following = int(following)
            reached = distance + step(xyz, following, node)
            if reached <= REACH_UM and reached < best.get(following, np.inf):
                best[following] = reached
                heapq.heappush(heap, (reached, following))
    return np.inf, -1


def free_region(graph, xyz, gt_dist, junction, firsts):
    """The fragment behind the free arms, up to where it comes back within ``GT_UM`` of a GT node."""
    cable, leaves, contacts, visited = 0.0, [], 0, 0
    stack = [(first, junction, step(xyz, first, junction)) for first in firsts]
    cable += sum(item[2] for item in stack)
    while stack:
        node, previous, travelled = stack.pop()
        visited += 1
        if visited > REGION_CAP:
            return cable, leaves, contacts, True
        if travelled > NEAR_UM and gt_dist[node] < GT_UM:
            contacts += 1
            continue
        onward = [int(n) for n in graph.adj[node] if n != previous]
        if not onward:
            leaves.append(node)
        for following in onward:
            length = step(xyz, following, node)
            cable += length
            stack.append((following, node, travelled + length))
    return cable, leaves, contacts, False


def same_label_run(gt, gt_label, start, label, cap=60):
    """Connected GT nodes carrying ``label`` around ``start`` (0 when ``start`` carries another label)."""
    if int(gt_label[start]) != label:
        return 0
    seen, stack = {start}, [start]
    while stack and len(seen) < cap:
        for following in gt.adj[stack.pop()]:
            if following not in seen and int(gt_label[following]) == label:
                seen.add(following)
                stack.append(following)
    return len(seen)


def site_cable_distances(graph, xyz, segment, kept, sites):
    """Cable distance from every node within ``REACH_UM`` to the nearest recorded site; a site snaps
    to the nearest node of its own segment within ``SITE_SNAP_MAX_UM`` as in ``candidate_rules``."""
    from scipy.spatial import cKDTree
    from candidate_rules import SITE_SNAP_MAX_UM

    tree = cKDTree(xyz[kept])
    best, source_of, heap = {}, {}, []
    for site in sites:
        point = np.asarray(site["xyz"], dtype=np.float64)
        near = kept[np.asarray(tree.query_ball_point(point, r=SITE_SNAP_MAX_UM), dtype=np.int64)]
        own = near[segment[near] == int(site["segment_id"])]
        if not len(own):
            continue
        start = int(own[int(np.argmin(np.linalg.norm(xyz[own] - point, axis=1)))])
        best[start] = 0.0
        source_of[start] = str(site.get("source", "geometric_walk"))
        heap.append((0.0, start, start))
    heapq.heapify(heap)
    origin = {}
    while heap:
        distance, node, start = heapq.heappop(heap)
        if distance > best.get(node, np.inf):
            continue
        origin[node] = start
        for following in graph.adj[node]:
            following = int(following)
            reached = distance + step(xyz, following, node)
            if reached <= REACH_UM and reached < best.get(following, np.inf):
                best[following] = reached
                heapq.heappush(heap, (reached, following, start))
    return best, {node: source_of[start] for node, start in origin.items()}


def local_fragment(graph, xyz, junction, limit_um=150.0):
    """Nodes (with their cable distance) and edges of the fragment within ``limit_um`` of the candidate."""
    reach, edges, stack = {junction: 0.0}, [], [(junction, -1, 0.0)]
    while stack:
        node, previous, travelled = stack.pop()
        for following in graph.adj[node]:
            following = int(following)
            reached = travelled + step(xyz, following, node)
            if following != previous and reached <= limit_um:
                reach[following] = reached
                edges.append((node, following))
                stack.append((following, node, reached))
    return reach, edges


def draw_example(axes, image, anisotropy, graph, xyz, component, gt_xyz, gt_neuron, gt_label, sites,
                 junction, segment_id, title, marks=()):
    """Three panels for one candidate, one view: the image (maximum projection of a ``PATCH_UM`` by
    ``PATCH_UM`` by ``SLAB_UM`` box, projected along the axis the arms spread least in), the same
    image with the fragments in the box, and with the GT nodes in the box. Follows
    ``scripts/plot_top_merge_predictions.py`` (reader, intensity range, colours). ``marks`` holds
    ``(xyz, scatter keywords)`` pairs drawn on the GT panel (used by ``merge_added_sites_check.py``)."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize, PowerNorm
    from scripts.plot_top_merge_predictions import JUNCTION_COLOR, SEGMENT_COLOR, read_patch

    reach, edges = local_fragment(graph, xyz, junction)
    arms = np.array([node for node, distance in reach.items() if distance <= WALK_UM])
    drop = int(np.argmin(xyz[arms].std(axis=0)))
    horizontal, vertical = [axis for axis in range(3) if axis != drop]
    patch_um = np.full(3, PATCH_UM)
    patch_um[drop] = SLAB_UM
    for axis, name in zip(axes, (title, "fragments in the box (cyan = candidate's segment)",
                                 "GT (dot = this segment's label, x = other / none; faint = outside the slab)")):
        axis.set_title(name, fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_facecolor("black")
    try:
        patch, origin = read_patch(image, xyz[junction], anisotropy, patch_um)
    except Exception as error:  # a failed read leaves the panels empty, with the reason
        axes[0].text(0.5, 0.5, f"image read failed: {type(error).__name__}: {str(error)[:80]}", color="white",
                     transform=axes[0].transAxes, ha="center", fontsize=7, wrap=True)
        return False
    projection = np.max(patch, axis=2 - drop)
    low, high = float(np.min(patch)), float(np.percentile(patch, 99.9))
    height, width = projection.shape
    extent = (-0.5 * anisotropy[horizontal], (width - 0.5) * anisotropy[horizontal],
              (height - 0.5) * anisotropy[vertical], -0.5 * anisotropy[vertical])
    lower = origin[::-1] * anisotropy                       # um of the box's first voxel, XYZ
    upper = (origin + np.asarray(patch.shape))[::-1] * anisotropy
    flat = lambda points: (np.asarray(points, dtype=np.float64) - lower)[..., [horizontal, vertical]]  # noqa: E731
    for index, axis in enumerate(axes):
        # The first panel is linear as in the repository's plotters; under the overlays a square-root
        # scale keeps a neurite ten times dimmer than the brightest one visible.
        axis.imshow(projection, cmap="viridis" if index == 0 else "gray", origin="upper", extent=extent,
                    norm=Normalize(low, max(high, low + 1)) if index == 0 else PowerNorm(0.5, low, max(high, low + 1)),
                    interpolation="nearest")
        if index == 0:
            axis.scatter(*flat(xyz[junction]), marker="+", s=160, linewidths=1.8, color=JUNCTION_COLOR, zorder=6)
        else:
            axis.scatter(*flat(xyz[junction]), marker="o", s=160, linewidths=1.8, facecolor="none",
                         edgecolor="red", zorder=6)
        axis.set_xlim(extent[:2])
        axis.set_ylim(extent[2:])
        axis.set_aspect("equal")
    inside = np.flatnonzero(np.all((xyz >= lower) & (xyz < upper), axis=1))
    own = component[inside] == component[junction]
    others = np.unique(component[inside[~own]])
    for index, other in enumerate(others):
        chosen = inside[component[inside] == other]
        axes[1].scatter(*flat(xyz[chosen]).T, s=5, color=colorsys.hsv_to_rgb((index / max(len(others), 1) + 0.05) % 1.0,
                                                                              0.9, 1.0), zorder=3)
    axes[1].add_collection(LineCollection(flat(xyz[np.asarray(edges, dtype=np.int64).reshape(-1, 2)]),
                                          colors=[SEGMENT_COLOR], linewidths=1.6, zorder=4))
    axes[1].scatter(*flat(xyz[inside[own]]).T, s=7, color=SEGMENT_COLOR, zorder=4)
    # GT nodes in the slab, and faintly those in front of or behind it (to PATCH_UM / 2 from the
    # candidate), so that a tracing leaving the slab is not read as a tracing that ends.
    deep_lower, deep_upper = lower.copy(), upper.copy()
    deep_lower[drop], deep_upper[drop] = xyz[junction][drop] - PATCH_UM / 2, xyz[junction][drop] + PATCH_UM / 2
    around = np.flatnonzero(np.all((gt_xyz >= deep_lower) & (gt_xyz < deep_upper), axis=1))
    in_slab = np.all((gt_xyz[around] >= lower) & (gt_xyz[around] < upper), axis=1)
    for index, neuron in enumerate(np.unique(gt_neuron[around])):
        colour = TAB10[index % len(TAB10)]
        for slab, alpha in ((True, 1.0), (False, 0.3)):
            chosen = around[(gt_neuron[around] == neuron) & (in_slab == slab)]
            carries = gt_label[chosen] == segment_id
            axes[2].scatter(*flat(gt_xyz[chosen[carries]]).T, s=10, color=colour, alpha=alpha, zorder=4)
            axes[2].scatter(*flat(gt_xyz[chosen[~carries]]).T, s=16, color=colour, marker="x", linewidths=0.9,
                            alpha=alpha, zorder=4)
    for site in sites.get(segment_id, []):
        point = np.asarray(site["xyz"], dtype=np.float64)
        if np.all((point >= lower) & (point < upper)):
            axes[2].scatter(*flat(point), s=220, marker="*", color="gold", edgecolor="black", zorder=5)
    for point, style in marks:  # further points of interest, drawn when inside the deeper box
        point = np.asarray(point, dtype=np.float64)
        if np.all((point >= deep_lower) & (point < deep_upper)):
            axes[2].scatter(*flat(point), zorder=6, **style)
    return True


def brain_examples(brain):
    """Sheets of ``EXAMPLES`` junction candidates per group (fixed seed) with the image: needs cloud access."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from agentic_neuron_proofreader.utils.img_util import TensorStoreImage
    from proofreader_evolve.harness.dataset import load_cached_graphs
    from proofreader_evolve.harness.native_pool import cache_path
    from scripts.dataset_config import get_cache_img_path

    started = time.monotonic()
    stored = np.load(OUT / f"evidence_{brain}.npz", allow_pickle=False)
    y, category, family, node = stored["truth"].astype(int), stored["category"], stored["family"], stored["node"]
    value = {name: stored["values"][:, i] for i, name in enumerate(stored["names"].tolist())}
    graph, gt, payload = load_cached_graphs(str(cache_path(brain, 100)), expect_brain=brain, expect_mcl=100)
    xyz = np.asarray(graph.node_xyz, dtype=np.float64)
    component = np.asarray(graph.node_component_id, dtype=np.int64)
    gt_xyz = np.asarray(gt.node_xyz, dtype=np.float64)
    gt_neuron = np.asarray(gt.node_component_id, dtype=np.int64)
    gt_label = np.asarray(payload["gt_node_canonical_label"], dtype=np.int64)
    anisotropy = np.asarray(payload["anisotropy"], dtype=np.float64)
    sites = {}
    for site in payload.get("gt_merge_sites") or []:
        sites.setdefault(int(site["segment_id"]), []).append(site)
    image_path = get_cache_img_path(cache_path(brain, 100), payload)
    image = TensorStoreImage(image_path)
    log(f"[{brain}] image {image_path} [{time.monotonic() - started:.0f}s]")
    rng = np.random.default_rng(0)
    (OUT / "examples").mkdir(parents=True, exist_ok=True)
    for name, chosen in (("no_gt_negatives", (category == "no_gt") & (y == 0)),
                         ("leaves_gt_negatives", (category == "leaves_gt") & (y == 0)),
                         ("leaves_gt_positives", (category == "leaves_gt") & (y == 1))):
        picked = np.flatnonzero(chosen & (family == "junction"))
        picked = np.sort(rng.choice(picked, size=min(EXAMPLES, len(picked)), replace=False))
        figure = Figure(figsize=(30, 5.3 * 4), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(4, 6)
        for axis in axes.ravel():
            axis.axis("off")
        read = 0
        for slot, i in enumerate(picked):
            panels = axes[slot // 2, 3 * (slot % 2):3 * (slot % 2) + 3]
            for axis in panels:
                axis.axis("on")
            junction = int(node[i])
            segment_id = int(str(graph.component_id_to_swc_id[int(component[junction])]).split(".")[0])
            title = (f"row {stored['rows'][i]}  GT {value['d_gt'][i]:.0f} um  site {value['site_cable'][i]:.0f} um  "
                     f"contact {value['contact_cable'][i]:.0f} um / run {value['contact_run'][i]:.0f}  "
                     f"seg merge {int(stored['seg_merge'][i])}")
            read += draw_example(panels, image, anisotropy, graph, xyz, component, gt_xyz, gt_neuron, gt_label,
                                 sites, junction, segment_id, title)
        figure.suptitle(f"{brain} {name}: image maximum projection ({PATCH_UM:g} x {PATCH_UM:g} um, {SLAB_UM:g} um deep), "
                        "fragments and GT in the same box (square-root grey scale); candidate = yellow + / red ring, site = star", fontsize=13)
        figure.savefig(OUT / "examples" / f"{brain}_{name}.png", dpi=80)
        log(f"[{brain}] {name}: {len(picked)} examples, {read} image reads [{time.monotonic() - started:.0f}s]")


def brain_evidence(brain):
    from scipy.spatial import cKDTree
    from candidate_rules import segment_of_component
    from proofreader_evolve.harness import label_availability
    from proofreader_evolve.harness.dataset import load_cached_graphs
    from proofreader_evolve.harness.native_pool import cache_path
    from proofreader_evolve.harness.raw_pool import ensure_raw_tables

    started = time.monotonic()
    bank = ensure_raw_tables(brain, cache_path(brain, 100), ROOT / "proofreader_evolve" / "feature_tables",
                             prepare=False, mcl=100)
    label_availability.apply({brain: bank})
    table = bank.tables["merge"]
    truth = np.asarray(table.truth, dtype=float)
    rows = np.flatnonzero(np.isfinite(truth))
    candidates = [table.candidates[i] for i in rows]
    family = np.array([c["candidate_family"] for c in candidates])
    node = np.array([c["node_id"] for c in candidates], dtype=np.int64)
    log(f"[{brain}] {len(rows)} labeled merge rows ({int(truth[rows].sum())} positives); loading the cache")
    graph, gt, payload = load_cached_graphs(str(cache_path(brain, 100)), expect_brain=brain, expect_mcl=100)
    xyz = np.asarray(graph.node_xyz, dtype=np.float64)
    component = np.asarray(graph.node_component_id, dtype=np.int64)
    to_segment = np.full(int(component.max()) + 1, -1, dtype=np.int64)
    for key, value in segment_of_component(graph).items():
        to_segment[int(key)] = int(value)
    segment = to_segment[component]
    gt_xyz = np.asarray(gt.node_xyz, dtype=np.float64)
    gt_neuron = np.asarray(gt.node_component_id, dtype=np.int64)
    gt_label = np.asarray(payload["gt_node_canonical_label"], dtype=np.int64)
    merge_segments = {int(s) for s in payload["gt_merge_labels"]}
    all_sites = list(payload.get("gt_merge_sites") or [])
    sites = {}
    for site in all_sites:
        sites.setdefault(int(site["segment_id"]), []).append(site)
    log(f"[{brain}] label policy {payload.get('gt_label_policy')}")
    gt_tree = cKDTree(gt_xyz)
    kept = np.flatnonzero(np.isin(component, np.unique(component[node])))
    distance, index = gt_tree.query(xyz[kept], k=1, distance_upper_bound=WIDE_UM, workers=8)
    gt_dist = np.full(len(xyz), np.inf)
    gt_index = np.full(len(xyz), -1, dtype=np.int64)
    gt_dist[kept] = distance
    gt_index[kept] = np.where(np.isfinite(distance), index, -1)
    site_cable, site_source = site_cable_distances(graph, xyz, segment, kept, all_sites)
    log(f"[{brain}] fragment graph {len(xyz)} nodes ({len(kept)} in the fragments of labeled candidates), "
        f"GT {len(gt_neuron)} nodes, {len(all_sites)} sites [{time.monotonic() - started:.0f}s]")

    names = ["site_cable", "d_gt", "contact_cable", "contact_run", "free_cable", "free_leaves",
             "free_leaf_max_gt", "free_contacts", "free_capped"]
    values = np.full((len(rows), len(names)), np.nan)
    column = {name: i for i, name in enumerate(names)}
    category, wide, source, seg_merge, n_arms = [], [], [], [], []
    for position, candidate in enumerate(candidates):
        junction, segment_id = int(candidate["node_id"]), int(candidate["segment_id"])
        firsts = [int(first) for first in graph.adj[junction]]
        walked = [arm_nodes(graph, xyz, junction, first) for first in firsts]
        arms = [read_arm(nodes, long, gt_dist, gt_index, gt_neuron, GT_UM) for nodes, long in walked]
        category.append(categorize(arms))
        wide.append(categorize([read_arm(nodes, long, gt_dist, gt_index, gt_neuron, WIDE_UM) for nodes, long in walked]))
        n_arms.append(len(arms))
        seg_merge.append(segment_id in merge_segments)
        source.append(site_source.get(junction, ""))
        row = values[position]
        row[column["site_cable"]] = site_cable.get(junction, np.inf)
        row[column["d_gt"]] = gt_tree.query(xyz[junction])[0]
        reach, contact = nearest_contact(graph, xyz, gt_dist, junction)
        row[column["contact_cable"]] = reach
        if contact >= 0:
            row[column["contact_run"]] = same_label_run(gt, gt_label, int(gt_index[contact]), segment_id)
        free_long = [first for first, (_, free, long) in zip(firsts, arms) if free and long]
        if category[-1] == "leaves_gt" and free_long:
            cable, leaves, contacts, capped = free_region(graph, xyz, gt_dist, junction, free_long)
            row[column["free_cable"]], row[column["free_leaves"]] = cable, len(leaves)
            row[column["free_contacts"]], row[column["free_capped"]] = contacts, capped
            row[column["free_leaf_max_gt"]] = float(gt_tree.query(xyz[leaves])[0].max()) if leaves else 0.0
    category, wide = np.array(category), np.array(wide)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"evidence_{brain}.npz", rows=rows, truth=truth[rows].astype(np.int8), family=family,
                        node=node, category=category, category_wide=wide, site_source=np.array(source),
                        seg_merge=np.array(seg_merge), n_arms=np.array(n_arms), values=values, names=np.array(names))
    y = truth[rows].astype(int)
    junction = family == "junction"
    counts = Counter(zip(y[junction].tolist(), category[junction].tolist()))
    log(f"[{brain}] wrote evidence; junction positives {[(c, counts[(1, c)]) for c in CATEGORIES]}; "
        f"negatives {[(c, counts[(0, c)]) for c in CATEGORIES]} [{time.monotonic() - started:.0f}s]")


# --------------------------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------------------------

def leaves_reason(site_cable, contact_run, leaves, leaf_max, contacts, capped, passthru):
    """Why a ``leaves_gt`` candidate has, or lacks, a recorded site within 20 um (first that applies)."""
    if site_cable <= 20:
        return "site within 20 um"
    if site_cable <= 60:
        return "site 20-60 um away"
    if contact_run < passthru:
        return "contact is a pass-through"
    if capped:
        return "free region too large to read"
    if leaves == 0 or leaf_max <= WALK_START_UM:
        return "no leaf > 50 um from GT"
    if contacts > 0:
        return "free region meets GT elsewhere"
    return "walk should record a site"


REASONS = ["site within 20 um", "site 20-60 um away", "contact is a pass-through", "no leaf > 50 um from GT",
           "free region meets GT elsewhere", "free region too large to read", "walk should record a site"]


def table(header, body):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in row) + " |"
                                                                               for row in body]


def binned(values, edges, labels):
    index = np.searchsorted(edges, values, side="left")
    return [int((index == i).sum()) for i in range(len(labels))]


def summarize(args):
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score

    experiment, state = accepted_merge_scores(Path(args.run))
    data, counts = {}, []
    for brain in args.brains:
        stored = np.load(OUT / f"evidence_{brain}.npz", allow_pickle=False)
        item = {key: stored[key] for key in ("truth", "family", "category", "category_wide", "site_source", "seg_merge")}
        item.update({name: stored["values"][:, i] for i, name in enumerate(stored["names"].tolist())})
        item["y"] = item.pop("truth").astype(int)
        junction = item["family"] == "junction"
        scores = state[f"{brain}/merge:scores"][stored["rows"]]
        item["rank"] = np.full(len(scores), np.nan)  # fold models differ in scale: rank within brain and family
        item["rank"][junction] = rankdata(scores[junction]) / int(junction.sum())
        item["brain"] = np.full(len(scores), brain)
        data[brain] = item
    pooled = {key: np.concatenate([data[b][key] for b in args.brains]) for key in data[args.brains[0]]}
    junction = pooled["family"] == "junction"
    y, category, rank = pooled["y"], pooled["category"], pooled["rank"]

    lines = ["# Merge labels of the candidates against the GT tracing around them", "",
             f"Script `notebooks/merge_junction_study/merge_junction_label_localization.py`; walk {WALK_UM:g} um, GT within {GT_UM:g} um, "
             f"arm nodes beyond {NEAR_UM:g} um, at least {MIN_NODES} supported nodes. Scores: holdout scores of "
             f"`{experiment}` in `{Path(args.run).name}`.", ""]

    def category_table(title, chosen_family, key="category"):
        body = []
        for brain in [*args.brains, "all"]:
            for label in (1, 0):
                chosen = chosen_family & (y == label) & ((pooled["brain"] == brain) | (brain == "all"))
                cells = [int((chosen & (pooled[key] == c)).sum()) for c in CATEGORIES]
                if brain == "all":
                    cells = [f"{c} ({100 * c / max(int(chosen.sum()), 1):.0f}%)" for c in cells]
                elif key == "category":
                    counts.append({"brain": brain, "family": title, "label": label, "rows": int(chosen.sum()),
                                   **dict(zip(CATEGORIES, cells))})
                body.append([brain, label, int(chosen.sum()), *cells])
        return [f"## {title}", "", *table(["brain", "label", "rows", *CATEGORIES], body), ""]

    lines += category_table("Label x category, junction family", junction)
    lines += category_table(f"Label x category, junction family, GT within {WIDE_UM:g} um", junction, "category_wide")
    lines += category_table("Label x category, tip_to_shaft family", ~junction)

    body = []
    for name in sorted(set(pooled["site_source"][junction & (y == 1)].tolist())):
        chosen = junction & (y == 1) & (pooled["site_source"] == name)
        body.append([name or "(none within reach)", int(chosen.sum()),
                     *[int((chosen & (category == c)).sum()) for c in CATEGORIES]])
    lines += ["## Junction positives by the source of the nearest recorded site", "",
              *table(["site source", "rows", *CATEGORIES], body), ""]

    body = []
    for label in (1, 0):
        cells = []
        for c in CATEGORIES:
            chosen = junction & (y == label) & (category == c)
            cells.append(f"{rank[chosen].mean():.2f} ({int(chosen.sum())})" if chosen.any() else "")
        body.append([label, *cells])
    lines += ["## Mean within-brain rank of the accepted scorer, junction family (1 = top), rows in brackets", "",
              *table(["label", *CATEGORIES], body), ""]

    def auroc_row(name, chosen):
        labels, ranks, kind = y[chosen], rank[chosen], category[chosen]
        agree = ((labels == 1) & np.isin(kind, MERGE_LIKE)) | ((labels == 0) & np.isin(kind, BRANCH_LIKE))
        return [name, f"{int(labels.sum())} / {int((labels == 0).sum())}", f"{roc_auc_score(labels, ranks):.3f}",
                f"{int(labels[agree].sum())} / {int((labels[agree] == 0).sum())}",
                f"{roc_auc_score(labels[agree], ranks[agree]):.3f}" if 0 < labels[agree].sum() < agree.sum() else ""]

    body = [auroc_row(brain, junction & (pooled["brain"] == brain)) for brain in args.brains]
    body.append(auroc_row("all (within-brain ranks)", junction))
    lines += ["## Junction AUROC of the accepted scorer", "",
              *table(["brain", "all rows (pos / neg)", "AUROC", "label and category agree (pos / neg)", "AUROC"], body), ""]

    # ---- leaves_gt: why is there, or is there not, a site within 20 um?
    leaves = junction & (category == "leaves_gt")
    reasons = {}
    for passthru in args.passthru:
        reasons[passthru] = np.array([
            leaves_reason(*item, passthru) if use else ""
            for use, item in zip(leaves, zip(pooled["site_cable"], np.nan_to_num(pooled["contact_run"]),
                                             pooled["free_leaves"], pooled["free_leaf_max_gt"],
                                             pooled["free_contacts"], pooled["free_capped"] == 1))])
    for passthru, reason in reasons.items():
        body = []
        for name in REASONS:
            row = [name]
            for label in (1, 0):
                chosen = leaves & (y == label) & (reason == name)
                row += [int(chosen.sum()), f"{rank[chosen].mean():.2f}" if chosen.any() else ""]
            row += [int((leaves & (y == 0) & (reason == name) & (pooled["brain"] == b)).sum()) for b in args.brains]
            body.append(row)
        lines += [f"## `leaves_gt` junctions: site rule read with a pass-through threshold of {passthru} GT nodes", "",
                  *table(["reason (first that applies)", "positives", "mean rank", "negatives", "mean rank",
                          *[f"neg {b}" for b in args.brains]], body), ""]
    negatives = leaves & (y == 0)
    edges, labels = [150, 500, 2000], ["<= 150", "150-500", "500-2000", "> 2000"]
    body = [[f"label {label}", *binned(pooled["free_cable"][leaves & (y == label)], edges, labels),
             f"{np.nanmedian(pooled['free_cable'][leaves & (y == label)]):.0f}"] for label in (1, 0)]
    lines += ["Cable of the free region behind the long free arms (um):", "",
              *table(["`leaves_gt` junctions", *labels, "median"], body), "",
              f"`leaves_gt` negatives on a segment in `gt_merge_labels`: {int((negatives & pooled['seg_merge']).sum())} "
              f"of {int(negatives.sum())}.", ""]

    # ---- no_gt negatives: where do they sit?
    empty = junction & (category == "no_gt") & (y == 0)
    d_edges, d_labels = [6, 12, 50], ["<= 6", "6-12", "12-50", "50-150"]
    c_edges, c_labels = [20, 60, 150, REACH_UM], ["<= 20", "20-60", "60-150", "150-300", "> 300"]
    s_edges, s_labels = [60, 150, REACH_UM], ["<= 60", "60-150", "150-300", "> 300"]
    body = []
    for brain in [*args.brains, "all"]:
        chosen = empty & ((pooled["brain"] == brain) | (brain == "all"))
        if not chosen.any():
            continue
        wide = pooled["category_wide"][chosen]
        body.append([brain, int(chosen.sum()),
                     " / ".join(map(str, binned(pooled["d_gt"][chosen], d_edges, d_labels))),
                     " / ".join(map(str, binned(pooled["contact_cable"][chosen], c_edges, c_labels))),
                     " / ".join(map(str, binned(pooled["site_cable"][chosen], s_edges, s_labels))),
                     int(pooled["seg_merge"][chosen].sum()),
                     int((wide != "no_gt").sum()), int(np.isin(wide, BRANCH_LIKE).sum()),
                     f"{rank[chosen].mean():.2f}"])
    lines += ["## `no_gt` junction negatives: where they sit", "",
              *table(["brain", "rows", "GT node (Euclidean um) " + " / ".join(d_labels),
                      "nearest contact (cable um) " + " / ".join(c_labels),
                      "nearest site (cable um) " + " / ".join(s_labels), "segment in gt_merge_labels",
                      f"not `no_gt` with GT within {WIDE_UM:g} um", "of them branch / spur", "mean rank"], body), ""]
    far = empty & (pooled["contact_cable"] > 20)
    beyond = far & np.isfinite(pooled["site_cable"]) & (pooled["site_cable"] <= pooled["contact_cable"] + 20)
    lines += [f"Of the {int(empty.sum())} rows, {int(far.sum())} are more than 20 um of cable from any contact with a "
              f"tracing; {int(beyond.sum())} of those have a recorded site no farther than the contact (+20 um), "
              f"{int((far & ~beyond & pooled['seg_merge']).sum())} others are on a segment in `gt_merge_labels`, and "
              f"{int((far & ~beyond & ~pooled['seg_merge']).sum())} on a segment that is not.", ""]

    # ---- what the junction AUROC would read under other definitions of a judgeable negative
    on_tracing = (pooled["contact_cable"] <= 20) & (np.nan_to_num(pooled["contact_run"]) >= min(args.passthru))
    policies = {
        "current labels": np.ones(len(y), dtype=bool),
        "without `no_gt` negatives": (y == 1) | (category != "no_gt"),
        "without `no_gt` and `leaves_gt` negatives": (y == 1) | ~np.isin(category, ("no_gt", "leaves_gt")),
        f"negatives only within 20 um of cable of a contact whose GT carries the segment (run >= {min(args.passthru)})":
            (y == 1) | on_tracing,
    }
    body = []
    for name, keep in policies.items():
        row = [name, int((junction & keep & (y == 0)).sum()), int((~junction & keep & (y == 0)).sum())]
        for brain in [*args.brains, "all"]:
            chosen = junction & keep & ((pooled["brain"] == brain) | (brain == "all"))
            row.append(f"{roc_auc_score(y[chosen], rank[chosen]):.3f}")
        body.append(row)
    lines += ["## Junction AUROC of the accepted scorer under other sets of judgeable negatives (positives unchanged)", "",
              *table(["negatives kept", "junction negatives", "tip_to_shaft negatives", *args.brains,
                      "all (within-brain ranks)"], body), "",
              "The scorer was fitted with the current labels; only the rows it is measured on change here.", ""]

    with (OUT / "counts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(counts[0]))
        writer.writeheader()
        writer.writerows(counts)
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("evidence", help="GT tracing around every labeled merge candidate")
    p.add_argument("--brains", nargs="+", default=TRAIN_BRAINS, help="TRAIN brains only")
    p = sub.add_parser("examples", help="image, fragments and GT around a few candidates per group (cloud access)")
    p.add_argument("--brains", nargs="+", default=TRAIN_BRAINS, help="TRAIN brains only")
    p = sub.add_parser("summarize", help="tables of label x category and the scorer's ranks")
    p.add_argument("--brains", nargs="+", default=TRAIN_BRAINS)
    p.add_argument("--run", required=True, help="finished run whose accepted merge scorer is read")
    p.add_argument("--passthru", nargs="+", type=int, default=[12],
                   help="pass-through thresholds (GT nodes) to read the site rule with; see the label policy in the log")
    args = parser.parse_args()
    unknown = set(args.brains) - set(TRAIN_BRAINS)
    if unknown:
        parser.error(f"Not TRAIN brains: {sorted(unknown)}; this analysis must not open validation or reserved brains")
    if args.stage == "evidence":
        for brain in args.brains:
            brain_evidence(brain)
    elif args.stage == "examples":
        for brain in args.brains:
            brain_examples(brain)
    else:
        summarize(args)


if __name__ == "__main__":
    main()

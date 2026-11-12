#!/usr/bin/env python3
"""Are the merge sites that the ``length_v2`` relabel added the ones the metric reports?

The relabel of 2026-11-11 changed the cache's merge sites (1,181 to 1,309 on twelve brains). This
script compares, per TRAIN brain, the sites before (``length_v1``, kept in
``merge_site_walk_replay/sites_<brain>.npz`` as ``cache``), the sites now in the cache and the
canonical sites of segmentation-skeleton-metrics (``metrics_out/<brain>/*/merge_sites.csv``):

* a new site with an old site within ``SAME_UM`` is the same event (kept, possibly moved to a
  branching node); a new site without one is **added**; an old site without a new site within
  ``SAME_UM`` is **dropped**;
* for each group, the share with a canonical site within 20 um and within 50 um (Euclidean).

``--examples`` draws every added and every dropped site with the image, the fragments and the
GT (the three panels of ``merge_junction_label_localization.draw_example``, centred on the
fragment node the site snaps to): gold stars are the sites now in the cache, magenta diamonds the
canonical sites, white crosses the sites before the relabel. Needs cloud access.

Run on a compute node in the ``panda`` environment from the repository root::

    python -u notebooks/merge_junction_study/merge_added_sites_check.py --brains 747807 802449
    python -u notebooks/merge_junction_study/merge_added_sites_check.py --summarize
    AWS_EC2_METADATA_DISABLED=true python -u notebooks/merge_junction_study/merge_added_sites_check.py --examples --brains 802449

Outputs under ``notebooks/merge_junction_study/merge_added_sites_check/`` (gitignored):
``sites_<brain>.json``, ``summary.md``, ``examples/<brain>_{added,dropped}_<n>.png``. TRAIN brains only;
read-only.

Limits. A canonical site is the reference, not independent truth: both lists rest on the same
tracing. The metric ran on its own graphs, so a correct cache site can lie more than 20 um from
its canonical counterpart, and one canonical site can stand for several nearby cache sites.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT), str(ROOT / "notebooks")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from merge_junction_fragment_context import TRAIN_BRAINS, log  # noqa: E402

OUT = Path(__file__).resolve().parent / "merge_added_sites_check"
BEFORE = Path(__file__).resolve().parent / "merge_site_walk_replay"
SAME_UM = 30.0   # the walk's own de-duplication distance


def near(points, others, radius):
    from scipy.spatial import cKDTree
    if not len(points) or not len(others):
        return np.zeros(len(points), dtype=bool)
    return cKDTree(others).query(points)[0] <= radius


def brain_check(brain):
    from merge_canonical_gap import load_canonical
    from proofreader_evolve.harness.native_pool import cache_path

    old = np.load(BEFORE / f"sites_{brain}.npz", allow_pickle=False)["cache"]
    with open(cache_path(brain, 100), "rb") as handle:
        payload = pickle.load(handle)
    record = payload["gt_label_policy"]
    if record.get("walk") != "per_neuron":
        raise ValueError(f"{brain}: the cache does not carry length_v2 labels ({record})")
    sites = list(payload["gt_merge_sites"])
    new = np.array([s["xyz"] for s in sites], dtype=np.float64).reshape(-1, 3)
    source = np.array([str(s.get("source", "geometric_walk")) for s in sites])
    canonical = np.asarray(load_canonical(ROOT / "metrics_out", brain, payload["anisotropy"])["xyz"], dtype=np.float64)
    kept, dropped = near(new, old, SAME_UM), ~near(old, new, SAME_UM)
    result = {"brain": brain, "old_sites": len(old), "new_sites": len(new), "canonical_sites": len(canonical)}
    for name, points in (("kept", new[kept]), ("added", new[~kept]), ("dropped", old[dropped]), ("all_new", new),
                         ("all_old", old)):
        result[name] = {"sites": len(points), "canonical_within_20": int(near(points, canonical, 20.0).sum()),
                        "canonical_within_50": int(near(points, canonical, 50.0).sum())}
    result["added"]["by_source"] = {s: int(((~kept) & (source == s)).sum()) for s in np.unique(source)}
    result["canonical_reached_within_50"] = {"before": int(near(canonical, old, 50.0).sum()),
                                             "after": int(near(canonical, new, 50.0).sum())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"sites_{brain}.json").write_text(json.dumps(result, indent=1))
    log(f"[{brain}] {json.dumps({k: result[k] for k in ('old_sites', 'new_sites', 'kept', 'added', 'dropped')})}")


def brain_examples(brain):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from scipy.spatial import cKDTree
    from agentic_neuron_proofreader.utils.img_util import TensorStoreImage
    from candidate_rules import SITE_SNAP_MAX_UM, segment_of_component
    from merge_canonical_gap import load_canonical
    from merge_canonical_site_labels import snap
    from merge_junction_label_localization import PATCH_UM, SLAB_UM, draw_example
    from proofreader_evolve.harness.dataset import load_cached_graphs
    from proofreader_evolve.harness.native_pool import cache_path
    from scripts.dataset_config import get_cache_img_path

    old = np.load(BEFORE / f"sites_{brain}.npz", allow_pickle=False)["cache"]
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
    anisotropy = np.asarray(payload["anisotropy"], dtype=np.float64)
    sites = list(payload["gt_merge_sites"])
    new = np.array([s["xyz"] for s in sites], dtype=np.float64).reshape(-1, 3)
    canonical = load_canonical(ROOT / "metrics_out", brain, payload["anisotropy"])
    canonical_xyz = np.asarray(canonical["xyz"], dtype=np.float64)
    by_segment = {}
    for site in sites:
        by_segment.setdefault(int(site["segment_id"]), []).append(site)
    wanted = set(by_segment) | set(canonical["segment"].tolist())
    kept_nodes = np.flatnonzero(np.isin(segment, list(wanted)))
    tree = cKDTree(xyz[kept_nodes])
    canonical_tree = cKDTree(canonical_xyz)
    marks = [(point, {"s": 150, "marker": "D", "facecolor": "none", "edgecolor": "magenta", "linewidths": 1.8})
             for point in canonical_xyz]
    marks += [(point, {"s": 110, "marker": "x", "color": "white", "linewidths": 2.0}) for point in old]
    cases = []
    for index in np.flatnonzero(~near(new, old, SAME_UM)):
        site = sites[index]
        node = snap(tree, kept_nodes, xyz, segment, new[index], int(site["segment_id"]), SITE_SNAP_MAX_UM)
        cases.append(("added", node, int(site["segment_id"]), new[index], str(site.get("source", "geometric_walk"))))
    for point in old[~near(old, new, SAME_UM)]:
        distance, nearest = tree.query(point)
        node = int(kept_nodes[nearest]) if distance <= SITE_SNAP_MAX_UM else -1
        cases.append(("dropped", node, int(segment[node]) if node >= 0 else -1, point, "site before the relabel"))
    image = TensorStoreImage(get_cache_img_path(cache_path(brain, 100), payload))
    (OUT / "examples").mkdir(parents=True, exist_ok=True)
    for group in ("added", "dropped"):
        chosen = [case for case in cases if case[0] == group and case[1] >= 0]
        for start in range(0, len(chosen), 8):
            figure = Figure(figsize=(30, 5.3 * 4), layout="constrained")
            FigureCanvasAgg(figure)
            axes = figure.subplots(4, 6)
            for axis in axes.ravel():
                axis.axis("off")
            for slot, (_, node, segment_id, point, source) in enumerate(chosen[start:start + 8]):
                panels = axes[slot // 2, 3 * (slot % 2):3 * (slot % 2) + 3]
                for axis in panels:
                    axis.axis("on")
                distance = float(canonical_tree.query(point)[0])
                title = (f"{group} #{start + slot + 1}  segment {segment_id}  {source}  "
                         f"nearest canonical site {distance:.0f} um")
                draw_example(panels, image, anisotropy, graph, xyz, component, gt_xyz, gt_neuron, gt_label, by_segment,
                             node, segment_id, title, marks=marks)
            figure.suptitle(f"{brain} merge sites {group} by the length_v2 relabel: image maximum projection "
                            f"({PATCH_UM:g} x {PATCH_UM:g} um, {SLAB_UM:g} um deep); red ring = the site's fragment node, "
                            "gold star = site now in the cache, magenta diamond = canonical site, white x = site before",
                            fontsize=13)
            figure.savefig(OUT / "examples" / f"{brain}_{group}_{start // 8 + 1}.png", dpi=80)
    log(f"[{brain}] drew {sum(c[0] == 'added' for c in cases)} added and {sum(c[0] == 'dropped' for c in cases)} dropped sites")


def summarize(brains):
    results = [json.loads((OUT / f"sites_{brain}.json").read_text()) for brain in brains]
    groups = ("kept", "added", "dropped")
    header = ["brain", "sites before", "now"]
    for name in groups:
        header += [name, "canonical within 20 um", "within 50 um"]
    header += ["canonical sites reached within 50 um: before", "after"]
    lines = ["# Merge sites added by the `length_v2` relabel against the canonical sites", "",
             f"Script `notebooks/merge_junction_study/merge_added_sites_check.py`; same event = an old and a new site "
             f"within {SAME_UM:g} um; distances Euclidean.", "",
             "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    total = np.zeros(len(header) - 1, dtype=int)
    for result in results:
        row = [result["old_sites"], result["new_sites"]]
        for name in groups:
            row += [result[name]["sites"], result[name]["canonical_within_20"], result[name]["canonical_within_50"]]
        row += [result["canonical_reached_within_50"]["before"], result["canonical_reached_within_50"]["after"]]
        total += np.array(row)
        lines.append(f"| {result['brain']} | " + " | ".join(str(v) for v in row) + " |")
    lines.append("| all | " + " | ".join(str(v) for v in total.tolist()) + " |")
    added = total[5:8]
    kept = total[2:5]
    lines += ["", f"Added sites with a canonical site within 50 um: {added[2]} of {added[0]} "
                  f"({100 * added[2] / max(added[0], 1):.0f}%); within 20 um: {added[1]} "
                  f"({100 * added[1] / max(added[0], 1):.0f}%). Kept sites: {kept[2]} of {kept[0]} "
                  f"({100 * kept[2] / max(kept[0], 1):.0f}%) within 50 um.", ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brains", nargs="+", default=TRAIN_BRAINS, help="TRAIN brains only")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--examples", action="store_true", help="draw the added and dropped sites (cloud access)")
    args = parser.parse_args()
    unknown = set(args.brains) - set(TRAIN_BRAINS)
    if unknown:
        parser.error(f"Not TRAIN brains: {sorted(unknown)}")
    if args.summarize:
        summarize(args.brains)
    elif args.examples:
        for brain in args.brains:
            brain_examples(brain)
    else:
        for brain in args.brains:
            brain_check(brain)


if __name__ == "__main__":
    main()

"""Pictures of joins that ground truth names, apart by the 50 um support condition.

Two runs of ``join_split_candidates.py --joins oracle --truth labels`` after the same merge cuts
differ only in ``--min-support-um``: ``--every`` joins every pair that ground truth names,
``--kept`` only the pairs in which both pieces label at least that length of the neuron. This
script rebuilds the cut graph, measures for every join of ``--every`` how much of the naming
neuron each of the two pieces labels, and draws a few joins of each group in the layout of
``proofreader_evolve/plotting/plot_error_contexts.py`` (its panels are used as they are):

* ``join_context_kept_<brain>_level0.png``: joins that the support condition keeps (made in
  both runs);
* ``join_context_dropped_<brain>_level0.png``: joins made only when every named pair is joined.

One column per join, four rows: the image patch (XY maximum projection) with the two pieces'
skeletons; the wider neighbourhood of the two pieces with the image footprint; the ground-truth
tracings in the same voxel box, one colour per neuron; the dense segmentation in the same box,
the two pieces' segments in the skeleton colours. Within a group the joins are ordered by the
support of the weaker piece and taken at evenly spaced quantiles, so the columns span the group
and are not picked by eye. ``examples.csv`` holds the numbers of the columns, ``joins.csv``
those of every join, ``summary.json`` the groups' sizes and medians.

Nothing is edited or evaluated here; the cache and the pools are read only. Run on a compute
node in panda, from the repository root (the image is read from S3, the segmentation from the
private GCS store):

    AWS_EC2_METADATA_DISABLED=true python -u evaluation/plot_oracle_join_examples.py
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cut_merge_sites import apply_cuts, oracle_sites
from export_cache_fragments import REPO, load_cache, read_segments
from join_split_candidates import ground_truth_pieces, node_state
from plot_merge_policy_cuts import RESULTS

sys.path.insert(0, str(REPO))
from proofreader_evolve.harness import candidate_pools  # noqa: E402
from proofreader_evolve.harness.image_context import ImageSource  # noqa: E402
from proofreader_evolve.plotting import plot_error_contexts as showcase  # noqa: E402
from scripts.dataset_config import get_cache_img_path  # noqa: E402

# The showcase's two skeleton colours: the piece that labels more of the neuron, the piece that labels less.
STRONGER, WEAKER = plt.cm.tab10(0), plt.cm.tab10(1)
HEADINGS = {"kept": "made with and without the 50 um condition",
            "dropped": "made only without the 50 um condition"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brain", default="794495")
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--every", type=Path, default=RESULTS / "794495_split_oracle_20261122/remove12_labels",
                        help="the run that joins every pair ground truth names")
    parser.add_argument("--kept", type=Path, default=RESULTS / "794495_split_oracle_20261122/remove12_labels_s50",
                        help="the run with --min-support-um")
    parser.add_argument("--segments", type=Path,
                        default=RESULTS / "794495_baseline_alignment_20261121/raw_swcs/segments_under_gt.txt")
    parser.add_argument("--per-group", type=int, default=4)
    parser.add_argument("--image-um", type=float, default=30.0, help="radius of the level-0 image patch")
    parser.add_argument("--neighbourhood-um", type=float, default=150.0, help="radius of the fragment neighbourhood")
    parser.add_argument("--no-segmentation", action="store_true", help="skip the segmentation row (no GCS read)")
    parser.add_argument("--out", type=Path, default=RESULTS / "794495_join_examples_20261123")
    return parser.parse_args()


def records(options):
    """The two runs' records, checked to be the same cuts and the same truth apart from the support."""
    every = json.loads((options.every / "fragments.json").read_text())
    kept = json.loads((options.kept / "fragments.json").read_text())
    for name, record in (("--every", every), ("--kept", kept)):
        if record["joins"]["selection"] != "oracle" or record["truth"]["reading"] != "labels" or record["across_cuts"]:
            sys.exit(f"{name}: not the joins that ground truth names (labels)")
        if record["brain"] != options.brain or record["sites"]["selection"] != "oracle":
            sys.exit(f"{name}: not oracle cuts on brain {options.brain}")
    if (every["sites"], every["merge_edit"]) != (kept["sites"], kept["merge_edit"]):
        sys.exit("the two runs do not follow the same cuts")
    if every["truth"]["min_support_um"] != 0 or kept["truth"]["min_support_um"] <= 0:
        sys.exit("--every must be the run without --min-support-um and --kept the run with it")
    return every, kept


def supports(rules, payload, canonical, gt_piece):
    """Per ground-truth node its neuron, ``{piece: {neuron: um of that neuron the piece labels}}``
    and ``{pair of pieces: neurons that name it}``: the split truth on pieces and the support, as
    ``join_split_candidates.piece_truth`` builds them for ``labels``."""
    gt = payload["gt_graph"]
    edges = np.array(list(gt.edges()), dtype=np.int64)
    u, v = gt_piece[edges[:, 0]], gt_piece[edges[:, 1]]
    error = np.asarray(payload["gt_edge_error"]).copy()
    error[(u != v) & (u != 0) & (v != 0) & (canonical[edges[:, 0]] == canonical[edges[:, 1]])] = 1
    naming = {}
    for neuron_id, pair in rules.split_truth({"gt_graph": gt, "gt_node_canonical_label": gt_piece, "gt_edge_error": error}):
        naming.setdefault(tuple(sorted(pair)), []).append(neuron_id)
    membership = rules.gt_neuron_membership(gt)
    neuron = np.fromiter((membership[node] for node in range(len(gt_piece))), dtype=np.int64, count=len(gt_piece))
    gt_xyz = np.asarray(gt.node_xyz, dtype=float)
    spacing = float(np.median(np.linalg.norm(gt_xyz[edges[:, 0]] - gt_xyz[edges[:, 1]], axis=1)))
    on_piece = gt_piece > 0
    keys, counts = np.unique(np.stack([gt_piece[on_piece], neuron[on_piece]], axis=1), axis=0, return_counts=True)
    support = {}
    for (piece, neuron_id), count in zip(keys.tolist(), counts.tolist()):
        support.setdefault(piece, {})[neuron_id] = count * spacing
    return neuron, support, naming


class Pieces:
    """Nodes and edges of the pieces of the edited graph."""
    def __init__(self, graph, piece, xyz):
        self.graph, self.piece, self.xyz = graph, piece, xyz
        self.order = np.argsort(piece, kind="stable")
        self.sorted = piece[self.order]

    def nodes(self, piece):
        return self.order[np.searchsorted(self.sorted, piece, "left"):np.searchsorted(self.sorted, piece, "right")]

    def cable_um(self, piece):
        edges = np.array(list(nx.Graph.subgraph(self.graph, self.nodes(piece).tolist()).edges()), dtype=np.int64).reshape(-1, 2)
        return float(np.linalg.norm(self.xyz[edges[:, 0]] - self.xyz[edges[:, 1]], axis=1).sum())

    def fragment(self, nodes, anchors, origin_um, zyx=None):
        """The showcase's fragment record for some nodes: positions from ``origin_um``, the piece
        as the colour key, local edges and anchors, and patch voxel positions when given."""
        local = {int(node): index for index, node in enumerate(nodes.tolist())}
        record = {"xyz_um": self.xyz[nodes] - origin_um, "segment": self.piece[nodes],
                  "edges": [(local[a], local[b]) for a, b in nx.Graph.subgraph(self.graph, nodes.tolist()).edges()],
                  "outside_radius": np.zeros(len(nodes), dtype=bool),
                  "anchor_nodes": [local[int(anchor)] for anchor in anchors if int(anchor) in local]}
        if zyx is not None:
            record["nodes_zyx"] = zyx
        return record


def measure(joins, piece, support, naming, pieces):
    """One row per join: of the neurons that name the pair, the one on which the weaker piece
    labels most (the support condition asks whether that reaches the limit), both pieces'
    supports on it and their cable."""
    rows, cable = [], {}
    for join in joins.itertuples():
        a, b = int(piece[join.node_id_a]), int(piece[join.node_id_b])
        named = naming.get(tuple(sorted((a, b))))
        if not named:
            sys.exit(f"join {join.node_id_a}-{join.node_id_b}: the split truth on pieces does not name this pair")
        on = {item: support.get(item, {}) for item in (a, b)}
        neuron = max(named, key=lambda n: (min(on[a].get(n, 0.0), on[b].get(n, 0.0)), -n))
        for item in (a, b):
            if item not in cable:
                cable[item] = pieces.cable_um(item)
        strong, weak = (a, b) if on[a].get(neuron, 0.0) >= on[b].get(neuron, 0.0) else (b, a)
        rows.append({"node_id_a": join.node_id_a, "node_id_b": join.node_id_b, "gap_um": join.gap_um, "truth": join.truth,
                     "x_um": join.x_um, "y_um": join.y_um, "z_um": join.z_um, "neuron": neuron,
                     "neurons_naming_the_pair": len(named), "stronger_piece": strong, "weaker_piece": weak,
                     "stronger_support_um": on[strong].get(neuron, 0.0), "weaker_support_um": on[weak].get(neuron, 0.0),
                     "stronger_cable_um": cable[strong], "weaker_cable_um": cable[weak],
                     "weaker_neurons_labeled": len(on[weak])})
    table = pd.DataFrame(rows)
    table["weaker_cable_elsewhere_um"] = (table["weaker_cable_um"] - table["weaker_support_um"]).clip(lower=0)
    return table


def draw_column(axes, example, options, context):
    """The showcase's four panels for one join."""
    pieces, gt, segment, spacing0, source, segmentation = context
    xyz, geometry = pieces.xyz, source.level(0)
    anchors = np.array([example.node_id_a, example.node_id_b], dtype=np.int64)
    middle = xyz[anchors].mean(axis=0)
    colours = {int(example.stronger_piece): STRONGER, int(example.weaker_piece): WEAKER}
    both = np.concatenate([pieces.nodes(example.stronger_piece), pieces.nodes(example.weaker_piece)])
    origin, shape, anchors_zyx = geometry.crop(xyz[anchors], options.image_um)

    header = (f"{example.neuron_name}: pieces label {example.stronger_support_um:,.0f} and "
              f"{example.weaker_support_um:,.0f} um, gap {example.gap_um:.1f} um\n"
              f"level0 patch, radius {options.image_um:.0f} um, XY max projection")
    try:
        patch, _ = source.read(xyz[anchors], radius_um=options.image_um, level=0)
    except Exception as error:                                  # a stalled or missing read leaves the other rows
        print(f"join {example.node_id_a}-{example.node_id_b}: image not read: {type(error).__name__}: {error}", flush=True)
        patch = None
        axes[0].text(0.5, 0.5, "no image", ha="center", va="center")
        axes[0].set_title(header, fontsize=9)
        axes[0].axis("off")
    else:
        zyx = geometry.graph_to_voxel(xyz[both]) - origin
        within = np.all((zyx >= -1) & (zyx <= shape), axis=1)
        showcase.draw_image_panel(axes[0], patch, pieces.fragment(both[within], anchors, middle, zyx[within]), colours, header)

    near = both[np.all(np.abs(xyz[both] - middle) <= options.neighbourhood_um, axis=1)]
    showcase.draw_fragment_panel(
        axes[1], pieces.fragment(near, anchors, middle), colours, options.image_um, options.neighbourhood_um,
        f"{showcase.ROW_NAMES[1]} ({options.neighbourhood_um:.0f} um); dashed box = image patch\n"
        f"cable of the pieces: {example.stronger_cable_um:,.0f} and {example.weaker_cable_um:,.0f} um")
    showcase.draw_ground_truth_panel(axes[2], gt, origin, shape, spacing0, patch, anchors_zyx, showcase.ROW_NAMES[2])
    if segmentation is None:
        axes[3].text(0.5, 0.5, "segmentation not read", ha="center", va="center")
        axes[3].set_title(showcase.ROW_NAMES[3], fontsize=9)
        axes[3].axis("off")
    else:
        labels = showcase.read_segmentation(segmentation, origin, shape)
        own = [int(segment[pieces.nodes(example.stronger_piece)[0]]), int(segment[pieces.nodes(example.weaker_piece)[0]])]
        showcase.draw_segmentation_panel(axes[3], labels, None if patch is None else patch["image_zyx"], spacing0, own,
                                         anchors_zyx, showcase.ROW_NAMES[3])


def draw_group(group, examples, options, context, path):
    figure, axes = plt.subplots(4, len(examples), figsize=(4.6 * len(examples), 4.3 * 4), constrained_layout=True,
                                squeeze=False)
    for column, example in enumerate(examples):
        draw_column(axes[:, column], example, options, context)
    handles = [Line2D([0], [0], color="red", marker="x", linestyle="", markersize=8, label="the two ends of the join"),
               Line2D([0], [0], color=STRONGER, linewidth=2, label="piece that labels more of the neuron"),
               Line2D([0], [0], color=WEAKER, linewidth=2, label="piece that labels less of the neuron"),
               Line2D([0], [0], color="black", linestyle="--", label="image patch footprint"),
               Line2D([0], [0], color=showcase.GT_PALETTE(0), linewidth=2, label="ground truth: one colour per neuron"),
               Patch(facecolor=np.asarray(plt.cm.Pastel2(0))[:3] * 0.55, label="segmentation: other segments")]
    figure.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    figure.suptitle(f"Brain {options.brain}: joins that ground truth names, {HEADINGS[group]} - rows: "
                    f"{', '.join(showcase.ROW_NAMES)}.\nHuman review only; display scaling does not change model pixels.",
                    fontsize=11)
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)


def main():
    options = parse_args()
    every_record, kept_record = records(options)
    cache, payload = load_cache(options.brain, options.mcl)
    graph = payload["fragments_graph"]
    sites, _ = oracle_sites(options.brain, cache, graph, read_segments(options.segments))
    apply_cuts(graph, sites, every_record["merge_edit"]["rule"], every_record["merge_edit"]["radius_um"])
    _, segment, present, _, piece, _ = node_state(graph, sites)
    xyz = np.asarray(graph.node_xyz, dtype=float)
    canonical, gt_piece, _ = ground_truth_pieces(payload, xyz, segment, present, piece, "labels")
    gt = payload["gt_graph"]
    gt_neuron, support, naming = supports(candidate_pools.rules(), payload, canonical, gt_piece)
    pieces = Pieces(graph, piece, xyz)

    every = pd.read_csv(options.every / "fragments_joins.csv")
    kept = pd.read_csv(options.kept / "fragments_joins.csv")
    table = measure(every, piece, support, naming, pieces)
    in_kept = set(zip(kept["node_id_a"], kept["node_id_b"]))
    table["group"] = ["kept" if key in in_kept else "dropped" for key in zip(table["node_id_a"], table["node_id_b"])]
    limit = kept_record["truth"]["min_support_um"]
    crossed = table[(table["group"] == "dropped") != (table["weaker_support_um"] < limit)]
    print(f"{len(table)} joins when every named pair is joined: {int((table['group'] == 'kept').sum())} also made with "
          f"the {limit:g} um condition, {int((table['group'] == 'dropped').sum())} only without it; "
          f"{len(crossed)} whose weaker support is on the other side of the limit; "
          f"{len(in_kept - set(zip(every['node_id_a'], every['node_id_b'])))} joins of the run with the condition "
          f"not made without it", flush=True)

    swc_ids = getattr(gt, "component_id_to_swc_id", None)
    component = np.asarray(gt.node_component_id)
    first = {int(n): int(np.flatnonzero(gt_neuron == n)[0]) for n in table["neuron"].unique()}
    table["neuron_name"] = [str(swc_ids[int(component[first[int(n)]])]) if swc_ids is not None else f"#{int(n)}"
                            for n in table["neuron"]]
    options.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(options.out / "joins.csv", index=False)
    source = ImageSource(get_cache_img_path(cache, payload))
    summary = {"brain": options.brain, "every": str(options.every), "kept": str(options.kept), "min_support_um": limit,
               "selection": "within a group, ordered by the weaker piece's support and taken at the quantiles "
                            "(i + 0.5) / n, i = 0 .. n - 1",
               "image": source.root, "image_radius_um": options.image_um, "neighbourhood_radius_um": options.neighbourhood_um,
               "groups": {}}
    for group, rows in table.groupby("group"):
        summary["groups"][group] = {
            "joins": len(rows), "by_truth_kind": rows["truth"].value_counts().to_dict(),
            **{f"median_{column}": round(float(rows[column].median()), 1)
               for column in ("gap_um", "weaker_support_um", "weaker_cable_um", "weaker_cable_elsewhere_um",
                              "stronger_support_um", "stronger_cable_um")},
            "weaker_piece_labels_more_than_one_neuron": int((rows["weaker_neurons_labeled"] > 1).sum())}
    print(json.dumps(summary["groups"], indent=2), flush=True)

    segmentation = None
    if not options.no_segmentation:
        summary["segmentation"] = showcase.segmentation_path(options.brain)
        segmentation = showcase.open_segmentation(summary["segmentation"])
    context = (pieces, gt, segment, np.asarray(payload["anisotropy"], dtype=float)[::-1], source, segmentation)
    chosen = []
    for group in ("kept", "dropped"):
        rows = table[table["group"] == group].sort_values(["weaker_support_um", "node_id_a"]).reset_index(drop=True)
        count = min(options.per_group, len(rows))
        examples = [rows.iloc[int((index + 0.5) / count * len(rows))] for index in range(count)]
        path = options.out / f"join_context_{group}_{options.brain}_level0.png"
        draw_group(group, examples, options, context, path)
        chosen += [{"file": path.name, "column": column + 1, **example.to_dict()} for column, example in enumerate(examples)]
        print(f"{path.name}: weaker pieces label " + ", ".join(f"{e.weaker_support_um:.0f}" for e in examples) + " um", flush=True)
    pd.DataFrame(chosen).to_csv(options.out / "examples.csv", index=False)
    (options.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"written to {options.out}", flush=True)


if __name__ == "__main__":
    main()

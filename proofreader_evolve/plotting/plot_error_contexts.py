"""Show what the data look like around true split errors and true merge errors.

One figure per kind, one column per example, four rows:

1. the cached image patch (XY maximum-intensity projection) with the cached
   fragment skeleton drawn on top, coloured by segment;
2. the wider 50 um fragment neighbourhood in graph coordinates, with the
   image footprint marked;
3. the ground-truth tracings inside the same voxel box, one colour per GT
   neuron, over the dimmed image;
4. the dense UNet segmentation inside the same voxel box (segment id at the
   brightest voxel along z); the candidate's own segment(s) keep the skeleton
   colours, every other segment is muted.

Rows 1-2 come from the context cache. Rows 3-4 need the brain's labelled
``_add.pkl`` (ground-truth graph, > 20 GB RAM, compute node) and, for the
segmentation, GCS credentials for the private segmentation store. Examples are
chosen from evaluator-labelled errors inside the cached band, so this module
reads labels and is for human review only. It is never copied into a
generation directory and nothing here is visible to the reviser.

Usage (panda environment, repository root, compute node):

    python -m proofreader_evolve.plotting.plot_error_contexts --brain 802449 \
        --examples 3 --tier level0
    # -> figs/error_context_split_802449_level0.png,
    #    figs/error_context_merge_802449_level0.png
"""

import argparse
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from proofreader_evolve.cli import precompute_error_scores as pc
from proofreader_evolve.harness.context_cache import CHUNK_ROWS, ContextCache
from proofreader_evolve.harness.dataset import load_cached_graphs
from proofreader_evolve.harness.native_pool import cache_path, ensure_native_tables

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = PROJECT_ROOT / "proofreader_evolve" / "context_cache"
DEFAULT_FIGS = PROJECT_ROOT / "figs"
# configs/zihan_gcs_token.json is rejected by Google (invalid_grant) since 2026-10-06.
DEFAULT_CREDENTIALS = (PROJECT_ROOT / "configs" / "allen-nd-goog-f5d46dbfa2cd.json",)
KINDS = ("split", "merge")
ROW_NAMES = ("image + fragment skeleton", "fragment neighbourhood", "ground truth", "segmentation")
GT_PALETTE = plt.cm.Dark2


# ----------------------------------------------------------------------------- data

def load_tables(brain, mcl=100, tables_dir=None, merge_dir=None, split_dir=None):
    """Frozen native tables (with evaluator labels) for one brain."""
    selected = pc.resolve_detector_runs(merge_dir, split_dir)
    return ensure_native_tables(str(brain), cache_path(str(brain), mcl), selected,
                                tables_dir or pc.DEFAULT_OUT, prepare=False, mcl=mcl)


def load_graphs(brain, mcl=100):
    """Fragment graph, ground-truth graph and xyz anisotropy from the labelled cache."""
    fragments, gt, payload = load_cached_graphs(str(cache_path(str(brain), mcl)),
                                                expect_brain=str(brain), expect_mcl=mcl)
    anisotropy = np.asarray(payload["anisotropy"], dtype=float)
    del payload
    return fragments, gt, anisotropy


def segmentation_path(brain):
    """Explicit configs/segmentation_paths.json entry, else the legacy GCS layout."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from dataset_config import get_segmentation_path
    return get_segmentation_path(str(brain), str(PROJECT_ROOT / "configs" / "segmentation_datasets.rtf"))


def open_segmentation(path, credentials=None):
    """TensorStore handle on the dense segmentation (5D: t, c, z, y, x)."""
    if credentials:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials)
    elif not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        for candidate in DEFAULT_CREDENTIALS:
            if candidate.is_file():
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(candidate)
                break
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    from agentic_neuron_proofreader.utils import img_util
    return img_util.TensorStoreImage(path)


def read_segmentation(store, origin, shape):
    """Segment ids in a level-0 voxel box; zeros outside the stored domain."""
    full = np.asarray(store.img.shape[-3:], dtype=int)
    lo, hi = np.maximum(origin, 0), np.minimum(origin + shape, full)
    block = np.zeros(tuple(shape), dtype=np.uint64)
    if (hi > lo).all():
        data = np.asarray(store.img[(0, 0, *[slice(int(a), int(b)) for a, b in zip(lo, hi)])].read().result())
        block[tuple(slice(int(a), int(b)) for a, b in zip(lo - origin, hi - origin))] = data
    return block


def labelled_band_rows(table, entry, tier):
    """True-error rows inside the cached band that have the requested image tier,
    in detector-rank order."""
    positives = np.flatnonzero(np.asarray(table.truth) == 1)
    tier_rows = entry.manifest["image_tiers"].get(tier, {}).get("rows", 0)
    chosen = sorted((entry.position[int(r)], int(r)) for r in positives
                    if entry.position.get(int(r)) is not None and entry.position[int(r)] < tier_rows)
    return [row for _, row in chosen]


def pick_examples(rows, count):
    """Evenly spaced picks across the eligible list (first, middle, last for 3)."""
    if len(rows) <= count:
        return list(rows)
    picks = np.unique(np.round(np.linspace(0, len(rows) - 1, count)).astype(int))
    return [rows[i] for i in picks]


def candidate_anchors(table, row):
    """Fragment-graph node ids and segmentation ids of a candidate, anchor order."""
    candidate = table.candidates[row]
    if table.kind == "merge":
        return [int(candidate["node_id"])], [int(candidate["segment_id"])]
    site = (candidate.get("occurrences") or [candidate])[0]
    return ([int(site["node_id_a"]), int(site["node_id_b"])],
            [int(candidate["segment_id_a"]), int(candidate["segment_id_b"])])


def cached_patch_box(entry, row, tier):
    """Level-0 voxel origin and shape of the cached patch (the cache keeps the
    absolute origin host-side only; level-1 boxes are doubled to level 0)."""
    info = entry.manifest["image_tiers"][tier]
    chunk_rows = info.get("chunk_rows", CHUNK_ROWS)
    index = entry.position[int(row)]
    chunk = entry._chunk(tier, index // chunk_rows)
    local = index % chunk_rows
    origin = np.asarray(chunk["origin_zyx"][local], dtype=int)
    shape = np.asarray(chunk["shape_zyx"][local], dtype=int)
    factor = 2 if tier == "level1" else 1
    return origin * factor, shape * factor


# ----------------------------------------------------------------------------- drawing

def segment_colours(segments):
    order = {segment: index for index, segment in enumerate(sorted(set(int(s) for s in segments)))}
    return {segment: plt.cm.tab10(index % 10) for segment, index in order.items()}


def intensity_limits(image, valid):
    finite = image[valid] if valid.any() else image.ravel()
    low, high = np.percentile(finite, [1, 99.8])
    return float(low), max(float(high), float(low) + 1.0)


def draw_anchors(ax, anchors_zyx, spacing):
    anchors = np.asarray(anchors_zyx, dtype=float).reshape(-1, 3)
    if len(anchors):
        ax.scatter((anchors[:, 2] + 0.5) * spacing[2], (anchors[:, 1] + 0.5) * spacing[1],
                   marker="x", color="red", s=60, linewidths=1.5, zorder=5)


def finish_patch_axes(ax, shape, spacing, title):
    ax.set_xlim(0, shape[2] * spacing[2])
    ax.set_ylim(0, shape[1] * spacing[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x (um from patch edge)")
    ax.set_ylabel("y (um from patch edge)")
    ax.set_title(title, fontsize=9)


def draw_image_panel(ax, patch, fragment, colours, title):
    image, valid = patch["image_zyx"], patch["valid_zyx"].astype(bool)
    spacing = np.asarray(patch["spacing_zyx_um"], dtype=float)
    low, high = intensity_limits(image, valid)
    extent = [0, image.shape[2] * spacing[2], 0, image.shape[1] * spacing[1]]
    ax.imshow(image.max(axis=0), cmap="gray", vmin=low, vmax=high, origin="lower", extent=extent,
              interpolation="nearest")
    nodes = np.asarray(fragment.get("nodes_zyx", []), dtype=float).reshape(-1, 3)
    segments = fragment["segment"]
    for a, b in fragment["edges"]:
        if a < len(nodes) and b < len(nodes):
            pair = nodes[[a, b]]
            ax.plot((pair[:, 2] + 0.5) * spacing[2], (pair[:, 1] + 0.5) * spacing[1],
                    color=colours[int(segments[a])], linewidth=1.0, alpha=0.9)
    draw_anchors(ax, patch["anchors_zyx"], spacing)
    finish_patch_axes(ax, image.shape, spacing, title)


def draw_fragment_panel(ax, fragment, colours, image_radius_um, neighbourhood_radius_um, title):
    xyz = np.asarray(fragment["xyz_um"], dtype=float).reshape(-1, 3)
    segments, outside = fragment["segment"], np.asarray(fragment["outside_radius"], dtype=bool)
    for a, b in fragment["edges"]:
        pair = xyz[[a, b]]
        faint = outside[a] or outside[b]
        ax.plot(pair[:, 0], pair[:, 1], color=colours[int(segments[a])],
                linewidth=0.8 if faint else 1.3, alpha=0.35 if faint else 0.95)
    ax.scatter(xyz[:, 0], xyz[:, 1], s=3, color=[colours[int(s)] for s in segments], alpha=0.6, zorder=3)
    anchors = [int(a) for a in fragment["anchor_nodes"] if 0 <= int(a) < len(xyz)]
    if anchors:
        ax.scatter(xyz[anchors, 0], xyz[anchors, 1], marker="x", color="red", s=70, linewidths=1.5, zorder=5)
    if image_radius_um:
        r = float(image_radius_um)
        for colour, width in (("white", 1.0), ("black", 0.6)):
            ax.add_patch(plt.Rectangle((-r, -r), 2 * r, 2 * r, fill=False, linestyle="--",
                                       edgecolor=colour, linewidth=width, alpha=0.9))
    r = float(neighbourhood_radius_um)
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_aspect("equal")
    ax.set_facecolor("#f2f2f2")
    ax.set_xlabel("x (um from anchor midpoint)")
    ax.set_ylabel("y (um from anchor midpoint)")
    ax.set_title(title, fontsize=9)


def draw_ground_truth_panel(ax, gt, origin, shape, spacing, background, anchors_zyx, title):
    """GT tracings inside the voxel box, one colour per GT neuron."""
    extent = [0, shape[2] * spacing[2], 0, shape[1] * spacing[1]]
    if background is not None:
        low, high = intensity_limits(background["image_zyx"], background["valid_zyx"].astype(bool))
        ax.imshow(background["image_zyx"].max(axis=0), cmap="gray", vmin=low, vmax=high * 1.6,
                  origin="lower", extent=extent, interpolation="nearest")
    else:
        ax.set_facecolor("#404040")
    edges, components = gt.edges_in_patch(origin, shape, return_components=True)
    nodes, node_components = gt.nodes_in_patch(origin, shape, return_components=True)
    order = {component: index for index, component in enumerate(dict.fromkeys(int(c) for c in node_components))}
    for (start, end), component in zip(edges, components):
        colour = GT_PALETTE(order.get(int(component), 0) % 8)
        ax.plot([(start[2] + 0.5) * spacing[2], (end[2] + 0.5) * spacing[2]],
                [(start[1] + 0.5) * spacing[1], (end[1] + 0.5) * spacing[1]],
                color=colour, linewidth=1.6, alpha=0.95)
    if len(nodes):
        ax.scatter((nodes[:, 2] + 0.5) * spacing[2], (nodes[:, 1] + 0.5) * spacing[1], s=6,
                   color=[GT_PALETTE(order[int(c)] % 8) for c in node_components], zorder=4)
    draw_anchors(ax, anchors_zyx, spacing)
    finish_patch_axes(ax, shape, spacing, f"{title}: {len(order)} GT neuron(s) in box")
    return len(order)


def segmentation_projection(labels, image=None):
    """2D label map: the id at the brightest image voxel along z, else the max id."""
    if image is not None and image.shape == labels.shape:
        depth = image.argmax(axis=0)
        return np.take_along_axis(labels, depth[None], axis=0)[0]
    return labels.max(axis=0)


def draw_segmentation_panel(ax, labels, image, spacing, candidate_segments, anchors_zyx, title):
    shape = labels.shape
    extent = [0, shape[2] * spacing[2], 0, shape[1] * spacing[1]]
    projected = segmentation_projection(labels, image)
    rgb = np.zeros(projected.shape + (3,), dtype=float)
    others = [int(s) for s in np.unique(projected) if s != 0 and int(s) not in candidate_segments]
    for index, segment in enumerate(others):
        rgb[projected == segment] = np.asarray(plt.cm.Pastel2(index % 8))[:3] * 0.55
    for index, segment in enumerate(candidate_segments):
        rgb[projected == segment] = plt.cm.tab10(index % 10)[:3]
    ax.imshow(rgb, origin="lower", extent=extent, interpolation="nearest")
    draw_anchors(ax, anchors_zyx, spacing)
    present = [s for s in candidate_segments if (labels == s).any()]
    finish_patch_axes(ax, shape, spacing,
                      f"{title}: {len(others) + len(present)} segment(s) in box; "
                      f"candidate segment(s) present: {len(present)}/{len(candidate_segments)}")


# ----------------------------------------------------------------------------- figure

def plot_kind(kind, table, entry, graphs, segmentation, brain, *, examples=3, tier="level0", rows=None, out=None):
    """Four-row figure for one kind. Returns (figure, chosen_rows)."""
    fragments, gt, anisotropy = graphs
    spacing0 = anisotropy[::-1]  # zyx microns per level-0 voxel
    eligible = labelled_band_rows(table, entry, tier)
    picks = [int(r) for r in (rows or [])] or pick_examples(eligible, examples)
    scores = np.asarray(table.features["detector_score"], dtype=float)
    figure, axes = plt.subplots(4, examples, figsize=(4.6 * examples, 4.3 * 4), constrained_layout=True,
                                squeeze=False)
    for column in range(examples):
        if column >= len(picks):
            for ax in axes[:, column]:
                ax.axis("off")
            continue
        row = picks[column]
        used_tier = tier
        contexts, patches = entry.contexts(table, [row], inputs="both", tier=tier)
        if patches[0] is None and tier != "level1":
            used_tier = "level1"
            contexts, patches = entry.contexts(table, [row], inputs="both", tier=used_tier)
        context, patch = contexts[0], patches[0]
        fragment = context["fragment"]
        colours = segment_colours(fragment["segment"])
        rank = entry.position[int(row)] + 1
        header = (f"{kind} error, row {row}, detector rank {rank}/{len(entry.rows)}, score {scores[row]:.3f}\n"
                  f"{used_tier} patch, radius {context['image_radius_um']:.0f} um, XY max projection")
        if patch is None:
            axes[0, column].text(0.5, 0.5, "no cached image", ha="center", va="center")
            axes[0, column].set_title(header, fontsize=9)
            axes[0, column].axis("off")
        else:
            draw_image_panel(axes[0, column], patch, fragment, colours, header)
        draw_fragment_panel(axes[1, column], fragment, colours, context["image_radius_um"], context["radius_um"],
                            f"{ROW_NAMES[1]} ({context['radius_um']:.0f} um); dashed box = image patch")

        # Rows 3-4 use the absolute level-0 voxel box of the cached patch.
        anchor_nodes, candidate_segments = candidate_anchors(table, row)
        origin, shape = cached_patch_box(entry, row, used_tier)
        anchors_zyx = (np.asarray(fragments.node_xyz[anchor_nodes], dtype=float) / anisotropy)[:, ::-1] - origin
        background = patch if (patch is not None and used_tier == "level0") else None
        draw_ground_truth_panel(axes[2, column], gt, origin, shape, spacing0, background, anchors_zyx, ROW_NAMES[2])
        if segmentation is None:
            axes[3, column].text(0.5, 0.5, "segmentation not read", ha="center", va="center")
            axes[3, column].set_title(ROW_NAMES[3], fontsize=9)
            axes[3, column].axis("off")
        else:
            labels = read_segmentation(segmentation, origin, shape)
            image = background["image_zyx"] if background is not None else None
            draw_segmentation_panel(axes[3, column], labels, image, spacing0, candidate_segments, anchors_zyx,
                                    ROW_NAMES[3])
    handles = [Line2D([0], [0], color="red", marker="x", linestyle="", markersize=8, label="candidate anchor(s)"),
               Line2D([0], [0], color=plt.cm.tab10(0), linewidth=2, label="candidate segment (first anchor)"),
               Line2D([0], [0], color=plt.cm.tab10(1), linewidth=2, label="candidate segment (second anchor)"),
               Line2D([0], [0], color="black", linestyle="--", label="image patch footprint"),
               Line2D([0], [0], color=GT_PALETTE(0), linewidth=2, label="ground truth: one colour per neuron"),
               Patch(facecolor=np.asarray(plt.cm.Pastel2(0))[:3] * 0.55, label="segmentation: other segments")]
    figure.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    figure.suptitle(f"Brain {brain}: {kind} errors (evaluator-labelled) - rows: {', '.join(ROW_NAMES)}.\n"
                    f"Human review only; display scaling does not change model pixels.", fontsize=11)
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out, dpi=130, bbox_inches="tight")
    return figure, picks


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brain", default="802449")
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--kinds", default="split,merge")
    parser.add_argument("--tier", choices=("level0", "level1"), default="level0",
                        help="image tier to show; a row without a cached patch at that tier falls back to level1")
    parser.add_argument("--context-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--feature-tables-dir", type=Path, default=pc.DEFAULT_OUT)
    parser.add_argument("--merge-dir", action="append")
    parser.add_argument("--split-dir", action="append")
    parser.add_argument("--rows-split", help="comma-separated table rows to show instead of automatic picks")
    parser.add_argument("--rows-merge", help="comma-separated table rows to show instead of automatic picks")
    parser.add_argument("--segmentation-path", default=None, help="override the GCS segmentation store")
    parser.add_argument("--gcs-credentials", type=Path, default=None)
    parser.add_argument("--no-segmentation", action="store_true", help="skip the segmentation row")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIGS,
                        help="output directory; files are error_context_<kind>_<brain>_<tier>.png")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    bank = load_tables(args.brain, args.mcl, args.feature_tables_dir, args.merge_dir, args.split_dir)
    cache = ContextCache(args.context_cache)
    print("loading labelled graphs ...", flush=True)
    graphs = load_graphs(args.brain, args.mcl)
    segmentation = None
    if not args.no_segmentation:
        path = args.segmentation_path or segmentation_path(args.brain)
        segmentation = open_segmentation(path, args.gcs_credentials)
        print(f"segmentation store: {path} shape {tuple(segmentation.img.shape)}", flush=True)
    rows = {}
    for kind, text in (("split", args.rows_split), ("merge", args.rows_merge)):
        if text:
            rows[kind] = [int(v) for v in text.split(",") if v.strip()]
    for kind in kinds:
        table = bank.tables[kind]
        entry = cache.entry(table)
        out = args.out_dir / f"error_context_{kind}_{args.brain}_{args.tier}.png"
        figure, picks = plot_kind(kind, table, entry, graphs, segmentation, args.brain, examples=args.examples,
                                  tier=args.tier, rows=rows.get(kind), out=out)
        plt.close(figure)
        print(f"{kind}: rows {picks} -> {out}", flush=True)


if __name__ == "__main__":
    main()

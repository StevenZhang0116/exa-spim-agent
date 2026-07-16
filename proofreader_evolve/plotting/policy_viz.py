"""
Shared rendering for compare_proofreader_policy.ipynb — apply an evolved policy to a
brain and visualize its CORRECT repairs (repaired splits) and its MERGE ERRORS (false
merges), before/after, in image + skeleton MIPs.

This module holds the logic that used to be duplicated inline in the notebook so the
SAME code can run on the TEST (held-out) brain AND the TRAIN brain — the notebook just
builds a ``BrainViz`` per brain and calls the render helpers. Nothing here is specific
to one brain; every brain-dependent object (graphs, image reader, edits, GT map) is
carried on ``BrainViz``.

Coordinate convention (must match agentic_neuron_proofreader.utils.img_util):
  * plot_skeleton_mips draws a 1x3 (XY/XZ/YZ) panel; panel i uses ``plane_axes[i] =
    (a, b)`` over local (z, y, x): x = coords[b], y = coords[a], xlim=(0,patch[b]),
    ylim=(patch[a],0). The error box below reuses exactly this, so a box lands on the
    real merge location on all three projections.

Not a standalone CLI — imported by the notebook (which sets sys.path and passes in the
already-loaded graphs / readers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# plot_skeleton_mips' per-panel (row_axis a, col_axis b) over local (z, y, x).
_PLANE_AXES = [(1, 2), (0, 2), (0, 1)]
_SLICE_FRACTIONS = (0.2, 0.4, 0.5, 0.6, 0.8)


def pair_key(a, b) -> tuple[str, str]:
    """Canonical unordered label-pair key (matches the notebook's _pair_key)."""
    a, b = str(a), str(b)
    return (a, b) if a <= b else (b, a)


@dataclass
class BrainViz:
    """Everything needed to render one brain's repairs / errors. Built by the notebook
    per brain (train or test) so the render helpers are brain-agnostic.

    role          — "test" or "train" (labelling + output-subdir naming only).
    brain_id      — e.g. "794491".
    split_sites   — enumerated SplitSites the policy reasoned over on this brain.
    edits         — the merge_labels edits the policy emitted on this brain.
    label_gt_map  — {label: {neuron: node_count}} over this brain's GT (for correct/
                    false classification + the pre-existing blame test).
    bd/gt_viz/fr_viz/handler/aniso — the *_add visualization cache graphs + the
                    EditHandler over ``edits`` + anisotropy, all for THIS brain.
    patch_shape   — (z, y, x) receptive-field cube for the MIPs.
    preexist_min  — nodes-on-both-neurons threshold for the pre-existing test
                    (== incremental_scoring._MERGE_MIN_NODES; the gate's rule).
    show_slices   — also render multi-depth raw-image slices.
    """
    role: str
    brain_id: str
    split_sites: list
    edits: list
    label_gt_map: dict
    bd: object
    gt_viz: object
    fr_viz: object
    handler: object
    aniso: object
    patch_shape: tuple
    preexist_min: int
    show_slices: bool = True

    accepted: set = field(default=None)  # {pair_key} the policy merged (built in __post_init__)

    def __post_init__(self):
        self.accepted = {pair_key(e["label_a"], e["label_b"]) for e in self.edits}

    # --- GT lookups -----------------------------------------------------------------
    def dominant(self, label):
        c = self.label_gt_map.get(str(label))
        return max(c, key=c.get) if c else None

    def _spans_both(self, label, n1, n2) -> bool:
        """True iff ``label`` already lands on BOTH neurons with >= preexist_min nodes
        each — a pre-existing two-neuron merge (matches classify_merge_edits._spans_both)."""
        c = self.label_gt_map.get(str(label)) or {}
        return c.get(n1, 0) >= self.preexist_min and c.get(n2, 0) >= self.preexist_min

    def blame(self, la, lb, da, db) -> str:
        """'PREEXISTING' iff either fragment already straddles both neurons; else 'POLICY'."""
        return ("PREEXISTING" if (self._spans_both(la, da, db) or self._spans_both(lb, da, db))
                else "POLICY")

    # --- site collection ------------------------------------------------------------
    def repaired_sites(self, allowed_neurons: set | None = None) -> list:
        """SplitSites the policy MERGED whose two fragments share a dominant neuron (a
        correctly-closed real split). Optional ``allowed_neurons`` restricts which GT
        neurons the shown repairs may sit on (does not affect scoring)."""
        out = []
        for s in self.split_sites:
            la, lb = str(s.label_a), str(s.label_b)
            if pair_key(la, lb) not in self.accepted:
                continue
            da, db = self.dominant(la), self.dominant(lb)
            if da is None or da != db:
                continue
            if allowed_neurons is not None and da not in allowed_neurons:
                continue
            out.append(s)
        return out

    def merge_error_sites(self) -> list:
        """(site, da, db, blame) for every false merge: accepted AND cross-neuron
        (da != db). Ordered POLICY-first, then by gap."""
        out = []
        for s in self.split_sites:
            la, lb = str(s.label_a), str(s.label_b)
            if pair_key(la, lb) not in self.accepted:
                continue
            da, db = self.dominant(la), self.dominant(lb)
            if da is None or db is None or da == db:
                continue
            out.append((s, da, db, self.blame(la, lb, da, db)))
        out.sort(key=lambda t: ({"POLICY": 0, "PREEXISTING": 1}[t[3]], t[0].gap_um))
        return out


def pick_spread(sites: list, n: int, seed: int) -> list:
    """Deterministically pick ``n`` sites spread evenly by gap across the middle 10-90%
    band (skips the extreme-near/far tails); ``seed`` shifts the (reproducible) set."""
    if not sites:
        return []
    sites = sorted(sites, key=lambda s: s.gap_um)
    band = sites[len(sites) // 10: 9 * len(sites) // 10] or sites
    idxs = [int((k + 0.5) * len(band) / n) + seed for k in range(n)]
    return [band[i % len(band)] for i in idxs]


# --- low-level rendering ------------------------------------------------------------
def _plot_slices(img_util, patch, fractions=_SLICE_FRACTIONS):
    """Multi-depth single-plane slices (rows = depth fraction; cols = XY/XZ/YZ)."""
    nz, ny, nx = patch.shape
    names = ["XY (vary z)", "XZ (vary y)", "YZ (vary x)"]
    sizes = [nz, ny, nx]
    vmax = np.percentile(patch, 99.9)
    fig, axs = plt.subplots(len(fractions), 3, figsize=(10, 3.2 * len(fractions)))
    axs = np.atleast_2d(axs)
    for r, frac in enumerate(fractions):
        idx = [min(int(frac * s), s - 1) for s in sizes]
        planes = [patch[idx[0]], patch[:, idx[1]], patch[:, :, idx[2]]]
        for c, (ax, plane, nm) in enumerate(zip(axs[r], planes, names)):
            ax.imshow(plane, cmap="gray", vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(nm, fontsize=14)
            if c == 0:
                ax.set_ylabel(f"{int(frac*100)}% depth\n(plane {idx[0]})", fontsize=12)
    plt.tight_layout(); plt.show()


def _box_current_skeleton_fig(lo, hi):
    """Draw an error box (local-voxel [lo, hi] in (z,y,x)) on the 3 panels of the
    skeleton-MIP figure just produced. plot_skeleton_mips called plt.show() (a no-op in
    the PDF loop), so that figure is still current; its axes are the XY/XZ/YZ panels."""
    from matplotlib.patches import Rectangle
    for ax, (a, b) in zip(plt.gcf().axes, _PLANE_AXES):
        ax.add_patch(Rectangle((lo[b], lo[a]), hi[b] - lo[b], hi[a] - lo[a],
                               fill=False, edgecolor="yellow", linewidth=1.8, zorder=10))


def _patch_at(bv: BrainViz, site, img_util):
    """Common patch setup for a site: center voxel, offset, image cube, and the GT /
    fragment nodes+edges+components in the patch. Returns a dict of everything the
    renderers need."""
    center_um = tuple((np.asarray(site.xyz_a, float) + np.asarray(site.xyz_b, float)) / 2.0)
    center_voxel = img_util.to_voxels(center_um, bv.aniso)
    offset = tuple(c - s // 2 for c, s in zip(center_voxel, bv.patch_shape))
    img_patch = np.squeeze(bv.bd.img.read(center_voxel, bv.patch_shape))
    gt_nodes, gt_ncomp = bv.gt_viz.nodes_in_patch(offset, bv.patch_shape, return_components=True)
    gt_edges, gt_ecomp = bv.gt_viz.edges_in_patch(offset, bv.patch_shape, return_components=True)
    fr_nodes, fr_nid, fr_ncomp = bv.fr_viz.nodes_in_patch(
        offset, bv.patch_shape, return_ids=True, return_components=True)
    fr_edges, fr_ecomp = bv.fr_viz.edges_in_patch(offset, bv.patch_shape, return_components=True)
    # Fragment component-id -> the policy's edited equivalence class (so the merged pair
    # shares a colour AFTER; that shared colour is the repair/fusion).
    comp_rep = {}
    for nid, comp in zip(fr_nid, fr_ncomp):
        comp_rep.setdefault(int(comp), int(nid))
    comp_edited = {c: bv.handler.get(str(bv.fr_viz.node_segment_id(rep)))
                   for c, rep in comp_rep.items()}
    cls_int = {cls: i for i, cls in enumerate(sorted(set(comp_edited.values())))}
    fr_ncomp_after = np.array([cls_int[comp_edited[int(c)]] for c in fr_ncomp], dtype=int)
    fr_ecomp_after = np.array([cls_int[comp_edited[int(c)]] for c in fr_ecomp], dtype=int)
    return dict(center_um=center_um, offset=offset, img_patch=img_patch,
                gt_nodes=gt_nodes, gt_ncomp=gt_ncomp, gt_edges=gt_edges, gt_ecomp=gt_ecomp,
                fr_nodes=fr_nodes, fr_ncomp=fr_ncomp, fr_edges=fr_edges, fr_ecomp=fr_ecomp,
                fr_ncomp_after=fr_ncomp_after, fr_ecomp_after=fr_ecomp_after)


def _error_box_bounds(bv: BrainViz, site, img_util, margin=12.0):
    """[lo, hi] local-voxel bounds around the two joined tips (+ margin), for the box."""
    offset = tuple(c - s // 2 for c, s in
                   zip(img_util.to_voxels(tuple((np.asarray(site.xyz_a, float)
                       + np.asarray(site.xyz_b, float)) / 2.0), bv.aniso), bv.patch_shape))
    pa = np.asarray(img_util.to_voxels(site.xyz_a, bv.aniso), float) - np.asarray(offset, float)
    pb = np.asarray(img_util.to_voxels(site.xyz_b, bv.aniso), float) - np.asarray(offset, float)
    return np.minimum(pa, pb) - margin, np.maximum(pa, pb) + margin


def show_repair(bv: BrainViz, site, img_util):
    """Render ONE correctly-repaired SplitSite: raw MIP (+slices), GT (once), fragments
    BEFORE (two colours = the split) and AFTER (one colour = the repair)."""
    P = _patch_at(bv, site, img_util)
    tag = (f"[{bv.role} {bv.brain_id}] REPAIR {site.label_a}+{site.label_b}  "
           f"gap {site.gap_um:.2f} µm  GT {bv.dominant(site.label_a)}  "
           f"@ {tuple(round(v,1) for v in P['center_um'])} µm")
    print(f"\n=== {tag} ===")
    print("Raw image (MIP):"); img_util.plot_mips(P["img_patch"])
    if bv.show_slices:
        _plot_slices(img_util, P["img_patch"])
    img_util.plot_skeleton_mips({"GT": {
        "nodes": P["gt_nodes"], "node_components": P["gt_ncomp"],
        "edges": P["gt_edges"], "components": P["gt_ecomp"], "color": "lime"}},
        bv.patch_shape, separate_rows=True)
    img_util.plot_skeleton_mips({"Fragments (before)": {
        "nodes": P["fr_nodes"], "node_components": P["fr_ncomp"],
        "edges": P["fr_edges"], "components": P["fr_ecomp"], "color": "cyan"}},
        bv.patch_shape, separate_rows=True)
    img_util.plot_skeleton_mips({"Fragments (after)": {
        "nodes": P["fr_nodes"], "node_components": P["fr_ncomp_after"],
        "edges": P["fr_edges"], "components": P["fr_ecomp_after"], "color": "cyan"}},
        bv.patch_shape, separate_rows=True)


def show_merge_error(bv: BrainViz, site, da, db, blame, img_util):
    """Render ONE false-merge SplitSite framed as an ERROR, tagged with its BLAME, with
    a yellow box marking the merge location on each skeleton panel (GT / before / after)."""
    P = _patch_at(bv, site, img_util)
    lo, hi = _error_box_bounds(bv, site, img_util)
    _why = ("POLICY-CAUSED (a NEW fusion of two clean fragments — penalized by the gate)"
            if blame == "POLICY" else
            "PRE-EXISTING (a fragment already spanned both neurons before the merge — "
            "NOT the policy's fault, NOT penalized)")
    tag = (f"[{bv.role} {bv.brain_id}][{blame}] FALSE MERGE {site.label_a}+{site.label_b}  "
           f"gap {site.gap_um:.2f} µm  neurons {da} + {db}  "
           f"@ {tuple(round(v,1) for v in P['center_um'])} µm")
    print(f"\n=== {tag} ===\n    {_why}")
    print("    (yellow box on skeleton panels = the merge location: joined tips + gap)")
    print("Raw image (MIP):"); img_util.plot_mips(P["img_patch"])
    if bv.show_slices:
        _plot_slices(img_util, P["img_patch"])
    img_util.plot_skeleton_mips({"GT (distinct neurons)": {
        "nodes": P["gt_nodes"], "node_components": P["gt_ncomp"],
        "edges": P["gt_edges"], "components": P["gt_ecomp"], "color": "lime"}},
        bv.patch_shape, separate_rows=True)
    _box_current_skeleton_fig(lo, hi)
    img_util.plot_skeleton_mips({"Fragments (before)": {
        "nodes": P["fr_nodes"], "node_components": P["fr_ncomp"],
        "edges": P["fr_edges"], "components": P["fr_ecomp"], "color": "red"}},
        bv.patch_shape, separate_rows=True)
    _box_current_skeleton_fig(lo, hi)
    img_util.plot_skeleton_mips({"Fragments (after = ERROR)": {
        "nodes": P["fr_nodes"], "node_components": P["fr_ncomp_after"],
        "edges": P["fr_edges"], "components": P["fr_ecomp_after"], "color": "red"}},
        bv.patch_shape, separate_rows=True)
    _box_current_skeleton_fig(lo, hi)


def _render_to_pdfs(items, render_fn, out_dir: Path, name_fn, suptitle_fn=None):
    """Shared PDF loop: for each item, isolate its figures, render, optionally stamp a
    suptitle, and save all figures into ONE per-item PDF under ``out_dir``. Suppresses
    the plotters' internal plt.show() (which the inline backend would otherwise use to
    close figures before we can save). Returns the list of written paths."""
    from matplotlib.backends.backend_pdf import PdfPages
    out_dir.mkdir(parents=True, exist_ok=True)
    written, real_show = [], plt.show
    plt.show = lambda *a, **k: None
    try:
        for k, item in enumerate(items):
            plt.close("all")
            render_fn(item)
            figs = [plt.figure(n) for n in plt.get_fignums()]
            if suptitle_fn is not None:
                st, color = suptitle_fn(item)
                for f in figs:
                    f.suptitle(st, fontsize=11, color=color)
            pdf_path = out_dir / name_fn(k, item)
            with PdfPages(pdf_path) as pdf:
                for f in figs:
                    pdf.savefig(f, bbox_inches="tight")
            written.append(pdf_path)
            print(f"  [{k+1}/{len(items)}] saved {len(figs)} figures -> {pdf_path}")
    finally:
        plt.show = real_show
        plt.close("all")
    return written


def render_repairs(bv: BrainViz, sites, out_dir: Path, img_util) -> list:
    """Render each repaired SplitSite to its own PDF under ``out_dir`` (one per site)."""
    _pad = max(2, len(str(max(len(sites) - 1, 0))))
    return _render_to_pdfs(
        list(enumerate(sites)),
        render_fn=lambda it: show_repair(bv, it[1], img_util),
        out_dir=out_dir,
        name_fn=lambda k, it: (f"ex{k:0{_pad}d}_{it[1].label_a}-{it[1].label_b}_"
                               f"{bv.dominant(it[1].label_a)}.pdf"),
    )


def render_merge_errors(bv: BrainViz, errors, out_dir: Path, img_util) -> list:
    """Render EVERY false merge to its own PDF (blame in filename + coloured suptitle)."""
    _pad = max(2, len(str(max(len(errors) - 1, 0))))
    def _name(k, it):
        s, da, db, bl = it[1]
        return f"err{k:0{_pad}d}_{bl}_{s.label_a}-{s.label_b}_{da}__{db}.pdf"
    def _suptitle(it):
        s, da, db, bl = it[1]
        return (f"[{bv.role} {bv.brain_id}][{bl}] false merge {s.label_a}+{s.label_b}  "
                f"neurons {da}+{db}  gap {s.gap_um:.2f} um",
                "#d62728" if bl == "POLICY" else "#7f7f7f")
    return _render_to_pdfs(
        list(enumerate(errors)),
        render_fn=lambda it: show_merge_error(bv, *it[1], img_util=img_util),
        out_dir=out_dir, name_fn=_name, suptitle_fn=_suptitle)

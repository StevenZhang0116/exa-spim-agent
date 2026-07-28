"""Verify the metrics_out merge_sites.csv coordinates align with the agentic *_add viz
cache: for each row, does the flagged Segment_ID land in the patch centered at
reverse(CSV Voxel)? Reports per-frame landing counts. Run on a compute node."""
import sys
from pathlib import Path
REPO = "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent"
sys.path.insert(0, REPO)
sys.path.insert(0, REPO.rsplit("/", 1)[0] + "/segmentation-skeleton-metrics/src")
_OUT = open(f"{REPO}/proofreader_evolve/scratch_frame_check.out", "w")
def P(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); _OUT.write(m+"\n"); _OUT.flush()

import numpy as np, pandas as pd
from agentic_neuron_proofreader.utils import img_util
from agentic_neuron_proofreader.data_modules.datasets import BrainDataset

BRAIN, MCL, PATCH = "794495", 100, 160
SEG = "raw.unet_449_792_202_splits_and_merges_831600"
PATCH_SHAPE = (PATCH, PATCH, PATCH)

CSV = Path(REPO) / "metrics_out" / BRAIN / SEG / "merge_sites.csv"
merges = pd.read_csv(CSV)
def _pt(s): return tuple(float(x) for x in str(s).strip("()[] ").split(","))
merges["Voxel_t"] = merges["Voxel"].map(_pt)
merges["World_t"] = merges["World"].map(_pt)
P(f"{len(merges)} merge rows in {CSV.name}")

P("loading *_add viz cache (~2.3 GB) ...")
bd = BrainDataset.load_from_cache(f"{REPO}/cache/dataset_cache_{BRAIN}_mcl{MCL}_add.pkl")
fr_viz, gt_viz = bd.fragments_graph, bd.gt_graph
aniso = fr_viz.anisotropy
P(f"  fragments={fr_viz.number_of_nodes()} nodes, GT={gt_viz.number_of_nodes()} nodes, aniso={tuple(aniso)}")

def seg_count(center_voxel, seg):
    offset = tuple(int(c) - d // 2 for c, d in zip(center_voxel, PATCH_SHAPE))
    nodes, node_ids = fr_viz.nodes_in_patch(offset, PATCH_SHAPE, return_ids=True)[:2]
    return sum(1 for n in node_ids if str(fr_viz.node_segment_id(int(n))) == str(seg))

def gt_neurons(center_voxel):
    offset = tuple(int(c) - d // 2 for c, d in zip(center_voxel, PATCH_SHAPE))
    _, ncomp = gt_viz.nodes_in_patch(offset, PATCH_SHAPE, return_components=True)
    return len(set(ncomp)) if len(ncomp) else 0

frames = {"reverse(Voxel)": lambda v, w: tuple(int(x) for x in v[::-1]),
          "to_voxels(World)": lambda v, w: tuple(img_util.to_voxels(tuple(w), aniso)),
          "raw Voxel":       lambda v, w: tuple(int(x) for x in v)}

land = {k: 0 for k in frames}          # rows whose Segment_ID lands, per frame
land_any = 0
rev_gtn = []                            # GT-neuron count in patch (reverse(Voxel) frame)
P("\nscanning all rows ...")
for i, row in merges.iterrows():
    v, w, seg = row["Voxel_t"], row["World_t"], row["Segment_ID"]
    hit_any = False
    for name, fn in frames.items():
        if seg_count(fn(v, w), seg) > 0:
            land[name] += 1; hit_any = True
    land_any += int(hit_any)
    rev_gtn.append(gt_neurons(frames["reverse(Voxel)"](v, w)))

n = len(merges)
P("\n=== Segment_ID lands in patch, by coordinate frame ===")
for k in frames:
    P(f"  {k:18}: {land[k]:3}/{n}  ({100*land[k]/n:.0f}%)")
P(f"  landed by SOME frame: {land_any}/{n}")
rev_gtn = np.array(rev_gtn)
P("\n=== using reverse(Voxel) — GT neurons in patch ===")
P(f"  patches with >=1 GT neuron: {(rev_gtn>=1).sum()}/{n}")
P(f"  patches with >=2 GT neurons: {(rev_gtn>=2).sum()}/{n}")
P(f"  patches with 0 GT (untraced): {(rev_gtn==0).sum()}/{n}")
P("=== DONE ===")

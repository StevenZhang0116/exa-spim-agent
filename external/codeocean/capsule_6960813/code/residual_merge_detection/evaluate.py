#!/usr/bin/env python3
"""Residual merge detection: score merge-site predictions made on the final
(post error-correction) SWCs against the merge sites the pipeline evaluation found
in those same SWCs.

Inputs
  --predictions-dir  folder with site_predictions_<cfg>.csv
                     (columns: node, xyz "(x, y, z)" in world um, Segment_ID, Prediction, Degree)
  --eval-root        folder with arborist_v2_eval_<run>/<result dir>/<step>/ from the pipeline evaluation
  --step             evaluation step whose SWCs the predictions were made on (default step4_learned_split_correction)

Per configuration the script
  1. keeps predictions within --gt-filter-um of any ground-truth skeleton node and
     farther than --ignore-um from any reviewed false-positive site (ignored_merges.swc);
  2. takes the step's *_merge_sites.csv (World column) as ground truth and keeps only
     sites within --site-radius-um of some kept prediction, since the detector only
     scores branch points and the other sites have no branch point to be found at;
  3. for thresholds 0.00..1.00 counts positives (score >= t), true positives (positive
     within --match-um of a kept GT site), false positives (total and per GT neuron),
     recall over reachable (kept) merge sites plus recall over all sites, and false
     positives farther than --soma-um from the soma (node 0 of the nearest GT neuron's SWC).

Outputs (in --out)
  pr_curves.csv                 one row per (config, threshold)
  table.md                      FP per neuron and reachable recall at 0.1 steps
  gt_merge_sites_<cfg>.csv      merge sites with nearest-branch-point distance and kept flag
  predictions_labeled.csv       kept predictions with is_tp, distances, GT neuron, soma distance
  summary.json                  counts per config
  slide_data.json               compact series for make_slide.py / make_fig.py
"""
import argparse, ast, glob, io, json, os, zipfile
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

RUN_DIR = {"it2": "arborist_v2_eval_baseline"}   # others resolve to arborist_v2_eval_<cfg>
THRESHOLDS = np.round(np.arange(0.0, 1.0001, 0.01), 2)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions-dir", default="/root/capsule/results")
    p.add_argument("--eval-root", default="/root/capsule/data/arborist_v2_agg08_eval")
    p.add_argument("--step", default="step4_learned_split_correction")
    p.add_argument("--configs", nargs="*", default=None, help="default: every site_predictions_<cfg>.csv found")
    p.add_argument("--out", default="/root/capsule/results/residual_merge_detection")
    p.add_argument("--gt-filter-um", type=float, default=5.0)
    p.add_argument("--site-radius-um", type=float, default=10.0)
    p.add_argument("--match-um", type=float, default=10.0)
    p.add_argument("--ignore-um", type=float, default=30.0)
    p.add_argument("--soma-um", type=float, default=1000.0)
    p.add_argument("--mark-threshold", type=float, default=0.1, help="threshold marked with a vertical line and value labels on the slide and PNG")
    return p.parse_args()


def step_dir(eval_root, cfg, step):
    run = RUN_DIR.get(cfg, f"arborist_v2_eval_{cfg}")
    hits = glob.glob(f"{eval_root}/{run}/*/{step}") or glob.glob(f"{eval_root}/{run}/{step}")
    if not hits:
        raise FileNotFoundError(f"no {step} folder under {eval_root}/{run}")
    return hits[0]


def load_gt(sdir, step):
    """GT skeleton nodes (world um), neuron label per node, soma (node 0) per neuron."""
    pts, lab, soma = [], [], {}
    for zp in sorted(glob.glob(f"{sdir}/{step}_fragments/N*.zip")):
        z = zipfile.ZipFile(zp)
        name = [n for n in z.namelist() if n.startswith("N0")][0]
        a = np.loadtxt(io.StringIO(z.read(name).decode()), comments="#")
        neuron = name.split(".")[0]
        pts.append(a[:, 2:5]); lab += [neuron] * len(a); soma[neuron] = a[0, 2:5]
    return np.vstack(pts), np.array(lab), soma


def load_ignored(sdir):
    p = f"{sdir}/ignored_merges.swc"
    if not os.path.exists(p):
        return np.zeros((0, 3))
    return pd.read_csv(p, sep=r"\s+", comment="#", header=None, usecols=[2, 3, 4]).values


def xyz_col(series):
    return np.array(series.map(ast.literal_eval).tolist(), float)


def evaluate(cfg, args):
    sdir = step_dir(args.eval_root, cfg, args.step)
    preds = pd.read_csv(f"{args.predictions_dir}/site_predictions_{cfg}.csv")
    pxyz = xyz_col(preds.xyz)
    G, glab, soma = load_gt(sdir, args.step)
    gtree = cKDTree(G)
    d_skel, j_skel = gtree.query(pxyz)
    keep = d_skel <= args.gt_filter_um
    ign = load_ignored(sdir)
    keep_ign = cKDTree(ign).query(pxyz)[0] > args.ignore_um if len(ign) else np.ones(len(pxyz), bool)
    sel = keep & keep_ign
    P, S = pxyz[sel], preds.Prediction.values[sel]

    sites = pd.read_csv(f"{sdir}/{args.step}_merge_sites.csv")
    site_xyz = xyz_col(sites.World)
    d_site = cKDTree(P).query(site_xyz)[0]
    kept_site = d_site <= args.site_radius_um
    gxyz = site_xyz[kept_site]
    sites.insert(0, "config", cfg)
    sites["nearest_branch_pred_um"] = d_site.round(2); sites["kept"] = kept_site
    sites.to_csv(f"{args.out}/gt_merge_sites_{cfg}.csv", index=False)

    d_gt, j_gt = cKDTree(gxyz).query(P)
    is_tp = d_gt <= args.match_um
    best_all = np.array([S[n].max() if n else -np.inf for n in cKDTree(P).query_ball_point(site_xyz, args.match_um)])
    best = best_all[kept_site]
    n_neurons = len(soma)
    neuron = glab[j_skel[sel]]
    d_soma = np.linalg.norm(P - np.array([soma[n] for n in neuron]), axis=1)
    far = d_soma > args.soma_um

    rows = []
    for t in THRESHOLDS:
        pos = S >= t
        npos, tp = int(pos.sum()), int((pos & is_tp).sum())
        rows.append(dict(config=cfg, threshold=t, n_pred_pos=npos, tp=tp, fp=npos - tp,
                         fp_far_from_soma=int((pos & ~is_tp & far).sum()),
                         fp_per_neuron=(npos - tp) / n_neurons, fp_far_per_neuron=int((pos & ~is_tp & far).sum()) / n_neurons,
                         precision=tp / npos if npos else np.nan,
                         recall=float((best_all >= t).mean()), recall_reachable=float((best >= t).mean()),
                         n_gt_sites_kept=len(gxyz), n_gt_sites_all=len(site_xyz), n_neurons=n_neurons))
    lab = preds[sel].copy()
    lab.insert(0, "config", cfg)
    lab["dist_to_gt_skeleton_um"] = d_skel[sel].round(2)
    lab["gt_neuron"] = neuron
    lab["dist_to_soma_um"] = d_soma.round(1)
    lab["dist_to_nearest_kept_gt_merge_um"] = d_gt.round(2)
    lab["nearest_gt_merge_id"] = sites.Merge_ID.values[kept_site][j_gt]
    lab["is_tp"] = is_tp
    info = dict(n_pred_total=len(preds), n_pred_within_gt=int(keep.sum()), n_pred_near_ignored=int((keep & ~keep_ign).sum()),
                n_pred_used=int(sel.sum()), n_gt_sites_all=len(site_xyz), n_gt_sites_kept=int(kept_site.sum()),
                n_ignored_sites=len(ign), n_tp_far_from_soma=int((is_tp & far).sum()), n_tp_sites=int(is_tp.sum()), n_neurons=n_neurons)
    return pd.DataFrame(rows), lab, info


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfgs = args.configs or sorted(os.path.basename(f)[len("site_predictions_"):-4]
                                  for f in glob.glob(f"{args.predictions_dir}/site_predictions_*.csv"))
    order = [c for c in ("it2", "dt5k", "dt20k", "dt50k") if c in cfgs] + [c for c in cfgs if c not in ("it2", "dt5k", "dt20k", "dt50k")]
    curves, labeled, summary = [], [], {"params": vars(args), "configs": {}}
    for cfg in order:
        C, L, info = evaluate(cfg, args)
        curves.append(C); labeled.append(L); summary["configs"][cfg] = info
        print(cfg, json.dumps(info))
    C = pd.concat(curves, ignore_index=True)
    C.to_csv(f"{args.out}/pr_curves.csv", index=False)
    pd.concat(labeled, ignore_index=True).to_csv(f"{args.out}/predictions_labeled.csv", index=False)
    json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1)

    slide = {"thresholds": THRESHOLDS.tolist(), "params": {k: v for k, v in vars(args).items() if k.endswith("_um")},
             "n_neurons": int(C.n_neurons.iloc[0]), "mark_threshold": args.mark_threshold, "configs": {}}
    for cfg in order:
        c = C[C.config == cfg]
        slide["configs"][cfg] = dict(recall=c.recall_reachable.round(4).tolist(), recall_all=c.recall.round(4).tolist(), fp=c.fp_per_neuron.round(3).tolist(),
                                     n_gt=int(c.n_gt_sites_kept.iloc[0]), n_gt_all=int(c.n_gt_sites_all.iloc[0]), n_cand=int(c.n_pred_pos.iloc[0]))
    json.dump(slide, open(f"{args.out}/slide_data.json", "w"))

    T = C[C.threshold.isin(np.round(np.arange(0.1, 0.91, 0.1), 2))]
    hdr = "| threshold | " + " | ".join(f"{c} FP/neuron | {c} recall" for c in order) + " |"
    lines = [hdr, "|" + "---|" * (1 + 2 * len(order))]
    for t, g in T.groupby("threshold"):
        g = g.set_index("config")
        lines.append(f"| {t:.1f} | " + " | ".join(f"{g.fp_per_neuron[c]:.1f} | {g.recall_reachable[c]:.2f}" for c in order) + " |")
    lines.append("")
    lines.append("Recall over reachable merge sites, those within the site radius of a branch point ("
                 + ", ".join(f"{c} {summary['configs'][c]['n_gt_sites_kept']} of {summary['configs'][c]['n_gt_sites_all']}" for c in order)
                 + f"); false positives are totals divided by {int(C.n_neurons.iloc[0])} ground-truth neurons.")
    open(f"{args.out}/table.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

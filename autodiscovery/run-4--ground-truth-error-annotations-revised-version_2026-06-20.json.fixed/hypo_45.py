# CORRECTED TEST for hypothesis id 45 (omit errors ~2.4x more likely at extreme
# Z-depths than central depths).
#
# FAULTS (verifier MAJOR): (1) the CMH "controlling for brain" is illusory on the
# origin -- there is only ONE brain in the pkl, so the stratified test reduces to
# a single 2x2 with no actual confounder control; reporting it as CMH overstates
# what was done. (2) NODE NON-INDEPENDENCE -- adjacent nodes share edges and run
# along the same neurite, so the test's effective n is far smaller than the raw
# node count, inflating significance. (3) "Extreme Z" is per-volume min-max
# normalization, conflating optical depth with FOV-boundary truncation (a
# construct caveat we cannot remove, but we drop the causal language).
#
# CORRECTION: keep the SAME Extreme-vs-Center x omit quantities but report the
# odds ratio WITH a proper 95% CI as the effect size, and replace the
# node-independent CMH p with a NEURON-CLUSTER permutation p-value (permute the
# Extreme/Center label at the NEURON level -- the independent unit -- and recompute
# the omit-rate odds ratio). When >1 brain is present we still stratify by brain.
#
# ORIGINAL RECORDED NUMBERS (for the driver to compare):
#   Extreme 4.44% (2283/51369) vs Center 1.94% (11342/585909), CMH OR=2.3561, p~0.

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

for pkg in ["psutil", "tensorstore", "statsmodels"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import pickle
import glob
import os
import re
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import StratifiedTable, Table2x2

def brain_id_from_path(path):
    m = re.search(r"dataset_cache_(\d+)_mcl(\d+)_add\.pkl$", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)

data_paths = sorted(glob.glob("../data/dataset_cache_*_add.pkl"))

records = []

for path in data_paths:
    brain_id = brain_id_from_path(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)

    gt = payload["gt_graph"]
    node_label = np.asarray(payload["gt_node_canonical_label"])

    nodes = list(gt.nodes)
    if not nodes:
        continue

    z_coords = []
    for n in nodes:
        if hasattr(gt, "node_xyz"):
            z_coords.append(gt.node_xyz[n][2])
        elif "node_xyz" in gt.nodes[n]:
            z_coords.append(gt.nodes[n]["node_xyz"][2])
        else:
            raise ValueError(f"Could not find node_xyz for node {n}")

    z_coords = np.array(z_coords)
    z_min = z_coords.min()
    z_max = z_coords.max()
    z_range = z_max - z_min
    if z_range == 0:
        z_norm = np.zeros_like(z_coords)
    else:
        z_norm = (z_coords - z_min) / z_range

    for i, n in enumerate(nodes):
        z = z_norm[i]
        if z <= 0.1 or z >= 0.9:
            bin_label = "Extreme"
        elif 0.4 <= z <= 0.6:
            bin_label = "Center"
        else:
            bin_label = "Other"

        is_omit = 1 if node_label[n] == 0 else 0

        records.append({
            "brain_id": brain_id,
            "bin": bin_label,
            "is_omit": is_omit,
            "neuron_id": gt.node_segment_id(n),  # cluster / independent unit
        })

df = pd.DataFrame(records)

# Omit-rate summary (unchanged).
print("=== Omit Rates by Z-Depth Bin ===")
summary = df.groupby("bin")["is_omit"].agg(["sum", "count", "mean"])
summary.rename(columns={"sum": "Omit Count", "count": "Total Nodes", "mean": "Omit Rate"}, inplace=True)
summary_disp = summary.copy()
summary_disp["Omit Rate"] = summary_disp["Omit Rate"].apply(lambda x: f"{x*100:.2f}%")
summary_disp = summary_disp.reindex(["Extreme", "Center", "Other"])
print(summary_disp.to_string())

df_test = df[df["bin"].isin(["Extreme", "Center"])].copy()
n_brains = df_test["brain_id"].nunique()

# === EFFECT SIZE: pooled 2x2 OR with a proper 95% CI ===
ext = df_test[df_test["bin"] == "Extreme"]["is_omit"]
cen = df_test[df_test["bin"] == "Center"]["is_omit"]
a = int(ext.sum()); b = int((1 - ext).sum())
c = int(cen.sum()); d = int((1 - cen).sum())
tbl = Table2x2(np.array([[a, b], [c, d]]))
or_point = tbl.oddsratio
or_lo, or_hi = tbl.oddsratio_confint()
print("\n=== Effect size (Extreme vs Center, single 2x2) ===")
print(f"Number of brains (strata)        : {n_brains}"
      + ("  [single stratum -> CMH gives no real confounder control]" if n_brains == 1 else ""))
print(f"Odds ratio (Extreme vs Center)   : {or_point:.4f}")
print(f"OR 95% CI (Woolf)                : [{or_lo:.4f}, {or_hi:.4f}]")

# If more than one brain, ALSO report the real CMH (kept honest).
if n_brains > 1:
    ct = pd.crosstab(index=[df_test["brain_id"], df_test["bin"]], columns=df_test["is_omit"])
    tables = []
    for bid in df_test["brain_id"].unique():
        sub = ct.loc[bid]
        e0 = sub.loc["Extreme", 0] if ("Extreme" in sub.index and 0 in sub.columns) else 0
        e1 = sub.loc["Extreme", 1] if ("Extreme" in sub.index and 1 in sub.columns) else 0
        c0 = sub.loc["Center", 0] if ("Center" in sub.index and 0 in sub.columns) else 0
        c1 = sub.loc["Center", 1] if ("Center" in sub.index and 1 in sub.columns) else 0
        tables.append(np.array([[e1, e0], [c1, c0]]))
    st = StratifiedTable(np.dstack(tables))
    print(f"True CMH pooled OR ({n_brains} strata): {st.oddsratio_pooled:.4f}, "
          f"p={st.test_null_odds().pvalue:.4e}")

# === NEURON-CLUSTER permutation test (neuron = independent unit) ===
# Permute the Extreme/Center bin label across NEURONS (block permutation): each
# neuron keeps its omit values but its bin assignment is shuffled at the neuron
# level, so within-neuron correlation cannot inflate the test.
print("\n=== Neuron-cluster permutation test (neuron = independent unit) ===")
rng = np.random.default_rng(42)
# Per-neuron dominant bin and omit values.
grp = df_test.groupby("neuron_id")
neuron_bins = grp["bin"].agg(lambda s: s.value_counts().idxmax())
neuron_units = []
for nid, g in grp:
    neuron_units.append((neuron_bins[nid], g["is_omit"].values))

def or_from_assignment(assign_bins):
    ext_vals = np.concatenate([vals for bn, (b0, vals) in zip(assign_bins, neuron_units) if bn == "Extreme"]) \
        if any(bn == "Extreme" for bn in assign_bins) else np.array([])
    cen_vals = np.concatenate([vals for bn, (b0, vals) in zip(assign_bins, neuron_units) if bn == "Center"]) \
        if any(bn == "Center" for bn in assign_bins) else np.array([])
    if len(ext_vals) == 0 or len(cen_vals) == 0:
        return np.nan
    aa = ext_vals.sum() + 0.5; bb = (1 - ext_vals).sum() + 0.5
    cc = cen_vals.sum() + 0.5; dd = (1 - cen_vals).sum() + 0.5
    return (aa * dd) / (bb * cc)

obs_bins = [b for b, _ in neuron_units]
obs_or = or_from_assignment(obs_bins)
n_perm = 5000
perm_ors = np.empty(n_perm)
bins_arr = np.array(obs_bins, dtype=object)
for i in range(n_perm):
    perm_ors[i] = or_from_assignment(list(rng.permutation(bins_arr)))
# Two-sided p on log-OR (effect = Extreme higher => OR>1).
log_obs = np.log(obs_or)
log_perm = np.log(perm_ors)
perm_p = (np.sum(np.abs(log_perm) >= abs(log_obs)) + 1) / (n_perm + 1)
print(f"  Neuron units = {len(neuron_units)}")
print(f"  Observed neuron-clustered OR (Extreme vs Center) = {obs_or:.4f}")
print(f"  Cluster-permutation two-sided p = {perm_p:.4g}")
print("  (Compare to ORIGINAL: CMH OR=2.3561, p~0 treating all nodes as independent.)")
print("\nNote: 'Extreme Z' is per-volume min-max normalized, so it cannot separate")
print("optical depth from FOV-boundary truncation; no causal attribution is made.")

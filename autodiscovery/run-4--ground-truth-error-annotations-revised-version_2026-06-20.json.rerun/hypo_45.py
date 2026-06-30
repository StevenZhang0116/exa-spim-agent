import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Install missing dependencies gracefully
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
from statsmodels.stats.contingency_tables import StratifiedTable

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
    
    # Extract Z-coordinates (assuming node_xyz is available either as an attribute or in node data)
    z_coords = []
    for n in nodes:
        if hasattr(gt, "node_xyz"):
            z_coords.append(gt.node_xyz[n][2])
        elif "node_xyz" in gt.nodes[n]:
            z_coords.append(gt.nodes[n]["node_xyz"][2])
        else:
            raise ValueError(f"Could not find node_xyz for node {n}")
            
    z_coords = np.array(z_coords)
    
    # Min-max normalize per brain
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
            "is_omit": is_omit
        })

df = pd.DataFrame(records)

# Prepare data for Cochran-Mantel-Haenszel test (Extreme vs Center)
df_test = df[df["bin"].isin(["Extreme", "Center"])]
ct = pd.crosstab(index=[df_test["brain_id"], df_test["bin"]], columns=df_test["is_omit"])

brain_ids = df_test["brain_id"].unique()
tables = []

for bid in brain_ids:
    if bid in ct.index.get_level_values("brain_id"):
        sub = ct.loc[bid]
        extreme_0 = sub.loc["Extreme", 0] if "Extreme" in sub.index and 0 in sub.columns else 0
        extreme_1 = sub.loc["Extreme", 1] if "Extreme" in sub.index and 1 in sub.columns else 0
        center_0 = sub.loc["Center", 0] if "Center" in sub.index and 0 in sub.columns else 0
        center_1 = sub.loc["Center", 1] if "Center" in sub.index and 1 in sub.columns else 0
        
        # Construct 2x2 table:
        # Rows: Extreme, Center
        # Cols: Omit (1), Not Omit (0)
        table = np.array([
            [extreme_1, extreme_0],
            [center_1, center_0]
        ])
        tables.append(table)

if tables:
    tables_array = np.dstack(tables)
    st = StratifiedTable(tables_array)

    # Generate summary statistics
    print("=== Omit Rates by Z-Depth Bin ===")
    summary = df.groupby("bin")["is_omit"].agg(["sum", "count", "mean"])
    summary.rename(columns={"sum": "Omit Count", "count": "Total Nodes", "mean": "Omit Rate"}, inplace=True)
    summary["Omit Rate"] = summary["Omit Rate"].apply(lambda x: f"{x*100:.2f}%")
    summary = summary.reindex(["Extreme", "Center", "Other"])
    print(summary.to_string())

    print("\n=== Cochran-Mantel-Haenszel Test (Extreme vs Center) ===")
    print(f"Pooled Odds Ratio (Extreme vs Center): {st.oddsratio_pooled:.4f}")
    print(f"p-value: {st.test_null_odds().pvalue:.4e}")
else:
    print("Not enough data to perform CMH test.")
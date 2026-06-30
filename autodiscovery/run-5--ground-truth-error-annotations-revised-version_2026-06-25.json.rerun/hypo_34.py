import sys
import subprocess
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

try:
    import psutil
except ImportError:
    install("psutil")

try:
    import tensorstore
except ImportError:
    install("tensorstore")

try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

try:
    import pandas as pd
except ImportError:
    install("pandas")
    import pandas as pd

try:
    from sklearn.metrics import roc_curve, auc
except ImportError:
    install("scikit-learn")
    from sklearn.metrics import roc_curve, auc

dataset_paths = glob.glob("../data/dataset_cache_*_add.pkl") + \
                glob.glob("../dataset_cache_*_add.pkl") + \
                glob.glob("data/dataset_cache_*_add.pkl") + \
                glob.glob("dataset_cache_*_add.pkl") + \
                glob.glob("../*/dataset_cache_*_add.pkl")

if not dataset_paths:
    print("Dataset not found.")
    sys.exit(1)
dataset_path = dataset_paths[0]

print(f"Loading dataset from {dataset_path}...")
with open(dataset_path, "rb") as f:
    payload = pickle.load(f)

fg = payload["fragments_graph"]

# Step 1 & 2: Calculate the total Euclidean cable length of each segment
edges = list(fg.edges)
if edges:
    edges_arr = np.array(edges)
    u = edges_arr[:, 0]
    v = edges_arr[:, 1]
    
    # Compute edge lengths
    dists = np.linalg.norm(fg.node_xyz[u] - fg.node_xyz[v], axis=1)
    
    # Map edges to segment IDs
    comp_u = fg.node_component_id[u]
    if isinstance(fg.component_id_to_swc_id, dict):
        seg_u = [fg.component_id_to_swc_id.get(c, c) for c in comp_u]
    else:
        seg_u = fg.component_id_to_swc_id[comp_u]
        
    # Sum lengths per segment ID, handling string-to-int conversion safely
    df = pd.DataFrame({'seg_id': seg_u, 'dist': dists})
    
    def to_int(x):
        try:
            return int(float(x))
        except (ValueError, TypeError):
            return None
            
    df['seg_id'] = df['seg_id'].apply(to_int)
    df = df.dropna(subset=['seg_id'])
    df['seg_id'] = df['seg_id'].astype(int)
    
    seg_lengths = df.groupby('seg_id')['dist'].sum().to_dict()
else:
    seg_lengths = {}

# Step 3: Divide segments into 'Merge-causing' and 'Correct' groups
node_label = np.asarray(payload["gt_node_canonical_label"])
valid_labels = set(int(x) for x in node_label if x != 0)
merge_labels = set(int(x) for x in payload["gt_merge_labels"])

correct_labels = valid_labels - merge_labels

# Extract lengths for merge vs correct segments (filtering out un-reconstructed/dropped short fragments)
merge_lengths = [seg_lengths[seg] for seg in merge_labels if seg in seg_lengths]
correct_lengths = [seg_lengths[seg] for seg in correct_labels if seg in seg_lengths]

print(f"Number of Merge-causing segments: {len(merge_lengths)}")
print(f"Number of Correct (non-merge) segments: {len(correct_lengths)}\n")

avg_merge = np.mean(merge_lengths) if merge_lengths else 0
med_merge = np.median(merge_lengths) if merge_lengths else 0
avg_corr = np.mean(correct_lengths) if correct_lengths else 0
med_corr = np.median(correct_lengths) if correct_lengths else 0

print(f"Merge segments - Average length: {avg_merge:.2f} \u00b5m, Median length: {med_merge:.2f} \u00b5m")
print(f"Correct segments - Average length: {avg_corr:.2f} \u00b5m, Median length: {med_corr:.2f} \u00b5m")

if avg_corr > 0:
    print(f"Ratio of average lengths (Merge / Correct): {avg_merge / avg_corr:.2f}")

# Step 4: Compare distributions and Generate ROC
if merge_lengths and correct_lengths:
    y_true = [1] * len(merge_lengths) + [0] * len(correct_lengths)
    y_scores = merge_lengths + correct_lengths

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Segment Cable Length for Predicting Merges')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    c_len_log = [x for x in correct_lengths if x > 0]
    m_len_log = [x for x in merge_lengths if x > 0]
    plt.boxplot([c_len_log, m_len_log], labels=['Correct (1-to-1)', 'Merge-causing'])
    plt.yscale('log')
    plt.ylabel('Cable Length (\u00b5m) [Log Scale]')
    plt.title('Cable Length Distribution of Segments')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()
else:
    print("Not enough data to plot ROC curve and boxplots.")

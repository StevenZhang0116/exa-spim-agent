#!/usr/bin/env python3
"""PNG of recall and false positives vs threshold from slide_data.json (same content as the slide)."""
import json, sys, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = sys.argv[1] if len(sys.argv) > 1 else "/root/capsule/results/residual_merge_detection"
D = json.load(open(f"{out}/slide_data.json")); T = D["thresholds"]
COL = {"it2": "#2a78d6", "dt5k": "#1baf7a", "dt20k": "#eda100", "dt50k": "#eb6834"}
NAME = {"it2": "base (it2)", "dt5k": "agg mt08_st08_dt5k", "dt20k": "agg mt08_st08_dt20k", "dt50k": "agg mt08_st08_dt50k"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
for cfg, d in D["configs"].items():
    c = COL.get(cfg, "#555"); n = NAME.get(cfg, cfg)
    ax[0].plot(T, d["recall"], color=c, lw=2, label=n)
    ax[1].plot(T, d["fp"], color=c, lw=2, label=n)
M = D.get("mark_threshold", 0.1); mi = min(range(len(T)), key=lambda i: abs(T[i] - M))
ax[0].set_ylim(0, 1.02 if max(max(d["recall"]) for d in D["configs"].values()) > 0.5 else 0.5); ax[1].set_ylim(0, None)
for a, key, fmt in zip(ax, ["recall", "fp"], ["{:.2f}", "{:.1f}"]):
    a.axvline(M, color="#4e5661", lw=1, ls=(0, (4, 4)))
    pts = sorted(((d[key][mi], cfg) for cfg, d in D["configs"].items()), reverse=True)
    disp = [a.transData.transform((M, v))[1] / fig.dpi * 72 for v, _ in pts]   # y in points, top first
    tgt = list(disp); top = a.transData.transform((M, a.get_ylim()[1]))[1] / fig.dpi * 72 - 5
    tgt[0] = min(tgt[0], top)
    for i in range(1, len(tgt)): tgt[i] = min(tgt[i], tgt[i - 1] - 10.5)
    for (v, cfg), y0, y1 in zip(pts, disp, tgt):
        a.plot(M, v, "o", color=COL.get(cfg, "#555"), ms=5, mec="white", mew=1)
        a.annotate(fmt.format(v), (M, v), xytext=(8, y1 - y0), textcoords="offset points", fontsize=8.5, va="center", color="#15181c", family="monospace")
    lo = a.get_ylim()[0]; near_bottom = min(disp) - a.transData.transform((M, lo))[1] / fig.dpi * 72 < 24
    a.annotate(f"t = {M:g}", (M, a.get_ylim()[1] if near_bottom else lo), xytext=(6, -10 if near_bottom else 8), textcoords="offset points", fontsize=8.5, color="#4e5661", family="monospace")
for a, t in zip(ax, ["Recall (reachable merge sites)", "False positives per GT neuron"]):
    a.set_title(t, loc="left", fontsize=11); a.set_xlabel("prediction threshold"); a.set_xlim(0, 1); a.grid(alpha=.25)
    [a.spines[s].set_visible(False) for s in ("top", "right")]
ax[0].legend(frameon=False, fontsize=9); ax[1].legend(frameon=False, fontsize=9)
p = D["params"]; g = D["configs"]
fig.text(0.01, 0.005, f"Sites within {p['gt_filter_um']:g} µm of GT; GT = merge sites within {p['site_radius_um']:g} µm of a branch point (" + ", ".join(f"{k} {v['n_gt']} of {v['n_gt_all']}" for k, v in g.items())
         + f"); match {p['match_um']:g} µm; FP = total / {D['n_neurons']} neurons.", fontsize=7.5, color="#555")
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f"{out}/recall_fp_vs_threshold.png"); print("wrote", f"{out}/recall_fp_vs_threshold.png")

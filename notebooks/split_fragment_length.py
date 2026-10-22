#!/usr/bin/env python3
"""Fragment cable length of split candidate pairs, per brain.

For every row of the declared split candidate pool (``proofreader_evolve/candidate_pools.json``,
``tip_to_any_node|r=80|k=2``, enumerated by ``candidate_rules.py`` through the same pool-only path
as the companions, ``export_rerank_companion.pool_only_bank``) this script records the cable
length of both fragments (segments) of the pair, the row's three-valued label (1 split,
0 GT-judged non-split, NaN not judgeable), its gap and whether the partner node is a tip. The
cable length of every fragment of the brain is kept as the background distribution. A fragment
is a segment id; a segment that the cache holds as several connected components (the swc id
suffix after the first ``.``) is summed, and the number of such segments is reported.

Three steps:

1. ``--measure`` (compute node, panda env, one cache per brain; the split enumeration of the
   largest brains takes 15-25 min and needs ~40 GB):
       srun --jobid=<job> --overlap -n1 -c4 bash -lc 'cd exa-spim-agent && \
           /home/zihan.zhang/.conda/envs/panda/bin/python -u notebooks/split_fragment_length.py \
           --measure --brains 794495 802449'
2. ``--plot`` (seconds, reads the per-brain files): ECDFs of the shorter and of the longer
   fragment of the pair per label class against the background, the AUC of fragment length
   for label 1 versus 0 pooled and within gap tertiles (the specifications' matched-gap rule),
   ``summary.csv`` and ``auc.csv``.
3. ``--canonical`` (brains with a split companion in ``markdowns/``): the canonical measurement
   (``agentic/canonical_stats.py``) of per-fold logistic fits of fragment length alone, of gap
   alone (the refit control) and of gap plus both lengths, against the raw gap ranking, with the
   paired bootstrap and the tie-breaking null.

Outputs, all under ``notebooks/split_fragment_length/``: ``<brain>_mcl<N>.npz`` and ``.json``,
``summary.csv``, ``auc.csv``, ``split_fragment_length_min.svg``, ``split_fragment_length_max.svg``,
``split_fragment_length_auc.svg``, ``canonical_<brain>.json``. Read-only with respect to the
caches, companions and tables.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agentic"))

ALL_BRAINS = ["794495", "802449", "789202", "794493", "794491", "747807", "750318",
              "751473", "754610", "754612", "754613"]
OUT_DIR = ROOT / "notebooks" / "split_fragment_length"
COMPANION_DIR = ROOT / "markdowns"
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
CLASSES = ("1", "0", "nan")
# Same slots as notebooks/plot_label_availability.py: 1 = orange, 0 = blue, NaN = recessive gray.
COLORS = {"1": "#eb6834", "0": "#2a78d6", "nan": "#c9c8c2", "background": "#52514e"}
CLASS_TEXT = {"1": "1  split", "0": "0  GT-judged, no split", "nan": "NaN  not judgeable",
              "background": "all fragments of the brain"}
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ----------------------------------------------------------------------------- measure
def segment_lengths(graph):
    """``(segment_ids, cable_um, n_components)`` of every fragment segment of the graph.

    Edge lengths are summed per connected component with a remapped (``np.unique``) index,
    never an array indexed by raw segment ids, then per segment id.
    """
    import candidate_rules as rules

    n_nodes = int(graph.number_of_nodes())
    xyz = np.asarray(graph.node_xyz, dtype=np.float64)
    components = np.asarray(graph.node_component_id, dtype=np.int64)
    if xyz.shape[0] != n_nodes or components.shape[0] != n_nodes:
        raise ValueError("node_xyz and node_component_id must align with contiguous node ids")
    if n_nodes and (not graph.has_node(0) or not graph.has_node(n_nodes - 1)):
        raise ValueError("fragment graph node ids must be 0..N-1")
    n_edges = int(graph.number_of_edges())
    edges = np.fromiter(itertools.chain.from_iterable(graph.edges()), dtype=np.int64,
                        count=2 * n_edges).reshape(n_edges, 2)
    if (components[edges[:, 0]] != components[edges[:, 1]]).any():
        raise ValueError("an edge crosses connected components")
    edge_um = np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1)
    component_ids, component_index = np.unique(components, return_inverse=True)
    to_segment = rules.segment_of_component(graph)
    component_segment = np.fromiter((int(to_segment.get(int(c), -1)) for c in component_ids),
                                    dtype=np.int64, count=len(component_ids))
    component_um = np.bincount(component_index[edges[:, 0]], weights=edge_um,
                               minlength=len(component_ids))
    segment_ids, segment_index = np.unique(component_segment, return_inverse=True)
    segment_um = np.bincount(segment_index, weights=component_um, minlength=len(segment_ids))
    n_components = np.bincount(segment_index, minlength=len(segment_ids))
    keep = segment_ids > 0
    return segment_ids[keep], segment_um[keep], n_components[keep]


def lookup(segment_ids, values, query):
    """``values`` of ``query`` segment ids; every query must be a known segment."""
    position = np.searchsorted(segment_ids, query)
    position = np.clip(position, 0, len(segment_ids) - 1)
    if not np.array_equal(segment_ids[position], query):
        missing = np.unique(query[segment_ids[position] != query])
        raise KeyError(f"{len(missing)} candidate segments are not in the fragment graph: {missing[:5]}")
    return values[position]


def measure_brain(brain: str, mcl: int) -> tuple[dict, dict]:
    from proofreader_evolve.cli import export_rerank_companion as ex
    from proofreader_evolve.harness import label_availability as la

    started = time.monotonic()
    bank, graph = ex.pool_only_bank(brain, mcl, ["split"])
    table = bank.tables["split"]
    rows = table.candidates
    labels = np.asarray(table.truth, dtype=np.float64)
    if len(rows) != len(labels):
        raise ValueError("rows and labels are not aligned")
    segment_ids, segment_um, n_components = segment_lengths(graph)
    seg_a = np.fromiter((int(r["segment_id_a"]) for r in rows), dtype=np.int64, count=len(rows))
    seg_b = np.fromiter((int(r["segment_id_b"]) for r in rows), dtype=np.int64, count=len(rows))
    gap = np.fromiter((float(r["gap_um"]) for r in rows), dtype=np.float64, count=len(rows))
    partner_tip = np.fromiter((r["node_role_a"] == "tip" and r["node_role_b"] == "tip" for r in rows),
                              dtype=bool, count=len(rows))
    arrays = {
        "label": labels.astype(np.float32),
        "gap_um": gap.astype(np.float32),
        "segment_id_a": seg_a, "segment_id_b": seg_b,
        "len_a": lookup(segment_ids, segment_um, seg_a).astype(np.float32),
        "len_b": lookup(segment_ids, segment_um, seg_b).astype(np.float32),
        "partner_tip": partner_tip,
        "background_segment_id": segment_ids,
        "background_len": segment_um.astype(np.float32),
        "background_components": n_components.astype(np.int32),
    }
    counts = la.label_counts(table.truth)
    shorter = np.minimum(arrays["len_a"], arrays["len_b"])
    longer = np.maximum(arrays["len_a"], arrays["len_b"])
    summary = {
        "brain": brain, "mcl": mcl,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_cache": table.meta["provenance"]["source_cache"],
        "candidate_policy": table.meta.get("candidate_policy"),
        "candidate_policy_source": table.meta.get("candidate_policy_source"),
        "pool_sha256": table.meta["pool_sha256"],
        "labels_sha256": la.labels_digest(table.truth),
        "rows": int(len(rows)), "label_counts": counts,
        "n_segments": int(len(segment_ids)),
        "n_segments_in_pool": int(len(np.unique(np.concatenate([seg_a, seg_b])))),
        "segments_with_several_components": int((n_components > 1).sum()),
        "background_len_um": quantile_record(segment_um),
        "shorter_fragment_um": {c: quantile_record(shorter[class_mask(labels, c)]) for c in CLASSES},
        "longer_fragment_um": {c: quantile_record(longer[class_mask(labels, c)]) for c in CLASSES},
        "partner_tip_share": {c: float(partner_tip[class_mask(labels, c)].mean())
                              for c in CLASSES if class_mask(labels, c).any()},
        "seconds": round(time.monotonic() - started, 1),
    }
    del bank, graph, table, rows
    gc.collect()
    return arrays, summary


def class_mask(labels, cls):
    labels = np.asarray(labels, dtype=float)
    if cls == "nan":
        return np.isnan(labels)
    if cls == "background":
        return np.ones(len(labels), dtype=bool)
    return labels == float(cls)


def quantile_record(values) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    record = {"n": int(len(values)), "min": float(values.min()), "max": float(values.max())}
    record.update({f"q{int(q * 100)}": float(np.quantile(values, q)) for q in QUANTILES})
    return record


def brain_paths(brain, mcl, out_dir=OUT_DIR):
    return out_dir / f"{brain}_mcl{mcl}.npz", out_dir / f"{brain}_mcl{mcl}.json"


def run_measure(brains, mcl, out_dir, force):
    out_dir.mkdir(parents=True, exist_ok=True)
    for brain in brains:
        npz_path, json_path = brain_paths(brain, mcl, out_dir)
        if npz_path.exists() and json_path.exists() and not force:
            print(f"{brain}: kept {json_path.name} (use --force to recompute)", flush=True)
            continue
        print(f"{brain}: measuring", flush=True)
        arrays, summary = measure_brain(brain, mcl)
        np.savez_compressed(npz_path, **arrays)
        json_path.write_text(json.dumps(summary, indent=1) + "\n")
        short = summary["shorter_fragment_um"]
        print(f"{brain}: {summary['rows']} rows, labels {summary['label_counts']}, "
              f"{summary['n_segments']} fragments, median shorter fragment "
              f"1={short['1'].get('q50', float('nan')):.0f} 0={short['0'].get('q50', float('nan')):.0f} "
              f"nan={short['nan'].get('q50', float('nan')):.0f} um, "
              f"{summary['seconds']} s", flush=True)


# ----------------------------------------------------------------------------- plot
def load_brain(brain, mcl, out_dir=OUT_DIR):
    npz_path, json_path = brain_paths(brain, mcl, out_dir)
    if not (npz_path.exists() and json_path.exists()):
        return None
    with np.load(npz_path, allow_pickle=False) as npz:
        arrays = {name: npz[name] for name in npz.files}
    return arrays, json.loads(json_path.read_text())


def auc_of(values, labels) -> tuple[float, int, int]:
    """Probability that a label-1 row has a larger value than a label-0 row (ties half)."""
    from scipy.stats import mannwhitneyu

    labels = np.asarray(labels, dtype=float)
    pos, neg = values[labels == 1], values[labels == 0]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan"), int(len(pos)), int(len(neg))
    u = mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg))), int(len(pos)), int(len(neg))


def auc_rows(brain, arrays) -> list[dict]:
    """AUC of each length measure (shorter first) for 1 vs 0, pooled and within gap tertiles."""
    labels = arrays["label"].astype(float)
    labeled = ~np.isnan(labels)
    gap = arrays["gap_um"].astype(float)
    shorter = np.minimum(arrays["len_a"], arrays["len_b"]).astype(float)
    longer = np.maximum(arrays["len_a"], arrays["len_b"]).astype(float)
    measures = {"shorter_fragment": -shorter, "longer_fragment": -longer, "gap": -gap}
    edges = np.quantile(gap[labeled], [1 / 3, 2 / 3])
    strata = {"pooled": labeled,
              "gap_tertile_1": labeled & (gap <= edges[0]),
              "gap_tertile_2": labeled & (gap > edges[0]) & (gap <= edges[1]),
              "gap_tertile_3": labeled & (gap > edges[1])}
    out = []
    for measure, values in measures.items():
        for stratum, mask in strata.items():
            auc, n1, n0 = auc_of(values[mask], labels[mask])
            out.append({"brain": brain, "measure": measure, "stratum": stratum,
                        "gap_range_um": (f"<= {edges[0]:.1f}" if stratum == "gap_tertile_1" else
                                         f"{edges[0]:.1f}-{edges[1]:.1f}" if stratum == "gap_tertile_2" else
                                         f"> {edges[1]:.1f}" if stratum == "gap_tertile_3" else "all"),
                        "n_1": n1, "n_0": n0, "auc": auc})
    return out


def ecdf(ax, values, color, label, linestyle="-", linewidth=1.6):
    values = np.sort(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    if not len(values):
        return
    y = np.arange(1, len(values) + 1) / len(values)
    ax.step(values, y, where="post", color=color, lw=linewidth, ls=linestyle, label=label)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=3)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def plot_ecdfs(data, which, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    brains = list(data)
    n_cols = 4
    n_rows = int(np.ceil(len(brains) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.7 * n_rows + 0.9),
                             sharex=True, sharey=True, squeeze=False)
    for ax in axes.ravel()[len(brains):]:
        ax.axis("off")
    for ax, brain in zip(axes.ravel(), brains):
        arrays, summary = data[brain]
        labels = arrays["label"].astype(float)
        pick = np.minimum if which == "min" else np.maximum
        values = pick(arrays["len_a"], arrays["len_b"]).astype(float)
        ecdf(ax, arrays["background_len"], COLORS["background"], CLASS_TEXT["background"], ":", 1.3)
        for cls in ("nan", "0", "1"):
            mask = class_mask(labels, cls)
            ecdf(ax, values[mask], COLORS[cls], CLASS_TEXT[cls])
        counts = summary["label_counts"]
        ax.set_title(f"{brain}   1: {counts['1']:,}   0: {counts['0']:,}   NaN: {counts['nan']:,}",
                     fontsize=8.5, color=TEXT_PRIMARY, loc="left")
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("cable length (um)", fontsize=8.5, color=TEXT_SECONDARY)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of rows", fontsize=8.5, color=TEXT_SECONDARY)
    handles, labels_ = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0))
    side = "shorter" if which == "min" else "longer"
    fig.suptitle(f"Split candidate pairs: cable length of the {side} fragment of the pair, per brain "
                 f"(ECDF; declared pool tip_to_any_node|r=80|k=2)", fontsize=10, color=TEXT_PRIMARY, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def plot_auc(auc_table, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    table = pd.DataFrame(auc_table)
    brains = list(dict.fromkeys(table["brain"]))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.42 * len(brains) + 1.8), sharey=True)
    markers = {"gap_tertile_1": "v", "gap_tertile_2": "s", "gap_tertile_3": "^"}
    names = {"gap_tertile_1": "shortest-gap tertile", "gap_tertile_2": "middle tertile",
             "gap_tertile_3": "longest-gap tertile"}
    for ax, measure in zip(axes, ("shorter_fragment", "longer_fragment")):
        sub = table[table["measure"] == measure]
        ys = {b: i for i, b in enumerate(brains)}
        for stratum, marker in markers.items():
            part = sub[sub["stratum"] == stratum]
            ax.scatter(part["auc"], [ys[b] for b in part["brain"]], marker=marker, s=28,
                       facecolors="none", edgecolors=COLORS["0"], lw=1.2, label=names[stratum], zorder=3)
        pooled = sub[sub["stratum"] == "pooled"]
        ax.scatter(pooled["auc"], [ys[b] for b in pooled["brain"]], marker="o", s=42,
                   color=COLORS["1"], label="pooled, all labeled rows", zorder=4)
        gap_pooled = table[(table["measure"] == "gap") & (table["stratum"] == "pooled")]
        ax.scatter(gap_pooled["auc"], [ys[b] for b in gap_pooled["brain"]], marker="x", s=36,
                   color=COLORS["background"], lw=1.2, label="gap itself, pooled (reference)", zorder=4)
        ax.axvline(0.5, color=GRID, lw=1.0, zorder=1)
        ax.set_yticks(range(len(brains)))
        ax.set_yticklabels(brains, fontsize=8.5, color=TEXT_PRIMARY)
        ax.invert_yaxis()
        ax.set_xlim(0.3, 1.0)
        ax.set_xlabel("AUC: label-1 rows rank above label-0 rows (shorter length first)",
                      fontsize=8.5, color=TEXT_SECONDARY)
        ax.set_title("shorter fragment of the pair" if measure == "shorter_fragment"
                     else "longer fragment of the pair", fontsize=9.5, color=TEXT_PRIMARY, loc="left")
        style(ax)
        ax.grid(True, axis="x", color=GRID, lw=0.6)
        ax.grid(False, axis="y")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle("Fragment length as a split ranking: pooled and within gap tertiles of the labeled rows",
                 fontsize=10, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)


def run_plot(brains, mcl, out_dir):
    import pandas as pd

    data = {}
    for brain in brains:
        loaded = load_brain(brain, mcl, out_dir)
        if loaded is not None:
            data[brain] = loaded
    if not data:
        raise FileNotFoundError(f"no per-brain files under {out_dir}; run --measure first")
    summary_rows, auc_table = [], []
    for brain, (arrays, summary) in data.items():
        for which, key in (("shorter", "shorter_fragment_um"), ("longer", "longer_fragment_um")):
            for cls in CLASSES:
                record = summary[key][cls]
                summary_rows.append({"brain": brain, "fragment": which, "class": cls, **record})
        summary_rows.append({"brain": brain, "fragment": "background", "class": "background",
                             **summary["background_len_um"]})
        auc_table.extend(auc_rows(brain, arrays))
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(auc_table).to_csv(out_dir / "auc.csv", index=False)
    plot_ecdfs(data, "min", out_dir / "split_fragment_length_min.svg")
    plot_ecdfs(data, "max", out_dir / "split_fragment_length_max.svg")
    plot_auc(auc_table, out_dir / "split_fragment_length_auc.svg")
    table = pd.DataFrame(auc_table)
    print(f"{len(data)} brains: {', '.join(data)}")
    print("AUC of the shorter fragment's length (shorter first), 1 vs 0:")
    pivot = table[table["measure"] == "shorter_fragment"].pivot(index="brain", columns="stratum", values="auc")
    gap_auc = table[(table["measure"] == "gap") & (table["stratum"] == "pooled")].set_index("brain")["auc"]
    pivot["gap_pooled"] = gap_auc
    print(pivot[["pooled", "gap_tertile_1", "gap_tertile_2", "gap_tertile_3", "gap_pooled"]]
          .round(3).to_string())
    print(f"wrote {out_dir / 'summary.csv'}, {out_dir / 'auc.csv'} and three SVGs under {out_dir}")


# ----------------------------------------------------------------------------- canonical
def fold_logistic(comp, X):
    """Per-fold logistic scores (decision function) of standardized columns ``X``."""
    import canonical_stats
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scores = np.full(len(comp["label"]), np.nan)
    for f in np.unique(comp["fold"]):
        fit = canonical_stats.fit_rows(comp, f)
        held = canonical_stats.held_rows(comp, f)
        scaler = StandardScaler().fit(X[fit])
        model = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(X[fit]), comp["label"][fit])
        scores[held] = model.decision_function(scaler.transform(X[held]))
    return scores


def run_canonical(brains, mcl, out_dir):
    import canonical_stats

    done = 0
    for brain in brains:
        companion = COMPANION_DIR / f"rerank_companion_{brain}_split.npz"
        manifest = COMPANION_DIR / f"rerank_companion_{brain}_split.json"
        loaded = load_brain(brain, mcl, out_dir)
        if not companion.exists() or loaded is None:
            continue
        arrays, summary = loaded
        meta = json.loads(manifest.read_text())
        if meta.get("pool_sha256") != summary["pool_sha256"]:
            print(f"{brain}: companion pool {meta.get('pool_sha256', '')[:12]} differs from the measured pool "
                  f"{summary['pool_sha256'][:12]}; skipped", flush=True)
            continue
        comp = canonical_stats.load_companion(companion)
        if not (np.array_equal(comp["segment_id_a"], arrays["segment_id_a"])
                and np.array_equal(comp["segment_id_b"], arrays["segment_id_b"])):
            raise ValueError(f"{brain}: companion rows are not the measured rows")
        gap = np.asarray(comp["gap_um"], dtype=float)
        shorter = np.log10(np.minimum(arrays["len_a"], arrays["len_b"]).astype(float))
        longer = np.log10(np.maximum(arrays["len_a"], arrays["len_b"]).astype(float))
        print(f"INPUTS READ brain={brain} companion_rows={len(gap)} labeled={int((~np.isnan(comp['label'])).sum())} "
              f"fragments={summary['n_segments']}", flush=True)
        variants = {
            "shorter fragment length alone (per-fold logistic)": np.column_stack([shorter]),
            "both fragment lengths (per-fold logistic)": np.column_stack([shorter, longer]),
            "REFIT CONTROL gap alone (per-fold logistic)": np.column_stack([gap]),
            "gap + both fragment lengths (per-fold logistic)": np.column_stack([gap, shorter, longer]),
        }
        results = {}
        for name, X in variants.items():
            results[name] = canonical_stats.report(comp, fold_logistic(comp, X), name=f"{brain}: {name}")
        print(f"VARIANTS COMPARED {len(variants)} (primary: gap + both fragment lengths)", flush=True)
        (out_dir / f"canonical_{brain}.json").write_text(json.dumps(
            {"brain": brain, "created": datetime.now().astimezone().isoformat(timespec="seconds"),
             "companion": str(companion), "pool_sha256": summary["pool_sha256"], "results": results},
            indent=1, default=float) + "\n")
        done += 1
    if not done:
        raise FileNotFoundError("no brain with both a companion and a measurement (run --measure first)")


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measure", action="store_true", help="enumerate the pool and record fragment lengths")
    parser.add_argument("--plot", action="store_true", help="figures, summary.csv and auc.csv from the per-brain files")
    parser.add_argument("--canonical", action="store_true",
                        help="canonical AP measurement on brains with a split companion")
    parser.add_argument("--brains", nargs="+", default=ALL_BRAINS)
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--force", action="store_true", help="recompute brains that already have files")
    args = parser.parse_args(argv)
    if not (args.measure or args.plot or args.canonical):
        parser.error("choose --measure, --plot and/or --canonical")
    if args.measure:
        run_measure(args.brains, args.mcl, args.out_dir, args.force)
    if args.plot:
        run_plot(args.brains, args.mcl, args.out_dir)
    if args.canonical:
        run_canonical(args.brains, args.mcl, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

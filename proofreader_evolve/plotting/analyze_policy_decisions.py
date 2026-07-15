"""
Compare policies BEHAVIORALLY across generations — as decision procedures, not prose.

The evolved policy is a decision tree in disguise: ``propose_edits`` is a per-site
guard chain (``if gap > GAP_THRESHOLD_UM: continue`` … ``if deg_b >= 3: continue`` …)
ending in accept/reject. So the faithful way to ask "how did the policy change between
generations" is to compare the DECISIONS themselves, not a sentence-embedding of the
``rules.md`` prose (which a paraphrase can move without any behavior change, and a big
code change can leave nearly unchanged). This module does that two ways:

  (1) BEHAVIORAL DECISION-AGREEMENT (exact; the real "how different are these policies").
      Enumerate ONE fixed set of candidate SplitSites, then run EVERY generation's
      ``propose_edits`` on that identical set and record which pairs each ACCEPTS
      (emits a ``merge_labels`` edit for). Similarity between two generations = Jaccard
      of their accepted-pair sets: |A_i ∩ A_j| / |A_i ∪ A_j|. This is exact, needs no
      model, and is invariant to code reordering / paraphrase / helper refactors —
      two policies that emit the same edits ARE the same policy operationally. Rendered
      as a gen×gen heatmap plus a consecutive / to-seed / to-final trajectory (the same
      layout as ``analyze_policy_similarity.py`` so the two are directly comparable).

  (2) SURROGATE DECISION TREE (interpretable; a literal tree per generation).
      Features are computed ONCE per candidate site (the deployable geometry the policy
      can see: ``gap_um``, ``colinear_cos``, ``deg_b``, ``rad_ratio``, ``mutual_nearest``,
      reciprocal ranks, cable lengths, …). Then for each generation we fit a shallow
      ``DecisionTreeClassifier`` on (features → the policy's accept/reject label on the
      SAME sites). This distills however-the-policy-is-written into a CANONICAL tree over
      one shared feature space, so you can read "what rule does gen N actually implement"
      and watch the driving features / thresholds drift. We report each surrogate's
      FIDELITY (train accuracy reproducing the policy) — a low fidelity means the policy
      keys on something outside the feature set (e.g. an image read), so trust the tree
      only where fidelity is high.

Why this complements ``analyze_policy_similarity.py`` rather than replacing it:
  * (1)/(2) require RUNNING the policies. Geometry-only is cheap and exact for
    geometry-gated policies; an IMAGE-gated policy needs cloud reads to reproduce
    faithfully (``--with-image``, costly) or is approximated geometry-only (flagged).
  * The embedding is read-only over saved ``rules.md`` and works for REJECTED candidates
    you may not want to re-run. Use it as a cheap prose proxy; use THIS for behavior.

FIXED CANDIDATE STREAM (important). All generations are compared on ONE candidate set
(by default the FINAL accepted generation's resolved ``ENUM_PARAMS``). This isolates the
ACCEPT/REJECT decision from the enumeration prior: a generation that would itself have
enumerated more/fewer sites is still asked to decide on this reference set, so the
comparison is "given the same candidates, do they decide alike?" Changes to the
enumeration prior (``max_gap_um`` etc.) are a SEPARATE axis, not captured here.

NOT wired into the evolution workflow — run standalone (needs scikit-learn; the panda
env has it):

    conda run -n panda python proofreader_evolve/plotting/analyze_policy_decisions.py \\
        794491_train1_test1_20260714_011120

    ... [--source candidate|accepted] [--max-sites N] [--with-image]
        [--enum-from final|seed|default] [--out fig.png]

``run_name`` is the directory under ``proofreader_evolve/runs/`` (a full path also
works). Default outputs, alongside the run: ``policy_decisions.png``,
``policy_decisions_jaccard.csv``, ``policy_decisions_trees.txt``.

HOW TO HAVE A COMPLETE RUN
--------------------------
"Complete" = every generation decides on the FULL enumerated candidate set (not the
default subsample), with the image reader active so image-gated policy tiers run
faithfully. Run it from the project root in the ``panda`` env (which has scikit-learn +
the SDK + the GCS token), DETACHED, since it is heavy:

    cd /path/to/exa-spim-agent
    nohup conda run -n panda python \\
        proofreader_evolve/plotting/analyze_policy_decisions.py \\
        794491_train1_test1_20260714_011120 --max-sites 0 --with-image \\
        > /tmp/policy_decisions.log 2>&1 &
    tail -f /tmp/policy_decisions.log

  * ``--max-sites 0`` uses ALL candidate sites (no subsample); ``--with-image`` builds
    the GCS ``LazyImagePatchReader`` so an image-gated ``propose_edits`` reproduces its
    real behaviour (this is what the evolution loop did).

MEMORY (the main constraint). ``site_feature_matrix`` does a per-site graph walk
(``split_geom`` -> ``rooted_subgraph``) over the whole-brain fragment graph (e.g.
794495 is ~20.8M nodes), once per site, plus a KD-tree over all nodes for the spatial
helpers. At ``--max-sites 0`` that is one walk per candidate (~5000) — run it on the
SAME machine that can load the graph in ``compare_proofreader_policy.ipynb``. If the
process is Killed (OOM), step down: ``--max-sites 2000`` (the default) is already a
large, representative comparison and ~2.5x lighter; ``--max-sites 800`` lighter still.
The Jaccard / surrogate-tree conclusions are stable across a few-thousand-site sample.

COST. ``--with-image`` issues cloud reads per gated site PER generation, so it is slow
and costs money. If you only care about the GEOMETRY-gated behaviour, OMIT
``--with-image``: the geometry-only path is EXACT for geometry tiers and a flagged
subset for image tiers, and is far faster/cheaper.

FIDELITY CHECK (do image only if needed). The run prints ``mean surrogate fidelity`` at
the end. If it is >= ~0.95, the geometry features already captured the policy's
decisions and ``--with-image`` adds nothing — skip it. A LOW fidelity means the policy
keys on something outside the geometry feature set (most likely an image read), so
rerun WITH ``--with-image`` for a faithful comparison. Recommended first pass — full
sites, geometry-only, then add image only if fidelity is low:

    conda run -n panda python \\
        proofreader_evolve/plotting/analyze_policy_decisions.py \\
        794491_train1_test1_20260714_011120 --max-sites 0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

# HERE = proofreader_evolve/. This file is in proofreader_evolve/plotting/.
HERE = Path(__file__).resolve().parent.parent
RUNS_DIR = HERE / "runs"
# PROJECT_ROOT = exa-spim-agent/. Ensure the package + the metrics src are importable
# when this file is run directly as a CLI (not only when imported from a notebook that
# already set sys.path). Harmless if already present.
_PROJECT_ROOT = HERE.parent
for _p in (str(_PROJECT_ROOT),
           str(_PROJECT_ROOT.parent / "segmentation-skeleton-metrics" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The deployable per-site feature set the surrogate tree is fit over. Each name is
# either a SplitSite attribute or a key of ``candidate._split_site_geom`` — i.e. exactly
# the GT-free geometry/topology a policy is allowed to threshold on. Booleans are cast
# to 0/1; missing values become NaN (sklearn >= 1.3 splits on NaN natively).
_SITE_FEATURES = ["gap_um", "mutual_nearest", "recip_rank_a", "recip_rank_b"]
_GEOM_FEATURES = ["colinear_cos", "cos_a", "cos_b", "tip_tangent_cos",
                  "deg_a", "deg_b", "rad_a", "rad_b", "rad_ratio",
                  "same_component", "graph_path_um",
                  "cable_a", "cable_b", "cable_min", "dist_to_branch_b"]
FEATURE_NAMES = _SITE_FEATURES + _GEOM_FEATURES


def _resolve_run_dir(run_name: str) -> Path:
    p = Path(run_name)
    if p.is_dir():
        return p
    cand = RUNS_DIR / run_name
    if cand.is_dir():
        return cand
    raise FileNotFoundError(f"run not found: {run_name!r} (looked for {p} and {cand})")


def _gen_dirs(run_dir: Path):
    import re
    gens = [d for d in run_dir.glob("gen*") if d.is_dir()
            and re.fullmatch(r"gen\d+", d.name)]
    return sorted(gens, key=lambda d: int(d.name[3:]))


def _accepted_gens(run_dir: Path) -> set[int]:
    try:
        rows = [json.loads(l) for l in (run_dir / "ledger.jsonl").read_text().splitlines()
                if l.strip()]
        return {int(r["generation"]) for r in rows if r.get("accepted")}
    except Exception:
        return set()


def _pair_key(a, b) -> tuple[str, str]:
    a, b = str(a), str(b)
    return (a, b) if a <= b else (b, a)


def _policy_file(gen_dir: Path, source: str) -> Path | None:
    """The heuristics file to run for a generation. ``candidate`` (default) = the policy
    that generation PROPOSED (every gen has one, accepted or not — the search-behavior
    view); ``accepted`` = only the kept lineage (gen has ``heuristics.accepted.py``)."""
    f = gen_dir / ("heuristics.accepted.py" if source == "accepted"
                   else "heuristics.candidate.py")
    return f if f.exists() else None


def _to_float(v) -> float:
    """Coerce a feature value to float for the surrogate tree: bool -> 0/1, None -> NaN."""
    if v is None:
        return float("nan")
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def build_reference_stream(run_dir: Path, gens: list[int], enum_from: str):
    """Enumerate the ONE shared candidate SplitSite set every policy will decide on.

    ``enum_from`` chooses whose ``ENUM_PARAMS`` define the stream:
      * ``final``   — the last accepted generation's policy (the reference; default),
      * ``seed``    — generation 1's policy,
      * ``default`` — the framework defaults (ignore any policy ENUM_PARAMS).
    Returns ``(fragments_graph, split_sites, enum_params)``. Uses the same cached
    whole-brain enumeration the loop uses, so the site set is identical to a real run's.
    """
    from proofreader_evolve.harness import dataset as ds
    from proofreader_evolve.harness import candidate as cand

    split_cfg = json.loads((run_dir / "split.json").read_text())
    # Cross-brain test brain, else the single brain.
    test = split_cfg.get("test_brains") or []
    brain = str(test[0]) if test else str(split_cfg.get("brains", ["789202"])[0])
    mcl = int(split_cfg.get("mcl", 100))
    cache_path = ds.default_cache_path(brain, min_cable_length=mcl)
    fragments_graph, _gt, _payload = ds.load_cached_graphs(cache_path)

    raw_enum = {}
    if enum_from in ("final", "seed"):
        pick = max(gens) if enum_from == "final" else min(gens)
        # prefer accepted policy for 'final', candidate otherwise
        pol = (run_dir / f"gen{pick:02d}" / "heuristics.accepted.py")
        if not pol.exists():
            pol = run_dir / f"gen{pick:02d}" / "heuristics.candidate.py"
        _, mod = cand._load_policy(str(pol))
        raw = getattr(mod, "ENUM_PARAMS", None)
        raw_enum = {**raw} if isinstance(raw, dict) else {}
    raw_enum.setdefault("max_gap_um", 15.0)
    enum_params = ds.resolve_enum_params(raw_enum)
    # splits-only enumeration (these runs are splits-only): SplitSites only.
    split_sites, _merge, _stats = cand._enumerate_sites_cached(
        fragments_graph, enum_params, enumerate_merges=False)
    return fragments_graph, list(split_sites), enum_params, brain


def build_ctx(fragments_graph, enum_params, split_sites, image_reader):
    """The ctx dict passed to every policy — mirrors ``candidate.run_candidate`` exactly
    (spatial helpers, split_geom, node_radius, image reader), so any policy runs
    faithfully. ``image_reader`` is None for the cheap geometry-only path."""
    from proofreader_evolve.harness import candidate as cand
    return {
        "max_gap_um": enum_params["max_gap_um"],
        "enum_params": dict(enum_params),
        "fragments_graph": fragments_graph,
        "n_split_sites": len(split_sites),
        "n_merge_sites": 0,
        "node_radius": getattr(fragments_graph, "node_radius", None),
        "split_geom": (lambda site: cand._split_site_geom(fragments_graph, site)),
        "nodes_within": (lambda xyz, radius: cand._nodes_within(fragments_graph, xyz, radius)),
        "foreign_labels_near": (
            lambda xyz, radius, exclude=(): cand._foreign_labels_near(
                fragments_graph, xyz, radius, exclude)),
        "read_image_patch": image_reader,
        "image_patch_shape": (16, 16, 16),
    }


def policy_decisions(gen_dirs_map: dict[int, Path], source: str, ctx, sites) -> dict[int, set]:
    """Run each generation's ``propose_edits`` on the SHARED ``sites`` and return
    ``{gen: set of accepted (label_a,label_b) pair-keys}``.

    Only ``merge_labels`` edits count (these are splits-only runs). A generation whose
    policy fails to import/run is skipped (absent from the returned dict) with a warning
    — the comparison then simply omits it rather than crashing the whole analysis.
    """
    from proofreader_evolve.harness import candidate as cand
    from proofreader_evolve.harness.edit_handler import normalize_edits

    out: dict[int, set] = {}
    for g, gd in gen_dirs_map.items():
        pol = _policy_file(gd, source)
        if pol is None:
            continue
        try:
            propose_edits, _mod = cand._load_policy(str(pol))
            raw = propose_edits(sites, ctx)
            edits = normalize_edits(raw)
            accepted = {
                _pair_key(e["label_a"], e["label_b"])
                for e in edits if e.get("kind") == "merge_labels"
            }
            out[g] = accepted
        except Exception as e:
            print(f"[policy_decisions] WARN gen{g:02d} policy failed to run ({e}); skipped")
    return out


def jaccard_matrix(gens: list[int], decisions: dict[int, set]) -> np.ndarray:
    """Pairwise Jaccard of accepted-pair sets. J(i,j)=|A_i∩A_j|/|A_i∪A_j|; two empty
    sets are defined as J=1.0 (both decided "accept nothing" — identical behavior)."""
    n = len(gens)
    m = np.zeros((n, n), dtype=float)
    for i, gi in enumerate(gens):
        for j, gj in enumerate(gens):
            a, b = decisions[gi], decisions[gj]
            if not a and not b:
                m[i, j] = 1.0
            else:
                inter = len(a & b)
                union = len(a | b)
                m[i, j] = inter / union if union else 1.0
    return m


def site_feature_matrix(sites, ctx) -> np.ndarray:
    """Compute the ``FEATURE_NAMES`` matrix ONCE for the shared sites (policy-independent
    — the surrogate trees all reuse it). Shape ``(n_sites, n_features)``, NaN where a
    feature is undefined. split_geom does per-site graph walks, so this is the main cost;
    it is paid a single time here, not per generation."""
    split_geom = ctx["split_geom"]
    X = np.full((len(sites), len(FEATURE_NAMES)), np.nan, dtype=float)
    for i, s in enumerate(sites):
        geom = {}
        try:
            geom = split_geom(s)
        except Exception:
            geom = {}
        row = {
            "gap_um": getattr(s, "gap_um", None),
            "mutual_nearest": getattr(s, "mutual_nearest", None),
            "recip_rank_a": getattr(s, "recip_rank_a", None),
            "recip_rank_b": getattr(s, "recip_rank_b", None),
        }
        for k in _GEOM_FEATURES:
            row[k] = geom.get(k)
        for j, name in enumerate(FEATURE_NAMES):
            X[i, j] = _to_float(row.get(name))
    return X


def surrogate_trees(gens: list[int], decisions: dict[int, set], sites, X: np.ndarray,
                    max_depth: int = 4):
    """Fit one shallow DecisionTreeClassifier per generation: features -> the policy's
    accept(1)/reject(0) label on the SAME sites. Returns ``(importances, fidelity,
    trees)``:
      * ``importances[g]`` — length-``FEATURE_NAMES`` array of Gini importances (which
        features drive that generation's decision),
      * ``fidelity[g]``    — train accuracy of the surrogate reproducing the policy
        (1.0 = the tree captures the policy exactly on this feature set; low => the
        policy keys on something absent here, e.g. an image read — distrust the tree),
      * ``trees[g]``       — the fitted classifier (for export_text / plotting).
    A degenerate generation (accepts none or all sites) yields zero importances and
    fidelity 1.0 (the constant tree trivially reproduces a constant decision).
    """
    from sklearn.tree import DecisionTreeClassifier

    site_pairs = [_pair_key(s.label_a, s.label_b) for s in sites]
    importances, fidelity, trees = {}, {}, {}
    for g in gens:
        acc = decisions[g]
        y = np.array([1 if pk in acc else 0 for pk in site_pairs], dtype=int)
        if y.min() == y.max():  # constant decision -> nothing for a tree to split on
            importances[g] = np.zeros(len(FEATURE_NAMES))
            fidelity[g] = 1.0
            trees[g] = None
            continue
        clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=5,
                                     random_state=0)
        clf.fit(X, y)
        importances[g] = clf.feature_importances_
        fidelity[g] = float(clf.score(X, y))
        trees[g] = clf
    return importances, fidelity, trees


def make_figure(gens, jac, importances, fidelity, accepted, decisions,
                run_name, brain, out_path) -> Path:
    """Three panels: (1) decision Jaccard heatmap, (2) decision-agreement trajectory
    (consecutive / to-seed / to-final), (3) surrogate feature-importance heatmap."""
    n = len(gens)
    acc_mask = [g in accepted for g in gens]
    consec = [float("nan")] + [float(jac[i, i - 1]) for i in range(1, n)]
    to_seed = [float(jac[i, 0]) for i in range(n)]
    final_idx = max((i for i, a in enumerate(acc_mask) if a), default=n - 1)
    to_final = [float(jac[i, final_idx]) for i in range(n)]
    n_accepted_edits = [len(decisions[g]) for g in gens]

    fig = plt.figure(figsize=(16, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15])

    # (1) decision Jaccard heatmap
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(jac, cmap="viridis", vmin=float(np.min(jac)), vmax=1.0)
    ax0.set_xticks(range(n)); ax0.set_yticks(range(n))
    ax0.set_xticklabels(gens, fontsize=6, rotation=90)
    ax0.set_yticklabels(gens, fontsize=6)
    ax0.set_xlabel("generation"); ax0.set_ylabel("generation")
    ax0.set_title("decision agreement (Jaccard of\naccepted merge pairs)", fontsize=10)
    for i, a in enumerate(acc_mask):
        if a:
            ax0.add_patch(plt.Rectangle((i - 0.5, -0.5), 1, n, fill=False,
                                        edgecolor="#2ca02c", lw=0.8, alpha=0.5))
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04).set_label("Jaccard", fontsize=8)

    # (2) decision-agreement trajectory
    ax1 = fig.add_subplot(gs[0, 1])
    x = np.arange(n)
    ax1.plot(x, consec, "-o", color="#d62728", ms=4, lw=1.3,
             label="consecutive J(gen, gen−1)")
    ax1.plot(x, to_seed, "-s", color="#1f77b4", ms=3, lw=1.1, label="J to seed (gen 1)")
    ax1.plot(x, to_final, "-^", color="#2ca02c", ms=3, lw=1.1,
             label=f"J to final ({gens[final_idx]})")
    for i, a in enumerate(acc_mask):
        if a:
            ax1.axvline(i, color="#2ca02c", lw=0.8, ls=":", alpha=0.6)
    ax1.set_xticks(x); ax1.set_xticklabels(gens, fontsize=6, rotation=90)
    ax1.set_xlabel("generation"); ax1.set_ylabel("Jaccard")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("decision drift across generations\n(low consecutive = big behavior "
                  "change; green = accepted)", fontsize=9)
    ax1.legend(fontsize=7, loc="lower right", framealpha=0.9)

    # (3) surrogate feature-importance heatmap (feature × generation)
    ax2 = fig.add_subplot(gs[0, 2])
    imp = np.array([importances[g] for g in gens]).T  # (n_features, n_gens)
    # drop features never used by any generation, to declutter
    used = imp.sum(axis=1) > 0
    feats_used = [f for f, u in zip(FEATURE_NAMES, used) if u]
    imp_used = imp[used] if used.any() else imp
    im2 = ax2.imshow(imp_used, cmap="magma", aspect="auto", vmin=0.0, vmax=1.0)
    ax2.set_yticks(range(len(feats_used)))
    ax2.set_yticklabels(feats_used, fontsize=7)
    ax2.set_xticks(range(n)); ax2.set_xticklabels(gens, fontsize=6, rotation=90)
    ax2.set_xlabel("generation")
    _meanfid = np.mean([fidelity[g] for g in gens])
    ax2.set_title(f"surrogate-tree feature importance\n(what drives accept/reject; "
                  f"mean fidelity {_meanfid:.2f})", fontsize=9)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).set_label("Gini importance", fontsize=8)

    fig.suptitle(f"Run {run_name} — policy DECISION comparison (brain {brain}, "
                 f"geometry features; {n} generations)", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_jaccard_csv(gens, jac, path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gen"] + list(gens))
        for i, g in enumerate(gens):
            w.writerow([g] + [f"{jac[i, j]:.4f}" for j in range(len(gens))])
    return path


def write_trees_txt(gens, trees, fidelity, decisions, accepted, path) -> Path:
    """Human-readable dump: for each generation, the surrogate tree (export_text over the
    real feature names) + its fidelity + how many pairs it accepted. This is the literal
    "what decision tree does gen N implement" artifact."""
    from sklearn.tree import export_text
    lines = ["# Surrogate decision trees per generation", "",
             "Each tree distills that generation's propose_edits into a canonical tree "
             "over the shared geometry features. FIDELITY = train accuracy reproducing "
             "the policy (low => the policy keys on features absent here, e.g. image).",
             ""]
    for g in gens:
        tag = "ACCEPTED" if g in accepted else "rejected"
        lines.append(f"\n{'='*70}\n## gen{g:02d} [{tag}] — accepts {len(decisions[g])} "
                     f"pair(s), surrogate fidelity {fidelity[g]:.3f}\n{'='*70}")
        clf = trees[g]
        if clf is None:
            lines.append("(constant decision — accepts none or all sites; no tree)")
            continue
        try:
            lines.append(export_text(clf, feature_names=FEATURE_NAMES, max_depth=6))
        except Exception as e:
            lines.append(f"(could not export tree: {e})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def analyze(run_dir: Path, source: str, max_sites: int, with_image: bool,
            enum_from: str, max_depth: int):
    """Full pipeline. Returns everything the figure/CSV/txt writers need."""
    gen_dirs = _gen_dirs(run_dir)
    gen_dirs_map = {int(d.name[3:]): d for d in gen_dirs
                    if _policy_file(d, source) is not None}
    gens = sorted(gen_dirs_map)
    if len(gens) < 2:
        raise SystemExit(f"need >=2 generations with a {source} policy (found {len(gens)}).")

    fragments_graph, split_sites, enum_params, brain = build_reference_stream(
        run_dir, gens, enum_from)

    # Optional image reader (faithful for image-gated policies; costly). Off by default.
    image_reader = None
    if with_image:
        import sys as _sys
        _sys.path.insert(0, str(HERE.parent / "scripts"))
        try:
            from proofreader_evolve.harness.image_features import LazyImagePatchReader
            from dataset_config import get_img_path
            _prefixes = str(HERE.parent / "configs" / "exaspim_image_prefixes.json")
            image_reader = LazyImagePatchReader(get_img_path(brain, prefixes_path=_prefixes),
                                                fragments_graph)
            print(f"[policy_decisions] image reader ENABLED for brain {brain}")
        except Exception as e:
            print(f"[policy_decisions] WARN could not build image reader ({e}); "
                  f"proceeding geometry-only (image-gated tiers will be a subset).")

    # Deterministically subsample the shared site set (keeps per-gen decision + one-time
    # feature extraction tractable). Evenly spaced by gap so the sample spans the range.
    sites = list(split_sites)
    if max_sites and len(sites) > max_sites:
        sites.sort(key=lambda s: s.gap_um)
        idx = np.linspace(0, len(sites) - 1, max_sites).astype(int)
        sites = [sites[i] for i in idx]
    print(f"[policy_decisions] {len(gens)} gens, {len(sites)} shared candidate sites "
          f"(enum_from={enum_from}, image={'on' if image_reader else 'off'})")

    ctx = build_ctx(fragments_graph, enum_params, sites, image_reader)
    decisions = policy_decisions(gen_dirs_map, source, ctx, sites)
    gens = [g for g in gens if g in decisions]  # keep only gens that ran
    jac = jaccard_matrix(gens, decisions)
    X = site_feature_matrix(sites, ctx)
    importances, fidelity, trees = surrogate_trees(gens, decisions, sites, X, max_depth)
    return dict(gens=gens, jac=jac, importances=importances, fidelity=fidelity,
                trees=trees, decisions=decisions, accepted=_accepted_gens(run_dir),
                brain=brain, n_sites=len(sites))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_name", help="run dir under proofreader_evolve/runs/ (or a path)")
    ap.add_argument("--source", default="candidate", choices=["candidate", "accepted"],
                    help="which policy per gen: 'candidate' (proposed, every gen — the "
                         "search view; default) or 'accepted' (kept lineage only)")
    ap.add_argument("--max-sites", type=int, default=2000,
                    help="subsample the shared candidate stream to this many sites "
                         "(evenly spaced by gap). Default 2000; 0 = use all.")
    ap.add_argument("--with-image", action="store_true",
                    help="build the image reader so image-gated policies run faithfully "
                         "(needs GCS; costs cloud reads PER SITE PER GEN). Default off = "
                         "geometry-only (exact for geometry-gated policies; a subset for "
                         "image-gated ones — flagged).")
    ap.add_argument("--enum-from", default="final", choices=["final", "seed", "default"],
                    help="whose ENUM_PARAMS define the ONE shared candidate stream "
                         "(default: final accepted gen). Decisions are compared on this "
                         "fixed stream; enumeration-prior changes are a separate axis.")
    ap.add_argument("--max-depth", type=int, default=4,
                    help="max depth of each surrogate DecisionTree (default 4 — shallow, "
                         "readable; the policies are shallow guard chains)")
    ap.add_argument("--out", default=None,
                    help="output PNG (default: runs/<run>/policy_decisions.png)")
    args = ap.parse_args(argv)

    run_dir = _resolve_run_dir(args.run_name)
    R = analyze(run_dir, args.source, args.max_sites, args.with_image,
                args.enum_from, args.max_depth)

    out_path = Path(args.out) if args.out else (run_dir / "policy_decisions.png")
    make_figure(R["gens"], R["jac"], R["importances"], R["fidelity"], R["accepted"],
                R["decisions"], run_dir.name, R["brain"], out_path)
    csv_path = out_path.with_name("policy_decisions_jaccard.csv")
    write_jaccard_csv(R["gens"], R["jac"], csv_path)
    txt_path = out_path.with_name("policy_decisions_trees.txt")
    write_trees_txt(R["gens"], R["trees"], R["fidelity"], R["decisions"],
                    R["accepted"], txt_path)

    gens, dec, fid = R["gens"], R["decisions"], R["fidelity"]
    consec = [(gens[i], float(R["jac"][i, i - 1])) for i in range(1, len(gens))]
    if consec:
        bg, bs = min(consec, key=lambda t: t[1])
        mean_j = sum(s for _, s in consec) / len(consec)
        print(f"[policy_decisions] mean consecutive decision-Jaccard = {mean_j:.3f} "
              f"(1.0 = identical decisions); biggest behavior change at gen {bg} "
              f"(J {bs:.3f} vs prev)")
    print(f"[policy_decisions] mean surrogate fidelity = "
          f"{np.mean([fid[g] for g in gens]):.3f} "
          f"(1.0 = tree captures the policy on the geometry features)")
    print(f"[policy_decisions] figure -> {out_path}")
    print(f"[policy_decisions] jaccard -> {csv_path}")
    print(f"[policy_decisions] trees   -> {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

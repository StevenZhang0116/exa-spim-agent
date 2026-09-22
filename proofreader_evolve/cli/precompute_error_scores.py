"""
Precompute per-brain detector score tables for the evolution FeatureBank.

Runs OFFLINE (compute node, `panda` env, >20 GB RAM for the pkl) and writes the
tables `harness.feature_bank.FeatureBank` serves at evolution time:

    proofreader_evolve/feature_tables/<brain>/merge_scores.pkl
    proofreader_evolve/feature_tables/<brain>/split_scores.pkl
    proofreader_evolve/feature_tables/<brain>/meta.json
    proofreader_evolve/feature_tables/<brain>/threshold_sweep.md   (with GT)

The detector priors are read DIRECTLY from their detector-build deliverable
folders under `autodiscovery-application/` — nothing is copied into
proofreader_evolve/. `--merge-dir` and `--split-dir` each name one such folder
(one merge run + one split run), and each folder is the single source of truth
for BOTH the frozen feature-extraction script (`<kind>_site_detector.py`) and
the fitted model it produced (`<kind>_detector_*.joblib`). meta.json records
the folder, script and joblib SHA-256s so every score table is traceable to
its exact detector run.

Training provenance / leakage rules for the DEFAULT folders (both fitted on
brain 794495, all its GT labels, nested CV; 794491 used during discovery only
as a generalization check; 789202 never touched): 794495 may serve as an
evolution TRAIN brain; the held-out GATE brain should be 794491 (mild
feature-selection leakage) or 789202 (none). Never gate a within-794495
train/heldout split against these priors. Point the `--*-dir` flags at other
detector runs and these rules travel with THOSE runs' READMEs instead.

Key design point — SCORE THE LOOP'S CANDIDATES, NOT THE DETECTOR'S UNIVERSE:
feature extraction reuses the frozen detector scripts verbatim (imported from
the deliverable folders), but the rows are the candidates the EVOLUTION
enumerators surface (`dataset.candidate_split_sites` / `candidate_merge_sites`),
enumerated here with generous bounds so any ENUM_PARAMS a policy can legally set
stays inside table coverage. That is ~50k split pairs + ~5k merge labels per
brain instead of the detectors' 500k-row universe — two orders of magnitude
cheaper, and exactly the rows the policy will ever be asked about.

Injection seams into the frozen scripts (no detector code is modified):
  * split: `extract_features` reads its candidate universe from
    `payload["__detector_sample_universe_cache__"]` — we pre-fill it with rows
    built from the loop's SplitSites (detector row schema, labels all 0).
  * merge: `extract_features` derives its segment set from
    `payload["gt_node_canonical_label"]` — we swap in a synthetic array that
    marks exactly the candidate labels (and empty `gt_merge_labels`), then
    restore the real arrays for the threshold sweep.

The sweep section (train brain only!) reports, per detector, the zero-false-
positive operating point on this brain's GT and the corresponding brain-relative
score QUANTILE — the number the seed policy's `*_QUANTILE` constants should be
calibrated from. Never calibrate thresholds on a brain the evolution gate will
score.

Example (Slurm):
    srun --partition=aind --mem=160G --cpus-per-task=4 --time=24:00:00 \
      /home/zihan.zhang/.conda/envs/panda/bin/python -m \
      proofreader_evolve.cli.precompute_error_scores \
      --brain 794495 --pkl cache/dataset_cache_794495_mcl100_add.pkl
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Make `proofreader_evolve` importable when run as a script from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from proofreader_evolve.harness import dataset as ds  # noqa: E402

HERE = Path(__file__).resolve().parent.parent          # proofreader_evolve/
DEFAULT_OUT = HERE / "feature_tables"
APP_DIR = PROJECT_ROOT / "autodiscovery-application"
# Default detector deliverable folders (one merge run + one split run). Each is
# the single source of truth for both the frozen script and its fitted joblib.
DEFAULT_MERGE_DIR = APP_DIR / "merge-error-794495-mcl100_2026-08-04"
DEFAULT_SPLIT_DIR = APP_DIR / "split-error-794495-mcl100-run-3_2026-08-24"


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _import_module(path: Path, name: str):
    """Import a frozen detector script by file path (its dir name has hyphens)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # so any internal pickling/introspection resolves
    spec.loader.exec_module(module)
    return module


def _score(estimator, X: np.ndarray) -> np.ndarray:
    """Ranking score from a fitted pipeline (same convention as the detectors)."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    raise TypeError(f"estimator {type(estimator).__name__} has no scoring method")


def _load_joblib(path: Path) -> dict:
    import joblib
    bundle = joblib.load(path)
    for key in ("pipeline", "feature_order"):
        if key not in bundle:
            raise KeyError(f"{path} lacks '{key}' — not a detector joblib bundle")
    return bundle


def _resolve_detector_dir(dir_arg: str, kind: str) -> tuple[Path, Path]:
    """Resolve one detector deliverable folder to ``(script_path, joblib_path)``.

    ``kind`` is ``"merge"`` or ``"split"``. The folder (an
    ``autodiscovery-application/<run>/`` detector-build deliverable) must hold
    the frozen ``<kind>_site_detector.py`` and exactly one fitted
    ``<kind>_detector_*.joblib`` produced by running that script (see the
    folder's RUN_COMMANDS.md if the joblib is missing).
    """
    d = Path(dir_arg)
    if not d.is_dir():
        raise SystemExit(f"--{kind}-dir is not a directory: {d}")
    script = d / f"{kind}_site_detector.py"
    if not script.exists():
        raise SystemExit(
            f"--{kind}-dir has no {script.name} — not a {kind}-detector "
            f"deliverable folder: {d}")
    joblibs = sorted(d.glob(f"{kind}_detector_*.joblib"))
    if not joblibs:
        raise SystemExit(
            f"--{kind}-dir has no {kind}_detector_*.joblib — run the detector "
            f"first (see {d / 'RUN_COMMANDS.md'})")
    if len(joblibs) > 1:
        raise SystemExit(
            f"--{kind}-dir holds {len(joblibs)} {kind}_detector_*.joblib files "
            f"({', '.join(p.name for p in joblibs)}) — remove all but the "
            f"intended one: {d}")
    return script, joblibs[0]


def _base_int_label(label) -> int:
    """Loop label (str, possibly 'L#a') -> the detector's integer segment id."""
    return int(str(label).split("#", 1)[0])


# --------------------------------------------------------------------------- #
# split side
# --------------------------------------------------------------------------- #

def _build_split_universe(frag, sites) -> tuple[list, np.ndarray, list]:
    """Detector-schema candidate rows from the loop's SplitSites.

    Returns ``(samples, labels_zeros, pair_meta)`` where ``pair_meta[i]`` carries
    the loop-facing identity of row i: (label_a, label_b, node_a, node_b, gap_um)
    with labels as the LOOP's strings (order-free pair; representative = closest
    gap). One row per unordered label pair, all the pair's gaps as occurrences —
    exactly the shape ``build_sample_universe`` produces, so the frozen feature
    code treats these rows identically.
    """
    def _role(node) -> str:
        return "tip" if int(frag.degree(node)) == 1 else "non_tip"

    grouped: dict = defaultdict(list)   # (seg_a_int, seg_b_int) -> [occurrence...]
    loop_ids: dict = {}                 # same key -> (label_a, label_b) loop strs
    for s in sites:
        seg_anchor = _base_int_label(s.label_a)
        seg_partner = _base_int_label(s.label_b)
        if seg_anchor == seg_partner:
            continue
        seg_a, seg_b = sorted((seg_anchor, seg_partner))
        if seg_anchor == seg_a:
            node_a, node_b, anchor_side = int(s.node_a), int(s.node_b), "a"
        else:
            node_a, node_b, anchor_side = int(s.node_b), int(s.node_a), "b"
        gaps = [{"gap_um": float(s.gap_um), "node_a": int(s.node_a),
                 "node_b": int(s.node_b)}]
        gaps += [{"gap_um": float(g["gap_um"]), "node_a": int(g["node_a"]),
                  "node_b": int(g["node_b"])} for g in (s.alt_gaps or [])]
        key = (seg_a, seg_b)
        loop_ids.setdefault(key, (str(s.label_a), str(s.label_b)))
        for g in gaps:
            # occurrence node ids oriented to the SORTED pair, like the detector's
            a_node, b_node = ((g["node_a"], g["node_b"]) if anchor_side == "a"
                              else (g["node_b"], g["node_a"]))
            grouped[key].append({
                "node_id_a": int(a_node),
                "node_id_b": int(b_node),
                "gap_um": float(g["gap_um"]),
                "anchor_side": anchor_side,
                "partner_rank": max(1, int(getattr(s, "recip_rank_a", 1) or 1)),
                "node_role_a": _role(a_node),
                "node_role_b": _role(b_node),
            })

    samples, pair_meta = [], []
    for idx, pair in enumerate(sorted(grouped)):
        occurrences = tuple(sorted(
            grouped[pair],
            key=lambda row: (row["gap_um"], row["node_id_a"], row["node_id_b"],
                             row["anchor_side"], row["partner_rank"]),
        ))
        rep = occurrences[0]
        samples.append({
            "candidate_id": int(idx),
            "segment_id_a": int(pair[0]),
            "segment_id_b": int(pair[1]),
            "node_id_a": int(rep["node_id_a"]),
            "node_id_b": int(rep["node_id_b"]),
            "gap_um": float(rep["gap_um"]),
            "anchor_side": rep["anchor_side"],
            "partner_rank": int(rep["partner_rank"]),
            "node_role_a": rep["node_role_a"],
            "node_role_b": rep["node_role_b"],
            "is_merge_creating": 0,             # audit-only field, not a predictor
            "contains_known_merge_segment": 0,  # audit-only field, not a predictor
            "occurrences": occurrences,
        })
        la, lb = loop_ids[pair]
        pair_meta.append((la, lb, int(rep["node_id_a"]), int(rep["node_id_b"]),
                          float(rep["gap_um"])))
    return samples, np.zeros(len(samples), dtype=np.int64), pair_meta


def compute_split_table(payload, sites, split_mod, bundle, log=log):
    """Score the loop's split candidates with the frozen split detector."""
    import pandas as pd

    frag = payload["fragments_graph"]
    samples, zero_labels, pair_meta = _build_split_universe(frag, sites)
    log(f"[split] {len(sites)} loop sites -> {len(samples)} unordered pair rows")
    if not samples:
        return None

    # Injection seam: extract_features reads the universe from this cache key and
    # then never touches GT (GT is consulted only by the universe builder we skip).
    payload["__detector_sample_universe_cache__"] = (samples, zero_labels)
    try:
        t0 = time.monotonic()
        rows, _labels, acc = split_mod.extract_features(payload, verbose=True)
        log(f"[split] feature extraction: {time.monotonic() - t0:.0f}s "
            f"({len(rows)} rows)")
        df = acc.to_frame()
    finally:
        payload.pop("__detector_sample_universe_cache__", None)
    if len(df) != len(samples):
        raise RuntimeError(f"[split] extractor returned {len(df)} rows for "
                           f"{len(samples)} injected candidates")

    feature_order = list(bundle["feature_order"])
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise RuntimeError(f"[split] extracted frame lacks joblib features: "
                           f"{missing[:5]}...")
    scores = _score(bundle["pipeline"], df[feature_order].values.astype(float))

    table = pd.DataFrame({
        "label_a": [m[0] for m in pair_meta],
        "label_b": [m[1] for m in pair_meta],
        "node_a": [m[2] for m in pair_meta],
        "node_b": [m[3] for m in pair_meta],
        "gap_um": [m[4] for m in pair_meta],
        "split_score": scores.astype(float),
    })
    table = pd.concat([table, df.reset_index(drop=True)], axis=1)
    return table


# --------------------------------------------------------------------------- #
# merge side
# --------------------------------------------------------------------------- #

def _node_segment_array(frag, merge_mod) -> np.ndarray:
    """Per-node integer segment id (the detectors' convention), -1 when unmapped."""
    comp_to_seg = merge_mod.build_comp_to_seg(frag)
    node_components = np.asarray(frag.node_component_id, dtype=np.int64)
    return np.fromiter(
        (int(comp_to_seg.get(int(c), -1)) for c in node_components),
        dtype=np.int64, count=len(node_components))


def compute_merge_table(payload, merge_sites, merge_mod, bundle, log=log):
    """Score the loop's merge-candidate LABELS with the frozen merge detector.

    Swaps a synthetic ``gt_node_canonical_label`` (candidate labels marked, all
    else 0) plus empty ``gt_merge_labels`` into the payload so the frozen
    extractor walks exactly the candidate set, then restores the real arrays.
    """
    import pandas as pd

    cand_ints = sorted({_base_int_label(s.label) for s in merge_sites})
    log(f"[merge] {len(merge_sites)} loop sites -> {len(cand_ints)} candidate labels")
    if not cand_ints:
        return None
    # Remember the loop's exact label strings (first site per label wins).
    label_str: dict = {}
    for s in merge_sites:
        label_str.setdefault(_base_int_label(s.label), str(s.label))

    frag = payload["fragments_graph"]
    node_segments = _node_segment_array(frag, merge_mod)
    synthetic = np.where(np.isin(node_segments,
                                 np.asarray(cand_ints, dtype=np.int64)),
                         node_segments, 0)

    real_canonical = payload["gt_node_canonical_label"]
    real_merges = payload["gt_merge_labels"]
    payload["gt_node_canonical_label"] = synthetic
    payload["gt_merge_labels"] = []
    try:
        t0 = time.monotonic()
        adjudicable, _is_merge, acc = merge_mod.extract_features(payload,
                                                                 verbose=True)
        log(f"[merge] feature extraction: {time.monotonic() - t0:.0f}s "
            f"({len(adjudicable)} labels)")
        df = acc.to_frame()
    finally:
        payload["gt_node_canonical_label"] = real_canonical
        payload["gt_merge_labels"] = real_merges

    feature_order = list(bundle["feature_order"])
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise RuntimeError(f"[merge] extracted frame lacks joblib features: "
                           f"{missing[:5]}...")
    scores = _score(bundle["pipeline"], df[feature_order].values.astype(float))

    table = df.reset_index(drop=True)
    table.insert(0, "merge_score", scores.astype(float))
    table.index = [label_str.get(seg) or str(seg) for seg in adjudicable]
    table.index.name = "label"
    return table


# --------------------------------------------------------------------------- #
# threshold sweep (train brain only)
# --------------------------------------------------------------------------- #

def _sweep(scores: np.ndarray, is_pos: np.ndarray, name: str) -> tuple[list, dict]:
    """Zero-FP operating point + a small precision/recall sweep table."""
    order = np.argsort(-scores)
    s, y = scores[order], is_pos[order]
    lines = [f"### {name}", "",
             f"{int(y.sum())} positives / {len(y)} candidates "
             f"(GT-adjudicable subset)", "",
             "| threshold | quantile | TP | FP | precision | recall |",
             "|---:|---:|---:|---:|---:|---:|"]
    tp = fp = 0
    best_zero_fp = None
    checkpoints = set(np.linspace(0, len(s) - 1, 12, dtype=int).tolist())
    n_pos = max(1, int(y.sum()))
    for i in range(len(s)):
        tp += int(y[i]); fp += 1 - int(y[i])
        if fp == 0:
            best_zero_fp = (float(s[i]), tp)
        if i in checkpoints:
            q = 1.0 - (i + 1) / len(s)
            lines.append(f"| {s[i]:.4f} | {q:.4f} | {tp} | {fp} | "
                         f"{tp / (tp + fp):.3f} | {tp / n_pos:.3f} |")
    summary = {}
    if best_zero_fp is not None:
        thr, tp0 = best_zero_fp
        q0 = float((scores < thr).mean())
        summary = {"zero_fp_threshold": thr, "zero_fp_tp": int(tp0),
                   "zero_fp_recall": tp0 / n_pos, "zero_fp_quantile": q0}
        lines += ["", f"**Zero-FP operating point:** threshold {thr:.4f} "
                  f"(score quantile {q0:.4f}) -> {tp0} true positives, recall "
                  f"{tp0 / n_pos:.3f}. Calibrate the seed's quantile constant "
                  f"from this (add safety margin: cross-brain scores drift)."]
    lines.append("")
    return lines, summary


def sweep_tables(payload, split_table, merge_table, split_mod, log=log):
    """GT sweep for both tables; returns (markdown_lines, summary_dict)."""
    lines = ["# Detector score threshold sweep (train-brain GT)", "",
             "Calibration source for the seed policy's quantile thresholds. "
             "USE ONLY ON A TRAIN BRAIN — never calibrate on a gate brain.", ""]
    summary = {}

    if merge_table is not None:
        real_merges = {int(x) for x in payload["gt_merge_labels"]}
        traced = {int(x) for x in
                  np.unique(np.asarray(payload["gt_node_canonical_label"]))
                  if int(x) != 0}
        labels_int = np.asarray([_base_int_label(l) for l in merge_table.index])
        adjudicable_mask = np.isin(labels_int, np.asarray(sorted(traced)))
        y = np.isin(labels_int, np.asarray(sorted(real_merges)))
        if adjudicable_mask.any():
            sw, sm = _sweep(
                merge_table["merge_score"].to_numpy(dtype=float)[adjudicable_mask],
                y[adjudicable_mask].astype(int), "merge detector (label-level)")
            lines += sw
            summary["merge"] = sm

    if split_table is not None:
        truth_pairs = {tuple(sorted(map(int, pair)))
                       for _, pair in split_mod._derive_split_truth(payload)}
        pairs = [tuple(sorted((_base_int_label(a), _base_int_label(b))))
                 for a, b in zip(split_table["label_a"], split_table["label_b"])]
        y = np.asarray([p in truth_pairs for p in pairs], dtype=int)
        sw, sm = _sweep(split_table["split_score"].to_numpy(dtype=float), y,
                        "split detector (pair-level)")
        lines += sw
        summary["split"] = sm
        # persist the audit-only label onto the table for later analysis
        split_table["is_split"] = y

    return lines, summary


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--brain", required=True)
    p.add_argument("--pkl", required=True,
                   help="labeled cache: cache/dataset_cache_<brain>_mcl<N>_add.pkl "
                        "(the _add variant — the sweep needs its GT fields)")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--merge-dir", default=str(DEFAULT_MERGE_DIR),
                   help="merge detector deliverable folder under "
                        "autodiscovery-application/ (must hold the frozen "
                        "merge_site_detector.py and its fitted "
                        "merge_detector_*.joblib)")
    p.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR),
                   help="split detector deliverable folder under "
                        "autodiscovery-application/ (must hold the frozen "
                        "split_site_detector.py and its fitted "
                        "split_detector_*.joblib)")
    # Enumeration bounds: GENEROUS on purpose — cover anything a policy can set
    # via ENUM_PARAMS (dataset.ENUM_PARAM_SPEC rails), so evolution stays inside
    # table coverage.
    p.add_argument("--max-gap-um", type=float, default=40.0)
    p.add_argument("--per-tip-k", type=int, default=8)
    p.add_argument("--split-max-sites", type=int, default=50000)
    p.add_argument("--tip-to-tip-only", action="store_true")
    p.add_argument("--min-arm-cable-um", type=float, default=2.0)
    p.add_argument("--seed-depth-um", type=float, default=8.0)
    p.add_argument("--merge-max-sites", type=int, default=5000)
    p.add_argument("--max-per-label", type=int, default=8)
    p.add_argument("--skip-merge", action="store_true")
    p.add_argument("--skip-split", action="store_true")
    p.add_argument("--no-sweep", dest="sweep", action="store_false")
    args = p.parse_args(argv)

    if str(args.brain) not in os.path.basename(args.pkl):
        raise SystemExit(f"--pkl basename does not mention brain {args.brain}: "
                         f"{args.pkl}")
    out_dir = Path(args.out_dir) / str(args.brain)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the detector deliverable folders BEFORE the multi-GB payload load
    # so a wrong/incomplete folder fails in milliseconds, not minutes.
    split_script = split_joblib_p = merge_script = merge_joblib_p = None
    if not args.skip_split:
        split_script, split_joblib_p = _resolve_detector_dir(args.split_dir,
                                                             "split")
        log(f"[split] detector folder: {Path(args.split_dir).resolve()} "
            f"(script {split_script.name}, model {split_joblib_p.name})")
    if not args.skip_merge:
        merge_script, merge_joblib_p = _resolve_detector_dir(args.merge_dir,
                                                             "merge")
        log(f"[merge] detector folder: {Path(args.merge_dir).resolve()} "
            f"(script {merge_script.name}, model {merge_joblib_p.name})")

    import agentic_neuron_proofreader  # noqa: F401 — registers SkeletonGraph
    log(f"loading payload: {args.pkl} (large; needs a compute node)")
    with open(args.pkl, "rb") as f:
        payload = pickle.load(f)
    frag = payload["fragments_graph"]
    n_nodes = int(frag.number_of_nodes())
    log(f"fragments graph: {n_nodes} nodes")

    merge_mod = (_import_module(merge_script, "frozen_merge_site_detector")
                 if merge_script is not None else None)
    split_mod = (_import_module(split_script, "frozen_split_site_detector")
                 if split_script is not None else None)

    split_table = merge_table = None
    enum_meta = {}

    if not args.skip_split:
        bundle = _load_joblib(split_joblib_p)
        log(f"[split] enumerating loop candidates (max_gap={args.max_gap_um}µm, "
            f"per_tip_k={args.per_tip_k}, cap={args.split_max_sites})")
        sites, stats = ds.candidate_split_sites(
            frag, max_gap_um=args.max_gap_um, max_sites=args.split_max_sites,
            tip_to_shaft=not args.tip_to_tip_only, per_tip_k=args.per_tip_k,
            return_stats=True)
        enum_meta["split"] = {
            "max_gap_um": args.max_gap_um, "per_tip_k": args.per_tip_k,
            "max_sites": args.split_max_sites,
            "tip_to_shaft": not args.tip_to_tip_only,
            "enum_stats": {k: v for k, v in (stats or {}).items()
                           if isinstance(v, (int, float, str, bool))},
        }
        split_table = compute_split_table(payload, sites, split_mod, bundle)

    if not args.skip_merge:
        bundle = _load_joblib(merge_joblib_p)
        log(f"[merge] enumerating loop candidates (min_arm={args.min_arm_cable_um}"
            f"µm, cap={args.merge_max_sites}) — the whole-graph scan is slow")
        merge_sites = ds.candidate_merge_sites(
            frag, min_arm_cable_um=args.min_arm_cable_um,
            seed_depth_um=args.seed_depth_um, max_sites=args.merge_max_sites,
            max_per_label=args.max_per_label)
        enum_meta["merge"] = {
            "min_arm_cable_um": args.min_arm_cable_um,
            "seed_depth_um": args.seed_depth_um,
            "max_sites": args.merge_max_sites, "max_per_label": args.max_per_label,
        }
        merge_table = compute_merge_table(payload, merge_sites, merge_mod, bundle)

    sweep_summary = {}
    if args.sweep and (split_table is not None or merge_table is not None):
        try:
            lines, sweep_summary = sweep_tables(payload, split_table, merge_table,
                                                split_mod)
            (out_dir / "threshold_sweep.md").write_text("\n".join(lines))
            log(f"wrote {out_dir / 'threshold_sweep.md'}")
        except Exception as e:
            log(f"[sweep] failed ({e!r}) — tables are still written")

    if split_table is not None:
        split_table.to_pickle(out_dir / "split_scores.pkl")
        log(f"wrote {out_dir / 'split_scores.pkl'} ({len(split_table)} pairs)")
    if merge_table is not None:
        merge_table.to_pickle(out_dir / "merge_scores.pkl")
        log(f"wrote {out_dir / 'merge_scores.pkl'} ({len(merge_table)} labels)")

    meta = {
        "brain": str(args.brain),
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_pkl": os.path.abspath(args.pkl),
        "n_graph_nodes": n_nodes,
        "models": {
            "split": None if args.skip_split else {
                "dir": str(Path(args.split_dir).resolve()),
                "script": split_script.name,
                "script_sha256": _sha256(split_script),
                "file": split_joblib_p.name,
                "sha256": _sha256(split_joblib_p),
            },
            "merge": None if args.skip_merge else {
                "dir": str(Path(args.merge_dir).resolve()),
                "script": merge_script.name,
                "script_sha256": _sha256(merge_script),
                "file": merge_joblib_p.name,
                "sha256": _sha256(merge_joblib_p),
            },
        },
        "enumeration": enum_meta,
        "rows": {"split": None if split_table is None else int(len(split_table)),
                 "merge": None if merge_table is None else int(len(merge_table))},
        "threshold_sweep": sweep_summary,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"wrote {out_dir / 'meta.json'}")
    log("# OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

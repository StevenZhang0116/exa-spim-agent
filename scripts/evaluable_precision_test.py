#!/usr/bin/env python
"""Does native Precision@K rank scorers the same way as precision on GT-evaluable rows?

Label 0 in the detector-native tables mixes "GT says no error" with "no GT nearby".
A candidate is GT-evaluable when its segment(s) carry a GT node label: merge -> the
candidate segment; split -> both segments of the pair. Every positive is evaluable
by construction. For each scorer this script reports, per brain and kind:

    p_all@K      native Precision@K over all rows (what evolution promotes on)
    eval_frac@K  fraction of the top-K rows that are GT-evaluable
    p_eval@K     Precision@K after restricting the ranking to evaluable rows

Scorers: the frozen ``detector_score`` and each run's ``best_scorer.py`` component,
when it can be reproduced from the table alone. Code components run directly; v2
trained models run their own ``predict`` against the run's ``model_artifacts``.
Components that need columns absent from ``features.pkl`` (agent descriptors,
context-cache columns) are skipped and reported, never imputed.

Read-only; imports nothing from proofreader_evolve. Run with panda on a compute node:

    python -u scripts/evaluable_precision_test.py
"""
import argparse
import ast
import gc
import json
from pathlib import Path
import pickle
import sys
import tempfile

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent / "agentic-neuron-proofreader" / "src"))
import agentic_neuron_proofreader  # noqa: E402,F401  (registers SkeletonGraph for pickle)

BRAINS = ("789202", "794491", "794493", "794495", "802449")


# --- scorers ---------------------------------------------------------------------------
def run_components(run_dir):
    """kind -> component source from a best_scorer.py (its _COMPONENT_SOURCES literal)."""
    tree = ast.parse((run_dir / "best_scorer.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_COMPONENT_SOURCES"
                                                for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"{run_dir}: no _COMPONENT_SOURCES")


def trained_manifest(source):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_TRAINED_MODEL"
                                                for t in node.targets):
            return ast.literal_eval(node.value)
    return None


def make_scorer(source, run_dir):
    """Return (callable(features, kind) -> scores, required_columns or None, description)."""
    model = trained_manifest(source)
    if model is None:
        namespace = {}
        exec(compile(source, "<component>", "exec"), namespace)
        fn = namespace["score_candidates"]
        return (lambda features, kind: np.asarray(fn(features, {"kind": kind}), float)), None, "code"
    if model.get("version") != "agent-trained-model-v2":
        raise ValueError(f"unsupported trained model version {model.get('version')}")
    namespace = {}
    exec(compile(model["program"], "<program>", "exec"), namespace)
    artifact_dir = run_dir / "model_artifacts" / model["artifact_sha256"]
    params = model["config"]["parameters"]

    def predict(features, kind):
        # predict() may write nothing, but give it a private copy of the inputs only.
        X = features[[c for c in model["columns"] if c in features.columns]]
        return np.asarray(namespace["predict"](X, str(artifact_dir), params), float)
    return predict, list(model["columns"]), f"trained({','.join(model['training_brains'])})"


# --- metrics ---------------------------------------------------------------------------
def metrics(scores, labels, evaluable, k):
    order = np.argsort(-scores, kind="stable")
    top = order[:k]
    ev_order = order[evaluable[order]]
    top_ev = ev_order[:k]
    return {"p_all": float(labels[top].mean()), "eval_frac": float(evaluable[top].mean()),
            "p_eval": float(labels[top_ev].mean()) if len(top_ev) else float("nan"),
            "n_eval": int(evaluable.sum())}


def load_table(tables_dir, brain, kind):
    for meta in sorted((tables_dir / brain).glob("*/meta.json")):
        m = json.loads(meta.read_text())
        if m.get("provenance", {}).get("kind") == kind:
            d = meta.parent
            return (pd.read_pickle(d / "features.pkl"), pickle.load(open(d / "candidates.pkl", "rb")),
                    np.load(d / "evaluator_labels.npy").astype(int))
    raise FileNotFoundError(f"no {kind} table for {brain}")


def gt_segments(cache):
    with open(cache, "rb") as f:
        payload = pickle.load(f)
    labels = np.asarray(payload["gt_node_canonical_label"])
    del payload
    gc.collect()
    return {int(x) for x in np.unique(labels) if int(x) != 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brains", nargs="+", default=list(BRAINS))
    parser.add_argument("--k", type=int, default=2000)
    parser.add_argument("--runs", nargs="*", help="run directories (default: every precision_* run)")
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "notebooks" / "evaluable_precision_test")
    args = parser.parse_args()
    runs_root = PROJECT_ROOT / "proofreader_evolve" / "runs"
    run_dirs = [Path(r) for r in args.runs] if args.runs else sorted(runs_root.glob("precision_*"))
    run_dirs = [r for r in run_dirs if (r / "best_scorer.py").exists()]
    tables_dir = PROJECT_ROOT / "proofreader_evolve" / "feature_tables"
    args.out.mkdir(parents=True, exist_ok=True)

    scorers = {}  # (name, kind) -> (fn, required, desc)
    for kind in ("merge", "split"):
        scorers[("frozen_detector", kind)] = (lambda f, k: f["detector_score"].to_numpy(float), None, "detector")
    for run in run_dirs:
        for kind, source in run_components(run).items():
            try:
                scorers[(run.name, kind)] = make_scorer(source, run)
            except Exception as exc:
                print(f"skip {run.name}/{kind}: {type(exc).__name__}: {exc}", flush=True)

    rows = []
    for brain in args.brains:
        segs = gt_segments(PROJECT_ROOT / "cache" / f"dataset_cache_{brain}_mcl{args.mcl}_add.pkl")
        for kind in ("merge", "split"):
            features, candidates, labels = load_table(tables_dir, brain, kind)
            if kind == "merge":
                evaluable = np.array([int(c["segment_id"]) in segs for c in candidates])
            else:
                evaluable = np.array([int(c["segment_id_a"]) in segs and int(c["segment_id_b"]) in segs
                                      for c in candidates])
            assert labels[~evaluable].sum() == 0, "a positive outside the evaluable set"
            for (name, k), (fn, required, desc) in scorers.items():
                if k != kind:
                    continue
                missing = sorted(set(required or []) - set(features.columns))
                if missing:
                    rows.append({"brain": brain, "kind": kind, "scorer": name, "type": desc,
                                 "status": f"skipped: needs {len(missing)} absent columns "
                                           f"(e.g. {missing[0]})"})
                    continue
                try:
                    scores = fn(features, kind)
                    if scores.shape != (len(features),) or not np.isfinite(scores).all():
                        raise ValueError("non-finite or misaligned scores")
                    m = metrics(scores, labels, evaluable, args.k)
                    rows.append({"brain": brain, "kind": kind, "scorer": name, "type": desc,
                                 "status": "ok", **m})
                except Exception as exc:
                    rows.append({"brain": brain, "kind": kind, "scorer": name, "type": desc,
                                 "status": f"error: {type(exc).__name__}: {str(exc)[:120]}"})
            print(f"{brain}/{kind}: rows {len(labels)}, evaluable {evaluable.mean():.3f}", flush=True)
            del features, candidates
            gc.collect()

    table = pd.DataFrame(rows)
    table.to_csv(args.out / f"results_k{args.k}.csv", index=False)
    ok = table[table.status == "ok"]
    pd.set_option("display.width", 220)
    print("\nPer brain/kind (ok rows)\n" + ok.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nSkipped / errors\n" + table[table.status != "ok"][["brain", "kind", "scorer", "status"]]
          .drop_duplicates(["kind", "scorer", "status"]).to_string(index=False))
    # Do the two metrics order the scorers the same way? (mean over brains, per kind)
    for kind in ("merge", "split"):
        sub = ok[ok.kind == kind].groupby("scorer")[["p_all", "p_eval", "eval_frac"]].mean()
        if len(sub) > 1:
            sub["rank_all"] = sub.p_all.rank(ascending=False)
            sub["rank_eval"] = sub.p_eval.rank(ascending=False)
            print(f"\n{kind}: mean over brains, scorer ranking by each metric\n"
                  + sub.sort_values("p_all", ascending=False).to_string(float_format=lambda v: f"{v:.3f}"))
            print(f"Spearman(p_all, p_eval) = {sub.p_all.corr(sub.p_eval, method='spearman'):.3f}; "
                  f"Spearman(p_all, eval_frac) = {sub.p_all.corr(sub.eval_frac, method='spearman'):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

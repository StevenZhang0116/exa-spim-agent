"""Compact TRAIN-only reviser input; full evaluator reports remain unchanged."""

import json


FEEDBACK_VERSION = "stratified-train-v6-per-kind-objective"
MAX_FEEDBACK_BYTES = 128_000  # 48 KB -> 96 KB (2026-10-05) -> 128 KB (2026-10-07, with the SDK file-read cap raised);
# in run precision_20261007_112554 six of ten generations were pruned to one or two examples per group at 96 KB.
METRICS = ("precision", "tp", "fp", "requested_k", "effective_k", "pool_size", "n_labeled", "n_unlabeled",
           "positives", "recall", "in_mean")
SELECTION_METRICS = ("protocol", "fold_budgets", "fold_tp", "fold_held_rows")


def metrics_only(report):
    if report is None:
        return None
    return {"per_kind_precision": report["per_kind_precision"],
            **({"mode": report["mode"]} if "mode" in report else {}),
            **({"aggregation": report["aggregation"]} if "aggregation" in report else {}),
            **({"holdout_cells": report["holdout_cells"], "holdout_precision": report["holdout_precision"]}
               if report.get("holdout_cells") else {}),
            "cells": {name: {key: cell[key] for key in (*METRICS, *SELECTION_METRICS) if key in cell}
                      | ({'local_context': cell['local_context']} if 'local_context' in cell else {})
                      | ({'image_context': cell['image_context']} if 'image_context' in cell else {})
                      for name, cell in report["cells"].items()}}


def _display_number(value):
    # Display precision only: scoring still uses the original table values.
    return None if value is None else float(format(value, ".6g"))


def render_compact_json(value, depth=0):
    """Readable JSON with numeric vectors on one line, instead of one line/value."""
    if isinstance(value, dict) and value:
        entries = ["  " * (depth + 1) + json.dumps(key) + ": " + render_compact_json(item, depth + 1)
                   for key, item in value.items()]
        return "{\n" + ",\n".join(entries) + "\n" + "  " * depth + "}"
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def write_train_feedback(path, parent_train, history, budgets, *, target_kind=None, memory=(),
                         statistics_file=None, report_role='parent', selection=None, protocol=None,
                         branch=None):
    """Bound the read size, retaining all feature names for the selected examples.

    Historical attempts carry metrics and diagnostic TRAIN comparisons, never repeated
    examples or validation diagnostics. Reduce examples symmetrically by group
    before dropping old history, then drop whole cells' examples (fewest GT positives
    first, metrics kept), and disclose that sampling in the report.
    When a selection report is supplied, examples and ranking deltas come from the
    official selection measurement (out-of-fold on selection brains); in-sample
    TRAIN numbers remain visible as diagnostics only.
    """
    example_source = selection if selection is not None else parent_train
    # The agent-facing `budgets` is one literal K per kind: the kind's cap, or for a kind
    # without a cap the largest K resolved on a TRAIN pool (2026-10-26); each pool's resolved K
    # is its requested_k, and the rule that produced it is spelled out in k_rule.
    caps = (budgets.kind_budgets(parent_train["cells"]) if hasattr(budgets, "kind_budgets")
            else dict(budgets))
    k_rule = budgets.describe() if hasattr(budgets, "describe") else None
    attempts = [{"generation": h["generation"],
                 "train": metrics_only(h["train"]), "train_gate": h.get("train_gate")}
                for h in history[-5:]]
    example_cells = [name for name in example_source["cells"]
                     if not target_kind or name.rsplit('/', 1)[-1] == target_kind]
    # Last resort once one example per group still exceeds the budget: drop whole cells' examples,
    # fewest GT positives first (2026-10-27: five selection cells x about 290 frozen feature
    # columns need 155 KB for one example per group). The cells keep their metrics and ranking_delta.
    drop_order = sorted(example_cells, key=lambda name: example_source["cells"][name].get("positives") or 0)
    while True:
        omitted = [name for name in drop_order if name not in example_cells]
        for per_class, prune in ((4, False), (4, True), (2, True), (1, True)):
            report = {
                "format": FEEDBACK_VERSION,
                "reading_guide": (
                    "TRAIN only. Labels are three-valued: 1 = error, 0 = GT-confirmed no error, NaN = not "
                    "judgeable by GT (n_unlabeled). Top-K, precision and recall use GT-labeled rows only "
                    "(n_labeled); NaN rows are scored but never ranked or counted, and every example is a "
                    "labeled row. Feature vectors and candidate_refs align with labels and scores by index. "
                    "Use inspect_candidate with a candidate_ref to inspect TRAIN fragment geometry. "
                    "Each example has a sampling group: selected positives/label0, just-below-K boundary "
                    "positives/label0, and missed positives; candidate reports also show gained/lost positives. "
                    "These are bounded diagnostic examples, not a representative dataset. "
                    "Do not infer global score/label correlation from selected errors. Preserve current true positives. "
                    "Numbers are rounded to 6 significant digits for display only; null means missing. "
                    "All feature columns of the selected examples are included. When examples_omitted_cells "
                    "is non-empty, those cells' examples were dropped to fit the read budget (fewest GT positives "
                    "first); their metrics and ranking_delta remain. "
                    "budgets are one K per kind (the kind's cap, or for a kind without a cap the largest K "
                    "resolved on a TRAIN pool); requested_k is the requested budget and effective_k is the "
                    "number of GT-labeled rows selected (k_rule explains the budget); in_mean says whether the "
                    "pool counts toward the objective. per_kind_precision combines the counted cells separately for each kind by the "
                    "k_rule's aggregation (positive-weighted: hits summed over brains / effective K summed over brains; "
                    "equal-brain: mean of the cells). Optimize the target kind's selection precision. When holdout_cells are listed, "
                    "those selection brains were held out whole (a model that never saw the brain scored them) and "
                    "holdout_precision summarizes only those cells under the same aggregation; use it to diagnose "
                    "whether selection gains transfer across brains. "
                    "The scorer contract is in the system prompt and rules.md."
                    + (" Examples, ranking_delta and the official selection score follow selection_protocol. "
                       "grouped_oof uses out-of-fold scores on selection brains; in_sample uses full-TRAIN scores. "
                       f"Branch ranking uses {report_role}_selection's metric; {report_role}_train records "
                       "the separate full-TRAIN measurement." if selection is not None else "")
                    + (" Examples and ranking_delta describe the assigned search branch (search_branch); "
                       f"{report_role}_selection/{report_role}_train describe the accepted scorer, and ranking_delta "
                       "is the branch minus the accepted scorer." if branch is not None else "")
                ),
                **({"search_branch": branch} if branch is not None else {}),
                "budgets": caps,
                "k_rule": k_rule,
                "target_kind": target_kind,
                f"{report_role}_train": metrics_only(parent_train),
                **({f"{report_role}_selection": metrics_only(selection),
                    "selection_protocol": protocol} if selection is not None else {}),
                "examples_per_group_limit": per_class,
                "columns_pruned": prune,
                "examples_omitted_cells": omitted,
                "examples": {},
                "ranking_delta": {name: c.get('ranking_delta') for name, c in example_source['cells'].items()},
                "attempts": attempts,
                "relevant_experiments": list(memory),
                "feature_statistics_file": str(statistics_file) if statistics_file else None,
                "history_attempts_omitted": len(history) - len(attempts),
            }
            for name in example_cells:
                cell = example_source["cells"][name]
                available = cell.get("training_examples", [])
                grouped = {}
                for e in available:
                    group = e.get('group', 'selected_label0' if e['label'] == 0 else 'missed_positive')
                    grouped.setdefault(group, []).append(e)
                examples = [e for group in grouped.values() for e in group[:per_class]]
                features = sorted({key for e in examples for key in e["features"]
                                   if not prune or e["features"].get(key) is not None})
                report["examples"][name] = {
                    "available": len(available), "included": len(examples),
                    "labels": [e["label"] for e in examples],
                    "candidate_refs": [e.get('candidate_ref') for e in examples],
                    "groups": [e.get('group', 'selected_label0' if e['label'] == 0 else 'missed_positive') for e in examples],
                    "group_sizes": cell.get('group_sizes', {}),
                    "scores": [_display_number(e["score"]) for e in examples],
                    "features": {key: [_display_number(e["features"].get(key)) for e in examples]
                                 for key in features},
                }
            text = render_compact_json(report) + "\n"
            size = len(text.encode("utf-8"))
            if size <= MAX_FEEDBACK_BYTES:
                path.write_text(text, encoding="utf-8")
                return {"bytes": size, "examples_per_group_limit": per_class,
                        "history_attempts": len(attempts), "examples_omitted_cells": omitted}
        if memory:
            memory = memory[:-1]
            continue
        if attempts:
            attempts = attempts[1:]
            continue
        if len(example_cells) > 1:
            dropped = next(name for name in drop_order if name in example_cells)
            example_cells = [name for name in example_cells if name != dropped]
            continue
        raise ValueError(f"Compact TRAIN feedback exceeds the {MAX_FEEDBACK_BYTES // 1000} KB read budget even with one example "
                         "per group from a single cell; reduce the feature columns or training brains in this run")

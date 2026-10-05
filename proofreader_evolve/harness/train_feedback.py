"""Compact TRAIN-only reviser input; full evaluator reports remain unchanged."""

import json


MAX_FEEDBACK_BYTES = 48_000
METRICS = ("precision", "tp", "fp", "requested_k", "effective_k", "pool_size", "positives", "recall")
SELECTION_METRICS = ("protocol", "fold_budgets", "fold_tp", "fold_held_rows")


def metrics_only(report):
    if report is None:
        return None
    return {"macro_precision": report["macro_precision"],
            **({"mode": report["mode"]} if "mode" in report else {}),
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
                         statistics_file=None, report_role='parent', selection=None, protocol=None):
    """Bound the read size, retaining all feature names for the selected examples.

    Historical attempts carry metrics and diagnostic TRAIN comparisons, never repeated
    examples or validation diagnostics. Reduce examples symmetrically by group
    before dropping old history, and disclose that sampling in the report.
    When a selection report is supplied, examples and ranking deltas come from the
    official selection measurement (out-of-fold on selection brains); in-sample
    TRAIN numbers remain visible as diagnostics only.
    """
    example_source = selection if selection is not None else parent_train
    attempts = [{"generation": h["generation"],
                 "train": metrics_only(h["train"]), "train_gate": h.get("train_gate")}
                for h in history[-5:]]
    while True:
        for per_class, prune in ((4, False), (4, True), (2, True), (1, True)):
            report = {
                "format": "stratified-train-v4" if selection is not None else "stratified-train-v3",
                "reading_guide": (
                    "TRAIN only. Feature vectors and candidate_refs align with labels and scores by index. "
                    "Use inspect_candidate with a candidate_ref to inspect TRAIN fragment geometry. "
                    "Each example has a sampling group: selected positives/label0, just-below-K boundary "
                    "positives/label0, and missed positives; candidate reports also show gained/lost positives. "
                    "These are bounded diagnostic examples, not a representative dataset. "
                    "Do not infer global score/label correlation from selected errors. Preserve current true positives. "
                    "Numbers are rounded to 6 significant digits for display only; null means missing. "
                    "All feature columns of the selected examples are included. "
                    "The scorer contract is in the system prompt and rules.md."
                    + (" Examples, ranking_delta and the official selection score come from "
                       "selection_protocol (out-of-fold on selection brains, not in-sample). "
                       f"{report_role}_train lists in-sample TRAIN metrics as diagnostics only; "
                       "they do not rank branches." if selection is not None else "")
                ),
                "budgets": budgets,
                "target_kind": target_kind,
                f"{report_role}_train": metrics_only(parent_train),
                **({f"{report_role}_selection": metrics_only(selection),
                    "selection_protocol": protocol} if selection is not None else {}),
                "examples_per_group_limit": per_class,
                "columns_pruned": prune,
                "examples": {},
                "ranking_delta": {name: c.get('ranking_delta') for name, c in example_source['cells'].items()},
                "attempts": attempts,
                "relevant_experiments": list(memory),
                "feature_statistics_file": str(statistics_file) if statistics_file else None,
                "history_attempts_omitted": len(history) - len(attempts),
            }
            for name, cell in example_source["cells"].items():
                if target_kind and name.rsplit('/', 1)[-1] != target_kind:
                    continue
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
                        "history_attempts": len(attempts)}
        if memory:
            memory = memory[:-1]
            continue
        if not attempts:
            raise ValueError(f"Compact TRAIN feedback exceeds the {MAX_FEEDBACK_BYTES // 1000} KB read budget even with one example "
                             "per group; reduce the number of training brains in this run")
        attempts = attempts[1:]

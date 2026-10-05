"""Durable, human-readable traces of observable reviser actions and evaluation."""

from datetime import datetime
import difflib
import json
import hashlib


def image_safe_content(value):
    """Retain image provenance, without duplicating base64 pixels in text logs."""
    if isinstance(value, list):
        return [image_safe_content(v) for v in value]
    if isinstance(value, dict):
        if value.get('type') == 'image':
            source = value.get('source') or {}
            data = value.get('data', source.get('data', ''))
            return {'type': 'image', 'mimeType': value.get('mimeType', source.get('media_type')),
                    'encoded_bytes': len(data), 'encoded_sha256': hashlib.sha256(data.encode()).hexdigest(),
                    'note': 'Pixels omitted from text trace; original preview is in image_inspections/'}
        return {k: image_safe_content(v) for k, v in value.items()}
    return value


# All events remain durable in both trace files. The terminal is a progress
# surface; raw SDK payloads and repetitive measurements stay in the trace files.
DETAIL_EVENTS = {'sdk_status', 'agent_text', 'user_message', 'tool_call', 'tool_result',
                 'agent_result', 'candidate_formula', 'feedback_ready', 'metrics',
                 'candidate_cache_hit'}


class Trajectory:
    def __init__(self, directory):
        self.directory = directory

    def emit(self, event, message, **details):
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        record = {"timestamp": timestamp, "event": event, "message": message, **details}
        # Open/close each append so a stopped run retains all received events.
        with (self.directory / "trajectory.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        heading = f"[{timestamp}] [{self.directory.name}] {event}: {message}"
        with (self.directory / "trajectory.txt").open("a", encoding="utf-8") as stream:
            stream.write(heading + "\n")
            if details:
                stream.write(json.dumps(details, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        if event in DETAIL_EVENTS:
            return
        if event == 'metrics_cell' and str(details.get('label', '')).startswith(('TRAIN ', 'Parent ')):
            return
        # Full tool payloads stay in the trace; console output stays readable.
        print(heading[:2000] + (" … [full event in trajectory.txt]" if len(heading) > 2000 else ""),
              flush=True)

    def sdk_message(self, message):
        from claude_agent_sdk import (AssistantMessage, UserMessage, SystemMessage,
                                     ResultMessage, TextBlock, ToolUseBlock, ToolResultBlock)

        if isinstance(message, (AssistantMessage, UserMessage)):
            if isinstance(message, AssistantMessage) and getattr(message, "error", None):
                self.emit("agent_error", str(message.error))
            blocks = message.content
            if isinstance(blocks, str):
                self.emit("user_message", blocks)
                return
            for block in blocks:
                if isinstance(block, TextBlock):
                    role = "agent_text" if isinstance(message, AssistantMessage) else "user_message"
                    self.emit(role, block.text)
                elif isinstance(block, ToolUseBlock):
                    path = block.input.get("file_path", "")
                    self.emit("tool_call", f"{block.name} {path} (id={block.id})",
                              tool_use_id=block.id, tool=block.name, input=block.input)
                elif isinstance(block, ToolResultBlock):
                    status = "error" if block.is_error else "ok"
                    self.emit("tool_result", f"{block.tool_use_id}: {status}",
                              tool_use_id=block.tool_use_id, is_error=block.is_error,
                              content=image_safe_content(block.content))
            # Thinking/signature blocks are deliberately not serialized.
        elif isinstance(message, SystemMessage):
            # Do not dump arbitrary session metadata (environment, credentials).
            self.emit("sdk_status", message.subtype)
        elif isinstance(message, ResultMessage):
            self.emit("agent_result", message.result or message.subtype,
                      is_error=message.is_error, subtype=message.subtype,
                      session_id=message.session_id, num_turns=message.num_turns,
                      duration_ms=message.duration_ms, usage=message.usage,
                      cost_usd=message.total_cost_usd,
                      errors=getattr(message, "errors", None))


def metric_summary(report):
    """Keep progress logs small; detailed training examples live in feedback."""
    return {"macro_precision": report["macro_precision"],
            "aggregation": report.get("aggregation"),
            "per_brain_precision": report.get("per_brain_precision", {}),
            "cells": {name: {k: cell[k] for k in (
                "pool_size", "positives", "requested_k", "effective_k",
                "tp", "fp", "precision", "recall")}
                      for name, cell in report["cells"].items()}}


def log_metrics(trace, label, report, parent=None):
    """Log pool-positive counts and Top-K hits for each brain and error kind.

    Discoverable errors are positive candidate rows in the fixed pool, not all
    GT errors or all GT pairs with available fragments. Emit a separate line per
    cell so console truncation cannot hide datasets when many brains are used.
    """
    summary = metric_summary(report)
    lines = [f"{label}: macro Precision@K={report['macro_precision']:.4f}"]
    if parent is not None:
        summary["delta_macro_precision"] = report["macro_precision"] - parent["macro_precision"]
        lines[0] += f" (delta={summary['delta_macro_precision']:+.4f})"
    for brain, value in summary["per_brain_precision"].items():
        lines.append(f"{brain} mean={value:.4f}")
    for name, cell in summary["cells"].items():
        if parent is not None:
            cell["delta_precision"] = cell["precision"] - parent["cells"][name]["precision"]
            cell["delta_tp"] = cell["tp"] - parent["cells"][name]["tp"]
    trace.emit("metrics", "; ".join(lines), label=label, metrics=summary)
    for name, cell in summary["cells"].items():
        brain, kind = name.rsplit("/", 1)
        recall = f"{cell['recall']:.4f}" if cell["recall"] is not None else "N/A (no pool positives)"
        line = (f"{label} | brain={brain} kind={kind} | pool_candidates={cell['pool_size']} "
                f"discoverable_pool_positives={cell['positives']} | "
                f"top_k_hits={cell['tp']} effective_k={cell['effective_k']} "
                f"requested_k={cell['requested_k']} | Precision@K={cell['precision']:.4f} "
                f"Recall@K={recall}")
        if parent is not None:
            line += f" delta={cell['delta_precision']:+.4f} delta_hits={cell['delta_tp']:+d}"
        trace.emit("metrics_cell", line, label=label, brain=brain, kind=kind,
                   positive_scope="fixed_candidate_pool", metrics=cell)


def save_diffs(directory):
    """Preserve even partial edits when revision or evaluation fails."""
    for name, parent in (("scorer.py", "parent_scorer.py"), ("rules.md", "parent_rules.md"),
                         ("training.py", "parent_training.py")):
        if not (directory / parent).is_file():
            continue
        candidate = directory / name
        after = candidate.read_text() if candidate.exists() else ""
        diff = difflib.unified_diff((directory / parent).read_text().splitlines(keepends=True),
                                    after.splitlines(keepends=True), fromfile=parent, tofile=name)
        (directory / (name + ".diff")).write_text("".join(diff))

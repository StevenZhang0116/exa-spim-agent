#!/usr/bin/env python3
"""Which of the information handed to a reviser session reached it, and which did it write about?

Question (2026-11-13, after arXiv 2608.10424, whose agents ignored the data analysis they were
given): every generation the host hands the fresh session a prompt, a set of files (feedback,
failure cases, hypothesis memory, candidate pool, feature statistics, guides) and the replies of
its tools. Which of these did the sessions of finished runs open, how much of each tool reply
reached them, and which topics do the sessions' own notes mention?

Post-hoc and read-only: the script reads the records of finished runs and changes nothing in a
run directory or in the workflow. No language model is involved; everything is counted from

* ``gen*/trajectory.jsonl`` (every tool call of the session with its input, every tool reply
  with the text the session received, and the session's own messages),
* ``gen*/reviser_prompt.txt``, the final ``gen*/proposal.json`` and ``gen*/rules.md``,
* ``ledger.jsonl`` (the generation's kind, mode and outcome) and ``manifest.json``.

Four measurements per generation:

1. files: which host-written files the session opened with Read, and whether the prompt named
   them;
2. tool replies: the size of each reply as the session received it. A reply above the SDK's
   size limit arrives as a 2 KB preview plus the path of a file holding the rest
   (``<persisted-output>``); the script counts those and whether the session then read the file;
3. feedback fields: for every reply of a training tool (``train_classifier``, ``evaluate_train``,
   ``search_parameters``), whether the name of a feedback field occurs in the full reply (read
   from the persisted file while it is still on disk), in the preview, and in what the session
   received in the end (the preview plus whatever it read of the persisted file). A field that
   does not occur in the received text was not in the session's context for that reply;
4. mentions: whether the session's own prose (its messages, and what it wrote to
   ``proposal.json`` and ``rules.md``) contains terms of a topic. Code it wrote is left out, so
   that a variable name does not count as a mention.

One behavioural link is exact: the candidate references a session passed to ``probe_descriptor``
are compared with the references listed in the generation's ``failure_cases.json``.

Limits. "Opened" and "in context" are facts of the log. "Mentions" is a term search: it can
miss a paraphrase and count an unrelated use of a word, and a mention does not show that the
information changed the submitted program. A file the session did not open may still have been
unnecessary. Generation outcomes are listed beside the counts for reading only; thirty
generations do not support a comparison of outcomes between sessions that opened a file and
sessions that did not.

Usage (a compute node; about a second per run)::

    python notebooks/feedback_use_audit.py proofreader_evolve/runs/RUN [RUN ...]

Outputs in ``notebooks/feedback_use_audit/`` (gitignored): ``audit.json`` (one record per
generation) and ``summary.md``.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics

REPO = Path(__file__).resolve().parents[1]

# Host-written files of a generation directory, in the order of the report.
FILES = ("train_feedback.json", "search_plan.json", "research_status.json", "hypothesis_memory.json",
         "failure_cases.json", "candidate_pool.json", "feature_statistics.json", "image_evidence.json",
         "model_environment.json", "classifier_guide.md", "descriptor_guide.md", "feature_discovery_guide.md",
         "local_context_guide.md", "image_context_guide.md", "volume_analysis_guide.md")

# Feedback fields searched for in the replies of the training tools, in the order they are discussed.
FIELDS = ("target_objective", "selection_gate", "resampling", "train_gate", "selection_comparison", "per_brain",
          "per_kind_by_family", "per_kind_statistics", "ranking_delta", "examples", "feature_evidence")
TRAINING_TOOLS = ("train_classifier", "evaluate_train", "search_parameters")

# Topics searched for in a session's own prose. The patterns are deliberately narrow.
TOPICS = {
    "outer result of earlier submissions": r"outer|promot|reject|significantly_worse|within_noise|gain_too_small|not[_ ]measured|held-?out",
    "TRAIN noise interval": r"interval|resampling|bootstrap|within (?:TRAIN |resampling |selection )*noise|beyond (?:resampling )?noise|selection noise",
    "per-brain results": r"per[- ]brain|hold-?out brain|{brains}",
    "candidate families": r"tip[_ -]to[_ -]shaft|per_kind_by_family|within[- ]junction|junction famil|by famil",
    "failure cases": r"failure[_ ]cases?|missed[_ ]positives?|selected_label0|false (?:positive|negative)s?|\bpair\d\d\b",
    "feature statistics": r"feature[_ ]statistics",
    "an earlier generation": None,  # an experiment id of another generation; see mentions()
}
PROSE_TARGETS = ("proposal.json", "rules.md")
PERSISTED = re.compile(r"Output too large \(([\d.]+)\s*([KM]?B)\)\. Full output saved to: (\S+)")
UNITS = {"B": 1, "KB": 1024, "MB": 1024 ** 2}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def events(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def as_text(value):
    return value if isinstance(value, str) else json.dumps(value)


def find_key(value, key):
    """The first string stored under ``key`` anywhere in a nested record."""
    if isinstance(value, dict):
        if isinstance(value.get(key), str):
            return value[key]
        value = list(value.values())
    if isinstance(value, list):
        for item in value:
            found = find_key(item, key)
            if found:
                return found
    return None


def outcome_of(row):
    """The generation's outcome in the words of the current workflow, from the ledger row."""
    reason = str(row.get("reason", ""))
    if row.get("accepted"):
        return "promoted"
    if row.get("failure_stage") or reason.startswith("Failed"):
        return f"failed ({row.get('failure_stage') or 'stage not recorded'})"
    if row.get("submission_within_noise") or "within TRAIN resampling noise" in reason:
        return "not_measured"
    if row.get("submission_repeat") or "Repeated submission" in reason:
        return "repeated"
    if "Unchanged submission" in reason:
        return "unchanged"
    rejection = find_key(row, "rejection")
    if rejection:
        return rejection
    if "significantly worse" in reason:
        return "significantly_worse"
    if "rejected by the LLM judge" in reason:
        return "rejected by the judge"
    return "rejected" if "rejected" in reason else "no submission"


def mentions(prose, generation, brains):
    found = {}
    for topic, pattern in TOPICS.items():
        if pattern is None:
            ids = {int(number) for number in re.findall(r"\bgen0*(\d+)(?:/attempt\d+)?\b", prose)}
            found[topic] = bool(ids - {generation})
        else:
            found[topic] = bool(re.search(pattern.replace("{brains}", "|".join(brains)), prose, re.IGNORECASE))
    return found


def audit_generation(run, directory, row, brains):
    generation = int(directory.name[3:])
    log = events(directory / "trajectory.jsonl")
    calls = {event["tool_use_id"]: event for event in log if event.get("event") == "tool_call"}
    replies = {event["tool_use_id"]: event for event in log if event.get("event") == "tool_result"}
    prompt = (directory / "reviser_prompt.txt").read_text() if (directory / "reviser_prompt.txt").exists() else ""

    opened = defaultdict(lambda: {"reads": 0, "chars": 0, "partial": 0})
    read_text = defaultdict(str)  # path -> everything a Read of it returned
    for use, call in calls.items():
        if call.get("tool") != "Read":
            continue
        path = Path(str(call["input"].get("file_path", "")))
        read_text[str(path)] += as_text(replies.get(use, {}).get("content", ""))
        if path.parent == directory.resolve() or path.parent.name == directory.name:
            entry = opened[path.name]
            entry["reads"] += 1
            entry["chars"] += len(as_text(replies.get(use, {}).get("content", "")))
            entry["partial"] += int("limit" in call["input"] or "offset" in call["input"])
    files = {}
    for name in FILES:
        path = directory / name
        files[name] = {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None,
                       "named_in_prompt": name in prompt, **opened.get(name, {"reads": 0, "chars": 0, "partial": 0})}

    tools = defaultdict(lambda: {"calls": 0, "received_chars": [], "truncated": 0, "full_bytes": [], "file_read": 0})
    training_replies = []
    for use, call in calls.items():
        name = str(call.get("tool", "")).replace("mcp__training__", "")
        if name in ("Read", "Edit", "Write"):
            continue
        entry = tools[name]
        entry["calls"] += 1
        content = as_text(replies.get(use, {}).get("content", ""))
        entry["received_chars"].append(len(content))
        match = PERSISTED.search(content)
        full, later = (None if match else content), ""
        if match:
            entry["truncated"] += 1
            entry["full_bytes"].append(int(float(match.group(1)) * UNITS[match.group(2)]))
            entry["file_read"] += int(match.group(3) in read_text)
            later = read_text.get(match.group(3), "")
            try:
                full = Path(match.group(3)).read_text()
            except OSError:
                full = None  # the SDK's persisted file is gone; the full reply is unknown
        if name in TRAINING_TOOLS:
            training_replies.append({
                "tool": name, "truncated": bool(match), "file_read": bool(later),
                "full": None if full is None else {field: field in full for field in FIELDS},
                "preview": {field: field in content for field in FIELDS},
                "received": {field: field in content or field in later for field in FIELDS}})

    prose = [str(event.get("message", "")) for event in log if event.get("event") == "agent_text"]
    for call in calls.values():
        target = Path(str(call.get("input", {}).get("file_path", ""))).name
        if call.get("tool") == "Write" and target in PROSE_TARGETS:
            prose.append(str(call["input"].get("content", "")))
        if call.get("tool") == "Edit" and target in PROSE_TARGETS:
            prose.append(str(call["input"].get("new_string", "")))
    for name in PROSE_TARGETS:
        if (directory / name).exists():
            prose.append((directory / name).read_text())
    proposal = read_json(directory / "proposal.json", {}) or {}

    cases = read_json(directory / "failure_cases.json", {}) or {}
    case_refs = {case.get("candidate_ref") for pair in cases.get("pairs", []) for case in pair.get("cases", [])} - {None}
    probed = set()
    probes = 0
    for call in calls.values():
        if str(call.get("tool", "")).endswith("probe_descriptor"):
            probes += 1
            refs = call["input"].get("candidate_refs") or []
            probed.update(refs if isinstance(refs, list) else re.findall(r"ctx:[0-9a-f]+:\w+:\d+", str(refs)))

    return {"run": run.name, "generation": generation, "kind": row.get("target_kind"), "mode": row.get("search_mode"),
            "outcome": outcome_of(row), "prompt_chars": len(prompt), "tool_calls": len(calls),
            "files": files, "tools": {name: dict(entry) for name, entry in tools.items()},
            "training_replies": training_replies,
            "mentions": mentions("\n".join(prose), generation, brains),
            "prose_chars": sum(len(part) for part in prose),
            "handoff_written": bool(proposal.get("handoff")),
            "memory_searches": tools.get("search_memory", {}).get("calls", 0),
            "failure_cases": {"listed_refs": len(case_refs), "probe_calls": probes, "probed_refs": len(probed),
                              "probed_from_failure_cases": len(probed & case_refs)}}


def median(values):
    return statistics.median(values) if values else None


def kb(value):
    return "n/a" if value is None else f"{value / 1024:.1f}"


def write_summary(records, path):
    total = len(records)
    runs = sorted({record["run"] for record in records})
    short = {run: run.split("_", 3)[-1] for run in runs}  # precision_<date>_<time>_<id>
    per_run = {run: [record for record in records if record["run"] == run] for run in runs}

    def share(predicate, group=records):
        return f"{sum(1 for record in group if predicate(record))}/{len(group)}"

    lines = ["# Use of the information handed to reviser sessions", "",
             f"{total} generations of {len(runs)} runs ({', '.join(short.values())}). Counted from the session logs; "
             "no language model. See the module docstring of `notebooks/feedback_use_audit.py` for definitions and limits.", "",
             "## 1. Files opened", "",
             "| file | size, KB (median) | named in the prompt | opened, generations | " + " | ".join(short.values())
             + " | received when opened, KB (median) | reads with limit/offset |",
             "|---|---|---|---|" + "---|" * len(runs) + "---|---|"]
    for name in FILES:
        present = [record for record in records if record["files"][name]["exists"]]
        if not present:
            continue
        lines.append(
            f"| `{name}` | {kb(median([record['files'][name]['bytes'] for record in present]))} | "
            f"{share(lambda r: r['files'][name]['named_in_prompt'], present)} | "
            f"{share(lambda r: r['files'][name]['reads'] > 0, present)} | "
            + " | ".join(share(lambda r: r["files"][name]["reads"] > 0,
                               [record for record in per_run[run] if record["files"][name]["exists"]]) for run in runs)
            + f" | {kb(median([record['files'][name]['chars'] for record in present if record['files'][name]['reads']]))}"
            + f" | {sum(record['files'][name]['partial'] for record in present)} |")

    lines += ["", "## 2. Tool replies as the session received them", "",
              "| tool | calls | received, KB (median) | replies cut to a preview | full size of those, KB (median) | "
              "persisted file then read |", "|---|---|---|---|---|---|"]
    names = Counter()
    for record in records:
        for name, entry in record["tools"].items():
            names[name] += entry["calls"]
    for name, count in names.most_common():
        entries = [record["tools"][name] for record in records if name in record["tools"]]
        received = [value for entry in entries for value in entry["received_chars"]]
        truncated = sum(entry["truncated"] for entry in entries)
        full = [value for entry in entries for value in entry["full_bytes"]]
        lines.append(f"| `{name}` | {count} | {kb(median(received))} | {truncated} | {kb(median(full))} | "
                     f"{sum(entry['file_read'] for entry in entries)} of {truncated} |")

    replies = [reply for record in records for reply in record["training_replies"]]
    known = [reply for reply in replies if reply["full"] is not None]
    lines += ["", "## 3. Feedback fields of the training replies", "",
              f"{len(replies)} replies of the training tools ({', '.join(TRAINING_TOOLS)}); {sum(r['truncated'] for r in replies)} "
              f"cut to a preview, of which the session read the persisted file for {sum(r['file_read'] for r in replies)}. "
              f"The full reply is still on disk for {len(known)}.", "",
              "| field | in the full reply | in the preview | received by the session (preview or file read) |",
              "|---|---|---|---|"]
    for name in FIELDS:
        lines.append(f"| `{name}` | {sum(r['full'][name] for r in known)}/{len(known)} | "
                     f"{sum(r['preview'][name] for r in replies)}/{len(replies)} | "
                     f"{sum(r['received'][name] for r in replies)}/{len(replies)} |")

    lines += ["", "## 4. Topics in the sessions' own prose", "",
              "| topic | mentioned, generations | merge | split | " + " | ".join(short.values()) + " |",
              "|---|---|---|---|" + "---|" * len(runs)]
    for topic in TOPICS:
        lines.append(f"| {topic} | {share(lambda r: r['mentions'][topic])} | "
                     + " | ".join(share(lambda r: r["mentions"][topic], [x for x in records if x["kind"] == kind])
                                  for kind in ("merge", "split")) + " | "
                     + " | ".join(share(lambda r: r["mentions"][topic], per_run[run]) for run in runs) + " |")

    with_cases = [record for record in records if record["failure_cases"]["listed_refs"]]
    probing = [record for record in with_cases if record["failure_cases"]["probe_calls"]]
    lines += ["", "## 5. Failure cases and probes", "",
              f"- generations with listed failure cases: {len(with_cases)}/{total}; "
              f"file opened in {share(lambda r: r['files']['failure_cases.json']['reads'] > 0, with_cases)}",
              f"- of those, generations that called `probe_descriptor`: {len(probing)}; "
              f"with at least one probed candidate taken from the failure cases: "
              f"{sum(1 for record in probing if record['failure_cases']['probed_from_failure_cases'])}",
              f"- probed candidate references: {sum(r['failure_cases']['probed_refs'] for r in probing)}, "
              f"of which from the failure cases: {sum(r['failure_cases']['probed_from_failure_cases'] for r in probing)}",
              f"- memory: `search_memory` called in {share(lambda r: r['memory_searches'] > 0)} generations; "
              f"a handoff written in {share(lambda r: r['handoff_written'])}"]

    flags = (("TF", "train_feedback.json"), ("HM", "hypothesis_memory.json"), ("FC", "failure_cases.json"),
             ("CP", "candidate_pool.json"), ("FS", "feature_statistics.json"), ("CG", "classifier_guide.md"))
    lines += ["", "## 6. Per generation", "",
              "Opened: " + ", ".join(f"{flag} = `{name}`" for flag, name in flags)
              + ". Training replies: read in full / cut to a preview (`train_classifier`, `evaluate_train`, "
              "`search_parameters`).", "",
              "| run | gen | kind | mode | outcome | opened | training replies read / cut | noise interval mentioned | "
              "outer result mentioned |", "|---|---|---|---|---|---|---|---|---|"]
    for record in records:
        training = [record["tools"][name] for name in TRAINING_TOOLS if name in record["tools"]]
        lines.append(
            f"| {short[record['run']]} | {record['generation']} | {record['kind']} | {record['mode']} | {record['outcome']} | "
            + (" ".join(flag for flag, name in flags if record["files"][name]["reads"]) or "none") + " | "
            f"{sum(entry['file_read'] for entry in training)} / {sum(entry['truncated'] for entry in training)} | "
            f"{'yes' if record['mentions']['TRAIN noise interval'] else 'no'} | "
            f"{'yes' if record['mentions']['outer result of earlier submissions'] else 'no'} |")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path, help="finished run directories")
    parser.add_argument("--output-dir", type=Path, default=REPO / "notebooks" / "feedback_use_audit")
    args = parser.parse_args()
    records = []
    for run in args.runs:
        run = run.resolve()
        manifest = read_json(run / "manifest.json", {}) or {}
        brains = [str(brain) for brain in manifest.get("train_brains", [])] or ["no-brain-recorded"]
        ledger = {}
        for line in (run / "ledger.jsonl").read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                ledger[int(row["generation"])] = row
        for directory in sorted(run.glob("gen[0-9]*")):
            if not (directory / "trajectory.jsonl").exists():
                continue
            records.append(audit_generation(run, directory, ledger.get(int(directory.name[3:]), {}), brains))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(json.dumps(records, indent=1))
    write_summary(records, args.output_dir / "summary.md")
    print((args.output_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()

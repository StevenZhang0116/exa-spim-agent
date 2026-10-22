#!/usr/bin/env python3
"""An agent as the discovery engine: the AutoDiscovery materials in, an AD-shaped export out.

An ablation of AutoDiscovery. A coding agent gets exactly what AutoDiscovery gets,
one task specification and three data files, proposes hypotheses, writes and runs
one experiment script per hypothesis on this node, and records an analysis and a
review of the result. The driver turns the records into ``autodiscovery/<run>.json``
with the same fields as an AutoDiscovery export (``id``, ``status``, ``hypothesis``,
``surprisal``, ``isSurprising``, ``prior``, ``posterior``, ``priorBelief``,
``posteriorBelief``, ``analysis``, ``review``, ``experimentPlan``, ``code``,
``codeOutput``, ``richOutputs``, ``runtimeMs``, ``createdAt``), so
``run_discovery_workflow.py`` and the detector build read it unchanged. The belief
fields are AutoDiscovery's self-assessment and play no part in the predictive
workflow (which keeps every non-excluded hypothesis), so they are written as
neutral constants (prior = posterior = 0.5, surprisal 0, not surprising) rather
than asked of the agent.

Two engines, the same files and the same researcher instructions
(``.claude/agents/discovery-researcher.md``):

- ``--engine claude``: a Claude Agent SDK session per batch that invokes the
  ``discovery-researcher`` subagent (cost from the SDK's result message);
- ``--engine codex``: ``codex exec`` (OpenAI Codex CLI) per batch with the
  researcher instructions as the prompt (tokens from its ``turn.completed`` events;
  cost = tokens x ``--price-in`` / ``--price-out`` per million, 0 when unpriced).

Materials (all by their AutoDiscovery names, linked into the work directory):
``markdowns/<kind>_features_for_proofreader_evolve.md`` as ``task.md``,
``cache/dataset_cache_<brain>_mcl<N>_add.pkl``,
``markdowns/rerank_companion_<brain>_<kind>.{npz,json}``. Nothing else is given.

Budget: hypotheses are produced in batches, each batch one fresh session (so
context never accumulates); the driver stops at ``--hypotheses``, ``--max-cost-usd``
(sum of the sessions' cost) or ``--max-hours``, whichever first, and assembles the
export from whatever records exist. Re-running the same ``--run-name`` resumes after
the last complete id. Scripts are run through the driver-written ``run_hypo.sh``
(timeout, captured stdout/stderr, runtime), never by hand, so the record's
``codeOutput`` and ``runtimeMs`` are the driver's, not the agent's.

Run on a compute node under ``panda`` (each script loads the pkl; budget 40 GB);
the Claude engine needs ``ANTHROPIC_API_KEY``, the Codex engine a logged-in
``codex`` CLI:

    python agentic/run_agent_discovery.py --engine claude --kind merge --brain 794495 \\
        --run-name claude-merge-1_2026-10-22 --hypotheses 100 --batch 5 \\
        --max-cost-usd 150 --max-hours 12

    python agentic/run_agent_discovery.py --engine codex --model gpt-6-astra \\
        --kind merge --brain 794495 --run-name codex-merge-1_2026-10-22 \\
        --hypotheses 100 --batch 5 --max-hours 12

    python agentic/run_agent_discovery.py --run-name claude-merge-1_2026-10-22 \\
        --kind merge --brain 794495 --assemble-only      # export from the records only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

try:
    from agentic.detector_build.agent_session import load_claude_sdk, open_agent_session
except ImportError:  # pragma: no cover - script-style import
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from detector_build.agent_session import load_claude_sdk, open_agent_session  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUTODISCOVERY_DIR = PROJECT_ROOT / "autodiscovery"
RESEARCHER_SPEC = PROJECT_ROOT / ".claude" / "agents" / "discovery-researcher.md"
KINDS = ("merge", "split")
ENGINES = ("claude", "codex")
DEFAULT_MODEL = {"claude": os.environ.get("AGENT_DISCOVERY_CLAUDE_MODEL", "claude-opus-5-5"),
                 "codex": os.environ.get("AGENT_DISCOVERY_CODEX_MODEL", "gpt-6-astra")}
DEFAULT_EFFORT = "high"
# AutoDiscovery's belief fields, written as neutral constants: no agent is asked
# for a prior or a posterior (the predictive workflow never ranks by them).
NEUTRAL_BELIEF = 0.5
BELIEF_BUCKETS = (("definitely_false", 0.2), ("maybe_false", 0.4), ("uncertain", 0.6),
                  ("maybe_true", 0.8), ("definitely_true", 1.01))
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def run_name_kind(run_name: str) -> str | None:
    """The kind the downstream reads off a run name: exactly one of the tokens
    ``split`` / ``merge`` (split on ``-``, ``_`` and ``.``), as
    ``run_discovery_workflow._is_split_run`` and the detector build do."""
    tokens = set(re.split(r"[-_.]", run_name.lower()))
    found = tokens & set(KINDS)
    return next(iter(found)) if len(found) == 1 else None


# --------------------------------------------------------------------------- records
def belief(probability: float) -> dict:
    """An AutoDiscovery-shaped ``boolean_cat`` belief from one probability (five votes)."""
    p = min(max(float(probability), 0.0), 1.0)
    votes = {name: 0 for name, _ in BELIEF_BUCKETS}
    bucket = next(name for name, upper in BELIEF_BUCKETS if p < upper)
    votes[bucket] = 5
    return {"_type": "boolean_cat", "n": 5, "mean": p, "_empirical_mean": p,
            "prior_params": [0.5, 0.5], **votes}


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


PLAN_KEYS = ("objective", "steps", "deliverables")


def experiment_plan(value) -> dict:
    """AutoDiscovery's ``experimentPlan`` is an object with objective / steps /
    deliverables (the ranker reads ``objective``); a plain string becomes the objective."""
    if isinstance(value, dict):
        return {key: str(value.get(key, "") or "").strip() for key in PLAN_KEYS}
    return {"objective": str(value or "").strip(), "steps": "", "deliverables": ""}


def rich_outputs(outputs_dir: Path, rid: int) -> list[dict]:
    """Figures the script saved as ``outputs/hypo_<id>_*.png``, embedded the way
    AutoDiscovery embeds them (``{"image/png": <base64>}``)."""
    import base64

    if not outputs_dir.is_dir():
        return []
    return [{"image/png": base64.b64encode(path.read_bytes()).decode("ascii")}
            for path in sorted(outputs_dir.glob(f"hypo_{rid}_*.png"))]


def record_state(work_dir: Path, rid: int) -> str:
    """``absent`` / ``proposed`` (proposal only) / ``complete`` (proposal + result)."""
    proposal = work_dir / "records" / f"{rid}.proposal.json"
    result = work_dir / "records" / f"{rid}.result.json"
    if not proposal.is_file():
        return "absent"
    return "complete" if result.is_file() else "proposed"


def next_batch(work_dir: Path, n_hypotheses: int, batch: int) -> list[int]:
    """The next ids to work on: unfinished ones first, then new ids, ``batch`` at most."""
    pending = [rid for rid in range(1, n_hypotheses + 1) if record_state(work_dir, rid) != "complete"]
    return pending[:batch]


def assemble_record(work_dir: Path, rid: int) -> dict | None:
    proposal = _read_json(work_dir / "records" / f"{rid}.proposal.json")
    if proposal is None:
        return None
    result = _read_json(work_dir / "records" / f"{rid}.result.json") or {}
    meta = _read_json(work_dir / "outputs" / f"hypo_{rid}.meta.json") or {}
    code_path = work_dir / f"hypo_{rid}.py"
    stdout_path = work_dir / "outputs" / f"hypo_{rid}.stdout.txt"
    stderr_path = work_dir / "outputs" / f"hypo_{rid}.stderr.txt"
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
    output = stdout + (f"\nSTDERR:\n{stderr}" if stderr.strip() else "")
    ran_ok = meta.get("exit_code") == 0
    succeeded = ran_ok and bool(result)
    return {
        "id": rid,
        "status": "SUCCEEDED" if succeeded else "FAILED",
        "hypothesis": str(proposal.get("hypothesis", "")).strip(),
        "surprisal": 0.0,
        "isSurprising": False,
        "prior": NEUTRAL_BELIEF,
        "posterior": NEUTRAL_BELIEF,
        "priorBelief": belief(NEUTRAL_BELIEF),
        "posteriorBelief": belief(NEUTRAL_BELIEF),
        "analysis": str(result.get("analysis", "")).strip(),
        "review": str(result.get("review", "")).strip(),
        "experimentPlan": experiment_plan(proposal.get("experimentPlan")),
        "code": code_path.read_text(encoding="utf-8") if code_path.is_file() else "",
        "codeOutput": output,
        "richOutputs": rich_outputs(work_dir / "outputs", rid),
        "runtimeMs": int(meta.get("runtime_ms", 0) or 0),
        "createdAt": meta.get("finished_at") or datetime.now(timezone.utc).isoformat(),
        "exitCode": meta.get("exit_code"),
    }


def assemble_export(work_dir: Path, n_hypotheses: int) -> list[dict]:
    records = []
    for rid in range(1, n_hypotheses + 1):
        record = assemble_record(work_dir, rid)
        if record is not None:
            records.append(record)
    return records


# --------------------------------------------------------------------------- work dir
def materials(kind: str, brain: str, mcl: int) -> dict[str, Path]:
    return {
        "task.md": PROJECT_ROOT / "markdowns" / f"{kind}_features_for_proofreader_evolve.md",
        f"dataset_cache_{brain}_mcl{mcl}_add.pkl": PROJECT_ROOT / "cache" / f"dataset_cache_{brain}_mcl{mcl}_add.pkl",
        f"rerank_companion_{brain}_{kind}.npz": PROJECT_ROOT / "markdowns" / f"rerank_companion_{brain}_{kind}.npz",
        f"rerank_companion_{brain}_{kind}.json": PROJECT_ROOT / "markdowns" / f"rerank_companion_{brain}_{kind}.json",
    }


RUN_HYPO_SH = """#!/bin/bash
# Driver-owned runner: the only way a hypothesis script is executed. Captures stdout,
# stderr, exit code and wall time so the export's codeOutput/runtimeMs are the driver's.
set -u
id="$1"
cd "$(dirname "$0")"
mkdir -p outputs
export AWS_EC2_METADATA_DISABLED=true MPLBACKEND=Agg OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
start_ns=$(date +%s%N)
started=$(date -u +%Y-%m-%dT%H:%M:%S.%6N+00:00)
timeout --signal=TERM --kill-after=30 {timeout}s "{python}" -u "hypo_${{id}}.py" \\
    > "outputs/hypo_${{id}}.stdout.txt" 2> "outputs/hypo_${{id}}.stderr.txt"
code=$?
end_ns=$(date +%s%N)
finished=$(date -u +%Y-%m-%dT%H:%M:%S.%6N+00:00)
printf '{{"id": %s, "exit_code": %s, "runtime_ms": %s, "started_at": "%s", "finished_at": "%s", "timeout_s": {timeout}}}\\n' \\
    "$id" "$code" "$(( (end_ns - start_ns) / 1000000 ))" "$started" "$finished" > "outputs/hypo_${{id}}.meta.json"
echo "hypo_${{id}}.py exit=$code runtime_s=$(( (end_ns - start_ns) / 1000000000 ))"
echo "--- stdout (last 60 lines) ---"; tail -n 60 "outputs/hypo_${{id}}.stdout.txt"
echo "--- stderr (last 20 lines) ---"; tail -n 20 "outputs/hypo_${{id}}.stderr.txt"
exit $code
"""


def prepare_work_dir(work_dir: Path, kind: str, brain: str, mcl: int, script_timeout: int,
                     python: str | None = None) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "records").mkdir(exist_ok=True)
    (work_dir / "outputs").mkdir(exist_ok=True)
    for name, source in materials(kind, brain, mcl).items():
        if not source.is_file():
            raise SystemExit(f"Missing material {source}")
        link = work_dir / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(source.resolve())
    runner = work_dir / "run_hypo.sh"
    runner.write_text(RUN_HYPO_SH.format(timeout=int(script_timeout), python=python or sys.executable))
    runner.chmod(0o755)


def write_ledger(work_dir: Path, n_hypotheses: int) -> Path:
    """One line per recorded hypothesis so a fresh session does not repeat them."""
    lines = []
    for rid in range(1, n_hypotheses + 1):
        proposal = _read_json(work_dir / "records" / f"{rid}.proposal.json")
        if proposal is None:
            continue
        result = _read_json(work_dir / "records" / f"{rid}.result.json") or {}
        state = "done" if result else "unfinished"
        lines.append(f"{rid} [{state}]: {str(proposal.get('hypothesis', '')).strip()[:220]}")
    ledger = work_dir / "records" / "ledger.txt"
    ledger.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return ledger


# --------------------------------------------------------------------------- prompts
def batch_instruction(work_dir: Path, ids: list[int], kind: str, brain: str, mcl: int) -> str:
    """What one session is asked to do, engine-independent (no agent name in it)."""
    unfinished = [rid for rid in ids if record_state(work_dir, rid) == "proposed"]
    resume = (f"Ids {unfinished} already have a proposal but no result: finish those first "
              "(run or fix their script, then write the result file). " if unfinished else "")
    w = str(work_dir)
    return f"""
Research hypotheses {ids[0]} to {ids[-1]} (ids {ids}) in the work directory {w}. The
task is {w}/task.md; the data are dataset_cache_{brain}_mcl{mcl}_add.pkl,
rerank_companion_{brain}_{kind}.npz and rerank_companion_{brain}_{kind}.json in that
directory, which every script opens by exactly those names (the runner changes into
the directory before running it). {w}/records/ledger.txt lists every hypothesis
already recorded in this run: do not repeat one. {resume}For each id, in this order:
write {w}/records/<id>.proposal.json (hypothesis; experimentPlan as an object with
objective, steps and deliverables) before any code; write {w}/hypo_<id>.py; run it
only with `{w}/run_hypo.sh <id>`; read {w}/outputs/hypo_<id>.stdout.txt and
.stderr.txt; write {w}/records/<id>.result.json (analysis, review). Work the ids
one at a time, serially (every script
loads the full dataset), and stop when all assigned ids have a result file. Do not
touch any other id and do not edit anything under {w}/outputs/.
""".strip()


def researcher_instructions(spec_path: Path = RESEARCHER_SPEC) -> str:
    """The researcher agent's instructions without their YAML front matter."""
    text = spec_path.read_text(encoding="utf-8")
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end >= 0:
            text = text[end + 4:]
    return text.strip()


# --------------------------------------------------------------------------- claude engine
async def run_batch_claude(work_dir: Path, ids: list[int], kind: str, brain: str, mcl: int, *,
                           model: str, effort: str, verbose: bool, max_budget_usd: float | None,
                           timeout_s: int) -> dict:
    """One fresh SDK session for one batch; the session's reported cost."""
    sdk = load_claude_sdk()
    options = sdk.agent_options(
        cwd=str(PROJECT_ROOT),   # the project's .claude/agents must be in scope
        setting_sources=["project"],
        allowed_tools=["Task", "Bash", "Read", "Write", "Edit", "Glob"],
        permission_mode="bypassPermissions",
        model=model,
        effort=effort,
        **({"max_budget_usd": float(max_budget_usd)} if max_budget_usd else {}),
    )
    prompt = "Use the discovery-researcher subagent. " + batch_instruction(work_dir, ids, kind, brain, mcl)
    cost = 0.0
    async with open_agent_session(sdk.client, options=options, connect_timeout_s=60) as client:
        await client.query(prompt)

        async def receive() -> None:
            nonlocal cost
            async for message in client.receive_response():
                if isinstance(message, sdk.assistant_message):
                    for block in message.content:
                        if isinstance(block, sdk.text_block) and verbose:
                            print(block.text, end="", flush=True)
                        elif sdk.tool_use_block and isinstance(block, sdk.tool_use_block):
                            print(f"[researcher] {getattr(block, 'name', 'tool')}", flush=True)
                elif isinstance(message, sdk.result_message):
                    reported = getattr(message, "total_cost_usd", None)
                    cost = float(reported) if reported is not None else cost
                    print(f"[researcher] batch {ids[0]}-{ids[-1]} session ended: cost ${cost:.4f}", flush=True)

        try:
            await asyncio.wait_for(receive(), timeout=timeout_s)
        except asyncio.TimeoutError:
            print(f"[researcher] batch {ids[0]}-{ids[-1]} timed out after {timeout_s}s; "
                  "its records so far are kept", flush=True)
    return {"cost_usd": cost, "tokens": None}


# --------------------------------------------------------------------------- codex engine
def codex_command(work_dir: Path, model: str, effort: str, last_message: Path, sandbox: str) -> list[str]:
    """``codex exec`` argv; the prompt goes in on stdin (``-``)."""
    argv = [shutil.which("codex") or "codex", "exec", "--json", "--skip-git-repo-check",
            "-C", str(work_dir), "-m", model, "-c", f'model_reasoning_effort="{effort}"',
            "-o", str(last_message)]
    if sandbox == "bypass":
        # The compute node (an externally bounded Slurm job) is the sandbox; the
        # scripts need network for the image volume and write under the work dir.
        argv.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        argv += ["-s", sandbox]
    return argv + ["-"]


def parse_codex_events(lines) -> dict:
    """Token usage summed over ``turn.completed`` events, plus the agent's messages."""
    usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_output_tokens": 0}
    messages, commands, turns = [], 0, 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "turn.completed":
            turns += 1
            for key in usage:
                usage[key] += int((event.get("usage") or {}).get(key, 0) or 0)
        elif kind == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message" and item.get("text"):
                messages.append(str(item["text"]))
            elif item.get("type") == "command_execution":
                commands += 1
    return {"usage": usage, "messages": messages, "commands": commands, "turns": turns}


def codex_cost(usage: dict, price_in: float, price_out: float) -> float:
    """USD from tokens and per-million prices; reasoning tokens are billed as output."""
    return (usage.get("input_tokens", 0) * price_in
            + (usage.get("output_tokens", 0) + usage.get("reasoning_output_tokens", 0)) * price_out) / 1e6


def run_batch_codex(work_dir: Path, ids: list[int], kind: str, brain: str, mcl: int, *,
                    model: str, effort: str, verbose: bool, timeout_s: int,
                    price_in: float, price_out: float, sandbox: str) -> dict:
    """One ``codex exec`` session for one batch; tokens from its JSONL events."""
    codex_dir = work_dir / "codex"
    codex_dir.mkdir(exist_ok=True)
    tag = f"{ids[0]}-{ids[-1]}_{datetime.now():%Y%m%d-%H%M%S}"
    events_path = codex_dir / f"events_{tag}.jsonl"
    last_message = codex_dir / f"last_message_{tag}.txt"
    prompt = researcher_instructions() + "\n\n# This batch\n\n" + batch_instruction(work_dir, ids, kind, brain, mcl)
    argv = codex_command(work_dir, model, effort, last_message, sandbox)
    lines: list[str] = []
    with events_path.open("w", encoding="utf-8") as events:
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, cwd=str(work_dir))
        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
            deadline = time.monotonic() + timeout_s
            for line in proc.stdout:
                events.write(line)
                events.flush()
                lines.append(line)
                if verbose:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        event = {}
                    item = event.get("item") or {}
                    if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                        print(item.get("text", ""), flush=True)
                    elif event.get("type") == "item.completed" and item.get("type") == "command_execution":
                        print(f"[codex] $ {str(item.get('command', ''))[:160]}", flush=True)
                if time.monotonic() > deadline:
                    proc.kill()
                    print(f"[codex] batch {ids[0]}-{ids[-1]} timed out after {timeout_s}s; "
                          "its records so far are kept", flush=True)
                    break
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        finally:
            stderr_text = proc.stderr.read() if proc.stderr else ""
            if stderr_text.strip():
                (codex_dir / f"stderr_{tag}.txt").write_text(stderr_text, encoding="utf-8")
    parsed = parse_codex_events(lines)
    cost = codex_cost(parsed["usage"], price_in, price_out)
    print(f"[codex] batch {ids[0]}-{ids[-1]} exit={proc.returncode} turns={parsed['turns']} "
          f"commands={parsed['commands']} tokens in={parsed['usage']['input_tokens']} "
          f"out={parsed['usage']['output_tokens']} reasoning={parsed['usage']['reasoning_output_tokens']}"
          + (f" cost ${cost:.4f}" if price_in or price_out else " (unpriced)"), flush=True)
    return {"cost_usd": cost, "tokens": parsed["usage"], "exit_code": proc.returncode}


# --------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=ENGINES, default="claude")
    parser.add_argument("--model", default=None, help="claude: claude-opus-5-5; codex: gpt-6-astra")
    parser.add_argument("--effort", default=DEFAULT_EFFORT,
                        help="reasoning effort passed to the engine (claude effort / codex model_reasoning_effort)")
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--brain", required=True)
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--run-name", required=True,
                        help="export stem carrying the kind as one -/_ token, e.g. claude-merge-1_2026-10-22")
    parser.add_argument("--hypotheses", type=int, default=100)
    parser.add_argument("--batch", type=int, default=5, help="hypotheses per fresh agent session")
    parser.add_argument("--max-cost-usd", type=float, default=150.0)
    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument("--script-timeout", type=int, default=1800, help="seconds per hypothesis script")
    parser.add_argument("--batch-timeout", type=int, default=4 * 3600, help="seconds per agent session")
    parser.add_argument("--price-in", type=float, default=0.0, help="codex: USD per million input tokens")
    parser.add_argument("--price-out", type=float, default=0.0, help="codex: USD per million output tokens")
    parser.add_argument("--codex-sandbox", default="bypass",
                        choices=("bypass", "workspace-write", "danger-full-access"),
                        help="codex: 'bypass' skips approvals and sandboxing (the Slurm job is the sandbox)")
    parser.add_argument("--assemble-only", action="store_true", help="write the export from existing records")
    parser.add_argument("--autodiscovery-dir", type=Path, default=AUTODISCOVERY_DIR,
                        help="where <run-name>.<engine>/ and <run-name>.json live")
    parser.add_argument("--verbose", action="store_true")
    return parser


def run_batch(args, work_dir: Path, ids: list[int], spent: float) -> dict:
    if args.engine == "claude":
        return asyncio.run(run_batch_claude(
            work_dir, ids, args.kind, args.brain, args.mcl, model=args.model, effort=args.effort,
            verbose=args.verbose, max_budget_usd=max(args.max_cost_usd - spent, 1.0),
            timeout_s=args.batch_timeout))
    return run_batch_codex(
        work_dir, ids, args.kind, args.brain, args.mcl, model=args.model, effort=args.effort,
        verbose=args.verbose, timeout_s=args.batch_timeout, price_in=args.price_in,
        price_out=args.price_out, sandbox=args.codex_sandbox)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not RUN_NAME_RE.match(args.run_name) or run_name_kind(args.run_name) != args.kind:
        parser.error("--run-name must be a plain file stem carrying the kind as one token "
                     "(split or merge, separated by - _ or .), e.g. claude-merge-1_2026-10-22")
    if args.hypotheses < 1 or args.batch < 1:
        parser.error("--hypotheses and --batch must be positive")
    args.model = args.model or DEFAULT_MODEL[args.engine]
    if args.engine == "codex" and not args.assemble_only and shutil.which("codex") is None:
        parser.error("the codex CLI is not on PATH")

    work_dir = args.autodiscovery_dir / f"{args.run_name}.{args.engine}"
    export_path = args.autodiscovery_dir / f"{args.run_name}.json"
    manifest_path = work_dir / "manifest.json"
    manifest = _read_json(manifest_path) or {
        "run_name": args.run_name, "kind": args.kind, "brain": args.brain, "mcl": args.mcl,
        "sessions": [], "cost_usd": 0.0, "seconds": 0.0}
    for key, value in (("engine", args.engine), ("model", args.model), ("effort", args.effort)):
        manifest.setdefault(key, value)
    if not args.assemble_only:
        prepare_work_dir(work_dir, args.kind, args.brain, args.mcl, args.script_timeout)
        if args.engine == "codex" and not (args.price_in or args.price_out):
            print("[driver] codex engine without --price-in/--price-out: tokens are recorded, "
                  "the cost budget cannot stop the run; --hypotheses and --max-hours do", flush=True)
        started = time.monotonic()
        log = work_dir / "agent_discovery.log.txt"
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"# {datetime.now():%Y-%m-%d %H:%M:%S} argv: {' '.join(sys.argv)} host: {platform.node()}\n")
        while True:
            ids = next_batch(work_dir, args.hypotheses, args.batch)
            spent = float(manifest["cost_usd"])
            elapsed_h = (time.monotonic() - started) / 3600 + float(manifest["seconds"]) / 3600
            if not ids:
                print("[driver] every hypothesis has a result; done", flush=True)
                break
            if spent >= args.max_cost_usd:
                print(f"[driver] cost budget reached (${spent:.2f} >= ${args.max_cost_usd:.2f}); stopping", flush=True)
                break
            if elapsed_h >= args.max_hours:
                print(f"[driver] time budget reached ({elapsed_h:.2f} h >= {args.max_hours} h); stopping", flush=True)
                break
            write_ledger(work_dir, args.hypotheses)
            print(f"[driver] {args.engine}/{args.model} batch ids {ids} (spent ${spent:.2f}, {elapsed_h:.2f} h)",
                  flush=True)
            batch_started = time.monotonic()
            try:
                outcome = run_batch(args, work_dir, ids, spent)
            except BaseException:
                traceback.print_exc()
                raise
            finally:
                manifest["seconds"] = float(manifest["seconds"]) + (time.monotonic() - batch_started)
            completed = [rid for rid in ids if record_state(work_dir, rid) == "complete"]
            manifest["sessions"].append({"ids": ids, "cost_usd": outcome["cost_usd"],
                                         "tokens": outcome.get("tokens"), "completed": completed,
                                         "finished_at": datetime.now(timezone.utc).isoformat()})
            manifest["cost_usd"] = float(manifest["cost_usd"]) + float(outcome["cost_usd"])
            manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
            with log.open("a", encoding="utf-8") as handle:
                handle.write(f"batch {ids}: cost ${outcome['cost_usd']:.4f}, tokens {outcome.get('tokens')}, "
                             f"completed {completed}\n")
            if not completed:
                print("[driver] the batch completed no hypothesis; stopping to avoid spending on a stuck loop",
                      flush=True)
                break
    records = assemble_export(work_dir, args.hypotheses)
    export_path.write_text(json.dumps(records, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["export"] = str(export_path)
    manifest["records"] = len(records)
    manifest["succeeded"] = sum(r["status"] == "SUCCEEDED" for r in records)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"[driver] export {export_path}: {len(records)} records "
          f"({manifest['succeeded']} SUCCEEDED), ${float(manifest['cost_usd']):.2f}, "
          f"{float(manifest['seconds']) / 3600:.2f} h of agent time", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

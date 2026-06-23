"""
Proofreader self-improvement loop (AlphaEvolve-shaped) via the Claude Agent SDK.

This implements exactly the cycle the reviewer asked for:

    1. give the system a subset of data        -> TRAIN ground-truth skeletons
    2. let it make proofreading decisions       -> run the evolved policy -> edits
    3. show it where it was wrong               -> failure report vs baseline
    4. ask it to explain why it was wrong       -> proofreader-reviser subagent
    5. let it revise its rules/prompts/tools    -> edits heuristics.py + rules.md
    6. evaluate on held-out cases               -> score on HELD-OUT skeletons
    7. retain only verifiable improvements      -> gate: beat the PARENT (current
                                                   accepted policy) on held-out
       ...and measure compute/time/human effort -> ledger.jsonl

The seed policy proposes no edits (== baseline), and the gate is parent-relative
(each generation must beat the last accepted policy, not the fixed baseline), so
accepted improvements accumulate generation over generation.

The evolved "program" is the pair (artifacts/heuristics.py, artifacts/rules.md).
Each generation snapshots them, lets the agent revise, re-scores on held-out, and
either keeps or reverts. Every generation's cost is recorded.

Run from the project root (exa-spim-agent/):
    python proofreader_evolve/run_evolution.py --brain 789202 --generations 5
    python proofreader_evolve/run_evolution.py --brain 789202 --generations 5 --human-gate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Make `proofreader_evolve` importable when run as a script from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)
try:
    from claude_agent_sdk import ToolUseBlock
except ImportError:  # pragma: no cover
    ToolUseBlock = ()  # type: ignore

from proofreader_evolve.harness import (
    scoring,
    dataset as ds,
    candidate as cand,
    incremental_scoring as inc,
)
from proofreader_evolve.harness.ledger import Ledger, GenerationCost

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
HEURISTICS = ARTIFACTS / "heuristics.py"
RULES = ARTIFACTS / "rules.md"
# Validated, cross-run error-regularity knowledge base produced by the
# AutoDiscovery workflow (agentic/run_discovery_workflow.py). When present, the
# reviser is pointed at it as a PRIOR — but told to trust ONLY the findings that
# GENERALIZE across brains and were UPHELD after statistical correction, and to
# distrust DOES-NOT-GENERALIZE / OVERTURNED ones. Optional: missing file => no-op.
DISCOVERY_PRIORS = PROJECT_ROOT / "autodiscovery" / "all-runs.combined.md"


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


class Heartbeat:
    """A background thread that logs ``... still <phase> (N.Ns elapsed)`` on an
    interval, so long silent stretches (per-candidate scoring, the reviser LLM
    call) visibly stay alive when output is captured to a file or piped.

    Why a thread and not an asyncio task: the scoring path (``run_candidate`` /
    ``score_incremental``) is synchronous and CPU-bound — it blocks the event
    loop, so an async heartbeat would be starved exactly when a pulse matters
    most. A daemon thread ticks regardless of what the main thread is doing.

    Use as a context manager around any long step::

        with Heartbeat("scoring held-out"):
            heldout_run = cand.run_candidate(...)
    """

    def __init__(self, phase: str, every_seconds: float = 30.0) -> None:
        self._phase = phase
        self._every = every_seconds
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        # Wait in small slices so stop is responsive, but only LOG every _every.
        while not self._stop.wait(self._every):
            log(f"   ... still {self._phase} ({time.monotonic() - self._t0:.0f}s elapsed)")

    def __enter__(self) -> "Heartbeat":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


# Default model for the evolution loop (and the inherit-ing reviser subagent).
# Opus 4.8 via the Anthropic API — the reviser is the "mutation operator" doing
# the diagnostic reasoning, so we want the strongest model there.
DEFAULT_MODEL = "claude-opus-4-8"


def _anthropic_api_env() -> dict[str, str]:
    """Provider env that pins the run to the Anthropic API (not Bedrock/Vertex).

    The surrounding shell may set CLAUDE_CODE_USE_BEDROCK=1, which would route to
    Bedrock — where the Anthropic model id ``claude-opus-4-8`` does not resolve
    and no ANTHROPIC_API_KEY is used. We override those vars for this session so
    the saved ANTHROPIC_API_KEY is what authenticates. Raises if the key is
    absent so the failure is explicit rather than a silent provider fallback.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; it is required to run the evolution "
            "loop on the Anthropic API. Export it (the saved key) and retry."
        )
    return {
        "ANTHROPIC_API_KEY": api_key,
        # Force the Anthropic API: disable the cloud-provider routes that would
        # otherwise take precedence and ignore the API key / model id.
        "CLAUDE_CODE_USE_BEDROCK": "0",
        "CLAUDE_CODE_USE_VERTEX": "0",
        # Run Opus 4.8 at maximum reasoning effort for the evolution/reviser work.
        # Levels: low|medium|high|xhigh|max; Opus 4.8 thinks adaptively and at
        # xhigh almost always reasons deeply. Applies to the session + subagents.
        "CLAUDE_EFFORT": "xhigh",
    }


def _make_isolation_guard(run_dir: Path, audit_path: Path):
    """Build a ``can_use_tool`` callback that ENFORCES run isolation via an ALLOWLIST.

    Two leak channels exist for a file-tool reviser living in the ``runs/`` tree:
      (1) cross-run: reading a SIBLING run's accepted policy / ledger and copying
          the answer (destroys "independent rediscovery"); and
      (2) intra-run: reading THIS run's own ``split.json`` (reveals which neurons
          are held-out), ``ledger.jsonl`` / ``attempts.md`` (run state), or prior
          ``gen*/heuristics.candidate.py`` — none of which the prompt permits.
    The prompt only restricts writes, so we close BOTH at the permission layer with
    an explicit allowlist: under ``runs/`` the reviser may touch ONLY this run's
    working artifacts and the per-generation failure reports; every other ``runs/``
    path (sibling OR intra-run) is DENIED and recorded. Paths OUTSIDE ``runs/`` (the
    harness modules it imports, etc.) are unaffected.

    Allowed under ``runs/`` (this run only):
      * Read/Write/Edit:  <run>/artifacts/heuristics.py, <run>/artifacts/rules.md
      * Read:             <run>/gen*/failure_report.md   (train-only, leak-free)

    Returns ``(callback, state)`` where ``state['violations']`` accumulates the
    denied attempts (also appended to ``audit_path``).
    """
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    runs_root = (HERE / "runs").resolve()
    this_run = run_dir.resolve()
    work_dir = (this_run / "artifacts").resolve()
    allowed_rw = {(work_dir / "heuristics.py"), (work_dir / "rules.md")}
    state = {"violations": []}

    # Which tools WRITE (so reading a report is fine but writing it is not).
    WRITE_TOOLS = {"Write", "Edit", "NotebookEdit"}
    # Tool-input fields that carry file paths / shell text, by tool name.
    PATH_FIELDS = {
        "Read": ("file_path",), "Write": ("file_path",), "Edit": ("file_path",),
        "Glob": ("path",), "NotebookEdit": ("notebook_path",),
        "Grep": ("path",),
    }

    def _path_allowed(tool_name: str, rp: Path) -> bool:
        """Allowlist test for a single resolved path that lies under ``runs/``."""
        # The two working artifacts: read or write.
        if rp in allowed_rw:
            return True
        # Failure reports of THIS run: read-only (never writable by the reviser).
        if (tool_name not in WRITE_TOOLS
                and this_run in rp.parents
                and rp.name == "failure_report.md"
                and rp.parent.parent == this_run
                and rp.parent.name.startswith("gen")):
            return True
        return False

    def _offending_paths(tool_name: str, ti: dict) -> list[str]:
        """Return runs/-tree paths in this call that are NOT on the allowlist."""
        candidates = []
        for f in PATH_FIELDS.get(tool_name, ()):
            v = ti.get(f)
            if isinstance(v, str):
                candidates.append(v)
        if tool_name == "Bash":
            # Can't parse shell reliably; scan any runs/ mention in the command.
            candidates.append(ti.get("command", ""))
        bad = []
        for c in candidates:
            if not c:
                continue
            if tool_name == "Bash":
                # Bash bypasses the allowlist entirely (it isn't even granted to the
                # reviser); flag any runs/ mention so a misconfig is loud, not silent.
                if "runs/" in c or str(runs_root) in c:
                    import re
                    for m in re.findall(r"\S*runs/\S*", c):
                        try:
                            rp = Path(m).resolve()
                        except Exception:
                            rp = Path(m)
                        if runs_root in rp.parents or rp == runs_root:
                            bad.append(m)
                continue
            try:
                rp = Path(c).resolve()
            except Exception:
                continue
            # Only police the runs/ tree; paths elsewhere (harness imports) are free.
            if runs_root in rp.parents and not _path_allowed(tool_name, rp):
                bad.append(c)
        return bad

    async def can_use_tool(tool_name: str, tool_input: dict, context):
        bad = _offending_paths(tool_name, tool_input or {})
        if bad:
            rec = {"tool": tool_name, "paths": bad}
            state["violations"].append(rec)
            try:
                with open(audit_path, "a") as f:
                    f.write(json.dumps({"DENIED": rec}) + "\n")
            except Exception:
                pass
            return PermissionResultDeny(
                behavior="deny",
                message=(f"Run isolation: {tool_name} may only touch this run's "
                         f"working heuristics.py / rules.md (read/write) or a "
                         f"gen*/failure_report.md (read). Denied: {bad}."),
                interrupt=False,
            )
        return PermissionResultAllow(behavior="allow")

    return can_use_tool, state


def build_options(model: str = DEFAULT_MODEL, run_dir: Path | None = None,
                  audit_path: Path | None = None):
    """SDK session config. setting_sources=['project'] auto-discovers the
    subagent in proofreader_evolve/agents via the project .claude.

    The model defaults to Opus 4.8 on the Anthropic API (see DEFAULT_MODEL); the
    reviser subagent's frontmatter says ``model: inherit``, so it uses this same
    model. Pass a different ``model`` to override.

    Run isolation (durable integrity): when ``run_dir`` is given we attach a
    ``can_use_tool`` callback that ALLOWLISTS only this run's working
    heuristics.py / rules.md (read/write) and its gen*/failure_report.md (read),
    DENYING every other ``runs/``-tree path — sibling runs AND this run's own
    split.json / ledger / attempts / prior candidates. The permission mode is
    ``acceptEdits`` (not ``bypassPermissions``, which would skip the callback) so
    it still fires non-interactively. The tool set is also trimmed to what the
    reviser needs. Returns ``(options, guard_state)``; None guard when no run_dir.
    """
    guard = state = None
    if run_dir is not None:
        guard, state = _make_isolation_guard(run_dir, audit_path or (run_dir / "tool_audit.jsonl"))
    opts = ClaudeAgentOptions(
        cwd=str(run_dir.resolve()) if run_dir is not None else str(PROJECT_ROOT),
        setting_sources=["project"],
        model=model,
        env=_anthropic_api_env(),
        # Reviser needs Task (it IS a subagent), Read/Write/Edit (its two files +
        # the report). Bash/Glob removed: they widen the read surface and aren't
        # needed (the import sanity-check can be dropped or run by the harness).
        allowed_tools=["Task", "Read", "Write", "Edit"],
        # acceptEdits (not bypassPermissions) so can_use_tool actually fires;
        # still non-interactive. The human gate is enforced in Python post-scoring.
        permission_mode="acceptEdits" if guard else "bypassPermissions",
        can_use_tool=guard,
    )
    return opts, state


def _find_existing_prepared(brain: str, exclude: Path | None = None) -> Path | None:
    """Find a reusable prepared-brain pickle for ``brain`` from a prior run.

    The prepared brain is candidate-invariant and identical across runs of the
    same brain, so the ~30 min / 1.6 GB build is paid once and copied forward.
    Searches this run-tree for ``prepared_<brain>.pkl`` — both the per-run
    location (``runs/<id>/prepared_<brain>.pkl``) and the legacy flat location
    (``runs/prepared_<brain>.pkl``) — and returns the newest match, or None.
    """
    runs_root = HERE / "runs"
    name = f"prepared_{brain}.pkl"
    candidates = []
    legacy = runs_root / name                      # legacy flat location
    if legacy.exists():
        candidates.append(legacy)
    candidates += [p for p in runs_root.glob(f"*/{name}")
                   if exclude is None or exclude not in p.parents]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_seed_source(seed_from: str) -> tuple[Path, Path]:
    """Resolve ``--seed-from`` to a prior run's latest ACCEPTED (heuristics, rules).

    ``seed_from`` is a run-id folder name (or a brain id -> newest matching run).
    Returns the paths of that run's last ``gen*/heuristics.accepted.py`` and
    ``rules.accepted.md``. Raises if the run has no accepted generation (nothing
    to continue from).
    """
    runs_root = HERE / "runs"
    run_dir = runs_root / seed_from
    if not run_dir.is_dir():
        matches = sorted(runs_root.glob(f"{seed_from}_*"), key=lambda p: p.stat().st_mtime)
        if not matches:
            raise SystemExit(f"--seed-from: no run folder matches {seed_from!r} under {runs_root}")
        run_dir = matches[-1]
    accepted = sorted(run_dir.glob("gen*/heuristics.accepted.py"))
    if not accepted:
        raise SystemExit(
            f"--seed-from {run_dir.name}: that run has no accepted generation "
            f"(no gen*/heuristics.accepted.py) — nothing to continue from."
        )
    h = accepted[-1]
    r = h.parent / "rules.accepted.md"
    if not r.exists():
        raise SystemExit(f"--seed-from {run_dir.name}: {h.name} found but {r.name} missing.")
    return h, r


def make_working_artifacts(run_dir: Path, seed_from: str | None = None) -> tuple[Path, Path]:
    """Copy the starting artifacts into this run's timestamped dir and return the
    copies' paths. The run reads/revises ONLY these copies; the originals under
    ``artifacts/`` are never touched, so the run is reproducible and the
    before/after files are trivially identifiable (original = ``artifacts/``,
    this run = ``runs/<brain>_<timestamp>/artifacts/``).

    By DEFAULT the start is the pristine seed (``artifacts/``), i.e. from scratch.
    If ``seed_from`` is given (a prior run-id), the run instead CONTINUES from that
    run's latest accepted policy, so lineages can compound across runs. The
    pristine ``artifacts/`` seed is never modified either way.
    """
    work_dir = run_dir / "artifacts"
    work_dir.mkdir(parents=True, exist_ok=True)
    work_heuristics = work_dir / "heuristics.py"
    work_rules = work_dir / "rules.md"
    if seed_from:
        src_h, src_r = _resolve_seed_source(seed_from)
        log(f"Seeding from prior run's accepted policy: {src_h}")
    else:
        src_h, src_r = HERE / "artifacts" / "heuristics.py", HERE / "artifacts" / "rules.md"
    # HERE/artifacts == ARTIFACTS; use the module constants for the default seed.
    shutil.copy2(src_h if seed_from else HEURISTICS, work_heuristics)
    shutil.copy2(src_r if seed_from else RULES, work_rules)
    return work_heuristics, work_rules


def snapshot(gen_dir: Path, heuristics: Path, rules: Path) -> None:
    """Save the PARENT artifacts (start-of-generation) so a rejected revision can
    be reverted. These are the pre-edit files; the post-edit ones are saved by
    ``save_candidate`` after the reviser runs."""
    gen_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(heuristics, gen_dir / "heuristics.py")
    shutil.copy2(rules, gen_dir / "rules.md")


def save_candidate(gen_dir: Path, heuristics: Path, rules: Path) -> tuple[str, str]:
    """Persist what the reviser ACTUALLY wrote this generation, accepted or not.

    Without this, a rejected generation's edited policy is destroyed on revert and
    you can never see what the agent tried. Returns (candidate_path, diffstat),
    where diffstat is '+added -removed' lines vs the parent snapshot (gen_dir's
    pre-edit heuristics.py saved by ``snapshot``).
    """
    cand_h = gen_dir / "heuristics.candidate.py"
    cand_r = gen_dir / "rules.candidate.md"
    shutil.copy2(heuristics, cand_h)
    shutil.copy2(rules, cand_r)

    # Diffstat of the edited policy vs the parent (the snapshot taken at gen start).
    parent_h = gen_dir / "heuristics.py"
    try:
        import difflib
        a = parent_h.read_text().splitlines()
        b = Path(heuristics).read_text().splitlines()
        added = removed = 0
        for line in difflib.unified_diff(a, b, lineterm=""):
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
        diffstat = f"+{added} -{removed}"
    except Exception:
        diffstat = "?"
    return str(cand_h), diffstat


def revert(gen_dir: Path, heuristics: Path, rules: Path) -> None:
    """Restore working artifacts from a snapshot (when a revision fails the gate)."""
    shutil.copy2(gen_dir / "heuristics.py", heuristics)
    shutil.copy2(gen_dir / "rules.md", rules)


def _format_attempts(attempts: list[dict]) -> str:
    """Render the prior-attempts archive (B: reviser memory) for the prompt.

    ``attempts`` are the revisions tried AGAINST THE CURRENT PARENT (cleared each
    time the parent advances). Showing them stops the reviser from re-proposing a
    change that was already rejected from the same starting point — the cause of
    the observed gen2==gen3 repetition.
    """
    if not attempts:
        return ""
    lines = ["\nAttempts already tried against the CURRENT policy "
             "(do NOT repeat these — they did not beat it):"]
    for a in attempts:
        lines.append(
            f"  - gen{a['gen']}: {a['summary']} -> held-out "
            f"{a['heldout']:+.3f} vs parent ({'kept' if a['accepted'] else 'rejected'})"
        )
    return "\n".join(lines) + "\n"


def _format_priors(priors_path: str | None) -> str:
    """Prompt fragment pointing the reviser at the validated discovery priors.

    Returns "" when no priors file is configured/exists (so the prompt is
    unchanged). When present, the reviser is told to READ the file and use it as a
    prior, with an ASYMMETRIC discipline that is deliberate:

      * The TRUST rule is stated abstractly (GENERALIZES + UPHELD) WITHOUT naming
        specific findings or quoting their thresholds. Naming a favoured few would
        anchor the agent onto them (and let it copy the numbers without ever
        reading the file), starving the other qualifying findings and collapsing
        the multi-lever exploration diversity that makes the search work. So the
        agent must survey ALL qualifying findings itself and pick by relevance.
      * The DISTRUST list IS explicit, because naming what to avoid is a guardrail
        that blocks a known failure (baking in a single-brain artifact) WITHOUT
        narrowing the useful search space — the opposite effect of an anchor.
    """
    if not priors_path:
        return ""
    return (
        f"\n\nA validated, cross-run knowledge base of U-Net error regularities is "
        f"available at {priors_path} (produced by the AutoDiscovery workflow: each "
        f"finding carries Reproduction / Generalization / Verdict / Post-correction "
        f"tokens). READ it and use it as a PRIOR to ground your improvement in "
        f"already-verified geometry/topology — do not re-derive from scratch what it "
        f"already establishes. DISCIPLINE on which findings to trust:\n"
        f"  • Survey ALL the findings and USE only those whose Generalization is "
        f"GENERALIZES AND whose post-correction Verdict is UPHELD or OK (robust "
        f"across every brain AND after cluster-robust statistical correction). Do "
        f"not privilege any particular finding — read the file, judge each by its "
        f"verdict tokens, and pick the one(s) most relevant to THIS generation's "
        f"failure report.\n"
        f"  • DISTRUST and do NOT bake in any finding marked DOES-NOT-GENERALIZE, "
        f"PARTIAL, WEAKENED, or OVERTURNED (e.g. Z-axis anisotropy, centrifugal "
        f"branch-order, omit/split-near-merge co-location) — those held only on one "
        f"brain or collapsed under correction, so a policy built on them would "
        f"overfit a single brain.\n"
        f"  • Prefer a lever that you and prior generations (see the attempts list) "
        f"have NOT tried yet — breadth across the qualifying findings beats "
        f"re-tuning the same one.\n"
        f"  • The priors are GT-derived population statistics about error geometry, "
        f"NOT this run's labels — using their thresholds/discriminators is fair game "
        f"and does NOT violate the no-hardcoded-label rule; never copy a raw "
        f"segment-id literal.\n"
        f"Cite the finding number(s) you relied on in your rules.md change log."
    )


async def ask_reviser(
    client: ClaudeSDKClient, report_path: str,
    heuristics_path: str, rules_path: str, verbose: bool,
    attempts: list[dict] | None = None,
    priors_path: str | None = None,
):
    """Run the proofreader-reviser subagent on the failure report. Returns
    (text, input_tokens, output_tokens, cost_usd).

    The agent is told to edit THIS RUN's working copies (under runs/<id>/artifacts),
    not the pristine originals under proofreader_evolve/artifacts. ``attempts`` is
    the memory of revisions already tried against the current parent (B), woven
    into the prompt so the agent proposes something NEW. ``priors_path``, when
    given, points the agent at the validated discovery knowledge base (see
    ``_format_priors`` for the trust discipline applied to it).
    """
    instruction = (
        "Use the proofreader-reviser subagent to improve the evolved proofreading "
        f"program. The current policy lives in {heuristics_path} and its theory in "
        f"{rules_path}. Edit THOSE files in place (do not touch any other files). "
        f"The failure report for the latest candidate is at {report_path}. "
        "Diagnose why the policy lost accuracy, make ONE concrete improvement to "
        "propose_edits (keeping its call signature), and update the rules file and "
        "its change log. Do NOT claim you ran, imported, or tested anything — you "
        "have no Bash; the harness import-checks and lint-checks your edit after "
        "you finish."
        + _format_priors(priors_path)
        + _format_attempts(attempts or [])
        + ("\nPropose a DIFFERENT improvement from any listed above."
           if attempts else "")
    )
    await client.query(instruction)
    # Capture the SUBAGENT's text, not the orchestrator's. The reviser runs inside
    # a Task tool-use; its AssistantMessages carry a non-None ``parent_tool_use_id``
    # (the Task's id), whereas the orchestrator's own narration ("I'll delegate
    # this to the … subagent") has ``parent_tool_use_id is None``. Collecting only
    # the parented text gives us the actual diagnosis/reasoning; the orchestrator
    # text is kept separately as a fallback in case no subagent text is surfaced.
    sub_chunks, orch_chunks = [], []
    in_tok = out_tok = 0
    cost = 0.0
    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            is_sub = getattr(message, "parent_tool_use_id", None) is not None
            for block in message.content:
                if isinstance(block, TextBlock):
                    (sub_chunks if is_sub else orch_chunks).append(block.text)
                    if verbose:
                        print(block.text, end="", flush=True)
                elif ToolUseBlock and isinstance(block, ToolUseBlock):
                    log(f"    → {getattr(block, 'name', 'tool')}"
                        f"{' [subagent]' if is_sub else ''}")
            # Sum token usage across ALL assistant messages (orchestrator +
            # subagent), so the ledger reflects the subagent's real consumption,
            # not just the parent's final ResultMessage.
            usage = getattr(message, "usage", None) or {}
            in_tok += usage.get("input_tokens", 0) or 0
            out_tok += usage.get("output_tokens", 0) or 0
        elif isinstance(message, ResultMessage):
            cost = getattr(message, "total_cost_usd", 0.0) or 0.0
            # Fall back to the result usage only if no per-message usage was seen.
            if in_tok == 0 and out_tok == 0:
                ru = getattr(message, "usage", None) or {}
                in_tok = ru.get("input_tokens", 0) or 0
                out_tok = ru.get("output_tokens", 0) or 0
    # Prefer the subagent's diagnosis; fall back to orchestrator text if the SDK
    # surfaced none (older SDKs / different routing).
    text = "".join(sub_chunks) or "".join(orch_chunks)
    return text, in_tok, out_tok, cost


def lint_no_hardcoded_labels(
    heuristics_path: str, report_labels: set, min_label_len: int = 4
) -> tuple[bool, str]:
    """Reject a policy that HARDCODES raw segment-id labels from the failure report.

    The failure report lists, for diagnosis, the exact raw segment ids that span
    multiple GT neurons (the `split_label` repair targets) and the label pairs an
    edit merged. Those ids are TRAIN-split, GT-derived facts. A policy that branches
    on a specific id — e.g. ``if s.label == "123456": split_label(...)`` — scores
    well on train by construction but cannot generalize to held-out/test, where
    those ids never appear. The reviser prompt forbids this, but instruction is not
    enforcement; this lint makes it a hard, auditable rule (the reviser has no Bash
    to self-check, so the harness must).

    Method (AST, precise — does NOT flag legitimate numeric thresholds like
    ``gap < 5.0`` or ``angle > 120``): collect every numeric and string CONSTANT in
    the policy whose normalized text equals one of the report's raw labels. Only
    labels of at least ``min_label_len`` characters are considered, so small tuning
    constants can never trip it; segment ids are long integers. Returns
    ``(ok, reason)`` — ``ok=False`` lists the offending labels.

    ``report_labels`` is the set of raw label strings the report exposed this run
    (merge targets + any fused/edited labels). Empty set => the lint is a no-op.
    """
    import ast as _ast

    targets = {str(x) for x in report_labels if len(str(x)) >= min_label_len}
    if not targets:
        return True, "no hardcode lint targets (no long raw labels in report)"
    try:
        tree = _ast.parse(open(heuristics_path).read())
    except Exception as e:  # a non-parsing policy is caught by the import check
        return True, f"lint skipped (policy did not parse: {e})"

    found = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Constant):
            v = node.value
            if isinstance(v, bool):
                continue
            if isinstance(v, int):
                text = str(v)
            elif isinstance(v, str):
                text = v
            else:
                continue  # floats are thresholds, never segment ids
            if text in targets:
                found.add(text)

    if found:
        sample = ", ".join(sorted(found)[:5])
        return False, (f"policy hardcodes {len(found)} raw segment-id label(s) from "
                       f"the failure report ({sample}) — overfits train, will not "
                       f"generalize. Decide from site features / ctx, never a "
                       f"specific label literal.")
    return True, f"no hardcoded labels (checked {len(targets)} report labels)"


def collect_report_labels(merge_labels: dict, train_run) -> set:
    """The raw segment-id labels the failure report exposes this generation.

    These are the ids a policy could copy to cheat: the baseline merge targets
    (``merge_labels`` keys) plus the labels named in the candidate's own edits
    (merge endpoints / split labels). The lint forbids the policy SOURCE from
    containing any of these as a literal.

    Mirrors the failure report's TRAIN-only isolation: the report only ever names
    a merge label when >=2 of its GT neurons fall in this run's TRAIN skeletons
    (see ``candidate._failure_report_body``). We restrict the merge-target ids to
    that same set so the lint key-set is a pure function of train GT — never the
    full-brain ``merge_labels`` keys, which would leak held-out label membership.
    """
    per_swc = getattr(getattr(train_run, "score", None), "per_swc", None)
    report_gt = set(per_swc.index) if per_swc is not None else set()
    labels: set = {
        str(label) for label, info in (merge_labels or {}).items()
        if sum(1 for n in info.get("gt_skeletons", []) if n in report_gt) >= 2
    }
    for e in (getattr(train_run, "edits", None) or []):
        if not isinstance(e, dict):
            continue
        for k in ("label", "label_a", "label_b"):
            if k in e and e[k] is not None:
                labels.add(str(e[k]))
    return labels


def evaluate_gate(
    cand_metrics: dict,
    parent_metrics: dict,
    has_split_edit: bool,
    gate_eps: float,
    split_tol: float = 0.05,
    merge_tol: float = 0.0,
) -> tuple[bool, str]:
    """Decide acceptance from the FULL metric vector, not Edge Accuracy alone.

    Edge Accuracy (= 100 - %Split - %Omit - %Merged) is the primary fitness: every
    generation must first beat the parent by ``gate_eps``.

    NO-NEW-MERGE guard (EVERY generation). The reviser's stated objective is
    "maximize Edge Accuracy WITHOUT creating merge errors", but Edge Accuracy alone
    does not enforce the second clause: a candidate that repairs many splits while
    introducing a few merges can still raise net Edge Accuracy and slip through. And
    ``merge_labels`` — the merge-only path's only edit — is precisely the action that
    creates merge errors (it fuses two labels; if they belong to different GT
    neurons that fusion IS a merge error). So we require, on EVERY generation:
      * # Merges did not increase (beyond ``merge_tol``), and
      * % Merged Edges did not increase (beyond ``merge_tol``).
    This is safe — it can never reject a CORRECT merge repair, because unifying two
    fragments of the SAME neuron never raises # Merges or % Merged Edges; only a
    wrong fusion does. Both components are checked, not just # Merges: folding a
    clean label into an ALREADY-merged one leaves # Merges flat while % Merged Edges
    climbs (the bad label now spans more edges), so # Merges alone would miss it.
    (This replaces the old design, which guarded merge components only on the
    split_label path and left merge-only generations gated on Edge Accuracy alone —
    the exact hole this closes. The old rationale conflated the %Split guard, which
    CAN wrongly reject good merge repairs that benignly raise #Splits via
    fix_label_misalignments, with the #Merges/%Merged guard, which cannot.)

    The over-split watchdog applies ONLY when the candidate emits at least one
    ``split_label`` edit — the action that can trade a merge penalty for a split
    penalty and over-split a real neuron while still looking net-positive on Edge
    Accuracy on a small held-out set: % Split Edges must not rise by more than
    ``split_tol`` above the parent (a small tolerance absorbs the benign
    misalignment-fill effect; a real over-split blows past it). It is NOT applied to
    merge-only generations, where a legitimate ``merge_labels`` repair can raise
    BOTH #Splits and Edge Accuracy together (fix_label_misalignments fills
    background gaps, adding distinct labels and coverage at once; see verify.py
    handler-parity notes), so a blanket %Split guard would wrongly reject it.

    Returns ``(keep, reason)``; ``reason`` is a short human-readable string for the
    log / attempts archive, naming the specific guard that fired.
    """
    def m(d, k):
        v = d.get(k, float("nan"))
        return v

    cand_acc = m(cand_metrics, "Edge Accuracy")
    parent_acc = m(parent_metrics, "Edge Accuracy")

    # Primary gate (applies to every generation).
    if not (cand_acc > parent_acc + gate_eps):
        return False, (f"Edge Accuracy {cand_acc:.3f} did not beat parent "
                       f"{parent_acc:.3f} by eps {gate_eps:.3f}")

    # No-new-merge guard (EVERY generation). A correct repair never trips this; only
    # a candidate that introduces / grows a merge error does. The 1e-9 absorbs
    # float noise when merge_tol is 0.
    cand_merged = m(cand_metrics, "% Merged Edges")
    parent_merged = m(parent_metrics, "% Merged Edges")
    cand_nmerge = m(cand_metrics, "# Merges")
    parent_nmerge = m(parent_metrics, "# Merges")
    if cand_nmerge > parent_nmerge + merge_tol + 1e-9:
        return False, (f"created merge error: # Merges {parent_nmerge:.2f} -> "
                       f"{cand_nmerge:.2f} (exceeds parent + tol {merge_tol:.3f})")
    if cand_merged > parent_merged + merge_tol + 1e-9:
        return False, (f"created merge error: % Merged Edges {parent_merged:.3f} -> "
                       f"{cand_merged:.3f} (exceeds parent + tol {merge_tol:.3f})")

    if not has_split_edit:
        return True, (f"Edge Accuracy {cand_acc:.3f} > parent {parent_acc:.3f} "
                      f"+ {gate_eps:.3f}; no new merge (#Merges {parent_nmerge:.2f}->"
                      f"{cand_nmerge:.2f}, %Merged {parent_merged:.3f}->"
                      f"{cand_merged:.3f}) (merge-only path)")

    # --- split_label generation: additionally enforce the over-split watchdog ---
    cand_split = m(cand_metrics, "% Split Edges")
    parent_split = m(parent_metrics, "% Split Edges")
    if cand_split > parent_split + split_tol:
        return False, (f"split_label over-split: % Split Edges {parent_split:.3f} "
                       f"-> {cand_split:.3f} exceeds parent + tol {split_tol:.3f}")

    return True, (f"Edge Accuracy {cand_acc:.3f} > parent {parent_acc:.3f}; "
                  f"merge not worsened (%Merged {parent_merged:.3f}->{cand_merged:.3f}, "
                  f"#Merges {parent_nmerge:.2f}->{cand_nmerge:.2f}) without "
                  f"over-splitting (%Split {parent_split:.3f}->{cand_split:.3f})")


def no_per_skeleton_merge_regression(cand_per_swc, parent_per_swc, merge_tol: float = 0.0):
    """Per-skeleton hard guard: no held-out skeleton may GAIN a merge error.

    The pooled / per-fold merge guard in ``evaluate_gate`` compares run-length-
    weighted AVERAGES, so a candidate that creates a merge on one skeleton while
    removing one on another can net to flat (or even improve) and slip through —
    yet a merge error WAS created. This guard is strictly stronger: it requires
    that NO individual held-out skeleton regress on either merge component
    (``# Merges`` or ``% Merged Edges``).

    Correct repairs never trip it: unifying two fragments of the SAME neuron never
    raises that neuron's ``# Merges`` or ``% Merged Edges`` (only a wrong fusion
    does), so the per-skeleton check inherits the same can't-reject-a-good-repair
    invariant the pooled guard relies on — just enforced row by row. The ``1e-9``
    absorbs float noise when ``merge_tol`` is 0; skeletons missing from the parent
    (NaN after reindex) compare False and are never flagged.

    Returns the sub-frame of ``cand_per_swc`` rows that regressed (empty == pass).
    """
    parent = parent_per_swc.reindex(cand_per_swc.index)
    bad_nmerge = cand_per_swc["# Merges"] > parent["# Merges"] + merge_tol + 1e-9
    bad_pct = cand_per_swc["% Merged Edges"] > parent["% Merged Edges"] + merge_tol + 1e-9
    return cand_per_swc[bad_nmerge | bad_pct]


_FOLD_METRIC_COLS = ("Edge Accuracy", "% Merged Edges", "# Merges", "% Split Edges")


def fold_metrics(per_swc, fold_names: list[str]) -> dict:
    """Run-length-weighted metric vector over just one fold's held-out skeletons.

    ``per_swc`` is a ScoreResult.per_swc frame (one row per GT skeleton, scored on
    the FULL held-out set in a single pass); ``fold_names`` selects this fold's rows.
    Returns the same metric keys ``evaluate_gate`` consumes, so a fold is gated
    exactly like the old single held-out set.
    """
    rows = per_swc.loc[per_swc.index.isin(fold_names)]
    return {m: scoring._weighted_avg(rows, m) for m in _FOLD_METRIC_COLS}


def evaluate_gate_kfold(
    cand_folds: list[dict], parent_folds: list[dict], has_split_edit: bool,
    gate_eps: float, split_tol: float = 0.05, merge_tol: float = 0.0,
) -> tuple[bool, str, dict]:
    """Aggregate the per-fold gate into one accept/reject across K folds.

    A candidate is kept only when BOTH hold:
      * AGGREGATE improvement: mean held-out Edge Accuracy across folds beats the
        parent's mean by ``gate_eps`` (the stable signal K-fold exists to provide —
        a single fold is too noisy to gate on), AND
      * NO PER-FOLD REGRESSION on the hard guards: ``evaluate_gate`` must pass on
        EVERY fold (no fold may gain Edge Accuracy by creating a merge there, and a
        split_label generation must not over-split in any fold). Per-fold for the
        guards is deliberately strict — a merge error created in one fold is a real
        regression even if the mean still rises.

    Returns ``(keep, reason, summary)`` where ``summary`` carries the per-fold and
    mean Edge Accuracy for the log / ledger.
    """
    import statistics
    cand_accs = [f["Edge Accuracy"] for f in cand_folds]
    parent_accs = [f["Edge Accuracy"] for f in parent_folds]
    cand_mean = statistics.fmean(cand_accs)
    parent_mean = statistics.fmean(parent_accs)
    stdev = statistics.pstdev(cand_accs) if len(cand_accs) > 1 else 0.0
    summary = {
        "cand_fold_acc": cand_accs, "parent_fold_acc": parent_accs,
        "cand_mean": cand_mean, "parent_mean": parent_mean, "cand_stdev": stdev,
    }

    # Aggregate gate: mean must beat parent mean by eps.
    if not (cand_mean > parent_mean + gate_eps):
        return (False,
                f"mean Edge Accuracy {cand_mean:.3f} did not beat parent "
                f"{parent_mean:.3f} by eps {gate_eps:.3f} "
                f"(folds {[round(a,2) for a in cand_accs]})",
                summary)

    # Per-fold hard guards: every fold must individually pass evaluate_gate. Use
    # gate_eps=-inf there so the per-fold Edge-Accuracy beat is NOT re-imposed (the
    # mean already enforces improvement); we only want the no-new-merge / over-split
    # guards to fire per fold.
    for i, (cf, pf) in enumerate(zip(cand_folds, parent_folds)):
        ok, why = evaluate_gate(
            cf, pf, has_split_edit=has_split_edit, gate_eps=float("-inf"),
            split_tol=split_tol, merge_tol=merge_tol,
        )
        if not ok:
            return (False, f"fold {i} guard failed: {why}", summary)

    return (True,
            f"mean Edge Accuracy {cand_mean:.3f} > parent {parent_mean:.3f} "
            f"+ {gate_eps:.3f} (folds {[round(a,2) for a in cand_accs]}, "
            f"±{stdev:.3f}); no per-fold merge/over-split regression",
            summary)


def human_gate(gen: int, train_acc: float, heldout_acc: float, parent_acc: float) -> bool:
    """Optional human checkpoint before committing a revision (reviewer's
    'how much human feedback was required' — each call is one intervention)."""
    print(f"\n--- Human gate (generation {gen}) ---")
    print(f"  parent held-out Edge Accuracy:   {parent_acc:.4f}")
    print(f"  candidate train  Edge Accuracy:  {train_acc:.4f}")
    print(f"  candidate held-out Edge Accuracy:{heldout_acc:.4f}")
    ans = input("  Keep this revision? [y/N] ").strip().lower()
    return ans == "y"


@dataclass
class BrainContext:
    """Everything needed to score one brain, kept separate per brain.

    Brains are NEVER pooled at the data level: a raw segment-id label is not
    comparable across brains, and each brain has its own ~1.6 GB prepared state,
    fragment graph, and image reader. We score each brain independently and pool
    only the per-skeleton RESULTS (the per_swc rows, whose index — e.g.
    ``N005-789202-SP`` — already carries the brain id, so pooled rows never collide).
    """
    brain: str
    fragments_graph: object
    image_reader: object
    prepared: object
    baseline_full: object                 # ScoreResult on the whole brain (no edits)
    train_names: list                     # this brain's train skeletons
    heldout_names: list                   # this brain's held-out skeletons
    merge_labels: dict                     # this brain's baseline merge targets
    base_train: object = None              # per_swc baseline rows for train
    base_heldout: object = None            # per_swc baseline rows for held-out
    label_gt_map: dict = None              # TRAIN-only {label: {gt_neuron: count}}
                                           # for the leak-free SplitSite audit
    heldout_label_gt_map: dict = None      # HELD-OUT {label: {gt_neuron: count}} —
                                           # GATE-ONLY split-repair scoring. LEAK
                                           # BOUNDARY: never goes into the failure
                                           # report / reviser; only the gate reads it.


def _setup_brain(brain: str, run_dir: Path, heldout_fraction: float,
                 split_seed: int, with_image: bool, verbose: bool) -> BrainContext:
    """Load + prepare ONE brain and compute its train/held-out split + baseline.

    Mirrors the original single-brain setup, factored out so a run can hold several
    brains at once. The prepared-brain pickle (expensive) is reused from any prior
    run if present, exactly as before.
    """
    paths = scoring.BrainPaths(brain)
    cache_path = ds.default_cache_path(brain)
    log(f"[{brain}] Loading cached fragment graph: {cache_path}")
    fragments_graph, _gt_graph, _ = ds.load_cached_graphs(cache_path)

    image_reader = None
    if with_image:
        from proofreader_evolve.harness.image_features import LazyImagePatchReader
        import sys as _sys
        _scripts = str(PROJECT_ROOT / "scripts")
        _sys.path.insert(0, _scripts)
        from dataset_config import get_img_path  # noqa: E402
        _prefixes = str(PROJECT_ROOT / "configs" / "exaspim_image_prefixes.json")
        img_path = get_img_path(brain, prefixes_path=_prefixes)
        image_reader = LazyImagePatchReader(img_path, fragments_graph)
        log(f"[{brain}] Image patch reader ENABLED (lazy): {img_path}")

    prepared_cache = str(run_dir / f"prepared_{brain}.pkl")
    if not os.path.exists(prepared_cache):
        reuse = _find_existing_prepared(brain, exclude=run_dir)
        if reuse:
            log(f"[{brain}] Reusing prepared brain from a prior run: {reuse}")
            shutil.copy2(reuse, prepared_cache)
    log(f"[{brain}] Preparing brain for incremental scoring (cache: {prepared_cache})")
    with Heartbeat(f"[{brain}] preparing brain (load/build — can be ~30 min cold)"):
        prepared = inc.get_or_build(paths, prepared_cache, verbose=verbose)

    with Heartbeat(f"[{brain}] scoring baseline"):
        baseline_full = inc.score_incremental(prepared, label_pairs=None, verbose=verbose)
    all_gt_names = list(baseline_full.per_swc.index)
    # Per-brain 70/30 (or as configured) split, seeded so it is reproducible. Each
    # brain splits its OWN skeletons, so no brain lands wholly in train or held-out.
    train_names, heldout_names = ds.train_heldout_split(
        all_gt_names, heldout_fraction=heldout_fraction, seed=split_seed
    )
    merge_labels = inc.collect_merge_labels(prepared)
    # TRAIN-ONLY label->{gt_neuron: count} map for the SplitSite audit. Restricting
    # to train_names keeps the SplitSite TRUE/NON verdict leak-free (held-out
    # membership is never revealed).
    label_gt_map = inc.label_gt_counts(prepared, gt_names=train_names)
    # HELD-OUT label->neuron map: GATE-ONLY (the dense split-repair gate signal).
    # Kept strictly separate from the train-only map above — only the gate reads it,
    # never the failure report / reviser, so held-out membership is never leaked.
    heldout_label_gt_map = inc.label_gt_counts(prepared, gt_names=heldout_names)
    log(f"[{brain}] {len(all_gt_names)} GT -> {len(train_names)} train / "
        f"{len(heldout_names)} held-out; {len(merge_labels)} baseline merge target(s)")

    base_train = baseline_full.per_swc.loc[
        baseline_full.per_swc.index.isin(train_names)]
    base_heldout = baseline_full.per_swc.loc[
        baseline_full.per_swc.index.isin(heldout_names)]
    return BrainContext(
        brain=brain, fragments_graph=fragments_graph, image_reader=image_reader,
        prepared=prepared, baseline_full=baseline_full,
        train_names=train_names, heldout_names=heldout_names,
        merge_labels=merge_labels, base_train=base_train, base_heldout=base_heldout,
        label_gt_map=label_gt_map, heldout_label_gt_map=heldout_label_gt_map,
    )


def _score_pooled(brains: list, names_attr: str, work_heuristics: str,
                  split_name: str, max_class_size, verbose: bool):
    """Run the policy on every brain's own (train|heldout) skeletons and POOL results.

    Each brain is scored with ITS OWN prepared state + fragment graph + image reader
    (data is never crossed). Returns ``(pooled_per_swc, per_brain_runs)`` where
    ``pooled_per_swc`` concatenates each brain's per-skeleton rows (indices carry the
    brain id; ``verify_integrity`` enforces no collision) and ``per_brain_runs`` is the list of each
    brain's CandidateRun (kept for the per-brain failure report). The pooled frame is
    what the run-length-weighted gate metrics are computed over, so a brain with more
    cable carries proportionally more weight — the same weighting the metric uses
    within a brain, applied across the pool.
    """
    import pandas as pd
    per_brain_runs = []
    frames = []
    for bc in brains:
        names = getattr(bc, names_attr)
        run = cand.run_candidate(
            bc.prepared, bc.fragments_graph, names, split_name,
            work_heuristics, max_class_size=max_class_size,
            image_reader=bc.image_reader, verbose=verbose,
        )
        per_brain_runs.append(run)
        frames.append(run.score.per_swc)
    # verify_integrity: pooling weights metrics by name, so a cross-brain name
    # collision would double-count silently — fail fast at the concat instead.
    pooled = pd.concat(frames, verify_integrity=True) if frames else None
    return pooled, per_brain_runs


def _pooled_split_repair(brains: list, per_brain_runs: list) -> dict:
    """Pool the dense split-repair gate signal across brains.

    For each brain, classify that brain's held-out edits against ITS OWN held-out
    label→neuron map (raw labels are not comparable across brains, so each brain is
    classified with its own map and only the COUNTS are pooled). Returns
    ``{"correct": int, "false": int, "unscored": int, "score": int}`` where
    ``score = correct - false`` is the primary fitness (dense: every repaired split
    counts, unlike Edge Accuracy which only moves on a bridged split edge).

    GATE-ONLY: reads each brain's ``heldout_label_gt_map``; never exposed to the
    reviser.
    """
    tot = {"correct": 0, "false": 0, "unscored": 0}
    for bc, run in zip(brains, per_brain_runs):
        c = inc.classify_merge_edits(run.edits, bc.heldout_label_gt_map)
        tot["correct"] += c["correct"]
        tot["false"] += c["false"]
        tot["unscored"] += c["unscored"]
    tot["score"] = tot["correct"] - tot["false"]
    return tot


async def run_evolution(
    brain: str, generations: int, heldout_fraction: float,
    human: bool, verbose: bool, model: str = DEFAULT_MODEL,
    gate_eps: float = 0.05, max_class_size=None, seed_from: str | None = None,
    split_seed: int | None = None, with_image: bool = True,
    split_tol: float = 0.05, merge_tol: float = 0.0, k_folds: int = 1,
    brains: list | None = None,
) -> None:
    # Brain set: --brains (list) takes precedence; else the single --brain. The run
    # id uses the first brain + a tag of the count so multi-brain runs are obvious.
    brain_list = [str(b) for b in (brains or [brain])]
    primary_brain = brain_list[0]
    run_tag = primary_brain if len(brain_list) == 1 else f"{primary_brain}+{len(brain_list)-1}"
    run_id = f"{run_tag}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = HERE / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(str(run_dir / "ledger.jsonl"))

    # Work on TIMESTAMPED COPIES of the artifacts, never the originals. The agent
    # reads/revises only these; artifacts/ stays pristine. By default the start is
    # the pristine seed (from scratch); --seed-from continues a prior run's policy.
    work_heuristics, work_rules = make_working_artifacts(run_dir, seed_from=seed_from)
    log(f"Run {run_id}")
    log(f"  start policy: {'continuing run ' + seed_from if seed_from else 'from-scratch seed'}")
    log(f"  originals (untouched): {HEURISTICS}")
    log(f"  working copies (revised this run): {work_heuristics}")

    # --- One-time setup: load+prepare EVERY brain (expensive, cached) -----------
    # Split seed: RANDOM by default; drawn once and shared across brains so every
    # brain's 70/30 split is reproducible from this one recorded seed.
    if split_seed is None:
        split_seed = int.from_bytes(os.urandom(4), "little")
    log(f"Brains: {brain_list} (split_seed={split_seed}, "
        f"heldout_fraction={heldout_fraction})")
    brain_ctxs = [
        _setup_brain(b, run_dir, heldout_fraction, split_seed, with_image, verbose)
        for b in brain_list
    ]

    # Pooled held-out names across brains. Pooling indexes skeletons BY NAME (concat
    # / isin / loc / fold slicing), which silently corrupts — duplicate rows, double-
    # counted weights — if two brains share a skeleton name. The name is expected to
    # embed the brain id (e.g. N005-789202-SP), but the harness never enforces that,
    # and passing the same brain twice (--brains 789202,789202) would collide too. So
    # fail FAST here, at the pool boundary, turning a silent miscount into a clear
    # startup error before any scoring runs.
    import pandas as _pd
    pooled_heldout = [n for bc in brain_ctxs for n in bc.heldout_names]
    _dupes = sorted({n for n in pooled_heldout if pooled_heldout.count(n) > 1})
    assert not _dupes, (
        f"pooled held-out skeleton names collide across brains: "
        f"{_dupes[:5]}{' …' if len(_dupes) > 5 else ''} "
        f"({len(_dupes)} total). Pooling indexes by name, so names must be globally "
        f"unique — check that each brain's skeletons embed its brain id and that no "
        f"brain was passed twice."
    )
    # verify_integrity re-checks the same invariant at the frame level (the rows that
    # actually get weighted), so a future name-source change can't reintroduce it.
    pooled_base_heldout = _pd.concat(
        [bc.base_heldout for bc in brain_ctxs], verify_integrity=True)
    baseline_heldout = scoring._weighted_avg(pooled_base_heldout, "Edge Accuracy")
    log(f"Pooled held-out: {len(pooled_heldout)} neurons across {len(brain_ctxs)} "
        f"brain(s); baseline (no-edit) Edge Accuracy = {baseline_heldout:.4f}")

    # K-fold the POOLED held-out set for gating (per-brain train reports unchanged).
    # k_folds=1 => one fold == the whole pooled held-out, i.e. the single-split gate
    # over the pool. The folds are name lists; scoring slices the pooled per_swc.
    k_eff = max(1, min(k_folds, len(pooled_heldout)))
    if k_eff != k_folds:
        log(f"k_folds {k_folds} clamped to {k_eff} (only {len(pooled_heldout)} "
            f"pooled held-out neurons)")
    if k_eff <= 1:
        heldout_folds = [{"fold": 0, "heldout": list(pooled_heldout)}]
    else:
        heldout_folds = ds.kfold_split(pooled_heldout, k=k_eff, seed=split_seed)
    log(f"Held-out K-fold: {k_eff} fold(s), sizes "
        f"{[len(f['heldout']) for f in heldout_folds]}")

    # Persist the seed + the exact per-brain partition so the run is reproducible.
    (run_dir / "split.json").write_text(json.dumps({
        "split_seed": split_seed,
        "heldout_fraction": heldout_fraction,
        "brains": brain_list,
        "per_brain": {bc.brain: {"train": bc.train_names,
                                 "heldout": bc.heldout_names} for bc in brain_ctxs},
        "k_folds": k_eff,
        "heldout_folds": [f["heldout"] for f in heldout_folds],
    }, indent=2))

    # PARENT-RELATIVE gate: each generation must beat the CURRENT policy (its
    # parent). Score the SEED on the pooled held-out once to set the bar.
    log("Scoring SEED policy on pooled held-out — sets the bar gen 1 must beat...")
    with Heartbeat("scoring seed policy on pooled held-out"):
        seed_pooled, seed_runs = _score_pooled(
            brain_ctxs, "heldout_names", str(work_heuristics), "heldout",
            max_class_size, verbose,
        )
    # Dense split-repair fitness of the SEED (the PRIMARY gate signal): correct
    # held-out merges minus false ones. Edge Accuracy stays computed/recorded but is
    # no longer the bar — it reads +0.000 for most real repairs (only a bridged
    # split EDGE moves it), which is why evolution flat-lined. See classify_merge_edits.
    parent_repair = _pooled_split_repair(brain_ctxs, seed_runs)
    log(f"Seed split-repair: correct={parent_repair['correct']} "
        f"false={parent_repair['false']} score={parent_repair['score']} "
        f"(unscored={parent_repair['unscored']}) — the bar gen 1 must beat")
    # Full pooled held-out metric vector of the parent (run-length-weighted over the
    # pool). Advances on every accept.
    parent_metrics = {m: scoring._weighted_avg(seed_pooled, m)
                      for m in _FOLD_METRIC_COLS}
    parent_heldout = parent_metrics["Edge Accuracy"]
    # Per-skeleton parent rows (the SAME pooled pass), kept so the per-skeleton merge
    # guard can compare each held-out skeleton against the accepted parent. Advances
    # on every accept, in lockstep with parent_metrics / parent_fold_metrics.
    parent_pooled = seed_pooled
    # Per-fold parent metric vectors — sliced from the SAME pooled scoring pass.
    parent_fold_metrics = [
        fold_metrics(seed_pooled, f["heldout"]) for f in heldout_folds
    ]
    import statistics as _st
    parent_mean_acc = _st.fmean(m["Edge Accuracy"] for m in parent_fold_metrics)
    log(f"Seed policy pooled held-out Edge Accuracy = {parent_heldout:.4f} "
        f"(full); K-fold mean = {parent_mean_acc:.4f} over {k_eff} fold(s) "
        f"(the bar generation 1 must beat)")

    options, guard_state = build_options(model=model, run_dir=run_dir)
    log(f"Reviser model: {model} (Anthropic API)")
    log("Run isolation: reviser tools = Read/Write/Edit/Task; cross-run file "
        "access DENIED via can_use_tool (audit -> tool_audit.jsonl)")
    # Validated discovery priors: pass the ABSOLUTE path (the reviser's cwd is the
    # run dir, and the file lives outside runs/, so the isolation guard allows
    # reading it). None when the knowledge base is absent => prompt unchanged.
    priors_path = str(DISCOVERY_PRIORS.resolve()) if DISCOVERY_PRIORS.is_file() else None
    if priors_path:
        log(f"Discovery priors: reviser will read {priors_path} "
            f"(trusting GENERALIZES+UPHELD findings only)")
    else:
        log(f"Discovery priors: none found at {DISCOVERY_PRIORS} (reviser runs without)")
    # B: memory of revisions tried against the CURRENT parent; cleared when the
    # parent advances (an accept), since past rejections no longer apply.
    attempts_vs_parent: list[dict] = []
    # Durable mirror of every attempt (survives crashes/restarts), tagged with the
    # parent bar each was tried against.
    attempts_log = run_dir / "attempts.md"
    async with ClaudeSDKClient(options=options) as client:
        for gen in range(1, generations + 1):
            print(f"\n=== Generation {gen}/{generations} ===")
            gen_wall0 = time.monotonic()
            gen_dir = run_dir / f"gen{gen:02d}"
            snapshot(gen_dir, work_heuristics, work_rules)  # revert source if rejected

            # (1-3) Run CURRENT policy on each brain's TRAIN; build the failure
            # report. Per brain so raw labels / merge targets stay brain-local.
            log("Step 1-3: run current policy on train, build failure report...")
            with Heartbeat(f"gen {gen}: running policy on train"):
                _, train_runs = _score_pooled(
                    brain_ctxs, "train_names", str(work_heuristics), "train",
                    max_class_size, verbose,
                )
            report_path = str(gen_dir / "failure_report.md")
            per_brain_report = [
                (bc.brain, tr,
                 scoring.ScoreResult(  # baseline on this brain's train, for the report
                     primary=scoring._weighted_avg(bc.base_train, "Edge Accuracy"),
                     metrics={}, per_swc=bc.base_train, output_dir="", seconds=0.0),
                 bc.merge_labels, bc.label_gt_map)
                for bc, tr in zip(brain_ctxs, train_runs)
            ]
            if len(per_brain_report) == 1:
                _, tr, base_sr, ml, lgm = per_brain_report[0]
                cand.write_failure_report(tr, base_sr, report_path,
                                          merge_labels=ml, label_gt_map=lgm)
            else:
                cand.write_multibrain_failure_report(per_brain_report, report_path)
            train_acc = scoring._weighted_avg(
                _pd.concat([tr.score.per_swc for tr in train_runs]), "Edge Accuracy")
            n_edits_total = sum(tr.n_edits for tr in train_runs)
            log(f"   train Edge Accuracy={train_acc:.4f} "
                f"({n_edits_total} edits across {len(train_runs)} brain(s)); "
                f"report -> {report_path}")

            # (4-5) Ask the agent to explain and revise the WORKING-COPY artifacts.
            log("Step 4-5: proofreader-reviser diagnoses and revises artifacts...")
            with Heartbeat(f"gen {gen}: waiting on reviser (LLM)"):
                diagnosis, in_tok, out_tok, cost = await ask_reviser(
                    client, report_path, str(work_heuristics), str(work_rules), verbose,
                    attempts=attempts_vs_parent, priors_path=priors_path,
                )

            # (A) Persist what the reviser wrote THIS generation — before scoring or
            # any revert — so even rejected candidates are inspectable afterward.
            candidate_path, diffstat = save_candidate(gen_dir, work_heuristics, work_rules)
            log(f"   candidate saved -> {candidate_path} (diffstat vs parent: {diffstat})")

            # Harness-side import check (the reviser no longer has Bash to do it).
            # A revision that doesn't import is a dead candidate -> revert to parent.
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location("_cand_check", str(work_heuristics))
            try:
                _m = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_m)
                assert hasattr(_m, "propose_edits"), "policy lost propose_edits"
                import_ok = True
            except Exception as _e:
                import_ok = False
                log(f"   [WARN] revised policy does not import ({_e}); reverting this gen")
                revert(gen_dir, work_heuristics, work_rules)

            # No-hardcode lint (P1-4): a policy that copies raw segment-id labels
            # from the failure report overfits train and won't generalize, so it is
            # rejected exactly like a failed import. Uses THIS generation's report
            # labels (baseline merge targets + the candidate's own edited labels).
            if import_ok:
                report_labels = set()
                for bc, tr in zip(brain_ctxs, train_runs):
                    report_labels |= collect_report_labels(bc.merge_labels, tr)
                lint_ok, lint_reason = lint_no_hardcoded_labels(
                    str(work_heuristics), report_labels
                )
                if not lint_ok:
                    import_ok = False
                    log(f"   [WARN] no-hardcode lint FAILED: {lint_reason}; "
                        f"reverting this gen")
                    revert(gen_dir, work_heuristics, work_rules)

            # The bar this gen tried to beat (pre-update): the split-repair score,
            # which is now the gate's decision variable.
            parent_bar = parent_repair["score"]
            train_seconds = sum(tr.score.seconds for tr in train_runs)
            if not import_ok:
                # Revision was already reverted to the parent above; don't waste a
                # held-out scoring pass on it. Record as a non-improving gen.
                heldout_acc = parent_mean_acc
                cand_repair = dict(parent_repair)  # no change: candidate == parent
                heldout_n_edits = 0                # nothing scored on held-out
                eval_seconds = train_seconds
                keep = False
                human_touches = 0
            else:
                # (6) Re-run the REVISED policy on each brain's HELD-OUT, pool results.
                log("Step 6: score revised policy on pooled held-out...")
                with Heartbeat(f"gen {gen}: scoring revised policy on pooled held-out"):
                    heldout_pooled, heldout_runs = _score_pooled(
                        brain_ctxs, "heldout_names", str(work_heuristics), "heldout",
                        max_class_size, verbose,
                    )
                heldout_acc = scoring._weighted_avg(heldout_pooled, "Edge Accuracy")
                eval_seconds = train_seconds + sum(
                    hr.score.seconds for hr in heldout_runs)

                # Per-fold candidate metrics, sliced from the pooled scoring pass —
                # still computed + recorded (Edge Accuracy is informative), but no
                # longer the bar. They feed the ledger / log, not the keep decision.
                cand_fold_metrics = [
                    fold_metrics(heldout_pooled, f["heldout"])
                    for f in heldout_folds
                ]
                cand_pooled_metrics = {m: scoring._weighted_avg(heldout_pooled, m)
                                       for m in _FOLD_METRIC_COLS}

                # (7) PRIMARY GATE = dense split-repair score on pooled held-out.
                # Edge Accuracy only moves when a merge bridges a true split EDGE, so
                # most correct repairs read +0.000 and evolution flat-lined. The
                # split-repair score counts EVERY correctly-repaired held-out split
                # (correct) and penalizes every wrong fusion (false), giving a
                # gradient that responds to each policy change. A candidate is kept
                # iff it makes MORE net correct repairs than the parent AND creates
                # ZERO false merges on held-out (the no-new-merge guard, now exact:
                # a false merge is a fusion of two different held-out neurons).
                cand_repair = _pooled_split_repair(brain_ctxs, heldout_runs)
                heldout_n_edits = sum(hr.n_edits for hr in heldout_runs)
                has_split_edit = any(
                    isinstance(e, dict) and e.get("kind") == "split_label"
                    for hr in heldout_runs for e in (hr.edits or [])
                )
                if cand_repair["false"] > 0:
                    improved = False
                    gate_reason = (
                        f"created {cand_repair['false']} false merge(s) on held-out "
                        f"(fused different neurons) — rejected regardless of repairs "
                        f"(correct={cand_repair['correct']})")
                elif cand_repair["score"] > parent_repair["score"]:
                    improved = True
                    gate_reason = (
                        f"split-repair score {parent_repair['score']} -> "
                        f"{cand_repair['score']} (correct {parent_repair['correct']}"
                        f"->{cand_repair['correct']}, false 0); no false merges")
                else:
                    improved = False
                    gate_reason = (
                        f"split-repair score {cand_repair['score']} did not beat "
                        f"parent {parent_repair['score']} "
                        f"(correct={cand_repair['correct']}, false=0)")
                heldout_acc = fold_summary_mean = cand_pooled_metrics["Edge Accuracy"]
                fold_summary = {"cand_mean": heldout_acc,
                                "cand_fold_acc": [m["Edge Accuracy"] for m in cand_fold_metrics],
                                "cand_stdev": 0.0}
                log(f"   gate: {gate_reason}")
                log(f"   (recorded: Edge Accuracy {heldout_acc:.4f}, "
                    f"%Merged {cand_pooled_metrics['% Merged Edges']:.4f}, "
                    f"#Merges {cand_pooled_metrics['# Merges']:.2f})")
                human_touches = 0
                if human:
                    human_touches = 1
                    keep = human_gate(gen, train_acc, heldout_acc, parent_heldout)
                else:
                    keep = improved

            # B: one-line summary of what this generation tried, for the memory.
            attempt_summary = (
                f"{diffstat} lines; " + (diagnosis or "").strip().split("\n", 1)[0][:120]
            ) or "(no diagnosis text)"
            if keep:
                parent_heldout = cand_pooled_metrics["Edge Accuracy"]  # pooled primary
                parent_mean_acc = heldout_acc                # recorded Edge Accuracy
                # Advance the metric vectors AND the split-repair bar so the next
                # generation's gate measures against THIS accepted child, not a stale
                # baseline. Only set when we actually scored held-out (import-failed
                # gens keep the prior parent).
                if import_ok:
                    parent_metrics = dict(cand_pooled_metrics)
                    parent_fold_metrics = cand_fold_metrics
                    parent_pooled = heldout_pooled
                    parent_repair = cand_repair       # advance the primary gate bar
                shutil.copy2(work_heuristics, gen_dir / "heuristics.accepted.py")
                shutil.copy2(work_rules, gen_dir / "rules.accepted.md")
                note = "accepted (new parent)"
                if k_eff > 1 and import_ok:
                    note += (f" [folds {[round(a,2) for a in fold_summary['cand_fold_acc']]}"
                             f" ±{fold_summary['cand_stdev']:.3f}]")
                # Parent advanced: prior rejections were against the OLD parent and
                # no longer apply, so clear the in-prompt memory.
                attempts_vs_parent = []
            else:
                revert(gen_dir, work_heuristics, work_rules)  # restore the parent
                note = "reverted (did not beat parent)"
                # Remember this rejected attempt so the next gen proposes something new.
                attempts_vs_parent.append({
                    "gen": gen,
                    "summary": attempt_summary,
                    "heldout": cand_repair["score"] - parent_bar,
                    "accepted": False,
                })
            # B (durable): append every generation to a file that survives crashes
            # and restarts. Tagged with the split-repair bar it was tried against, so
            # a later reader can tell which attempts are still relevant (same parent).
            with open(attempts_log, "a") as f:
                f.write(f"- gen{gen:02d} [vs parent split-repair {parent_bar}]: "
                        f"score {cand_repair['score']} "
                        f"({cand_repair['score'] - parent_bar:+d}; "
                        f"correct={cand_repair['correct']}, false={cand_repair['false']}) "
                        f"-> {note}; {attempt_summary}\n")
            log(f"Step 7: split-repair score={cand_repair['score']} "
                f"(correct={cand_repair['correct']}, false={cand_repair['false']}; "
                f"parent={parent_bar}); Edge Accuracy={heldout_acc:.4f} -> {note}")

            ledger.record(GenerationCost(
                generation=gen,
                wall_seconds=time.monotonic() - gen_wall0,
                eval_seconds=eval_seconds,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=cost,
                n_evaluations=2,
                human_interventions=human_touches,
                train_primary=train_acc,
                heldout_primary=heldout_acc,
                parent_heldout=parent_bar,
                accepted=keep,
                note=note,
                heldout_n_edits=heldout_n_edits,
                heldout_correct_merges=cand_repair["correct"],
                heldout_false_merges=cand_repair["false"],
                heldout_split_repair_score=cand_repair["score"],
                candidate_path=candidate_path,
                heuristics_diffstat=diffstat,
                diagnosis=(diagnosis or "")[:2000],  # truncate; full text is in stdout
            ))

    # Run-isolation audit: if the reviser ever attempted to touch a sibling run's
    # files, the guard denied it — but flag the run loudly so the result isn't
    # trusted as an independent sample.
    n_viol = len(guard_state["violations"]) if guard_state else 0
    if n_viol:
        log(f"[POLLUTION WARNING] {n_viol} denied cross-run file access attempt(s) "
            f"by the reviser — see {run_dir / 'tool_audit.jsonl'}. The edits were "
            f"NOT informed by other runs (access was blocked), but treat this run's "
            f"'independence' with suspicion.")
        (run_dir / "POLLUTION_ATTEMPTED").write_text(
            json.dumps(guard_state["violations"], indent=2))

    print("\n=== Evolution complete ===")
    print(ledger.summarize())
    print(f"Cross-run access attempts (denied): {n_viol}")
    print(f"Originals (unchanged)        -> {ARTIFACTS}")
    print(f"Evolved artifacts (this run) -> {work_heuristics.parent}")
    print(f"Run log + ledger             -> {run_dir}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--brain", default="789202", help="brain_id (must have a cache pkl)")
    p.add_argument("--brains", default=None,
                   help="comma-separated brain_ids to POOL (e.g. "
                        "'789202,794491,794492'). Each brain is loaded + prepared + "
                        "split 70/30 separately; the gate scores the policy on every "
                        "brain's own train/held-out and pools the per-skeleton "
                        "results (data is never crossed — raw labels are not "
                        "comparable across brains). Overrides --brain. NOTE: each "
                        "brain holds ~1.6 GB prepared state in memory simultaneously.")
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--heldout-fraction", type=float, default=0.5,
                   help="fraction of EACH brain's GT skeletons reserved for held-out "
                        "gating (the rest are train). Default 0.5; pass 0.3 for the "
                        "70/30 train/test split.")
    p.add_argument("--k-folds", type=int, default=1,
                   help="K-fold cross-validation WITHIN the held-out set for gating. "
                        "The candidate must beat the parent on the MEAN held-out "
                        "Edge Accuracy across K disjoint folds AND not regress on the "
                        "no-new-merge / over-split guards in ANY fold. Default 1 "
                        "(single held-out set == legacy behavior). With few held-out "
                        "neurons, K>1 (e.g. 4) turns a noisy single split into a "
                        "stable mean so small true improvements become detectable. "
                        "Clamped to the held-out neuron count. Scoring cost is "
                        "unchanged — folds are row-slices of one scoring pass.")
    p.add_argument("--human-gate", action="store_true",
                   help="ask a human before keeping each revision")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"reviser model id (default: {DEFAULT_MODEL}, Anthropic API)")
    p.add_argument("--gate-eps", type=float, default=0.05,
                   help="held-out Edge Accuracy margin a generation must beat the "
                        "parent by to be accepted (guards against noise-level wins)")
    p.add_argument("--max-class-size", type=int, default=None,
                   help="hard cap on labels fused into one merge class (guardrail "
                        "against brain-spanning mega-merges); default: no cap")
    p.add_argument("--split-tol", type=float, default=0.05,
                   help="for generations that emit split_label edits: max amount "
                        "%% Split Edges may rise above the parent before the gate "
                        "rejects it as over-splitting. Merge-only gens are "
                        "unaffected by this over-split watchdog.")
    p.add_argument("--merge-tol", type=float, default=0.0,
                   help="no-new-merge guard (EVERY generation): max amount # Merges "
                        "and %% Merged Edges may rise above the parent before the "
                        "gate rejects the candidate as creating a merge error. "
                        "Default 0.0 (strict — accept only if merge error does not "
                        "grow); raise to allow a split-for-merge trade, set very "
                        "high to disable and recover the old Edge-Accuracy-only "
                        "behavior on the merge-only path.")
    p.add_argument("--seed-from", default=None,
                   help="CONTINUE from a prior run's latest accepted policy "
                        "(run-id folder name, or brain id for its newest run) "
                        "instead of the from-scratch seed; default: from scratch")
    p.add_argument("--split-seed", type=int, default=None,
                   help="RNG seed for the train/held-out split. Default: random "
                        "per run (recorded in the run's split.json). Pass an int "
                        "to pin a reproducible split.")
    p.add_argument("--with-image", action=argparse.BooleanOptionalAction, default=True,
                   help="give the policy a LAZY raw-image patch reader in ctx "
                        "(ctx['read_image_patch']) so it can test fluorescence "
                        "continuity at a gap; each read is a cloud fetch, so the "
                        "policy must gate reads behind cheap filters. Default ON; "
                        "pass --no-with-image to disable (skeleton-only, no cloud "
                        "reads).")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    brains = None
    if args.brains:
        brains = [b.strip() for b in args.brains.split(",") if b.strip()]
    asyncio.run(run_evolution(
        args.brain, args.generations, args.heldout_fraction,
        args.human_gate, args.verbose, args.model,
        gate_eps=args.gate_eps, max_class_size=args.max_class_size,
        seed_from=args.seed_from, split_seed=args.split_seed,
        with_image=args.with_image, split_tol=args.split_tol,
        merge_tol=args.merge_tol, k_folds=args.k_folds, brains=brains,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

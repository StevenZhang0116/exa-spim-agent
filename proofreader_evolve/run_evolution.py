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
import time
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


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


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
    }


def _make_isolation_guard(run_dir: Path, audit_path: Path):
    """Build a ``can_use_tool`` callback that ENFORCES run isolation.

    Cross-run pollution channel: the reviser runs with file tools inside the
    ``runs/`` tree, and nothing structurally stops it from READING a sibling run's
    accepted policy / ledger / SUMMARY and copying the answer (destroying the
    "independent rediscovery" claim and any honest cross-run variance estimate).
    The prompt only restricts writes. This callback closes the channel at the
    permission layer: any tool call whose arguments reference a path under
    ``runs/`` OTHER than the current run is DENIED, and every denial is recorded
    so a tainted generation is detectable.

    Returns ``(callback, state)`` where ``state['violations']`` accumulates the
    denied attempts (also appended to ``audit_path``).
    """
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    runs_root = (HERE / "runs").resolve()
    this_run = run_dir.resolve()
    state = {"violations": []}

    # Tool-input fields that carry file paths / shell text, by tool name.
    PATH_FIELDS = {
        "Read": ("file_path",), "Write": ("file_path",), "Edit": ("file_path",),
        "Glob": ("path",), "NotebookEdit": ("notebook_path",),
        "Grep": ("path",),
    }

    def _offending_paths(tool_name: str, ti: dict) -> list[str]:
        """Return path-like strings in this tool call that point at a SIBLING run."""
        candidates = []
        for f in PATH_FIELDS.get(tool_name, ()):
            v = ti.get(f)
            if isinstance(v, str):
                candidates.append(v)
        if tool_name == "Bash":
            # Can't parse shell reliably; flag any mention of the runs/ tree.
            candidates.append(ti.get("command", ""))
        bad = []
        for c in candidates:
            if not c:
                continue
            # Bash: substring check on the runs root (path may be embedded in a cmd).
            if tool_name == "Bash":
                if "runs/" in c or str(runs_root) in c:
                    # allow only if every runs/ mention is this run
                    import re
                    for m in re.findall(r"\S*runs/\S*", c):
                        try:
                            rp = Path(m).resolve()
                        except Exception:
                            rp = Path(m)
                        if runs_root in rp.parents and this_run not in (rp, *rp.parents):
                            bad.append(m)
                continue
            try:
                rp = Path(c).resolve()
            except Exception:
                continue
            if runs_root in rp.parents and this_run != rp and this_run not in rp.parents:
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
                message=(f"Run isolation: {tool_name} may not access another run's "
                         f"files ({bad}). Only the current run dir is permitted."),
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
    ``can_use_tool`` callback that DENIES any file/shell tool call referencing a
    SIBLING run's files, and switch the permission mode off ``bypassPermissions``
    (which would skip the callback) to ``acceptEdits`` (non-interactive, but the
    callback still fires). The tool set is also trimmed to what the reviser needs.
    Returns ``(options, guard_state)``; ``guard_state`` is None when no run_dir.
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


async def ask_reviser(
    client: ClaudeSDKClient, report_path: str,
    heuristics_path: str, rules_path: str, verbose: bool,
    attempts: list[dict] | None = None,
):
    """Run the proofreader-reviser subagent on the failure report. Returns
    (text, input_tokens, output_tokens, cost_usd).

    The agent is told to edit THIS RUN's working copies (under runs/<id>/artifacts),
    not the pristine originals under proofreader_evolve/artifacts. ``attempts`` is
    the memory of revisions already tried against the current parent (B), woven
    into the prompt so the agent proposes something NEW.
    """
    instruction = (
        "Use the proofreader-reviser subagent to improve the evolved proofreading "
        f"program. The current policy lives in {heuristics_path} and its theory in "
        f"{rules_path}. Edit THOSE files in place (do not touch any other files). "
        f"The failure report for the latest candidate is at {report_path}. "
        "Diagnose why the policy lost accuracy, make ONE concrete improvement to "
        "propose_edits (keeping its call signature), update the rules file and its "
        "change log, and confirm the module imports."
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
    """
    labels: set = set(map(str, (merge_labels or {}).keys()))
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
) -> tuple[bool, str]:
    """Decide acceptance from the FULL metric vector, not Edge Accuracy alone.

    Edge Accuracy (= 100 - %Split - %Omit - %Merged) is the primary fitness, and a
    pure merge-only generation is gated on it exactly as before: it must beat the
    parent by ``gate_eps``. We do NOT add component guards to the merge-only path —
    a legitimate ``merge_labels`` repair can raise BOTH #Splits and Edge Accuracy
    together (fix_label_misalignments fills background gaps, adding distinct labels
    and coverage at once; see verify.py handler-parity notes), so a blanket
    %Split/#Merges guard would wrongly reject good merge repairs.

    The extra guards apply ONLY when the candidate emits at least one ``split_label``
    edit — the action that can trade a merge penalty for a split penalty and still
    look net-positive on Edge Accuracy while over-splitting a real neuron on a small
    held-out set. For such a generation we additionally require:
      * the merge component actually improved (% Merged Edges did not get worse, and
        # Merges did not increase) — i.e. the split repaired what it claimed to, and
      * the over-split watchdog: % Split Edges did not rise by more than ``split_tol``
        above the parent (a small tolerance absorbs the benign misalignment-fill
        effect; a real over-split blows past it).

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

    if not has_split_edit:
        return True, (f"Edge Accuracy {cand_acc:.3f} > parent {parent_acc:.3f} "
                      f"+ {gate_eps:.3f} (merge-only path)")

    # --- split_label generation: enforce the merge-repair + over-split guards ---
    cand_merged = m(cand_metrics, "% Merged Edges")
    parent_merged = m(parent_metrics, "% Merged Edges")
    cand_nmerge = m(cand_metrics, "# Merges")
    parent_nmerge = m(parent_metrics, "# Merges")
    cand_split = m(cand_metrics, "% Split Edges")
    parent_split = m(parent_metrics, "% Split Edges")

    # The split must not WORSEN the merge component it exists to repair.
    if cand_merged > parent_merged + 1e-9:
        return False, (f"split_label raised % Merged Edges "
                       f"{parent_merged:.3f} -> {cand_merged:.3f} (a merge repair "
                       f"must not increase merge error)")
    if cand_nmerge > parent_nmerge + 1e-9:
        return False, (f"split_label raised # Merges {parent_nmerge:.2f} -> "
                       f"{cand_nmerge:.2f}")
    # Over-split watchdog: % Split Edges must not climb beyond tolerance.
    if cand_split > parent_split + split_tol:
        return False, (f"split_label over-split: % Split Edges {parent_split:.3f} "
                       f"-> {cand_split:.3f} exceeds parent + tol {split_tol:.3f}")

    return True, (f"Edge Accuracy {cand_acc:.3f} > parent {parent_acc:.3f}; "
                  f"merge repaired (%Merged {parent_merged:.3f}->{cand_merged:.3f}, "
                  f"#Merges {parent_nmerge:.2f}->{cand_nmerge:.2f}) without "
                  f"over-splitting (%Split {parent_split:.3f}->{cand_split:.3f})")


def human_gate(gen: int, train_acc: float, heldout_acc: float, parent_acc: float) -> bool:
    """Optional human checkpoint before committing a revision (reviewer's
    'how much human feedback was required' — each call is one intervention)."""
    print(f"\n--- Human gate (generation {gen}) ---")
    print(f"  parent held-out Edge Accuracy:   {parent_acc:.4f}")
    print(f"  candidate train  Edge Accuracy:  {train_acc:.4f}")
    print(f"  candidate held-out Edge Accuracy:{heldout_acc:.4f}")
    ans = input("  Keep this revision? [y/N] ").strip().lower()
    return ans == "y"


async def run_evolution(
    brain: str, generations: int, heldout_fraction: float,
    human: bool, verbose: bool, model: str = DEFAULT_MODEL,
    gate_eps: float = 0.05, max_class_size=None, seed_from: str | None = None,
    split_seed: int | None = None, with_image: bool = True,
    split_tol: float = 0.05,
) -> None:
    run_id = f"{brain}_{datetime.now():%Y%m%d_%H%M%S}"
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

    # --- One-time setup: prepared brain (expensive, cached), split, baseline --
    paths = scoring.BrainPaths(brain)

    # Cached fragment graph drives candidate-site GEOMETRY (fast, no GCS).
    cache_path = ds.default_cache_path(brain)
    log(f"Loading cached fragment graph (site geometry): {cache_path}")
    fragments_graph, _gt_graph, _ = ds.load_cached_graphs(cache_path)

    # Optional lazy image-patch reader for the policy (off unless --with-image).
    # Built once and shared across generations; opens the cloud client only on the
    # first read, so enabling it is free until the policy actually queries it.
    image_reader = None
    if with_image:
        from proofreader_evolve.harness.image_features import LazyImagePatchReader
        import sys as _sys
        # configs/ and scripts/ live in PROJECT_ROOT (exa-spim-agent/), one level
        # above proofreader_evolve/ -- not at the outer repo root.
        _scripts = str(PROJECT_ROOT / "scripts")
        _sys.path.insert(0, _scripts)
        from dataset_config import get_img_path  # noqa: E402
        _prefixes = str(PROJECT_ROOT / "configs" / "exaspim_image_prefixes.json")
        img_path = get_img_path(brain, prefixes_path=_prefixes)
        image_reader = LazyImagePatchReader(img_path, fragments_graph)
        log(f"Image patch reader ENABLED (lazy): {img_path}")

    # PreparedBrain drives SCORING. Built once (~30 min, ~1.6 GB), then pickled.
    # It lives in THIS run's folder, but the ~30 min build is candidate-invariant
    # and identical across runs of the same brain, so we REUSE an existing pickle
    # from any prior run folder (or the legacy runs/ location) rather than rebuild.
    prepared_cache = str(run_dir / f"prepared_{brain}.pkl")
    if not os.path.exists(prepared_cache):
        reuse = _find_existing_prepared(brain, exclude=run_dir)
        if reuse:
            log(f"Reusing prepared brain from a prior run: {reuse}")
            shutil.copy2(reuse, prepared_cache)
    log(f"Preparing brain for incremental scoring (cache: {prepared_cache})")
    prepared = inc.get_or_build(paths, prepared_cache, verbose=verbose)

    log("Scoring BASELINE (no edits) incrementally — sets the bar and GT split...")
    baseline_full = inc.score_incremental(prepared, label_pairs=None, verbose=verbose)
    all_gt_names = list(baseline_full.per_swc.index)
    # Split seed: RANDOM by default (different train/held-out partition each run,
    # so the gate isn't perpetually optimizing one fixed split). We draw a concrete
    # seed and RECORD it, so the run is still reproducible after the fact; pass
    # --split-seed to pin it.
    if split_seed is None:
        split_seed = int.from_bytes(os.urandom(4), "little")
    train_names, heldout_names = ds.train_heldout_split(
        all_gt_names, heldout_fraction=heldout_fraction, seed=split_seed
    )
    log(f"GT skeletons: {len(all_gt_names)} total -> "
        f"{len(train_names)} train, {len(heldout_names)} held-out "
        f"(split_seed={split_seed})")
    # Persist the seed + the exact partition so the run can be reproduced/audited.
    (run_dir / "split.json").write_text(json.dumps({
        "split_seed": split_seed,
        "heldout_fraction": heldout_fraction,
        "train": train_names,
        "heldout": heldout_names,
    }, indent=2))

    # Baseline (no-edit) restricted to each split — the reference floor and the
    # comparison shown in the failure report.
    base_train = baseline_full.per_swc.loc[
        baseline_full.per_swc.index.isin(train_names)]
    base_heldout = baseline_full.per_swc.loc[
        baseline_full.per_swc.index.isin(heldout_names)]
    baseline_heldout = scoring._weighted_avg(base_heldout, "Edge Accuracy")
    log(f"Baseline (no-edit) held-out Edge Accuracy = {baseline_heldout:.4f}")

    # Baseline (pre-existing) merge errors by raw label — the split_label repair
    # targets surfaced in the failure report. Candidate-invariant (GT-derived), so
    # compute ONCE here. DIAGNOSIS-ONLY: passed to the report, which restricts it to
    # the TRAIN skeletons; it never reaches a held-out policy.
    merge_labels = inc.collect_merge_labels(prepared)
    log(f"Baseline merge errors: {len(merge_labels)} raw label(s) span >=2 GT "
        f"neurons (the split_label repair targets)")

    # PARENT-RELATIVE gate: each generation must beat the CURRENT policy (its
    # parent), not the global no-edit baseline. The parent is the last accepted
    # policy (initially the seed in the working copy). Score the seed once to set
    # the bar — with a no-op seed this equals the baseline, but scoring it makes
    # the gate correct for ANY seed and lets sub-baseline lineages still climb.
    seed_heldout_run = cand.run_candidate(
        prepared, fragments_graph, heldout_names, "heldout",
        str(work_heuristics), max_class_size=max_class_size,
                image_reader=image_reader, verbose=verbose,
    )
    parent_heldout = seed_heldout_run.score.primary
    # Full held-out metric vector of the current parent, so the gate can enforce
    # component guards (% Merged Edges / # Merges / % Split Edges) for split_label
    # generations — not just the Edge Accuracy scalar. Advances on every accept.
    parent_metrics = dict(seed_heldout_run.score.metrics)
    log(f"Seed policy held-out Edge Accuracy = {parent_heldout:.4f} "
        f"(the bar generation 1 must beat)")

    options, guard_state = build_options(model=model, run_dir=run_dir)
    log(f"Reviser model: {model} (Anthropic API)")
    log("Run isolation: reviser tools = Read/Write/Edit/Task; cross-run file "
        "access DENIED via can_use_tool (audit -> tool_audit.jsonl)")
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

            # (1-3) Run CURRENT policy on TRAIN; build the failure report.
            log("Step 1-3: run current policy on train, build failure report...")
            train_run = cand.run_candidate(
                prepared, fragments_graph, train_names, "train",
                str(work_heuristics), max_class_size=max_class_size,
                image_reader=image_reader, verbose=verbose,
            )
            report_path = str(gen_dir / "failure_report.md")
            cand.write_failure_report(
                train_run,
                scoring.ScoreResult(  # baseline on train, wrapped for the report
                    primary=scoring._weighted_avg(base_train, "Edge Accuracy"),
                    metrics={}, per_swc=base_train, output_dir="", seconds=0.0,
                ),
                report_path,
                merge_labels=merge_labels,  # baseline merge targets (train-restricted in-report)
            )
            log(f"   train Edge Accuracy={train_run.score.primary:.4f} "
                f"({train_run.n_edits} edits); report -> {report_path}")

            # (4-5) Ask the agent to explain and revise the WORKING-COPY artifacts.
            log("Step 4-5: proofreader-reviser diagnoses and revises artifacts...")
            diagnosis, in_tok, out_tok, cost = await ask_reviser(
                client, report_path, str(work_heuristics), str(work_rules), verbose,
                attempts=attempts_vs_parent,
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
                report_labels = collect_report_labels(merge_labels, train_run)
                lint_ok, lint_reason = lint_no_hardcoded_labels(
                    str(work_heuristics), report_labels
                )
                if not lint_ok:
                    import_ok = False
                    log(f"   [WARN] no-hardcode lint FAILED: {lint_reason}; "
                        f"reverting this gen")
                    revert(gen_dir, work_heuristics, work_rules)

            parent_bar = parent_heldout  # the bar this gen tried to beat (pre-update)
            if not import_ok:
                # Revision was already reverted to the parent above; don't waste a
                # held-out scoring pass on it. Record as a non-improving gen.
                heldout_acc = parent_heldout
                eval_seconds = train_run.score.seconds
                keep = False
                human_touches = 0
            else:
                # (6) Re-run the REVISED policy on HELD-OUT and score.
                log("Step 6: score revised policy on held-out...")
                heldout_run = cand.run_candidate(
                    prepared, fragments_graph, heldout_names, "heldout",
                    str(work_heuristics), max_class_size=max_class_size,
                    image_reader=image_reader, verbose=verbose,
                )
                heldout_acc = heldout_run.score.primary
                eval_seconds = train_run.score.seconds + heldout_run.score.seconds

                # (7) Gate: held-out Edge Accuracy must beat the PARENT by a margin
                # (gate_eps; few held-out skeletons -> strict '>' would lock in
                # noise-level wins). For a generation that emits split_label, ALSO
                # require the merge component to improve and the over-split watchdog
                # to hold — Edge Accuracy alone can mask a merge-for-split trade on a
                # small held-out set. Merge-only generations are gated on Edge
                # Accuracy exactly as before (no component guards). See evaluate_gate.
                has_split_edit = any(
                    isinstance(e, dict) and e.get("kind") == "split_label"
                    for e in (heldout_run.edits or [])
                )
                improved, gate_reason = evaluate_gate(
                    heldout_run.score.metrics, parent_metrics,
                    has_split_edit=has_split_edit, gate_eps=gate_eps,
                    split_tol=split_tol,
                )
                log(f"   gate: {gate_reason}")
                human_touches = 0
                if human:
                    human_touches = 1
                    keep = human_gate(gen, train_run.score.primary, heldout_acc, parent_heldout)
                else:
                    keep = improved

            # B: one-line summary of what this generation tried, for the memory.
            attempt_summary = (
                f"{diffstat} lines; " + (diagnosis or "").strip().split("\n", 1)[0][:120]
            ) or "(no diagnosis text)"
            if keep:
                parent_heldout = heldout_acc  # this child becomes the new parent
                # Advance the full metric vector too, so the next generation's gate
                # (esp. the split_label component guards) measures against THIS
                # accepted child, not a stale baseline. Only set when we actually
                # scored held-out (import-failed gens keep the prior parent).
                if import_ok:
                    parent_metrics = dict(heldout_run.score.metrics)
                shutil.copy2(work_heuristics, gen_dir / "heuristics.accepted.py")
                shutil.copy2(work_rules, gen_dir / "rules.accepted.md")
                note = "accepted (new parent)"
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
                    "heldout": heldout_acc - parent_bar,
                    "accepted": False,
                })
            # B (durable): append every generation to a file that survives crashes
            # and restarts. Tagged with the parent bar it was tried against, so a
            # later reader can tell which attempts are still relevant (same parent).
            with open(attempts_log, "a") as f:
                f.write(f"- gen{gen:02d} [vs parent {parent_bar:.3f}]: "
                        f"held-out {heldout_acc:.3f} ({heldout_acc - parent_bar:+.3f}) "
                        f"-> {note}; {attempt_summary}\n")
            log(f"Step 7: held-out Edge Accuracy={heldout_acc:.4f} "
                f"(parent={parent_heldout:.4f}, baseline={baseline_heldout:.4f}) -> {note}")

            ledger.record(GenerationCost(
                generation=gen,
                wall_seconds=time.monotonic() - gen_wall0,
                eval_seconds=eval_seconds,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=cost,
                n_evaluations=2,
                human_interventions=human_touches,
                train_primary=train_run.score.primary,
                heldout_primary=heldout_acc,
                parent_heldout=parent_bar,
                accepted=keep,
                note=note,
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
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--heldout-fraction", type=float, default=0.33)
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
                        "% Split Edges may rise above the parent before the gate "
                        "rejects it as over-splitting (also requires % Merged Edges "
                        "/ # Merges not to worsen). Merge-only gens are unaffected.")
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
    asyncio.run(run_evolution(
        args.brain, args.generations, args.heldout_fraction,
        args.human_gate, args.verbose, args.model,
        gate_eps=args.gate_eps, max_class_size=args.max_class_size,
        seed_from=args.seed_from, split_seed=args.split_seed,
        with_image=args.with_image, split_tol=args.split_tol,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

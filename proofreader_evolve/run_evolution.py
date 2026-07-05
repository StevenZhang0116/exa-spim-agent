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

The gate's decision variable is the DENSE held-out split-repair fitness
``(correct - false) - merge_penalty * false`` (see ``_fitness`` / step 7), kept
iff it beats the parent by ``score_margin``. Edge Accuracy is recorded for
diagnosis but is NOT the bar — it reads +0.000 for most real repairs, so it
flat-lined as a gate. A false merge is penalized (default 100 each), not
hard-rejected. The seed policy is a conservative COLINEAR SPLIT-REPAIR policy
(not a no-op — see artifacts/heuristics.py::propose_edits): it emits a small,
high-precision set of ``merge_labels`` so the score starts off the flat no-edit
baseline and the loop has a gradient. The gate is parent-relative (each
generation must beat the last accepted policy, not the fixed baseline), so
accepted improvements accumulate generation over generation.

The hard fitness only sees GT-covered edits; in a sparsely-traced brain most
edits are ``unscored`` (no GT at either endpoint). Rather than penalize those
(which would just teach the policy to avoid untraced neurons), a CONFIDENCE layer
attaches a GT-INDEPENDENT soft verdict to that blind spot — the image bridge_ratio
the policy already fetched, replayed at zero extra cloud reads (see
``_image_confidence`` / ``_confidence_level``). It reports how much of each
decision is GT-backed vs. resting on an unverified blind spot (high/medium/low),
logged and recorded in the ledger as an advisory label. It NEVER enters the hard
fitness; only the opt-in ``--confidence-veto`` may block an accepted generation,
and only on POSITIVE image evidence of a likely-false merge — never on absence of
GT.

The evolved "program" is the pair (artifacts/heuristics.py, artifacts/rules.md).
Each generation snapshots them, lets the agent revise, re-scores on held-out, and
either keeps or reverts. Every generation's cost is recorded.

Run from the project root (exa-spim-agent/):
    python proofreader_evolve/run_evolution.py --brain 789202 --generations 5
    python proofreader_evolve/run_evolution.py --brain 789202 --generations 5 --human-gate
    # split-error-only, mega-merge guard on, wider candidate stream (mcl10), no priors:
    python proofreader_evolve/run_evolution.py --brain 789202 --generations 40 --no-priors --splits-only --max-class-size 6 --mcl 10
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
try:
    # Extended-thinking / reasoning block. Present only on SDKs that surface
    # thinking; ``()`` makes the isinstance check below a harmless no-op otherwise.
    from claude_agent_sdk import ThinkingBlock
except ImportError:  # pragma: no cover
    ThinkingBlock = ()  # type: ignore
try:
    # Tool RESULT block (what a tool returned) — captured into the transcript so the
    # saved record shows what the reviser's Read/Edit calls actually produced.
    from claude_agent_sdk import ToolResultBlock
except ImportError:  # pragma: no cover
    ToolResultBlock = ()  # type: ignore

from proofreader_evolve.harness import (
    scoring,
    dataset as ds,
    candidate as cand,
    incremental_scoring as inc,
    priors as priors_kb,
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


class _Tee:
    """A write-through stream that mirrors everything to the real console stream AND
    an open log file, so the full run output is persisted without losing the live
    terminal view. Wraps ONE console stream (stdout or stderr); both are wrapped so
    ``log()``/heartbeat (stderr), the banners/summary and the reviser stream
    (stdout), and any crashing traceback all land in the same file, in order.

    Only the methods callers actually use are proxied; ``isatty`` reports the console
    stream's value so downstream code that checks for a TTY still behaves correctly.
    Failures writing to the file are swallowed — a logging problem must never take
    down the run.
    """

    def __init__(self, console, fh) -> None:
        self._console = console
        self._fh = fh

    def write(self, s):
        n = self._console.write(s)
        try:
            self._fh.write(s)
        except Exception:
            pass
        return n

    def flush(self):
        self._console.flush()
        try:
            self._fh.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self._console, "isatty", lambda: False)()

    def __getattr__(self, name):
        # Delegate anything not overridden (encoding, fileno, …) to the console.
        return getattr(self._console, name)


# The currently-installed run-log capture (single-shot CLI => at most one). main()'s
# finally tears it down via detach_active_run_log() so the file is flushed/closed and
# the streams restored on BOTH the normal and the crash path.
_ACTIVE_LOG_CAP = None


class run_log_capture:
    """Tees stdout+stderr to ``run_dir/run.log`` for the run.

    A header line records the wall-clock start and argv so a saved log is
    self-describing. Append mode ('a') so a resumed/re-run into the same dir extends
    rather than truncates the record. Register-on-enter so ``detach_active_run_log()``
    (called from main()'s finally) restores the streams and closes the file even when
    the run raises — the interpreter prints any traceback to ``sys.stderr`` (still the
    tee) before that finally runs, so the crash is captured.
    """

    def __init__(self, run_dir: Path) -> None:
        self._path = Path(run_dir) / "run.log"
        self._fh = None
        self._saved = None

    def __enter__(self) -> Path:
        global _ACTIVE_LOG_CAP
        self._fh = open(self._path, "a", buffering=1)  # line-buffered
        self._fh.write(
            f"\n{'='*70}\n# run.log — started {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"# argv: {' '.join(sys.argv)}\n{'='*70}\n")
        self._saved = (sys.stdout, sys.stderr)
        sys.stdout = _Tee(sys.stdout, self._fh)
        sys.stderr = _Tee(sys.stderr, self._fh)
        _ACTIVE_LOG_CAP = self
        return self._path

    def __exit__(self, *exc) -> None:
        global _ACTIVE_LOG_CAP
        if self._saved is not None:
            sys.stdout, sys.stderr = self._saved
            self._saved = None
        if self._fh is not None:
            try:
                self._fh.flush(); self._fh.close()
            except Exception:
                pass
            self._fh = None
        if _ACTIVE_LOG_CAP is self:
            _ACTIVE_LOG_CAP = None
        return False  # never suppress exceptions


def detach_active_run_log() -> None:
    """Tear down the active run-log capture, if any (idempotent). Called by main()'s
    finally so streams are restored and the log flushed on every exit path."""
    if _ACTIVE_LOG_CAP is not None:
        _ACTIVE_LOG_CAP.__exit__(None, None, None)


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
        # Reasoning effort for the evolution/reviser work. Levels:
        # low|medium|high|xhigh|max; Opus 4.8 thinks adaptively. ``high`` keeps strong
        # diagnostic reasoning at lower latency/cost than ``xhigh``. Applies to the
        # session + subagents.
        "CLAUDE_EFFORT": "high",
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


STRUCTURAL_LESSON_CAP = 12  # max over-merge taboos shown, to bound prompt tokens


def _format_structural_lessons(lessons: list[dict]) -> str:
    """Render the cross-parent 'over-merge taboo' list for the prompt.

    Unlike ``_format_attempts`` (which is scoped to the CURRENT parent and cleared
    on every accept), this list holds revisions that were REJECTED and had also
    created a false merge on held-out. A false merge means the change fused two
    DIFFERENT neurons — that is a property of the geometric condition the change
    loosened, NOT of the parent it was competing against — so the lesson must PERSIST
    across parents. Otherwise the reviser re-derives the same over-merge direction
    against each new parent and the gate's fitness penalty has to catch it again.
    Only scalar counts + the change summary are shown (no neuron identity), so this is
    the same leak-safety level as the attempts list.
    """
    if not lessons:
        return ""
    lines = ["\nOVER-MERGE TABOO (persists across accepted generations). Each change "
             "below CREATED >=1 false merge on held-out (fused two DIFFERENT neurons) "
             "and was rejected. Under the smoothed gate a false merge is not an "
             "automatic reject, but each costs a heavy fitness penalty (~merge_penalty "
             "correct repairs), so these are high over-merge-RISK directions, NOT "
             "parent-specific misses: only revisit one with MATERIALLY STRONGER "
             "precision evidence (e.g. an image bridge_ratio confirmation) or a much "
             "larger recall payoff, never by merely re-loosening the same geometric "
             "threshold:"]
    for L in lessons:
        lines.append(f"  - gen{L['gen']}: {L['summary']} -> created {L['false']} "
                     f"false merge(s)")
    return "\n".join(lines) + "\n"


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
             "(do NOT repeat these — they did not beat it). The number is the "
             "held-out penalized FITNESS delta vs the parent (score - "
             "merge_penalty*false, the gate's actual decision variable — NOT Edge "
             "Accuracy):"]
    for a in attempts:
        lines.append(
            f"  - gen{a['gen']}: {a['summary']} -> held-out fitness "
            f"{a['heldout']:+g} vs parent ({'kept' if a['accepted'] else 'rejected'})"
        )
    return "\n".join(lines) + "\n"


def _grounded_vs_exploratory(gen_gap: list[dict]) -> str:
    """Compare the generalization gap of GROUNDED vs EXPLORATORY accepted gens.

    Layer 2 of the priors overhaul: each accepted gen carries ``grounded`` (did the
    reviser cite a discovery finding?). If we have >=1 of EACH kind, report the mean
    train→held-out gap for each group so the reviser learns EMPIRICALLY whether
    grounding its changes in the validated priors is paying off on held-out — the
    feedback loop that makes the priors a measured bias, not a blind mandate. Returns
    "" when we lack both groups (nothing to compare) or ``grounded`` is unknown.
    """
    g = [r for r in gen_gap if r.get("grounded") is True]
    e = [r for r in gen_gap if r.get("grounded") is False]
    if not g or not e:
        return ""
    def _gap(rows):
        return sum(r["train"] - r["heldout"] for r in rows) / len(rows)
    gg, eg = _gap(g), _gap(e)
    verdict = (
        "GROUNDED changes are generalizing BETTER (smaller train→held-out gap) — keep "
        "citing validated findings" if gg < eg else
        "grounded and exploratory generalize about the same" if abs(gg - eg) < 1e-9 else
        "EXPLORATORY changes generalized better here — the inlined priors may not fit "
        "this brain; weigh them, don't follow blindly")
    return (
        f"\nGROUNDING PAYOFF (accepted gens, by whether they cited a discovery "
        f"finding): grounded (n={len(g)}) mean train→held-out gap = {gg:+.1f}; "
        f"exploratory (n={len(e)}) gap = {eg:+.1f}. {verdict}.\n"
    )


def _format_gen_gap(gen_gap: list[dict], window: int = 5) -> str:
    """Render the train→held-out generalization-gap meta-signal for the prompt.

    ``gen_gap`` holds one ``{"gen", "train", "heldout", "grounded"}`` per ACCEPTED
    generation — the candidate's TRAIN and HELD-OUT split-repair scores (and whether
    it cited a prior). We show only the AGGREGATE over the last ``window`` accepts
    (mean train score, mean held-out score, and the gap), so the reviser can SEE
    whether its accepted changes generalize — without ever revealing which neurons
    are held-out. A train score persistently far above held-out = overfitting the
    train split; the reviser should then favor changes grounded in generalizable
    geometry over ones that only chase train-specific sites. Empty until there are
    >=2 accepted generations (a gap needs history). A grounded-vs-exploratory
    breakdown is appended once both kinds have been accepted.
    """
    if len(gen_gap) < 2:
        return ""
    recent = gen_gap[-window:]
    mt = sum(r["train"] for r in recent) / len(recent)
    mh = sum(r["heldout"] for r in recent) / len(recent)
    gap = mt - mh
    trend = ("widening (train pulling ahead of held-out)" if gap > 0
             else "held-out keeping pace" if abs(gap) < 1e-9
             else "held-out ahead (unusual; not overfitting)")
    return (
        f"\n\nGENERALIZATION CHECK (aggregate over the last {len(recent)} ACCEPTED "
        f"generations; no neuron identities, just scores): mean TRAIN split-repair "
        f"score = {mt:.1f}, mean HELD-OUT split-repair score = {mh:.1f}, "
        f"gap (train - held-out) = {gap:+.1f} — {trend}. The gate scores you on "
        f"HELD-OUT, so a large positive gap means your recent accepted changes "
        f"repair train splits that DON'T carry over. If the gap is widening, prefer "
        f"improvements grounded in GENERALIZABLE geometry/topology (tangent "
        f"continuity, caliber match, endpoint degree) over ones that chase "
        f"train-specific recall.\n"
        + _grounded_vs_exploratory(gen_gap)
    )


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
        f"\n\nThe failure report now INLINES (near its top, 'Grounding priors' "
        f"section) the FEW validated findings matched to THIS generation's dominant "
        f"failure mode — start there. They come from a validated, cross-run knowledge "
        f"base of U-Net error regularities at {priors_path} (AutoDiscovery workflow: "
        f"each finding carries Reproduction / Generalization / Verdict / "
        f"Post-correction tokens). Use them as a PRIOR to ground your improvement in "
        f"already-verified geometry/topology rather than re-deriving from scratch. "
        f"DISCIPLINE on which findings to trust:\n"
        f"  • The inlined findings are ALREADY pre-filtered to those whose "
        f"Generalization is GENERALIZES and whose post-correction Verdict is UPHELD/OK "
        f"(robust across every brain AND after cluster-robust correction). If none "
        f"fits this failure, open {priors_path} and pick another QUALIFYING finding "
        f"by the same rule — do not privilege any particular one.\n"
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
        f"Cite the finding number(s) you relied on in your rules.md change log — "
        f"e.g. 'Finding #8'. This is MEASURED: generations that cite a finding are "
        f"tracked as 'grounded' and their held-out generalization is compared to "
        f"un-cited ('exploratory') ones (see the GROUNDING PAYOFF line), so an honest "
        f"citation feeds the loop that decides whether grounding is helping."
    )


async def ask_reviser(
    client: ClaudeSDKClient, report_path: str,
    heuristics_path: str, rules_path: str, verbose: bool,
    attempts: list[dict] | None = None,
    priors_path: str | None = None,
    splits_only: bool = False,
    gen_gap: list[dict] | None = None,
    transcript_path: str | None = None,
    structural_lessons: list[dict] | None = None,
):
    """Run the proofreader-reviser subagent on the failure report. Returns
    (text, input_tokens, output_tokens, cost_usd, read_priors). ``read_priors`` is
    True/False when a priors file was configured (did the reviser actually Read it
    this generation?), or None when no priors were configured.

    The agent is told to edit THIS RUN's working copies (under runs/<id>/artifacts),
    not the pristine originals under proofreader_evolve/artifacts. ``attempts`` is
    the memory of revisions already tried against the current parent (B), woven
    into the prompt so the agent proposes something NEW. ``priors_path``, when
    given, points the agent at the validated discovery knowledge base (see
    ``_format_priors`` for the trust discipline applied to it).

    ``structural_lessons`` (optional): the cross-parent 'over-merge taboo' list —
    revisions that created a false merge on held-out. Unlike ``attempts`` it is NOT
    cleared when the parent advances (a false merge is a geometric fact, not a
    parent-relative miss), so the reviser stops re-deriving the same over-merge
    direction against every new parent. See ``_format_structural_lessons``.

    ``transcript_path`` (optional): when given, the FULL per-generation reviser
    record — the prompt, every assistant THINKING block (the chain-of-thought, which
    the ledger's truncated ``diagnosis`` field drops), every TextBlock, and every
    tool call + result, in stream order — is written to that markdown file. This is
    the durable, untruncated audit trail of WHY the reviser made each edit; nothing
    else in the run persists the reasoning. Subagent vs orchestrator messages are
    tagged so the subagent's reasoning is distinguishable.
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
        + ("\n\nIMPORTANT — THIS RUN: merge-error repair is DISABLED. The candidate "
           "stream contains SplitSites only (no MergeSite is enumerated), and any "
           "`split_label` edit you emit is dropped before scoring. Do NOT write or "
           "tune `split_label` logic; focus entirely on the `merge_labels` "
           "(split-error) policy. The failure report's merge sections are omitted "
           "accordingly."
           if splits_only else "")
        + (
           "\n\nIMAGE CURRICULUM — couple reading image to PASSING THE GATE in ONE "
           "revision. The gate keeps a candidate only if it raises the penalized "
           "FITNESS (net correct `merge_labels` repairs minus a heavy per-false-merge "
           "penalty) above the parent — so a false merge is very costly but not an "
           "automatic reject. A revision that merely STARTS calling "
           "`gap_bridge_evidence` without changing which labels you merge ties the "
           "parent's fitness and is REVERTED — "
           "so its reads (and any image signal) are thrown away and never reach a "
           "future report. Therefore, if you decide image is worth using, you MUST "
           "spend it to RAISE RECALL in the SAME revision: take SplitSites in the "
           "report's 'MISSED real splits (rejected REAL)' bucket — real splits your "
           "current geometry is too conservative to accept — and ACCEPT the ones a "
           "high `gap_bridge_evidence.bridge_ratio` confirms are a continuous bright "
           "bridge across the gap. Gate the (cloud) read behind your cheap geometric "
           "filters so you only read the handful of ambiguous candidates. Consult the "
           "report's 'Image warm-start probe' section FIRST: it already measured "
           "whether `bridge_ratio` separates REAL from FALSE on this brain (an AUC), "
           "so you know BEFORE writing the rule whether image is worth gating on and "
           "roughly where the threshold sits. If the AUC is weak, do not over-invest "
           "in image — improve the geometric `merge_labels` policy instead.")
        + _format_priors(priors_path)
        + _format_gen_gap(gen_gap or [])
        + _format_structural_lessons(structural_lessons or [])
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
    # THINKING (chain-of-thought) collected separately, subagent vs orchestrator.
    sub_think, orch_think = [], []
    # Full stream-ordered transcript events for transcript_path (thinking, text,
    # tool calls + results), so the saved record reads in the order it happened.
    transcript_events: list[str] = []
    in_tok = out_tok = 0
    cost = 0.0
    # Did the reviser actually READ the discovery priors this generation? We can
    # grant the tool + allow the path, but only the tool-call stream tells us it was
    # used — so verify, don't assume. None when no priors were configured.
    priors_name = os.path.basename(priors_path) if priors_path else None
    read_priors = False if priors_name else None

    def _ev(tag: str, body: str) -> None:
        transcript_events.append(f"### {tag}\n\n{body.rstrip()}\n")

    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            is_sub = getattr(message, "parent_tool_use_id", None) is not None
            who = "subagent" if is_sub else "orchestrator"
            for block in message.content:
                # THINKING first: the chain-of-thought the ledger drops. The SDK
                # exposes it as a ThinkingBlock with a ``.thinking`` str (older SDKs
                # may use ``.text``); guard both.
                if ThinkingBlock and isinstance(block, ThinkingBlock):
                    tb = getattr(block, "thinking", None) or getattr(block, "text", "") or ""
                    (sub_think if is_sub else orch_think).append(tb)
                    _ev(f"💭 thinking [{who}]", tb)
                    if verbose:
                        print(tb, end="", flush=True)
                elif isinstance(block, TextBlock):
                    (sub_chunks if is_sub else orch_chunks).append(block.text)
                    _ev(f"📝 text [{who}]", block.text)
                    if verbose:
                        print(block.text, end="", flush=True)
                elif ToolUseBlock and isinstance(block, ToolUseBlock):
                    tname = getattr(block, "name", "tool")
                    log(f"    → {tname}{' [subagent]' if is_sub else ''}")
                    ti = getattr(block, "input", {}) or {}
                    _ev(f"🔧 tool call [{who}]: {tname}",
                        "```json\n" + json.dumps(ti, indent=2, default=str)[:4000] + "\n```")
                    # Flag a Read whose target is the priors file (any field that
                    # carries a path), so we can confirm the prior was consulted.
                    if priors_name and tname == "Read":
                        if any(priors_name in str(v) for v in ti.values()):
                            read_priors = True
                elif ToolResultBlock and isinstance(block, ToolResultBlock):
                    _content = getattr(block, "content", "")
                    _ev(f"📤 tool result [{who}]", str(_content)[:4000])
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
    # surfaced none (older SDKs / different routing). Same precedence for thinking.
    text = "".join(sub_chunks) or "".join(orch_chunks)
    thinking = "".join(sub_think) or "".join(orch_think)

    # Persist the FULL, untruncated reviser record (prompt + thinking + text + tool
    # I/O) so the reasoning behind each edit survives the run — the ledger only keeps
    # a 2000-char slice of the final text and no thinking at all.
    if transcript_path:
        try:
            header = [
                "# Reviser transcript\n",
                f"- thinking captured: {'yes' if thinking else 'no'} "
                f"({len(thinking)} chars)",
                f"- final text: {len(text)} chars",
                f"- tokens: in={in_tok} out={out_tok}; cost_usd={cost}",
                f"- priors read: {read_priors}\n",
                "## Prompt sent to the reviser\n",
                "```\n" + instruction + "\n```\n",
                "## Stream (thinking / text / tool calls, in order)\n",
            ]
            os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
            with open(transcript_path, "w") as f:
                f.write("\n".join(header) + "\n" + "\n".join(transcript_events))
        except Exception as _e:
            log(f"   [WARN] could not write reviser transcript "
                f"({transcript_path}): {_e}")

    return text, in_tok, out_tok, cost, read_priors


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


_FOLD_METRIC_COLS = ("Edge Accuracy", "% Merged Edges", "# Merges", "% Split Edges")


def fold_metrics(per_swc, fold_names: list[str]) -> dict:
    """Run-length-weighted metric vector over just one fold's held-out skeletons.

    ``per_swc`` is a ScoreResult.per_swc frame (one row per GT skeleton, scored on
    the FULL held-out set in a single pass); ``fold_names`` selects this fold's rows.
    DIAGNOSTIC ONLY: the per-fold Edge-Accuracy vectors this produces are logged and
    written to split.json so a generation's stability across folds is visible; they
    do NOT feed the accept/reject decision (that is the pooled split-repair score —
    see the gate at step 7).
    """
    rows = per_swc.loc[per_swc.index.isin(fold_names)]
    return {m: scoring._weighted_avg(rows, m) for m in _FOLD_METRIC_COLS}


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
    role: str = "split"                    # "split" (within-brain), "train" (all train),
                                           # or "heldout" (all held-out) — see _setup_brain
    base_train: object = None              # per_swc baseline rows for train
    base_heldout: object = None            # per_swc baseline rows for held-out
    label_gt_map: dict = None              # TRAIN-only {label: {gt_neuron: count}}
                                           # for the leak-free SplitSite audit
    heldout_label_gt_map: dict = None      # HELD-OUT {label: {gt_neuron: count}} —
                                           # GATE-ONLY split-repair scoring. LEAK
                                           # BOUNDARY: never goes into the failure
                                           # report / reviser; only the gate reads it.
    image_probe_section: list = None       # pre-rendered image warm-start probe
                                           # markdown (run-cached, train-only, leak-
                                           # free); None if no image reader. Appended
                                           # to every generation's failure report.
    confidence_model: object = None        # CALIBRATED multi-feature image confidence
                                           # model (image_confidence.ConfidenceModel),
                                           # fit ONCE on this brain's train warm-start
                                           # REAL/FALSE examples. GATE-side per-edit
                                           # confidence input; None -> fallback used.


def _setup_brain(brain: str, run_dir: Path, heldout_fraction: float,
                 split_seed: int, verbose: bool, mcl: int = 100,
                 role: str = "split") -> BrainContext:
    """Load + prepare ONE brain and compute its train/held-out split + baseline.

    Mirrors the original single-brain setup, factored out so a run can hold several
    brains at once. The prepared-brain pickle (expensive) is reused from any prior
    run if present, exactly as before. ``mcl`` selects which fragment cache to load
    (``dataset_cache_<brain>_mcl<mcl>.pkl``); default 100.

    ``role`` controls how this brain's neurons are assigned:
      * ``"split"`` (default): the legacy per-brain split — this brain's own
        skeletons are partitioned ``train_heldout_split`` (train feeds the failure
        report, held-out feeds the gate), so a single brain serves BOTH roles.
      * ``"train"``: ALL of this brain's neurons go to TRAIN (feedback); none are
        held out. Used in CROSS-BRAIN mode where a different brain is the gate.
      * ``"heldout"``: ALL of this brain's neurons go to the HELD-OUT gate; none
        are train. The gate then scores on a WHOLE brain the reviser never saw —
        a strictly cleaner generalization signal than a within-brain split (no
        fragment of a test neuron ever appears in the train feedback).
    """
    paths = scoring.BrainPaths(brain)
    cache_path = ds.default_cache_path(brain, min_cable_length=mcl)
    if not os.path.exists(cache_path):
        raise SystemExit(
            f"[{brain}] fragment cache not found: {cache_path}\n"
            f"  (check --brain / --mcl; build it via notebooks/load_skeletons.ipynb "
            f"with min_cable_length={mcl} if missing)")
    log(f"[{brain}] Loading cached fragment graph (mcl={mcl}): {cache_path}")
    # Validate the cache CONTENTS against the requested brain + mcl, not just the
    # filename convention — a swapped/stale pickle would otherwise train/gate on the
    # wrong data silently. Fails fast on mismatch (see load_cached_graphs).
    fragments_graph, _gt_graph, _ = ds.load_cached_graphs(
        cache_path, expect_brain=brain, expect_mcl=mcl)

    # The policy always gets a LAZY raw-image patch reader in ctx (each read is a
    # cloud fetch, so the policy gates reads behind cheap filters). Lazy: no cloud
    # access happens until a read is actually requested.
    from proofreader_evolve.harness.image_features import LazyImagePatchReader
    import sys as _sys
    _scripts = str(PROJECT_ROOT / "scripts")
    _sys.path.insert(0, _scripts)
    from dataset_config import get_img_path  # noqa: E402
    _prefixes = str(PROJECT_ROOT / "configs" / "exaspim_image_prefixes.json")
    img_path = get_img_path(brain, prefixes_path=_prefixes)
    image_reader = LazyImagePatchReader(img_path, fragments_graph)
    log(f"[{brain}] Image patch reader ENABLED (lazy): {img_path}")

    # RUN ISOLATION for EXPERIMENT state: every run's split/gate/feedback artifacts
    # stay in THIS run's own directory — a run never reads a sibling run's results,
    # so no experiment state leaks across runs. The prepared-brain pickle is the ONE
    # exception, and safely so: it is a pure function of the (immutable) dataset and
    # holds NO run-specific state (the train/held-out split, seed, gt maps, and edits
    # are all computed AFTER loading, below). So instead of rebuilding it (~30 min)
    # every run, get_or_build keeps a SINGLE persisted copy in a SHARED, validated
    # cross-run cache under proofreader_evolve/prepared_cache/ — schema-version +
    # brain-id + content-checksum are verified on load, and a stale/corrupt/mismatched
    # artifact is rebuilt rather than trusted. NO per-run duplicate is written: a
    # resume just re-reads the shared copy (seconds), and any leftover per-run pickle
    # from older runs is promoted into the shared cache and then deleted. The
    # ``cache_path`` below is only used as a fallback if the shared write fails.
    # NOTE the prepared state is built from the RAW (unfiltered) GCS graphs, so its
    # content does NOT depend on mcl — the filename intentionally omits mcl.
    prepared_cache = str(run_dir / f"prepared_{brain}.pkl")
    log(f"[{brain}] Preparing brain for incremental scoring (per-run cache: "
        f"{prepared_cache}; shared cache: {inc.shared_prepared_cache_path(brain)})")
    with Heartbeat(f"[{brain}] preparing brain (load/build — can be ~30 min cold)"):
        prepared = inc.get_or_build(paths, prepared_cache, verbose=verbose)

    with Heartbeat(f"[{brain}] scoring baseline"):
        baseline_full = inc.score_incremental(prepared, label_pairs=None, verbose=verbose)
    all_gt_names = list(baseline_full.per_swc.index)
    # Neuron assignment depends on this brain's ROLE:
    #   * "split"   -> per-brain 70/30 (or as configured), seeded + reproducible. Each
    #                  brain splits its OWN skeletons, so no brain lands wholly on a side.
    #   * "train"   -> ALL neurons are TRAIN (cross-brain mode; a different brain gates).
    #   * "heldout" -> ALL neurons are HELD-OUT (cross-brain mode; the gate scores a
    #                  whole brain the reviser's feedback never touched).
    if role == "train":
        train_names, heldout_names = list(all_gt_names), []
    elif role == "heldout":
        train_names, heldout_names = [], list(all_gt_names)
    else:
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

    # Image WARM-START probe (one-time, run-cached): break the image-signal cold
    # start by having the HARNESS read gap_bridge_evidence on a small, balanced,
    # leak-free (train-only) sample of REAL vs FALSE SplitSites in the ambiguous
    # gap-overlap band, and report whether bridge_ratio separates them (AUC). It is a
    # MEASUREMENT, never a policy/gate input. Only runs when an image reader exists;
    # the cloud reads are paid ONCE per brain here and reused in every generation's
    # report. See candidate.image_warmstart_probe.
    image_probe_section = None
    confidence_model = None
    if image_reader is not None:
        with Heartbeat(f"[{brain}] image warm-start probe (one-time)"):
            probe_splits = ds.candidate_split_sites(fragments_graph)
            probe = cand.image_warmstart_probe(
                probe_splits, label_gt_map, image_reader, train_names)
        if probe is not None:
            image_probe_section = probe["section"]
            # Calibrated multi-feature confidence model, fit on THIS brain's train
            # warm-start REAL/FALSE reads (leak-free). Threaded to the gate for
            # per-edit confidence on blind-spot merges (option #1).
            confidence_model = probe.get("model")
            log(f"[{brain}] image warm-start probe: bridge_ratio AUC="
                f"{probe['auc'] if probe['auc'] is not None else float('nan'):.2f} "
                f"(REAL {probe['n_real']} / FALSE {probe['n_false']} probed); "
                f"confidence model = "
                f"{confidence_model.kind if confidence_model else 'none'}")
        else:
            log(f"[{brain}] image warm-start probe: nothing to probe "
                f"(no train-classifiable SplitSite pair)")

    return BrainContext(
        brain=brain, fragments_graph=fragments_graph, image_reader=image_reader,
        prepared=prepared, baseline_full=baseline_full,
        train_names=train_names, heldout_names=heldout_names, role=role,
        merge_labels=merge_labels, base_train=base_train, base_heldout=base_heldout,
        label_gt_map=label_gt_map, heldout_label_gt_map=heldout_label_gt_map,
        image_probe_section=image_probe_section,
        confidence_model=confidence_model,
    )


def _score_pooled(brains: list, names_attr: str, work_heuristics: str,
                  split_name: str, max_class_size, verbose: bool,
                  splits_only: bool = False):
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
            splits_only=splits_only,
        )
        per_brain_runs.append(run)
        frames.append(run.score.per_swc)
    # verify_integrity: pooling weights metrics by name, so a cross-brain name
    # collision would double-count silently — fail fast at the concat instead.
    pooled = pd.concat(frames, verify_integrity=True) if frames else None
    return pooled, per_brain_runs


def _pooled_split_repair(brains: list, per_brain_runs: list,
                         map_attr: str = "heldout_label_gt_map") -> dict:
    """Pool the dense split-repair signal across brains.

    For each brain, classify that brain's edits against ITS OWN label→neuron map
    (raw labels are not comparable across brains, so each brain is classified with
    its own map and only the COUNTS are pooled). Returns
    ``{"correct": int, "false": int, "unscored": int, "score": int}`` where
    ``score = correct - false`` is the primary fitness (dense: every repaired split
    counts, unlike Edge Accuracy which only moves on a bridged split edge).

    ``map_attr`` selects which per-brain map to classify against:
      * ``"heldout_label_gt_map"`` (default) — the GATE signal; GATE-ONLY, never
        exposed to the reviser.
      * ``"label_gt_map"`` — the TRAIN signal; used only to compute the AGGREGATE
        train→held-out generalization gap (a scalar trend), which IS shown to the
        reviser. Pass the TRAIN per-brain runs with this so train edits are scored
        on the train map.
    """
    tot = {"correct": 0, "false": 0, "unscored": 0}
    # Per-brain blind-spot detail, kept ALONGSIDE the pooled counts so a caller can
    # corroborate the unscored edits with a GT-independent signal (see
    # ``_image_confidence``). Each entry is (BrainContext, CandidateRun, unscored
    # label-pairs) — the pair→node lookup needs that brain's own run + graph.
    tot["_unscored_by_brain"] = []
    # (BrainContext, CandidateRun) for EVERY brain — lets a caller re-classify each
    # brain's edits (correct/false/unscored) with brain-local labels for a per-EDIT
    # confidence table (see ``_per_edit_confidence``). Also carries which map was
    # used, so the per-edit re-classification matches this pooled pass.
    tot["_by_brain"] = list(zip(brains, per_brain_runs))
    tot["_map_attr"] = map_attr
    for bc, run in zip(brains, per_brain_runs):
        c = inc.classify_merge_edits(run.edits, getattr(bc, map_attr))
        tot["correct"] += c["correct"]
        tot["false"] += c["false"]
        tot["unscored"] += c["unscored"]
        if c.get("unscored_pairs"):
            tot["_unscored_by_brain"].append((bc, run, c["unscored_pairs"]))
    tot["score"] = tot["correct"] - tot["false"]
    return tot


def _fitness(repair: dict, merge_penalty: float) -> float:
    """The gate's decision variable: split-repair score MINUS a heavy per-false-merge
    penalty. SMOOTH replacement for the old hard 'false == 0' gate.

    ``fitness = repair["score"] - merge_penalty * repair["false"]``
             ``= (correct - false) - merge_penalty * false``.

    A false merge (fusing two DIFFERENT neurons) is still strongly discouraged — at
    the default ``merge_penalty=100`` a single one costs ~100 correct repairs to
    offset — but it no longer AUTOMATICALLY rejects the candidate. A change that
    creates one false merge while repairing 120 real splits can now be kept, whereas
    the old gate reverted it outright. This trades a small, controlled amount of
    merge error for the split-recall it unlocks, and keeps the fitness landscape
    continuous (a near-miss is scored just below a clean win, not slammed to reject),
    which gives the search a usable gradient around the precision boundary.
    """
    return repair["score"] - merge_penalty * repair["false"]


# --- Confidence layer (option ①): a GT-INDEPENDENT soft verdict on the blind spot -
# The hard fitness above rests only on GT-covered edges; in a sparsely-traced brain
# most edits are ``unscored`` (no GT at either endpoint) and carry NO verdict. We do
# NOT fold them into the score (penalizing "no GT" would just teach the policy to
# avoid untraced neurons — exactly the overfitting we want to avoid). Instead we
# attach an independent, image-derived SOFT verdict and turn it into a CONFIDENCE
# LEVEL for the accept/reject decision, so the reviewer can see how much of a
# generation's fitness is GT-backed vs. resting on an unverified blind spot.
#
# Independence is the whole point: the image features are raw fluorescence along the
# gap, NOT the geometry the policy decided with, so it is a genuine second opinion
# (not circular). Leak-safe: the features are not GT, and the model is fit only on
# TRAIN warm-start labels; the soft verdict stays in the gate/ledger, never shown to
# the reviser (only the aggregate per-feature separability is).
#
# MULTI-FEATURE + CALIBRATED: rather than thresholding a single bridge_ratio scalar,
# we run the FULL feature vector the reader already returns (bridge_ratio,
# bridge_mean_ratio, valley_frac, profile_cv, bridge_pos) through a model calibrated
# on the warm-start REAL/FALSE examples (image_confidence.ConfidenceModel). See #1/#2.
#
# ZERO extra cloud reads: we REPLAY the ``gap_bridge_evidence`` reads the policy
# already made this generation (recorded in ``run.image_reads``). An unscored edit
# whose SplitSite the policy imaged gets a soft verdict; one it never imaged stays
# ``unknown`` (counted, never guessed).

# Probability bands for the calibrated model's P(REAL) -> verdict label. The model
# (image_confidence.ConfidenceModel) outputs a probability; >= _P_LIKELY_CORRECT is a
# confident "likely correct" merge, <= _P_LIKELY_FALSE a confident "likely false"
# fusion, the middle band "ambiguous". These are PROBABILITY thresholds (post-
# calibration), not the raw-bridge_ratio band the fallback model uses internally.
_P_LIKELY_CORRECT = 0.65
_P_LIKELY_FALSE = 0.35


def _select_confidence_model(repair: dict):
    """Pick the confidence model to judge this pooled pass's blind-spot merges.

    In cross-brain mode the HELD-OUT brain has no train GT, so its own warm-start
    probe can't calibrate — it carries only a fallback. Prefer a genuinely CALIBRATED
    (``logistic``) model fit on ANY brain's train examples: image features are
    normalized/brain-agnostic, so a train-fit model transfers to score the held-out
    brain's edits (and keeps the confidence signal independent of the held-out GT the
    gate uses). Falls back to a bridge_ratio sigmoid if no brain calibrated one.
    """
    from proofreader_evolve.harness import image_confidence as _imgconf
    best = None
    for bc, _run in repair.get("_by_brain", []):
        m = getattr(bc, "confidence_model", None)
        if m is None:
            continue
        if getattr(m, "kind", None) == "logistic":
            return m  # a real calibrated model — use it
        best = best or m
    return best or _imgconf.ConfidenceModel.fallback()


def _imaged_pair_proba(run, want: set, model) -> dict:
    """Map each wanted label-pair the policy IMAGED this run -> its P(REAL).

    Replays ``gap_bridge_evidence`` reads the policy already made (zero extra cloud
    reads), runs the FULL feature vector through the calibrated ``model``, and keeps
    the MOST favorable (max P) read when a pair was imaged more than once. Pairs the
    policy never imaged are simply absent from the returned dict.
    """
    node_to_pair = {}
    for s in (getattr(run, "split_sites", None) or []):
        pr = tuple(sorted((str(getattr(s, "label_a", "")),
                           str(getattr(s, "label_b", "")))))
        for nd in (getattr(s, "node_a", None), getattr(s, "node_b", None)):
            if nd is not None:
                node_to_pair[int(nd)] = pr
    pair_p: dict = {}
    for r in (getattr(run, "image_reads", None) or []):
        if r.get("method") != "gap_bridge_evidence":
            continue
        pr = node_to_pair.get(r.get("node_a")) or node_to_pair.get(r.get("node_b"))
        if pr is None or pr not in want:
            continue
        p = model.predict_proba(r.get("result") or {})
        if p != p:  # NaN (unreadable features)
            continue
        if pr not in pair_p or p > pair_p[pr]:
            pair_p[pr] = p
    return pair_p


def _image_confidence(repair: dict) -> dict:
    """Soft, GT-independent verdict on the unscored (blind-spot) merges.

    Runs each imaged unscored merge's FULL image feature vector through the CALIBRATED
    multi-feature model (image_confidence.ConfidenceModel; fit on the warm-start
    REAL/FALSE examples) to get P(REAL), then buckets:
      * P >= ``_P_LIKELY_CORRECT`` -> ``soft_correct`` (likely one neuron)
      * P <= ``_P_LIKELY_FALSE``   -> ``soft_false``   (likely two neurons)
      * in between                 -> ``soft_ambiguous``
      * SplitSite never imaged     -> ``unknown``

    Returns those counts plus:
      * ``covered``   = unscored edits with ANY image verdict (soft_* ),
      * ``coverage``  = covered / unscored (how much of the blind spot we can see),
      * ``gt_fraction`` = GT-verified edits / total classified edits (correct+false),
      * ``model_kind`` = which model judged them ("logistic" | "fallback").
    No cloud reads; NaN-safe. ``soft_*`` counts are advisory — they never enter the
    hard fitness; only ``--confidence-veto`` may act on ``soft_false`` (opt-in).
    """
    out = {"soft_correct": 0, "soft_false": 0, "soft_ambiguous": 0, "unknown": 0,
           "covered": 0, "coverage": float("nan"), "gt_fraction": float("nan"),
           "model_kind": "none"}
    model = _select_confidence_model(repair)
    out["model_kind"] = getattr(model, "kind", "none")
    total_unscored = 0
    for bc, run, pairs in repair.get("_unscored_by_brain", []):
        want = {tuple(sorted(p)) for p in pairs}
        total_unscored += len(pairs)
        pair_p = _imaged_pair_proba(run, want, model)
        for pr in want:
            p = pair_p.get(pr)
            if p is None:
                out["unknown"] += 1
            elif p >= _P_LIKELY_CORRECT:
                out["soft_correct"] += 1
            elif p <= _P_LIKELY_FALSE:
                out["soft_false"] += 1
            else:
                out["soft_ambiguous"] += 1
    out["covered"] = out["soft_correct"] + out["soft_false"] + out["soft_ambiguous"]
    if total_unscored:
        out["coverage"] = out["covered"] / total_unscored
    gt_verified = repair.get("correct", 0) + repair.get("false", 0)
    denom = gt_verified + total_unscored
    if denom:
        out["gt_fraction"] = gt_verified / denom
    return out


def _per_edit_confidence(repair: dict) -> list:
    """A CONFIDENCE row for EVERY held-out merge edit this generation.

    Combines the two evidence sources into one per-edit verdict so a human reviewer
    (or downstream tooling) can rank/triage individual merges, not just the whole
    generation. Each row is a dict:

        {brain, label_a, label_b, verdict, source, confidence, confidence_score}

    where the verdict comes from the STRONGEST evidence available for that edit:
      * GT-verified (``correct``/``false``) -> source="gt", confidence="high",
        verdict in {"correct","false"}. GT is authoritative, so it wins over image.
      * unscored but IMAGED -> source="image", confidence in
        {"likely_correct","likely_false","ambiguous"} from the CALIBRATED multi-
        feature model's P(REAL) (an independent, GT-free signal); ``confidence_score``
        is that probability.
      * unscored and NOT imaged -> source="none", confidence="unknown".

    LEAK-SAFE: a GATE/ledger-side artifact (uses the held-out GT map + raw image
    features); NEVER fed to the reviser. Zero extra cloud reads — image reads are
    replayed from what the policy already fetched (same model as ``_image_confidence``).
    """
    rows = []
    map_attr = repair.get("_map_attr", "heldout_label_gt_map")
    model = _select_confidence_model(repair)
    # Classify per brain so EVERY edit (correct / false / unscored) gets a row with
    # its brain id. Cheap: reuses the same maps + edits already in memory, and the
    # SAME map this pooled pass used (so the per-edit verdicts match the counts).
    for bc, run in repair.get("_by_brain", []):
        c = inc.classify_merge_edits(run.edits, getattr(bc, map_attr))
        brain = getattr(bc, "brain", "?")
        for a, b, da, db in c.get("correct_pairs", []):
            rows.append({"brain": brain, "label_a": a, "label_b": b,
                         "verdict": "correct", "source": "gt", "confidence": "high",
                         "confidence_score": 1.0})
        for a, b, da, db in c.get("false_pairs", []):
            rows.append({"brain": brain, "label_a": a, "label_b": b,
                         "verdict": "false", "source": "gt", "confidence": "high",
                         "confidence_score": 0.0})
        # unscored edits for THIS brain, corroborated by replayed image evidence run
        # through the calibrated multi-feature model.
        unscored = {tuple(sorted(p)) for p in c.get("unscored_pairs", [])}
        if not unscored:
            continue
        pair_p = _imaged_pair_proba(run, unscored, model)
        for pr in sorted(unscored):
            p = pair_p.get(pr)
            label = model.label(p if p is not None else float("nan"),
                                lo=_P_LIKELY_FALSE, hi=_P_LIKELY_CORRECT)
            rows.append({"brain": brain, "label_a": pr[0], "label_b": pr[1],
                         "verdict": "unscored",
                         "source": "image" if label != "unknown" else "none",
                         "confidence": label,
                         "confidence_score": p if p is not None else float("nan")})
    return rows


def _confidence_level(repair: dict, conf: dict) -> str:
    """A coarse CONFIDENCE LEVEL for this generation's accept/reject decision.

    Combines how much of the decision is GT-backed with what the independent image
    signal says about the blind spot:
      * ``high``   — most edits GT-verified (``gt_fraction >= 0.5``): the hard
        fitness rests on solid ground regardless of the blind spot.
      * ``low``    — decision rests on a large blind spot AND the image signal flags
        likely-false merges in it (``soft_false`` present with little GT backing).
      * ``medium`` — everything else (blind spot large but image evidence is clean
        or unknown; no positive reason to distrust the decision).
    Purely a REPORTED label unless ``--confidence-veto`` is set (then a ``low`` with
    image-flagged false merges can block an otherwise-accepted generation).
    """
    gtf = conf.get("gt_fraction")
    if gtf == gtf and gtf is not None and gtf >= 0.5:
        return "high"
    if conf.get("soft_false", 0) > 0 and (gtf != gtf or gtf is None or gtf < 0.25):
        return "low"
    return "medium"


def _dominant_failure_mode(train_repair: dict) -> str:
    """Coarse label for THIS generation's dominant failure mode, from the TRAIN
    split-repair counts, used to MATCH grounding priors to the report (Layer 1).

    Leak-free: uses only the train-classified correct/false/unscored counts already
    computed for the generalization-gap signal — no held-out, no neuron identity.
      * no correct AND no false yet          -> "cold_start" (policy not merging)
      * any false merges present             -> "precision" (over-merging dominates)
      * correct merges, zero false           -> "recall"    (safe so far; push recall)
      * otherwise                            -> "both"
    Priors only BIAS the proposal, so a coarse mode is enough; the held-out gate,
    not this label, still decides accept/reject.
    """
    correct = train_repair.get("correct", 0)
    false = train_repair.get("false", 0)
    if correct == 0 and false == 0:
        return "cold_start"
    if false > 0:
        return "precision"
    if correct > 0:
        return "recall"
    return "both"


def _failure_signature(train_repair: dict, train_runs: list) -> str:
    """A free-text description of THIS generation's failure, for CONTENT-matching the
    grounding priors (priors.relevance) to the actual situation — not just a coarse
    mode label. Leak-free: built only from TRAIN split-repair counts + the geometry of
    the edits the policy MISSED / got WRONG (the same train-side signal already used
    for the failure mode). Names the geometric axes in play so a finding whose text
    talks about that geometry scores higher.
    """
    correct = train_repair.get("correct", 0)
    false = train_repair.get("false", 0)
    terms = []
    if false > 0:
        terms += ["over-merge", "false merge", "crossing", "angle", "collinear",
                  "radius", "caliber", "precision"]
    if correct == 0 and false == 0:
        terms += ["proximity", "gap", "threshold", "reconnect", "not merging"]
    # Recall pressure: many enumerated split sites but few edits emitted.
    n_sites = sum(getattr(r, "n_sites", 0) for r in (train_runs or []))
    n_edits = sum(getattr(r, "n_edits", 0) for r in (train_runs or []))
    if n_sites and n_edits < 0.5 * n_sites:
        terms += ["missed", "recall", "gap", "distance", "threshold", "reach", "branch"]
    return " ".join(terms)


async def run_evolution(
    brain: str, generations: int, heldout_fraction: float,
    human: bool, verbose: bool, model: str = DEFAULT_MODEL,
    score_margin: int = 1, max_class_size=None, seed_from: str | None = None,
    split_seed: int | None = None,
    k_folds: int = 1,
    brains: list | None = None, splits_only: bool = False,
    use_priors: bool = True, mcl: int = 100,
    converge_patience: int = 0, converge_eps: int = 1,
    train_brains: list | None = None, test_brains: list | None = None,
    merge_penalty: float = 100.0,
    confidence_veto: bool = False,
) -> None:
    # CROSS-BRAIN mode: --train-brains + --test-brains assign WHOLE brains to roles —
    # every train brain's neurons feed the failure report, every test brain's neurons
    # feed the gate, with NO within-brain split. This is the cleanest generalization
    # signal (the gate scores brains the reviser's feedback never touched) and is the
    # natural way to "focus the gate on held-out neurons' merge errors". When neither
    # is given, behavior is the legacy per-brain split over --brains / --brain.
    cross_brain = bool(train_brains) and bool(test_brains)
    if bool(train_brains) ^ bool(test_brains):
        raise SystemExit(
            "--train-brains and --test-brains must be given TOGETHER (cross-brain "
            "mode) — got only one. Pass both, or neither (to use the per-brain "
            "split over --brain/--brains).")
    if cross_brain:
        train_brains = [str(b) for b in train_brains]
        test_brains = [str(b) for b in test_brains]
        # (a) No brain may repeat WITHIN a list — a duplicate would load the same
        # brain twice, double-count it in pooling, and only surface later as a
        # cryptic skeleton-name collision at the verify_integrity concat. Catch it
        # here with a clear message.
        def _dups(seq):
            seen, dup = set(), []
            for b in seq:
                (dup.append(b) if b in seen else seen.add(b))
            return sorted(set(dup))
        train_dups, test_dups = _dups(train_brains), _dups(test_brains)
        if train_dups or test_dups:
            raise SystemExit(
                f"--train-brains / --test-brains contain duplicate brain(s) "
                f"(train: {train_dups or 'none'}; test: {test_dups or 'none'}); each "
                f"brain may appear at most once — a repeat double-counts that brain "
                f"and collides on skeleton names downstream.")
        # (b) No brain may be in BOTH sets — that would leak a brain's neurons across
        # the train/gate boundary (the whole point of cross-brain mode is disjointness).
        overlap = sorted(set(train_brains) & set(test_brains))
        if overlap:
            raise SystemExit(
                f"--train-brains and --test-brains overlap ({overlap}); a brain "
                f"cannot be both the train and the test set (that would leak its "
                f"neurons across the gate boundary).")
        # role per brain, in a stable train-then-test order; the run id flags it.
        brain_roles = ([(b, "train") for b in train_brains]
                       + [(b, "heldout") for b in test_brains])
        brain_list = [b for b, _ in brain_roles]
        primary_brain = train_brains[0]
        run_tag = f"{primary_brain}_train{len(train_brains)}_test{len(test_brains)}"
    else:
        # Brain set: --brains (list) takes precedence; else the single --brain.
        brain_list = [str(b) for b in (brains or [brain])]
        brain_roles = [(b, "split") for b in brain_list]
        primary_brain = brain_list[0]
        run_tag = primary_brain if len(brain_list) == 1 else f"{primary_brain}+{len(brain_list)-1}"
    run_id = f"{run_tag}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = HERE / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Tee ALL console output (this function's log()/prints, the reviser stream, any
    # traceback) to runs/<id>/run.log. Installed here — as soon as run_dir exists —
    # and torn down by main()'s finally, so even a crash is captured. Streams are
    # restored there too. (Setup log lines emitted BEFORE this point are the generic
    # split/brain-assignment messages, not run-specific.)
    run_log_capture(run_dir).__enter__()  # torn down in main()'s finally
    log(f"  full console log -> {run_dir / 'run.log'}")
    ledger = Ledger(str(run_dir / "ledger.jsonl"))

    # Work on TIMESTAMPED COPIES of the artifacts, never the originals. The agent
    # reads/revises only these; artifacts/ stays pristine. By default the start is
    # the pristine seed (from scratch); --seed-from continues a prior run's policy.
    work_heuristics, work_rules = make_working_artifacts(run_dir, seed_from=seed_from)
    log(f"Run {run_id}")
    log(f"  start policy: {'continuing run ' + seed_from if seed_from else 'from-scratch seed'}")
    log(f"  originals (untouched): {HEURISTICS}")
    log(f"  working copies (revised this run): {work_heuristics}")
    if splits_only:
        log("  SPLIT-ERROR-ONLY mode: split_label (merge-error) edits are dropped "
            "before scoring — faster, and the gate metric is unchanged (it scores "
            "only merge_labels). Merge-error repair is deferred.")

    # --- One-time setup: load+prepare EVERY brain (expensive, cached) -----------
    # Split seed: RANDOM by default; drawn once and shared across brains so every
    # brain's 70/30 split is reproducible from this one recorded seed.
    if split_seed is None:
        split_seed = int.from_bytes(os.urandom(4), "little")
    if cross_brain:
        log(f"CROSS-BRAIN mode: train={train_brains} -> feedback, "
            f"test={test_brains} -> gate (whole-brain held-out; no within-brain split)")
    else:
        log(f"Brains: {brain_list} (split_seed={split_seed}, "
            f"heldout_fraction={heldout_fraction})")
    brain_ctxs = [
        _setup_brain(b, run_dir, heldout_fraction, split_seed, verbose, mcl=mcl,
                     role=r)
        for b, r in brain_roles
    ]
    # Brains that carry TRAIN neurons (feedback) and those that carry HELD-OUT
    # neurons (gate). In the per-brain "split" mode every brain carries both, so both
    # lists equal brain_ctxs and behavior is unchanged. In cross-brain mode they are
    # disjoint, so the train report never sees a test brain and the gate never sees a
    # train brain.
    train_ctxs = [bc for bc in brain_ctxs if bc.train_names]
    heldout_ctxs = [bc for bc in brain_ctxs if bc.heldout_names]
    assert train_ctxs, "no brain contributes TRAIN neurons — nothing to learn from"
    assert heldout_ctxs, "no brain contributes HELD-OUT neurons — nothing to gate on"

    # Pooled held-out names across brains. Pooling indexes skeletons BY NAME (concat
    # / isin / loc / fold slicing), which silently corrupts — duplicate rows, double-
    # counted weights — if two brains share a skeleton name. The name is expected to
    # embed the brain id (e.g. N005-789202-SP), but the harness never enforces that,
    # and passing the same brain twice (--brains 789202,789202) would collide too. So
    # fail FAST here, at the pool boundary, turning a silent miscount into a clear
    # startup error before any scoring runs.
    import pandas as _pd
    pooled_heldout = [n for bc in heldout_ctxs for n in bc.heldout_names]
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
        [bc.base_heldout for bc in heldout_ctxs], verify_integrity=True)
    baseline_heldout = scoring._weighted_avg(pooled_base_heldout, "Edge Accuracy")
    log(f"Pooled held-out: {len(pooled_heldout)} neurons across {len(heldout_ctxs)} "
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
        # min_cable_length of the fragment cache this run enumerated candidates over.
        # Recorded so downstream analysis (compare_proofreader_policy.ipynb) reloads
        # the SAME cache — using mcl100 to replay an mcl10 run would enumerate a
        # different candidate stream and the numbers would not match this run.
        "mcl": mcl,
        # Cross-brain vs per-brain split, and each brain's role, so a run is fully
        # reproducible and downstream analysis can tell the two modes apart.
        "cross_brain": cross_brain,
        "train_brains": train_brains if cross_brain else None,
        "test_brains": test_brains if cross_brain else None,
        "per_brain": {bc.brain: {"role": bc.role,
                                 "train": bc.train_names,
                                 "heldout": bc.heldout_names} for bc in brain_ctxs},
        "k_folds": k_eff,
        "heldout_folds": [f["heldout"] for f in heldout_folds],
    }, indent=2))

    # PARENT-RELATIVE gate: each generation must beat the CURRENT policy (its
    # parent). Score the SEED on the pooled held-out once to set the bar.
    log("Scoring SEED policy on pooled held-out — sets the bar gen 1 must beat...")
    with Heartbeat("scoring seed policy on pooled held-out"):
        seed_pooled, seed_runs = _score_pooled(
            heldout_ctxs, "heldout_names", str(work_heuristics), "heldout",
            max_class_size, verbose, splits_only=splits_only,
        )
    # Dense split-repair fitness of the SEED (the PRIMARY gate signal): correct
    # held-out merges minus false ones. Edge Accuracy stays computed/recorded but is
    # no longer the bar — it reads +0.000 for most real repairs (only a bridged
    # split EDGE moves it), which is why evolution flat-lined. See classify_merge_edits.
    parent_repair = _pooled_split_repair(heldout_ctxs, seed_runs)
    log(f"Seed split-repair: correct={parent_repair['correct']} "
        f"false={parent_repair['false']} score={parent_repair['score']} "
        f"(unscored={parent_repair['unscored']}); penalized FITNESS = score - "
        f"{merge_penalty:g}*false = {_fitness(parent_repair, merge_penalty):g} — the "
        f"bar gen 1 must beat by score_margin={score_margin} (keep iff fitness gain "
        f">= {score_margin}; a false merge costs {merge_penalty:g} each, so it is "
        f"heavily discouraged but no longer an automatic reject)")
    # Full pooled held-out metric vector of the parent (run-length-weighted over the
    # pool). Advances on every accept.
    parent_metrics = {m: scoring._weighted_avg(seed_pooled, m)
                      for m in _FOLD_METRIC_COLS}
    # Parent's held-out Edge Accuracy — a DIAGNOSTIC shown at the human gate, NOT the
    # accept/reject bar (that bar is parent_repair["score"], the split-repair score).
    parent_edge_accuracy = parent_metrics["Edge Accuracy"]
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
    log(f"Seed policy pooled held-out Edge Accuracy = {parent_edge_accuracy:.4f} "
        f"(full); K-fold mean = {parent_mean_acc:.4f} over {k_eff} fold(s) "
        f"(diagnostic; the actual bar gen 1 must beat is the split-repair score above)")

    options, guard_state = build_options(model=model, run_dir=run_dir)
    log(f"Reviser model: {model} (Anthropic API)")
    log("Run isolation: reviser tools = Read/Write/Edit/Task; cross-run file "
        "access DENIED via can_use_tool (audit -> tool_audit.jsonl)")
    # Validated discovery priors: pass the ABSOLUTE path (the reviser's cwd is the
    # run dir, and the file lives outside runs/, so the isolation guard allows
    # reading it). None when the knowledge base is absent => prompt unchanged.
    # --no-priors disables the discovery knowledge base entirely (ablation: does the
    # reviser improve WITHOUT the cross-run priors?). When off, priors_path is None so
    # _format_priors emits nothing and the prompt is unchanged.
    if not use_priors:
        priors_path = None
        log("Discovery priors: DISABLED by --no-priors (reviser runs without the "
            "knowledge base)")
    else:
        priors_path = str(DISCOVERY_PRIORS.resolve()) if DISCOVERY_PRIORS.is_file() else None
        if priors_path:
            log(f"Discovery priors: reviser will read {priors_path} "
                f"(trusting GENERALIZES+UPHELD findings only)")
        else:
            # Priors were WANTED (no --no-priors) but the knowledge base file is
            # absent. This is an UNINTENDED degradation (≠ the deliberate --no-priors
            # ablation), so flag it loudly: the run will proceed un-grounded, and the
            # fix is to build the file via run_consolidation_workflow.py.
            log(f"   [WARN] Discovery priors WANTED (no --no-priors) but the knowledge "
                f"base is MISSING at {DISCOVERY_PRIORS} — proceeding WITHOUT priors "
                f"(reviser is un-grounded this run). Build it via "
                f"`python agentic/run_consolidation_workflow.py`, or pass --no-priors "
                f"to silence this if running un-grounded is intentional.")
    # B: memory of revisions tried against the CURRENT parent; cleared when the
    # parent advances (an accept), since past rejections no longer apply.
    attempts_vs_parent: list[dict] = []
    # Over-merge taboo: revisions that CREATED a false merge on held-out. A false
    # merge is a geometric fact about the loosened condition, not a parent-relative
    # miss, so this list PERSISTS across parents (never cleared on accept) — it stops
    # the reviser re-deriving the same over-merge direction against each new parent.
    structural_lessons: list[dict] = []
    # Convergence / early-stop (opt-in via --converge-patience > 0): record the
    # ACCEPTED parent penalized FITNESS at the END of each generation. We stop when
    # the parent has gained < converge_eps over the last converge_patience gens — a
    # plateau where further reviser calls (one LLM + two scoring passes each) are
    # unlikely to pay off. Disabled (0) reproduces the run-all-generations behavior.
    parent_score_history: list[float] = [_fitness(parent_repair, merge_penalty)]  # gen 0
    # Generalization-gap meta-signal (shown to the reviser): for each ACCEPTED
    # generation, the candidate's TRAIN split-repair score vs its HELD-OUT score.
    # A persistently larger train gain than held-out gain = overfitting the train
    # split. We expose ONLY the aggregate scalars (train_score, heldout_score per
    # accepted gen) — never which neurons — so held-out identity stays hidden while
    # "are my changes generalizing?" becomes visible feedback. Each entry:
    # {"gen": int, "train": int, "heldout": int}.
    gen_gap_history: list[dict] = []
    # CLOSED priors feedback loop (fixes the "measured but ignored" gap): how often
    # each finding has been SURFACED (rotation — fade repeats so the whole qualifying
    # set gets exposure) and which findings were CITED in a generation that was then
    # REVERTED (down-weight tried-and-failed levers). Both feed select_for_report.
    priors_shown_counts: dict[int, int] = {}
    priors_failed: set[int] = set()
    # TOKEN COST: open a FRESH ClaudeSDKClient PER GENERATION (not one shared session
    # for the whole loop). A shared session accumulates every prior generation's
    # failure report (~6-7K tokens each) + full reviser transcript in-context, so
    # input tokens grew ~QUADRATICALLY with generation count — yet none of that stale
    # history is needed: each generation is parent-relative and stateless, and the
    # only memory that must carry forward is re-encoded compactly and passed into
    # ask_reviser every gen (attempts_vs_parent, gen_gap_history, structural_lessons).
    # A clean per-gen session makes per-generation input cost FLAT (system prompt +
    # this gen's report + those small summaries) with zero loss of carried memory.
    for gen in range(1, generations + 1):
        async with ClaudeSDKClient(options=options) as client:
            print(f"\n=== Generation {gen}/{generations} ===")
            gen_wall0 = time.monotonic()
            gen_dir = run_dir / f"gen{gen:02d}"
            snapshot(gen_dir, work_heuristics, work_rules)  # revert source if rejected

            # (1-3) Run CURRENT policy on each brain's TRAIN; build the failure
            # report. Per brain so raw labels / merge targets stay brain-local.
            log("Step 1-3: run current policy on train, build failure report...")
            with Heartbeat(f"gen {gen}: running policy on train"):
                _, train_runs = _score_pooled(
                    train_ctxs, "train_names", str(work_heuristics), "train",
                    max_class_size, verbose, splits_only=splits_only,
                )
            # Train split-repair score of the CURRENT policy (this gen, on train).
            # Computed HERE (before the report) so it can BOTH pick the grounding
            # priors' failure mode below AND feed the generalization-gap signal later.
            # Same metric as the gate, but on the TRAIN map (leak-free); cheap (reuses
            # train_runs' edits).
            train_repair = _pooled_split_repair(train_ctxs, train_runs,
                                                map_attr="label_gt_map")
            # Layer 1 — inline the FEW validated priors matched to this generation's
            # dominant failure mode, so the reviser is grounded by DEFAULT (no reliance
            # on it opening the knowledge-base file). Empty list when priors are
            # unavailable/disabled, so the writers splice nothing.
            failure_mode = _dominant_failure_mode(train_repair)
            # Content-match the priors to this gen's actual failure (signature) and
            # thread the CLOSED loop: down-weight tried-and-reverted findings, rotate
            # by prior exposure. Returns which findings were surfaced so we can update
            # the rotation counter.
            priors_signature = _failure_signature(train_repair, train_runs)
            priors_section, priors_shown_now = (
                priors_kb.build_report_section(
                    priors_path, failure_mode, signature=priors_signature,
                    failed_findings=priors_failed, shown_counts=priors_shown_counts)
                if priors_path else ([], []))
            for _n in priors_shown_now:
                priors_shown_counts[_n] = priors_shown_counts.get(_n, 0) + 1
            report_path = str(gen_dir / "failure_report.md")
            # SPLIT-ERROR-ONLY: withhold merge_labels from the report so every
            # merge-diagnosis section (Baseline merge errors, MergeSite feature
            # tables, detector recall gap) is suppressed — otherwise the report would
            # keep coaching the reviser toward split_label repairs this run drops. The
            # train label→GT map is still passed: it drives the SplitSite audit, which
            # is exactly the split-error signal we DO want.
            report_merge_labels = (lambda bc: None if splits_only else bc.merge_labels)
            per_brain_report = [
                (bc.brain, tr,
                 scoring.ScoreResult(  # baseline on this brain's train, for the report
                     primary=scoring._weighted_avg(bc.base_train, "Edge Accuracy"),
                     metrics={}, per_swc=bc.base_train, output_dir="", seconds=0.0),
                 report_merge_labels(bc), bc.label_gt_map, bc.fragments_graph,
                 bc.image_probe_section)
                for bc, tr in zip(train_ctxs, train_runs)
            ]
            if len(per_brain_report) == 1:
                _, tr, base_sr, ml, lgm, fg, probe = per_brain_report[0]
                cand.write_failure_report(tr, base_sr, report_path,
                                          merge_labels=ml, label_gt_map=lgm,
                                          fragments_graph=fg, extra_sections=probe,
                                          priors_section=priors_section,
                                          merge_penalty=merge_penalty)
            else:
                cand.write_multibrain_failure_report(per_brain_report, report_path,
                                                     priors_section=priors_section,
                                                     merge_penalty=merge_penalty)
            if priors_section:
                log(f"   grounding priors inlined (failure mode: {failure_mode})")
            train_acc = scoring._weighted_avg(
                _pd.concat([tr.score.per_swc for tr in train_runs]), "Edge Accuracy")
            n_edits_total = sum(tr.n_edits for tr in train_runs)
            log(f"   train Edge Accuracy={train_acc:.4f} "
                f"({n_edits_total} edits across {len(train_runs)} brain(s)); "
                f"report -> {report_path}")
            # SPLIT-ERROR-ONLY visibility: with no MergeSite enumerated the policy
            # should emit no split_label, but if it hardcoded one we dropped it — say
            # so loudly rather than letting the discard be silent.
            n_dropped = sum(getattr(tr, "n_split_label_dropped", 0) for tr in train_runs)
            if n_dropped:
                log(f"   [WARN] splits-only: dropped {n_dropped} split_label edit(s) "
                    f"the policy emitted — merge repairs are disabled this run")
            # TRAIN-SIDE over-merge alert (diagnostic; the gate judges false merges on
            # HELD-OUT, so these never reject — but they are real over-merges the gate
            # is blind to). Pool each brain's train edits against ITS OWN train map.
            train_false = sum(
                inc.classify_merge_edits(tr.edits, bc.label_gt_map)["false"]
                for bc, tr in zip(train_ctxs, train_runs))
            if train_false:
                log(f"   [ALERT] {train_false} train-side false merge(s) — over-merges "
                    f"the held-out gate does NOT see (see failure report)")
            # (train_repair for the generalization-gap signal was computed above,
            # before the report, so it could also select the grounding priors.)

            # (4-5) Ask the agent to explain and revise the WORKING-COPY artifacts.
            log("Step 4-5: proofreader-reviser diagnoses and revises artifacts...")
            reviser_transcript = str(gen_dir / "reviser_transcript.md")
            with Heartbeat(f"gen {gen}: waiting on reviser (LLM)"):
                diagnosis, in_tok, out_tok, cost, read_priors = await ask_reviser(
                    client, report_path, str(work_heuristics), str(work_rules), verbose,
                    attempts=attempts_vs_parent, priors_path=priors_path,
                    splits_only=splits_only, gen_gap=gen_gap_history,
                    transcript_path=reviser_transcript,
                    structural_lessons=structural_lessons,
                )
            log(f"   reviser transcript (thinking + text + tools) -> {reviser_transcript}")

            # (A) Persist what the reviser wrote THIS generation — before scoring or
            # any revert — so even rejected candidates are inspectable afterward.
            candidate_path, diffstat = save_candidate(gen_dir, work_heuristics, work_rules)
            log(f"   candidate saved -> {candidate_path} (diffstat vs parent: {diffstat})")

            # Layer 2 — GROUNDED detection. The prompt asks the reviser to cite the
            # finding number(s) it relied on in the rules.md change log; a citation in
            # the lines it ADDED this generation means a prior actually SHAPED the edit
            # (strictly stronger than read_priors, which only proves the file was
            # opened). We diff the parent rules snapshot (gen_dir/rules.md, from
            # snapshot()) against what the reviser wrote (work_rules) and scan only the
            # added lines. grounded is None when no priors were configured.
            grounded = None
            cited = set()
            supported = set()          # citations whose CONTENT shows up in the edit
            if priors_path:
                try:
                    import difflib as _dl
                    def _added_lines(parent_path, cand_path):
                        p = parent_path.read_text().splitlines()
                        c = Path(cand_path).read_text().splitlines()
                        return "\n".join(
                            ln[1:] for ln in _dl.unified_diff(p, c, lineterm="")
                            if ln.startswith("+") and not ln.startswith("+++"))
                    _added_rules = _added_lines(gen_dir / "rules.md", work_rules)
                    # Also diff the CODE the reviser added — a finding is only truly
                    # "used" if its geometry shows up in the rule or the code, not just
                    # the change-log prose.
                    _added_code = _added_lines(gen_dir / "heuristics.py", work_heuristics)
                    cited = priors_kb.cited_findings(_added_rules)
                    # #5: strengthen the grounded proxy — a citation is SUPPORTED only
                    # if the finding's contentful vocabulary actually appears in what
                    # the reviser changed (rules + code), distinguishing "cited AND
                    # used" from "typed #N but changed something unrelated".
                    supported = priors_kb.supported_citations(
                        _added_rules, _added_code, priors_kb.load_findings(priors_path))
                except Exception:
                    cited = set(); supported = set()
                # grounded now requires a SUPPORTED citation (stronger than a bare #N).
                grounded = bool(supported)
            if grounded:
                _bare = sorted(cited - supported)
                log(f"   grounded: reviser cited AND used discovery finding(s) "
                    f"{sorted(supported)}"
                    + (f" (also cited but unsupported: {_bare})" if _bare else ""))
            elif grounded is False:
                if cited:
                    log(f"   [WARN] cited finding(s) {sorted(cited)} but none are "
                        f"SUPPORTED (the edit does not reflect the finding's geometry) "
                        f"— counted EXPLORATORY, not grounded")
                else:
                    log("   [WARN] priors were inlined but the reviser cited NO finding "
                        "— this generation is EXPLORATORY (un-grounded); recorded so we "
                        "can compare grounded vs exploratory generalization")

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
                for bc, tr in zip(train_ctxs, train_runs):
                    report_labels |= collect_report_labels(bc.merge_labels, tr)
                lint_ok, lint_reason = lint_no_hardcoded_labels(
                    str(work_heuristics), report_labels
                )
                if not lint_ok:
                    import_ok = False
                    log(f"   [WARN] no-hardcode lint FAILED: {lint_reason}; "
                        f"reverting this gen")
                    revert(gen_dir, work_heuristics, work_rules)

            # The bar this gen tried to beat (pre-update): the parent's penalized
            # FITNESS (split-repair score minus merge_penalty * false), which is the
            # gate's decision variable. Capture the raw parent score too, BEFORE any
            # accept advances parent_repair, so the ledger can record the true bar.
            parent_bar = _fitness(parent_repair, merge_penalty)
            parent_score_raw = parent_repair["score"]
            train_seconds = sum(tr.score.seconds for tr in train_runs)
            if not import_ok:
                # Revision was already reverted to the parent above; don't waste a
                # held-out scoring pass on it. Record as a non-improving gen.
                heldout_acc = parent_mean_acc
                cand_repair = dict(parent_repair)  # no change: candidate == parent
                cand_fitness = parent_bar          # == parent fitness (no change)
                heldout_n_edits = 0                # nothing scored on held-out
                heldout_dropped = 0                # nothing scored -> nothing dropped
                eval_seconds = train_seconds
                keep = False
                human_touches = 0
                # No held-out scoring ran, so there is no decision to gauge confidence
                # for; record neutral placeholders (the ledger call below is shared).
                cand_conf = {"soft_correct": 0, "soft_false": 0, "soft_ambiguous": 0,
                             "unknown": 0, "covered": 0, "coverage": float("nan"),
                             "gt_fraction": float("nan")}
                confidence = "n/a"
            else:
                # (6) Re-run the REVISED policy on each brain's HELD-OUT, pool results.
                log("Step 6: score revised policy on pooled held-out...")
                with Heartbeat(f"gen {gen}: scoring revised policy on pooled held-out"):
                    heldout_pooled, heldout_runs = _score_pooled(
                        heldout_ctxs, "heldout_names", str(work_heuristics), "heldout",
                        max_class_size, verbose, splits_only=splits_only,
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

                # (7) PRIMARY GATE = penalized split-repair FITNESS on pooled held-out.
                # Edge Accuracy only moves when a merge bridges a true split EDGE, so
                # most correct repairs read +0.000 and evolution flat-lined. The
                # split-repair score counts EVERY correctly-repaired held-out split
                # (correct) and penalizes every wrong fusion (false); the gate then
                # subtracts a HEAVY per-false-merge penalty to get the fitness:
                #   fitness = (correct - false) - merge_penalty * false.
                # A candidate is kept iff its fitness beats the parent's by at least
                # ``score_margin``. This REPLACES the old hard 'false == 0' reject: a
                # false merge is still heavily discouraged (at merge_penalty=100 it
                # costs ~100 correct repairs to offset), but a change that trades a
                # single false merge for a large recall gain is no longer rejected
                # outright — the landscape stays smooth around the precision boundary.
                #
                # MARGIN (score_margin, integer >= 1). The bar is
                # ``fitness >= parent + score_margin``, NOT a strict ``>`` — a strict
                # ``>`` accepts a +1 win, which on a small held-out set is within
                # the noise of a single repaired split flipping correct<->unscored
                # from one revision to the next. score_margin=1 accepts any net
                # improvement; 2-3 requires the gain to clear that single-repair noise
                # floor before it is locked in as the parent.
                cand_repair = _pooled_split_repair(heldout_ctxs, heldout_runs)
                heldout_n_edits = sum(hr.n_edits for hr in heldout_runs)
                heldout_dropped = sum(
                    getattr(hr, "n_split_label_dropped", 0) for hr in heldout_runs)
                has_split_edit = any(
                    isinstance(e, dict) and e.get("kind") == "split_label"
                    for hr in heldout_runs for e in (hr.edits or [])
                )
                cand_fitness = _fitness(cand_repair, merge_penalty)
                _gain = cand_fitness - parent_bar
                _fmerges = cand_repair["false"]
                _fnote = ("false 0" if _fmerges == 0 else
                          f"false {_fmerges} (penalty -{merge_penalty:g}*{_fmerges}="
                          f"{-merge_penalty * _fmerges:g})")
                if _gain >= score_margin:
                    improved = True
                    gate_reason = (
                        f"fitness {parent_bar:g} -> {cand_fitness:g} ({_gain:+g} >= "
                        f"margin {score_margin}; correct {parent_repair['correct']}"
                        f"->{cand_repair['correct']}, {_fnote})")
                else:
                    improved = False
                    gate_reason = (
                        f"fitness {cand_fitness:g} did not beat parent {parent_bar:g} "
                        f"by margin {score_margin} ({_gain:+g}; "
                        f"correct={cand_repair['correct']}, {_fnote})")
                heldout_acc = fold_summary_mean = cand_pooled_metrics["Edge Accuracy"]
                fold_summary = {"cand_mean": heldout_acc,
                                "cand_fold_acc": [m["Edge Accuracy"] for m in cand_fold_metrics],
                                "cand_stdev": 0.0}
                log(f"   gate: {gate_reason}")
                log(f"   (recorded: Edge Accuracy {heldout_acc:.4f}, "
                    f"%Merged {cand_pooled_metrics['% Merged Edges']:.4f}, "
                    f"#Merges {cand_pooled_metrics['# Merges']:.2f})")
                # BLIND SPOT: how many held-out merges the gate could not verify (no
                # GT coverage at either endpoint) — these carry NO correctness check.
                _unsc = cand_repair["unscored"]
                _frac = (_unsc / heldout_n_edits) if heldout_n_edits else float("nan")
                log(f"   blind spot: {_unsc}/{heldout_n_edits} held-out merges "
                    f"UNSCORED ({_frac:.0%}) — outside held-out GT coverage, neither "
                    f"rewarded nor penalized by the gate")
                # ① CONFIDENCE: independent, CALIBRATED multi-feature image verdict on
                # that blind spot, replayed from reads the policy already made (zero
                # extra reads), folded into a coarse decision-confidence level. Advisory
                # by default; only --confidence-veto lets an image-flagged false merge
                # block accept.
                cand_conf = _image_confidence(cand_repair)
                confidence = _confidence_level(cand_repair, cand_conf)
                log(f"   confidence: {confidence.upper()} "
                    f"(gt_fraction={cand_conf['gt_fraction']:.0%}; model="
                    f"{cand_conf['model_kind']}; blind-spot image verdict: "
                    f"{cand_conf['soft_correct']} likely-correct / "
                    f"{cand_conf['soft_false']} likely-FALSE / "
                    f"{cand_conf['soft_ambiguous']} ambiguous / {cand_conf['unknown']} "
                    f"unimaged; coverage={cand_conf['coverage']:.0%})")
                # PER-EDIT confidence: one verdict per held-out merge (GT-authoritative
                # where covered, calibrated multi-feature image model in the blind spot).
                # A GATE-side triage artifact (uses the held-out GT map) — written to the
                # gen dir, NEVER shown to the reviser. Zero extra cloud reads.
                edit_conf_rows = _per_edit_confidence(cand_repair)
                try:
                    (gen_dir / "edit_confidence.json").write_text(json.dumps(
                        edit_conf_rows, indent=2, default=str))
                    log(f"   per-edit confidence: {len(edit_conf_rows)} edit(s) -> "
                        f"{gen_dir / 'edit_confidence.json'}")
                except OSError as _e:
                    log(f"   [warn] could not write per-edit confidence ({_e})")
                human_touches = 0
                if human:
                    human_touches = 1
                    keep = human_gate(gen, train_acc, heldout_acc, parent_edge_accuracy)
                else:
                    keep = improved
                    # OPT-IN veto (--confidence-veto): if the hard gate would ACCEPT
                    # but the decision is LOW confidence — a large unverified blind spot
                    # AND the independent image signal flags likely-false merges in it —
                    # block it. This acts ONLY on POSITIVE evidence of a bad merge
                    # (soft_false), never on mere absence of GT, so it does not push the
                    # policy away from untraced neurons.
                    if confidence_veto and improved and confidence == "low":
                        keep = False
                        log(f"   [VETO] hard gate accepted but confidence is LOW "
                            f"({cand_conf['soft_false']} image-flagged likely-false "
                            f"merge(s) in a blind spot with gt_fraction="
                            f"{cand_conf['gt_fraction']:.0%}); reverting. Disable with "
                            f"no --confidence-veto to keep it as an advisory label only.")
                        revert(gen_dir, work_heuristics, work_rules)

            # B: one-line summary of what this generation tried, for the memory.
            attempt_summary = (
                f"{diffstat} lines; " + (diagnosis or "").strip().split("\n", 1)[0][:120]
            ) or "(no diagnosis text)"
            if keep:
                parent_edge_accuracy = cand_pooled_metrics["Edge Accuracy"]  # diagnostic
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
                    # Record this accepted policy's train vs held-out split-repair
                    # scores for the generalization-gap meta-signal (next gen's prompt).
                    # ``grounded`` lets the next gen compare grounded vs exploratory
                    # generalization (Layer 2).
                    gen_gap_history.append({
                        "gen": gen,
                        "train": train_repair["score"],
                        "heldout": cand_repair["score"],
                        "grounded": grounded,
                    })
                shutil.copy2(work_heuristics, gen_dir / "heuristics.accepted.py")
                shutil.copy2(work_rules, gen_dir / "rules.accepted.md")
                note = "accepted (new parent)"
                if k_eff > 1 and import_ok:
                    note += (f" [folds {[round(a,2) for a in fold_summary['cand_fold_acc']]}"
                             f" ±{fold_summary['cand_stdev']:.3f}]")
                # Parent advanced: prior rejections were against the OLD parent and
                # no longer apply, so clear the in-prompt memory.
                attempts_vs_parent = []
                # A cited finding that just got ACCEPTED is no longer a failed lever —
                # clear it so it can be offered again to a future generation.
                priors_failed -= cited
            else:
                revert(gen_dir, work_heuristics, work_rules)  # restore the parent
                note = "reverted (did not beat parent)"
                # CLOSE THE PRIORS LOOP (#2): a finding cited in a REVERTED generation
                # is a lever that was tried and did not beat the parent — down-weight it
                # in future selection so the reviser stops being handed the same failed
                # lever. (Cleared on a later accept above.)
                priors_failed |= cited
                # Remember this rejected attempt so the next gen proposes something new.
                # Delta is in FITNESS units (penalized), matching the gate.
                attempts_vs_parent.append({
                    "gen": gen,
                    "summary": attempt_summary,
                    "heldout": cand_fitness - parent_bar,
                    "accepted": False,
                })
                # Structural lesson: a rejection whose candidate ALSO created a false
                # merge is a fact about geometry, not about this parent, so it must
                # outlive the parent. (With the smoothed gate a false merge no longer
                # forces the reject — but if the candidate was rejected AND carried a
                # false merge, the over-merge direction is still worth remembering.)
                # Guard on import_ok: an import/lint failure copies parent_repair into
                # cand_repair (false==0), so it can never be mis-logged as an over-merge.
                if import_ok and cand_repair["false"] > 0:
                    structural_lessons.append({
                        "gen": gen,
                        "false": cand_repair["false"],
                        "summary": attempt_summary,
                    })
                    # Dedup by summary (keep most recent) and cap, to bound prompt size.
                    seen, dedup = set(), []
                    for L in reversed(structural_lessons):
                        if L["summary"] in seen:
                            continue
                        seen.add(L["summary"]); dedup.append(L)
                    structural_lessons = list(reversed(dedup))[-STRUCTURAL_LESSON_CAP:]
            # B (durable): every generation's attempt is persisted in ledger.jsonl
            # (the single source of truth) via ledger.record(...) just below. The old
            # per-run attempts.md is no longer written inline; render the same
            # human-readable timeline on demand from the ledger with
            # ``python -m proofreader_evolve.harness.ledger <run_dir>`` (see
            # ledger.render_attempts_timeline).
            log(f"Step 7: fitness={cand_fitness:g} (split-repair "
                f"score={cand_repair['score']}, correct={cand_repair['correct']}, "
                f"false={cand_repair['false']}; parent fitness={parent_bar:g}); "
                f"Edge Accuracy={heldout_acc:.4f} -> {note}")

            ledger.record(GenerationCost(
                generation=gen,
                wall_seconds=time.monotonic() - gen_wall0,
                eval_seconds=eval_seconds,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=cost,
                n_evaluations=2,
                human_interventions=human_touches,
                train_edge_accuracy=train_acc,
                heldout_edge_accuracy=heldout_acc,
                parent_split_repair_score=parent_score_raw,  # raw parent score, for plots
                accepted=keep,
                note=note,
                heldout_n_edits=heldout_n_edits,
                heldout_correct_merges=cand_repair["correct"],
                heldout_false_merges=cand_repair["false"],
                heldout_split_repair_score=cand_repair["score"],
                merge_penalty=merge_penalty,
                heldout_fitness=cand_fitness,
                parent_fitness=parent_bar,
                heldout_unscored_merges=cand_repair["unscored"],
                heldout_unscored_fraction=(
                    cand_repair["unscored"] / heldout_n_edits
                    if heldout_n_edits else float("nan")),
                decision_confidence=confidence,
                heldout_gt_fraction=cand_conf["gt_fraction"],
                heldout_soft_correct=cand_conf["soft_correct"],
                heldout_soft_false=cand_conf["soft_false"],
                heldout_soft_ambiguous=cand_conf["soft_ambiguous"],
                confidence_veto=confidence_veto,
                splits_only=splits_only,
                heldout_split_label_dropped=heldout_dropped,
                read_priors=read_priors,
                grounded=grounded,
                cited_findings=",".join(str(n) for n in sorted(cited)),
                supported_findings=",".join(str(n) for n in sorted(supported)),
                train_false_merges=train_false,
                candidate_path=candidate_path,
                heuristics_diffstat=diffstat,
                diagnosis=(diagnosis or "")[:2000],  # truncate; full text is in stdout
            ))

            # Convergence / early-stop (opt-in). The parent's penalized FITNESS AFTER
            # this gen's accept/revert is the bar, so the history tracks the accepted
            # frontier. Stop once it has gained < converge_eps over the last
            # converge_patience generations (a plateau).
            parent_score_history.append(_fitness(parent_repair, merge_penalty))
            if converge_patience > 0 and len(parent_score_history) > converge_patience:
                window_gain = parent_score_history[-1] - parent_score_history[-1 - converge_patience]
                if window_gain < converge_eps:
                    log(f"[converged] parent fitness gained {window_gain:g} "
                        f"(< eps {converge_eps}) over the last {converge_patience} "
                        f"generation(s) — stopping early at gen {gen}/{generations}.")
                    break

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

    # Auto-generate the per-generation performance figure from the ledger we just
    # wrote, so it appears next to the run with no separate step. Best-effort: a
    # plotting failure (e.g. headless matplotlib quirk) must never fail a finished run.
    perf_png = run_dir / "performance.png"
    try:
        from proofreader_evolve import plot_run_performance as _prp
        _rows = _prp.load_ledger(run_dir)
        if _rows:
            _prp.make_figure(_rows, run_dir.name, perf_png)
            log(f"Performance figure -> {perf_png}")
    except Exception as _e:
        log(f"[WARN] could not auto-generate performance figure ({_e}); "
            f"run `python proofreader_evolve/plot_run_performance.py {run_dir.name}` manually")

    # Auto-generate the policy-evolution summary figure (what lever each accepted
    # generation added, in plain English via the LLM). Best-effort, same as above.
    policy_png = run_dir / "policy_evolution.png"
    try:
        from proofreader_evolve import plot_policy_evolution as _ppe
        _summary = _ppe.collect(run_dir)
        if _summary["rows"]:
            _ppe.summarize_with_llm(_summary)        # plain-English summaries
            _ppe.make_figure(_summary, policy_png)
            log(f"Policy-evolution figure -> {policy_png}")
    except Exception as _e:
        log(f"[WARN] could not auto-generate policy-evolution figure ({_e}); "
            f"run `python proofreader_evolve/plot_policy_evolution.py {run_dir.name}` manually")

    print("\n=== Evolution complete ===")
    print(ledger.summarize())
    print(f"Cross-run access attempts (denied): {n_viol}")
    print(f"Originals (unchanged)        -> {ARTIFACTS}")
    print(f"Evolved artifacts (this run) -> {work_heuristics.parent}")
    print(f"Run log + ledger             -> {run_dir}")
    if perf_png.exists():
        print(f"Performance figure           -> {perf_png}")
    if policy_png.exists():
        print(f"Policy-evolution figure      -> {policy_png}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--brain", default="789202", help="brain_id (must have a cache pkl)")
    p.add_argument("--mcl", type=int, default=100,
                   help="min_cable_length of the fragment cache to load "
                        "(cache/dataset_cache_<brain>_mcl<MCL>.pkl). Default 100. A "
                        "smaller value (e.g. 10) keeps shorter fragments, widening the "
                        "candidate stream the policy can reach (raises the recall "
                        "ceiling) at the cost of more, noisier candidates. Applies to "
                        "every brain in --brains.")
    p.add_argument("--brains", default=None,
                   help="comma-separated brain_ids to POOL (e.g. "
                        "'789202,794491,794492'). Each brain is loaded + prepared + "
                        "split 70/30 separately; the gate scores the policy on every "
                        "brain's own train/held-out and pools the per-skeleton "
                        "results (data is never crossed — raw labels are not "
                        "comparable across brains). Overrides --brain. NOTE: each "
                        "brain holds ~1.6 GB prepared state in memory simultaneously.")
    p.add_argument("--train-brains", default=None,
                   help="CROSS-BRAIN mode: comma-separated brain_ids whose WHOLE "
                        "neuron set is the TRAIN/feedback split (the reviser learns "
                        "from these). Must be used WITH --test-brains, and the two "
                        "sets must not overlap. When given, there is NO within-brain "
                        "split — train brains feed the failure report, test brains "
                        "feed the gate. Overrides --brain/--brains.")
    p.add_argument("--test-brains", default=None,
                   help="CROSS-BRAIN mode: comma-separated brain_ids whose WHOLE "
                        "neuron set is the HELD-OUT gate (scored, never shown to the "
                        "reviser). The gate then measures generalization on brains the "
                        "feedback never touched — no fragment of a test neuron appears "
                        "in train. Must be used WITH --train-brains.")
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--heldout-fraction", type=float, default=0.5,
                   help="fraction of EACH brain's GT skeletons reserved for held-out "
                        "gating (the rest are train). Default 0.5; pass 0.3 for the "
                        "70/30 train/test split.")
    p.add_argument("--k-folds", type=int, default=1,
                   help="K-fold slicing of the held-out set, DIAGNOSTIC ONLY. The "
                        "accept/reject gate is the POOLED split-repair score (see "
                        "--score-margin), not a per-fold vote; K>1 only adds per-fold "
                        "Edge-Accuracy breakdown to the log / split.json so you can "
                        "see how stable a generation's result is across folds. "
                        "Default 1. Clamped to the held-out neuron count. Scoring "
                        "cost is unchanged — folds are row-slices of one scoring pass.")
    p.add_argument("--human-gate", action="store_true",
                   help="ask a human before keeping each revision")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"reviser model id (default: {DEFAULT_MODEL}, Anthropic API)")
    p.add_argument("--score-margin", type=int, default=1,
                   help="minimum amount a generation's pooled held-out penalized "
                        "FITNESS (split-repair score minus --merge-penalty*false) must "
                        "EXCEED the parent's by to be accepted: keep iff fitness gain "
                        ">= score-margin. Default 1 (accept any net improvement). "
                        "Raise to 2-3 so a single-repair swing (within noise on a "
                        "small held-out set) is not locked in as the new parent. "
                        "Integer.")
    p.add_argument("--merge-penalty", type=float, default=100.0,
                   help="per-false-merge penalty in the held-out FITNESS "
                        "(fitness = (correct - false) - merge_penalty*false). REPLACES "
                        "the old hard 'zero false merges' gate with a smooth one: at "
                        "the default 100, one false merge (fusing two different "
                        "neurons) costs ~100 correct repairs to offset — heavily "
                        "discouraged but no longer an automatic reject, so a change "
                        "that trades one false merge for a large recall gain can be "
                        "kept. Raise for stricter precision, lower to tolerate more "
                        "merge error. Set very high (e.g. 1e9) to recover the old "
                        "hard gate.")
    p.add_argument("--confidence-veto", action="store_true",
                   help="OPT-IN: let the ① confidence layer BLOCK an otherwise-accepted "
                        "generation when the decision is LOW confidence — i.e. it rests "
                        "on a large unverified blind spot AND the independent image "
                        "signal (bridge_ratio, replayed from reads the policy already "
                        "made — no extra cloud reads) flags likely-FALSE merges in it. "
                        "Acts ONLY on positive image evidence of a bad merge, never on "
                        "mere absence of GT, so it does not push the policy away from "
                        "untraced neurons. Default OFF: the confidence level is computed "
                        "and logged/recorded as an advisory label only.")
    p.add_argument("--max-class-size", type=int, default=None,
                   help="hard cap on labels fused into one merge class (guardrail "
                        "against brain-spanning mega-merges); default: no cap")
    p.add_argument("--converge-patience", type=int, default=0,
                   help="early-stop: stop if the ACCEPTED parent split-repair score "
                        "improves by < --converge-eps over this many CONSECUTIVE "
                        "generations (a plateau). Default 0 = disabled (run all "
                        "--generations). E.g. 4 stops once 4 gens in a row add < eps.")
    p.add_argument("--converge-eps", type=int, default=1,
                   help="minimum parent-score gain over a --converge-patience window "
                        "that counts as 'still improving'. Default 1 (the score is an "
                        "integer count of net repairs). Only used when "
                        "--converge-patience > 0.")
    p.add_argument("--seed-from", default=None,
                   help="CONTINUE from a prior run's latest accepted policy "
                        "(run-id folder name, or brain id for its newest run) "
                        "instead of the from-scratch seed; default: from scratch")
    p.add_argument("--split-seed", type=int, default=None,
                   help="RNG seed for the train/held-out split. Default: random "
                        "per run (recorded in the run's split.json). Pass an int "
                        "to pin a reproducible split.")
    p.add_argument("--splits-only", action="store_true",
                   help="SPLIT-ERROR-ONLY fast mode: drop every split_label "
                        "(merge-error) edit before scoring, so only merge_labels "
                        "(split-error repairs) are scored. Skips the expensive "
                        "coordinate-aware split path (Dijkstra + fragment rebuild). "
                        "The gate is UNCHANGED — its split-repair metric already "
                        "scores only merge_labels — so this is a faster, "
                        "metric-consistent test; merge-error repair is deferred.")
    p.add_argument("--no-priors", dest="use_priors", action="store_false",
                   help="do NOT point the reviser at the AutoDiscovery knowledge "
                        "base (autodiscovery/all-runs.combined.md). Default: read it. "
                        "Use this for an ablation — does the reviser improve WITHOUT "
                        "the cross-run priors?")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    brains = None
    if args.brains:
        brains = [b.strip() for b in args.brains.split(",") if b.strip()]
    train_brains = test_brains = None
    if args.train_brains:
        train_brains = [b.strip() for b in args.train_brains.split(",") if b.strip()]
    if args.test_brains:
        test_brains = [b.strip() for b in args.test_brains.split(",") if b.strip()]
    try:
        asyncio.run(run_evolution(
            args.brain, args.generations, args.heldout_fraction,
            args.human_gate, args.verbose, args.model,
            score_margin=args.score_margin, max_class_size=args.max_class_size,
            seed_from=args.seed_from, split_seed=args.split_seed,
            k_folds=args.k_folds, brains=brains,
            splits_only=args.splits_only, use_priors=args.use_priors,
            mcl=args.mcl,
            converge_patience=args.converge_patience, converge_eps=args.converge_eps,
            train_brains=train_brains, test_brains=test_brains,
            merge_penalty=args.merge_penalty,
            confidence_veto=args.confidence_veto,
        ))
    except KeyboardInterrupt:
        # Ctrl-C: note it in the (still-teed) log, then exit non-zero quietly.
        log("[interrupted] KeyboardInterrupt — run stopped by user")
        raise SystemExit(130)
    except BaseException:
        # Print the traceback HERE, while stdout/stderr are still teed, so the crash
        # is captured in run.log. (The interpreter's default hook would otherwise fire
        # only AFTER the finally below restored the streams, missing the log.) Exit
        # non-zero via SystemExit(1) so the traceback shows exactly ONCE (SystemExit
        # with an int prints no second traceback), not duplicated by re-raising.
        import traceback as _tb
        _tb.print_exc()
        raise SystemExit(1)
    finally:
        # Restore stdout/stderr and flush+close runs/<id>/run.log on EVERY exit path
        # (normal return, KeyboardInterrupt, or the crash handled just above). No-op
        # if no run log was installed.
        detach_active_run_log()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

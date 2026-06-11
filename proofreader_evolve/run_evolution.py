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


def build_options(model: str = DEFAULT_MODEL) -> ClaudeAgentOptions:
    """SDK session config. setting_sources=['project'] auto-discovers the
    subagent in proofreader_evolve/agents via the project .claude — but since the
    agents live under this folder, we point the agent dir explicitly via cwd and
    a relative agents path is loaded by the SDK's project settings. Tools are
    limited to what the reviser needs.

    The model defaults to Opus 4.8 on the Anthropic API (see DEFAULT_MODEL); the
    reviser subagent's frontmatter says ``model: inherit``, so it uses this same
    model. Pass a different ``model`` to override.
    """
    return ClaudeAgentOptions(
        cwd=str(PROJECT_ROOT),
        setting_sources=["project"],
        model=model,
        env=_anthropic_api_env(),
        allowed_tools=["Task", "Read", "Write", "Edit", "Bash", "Glob"],
        # Read-only/edit-only work; bypass keeps the loop non-interactive. The
        # human gate (if enabled) is enforced in Python after held-out scoring,
        # which is the consequential decision point.
        permission_mode="bypassPermissions",
    )


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


def make_working_artifacts(run_dir: Path) -> tuple[Path, Path]:
    """Copy the pristine artifacts into this run's timestamped dir and return the
    copies' paths. The run reads/revises ONLY these copies; the originals under
    ``artifacts/`` are never touched, so the run is reproducible and the
    before/after files are trivially identifiable (original = ``artifacts/``,
    this run = ``runs/<brain>_<timestamp>/artifacts/``).
    """
    work_dir = run_dir / "artifacts"
    work_dir.mkdir(parents=True, exist_ok=True)
    work_heuristics = work_dir / "heuristics.py"
    work_rules = work_dir / "rules.md"
    shutil.copy2(HEURISTICS, work_heuristics)
    shutil.copy2(RULES, work_rules)
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
    gate_eps: float = 0.05, max_class_size=None,
) -> None:
    run_id = f"{brain}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = HERE / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(str(run_dir / "ledger.jsonl"))

    # Work on TIMESTAMPED COPIES of the artifacts, never the originals. The agent
    # reads/revises only these; artifacts/ stays pristine, so the original vs.
    # evolved files are easy to tell apart afterward.
    work_heuristics, work_rules = make_working_artifacts(run_dir)
    log(f"Run {run_id}")
    log(f"  originals (untouched): {HEURISTICS}")
    log(f"  working copies (revised this run): {work_heuristics}")

    # --- One-time setup: prepared brain (expensive, cached), split, baseline --
    paths = scoring.BrainPaths(brain)

    # Cached fragment graph drives candidate-site GEOMETRY (fast, no GCS).
    cache_path = ds.default_cache_path(brain)
    log(f"Loading cached fragment graph (site geometry): {cache_path}")
    fragments_graph, _gt_graph, _ = ds.load_cached_graphs(cache_path)

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
    train_names, heldout_names = ds.train_heldout_split(
        all_gt_names, heldout_fraction=heldout_fraction
    )
    log(f"GT skeletons: {len(all_gt_names)} total -> "
        f"{len(train_names)} train, {len(heldout_names)} held-out")

    # Baseline (no-edit) restricted to each split — the reference floor and the
    # comparison shown in the failure report.
    base_train = baseline_full.per_swc.loc[
        baseline_full.per_swc.index.isin(train_names)]
    base_heldout = baseline_full.per_swc.loc[
        baseline_full.per_swc.index.isin(heldout_names)]
    baseline_heldout = scoring._weighted_avg(base_heldout, "Edge Accuracy")
    log(f"Baseline (no-edit) held-out Edge Accuracy = {baseline_heldout:.4f}")

    # PARENT-RELATIVE gate: each generation must beat the CURRENT policy (its
    # parent), not the global no-edit baseline. The parent is the last accepted
    # policy (initially the seed in the working copy). Score the seed once to set
    # the bar — with a no-op seed this equals the baseline, but scoring it makes
    # the gate correct for ANY seed and lets sub-baseline lineages still climb.
    seed_heldout_run = cand.run_candidate(
        prepared, fragments_graph, heldout_names, "heldout",
        str(work_heuristics), max_class_size=max_class_size, verbose=verbose,
    )
    parent_heldout = seed_heldout_run.score.primary
    log(f"Seed policy held-out Edge Accuracy = {parent_heldout:.4f} "
        f"(the bar generation 1 must beat)")

    options = build_options(model=model)
    log(f"Reviser model: {model} (Anthropic API)")
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
                str(work_heuristics), max_class_size=max_class_size, verbose=verbose,
            )
            report_path = str(gen_dir / "failure_report.md")
            cand.write_failure_report(
                train_run,
                scoring.ScoreResult(  # baseline on train, wrapped for the report
                    primary=scoring._weighted_avg(base_train, "Edge Accuracy"),
                    metrics={}, per_swc=base_train, output_dir="", seconds=0.0,
                ),
                report_path,
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

            # (6) Re-run the REVISED policy on HELD-OUT and score.
            log("Step 6: score revised policy on held-out...")
            heldout_run = cand.run_candidate(
                prepared, fragments_graph, heldout_names, "heldout",
                str(work_heuristics), max_class_size=max_class_size, verbose=verbose,
            )
            heldout_acc = heldout_run.score.primary
            eval_seconds = train_run.score.seconds + heldout_run.score.seconds

            # (7) Gate: keep only if held-out Edge Accuracy beats the PARENT (the
            # current accepted policy), so partial improvements accumulate across
            # generations instead of each child having to beat the global baseline
            # from the reverted seed.
            parent_bar = parent_heldout  # the bar this gen tried to beat (pre-update)
            # Require a margin (gate_eps) over the parent, not just any increase —
            # held-out has few skeletons, so a strict '>' would lock in noise-level
            # "wins" and inflate the bar the next generation must clear.
            improved = heldout_acc > parent_heldout + gate_eps
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

    print("\n=== Evolution complete ===")
    print(ledger.summarize())
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
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    asyncio.run(run_evolution(
        args.brain, args.generations, args.heldout_fraction,
        args.human_gate, args.verbose, args.model,
        gate_eps=args.gate_eps, max_class_size=args.max_class_size,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

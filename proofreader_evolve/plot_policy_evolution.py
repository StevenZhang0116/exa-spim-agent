"""
Summarize WHAT each accepted generation added to the policy, as a timeline figure.

Takes a run name and reconstructs, generation by generation, the one lever each
ACCEPTED revision introduced — read deterministically from the evolved
``heuristics.accepted.py`` files (no model): the top-level CONSTANT a generation
adds (or changes) is the rule it introduced, and that constant's inline comment is
its rationale. Cross-referenced with the ledger so each row also shows the
split-repair score it reached and the gain over the previous accepted policy.

Why constants (not the free-text diagnosis): every accepted generation in this
loop introduces its lever as a named module-level constant in heuristics.py (e.g.
``TIP_SHAFT_HIGH_RAD_RATIO = 2.0  # caliber FLOOR for ...``). Diffing the accepted
policies' constants against the previous accepted policy yields exactly the
rule(s) added that generation — precise and reproducible, where parsing prose is
not.

Output: a timeline figure ``runs/<run>/policy_evolution.png`` — one row per
accepted generation: the new/changed constant(s), a short rationale from the
comment, and the split-repair score (with the +gain).

Usage (from the ``exa-spim-agent/`` project root)
-------------------------------------------------
    python proofreader_evolve/plot_policy_evolution.py 789202_20260623_005939
    python proofreader_evolve/plot_policy_evolution.py <run_name> --out fig.png
    python proofreader_evolve/plot_policy_evolution.py <run_name> --md summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"

# Top-level constant assignment with an optional trailing comment:
#   NAME = value            # rationale...
_CONST_RE = re.compile(r"^([A-Z][A-Z0-9_]+)\s*=\s*(.+?)\s*(?:#\s*(.*))?$", re.M)


def _resolve_run_dir(run_name: str) -> Path:
    p = Path(run_name)
    if p.is_dir():
        return p
    cand = RUNS_DIR / run_name
    if cand.is_dir():
        return cand
    raise FileNotFoundError(f"run not found: {run_name!r} (looked for {p} and {cand})")


def _constants(path: Path) -> dict:
    """{NAME: (value_str, comment_str)} for every top-level constant in a policy file."""
    if not path.exists():
        return {}
    out = {}
    for m in _CONST_RE.finditer(path.read_text()):
        out[m.group(1)] = (m.group(2).strip(), (m.group(3) or "").strip())
    return out


def _gist(comment: str) -> str:
    """A concise SUMMARY of the lever: the comment's first sentence, COMPLETE.

    The constants' comments lead with a headline clause that already states the idea
    (e.g. "MICRO regime: at/below this gap accept on DISTANCE ALONE, waiving
    colinearity entirely."). We return that first sentence in full — summarized, but
    never truncated mid-word. Trailing parenthetical justifications / extra sentences
    are dropped (that is the summarization).
    """
    if not comment:
        return "(no comment)"
    # First sentence: up to the first '. ' or ';'. Keep it whole.
    head = re.split(r"\.\s|;\s", comment.strip(), maxsplit=1)[0].strip()
    return head.rstrip(".")


def collect(run_dir: Path) -> dict:
    """Reconstruct the per-accepted-generation lever timeline + scores."""
    ledger = [json.loads(l) for l in (run_dir / "ledger.jsonl").read_text().splitlines() if l.strip()]
    ledger.sort(key=lambda r: r.get("generation", 0))
    by_gen = {r["generation"]: r for r in ledger}
    accepted = [r["generation"] for r in ledger if r.get("accepted")]

    rows = []
    prev = {}  # constants of the previous ACCEPTED policy
    # Seed baseline = the bar gen 1 faced (its parent split-repair score).
    base_score = by_gen[accepted[0]].get("parent_split_repair_score",
                                         by_gen[accepted[0]].get("parent_heldout")) if accepted else None
    prev_score = base_score
    for g in accepted:
        cur = _constants(run_dir / f"gen{g:02d}" / "heuristics.accepted.py")
        added = [k for k in cur if k not in prev]
        changed = [k for k in cur if k in prev and cur[k][0] != prev[k][0]]
        # A lever is a SCALAR threshold constant. Drop container-valued ones (dicts /
        # lists / tuples, e.g. ENUM_PARAMS) — their value is multi-line and not the
        # threshold that names the lever; keep them only if nothing scalar moved.
        def _scalar(names):
            s = [k for k in names if not cur[k][0].startswith(("{", "[", "("))]
            return s or names
        # Describe the lever: prefer the newly-added constant(s); else the changed
        # one(s); else flag a logic-only revision (no constant moved).
        if added:
            lever = _scalar(added)
            kind = "added"
        elif changed:
            lever = _scalar(changed)
            kind = "changed"
        else:
            lever = []
            kind = "logic-only"
        # RAW material for the summary (constant name=value + its full comment, or a
        # logic-only note). The natural-language one-liner is filled in later by the
        # LLM (summarize_with_llm); the cleaned first-sentence is the offline fallback.
        if lever:
            raw = "; ".join(f"{k} = {cur[k][0]}  # {cur[k][1]}" for k in lever)
            fallback = f"{_gist(cur[lever[0]][1])}"
        else:
            raw = "Refactor / control-flow change with no new constant or threshold."
            fallback = "control-flow change (no new constant)"
        sr = by_gen[g].get("heldout_split_repair_score")
        gain = (sr - prev_score) if (sr is not None and prev_score is not None) else None
        rows.append({
            "gen": g, "kind": kind, "raw": raw, "summary": fallback,
            "score": sr, "gain": gain,
        })
        prev = cur
        prev_score = sr if sr is not None else prev_score

    return {
        "run": run_dir.name,
        "n_generations": len(ledger),
        "n_accepted": len(accepted),
        "baseline_score": base_score,
        "final_score": rows[-1]["score"] if rows else None,
        "rows": rows,
    }


_MODEL = "us.anthropic.claude-opus-4-8[1m]"


def _llm_text(prompt: str, model: str = _MODEL) -> str:
    """Run one prompt through claude_agent_sdk and return the reply text.

    Inherits the ambient provider config (this environment runs on Bedrock:
    CLAUDE_CODE_USE_BEDROCK=1 + AWS_REGION), so we do NOT override env — the model id
    is the Bedrock inference-profile id that resolves there. Raises on any failure;
    callers wrap in try/except and fall back.
    """
    import asyncio

    async def _run():
        from claude_agent_sdk import (ClaudeAgentOptions, ClaudeSDKClient,
                                      AssistantMessage, TextBlock)
        opts = ClaudeAgentOptions(model=model, permission_mode="bypassPermissions",
                                  allowed_tools=[])
        chunks = []
        async with ClaudeSDKClient(options=opts) as client:
            await client.query(prompt)
            async for msg in client.receive_response():
                content = getattr(msg, "content", None)
                if content:
                    for blk in content:
                        if isinstance(blk, TextBlock) or isinstance(getattr(blk, "text", None), str):
                            chunks.append(blk.text)
                res = getattr(msg, "result", None)
                if isinstance(res, str) and res.strip() and not chunks:
                    chunks.append(res)
        return "".join(chunks)

    return asyncio.run(_run())


def summarize_with_llm(summary: dict, model: str = _MODEL) -> None:
    """Fill each row's ``summary`` with a PLAIN-ENGLISH one-liner via claude_agent_sdk.

    We hand the model the raw lever material (constant name=value + its code comment)
    for every accepted generation at once and ask for one natural-language sentence
    per generation — describing the IDEA of the rule with NO parameter names, code
    symbols, or numbers. This avoids any hardcoded name→phrase mapping, so it works
    for runs that introduce rules we have never seen. Best-effort: on any failure
    (no SDK, no API key, parse error) the rows keep their offline first-sentence
    fallback, so the figure still renders.
    """
    rows = summary["rows"]
    if not rows:
        return
    items = "\n".join(f"[gen {r['gen']}] {r['raw']}" for r in rows)
    prompt = (
        "You are documenting how an automated neuron-proofreading policy evolved. "
        "Each line below is ONE accepted generation: the code constant(s) it added or "
        "changed, with the developer's inline comment. For EACH generation, write ONE "
        "short, plain-English sentence describing the IDEA of the rule it introduced — "
        "what kind of broken-apart neuron fragments it now reconnects, and on what "
        "intuition. STRICT RULES: no parameter/constant names, no code symbols "
        "(no '==', '>=', 'deg_a', etc.), no numbers/thresholds. Speak in terms a "
        "biologist would understand (fragment tips, gaps, cable thickness, straight "
        "continuation, T-junctions, etc.). Return ONLY a JSON object mapping the "
        "generation number (as a string) to its sentence, nothing else.\n\n"
        f"{items}"
    )
    try:
        text = _llm_text(prompt, model)
        cleaned = re.sub(r"```(?:json)?|```", "", text).strip()  # strip md fences
        m = re.search(r"\{.*\}", cleaned, re.S)                  # tolerate prose around JSON
        if not m:
            raise ValueError(f"no JSON object in reply ({len(text)} chars)")
        mapping = json.loads(m.group(0))
        n = 0
        for r in rows:
            s = mapping.get(str(r["gen"]))
            if s:
                r["summary"] = s.strip(); n += 1
        summary["summary_source"] = "llm"
        print(f"[plot_policy_evolution] LLM summarized {n}/{len(rows)} generations")
    except Exception as e:
        summary["summary_source"] = f"offline-fallback ({type(e).__name__})"
        print(f"[plot_policy_evolution] LLM summary unavailable ({e}); offline fallback")


def summarize_decision_hierarchy(summary: dict, run_dir: Path, model: str = _MODEL) -> None:
    """Summarize the FINAL policy's decision process as an ordered list of tiers.

    Reads the latest accepted ``heuristics.accepted.py``'s ``propose_edits`` source and
    asks the LLM for the sequential decision waterfall it implements — each tier a
    short plain-English 'when X → reconnect' rule, in evaluation order (first match
    wins). Stored as ``summary['hierarchy']`` (list of strings). Best-effort.
    """
    # Final accepted policy = the accepted heuristics of the LAST accepted generation.
    last_gen = summary["rows"][-1]["gen"] if summary["rows"] else None
    if last_gen is None:
        return
    pol = run_dir / f"gen{last_gen:02d}" / "heuristics.accepted.py"
    if not pol.exists():
        return
    src = pol.read_text()
    # Extract the propose_edits function body (def ... up to the next top-level def).
    m = re.search(r"\ndef propose_edits\(.*?(?=\Z|\ndef )", src, re.S)
    body = m.group(0) if m else src
    prompt = (
        "Below is the `propose_edits` function of an automated neuron-proofreading "
        "policy. It decides, for each candidate gap between two fragments, whether to "
        "RECONNECT them. The decision is a SEQUENTIAL waterfall: it tests tiers in "
        "order and the FIRST matching tier wins (then it moves to the next candidate); "
        "if no tier matches it does nothing. Summarize that decision process as an "
        "ORDERED list of tiers, in evaluation order. Each tier: one short plain-English "
        "line of the form 'when <condition> → <accept/skip>'. Describe conditions in "
        "biologist terms (gap distance near/far, both ends are tips, tip meets the side "
        "of a cable = T-junction, cable-thickness match/mismatch, straight continuation), "
        "with NO parameter names, code symbols, or specific numbers. Return ONLY a JSON "
        "list of strings (the tiers in order), nothing else.\n\n"
        f"{body}"
    )
    try:
        text = _llm_text(prompt, model)
        cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
        mm = re.search(r"\[.*\]", cleaned, re.S)
        if not mm:
            raise ValueError(f"no JSON list in reply ({len(text)} chars)")
        tiers = json.loads(mm.group(0))
        summary["hierarchy"] = [str(t).strip() for t in tiers if str(t).strip()]
        print(f"[plot_policy_evolution] decision hierarchy: {len(summary['hierarchy'])} tiers")
    except Exception as e:
        summary["hierarchy"] = []
        print(f"[plot_policy_evolution] decision hierarchy unavailable ({e})")


def make_figure(summary: dict, out_path: Path, wrap: int = 66) -> Path:
    """Compact 3-column table: generation | change text (auto-wrapped) | score.

    The ``change`` column shows the COMPLETE summary (never truncated); long text is
    wrapped at ``wrap`` characters and each row is given as many text lines as it
    needs, so the figure height grows with the wrapped content and nothing overlaps.
    The score sits just to the RIGHT of the change column (not pushed to the far
    edge), so the table reads tight.
    """
    import textwrap

    rows = summary["rows"]
    hierarchy = summary.get("hierarchy") or []
    # Pre-wrap each change cell; row height (in text lines) = its wrapped line count.
    wrapped = [textwrap.wrap(r["summary"], width=wrap) or [""] for r in rows]
    total_lines = sum(len(w) for w in wrapped)
    # Pre-wrap the decision-hierarchy tiers too (rendered as a panel below the table).
    hwrapped = [textwrap.wrap(f"{i+1}. {t}", width=wrap + 8) or [""]
                for i, t in enumerate(hierarchy)]
    h_lines = sum(len(w) for w in hwrapped)

    LINE = 0.24                       # inches per text line
    table_units = total_lines + len(rows) * 0.5
    hier_units = (h_lines + len(hierarchy) * 0.3 + 1.5) if hierarchy else 0.0
    fig_h = 0.9 + LINE * (table_units + hier_units + 1.5)
    X_GEN, X_CHANGE, X_SCORE = 0.0, 0.045, 0.83
    fig, ax = plt.subplots(figsize=(8.2, fig_h))
    ax.axis("off")

    ax.set_title(f"Policy evolution — {summary['run']}   "
                 f"(score {summary['baseline_score']} → {summary['final_score']}, "
                 f"{summary['n_accepted']} accepted)", fontsize=11, loc="left")

    total_units = table_units + hier_units
    y = total_units
    # --- evolution table ---
    ax.text(X_GEN, y + 0.7, "gen", fontsize=9, fontweight="bold")
    ax.text(X_CHANGE, y + 0.7, "change", fontsize=9, fontweight="bold")
    ax.text(X_SCORE, y + 0.7, "score", fontsize=9, fontweight="bold")
    for r, lines in zip(rows, wrapped):
        block = len(lines)
        top = y
        ax.text(X_GEN, top, f"{r['gen']}", fontsize=9, fontweight="bold", va="top")
        ax.text(X_CHANGE, top, "\n".join(lines), fontsize=8, va="top", linespacing=1.2)
        gain = f" (+{r['gain']})" if (r["gain"] is not None and r["gain"] > 0) else ""
        ax.text(X_SCORE, top, f"{r['score']}{gain}", fontsize=8.5, va="top")
        y -= block + 0.5

    # --- final-policy decision hierarchy panel ---
    if hierarchy:
        y -= 0.8
        ax.axhline(y + 0.4, xmin=0.0, xmax=1.0, color="#bbb", lw=0.8)
        ax.text(X_GEN, y, "Final policy — decision process (first matching tier wins, "
                "top → bottom):", fontsize=9.5, fontweight="bold", va="top")
        y -= 1.1
        for lines in hwrapped:
            ax.text(X_CHANGE, y, "\n".join(lines), fontsize=8, va="top",
                    color="#1a4f8a", linespacing=1.2)
            y -= len(lines) + 0.3

    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_units + 1.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def to_markdown(summary: dict) -> str:
    lines = [f"# Policy evolution — {summary['run']}",
             f"- {summary['n_accepted']} accepted / {summary['n_generations']} generations",
             f"- split-repair score {summary['baseline_score']} → {summary['final_score']}",
             "",
             "| gen | change | score |",
             "|---|---|---|"]
    for r in summary["rows"]:
        gain = f" (+{r['gain']})" if (r["gain"] is not None and r["gain"] > 0) else ""
        lines.append(f"| {r['gen']} | {r['summary'].replace('|', '/')} | {r['score']}{gain} |")
    hierarchy = summary.get("hierarchy") or []
    if hierarchy:
        lines += ["", "## Final policy — decision process (first matching tier wins)", ""]
        lines += [f"{i+1}. {t}" for i, t in enumerate(hierarchy)]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_name", help="run dir name under proofreader_evolve/runs/ (or a full path)")
    p.add_argument("--out", default=None, help="output PNG (default: runs/<run>/policy_evolution.png)")
    p.add_argument("--md", default=None, help="also write a markdown summary to this path")
    args = p.parse_args(argv)

    run_dir = _resolve_run_dir(args.run_name)
    summary = collect(run_dir)
    if not summary["rows"]:
        print(f"[plot_policy_evolution] no accepted generations in {run_dir.name}")
        return 1
    summarize_with_llm(summary)
    summarize_decision_hierarchy(summary, run_dir)

    out_path = Path(args.out) if args.out else (run_dir / "policy_evolution.png")
    make_figure(summary, out_path)
    print(f"[plot_policy_evolution] {summary['n_accepted']} accepted levers, "
          f"score {summary['baseline_score']} -> {summary['final_score']}")
    print(f"[plot_policy_evolution] figure -> {out_path}")
    for r in summary["rows"]:
        g = f" (+{r['gain']})" if (r["gain"] is not None and r["gain"] > 0) else ""
        print(f"  gen{r['gen']:>2}: {r['summary']}  -> score {r['score']}{g}")
    if args.md:
        Path(args.md).write_text(to_markdown(summary))
        print(f"[plot_policy_evolution] markdown -> {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

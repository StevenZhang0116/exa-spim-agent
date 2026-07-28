"""
Cost / effort ledger for the evolution loop.

The reviewer explicitly asked us to "measure how much compute/time/human
feedback was required." This records, per generation, the budget actually spent
so the final report can state the cost of each unit of accuracy gained.

One JSONL row per generation; cheap, append-only, resumable.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field


@dataclass
class GenerationCost:
    """Effort + outcome for one generation of the loop."""

    generation: int
    wall_seconds: float = 0.0            # total wall-clock for the generation
    eval_seconds: float = 0.0            # time inside evaluate() (the scorer)
    input_tokens: int = 0                # UNCACHED agent input tokens (ResultMessage.usage)
    output_tokens: int = 0               # agent output tokens (ResultMessage.usage)
    # Cache tokens from ResultMessage.usage. In the shared-session design almost ALL
    # input is served from cache, so ``input_tokens`` alone drastically understates the
    # input the model processed; the TRUE input is input_tokens + cache_read_input_tokens
    # + cache_creation_input_tokens. Cheap ($) but real, and the thing that grows with
    # generation count — so track it explicitly to see the shared-session cost curve.
    cache_read_input_tokens: int = 0     # input served from the prompt cache
    cache_creation_input_tokens: int = 0 # input written INTO the cache this turn
    cost_usd: float = 0.0                # agent $ (from ResultMessage if present)
    n_evaluations: int = 0               # how many evaluate() calls this gen
    human_interventions: int = 0         # # of human approvals/edits this gen
    # NOTE on naming: Edge Accuracy is a DIAGNOSTIC here, NOT the gate. The gate is
    # the split-repair score (see ``heldout_split_repair_score`` below and
    # ``parent_split_repair_score``). These two Edge-Accuracy fields are recorded for
    # observation only — they do not decide accept/reject.
    train_edge_accuracy: float = float("nan")    # train Edge Accuracy after revision
                                         # (diagnostic; can DROP as recall rises)
    heldout_edge_accuracy: float = float("nan")  # held-out Edge Accuracy (diagnostic;
                                         # NOT the gate, despite the legacy "primary")
    # Held-out STANDARD benchmark metrics (run-length-weighted, pooled) — DIAGNOSTIC
    # ONLY, never gate inputs (the gate is the penalized split-repair fitness). Recorded
    # in ABSOLUTE terms so the performance figure can plot Edge Accuracy and the merge
    # burden (% Merged Edges / # Merges) across generations. NaN on import/lint-failed
    # gens (no held-out scoring ran).
    heldout_merged_edges: float = float("nan")   # held-out % Merged Edges
    heldout_merges: float = float("nan")         # held-out # Merges (count)
    heldout_split_edges: float = float("nan")    # held-out % Split Edges
    # The parent policy's held-out RAW split-repair score (correct - false), recorded
    # for the plots. NOTE: this is NOT the gate bar anymore — the gate compares
    # PENALIZED fitness (see ``parent_fitness`` / ``heldout_fitness`` below). Kept as a
    # raw diagnostic and for backward-compatible plotting.
    parent_split_repair_score: float = float("nan")
    accepted: bool = False               # was the revision kept?
    note: str = ""
    # --- held-out edit activity (what the policy actually did on the gated set) --
    heldout_n_edits: int = 0             # total edits the policy emitted on held-out
    heldout_correct_merges: int = 0      # merges joining the SAME held-out neuron
    heldout_false_merges: int = 0        # merges fusing DIFFERENT held-out neurons (TOTAL)
    # Blame-attributed split of heldout_false_merges. Only false_policy is penalized by
    # the gate: a GENUINELY NEW fusion of two clean fragments. false_preexisting = a
    # fragment that ALREADY spanned both neurons before the merge (a pre-existing
    # segmentation/GT merge the policy merely joined onto — not its fault). See
    # incremental_scoring.classify_merge_edits. heldout_false_merges ==
    # false_policy + false_preexisting.
    heldout_false_merges_policy: int = 0       # policy-caused (penalized)
    heldout_false_merges_preexisting: int = 0  # pre-existing (NOT penalized)
    heldout_split_repair_score: int = 0  # correct - false_policy (pre-penalty; only the
                                         # policy-caused false merges are subtracted)
    # --- penalized FITNESS (the actual gate decision variable) -------------------
    # fitness = heldout_split_repair_score - merge_penalty * heldout_false_merges.
    # This REPLACED the hard 'false == 0' gate: a candidate is kept iff its fitness
    # beats the parent's by >= score_margin. Recorded so the accept/reject decision
    # is reconstructable from the ledger alone.
    merge_penalty: float = 100.0         # per-false-merge penalty in effect this gen
    heldout_fitness: float = 0.0         # candidate's penalized fitness on held-out
    parent_fitness: float = 0.0          # the parent fitness bar this gen had to beat
    # --- BLIND SPOT: merges the gate could NOT verify ---------------------------
    # A merge is "unscored" when neither endpoint's dominant neuron is in the
    # held-out GT, so the gate neither rewards nor penalizes it. High unscored ⇒ most
    # edits land in regions with no GT coverage and carry NO correctness guarantee
    # (deployable, but unverified). This is a coverage diagnostic, never a gate input.
    heldout_unscored_merges: int = 0     # merges with no held-out-GT verdict
    heldout_unscored_fraction: float = float("nan")  # unscored / total held-out edits
    # --- ① CONFIDENCE: GT-INDEPENDENT image verdict on the blind spot -------------
    # A coarse confidence level for this generation's accept/reject decision, plus
    # the image (bridge_ratio) verdict it is built from. Replayed from reads the
    # policy already made (zero extra cloud reads); advisory unless --confidence-veto.
    # gt_fraction = GT-verified edits / (GT-verified + unscored): how much of the
    # decision rests on GT vs. the blind spot. soft_* count the unscored merges the
    # image signal calls likely-correct / likely-false / ambiguous; the rest were
    # never imaged (unknown). NEVER an input to the hard fitness.
    decision_confidence: str = "n/a"     # "high" | "medium" | "low" | "n/a"
    heldout_gt_fraction: float = float("nan")
    heldout_soft_correct: int = 0        # blind-spot merges image calls likely-correct
    heldout_soft_false: int = 0          # blind-spot merges image calls likely-FALSE
    heldout_soft_ambiguous: int = 0      # blind-spot merges with ambiguous image signal
    confidence_veto: bool = False        # was the opt-in low-confidence veto active?
    # --- POLICY EXECUTION COST + BUDGET (options 1 & 2) --------------------------
    # propose_edits' OWN wall-clock on held-out (separate from scoring), and whether
    # it blew the time budget. A quadratic per-site feature over a whole-brain
    # candidate stream can run for an hour; timing makes that attributable to the
    # policy (not the scorer), and a timeout REJECTS the generation (heldout_policy_
    # timed_out True => note is a BUDGET-REJECT). 0.0 / False when no held-out policy
    # pass ran (import/lint failed first).
    heldout_policy_seconds: float = 0.0
    heldout_policy_timed_out: bool = False
    policy_time_budget: float = 0.0    # the budget in effect this run (seconds)
    # --- TRAIN-SIDE over-merge alert (diagnostic, NOT a gate input) --------------
    # Merges that fused two DIFFERENT neurons on a TRAIN skeleton. The gate judges
    # false merges on HELD-OUT only (kept isolated for an honest generalization
    # signal), so these never reject a candidate — but they are real over-merges the
    # gate is blind to (partners in train / no-GT regions). Recorded so train-side
    # precision regressions are visible per generation. This is the POLICY-CAUSED count
    # (blame-attributed the same way the gate is — see classify_merge_edits).
    train_false_merges: int = 0
    # Train-side NEUTRAL false merges: a fragment already spanned both neurons before
    # the edit (pre-existing merge). Not the policy's fault — recorded (not gated) so
    # post-analysis can extract/visualize the train-side neutral count symmetrically
    # with heldout_false_merges_preexisting.
    train_false_merges_preexisting: int = 0
    # --- run mode (split-error-only fast mode) ----------------------------------
    splits_only: bool = False            # True => merge-error repair disabled this
                                         # run (no MergeSite enumerated; split_label
                                         # edits dropped before scoring)
    heldout_split_label_dropped: int = 0 # split_label edits the policy emitted that
                                         # were discarded by splits_only (should be 0)
    # --- run mode (two-phase split-then-merge) -----------------------------------
    two_phase: bool = False              # True => the policy ran twice (merge-repair
                                         # pass -> re-enumerate -> split-repair pass)
    # Held-out MERGE-REPAIR (split_label) credit, symmetric with the split-repair
    # merge counts. Populated only in two-phase runs (0 otherwise). correct = fused
    # segments correctly cut; false = clean single-neuron segments wrongly cut (a
    # manufactured split error, penalized like a false merge in the fitness).
    heldout_mrepair_correct: int = 0
    heldout_mrepair_false: int = 0
    heldout_mrepair_unscored: int = 0
    # Did the reviser actually READ the discovery priors (all-runs.combined.md) this
    # generation? Verified from the tool-call stream (a Read of the priors file), not
    # assumed. True = read it; False = priors were available but not read; None = no
    # priors file was configured for this run.
    read_priors: "bool | None" = None
    # GROUNDED / citation tracking. NO LONGER POPULATED by run_evolution.py: the loop
    # was rewound to the earlier design (reviser reads the discovery file itself, no
    # inlined ranked menu and no grounded-vs-exploratory feedback loop), so these keep
    # their defaults (None / "") in new runs. They remain on the schema so ledgers from
    # the menu-era still parse; a re-enable would set them again. When it WAS populated:
    # grounded=True meant the reviser cited a finding whose content shaped the edit,
    # cited_findings listed the bare "#N" mentions, and supported_findings was the
    # subset whose content actually appeared in the edit (the stronger signal).
    grounded: "bool | None" = None
    cited_findings: str = ""
    supported_findings: str = ""
    # --- traceability (A): what the reviser actually did this generation --------
    candidate_path: str = ""             # gen<NN>/heuristics.candidate.py (always saved)
    heuristics_diffstat: str = ""        # "+A -B" lines changed vs the parent policy
    diagnosis: str = ""                  # the reviser's explanation text (truncated)

    def to_json(self) -> dict:
        return asdict(self)


class Ledger:
    """Append-only JSONL ledger at runs/<run>/ledger.jsonl."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def record(self, cost: GenerationCost) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(cost.to_json()) + "\n")

    def read_all(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def attempts_timeline(self) -> str:
        """Render the human-readable per-generation attempts timeline from the ledger.

        This is the on-demand replacement for the old, separately-maintained
        ``attempts.md`` file: the ledger is the single source of truth, and this
        reconstructs the same one-line-per-generation view from it (nothing is written
        during the run). Byte-compatible with the historical format:

          - gen07 [vs parent fitness 31]: fitness 31 (+0; score=31, correct=31,
            false=0) -> reverted (did not beat parent); +24 -1 lines; <diagnosis…>
        """
        return render_attempts_timeline(self.read_all())

    def summarize(self) -> str:
        rows = self.read_all()
        if not rows:
            return "no generations recorded yet."
        accepted = [r for r in rows if r["accepted"]]
        total_s = sum(r["wall_seconds"] for r in rows)
        # cost_usd (and, in the shared-session design, the ResultMessage.usage token
        # counts) may be SESSION-CUMULATIVE (monotonic across gens) rather than per-gen.
        # Disambiguate from the data: a non-decreasing series is already cumulative, so
        # take its LAST value; otherwise it is per-gen, so SUM. This handles both the
        # shared-session ledgers (cumulative) and the intervening per-gen-session ones.
        def _total(key):
            vals = [r.get(key, 0) or 0 for r in rows]
            if not vals:
                return 0
            return vals[-1] if all(b >= a for a, b in zip(vals, vals[1:])) else sum(vals)
        total_out = _total("output_tokens")
        total_in = _total("input_tokens")
        total_cache_read = _total("cache_read_input_tokens")
        total_cache_creation = _total("cache_creation_input_tokens")
        total_cost = _total("cost_usd")
        total_human = sum(r["human_interventions"] for r in rows)
        # Read the current key; fall back to the legacy "heldout_primary" so old
        # ledgers still summarize.
        def _hea(r):
            return r.get("heldout_edge_accuracy", r.get("heldout_primary", float("nan")))
        best = max((_hea(r) for r in rows if _hea(r) == _hea(r)), default=float("nan"))
        total_input_all = total_in + total_cache_read + total_cache_creation
        return (
            f"{len(rows)} generations, {len(accepted)} accepted. "
            f"best held-out Edge Accuracy={best:.4f}. "
            f"cost: {total_s:.0f}s wall, "
            f"in={total_input_all} tok (uncached {total_in} + cache_read "
            f"{total_cache_read} + cache_creation {total_cache_creation}), "
            f"out={total_out} tok, "
            f"${total_cost:.4f}, {total_human} human interventions."
        )


def _fmt_num(x) -> str:
    """Format a number like the ``{:g}`` used in the original attempts line."""
    try:
        return f"{float(x):g}"
    except (TypeError, ValueError):
        return str(x)


def render_attempts_timeline(rows: list[dict]) -> str:
    """Reconstruct the attempts timeline (one line per generation) from ledger rows.

    Single source of truth = ledger.jsonl. Each field maps directly:
      parent_fitness -> "vs parent fitness"; heldout_fitness -> "fitness";
      (heldout_fitness - parent_fitness) -> the +/- gain; heldout_split_repair_score,
      heldout_correct_merges, heldout_false_merges -> score/correct/false; note ->
      the accept/revert verdict; heuristics_diffstat + diagnosis(1st line, 120 chars)
      -> the trailing one-line summary. Rows are sorted by generation.
    """
    out = []
    for r in sorted(rows, key=lambda r: r.get("generation", 0)):
        parent = r.get("parent_fitness", 0.0)
        fit = r.get("heldout_fitness", 0.0)
        try:
            gain = float(fit) - float(parent)
            gain_s = f"{gain:+g}"
        except (TypeError, ValueError):
            gain_s = "?"
        # Rebuild the trailing summary EXACTLY as the loop did (byte-compatible with
        # the historical attempts.md): "<diffstat> lines; <first line of diagnosis,
        # 120 chars>". The ledger stores the raw (un-stripped) diagnosis, so applying
        # the same strip -> first-line -> [:120] here reproduces the old string.
        diffstat = r.get("heuristics_diffstat", "?")
        diag = (r.get("diagnosis", "") or "").strip().split("\n", 1)[0][:120]
        summary = f"{diffstat} lines; {diag}"
        out.append(
            f"- gen{int(r.get('generation', 0)):02d} "
            f"[vs parent fitness {_fmt_num(parent)}]: "
            f"fitness {_fmt_num(fit)} "
            f"({gain_s}; score={r.get('heldout_split_repair_score', '?')}, "
            f"correct={r.get('heldout_correct_merges', '?')}, "
            f"false={r.get('heldout_false_merges', '?')}) "
            f"-> {r.get('note', '')}; {summary}"
        )
    return "\n".join(out)


if __name__ == "__main__":
    # Render the attempts timeline from a run's ledger, on demand (replaces the old
    # inline attempts.md). Usage:
    #   python -m proofreader_evolve.harness.ledger <run_dir | ledger.jsonl>
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("usage: ledger.py <run_dir | path/to/ledger.jsonl>")
    arg = sys.argv[1]
    path = arg if arg.endswith(".jsonl") else os.path.join(arg, "ledger.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"no ledger at {path}")
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(render_attempts_timeline(rows))

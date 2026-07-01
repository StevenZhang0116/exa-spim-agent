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
    input_tokens: int = 0                # agent input tokens (from ResultMessage)
    output_tokens: int = 0               # agent output tokens
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
    heldout_false_merges: int = 0        # merges fusing DIFFERENT held-out neurons
    heldout_split_repair_score: int = 0  # correct - false (raw, pre-penalty)
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
    # --- TRAIN-SIDE over-merge alert (diagnostic, NOT a gate input) --------------
    # Merges that fused two DIFFERENT neurons on a TRAIN skeleton. The gate judges
    # false merges on HELD-OUT only (kept isolated for an honest generalization
    # signal), so these never reject a candidate — but they are real over-merges the
    # gate is blind to (partners in train / no-GT regions). Recorded so train-side
    # precision regressions are visible per generation.
    train_false_merges: int = 0
    # --- run mode (split-error-only fast mode) ----------------------------------
    splits_only: bool = False            # True => merge-error repair disabled this
                                         # run (no MergeSite enumerated; split_label
                                         # edits dropped before scoring)
    heldout_split_label_dropped: int = 0 # split_label edits the policy emitted that
                                         # were discarded by splits_only (should be 0)
    # Did the reviser actually READ the discovery priors (all-runs.combined.md) this
    # generation? Verified from the tool-call stream (a Read of the priors file), not
    # assumed. True = read it; False = priors were available but not read; None = no
    # priors file was configured for this run.
    read_priors: "bool | None" = None
    # Was this generation GROUNDED — did the reviser CITE a discovery finding number
    # in the rules.md change log (a prior actually shaped the edit)? Strictly stronger
    # than read_priors. True = cited >=1 finding; False = priors inlined but none
    # cited (exploratory); None = no priors configured. Grouping accepted gens by this
    # lets us measure whether grounding generalizes better than free exploration.
    grounded: "bool | None" = None
    cited_findings: str = ""             # comma-joined finding numbers cited, if any
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

    def summarize(self) -> str:
        rows = self.read_all()
        if not rows:
            return "no generations recorded yet."
        accepted = [r for r in rows if r["accepted"]]
        total_s = sum(r["wall_seconds"] for r in rows)
        total_out = sum(r["output_tokens"] for r in rows)
        # cost_usd is the running SESSION total (cumulative), so the run total is the
        # LAST/MAX value, NOT a sum over generations.
        total_cost = max((r.get("cost_usd", 0.0) for r in rows), default=0.0)
        total_human = sum(r["human_interventions"] for r in rows)
        # Read the current key; fall back to the legacy "heldout_primary" so old
        # ledgers still summarize.
        def _hea(r):
            return r.get("heldout_edge_accuracy", r.get("heldout_primary", float("nan")))
        best = max((_hea(r) for r in rows if _hea(r) == _hea(r)), default=float("nan"))
        return (
            f"{len(rows)} generations, {len(accepted)} accepted. "
            f"best held-out Edge Accuracy={best:.4f}. "
            f"cost: {total_s:.0f}s wall, {total_out} out-tokens, "
            f"${total_cost:.4f}, {total_human} human interventions."
        )

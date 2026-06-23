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
    # The gate threshold this generation had to BEAT: the parent policy's held-out
    # split-repair score (correct - false), an INTEGER — NOT an Edge Accuracy. A
    # candidate is kept iff its heldout_split_repair_score > this AND false == 0.
    parent_split_repair_score: float = float("nan")
    accepted: bool = False               # was the revision kept?
    note: str = ""
    # --- held-out edit activity (what the policy actually did on the gated set) --
    heldout_n_edits: int = 0             # total edits the policy emitted on held-out
    heldout_correct_merges: int = 0      # merges joining the SAME held-out neuron
    heldout_false_merges: int = 0        # merges fusing DIFFERENT held-out neurons
    heldout_split_repair_score: int = 0  # correct - false (the gate's primary signal)
    # --- run mode (split-error-only fast mode) ----------------------------------
    splits_only: bool = False            # True => merge-error repair disabled this
                                         # run (no MergeSite enumerated; split_label
                                         # edits dropped before scoring)
    heldout_split_label_dropped: int = 0 # split_label edits the policy emitted that
                                         # were discarded by splits_only (should be 0)
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

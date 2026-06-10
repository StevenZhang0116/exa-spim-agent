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
    train_primary: float = float("nan")  # train Edge Accuracy after revision
    heldout_primary: float = float("nan")# held-out Edge Accuracy (the gate)
    accepted: bool = False               # was the revision kept?
    note: str = ""

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
        total_cost = sum(r["cost_usd"] for r in rows)
        total_human = sum(r["human_interventions"] for r in rows)
        best = max((r["heldout_primary"] for r in rows
                    if r["heldout_primary"] == r["heldout_primary"]), default=float("nan"))
        return (
            f"{len(rows)} generations, {len(accepted)} accepted. "
            f"best held-out Edge Accuracy={best:.4f}. "
            f"cost: {total_s:.0f}s wall, {total_out} out-tokens, "
            f"${total_cost:.4f}, {total_human} human interventions."
        )

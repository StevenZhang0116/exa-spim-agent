"""Adaptive allocation of generations to scorer kinds (merge / split).

Host-side only. Inputs are the accepted parent's development-validation cells
(positives, K, precision), this run's ledger rows and the candidate pool's
per-kind progress. The output is the order in which kinds are offered to
``CandidatePool.next_plan`` plus a record for the ledger and trajectory. No
number computed here is written to an agent-visible file; the generation plan
only carries the rule name.

Rules, in priority order:

1. ``pending_followup``: a kind with a pending promotion follow-up or a pending
   investigation keeps its slot (unchanged semantics from the per-kind scheduler).
2. ``floor``: a kind not revised in the last ``floor_every - 1`` completed
   generations is revised now, so every active kind appears at least once in any
   window of ``floor_every`` generations.
3. ``momentum``: among unsaturated kinds, the largest mean gate-passing
   validation gain over the kind's last ``MOMENTUM_WINDOW`` scheduled
   generations; rejected or failed generations count as zero.
4. ``headroom``: the largest equal-brain mean of ``min(positives, K) / K -
   precision`` over the validation cells of that kind.

A kind whose headroom is below ``SATURATION_HITS / K`` (half a Top-K hit) is
saturated: rules 3 and 4 skip it and it is revised only through rules 1 and 2.
"""

import math

SCHEDULE_VERSION = "adaptive-kind-allocation-v1"
MODES = ("adaptive", "alternate")
MOMENTUM_WINDOW = 2
SATURATION_HITS = .5
DEFAULT_FLOOR_EVERY = 4


def headroom(cells, budgets, kinds):
    """Equal-brain mean remaining precision per kind on the fixed validation pools."""
    per_kind = {kind: [] for kind in kinds}
    for name, cell in cells.items():
        kind = name.rsplit("/", 1)[-1]
        if kind in per_kind:
            k = int(cell["requested_k"])
            if k < 1:
                raise ValueError("Requested K must be positive")
            cap = min(int(cell["positives"]), k) / k
            per_kind[kind].append(cap - float(cell["precision"]))
    return {kind: (math.fsum(values) / len(values) if values else 0.) for kind, values in per_kind.items()}


def momentum(records, kinds, window=MOMENTUM_WINDOW):
    """Mean gate-passing validation gain over each kind's last `window` generations."""
    result = {}
    for kind in kinds:
        rows = [row for row in records if row.get("target_kind") == kind][-window:]
        gains = []
        for row in rows:
            gate = row.get("validation_gate") or {}
            gain = 0.
            if (row.get("accepted") and gate.get("candidate_mean") is not None
                    and gate.get("parent_mean") is not None):
                gain = max(0., float(gate["candidate_mean"]) - float(gate["parent_mean"]))
            gains.append(gain)
        result[kind] = math.fsum(gains) / len(gains) if gains else 0.
    return result


def generations_since(records, kinds):
    """Completed generations since each kind was last scheduled (all of them if never)."""
    result = {}
    for kind in kinds:
        since = 0
        for row in reversed(records):
            if row.get("target_kind") == kind:
                break
            since += 1
        result[kind] = since
    return result


def allocate(kinds, *, cells, budgets, records, paused, pending, floor_every=DEFAULT_FLOOR_EVERY):
    """Order the kinds for the next generation and explain the choice."""
    if type(floor_every) is not int or floor_every < 1:
        raise ValueError("floor_every must be a positive integer")
    kinds = list(kinds)
    room = headroom(cells, budgets, kinds)
    gain = momentum(records, kinds)
    since = generations_since(records, kinds)
    saturated = {kind: room[kind] < SATURATION_HITS / budgets[kind] for kind in kinds}
    active = [kind for kind in kinds if not paused.get(kind)]

    def by_headroom(candidates):
        return sorted(candidates, key=lambda kind: (-room[kind], kinds.index(kind)))

    pending_kinds = [kind for kind in active if pending.get(kind)]
    starved = [kind for kind in active if since[kind] >= floor_every - 1 and kind not in pending_kinds]
    moving = [kind for kind in active if gain[kind] > 0. and not saturated[kind]
              and kind not in pending_kinds and kind not in starved]
    if not active:
        lead, rule = None, "all_paused"
    elif pending_kinds:
        lead, rule = pending_kinds[0], "pending_followup"
    elif starved:
        lead, rule = max(starved, key=lambda kind: (since[kind], room[kind], -kinds.index(kind))), "floor"
    elif moving:
        lead, rule = max(moving, key=lambda kind: (gain[kind], room[kind], -kinds.index(kind))), "momentum"
    else:
        unsaturated = [kind for kind in active if not saturated[kind]]
        lead, rule = by_headroom(unsaturated or active)[0], "headroom"
    order = ([lead] if lead is not None else []) + [kind for kind in by_headroom(active) if kind != lead]
    return {"version": SCHEDULE_VERSION, "mode": "adaptive", "rule": rule, "chosen": lead, "order": order,
            "floor_every": floor_every, "momentum_window": MOMENTUM_WINDOW, "saturation_hits": SATURATION_HITS,
            "kinds": {kind: {"headroom": room[kind], "momentum": gain[kind], "generations_since": since[kind],
                             "saturated": saturated[kind], "paused": bool(paused.get(kind)),
                             "pending": bool(pending.get(kind))} for kind in kinds},
            "scope": "Host-side allocation over development-validation cells and ledger rows; "
                     "only the rule name reaches the generation plan."}

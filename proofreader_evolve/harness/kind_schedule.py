"""Adaptive allocation of generations to scorer kinds (merge / split).

Host-side only. Inputs are the accepted parent's development-validation cells
(positives, K, precision), this run's ledger rows and the candidate pool's
per-kind progress. The output is the order in which kinds are offered to
``CandidatePool.next_plan`` plus a record for the ledger and trajectory. No
number computed here is written to an agent-visible file; the generation plan
only carries the rule name.

Rules, in priority order:

0. ``initial_exploration``: give each active kind one generation before returning
   to a previously scheduled kind. Failed generations still count as an opportunity.
1. ``floor``: a kind not revised in the last ``floor_every - 1`` completed
   generations is revised now, so every active kind appears at least once in any
   window of ``floor_every`` generations.
2. ``pending_followup``: a pending promotion follow-up or investigation takes the
   next available slot. Initial exploration or a floor allocation defers it intact.
3. ``momentum``: among unsaturated kinds, the largest mean validation point
   gain (candidate minus parent, clipped at zero) over the kind's last
   ``MOMENTUM_WINDOW`` scheduled generations, whether or not the promotion gate
   accepted the candidate; a failed generation without a validation measurement
   counts as zero. Until 2026-11-01 only gate-passing gains counted, which starved
   the kind with the coarser gate: merge needs about ten times the gain that
   split needs to pass the bootstrap, so its rejected +0.02 gains counted as
   nothing while split's accepted +0.006 kept it scheduled at the detector level.
4. ``headroom``: the largest remaining objective under the configured per-kind
   aggregation: remaining precision using each cell's effective K, or 1 minus the
   kind's AUROC / partial AUROC / average precision when that is the objective (v5,
   2026-11-02). Cells flagged ``in_mean: false`` (below the GT-positive floor) are
   skipped, as in the promotion objective.

A kind is saturated when its headroom is below half the smallest resolvable step of
its objective: one hit under its aggregation and effective K for precision, one
positive passing one negative (1 / sum over cells of positives x negatives) for the
rank statistics, which in practice never trips. Rules 3 and 4 skip saturated kinds;
rules 1 and 2 can still revise them. Momentum reads the gate's objective gain
(``candidate_objective`` minus ``parent_objective``), falling back to the precision
fields of ledgers written before v7.
"""

import math

SCHEDULE_VERSION = "adaptive-kind-allocation-v7-initial-exploration"
MODES = ("adaptive", "alternate")
MOMENTUM_WINDOW = 2
SATURATION_HITS = .5
DEFAULT_FLOOR_EVERY = 4


def _counted(cells, kinds):
    """Cells of the requested kinds that count toward the objective, keyed by kind."""
    per_kind = {kind: [] for kind in kinds}
    for name, cell in cells.items():
        kind = name.rsplit("/", 1)[-1]
        if kind in per_kind and cell.get("in_mean", True):
            k = int(cell["requested_k"])
            if k < 1:
                raise ValueError("Requested K must be positive")
            per_kind[kind].append(cell)
    return per_kind


def _objective(budgets):
    return getattr(budgets, "objective", "precision_at_k") or "precision_at_k"


def headroom(cells, budgets, kinds):
    """Remaining objective per kind on the counted validation pools, under the budgets'
    aggregation: for precision the equal-brain mean or pooled hits / effective K when
    positive-weighted; for a rank statistic 1 minus its positive-weighted (or equal) mean."""
    pooled = getattr(budgets, "aggregation", "equal-brain") == "positive-weighted"
    objective = _objective(budgets)
    result = {}
    for kind, counted in _counted(cells, kinds).items():
        if objective != "precision_at_k":
            values = [(float(cell[objective]), int(cell["positives"])) for cell in counted
                      if cell.get(objective) is not None]
            if not values:
                result[kind] = 0.
            elif pooled:
                total = sum(weight for _, weight in values)
                result[kind] = 1. - (sum(value * weight for value, weight in values) / total if total else 0.)
            else:
                result[kind] = 1. - math.fsum(value for value, _ in values) / len(values)
            continue
        if pooled and counted:
            total_k = sum(int(cell.get('effective_k', cell['requested_k'])) for cell in counted)
            attainable = sum(min(int(cell["positives"]), int(cell.get('effective_k', cell['requested_k']))) for cell in counted)
            result[kind] = (attainable - sum(int(cell["tp"]) for cell in counted)) / total_k
            continue
        values = [min(int(cell["positives"]), int(cell.get('effective_k', cell['requested_k'])))
                  / int(cell.get('effective_k', cell['requested_k']))
                  - float(cell["precision"]) for cell in counted]
        result[kind] = math.fsum(values) / len(values) if values else 0.
    return result


def saturation_thresholds(cells, budgets, kinds):
    """Half the smallest one-hit change in each kind's objective.

    Pooled precision changes by 1 / sum(effective K) per hit. Equal-brain
    precision changes by 1 / (number of counted brains * that brain's K),
    so its smallest step uses the largest K. No counted cells means no
    evidence of saturation; a zero threshold keeps that kind unsaturated.
    """
    pooled = getattr(budgets, "aggregation", "equal-brain") == "positive-weighted"
    objective = _objective(budgets)
    result = {}
    for kind, counted in _counted(cells, kinds).items():
        if not counted:
            result[kind] = 0.
            continue
        if objective != "precision_at_k":
            pairs = sum(int(cell["positives"]) * max(0, int(cell.get("n_labeled", cell["positives"])) - int(cell["positives"]))
                        for cell in counted)
            result[kind] = SATURATION_HITS / pairs if pairs else 0.
            continue
        ks = [int(cell.get('effective_k', cell['requested_k'])) for cell in counted]
        denominator = sum(ks) if pooled else len(ks) * max(ks)
        result[kind] = SATURATION_HITS / denominator
    return result


def momentum(records, kinds, window=MOMENTUM_WINDOW):
    """Mean validation point gain of the objective (clipped at zero) over each kind's last
    `window` generations, accepted or rejected; a generation without a gate measurement counts
    as zero. Ledgers before v7 carry precision fields only."""
    result = {}
    for kind in kinds:
        rows = [row for row in records if row.get("target_kind") == kind][-window:]
        gains = []
        for row in rows:
            gate = row.get("validation_gate") or {}
            gain = 0.
            after, before = gate.get("candidate_objective"), gate.get("parent_objective")
            if after is None or before is None:
                after, before = gate.get("candidate_precision"), gate.get("parent_precision")
            if gate.get("target_kind") == kind and after is not None and before is not None:
                gain = max(0., float(after) - float(before))
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
    thresholds = saturation_thresholds(cells, budgets, kinds)
    saturated = {kind: room[kind] < thresholds[kind] for kind in kinds}
    active = [kind for kind in kinds if not paused.get(kind)]

    def by_headroom(candidates):
        return sorted(candidates, key=lambda kind: (-room[kind], kinds.index(kind)))

    pending_kinds = [kind for kind in active if pending.get(kind)]
    scheduled = {row.get('target_kind') for row in records}
    untried = [kind for kind in active if kind not in scheduled]
    starved = [kind for kind in active if since[kind] >= floor_every - 1]
    moving = [kind for kind in active if gain[kind] > 0. and not saturated[kind]
              and kind not in pending_kinds and kind not in starved]
    if not active:
        lead, rule = None, "all_paused"
    elif untried:
        lead, rule = by_headroom(untried)[0], "initial_exploration"
    elif starved:
        lead, rule = max(starved, key=lambda kind: (since[kind], room[kind], -kinds.index(kind))), "floor"
    elif pending_kinds:
        lead, rule = pending_kinds[0], "pending_followup"
    elif moving:
        lead, rule = max(moving, key=lambda kind: (gain[kind], room[kind], -kinds.index(kind))), "momentum"
    else:
        unsaturated = [kind for kind in active if not saturated[kind]]
        lead, rule = by_headroom(unsaturated or active)[0], "headroom"
    order = ([lead] if lead is not None else []) + [kind for kind in by_headroom(active) if kind != lead]
    return {"version": SCHEDULE_VERSION, "mode": "adaptive", "rule": rule, "chosen": lead, "order": order,
            "floor_every": floor_every, "momentum_window": MOMENTUM_WINDOW, "saturation_hits": SATURATION_HITS,
            "kinds": {kind: {"headroom": room[kind], "momentum": gain[kind], "generations_since": since[kind],
                             "saturation_threshold": thresholds[kind],
                             "saturated": saturated[kind], "paused": bool(paused.get(kind)),
                             "pending": bool(pending.get(kind))} for kind in kinds},
            "scope": "Host-side allocation over development-validation cells and ledger rows; "
                     "only the rule name reaches the generation plan."}

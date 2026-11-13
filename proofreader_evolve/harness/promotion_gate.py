"""The promotion gate (``--promotion-gate rule``, the default since 2026-11-12).

A submitted candidate replaces the accepted scorer of its kind when, on the
development-validation pools,

* no guarded rank statistic is significantly worse (the guard rail: the paired-bootstrap
  interval of its delta lies entirely below zero), and
* the run's objective is better beyond resampling noise (the interval of its delta lies above
  zero), by more than a minimum effect (``--precision-margin``).

The host computes one table: every rank statistic of both scorers and the interval of each
delta from one paired Poisson(1) bootstrap over fragment groups
(``fixed_pool_scoring.resampling_groups``; the same weights, drawn in the same order, as
``fixed_pool_scoring.paired_bootstrap``, so the objective's interval is that function's). The
table, the guard rail and the rule go into the gate record. A rejected candidate's
``rejection_category`` is the only thing a later reviser session learns beyond "not promoted":
a word from a fixed list, derived from the intervals, never a number.

The validation brains remain development data: a gate that queries them every generation
selects on them. The interval is conditional on those brains and is not corrected for
repeated tries.

History. From 2026-11-10 to 2026-11-12 a language model voted on this table
(``--promotion-gate judge``; five votes, strict majority, inside the guard rail). In the two
runs of 2026-11-12 its fourteen voted decisions, thirteen of them unanimous, all equalled "the
pAUC interval lies above zero" while the declared objective, which TRAIN selection and the
reviser optimise, was average precision. The votes were replaced by the rule on the declared
objective and removed; a second table restricted to merge junctions, shown to the judge, never
changed a decision in eight merge submissions and was removed with it. The earlier
``bootstrap`` mode (the objective's interval and a point-estimate head guard) is covered by the
rule and the guard rail. The code is in the repository history (commit 17473f7).
"""
import math

import numpy as np

from . import fixed_pool_scoring as scoring
from .label_availability import labeled_rows
from .rank_statistics import ranking, ranking_statistics

VERSION = "promotion-gate-v4-rule-fragment-groups"
GATE_VERSION = "target-kind-validation-rule-v2"
MODE = "rule"
STATISTICS = ("auroc", "pauc", "average_precision", "precision_depth_mean",
              "precision", "precision_at_half_k", "precision_at_double_k")
# Single-cut precisions at K/2 and 2K enter the guard through their mean with precision at K.
GUARDED = ("auroc", "pauc", "average_precision", "precision_depth_mean", "precision")
# The only words about a rejection that reach the reviser (``rejection_category``).
REJECTION_CATEGORIES = ("significantly_worse", "within_noise", "gain_too_small")


def gate_modes():
    return (MODE, *scoring.GATE_MODES)


def gate_version(mode):
    return GATE_VERSION if mode == MODE else scoring.gate_version(mode)


# --- the table -------------------------------------------------------------------------------

def _kind_statistics(per_cell, pooled):
    """Kind statistics from the counted cells' ``ranking_statistics`` and (resampled) positive
    weights, under the rule of ``rank_statistics.aggregate_statistics``."""
    out = {}
    for name, key in (("precision_at_half_k", "half_k"), ("precision", "k"), ("precision_at_double_k", "double_k")):
        pairs = [stats[key] for stats, _ in per_cell]
        if pooled:
            depth = sum(d for _, d in pairs)
            out[name] = sum(h for h, _ in pairs) / depth if depth else None
        else:
            out[name] = math.fsum((h / d if d else 0.) for h, d in pairs) / len(pairs) if pairs else None
    depths = [out["precision_at_half_k"], out["precision"], out["precision_at_double_k"]]
    out["precision_depth_mean"] = math.fsum(depths) / 3 if all(v is not None for v in depths) else None
    for name in ("average_precision", "auroc", "pauc"):
        defined = [(stats[name], weight) for stats, weight in per_cell if stats[name] is not None]
        if not defined:
            out[name] = None
        elif pooled:
            total = sum(weight for _, weight in defined)
            out[name] = sum(value * weight for value, weight in defined) / total if total else None
        else:
            out[name] = math.fsum(value for value, _ in defined) / len(defined)
    return out


def intervals_from_rankings(cells, *, draws, seed, alpha=.05, pauc_fpr=scoring.DEFAULT_PAUC_FPR, pooled=True):
    """Both arms' statistics and the paired interval of every delta.

    ``cells`` maps a cell name to ``{"arms": (parent, candidate), "size", "k", "counted"}`` in
    pool order; an arm is ``(positions, labels, starts)``: each ranked row's index among the
    cell's ``size`` labeled rows, its label, and the tie-group starts. One Poisson(1) weight per
    fragment group (``groups``: the group index of each labeled row, ``n_groups``; one group per
    row when absent) is drawn for every cell, counted or not, exactly as the bootstrap gate draws.
    """
    if type(draws) is not int or draws < 1:
        raise ValueError("Bootstrap draws must be a positive integer")
    counted = [name for name, cell in cells.items() if cell["counted"]]
    if not counted:
        raise ValueError("No candidate pool counts toward the gate's statistics")

    def measure(weights):
        result = []
        for arm in (0, 1):
            per_cell = []
            for name in counted:
                positions, labels, starts = cells[name]["arms"][arm]
                cell_weights = None if weights is None else weights[name][positions].astype(float)
                stats = ranking_statistics(labels, starts, cells[name]["k"], weights=cell_weights, pauc_fpr=pauc_fpr)
                positives = float(labels.sum()) if cell_weights is None else float((labels * cell_weights).sum())
                per_cell.append((stats, positives))
            result.append(_kind_statistics(per_cell, pooled))
        return result

    point_parent, point_candidate = measure(None)
    rng = np.random.default_rng(seed)
    deltas = {name: np.full(draws, np.nan) for name in STATISTICS}
    for draw in range(draws):
        weights = {name: (rng.poisson(1., cell["n_groups"]).astype(np.int64)[cell["groups"]] if "groups" in cell
                          else rng.poisson(1., cell["size"]).astype(np.int64)) for name, cell in cells.items()}
        before, after = measure(weights)
        for name in STATISTICS:
            if before[name] is not None and after[name] is not None:
                deltas[name][draw] = after[name] - before[name]
    bounds = [100. * alpha / 2., 100. * (1. - alpha / 2.)]
    out = {}
    for name in STATISTICS:
        parent, candidate = point_parent[name], point_candidate[name]
        low, high = ((float(v) for v in np.nanpercentile(deltas[name], bounds))
                     if np.isfinite(deltas[name]).any() else (None, None))
        out[name] = {"parent": None if parent is None else float(parent),
                     "candidate": None if candidate is None else float(candidate),
                     "delta": None if parent is None or candidate is None else float(candidate - parent),
                     "low": low, "high": high}
    return out


def joint_intervals(parent_state, candidate_state, tables_by_brain, budgets, *, target_kind, draws, seed, alpha=.05):
    """``intervals_from_rankings`` over the target-kind cells of scored tables (the inputs of
    ``fixed_pool_scoring.paired_bootstrap``)."""
    budgets = scoring.Budgets.coerce(budgets)
    cells = {}
    for brain, banks in tables_by_brain.items():
        for kind, table in banks.tables.items():
            if kind != target_kind:
                continue
            name = f"{brain}/{kind}"
            truth = np.asarray(table.truth, dtype=float)
            rows = labeled_rows(truth)
            position = np.full(len(truth), -1, dtype=np.int64)
            position[rows] = np.arange(len(rows))
            arms = []
            for state in (parent_state, candidate_state):
                scores = np.asarray(state[name]["scores"], dtype=float)
                if scores.shape != truth.shape:
                    raise ValueError(f"Gate scores do not align with {name}")
                order, labels, starts = ranking(scores, truth, table.keys)
                arms.append((position[order], labels.astype(float), starts))
            groups, n_groups = scoring.labeled_groups(kind, table, rows)
            cells[name] = {"arms": tuple(arms), "size": len(rows), "k": budgets.k_for(kind, truth),
                           "counted": budgets.in_mean(truth), "positives": int(np.nansum(truth)),
                           "groups": groups, "n_groups": n_groups}
    statistics = intervals_from_rankings(cells, draws=draws, seed=seed, alpha=alpha, pauc_fpr=budgets.pauc_fpr,
                                         pooled=budgets.aggregation == "positive-weighted")
    counted = [cell for cell in cells.values() if cell["counted"]]
    pools = {"pools": len(counted), "labeled": sum(cell["size"] for cell in counted),
             "positives": sum(cell["positives"] for cell in counted),
             "groups": sum(cell["n_groups"] for cell in counted)}
    return statistics, pools


def guard_rail(statistics):
    """No guarded statistic significantly worse; an undefined interval fails the guard."""
    return all(statistics[name]["high"] is not None and statistics[name]["high"] >= 0 for name in GUARDED)


def rule_check(statistics, objective, margin=0.):
    """The rule on the run's objective: the interval of its delta above zero and the point gain
    above ``margin``. The guard rail is separate."""
    name = "precision" if objective == "precision_at_k" else objective
    row = statistics[name]
    significant = row["low"] is not None and row["low"] > 0
    large_enough = row["delta"] is not None and row["delta"] > max(float(margin), 1e-12)
    return {"statistic": name, "delta": row["delta"], "low": row["low"], "high": row["high"],
            "margin": float(margin), "significant": bool(significant), "large_enough": bool(large_enough),
            "passed": bool(significant and large_enough)}


def rejection_category(statistics, rule):
    """Why a candidate was not promoted, as one of ``REJECTION_CATEGORIES``; a function of the
    table's intervals alone, so no statistic's value is passed on.

    ``significantly_worse``: a guarded statistic's interval lies below zero (the guard rail);
    ``within_noise``: the objective's interval does not lie above zero; ``gain_too_small``: it
    does, and the gain is not above the margin. ``rule`` is ``rule_check``'s result.
    """
    if not guard_rail(statistics):
        return "significantly_worse"
    return "gain_too_small" if rule["significant"] else "within_noise"


# --- the gate --------------------------------------------------------------------------------

def gate(parent_state, candidate_state, tables_by_brain, budgets, *, target_kind, draws, seed, alpha=.05, margin=0.):
    """The gate record of one submission; ``passed`` and ``reason`` are what the driver applies.
    Never promotes on an internal failure."""
    budgets = scoring.Budgets.coerce(budgets)
    record = {"version": VERSION, "draws": draws, "seed": seed, "alpha": alpha,
              "guarded_statistics": list(GUARDED), "resampling_unit": scoring.RESAMPLING_UNIT}
    try:
        statistics, pools = joint_intervals(parent_state, candidate_state, tables_by_brain, budgets,
                                            target_kind=target_kind, draws=draws, seed=seed, alpha=alpha)
    except Exception as exc:  # fail closed: no table, no promotion
        record.update(passed=False, status="unavailable", error=f"{type(exc).__name__}: {exc}", rejection_category=None,
                      reason=f"Gate statistics unavailable; candidate not promoted: {type(exc).__name__}: {exc}")
        return record
    guard = guard_rail(statistics)
    rule = rule_check(statistics, budgets.objective, margin)
    record.update(statistics=statistics, pools=pools, guard_rail=guard, rule=rule)
    row = statistics[rule["statistic"]]
    detail = (f"{scoring.OBJECTIVE_LABELS[budgets.objective]} "
              + (f"{row['parent']:.6f} -> {row['candidate']:.6f}; delta={row['delta']:+.6f}, "
                 f"interval [{row['low']:+.6f}, {row['high']:+.6f}]" if row["delta"] is not None and row["low"] is not None
                 else "undefined"))
    if not guard:
        worse = [name for name in GUARDED if statistics[name]["high"] is None or statistics[name]["high"] < 0]
        record.update(passed=False, status="guard_rail", rejection_category=rejection_category(statistics, rule),
                      reason=f"{target_kind} rejected by the guard rail (significantly worse: {', '.join(worse)}): {detail}")
        return record
    accepted = rule["passed"]
    record.update(passed=accepted, status="rule",
                  rejection_category=None if accepted else rejection_category(statistics, rule),
                  reason=f"{target_kind} {'accepted' if accepted else 'rejected'} by the promotion rule (guard rail "
                         f"passed; the interval must lie above zero and the gain above {rule['margin']:g}): {detail}")
    return record

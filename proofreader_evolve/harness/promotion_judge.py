"""LLM judge for the promotion gate (``--promotion-gate judge``, 2026-11-10).

The fixed gates compare one statistic of the candidate with the accepted scorer. Under this
gate a language model reads a table of every rank statistic of both scorers on the
development-validation pools, with the paired-bootstrap interval of each delta, and votes
accept or reject several times; the host decides from the votes inside fixed limits:

* the judge sees numbers only: no run or generation id, no code, no reviser summary;
* a candidate with any guarded statistic significantly worse (interval entirely below zero)
  is rejected whatever the votes say;
* acceptance needs a strict majority of the *requested* votes, so a failed call, a refusal or
  an unparseable answer counts against the candidate (fail closed);
* every vote, its reason, the table and the token use go into the gate record;
* the reviser learns one thing more than "not promoted": a category from a fixed list of five,
  derived by the host from the table's intervals (``rejection_category``), never the model's
  words and never a number.

The statistics are those of ``rank_statistics.ranking_statistics`` on the same Poisson(1) row
weights, drawn in the same order, as ``fixed_pool_scoring.paired_bootstrap``, so the interval of
the run's objective equals the one the bootstrap gate would have used. The validation brains
remain development data: a model judging them repeatedly selects on them like any other gate.

Evidence for the design (``notebooks/acceptance_rule_replay.py``, 2026-11-10, 19 validated
candidates of two raw runs): with intervals the judge agreed with an average-precision gate on
14 of 15 non-seed pairs and accepted 1 of 60 null pairs; without intervals it accepted fewer
real improvements and its votes split more often, which is why the table always carries them.

``--promotion-gate rule`` (2026-11-12) decides from the same table without a model: the guard
rail, the interval of the run's objective above zero, and its gain above ``--precision-margin``.
In the two runs of 2026-11-12 the judge's fourteen voted decisions (thirteen unanimous) all
equalled "the pAUC interval lies above zero", a criterion the run had not declared: the agent
optimised average precision. The rule gate decides on the declared objective instead, so what
the agent optimises on TRAIN is what promotes; choose the objective to choose the criterion.
With ``--judge-shadow`` the judge's votes are still cast and recorded, without effect.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os

import numpy as np

from . import fixed_pool_scoring as scoring
from .label_availability import labeled_rows
from .rank_statistics import FAMILY_FIELD, ranking, ranking_statistics

VERSION = "promotion-judge-v3-rule-binding"
GATE_VERSION = "target-kind-validation-judge-v2"
RULE_GATE_VERSION = "target-kind-validation-rule-v1"
MODE = "judge"
RULE_MODE = "rule"
DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_VOTES = 5
DEFAULT_EFFORT = "medium"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
STATISTICS = ("auroc", "pauc", "average_precision", "precision_depth_mean",
              "precision", "precision_at_half_k", "precision_at_double_k")
LABELS = {"auroc": "AUROC", "pauc": "pAUC (FPR <= {fpr:g})", "average_precision": "average precision",
          "precision_depth_mean": "mean precision at K/2, K, 2K", "precision": "precision at K",
          "precision_at_half_k": "precision at K/2", "precision_at_double_k": "precision at 2K"}
# Single-cut precisions at K/2 and 2K enter the guard through their mean with precision at K.
GUARDED = ("auroc", "pauc", "average_precision", "precision_depth_mean", "precision")
# Statistics of the top of the ranking, for the rejection category under the votes. Average
# precision and precision at K are left out since 2026-11-12: the judge rejected candidates whose
# gains in those two were beyond noise while pAUC and precision at K/2 were not, and the category
# then told the reviser that the top had improved.
HEAD = ("pauc", "precision_at_half_k", "precision_depth_mean")
# Kinds whose candidates come from several rules get a second table restricted to the family that
# holds nearly all true errors: ranking that family above the other raises every pooled statistic
# without telling errors from non-errors inside it. The guard rail and the rejection category keep
# using the whole-pool table.
FAMILY_TABLES = {"merge": "junction"}
# The only words about a rejection that reach the reviser (``rejection_category``).
REJECTION_CATEGORIES = ("significantly_worse", "within_noise", "head_not_improved", "not_convincing",
                        "gain_too_small")

SYSTEM = """You review one proposed change to a ranking model. The model ranks candidate \
reconstruction errors in neuron tracings so that proofreaders inspect the highest-ranked candidates \
first; true errors are rare among the candidates. A new candidate model was measured against the \
currently accepted model on the same held-out pools, and you decide whether the candidate replaces it.

Accept only when the numbers show that the candidate is really better for finding errors, with the \
top of the ranking mattering most because that is what gets inspected. Reject when the differences \
could be resampling noise or when the candidate trades the top of the ranking away. Both mistakes \
cost something: accepting a change that is not an improvement makes all later work build on it, and \
rejecting a real improvement loses the work behind it. Judge from the table alone; you are given no \
other information on purpose. When a second table is given, it holds the same statistics on the one \
group of candidates that contains nearly all true errors; a model can raise the first table merely \
by ranking that group above the rest, so the second table shows whether errors are separated from \
non-errors where it matters."""

SCHEMA = {"type": "object",
          "properties": {"decision": {"type": "string", "enum": ["accept", "reject"]},
                         "reason": {"type": "string"}},
          "required": ["decision", "reason"], "additionalProperties": False}


def gate_modes():
    return (*scoring.GATE_MODES, MODE, RULE_MODE)


def gate_version(mode):
    return {MODE: GATE_VERSION, RULE_MODE: RULE_GATE_VERSION}.get(mode) or scoring.gate_version(mode)


# --- statistics ------------------------------------------------------------------------------

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
    labeled row is drawn for every cell, counted or not, exactly as the bootstrap gate draws.
    """
    if type(draws) is not int or draws < 1:
        raise ValueError("Bootstrap draws must be a positive integer")
    counted = [name for name, cell in cells.items() if cell["counted"]]
    if not counted:
        raise ValueError("No candidate pool counts toward the judge's statistics")

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
        weights = {name: rng.poisson(1., cell["size"]).astype(np.int64) for name, cell in cells.items()}
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


def joint_intervals(parent_state, candidate_state, tables_by_brain, budgets, *, target_kind, draws, seed, alpha=.05,
                    family=None):
    """``intervals_from_rankings`` over the target-kind cells of scored tables (the inputs of
    ``fixed_pool_scoring.paired_bootstrap``). ``family`` restricts every pool to the candidates of
    one family (other rows count as unlabeled; K and the positives floor follow the restricted rows)."""
    budgets = scoring.Budgets.coerce(budgets)
    cells = {}
    for brain, banks in tables_by_brain.items():
        for kind, table in banks.tables.items():
            if kind != target_kind:
                continue
            name = f"{brain}/{kind}"
            truth = np.asarray(table.truth, dtype=float)
            if family is not None:
                candidates = getattr(table, 'candidates', None)
                if not candidates or FAMILY_FIELD not in candidates[0]:
                    raise ValueError(f'{name} carries no candidate families')
                member = np.array([candidate[FAMILY_FIELD] == family for candidate in candidates], dtype=bool)
                truth = np.where(member, truth, np.nan)
                if not np.nansum(truth) or not (truth == 0).any():
                    continue
            rows = labeled_rows(truth)
            position = np.full(len(truth), -1, dtype=np.int64)
            position[rows] = np.arange(len(rows))
            arms = []
            for state in (parent_state, candidate_state):
                scores = np.asarray(state[name]["scores"], dtype=float)
                if scores.shape != truth.shape:
                    raise ValueError(f"Judge scores do not align with {name}")
                order, labels, starts = ranking(scores, truth, table.keys)
                arms.append((position[order], labels.astype(float), starts))
            cells[name] = {"arms": tuple(arms), "size": len(rows), "k": budgets.k_for(kind, truth),
                           "counted": budgets.in_mean(truth), "positives": int(np.nansum(truth))}
    statistics = intervals_from_rankings(cells, draws=draws, seed=seed, alpha=alpha, pauc_fpr=budgets.pauc_fpr,
                                         pooled=budgets.aggregation == "positive-weighted")
    counted = [cell for cell in cells.values() if cell["counted"]]
    pools = {"pools": len(counted), "labeled": sum(cell["size"] for cell in counted),
             "positives": sum(cell["positives"] for cell in counted)}
    return statistics, pools


def guard_rail(statistics):
    """No guarded statistic significantly worse; an undefined interval fails the guard."""
    return all(statistics[name]["high"] is not None and statistics[name]["high"] >= 0 for name in GUARDED)


def rule_check(statistics, objective, margin=0.):
    """The deterministic rule of ``--promotion-gate rule`` on the run's objective: the interval
    of its delta above zero and the point gain above ``margin``. The guard rail is separate."""
    name = "precision" if objective == "precision_at_k" else objective
    row = statistics[name]
    significant = row["low"] is not None and row["low"] > 0
    large_enough = row["delta"] is not None and row["delta"] > max(float(margin), 1e-12)
    return {"statistic": name, "delta": row["delta"], "low": row["low"], "high": row["high"],
            "margin": float(margin), "significant": bool(significant), "large_enough": bool(large_enough),
            "passed": bool(significant and large_enough)}


def rejection_category(statistics, rule=None):
    """Why a candidate was not promoted, as one of ``REJECTION_CATEGORIES``; a function of the
    table's intervals alone, so no statistic's value and no text of the judge is passed on.

    ``significantly_worse``: a guarded statistic's interval lies below zero (the guard rail).
    Under the votes: ``within_noise``: no guarded statistic's interval lies above zero;
    ``head_not_improved``: something improved beyond noise but no statistic of the top of the
    ranking (``HEAD``) did; ``not_convincing``: a head statistic improved beyond noise and the
    votes still rejected. Under the rule (``rule`` = ``rule_check``'s result): ``within_noise``:
    the objective's interval does not lie above zero; ``gain_too_small``: it does, and the gain
    is not above the margin.
    """
    if not guard_rail(statistics):
        return "significantly_worse"
    if rule is not None:
        return "gain_too_small" if rule["significant"] else "within_noise"
    if not any(statistics[name]["low"] is not None and statistics[name]["low"] > 0 for name in GUARDED):
        return "within_noise"
    top = any(statistics[name]["low"] is not None and statistics[name]["low"] > 0 for name in HEAD)
    return "not_convincing" if top else "head_not_improved"


# --- the judge -------------------------------------------------------------------------------

def prompt(kind, pools, statistics, *, alpha=.05, pauc_fpr=scoring.DEFAULT_PAUC_FPR, pooled=True,
           with_intervals=True, family=None):
    """The user message: the held-out sample sizes and the table. Nothing identifies the run.
    ``family`` = ``(name, pools, statistics)`` appends the same table restricted to one family."""
    level = f"{100. * (1. - alpha):g}%"
    lines = [f"Error kind: {kind}. Held-out data: {pools['pools']} pools, {pools['labeled']} labeled candidates "
             f"of which {pools['positives']} are true errors. K in a pool is its number of true errors, so "
             "precision at K is also recall at K. Statistics are "
             + ("pooled over the pools weighted by their true errors." if pooled else "averaged over the pools."), "",
             "| statistic | accepted model | candidate | delta |"
             + (f" {level} interval of the delta (paired bootstrap) |" if with_intervals else ""),
             "|---|---|---|---|" + ("---|" if with_intervals else "")]

    def number(value, signed=False):
        return "undefined" if value is None else (f"{value:+.4f}" if signed else f"{value:.4f}")

    for name in STATISTICS:
        row = statistics[name]
        line = (f"| {LABELS[name].format(fpr=pauc_fpr)} | {number(row['parent'])} | {number(row['candidate'])} | "
                f"{number(row['delta'], True)} |")
        if with_intervals:
            line += f" [{number(row['low'], True)}, {number(row['high'], True)}] |"
        lines.append(line)
    if family is not None:
        name, family_pools, family_statistics = family
        lines += ["", f"Second table: {name} candidates only ({family_pools['pools']} pools, {family_pools['labeled']} "
                      f"labeled candidates of which {family_pools['positives']} are true errors; K is again the number "
                      "of true errors).", "",
                  "| statistic | accepted model | candidate | delta |"
                  + (f" {level} interval of the delta (paired bootstrap) |" if with_intervals else ""),
                  "|---|---|---|---|" + ("---|" if with_intervals else "")]
        for key in STATISTICS:
            row = family_statistics[key]
            line = (f"| {LABELS[key].format(fpr=pauc_fpr)} | {number(row['parent'])} | {number(row['candidate'])} | "
                    f"{number(row['delta'], True)} |")
            if with_intervals:
                line += f" [{number(row['low'], True)}, {number(row['high'], True)}] |"
            lines.append(line)
    lines += ["", "Does the candidate replace the accepted model? Answer with the JSON object."]
    return "\n".join(lines)


def vote(client, text, *, model, effort):
    """One vote: ``{"decision", "reason", ...}`` or ``{"error": ...}`` (never raises on API faults)."""
    import anthropic
    try:
        response = client.messages.create(
            model=model, max_tokens=8000, system=SYSTEM, messages=[{"role": "user", "content": text}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}})
    except anthropic.RateLimitError as exc:
        return {"error": f"RateLimitError: {exc.message}"}
    except anthropic.APIStatusError as exc:
        return {"error": f"APIStatusError {exc.status_code}: {exc.message}"}
    except anthropic.APIConnectionError as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    if response.stop_reason != "end_turn":  # a refusal or a truncated answer is no vote
        return {"error": f"stop_reason {response.stop_reason}", "usage": usage}
    answer = next((block.text for block in response.content if block.type == "text"), None)
    try:
        parsed = json.loads(answer)
        decision, reason = parsed["decision"], str(parsed["reason"])
    except (TypeError, ValueError, KeyError):
        return {"error": "unparseable answer", "usage": usage}
    if decision not in ("accept", "reject"):
        return {"error": f"unknown decision {decision!r}", "usage": usage}
    return {"decision": decision, "reason": reason, "usage": usage, "request_id": response._request_id}


def cast_votes(text, *, votes, model=DEFAULT_MODEL, effort=DEFAULT_EFFORT):
    """``votes`` independent votes on one table. Tests replace this function."""
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return [{"error": "ANTHROPIC_API_KEY is not set"} for _ in range(votes)]
    client = anthropic.Anthropic(max_retries=6, timeout=300.)
    with ThreadPoolExecutor(max_workers=votes) as pool:
        return list(pool.map(lambda _: vote(client, text, model=model, effort=effort), range(votes)))


def decide(statistics, ballots, requested):
    """(accepted, accept votes, votes cast): a strict majority of the requested votes and the guard."""
    cast = [ballot for ballot in ballots if "decision" in ballot]
    accepts = sum(ballot["decision"] == "accept" for ballot in cast)
    return bool(guard_rail(statistics) and accepts * 2 > requested), accepts, len(cast)


def judge(parent_state, candidate_state, tables_by_brain, budgets, *, target_kind, draws, seed, alpha=.05,
          votes=DEFAULT_VOTES, model=DEFAULT_MODEL, effort=DEFAULT_EFFORT, binding=MODE, margin=0., shadow=False):
    """The gate record of one submission; ``passed`` and ``reason`` are what the driver applies.
    Never promotes on an internal failure.

    ``binding`` = ``"judge"``: the votes decide inside the guard rail. ``binding`` = ``"rule"``:
    ``rule_check`` on the run's objective decides inside the guard rail, no model is needed, and
    with ``shadow`` the votes are cast and recorded without effect (a failed shadow call changes
    nothing)."""
    if binding not in (MODE, RULE_MODE):
        raise ValueError(f"Unknown binding {binding!r}")
    budgets = scoring.Budgets.coerce(budgets)
    by_rule = binding == RULE_MODE
    record = {"version": VERSION, "binding": binding, "model": model, "effort": effort,
              "votes_requested": 0 if by_rule and not shadow else votes,
              "draws": draws, "seed": seed, "alpha": alpha, "guarded_statistics": list(GUARDED)}
    try:
        statistics, pools = joint_intervals(parent_state, candidate_state, tables_by_brain, budgets,
                                            target_kind=target_kind, draws=draws, seed=seed, alpha=alpha)
    except Exception as exc:  # fail closed: no table, no promotion
        record.update(passed=False, status="unavailable", error=f"{type(exc).__name__}: {exc}", rejection_category=None,
                      reason=f"Judge statistics unavailable; candidate not promoted: {type(exc).__name__}: {exc}")
        return record
    guard = guard_rail(statistics)
    family = None
    if target_kind in FAMILY_TABLES:
        try:  # a missing second table never blocks the decision: the first table is the gate's basis
            family = (FAMILY_TABLES[target_kind],
                      *joint_intervals(parent_state, candidate_state, tables_by_brain, budgets, target_kind=target_kind,
                                       draws=draws, seed=seed, alpha=alpha, family=FAMILY_TABLES[target_kind])[::-1])
        except Exception as exc:
            record["family_table_error"] = f"{type(exc).__name__}: {exc}"
    text = prompt(target_kind, pools, statistics, alpha=alpha, pauc_fpr=budgets.pauc_fpr,
                  pooled=budgets.aggregation == "positive-weighted", family=family)
    if family is not None:
        record.update(family=family[0], family_pools=family[1], family_statistics=family[2])
    rule = rule_check(statistics, budgets.objective, margin)
    record.update(statistics=statistics, pools=pools, guard_rail=guard, rule=rule,
                  prompt_sha256=hashlib.sha256((SYSTEM + "\n" + text).encode()).hexdigest())
    worse = [name for name in GUARDED if statistics[name]["high"] is None or statistics[name]["high"] < 0]
    row = statistics[rule["statistic"]]
    detail = (f"{scoring.OBJECTIVE_LABELS[budgets.objective]} "
              + (f"{row['parent']:.6f} -> {row['candidate']:.6f}; delta={row['delta']:+.6f}, "
                 f"interval [{row['low']:+.6f}, {row['high']:+.6f}]" if row["delta"] is not None and row["low"] is not None
                 else "undefined"))
    if not guard:
        # Neither the votes nor the rule could change the outcome, so the model is not called.
        record.update(passed=False, status="guard_rail", ballots=[], accept_votes=0, votes_cast=0,
                      rejection_category=rejection_category(statistics),
                      reason=f"{target_kind} rejected by the {'rule' if by_rule else 'judge'}'s guard rail "
                             f"(significantly worse: {', '.join(worse)}): {detail}")
        return record
    ballots = []
    if not by_rule or shadow:
        try:
            ballots = cast_votes(text, votes=votes, model=model, effort=effort)
        except Exception as exc:  # fail closed under the votes; without effect under the rule
            ballots = [{"error": f"{type(exc).__name__}: {exc}"} for _ in range(votes)]
    voted, accepts, cast = decide(statistics, ballots, votes)
    usage = {key: sum((ballot.get("usage") or {}).get(key, 0) for ballot in ballots)
             for key in ("input_tokens", "output_tokens")}
    record.update(ballots=ballots, accept_votes=accepts, votes_cast=cast, usage=usage)
    if by_rule:
        accepted = rule["passed"]
        record.update(passed=accepted, status="rule", shadow_votes_passed=voted if shadow and cast else None,
                      rejection_category=None if accepted else rejection_category(statistics, rule),
                      reason=f"{target_kind} {'accepted' if accepted else 'rejected'} by the promotion rule "
                             f"(guard rail passed; the interval must lie above zero and the gain above "
                             f"{rule['margin']:g}): {detail}"
                             + (f"; shadow judge {accepts} accept of {cast} votes cast" if shadow else ""))
        return record
    # No category without a vote cast: failed calls say nothing about the candidate.
    record.update(passed=voted, status="voted",
                  rejection_category=None if voted or not cast else rejection_category(statistics),
                  reason=f"{target_kind} {'accepted' if voted else 'rejected'} by the LLM judge "
                         f"({accepts} accept of {cast} votes cast, {votes} requested; guard rail passed): {detail}")
    return record

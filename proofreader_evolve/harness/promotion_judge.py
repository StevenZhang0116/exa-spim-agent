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
* the reviser learns one thing more than "not promoted": a category from a fixed list of four,
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

VERSION = "promotion-judge-v2-family-table"
GATE_VERSION = "target-kind-validation-judge-v2"
MODE = "judge"
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
# Guarded statistics that weigh the top of the ranking (everything but the whole-ranking AUROC).
HEAD = ("pauc", "average_precision", "precision_depth_mean", "precision")
# Kinds whose candidates come from several rules get a second table restricted to the family that
# holds nearly all true errors: ranking that family above the other raises every pooled statistic
# without telling errors from non-errors inside it. The guard rail and the rejection category keep
# using the whole-pool table.
FAMILY_TABLES = {"merge": "junction"}
# The only words about a rejection that reach the reviser (``rejection_category``).
REJECTION_CATEGORIES = ("significantly_worse", "within_noise", "head_not_improved", "not_convincing")

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
    return (*scoring.GATE_MODES, MODE)


def gate_version(mode):
    return GATE_VERSION if mode == MODE else scoring.gate_version(mode)


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


def rejection_category(statistics):
    """Why a candidate was not promoted, as one of ``REJECTION_CATEGORIES``; a function of the
    table's intervals alone, so no statistic's value and no text of the judge is passed on.

    ``significantly_worse``: a guarded statistic's interval lies below zero (the guard rail);
    ``within_noise``: no guarded statistic's interval lies above zero;
    ``head_not_improved``: only the whole-ranking AUROC improved beyond noise;
    ``not_convincing``: a head statistic improved beyond noise and the votes still rejected.
    """
    if not guard_rail(statistics):
        return "significantly_worse"
    better = [name for name in GUARDED if statistics[name]["low"] is not None and statistics[name]["low"] > 0]
    if not better:
        return "within_noise"
    return "not_convincing" if any(name in HEAD for name in better) else "head_not_improved"


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
          votes=DEFAULT_VOTES, model=DEFAULT_MODEL, effort=DEFAULT_EFFORT):
    """The gate record of one submission; ``passed`` and ``reason`` are what the driver applies.
    Never promotes on an internal failure."""
    budgets = scoring.Budgets.coerce(budgets)
    record = {"version": VERSION, "model": model, "effort": effort, "votes_requested": votes,
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
        try:  # a missing second table never blocks the vote: the first table is the gate's basis
            family = (FAMILY_TABLES[target_kind],
                      *joint_intervals(parent_state, candidate_state, tables_by_brain, budgets, target_kind=target_kind,
                                       draws=draws, seed=seed, alpha=alpha, family=FAMILY_TABLES[target_kind])[::-1])
        except Exception as exc:
            record["family_table_error"] = f"{type(exc).__name__}: {exc}"
    text = prompt(target_kind, pools, statistics, alpha=alpha, pauc_fpr=budgets.pauc_fpr,
                  pooled=budgets.aggregation == "positive-weighted", family=family)
    if family is not None:
        record.update(family=family[0], family_pools=family[1], family_statistics=family[2])
    record.update(statistics=statistics, pools=pools, guard_rail=guard,
                  prompt_sha256=hashlib.sha256((SYSTEM + "\n" + text).encode()).hexdigest())
    worse = [name for name in GUARDED if statistics[name]["high"] is None or statistics[name]["high"] < 0]
    objective = "precision" if budgets.objective == "precision_at_k" else budgets.objective
    row = statistics[objective]
    detail = (f"{scoring.OBJECTIVE_LABELS[budgets.objective]} "
              + (f"{row['parent']:.6f} -> {row['candidate']:.6f}; delta={row['delta']:+.6f}, "
                 f"interval [{row['low']:+.6f}, {row['high']:+.6f}]" if row["delta"] is not None and row["low"] is not None
                 else "undefined"))
    if not guard:
        # The votes could not change the outcome, so the model is not called.
        record.update(passed=False, status="guard_rail", ballots=[], accept_votes=0, votes_cast=0,
                      rejection_category=rejection_category(statistics),
                      reason=f"{target_kind} rejected by the judge's guard rail (significantly worse: "
                             f"{', '.join(worse)}): {detail}")
        return record
    try:
        ballots = cast_votes(text, votes=votes, model=model, effort=effort)
    except Exception as exc:  # fail closed
        ballots = [{"error": f"{type(exc).__name__}: {exc}"} for _ in range(votes)]
    accepted, accepts, cast = decide(statistics, ballots, votes)
    usage = {key: sum((ballot.get("usage") or {}).get(key, 0) for ballot in ballots)
             for key in ("input_tokens", "output_tokens")}
    # No category without a vote cast: failed calls say nothing about the candidate.
    record.update(passed=accepted, status="voted", ballots=ballots, accept_votes=accepts, votes_cast=cast, usage=usage,
                  rejection_category=None if accepted or not cast else rejection_category(statistics),
                  reason=f"{target_kind} {'accepted' if accepted else 'rejected'} by the LLM judge "
                         f"({accepts} accept of {cast} votes cast, {votes} requested; guard rail passed): {detail}")
    return record

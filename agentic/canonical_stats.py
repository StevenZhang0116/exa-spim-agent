"""
Canonical scoring for AutoDiscovery feature evaluation.

One implementation of the scoring the feature specifications define
(``markdowns/{merge,split}_features_for_proofreader_evolve.md``, "Evaluation"):
the tie-aware out-of-fold average precision, the precision of the top fraction
of labeled rows, the labeled share at the top of the full ranking, the raw
baseline of each kind (node degree for merge candidates, gap distance for split
candidates, both as the ranking itself and never a model fitted to it), the
paired bootstrap of the increment in average precision over that baseline, and
the tie-breaking null (the AP that random tie-breaks of the baseline reach; integer
degree ties every labeled merge row, so a score that only reorders those ties gains
AP without information).

Why one module: the verifier of new-merge-1 (2026-10-19) found corrected tests
measured against five different "baselines" (a logistic fit of degree at 0.0258
or 0.0286, a boosted degree model at 0.0185, a non-tie-aware AP at 0.0319 and
0.0410) where the raw tie-aware degree ranking gives 0.0278, so increments were
not comparable across hypotheses. Corrected scripts import this module and print
``report()``, so every corrected increment is measured the same way; the driver
prints ``baseline_card`` into its instructions so every agent cites the same
figure to beat.

The runner links this file into each script's sandbox when the script mentions
``canonical_stats`` and folds its digest into the checkpoint key, so a change
here re-measures the scripts that use it.

Usage
-----
In a corrected script (the companion npz is beside the script)::

    import numpy as np, canonical_stats as cs
    comp = cs.load_companion("rerank_companion_794495_merge.npz")
    scores = cs.finite(my_feature_scores)           # NaN ranks last
    cs.report(comp, scores, name="degree+feature")  # prints the CANONICAL block

Baseline card of a companion (what the driver runs)::

    python agentic/canonical_stats.py card markdowns/rerank_companion_794495_merge.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

STANDARD_FRACTIONS: tuple[float, ...] = (0.01, 0.025, 0.05, 0.10, 0.15, 0.20)
SHARE_FRACTIONS: tuple[float, ...] = (0.01, 0.05, 0.20)
N_DRAWS: int = 1000
TIE_BREAK_DRAWS: int = 200
SEED: int = 0
# Columns the card and the scoring read; the rest of a companion stays on disk.
_MERGE_COLUMNS = ("row", "label", "fold", "segment_id", "node_id", "degree")
_SPLIT_COLUMNS = ("row", "label", "fold", "segment_id_a", "segment_id_b", "gap_um")


def load_companion(path, columns=None) -> dict:
    """The companion's arrays as a dict (``kind`` added: ``merge`` or ``split``).

    Loads the identity, label, fold, fit-mask and baseline columns, plus
    ``columns``; other arrays (the split occurrences, image coordinates) are
    left on disk unless named.
    """
    with np.load(path, allow_pickle=False) as npz:
        names = set(npz.files)
        kind = companion_kind(names)
        wanted = list(_MERGE_COLUMNS if kind == "merge" else _SPLIT_COLUMNS)
        wanted += [n for n in names if n.startswith("fit_mask_fold")]
        wanted += list(columns or ())
        comp = {name: npz[name] for name in dict.fromkeys(wanted) if name in names}
    comp["kind"] = kind
    return comp


def companion_kind(names) -> str:
    if "segment_id_a" in names and "gap_um" in names:
        return "split"
    if "segment_id" in names and "degree" in names:
        return "merge"
    raise ValueError("not a merge or split companion: no degree / gap_um column")


def candidate_keys(comp) -> np.ndarray:
    """Tie-break key of every row (the specifications' ``candidate_keys``)."""
    if comp["kind"] == "merge":
        return np.char.add(np.char.add("merge:", comp["segment_id"].astype(str)),
                           np.char.add(":", comp["node_id"].astype(str)))
    return np.char.add(np.char.add("split:", comp["segment_id_a"].astype(str)),
                       np.char.add(":", comp["segment_id_b"].astype(str)))


def finite(scores, fill=-np.inf) -> np.ndarray:
    """Scores with NaN replaced so the row ranks last (the specifications' rule)."""
    s = np.asarray(scores, dtype=float).copy()
    s[~np.isfinite(s)] = fill
    return s


def precision_at_fraction(comp, scores, fraction, keys=None, counts=False):
    """Out-of-fold precision of the top ``fraction`` of labeled rows, summed over folds.

    Pass ``keys=candidate_keys(comp)`` when calling repeatedly; ``counts=True``
    returns ``(hits, rows ranked)`` instead of their ratio. Ties rank in key order.
    """
    label, fold = comp["label"], comp["fold"]
    keys = candidate_keys(comp) if keys is None else keys
    tp = k_total = 0
    for f in np.unique(fold):
        rows = np.flatnonzero((fold == f) & ~np.isnan(label))
        k = max(1, int(round(fraction * len(rows))))
        order = rows[np.lexsort((keys[rows], -np.asarray(scores, dtype=float)[rows]))]
        top = order[:k]
        tp += int(label[top].sum())
        k_total += len(top)
    return (tp, k_total) if counts else tp / k_total


def _ap(hits: np.ndarray, scores: np.ndarray) -> float:
    """Tie-aware average precision of boolean ``hits`` ranked by ``scores``, high first."""
    s = -np.asarray(scores, dtype=float)
    order = np.argsort(s, kind="stable")
    s, hits = s[order], np.asarray(hits, dtype=bool)[order]
    last = np.flatnonzero(np.r_[s[1:] != s[:-1], True])   # last row of every tie group
    tp = np.cumsum(hits)[last]
    return float(np.sum(tp / (last + 1) * np.diff(np.r_[0, tp])) / max(hits.sum(), 1))


def average_precision(comp, scores) -> float:
    """Average precision over labeled rows. Tied scores form one threshold (as in
    sklearn.metrics.average_precision_score), so a score that ties many rows is not
    credited for the row order inside its ties."""
    label = comp["label"]
    rows = np.flatnonzero(~np.isnan(label))
    return _ap(label[rows] == 1, np.asarray(scores, dtype=float)[rows])


def fold_average_precision(comp, scores) -> dict[int, float]:
    """Average precision of the labeled rows of each fold."""
    label, fold = comp["label"], comp["fold"]
    out = {}
    for f in np.unique(fold):
        rows = np.flatnonzero((fold == f) & ~np.isnan(label))
        out[int(f)] = _ap(label[rows] == 1, np.asarray(scores, dtype=float)[rows])
    return out


def labeled_share_at_fraction(comp, scores, fraction) -> float:
    """Share of GT-labeled rows among the top ``fraction`` of ALL rows (labeled or not)."""
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    top = order[:max(1, int(round(fraction * len(order))))]
    return float((~np.isnan(comp["label"][top])).mean())


def fit_rows(comp, fold) -> np.ndarray:
    """Rows a model may be fitted on when ``fold`` is held out: purged mask and a label."""
    return np.flatnonzero(comp[f"fit_mask_fold{int(fold)}"] & ~np.isnan(comp["label"]))


def held_rows(comp, fold) -> np.ndarray:
    """Rows of ``fold`` (labeled or not) that the fold's model scores."""
    return np.flatnonzero(comp["fold"] == fold)


def baseline(comp) -> tuple[str, np.ndarray]:
    """The kind's raw baseline ranking: ``("degree", degree)`` or ``("gap", -gap_um)``.

    This is the ranking itself (higher score first), not a model fitted to it:
    a logistic or tree fit of the same column scores lower out of fold, and an
    increment measured against such a fit is not an increment over the baseline.
    """
    if comp["kind"] == "merge":
        return "degree", np.asarray(comp["degree"], dtype=float)
    return "gap", -np.asarray(comp["gap_um"], dtype=float)


def paired_bootstrap_delta_ap(
    comp, scores, reference, *, n_draws: int = N_DRAWS, seed: int = SEED,
) -> dict:
    """Paired bootstrap of ``AP(scores) - AP(reference)`` over labeled rows.

    Labeled rows are resampled with replacement within each fold, the same
    draws for both scores (the specifications' interval). Returns the point
    delta, the mean and 95% interval of the draws, the share of draws at or
    below zero (one-sided p of "no gain"), and its two-sided version.
    """
    label, fold = comp["label"], comp["fold"]
    scores = np.asarray(scores, dtype=float)
    reference = np.asarray(reference, dtype=float)
    labeled = np.flatnonzero(~np.isnan(label))
    hits_all = label == 1
    by_fold = [labeled[fold[labeled] == f] for f in np.unique(fold[labeled])]
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_draws)
    for i in range(n_draws):
        rows = np.concatenate([rng.choice(r, size=len(r), replace=True) for r in by_fold])
        deltas[i] = _ap(hits_all[rows], scores[rows]) - _ap(hits_all[rows], reference[rows])
    point = _ap(hits_all[labeled], scores[labeled]) - _ap(hits_all[labeled], reference[labeled])
    p_one = float(np.mean(deltas <= 0))
    return {
        "delta": point,
        "mean": float(deltas.mean()),
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "p_one_sided": p_one,
        "p_two_sided": float(min(1.0, 2 * min(p_one, 1 - p_one))),
        "n_draws": n_draws,
        "seed": seed,
    }


def tie_break_null(
    comp, reference, model_ap: float, *, n_draws: int = TIE_BREAK_DRAWS, seed: int = SEED,
) -> dict:
    """Share of random tie-breaks of ``reference`` whose AP reaches ``model_ap``.

    Each draw adds uniform noise smaller than half the smallest step between distinct
    reference values, so only tied rows are reordered. Degree ties 9,742 of the 9,745
    labeled merge rows of 794495 and random tie-breaks average AP 0.035 (95% range
    0.027 to 0.048) against the tie-aware 0.0278, so the specifications count a gain
    only when this share is at most 0.05. Gap ties few split rows and the null stays
    within 0.001 of the baseline there.
    """
    label = comp["label"]
    rows = np.flatnonzero(~np.isnan(label))
    hits = label[rows] == 1
    base = np.asarray(reference, dtype=float)[rows]
    distinct = np.unique(base[np.isfinite(base)])
    step = float(np.min(np.diff(distinct))) if len(distinct) > 1 else 1.0
    rng = np.random.default_rng(seed)
    aps = np.array([_ap(hits, base + rng.uniform(0.0, 0.5 * step, len(rows)))
                    for _ in range(n_draws)])
    return {
        "share_null_ge_model": float(np.mean(aps >= model_ap)),
        "null_mean": float(aps.mean()),
        "null_95": [float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))],
        "n_draws": int(n_draws),
        "seed": seed,
    }


def label_shift(comp, values) -> dict:
    """How ``values`` differ between labeled and unlabeled rows (KS statistic, medians)."""
    from scipy.stats import ks_2samp

    v = np.asarray(values, dtype=float)
    labeled = ~np.isnan(comp["label"])
    a, b = v[labeled & np.isfinite(v)], v[~labeled & np.isfinite(v)]
    if len(a) == 0 or len(b) == 0:
        return {"ks": None, "median_labeled": None, "median_unlabeled": None}
    return {
        "ks": float(ks_2samp(a, b).statistic),
        "median_labeled": float(np.median(a)),
        "median_unlabeled": float(np.median(b)),
    }


def evaluate(
    comp, scores, *, name: str, fractions=STANDARD_FRACTIONS, n_draws: int = N_DRAWS,
    seed: int = SEED, bootstrap: bool = True,
) -> dict:
    """Every figure the specifications ask for, for ``scores`` against the baseline."""
    scores = finite(scores)
    bname, bscores = baseline(comp)
    keys = candidate_keys(comp)
    label = comp["label"]
    labeled = ~np.isnan(label)
    folds = {}
    for f in np.unique(comp["fold"]):
        rows = (comp["fold"] == f) & labeled
        folds[int(f)] = {"n_labeled": int(rows.sum()), "n_pos": int((label[rows] == 1).sum())}
    model_fold, base_fold = fold_average_precision(comp, scores), fold_average_precision(comp, bscores)
    for f in folds:
        folds[f]["ap"] = model_fold[f]
        folds[f]["baseline_ap"] = base_fold[f]
    result = {
        "name": name,
        "baseline": bname,
        "n_rows": int(len(label)),
        "n_labeled": int(labeled.sum()),
        "n_pos": int((label == 1).sum()),
        "ap": average_precision(comp, scores),
        "baseline_ap": average_precision(comp, bscores),
        "folds": folds,
        "precision": {
            str(fr): {
                "model": list(precision_at_fraction(comp, scores, fr, keys, counts=True)),
                "baseline": list(precision_at_fraction(comp, bscores, fr, keys, counts=True)),
            }
            for fr in fractions
        },
        "labeled_share": {
            str(fr): {
                "model": labeled_share_at_fraction(comp, scores, fr),
                "baseline": labeled_share_at_fraction(comp, bscores, fr),
            }
            for fr in SHARE_FRACTIONS
        },
        "labeled_share_overall": float(labeled.mean()),
        # The Firewall check: a score whose labeled and unlabeled distributions
        # differ strongly may measure how traced-neuron-like a candidate is.
        "label_shift": label_shift(comp, scores),
        "baseline_label_shift": label_shift(comp, bscores),
    }
    result["delta_ap"] = result["ap"] - result["baseline_ap"]
    if bootstrap:
        result["bootstrap"] = paired_bootstrap_delta_ap(
            comp, scores, bscores, n_draws=n_draws, seed=seed
        )
        result["tie_break_null"] = tie_break_null(comp, bscores, result["ap"], seed=seed)
    return result


def format_report(result: dict) -> str:
    """The CANONICAL block: one labelled line per figure, stable across scripts."""
    lines = [
        f"CANONICAL model={result['name']} baseline={result['baseline']} "
        f"n_labeled={result['n_labeled']} n_pos={result['n_pos']} n_rows={result['n_rows']}",
        f"CANONICAL ap model={result['ap']:.5f} baseline={result['baseline_ap']:.5f} "
        f"delta={result['delta_ap']:+.5f}",
    ]
    for f, fold in sorted(result["folds"].items()):
        lines.append(
            f"CANONICAL fold {f} model={fold['ap']:.5f} baseline={fold['baseline_ap']:.5f} "
            f"n_labeled={fold['n_labeled']} n_pos={fold['n_pos']}"
        )
    for fr, row in result["precision"].items():
        m, b = row["model"], row["baseline"]
        lines.append(f"CANONICAL precision@{fr} model={m[0]}/{m[1]} baseline={b[0]}/{b[1]}")
    boot = result.get("bootstrap")
    if boot:
        lines.append(
            f"CANONICAL bootstrap delta_mean={boot['mean']:+.5f} "
            f"ci95=[{boot['ci95'][0]:+.5f}, {boot['ci95'][1]:+.5f}] "
            f"p_one_sided={boot['p_one_sided']:.4f} p_two_sided={boot['p_two_sided']:.4f} "
            f"draws={boot['n_draws']} seed={boot['seed']}"
        )
    null = result.get("tie_break_null")
    if null:
        lines.append(
            f"CANONICAL tie_break_null share_null_ge_model={null['share_null_ge_model']:.3f} "
            f"null_mean={null['null_mean']:.5f} "
            f"null_95=[{null['null_95'][0]:.5f}, {null['null_95'][1]:.5f}] "
            f"draws={null['n_draws']}"
        )
    for fr, row in result["labeled_share"].items():
        lines.append(
            f"CANONICAL labeled_share@{fr} model={row['model']:.4f} "
            f"baseline={row['baseline']:.4f} overall={result['labeled_share_overall']:.4f}"
        )
    shift = result.get("label_shift") or {}
    if shift.get("ks") is not None:
        lines.append(
            f"CANONICAL label_shift ks={shift['ks']:.4f} "
            f"median_labeled={shift['median_labeled']:.6g} "
            f"median_unlabeled={shift['median_unlabeled']:.6g}"
        )
    return "\n".join(lines)


def report(comp, scores, *, name: str, **kwargs) -> dict:
    """``evaluate`` and print its CANONICAL block; returns the figures."""
    result = evaluate(comp, scores, name=name, **kwargs)
    print(format_report(result), flush=True)
    return result


# --- Baseline card -------------------------------------------------------------

def baseline_card(npz_path) -> dict:
    """The baseline figures of a companion, for the driver's instructions."""
    npz_path = Path(npz_path)
    comp = load_companion(npz_path)
    bname, bscores = baseline(comp)
    result = evaluate(comp, bscores, name=bname, bootstrap=False)
    manifest_path = npz_path.with_suffix(".json")
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
    return {
        "companion": npz_path.name,
        "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "brain": manifest.get("brain"),
        "kind": comp["kind"],
        "candidate_policy": manifest.get("candidate_policy"),
        "baseline": bname,
        "n_rows": result["n_rows"],
        "n_labeled": result["n_labeled"],
        "n_pos": result["n_pos"],
        "random_rate": result["n_pos"] / max(result["n_labeled"], 1),
        "baseline_ap": result["baseline_ap"],
        "folds": {
            str(f): {"ap": fold["baseline_ap"], "n_labeled": fold["n_labeled"],
                     "n_pos": fold["n_pos"]}
            for f, fold in result["folds"].items()
        },
        "precision": {fr: row["baseline"] for fr, row in result["precision"].items()},
        "labeled_share": {fr: row["baseline"] for fr, row in result["labeled_share"].items()},
        "labeled_share_overall": result["labeled_share_overall"],
        # What random tie-breaks of the baseline alone reach: the floor a model's
        # AP must clear before an increment means anything.
        "tie_break_null": tie_break_null(comp, bscores, result["baseline_ap"]),
    }


def card_text(card: dict) -> str:
    """The card as the sentences an instruction carries."""
    folds = ", ".join(f"fold {f}: {v['ap']:.4f}" for f, v in sorted(card["folds"].items()))
    precision = ", ".join(f"{h}/{k} at {float(fr):.1%}" for fr, (h, k) in card["precision"].items())
    shares = ", ".join(f"{v:.1%} at {float(fr):.0%}" for fr, v in card["labeled_share"].items())
    what = ("node degree, higher first" if card["baseline"] == "degree"
            else "gap distance, smaller first")
    # Cards written before 2026-10-21 (and hand-built ones) carry no null figures.
    null = card.get("tie_break_null")
    null_text = ""
    if null and null["null_95"][1] - null["null_95"][0] >= 0.005:
        null_text = (
            f" Random tie-breaks of the baseline alone average AP {null['null_mean']:.4f} "
            f"(95% range {null['null_95'][0]:.4f} to {null['null_95'][1]:.4f}): a model "
            "whose AP lies inside that range has only reordered the baseline's ties, so "
            "a gain counts only when the CANONICAL tie_break_null share is at most 0.05."
        )
    elif null:
        null_text = (
            f" The baseline ties few rows: random tie-breaks move its AP only between "
            f"{null['null_95'][0]:.4f} and {null['null_95'][1]:.4f}, so the CANONICAL "
            "tie_break_null share matters only for scores that tie many rows."
        )
    return (
        f"BASELINE CARD ({card['companion']}, brain {card['brain']}, {card['kind']}): "
        f"{card['n_labeled']:,} labeled rows with {card['n_pos']:,} positives "
        f"(random rate {card['random_rate']:.4f}) among {card['n_rows']:,} candidates. "
        f"The baseline is the raw ranking by {what}: tie-aware out-of-fold AP "
        f"{card['baseline_ap']:.4f} ({folds}); precision {precision}; its top of ALL "
        f"rows holds {shares} labeled rows against {card['labeled_share_overall']:.1%} "
        "overall. An increment is measured against this ranking, not against a model "
        "fitted to the same column, and with the tie-aware average precision of "
        "agentic/canonical_stats.py." + null_text
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    card = sub.add_parser("card", help="Print the baseline card of a companion npz.")
    card.add_argument("npz", type=Path)
    card.add_argument("--text", action="store_true", help="Print the sentences, not JSON.")
    args = parser.parse_args(argv)
    if args.command == "card":
        result = baseline_card(args.npz)
        print(card_text(result) if args.text else json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

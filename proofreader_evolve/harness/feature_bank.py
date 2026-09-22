"""
FeatureBank: per-brain candidate score tables from the frozen AutoDiscovery
detectors (layer 1+2 of the detector-seeded evolution stack).

The two fitted detectors — read directly from their detector-build deliverable
folders under ``autodiscovery-application/`` (one merge run + one split run,
chosen via ``precompute_error_scores`` ``--merge-dir``/``--split-dir``; exact
provenance in each table's ``meta.json``) — rank error candidates on 129
validated geometric features:

  - the MERGE detector scores a raw segment LABEL for "this label fuses two
    neurons" (segment-level — it does NOT localize the cut; cut placement stays
    with ``dataset.candidate_merge_sites``);
  - the SPLIT detector scores a candidate segment PAIR for "these two labels
    are one broken neuron" (pair-level — the same granularity as a
    ``merge_labels`` edit).

Because full feature extraction over a brain costs hours on a compute node, it
runs OFFLINE once per brain (``cli/precompute_error_scores.py``) and lands here
as two pandas tables under ``feature_tables/<brain>/``:

  merge_scores.pkl : DataFrame indexed by label (str);
                     columns = ["merge_score"] + 39 features + is_defined flags
  split_scores.pkl : DataFrame with label_a/label_b/node_a/node_b key columns;
                     columns include "split_score" + 90 features + flags
  meta.json        : provenance (source pkl, joblib SHAs, enum params,
                     graph fingerprint, threshold sweep summary)

The harness annotates every enumerated site with its scalar score
(``SplitSite.split_score`` / ``MergeSite.label_merge_score``) and hands the
bank itself to the policy as ``ctx["feature_bank"]`` so it can pull full
feature rows on demand. Scores are UNCALIBRATED ranking scores (not
probabilities) and ``nan`` means "candidate not in the table" — a policy must
treat nan explicitly (typically: fall back to pure-geometry rules).

Post-split pseudo-labels (``L#a``): pair lookups strip the ``#suffix`` and read
the base pair's row. Node geometry is unchanged by a relabel (nodes never
move), so the score is exact for those columns and only slightly stale for the
few segment-membership columns (component cable length etc.). Good enough as a
phase-2 prior; an exact live recompute is a later refinement.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

_NAN = float("nan")

# Default location of the per-brain tables, relative to this package.
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TABLES_DIR = os.path.join(_PKG_DIR, "feature_tables")

# Key columns of split_scores.pkl that are NOT detector features.
_SPLIT_KEY_COLS = ("label_a", "label_b", "node_a", "node_b", "gap_um",
                   "split_score", "is_split")
_MERGE_KEY_COLS = ("merge_score", "is_merge")


def base_label(label: str) -> str:
    """Raw label behind a (possibly) post-split pseudo-label: ``'L#a' -> 'L'``."""
    return str(label).split("#", 1)[0]


def pair_key(label_a: str, label_b: str) -> tuple:
    """Order-free key for a label pair, pseudo-labels reduced to their base."""
    a, b = base_label(label_a), base_label(label_b)
    return (a, b) if a <= b else (b, a)


class FeatureBank:
    """Read-only façade over one brain's precomputed detector score tables.

    Construct via :meth:`load`; it returns ``None`` when the brain has no
    tables yet, and the harness then simply leaves every site's score at nan —
    the loop runs fine without a bank, only the model prior is absent.
    """

    def __init__(self, brain_id: str, merge_df, split_df, meta: dict):
        self.brain_id = str(brain_id)
        self.meta = dict(meta or {})
        self._merge_df = merge_df          # index: label str
        self._split_df = split_df          # RangeIndex; key columns above

        # ---- fast lookups --------------------------------------------------
        # merge: label -> score (plain dict; <= a few thousand labels)
        self._merge_score = {}
        if merge_df is not None and len(merge_df):
            scores = merge_df["merge_score"].to_numpy(dtype=float)
            for label, score in zip(merge_df.index.astype(str), scores):
                self._merge_score[label] = float(score)

        # split: unordered base-label pair -> row position of the BEST
        # (highest-scoring) gap between that pair. Pair-level is the edit
        # granularity: one merge_labels unifies the pair across all its gaps.
        self._split_best_row: dict = {}
        if split_df is not None and len(split_df):
            la = split_df["label_a"].astype(str).to_numpy()
            lb = split_df["label_b"].astype(str).to_numpy()
            sc = split_df["split_score"].to_numpy(dtype=float)
            for pos in range(len(split_df)):
                key = pair_key(la[pos], lb[pos])
                cur = self._split_best_row.get(key)
                s = sc[pos]
                if cur is None or (not math.isnan(s) and
                                   (math.isnan(cur[1]) or s > cur[1])):
                    self._split_best_row[key] = (pos, float(s))

        self.merge_feature_names = (
            [c for c in merge_df.columns if c not in _MERGE_KEY_COLS]
            if merge_df is not None else [])
        self.split_feature_names = (
            [c for c in split_df.columns if c not in _SPLIT_KEY_COLS]
            if split_df is not None else [])

    # ---- construction -------------------------------------------------------

    @classmethod
    def load(cls, brain_id: str, tables_dir: str | None = None):
        """Load the bank for ``brain_id`` or return None when absent/unreadable."""
        import pandas as pd

        d = os.path.join(tables_dir or DEFAULT_TABLES_DIR, str(brain_id))
        merge_p = os.path.join(d, "merge_scores.pkl")
        split_p = os.path.join(d, "split_scores.pkl")
        meta_p = os.path.join(d, "meta.json")
        if not (os.path.exists(merge_p) or os.path.exists(split_p)):
            return None
        try:
            merge_df = pd.read_pickle(merge_p) if os.path.exists(merge_p) else None
            split_df = pd.read_pickle(split_p) if os.path.exists(split_p) else None
            meta = {}
            if os.path.exists(meta_p):
                with open(meta_p) as f:
                    meta = json.load(f)
            return cls(brain_id, merge_df, split_df, meta)
        except Exception as e:  # a corrupt table must never kill the loop
            print(f"[feature_bank] failed to load tables for {brain_id}: {e!r} "
                  f"— continuing without detector priors")
            return None

    def validate_graph(self, fragments_graph) -> list:
        """Cheap provenance check; returns a list of WARNING strings (may be [])."""
        warnings = []
        expect = self.meta.get("n_graph_nodes")
        if expect is not None:
            try:
                actual = int(fragments_graph.number_of_nodes())
            except Exception:
                actual = len(getattr(fragments_graph, "node_xyz", []))
            if int(expect) != int(actual):
                warnings.append(
                    f"feature_bank[{self.brain_id}]: table built on a graph with "
                    f"{expect} nodes but this run's graph has {actual} — scores "
                    f"may be misaligned; rebuild with precompute_error_scores")
        return warnings

    # ---- scalar scores (cheap; used to annotate every site) -----------------

    def merge_score(self, label: str) -> float:
        """Frozen merge-detector score for a raw label (nan = not in table)."""
        return self._merge_score.get(base_label(label), _NAN)

    def split_score(self, label_a: str, label_b: str) -> float:
        """Best frozen split-detector score for an unordered label pair."""
        hit = self._split_best_row.get(pair_key(label_a, label_b))
        return hit[1] if hit is not None else _NAN

    # ---- full feature rows (on demand; policy/report use) -------------------

    def merge_features(self, label: str) -> dict:
        """Full 39-column feature row for a label ({} when absent)."""
        if self._merge_df is None:
            return {}
        key = base_label(label)
        if key not in self._merge_df.index:
            return {}
        row = self._merge_df.loc[key]
        return {c: row[c] for c in self.merge_feature_names}

    def split_features(self, label_a: str, label_b: str) -> dict:
        """Full 90-column feature row for the pair's best gap ({} when absent)."""
        hit = self._split_best_row.get(pair_key(label_a, label_b))
        if hit is None or self._split_df is None:
            return {}
        row = self._split_df.iloc[hit[0]]
        return {c: row[c] for c in self.split_feature_names}

    # ---- harness annotation --------------------------------------------------

    def annotate_split_sites(self, sites) -> int:
        """Fill ``site.split_score`` on every SplitSite; returns #scored."""
        n = 0
        for s in sites:
            score = self.split_score(s.label_a, s.label_b)
            s.split_score = score
            if not math.isnan(score):
                n += 1
        return n

    def annotate_merge_sites(self, sites) -> int:
        """Fill ``site.label_merge_score`` on every MergeSite; returns #scored."""
        n = 0
        for s in sites:
            score = self.merge_score(s.label)
            s.label_merge_score = score
            if not math.isnan(score):
                n += 1
        return n

    # ---- reporting -----------------------------------------------------------

    def describe(self) -> str:
        """One-line summary for run logs / the failure report."""
        n_m = len(self._merge_score)
        n_s = len(self._split_best_row)
        prov = self.meta.get("built_utc", "unknown build time")
        return (f"FeatureBank[{self.brain_id}]: {n_m} merge-labels scored, "
                f"{n_s} split-pairs scored ({prov})")

    def score_quantile(self, kind: str, q: float) -> float:
        """Quantile of the stored score distribution ('merge'|'split').

        Lets a policy set brain-relative (quantile) thresholds instead of
        absolute score cutoffs, which do not transfer across brains.
        """
        if kind == "merge":
            vals = np.asarray(list(self._merge_score.values()), dtype=float)
        elif kind == "split":
            vals = np.asarray([s for _, s in self._split_best_row.values()],
                              dtype=float)
        else:
            raise ValueError("kind must be 'merge' or 'split'")
        vals = vals[~np.isnan(vals)]
        if not len(vals):
            return _NAN
        return float(np.quantile(vals, q))

"""Calibrated, MULTI-FEATURE image confidence for merge edits.

Why this exists
---------------
The first version of the confidence layer collapsed a ``gap_bridge_evidence`` read
to a single ``bridge_ratio`` scalar and thresholded it at hand-picked 0.70/0.35.
That is too simple: a single noise dip on a real neurite reads as a fake gap, and a
straight chord between two curved tips can miss the real bridge. But the reader
ALREADY returns a richer vector — ``bridge_ratio``, ``bridge_mean_ratio``,
``valley_frac``, ``profile_cv``, ``bridge_pos`` — at ZERO extra cloud cost. This
module turns that whole vector into ONE calibrated probability that a merge is a
REAL split repair (both fragments are one neuron) vs a FALSE fusion.

Calibration, not hand-tuning
----------------------------
The threshold is fit from DATA, not guessed. The harness's warm-start probe reads a
balanced, leak-free (train-only) sample of REAL vs FALSE SplitSites; those labeled
feature vectors train a small logistic regression here. The fitted model is then
applied to the per-edit blind-spot verdict (option #1) AND its per-feature
separation is reported to the reviser (option #2), so the agent evolves image rules
from the real signal instead of a fixed scalar.

Independence (leak / circularity boundary)
-------------------------------------------
These features are RAW FLUORESCENCE along the gap chord — an independent second
opinion, not the geometry the policy decided with. The model is trained ONLY on the
train-scoped warm-start labels and applied as a confidence signal; it is never a
policy input and (like the rest of the confidence layer) never shown to the reviser
as a held-out verdict — only the aggregate per-feature separability is.

Fallbacks (never crash a run)
-----------------------------
If scikit-learn is unavailable, or there are too few / single-class labels, we fall
back to a monotone ``bridge_ratio`` sigmoid around the legacy 0.70/0.35 band — i.e.
the old behavior — so scoring always has a usable confidence. ``FEATURES`` is the
single source of truth for which keys are used, in order.
"""
from __future__ import annotations

import math

# The image-evidence features used for confidence, in a fixed order. Every one is
# already returned by LazyImagePatchReader.gap_bridge_evidence (no new cloud reads).
# All are normalized by endpoint brightness inside the reader, so they are
# brain-agnostic. HIGHER bridge_ratio / bridge_mean_ratio ⇒ more REAL; HIGHER
# valley_frac / profile_cv ⇒ more FALSE (a wide/uneven dark valley).
FEATURES = ("bridge_ratio", "bridge_mean_ratio", "valley_frac", "profile_cv", "bridge_pos")

# Legacy single-feature band (the pre-calibration thresholds), reused by the
# fallback and as human-readable anchors.
_BRIDGE_LIKELY_CORRECT = 0.70
_BRIDGE_LIKELY_FALSE = 0.35


def extract_features(result: dict) -> list:
    """Pull the FEATURES vector out of a gap_bridge_evidence result dict.

    Returns a list of floats aligned to ``FEATURES``; missing/invalid entries are
    NaN. ``result`` is the ``{...}`` a recorded image read stores under "result".
    """
    out = []
    for k in FEATURES:
        try:
            v = float(result.get(k, float("nan")))
        except (TypeError, ValueError):
            v = float("nan")
        out.append(v)
    return out


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


class ConfidenceModel:
    """Maps a gap_bridge_evidence result -> P(REAL split repair) in [0, 1].

    Two flavors, chosen automatically by ``fit``:
      * ``"logistic"`` — a scikit-learn LogisticRegression over the standardized
        FEATURES vector, fit on the warm-start REAL/FALSE labels. Used when there are
        enough two-class examples.
      * ``"fallback"`` — a monotone bridge_ratio sigmoid centered on the legacy band.
        Used when sklearn is missing or the labels are too few / single-class.

    ``predict_proba(result) -> float`` is NaN-safe (missing features imputed to the
    training mean for logistic, or ignored for the fallback). ``label(p)`` buckets a
    probability into "likely_correct" / "ambiguous" / "likely_false".
    """

    def __init__(self, kind, payload=None, n_pos=0, n_neg=0):
        self.kind = kind          # "logistic" | "fallback"
        self.payload = payload or {}
        self.n_pos = n_pos
        self.n_neg = n_neg

    # --- construction ---
    @staticmethod
    def fallback() -> "ConfidenceModel":
        """Single-feature bridge_ratio sigmoid (the pre-calibration behavior)."""
        return ConfidenceModel("fallback")

    @classmethod
    def fit(cls, real_feats: list, false_feats: list, min_per_class: int = 6):
        """Fit a logistic model on labeled feature vectors; fall back if unable.

        ``real_feats`` / ``false_feats`` are lists of FEATURES-aligned float vectors
        (REAL = merge SHOULD happen, FALSE = must NOT). Rows that are entirely NaN
        are dropped; remaining NaNs are mean-imputed per feature. Requires at least
        ``min_per_class`` usable rows in EACH class, else returns the fallback.
        """
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except Exception:
            return cls.fallback()

        def _clean(rows):
            arr = np.asarray([r for r in rows if any(v == v for v in r)], dtype=float)
            return arr
        pos = _clean(real_feats)
        neg = _clean(false_feats)
        if len(pos) < min_per_class or len(neg) < min_per_class:
            return cls.fallback()

        X = np.vstack([pos, neg])
        y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        # Mean-impute NaNs per column (from the combined training mean).
        col_mean = np.nanmean(X, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xs, y)
        payload = {
            "coef": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]),
            "mean": scaler.mean_.tolist(),
            "scale": [s if s else 1.0 for s in scaler.scale_.tolist()],
            "impute": col_mean.tolist(),
        }
        return cls("logistic", payload, n_pos=int(len(pos)), n_neg=int(len(neg)))

    # --- inference ---
    def predict_proba(self, result: dict) -> float:
        """P(REAL) in [0,1] for one gap_bridge_evidence result. NaN-safe."""
        feats = extract_features(result)
        if self.kind == "logistic":
            p = self.payload
            z = p["intercept"]
            for i, v in enumerate(feats):
                if v != v:  # NaN -> training-mean impute (standardizes to ~0)
                    v = p["impute"][i]
                z += p["coef"][i] * ((v - p["mean"][i]) / p["scale"][i])
            return _sigmoid(z)
        # fallback: monotone bridge_ratio sigmoid centered on the legacy band mid.
        br = feats[0]  # bridge_ratio is FEATURES[0]
        if br != br:
            return float("nan")
        mid = 0.5 * (_BRIDGE_LIKELY_CORRECT + _BRIDGE_LIKELY_FALSE)
        span = max(1e-6, _BRIDGE_LIKELY_CORRECT - _BRIDGE_LIKELY_FALSE)
        # scale so br=_BRIDGE_LIKELY_CORRECT -> ~0.88, br=_BRIDGE_LIKELY_FALSE -> ~0.12
        return _sigmoid(4.0 * (br - mid) / span)

    @staticmethod
    def label(p: float, lo: float = 0.35, hi: float = 0.65):
        """Bucket a probability into a verdict label (NaN -> 'unknown')."""
        if p != p:
            return "unknown"
        if p >= hi:
            return "likely_correct"
        if p <= lo:
            return "likely_false"
        return "ambiguous"

    def describe(self) -> str:
        if self.kind == "logistic":
            terms = ", ".join(f"{f}:{c:+.2f}"
                              for f, c in zip(FEATURES, self.payload["coef"]))
            return (f"logistic over {len(FEATURES)} features "
                    f"(n_real={self.n_pos}, n_false={self.n_neg}); standardized "
                    f"weights [{terms}]")
        return ("fallback: bridge_ratio sigmoid on the legacy "
                f"{_BRIDGE_LIKELY_FALSE:.2f}/{_BRIDGE_LIKELY_CORRECT:.2f} band "
                "(too few labeled examples to calibrate multi-feature)")

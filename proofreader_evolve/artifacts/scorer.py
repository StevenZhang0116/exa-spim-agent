"""Baseline for fixed detector-native pools: preserve the frozen ranking."""


def score_candidates(features, ctx):
    """One finite score per input row; ctx['kind'] is 'merge' or 'split'."""
    return features["detector_score"].to_numpy(dtype=float)

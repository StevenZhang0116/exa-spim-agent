"""Seed for --detector-prior no-score: no frozen ranking, a constant score.

Ties resolve by candidate identity, a label-independent order, so the starting point
is equivalent to an arbitrary ranking of the GT-labeled rows.
"""
import numpy as np


def score_candidates(features, ctx):
    """One finite score per input row; ctx['kind'] is 'merge' or 'split'."""
    return np.zeros(len(features), dtype=float)

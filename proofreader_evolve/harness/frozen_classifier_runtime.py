"""Inference-only NumPy code embedded verbatim in every frozen classifier export."""

import numpy as np


def predict_frozen(model, features, kind):
    if kind != model['kind']:
        raise ValueError('Frozen classifier belongs to another scoring kind')
    columns = model['config']['features']
    missing = set(columns) - set(features.columns)
    if missing:
        raise ValueError('Missing fitted feature columns: ' + repr(sorted(missing)))
    result = np.empty(len(features), dtype=float)
    for start in range(0, len(features), 65536):
        stop = min(start + 65536, len(features))
        x = features.iloc[start:stop][columns].to_numpy(dtype=float, copy=True)
        x = np.where(np.isfinite(x), x, np.asarray(model['impute']))
        x = (x - np.asarray(model['mean'])) / np.asarray(model['scale'])
        if model['config']['algorithm'] == 'logistic_regression':
            logits = x @ np.asarray(model['coef']) + model['intercept']
            # Stable sigmoid; no prediction-time fitting or batch statistics.
            positive = logits >= 0
            scores = np.empty(len(x), dtype=float)
            scores[positive] = 1 / (1 + np.exp(-logits[positive]))
            exp_logits = np.exp(logits[~positive])
            scores[~positive] = exp_logits / (1 + exp_logits)
        else:
            # sklearn trees compare float32 inputs against stored float64 thresholds.
            x = x.astype(np.float32)
            if not np.isfinite(x).all():
                raise ValueError('Predictor values exceed the fitted forest float32 input range')
            scores = np.zeros(len(x), dtype=float)
            for tree in model['trees']:
                left, right = np.asarray(tree['left']), np.asarray(tree['right'])
                feature, threshold = np.asarray(tree['feature']), np.asarray(tree['threshold'])
                nodes = np.zeros(len(x), dtype=int)
                for _ in range(model['config']['parameters']['max_depth']):
                    rows = np.flatnonzero(left[nodes] != -1)
                    if not len(rows):
                        break
                    current = nodes[rows]
                    go_left = x[rows, feature[current]] <= threshold[current]
                    nodes[rows] = np.where(go_left, left[current], right[current])
                if np.any(left[nodes] != -1):
                    raise ValueError('Frozen tree exceeds its declared depth')
                scores += np.asarray(tree['probability'])[nodes]
            scores /= len(model['trees'])
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite frozen classifier prediction')
        result[start:stop] = scores
    return result


def score_candidates(features, ctx):
    return predict_frozen(_FROZEN_CLASSIFIER, features, ctx['kind'])

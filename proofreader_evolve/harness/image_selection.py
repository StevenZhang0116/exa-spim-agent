"""Label-free image sampling and host-only coverage measurements."""
import numpy as np

from .local_features import select_rows


def ranked_rows(frame, spec, keys=None):
    feature = spec['selection_feature']
    if feature not in frame:
        raise ValueError('Image selection_feature must name an existing base predictor')
    values = frame[feature].to_numpy(dtype=float, copy=True)
    priority = -values if spec['selection_largest'] else values
    priority[~np.isfinite(priority)] = np.inf
    tie = (np.asarray(keys, dtype=str) if spec['selection_mode'] == 'top_k_boundary' and keys is not None
           else np.arange(len(frame)))
    if len(tie) != len(frame):
        raise ValueError('Image selection keys must align with the fixed pool')
    return np.lexsort((tie, priority))


def select_image_rows(frame, spec, keys=None):
    """Split the budget across the selection predictor's K boundary, then backfill."""
    if spec['selection_mode'] == 'top':
        # Preserve the original row-order tie behavior of existing contracts.
        return select_rows(frame, spec)
    order = ranked_rows(frame, spec, keys)
    k = min(spec['selection_k'], len(order))
    count = min(spec['max_candidates'], len(order))
    inside = min(k, (count + 1) // 2)
    outside = min(len(order) - k, count - inside)
    inside = min(k, count - outside)
    outside = min(len(order) - k, count - inside)
    return order[k - inside:k + outside]


def selection_coverage(frame, spec, selected, keys=None):
    order = ranked_rows(frame, spec, keys)
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    result = {'mode': spec['selection_mode'], 'feature': spec['selection_feature'],
              'selected_rows': len(selected), 'pool_rows': len(frame),
              'rank_min': int(ranks[selected].min()) if len(selected) else None,
              'rank_max': int(ranks[selected].max()) if len(selected) else None,
              'ranking_scope': 'Frozen base predictor, not the evolving scorer; no labels used'}
    if spec['selection_mode'] == 'top_k_boundary':
        k = min(spec['selection_k'], len(frame))
        inside = int(np.count_nonzero(ranks[selected] <= k))
        result.update(selection_k=spec['selection_k'], effective_selection_k=k,
                      inside_top_k=inside, outside_top_k=len(selected) - inside,
                      inside_top_k_fraction=inside / k if k else None)
    return result


def scoring_coverage(frame, chosen, truth, requested_k):
    """Labels are used only after selection/scoring, inside the host evaluator."""
    available = frame['image_available'].to_numpy() == 1
    selected = np.flatnonzero(available)
    chosen_with_images = np.asarray(chosen)[available[chosen]]
    return {'requested_k': requested_k, 'effective_k': len(chosen),
            'selected_rows': len(selected), 'selected_pool_positives': int(np.asarray(truth)[selected].sum()),
            'top_k_with_images': len(chosen_with_images),
            'top_k_image_fraction': len(chosen_with_images) / len(chosen) if len(chosen) else None,
            'top_k_hits_with_images': int(np.asarray(truth)[chosen_with_images].sum()),
            'scope': 'Input coverage only; image contribution requires a paired removal measurement'}

"""Labeled dataset caches (``dataset_cache_<brain>_mcl<N>_add.pkl``): one builder.

``python -m labeled_cache --brains <b> --mcl 100 --policy length_v1`` runs, per brain,
the stages its files still need: build the base cache from the GCS skeletons, bake the
GT labels from the dense segmentation, and apply the merge-threshold policy to an
existing labeled cache in place. See ``markdowns/labeled_dataset_cache.md``.

``python -m labeled_cache.source_check --brains <b>`` checks, before a brain is built, that
its fragment SWCs and dense segmentation are the same segmentation run (fragment node voxels
read back from the segmentation carry the fragment's own id).
"""

from labeled_cache.policy import (  # noqa: F401
    DEFAULT_LENGTH_UM, POLICIES, POLICY_LENGTH_V1, POLICY_ORIGINAL, policy_matches, thresholds_for,
)

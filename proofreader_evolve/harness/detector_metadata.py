"""Explicit, GT-free metadata for frozen-detector feature extraction."""

from numbers import Integral


def inference_metadata(payload, expected_mcl=None):
    """Whitelist MCL and the image location; never retain source payloads or GT fields.

    Older caches may omit MCL: an explicit, validated run setting is required
    then. Never let a frozen detector silently substitute its own default.
    """
    actual = payload.get("min_cable_length")
    for value in (actual, expected_mcl):
        if value is not None and (isinstance(value, bool)
                                  or not isinstance(value, Integral) or value < 0):
            raise ValueError("min_cable_length must be a nonnegative integer")
    if actual is not None and expected_mcl is not None and actual != expected_mcl:
        raise ValueError("Detector min_cable_length disagrees with requested MCL")
    value = actual if actual is not None else expected_mcl
    if value is None:
        raise ValueError("Explicit min_cable_length is required for detector inference")
    metadata = {"min_cable_length": int(value)}
    # The raw image location is GT-free and is the only way a detector's image features
    # can run on a table (a detector has no companion manifest). Absent or empty, the
    # detector leaves its image features undefined; it never substitutes a path.
    img_path = payload.get("img_path")
    if isinstance(img_path, str) and img_path:
        metadata["img_path"] = img_path
    return metadata

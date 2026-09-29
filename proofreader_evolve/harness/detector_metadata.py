"""Explicit, GT-free metadata for frozen-detector feature extraction."""

from numbers import Integral


def inference_metadata(payload, expected_mcl=None):
    """Whitelist only MCL; never retain source payloads or ground-truth fields.

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
    return {"min_cable_length": int(value)}

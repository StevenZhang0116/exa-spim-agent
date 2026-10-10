"""Driver-test helper: mocked native tables have no availability sidecars on disk.

``label_availability.apply`` (the real sidecar loader) is tested in
test_label_availability.py. Driver tests patch it for the whole module so their
mocked tables become three-valued without sidecars: every row stays labeled, or,
with ``nan_every``, every n-th label-0 row becomes NaN (not judgeable by GT).
"""
from unittest.mock import patch

import numpy as np

from proofreader_evolve.harness import label_availability


def labeled_tables(nan_every=None):
    def apply(banks):
        summary = {}
        for brain, bank in banks.items():
            for kind, table in bank.tables.items():
                if getattr(table, "availability", None) is None:
                    truth = np.asarray(table.truth, dtype=float).copy()
                    if nan_every:
                        zeros = np.flatnonzero(truth == 0)
                        truth[zeros[::nan_every]] = np.nan
                    table.truth = truth
                    table.availability = {"definition": "test", "version": label_availability.AVAILABILITY_VERSION,
                                          "path": None, "counts": label_availability.label_counts(truth)}
                summary[f"{brain}/{kind}"] = dict(table.availability)
        return summary
    return patch.object(label_availability, "apply", side_effect=apply)


_active = []


def setUpModule():
    _active.append(labeled_tables())
    _active[-1].start()


def tearDownModule():
    while _active:
        _active.pop().stop()

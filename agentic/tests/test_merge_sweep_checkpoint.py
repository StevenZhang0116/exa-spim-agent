import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


NOTEBOOKS = Path(__file__).resolve().parents[2] / "notebooks"
with patch.object(sys, "path", [str(NOTEBOOKS), *sys.path]):
    import merge_candidate_pool_sweep as sweep


class MergeSweepCheckpointTests(unittest.TestCase):
    def test_reuse_requires_same_cache_and_complete_current_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "dataset_cache_794495_mcl100_add.pkl"
            cache.write_bytes(b"first cache")
            run = root / "run"
            result = (pd.DataFrame({"n_candidates": [3]}), pd.DataFrame({"site_index": [0]}), {})
            with patch.object(sweep, "load_payload", return_value={}), \
                    patch.object(sweep, "sweep_payload", side_effect=lambda *args: result[:2] + ({},)) as compute:
                sweep.sweep_brain(cache, sweep.SweepConfig(), run)
                sweep.sweep_brain(cache, sweep.SweepConfig(), run)
                self.assertEqual(compute.call_count, 1)
                cache.write_bytes(b"refreshed cache with new sites")
                sweep.sweep_brain(cache, sweep.SweepConfig(), run)
                self.assertEqual(compute.call_count, 2)
                meta_path = run / "per_brain" / "794495_meta.json"
                meta = json.loads(meta_path.read_text())
                del meta["cache_identity"]
                meta_path.write_text(json.dumps(meta))
                sweep.sweep_brain(cache, sweep.SweepConfig(), run)
                self.assertEqual(compute.call_count, 3)
                (run / "per_brain" / "794495_site_distances.csv").unlink()
                sweep.sweep_brain(cache, sweep.SweepConfig(), run)
                self.assertEqual(compute.call_count, 4)

    def test_changed_cache_during_sweep_is_not_checkpointed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "dataset_cache_794495_mcl100_add.pkl"
            cache.write_bytes(b"old")

            def compute(*args):
                cache.write_bytes(b"new cache")
                return pd.DataFrame({"count": [1]}), pd.DataFrame(), {}

            with patch.object(sweep, "load_payload", return_value={}), \
                    patch.object(sweep, "sweep_payload", side_effect=compute), \
                    self.assertRaisesRegex(RuntimeError, "changed during sweep"):
                sweep.sweep_brain(cache, sweep.SweepConfig(), root / "run")
            self.assertFalse((root / "run" / "per_brain" / "794495_meta.json").exists())


if __name__ == "__main__":
    unittest.main()
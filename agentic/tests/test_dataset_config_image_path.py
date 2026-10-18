"""Tests of the image-path resolver in ``scripts/dataset_config.py``: a reviewed alignment URI
wins over a cache's stored ``img_path``; registrations other than identity are refused."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import dataset_config  # noqa: E402

IDENTITY = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


class ReviewedImagePathTests(unittest.TestCase):
    def registry(self, directory, brains):
        path = Path(directory) / "image_alignment.json"
        path.write_text(json.dumps({"version": 1, "brains": brains}))
        return str(path)

    def test_reviewed_uri_wins_and_points_at_level_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.registry(directory, {
                "111": {"status": "reviewed", "image_uri": "s3://b/live/fused.zarr/", "graph_to_world_um": IDENTITY},
                "222": {"status": "pending", "image_uri": "s3://b/unreviewed/fused.zarr/"},
                "333": {"status": "reviewed", "image_uri": "s3://b/shifted/fused.zarr/",
                        "graph_to_world_um": [[1, 0, 0, 5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
            })
            resolve = dataset_config.get_reviewed_img_path
            self.assertEqual(resolve("111", fallback="s3://b/dead/0", alignment_path=path), "s3://b/live/fused.zarr/0")
            self.assertEqual(resolve(111, alignment_path=path), "s3://b/live/fused.zarr/0")
            self.assertEqual(resolve("222", fallback="cached", alignment_path=path), "cached")
            self.assertEqual(resolve("444", fallback="cached", alignment_path=path), "cached")
            self.assertEqual(resolve("444", fallback="cached", alignment_path=str(Path(directory) / "none.json")),
                             "cached")
            with self.assertRaises(ValueError):
                resolve("333", fallback="cached", alignment_path=path)

    def test_cache_file_name_supplies_the_brain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.registry(directory, {"111": {"status": "reviewed", "image_uri": "s3://b/live/fused.zarr"}})
            payload = {"img_path": "s3://b/dead/0"}
            resolve = lambda name: dataset_config.get_cache_img_path(name, payload, alignment_path=path)  # noqa: E731
            self.assertEqual(resolve("cache/dataset_cache_111_mcl100_add.pkl"), "s3://b/live/fused.zarr/0")
            self.assertEqual(resolve(Path("x/dataset_cache_999_mcl10.pkl")), "s3://b/dead/0")
            self.assertEqual(resolve("other.pkl"), "s3://b/dead/0")


if __name__ == "__main__":
    unittest.main()

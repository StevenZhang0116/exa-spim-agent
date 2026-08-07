from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

AGENTIC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENTIC_DIR))

from rank_by_surprise import (  # noqa: E402
    load_predictive_ids,
    main as rank_main,
    rank_records,
    resolve_paths,
)
from rerun_experiments import main as rerun_main  # noqa: E402


class PredictiveDirectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {
                "id": 1,
                "surprisal": 0.4,
                "prior": 0.4,
                "posterior": 0.8,
                "_run": "test",
            },
            {
                "id": 2,
                "surprisal": -0.7,
                "prior": 0.8,
                "posterior": 0.2,
                "_run": "test",
            },
            {
                "id": 3,
                "surprisal": -0.2,
                "prior": 0.8,
                "posterior": 0.7,
                "_run": "test",
            },
            {
                "id": 4,
                "surprisal": None,
                "prior": None,
                "posterior": None,
                "_run": "test",
            },
        ]

    def test_predictive_direction_uses_manifest_ids_not_belief_direction(self) -> None:
        ranked, dropped, excluded = rank_records(
            self.records,
            direction="predictive",
            predictive_ids={"2", "3", "4"},
        )

        self.assertEqual([record["id"] for record in ranked], [2, 3, 4])
        self.assertEqual(dropped, [])
        self.assertEqual([record["id"] for record in excluded], [1])

    def test_predictive_direction_requires_manifest_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "selection manifest"):
            rank_records(self.records, direction="predictive")

    def test_load_predictive_ids_normalizes_id_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "selection.json"
            manifest.write_text(json.dumps({"selected_ids": [2, "feature-3"]}))

            self.assertEqual(load_predictive_ids(manifest), {"2", "feature-3"})

    def test_directory_discovery_ignores_predictive_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            run = directory / "run.json"
            manifest = directory / "run.predictive-selection.json"
            run.write_text("[]")
            manifest.write_text(json.dumps({"selected_ids": []}))

            self.assertEqual(resolve_paths([directory]), [run])

    def test_predictive_rank_rejects_top_cutoff(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                rank_main(
                    [
                        "--direction",
                        "predictive",
                        "--predictive-manifest",
                        "unused.json",
                        "--top",
                        "1",
                    ]
                )
        self.assertEqual(raised.exception.code, 2)

    def test_predictive_rerun_rejects_top_cutoff(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                rerun_main(
                    [
                        "unused-run.json",
                        "--pkl",
                        "unused.pkl",
                        "--direction",
                        "predictive",
                        "--predictive-manifest",
                        "unused-selection.json",
                        "--top",
                        "1",
                    ]
                )
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

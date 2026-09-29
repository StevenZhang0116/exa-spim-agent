import contextlib
import hashlib
import io
import tempfile
import unittest
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from notebooks import compare_add_cache_metrics_across_datasets as compare


def rows(brain="111", mcl=100, value=1.):
    row = {"brain_id": brain, "neuron": "N001", "min_cable_length": mcl}
    row.update({f"{column}_{kind}": value for column in compare.COMPARE_COLS
                for kind in ("cache", "canon")})
    return pd.DataFrame([row])


class CompareCacheMetricsTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()

    def tearDown(self):
        self.redirect.__exit__(None, None, None)

    def test_legacy_mcl_summary_and_named_row_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows(value=9.).drop(columns="min_cable_length").to_csv(root / "111_per_neuron.csv", index=False)
            pd.DataFrame([{"brain_id": "111", "min_cable_length": 100}]).to_csv(
                root / "111_summary.csv", index=False)
            rows(value=2.).drop(columns="min_cable_length").to_csv(root / "111_mcl100_per_neuron.csv", index=False)
            rows(mcl=10, value=3.).to_csv(root / "111_mcl10_per_neuron.csv", index=False)
            legacy = compare.load_per_neuron(root / "111_per_neuron.csv")
            self.assertEqual(legacy["min_cable_length"].iloc[0], 100)
            self.assertFalse(legacy["_mcl_named"].iloc[0])
            combined = compare.load_saved_results(root)
            self.assertEqual(len(combined), 2)
            self.assertEqual(combined.loc[combined["min_cable_length"] == 100, "# Splits_cache"].iloc[0], 2.)
            self.assertNotIn("_mcl_named", combined)

    def test_explicit_legacy_mcl_needs_no_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "111_per_neuron.csv"
            rows().to_csv(path, index=False)
            self.assertEqual(compare.load_per_neuron(path)["min_cable_length"].iloc[0], 100)

    def test_missing_legacy_summary_is_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "111_per_neuron.csv"
            rows().drop(columns="min_cable_length").to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "matching"):
                compare.load_per_neuron(path)

    def test_invalid_schema_mcl_and_brain_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "111_mcl100_per_neuron.csv"
            invalid = [rows().drop(columns="neuron"), rows(mcl=10), rows(mcl=10.5),
                       rows(brain="../111"), rows().iloc[:0]]
            for frame in invalid:
                with self.subTest(columns=list(frame.columns), values=frame.to_dict()):
                    frame.to_csv(path, index=False)
                    with self.assertRaises(ValueError):
                        compare.load_per_neuron(path)

    def test_all_figures_saved_and_inputs_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for brain, mcl in (("111", 100), ("111", 10), ("222", 100), ("222", 10), ("333", 100)):
                rows(brain, mcl).to_csv(root / f"{brain}_mcl{mcl}_per_neuron.csv", index=False)
            old_scatter = root / "111_mcl100_scatter.png"
            old_scatter.write_bytes(b"original verifier scatter")
            before = {path: hashlib.sha256(path.read_bytes()).digest() for path in root.iterdir()}
            self.assertEqual(compare.main(["--stats-dir", str(root), "--dpi", "60"]), 0)
            expected = {"compare_mcl100_across_brains.png", "compare_mcl10_across_brains.png",
                        "compare_111_mcl100_vs_mcl10.png", "compare_222_mcl100_vs_mcl10.png"}
            self.assertEqual({path.name for path in root.glob("compare_*.png")}, expected)
            self.assertIn("SKIP 333: missing mcl10", self.output.getvalue())
            for name in expected:
                image = mpimg.imread(root / name)
                self.assertGreater(image.shape[0], 100)
                self.assertGreater(np.std(image[:, :, :3]), 0.01)
            for path, digest in before.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), digest)

    def test_output_override_and_single_mcl_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = root / "stats"
            stats.mkdir()
            rows().to_csv(stats / "111_mcl100_per_neuron.csv", index=False)
            output = root / "figures"
            self.assertEqual(compare.main(["--stats-dir", str(stats), "--output-dir", str(output),
                                           "--dpi", "60"]), 0)
            self.assertEqual([path.name for path in output.iterdir()], ["compare_mcl100_across_brains.png"])
            self.assertFalse(list(stats.glob("*.png")))

    def test_plot_filters_nonfinite_values_and_keeps_legend(self):
        frame = rows()
        frame["# Splits_cache"] = np.nan
        frame["% Omit Edges_canon"] = np.inf
        figure = compare.plot_metric_grid(frame, "brain_id", ["111"], {"111": "red"}, "test", "brain")
        self.assertEqual(len(figure.axes), 5)
        self.assertEqual(figure.axes[0].texts[0].get_text(), "No comparable data")
        self.assertEqual(figure.legends[0].texts[0].get_text(), "111")
        self.assertEqual(len(figure.axes[1].collections[0].get_offsets()), 1)
        np.testing.assert_allclose(figure.axes[1].lines[0].get_xdata(), [0.5, 1.5])
        figure.canvas.draw()
        self.assertGreater(np.std(np.asarray(figure.canvas.buffer_rgba())[:, :, :3]), 1.)
        figure.clear()

    def test_no_csv_and_invalid_dpi_fail_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            for options in ([], ["--dpi", "0"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    compare.main(["--stats-dir", directory, *options])
                self.assertEqual(error.exception.code, 2)
            self.assertFalse(list(Path(directory).iterdir()))

    def test_no_supported_mcl_produces_no_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows(mcl=20).to_csv(root / "111_mcl20_per_neuron.csv", index=False)
            self.assertEqual(compare.main(["--stats-dir", str(root)]), 1)
            self.assertFalse(list(root.glob("*.png")))


if __name__ == "__main__":
    unittest.main()
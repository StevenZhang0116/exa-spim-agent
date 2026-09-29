"""Generic cost checks, with synthetic inputs only; no brain caches or APIs."""

import ast
from pathlib import Path
import textwrap
import unittest
from unittest.mock import patch

import numpy as np

from agentic.detector_build.cost_validation import analyze_row_cost
from agentic.detector_build.assembly import _validate_row_path_cost


def fragment(row, setup="", helpers="", prelude=""):
    return ("import numpy as np\n" + textwrap.dedent(prelude) + textwrap.dedent(helpers)
            + "\ndef extract_features(payload, timing=None):\n"
            + "    geometry = payload['fragments_graph']\n"
            + "    coordinates = np.asarray(geometry.node_xyz)\n"
            + "    samples, labels = build_sample_universe(payload)\n"
            + textwrap.indent(textwrap.dedent(setup), "    ") + "\n"
            + "    for sample in samples:\n"
            + textwrap.indent(textwrap.dedent(row), "        ") + "\n")


class CostDataflowTests(unittest.TestCase):
    def reject(self, source, message):
        report = analyze_row_cost(ast.parse(source))
        self.assertEqual(report.status, "rejected", report)
        self.assertIn(message, "\n".join(report.violations))

    def accept(self, source):
        report = analyze_row_cost(ast.parse(source))
        self.assertFalse(report.violations, report)
        return report

    def test_global_operations_through_aliases_and_multidimensional_slices(self):
        cases = [
            "column = coordinates[:, 2]\nvalue = np.min(column)",
            "column = coordinates[..., 2]\nvalue = column.max()",
            "column = coordinates[:, 1]\nvalue = np.mean(a=column)",
            "renamed = coordinates\nvalue = np.percentile(renamed[:, 2], 50)",
            "value = np.sort(coordinates[:, 2])",
            "value = np.unique(coordinates[:, 2])",
            "value = np.min(coordinates[::2, 2])",
            "mask = coordinates[:, 2] > 0\nvalue = coordinates[mask]",
            "value = cKDTree(coordinates)",
            "value = low(coordinates[:, 2])",
            "total = 0\ntotal += np.sum(coordinates[:, 2])",
            "value = np.min(column := coordinates[:, 2])",
        ]
        for row in cases:
            with self.subTest(row=row):
                self.reject(fragment(row, prelude="from numpy import min as low\n"), "repeated global")

    def test_local_slices_and_precomputed_scalars_are_allowed(self):
        for row, setup in [
            ("value = np.min(coordinates[sample['node'], :])", ""),
            ("value = np.max(coordinates[:32, 2])", ""),
            ("value = cached_z", "cached_z = np.min(coordinates[:, 2])"),
            ("value = np.min(coordinates[[0, 1, 2], 2])", ""),
        ]:
            with self.subTest(row=row):
                self.accept(fragment(row, setup))

    def test_helpers_parameters_returns_and_instance_attributes(self):
        self.reject(fragment("value = outer(coordinates)", helpers="""
            def inner(points):
                return np.sum(points[:, 1])
            def outer(renamed):
                return inner(renamed)
        """), "outer -> inner")
        self.reject(fragment("value = np.max(column(coordinates))", helpers="""
            def column(values):
                return values[:, 2]
        """), "repeated global")
        self.reject(fragment("value = context.boundary()", "context = Context(geometry)", helpers="""
            class Context:
                def __init__(self, graph):
                    self.any_name = np.asarray(graph.node_xyz)
                def boundary(self):
                    return np.min(self.any_name[:, 2])
        """), "boundary")
        self.accept(fragment("value = context.boundary()", "context = Context(geometry)", helpers="""
            class Context:
                def __init__(self, graph):
                    self.minimum = np.min(graph.node_xyz[:, 2])
                def boundary(self):
                    return self.minimum
        """))

    def test_reductions_preserve_remaining_global_axes(self):
        self.reject(fragment("value = np.sum(lengths)",
                             "lengths = np.mean(coordinates, axis=1)"), "repeated global")
        self.accept(fragment("value = lengths[sample['node']]",
                             "lengths = np.mean(coordinates, axis=1)"))
        self.accept(fragment("value = np.sum(center)",
                             "center = np.mean(coordinates, axis=0)"))

    def test_variable_names_do_not_control_classification(self):
        for variable in ("points", "unrelated_name", "data_17"):
            source = fragment("value = np.min(coordinates[:, 2])").replace("coordinates", variable)
            self.reject(source, "payload.fragments_graph.node_xyz")

    def test_local_method_arguments_are_not_mistaken_for_whole_graph(self):
        self.accept(fragment("value = context.reduce(coordinates[sample['node'], :])",
                             "context = Context()", helpers="""
            class Context:
                def reduce(self, data):
                    return data.sum()
        """))

    def test_persistent_global_cache_must_be_warmed(self):
        helper = """
            def bounds():
                return _memoized(cache, 'global', lambda: np.min(coordinates[:, 2]))
        """
        setup = "cache = {}\n" + textwrap.dedent(helper)
        self.reject(fragment("value = bounds()", setup), "repeated global")
        self.accept(fragment("value = bounds()", setup + "\nbounds()\n"))
        self.reject(fragment("value = bounds()", setup + "\nif payload['warm']:\n    bounds()\n"),
                    "repeated global")
        self.reject(fragment("value = _memoized(cache, sample['id'], lambda: np.max(coordinates[:, 2]))",
                             "cache = {}"), "repeated global")
        self.reject(fragment("cache.clear()\nvalue = bounds()", setup + "\nbounds()\n"),
                    "repeated global")
        self.reject(fragment("del cache['global']\nvalue = bounds()", setup + "\nbounds()\n"),
                    "repeated global")

    def test_prewarming_a_different_constant_key_does_not_hide_global_work(self):
        setup = """
            cache = {}
            def bounds(key):
                return _memoized(cache, key, lambda: np.min(coordinates[:, 2]))
            bounds('one')
        """
        self.reject(fragment("value = bounds('two')", setup), "repeated global")
        self.accept(fragment("value = bounds('one')", setup))

    def test_cache_lifetime_and_result_determining_key_parameters(self):
        self.accept(fragment("cache = {}\nvalue = _memoized(cache, sample['id'], lambda: 1)"))
        report = self.accept(fragment("""
            cache = {}
            value = _memoized(cache, sample['component'], lambda:
                np.sum(node_connected_component(geometry, sample['component'])))
        """))
        self.assertEqual(report.status, "partial")
        self.reject(fragment("""
            value = _memoized(cache, sample['component'],
                              lambda: sample['component'] + sample['radius'])
        """, "cache = {}"), "key omits")
        self.accept(fragment("""
            value = _memoized(cache, (sample['component'], sample['radius']),
                              lambda: sample['component'] + sample['radius'])
        """, "cache = {}"))
        self.reject(fragment("""
            def compute():
                if sample['radius'] > 0:
                    return 1
                return 0
            value = _memoized(cache, sample['component'], compute)
        """, "cache = {}"), "key omits")

    def test_component_and_density_work_are_not_certified_as_constant(self):
        for row in [
            "nodes = node_connected_component(geometry, sample['node'])\nvalue = np.sum(nodes)",
            "nodes = tree.query_ball_point(sample['position'], 50)\nvalue = np.mean(nodes)",
            "value = opaque(coordinates)",
            "value = np.sum(opaque())",
        ]:
            with self.subTest(row=row):
                report = self.accept(fragment(row))
                self.assertEqual(report.status, "partial")
                self.assertTrue(report.unverified)

    def test_recursion_is_reported_as_unverified(self):
        report = self.accept(fragment("value = helper(sample)", helpers="""
            def helper(value):
                return helper(value)
        """))
        self.assertEqual(report.status, "partial")
        self.assertIn("recursive call", "\n".join(report.unverified))

    def test_worker_callback_is_checked_without_timing_hooks(self):
        source = """
            import numpy as np
            def extract_features(payload):
                points = payload['fragments_graph'].node_xyz
                samples, labels = build_sample_universe(payload)
                def worker(sample):
                    return np.min(points[:, 2])
                return _bounded_thread_map(worker, samples, 2)
        """
        self.reject(textwrap.dedent(source), "repeated global")

    def test_component_cache_is_distinct_from_global_scan_per_component(self):
        self.accept(fragment("""
            value = _memoized(cache, sample['component'], lambda:
                np.sum(node_connected_component(geometry, sample['component'])))
        """, "cache = {}"))
        self.reject(fragment("""
            value = _memoized(cache, sample['component'], lambda:
                np.sum(geometry.node_component_id == sample['component']))
        """, "cache = {}"), "repeated global")

    def test_budget_exhaustion_is_explicit(self):
        report = analyze_row_cost(ast.parse(fragment("value = 1")), max_steps=1)
        self.assertEqual(report.status, "partial")
        self.assertIn("budget exhausted", "\n".join(report.unverified))

    def test_opaque_globals_are_not_certified_as_scalar(self):
        report = self.accept(fragment("value = np.min(EXTERNAL_DATA)"))
        self.assertEqual(report.status, "partial")

    def test_builder_emits_partial_verification_diagnostic(self):
        source = fragment("value = opaque(coordinates)")
        with self.assertWarnsRegex(RuntimeWarning, "Partial row-cost verification"):
            _validate_row_path_cost(ast.parse(source), Path("generated.py"))

    def test_builder_gate_rejects_original_pattern(self):
        source = fragment("value = context.calculate()", "context = Context(geometry)", helpers="""
            class Context:
                def __init__(self, graph):
                    self.xyz = np.asarray(graph.node_xyz)
                def calculate(self):
                    return np.min(self.xyz[:, 2])
        """)
        with self.assertRaisesRegex(SystemExit, "repeated global"):
            _validate_row_path_cost(ast.parse(source), Path("generated.py"))

    def test_precompute_is_equivalent_and_global_call_count_is_constant(self):
        # Execute only these small, trusted fixtures; never generated detectors.
        def original(points, locations):
            return np.array([min(z - np.min(points[:, 2]), np.max(points[:, 2]) - z)
                             for z in locations])

        def cached(points, locations):
            lower, upper = np.min(points[:, 2]), np.max(points[:, 2])
            return np.array([min(z - lower, upper - z) for z in locations])

        for n in (8, 40):
            points = np.arange(n * 3, dtype=float).reshape(n, 3)
            for count in (1, 25):
                locations = np.linspace(0, n * 3, count)
                expected = original(points, locations)
                with patch.object(np, "min", wraps=np.min) as low, \
                        patch.object(np, "max", wraps=np.max) as high:
                    actual = cached(points, locations)
                    self.assertEqual(low.call_count, 1)
                    self.assertEqual(high.call_count, 1)
                np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()

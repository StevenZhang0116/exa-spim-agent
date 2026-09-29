import ast
import tempfile
import unittest
from pathlib import Path

from agentic.detector_build.assembly import (
    _validate_accumulator_set_calls, validate_feature_implementation,
)
from agentic.detector_build.contracts import DetectorTarget


def fragment(signature, call, setup="acc = FeatureAccumulator()"):
    return f'''FEATURE_REGISTRY = [("value",), ("other",)]
class FeatureAccumulator:
    def set(self, {signature}):
        pass
def extract_features(payload):
    {setup}
    segment = 32323215387
    {call}
'''


class AccumulatorContractTests(unittest.TestCase):
    def validate(self, source):
        _validate_accumulator_set_calls(ast.parse(source), Path("feature.py"))

    def test_rejects_reversed_feature_first_call(self):
        source = fragment("segment_id, feature_name, value", 'acc.set("value", segment, 1.5)')
        with self.assertRaisesRegex(SystemExit, "argument order mismatch"):
            self.validate(source)

    def test_accepts_consistent_positional_and_keyword_orders(self):
        cases = [
            ("feature_name, segment_id, value", 'acc.set("value", segment, 1.5)'),
            ("segment_id, feature_name, value", 'acc.set(segment, "value", 1.5)'),
            ("segment_id, feature_name, value", 'acc.set(feature_name="value", segment_id=segment, value=1.5)'),
            ("*, feature_name, segment_id, value", 'acc.set(feature_name="value", segment_id=segment, value=1.5)'),
            ("feature_name, /, segment_id, value=0", 'acc.set("value", segment)'),
        ]
        for signature, call in cases:
            with self.subTest(signature=signature, call=call):
                self.validate(fragment(signature, call))

    def test_rejects_missing_duplicate_arguments_and_unknown_feature(self):
        for call in ('acc.set("value", segment)', 'acc.set("value", segment, 1., feature_name="value")',
                     'acc.set("not_in_registry", segment, 1.)', 'acc.set(123, segment, 1.)'):
            with self.subTest(call=call), self.assertRaises(SystemExit):
                self.validate(fragment("feature_name, segment_id, value", call))

    def test_detects_alias_and_closure_calls(self):
        source = fragment("segment_id, feature_name, value", '''alias = acc
    def write():
        alias.set("value", segment, 1.5)
    write()''')
        with self.assertRaisesRegex(SystemExit, "argument order mismatch"):
            self.validate(source)

    def test_shadowed_or_unrelated_receivers_are_not_accumulators(self):
        source = fragment("segment_id, feature_name, value", '''acc = payload
    acc.set("value", segment, 1.5)''')
        self.validate(source)
        self.validate(source + '\ndef unrelated(acc):\n    acc.set("value", 0, 1.)\n')

    def test_dynamic_feature_names_are_not_falsely_rejected(self):
        self.validate(fragment("feature_name, segment_id, value", 'acc.set(payload["feature"], segment, 1.5)'))

    def test_contract_is_enforced_by_feature_validation_before_assembly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feature.py"
            path.write_text(fragment("segment_id, feature_name, value", 'acc.set("value", segment, 1.5)'))
            with self.assertRaisesRegex(SystemExit, "argument order mismatch"):
                validate_feature_implementation(path, required_symbols={"FEATURE_REGISTRY", "FeatureAccumulator",
                                                                       "extract_features"},
                                                target=DetectorTarget.MERGE_SITE)


if __name__ == "__main__":
    unittest.main()
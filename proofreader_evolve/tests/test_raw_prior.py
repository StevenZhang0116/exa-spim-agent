"""--detector-prior raw (2026-10-27): the frozen score and every discovered h<n>_ feature column are
withheld; the feature frame is rebuilt from the label-blind candidate attributes. Synthetic
fixtures only; no brain caches, cloud reads or LLM calls."""

from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from proofreader_evolve.harness import detector_prior
from proofreader_evolve.harness.collect_run import collect
from proofreader_evolve.harness.native_pool import NativeTable, NativeTables, pool_digest
from proofreader_evolve.tests.test_detector_prior import parse, run_driver
from proofreader_evolve.tests.label_fixture import setUpModule, tearDownModule  # noqa: F401

HERE = Path(__file__).resolve().parents[1]


def raw_merge_table(brain, *, groups=8, offset=0.):
    """Like test_selection_protocol.brain_table, with the candidate attributes of candidate_rules
    and one discovered feature column; positives are tip_to_shaft rows with a small tip gap."""
    rows, labels = [], []
    for group in range(groups):
        for position in range(2):
            index = len(rows)
            label = position
            rows.append({'candidate_id': index, 'node_id': index, 'segment_id': group + 1, 'degree': 3,
                         'x_um': 10. * index, 'y_um': 0., 'z_um': 0.,
                         'candidate_family': 'tip_to_shaft', 'tip_gap_um': 3. + index if label else 40. + index})
            labels.append(label)
    frame = pd.DataFrame({'detector_score': np.linspace(1., 0., len(rows)),
                          'h3_caliber_ratio_15um': np.linspace(0., 1., len(rows)),
                          'signal': np.asarray(labels, dtype=float) + offset})
    table = NativeTable('merge', frame, rows, np.asarray(labels, dtype=np.int8),
                        {'pool_sha256': pool_digest('merge', rows), 'training_brain': '794495',
                         'provenance': {'brain': brain}})
    return NativeTables(brain, {'merge': table}, {})


def raw_banks():
    banks = {'794495': raw_merge_table('794495', offset=1000.), '802449': raw_merge_table('802449'),
             '2': raw_merge_table('2', offset=5000.)}
    return banks


SPLIT_ROWS = [
    {'candidate_id': 0, 'segment_id_a': 1, 'segment_id_b': 2, 'node_id_a': 10, 'node_id_b': 20, 'gap_um': 12.5,
     'anchor_side': 'a', 'partner_rank': 1, 'node_role_a': 'tip', 'node_role_b': 'non_tip',
     'occurrences': ({'node_id_a': 10, 'node_id_b': 20, 'gap_um': 12.5, 'anchor_side': 'a', 'partner_rank': 1,
                      'node_role_a': 'tip', 'node_role_b': 'non_tip'},
                     {'node_id_a': 11, 'node_id_b': 20, 'gap_um': 15., 'anchor_side': 'b', 'partner_rank': 2,
                      'node_role_a': 'tip', 'node_role_b': 'tip'})},
    {'candidate_id': 1, 'segment_id_a': 3, 'segment_id_b': 4, 'node_id_a': 30, 'node_id_b': 40, 'gap_um': 70.,
     'anchor_side': 'b', 'partner_rank': 2, 'node_role_a': 'non_tip', 'node_role_b': 'tip', 'occurrences': ()},
]


class RawColumnTests(unittest.TestCase):
    def test_merge_columns_follow_the_candidate_rows(self):
        table = raw_merge_table('1').tables['merge']
        rows = [dict(r) for r in table.candidates]
        rows[0]['candidate_family'], rows[0]['tip_gap_um'] = 'junction', None
        frame = detector_prior.raw_columns('merge', rows)
        self.assertEqual(list(frame.columns), list(detector_prior.RAW_COLUMNS['merge']))
        self.assertEqual(frame['raw_degree'].tolist(), [3.] * len(rows))
        self.assertEqual(frame['raw_candidate_family'].tolist()[:2], [0., 1.])
        self.assertTrue(math.isnan(frame['raw_tip_gap_um'][0]))
        self.assertEqual(frame['raw_tip_gap_um'][1], rows[1]['tip_gap_um'])
        rows[2]['candidate_family'] = 'bridge'
        with self.assertRaisesRegex(ValueError, 'candidate_family'):
            detector_prior.raw_columns('merge', rows)

    def test_split_columns_include_occurrence_aggregates(self):
        frame = detector_prior.raw_columns('split', SPLIT_ROWS)
        self.assertEqual(list(frame.columns), list(detector_prior.RAW_COLUMNS['split']))
        first, second = frame.iloc[0], frame.iloc[1]
        self.assertEqual((first['raw_gap_um'], first['raw_partner_rank'], first['raw_anchor_side']), (12.5, 1., 0.))
        self.assertEqual((first['raw_node_role_a'], first['raw_node_role_b']), (1., 0.))
        self.assertEqual((first['raw_occurrence_count'], first['raw_occurrence_gap_min_um'],
                          first['raw_occurrence_gap_max_um']), (2., 12.5, 15.))
        self.assertEqual((second['raw_anchor_side'], second['raw_node_role_a'], second['raw_occurrence_count']),
                         (1., 0., 0.))
        self.assertTrue(math.isnan(second['raw_occurrence_gap_min_um']))
        with self.assertRaises(ValueError):
            detector_prior.raw_columns('omit', [])

    def test_strip_to_raw_keeps_host_copies_and_converts_shared_tables_once(self):
        banks = raw_banks()
        banks['alias'] = banks['802449']
        table = banks['802449'].tables['merge']
        score, frame = table.features['detector_score'].to_numpy(copy=True), table.features
        cells = detector_prior.strip_to_raw(banks)
        self.assertEqual(cells, ['794495/merge', '802449/merge', '2/merge'])
        for bank in banks.values():
            self.assertEqual(list(bank.tables['merge'].features.columns), list(detector_prior.RAW_COLUMNS['merge']))
        np.testing.assert_array_equal(banks['alias'].tables['merge'].host_detector_score, score)
        self.assertIs(table.host_detector_features, frame)
        self.assertEqual(detector_prior.strip_to_raw(banks), [])

    def test_check_source_refuses_discovered_feature_names_only_under_raw(self):
        detector_prior.check_source("x = features['h3_caliber_ratio_15um']\n", 'scorer')
        with self.assertRaisesRegex(ValueError, 'h3_caliber_ratio_15um.*raw'):
            detector_prior.check_source("x = features['h3_caliber_ratio_15um']\n", 'scorer', raw=True)
        detector_prior.check_source("x = features['raw_gap_um']\n", 'scorer', raw=True)
        detector_prior.check_source(detector_prior.raw_heuristic_source(), 'heuristic', raw=True)
        self.assertTrue(detector_prior.is_detector_feature('h87_caliber_scaled_gap'))
        self.assertFalse(detector_prior.is_detector_feature('raw_gap_um'))
        self.assertFalse(detector_prior.is_detector_feature('bank_agent_h1'))


class ArgumentTests(unittest.TestCase):
    def test_raw_defaults_and_bank_refusals(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        raw = driver.parse_args(['--detector-prior', 'raw'])
        self.assertEqual(raw.start_from, HERE / 'artifacts' / 'scorer_no_score.py')
        self.assertEqual(raw.descriptor_bank, HERE / 'descriptor_bank_raw')
        self.assertTrue(driver._contract(raw).endswith(detector_prior.NOTICE_RAW))
        self.assertIn('raw_tip_gap_um', driver._contract(raw))
        no_score = driver.parse_args(['--detector-prior', 'no-score'])
        self.assertTrue(driver._contract(no_score).endswith(detector_prior.NOTICE))
        self.assertEqual(driver._contract(driver.parse_args([])), driver.CONTRACT)
        for bank in ('descriptor_bank', 'descriptor_bank_no_score'):
            with self.subTest(bank=bank), redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO()), \
                    self.assertRaises(SystemExit):
                driver.parse_args(['--detector-prior', 'raw', '--descriptor-bank', str(HERE / bank)])
        with redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
            driver.parse_args(['--detector-prior', 'no-score', '--descriptor-bank', str(HERE / 'descriptor_bank_raw')])


class DriverTests(unittest.TestCase):
    def test_raw_run_withholds_the_columns_and_records_both_baselines(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        banks = raw_banks()
        seen = {}
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = parse(driver, tmp, '--detector-prior', 'raw')

            async def revise(run_dir, policy, rules, report, model, *, experiments):
                seen['rules'] = rules.read_text()
                seen['visible'] = {p.name: p.read_text() for p in policy.parent.glob('*.json')}
                policy.write_text("def score_candidates(features, ctx):\n"
                                  "    assert list(features.columns) == "
                                  "['raw_degree', 'raw_candidate_family', 'raw_tip_gap_um']\n"
                                  "    return -features['raw_tip_gap_um'].fillna(1e6).to_numpy()\n")
                experiments.proposal_path.write_text(json.dumps({'hypothesis': 'small tip gaps are merges',
                    'strategy': 'rank by the raw tip gap', 'family': 'raw-gap', 'parameter_grid': {}}))
                experiments.evaluate()
                return {'summary': 'raw gap', 'cost_usd': 0}

            path = run_driver(driver, args, banks, revise)
            manifest = json.loads((path / 'manifest.json').read_text())
            baseline = json.loads((path / 'baseline.json').read_text())
            seed = json.loads((path / 'seed.json').read_text())
            record = json.loads((path / 'ledger.jsonl').read_text().splitlines()[0])
            split = json.loads((path / 'dataset_split.json').read_text())
            summary = collect(path)
        prior = manifest['detector_prior']
        self.assertEqual(prior['mode'], 'raw')
        self.assertEqual(prior['stripped_cells'], ['794495/merge', '802449/merge', '2/merge'])
        self.assertEqual(prior['raw_columns']['merge'], list(detector_prior.RAW_COLUMNS['merge']))
        self.assertEqual(split['detector_prior'], 'raw')
        self.assertEqual(summary['detector_prior'], 'raw')
        self.assertIn('Raw-candidate ablation', seen['rules'])
        for name, text in seen['visible'].items():
            with self.subTest(file=name):
                self.assertNotIn('detector_score', text)
                self.assertNotIn('h3_caliber_ratio_15um', text)
        # Frozen scores descend by row and labels alternate 0 / 1: the frozen Top 3 hits once; the raw
        # heuristic (degree, then the smaller tip gap) and the agent's gap ranking put the positives first.
        self.assertAlmostEqual(baseline['validation']['cells']['2/merge']['precision'], 1 / 3)
        self.assertEqual(baseline['raw_heuristics']['validation']['cells']['2/merge']['precision'], 1.)
        self.assertEqual(baseline['raw_heuristics']['train']['cells']['802449/merge']['precision'], 1.)
        self.assertAlmostEqual(seed['validation']['cells']['2/merge']['precision'], 1 / 3)  # constant seed, id order
        self.assertEqual(record['validation']['cells']['2/merge']['precision'], 1.)
        self.assertTrue(record['accepted'])
        for bank in banks.values():
            self.assertEqual(list(bank.tables['merge'].features.columns), list(detector_prior.RAW_COLUMNS['merge']))

    def test_a_seed_reading_a_discovered_feature_is_refused(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            seed = Path(tmp) / 'seed.py'
            seed.write_text("def score_candidates(features, ctx):\n"
                            "    return features['h3_caliber_ratio_15um'].to_numpy()\n")
            args = parse(driver, tmp, '--detector-prior', 'raw', '--start-from', str(seed))
            with self.assertRaisesRegex(ValueError, 'Starting scorer .* references h3_caliber_ratio_15um'):
                run_driver(driver, args, raw_banks(), None)


if __name__ == '__main__':
    unittest.main()

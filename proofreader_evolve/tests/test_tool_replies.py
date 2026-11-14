"""Tool replies fit the size budget and keep the conclusions (2026-11-13).

The CLI hands a session a reply above about 50 KB as a 2 KB preview; the records here have the
shape and size of the replies of the 2026-11-12 runs (six TRAIN brains, 26 descriptor columns).
Synthetic values only; no brain caches, cloud reads or LLM calls."""

from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from proofreader_evolve.harness import tool_replies as replies
from proofreader_evolve.harness.train_experiments import TrainingExperiments

BRAINS = ('747807', '750318', '754610', '754612', '754613', '802449')
STATISTICS = ('auroc', 'pauc', 'average_precision', 'precision_depth_mean', 'precision', 'precision_at_half_k',
              'precision_at_double_k')
PARKED_AT = 50_000  # the size from which the CLI was observed to hand over a preview


def cell(index):
    return {f'field_{number:02d}': 0.123456789 + index + number for number in range(25)}


def report(kinds=('merge',)):
    return {'per_kind_precision': {kind: 0.18 for kind in kinds}, 'objective': 'average_precision',
            'per_kind_objective': {kind: 0.150271357756 for kind in kinds},
            'per_kind_statistics': {kind: {name: 0.123456789012 for name in STATISTICS} for kind in kinds},
            'per_kind_by_family': {'merge': {family: {name: 0.123456789012 for name in ('auroc', 'average_precision')}
                                             for family in ('junction', 'tip_to_shaft')}},
            'aggregation': 'per-kind-positive-weighted-precision',
            'cells': {f'{brain}/{kind}': cell(index) for index, brain in enumerate(BRAINS) for kind in kinds}}


def comparison():
    pair = {'parent': 0.150271357756, 'candidate': 0.140863260070, 'delta': -0.009408097686}
    return {'diagnostic_only': True, 'reference': 'accepted_parent', 'objective': 'average_precision',
            'aggregation': 'per-kind-positive-weighted-precision', 'overall': pair, 'holdout': pair,
            'overall_precision': pair, 'holdout_precision': pair, 'warnings': [],
            'per_brain': {f'{brain}/merge': {**pair, 'positives': 120, 'precision_parent': 0.2, 'precision_candidate': 0.21}
                          for brain in BRAINS}}


def gate():
    return {'metric': 'grouped_cv_average_precision', 'objective': 'average_precision', 'mode': 'grouped_oof',
            'target_kind': 'merge', 'parent': 0.150271357756, 'candidate': 0.140863260070, 'delta': -0.0094080976,
            'resampling': {'status': 'ok', 'interval': [-0.0132070067, -0.0029713080], 'interval_level': 0.8,
                           'beyond_noise': False, 'draws': 1000, 'unit': 'fragment segment'}}


def feedback():
    return {'format': 'stratified-train-v8-selection-comparison', 'reading_guide': 'g' * 4300, 'budgets': {'merge': 3},
            'k_rule': {'k_rule': 'k' * 900}, 'target_kind': 'merge', 'selection_comparison': comparison(),
            'candidate_train': report(('merge', 'split')), 'candidate_selection': report(),
            'selection_protocol': {'partitions': 'p' * 4300}, 'examples_per_group_limit': 4, 'columns_pruned': False,
            'examples_omitted_cells': [],
            'examples': {f'{brain}/merge': {'candidate_refs': [f'ctx:{brain}:merge:{row}' for row in range(40)],
                                            'features': {'bank_agent_column': [0.5] * 40}, 'rows': 'x' * 9000}
                         for brain in BRAINS},
            'ranking_delta': {f'{brain}/merge': {'gained_positives': 1, 'lost_positives': 2, 'top_k_overlap': 0.9}
                              for brain in BRAINS},
            'attempts': [], 'relevant_experiments': [], 'feature_statistics_file': None, 'history_attempts_omitted': 0}


def training():
    return {'config': {'parameters': {'n_estimators': 300}}, 'training_brains': list(BRAINS), 'training_rows': 27934,
            'input_columns': [f'bank_agent_column_{number:02d}' for number in range(29)],
            'training_output_tail': 'log line\n' * 800, 'implementation_sha256': {f'module_{n}': 'a' * 64 for n in range(13)},
            'artifact_files': {'model.pkl': 'b' * 64}, 'selection_fold_fits': [{'fold': fold} for fold in range(3)],
            'fit_snapshot': '/run/gen003/classifier_fits/fit001'}


def entry(number=1):
    return {'hypothesis': 'h' * 300, 'strategy': 's' * 150, 'family': 'image', 'parameter_grid': {}, 'target_kind': 'merge',
            'sequence': number, 'experiment': f'gen003/attempt{number:03d}', 'search_mode': 'explore',
            'snapshot': f'/run/gen003/experiments/attempt{number:03d}', 'train': report(('merge', 'split')),
            'train_gate': {'passed': False, 'reason': 'average precision regressed on 754612/merge'},
            'ranking_delta': {f'{brain}/{kind}': {'gained_positives': 1} for brain in BRAINS for kind in ('merge', 'split')},
            'status': 'evaluated', 'cached': False, 'classifier': {'parameters': {'n_estimators': 300}},
            'training': training(), 'in_sample_precision': 0.4, 'selection': report(),
            'selection_comparison': comparison(), 'selection_gate': gate(), 'target_precision': 0.179487179487,
            'target_objective': 0.140863260070, 'specialist_profile': {'slices': 'z' * 1300}, 'feature_evidence': [],
            'timings': {name: 12.345678 for name in ('fit_seconds', 'oof_seconds', 'score_seconds', 'diagnostics_seconds')}}


def column_summaries(columns=26, labels=True):
    record = {'computed_rows': 11524, 'pool_rows': 79621, 'finite_count': 11524, 'missing_count': 0,
              'finite_fraction': 0.98, 'nonzero_count': 11524, 'nonzero_fraction': 1.0, 'min': 2.0, 'max': 4.123456,
              'std': 0.494137, 'constant': False, 'q25_q50_q75': [2.0, 3.123456, 3.5],
              'examples': {'min': 'ctx:4a5c0bb113828f8e:merge:3', 'max': 'ctx:4a5c0bb113828f8e:merge:5168',
                           'nonzero': 'ctx:4a5c0bb113828f8e:merge:2'}}
    if labels:
        record.update({'unlabeled_count': 0, '0': {'count': 11396, 'finite': 11396, 'q25_q50_q75': [2.0, 3.0, 3.0]},
                       '1': {'count': 128, 'finite': 128, 'q25_q50_q75': [3.0, 3.5, 4.0]}})
    return {brain: {f'bank_agent_column_{number:02d}': deepcopy(record) for number in range(columns)} for brain in BRAINS}


class AttemptReplyTests(unittest.TestCase):
    def test_the_reply_fits_and_keeps_the_conclusions(self):
        record, detail = entry(), feedback()
        self.assertGreater(replies.size({**record, 'feedback': detail}), PARKED_AT)  # what was sent before
        untouched = deepcopy(record)
        reply = replies.attempt_reply(record, detail, '/run/gen003/experiments/attempt001/feedback.json')
        self.assertLess(replies.size(reply), replies.REPLY_BUDGET_CHARS)
        self.assertEqual(record, untouched)  # the harness's own record keeps its cells
        self.assertEqual(list(reply)[:2], ['experiment', 'status'])
        self.assertLess(list(reply).index('selection_gate'), list(reply).index('hypothesis'))
        self.assertEqual(reply['target_objective'], record['target_objective'])
        self.assertEqual(reply['selection_gate']['resampling'], record['selection_gate']['resampling'])
        self.assertEqual(set(reply['selection_comparison']['per_brain']), {f'{brain}/merge' for brain in BRAINS})
        digest = reply['feedback']
        self.assertEqual(digest['candidate_selection']['per_kind_by_family'], detail['candidate_selection']['per_kind_by_family'])
        self.assertEqual(digest['candidate_selection']['per_kind_statistics'], detail['candidate_selection']['per_kind_statistics'])
        self.assertEqual(digest['ranking_delta'], detail['ranking_delta'])
        self.assertEqual(digest['file'], '/run/gen003/experiments/attempt001/feedback.json')
        for moved in ('examples', 'reading_guide', 'k_rule', 'selection_protocol', 'candidate_train.cells'):
            self.assertIn(moved, digest['file_holds'])
        for key in ('examples', 'reading_guide', 'selection_protocol'):
            self.assertNotIn(key, digest)
        for carrier in (reply['train'], reply['selection'], digest['candidate_train'], digest['candidate_selection']):
            self.assertNotIn('cells', carrier)
        self.assertNotIn('implementation_sha256', reply['training'])
        self.assertLessEqual(len(reply['training']['training_output_tail']), replies.OUTPUT_TAIL_CHARS)
        self.assertTrue(reply['training']['training_output_tail'].endswith('log line\n'))

    def test_a_failed_fit_keeps_its_whole_output_tail(self):
        # A failure row carries the tail at its top level, not under a training summary, and passes through.
        failed = {'experiment': 'gen003/attempt002', 'status': 'training_error', 'training_output_tail': 't' * 8000}
        reply = replies.grid_reply({'candidates': [failed], 'best': None})
        self.assertEqual(len(reply['candidates'][0]['training_output_tail']), 8000)


class GridReplyTests(unittest.TestCase):
    def rows(self, count):
        keys = ('experiment', 'status', 'classifier', 'target_precision', 'target_objective', 'train_gate',
                'selection_gate', 'selection_comparison', 'timings')
        return [{key: entry(number)[key] for key in keys} for number in range(1, count + 1)]

    def best(self):
        restored = replies.attempt_reply(entry(), feedback(), '/run/gen003/experiments/attempt001/feedback.json')
        return {key: restored[key] for key in ('classifier', 'target_objective', 'target_precision', 'train_gate',
                                               'selection_gate', 'selection_comparison', 'feedback', 'training')}

    def test_a_full_generation_grid_fits(self):
        reply = replies.grid_reply({'status': 'trained', 'candidates': self.rows(8), 'best': self.best()})
        self.assertLess(replies.size(reply), replies.REPLY_BUDGET_CHARS)
        self.assertNotIn('candidates_trimmed', reply)
        for row in reply['candidates']:
            self.assertNotIn('per_brain', row['selection_comparison'])
            self.assertIn('overall', row['selection_comparison'])
            self.assertIn('resampling', row['selection_gate'])
        self.assertIn('per_brain', reply['best']['selection_comparison'])
        self.assertIn('per_kind_by_family', reply['best']['feedback']['candidate_selection'])

    def test_a_long_cached_grid_is_shortened_rather_than_parked(self):
        reply = replies.grid_reply({'status': 'trained', 'candidates': self.rows(40), 'best': self.best()})
        self.assertLess(replies.size(reply), replies.REPLY_BUDGET_CHARS)
        self.assertEqual(reply['candidates_trimmed'][0], 'timings')
        self.assertEqual(len(reply['candidates']), 40)
        self.assertTrue(all('target_objective' in row and 'experiment' in row for row in reply['candidates']))


class DescriptorReplyTests(unittest.TestCase):
    def test_a_descriptor_run_reports_one_digest_per_column(self):
        public = {'run': 'gen003/descriptor001', 'status': 'registered', 'summaries': column_summaries(),
                  'coverage': {brain: {'computed_rows': 11524} for brain in BRAINS}}
        self.assertGreater(replies.size(public), PARKED_AT)
        reply = replies.descriptor_reply(public, '/run/gen003/descriptor_runs/run001/result.json')
        self.assertLess(replies.size(reply), replies.REPLY_BUDGET_CHARS)
        self.assertIn('summaries', public)  # the caller's record is not changed
        self.assertNotIn('summaries', reply)
        self.assertEqual(reply['summaries_file'], '/run/gen003/descriptor_runs/run001/result.json')
        digest = reply['column_summary']['bank_agent_column_00']
        self.assertEqual(digest['finite_fraction_min'], 0.98)
        self.assertEqual((digest['min'], digest['max']), (2.0, 4.123))
        self.assertEqual(digest['median'], {brain: 3.123 for brain in BRAINS})
        self.assertEqual(digest['median_label0'], {brain: 3.0 for brain in BRAINS})
        self.assertEqual(digest['median_label1'], {brain: 3.5 for brain in BRAINS})
        self.assertEqual(digest['constant_in'], [])

    def test_a_failed_run_passes_through(self):
        public = {'run': 'gen003/descriptor001', 'status': 'execution_error', 'error': 'ValueError: x'}
        self.assertEqual(replies.descriptor_reply(public, 'result.json'), public)

    def probe(self, columns, cases):
        summaries = column_summaries(columns, labels=False)
        values = {f'bank_agent_column_{number:02d}': 0.123456789 for number in range(columns)}
        return {'status': 'probed', 'probe': 'gen003/probe001', 'previews': [],
                'brains': {brain: {'rows': 64, 'summaries': summaries[brain], 'wall_seconds': 1.0,
                                   'cases': [{'candidate_ref': f'ctx:{brain}:merge:{row}', 'values': dict(values)}
                                             for row in range(cases)]} for brain in BRAINS}}

    def test_a_probe_reports_the_digest_and_rounded_cases(self):
        response = self.probe(26, 3)
        self.assertGreater(replies.size(response), PARKED_AT)
        untouched = deepcopy(response)
        reply = replies.probe_reply(response, '/run/gen003/descriptor_probes/probe001/result.json')
        self.assertEqual(response, untouched)
        self.assertLess(replies.size(reply), replies.REPLY_BUDGET_CHARS)
        self.assertNotIn('cases_trimmed_to_per_brain', reply)
        self.assertEqual(len(reply['brains']['802449']['cases']), 3)
        self.assertEqual(reply['brains']['802449']['cases'][0]['values']['bank_agent_column_00'], 0.1235)
        self.assertNotIn('summaries', reply['brains']['802449'])
        self.assertNotIn('median_label1', reply['column_summary']['bank_agent_column_00'])
        self.assertEqual(reply['result_file'], '/run/gen003/descriptor_probes/probe001/result.json')

    def test_a_wide_probe_drops_cases_before_it_is_parked(self):
        reply = replies.probe_reply(self.probe(60, 3), 'result.json')
        self.assertLess(replies.size(reply), replies.REPLY_BUDGET_CHARS)
        self.assertIn(reply['cases_trimmed_to_per_brain'], (0, 1))
        self.assertEqual(len(reply['column_summary']), 60)


class MemoryReplyTests(unittest.TestCase):
    def entries(self, count, diff_chars):
        return [{**entry(number), 'diff_file': 'training.py.diff', 'code_diff': 'd' * diff_chars, 'diff_truncated': False,
                 'hypothesis_evidence': {'version': 'v', 'scope': 's', 'handoff': [], 'handoff_guide': 'g' * 250,
                                         'records': [{'r': 'x' * 1800}], 'records_omitted': 0, 'note': 'n' * 250}}
                for number in range(1, count + 1)]

    def test_two_experiments_keep_their_diff_and_evidence(self):
        entries = self.entries(2, 3000)
        self.assertGreater(replies.size(entries), PARKED_AT)
        reply = replies.memory_reply(entries)
        self.assertLess(replies.size(reply), replies.REPLY_BUDGET_CHARS)
        for row, source in zip(reply, entries):
            self.assertEqual(row['code_diff'], source['code_diff'])
            self.assertEqual(set(row['hypothesis_evidence']), {'handoff', 'records', 'records_omitted'})

    def test_five_experiments_fit(self):
        entries = self.entries(5, 3000)
        reply = replies.memory_reply(entries)
        self.assertLess(replies.size(reply), replies.REPLY_BUDGET_CHARS)
        for row, source in zip(reply, entries):
            self.assertEqual(row['experiment'], source['experiment'])
            self.assertEqual(row['hypothesis'], source['hypothesis'])
            self.assertEqual(row['selection_gate'], source['selection_gate'])
            self.assertEqual(row['snapshot'], source['snapshot'])
            self.assertNotIn('train', row)
            self.assertNotIn('per_brain', row['selection_comparison'])
            self.assertGreaterEqual(len(row['code_diff']), 2000)
            self.assertNotIn('hypothesis_evidence', row)
            self.assertIn('hypothesis_memory.json', row['hypothesis_evidence_note'])

    def test_long_diffs_are_cut_further_when_needed(self):
        entries = [{**item, 'hypothesis': 'h' * 4000} for item in self.entries(5, 3000)]
        reply = replies.memory_reply(entries)
        self.assertLess(replies.size(reply), replies.REPLY_BUDGET_CHARS)
        self.assertTrue(all(len(row['code_diff']) == replies.MEMORY_STEPS[-1][0] and row['diff_truncated'] for row in reply))
        self.assertTrue(all(row['hypothesis'].endswith('[cut; see result.json]') and 'selection_gate' in row for row in reply))


class BudgetTests(unittest.TestCase):
    def test_the_budget_stays_below_the_observed_parking_size(self):
        self.assertLessEqual(replies.REPLY_BUDGET_CHARS, 0.7 * PARKED_AT)

    def test_reply_chars_counts_text_blocks_of_sdk_content(self):
        self.assertEqual(replies.reply_chars({'content': [{'type': 'text', 'text': 'abcd'}, {'type': 'image', 'data': 'x' * 99},
                                                          {'type': 'text', 'text': 'ef'}]}), 6)
        self.assertEqual(replies.reply_chars({'a': 1}), len(json.dumps({'a': 1})))

    def test_an_oversized_reply_is_recorded_in_the_trajectory(self):
        session = SimpleNamespace(trace=Mock())
        TrainingExperiments._check_reply_size(session, 'inspect_failure_cases', {'pairs': 'x' * 40_000})
        event, message = session.trace.emit.call_args.args
        self.assertEqual(event, 'tool_reply_over_budget')
        self.assertIn('inspect_failure_cases', message)
        self.assertEqual(session.trace.emit.call_args.kwargs['budget'], replies.REPLY_BUDGET_CHARS)
        session.trace.emit.reset_mock()
        TrainingExperiments._check_reply_size(session, 'evaluate_train', {'status': 'evaluated'})
        session.trace.emit.assert_not_called()

    def test_the_files_replies_name_are_in_trees_a_session_may_read(self):
        from proofreader_evolve.cli.run_precision_evolution import TRAIN_ARTEFACT_DIRS
        for name in ('experiments', 'descriptor_runs', 'descriptor_probes'):
            self.assertIn(name, TRAIN_ARTEFACT_DIRS)


if __name__ == '__main__':
    unittest.main()

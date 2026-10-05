"""Offline-report regression cases. Synthetic JSON only; no workers or SDK."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from proofreader_evolve.harness.evolution_report import (
    ArtifactReader, build_report_data, update_report, write_report,
)
from proofreader_evolve.harness.run_records import PRECISION


def measurement(score, policy='seed'):
    return {'policy_sha256': policy, 'macro_precision': score, 'cells': {
        '794495/split': {'pool_size': 100, 'positives': 20, 'requested_k': 10,
                         'effective_k': 10, 'tp': int(score * 10), 'fp': 10 - int(score * 10),
                         'precision': score, 'recall': score / 2}}}


class EvolutionReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.put('manifest.json', {'objective': PRECISION, 'train_brains': ['794495'],
                                  'development_validation_brains': ['794493'], 'budgets': {'split': 10}})
        self.put('seed.json', {'train': measurement(.4), 'validation': measurement(.3)})

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def generation(self, number, **kwargs):
        row = {'generation': number, 'objective': PRECISION, 'accepted': False,
               'train': None, 'validation': None, 'experiments': [], **kwargs}
        self.put(f'gen{number:03d}/evaluation.json', row)
        return row

    def test_rejected_submission_can_remain_in_pool_and_missing_scores_stay_null(self):
        attempt = {'experiment': 'gen001/attempt001', 'sequence': 1, 'status': 'evaluated',
                   'target_kind': 'split', 'component_sha256': 'specialist', 'target_precision': .2,
                   'train': measurement(.2), 'hypothesis': 'Recover long-gap positives'}
        self.generation(1, train=measurement(.2, 'candidate'), validation=measurement(.2, 'candidate'),
                        submitted_experiment=attempt['experiment'], experiments=[attempt])
        self.put('gen001/candidate_pool_after.json', {'kinds': {'split': {'candidates': [
            {'experiment': attempt['experiment'], 'component_sha256': 'specialist',
             'retention': {'reason': 'train_specialist'}}]}}})
        self.generation(2, failure_stage='revision', reason='Failed: SDK error',
                        reviser={'cost_usd': .12})
        report = build_report_data(self.root)
        first, second = report['generations']
        self.assertEqual(first['status'], 'rejected')
        self.assertEqual(first['pool_after']['kinds']['split']['candidates'][0]['retention']['reason'],
                         'train_specialist')
        self.assertEqual(second['status'], 'agent_error')
        self.assertIsNone(second['validation'])
        self.assertFalse(report['totals']['cost_complete'])
        self.assertEqual(report['totals']['recorded_cost_usd'], .12)
        self.assertIsNone(first['attempts'][0]['lineage'])  # No invented old ancestry.

    def test_partial_ledger_keeps_completed_records_and_interrupted_attempts(self):
        row = self.generation(1, accepted=True, train=measurement(.6, 'winner'),
                              validation=measurement(.5, 'winner'))
        (self.root / 'ledger.jsonl').write_text(json.dumps(row) + '\n{"generation":')
        self.put('gen002/search_plan.json', {'target_kind': 'split', 'search_parent': {'experiment': 'seed/split'}})
        self.put('gen002/experiments/attempt001/proposal.json', {'hypothesis': 'Unfinished measurement'})
        (self.root / 'gen002/trajectory.jsonl').write_text(json.dumps({
            'event': 'generation_interrupted', 'message': 'KeyboardInterrupt'}) + '\n')
        report = build_report_data(self.root)
        self.assertEqual(len(report['generations']), 2)
        second = report['generations'][1]
        self.assertEqual(second['status'], 'interrupted')
        self.assertEqual(second['attempts'][0]['status'], 'incomplete')
        self.assertEqual(second['parent']['validation']['macro_precision'], .5)
        self.assertTrue(any('incomplete/invalid' in w for w in report['warnings']))

    def test_protected_reference_and_scheduling_role_are_preserved(self):
        reference = {'experiment': 'seed/split', 'component_sha256': 'seed',
                     'target_precision': .3, 'retention': {'reason': 'protected_reference'}}
        self.put('candidate_pool.json', {'kinds': {'split': {
            'reference_branch': reference, 'candidates': []}}})
        self.generation(1, search_branch_role='reference', search_reason='reference_branch')
        report = build_report_data(self.root)
        saved = report['pool']['kinds']['split']['reference_branch']
        self.assertEqual(saved['experiment'], 'seed/split')
        self.assertEqual(saved['retention']['reason'], 'protected_reference')
        self.assertEqual(report['generations'][0]['search_branch_role'], 'reference')

    def test_safe_html_and_offline_assets(self):
        hostile = '</script><img src=x onerror=alert(1)>'
        self.generation(1, reason=hostile, failure_stage='submission_check')
        html = write_report(self.root).read_text()
        self.assertNotIn(hostile, html)
        self.assertIn('\\u003c/script>', html)
        self.assertNotIn('@@DATA@@', html)
        self.assertNotIn('@@CSS@@', html)
        self.assertNotIn('@@JS@@', html)
        self.assertNotIn('innerHTML', html)
        self.assertIn("connect-src 'none'", html)

    def test_no_outside_artifacts_or_binary_loading(self):
        self.generation(1, experiments=[{'experiment': 'gen001/attempt001', 'sequence': 1,
                                        'status': 'evaluated', 'snapshot': '/untrusted/path'}])
        self.put('unrelated.json', {'must_not_be_read': True})
        (self.root / 'cache.pkl').write_bytes(b'not a pickle')
        directory = self.root / 'gen001/experiments/attempt001'
        directory.mkdir(parents=True)
        with tempfile.TemporaryDirectory() as outside:
            secret = Path(outside) / 'source.py'
            secret.write_text('PRIVATE_OUTSIDE_SOURCE')
            (directory / 'scorer.py').symlink_to(secret)
            report = build_report_data(self.root)
        self.assertNotIn('PRIVATE_OUTSIDE_SOURCE', json.dumps(report))
        self.assertNotIn('must_not_be_read', json.dumps(report))
        self.assertTrue(any('escapes run directory' in w for w in report['warnings']))

    def test_non_precision_and_duplicate_generations_are_rejected(self):
        self.put('manifest.json', {'objective': 'structural_repair'})
        with self.assertRaisesRegex(ValueError, 'Only fixed-pool'):
            build_report_data(self.root)
        self.put('manifest.json', {'objective': PRECISION})
        row = self.generation(1)
        (self.root / 'ledger.jsonl').write_text((json.dumps(row) + '\n') * 2)
        with self.assertRaisesRegex(ValueError, 'Duplicate generation'):
            build_report_data(self.root)

    def test_report_failure_does_not_change_experiment_outcome(self):
        trace = Mock()
        with patch('proofreader_evolve.harness.evolution_report.write_report', side_effect=OSError('disk full')):
            self.assertIsNone(update_report(self.root, trace, status='failed', reason='original SDK error'))
        state = json.loads((self.root / 'run_status.json').read_text())
        self.assertEqual(state['reason'], 'original SDK error')
        trace.emit.assert_called_once()

    def test_preview_budget_does_not_claim_empty_file_is_complete(self):
        (self.root / 'source.txt').write_text('x' * 100)
        reader = ArtifactReader(self.root)
        reader.preview_remaining = 5
        result = reader.text('source.txt')
        self.assertEqual(result['text'], 'xxxxx')
        self.assertTrue(result['truncated'])

    def test_parent_hash_mismatch_is_not_used_for_comparison(self):
        self.generation(1, parent_policy_sha256='different_parent')
        parent = build_report_data(self.root)['generations'][0]['parent']
        self.assertIsNone(parent['train'])
        self.assertIsNone(parent['validation'])

    def test_removed_sections_do_not_embed_payloads_or_consume_source_previews(self):
        self.generation(1)
        trace = {'event': 'generation_interrupted', 'message': 'RAW_EVENT_PAYLOAD'}
        (self.root / 'gen001/trajectory.jsonl').write_text(json.dumps(trace) + '\n')
        (self.root / 'gen001/reviser_prompt.txt').write_text('UNUSED_GENERATION_PREVIEW')
        self.put('gen001/context_inspections/inspection001.json', {'private': 'UNUSED_CONTEXT_PREVIEW'})
        (self.root / 'trajectory.jsonl').write_text(json.dumps({'event': 'run_complete'}) + '\n')
        reader = ArtifactReader(self.root)
        reader.preview_remaining = 0
        self.assertEqual(reader.event_names('gen001'), {'generation_interrupted'})
        self.assertEqual(reader.preview_remaining, 0)
        report = build_report_data(self.root)
        self.assertEqual(report['state']['status'], 'completed')
        self.assertEqual(report['generations'][0]['status'], 'interrupted')
        self.assertNotIn('events', report)
        for name in ('events', 'files', 'context_inspections', 'pool_before'):
            self.assertNotIn(name, report['generations'][0])
        html = write_report(self.root).read_text()
        for removed in ('RAW_EVENT_PAYLOAD', 'UNUSED_GENERATION_PREVIEW', 'UNUSED_CONTEXT_PREVIEW',
                        'Generation Details', 'Run Events'):
            self.assertNotIn(removed, html)


if __name__ == '__main__':
    unittest.main()

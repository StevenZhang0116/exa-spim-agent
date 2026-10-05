"""Synthetic TRAIN coverage selection; no scoring workers, model fitting or APIs."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from proofreader_evolve.harness.candidate_pool import CandidatePool, candidate_metadata
from proofreader_evolve.harness.train_coverage import TrainingCoverage


class SpecialistPoolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.truth = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int8)
        self.table = SimpleNamespace(
            truth=self.truth, meta={'pool_sha256': 'fixed-train-pool'},
            candidates=[{'gap_um': 10 if index < 3 else 40} for index in range(10)])
        self.coverage = TrainingCoverage({'TRAIN': SimpleNamespace(tables={'split': self.table})})

    def entry(self, number, chosen, *, cached=False):
        chosen = np.asarray(chosen, dtype=int)
        tp = int(self.truth[chosen].sum())
        source = (f"PARAMS = {{'weight': {number}}}\n"
                  "def score_candidates(features, ctx):\n"
                  "    return features['detector_score'].to_numpy() * PARAMS['weight']\n")
        report = {'macro_precision': tp / len(chosen), 'cells': {'TRAIN/split': {
            'pool_sha256': 'fixed-train-pool',
            'labels_sha256': hashlib.sha256(self.truth.tobytes()).hexdigest(),
            'pool_size': len(self.truth), 'positives': 6,
            'requested_k': 4, 'effective_k': len(chosen), 'tp': tp,
            'fp': len(chosen) - tp, 'precision': tp / len(chosen), 'recall': tp / 6,
        }}}
        metadata = candidate_metadata(source, 'split', report,
                                      {'TRAIN/split': {'chosen': chosen}}, coverage=self.coverage)
        return {'experiment': f'gen001/attempt{number:03d}', 'sequence': number,
                'target_kind': 'split', 'component_sha256': hashlib.sha256(source.encode()).hexdigest(),
                'family': 'fixture', 'status': 'evaluated', 'cached': cached, 'train': report, **metadata}

    def pool(self, name='pool', **kwargs):
        return CandidatePool(self.root / f'{name}.json', ['split'], coverage=self.coverage, **kwargs)

    def test_specialist_bypasses_tolerance_and_beats_negative_only_diversity(self):
        champion = self.entry(1, [0, 1, 2, 6])
        redundant = self.entry(2, [0, 1, 2, 7])  # Different Top K; identical positive hits.
        specialist = self.entry(3, [3, 4, 6, 7])  # 0.25 below champion, beyond tolerance.
        pool = self.pool(size=2, score_tolerance=.02)
        pool.add(champion)
        pool.finish(pool.next_plan(), [redundant, specialist])
        self.assertEqual([e['sequence'] for e in pool.members['split']], [1, 3])
        saved = pool.summary()['kinds']['split']
        self.assertEqual(saved['retained_positive_hits'], 5)
        branch = saved['candidates'][1]
        self.assertEqual(branch['retention']['reason'], 'train_specialist')
        self.assertEqual(branch['vs_champion']['additional_positive_hits'], 2)
        self.assertEqual(branch['vs_champion']['lost_positive_hits'], 3)
        self.assertEqual(branch['specialist_profile']['slices']['TRAIN/split/gap_gt_25um'],
                         {'positives': 3, 'hits': 2, 'recall': 2 / 3})

    def test_positive_identity_matters_even_when_slice_scores_tie(self):
        # Both hit two short-gap positives, but the second recovers a different one.
        pool = self.pool(size=2, score_tolerance=0)
        pool.add(self.entry(1, [0, 1, 6, 7]))
        pool.add(self.entry(2, [1, 2, 6, 7]))
        branch = pool.summary()['kinds']['split']['candidates'][1]
        self.assertEqual(branch['retention']['reason'], 'train_specialist')
        self.assertEqual(branch['vs_champion']['additional_positive_hits'], 1)
        self.assertEqual(branch['vs_champion']['improved_slices'], [])

    def test_weaker_redundant_candidate_is_not_a_specialist(self):
        pool = self.pool()
        pool.add(self.entry(1, [0, 1, 2, 6]))
        pool.add(self.entry(2, [0, 1, 8, 9]))
        self.assertEqual(len(pool.members['split']), 1)

    def test_protected_reference_survives_train_eviction_and_has_only_train_diagnostics(self):
        pool = self.pool(size=1, score_tolerance=0)
        champion = self.entry(1, [0, 1, 2, 6])
        reference = self.entry(2, [0, 6, 7, 8])
        pool.add(champion)
        pool.add(reference)
        # The outer decision is deliberately not part of the exposed snapshot.
        pool.set_reference({**reference, 'validation': {'HELDOUT': 'private'},
                            'reason': 'private promotion reason'})
        reference['parameters']['weight'] = 999
        summary = pool.summary()['kinds']['split']
        self.assertEqual([e['experiment'] for e in summary['candidates']], [champion['experiment']])
        self.assertEqual(summary['reference_branch']['target_precision'], .25)
        self.assertEqual(summary['reference_branch']['parameters']['weight'], 2)
        self.assertEqual(summary['reference_branch']['retention']['reason'], 'protected_reference')
        plans = [pool.next_plan() for _ in range(3)]
        self.assertEqual(plans[-1]['branch_role'], 'reference')
        self.assertEqual(plans[-1]['search_parent']['experiment'], reference['experiment'])
        self.assertEqual(plans[-1]['complementary_candidates'][0]['experiment'], champion['experiment'])
        text = json.dumps({'pool': pool.summary(), 'plans': plans})
        for hidden in ('validation', 'HELDOUT', 'private', 'positive_rows', 'row_indices'):
            self.assertNotIn(hidden, text)

    def test_reference_must_receive_a_measured_round_before_plateau_pauses_kind(self):
        pool = self.pool(size=1, plateau_patience=1, exploration_patience=1, explore_every=99)
        champion = self.entry(1, [0, 1, 2, 6])
        reference = self.entry(2, [0, 6, 7, 8])
        pool.add(champion)
        pool.set_reference(reference)
        for number in (3, 4):
            pool.finish(pool.next_plan(), [self.entry(number, [0, 1, 2, 6])])
        self.assertFalse(pool.progress['split']['paused'])
        plan = pool.next_plan()
        self.assertEqual(plan['branch_role'], 'reference')
        pool.finish(plan, [{'status': 'execution_error'}])
        for _ in range(2):
            pool.finish(pool.next_plan(), [{**champion, 'cached': True}])
        self.assertFalse(pool.progress['split']['reference_measured'])
        plan = pool.next_plan()
        self.assertEqual(plan['branch_role'], 'reference')
        pool.finish(plan, [self.entry(5, [0, 6, 7, 8])])
        self.assertTrue(pool.progress['split']['reference_measured'])
        self.assertTrue(pool.progress['split']['paused'])
        self.assertIsNone(pool.next_plan())

    def test_new_reference_reopens_kind_and_replaces_only_the_protected_slot(self):
        pool = self.pool(size=1)
        champion = self.entry(1, [0, 1, 2, 6])
        reference = self.entry(2, [0, 6, 7, 8])
        pool.add(champion)
        pool.set_reference(champion)
        plan = pool.next_plan()
        pool.progress['split'].update(paused=True, stalled_rounds=8,
                                      stalled_explorations=3, reference_measured=True)
        result = pool.finish(plan, [reference], reference_entry=reference)
        self.assertFalse(result['paused'])
        self.assertFalse(result['reference_measured'])
        self.assertEqual(result['stalled_rounds'], 0)
        self.assertEqual(result['scheduled_rounds'], 1)
        self.assertEqual(pool.references['split']['experiment'], reference['experiment'])
        self.assertEqual(pool.members['split'][0]['experiment'], champion['experiment'])

    def test_exploration_uses_specialist_and_exposes_complement_counts_only(self):
        pool = self.pool(size=2, explore_every=1)
        pool.add(self.entry(1, [0, 1, 2, 6]))
        pool.add(self.entry(2, [3, 4, 6, 7]))
        plan = pool.next_plan()
        self.assertEqual(plan['search_parent']['sequence'], 2)
        self.assertEqual(plan['complementary_candidates'][0]['experiment'], 'gen001/attempt001')
        self.assertEqual(plan['complementary_candidates'][0]['additional_positive_hits'], 3)
        text = json.dumps({'plan': plan, 'pool': pool.summary()})
        for private_field in ('positive_rows', 'chosen', 'segment_id', 'node_id', 'row_indices'):
            self.assertNotIn(private_field, text)
        self.assertNotIn('validation', text.lower())

    def test_new_coverage_resets_plateau_without_mean_gain(self):
        pool = self.pool(size=2)
        pool.add(self.entry(1, [0, 1, 2, 6]))
        pool.progress['split'].update(stalled_rounds=2, stalled_explorations=1)
        progress = pool.finish(pool.next_plan(), [self.entry(2, [3, 4, 5, 6])])
        self.assertEqual(progress['new_positive_hits'], 3)
        self.assertEqual(progress['stalled_rounds'], 0)
        self.assertEqual(progress['stalled_explorations'], 0)
        self.assertFalse(progress['paused'])
        progress = pool.finish(pool.next_plan(), [self.entry(3, [1, 2, 3, 6])])
        self.assertEqual(progress['new_positive_hits'], 0)
        self.assertEqual(progress['improved_slices'], [])
        self.assertEqual(progress['stalled_rounds'], 1)

    def test_generation_selection_is_order_independent_and_bounded(self):
        entries = [self.entry(2, [3, 4, 6, 7]), self.entry(3, [4, 5, 6, 7]),
                   self.entry(4, [0, 1, 6, 7])]
        results = []
        for number, order in enumerate((entries, list(reversed(entries)))):
            pool = self.pool(str(number), size=2)
            pool.add(self.entry(1, [0, 1, 2, 6]))
            pool.finish(pool.next_plan(), order)
            results.append([e['sequence'] for e in pool.members['split']])
        self.assertEqual(results, [[1, 2], [1, 2]])

    def test_cached_or_failed_rounds_do_not_advance_plateau(self):
        pool = self.pool()
        champion = self.entry(1, [0, 1, 2, 6])
        pool.add(champion)
        pool.progress['split']['stalled_rounds'] = 1
        for trials in ([{**champion, 'cached': True}], [{'status': 'execution_error'}]):
            progress = pool.finish(pool.next_plan(), trials)
            self.assertEqual(progress['rounds'], 0)
            self.assertEqual(progress['stalled_rounds'], 1)
            self.assertEqual(progress['new_positive_hits'], 0)

    def test_rejects_other_brains_changed_labels_or_changed_k(self):
        entry = self.entry(1, [0, 1, 2, 6])
        cells = entry['train']['cells']
        with self.assertRaisesRegex(ValueError, 'configured TRAIN cells'):
            self.coverage.record('other', 'split', {'HELDOUT/split': cells['TRAIN/split']}, {})
        for key, value in [('labels_sha256', 'changed'), ('pool_sha256', 'changed'), ('requested_k', 5)]:
            altered = deepcopy(cells)
            altered['TRAIN/split'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.coverage.record('other', 'split', altered,
                                     {'TRAIN/split': {'chosen': np.array([0, 1, 2, 6])}})

    def test_missing_geometry_and_zero_positives_have_no_fake_slice_evidence(self):
        table = SimpleNamespace(truth=np.zeros(4, dtype=np.int8),
                                candidates=[{} for _ in range(4)], meta={'pool_sha256': 'empty'})
        coverage = TrainingCoverage({'TRAIN': SimpleNamespace(tables={'split': table})})
        cells = {'TRAIN/split': {'pool_sha256': 'empty', 'pool_size': 4, 'positives': 0,
                                'labels_sha256': hashlib.sha256(table.truth.tobytes()).hexdigest(),
                                'requested_k': 2, 'effective_k': 2, 'tp': 0}}
        profile = coverage.record('zero', 'split', cells, {'TRAIN/split': {'chosen': np.array([0, 1])}})
        entry = {'target_kind': 'split', 'component_sha256': 'zero', 'specialist_profile': profile}
        self.assertEqual(profile['slices'], {'TRAIN/split/all': {'positives': 0, 'hits': 0, 'recall': None}})
        self.assertEqual(coverage.compare(entry, [])['improved_slices'], [])
        self.assertEqual(coverage.compare(entry, [])['additional_macro_recall'], 0)

    def test_merge_degree_slices_use_same_global_top_k(self):
        table = SimpleNamespace(truth=np.array([1, 1, 1, 0], dtype=np.int8),
                                candidates=[{'degree': d} for d in (3, 4, 5, 3)],
                                meta={'pool_sha256': 'merge'})
        coverage = TrainingCoverage({'TRAIN': SimpleNamespace(tables={'merge': table})})
        cells = {'TRAIN/merge': {'pool_sha256': 'merge', 'pool_size': 4, 'positives': 3,
                                'labels_sha256': hashlib.sha256(table.truth.tobytes()).hexdigest(),
                                'requested_k': 2, 'effective_k': 2, 'tp': 1}}
        slices = coverage.record('merge', 'merge', cells,
                                 {'TRAIN/merge': {'chosen': np.array([1, 3])}})['slices']
        self.assertEqual(slices['TRAIN/merge/degree_3']['hits'], 0)
        self.assertEqual(slices['TRAIN/merge/degree_ge_4']['hits'], 1)
        self.assertEqual(slices['TRAIN/merge/degree_ge_4']['recall'], .5)


if __name__ == '__main__':
    unittest.main()

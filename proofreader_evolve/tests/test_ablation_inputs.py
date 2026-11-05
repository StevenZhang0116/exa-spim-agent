"""Effective ablation inputs and evidence accounting; synthetic data, isolated workers."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from proofreader_evolve.harness import feature_ablation as ablation
from proofreader_evolve.harness.image_evidence import descriptor_ablation_evidence
from proofreader_evolve.harness.train_experiments import ExperimentMemory, TrainingExperiments
from proofreader_evolve.tests.test_feature_discovery import training_fixture, FORMULA, RESEARCH


PROGRAM = '''def select_columns(columns, params):
    return params['columns']
def fit(X_train, y_train, artifact_dir, params, **kwargs):
    (artifact_dir / 'model.txt').write_text('synthetic model')
def predict(X, artifact_dir, params, **kwargs):
    if params.get('raw'):
        return [float(bool(patches)) for patches in kwargs['images']]
    return X.iloc[:, 0].to_numpy()
'''
RAW_PROGRAM = "LOCAL_IMAGE = {'raw_patches': True}\n" + PROGRAM


class AblationInputTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.train = training_fixture()
        self.table = self.train['794495'].tables['merge']
        self.frames = {'794495': self.table.features.copy()}
        self.units = []

    def measure(self, selected, removed, *, program=PROGRAM, raw=False):
        return ablation.measure_ablation(self.train, 'merge', program,
            {'parameters': {'columns': selected, 'raw': raw}}, self.frames, removed,
            ablation.grouped_folds(self.train, 'merge'),
            {'program_sha256': 'fixture', 'parameters_sha256': 'fixture'}, {'merge': 3},
            self.root / 'comparison', classifier=True, charge=lambda: self.units.append(1),
            memory_mb=1024, fit_timeout=60, score_timeout=60)

    def test_unused_inputs_skip_control_and_do_not_complete_research(self):
        gen = self.root / 'gen001'
        gen.mkdir()
        policy, rules = gen / 'scorer.py', gen / 'rules.md'
        policy.write_text(FORMULA)
        rules.write_text('Synthetic input check.')
        (gen / 'training.py').write_text(PROGRAM)
        (gen / 'proposal.json').write_text(json.dumps({'hypothesis': 'Test a feature.',
            'strategy': 'Paired removal.', 'research': RESEARCH,
            'classifier': {'parameters': {'columns': ['detector_score']}}}))
        memory = ExperimentMemory(self.root)
        session = TrainingExperiments(gen, policy, rules, 'merge', {'merge': FORMULA}, self.train, {},
            {'794495/merge': {'scores': self.table.features.detector_score.to_numpy(), 'chosen': np.arange(3)}},
            {'merge': 3}, 60, 0, memory, max_evaluations=2, classifier_memory_mb=1024)
        session.search_plan.update(investigation={'required_source': None}, followup_required=True)
        with patch.object(ablation, 'fit_classifier', wraps=ablation.fit_classifier) as fit:
            report = session.evaluate_feature_ablation()
        self.assertEqual(fit.call_count, 3)
        self.assertEqual(report['status'], 'not_applicable')
        self.assertEqual(report['reason'], 'no_effective_inputs')
        self.assertEqual(report['input_check']['effective_columns'], [])
        self.assertEqual(report['input_check']['ignored_columns'], ['local_signal'])
        self.assertNotIn('delta_objective', report)
        self.assertEqual((report['evaluations_used'], report['evaluations_remaining']), (1, 1))
        directory = gen / 'feature_ablations/ablation001'
        self.assertFalse((directory / 'without_features').exists())
        self.assertFalse(list(directory.glob('full/fold*/models')))
        with np.load(directory / 'predictions.npz') as arrays:
            self.assertEqual(arrays.files, ['full:794495'])
        status = session.research_status()
        self.assertFalse(status['investigation_complete'])
        self.assertFalse(status['followup_complete'])
        self.assertEqual(status['new_measurements'], 0)
        self.assertEqual(status['execution_failures'], 0)
        self.assertEqual(status['new_observations'], 1)
        self.assertFalse(memory.hypotheses.ablation_cache)
        self.assertFalse(memory.candidates)
        self.assertEqual(policy.read_text(), FORMULA)

    def test_partial_selection_reports_effective_and_ignored_columns(self):
        with patch.object(ablation, 'fit_classifier', wraps=ablation.fit_classifier) as fit:
            report = self.measure(['local_signal'], ['detector_score', 'local_signal'])
        self.assertEqual(fit.call_count, 6)
        self.assertEqual(len(self.units), 2)
        self.assertEqual(report['status'], 'measured')
        self.assertEqual(report['input_check']['effective_columns'], ['local_signal'])
        self.assertEqual(report['input_check']['ignored_columns'], ['detector_score'])
        self.assertGreater(report['delta_precision'], 0.)
        self.assertEqual(len(report['input_check']['folds']), 3)
        self.assertTrue(all(f['effective_columns'] == ['local_signal'] for f in report['input_check']['folds']))

    def test_feature_selected_only_in_later_fold_still_runs_control(self):
        models = [{'columns': columns} for columns in
                  (['detector_score'], ['local_signal'], ['detector_score'])] * 2
        with patch.object(ablation, 'fit_classifier', return_value=('mock', {})) as fit, \
                patch.object(ablation, 'frozen_model', side_effect=models), \
                patch.object(ablation, 'predict_model', side_effect=lambda model, store, frame, *a, **k:
                             frame[model['columns'][0]].to_numpy()):
            report = self.measure(['local_signal'], ['local_signal'])
        self.assertEqual(fit.call_count, 6)
        self.assertEqual(len(self.units), 2)
        self.assertEqual(report['status'], 'measured')
        self.assertEqual([f['effective_columns'] for f in report['input_check']['folds']],
                         [[], ['local_signal'], []])

    def test_raw_patch_removal_works_without_numeric_image_marker(self):
        path = self.root / 'pixels.npz'
        np.savez(path, image_zyx=np.ones((1, 1, 1)))
        frame = self.frames['794495']
        frame['image_available'] = self.table.truth.astype(float)
        frame.attrs['_image_rows'] = {int(i): [str(path)] for i in np.flatnonzero(self.table.truth)}
        report = self.measure([], ['image_available'], program=RAW_PROGRAM, raw=True)
        self.assertEqual(report['status'], 'measured')
        self.assertEqual(len(self.units), 2)
        self.assertEqual(report['input_check']['effective_columns'], ['image_available'])
        self.assertEqual(report['input_check']['ignored_columns'], [])
        self.assertTrue(all(f['raw_patches_removed'] for f in report['input_check']['folds']))
        self.assertTrue(report['image_evidence']['image_only_control'])
        self.assertGreater(report['delta_precision'], 0.)
        with np.load(self.root / 'comparison/predictions.npz') as arrays:
            np.testing.assert_array_equal(arrays['full:794495'], self.table.truth)
            np.testing.assert_array_equal(arrays['without_features:794495'], 0.)
        self.assertTrue(frame.attrs['_image_rows'])

    def test_empty_raw_patch_transport_does_not_make_removal_effective(self):
        frame = self.frames['794495']
        frame['image_available'] = 0.
        frame.attrs['_image_rows'] = {}
        report = self.measure([], ['image_available'], program=RAW_PROGRAM, raw=True)
        self.assertEqual(report['status'], 'not_applicable')
        self.assertEqual(len(self.units), 1)
        self.assertFalse(any(f['raw_patches_removed'] for f in report['input_check']['folds']))

    def test_ignored_image_descriptors_cannot_supply_image_comparison_evidence(self):
        report = {'feature_columns': ['local_signal', 'bank_agent_image'],
                  'input_check': {'effective_columns': ['local_signal']}}
        registry = {'bank_agent_image': {'inputs': 'image'}}
        self.assertFalse(descriptor_ablation_evidence(report, registry))
        report['input_check']['effective_columns'].append('bank_agent_image')
        self.assertIn('bank_agent_image', descriptor_ablation_evidence(report, registry)['descriptor_columns'])

    def test_ignored_direct_image_features_do_not_turn_geometry_ablation_into_image_evidence(self):
        program = ("LOCAL_IMAGE = {'feature_names': ['image_noise']}\n"
                   "def extract_image_features(context):\n    return {'image_noise': 0.}\n" + PROGRAM)
        self.frames['794495']['image_noise'] = 0.
        self.frames['794495']['image_available'] = 1.
        with patch.object(ablation, 'fit_classifier', return_value=('mock', {})), \
                patch.object(ablation, 'frozen_model', return_value={'columns': ['local_signal']}), \
                patch.object(ablation, 'predict_model', side_effect=lambda model, store, frame, *a, **k:
                             frame.local_signal.to_numpy()):
            report = self.measure(['local_signal'], ['local_signal', 'image_noise', 'image_available'],
                                  program=program)
        self.assertEqual(report['status'], 'measured')
        self.assertEqual(report['input_check']['effective_columns'], ['local_signal'])
        self.assertFalse(report['image_evidence']['all_image_inputs_removed_in_control'])
        self.assertFalse(report['image_evidence']['image_only_control'])


if __name__ == '__main__':
    unittest.main()

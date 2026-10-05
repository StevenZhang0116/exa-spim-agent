"""Model-program/artifact contract and TRAIN isolation regressions.

Synthetic fixtures only; no brain caches or live LLM requests.
The worker cases require Linux Landlock ABI >= 3 and libseccomp.
"""
import asyncio
from contextlib import redirect_stdout
from copy import deepcopy
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from proofreader_evolve.harness import fixed_pool_scoring as scoring
from proofreader_evolve.harness.classifier_contract import (
    classifier_info, classifier_configs, frozen_model, normalize_config, render_model, verify_artifacts)
from proofreader_evolve.harness.classifier_training import fit_classifier, ClassifierTrainingError
from proofreader_evolve.harness.isolated_scoring import score
from proofreader_evolve.harness.scorer_components import components
from proofreader_evolve.harness.train_experiments import ExperimentMemory, TrainingExperiments
from proofreader_evolve.tests.test_fixed_pool_scoring import fixture, BASELINE

# A custom learner with its own persistence format, not a framework-selected classifier.
PROGRAM = '''import numpy as np

def fit(X_train, y_train, artifact_dir, params):
    positive = X_train.loc[y_train == 1, 'evidence'].mean()
    np.save(artifact_dir / 'center.npy', np.array([positive]), allow_pickle=False)
    print('custom positive center:', positive)


def predict(X, artifact_dir, params):
    center = np.load(artifact_dir / 'center.npy', allow_pickle=False)[0]
    x = X['evidence'].fillna(0).to_numpy()
    return -params.get('scale', 1.0) * np.abs(x - center)
'''


def proposal(session, *, parameters=None, grid=None, program=PROGRAM):
    session.training_path.write_text(program)
    session.proposal_path.write_text(json.dumps({
        'hypothesis': 'Learn a ranking from TRAIN labels', 'strategy': 'Fit and freeze a custom learner',
        'family': 'custom-learner', 'classifier': {'parameters': parameters or {'scale': 1.0}},
        'parameter_grid': grid or {}}))


class ProgramContractTests(unittest.TestCase):
    def test_legacy_numpy_exports_still_validate_and_predict(self):
        from proofreader_evolve.harness import legacy_classifier as legacy
        # Historical exports embed this exact template, including comments.
        self.assertEqual(hashlib.sha256(legacy.RUNTIME_PATH.read_bytes()).hexdigest(),
                         '71b9047822e8adbc080a030ba67fb5a5233d584ec7e9604626710c3b9b995820')
        for algorithm in ('logistic_regression', 'random_forest'):
            with self.subTest(algorithm=algorithm):
                config = {'algorithm': algorithm, 'features': ['evidence'],
                          'parameters': {**legacy.DEFAULTS[algorithm]}}
                model = {'version': legacy.MODEL_VERSION, 'kind': 'split', 'config': config,
                         'training_brains': ['1'], 'training_rows': 3,
                         'training_fingerprint': '0' * 64, 'impute': [0.], 'mean': [0.], 'scale': [1.]}
                frame = pd.DataFrame({'evidence': [-1., 1., np.nan]})
                if algorithm == 'logistic_regression':
                    model.update(coef=[2.], intercept=0.)
                    expected = 1 / (1 + np.exp(-np.array([-2., 2., 0.])))
                else:
                    config['parameters'].update(n_estimators=1, max_depth=1)
                    model['trees'] = [{'left': [1, -1, -1], 'right': [2, -1, -1],
                                       'feature': [0, -2, -2], 'threshold': [0., -2., -2.],
                                       'probability': [.5, .2, .8]}]
                    expected = [.2, .8, .2]
                source = ('# Frozen classifier fitted by the harness on TRAIN only. Do not edit fitted arrays.\n'
                          f'_FROZEN_CLASSIFIER = {model!r}\n' + legacy.RUNTIME_PATH.read_text())
                self.assertEqual(frozen_model(source), model)
                np.testing.assert_allclose(score(source, frame, 'split'), expected)
                with self.assertRaisesRegex(ValueError, 'edited'):
                    frozen_model(source + '\n# changed\n')

    def test_arbitrary_settings_and_program_structure(self):
        config = normalize_config({'parameters': {'scale': 1, 'architecture': [8, 4], 'loss': 'custom',
                                                 'options': {'depth': None}}})
        tuned = {**config, 'parameters': {**config['parameters'], 'scale': 5}}
        self.assertEqual(classifier_info(config, PROGRAM)['formula_sha256'],
                         classifier_info(tuned, PROGRAM)['formula_sha256'])
        changed = PROGRAM.replace('np.abs(x - center)', '(x - center) ** 2')
        self.assertNotEqual(classifier_info(config, PROGRAM)['formula_sha256'],
                            classifier_info(config, changed)['formula_sha256'])
        self.assertEqual(len(classifier_configs(config, {'scale': [1, 1, 5]})), 2)
        with self.assertRaises(ValueError):
            normalize_config({'algorithm': 'random_forest'})
        with self.assertRaises(ValueError):
            classifier_configs(config, {'loss': [1, 2]})
        with self.assertRaises(ValueError):
            normalize_config({'parameters': {'nested': {'bad': float('nan')}}})

    def test_custom_artifacts_prediction_integrity_and_code_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, summary = fit_classifier({'1': fixture()}, 'split', {'parameters': {'scale': 2}},
                root / 'fit', program=PROGRAM, artifact_store=root / 'model_artifacts')
            model = frozen_model(source)
            self.assertEqual(model['program'], PROGRAM)
            self.assertEqual(model['training_brains'], ['1'])
            self.assertEqual(set(model['files']), {'center.npy'})
            self.assertIn('custom positive center:', summary['training_output_tail'])
            frame = fixture().tables['split'].features
            expected = -2 * np.abs(frame['evidence'].to_numpy() - .9)
            np.testing.assert_array_equal(score(source, frame, 'split', artifact_store=root / 'model_artifacts'), expected)
            extended = pd.concat([frame, frame.assign(evidence=10000)], ignore_index=True)
            np.testing.assert_array_equal(score(source, extended, 'split', artifact_store=root / 'model_artifacts')[:3], expected)
            with self.assertRaisesRegex(ValueError, 'edited'):
                frozen_model(source + '\n# changed\n')
            with self.assertRaises(ValueError):
                score(source, frame, 'split')  # Cannot silently ignore missing sidecar artifacts.
            artifact = verify_artifacts(model, root / 'model_artifacts') / 'center.npy'
            artifact.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'differ'):
                verify_artifacts(model, root / 'model_artifacts')

    def test_installed_estimator_and_serialized_custom_class(self):
        program = '''import joblib
from sklearn.tree import DecisionTreeClassifier
class CustomTree(DecisionTreeClassifier):
    pass

def fit(X_train, y_train, artifact_dir, params):
    model = CustomTree(max_depth=params['depth'], random_state=3)
    model.fit(X_train[['evidence']].fillna(0), y_train)
    joblib.dump(model, artifact_dir / 'tree.joblib')

def predict(X, artifact_dir, params):
    model = joblib.load(artifact_dir / 'tree.joblib')
    return model.predict_proba(X[['evidence']].fillna(0))[:, 1]
'''
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = fit_classifier({'1': fixture()}, 'split', {'parameters': {'depth': None}},
                root / 'fit', program=program, artifact_store=root / 'models')
            predicted = score(source, fixture().tables['split'].features, 'split', artifact_store=root / 'models')
            np.testing.assert_array_equal(predicted, [0, 1, 0])

    def test_filesystem_network_process_and_prediction_write_boundaries(self):
        probes = '''
def forbidden(operation):
    try:
        operation()
    except (PermissionError, OSError):
        return
    raise RuntimeError('Sandbox operation unexpectedly permitted')

'''
        fit_probe = '''    from pathlib import Path
    import socket, subprocess
    forbidden(lambda: Path(params['forbidden']).read_text())
    forbidden(lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM))
    forbidden(lambda: subprocess.run(['/bin/true'], check=True))
'''
        predict_probe = '''    from pathlib import Path
    forbidden(lambda: Path(params['forbidden']).read_text())
    forbidden(lambda: (artifact_dir / 'center.npy').write_bytes(b'changed'))
    forbidden(lambda: (artifact_dir.parent / 'y.npy').read_bytes())
'''
        program = probes + PROGRAM.replace('    positive =', fit_probe + '    positive =').replace(
            '    center =', predict_probe + '    center =')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            secret = root / 'heldout-sentinel.txt'
            secret.write_text('unavailable to fit and predict')
            source, _ = fit_classifier({'1': fixture()}, 'split', {'parameters': {'forbidden': str(secret)}},
                root / 'fit', program=program, artifact_store=root / 'models')
            values = score(source, fixture().tables['split'].features, 'split', artifact_store=root / 'models')
            self.assertTrue(np.isfinite(values).all())


class ClassifierToolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        redirect = redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        self.train = {'1': fixture()}
        self.state = {}
        self.parent = scoring.evaluate(BASELINE, self.train, {'split': 1}, state=self.state)
        self.memory = ExperimentMemory(self.root)
        self.session = self.make_session(1)

    def make_session(self, generation):
        directory = self.root / f'gen{generation:03d}'
        directory.mkdir()
        policy, rules = directory / 'scorer.py', directory / 'rules.md'
        policy.write_text(BASELINE)
        rules.write_text('TRAIN-only fixture')
        return TrainingExperiments(directory, policy, rules, 'split', components(BASELINE), self.train,
            self.parent, self.state, {'split': 1}, 120, 0, self.memory, max_evaluations=2, generation=generation)

    def test_grid_cache_restore_and_code_changes_refit(self):
        proposal(self.session, grid={'scale': [1, 2]})
        response = self.session.train_classifier()
        self.assertEqual(response['status'], 'trained')
        self.assertEqual(self.session.evaluations_used, 2)
        self.assertEqual(self.session.submitted()[0]['report']['macro_precision'], 1)
        with patch('proofreader_evolve.harness.train_experiments.fit_classifier') as fit:
            self.session.train_classifier()
            self.session.training_path.write_text('invalid pending edit')
            self.session.restore('gen001/attempt001')
        fit.assert_not_called()
        self.assertEqual(self.session.training_path.read_text(), PROGRAM)
        self.session = self.make_session(2)
        proposal(self.session, grid={'scale': [1, 2]})
        with patch('proofreader_evolve.harness.train_experiments.fit_classifier') as fit:
            self.session.train_classifier()
        fit.assert_not_called()
        self.assertEqual(self.session.evaluations_used, 0)
        changed = PROGRAM.replace('np.abs(x - center)', '(x - center) ** 2')
        proposal(self.session, program=changed)
        with patch('proofreader_evolve.harness.train_experiments.fit_classifier', wraps=fit_classifier) as fit:
            self.session.train_classifier()
        self.assertEqual(fit.call_count, 1)

    def test_invalid_requests_and_failed_fits(self):
        proposal(self.session, program='def fit(): pass')
        with self.assertRaisesRegex(ValueError, 'must define'):
            self.session.train_classifier()
        proposal(self.session, grid={'scale': [1, 2, 3]})
        with self.assertRaisesRegex(ValueError, 'needs 3'):
            self.session.train_classifier()
        self.assertEqual(self.session.evaluations_used, 0)
        proposal(self.session)
        with patch('proofreader_evolve.harness.train_experiments.fit_classifier',
                   side_effect=ClassifierTrainingError('timed out')):
            response = self.session.train_classifier()
        self.assertEqual(response['status'], 'no_successful_candidate')
        self.assertEqual(self.session.evaluations_used, 1)
        self.assertIsNone(self.session.attempts[0]['entry']['train'])
        self.assertIsNone(self.session.attempts[0]['entry']['candidate_sha256'])

    def test_tune_and_submission_bind_source_parameters_and_artifacts(self):
        proposal(self.session)
        self.session.train_classifier()
        fitted, _ = self.session.submitted()
        self.session.training_path.write_text(PROGRAM + '\n# unmeasured edit\n')
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.session.submitted()
        self.session.restore('best')
        self.session.search_plan = {'mode': 'tune', 'search_parent': fitted['entry']}
        proposal(self.session, program=PROGRAM.replace('np.abs(x - center)', '(x - center) ** 2'))
        with self.assertRaisesRegex(ValueError, 'Tune mode fixes'):
            self.session.train_classifier()
        proposal(self.session, parameters={'scale': 3})
        with self.assertRaisesRegex(ValueError, 'not been fitted'):
            self.session.evaluate()
        self.session.train_classifier()
        model = frozen_model(self.session.submitted()[0]['component'])
        self.assertEqual(model['config']['parameters']['scale'], 3)
        forged = deepcopy(model)
        forged['config']['parameters']['scale'] = 4
        self.session.policy_path.write_text(render_model(forged))
        with self.assertRaisesRegex(ValueError, 'immutable'):
            self.session.evaluate()


class ClassifierDriverTests(unittest.TestCase):
    def test_train_only_fit_frozen_validation_and_portable_resume(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            seed = Path(tmp) / 'seed.py'
            seed.write_text(BASELINE)
            args = driver.parse_args(['--selection-protocol', 'in_sample', '--train-brains', '1', '--validation-brains', '2', '--split-k', '1',
                                      '--generations', '1', '--start-from', str(seed), '--runs-dir', tmp])
            async def revise(run_dir, policy, rules, report, model, *, experiments):
                self.assertEqual(set(experiments.train), {'1'})
                proposal(experiments)
                self.assertEqual(experiments.train_classifier()['status'], 'trained')
                return {'summary': 'Fit TRAIN and submit frozen artifacts', 'cost_usd': 0}
            with patch.object(driver.pc, 'resolve_detector_runs', return_value={}), \
                    patch.object(driver, 'ensure_native_tables', side_effect=lambda brain, *a, **k: fixture(brain)), \
                    patch('proofreader_evolve.harness.train_experiments.fit_classifier', wraps=fit_classifier) as fit:
                path = asyncio.run(driver.run(args, revise_fn=revise))
                self.assertEqual(fit.call_count, 1)
                self.assertEqual(set(fit.call_args.args[0]), {'1'})
                exported = path / 'best_scorer.py'
                model = frozen_model(components(exported.read_text())['split'])
                self.assertEqual(model['training_brains'], ['1'])
                self.assertTrue(json.loads((path / 'ledger.jsonl').read_text().splitlines()[0])['accepted'])
                args.start_from, args.generations = exported, 0
                resumed = asyncio.run(driver.run(args, revise_fn=revise))
                self.assertEqual(fit.call_count, 1)
                verify_artifacts(model, resumed / 'model_artifacts')
                self.assertEqual(json.loads((resumed / 'final.json').read_text())['validation']['macro_precision'], 1)
                args.train_brains, args.validation_brains = ['4'], ['1']
                with patch.object(driver.scoring, 'evaluate') as evaluate:
                    with self.assertRaisesRegex(ValueError, 'fitted on a development-validation brain'):
                        asyncio.run(driver.run(args, revise_fn=revise))
                evaluate.assert_not_called()


if __name__ == '__main__':
    unittest.main()

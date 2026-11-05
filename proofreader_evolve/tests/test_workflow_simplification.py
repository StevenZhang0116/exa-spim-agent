"""Storage failure boundaries, prompt obligations and PNG-only access; no API."""
import asyncio
import io
import os
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from proofreader_evolve.cli import run_precision_evolution as driver
from proofreader_evolve.harness import selection_protocol as selection
from proofreader_evolve.harness.reviser_access import make_guard
from proofreader_evolve.harness.train_experiments import save_state
from proofreader_evolve.tests.test_trajectory import result_message


class StorageTests(unittest.TestCase):
    def test_selection_shares_bytes_without_overwriting_source_on_later_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fit, attempt = root / 'fit', root / 'attempt'
            fit.mkdir(); attempt.mkdir()
            report = {'cells': {'brain/merge': {}}}
            state = {'brain/merge': {'scores': np.array([0.1, 0.9]), 'chosen': np.array([1])}}
            selection.save_selection(fit, report, state)
            original = (fit / 'selection_state.npz').read_bytes()
            selection.save_selection(attempt, report, state, source=fit)
            self.assertTrue(os.path.samefile(fit / 'selection_state.npz', attempt / 'selection_state.npz'))
            _, loaded = selection.load_selection(attempt)
            for key in ('scores', 'chosen'):
                np.testing.assert_array_equal(loaded['brain/merge'][key], state['brain/merge'][key])
            state['brain/merge']['scores'] = np.array([0.7, 0.3])
            selection.save_selection(attempt, report, state)
            self.assertEqual((fit / 'selection_state.npz').read_bytes(), original)
            self.assertFalse(os.path.samefile(fit / 'selection_state.npz', attempt / 'selection_state.npz'))

    def test_fold_cleanup_requires_successful_score_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / 'selection_folds/fold0/models/digest'
            models.mkdir(parents=True)
            (models / 'tree.txt').write_text('temporary model')
            metadata = root / 'selection_folds/fold0/model.json'
            metadata.write_text('{}')
            final = root / 'model_artifacts/final.bin'
            final.parent.mkdir(); final.write_text('accepted model')
            with patch.object(selection.np, 'savez', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    selection.save_selection(root, {'cells': {}}, {})
            self.assertTrue(models.is_dir())
            selection.save_selection(root, {'cells': {}}, {})
            self.assertFalse(models.parent.exists())
            self.assertTrue(metadata.is_file())
            self.assertEqual(final.read_text(), 'accepted model')

    def test_train_snapshot_only_serializes_target_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train_state.npz'
            state = {f'brain/{kind}': {'scores': np.array([1., 2.]), 'chosen': np.array([1])}
                     for kind in ('merge', 'split')}
            save_state(path, state, kind='split')
            with np.load(path) as data:
                self.assertEqual(set(data.files), {'brain/split:scores', 'brain/split:chosen'})
                np.testing.assert_array_equal(data['brain/split:scores'], state['brain/split']['scores'])

    def test_ablation_keeps_fold_models_when_prediction_or_save_fails(self):
        from proofreader_evolve.harness import feature_ablation as ablation
        from proofreader_evolve.tests.test_feature_discovery import training_fixture, PROGRAM
        train = training_fixture()
        frames = {'794495': train['794495'].tables['merge'].features}
        split = ablation.grouped_folds(train, 'merge')
        with tempfile.TemporaryDirectory() as tmp:
            for phase in ('predict', 'save'):
                directory = Path(tmp) / phase
                def fit(*args, **kwargs):
                    store = Path(kwargs['artifact_store'])
                    store.mkdir(parents=True)
                    (store / 'model.bin').write_bytes(b'fold')
                    return 'mock', {}
                def predict(model, store, frame, kind, **kwargs):
                    if phase == 'predict':
                        raise ValueError('prediction failed')
                    return frame.local_signal.to_numpy()
                with patch.object(ablation, 'fit_classifier', side_effect=fit), \
                        patch.object(ablation, 'frozen_model', return_value={'columns': ['local_signal']}), \
                        patch.object(ablation, 'predict_model', side_effect=predict), \
                        patch.object(ablation.np, 'savez', side_effect=OSError('save failed')):
                    with self.assertRaisesRegex((ValueError, OSError), 'failed'):
                        ablation.measure_ablation(train, 'merge', PROGRAM, {'parameters': {}}, frames,
                            ['local_signal'], split, {'program_sha256': 'fixture'}, {'merge': 3}, directory,
                            classifier=True, charge=lambda: None)
                self.assertTrue(list(directory.glob('*/fold*/models/model.bin')))


class PreviewAccessTests(unittest.TestCase):
    def test_current_and_earlier_png_only_no_audit_or_alias_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = [root / f'gen{i:03d}/image_inspections' for i in (1, 2, 3)]
            for directory in dirs:
                directory.mkdir(parents=True)
                (directory / 'image.png').write_bytes(b'fixture')
                (directory / 'image.json').write_text('{"source": "host-only"}')
            guard, _ = make_guard(root, dirs[1].parent, [], root / 'audit.jsonl', readable_png_dirs=dirs[:2])
            def decision(path, tool='Read'):
                return asyncio.run(guard(tool, {'file_path': str(path)}, None)).behavior
            for directory in dirs[:2]:
                self.assertEqual(decision(directory / 'image.png'), 'allow')
                self.assertEqual(decision(directory / 'image.png', 'Write'), 'deny')
                self.assertEqual(decision(directory / 'image.json'), 'deny')
                (directory / 'audit.png').symlink_to(directory / 'image.json')
                self.assertEqual(decision(directory / 'audit.png'), 'deny')
            self.assertEqual(decision(dirs[2] / 'image.png'), 'deny')
            private = root / 'gen001/validation_descriptor_runs/leak.png'
            private.parent.mkdir(); private.write_bytes(b'private')
            (dirs[0] / 'leak.png').symlink_to(private)
            self.assertEqual(decision(dirs[0] / 'leak.png'), 'deny')
            # An allowed directory replaced after binding cannot expose a different tree.
            dirs[0].rename(dirs[0].with_name('original'))
            dirs[0].symlink_to(dirs[2], target_is_directory=True)
            self.assertEqual(decision(dirs[0] / 'image.png'), 'deny')


class PromptTests(unittest.TestCase):
    def test_short_prompt_retains_assignment_actual_budgets_and_submission_contract(self):
        captured = {}
        class Client:
            def __init__(self, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def query(self, prompt):
                captured['prompt'] = prompt
            async def receive_response(self):
                yield result_message()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); gen = root / 'gen002'; gen.mkdir()
            (root / 'gen001').mkdir()
            experiments = SimpleNamespace(target_kind='merge', remaining=5, max_evaluations=8,
                search_plan={'mode': 'explore', 'branch_role': 'reference',
                             'search_parent': {'experiment': 'gen001/attempt002'},
                             'investigation': {'required_source': 'raw_image'}, 'followup_required': True},
                descriptor_budget={'per_call_seconds': 700., 'generation_seconds': 1500.},
                descriptor_seconds_remaining=900., memory=SimpleNamespace(descriptor_runs=object()),
                mcp_server=Mock(return_value=object()))
            with patch('claude_agent_sdk.ClaudeSDKClient', Client), \
                    patch.object(driver, 'build_options', return_value=(object(), {})), \
                    patch.object(driver, 'bind_session_options', return_value=object()) as bind, \
                    redirect_stdout(io.StringIO()):
                asyncio.run(driver.revise(root, gen/'scorer.py', gen/'rules.md', gen/'train_feedback.json',
                                          'fixture', experiments=experiments))
            prompt = captured['prompt']
            for required in ('raw_image', 'followup_required', 'gen001/attempt002', 'NEW paired TRAIN',
                             'Incomplete assignments', 'successfully evaluated snapshot',
                             'Validation data and logs are unavailable', '5 remaining of 8',
                             '700 s/call', '1500 s/generation', '900 s remaining',
                             'classifier_guide.md', 'descriptor_guide.md', 'install_packages',
                             'mcp__training__search_memory'):
                self.assertIn(required, prompt)
            self.assertLess(len(prompt.encode()), 7000)
            self.assertEqual(set(bind.call_args.kwargs['readable_png_dirs']),
                             {gen/'image_inspections', root/'gen001/image_inspections'})

    def test_removed_context_cache_flags_are_rejected(self):
        from proofreader_evolve.cli import preflight
        from contextlib import redirect_stderr
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as driver_error:
                driver.parse_args(['--context-cache', '/unused'])
            with self.assertRaises(SystemExit) as preflight_error:
                preflight.main(['--brains', '802449', '--context-cache', '/unused'])
        self.assertEqual(driver_error.exception.code, 2)
        self.assertEqual(preflight_error.exception.code, 2)

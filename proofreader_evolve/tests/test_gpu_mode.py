"""GPU mode of the model sandbox and patches for every GT-labeled row (2026-11-17).

The switch, the selection, the transport limit, the early stop of an oversized read, the patch
cache's location and byte budget are tested on synthetic fixtures anywhere. `GpuWorkerTests`
runs the start-up check and trains and applies one fixed small 3D network through the real fit
and predict workers inside the sandbox; it needs a GPU allocated to the test process and is
skipped otherwise (run it on a GPU node: `python -m unittest proofreader_evolve.tests.test_gpu_mode`).
No brain caches, cloud reads or LLM calls."""

import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from proofreader_evolve.cli import run_precision_evolution as driver
from proofreader_evolve.harness import image_features, model_execution, model_sandbox
from proofreader_evolve.harness.classifier_contract import frozen_model
from proofreader_evolve.harness.classifier_training import fit_classifier
from proofreader_evolve.harness.image_context import CandidateImageStore, file_hash
from proofreader_evolve.harness.image_contract import SELECTION_MODES, image_spec
from proofreader_evolve.harness.model_execution import ModelExecutionError, predict_model
from proofreader_evolve.tests.test_feature_discovery import training_fixture

RAW = """LOCAL_IMAGE = {settings!r}
def fit(X_train, y_train, artifact_dir, params, *, images):
    pass
def predict(X, artifact_dir, params, *, images):
    return [0. for _ in images]
"""

GUIDE = Path(__file__).resolve().parents[1] / 'artifacts' / 'image_context_guide.md'


def guide_example():
    """The network program printed in the agents' image guide, so that what the guide shows is
    what the tests run."""
    text = GUIDE.read_text()
    begin = text.index('```python\n# Example: a small 3D network on image patches') + len('```python\n')
    return text[begin:text.index('\n```', begin)] + '\n'


# The guide's example falls back to the CPU; the GPU tests must fail when it does.
REQUIRE_GPU = """

_fit_of_the_example = fit


def fit(X_train, y_train, artifact_dir, params, *, images):
    if not torch.cuda.is_available():
        raise RuntimeError('no GPU inside the worker')
    _fit_of_the_example(X_train, y_train, artifact_dir, params, images=images)
"""


class SwitchTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(model_execution.set_gpu, False)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def launch(self, mode):
        """The request file and environment run_worker hands to a worker of this mode."""
        root = self.root / f'{mode}{len(list(self.root.iterdir()))}'
        root.mkdir()

        class Launched(Exception):
            pass
        seen = {}

        def popen(command, **kwargs):
            seen.update(env=kwargs['env'], request=json.loads((root / 'request.json').read_text()))
            raise Launched()
        with patch.object(model_execution.subprocess, 'Popen', side_effect=popen), \
                patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3'}), self.assertRaises(Launched):
            model_execution.run_worker(root, {'mode': mode, 'threads': 1})
        return seen

    def test_gpu_is_hidden_unless_the_run_enables_it(self):
        self.assertFalse(model_execution.gpu_enabled())
        for mode in ('fit', 'predict', 'describe'):
            seen = self.launch(mode)
            self.assertEqual(seen['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertFalse(seen['request']['gpu'])

    def test_only_fit_and_predict_workers_receive_the_gpu(self):
        model_execution.set_gpu(True)
        for mode in ('fit', 'predict'):
            seen = self.launch(mode)
            self.assertEqual(seen['env']['CUDA_VISIBLE_DEVICES'], '3')  # the scheduler's value, unchanged
            self.assertTrue(seen['request']['gpu'])
        for mode in ('describe', 'extract', 'extract_image', 'analyze_volume'):
            seen = self.launch(mode)
            self.assertEqual(seen['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertFalse(seen['request']['gpu'])

    def test_a_requested_gpu_that_is_not_there_fails_at_once(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), self.assertRaisesRegex(ValueError, 'no GPU is allocated'):
            model_execution.gpu_device()
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), \
                patch.object(model_execution.subprocess, 'run', side_effect=FileNotFoundError('nvidia-smi')), \
                self.assertRaisesRegex(ValueError, 'could not list a device'):
            model_execution.gpu_device()
        listing = SimpleNamespace(stdout='NVIDIA L40S, 46068\n')
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), \
                patch.object(model_execution.subprocess, 'run', return_value=listing):
            self.assertEqual(model_execution.gpu_device(),
                             {'name': 'NVIDIA L40S', 'memory_mb': 46068, 'devices': 1, 'cuda_visible_devices': '0'})

    def test_driver_options(self):
        base = ['--train-brains', '1', '--validation-brains', '2']
        args = driver.parse_args(base)
        self.assertEqual((args.gpu, args.image_transport_gb, args.image_cache_gb), (False, 1., 20.))
        self.assertIsNone(args.image_cache_dir)
        args = driver.parse_args([*base, '--gpu', '--image-transport-gb', '16', '--image-cache-gb', '64',
                                  '--image-cache-dir', 'local'])
        self.assertEqual((args.gpu, args.image_transport_gb, args.image_cache_gb, args.image_cache_dir),
                         (True, 16., 64., 'local'))
        with self.assertRaises(SystemExit):
            driver.parse_args([*base, '--image-transport-gb', '0'])

    def test_the_prompt_names_the_option_only_in_a_gpu_run(self):
        self.assertEqual(driver.gpu_prompt_text(), '')
        model_execution.set_gpu(True)
        text = driver.gpu_prompt_text()
        for phrase in ('A GPU is available to fit and predict', "selection_mode 'judgeable'",
                       'image_context_guide.md', 'an option, not a requirement'):
            self.assertIn(phrase, text)

    def test_the_guides_example_is_a_valid_raw_patch_program(self):
        program, spec = image_spec(guide_example())
        self.assertTrue(spec['raw_patches'])
        self.assertEqual((spec['selection_mode'], spec['level'], spec['radius_um']), ('judgeable', 1, 16.0))
        compile(program, 'guide_example', 'exec')
        for required in ('cudnn.deterministic = True', 'cudnn.benchmark = False', "torch.cuda.is_available()",
                         'weights_only=True'):
            self.assertIn(required, program)

    def test_the_sandbox_check_reports_a_worker_that_cannot_use_cuda(self):
        with self.assertRaisesRegex(ValueError, 'needs GPU mode'):
            model_execution.gpu_worker_check()
        model_execution.set_gpu(True)
        with patch.object(model_execution, 'run_worker', side_effect=ModelExecutionError('RuntimeError: no CUDA')), \
                self.assertRaisesRegex(ValueError, 'a sandboxed worker could not use CUDA: ModelExecutionError'):
            model_execution.gpu_worker_check()

    def test_the_patch_cache_location(self):
        run_dir = self.root / 'runs' / 'precision_x'
        run_dir.mkdir(parents=True)
        self.assertEqual(driver.image_cache_directory(None, run_dir, 1), run_dir / 'image_cache')
        with patch.object(driver.tempfile, 'gettempdir', return_value=str(self.root / 'node_local')):
            local = driver.image_cache_directory('local', run_dir, 1)
        self.assertEqual(local, (self.root / 'node_local' / 'image_cache_precision_x').resolve())
        self.assertTrue(local.is_dir())
        named = driver.image_cache_directory(str(self.root / 'elsewhere'), run_dir, 1)
        self.assertEqual(named, (self.root / 'elsewhere' / 'image_cache_precision_x').resolve())
        with self.assertRaisesRegex(ValueError, 'less than the cache budget'):
            driver.image_cache_directory(str(self.root / 'elsewhere'), run_dir, 2**62)
        with self.assertRaisesRegex(ValueError, 'cannot create'):
            (self.root / 'a_file').write_text('x')
            driver.image_cache_directory(str(self.root / 'a_file'), run_dir, 1)

    def test_the_sandbox_additions_are_the_probed_ones(self):
        self.assertEqual(model_sandbox.GPU_SYSCALLS, ('ioctl', 'eventfd2', 'setsockopt', 'bind', 'listen'))
        self.assertIn('/proc/self/task', model_sandbox.GPU_WRITE_PATHS)
        self.assertTrue(all(path.startswith('/dev/nvidia') for path in model_sandbox.gpu_device_paths()))


class JudgeableSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(image_features.set_transport_limit)
        self.root = Path(temporary.name)
        self.table = training_fixture()['794495'].tables['merge']
        truth = np.asarray(self.table.truth, dtype=float)
        truth[[1, 4, 9]] = np.nan  # not judgeable by GT
        self.table.truth = truth
        self.paths, self.calls = {}, []
        for row in range(len(self.table.features)):
            self.paths[row] = self.root / f'pixels{row}.npz'
            np.savez(self.paths[row], image_zyx=np.full((3, 4, 5), row + 1, dtype=np.uint16),
                     valid_zyx=np.ones((3, 4, 5), dtype=bool))
        self.table.image_context = SimpleNamespace(directory=self.root / 'cache', trace=Mock(),
                                                   patch=Mock(side_effect=self.get_patch))

    def get_patch(self, table, index, spec, occurrence, **options):
        self.calls.append((index, occurrence, options.get('trace', True)))
        return self.paths[index], {'patch_sha256': file_hash(self.paths[index]), 'occurrences_total': 1}

    def program(self, **settings):
        return RAW.format(settings={'raw_patches': True, 'selection_mode': 'judgeable', **settings})

    def test_contract(self):
        self.assertIn('judgeable', SELECTION_MODES)
        spec = image_spec(self.program(max_total_mb=32768))[1]
        self.assertEqual((spec['selection_mode'], spec['max_total_mb'], spec['selection_k']), ('judgeable', 32768, None))
        with self.assertRaisesRegex(ValueError, 'selection_k'):
            image_spec(self.program(selection_k=5))
        with self.assertRaisesRegex(ValueError, 'selection_mode'):
            image_spec(RAW.format(settings={'raw_patches': True, 'selection_mode': 'labeled'}))

    def test_every_labeled_row_and_no_other_gets_patches(self):
        frame = image_features.augmented_image_features(self.program(), self.table, self.table.features)
        labeled = [row for row in range(12) if row not in (1, 4, 9)]
        self.assertEqual(sorted(frame.attrs['_image_rows']), labeled)
        self.assertEqual(np.flatnonzero(frame['image_available'].to_numpy() == 1).tolist(), labeled)
        coverage = frame.attrs['image_context']['selection_coverage']
        self.assertEqual((coverage['mode'], coverage['selected_rows'], coverage['pool_rows']), ('judgeable', 9, 12))
        self.assertTrue(all(trace for _, _, trace in self.calls))  # a small selection keeps its per-patch events

    def test_a_large_selection_is_read_by_several_readers_first(self):
        with patch.object(image_features, 'PARALLEL_READ_ROWS', 4):
            frame = image_features.augmented_image_features(self.program(), self.table, self.table.features)
        self.assertEqual(len(frame.attrs['_image_rows']), 9)
        self.assertEqual(len(self.calls), 18)  # every row once by the readers, once by the loop
        self.assertFalse(any(trace for _, _, trace in self.calls))
        self.assertEqual(sorted(index for index, _, _ in self.calls[:9]), sorted(frame.attrs['_image_rows']))

    def test_a_failed_read_stops_the_readers_and_surfaces(self):
        def broken(table, index, spec, occurrence, **options):
            raise OSError('store unreachable')
        self.table.image_context.patch = Mock(side_effect=broken)
        with patch.object(image_features, 'PARALLEL_READ_ROWS', 4), self.assertRaisesRegex(OSError, 'store unreachable'):
            image_features.augmented_image_features(self.program(), self.table, self.table.features)

    def test_an_oversized_request_stops_reading_early(self):
        size = self.paths[0].stat().st_size
        image_features.set_transport_limit(3 * size)   # nine rows are selected; three fit
        with patch.object(image_features, 'PARALLEL_READ_ROWS', 4), patch.object(image_features, 'IMAGE_READERS', 1), \
                self.assertRaisesRegex(ValueError, 'raw-patch limit'):
            image_features.augmented_image_features(self.program(), self.table, self.table.features)
        # The fourth patch crosses the limit; at most one more read was already under way.
        self.assertLessEqual(len(self.calls), 5)
        self.assertLess(len(self.calls), 9)

    def test_a_slow_first_read_does_not_let_the_other_readers_run_past_the_limit(self):
        # Results come back in input order; the count must not wait for the first one.
        size = self.paths[0].stat().st_size
        first = min(row for row in range(len(self.table.features)) if row not in (1, 4, 9))
        released = threading.Event()

        def get_patch(table, index, spec, occurrence, **options):
            self.calls.append(index)
            if index == first:
                released.wait(10)   # held until the other reader has stopped
            return self.paths[index], {'patch_sha256': file_hash(self.paths[index]), 'occurrences_total': 1}
        self.table.image_context.patch = Mock(side_effect=get_patch)
        image_features.set_transport_limit(3 * size)   # nine rows are selected; three fit
        outcome = []

        def request():
            try:
                image_features.augmented_image_features(self.program(), self.table, self.table.features)
            except ValueError as exc:
                outcome.append(str(exc))
        with patch.object(image_features, 'PARALLEL_READ_ROWS', 4), patch.object(image_features, 'IMAGE_READERS', 2):
            thread = threading.Thread(target=request)
            thread.start()
            deadline = time.monotonic() + 10
            while len(self.calls) < 5 and time.monotonic() < deadline:
                time.sleep(0.01)
            time.sleep(0.3)             # room for further reads, if any were still being started
            reads_while_blocked = len(self.calls)
            released.set()
            thread.join(20)
        # The held read plus four by the other reader (the fourth crosses the limit); nine before the fix.
        self.assertEqual(reads_while_blocked, 5)
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(len(outcome), 1)
        self.assertIn('raw-patch limit', outcome[0])

    def test_a_failed_read_stops_the_other_readers(self):
        failing = min(row for row in range(len(self.table.features)) if row not in (1, 4, 9))

        def get_patch(table, index, spec, occurrence, **options):
            self.calls.append(index)
            if index == failing:
                raise TimeoutError('Image preparation budget exhausted')
            return self.paths[index], {'patch_sha256': file_hash(self.paths[index]), 'occurrences_total': 1}
        self.table.image_context.patch = Mock(side_effect=get_patch)
        with patch.object(image_features, 'PARALLEL_READ_ROWS', 4), patch.object(image_features, 'IMAGE_READERS', 1), \
                self.assertRaises(TimeoutError):
            image_features.augmented_image_features(self.program(), self.table, self.table.features)
        self.assertEqual(self.calls, [failing])

    def test_the_size_of_a_worker_call_is_checked_before_anything_is_copied(self):
        frame = self.table.features.copy()
        frame.attrs['_image_rows'] = {row: [str(self.paths[row])] for row in range(6)}
        image_features.set_transport_limit(4 * self.paths[0].stat().st_size)
        with self.assertRaisesRegex(ValueError, 'Image transport exceeds'):
            image_features.stage_images(self.root / 'worker', [(frame, list(range(6)))])
        self.assertFalse((self.root / 'worker' / 'images').exists())
        (self.root / 'fits').mkdir()
        staged = image_features.stage_images(self.root / 'fits', [(frame, list(range(4)))])
        self.assertEqual((staged['rows'], staged['rows_with_images']), (4, 4))

    def test_the_transport_limit_is_the_runs_setting(self):
        size = self.paths[0].stat().st_size
        image_features.set_transport_limit(4 * size)
        with self.assertRaisesRegex(ValueError, 'raw-patch limit'):
            image_features.augmented_image_features(self.program(), self.table, self.table.features)
        image_features.set_transport_limit(16 * size)
        frame = image_features.augmented_image_features(self.program(), self.table, self.table.features)
        self.assertEqual(len(frame.attrs['_image_rows']), 9)
        self.assertEqual(frame.attrs['image_context']['limits']['total_bytes'], 16 * size)
        with self.assertRaises(ValueError):
            image_features.set_transport_limit(0)


class CacheBudgetTests(unittest.TestCase):
    """The byte budget of the patch cache holds when several readers store patches at once."""
    def test_concurrent_commits_never_exceed_the_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staged = []
            for index in range(24):
                directory = root / 'staging' / str(index)
                directory.mkdir(parents=True)
                (directory / 'patch.npz').write_bytes(b'x' * 1000)
                staged.append(directory)
            store = object.__new__(CandidateImageStore)
            store.directory = root / 'cache'
            store.directory.mkdir()
            store.max_cache_bytes, store.cache_bytes = 10_500, 0   # room for ten patches
            store._cache_lock = threading.Lock()
            start, refused = threading.Barrier(len(staged)), []

            def commit(directory):
                start.wait()
                try:
                    store._commit(directory, store.directory / directory.name)
                except ValueError as exc:
                    refused.append(str(exc))
            threads = [threading.Thread(target=commit, args=(directory,)) for directory in staged]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            kept = list(store.directory.iterdir())
            self.assertEqual(len(kept), 10)
            self.assertEqual(store.cache_bytes, 10_000)
            self.assertEqual(sum((entry / 'patch.npz').stat().st_size for entry in kept), 10_000)
            self.assertEqual(len(refused), 14)
            self.assertTrue(all('budget exhausted' in message for message in refused))

    def test_a_patch_stored_twice_is_counted_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = object.__new__(CandidateImageStore)
            store.directory, store.max_cache_bytes, store.cache_bytes = root, 10_000, 0
            store._cache_lock = threading.Lock()
            for name in ('first', 'second'):
                (root / name).mkdir()
                (root / name / 'patch.npz').write_bytes(b'x' * 1000)
                store._commit(root / name, root / 'target')
            self.assertEqual(store.cache_bytes, 1000)


def gpu_allocated():
    return bool(os.environ.get('CUDA_VISIBLE_DEVICES')) and shutil.which('nvidia-smi') is not None


@unittest.skipUnless(gpu_allocated(), 'needs a GPU allocated to the test process')
class GpuWorkerTests(unittest.TestCase):
    """The guide's example network through the real sandboxed fit and predict workers."""
    def setUp(self):
        self.addCleanup(model_execution.set_gpu, False)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.train = training_fixture()
        table = self.train['794495'].tables['merge']
        self.truth = np.asarray(table.truth, dtype=float)
        self.frame = table.features.copy()
        generator = np.random.default_rng(0)
        rows = {}
        for row, label in enumerate(self.truth):
            # Noise everywhere; a positive has a bright block in the middle. The shape is not a
            # cube, as real patches are not (the example resamples them).
            volume = generator.integers(90, 110, size=(8, 12, 12)).astype(np.uint16)
            if label:
                volume[2:6, 4:8, 4:8] += 400
            path = self.root / f'patch{row}.npz'
            np.savez(path, image_zyx=volume, valid_zyx=np.ones(volume.shape, dtype=bool))
            rows[row] = [str(path)]
        self.frame.attrs['_image_rows'] = rows
        self.program = guide_example() + REQUIRE_GPU

    def fit(self, name):
        return fit_classifier(self.train, 'merge', {'parameters': {'epochs': 80, 'lr': 0.01}}, self.root / name,
                              program=self.program, artifact_store=self.root / 'models',
                              feature_frames={'794495': self.frame},
                              fit_rows={'794495': np.arange(len(self.frame))}, timeout=300)

    def test_fit_save_reload_and_deterministic_prediction_on_the_gpu(self):
        model_execution.set_gpu(True)
        component, summary = self.fit('fit')
        model = frozen_model(component)
        self.assertIn('network.pt', json.dumps(summary['artifact_files']))
        first = predict_model(model, self.root / 'models', self.frame, 'merge', timeout=300)
        second = predict_model(model, self.root / 'models', self.frame, 'merge', timeout=300)
        self.assertEqual(first.shape, (len(self.frame),))
        np.testing.assert_array_equal(first, second)  # two fresh workers, the same frozen state
        self.assertGreater(first[self.truth == 1].min(), first[self.truth == 0].max())

    def test_the_start_up_check_runs_cuda_in_a_sandboxed_worker(self):
        model_execution.set_gpu(True)
        device = model_execution.gpu_worker_check()
        self.assertTrue(device['name'])
        self.assertGreater(device['memory_mb'], 1000)
        self.assertTrue(device['cuda'])

    def test_the_frozen_network_also_runs_on_the_cpu(self):
        model_execution.set_gpu(True)
        component, _ = self.fit('fit')
        gpu = predict_model(frozen_model(component), self.root / 'models', self.frame, 'merge', timeout=300)
        model_execution.set_gpu(False)  # the same frozen model in a worker without a GPU
        cpu = predict_model(frozen_model(component), self.root / 'models', self.frame, 'merge', timeout=300)
        np.testing.assert_allclose(cpu, gpu, rtol=1e-3, atol=1e-3)  # close, not guaranteed identical

    def test_without_the_switch_the_same_worker_has_no_gpu(self):
        with self.assertRaisesRegex(ModelExecutionError, 'no GPU inside the worker'):
            self.fit('cpu_only')


if __name__ == '__main__':
    unittest.main()

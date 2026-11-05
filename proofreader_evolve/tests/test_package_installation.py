"""Dependency installation transport; mocked pip never changes the real environment."""

import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from proofreader_evolve.harness import package_installation as installer


class PackageInstallationTests(unittest.TestCase):
    def test_install_targets_current_environment_preserves_versions_and_refreshes_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = Path(tmp)
            (gen / 'model_environment.json').write_text(json.dumps({'device': 'cpu', 'packages': {}}))
            before = {'numpy': '2.2.0'}
            after = {**before, 'new-model': '1.0'}

            def launch(command, **kwargs):
                self.assertEqual(command[:6], [sys.executable, '-I', '-m', 'pip', '--isolated', 'install'])
                self.assertEqual(command[command.index('--prefix') + 1], sys.prefix)
                constraints = Path(command[command.index('--constraint') + 1])
                self.assertEqual(constraints.read_text(), 'numpy===2.2.0\n')
                self.assertEqual(command[-1], 'new-model>=1.0')
                self.assertTrue(kwargs['start_new_session'])
                kwargs['stdout'].write(b'Successfully installed new-model-1.0\n')
                return Mock(wait=Mock(return_value=0))

            with patch.object(installer, 'installed_packages', side_effect=[before, after]), \
                    patch.object(installer.subprocess, 'Popen', side_effect=launch):
                result = installer.install_packages(gen, ['new-model>=1.0'])
            self.assertEqual(result['status'], 'installed')
            self.assertEqual(result['changed_packages'], {'new-model': '1.0'})
            environment = json.loads((gen / 'model_environment.json').read_text())
            self.assertEqual(environment, {'device': 'cpu', 'packages': after})
            self.assertIn('Successfully installed', result['output'])
            self.assertEqual(json.loads((gen / 'package_installs/install001/result.json').read_text()), result)

    def test_failure_returns_pip_diagnostic_and_records_partial_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            def launch(command, **kwargs):
                kwargs['stdout'].write(b'ERROR: dependency conflict\n')
                return Mock(wait=Mock(return_value=1))
            with patch.object(installer, 'installed_packages', side_effect=[{}, {'dependency': '1.0'}]), \
                    patch.object(installer.subprocess, 'Popen', side_effect=launch):
                result = installer.install_packages(Path(tmp), ['new-model'])
            self.assertEqual(result['status'], 'error')
            self.assertEqual(result['changed_packages'], {'dependency': '1.0'})
            self.assertIn('dependency conflict', result['output'])

    def test_timeout_stops_installer_and_build_children(self):
        process = Mock(pid=12345)
        process.wait.side_effect = [subprocess.TimeoutExpired('pip', 600), -9]
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(installer, 'installed_packages', return_value={}), \
                patch.object(installer.subprocess, 'Popen', return_value=process), \
                patch.object(installer.os, 'killpg') as kill:
            result = installer.install_packages(Path(tmp), ['new-model'])
        kill.assert_called_once_with(12345, signal.SIGKILL)
        self.assertEqual(result['status'], 'error')
        self.assertIn('600 seconds', result['error'])

    def test_rejects_options_paths_and_urls_before_starting_pip(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(installer.subprocess, 'Popen') as run:
            for packages in ([], 'numpy', ['--target=/tmp'], ['/tmp/package'],
                             ['model @ https://example.com/model.whl'], ['numpy; echo secret'], [None]):
                with self.subTest(packages=packages), self.assertRaises(ValueError):
                    installer.install_packages(Path(tmp), packages)
            run.assert_not_called()
            self.assertFalse((Path(tmp) / 'package_installs').exists())


if __name__ == '__main__':
    unittest.main()

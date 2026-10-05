"""Landlock bootstrap regressions using mocked OS calls, without installing a sandbox."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from proofreader_evolve.harness import model_sandbox


class LandlockBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.stack = self.enterContext(ExitStack())
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.stack.enter_context(patch.object(model_sandbox.sys, 'platform', 'linux'))
        self.machine = self.stack.enter_context(
            patch.object(model_sandbox.platform, 'machine', return_value='x86_64'))
        # Reproduce the failing Python build without modifying the real os module.
        self.os = SimpleNamespace(open=Mock(return_value=102), close=Mock(), O_CLOEXEC=0o2000000)
        self.stack.enter_context(patch.object(model_sandbox, 'os', self.os))
        self.libc = Mock()
        self.libc.prctl.return_value = 0
        self.cdll = self.stack.enter_context(
            patch.object(model_sandbox.ctypes, 'CDLL', return_value=self.libc))

    def test_missing_o_path_uses_linux_flags_and_enforces_landlock(self):
        for architecture in ('x86_64', 'aarch64'):
            with self.subTest(architecture=architecture):
                self.machine.return_value = architecture
                self.os.open.reset_mock()
                self.os.close.reset_mock()
                self.libc.reset_mock()
                # ABI probe, ruleset creation, path rule, enforcement.
                self.libc.syscall.side_effect = [3, 101, 0, 0]
                model_sandbox.restrict_filesystem([self.root], [])
                self.os.open.assert_called_once_with(self.root, 0o12000000)
                self.assertEqual([c.args[0] for c in self.libc.syscall.call_args_list],
                                 [444, 444, 445, 446])
                self.libc.prctl.assert_called_once_with(38, 1, 0, 0, 0)
                self.assertEqual(self.os.close.call_args_list, [call(102), call(101)])

    def test_missing_close_on_exec_also_preserves_linux_flags(self):
        del self.os.O_CLOEXEC
        self.libc.syscall.side_effect = [3, 101, 0, 0]
        model_sandbox.restrict_filesystem([self.root], [])
        self.os.open.assert_called_once_with(self.root, 0o12000000)

    def test_native_flags_are_used_when_available(self):
        self.os.O_PATH = 0o10000000
        self.libc.syscall.side_effect = [3, 101, 0, 0]
        model_sandbox.restrict_filesystem([self.root], [])
        self.os.open.assert_called_once_with(self.root, self.os.O_PATH | self.os.O_CLOEXEC)

    def test_unsupported_platform_never_uses_linux_fallback(self):
        self.machine.return_value = 'unsupported'
        with self.assertRaisesRegex(RuntimeError, 'require Linux'):
            model_sandbox.restrict_filesystem([self.root], [])
        self.cdll.assert_not_called()
        self.os.open.assert_not_called()

    def test_unavailable_landlock_still_fails_closed(self):
        for abi in (-1, 2):
            with self.subTest(abi=abi):
                self.libc.syscall.side_effect = [abi]
                with self.assertRaisesRegex(RuntimeError, 'isolation unavailable'):
                    model_sandbox.restrict_filesystem([self.root], [])
                self.os.open.assert_not_called()

    def test_enforcement_failure_propagates_and_closes_descriptors(self):
        self.libc.syscall.side_effect = [3, 101, 0, -1]
        with self.assertRaisesRegex(OSError, 'Landlock enforcement failed'):
            model_sandbox.restrict_filesystem([self.root], [])
        self.assertEqual(self.os.close.call_args_list, [call(102), call(101)])

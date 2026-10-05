"""Observable SDK trajectory tests with synthetic messages; no API or real data."""

import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from claude_agent_sdk import (AssistantMessage, UserMessage, SystemMessage, ResultMessage,
                             TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock)

from proofreader_evolve.cli import run_precision_evolution as driver
from proofreader_evolve.harness.trajectory import Trajectory
from proofreader_evolve.harness.reviser_access import make_guard


def result_message(error=False):
    return ResultMessage(subtype="error_max_turns" if error else "success", duration_ms=12,
                         duration_api_ms=10, is_error=error, num_turns=2,
                         session_id="fixture-session", total_cost_usd=.01,
                         usage={"input_tokens": 10}, result="Used evidence feature")


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.gen = self.root / "gen001"
        self.gen.mkdir()
        self.output = io.StringIO()
        redirect = redirect_stdout(self.output)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def events(self):
        return [json.loads(line) for line in (self.gen / "trajectory.jsonl").read_text().splitlines()]

    def test_events_are_durable_ordered_and_tool_payloads_are_complete(self):
        trace = Trajectory(self.gen)
        trace.sdk_message(SystemMessage(subtype="init", data={"private_metadata": "excluded"}))
        trace.sdk_message(AssistantMessage(model="test", content=[
            TextBlock(text="I will adjust evidence weighting."),
            ThinkingBlock(thinking="excluded thinking", signature="excluded signature"),
            ToolUseBlock(id="tool-1", name="Edit", input={"file_path": "scorer.py",
                        "old_string": "before", "new_string": "after"})]))
        # Inspect before session completion: each event must already be on disk.
        self.assertEqual(len(self.events()), 3)
        trace.sdk_message(UserMessage(content=[ToolResultBlock(tool_use_id="tool-1",
                          content="Edit was denied", is_error=True)]))
        trace.sdk_message(result_message())
        events = self.events()
        self.assertEqual([e["event"] for e in events],
                         ["sdk_status", "agent_text", "tool_call", "tool_result", "agent_result"])
        self.assertEqual(events[2]["input"]["new_string"], "after")
        self.assertEqual(events[3]["tool_use_id"], events[2]["tool_use_id"])
        self.assertTrue(events[3]["is_error"])
        self.assertEqual(events[-1]["cost_usd"], .01)
        transcript = (self.gen / "trajectory.txt").read_text()
        self.assertIn("Edit was denied", transcript)
        self.assertNotIn("excluded", transcript)
        self.assertNotIn("Edit scorer.py", self.output.getvalue())
        trace.emit('generation_start', 'Generation 1: split explore')
        self.assertIn('Generation 1: split explore', self.output.getvalue())

    def test_revise_logs_messages_and_preserves_result_before_sdk_error(self):
        for error in (False, True):
            with self.subTest(error=error):
                class Client:
                    def __init__(self, **kwargs):
                        pass

                    async def __aenter__(self):
                        return self

                    async def __aexit__(self, *args):
                        pass

                    async def query(self, prompt):
                        pass

                    async def receive_response(self):
                        yield AssistantMessage(model="test", content=[TextBlock(text="Plan: adjust evidence")])
                        yield result_message(error)

                with patch("claude_agent_sdk.ClaudeSDKClient", Client), \
                        patch.object(driver, "build_options", return_value=(object(), {})) as options, \
                        patch.object(driver, "bind_session_options", return_value=object()):
                    call = driver.revise(self.root, self.gen / "scorer.py", self.gen / "rules.md",
                                         self.gen / "train_feedback.json", "fixture-model", max_turns=37)
                    if error:
                        with self.assertRaisesRegex(RuntimeError, "subtype=error_max_turns; turns=2; max_turns=37"):
                            asyncio.run(call)
                    else:
                        self.assertEqual(asyncio.run(call)["cost_usd"], .01)
                saved = json.loads((self.gen / "reviser_result.json").read_text())
                self.assertEqual(saved["cost_usd"], .01)
                self.assertEqual(saved["subtype"], "error_max_turns" if error else "success")
                self.assertEqual(saved["max_turns"], 37)
                self.assertEqual(options.call_args.kwargs["max_turns"], 37)
                self.assertEqual(self.events()[-1]["is_error"], error)
                self.assertIn("TRAIN feedback", (self.gen / "reviser_prompt.txt").read_text())

    def test_transport_failure_keeps_already_received_actions(self):
        class Client:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def query(self, prompt):
                pass

            async def receive_response(self):
                yield AssistantMessage(model="test", content=[
                    ToolUseBlock(id="read-1", name="Read", input={"file_path": "scorer.py"})])
                raise ConnectionError("fixture disconnect")

        with patch("claude_agent_sdk.ClaudeSDKClient", Client), \
                patch.object(driver, "build_options", return_value=(object(), {})), \
                patch.object(driver, "bind_session_options", return_value=object()), \
                self.assertRaises(ConnectionError):
            asyncio.run(driver.revise(self.root, self.gen / "scorer.py", self.gen / "rules.md",
                                      self.gen / "train_feedback.json", "fixture-model"))
        self.assertEqual(self.events()[-1]["tool_use_id"], "read-1")

    def test_reviser_cannot_read_validation_in_trajectories_or_snapshots(self):
        guard, _ = make_guard(self.root, self.gen, [self.gen / "train_feedback.json"],
                              self.root / "tool_audit.jsonl")
        for name in ("trajectory.txt", "trajectory.jsonl", "parent_scorer.py", "scorer.py.diff",
                     "report.html", "run_status.json", "candidate_pool_after.json"):
            for tool in ("Read", "Write", "Edit"):
                decision = asyncio.run(guard(tool, {"file_path": str(self.gen / name)}, None))
                self.assertEqual(decision.behavior, "deny")
        for name in ('report.html', 'run_status.json'):
            decision = asyncio.run(guard('Read', {'file_path': str(self.root / name)}, None))
            self.assertEqual(decision.behavior, 'deny')


if __name__ == "__main__":
    unittest.main()

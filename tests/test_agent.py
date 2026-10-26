import subprocess
import threading
import unittest
from unittest.mock import Mock, patch

from app import agent


class AgentTests(unittest.TestCase):
    def lane(self):
        with patch.object(agent, "Sandbox") as sandbox:
            lane = agent.Lane("plain", 0)
        lane.box.listing.return_value = {}
        lane.box.run.return_value = (0, "completed")
        return lane

    def test_cancelled_generation_does_not_commit_or_run_tools(self):
        lane = self.lane()
        cancel = threading.Event()

        def stream(*args):
            yield "text", "Partial reply"
            cancel.set()
            yield "calls", [{"name": "bash", "arguments": {"command": "touch changed"}}]

        model = Mock(agent_stream=stream, tool_args=lambda args: args)
        events = list(lane.turn(model, "Work", [], {}, cancel))

        lane.box.run.assert_not_called()
        self.assertEqual(lane.messages, [{"role": "user", "content": "Work"}])
        self.assertEqual(lane.calls, 0)
        self.assertEqual(events[-1]["type"], "idle")

    def test_cancel_between_tools_skips_remaining_calls_and_completes_history(self):
        lane = self.lane()
        cancel = threading.Event()

        def run(command):
            cancel.set()
            return 0, "first completed"

        lane.box.run.side_effect = run
        calls = [{"name": "bash", "arguments": {"command": command}} for command in ("first", "second")]
        model = Mock(agent_stream=lambda *args: iter([("calls", calls)]), tool_args=lambda args: args)
        events = list(lane.turn(model, "Work", [], {}, cancel))

        lane.box.run.assert_called_once_with("first")
        results = [event for event in events if event["type"] == "result"]
        self.assertEqual([result["ok"] for result in results], [True, False])
        self.assertIn("cancel", results[-1]["output"].lower())
        requested = lane.messages[1]["tool_calls"]
        responses = lane.messages[2:]
        self.assertEqual([call["id"] for call in requested], [msg["tool_call_id"] for msg in responses])

    def test_session_creation_failure_closes_created_lanes(self):
        first = Mock()
        with patch.object(agent, "Lane", side_effect=[first, RuntimeError("second VM failed")]):
            with self.assertRaisesRegex(RuntimeError, "second VM failed"):
                agent.Session(Mock(), {"plain": 0, "steered": 1})
        first.box.close.assert_called_once_with()

    def test_failed_vm_start_removes_partial_container(self):
        with patch.object(agent, "ensure_image"), patch.object(agent, "container") as container:
            container.side_effect = [subprocess.TimeoutExpired("container run", 120), Mock(returncode=0)]
            with self.assertRaises(subprocess.TimeoutExpired):
                agent.Sandbox()
        self.assertEqual(container.call_args_list[-1].args[:2], ("delete", "--force"))
        self.assertEqual(container.call_args_list[-1].args[2], container.call_args_list[0].args[4])

    def test_build_timeout_stops_builder(self):
        def container(*args, **kwargs):
            if args[0] == "build":
                raise subprocess.TimeoutExpired("container build", 900)
            return Mock(returncode=1 if args[:2] == ("image", "inspect") else 0)

        with patch.object(agent, "container", side_effect=container) as call:
            with self.assertRaises(subprocess.TimeoutExpired):
                agent.ensure_image()
        self.assertEqual(call.call_args_list[-1].args, ("builder", "stop"))

    def test_upload_reports_failed_write(self):
        lane = Mock()
        lane.box.write.return_value = (1, "permission denied")
        with patch.object(agent, "Lane", return_value=lane):
            session = agent.Session(Mock(), {"plain": 0})
        with self.assertRaisesRegex(RuntimeError, "permission denied"):
            session.put("config.json", "{}")

    def test_failed_file_write_does_not_report_an_applied_diff(self):
        lane = self.lane()
        lane.box.read.return_value = ("old", None)
        lane.box.write.return_value = (1, "permission denied")
        ok, output, extra = lane.tool("write_file", {"path": "file", "content": "new"})
        self.assertFalse(ok)
        self.assertEqual(output, "permission denied")
        self.assertNotIn("diff", extra)

    def test_session_close_attempts_every_lane_after_one_fails(self):
        first, second = Mock(), Mock()
        first.box.close.side_effect = subprocess.TimeoutExpired("container delete", 60)
        with patch.object(agent, "Lane", side_effect=[first, second]):
            session = agent.Session(Mock(), {"plain": 0, "steered": 1})
        with self.assertRaises(ExceptionGroup):
            session.close()
        second.box.close.assert_called_once_with()

    def test_vm_delete_failure_is_reported(self):
        box = agent.Sandbox.__new__(agent.Sandbox)
        box.name = "test-box"
        for error in ("permission denied", "notFound: runtime socket not found"):
            with self.subTest(error=error), patch.object(agent, "container", return_value=Mock(returncode=1, stderr=error)):
                with self.assertRaisesRegex(RuntimeError, error):
                    box.close()

    def test_vm_delete_is_idempotent_when_container_is_already_gone(self):
        box = agent.Sandbox.__new__(agent.Sandbox)
        box.name = "test-box"
        error = 'Error: internalError: "failed to delete container" (cause: "notFound: "container with ID test-box not found"")'
        with patch.object(agent, "container", return_value=Mock(returncode=1, stderr=error)):
            box.close()

    def test_vm_start_error_is_preserved_when_cleanup_also_fails(self):
        failure = subprocess.TimeoutExpired("container run", 120)
        with patch.object(agent, "ensure_image"), patch.object(agent, "container") as container:
            container.side_effect = [failure, Mock(returncode=1, stderr="delete failed")]
            with self.assertRaises(subprocess.TimeoutExpired) as raised:
                agent.Sandbox()
        self.assertIs(raised.exception, failure)
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)

    def test_session_start_error_is_preserved_when_cleanup_also_fails(self):
        first = Mock()
        first.box.close.side_effect = RuntimeError("delete failed")
        failure = RuntimeError("second VM failed")
        with patch.object(agent, "Lane", side_effect=[first, failure]):
            with self.assertRaisesRegex(RuntimeError, "second VM failed") as raised:
                agent.Session(Mock(), {"plain": 0, "steered": 1})
        self.assertIs(raised.exception, failure)
        self.assertIsInstance(raised.exception.__cause__, ExceptionGroup)


if __name__ == "__main__":
    unittest.main()

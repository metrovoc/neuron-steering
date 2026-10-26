import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from app.models import Flash, Small


class Response:
    def __init__(self, deltas):
        self.lines = [f'data: {json.dumps({"choices": [{"delta": d}]})}\n'.encode() for d in deltas]
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def __iter__(self):
        return iter(self.lines)


class FlashTests(unittest.TestCase):
    def test_calibration_is_only_required_for_steering(self):
        with tempfile.TemporaryDirectory() as root, patch("app.models.ROOT", root):
            model = Flash()
            self.assertIsNone(model.norms)
            np.testing.assert_array_equal(model.vector([]), 0)
            with self.assertRaisesRegex(RuntimeError, "out/rnorm.json"):
                model.vector([{"layer": 1, "neuron": 0, "dose": 1}])
            Path(root, "out").mkdir()
            Path(root, "out", "rnorm.json").write_text('{"1": 2}')
            model.column = Mock(return_value=np.ones(2560, np.float32))
            vector = model.vector([{"layer": 1, "neuron": 0, "dose": 3}])
            self.assertAlmostEqual(float(np.linalg.norm(vector[1])), 6, places=5)

    def test_reset_live_file_restores_same_stack(self):
        model = Flash()
        vector = np.ones((48, 2560), np.float32)
        model.vector = Mock(return_value=vector)
        pushes = [{"layer": 1, "neuron": 0, "dose": 1}]
        with tempfile.TemporaryDirectory() as root:
            model.live = str(Path(root, "live.f32"))
            model.set_vector(pushes, 2)
            model.set_vector(pushes, 2)
            self.assertEqual(model.vector.call_count, 1)
            model.write_live(np.zeros_like(vector))
            model.set_vector(pushes, 2)
            self.assertEqual(model.vector.call_count, 2)
            np.testing.assert_array_equal(np.fromfile(model.live, np.float32), 2)
            self.assertFalse(Path(model.live + ".tmp").exists())

    def test_cancelled_chat_does_not_start_later_columns(self):
        model = Flash()
        model.ensure = Mock()
        model.set_vector = Mock()
        response = Response([{"content": "hello"}])
        model.request = Mock(return_value=response)
        cancel = threading.Event()
        stream = model.chat([[], []], [], [0, 0], {}, cancel)
        self.assertEqual(next(stream), (0, "text", "hello"))
        cancel.set()
        self.assertEqual(list(stream), [])
        self.assertEqual(model.request.call_count, 1)
        self.assertTrue(response.closed)

    def test_cancelled_agent_does_not_return_partial_calls(self):
        model = Flash()
        model.ensure = Mock()
        model.set_vector = Mock()
        response = Response([
            {"tool_calls": [{"index": 0, "function": {"name": "bash", "arguments": '{"command":"true"}'}}]},
            {"content": "running"},
        ])
        model.request = Mock(return_value=response)
        cancel = threading.Event()
        stream = model.agent_stream([], [], [], 0, {}, cancel)
        self.assertEqual(next(stream), ("text", "running"))
        cancel.set()
        self.assertEqual(list(stream), [])
        self.assertTrue(response.closed)

    def test_already_cancelled_requests_do_not_load_models(self):
        cancel = threading.Event()
        cancel.set()
        flash = Flash()
        flash.ensure = Mock(side_effect=AssertionError("should not load"))
        self.assertEqual(list(flash.chat([[]], [], [0], {}, cancel)), [])
        self.assertEqual(list(flash.agent_stream([], [], [], 0, {}, cancel)), [])
        small = Small()
        small.runtime = Mock(side_effect=AssertionError("should not load"))
        small.ensure = Mock(side_effect=AssertionError("should not load"))
        self.assertEqual(list(small.chat([[]], [], [0], {}, cancel)), [])
        self.assertEqual(list(small.agent_stream([], [], [], 0, {}, cancel)), [])
        self.assertEqual(list(small.generate_rows([[1]], [], [0], 5, cancel)), [])


if __name__ == "__main__":
    unittest.main()

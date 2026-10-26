import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from neuron_steering import acts


class CaptureTests(unittest.TestCase):
    def test_empty_capture_has_empty_dense_activations(self):
        with tempfile.TemporaryDirectory() as root:
            capture = acts.Capture(str(Path(root, "absent.bin")))
            self.assertEqual(capture.positions(), [])
            self.assertEqual(capture.dense(0)["shared"].shape, (0, acts.F))

    def test_failed_request_preserves_error_and_removes_partial_capture(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root, "capture.bin")

            def fail(*args, **kwargs):
                path.write_bytes(b"incomplete")
                raise OSError("request failed")

            with patch("neuron_steering.acts.urllib.request.urlopen", side_effect=fail):
                with self.assertRaisesRegex(OSError, "request failed"):
                    acts.capture(8090, str(path), "hello")
            self.assertFalse(path.exists())

    def test_invalid_capture_is_removed_after_parse_failure(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root, "capture.bin")
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            response.read.side_effect = lambda: path.write_bytes(b"incomplete")
            with patch("neuron_steering.acts.urllib.request.urlopen", return_value=response), patch("neuron_steering.acts.time.sleep"):
                with self.assertRaises(ValueError):
                    acts.capture(8090, str(path), "hello")
            self.assertFalse(path.exists())
            response.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()

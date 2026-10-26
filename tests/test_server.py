import asyncio
import importlib.util
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


class ServerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.small, self.flash = Mock(), Mock()
        models = SimpleNamespace(Small=lambda: self.small, Flash=lambda: self.flash,
                                 WORK=self.tmp.name, ncvec=Mock())
        spec = importlib.util.spec_from_file_location(
            "server_under_test", Path(__file__).resolve().parents[1] / "app" / "server.py")
        self.server = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"models": models, "agent": Mock()}):
            spec.loader.exec_module(self.server)

    async def drain(self, response):
        return [event async for event in response.body_iterator]

    async def test_old_response_cannot_cancel_next_request(self):
        first_finished, second_started = threading.Event(), threading.Event()
        release_second = threading.Event()
        tokens = []

        def first(cancel):
            tokens.append(cancel)
            first_finished.set()
            return iter(())

        def second(cancel):
            tokens.append(cancel)
            second_started.set()
            release_second.wait(5)
            return iter(())

        first_response = self.server.stream(first)
        self.assertTrue(first_finished.wait(2))
        self.assertTrue(self.server.busy.acquire(timeout=2))
        self.server.busy.release()
        second_response = self.server.stream(second)
        try:
            self.assertTrue(second_started.wait(2))
            await self.drain(first_response)
            self.assertFalse(tokens[1].is_set())
            self.server.stop()
            self.assertTrue(tokens[1].is_set())
        finally:
            release_second.set()
            await self.drain(second_response)

    async def test_disconnected_client_cancels_its_worker(self):
        stopped = threading.Event()

        def work(cancel):
            yield self.server.sse({"delta": "hello"})
            if cancel.wait(5):
                stopped.set()

        response = self.server.stream(work)
        await anext(response.body_iterator)
        await response.body_iterator.aclose()
        self.assertTrue(stopped.wait(2))
        self.assertTrue(self.server.busy.acquire(timeout=2))
        self.server.busy.release()

    async def test_model_warming_is_exclusive(self):
        started, release = threading.Event(), threading.Event()

        def load():
            started.set()
            release.wait(5)

        self.small.ensure.side_effect = load
        worker = threading.Thread(target=self.server.warm, args=({"model": "small"},))
        worker.start()
        try:
            self.assertTrue(started.wait(2))
            self.assertIn("error", await asyncio.to_thread(self.server.warm, {"model": "flash"}))
            self.assertIn("error", await asyncio.to_thread(self.server.search, "flash", "test"))
            self.small.unload.assert_not_called()
            self.flash.ensure.assert_not_called()
        finally:
            release.set()
            worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertFalse(self.server.busy.locked())

    async def test_failed_warm_reports_error_and_releases_lock(self):
        self.small.ensure.side_effect = OSError("model unavailable")
        self.assertIn("model unavailable", self.server.warm({"model": "small"})["error"])
        self.assertFalse(self.server.busy.locked())

    async def test_export_removes_download_after_response(self):
        self.flash.vector.return_value = np.zeros((48, 2), np.float32)
        self.server.models.ncvec.write_cvec.side_effect = lambda path, *args: Path(path).write_bytes(b"GGUF")
        response = self.server.export({"model": "flash", "pushes": []})
        self.assertEqual(Path(response.path).read_bytes(), b"GGUF")
        await response.background()
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    async def test_failed_export_removes_temporary_directory(self):
        self.flash.vector.return_value = np.zeros((48, 2), np.float32)
        self.server.models.ncvec.write_cvec.side_effect = OSError("write failed")
        self.assertIn("write failed", self.server.export({"model": "flash"})["error"])
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])
        self.assertFalse(self.server.busy.locked())

    async def test_numeric_expert_zero_is_preserved(self):
        rows = self.server.pushes_of({"pushes": [
            {"layer": 14, "neuron": 305, "dose": 1, "expert": 0},
            {"layer": 14, "neuron": 305, "dose": 1, "expert": ""}]})
        self.assertEqual([row["expert"] for row in rows], ["0", None])

    async def test_failed_upload_returns_error_and_releases_lock(self):
        self.server.session = Mock()
        self.server.session.put.side_effect = RuntimeError("disk full")
        result = self.server.agent_put({"files": {"example.txt": "text"}})
        self.assertIn("disk full", result["error"])
        self.assertFalse(self.server.busy.locked())

    async def test_close_session_releases_backend_session(self):
        session = Mock()
        self.server.session = session
        self.assertEqual(self.server.agent_close(), {})
        session.close.assert_called_once()
        self.assertIsNone(self.server.session)


if __name__ == "__main__":
    unittest.main()

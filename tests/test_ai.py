from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request

from vdvr.ai import AiServer


class FakeController:
    def __init__(self) -> None:
        self.ready = True
        self.state = "idle"
        self.error = None
        self.prompt = None
        self.motion = None
        self.stopped = False

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "state": self.state,
            "error": self.error,
            "prompt": self.prompt,
            "motion": self.motion,
        }

    def perform(self, prompt: str, **kwargs) -> dict:
        self.prompt = prompt
        return {
            "ok": True,
            "prompt": prompt,
            "motion": "fake.npz",
            "frames": 1,
            "timings": {"encode_s": 0.1, "generate_s": 0.2},
            "clip_seconds": 4.0,
            "return_seconds": kwargs.get("return_seconds") or 1.2,
        }

    def stop(self) -> None:
        self.stopped = True


class AiServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = FakeController()
        self.server = AiServer(("127.0.0.1", 0), self.controller)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address[:2]
        self.base = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)

    def _get(self, path: str) -> tuple[int, dict]:
        with urllib.request.urlopen(self.base + path, timeout=3.0) as response:
            return response.status, json.load(response)

    def _post(self, path: str, payload: dict) -> tuple[int, dict]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base + path,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=3.0) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc)

    def test_health_and_status(self) -> None:
        status, body = self._get("/health")
        self.assertEqual(status, 200)
        self.assertTrue(body["ready"])
        self.assertEqual(body["state"], "idle")

        status, body = self._get("/status")
        self.assertEqual(status, 200)
        self.assertTrue(body["ready"])

    def test_perform(self) -> None:
        status, body = self._post(
            "/perform",
            {"prompt": "a person waves", "duration": 2, "steps": 6, "return_seconds": 0.5},
        )
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["prompt"], "a person waves")
        self.assertEqual(body["motion"], "fake.npz")
        self.assertEqual(self.controller.prompt, "a person waves")

    def test_invalid_prompt(self) -> None:
        status, body = self._post("/perform", {})
        self.assertEqual(status, 400)
        self.assertFalse(body["ok"])

    def test_stop(self) -> None:
        status, body = self._post("/stop", {})
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertTrue(self.controller.stopped)


if __name__ == "__main__":
    unittest.main()

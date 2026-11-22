from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from vdvr.live import LiveWorker

FAKE_WORKER = r'''
import json
import sys

print(json.dumps({"status": "ready"}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get("quit"):
        break
    print(json.dumps({"motion": "fake.npz", "frames": 1, "timings": {"encode_s": 0.1}}), flush=True)
'''


class LiveWorkerTests(unittest.TestCase):
    def test_line_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "worker.py"
            script.write_text(FAKE_WORKER, encoding="utf-8")
            worker = LiveWorker(Path(sys.executable), script, Path(tmp))
            try:
                worker.start()
                worker.wait_ready()
                response = worker.request("a test prompt", 4.0, 0, 10)
                self.assertEqual(response["motion"], "fake.npz")
                self.assertEqual(response["frames"], 1)
            finally:
                worker.close()


if __name__ == "__main__":
    unittest.main()

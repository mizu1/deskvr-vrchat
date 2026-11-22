"""Local HTTP performance API for AI-driven VRChat performances.

The AI calls ``POST /perform`` with a natural-language prompt.  The controller
generates one ARDY clip and plays it **once** through GlassVR + VRChat OSC,
then blends back to the rest pose and returns the result.  It does not loop.

Start it with::

    python -m vdvr.live --api --api-port 7878

See ``docs/AI_INTERFACE.md`` for the full protocol.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .ardy import ArdyPlayer, ArdySource
from .live import LiveWorker


class NotReadyError(RuntimeError):
    """The ARDY model/encoder is still loading."""


class BusyError(RuntimeError):
    """Another performance is running and ``wait`` was false."""


class PerformanceController:
    """Generate + play one clip at a time (one-shot, no loop)."""

    def __init__(
        self,
        cfg: dict,
        height: float,
        profile: str,
        worker: LiveWorker,
        *,
        dry_run: bool = False,
        fade_in: float = 0.4,
        default_return_seconds: float = 1.2,
    ) -> None:
        self.cfg = cfg
        self.height = float(height)
        self.profile = profile
        self.worker = worker
        self.dry_run = bool(dry_run)
        self.fade_in = float(fade_in)
        self.default_return_seconds = float(default_return_seconds)

        self.ready = False
        self.state = "loading"
        self.error: str | None = None
        self.prompt: str | None = None
        self.motion: str | None = None
        self.current: ArdyPlayer | None = None
        self._lock = threading.Lock()

    def initialize(self) -> None:
        self.state = "loading"
        print("[ai] starting ARDY worker (encoder + model)...", flush=True)
        try:
            self.worker.start()
            self.worker.wait_ready()
            self.ready = True
            self.state = "idle"
            print("[ai] ready", flush=True)
        except Exception as exc:  # noqa: BLE001 - surfaced through /status and logs
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = "error"
            print(f"[ai] worker failed: {self.error}", flush=True)

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "state": self.state,
            "error": self.error,
            "prompt": self.prompt,
            "motion": self.motion,
        }

    def stop(self) -> None:
        player = self.current
        if player is not None:
            player.request_stop()

    def perform(
        self,
        prompt: str,
        *,
        duration: float = 4.0,
        steps: int = 10,
        seed: int = 0,
        return_seconds: float | None = None,
        wait: bool = True,
    ) -> dict:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        if not self.ready:
            raise NotReadyError(self.error or "ARDY model is still loading")

        acquired = self._lock.acquire(blocking=wait)
        if not acquired:
            raise BusyError("another performance is running")
        try:
            if not self.ready:
                raise NotReadyError(self.error or "ARDY model is still loading")

            duration = float(min(max(float(duration), 1.0), 10.0))
            steps = int(min(max(int(steps), 1), 50))
            return_seconds = (
                self.default_return_seconds if return_seconds is None else float(return_seconds)
            )
            return_seconds = float(min(max(return_seconds, 0.0), 10.0))

            self.prompt = prompt
            self.state = "generating"
            response = self.worker.request(prompt, duration, seed, steps)
            if "error" in response:
                raise RuntimeError(response["error"])

            motion_path = Path(response["motion"])
            if not motion_path.is_file():
                raise RuntimeError(f"generated motion not found: {motion_path}")
            self.motion = str(motion_path)

            source = ArdySource(motion_path, height=self.height, profile=self.profile)
            player = ArdyPlayer(
                source,
                self.cfg,
                loop=False,
                fade_in=self.fade_in,
                return_seconds=return_seconds,
                dry_run=self.dry_run,
                verbose=False,
            )
            self.current = player
            self.state = "playing"
            try:
                player.run()  # one clip, then auto-return to rest and exit
            finally:
                self.current = None

            self.state = "idle"
            return {
                "ok": True,
                "prompt": prompt,
                "motion": str(motion_path),
                "frames": response.get("frames"),
                "encoder": response.get("encoder"),
                "timings": response.get("timings"),
                "clip_seconds": round(float(source.duration), 3),
                "return_seconds": return_seconds,
            }
        finally:
            if self.state != "error":
                self.state = "idle" if self.ready else self.state
            self._lock.release()

    def close(self) -> None:
        self.stop()
        self.worker.close()


class _Handler(BaseHTTPRequestHandler):
    server_version = "vdvr-ai/1.0"
    server: "AiServer"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[ai] {self.address_string()} {fmt % args}", flush=True)

    def _send(self, code: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:
        controller = self.server.controller
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send(200, {"ok": True, "ready": controller.ready, "state": controller.state})
        elif path == "/status":
            self._send(200, {"ok": True, **controller.status()})
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        controller = self.server.controller
        path = self.path.split("?", 1)[0]
        if path == "/stop":
            controller.stop()
            self._send(200, {"ok": True, "state": controller.state})
            return
        if path != "/perform":
            self._send(404, {"ok": False, "error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            request = json.loads(raw)
        except (ValueError, OSError):
            self._send(400, {"ok": False, "error": "invalid JSON body"})
            return

        prompt = str(request.get("prompt", "")).strip()
        if not prompt:
            self._send(400, {"ok": False, "error": "prompt is required"})
            return
        try:
            duration = float(request.get("duration", 4.0))
            steps = int(request.get("steps", 10))
            seed = int(request.get("seed", 0))
            return_seconds = request.get("return_seconds")
            wait = bool(request.get("wait", True))
        except (TypeError, ValueError):
            self._send(400, {"ok": False, "error": "invalid duration/steps/seed/wait"})
            return

        try:
            result = controller.perform(
                prompt,
                duration=duration,
                steps=steps,
                seed=seed,
                return_seconds=return_seconds,
                wait=wait,
            )
            self._send(200, result)
        except NotReadyError as exc:
            self._send(503, {"ok": False, "error": str(exc), "state": controller.state})
        except BusyError as exc:
            self._send(409, {"ok": False, "error": str(exc), "state": controller.state})
        except ValueError as exc:
            self._send(400, {"ok": False, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - report the failure to the caller
            self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


class AiServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], controller: PerformanceController) -> None:
        super().__init__(address, _Handler)
        self.controller = controller


def serve(controller: PerformanceController, host: str, port: int) -> None:
    server = AiServer((host, port), controller)
    bound_host, bound_port = server.server_address[:2]
    print(f"[ai] HTTP API listening on http://{bound_host}:{bound_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ai] interrupted", flush=True)
    finally:
        server.server_close()

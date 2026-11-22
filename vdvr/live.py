"""Natural-language live ARDY control for vdvr.

Keeps an ARDY text->motion worker loaded, asks for a prompt in the console,
generates a clip, and plays it through GlassVR (head/hands) + VRChat OSC
(body).  The current clip loops while you type the next prompt.

The worker must run in the ARDY python environment (torch + the ARDY model).
The live controller itself only needs the normal vdvr environment (numpy).

Example::

    python -m vdvr.live --height 1.70 --profile standard
    prompt> a person waves with their right hand
    prompt> a person walks forward and stops
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import config as config_module
from .ardy import ArdyPlayer, ArdySource

DEFAULT_INFER_REPO = Path(
    os.environ.get("ARDY_INFER_REPO", r"E:\360Downloads\SeedData\repos\ardy-0.5b-infer")
)
DEFAULT_WORKER_PYTHON = Path(
    os.environ.get(
        "ARDY_VENV_PY",
        r"D:\ProgramProject\ARDY\.venv\Scripts\python.exe",
    )
)


class LiveWorker:
    """Line-delimited JSON client for vdvr/prompt_worker.py."""

    def __init__(
        self,
        python: Path,
        script: Path,
        infer_repo: Path,
        *,
        model: str | None = None,
        checkpoints_dir: str | None = None,
        no_encoder: bool = False,
    ) -> None:
        self.python = Path(python)
        self.script = Path(script)
        self.infer_repo = Path(infer_repo)
        self.model = model
        self.checkpoints_dir = checkpoints_dir
        self.no_encoder = no_encoder
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        command = [
            str(self.python),
            "-u",
            str(self.script),
            "--infer-repo",
            str(self.infer_repo),
        ]
        if self.model:
            command += ["--model", self.model]
        if self.checkpoints_dir:
            command += ["--checkpoints-dir", self.checkpoints_dir]
        if self.no_encoder:
            command.append("--no-encoder")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

    def _read(self) -> dict | None:
        if self.proc is None or self.proc.stdout is None:
            raise RuntimeError("ARDY worker is not running")
        line = self.proc.stdout.readline()
        if line == "":
            code = self.proc.poll()
            raise RuntimeError(f"ARDY worker exited (code {code})")
        line = line.strip()
        if not line:
            return None
        start = line.find("{")
        if start >= 0:
            try:
                return json.loads(line[start:])
            except json.JSONDecodeError:
                pass
        print(f"[ardy] {line}", flush=True)
        return None

    def wait_ready(self) -> None:
        print("[live] loading ARDY encoder/model (first start can take ~20 s)", flush=True)
        while True:
            message = self._read()
            if message is None:
                continue
            if "status" in message:
                print(f"[ardy] {message['status']}", flush=True)
                if message["status"] == "ready":
                    return
                continue
            if "error" in message:
                raise RuntimeError(message["error"])

    def request(self, text: str, duration: float, seed: int, steps: int) -> dict:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("ARDY worker is not running")
        payload = {"text": text, "duration": duration, "seed": seed, "steps": steps}
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        while True:
            message = self._read()
            if message is None:
                continue
            if "status" in message:
                print(f"[ardy] {message['status']}", flush=True)
                continue
            if "error" in message or "motion" in message:
                return message
            print(f"[ardy] {message}", flush=True)

    def close(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.write('{"quit": true}\n')
                proc.stdin.flush()
        except (OSError, ValueError):
            pass
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


class LivePlayer:
    """One ArdyPlayer at a time; switching stops the old one first."""

    def __init__(
        self,
        cfg: dict,
        height: float,
        profile: str,
        *,
        fade_in: float = 0.4,
        return_seconds: float = 1.2,
        dry_run: bool = False,
    ) -> None:
        self.cfg = cfg
        self.dry_run = bool(dry_run)
        self.height = height
        self.profile = profile
        self.fade_in = fade_in
        self.return_seconds = return_seconds
        self.player: ArdyPlayer | None = None
        self.thread: threading.Thread | None = None

    def play(self, source: ArdySource, *, calibrate: float = 0.0) -> None:
        self.stop()
        player = ArdyPlayer(
            source,
            self.cfg,
            loop=True,
            fade_in=self.fade_in,
            return_seconds=self.return_seconds,
            dry_run=self.dry_run,
            verbose=False,
        )
        if calibrate > 0.0:
            print("[live] enter VRChat Calibrate FBT now (T-pose + both triggers)", flush=True)
            player.calibrate(calibrate)
        self.player = player
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            self.player.run()  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 - keep the live prompt alive
            print(f"[player] {type(exc).__name__}: {exc}", flush=True)

    def stop(self) -> None:
        player = self.player
        thread = self.thread
        self.player = None
        self.thread = None
        if player is None:
            return
        player.request_stop()
        if thread is not None:
            thread.join(timeout=10.0)
            if thread.is_alive():
                print("[live] warning: previous player did not stop in time", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vdvr.live",
        description="Generate ARDY motions from natural language and play them in VRChat.",
    )
    parser.add_argument("--config", type=str, default=None, help="vdvr config.json")
    parser.add_argument("--infer-repo", type=Path, default=DEFAULT_INFER_REPO)
    parser.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON)
    parser.add_argument(
        "--worker-script",
        type=Path,
        default=Path(__file__).with_name("prompt_worker.py"),
    )
    parser.add_argument("--height", type=float, default=None, help="your height in metres")
    parser.add_argument("--profile", choices=("minimal", "standard", "full"), default=None)
    parser.add_argument("--duration", type=float, default=4.0, help="generated clip length (seconds)")
    parser.add_argument("--steps", type=int, default=10, help="diffusion steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibrate", type=float, default=0.0, help="T-pose + triggers before the first clip")
    parser.add_argument("--fade-in", type=float, default=0.4)
    parser.add_argument("--return-seconds", type=float, default=1.2)
    parser.add_argument("--model", default=None)
    parser.add_argument("--checkpoints-dir", default=None)
    parser.add_argument("--no-encoder", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="generate and retarget without sending OSC/GlassVR")
    parser.add_argument("--prompt", default=None, help="first prompt (otherwise ask on stdin)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_module.load(args.config)
    if args.height is not None:
        cfg["body"]["height"] = args.height
    if args.profile is not None:
        cfg["body"]["profile"] = args.profile
    height = float(cfg["body"]["height"])
    profile = str(cfg["body"]["profile"])

    if not args.infer_repo.is_dir():
        print(f"[live] ARDY infer repo not found: {args.infer_repo}", file=sys.stderr)
        return 2
    if not args.worker_python.is_file():
        print(f"[live] ARDY python not found: {args.worker_python}", file=sys.stderr)
        return 2
    if not args.worker_script.is_file():
        print(f"[live] worker script not found: {args.worker_script}", file=sys.stderr)
        return 2

    if not args.dry_run:
        print(
            "[live] stop run.py/start.bat first: the live controller writes the same "
            "GlassVR UDP ports",
            flush=True,
        )
    worker = LiveWorker(
        args.worker_python,
        args.worker_script,
        args.infer_repo,
        model=args.model,
        checkpoints_dir=args.checkpoints_dir,
        no_encoder=args.no_encoder,
    )
    try:
        worker.start()
        worker.wait_ready()
    except Exception as exc:  # noqa: BLE001 - startup errors are user-facing
        print(f"[live] worker failed: {exc}", file=sys.stderr)
        worker.close()
        return 2

    live = LivePlayer(
        cfg,
        height,
        profile,
        fade_in=args.fade_in,
        return_seconds=args.return_seconds,
        dry_run=args.dry_run,
    )
    first = True
    try:
        while True:
            text = args.prompt if args.prompt is not None else input("prompt> ")
            args.prompt = None
            text = (text or "").strip()
            if not text:
                continue
            if text.lower() in ("quit", "exit"):
                break

            print("[live] generating...", flush=True)
            try:
                response = worker.request(text, args.duration, args.seed, args.steps)
            except Exception as exc:  # noqa: BLE001
                print(f"[live] generation failed: {exc}", file=sys.stderr)
                continue
            if "error" in response:
                print(f"[live] generation failed: {response['error']}", file=sys.stderr)
                continue

            motion_path = Path(response["motion"])
            if not motion_path.is_file():
                print(f"[live] motion file missing: {motion_path}", file=sys.stderr)
                continue
            try:
                source = ArdySource(motion_path, height=height, profile=profile)
            except Exception as exc:  # noqa: BLE001
                print(f"[live] retarget failed: {exc}", file=sys.stderr)
                continue

            print(
                f"[live] playing '{text}' ({response.get('frames')} frames, "
                f"{response.get('timings')})",
                flush=True,
            )
            live.play(source, calibrate=args.calibrate if first else 0.0)
            first = False
            print("[live] type the next prompt (quit / Ctrl+C to stop)", flush=True)
    except KeyboardInterrupt:
        print("\n[live] interrupted", flush=True)
    finally:
        live.stop()
        worker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

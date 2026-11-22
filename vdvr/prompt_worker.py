"""ARDY text -> motion worker. Run this with the ARDY python environment.

The vdvr live controller starts it once and keeps it running:

    {"text":"a person waves","duration":4.0,"seed":0,"steps":10}
    -> {"motion":"...npz","frames":120,"timings":{...}}

It loads the ARDY model once, starts the student encoder service if it is not
already running, and writes one SOMA77 motion.npz per request.  No GIF render.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


def _emit(message: dict) -> None:
    print(json.dumps(message, ensure_ascii=False), flush=True)


def _log(message: str) -> None:
    print(f"[ardy-worker] {message}", file=sys.stderr, flush=True)


def _frame_count(arrays: dict) -> int:
    joints = np.asarray(arrays["posed_joints"])
    if joints.ndim == 4:
        return int(joints.shape[1])
    if joints.ndim == 3:
        return int(joints.shape[0])
    return 0


def _healthy(encoder_url: str) -> bool:
    try:
        with urllib.request.urlopen(encoder_url.rstrip("/") + "/health", timeout=1.5) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def _ensure_encoder(repo: Path, config) -> subprocess.Popen | None:
    """Start the student encoder service when it is not already running."""
    if _healthy(config.ENCODER_URL):
        _log(f"using existing encoder at {config.ENCODER_URL}")
        return None

    parsed = urlparse(config.ENCODER_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 7871
    if host not in ("127.0.0.1", "localhost"):
        _log(f"encoder {config.ENCODER_URL} is remote; not starting it locally")
        return None

    command = [
        str(config.LLM2VEC_PY),
        str(repo / "encoder_server.py"),
        "--student-dir",
        str(config.STUDENT_DIR),
        "--host",
        host,
        "--port",
        str(port),
    ]
    env = dict(os.environ, PYTHONPATH=str(config.LLM2VEC_PYTHONPATH), HF_HUB_OFFLINE="1")
    _emit({"status": "starting_encoder"})
    process = subprocess.Popen(
        command,
        cwd=str(repo),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        if _healthy(config.ENCODER_URL):
            _log(f"encoder ready at {config.ENCODER_URL}")
            return process
        if process.poll() is not None:
            _log("encoder service exited during startup")
            return None
        time.sleep(0.5)
    _log("encoder service startup timed out")
    try:
        process.terminate()
    except OSError:
        pass
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARDY text-to-motion worker")
    parser.add_argument(
        "--infer-repo",
        default=os.environ.get(
            "ARDY_INFER_REPO",
            r"E:\360Downloads\SeedData\repos\ardy-0.5b-infer",
        ),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--checkpoints-dir", default=None)
    parser.add_argument("--no-encoder", action="store_true")
    args = parser.parse_args(argv)

    repo = Path(args.infer_repo).resolve()
    if not (repo / "generate_motion.py").is_file():
        _emit({"error": f"ARDY infer repo not found: {repo}"})
        return 2

    # Running this file directly puts the *vdvr* directory on sys.path.  A module
    # called ardy.py lives there and would shadow the real ARDY package.
    script_dir = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if not p or Path(p).resolve() != script_dir]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    import config  # ARDY infer repo config

    ardy_repo = Path(config.ARDY_REPO)
    if ardy_repo.is_dir() and str(ardy_repo) not in sys.path:
        sys.path.insert(0, str(ardy_repo))

    import pipeline
    from generate_motion import ArdyStudentGenerator

    encoder_process = None
    try:
        if not args.no_encoder:
            encoder_process = _ensure_encoder(repo, config)
        _emit({"status": "loading_model"})
        checkpoints = Path(args.checkpoints_dir) if args.checkpoints_dir else None
        generator = ArdyStudentGenerator(args.model, checkpoints)
        out_dir = config.WORK_DIR / "live"
        out_dir.mkdir(parents=True, exist_ok=True)
        _emit({"status": "ready"})

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                _emit({"error": "invalid JSON request"})
                continue
            if request.get("quit"):
                break

            text = str(request.get("text", "")).strip() or "a person walks forward"
            duration = float(np.clip(float(request.get("duration", 4.0)), 1.0, 10.0))
            seed = int(request.get("seed", 0))
            steps = int(np.clip(int(request.get("steps", 10)), 1, 50))

            try:
                started = time.perf_counter()
                stamp = f"{time.time_ns()}_{abs(hash((text, duration, seed, steps))) % 10**6}"
                vector_path = out_dir / f"vectors_{stamp}.npz"
                payload, encoder_source = pipeline.encode_prompts([text], vector_path)
                encode_s = time.perf_counter() - started

                started = time.perf_counter()
                generator.set_vectors([text], [payload["vectors"][0]])
                arrays = generator.generate(text, duration, seed, steps)
                generate_s = time.perf_counter() - started

                motion_path = out_dir / f"motion_{stamp}.npz"
                np.savez(motion_path, **arrays)
                _emit(
                    {
                        "motion": str(motion_path),
                        "frames": _frame_count(arrays),
                        "duration": duration,
                        "text": text,
                        "encoder": encoder_source,
                        "timings": {
                            "encode_s": round(encode_s, 3),
                            "generate_s": round(generate_s, 3),
                        },
                    }
                )
            except Exception as exc:  # noqa: BLE001 - report to the live controller
                _log(f"generation failed: {type(exc).__name__}: {exc}")
                _emit({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        if encoder_process is not None:
            try:
                encoder_process.terminate()
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

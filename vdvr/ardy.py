"""Play an ARDY SOMA77 motion through GlassVR and VRChat OSC trackers.

The normal vdvr app drives GlassVR's virtual HMD/controllers from the mouse.
This module swaps that input source for a pre-recorded ARDY clip:

    ARDY Head / LeftHand / RightHand      -> GlassVR UDP (9100 / 9102 / 9101)
    ARDY hip/chest/feet/knees/elbows      -> VRChat OSC trackers (9000)

VRChat's six-point IK (head + two hands + hip + two feet) then animates the
whole upper body, while the OSC trackers drive the lower body.  This avoids the
8-point FBT / FBIK limitations entirely and does not modify VRChat.

Usage::

    # stop run.py (the mouse simulator) first: both write the same GlassVR ports
    python -m vdvr.ardy E:\\path\\motion.npz --height 1.70 --profile standard

    # optional: hold a T-pose + both triggers for VRChat's Calibrate FBT
    python -m vdvr.ardy motion.npz --calibrate 5

The ARDY retargeter lives in a separate checkout; set ``ARDY_VRC_REPO`` or pass
``--ardy-repo``.  It needs ``numpy``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import config as config_module
from .body import PROFILES, sanitize_tracker
from .glassvr import GlassVrClient, IndexInput
from .math3d import q_from_euler_deg, qmul
from .vrchat import VrchatOsc

DEFAULT_ARDY_REPO = Path(
    os.environ.get("ARDY_VRC_REPO", str(Path.home() / "ardy-action-to-vrchat"))
)

# The OSC tracker slot numbers used by vdvr/body.py (also the order VRChat's
# tracker consumers expect in this setup).
NAME_TO_SLOT = {
    "hip": 1,
    "chest": 2,
    "lfoot": 3,
    "rfoot": 4,
    "lknee": 5,
    "rknee": 6,
    "lelbow": 7,
    "relbow": 8,
}
SLOT_TO_NAME = {slot: name for name, slot in NAME_TO_SLOT.items()}

_ARDY: SimpleNamespace | None = None


def _load_ardy(repo: str | Path | None = None) -> SimpleNamespace:
    """Import ardy_vrc lazily so the rest of vdvr does not depend on numpy."""
    global _ARDY
    if _ARDY is not None:
        return _ARDY

    root = Path(repo) if repo else DEFAULT_ARDY_REPO
    if not (root / "ardy_vrc").is_dir():
        raise FileNotFoundError(
            f"ardy_vrc not found under {root}. Pass --ardy-repo or set ARDY_VRC_REPO."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from ardy_vrc import mathutil
    from ardy_vrc.retarget import Retargeter, RetargetOptions
    from ardy_vrc.skeleton import JOINT, load_motion

    _ARDY = SimpleNamespace(
        root=root,
        mathutil=mathutil,
        Retargeter=Retargeter,
        RetargetOptions=RetargetOptions,
        JOINT=JOINT,
        load_motion=load_motion,
        euler=mathutil.euler_zxy_from_matrix,
        matrix_from_quat=mathutil.matrix_from_quat,
        quat_from_matrix=mathutil.quat_from_matrix,
        mirror_x_rotation=mathutil.mirror_x_rotation,
        slerp=mathutil.quat_slerp,
        lerp=mathutil.lerp,
        smoothstep=mathutil.smoothstep,
    )
    return _ARDY


@dataclass
class ArdyFrame:
    head_pos: np.ndarray
    head_quat: np.ndarray
    left_pos: np.ndarray
    left_quat: np.ndarray
    right_pos: np.ndarray
    right_quat: np.ndarray
    body_pos: dict[str, np.ndarray]
    body_quat: dict[str, np.ndarray]


class ArdySource:
    """Pre-sampled ARDY motion: head/hands + selected body tracker slots."""

    def __init__(
        self,
        motion_path: str | Path,
        height: float = 1.70,
        profile: str = "standard",
        ardy_repo: str | Path | None = None,
    ) -> None:
        ardy = _load_ardy(ardy_repo)
        if profile not in PROFILES:
            raise ValueError(f"unknown tracker profile: {profile!r}")
        self._ardy = ardy
        self.profile = profile
        self.height = float(height)
        self.motion = ardy.load_motion(motion_path)
        self.body_slots = tuple(SLOT_TO_NAME[n] for n in PROFILES[profile])
        self.slot_numbers = tuple(PROFILES[profile])

        options = ardy.RetargetOptions(
            height=self.height,
            slots=self.body_slots,
            send_head=False,
            ground_lock=True,
        )
        self.retargeter = ardy.Retargeter(self.motion, options)
        self.sequence = self.retargeter.sequence
        self.fps = float(self.motion.fps)
        self.frames = int(self.sequence.frames)
        self.duration = (self.frames - 1) / self.fps if self.frames > 1 else 0.0
        self._source_frames = int(self.motion.frames)

        joints = np.asarray(self.retargeter.joint_positions, dtype=np.float64)
        j = ardy.JOINT
        self._head_pos = 0.5 * (joints[:, j["LeftEye"]] + joints[:, j["RightEye"]])
        self._left_pos = joints[:, j["LeftHand"]]
        self._right_pos = joints[:, j["RightHand"]]

        rotations = self.motion.rotations
        self._head_quat = np.stack(
            [
                ardy.quat_from_matrix(ardy.mirror_x_rotation(rotations[t, j["Head"]]))
                for t in range(self._source_frames)
            ]
        )
        left_quat = np.stack(
            [
                ardy.quat_from_matrix(ardy.mirror_x_rotation(rotations[t, j["LeftHand"]]))
                for t in range(self._source_frames)
            ]
        )
        right_quat = np.stack(
            [
                ardy.quat_from_matrix(ardy.mirror_x_rotation(rotations[t, j["RightHand"]]))
                for t in range(self._source_frames)
            ]
        )
        # SOMA hand bones carry a large roll; remove the rest-pose roll so the
        # GlassVR controller points like a controller, not like a finger bone.
        self._left_quat = self._remove_rest_roll(left_quat)
        self._right_quat = self._remove_rest_roll(right_quat)

        self._body_pos = np.asarray(self.sequence.positions, dtype=np.float64)
        self._body_quat = np.asarray(self.sequence.quats, dtype=np.float64)

    # --------------------------------------------------------------- helpers
    def _remove_rest_roll(self, quats: np.ndarray) -> np.ndarray:
        if len(quats) == 0:
            return quats
        euler = self._ardy.euler(self._ardy.matrix_from_quat(quats[0]))
        offset = q_from_euler_deg(0.0, 0.0, -float(euler[2]))
        return np.stack([qmul(q, offset) for q in quats])

    @property
    def rest(self) -> ArdyFrame:
        return self.frame_at(0.0)

    def euler(self, quat: np.ndarray) -> np.ndarray:
        return self._ardy.euler(self._ardy.matrix_from_quat(quat))

    def frame_at(self, t: float) -> ArdyFrame:
        if self.frames <= 1 or self.duration <= 0.0:
            idx = 0
            f = 0.0
            idx2 = 0
        else:
            x = float(np.clip(t, 0.0, self.duration)) * self.fps
            idx = min(int(np.floor(x)), self.frames - 1)
            idx2 = min(idx + 1, self.frames - 1)
            f = x - idx

        body_pos = {
            name: self._ardy.lerp(self._body_pos[idx, i], self._body_pos[idx2, i], f)
            for i, name in enumerate(self.body_slots)
        }
        body_quat = {
            name: self._ardy.slerp(self._body_quat[idx, i], self._body_quat[idx2, i], f)
            for i, name in enumerate(self.body_slots)
        }
        return ArdyFrame(
            head_pos=self._ardy.lerp(self._head_pos[idx], self._head_pos[idx2], f),
            head_quat=self._ardy.slerp(self._head_quat[idx], self._head_quat[idx2], f),
            left_pos=self._ardy.lerp(self._left_pos[idx], self._left_pos[idx2], f),
            left_quat=self._ardy.slerp(self._left_quat[idx], self._left_quat[idx2], f),
            right_pos=self._ardy.lerp(self._right_pos[idx], self._right_pos[idx2], f),
            right_quat=self._ardy.slerp(self._right_quat[idx], self._right_quat[idx2], f),
            body_pos=body_pos,
            body_quat=body_quat,
        )

    def tpose_frame(self) -> ArdyFrame:
        """A static T-pose for VRChat's Calibrate FBT."""
        rest = self.rest
        tpose = self.retargeter.tpose_frame()
        eye_y = float(rest.head_pos[1])
        span = 0.44 * self.height
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        body_pos = {name: np.asarray(tpose.positions[name], dtype=np.float64) for name in self.body_slots}
        body_quat = {name: np.asarray(tpose.quats[name], dtype=np.float64) for name in self.body_slots}
        return ArdyFrame(
            head_pos=np.array([0.0, eye_y, 0.0], dtype=np.float64),
            head_quat=identity.copy(),
            left_pos=np.array([-span, eye_y - 0.07, 0.0], dtype=np.float64),
            left_quat=identity.copy(),
            right_pos=np.array([span, eye_y - 0.07, 0.0], dtype=np.float64),
            right_quat=identity.copy(),
            body_pos=body_pos,
            body_quat=body_quat,
        )


def _blend(source: ArdySource, a: ArdyFrame, b: ArdyFrame, u: float) -> ArdyFrame:
    return ArdyFrame(
        head_pos=source._ardy.lerp(a.head_pos, b.head_pos, u),
        head_quat=source._ardy.slerp(a.head_quat, b.head_quat, u),
        left_pos=source._ardy.lerp(a.left_pos, b.left_pos, u),
        left_quat=source._ardy.slerp(a.left_quat, b.left_quat, u),
        right_pos=source._ardy.lerp(a.right_pos, b.right_pos, u),
        right_quat=source._ardy.slerp(a.right_quat, b.right_quat, u),
        body_pos={k: source._ardy.lerp(a.body_pos[k], b.body_pos[k], u) for k in a.body_pos},
        body_quat={k: source._ardy.slerp(a.body_quat[k], b.body_quat[k], u) for k in a.body_quat},
    )


class ArdyTimeline:
    def __init__(
        self,
        source: ArdySource,
        loop: bool = False,
        fade_in: float = 0.5,
        return_seconds: float = 2.5,
    ) -> None:
        self.source = source
        self.loop = bool(loop)
        self.fade_in = max(0.0, float(fade_in))
        self.return_seconds = max(0.0, float(return_seconds))
        self.duration = source.duration
        self.rest = source.frame_at(0.0)
        self._return_time: float | None = None
        self._return_frame: ArdyFrame | None = None

    def phase(self, t: float) -> str:
        if t < 0.0:
            return "delay"
        if self._return_time is not None:
            if t < self._return_time:
                return "motion"
            return "return" if t < self._return_time + self.return_seconds else "hold"
        if not self.loop and t >= self.duration:
            return "return" if t < self.duration + self.return_seconds else "hold"
        if self.fade_in > 0.0 and t < self.fade_in:
            return "fade"
        return "motion"

    def sample(self, t: float) -> tuple[ArdyFrame, str]:
        p = self.phase(t)
        if p in ("delay", "hold"):
            return self.rest, p
        if p == "return":
            rt = self._return_time if self._return_time is not None else self.duration
            u = 1.0 if self.return_seconds <= 0.0 else (t - rt) / self.return_seconds
            start = self._return_frame if self._return_frame is not None else self.source.frame_at(self.duration)
            return _blend(self.source, start, self.rest, self.source._ardy.smoothstep(u)), p

        mt = t % self.duration if self.loop and self.duration > 0.0 else min(max(t, 0.0), self.duration)
        frame = self.source.frame_at(mt)
        if p == "fade":
            u = self.source._ardy.smoothstep(t / self.fade_in) if self.fade_in > 0.0 else 1.0
            return _blend(self.source, self.rest, frame, u), p
        return frame, p

    def begin_return(self, t: float) -> None:
        if self._return_time is not None:
            return
        if t < self.duration or not self.loop:
            self._return_frame = self.sample(t)[0]
        else:
            self._return_frame = self.source.frame_at(self.duration)
        self._return_time = float(t)


class ArdyPlayer:
    def __init__(
        self,
        source: ArdySource,
        cfg: dict,
        *,
        loop: bool = False,
        fade_in: float = 0.5,
        return_seconds: float = 2.5,
        rate: float = 90.0,
        head_anchor: bool = True,
        dry_run: bool = False,
        verbose: bool = True,
    ) -> None:
        self.source = source
        self.timeline = ArdyTimeline(source, loop=loop, fade_in=fade_in, return_seconds=return_seconds)
        self.rate = max(30.0, float(rate))
        self.dry_run = bool(dry_run)
        self.head_anchor = bool(head_anchor)
        self.verbose = bool(verbose)
        self.frames = 0
        self.tracker_frames = 0

        glass_cfg = cfg["glassvr"]
        osc_cfg = cfg["vrchat"]
        self.glass = GlassVrClient(
            host=str(glass_cfg.get("host", "127.0.0.1")),
            hmd_port=int(glass_cfg.get("hmd_port", 9100)),
            left_port=int(glass_cfg.get("left_port", 9102)),
            right_port=int(glass_cfg.get("right_port", 9101)),
            enabled=not self.dry_run and bool(glass_cfg.get("enabled", True)),
            protocol=str(glass_cfg.get("protocol", "both")),
        )
        self.osc = VrchatOsc(
            host=str(osc_cfg.get("host", "127.0.0.1")),
            port=int(osc_cfg.get("port", 9000)),
            enabled=not self.dry_run and bool(osc_cfg.get("enabled", True)),
        )
        self._last_status = time.perf_counter()

    # ------------------------------------------------------------------ send
    @staticmethod
    def _state(trigger: bool) -> IndexInput:
        return IndexInput(trigger=1.0 if trigger else 0.0)

    def _send(self, frame: ArdyFrame, *, trigger: bool = False) -> None:
        self.glass.hmd(tuple(frame.head_pos), tuple(frame.head_quat))
        self.glass.left_hand(tuple(frame.left_pos), tuple(frame.left_quat), self._state(trigger))
        self.glass.right_hand(tuple(frame.right_pos), tuple(frame.right_quat), self._state(trigger))

        for name, slot in NAME_TO_SLOT.items():
            if name not in frame.body_pos:
                continue
            euler = self.source.euler(frame.body_quat[name])
            position, euler = sanitize_tracker((tuple(frame.body_pos[name]), tuple(euler)))
            self.osc.tracker(slot, position, euler)
        self.tracker_frames += 1

        if self.head_anchor:
            euler = self.source.euler(frame.head_quat)
            self.osc.head(tuple(frame.head_pos), tuple(euler))
        self.frames += 1

    def calibrate(self, seconds: float) -> None:
        """Hold a T-pose and both triggers so VRChat can Calibrate FBT."""
        if seconds <= 0:
            return
        self._log(f"T-pose calibration for {seconds:.1f}s: open VRChat Calibrate FBT now")
        frame = self.source.tpose_frame()
        end = time.perf_counter() + float(seconds)
        period = 1.0 / self.rate
        while time.perf_counter() < end:
            self._send(frame, trigger=True)
            time.sleep(period)
        self._log("calibration pose finished")

    # ------------------------------------------------------------------- run
    def run(self, duration: float | None = None) -> int:
        period = 1.0 / self.rate
        start = time.perf_counter()
        next_tick = start
        hold_start: float | None = None
        if not self.dry_run:
            self._log(
                "make sure run.py (mouse simulator) is stopped; "
                "it writes the same GlassVR UDP ports"
            )
        self._log(
            f"ARDY playback: {self.source.frames} frames @ {self.source.fps:.0f} fps, "
            f"profile={self.source.profile}, slots={self.source.body_slots}, "
            f"height={self.source.height:.2f}m"
        )
        try:
            while True:
                try:
                    now = time.perf_counter()
                    elapsed = now - start
                    frame, phase = self.timeline.sample(elapsed)
                    self._send(frame)

                    if phase == "hold":
                        if hold_start is None:
                            hold_start = now
                        if now - hold_start >= 0.5:
                            break
                    if duration is not None and elapsed >= duration:
                        break

                    self._status(now, phase)
                    next_tick += period
                    sleep_for = next_tick - time.perf_counter()
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    elif sleep_for < -3.0 * period:
                        next_tick = time.perf_counter()
                except KeyboardInterrupt:
                    if self.timeline._return_time is None:
                        self.timeline.begin_return(time.perf_counter() - start)
                        self._log("interrupted: returning to the rest pose")
                    else:
                        break
        finally:
            self.osc.close()
            self.glass.close()
        self._log(
            f"stopped: {self.frames} frames, {self.tracker_frames} tracker ticks, "
            f"{self.osc.tracker_packets} OSC tracker packets, {self.glass.sent_packets} GlassVR packets"
        )
        return 0

    # --------------------------------------------------------------- logging
    def _status(self, now: float, phase: str) -> None:
        if not self.verbose or now - self._last_status < 2.0:
            return
        self._last_status = now
        self._log(
            f"phase={phase:<6} frames={self.frames} "
            f"osc={self.osc.tracker_packets} glass={self.glass.sent_packets}"
        )

    @staticmethod
    def _log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vdvr.ardy",
        description="Play an ARDY SOMA77 motion through GlassVR (head/hands) and VRChat OSC trackers (body).",
    )
    parser.add_argument("motion", type=Path, help="ARDY motion.npz")
    parser.add_argument("--config", type=str, default=None, help="vdvr config.json")
    parser.add_argument("--ardy-repo", type=str, default=None, help="checkout containing the ardy_vrc package")
    parser.add_argument("--height", type=float, default=None, help="your height in metres")
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None, help="OSC tracker count")
    parser.add_argument("--loop", action="store_true", help="loop until Ctrl+C")
    parser.add_argument("--fade-in", type=float, default=0.5, help="blend from the rest pose (seconds)")
    parser.add_argument("--return-seconds", type=float, default=2.5, help="blend back to rest (seconds)")
    parser.add_argument("--rate", type=float, default=90.0, help="pose update rate (Hz)")
    parser.add_argument("--no-head-anchor", action="store_true", help="do not send the OSC head anchor")
    parser.add_argument("--calibrate", type=float, default=0.0, help="hold T-pose + triggers for N seconds first")
    parser.add_argument("--duration", type=float, default=None, help="exit after N seconds (tests)")
    parser.add_argument("--dry-run", action="store_true", help="compute poses but send nothing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_module.load(args.config)
    if args.height is not None:
        cfg["body"]["height"] = args.height
    if args.profile is not None:
        cfg["body"]["profile"] = args.profile

    source = ArdySource(
        args.motion,
        height=float(cfg["body"]["height"]),
        profile=str(cfg["body"]["profile"]),
        ardy_repo=args.ardy_repo,
    )
    player = ArdyPlayer(
        source,
        cfg,
        loop=args.loop,
        fade_in=args.fade_in,
        return_seconds=args.return_seconds,
        rate=args.rate,
        head_anchor=not args.no_head_anchor,
        dry_run=args.dry_run,
    )
    player.calibrate(args.calibrate)
    return player.run(duration=args.duration)


if __name__ == "__main__":
    raise SystemExit(main())

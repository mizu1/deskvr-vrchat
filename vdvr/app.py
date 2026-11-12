"""Main application loop: desktop input -> virtual VR hardware + fake FBT."""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import sys
import time
from pathlib import Path

from . import config as config_module
from .body import BodyInput, BodyModel, sanitize_tracker
from .glassvr import GlassVrClient, IndexInput
from .math3d import (
    clamp,
    q_from_euler_deg,
    q_from_yaw,
    qrotate,
    right_from_yaw,
    v,
    vadd,
    wrap180,
)
from .vrchat import VrchatOsc

TRACKER_INTERVAL = 1.0 / 90.0
IDLE_INTERVAL = 1.0 / 120.0


def _scaled(vec: tuple[float, float, float], height: float) -> tuple[float, float, float]:
    scale = height / 1.70
    return (vec[0] * scale, vec[1] * scale, vec[2] * scale)


class App:
    def __init__(self, cfg: dict, *, dry_run: bool = False, demo: bool = False) -> None:
        self.cfg = cfg
        self.dry_run = dry_run
        self.demo = demo

        from .input_win import WinInput, focus_window  # lazy import: no user32 in tests

        self._focus_window = focus_window

        capture_cfg = cfg["capture"]
        self.input = WinInput(capture_cfg.get("processes", ["VRChat.exe"]))
        processes = capture_cfg.get("processes", ["VRChat.exe"])
        if isinstance(processes, str):
            processes = [part.strip() for part in processes.split(",")]
        self.capture_names = ", ".join(processes)

        self.sensitivity = float(capture_cfg.get("mouse_sensitivity", 0.045))
        self.invert_y = bool(capture_cfg.get("invert_y", False))
        self.recenter_mouse = bool(capture_cfg.get("recenter_mouse", True))

        glass_cfg = cfg["glassvr"]
        self.glass = GlassVrClient(
            host=glass_cfg.get("host", "127.0.0.1"),
            hmd_port=int(glass_cfg.get("hmd_port", 9100)),
            left_port=int(glass_cfg.get("left_port", 9102)),
            right_port=int(glass_cfg.get("right_port", 9101)),
            enabled=bool(glass_cfg.get("enabled", True)) and not dry_run,
            protocol=glass_cfg.get("protocol", "both"),
        )

        osc_cfg = cfg["vrchat"]
        self.osc = VrchatOsc(
            host=osc_cfg.get("host", "127.0.0.1"),
            port=int(osc_cfg.get("port", 9000)),
            enabled=bool(osc_cfg.get("enabled", True)) and not dry_run,
        )

        body_cfg = cfg["body"]
        self.height = float(body_cfg.get("height", 1.70))
        self.body = BodyModel(height=self.height, profile=body_cfg.get("profile", "standard"))

        self.movement_via = cfg["input"].get("movement_via", "controller").lower()
        self.menu_via = cfg["input"].get("menu_via", "controller").lower()
        self.jump_via = cfg["input"].get("jump_via", "both").lower()

        self.yaw = 0.0
        self.pitch = 0.0
        self.body_yaw = 0.0
        self.pointer_yaw = 0.0
        self.pointer_pitch = 0.0
        self.pointer_sensitivity = float(cfg["input"].get("pointer_sensitivity", 0.08))
        self.pointer_key = str(cfg["input"].get("pointer_key", "ALT")).upper()
        self.crouch = 0.0
        self.crouch_toggle = False
        self.capture = bool(cfg["start"].get("enabled", True))
        self.tpose_until = 0.0
        self.menu_pulse_until = 0.0
        self.system_pulse_until = 0.0
        self._menu_edge = False
        self.running_flag = True
        self.deadline: float | None = None
        self._warned_inactive = False
        self._inactive_since: float | None = None
        self._skip_next_mouse = False

        self._was_active = False
        self._tracker_accum = 0.0
        self._last_status = time.perf_counter()
        self._frames = 0
        self._loops = 0
        self._loop_rate = 0.0

    # ------------------------------------------------------------------ helpers
    def _movement(self, active: bool) -> tuple[float, float, bool]:
        inp = self.input
        forward = 0.0
        strafe = 0.0
        if active:
            forward = float(inp.down("W")) - float(inp.down("S"))
            strafe = float(inp.down("D")) - float(inp.down("A"))
        length = math.hypot(forward, strafe)
        if length > 1.0:
            forward /= length
            strafe /= length
        running = bool(active and inp.down("SHIFT"))
        return forward, strafe, running

    def _hands(self, head_pos, yaw: float, tposing: bool, now: float, pointer_yaw: float = 0.0, pointer_pitch: float = 0.0):
        """Return (left_pos, right_pos, left_rot, right_rot).

        The hands live in *body* space, not head space.  Standing still, the
        body yaw stays put, so turning the head makes the hand ray sweep across
        the view (this is what a real controller does).  The right hand also has
        a mouse-driven "pointer" offset (hold Alt), which aims the laser at UI
        without moving the head.
        """
        yaw_q = q_from_yaw(yaw)
        right = right_from_yaw(yaw)
        if tposing:
            span = 0.44 * self.height
            shoulder_y = head_pos[1] - 0.07
            left = v(head_pos[0] - right[0] * span, shoulder_y, head_pos[2] - right[2] * span)
            right_pos = v(head_pos[0] + right[0] * span, shoulder_y, head_pos[2] + right[2] * span)
            return left, right_pos, yaw_q, yaw_q

        sway_x = 0.004 * math.sin(now * 1.1)
        sway_y = 0.005 * math.sin(now * 1.7)
        base = _scaled((0.16, -0.22, 0.30), self.height)
        left = vadd(head_pos, qrotate(yaw_q, (-base[0] + sway_x, base[1] + sway_y, base[2])))

        # Shift the right hand along with the pointer so drivers that derive the
        # aim from the forearm/hand position also see the ray move.
        pointer_x = base[0] + 0.18 * math.sin(math.radians(pointer_yaw))
        pointer_y = base[1] - 0.12 * math.sin(math.radians(pointer_pitch))
        right_pos = vadd(head_pos, qrotate(yaw_q, (pointer_x - sway_x, pointer_y + sway_y, base[2])))

        left_rot = q_from_euler_deg(-5.0, yaw, 0.0)
        right_rot = q_from_euler_deg(-5.0 + pointer_pitch, yaw + pointer_yaw, 0.0)
        return left, right_pos, left_rot, right_rot

    def _hotkeys(self, now: float, active: bool) -> None:
        inp = self.input
        if inp.just_pressed("F8"):
            self.capture = not self.capture
            self._log(f"capture {'ENABLED' if self.capture else 'DISABLED'}")
        if inp.just_pressed("F6"):
            found = self._focus_window()
            if found:
                self._log(f"focused window: {found}")
                self._warned_inactive = False
            else:
                self._log("could not find 'Headset Window' / 'VRChat' - click the mirror window manually")
        if inp.just_pressed("F9"):
            self.yaw = 0.0
            self.pitch = 0.0
            self.body_yaw = 0.0
            self.pointer_yaw = 0.0
            self.pointer_pitch = 0.0
            self.body.phase = 0.0
            inp.reset_cursor_center()
            self._log("view + body + pointer recentered")
        if inp.just_pressed("F7"):
            self.tpose_until = now + 3.5
            self._log("FBT calibration pose (T-pose + both triggers) for 3.5 s")
        if inp.just_pressed("F10"):
            self.capture = False
            self.osc.release_all(now)
            self._log("EMERGENCY RELEASE - capture off (F8 to resume)")
        if not active:
            return
        if inp.just_pressed("B") or inp.just_pressed("ESC"):
            self.menu_pulse_until = now + 0.10
            self._menu_edge = True
        if inp.just_pressed("C"):
            self.crouch_toggle = not self.crouch_toggle
        if inp.just_pressed("M"):
            self.osc.tap("Voice")
        if inp.just_pressed("TAB"):
            self.system_pulse_until = now + 0.12
        if inp.down("CTRL") and inp.down("ALT") and inp.just_pressed("Q"):
            self.running_flag = False

    # ------------------------------------------------------------------ main loop
    def run(self) -> int:
        timer_period = False
        if os.name == "nt":
            try:
                ctypes.windll.winmm.timeBeginPeriod(1)
                timer_period = True
            except Exception:  # noqa: BLE001 - optional latency tweak
                pass

        countdown = int(self.cfg["start"].get("countdown", 0))
        if countdown > 0 and self.capture and not self.demo and not self.dry_run:
            self._log(f"focus {self.capture_names} now; capturing in {countdown}s (F8 toggles)")
            for remaining in range(countdown, 0, -1):
                print(f"  {remaining}...", flush=True)
                time.sleep(1.0)

        self._log(
            f"height={self.height:.2f}m profile={self.body.profile} "
            f"glassvr={'off' if self.dry_run else 'on'} osc={'off' if self.dry_run else 'on'}"
        )
        if self.demo:
            self._log("DEMO mode: scripted motion, no real input")

        if self.cfg["start"].get("focus_game", True) and not self.dry_run and not self.demo:
            for _ in range(15):
                found = self._focus_window()
                if found:
                    time.sleep(0.3)
                    if self.input.foreground_matches():
                        self._log(f"input is active on window: {found}")
                    else:
                        self._log(f"found {found}, but Windows blocked focus - press F6 or click it")
                    break
                time.sleep(1.0)

        last = time.perf_counter()
        start = last
        try:
            while self.running_flag:
                now = time.perf_counter()
                dt = clamp(now - last, 0.0, 0.05)
                last = now
                self._loops += 1

                self.input.snapshot()

                foreground = self.input.foreground_matches(now)
                active = self.capture and foreground and not self.demo
                self._hotkeys(now, active)

                if active and not self._was_active:
                    self.input.reset_cursor_center()
                    self._skip_next_mouse = True
                self._was_active = active

                if active:
                    self._inactive_since = None
                    self._warned_inactive = False
                elif self.capture and not self.demo:
                    if self._inactive_since is None:
                        self._inactive_since = now
                    elif not self._warned_inactive and now - self._inactive_since > 3.0:
                        self._log(
                            f"input paused: foreground={self.input.foreground_name or '-'}; "
                            "press F6 or click the Headset Window"
                        )
                        self._warned_inactive = True

                if active:
                    dx, dy = self.input.mouse_look(recenter=self.recenter_mouse)
                    if self._skip_next_mouse:
                        self._skip_next_mouse = False
                        dx = dy = 0.0
                    if self.input.down(self.pointer_key):
                        # Pointer mode: the mouse aims the right hand laser
                        # instead of turning the head.
                        self.pointer_yaw = clamp(
                            self.pointer_yaw + dx * self.pointer_sensitivity, -75.0, 75.0
                        )
                        self.pointer_pitch = clamp(
                            self.pointer_pitch + dy * self.pointer_sensitivity, -45.0, 45.0
                        )
                    else:
                        self.yaw = wrap180(self.yaw + dx * self.sensitivity)
                        sign = -1.0 if self.invert_y else 1.0
                        self.pitch = clamp(self.pitch + dy * self.sensitivity * sign, -80.0, 80.0)

                tposing = now < self.tpose_until

                # Crouch is a smooth blend; Ctrl is hold-to-crouch, C toggles.
                if tposing:
                    target_crouch = 0.0
                else:
                    target_crouch = 1.0 if (active and (self.input.down("CTRL") or self.crouch_toggle)) else 0.0
                self.crouch += (target_crouch - self.crouch) * clamp(dt * 12.0, 0.0, 1.0)

                forward, strafe, running = self._movement(active)
                jumping = bool(active and self.input.down("SPACE"))
                jump_edge = bool(active and self.input.just_pressed("SPACE"))
                if self.jump_via in ("osc", "both") and jump_edge:
                    self.osc.tap("Jump")
                if self.demo:
                    forward, strafe = self._demo_motion(now - start)
                    running = False
                    jumping = False

                if self.deadline is not None and now >= self.deadline:
                    self.running_flag = False

                # Standing: the body (and hands) stay put so turning the head
                # sweeps the hand ray.  Moving: the body follows the head.
                body_delta = wrap180(self.yaw - self.body_yaw)
                if (abs(forward) + abs(strafe) > 0.1) or abs(body_delta) > 120.0:
                    self.body_yaw = wrap180(
                        self.body_yaw + body_delta * clamp(dt * 3.0, 0.0, 1.0)
                    )

                # ---------------- head / camera ----------------
                eye_y = self.body.eye_height - 0.44 * self.crouch
                eye_y += 0.006 * math.sin(now * 1.8)
                head_pos = (0.0, eye_y, 0.0)
                head_quat = q_from_euler_deg(self.pitch, self.yaw, 0.0)

                left_wrist, right_wrist, left_hand_quat, right_hand_quat = self._hands(
                    head_pos, self.body_yaw, tposing, now, self.pointer_yaw, self.pointer_pitch
                )

                # ---------------- virtual controllers ----------------
                menu_down = active and now < self.menu_pulse_until
                turn = 0.0
                if active:
                    turn = float(self.input.down("RIGHT")) - float(self.input.down("LEFT"))

                right_state = IndexInput(
                    a=jumping and self.jump_via in ("controller", "both"),
                    b=menu_down if self.menu_via in ("controller", "both") else False,
                    system=now < self.system_pulse_until,
                    joystick_x=turn,
                    trigger=1.0
                    if (active and (self.input.down("LMB") or self.input.down("E") or tposing))
                    else 0.0,
                    grip_pull=1.0 if (active and (self.input.down("RMB") or self.input.down("F"))) else 0.0,
                )
                left_state = IndexInput(
                    joystick_x=strafe if self.movement_via in ("controller", "both") else 0.0,
                    joystick_y=forward if self.movement_via in ("controller", "both") else 0.0,
                    trigger=1.0 if (active and (self.input.down("Q") or tposing)) else 0.0,
                    grip_pull=1.0 if (active and self.input.down("R")) else 0.0,
                )

                self.glass.hmd(head_pos, head_quat)
                self.glass.left_hand(left_wrist, left_hand_quat, left_state)
                self.glass.right_hand(right_wrist, right_hand_quat, right_state)

                # ---------------- OSC inputs ----------------
                if self.movement_via in ("osc", "both"):
                    self.osc.axis("Vertical", forward, now)
                    self.osc.axis("Horizontal", strafe, now)
                self.osc.button("Run", running and active, now)
                if self._menu_edge and self.menu_via in ("osc", "both"):
                    self.osc.tap("QuickMenuToggleRight")
                self._menu_edge = False

                # ---------------- fake FBT ----------------
                body_state = BodyInput(
                    head_pos=head_pos,
                    head_pitch=self.pitch,
                    head_yaw=self.body_yaw,
                    left_wrist=left_wrist,
                    right_wrist=right_wrist,
                    move_forward=forward,
                    move_strafe=strafe,
                    running=running,
                    crouch=self.crouch,
                    jumping=jumping,
                )
                trackers = self.body.update(dt, body_state)

                self._tracker_accum += dt
                if self._tracker_accum >= TRACKER_INTERVAL:
                    # Carry the remainder so the average rate matches 90 Hz even
                    # when the main loop runs at a different frequency.
                    self._tracker_accum = min(self._tracker_accum - TRACKER_INTERVAL, TRACKER_INTERVAL)
                    for slot in sorted(trackers):
                        position, euler = sanitize_tracker(trackers[slot])
                        self.osc.tracker(slot, position, euler)
                    self.osc.head(head_pos, (self.pitch, self.yaw, 0.0))
                    self._frames += 1

                self._status(now, active, forward, strafe, trackers)

                # Keep CPU usage low but latency tight.
                sleep_for = IDLE_INTERVAL - (time.perf_counter() - now)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except KeyboardInterrupt:
            self._log("interrupted")
        finally:
            self.osc.release_all()
            self.osc.close()
            self.glass.close()
            if timer_period:
                try:
                    ctypes.windll.winmm.timeEndPeriod(1)
                except Exception:  # noqa: BLE001
                    pass
        return 0

    # ------------------------------------------------------------------ misc
    def _demo_motion(self, elapsed: float) -> tuple[float, float]:
        phase = elapsed % 12.0
        if phase < 4.0:
            return (0.8, 0.0)
        if phase < 6.0:
            return (0.0, 0.6)
        if phase < 8.0:
            return (0.0, 0.0)
        return (-0.6, 0.0)

    def _status(self, now: float, active: bool, forward: float, strafe: float, trackers: dict) -> None:
        if now - self._last_status < 2.0:
            return
        elapsed = now - self._last_status
        self._loop_rate = self._loops / elapsed if elapsed > 0 else 0.0
        self._loops = 0
        self._last_status = now
        self._log(
            f"fg={self.input.foreground_name or '-':<14} "
            f"capture={'ON ' if self.capture else 'OFF'} "
            f"active={'Y' if active else 'N'} loop={self._loop_rate:5.0f}Hz "
            f"yaw={self.yaw:7.1f} pitch={self.pitch:6.1f} crouch={self.crouch:4.2f} "
            f"move=({forward:+.1f},{strafe:+.1f}) ptr=({self.pointer_yaw:+.0f},{self.pointer_pitch:+.0f}) "
            f"trackers={len(trackers)} "
            f"pkt={self.osc.tracker_packets} glass={self.glass.sent_packets}"
        )

    @staticmethod
    def _log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vdvr",
        description="Run VRChat in VR mode using only keyboard/mouse, with synthetic full-body tracking.",
    )
    parser.add_argument("--config", type=str, default=None, help="path to config.json")
    parser.add_argument("--height", type=float, default=None, help="your real height in metres")
    parser.add_argument(
        "--profile",
        choices=("minimal", "standard", "full"),
        default=None,
        help="how many OSC trackers to send",
    )
    parser.add_argument("--sensitivity", type=float, default=None, help="mouse degrees per pixel")
    parser.add_argument("--dry-run", action="store_true", help="compute poses but send nothing")
    parser.add_argument("--demo", action="store_true", help="scripted motion, ignore real input")
    parser.add_argument("--duration", type=float, default=None, help="exit after N seconds")
    parser.add_argument("--no-countdown", action="store_true", help="skip the start countdown")
    parser.add_argument("--make-config", action="store_true", help="write a config.json and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.make_config:
        target = Path(args.config or "config.json")
        config_module.save_default(target)
        print(f"wrote {target.resolve()}")
        return 0

    if os.name != "nt":
        print("This tool currently requires Windows (user32 input + SteamVR).", file=sys.stderr)
        return 2

    cfg = config_module.load(args.config)
    if args.height is not None:
        cfg["body"]["height"] = args.height
    if args.profile is not None:
        cfg["body"]["profile"] = args.profile
    if args.sensitivity is not None:
        cfg["capture"]["mouse_sensitivity"] = args.sensitivity
    if args.no_countdown:
        cfg["start"]["countdown"] = 0

    app = App(cfg, dry_run=args.dry_run, demo=args.demo)
    if args.duration is not None:
        app.deadline = time.perf_counter() + float(args.duration)
    return app.run()

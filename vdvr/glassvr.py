"""Client for the GlassVR OpenVR driver's UDP interface.

GlassVR (https://github.com/DaniXmir/GlassVr) is an open source SteamVR driver
that can emulate an HMD, controllers and trackers from data supplied over UDP or
named pipes.  We use it as the "hardware", and drive it from this process.

Wire format (little endian, no padding), one UDP port per device.  GlassVR
changed the tag layout between v2.3 and v2.4:

* v2.3: one byte tag ``P`` / ``R`` / ``I`` / ``S`` / ``E`` + packed struct
* v2.4+: three byte tag ``pos`` / ``rot`` / ``idx`` / ``skl`` / ``ext``

``protocol="both"`` makes us work with either driver version (the unknown tag is
ignored, and the old/new length checks reject the wrong layout).
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

from .math3d import Quat, Vec3, qnorm

# v2.4+ layout (3 byte ASCII tag)
_POS = struct.Struct("<3s3d")
_ROT = struct.Struct("<3s4d")
_IDX = struct.Struct("<3s12?8d")
_EXT = struct.Struct("<3s?")

# v2.3 layout (single byte tag)
_POS_V23 = struct.Struct("<c3d")
_ROT_V23 = struct.Struct("<c4d")
_IDX_V23 = struct.Struct("<c12?8d")
_EXT_V23 = struct.Struct("<c?")

_POS_BODY = struct.Struct("<3d")
_ROT_BODY = struct.Struct("<4d")
_IDX_BODY = struct.Struct("<12?8d")


def to_openvr_position(position: Vec3) -> Vec3:
    """Unity (+Z forward, left handed) -> OpenVR (+Y up, -Z forward, right handed)."""
    return (float(position[0]), float(position[1]), -float(position[2]))


def to_openvr_rotation(rotation: Quat) -> Quat:
    """Reflect a Unity quaternion into OpenVR space: (w, x, y, z) -> (w, -x, -y, z)."""
    w, x, y, z = qnorm(rotation)
    return (w, -x, -y, z)


@dataclass
class IndexInput:
    """Valve Index controller state (what VRChat understands best)."""

    a: bool = False
    b: bool = False
    system: bool = False
    joystick_click: bool = False
    trigger_click: bool = False
    a_touch: bool = False
    b_touch: bool = False
    system_touch: bool = False
    joystick_touch: bool = False
    trigger_touch: bool = False
    touchpad_touch: bool = False
    grip_touch: bool = False
    joystick_x: float = 0.0
    joystick_y: float = 0.0
    touchpad_x: float = 0.0
    touchpad_y: float = 0.0
    trigger: float = 0.0
    touchpad_force: float = 0.0
    grip_pull: float = 0.0
    grip_force: float = 0.0

    def _values(self) -> list:
        trigger = max(0.0, min(1.0, self.trigger))
        grip = max(0.0, min(1.0, self.grip_pull))
        return [
            self.a,
            self.b,
            self.system,
            self.joystick_click,
            self.trigger_click or trigger >= 0.5,
            self.a_touch or self.a,
            self.b_touch or self.b,
            self.system_touch or self.system,
            self.joystick_touch or self.joystick_click,
            self.trigger_touch or trigger >= 0.5,
            self.touchpad_touch,
            self.grip_touch or grip >= 0.5,
            max(-1.0, min(1.0, self.joystick_x)),
            max(-1.0, min(1.0, self.joystick_y)),
            self.touchpad_x,
            self.touchpad_y,
            trigger,
            self.touchpad_force,
            grip,
            grip,
        ]

    def packet(self) -> bytes:
        """v2.4+ packet: b'idx' + struct."""
        return _IDX.pack(b"idx", *self._values())

    def packet_v23(self) -> bytes:
        """v2.3 packet: b'I' + struct."""
        return _IDX_V23.pack(b"I", *self._values())


class GlassVrClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        hmd_port: int = 9100,
        left_port: int = 9102,
        right_port: int = 9101,
        enabled: bool = True,
        protocol: str = "both",
    ) -> None:
        self.address = (host, 0)
        self.hmd_port = int(hmd_port)
        self.left_port = int(left_port)
        self.right_port = int(right_port)
        self.enabled = enabled
        self.protocol = protocol.lower()
        if self.protocol not in ("v23", "v24", "both"):
            raise ValueError("glassvr.protocol must be v23, v24 or both")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self.sent_packets = 0

    # ------------------------------------------------------------------ basics
    def _send(self, port: int, payload: bytes) -> None:
        if not self.enabled:
            return
        try:
            self._sock.sendto(payload, (self.address[0], port))
            self.sent_packets += 1
        except OSError:
            # A missing driver must never take the app down.
            pass

    def pose(self, port: int, position: Vec3, rotation: Quat) -> None:
        # GlassVR feeds these straight into vr::DriverPose_t, which is OpenVR
        # space, so convert from the Unity space used everywhere else in vdvr.
        ox, oy, oz = to_openvr_position(position)
        w, x, y, z = to_openvr_rotation(rotation)
        if self.protocol in ("v23", "both"):
            # GlassVR <= v2.3: 1 byte tag, then the packed doubles.
            self._send(port, _POS_V23.pack(b"P", ox, oy, oz))
            self._send(port, _ROT_V23.pack(b"R", w, x, y, z))
        if self.protocol in ("v24", "both"):
            # GlassVR >= v2.4: 3 byte ASCII tag.
            self._send(port, _POS.pack(b"pos", ox, oy, oz))
            self._send(port, _ROT.pack(b"rot", w, x, y, z))

    def _input(self, port: int, state: IndexInput) -> None:
        if self.protocol in ("v23", "both"):
            self._send(port, state.packet_v23())
        if self.protocol in ("v24", "both"):
            self._send(port, state.packet())

    def hmd(self, position: Vec3, rotation: Quat) -> None:
        self.pose(self.hmd_port, position, rotation)

    def left_hand(self, position: Vec3, rotation: Quat, state: IndexInput) -> None:
        self.pose(self.left_port, position, rotation)
        self._input(self.left_port, state)

    def right_hand(self, position: Vec3, rotation: Quat, state: IndexInput) -> None:
        self.pose(self.right_port, position, rotation)
        self._input(self.right_port, state)

    def reset(self, port: int | None = None) -> None:
        ports = [self.hmd_port, self.left_port, self.right_port] if port is None else [port]
        for item in ports:
            if self.protocol in ("v23", "both"):
                self._send(item, _EXT_V23.pack(b"E", True))
            if self.protocol in ("v24", "both"):
                self._send(item, _EXT.pack(b"ext", True))

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

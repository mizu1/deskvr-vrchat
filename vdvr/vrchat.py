"""VRChat OSC output (trackers, head reference, /input/* actions)."""

from __future__ import annotations

import socket
import time

from .osc import encode

AXIS_KEEPALIVE = 0.20
BUTTON_KEEPALIVE = 0.55


class VrchatOsc:
    def __init__(self, host: str = "127.0.0.1", port: int = 9000, enabled: bool = True) -> None:
        self.address = (host, int(port))
        self.enabled = enabled
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)

        self._axes: dict[str, float] = {}
        self._axis_sent: dict[str, float] = {}
        self._buttons: dict[str, bool] = {}
        self._button_sent: dict[str, float] = {}

        self.tracker_packets = 0

    # ------------------------------------------------------------------ low level
    def send(self, address: str, *args: object) -> None:
        if not self.enabled:
            return
        try:
            self._sock.sendto(encode(address, *args), self.address)
        except OSError:
            pass

    # ------------------------------------------------------------------ tracking
    def tracker(self, slot: int, position: tuple[float, float, float], euler: tuple[float, float, float]) -> None:
        self.send(
            f"/tracking/trackers/{slot}/position",
            float(position[0]),
            float(position[1]),
            float(position[2]),
        )
        self.send(
            f"/tracking/trackers/{slot}/rotation",
            float(euler[0]),
            float(euler[1]),
            float(euler[2]),
        )
        self.tracker_packets += 1

    def head(self, position: tuple[float, float, float], euler: tuple[float, float, float]) -> None:
        """Optional head reference: aligns OSC space with VRChat's head bone."""
        self.send(
            "/tracking/trackers/head/position",
            float(position[0]),
            float(position[1]),
            float(position[2]),
        )
        self.send(
            "/tracking/trackers/head/rotation",
            float(euler[0]),
            float(euler[1]),
            float(euler[2]),
        )

    # ------------------------------------------------------------------ actions
    def axis(self, name: str, value: float, now: float | None = None, force: bool = False) -> None:
        now = time.perf_counter() if now is None else now
        value = max(-1.0, min(1.0, float(value)))
        if value == 0.0:
            value = 0.0  # avoid -0.0
        previous = self._axes.get(name)
        last = self._axis_sent.get(name, -1e9)
        changed = previous is None or abs(previous - value) > 0.02
        if force or changed or (now - last) >= AXIS_KEEPALIVE:
            self.send(f"/input/{name}", value)
            self._axes[name] = value
            self._axis_sent[name] = now

    def button(self, name: str, down: bool, now: float | None = None, force: bool = False) -> None:
        now = time.perf_counter() if now is None else now
        down = bool(down)
        previous = self._buttons.get(name)
        last = self._button_sent.get(name, -1e9)
        if previous is None or previous != down or (down and now - last >= BUTTON_KEEPALIVE) or force:
            self.send(f"/input/{name}", 1 if down else 0)
            self._buttons[name] = down
            self._button_sent[name] = now

    def tap(self, name: str) -> None:
        """Press + release, for actions VRChat treats as an edge trigger."""
        self.send(f"/input/{name}", 1)
        self.send(f"/input/{name}", 0)
        self._buttons[name] = False
        self._button_sent[name] = time.perf_counter()

    def release_all(self, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        for name in list(self._axes):
            self.axis(name, 0.0, now, force=True)
        for name in list(self._buttons):
            self.button(name, False, now, force=True)

    def close(self) -> None:
        self.release_all()
        try:
            self._sock.close()
        except OSError:
            pass

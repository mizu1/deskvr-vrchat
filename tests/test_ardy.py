from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from vdvr import ardy
from vdvr.body import PROFILES

MOTION = Path(
    os.environ.get(
        "ARDY_MOTION",
        r"E:\360Downloads\SeedData\repos\ardy-0.5b-infer\outputs\pipeline\motion_44299933.npz",
    )
)
REPO = Path(
    os.environ.get(
        "ARDY_VRC_REPO",
        r"E:\360Downloads\SeedData\repos\ardy-action-to-vrchat",
    )
)


def _ardy_available() -> bool:
    return MOTION.exists() and (REPO / "ardy_vrc").is_dir()


@unittest.skipUnless(_ardy_available(), "ARDY motion/checkout not available")
class ArdySourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.source = ardy.ArdySource(MOTION, height=1.70, profile="standard", ardy_repo=REPO)
        except FileNotFoundError as exc:  # pragma: no cover - env dependent
            raise unittest.SkipTest(str(exc))

    def test_slots_match_body_profiles(self) -> None:
        self.assertEqual(self.source.body_slots, ("hip", "chest", "lfoot", "rfoot"))
        numbers = tuple(ardy.NAME_TO_SLOT[name] for name in self.source.body_slots)
        self.assertEqual(numbers, PROFILES["standard"])

    def test_frame_is_plausible(self) -> None:
        frame = self.source.frame_at(0.0)
        self.assertTrue(np.isfinite(frame.head_pos).all())
        self.assertTrue(np.isfinite(frame.head_quat).all())
        self.assertGreater(float(frame.head_pos[1]), 1.2)
        self.assertLess(float(frame.left_pos[0]), 0.0)
        self.assertGreater(float(frame.right_pos[0]), 0.0)
        self.assertEqual(set(frame.body_pos), set(self.source.body_slots))
        for value in frame.body_pos.values():
            self.assertTrue(np.isfinite(value).all())

    def test_tpose_has_horizontal_hands(self) -> None:
        frame = self.source.tpose_frame()
        self.assertAlmostEqual(float(frame.left_pos[1]), float(frame.right_pos[1]), places=6)
        self.assertLess(float(frame.left_pos[0]), -0.5)
        self.assertGreater(float(frame.right_pos[0]), 0.5)
        self.assertEqual(set(frame.body_pos), set(self.source.body_slots))

    def test_timeline_fade_return_hold(self) -> None:
        timeline = ardy.ArdyTimeline(self.source, loop=False, fade_in=0.1, return_seconds=0.5)
        self.assertEqual(timeline.phase(0.0), "fade")
        self.assertEqual(timeline.phase(self.source.duration + 0.1), "return")
        self.assertEqual(timeline.phase(self.source.duration + 0.6), "hold")
        frame, phase = timeline.sample(self.source.duration + 0.6)
        self.assertEqual(phase, "hold")
        np.testing.assert_allclose(frame.head_pos, timeline.rest.head_pos)

    def test_timeline_loop_and_interrupt(self) -> None:
        timeline = ardy.ArdyTimeline(self.source, loop=True, fade_in=0.0, return_seconds=0.5)
        self.assertEqual(timeline.phase(self.source.duration * 2 + 0.1), "motion")
        timeline.begin_return(0.5)
        self.assertEqual(timeline.phase(0.5), "return")

    def test_player_dry_run(self) -> None:
        cfg = {"glassvr": {"enabled": False}, "vrchat": {"enabled": False}}
        player = ardy.ArdyPlayer(self.source, cfg, dry_run=True, rate=120.0, verbose=False)
        player.run(duration=0.25)
        self.assertGreater(player.frames, 0)
        self.assertGreater(player.tracker_frames, 0)

    def test_player_sends_expected_slots(self) -> None:
        import socket

        from vdvr.osc import decode

        sockets = []

        def bind() -> socket.socket:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("127.0.0.1", 0))
            sock.settimeout(0.2)
            sockets.append(sock)
            return sock

        hmd = bind()
        left = bind()
        right = bind()
        osc = bind()
        cfg = {
            "glassvr": {
                "host": "127.0.0.1",
                "hmd_port": hmd.getsockname()[1],
                "left_port": left.getsockname()[1],
                "right_port": right.getsockname()[1],
                "protocol": "both",
                "enabled": True,
            },
            "vrchat": {
                "host": "127.0.0.1",
                "port": osc.getsockname()[1],
                "enabled": True,
            },
        }
        try:
            player = ardy.ArdyPlayer(self.source, cfg, dry_run=False, rate=120.0, verbose=False)
            player.run(duration=0.25)

            hmd_packets = 0
            while True:
                try:
                    hmd.recvfrom(8192)
                    hmd_packets += 1
                except socket.timeout:
                    break
            self.assertGreater(hmd_packets, 0)

            addresses = []
            while True:
                try:
                    data, _ = osc.recvfrom(8192)
                except socket.timeout:
                    break
                try:
                    message = decode(data)
                except Exception:  # noqa: BLE001 - ignore malformed packets
                    continue
                if isinstance(message, tuple):
                    addresses.append(message[0])
            self.assertIn("/tracking/trackers/1/position", addresses)
            self.assertIn("/tracking/trackers/2/position", addresses)
            self.assertIn("/tracking/trackers/3/position", addresses)
            self.assertIn("/tracking/trackers/4/position", addresses)
            self.assertIn("/tracking/trackers/head/position", addresses)
        finally:
            for sock in sockets:
                sock.close()


if __name__ == "__main__":
    unittest.main()

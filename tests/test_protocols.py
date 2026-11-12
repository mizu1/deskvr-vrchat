from __future__ import annotations

import socket
import unittest

from vdvr.glassvr import (
    GlassVrClient,
    IndexInput,
    _EXT,
    _EXT_V23,
    _IDX,
    _IDX_V23,
    _POS,
    _POS_V23,
    _ROT,
    _ROT_V23,
    to_openvr_position,
    to_openvr_rotation,
)
from vdvr.math3d import q_from_euler_deg, qrotate
from vdvr.osc import decode
from vdvr.vrchat import VrchatOsc


class CoordinateConversionTests(unittest.TestCase):
    def test_position_flips_z(self) -> None:
        self.assertEqual(to_openvr_position((1.0, 2.0, 3.0)), (1.0, 2.0, -3.0))

    def test_yaw_maps_to_openvr_forward(self) -> None:
        # Unity: yaw +90 turns forward (+Z) to right (+X).
        q_unity = q_from_euler_deg(0.0, 90.0, 0.0)
        q_ovr = to_openvr_rotation(q_unity)
        self.assertAlmostEqual(q_ovr[0], 0.7071068, places=5)
        self.assertAlmostEqual(q_ovr[2], -0.7071068, places=5)

        # In OpenVR space forward is -Z; rotating it must also land on +X.
        forward_ovr = to_openvr_position((0.0, 0.0, 1.0))
        rotated = qrotate(q_ovr, forward_ovr)
        self.assertAlmostEqual(rotated[0], 1.0, places=5)
        self.assertAlmostEqual(rotated[2], 0.0, places=5)

    def test_pitch_maps_to_openvr_down(self) -> None:
        q_unity = q_from_euler_deg(90.0, 0.0, 0.0)
        q_ovr = to_openvr_rotation(q_unity)
        forward_ovr = to_openvr_position((0.0, 0.0, 1.0))
        rotated = qrotate(q_ovr, forward_ovr)
        self.assertAlmostEqual(rotated[1], -1.0, places=5)


class GlassVrPacketTests(unittest.TestCase):
    def test_struct_sizes(self) -> None:
        self.assertEqual(_POS.size, 27)
        self.assertEqual(_ROT.size, 35)
        self.assertEqual(_IDX.size, 79)
        self.assertEqual(_EXT.size, 4)
        # GlassVR <= v2.3 uses a one byte tag.
        self.assertEqual(_POS_V23.size, 25)
        self.assertEqual(_ROT_V23.size, 33)
        self.assertEqual(_IDX_V23.size, 77)
        self.assertEqual(_EXT_V23.size, 2)

    def test_index_packet(self) -> None:
        state = IndexInput(a=True, trigger=1.0, grip_pull=1.0, joystick_y=1.0)
        packet = state.packet()
        self.assertEqual(len(packet), _IDX.size)
        self.assertEqual(packet[:3], b"idx")

        legacy = state.packet_v23()
        self.assertEqual(len(legacy), _IDX_V23.size)
        self.assertEqual(legacy[:1], b"I")

    def test_full_trigger_sets_click(self) -> None:
        state = IndexInput(trigger=1.0, grip_pull=0.9)
        unpacked = _IDX.unpack(state.packet())
        # tag, a, b, system, joystick_click, trigger_click, ...
        self.assertTrue(unpacked[5])  # trigger_click
        self.assertAlmostEqual(unpacked[17], 1.0)  # trigger analog
        self.assertAlmostEqual(unpacked[19], 0.9)  # grip pull


class GlassVrWireTests(unittest.TestCase):
    def _receiver(self) -> tuple[socket.socket, int]:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(2.0)
        return receiver, receiver.getsockname()[1]

    def test_v23_tags(self) -> None:
        receiver, port = self._receiver()
        client = GlassVrClient(hmd_port=port, protocol="v23")
        try:
            client.hmd((0.0, 1.5, 0.0), (1.0, 0.0, 0.0, 0.0))
            first = receiver.recvfrom(4096)[0]
            second = receiver.recvfrom(4096)[0]
        finally:
            client.close()
            receiver.close()
        self.assertEqual(len(first), _POS_V23.size)
        self.assertEqual(first[:1], b"P")
        self.assertEqual(len(second), _ROT_V23.size)
        self.assertEqual(second[:1], b"R")

    def test_v24_tags(self) -> None:
        receiver, port = self._receiver()
        client = GlassVrClient(hmd_port=port, protocol="v24")
        try:
            client.hmd((0.0, 1.5, 0.0), (1.0, 0.0, 0.0, 0.0))
            first = receiver.recvfrom(4096)[0]
            second = receiver.recvfrom(4096)[0]
        finally:
            client.close()
            receiver.close()
        self.assertEqual(first[:3], b"pos")
        self.assertEqual(second[:3], b"rot")

    def test_both_protocols(self) -> None:
        receiver, port = self._receiver()
        client = GlassVrClient(hmd_port=port, protocol="both")
        try:
            client.hmd((0.0, 1.5, 0.0), (1.0, 0.0, 0.0, 0.0))
            packets = [receiver.recvfrom(4096)[0] for _ in range(4)]
        finally:
            client.close()
            receiver.close()
        tags = {packet[:1] for packet in packets if packet[:1] in (b"P", b"R")}
        tags3 = {packet[:3] for packet in packets if packet[:3] in (b"pos", b"rot")}
        self.assertEqual(tags, {b"P", b"R"})
        self.assertEqual(tags3, {b"pos", b"rot"})


class VrchatSenderTests(unittest.TestCase):
    def test_tracker_messages_reach_udp_socket(self) -> None:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(2.0)
        port = receiver.getsockname()[1]

        sender = VrchatOsc(port=port)
        try:
            sender.tracker(3, (0.1, 0.038, -0.2), (0.0, 45.0, 0.0))
            first = receiver.recvfrom(4096)[0]
            second = receiver.recvfrom(4096)[0]
        finally:
            sender.close()
            receiver.close()

        address1, args1 = decode(first)  # type: ignore[misc]
        address2, args2 = decode(second)  # type: ignore[misc]
        self.assertEqual(address1, "/tracking/trackers/3/position")
        self.assertEqual(len(args1), 3)
        self.assertAlmostEqual(args1[1], 0.038, places=5)
        self.assertEqual(address2, "/tracking/trackers/3/rotation")
        self.assertAlmostEqual(args2[1], 45.0, places=4)

    def test_axes_reset_to_zero(self) -> None:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(2.0)
        port = receiver.getsockname()[1]
        sender = VrchatOsc(port=port)
        try:
            sender.axis("Vertical", 1.0, now=0.0, force=True)
            item = receiver.recvfrom(4096)[0]
            address, args = decode(item)  # type: ignore[misc]
            self.assertEqual(address, "/input/Vertical")
            self.assertAlmostEqual(args[0], 1.0, places=5)

            sender.axis("Vertical", 0.0, now=0.1, force=True)
            item = receiver.recvfrom(4096)[0]
            address, args = decode(item)  # type: ignore[misc]
            self.assertEqual(address, "/input/Vertical")
            self.assertEqual(args[0], 0.0)
        finally:
            sender.close()
            receiver.close()


if __name__ == "__main__":
    unittest.main()

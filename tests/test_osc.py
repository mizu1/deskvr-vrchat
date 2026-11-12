from __future__ import annotations

import unittest

from vdvr.osc import decode, encode


class OscTests(unittest.TestCase):
    def test_roundtrip_floats(self) -> None:
        packet = encode("/tracking/trackers/1/position", 1.5, -2.25, 0.0)
        address, args = decode(packet)  # type: ignore[misc]
        self.assertEqual(address, "/tracking/trackers/1/position")
        self.assertEqual(len(args), 3)
        self.assertAlmostEqual(args[0], 1.5, places=5)
        self.assertAlmostEqual(args[1], -2.25, places=5)
        self.assertAlmostEqual(args[2], 0.0, places=5)

    def test_roundtrip_mixed_types(self) -> None:
        packet = encode("/input/Jump", 1)
        address, args = decode(packet)  # type: ignore[misc]
        self.assertEqual(address, "/input/Jump")
        self.assertEqual(args, [1])

        packet = encode("/chatbox/input", "hi", True)
        address, args = decode(packet)  # type: ignore[misc]
        self.assertEqual(address, "/chatbox/input")
        self.assertEqual(args, ["hi", True])

    def test_alignment(self) -> None:
        # Address "abc" must be padded to 4 bytes + null terminator.
        packet = encode("/abc", 1.0)
        self.assertEqual(packet[:8], b"/abc\x00\x00\x00\x00")
        self.assertEqual(packet[8:12], b",f\x00\x00")

    def test_bundle(self) -> None:
        message = encode("/a", 1.0)
        bundle = (
            b"#bundle\x00"
            + b"\x00" * 8
            + len(message).to_bytes(4, "big")
            + message
        )
        decoded = decode(bundle)
        self.assertIsInstance(decoded, list)
        self.assertEqual(decoded[0][0], "/a")  # type: ignore[index]

    def test_bool_tags(self) -> None:
        packet = encode("/x", True, False)
        self.assertIn(b",TF", packet)


if __name__ == "__main__":
    unittest.main()

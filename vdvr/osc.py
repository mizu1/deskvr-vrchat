"""Minimal OSC 1.0 encoder/decoder (UDP friendly, stdlib only).

Only the types VRChat actually uses are supported: float ('f'), int ('i'),
bool ('T'/'F'), and string ('s').
"""

from __future__ import annotations

import struct
from typing import Any


def _align4(value: int) -> int:
    return (value + 3) & ~3


def _encode_string(value: str) -> bytes:
    raw = value.encode("utf-8") + b"\x00"
    return raw + (b"\x00" * (_align4(len(raw)) - len(raw)))


def encode(address: str, *args: Any) -> bytes:
    """Encode a single OSC message."""
    tags = ","
    payload = bytearray()

    for arg in args:
        if isinstance(arg, bool):
            tags += "T" if arg else "F"
        elif isinstance(arg, int):
            tags += "i"
            payload += struct.pack(">i", arg)
        elif isinstance(arg, float):
            tags += "f"
            payload += struct.pack(">f", arg)
        elif isinstance(arg, str):
            tags += "s"
            payload += _encode_string(arg)
        else:  # pragma: no cover - programmer error
            raise TypeError(f"Unsupported OSC argument type: {type(arg)!r}")

    return _encode_string(address) + _encode_string(tags) + bytes(payload)


def _read_string(data: bytes, offset: int) -> tuple[str, int]:
    end = data.find(b"\x00", offset)
    if end < 0:
        raise ValueError("Unterminated OSC string")
    value = data[offset:end].decode("utf-8", errors="replace")
    offset = _align4(end + 1)
    return value, offset


def decode(data: bytes) -> tuple[str, list[Any]] | list[Any]:
    """Decode one OSC packet.

    Returns ``(address, args)`` for a message, or a list of decoded messages
    for a bundle.
    """
    if data.startswith(b"#bundle\x00"):
        messages: list[Any] = []
        offset = 16  # "#bundle\0" + 8 byte time tag
        while offset + 4 <= len(data):
            (size,) = struct.unpack(">i", data[offset : offset + 4])
            offset += 4
            chunk = data[offset : offset + size]
            offset += size
            if chunk:
                messages.append(decode(chunk))
        return messages

    address, offset = _read_string(data, 0)
    if offset >= len(data):
        return address, []

    tags, offset = _read_string(data, offset)
    if not tags.startswith(","):
        # Some senders omit the comma. Be forgiving.
        tags = "," + tags

    args: list[Any] = []
    for tag in tags[1:]:
        if tag == "f":
            (value,) = struct.unpack(">f", data[offset : offset + 4])
            offset += 4
            args.append(float(value))
        elif tag == "i":
            (value,) = struct.unpack(">i", data[offset : offset + 4])
            offset += 4
            args.append(int(value))
        elif tag == "T":
            args.append(True)
        elif tag == "F":
            args.append(False)
        elif tag == "s":
            value, offset = _read_string(data, offset)
            args.append(value)
        else:  # pragma: no cover - unknown tag
            break

    return address, args

#!/usr/bin/env python3
"""Print every OSC packet received on a UDP port (debug helper).

Examples::

    python tools/osc_monitor.py                 # VRChat's outgoing port 9001
    python tools/osc_monitor.py --port 9000     # what this tool sends to VRChat
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vdvr.osc import decode  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--quiet", action="store_true", help="skip high-rate tracking messages")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    print(f"listening on {args.host}:{args.port} (Ctrl+C to stop)", flush=True)

    try:
        while True:
            data, addr = sock.recvfrom(8192)
            try:
                result = decode(data)
            except Exception as exc:  # noqa: BLE001
                print(f"{addr} undecodable ({exc}): {data[:48]!r}")
                continue

            messages = result if isinstance(result, list) else [result]
            for message in messages:
                address, values = message  # type: ignore[misc]
                if args.quiet and address.startswith("/tracking/trackers/"):
                    continue
                stamp = time.strftime("%H:%M:%S")
                printable = ", ".join(
                    f"{value:.3f}" if isinstance(value, float) else repr(value) for value in values
                )
                print(f"[{stamp}] {address} {printable}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

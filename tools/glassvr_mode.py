#!/usr/bin/env python3
"""Toggle GlassVR between "normal VR" and "vdvr desktop-fake-VR" modes.

Usage::

    python tools/glassvr_mode.py status
    python tools/glassvr_mode.py normal   # disable virtual HMD/controllers, for real VR
    python tools/glassvr_mode.py vdvr     # enable virtual HMD/controllers, for vdvr

The GlassVR driver reads %APPDATA%\\glassvr\\settings.json live (every ~1 ms),
but SteamVR HMD selection is happier after a restart.  This script never
restarts SteamVR itself; close vdvr and restart SteamVR after switching.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

SETTINGS = Path(os.environ.get("APPDATA", "")) / "glassvr" / "settings.json"

V23_PORTS = {"hmd port": 9100, "cr port": 9101, "cl port": 9102}
V23_MODES = {
    "hmdpos mode": "UDP",
    "hmdrot mode": "UDP",
    "crpos mode": "UDP",
    "crrot mode": "UDP",
    "crinput mode": "UDP",
    "clpos mode": "UDP",
    "clrot mode": "UDP",
    "clinput mode": "UDP",
}
ZERO_OFFSET_SUFFIXES = (
    "offset world x",
    "offset world y",
    "offset world z",
    "offset world yaw",
    "offset world pitch",
    "offset world roll",
    "offset local x",
    "offset local y",
    "offset local z",
    "offset local yaw",
    "offset local pitch",
    "offset local roll",
)


def load() -> dict:
    if not SETTINGS.exists():
        raise SystemExit(f"GlassVR settings not found: {SETTINGS}")
    with SETTINGS.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def save(data: dict) -> Path:
    backup = SETTINGS.with_name(SETTINGS.name + ".bak-" + time.strftime("%Y%m%d-%H%M%S"))
    shutil.copy2(SETTINGS, backup)
    with SETTINGS.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)
        handle.write("\n")
    return backup


def apply_normal(data: dict) -> None:
    data["enable hmd"] = False
    data["enable cr"] = False
    data["enable cl"] = False


def apply_vdvr(data: dict) -> None:
    data.update(V23_MODES)
    data.update(V23_PORTS)
    data["enable hmd"] = True
    data["enable cr"] = True
    data["enable cl"] = True
    data["ipd"] = max(float(data.get("ipd", 0.0) or 0.0), 0.064)
    for device in ("hmd", "cr", "cl"):
        data[f"{device} playspace yaw"] = 0.0
        for suffix in ZERO_OFFSET_SUFFIXES:
            data[f"{device} {suffix}"] = 0.0


def print_status(data: dict) -> None:
    print("GlassVR settings:", SETTINGS)
    for key in (
        "enable hmd",
        "enable cr",
        "enable cl",
        "hmdpos mode",
        "crinput mode",
        "clinput mode",
        "hmd port",
        "cr port",
        "cl port",
        "ipd",
    ):
        print(f"  {key:16} = {data.get(key)!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("status", "normal", "vdvr"), nargs="?", default="status")
    args = parser.parse_args()

    data = load()
    print_status(data)

    if args.mode == "status":
        return 0

    app_running = False
    try:
        import subprocess

        output = subprocess.check_output(
            ["tasklist", "/FI", "IMAGENAME eq GlassVR.exe"], text=True, errors="ignore"
        )
        app_running = "GlassVR.exe" in output
    except Exception:  # noqa: BLE001
        pass
    if app_running:
        print("\n[!] GlassVR.exe is running; close it first or it may overwrite these settings.")

    if args.mode == "normal":
        apply_normal(data)
        backup = save(data)
        print("\nMode: NORMAL VR")
        print("  virtual HMD/controllers disabled")
    else:
        apply_vdvr(data)
        backup = save(data)
        print("\nMode: VDVR (desktop fake VR)")
        print("  virtual HMD/controllers enabled on UDP 9100/9102/9101")

    print(f"  backup: {backup}")
    print("\nNext: close python run.py, then restart SteamVR (SteamVR menu -> Quit, start again).")
    if args.mode == "vdvr":
        print("Then run start.bat and launch VRChat in VR mode.")
    else:
        print("Then launch VRChat normally from Steam / your real headset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

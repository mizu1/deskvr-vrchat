#!/usr/bin/env python3
"""Environment checker for the SteamVR + VRChat + GlassVR stack.

Usage::

    python tools/check_env.py
    python tools/check_env.py --apply-drivers   # sets steamvr.activateMultipleDrivers=true

Nothing is modified unless --apply-drivers is passed; a timestamped backup of
steamvr.vrsettings is written first.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


def steam_root() -> Path | None:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
            value, _ = winreg.QueryValueEx(key, "SteamPath")
            candidate = Path(str(value))
            if candidate.exists():
                return candidate
    except OSError:
        pass
    for candidate in (Path(r"C:\Program Files (x86)\Steam"), Path(r"C:\Program Files\Steam"), Path(r"E:\Steam")):
        if (candidate / "steamapps").exists():
            return candidate
    return None


def libraries(root: Path) -> list[Path]:
    found = [root]
    vdf = root / "steamapps" / "libraryfolders.vdf"
    if vdf.exists():
        text = vdf.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r'"path"\s*"([^"]+)"', text):
            candidate = Path(match.group(1).replace("\\\\", "\\"))
            if (candidate / "steamapps").exists() and candidate not in found:
                found.append(candidate)
    return found


def find_app(root: Path, app_dir: str) -> Path | None:
    for lib in libraries(root):
        candidate = lib / "steamapps" / "common" / app_dir
        if candidate.exists():
            return candidate
    return None


def port_in_use(port: int) -> bool:
    """Windows-friendly check (a 0.0.0.0 bind can succeed even when the port is taken)."""
    try:
        output = subprocess.check_output(
            ["netstat", "-ano", "-p", "UDP"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:  # noqa: BLE001
        return False
    return re.search(rf":{port}\s", output) is not None


def load_settings(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply-drivers", action="store_true", help="enable activateMultipleDrivers")
    args = parser.parse_args()

    ok = True
    print("== Steam ==")
    root = steam_root()
    if not root:
        print("  [!] Steam not found")
        return 1
    print(f"  Steam: {root}")

    print("== SteamVR / VRChat ==")
    steamvr = find_app(root, "SteamVR")
    vrchat = find_app(root, "VRChat")
    print(f"  SteamVR: {steamvr or '[!] missing'}")
    print(f"  VRChat:  {vrchat or '[!] missing'}")
    ok = ok and bool(steamvr) and bool(vrchat)

    print("== GlassVR driver ==")
    glass = (steamvr / "drivers" / "glassvrdriver") if steamvr else None
    installed = bool(glass and glass.exists())
    print(f"  drivers/glassvrdriver: {'installed' if installed else '[!] not installed'}")
    if installed and glass is not None:
        manifest = glass / "driver.vrdrivermanifest"
        print(f"  manifest: {manifest if manifest.exists() else '[!] missing'}")
        dll = glass / "bin" / "win64" / "driver_glassvr.dll"
        if dll.exists():
            data = dll.read_bytes()
            # v2.4+ has the "pos mode" settings key as a string literal; v2.3
            # uses the older 1-byte UDP tag protocol and does not.
            if b"pos mode\x00" in data:
                print("  protocol hint: v2.4+ (3-byte tags) -> glassvr.protocol = v24 or both")
            else:
                print("  protocol hint: v2.3 (1-byte tags)  -> glassvr.protocol = v23 or both")
    else:
        print("  Install: https://github.com/DaniXmir/GlassVr/releases -> run main.exe -> Driver tab -> install")
        ok = False

    print("== steamvr.vrsettings ==")
    settings_path = root / "config" / "steamvr.vrsettings"
    try:
        settings = load_settings(settings_path)
    except Exception as exc:  # noqa: BLE001
        print(f"  [!] cannot parse {settings_path}: {exc}")
        settings = {}

    steamvr_section = settings.get("steamvr", {})
    multiple = bool(steamvr_section.get("activateMultipleDrivers", False))
    require_hmd = steamvr_section.get("requireHmd", "default(true)")
    forced = steamvr_section.get("forcedDriver", "")
    print(f"  activateMultipleDrivers: {multiple}")
    print(f"  requireHmd: {require_hmd}")
    print(f"  forcedDriver: {forced or '(empty)'}")

    if not multiple:
        ok = False
        if args.apply_drivers and settings_path.exists():
            backup = settings_path.with_suffix(f".vrsettings.bak-{time.strftime('%Y%m%d-%H%M%S')}")
            shutil.copy2(settings_path, backup)
            settings.setdefault("steamvr", {})["activateMultipleDrivers"] = True
            with settings_path.open("w", encoding="utf-8") as handle:
                json.dump(settings, handle, indent=3)
                handle.write("\n")
            print(f"  [fixed] activateMultipleDrivers=true (backup: {backup.name})")
        else:
            print("  Fix: python tools/check_env.py --apply-drivers")

    if forced and forced.lower() == "null":
        print("  [!] forcedDriver=null will fight GlassVR; remove it while using virtual HMD")
        ok = False

    print("== UDP ports ==")
    for name, port in (("VRChat OSC in", 9000), ("GlassVR HMD", 9100), ("GlassVR right", 9101), ("GlassVR left", 9102)):
        used = port_in_use(port)
        print(f"  {name:<14} {port}: {'in use' if used else 'free'}")

    print("== Summary ==")
    print("  ready" if ok else "  fix the items marked [!] above, then start SteamVR and VRChat")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

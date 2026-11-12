"""Configuration loading with nested defaults."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "vrchat": {
        "host": "127.0.0.1",
        "port": 9000,
        "enabled": True,
    },
    "glassvr": {
        "host": "127.0.0.1",
        "enabled": True,
        # Must match the ports assigned in the GlassVR UI.
        # Deliberately different from VRChat's 9000/9001 OSC ports.
        "hmd_port": 9100,
        "left_port": 9102,
        "right_port": 9101,
        # v23 = GlassVR <= 2.3 (1 byte tags), v24 = 2.4+ (3 byte tags), both = compatible
        "protocol": "both",
    },
    "capture": {
        "processes": ["VRChat.exe", "vrcompositor.exe", "vrmonitor.exe"],
        "mouse_sensitivity": 0.045,
        "invert_y": False,
        "recenter_mouse": True,
    },
    "body": {
        "height": 1.70,
        "profile": "standard",  # minimal | standard | full
    },
    "input": {
        "movement_via": "controller",  # controller | osc | both
        "menu_via": "controller",  # controller | osc | both
        "jump_via": "both",  # controller | osc | both
        "pointer_key": "ALT",  # hold to aim the right hand laser with the mouse
        "pointer_sensitivity": 0.08,  # degrees per pixel in pointer mode
    },
    "start": {
        "enabled": True,
        "countdown": 5,
        "focus_game": True,
    },
    "debug": False,
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        candidate = Path(__file__).resolve().parent.parent / "config.json"
    else:
        candidate = Path(path)
    data: dict[str, Any] = {}
    if candidate.exists():
        with candidate.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    return _deep_merge(DEFAULTS, data)


def save_default(path: str | Path) -> Path:
    target = Path(path)
    target.write_text(json.dumps(DEFAULTS, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target

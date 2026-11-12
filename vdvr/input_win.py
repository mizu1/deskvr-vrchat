"""Global keyboard / mouse access on Windows via user32 (no third-party deps).

We poll state instead of using hooks so the game keeps receiving its own input.
Mouse look is implemented by re-centering the cursor each frame, which gives
unbounded yaw like a normal FPS.

If VRChat runs elevated and this tool does not, Windows will hide its keystrokes
from us.  Run both at the same integrity level (preferably both normal, or both
as administrator).
"""

from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short
user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
user32.GetCursorPos.restype = wintypes.BOOL
user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
user32.SetCursorPos.restype = wintypes.BOOL
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetWindowRect.restype = wintypes.BOOL
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.GetSystemMetrics.argtypes = [ctypes.c_int]
user32.GetSystemMetrics.restype = ctypes.c_int
user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
user32.FindWindowW.restype = wintypes.HWND
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.ShowWindow.restype = wintypes.BOOL

kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.QueryFullProcessImageNameW.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.LPWSTR,
    ctypes.POINTER(wintypes.DWORD),
]
kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL

VK: dict[str, int] = {
    "W": 0x57,
    "A": 0x41,
    "S": 0x53,
    "D": 0x44,
    "E": 0x45,
    "F": 0x46,
    "Q": 0x51,
    "R": 0x52,
    "C": 0x43,
    "B": 0x42,
    "M": 0x4D,
    "V": 0x56,
    "SPACE": 0x20,
    "SHIFT": 0x10,
    "CTRL": 0x11,
    "ALT": 0x12,
    "LEFT": 0x25,
    "UP": 0x26,
    "RIGHT": 0x27,
    "DOWN": 0x28,
    "ESC": 0x1B,
    "TAB": 0x09,
    "F7": 0x76,
    "F8": 0x77,
    "F9": 0x78,
    "F10": 0x79,
    "F6": 0x75,
    "LMB": 0x01,
    "RMB": 0x02,
    "MMB": 0x04,
}

WATCHED: tuple[str, ...] = tuple(VK)

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


class WinInput:
    def __init__(self, capture_processes: str | tuple[str, ...] = "VRChat.exe") -> None:
        if isinstance(capture_processes, str):
            names = [part.strip() for part in capture_processes.split(",") if part.strip()]
        else:
            names = list(capture_processes)
        self.capture_processes = {name.lower() for name in names}
        self._states: dict[str, bool] = {}
        self._prev: dict[str, bool] = {}
        self._fg_checked = 0.0
        self._fg_name = ""
        self._fg_hwnd = 0
        self._center: tuple[int, int] | None = None
        self._warned_not_foreground = False

    # ------------------------------------------------------------- key state
    @staticmethod
    def _vk_down(vk: int) -> bool:
        return bool(user32.GetAsyncKeyState(vk) & 0x8000)

    def down(self, name: str) -> bool:
        vk = VK.get(name.upper())
        if vk is None:
            return False
        return self._vk_down(vk)

    def snapshot(self) -> None:
        """Call exactly once per frame before reading edges."""
        self._prev = self._states
        self._states = {name: self._vk_down(vk) for name, vk in VK.items()}

    def just_pressed(self, name: str) -> bool:
        name = name.upper()
        return bool(self._states.get(name)) and not bool(self._prev.get(name))

    def just_released(self, name: str) -> bool:
        name = name.upper()
        return (not self._states.get(name, False)) and bool(self._prev.get(name))

    # ------------------------------------------------------------- focus
    def _refresh_foreground(self, now: float) -> None:
        if now - self._fg_checked < 0.15:
            return
        self._fg_checked = now
        hwnd = user32.GetForegroundWindow()
        pid = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        name = ""
        process = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if process:
            buffer = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(1024)
            if kernel32.QueryFullProcessImageNameW(process, 0, buffer, ctypes.byref(size)):
                name = os.path.basename(buffer.value)
            kernel32.CloseHandle(process)
        self._fg_name = name.lower()
        self._fg_hwnd = int(hwnd or 0)

    @property
    def foreground_name(self) -> str:
        return self._fg_name

    def foreground_matches(self, now: float | None = None) -> bool:
        now = time.perf_counter() if now is None else now
        self._refresh_foreground(now)
        return self._fg_name in self.capture_processes

    # ------------------------------------------------------------- mouse
    def _foreground_center(self) -> tuple[int, int]:
        rect = wintypes.RECT()
        if self._fg_hwnd and user32.GetWindowRect(self._fg_hwnd, ctypes.byref(rect)):
            if rect.right > rect.left + 32 and rect.bottom > rect.top + 32:
                return ((rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2)
        return (user32.GetSystemMetrics(0) // 2, user32.GetSystemMetrics(1) // 2)

    def mouse_look(self, recenter: bool = True) -> tuple[float, float]:
        """Return (dx, dy) since the last call and re-center the OS cursor."""
        point = wintypes.POINT()
        if not user32.GetCursorPos(ctypes.byref(point)):
            return (0.0, 0.0)

        center = self._center or self._foreground_center()
        self._center = center
        dx = float(point.x - center[0])
        dy = float(point.y - center[1])

        if recenter:
            user32.SetCursorPos(center[0], center[1])

        # Ignore impossible jumps (resolution change, alt-tab, remote desktop).
        if abs(dx) > 450.0 or abs(dy) > 450.0:
            return (0.0, 0.0)
        return (dx, dy)

    def reset_cursor_center(self) -> None:
        self._center = self._foreground_center()


def backend_available() -> bool:
    return os.name == "nt"


def focus_window(titles: tuple[str, ...] = ("Headset Window", "VRChat")) -> str | None:
    """Bring a known game window to the foreground. Returns the title found."""
    for title in titles:
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
            return title
    return None

"""Capture the client area of a running Windows graphics demo."""

import argparse
import ctypes
import subprocess
import time
from ctypes import wintypes
from pathlib import Path

from PIL import ImageGrab


user32 = ctypes.windll.user32
try:
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
except OSError:
    pass


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


def windows_for_pid(pid):
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _):
        window_pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_pid))
        if window_pid.value == pid and user32.IsWindowVisible(hwnd):
            rect = wintypes.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rect))
            width = rect.right - rect.left
            height = rect.bottom - rect.top
            if width >= 320 and height >= 240:
                title = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, title, len(title))
                klass = ctypes.create_unicode_buffer(128)
                user32.GetClassNameW(hwnd, klass, len(klass))
                if klass.value != "ConsoleWindowClass":
                    found.append((width * height, hwnd, width, height, title.value))
        return True

    user32.EnumWindows(visit, 0)
    return sorted(found, reverse=True)


def capture(hwnd, width, height, path, desktop, window_capture):
    if not window_capture:
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x43)  # HWND_TOPMOST, keep size/position
        user32.BringWindowToTop(hwnd)
        user32.keybd_event(0x12, 0, 0, 0)
        user32.keybd_event(0x12, 0, 2, 0)
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.25)
        if user32.GetForegroundWindow() != hwnd:
            foreground = user32.GetForegroundWindow()
            foreground_thread = user32.GetWindowThreadProcessId(foreground, None)
            current_thread = ctypes.windll.kernel32.GetCurrentThreadId()
            user32.AttachThreadInput(current_thread, foreground_thread, True)
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
            user32.AttachThreadInput(current_thread, foreground_thread, False)
            time.sleep(0.25)
        if user32.GetForegroundWindow() != hwnd:
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            user32.SetCursorPos(rect.left + 120, rect.top + 15)
            user32.mouse_event(0x0002, 0, 0, 0, 0)
            user32.mouse_event(0x0004, 0, 0, 0, 0)
            time.sleep(0.25)
    if user32.GetForegroundWindow() != hwnd:
        print(f"warning: demo window {hwnd} is topmost but not foreground; inspect capture before accepting", flush=True)
    point = POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(point)):
        raise RuntimeError("ClientToScreen failed")
    bbox = (point.x, point.y, point.x + width, point.y + height)
    title = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, title, len(title))
    print(f"PID window={hwnd} title={title.value!r} foreground={user32.GetForegroundWindow()} bbox={bbox}", flush=True)
    try:
        if window_capture:
            image = ImageGrab.grab(window=hwnd)
        elif user32.GetForegroundWindow() == hwnd:
            image = ImageGrab.grab(bbox=bbox, all_screens=True)
        else:
            # A non-foreground client rectangle can be stale (especially when
            # DWM keeps a presentation surface cached). Ask PrintWindow for the
            # target's current client contents instead of sampling the desktop.
            image = ImageGrab.grab(window=hwnd)
    except OSError:
        image = ImageGrab.grab(window=hwnd)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    if desktop and path.stem.endswith("-01"):
        ImageGrab.grab(all_screens=True).save(path.with_name(path.stem + "-desktop.png"))
    print(f"{path}: {image.width}x{image.height}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--times", default="2,5,8", help="seconds after launch")
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--desktop", action="store_true")
    parser.add_argument(
        "--window-capture",
        action="store_true",
        help="capture the target window with PrintWindow without changing foreground focus",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing demo command")
    times = sorted(float(value) for value in args.times.split(","))
    proc = subprocess.Popen(command, cwd=args.cwd)
    start = time.monotonic()
    try:
        for index, seconds in enumerate(times, 1):
            while time.monotonic() - start < seconds:
                if proc.poll() is not None:
                    raise RuntimeError(f"demo exited early with code {proc.returncode}")
                time.sleep(0.05)
            candidates = windows_for_pid(proc.pid)
            if not candidates:
                raise RuntimeError(f"no visible window for PID {proc.pid}")
            print(f"window candidates: {candidates}", flush=True)
            _, hwnd, width, height, _ = candidates[0]
            capture(
                hwnd,
                width,
                height,
                args.output.with_name(f"{args.output.stem}-{index:02d}.png"),
                args.desktop,
                args.window_capture,
            )
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    main()

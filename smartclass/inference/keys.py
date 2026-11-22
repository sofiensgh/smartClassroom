"""Step 4 - send a slide key (tool only: pyautogui + Windows focus handling).

Keys always go to the window that has the keyboard focus. When the user ticks
"Send keys" in our app, the focus is on OUR window, so PowerPoint never saw the keys.
Fix: before each key, give the focus back to the PowerPoint slideshow window
(class "screenClass") if one is open. Otherwise the key goes to the focused window
(PDF viewer, browser...).
"""
import sys
import time

_pyautogui = None
SLIDESHOW_CLASSES = ("screenClass",)   # PowerPoint slide show window


def _focus_slideshow():
    """Returns 'PowerPoint' if a slideshow window is (now) in the foreground, else None."""
    if sys.platform != "win32":
        return None
    import ctypes
    u = ctypes.windll.user32
    for cls in SLIDESHOW_CLASSES:
        h = u.FindWindowW(cls, None)
        if not h:
            continue
        if u.GetForegroundWindow() != h:
            # Windows only lets a background app change the foreground window right after a key
            # event: a harmless Alt tap makes SetForegroundWindow work.
            u.keybd_event(0x12, 0, 0, 0)
            u.keybd_event(0x12, 0, 2, 0)
            u.SetForegroundWindow(h)
            time.sleep(0.05)
        return "PowerPoint" if u.GetForegroundWindow() == h else None
    return None


def send_key(key):
    """Press `key` for the presentation. Returns where it went ('PowerPoint' or 'focused window')."""
    global _pyautogui
    if _pyautogui is None:
        import pyautogui
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False   # we never move the mouse
        _pyautogui = pyautogui
    target = _focus_slideshow() or "focused window"
    _pyautogui.press(key)
    return target

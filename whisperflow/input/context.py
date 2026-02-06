"""Active window and context detection for Linux (X11/Wayland)."""

import json
import logging
import os
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _is_wayland() -> bool:
    return bool(os.environ.get("WAYLAND_DISPLAY"))


@dataclass
class WindowContext:
    """Information about the currently active window."""
    app_name: str  # e.g., "firefox", "slack", "code"
    window_title: str  # e.g., "Inbox - Gmail - Mozilla Firefox"
    window_class: str  # e.g., "Firefox", "Slack", "Code"

    @property
    def detected_app(self) -> str:
        """Normalized app name for tone/context matching."""
        name = self.app_name.lower()
        wclass = self.window_class.lower()
        title = self.window_title.lower()

        # Email clients
        if any(kw in name or kw in wclass for kw in ("thunderbird", "evolution", "geary")):
            return "email"
        if "gmail" in title or "outlook" in title or "mail" in title:
            return "email"

        # Chat/messaging
        if any(kw in name or kw in wclass for kw in ("slack", "discord", "telegram", "signal", "whatsapp", "element")):
            return name if name in ("slack", "discord", "telegram", "signal") else wclass
        if any(kw in title for kw in ("slack", "discord", "telegram", "whatsapp", "messages")):
            return next(kw for kw in ("slack", "discord", "telegram", "whatsapp", "messages") if kw in title)

        # Code editors / IDEs
        if any(kw in name or kw in wclass for kw in ("code", "cursor", "windsurf", "replit", "vim", "neovim", "emacs", "idea", "pycharm", "webstorm")):
            return "code"

        # Terminal
        if any(kw in name or kw in wclass for kw in ("terminal", "konsole", "gnome-terminal", "alacritty", "kitty", "wezterm", "xterm")):
            return "terminal"

        # Document editors
        if any(kw in name or kw in wclass for kw in ("libreoffice", "soffice")):
            return "document"
        if any(kw in title for kw in ("google docs", "notion", "obsidian")):
            return "document"

        # Browsers (check title for web app context)
        if any(kw in name or kw in wclass for kw in ("firefox", "chrome", "chromium", "brave", "vivaldi")):
            if "gmail" in title or "mail" in title:
                return "email"
            if "slack" in title:
                return "slack"
            if "discord" in title:
                return "discord"
            if "chatgpt" in title or "claude" in title:
                return "ai-chat"
            if "docs" in title or "notion" in title:
                return "document"
            return "browser"

        return "other"


def get_active_window() -> WindowContext:
    """Get information about the currently active window.

    Tries xdotool (X11) first, then gdbus (GNOME/Wayland), falls back to empty.
    """
    if _is_wayland():
        ctx = _get_active_window_wayland()
        if ctx.app_name or ctx.window_title:
            return ctx

    return _get_active_window_x11()


def _get_active_window_x11() -> WindowContext:
    """X11 window detection using xdotool + xprop."""
    try:
        # Get active window ID
        win_id = subprocess.run(
            ["xdotool", "getactivewindow"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()

        if not win_id:
            return WindowContext("", "", "")

        # Get window name
        win_name = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowname"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()

        # Get window PID and process name
        pid_result = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowpid"],
            capture_output=True, text=True, timeout=2,
        )
        pid = pid_result.stdout.strip()

        app_name = ""
        if pid:
            proc_result = subprocess.run(
                ["ps", "-p", pid, "-o", "comm="],
                capture_output=True, text=True, timeout=2,
            )
            app_name = proc_result.stdout.strip()

        # Get WM_CLASS
        wm_class = ""
        try:
            xprop_result = subprocess.run(
                ["xprop", "-id", win_id, "WM_CLASS"],
                capture_output=True, text=True, timeout=2,
            )
            if "=" in xprop_result.stdout:
                wm_class = xprop_result.stdout.split("=")[1].strip().strip('"').split('"')[0]
        except Exception:
            pass

        return WindowContext(
            app_name=app_name,
            window_title=win_name,
            window_class=wm_class or app_name,
        )

    except FileNotFoundError:
        logger.warning("xdotool not found - install with: sudo apt install xdotool")
        return WindowContext("", "", "")
    except Exception as e:
        logger.debug(f"Failed to get active window (X11): {e}")
        return WindowContext("", "", "")


def _get_active_window_wayland() -> WindowContext:
    """Wayland window detection using gdbus (GNOME) or sway IPC."""
    # Try GNOME Shell / Mutter via gdbus
    try:
        result = subprocess.run(
            [
                "gdbus", "call", "--session",
                "--dest", "org.gnome.Shell",
                "--object-path", "/org/gnome/Shell/Extensions/WindowsExt",
                "--method", "org.gnome.Shell.Extensions.WindowsExt.FocusTitle",
            ],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            title = result.stdout.strip().strip("()'\"")
            # Try to get app ID via eval
            app_result = subprocess.run(
                [
                    "gdbus", "call", "--session",
                    "--dest", "org.gnome.Shell",
                    "--object-path", "/org/gnome/Shell/Extensions/WindowsExt",
                    "--method", "org.gnome.Shell.Extensions.WindowsExt.FocusApp",
                ],
                capture_output=True, text=True, timeout=2,
            )
            app_name = app_result.stdout.strip().strip("()'\"") if app_result.returncode == 0 else ""
            return WindowContext(app_name=app_name, window_title=title, window_class=app_name)
    except (FileNotFoundError, Exception):
        pass

    # Try sway/i3 IPC
    try:
        result = subprocess.run(
            ["swaymsg", "-t", "get_tree"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            tree = json.loads(result.stdout)
            focused = _find_focused_sway(tree)
            if focused:
                app_id = focused.get("app_id", "") or ""
                title = focused.get("name", "") or ""
                wclass = focused.get("window_properties", {}).get("class", "") or app_id
                return WindowContext(app_name=app_id, window_title=title, window_class=wclass)
    except (FileNotFoundError, json.JSONDecodeError, Exception):
        pass

    # Try hyprland
    try:
        result = subprocess.run(
            ["hyprctl", "activewindow", "-j"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return WindowContext(
                app_name=data.get("class", ""),
                window_title=data.get("title", ""),
                window_class=data.get("class", ""),
            )
    except (FileNotFoundError, json.JSONDecodeError, Exception):
        pass

    return WindowContext("", "", "")


def _find_focused_sway(node: dict) -> dict | None:
    """Recursively find the focused node in sway's tree."""
    if node.get("focused"):
        return node
    for child in node.get("nodes", []) + node.get("floating_nodes", []):
        result = _find_focused_sway(child)
        if result:
            return result
    return None


def get_selected_text() -> str:
    """Get currently selected text from the active window."""
    if _is_wayland():
        # On Wayland, try wl-paste with primary selection
        try:
            result = subprocess.run(
                ["wl-paste", "--primary", "--no-newline"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                return result.stdout
        except (FileNotFoundError, Exception):
            pass
        return ""

    # X11: use xclip primary selection
    try:
        result = subprocess.run(
            ["xclip", "-selection", "primary", "-o"],
            capture_output=True, text=True, timeout=2,
        )
        return result.stdout
    except Exception:
        return ""

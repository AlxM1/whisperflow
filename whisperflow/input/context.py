"""Active window and context detection for Linux (X11/Wayland)."""

import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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

    Tries xdotool (X11) first, falls back to basic detection.
    """
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
        logger.debug(f"Failed to get active window: {e}")
        return WindowContext("", "", "")


def get_selected_text() -> str:
    """Get currently selected text from the active window using xclip."""
    try:
        result = subprocess.run(
            ["xclip", "-selection", "primary", "-o"],
            capture_output=True, text=True, timeout=2,
        )
        return result.stdout
    except Exception:
        return ""

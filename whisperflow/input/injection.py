"""Text injection into active applications on Linux."""

import logging
import os
import subprocess
import time

logger = logging.getLogger(__name__)

# Time to wait after paste before restoring clipboard (seconds).
# Longer values are safer for slow apps (SSH, remote desktop).
CLIPBOARD_RESTORE_DELAY = 0.5


def _detect_display_server() -> str:
    """Auto-detect X11 vs Wayland."""
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return "x11"  # fallback


class TextInjector:
    """Injects text into the currently active application."""

    def __init__(self, method: str = "auto", delay_ms: int = 50):
        if method == "auto":
            self.method = _detect_display_server()
        elif method in ("wl-copy", "wayland"):
            self.method = "wayland"
        else:
            self.method = "x11"
        self.delay_ms = delay_ms

    def inject(self, text: str) -> bool:
        """Inject text into the active text field.

        Uses clipboard-based injection for reliability:
        1. Save current clipboard
        2. Copy text to clipboard
        3. Wait for clipboard to settle
        4. Simulate Ctrl+V paste
        5. Wait for app to process paste
        6. Restore original clipboard

        Args:
            text: The text to inject

        Returns:
            True if successful
        """
        if not text:
            return False

        if self.method == "wayland":
            return self._inject_wayland(text)
        return self._inject_x11(text)

    def _inject_x11(self, text: str) -> bool:
        """Inject text using xclip + xdotool on X11."""
        try:
            # Save current clipboard
            old_clipboard = ""
            try:
                result = subprocess.run(
                    ["xclip", "-selection", "clipboard", "-o"],
                    capture_output=True, text=True, timeout=2,
                )
                if result.returncode == 0:
                    old_clipboard = result.stdout
            except Exception:
                pass

            # Set clipboard to our text
            proc = subprocess.Popen(
                ["xclip", "-selection", "clipboard"],
                stdin=subprocess.PIPE,
            )
            proc.communicate(input=text.encode("utf-8"))

            # Wait for clipboard to settle
            time.sleep(self.delay_ms / 1000.0)

            # Simulate Ctrl+V to paste
            subprocess.run(
                ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
                timeout=2,
            )

            # Wait for the target app to process the paste, then restore
            time.sleep(CLIPBOARD_RESTORE_DELAY)
            if old_clipboard:
                proc = subprocess.Popen(
                    ["xclip", "-selection", "clipboard"],
                    stdin=subprocess.PIPE,
                )
                proc.communicate(input=old_clipboard.encode("utf-8"))

            logger.debug(f"Injected {len(text)} chars via xdotool")
            return True

        except FileNotFoundError:
            logger.error("xdotool or xclip not found - install with: sudo apt install xdotool xclip")
            return False
        except Exception as e:
            logger.error(f"Text injection failed: {e}")
            return False

    def _inject_wayland(self, text: str) -> bool:
        """Inject text using wl-copy + wtype on Wayland."""
        try:
            # Save current clipboard
            old_clipboard = ""
            try:
                result = subprocess.run(
                    ["wl-paste", "--no-newline"],
                    capture_output=True, text=True, timeout=2,
                )
                if result.returncode == 0:
                    old_clipboard = result.stdout
            except Exception:
                pass

            # Set clipboard to our text
            proc = subprocess.Popen(
                ["wl-copy"],
                stdin=subprocess.PIPE,
            )
            proc.communicate(input=text.encode("utf-8"))

            # Wait for clipboard to settle
            time.sleep(self.delay_ms / 1000.0)

            # Simulate Ctrl+V using wtype
            subprocess.run(
                ["wtype", "-M", "ctrl", "-k", "v"],
                timeout=2,
            )

            # Wait for paste, then restore clipboard
            time.sleep(CLIPBOARD_RESTORE_DELAY)
            if old_clipboard:
                proc = subprocess.Popen(
                    ["wl-copy"],
                    stdin=subprocess.PIPE,
                )
                proc.communicate(input=old_clipboard.encode("utf-8"))

            logger.debug(f"Injected {len(text)} chars via wl-copy/wtype")
            return True

        except FileNotFoundError:
            logger.error("wl-copy or wtype not found - install wl-clipboard and wtype")
            return False
        except Exception as e:
            logger.error(f"Wayland text injection failed: {e}")
            return False

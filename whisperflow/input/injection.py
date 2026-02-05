"""Text injection into active applications on Linux."""

import logging
import subprocess
import time

logger = logging.getLogger(__name__)


class TextInjector:
    """Injects text into the currently active application."""

    def __init__(self, method: str = "xdotool", delay_ms: int = 50):
        self.method = method
        self.delay_ms = delay_ms

    def inject(self, text: str) -> bool:
        """Inject text into the active text field.

        Uses clipboard-based injection for reliability:
        1. Save current clipboard
        2. Copy text to clipboard
        3. Simulate Ctrl+V paste
        4. Restore original clipboard

        Args:
            text: The text to inject

        Returns:
            True if successful
        """
        if not text:
            return False

        if self.method == "wl-copy":
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
                old_clipboard = result.stdout
            except Exception:
                pass

            # Set clipboard to our text
            proc = subprocess.Popen(
                ["xclip", "-selection", "clipboard"],
                stdin=subprocess.PIPE,
            )
            proc.communicate(input=text.encode("utf-8"))

            # Small delay for clipboard to settle
            time.sleep(self.delay_ms / 1000.0)

            # Simulate Ctrl+V to paste
            subprocess.run(
                ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
                timeout=2,
            )

            # Wait a bit then restore clipboard
            time.sleep(0.2)
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
            # Use wl-copy for clipboard
            proc = subprocess.Popen(
                ["wl-copy"],
                stdin=subprocess.PIPE,
            )
            proc.communicate(input=text.encode("utf-8"))

            time.sleep(self.delay_ms / 1000.0)

            # Simulate Ctrl+V
            subprocess.run(["wtype", "-M", "ctrl", "-P", "v", "-p", "v", "-m", "ctrl"], timeout=2)

            logger.debug(f"Injected {len(text)} chars via wl-copy")
            return True

        except FileNotFoundError:
            logger.error("wl-copy or wtype not found - install wl-clipboard and wtype")
            return False
        except Exception as e:
            logger.error(f"Wayland text injection failed: {e}")
            return False

"""Global hotkey management using pynput."""

import logging
import threading
from typing import Callable

from pynput import keyboard

logger = logging.getLogger(__name__)

# Map config key names to pynput keys
KEY_MAP = {
    "ctrl_l": keyboard.Key.ctrl_l,
    "ctrl_r": keyboard.Key.ctrl_r,
    "ctrl": keyboard.Key.ctrl_l,
    "alt_l": keyboard.Key.alt_l,
    "alt_r": keyboard.Key.alt_r,
    "alt": keyboard.Key.alt_l,
    "shift_l": keyboard.Key.shift_l,
    "shift_r": keyboard.Key.shift_r,
    "shift": keyboard.Key.shift_l,
    "super": keyboard.Key.cmd,
    "space": keyboard.Key.space,
    "fn": keyboard.Key.f13,  # Fn mapped to F13 on some systems
}


def _parse_key(key_str: str):
    """Parse a key string to a pynput key."""
    key_str = key_str.strip().lower()
    if key_str in KEY_MAP:
        return KEY_MAP[key_str]
    if len(key_str) == 1:
        return keyboard.KeyCode.from_char(key_str)
    return None


def parse_hotkey(hotkey_str: str) -> set:
    """Parse a hotkey string like 'ctrl_l+space' into a set of keys."""
    parts = hotkey_str.split("+")
    keys = set()
    for part in parts:
        key = _parse_key(part)
        if key:
            keys.add(key)
        else:
            logger.warning(f"Unknown key: {part}")
    return keys


class HotkeyManager:
    """Manages global hotkeys for hold-to-talk and command mode."""

    def __init__(
        self,
        hold_key: str = "ctrl_l+space",
        command_key: str = "ctrl_l+shift+space",
        whisper_mode_key: str = "ctrl_l+alt+w",
        on_hold_start: Callable | None = None,
        on_hold_end: Callable | None = None,
        on_command_start: Callable | None = None,
        on_command_end: Callable | None = None,
        on_whisper_toggle: Callable | None = None,
    ):
        self._hold_keys = parse_hotkey(hold_key)
        self._command_keys = parse_hotkey(command_key)
        self._whisper_keys = parse_hotkey(whisper_mode_key)

        self.on_hold_start = on_hold_start
        self.on_hold_end = on_hold_end
        self.on_command_start = on_command_start
        self.on_command_end = on_command_end
        self.on_whisper_toggle = on_whisper_toggle

        self._pressed_keys: set = set()
        self._hold_active = False
        self._command_active = False
        self._listener: keyboard.Listener | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start listening for hotkeys."""
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        logger.info(f"Hotkey listener started (hold={self._hold_keys}, cmd={self._command_keys})")

    def stop(self) -> None:
        """Stop listening for hotkeys."""
        if self._listener:
            self._listener.stop()
            self._listener = None
            logger.info("Hotkey listener stopped")

    def _normalize_key(self, key):
        """Normalize a key for comparison."""
        if hasattr(key, "char"):
            return key
        return key

    def _on_press(self, key):
        with self._lock:
            self._pressed_keys.add(self._normalize_key(key))

            # Check whisper mode toggle (requires all keys pressed simultaneously)
            if self._whisper_keys and self._whisper_keys.issubset(self._pressed_keys):
                if self.on_whisper_toggle:
                    threading.Thread(target=self.on_whisper_toggle, daemon=True).start()
                return

            # Check command mode
            if self._command_keys and self._command_keys.issubset(self._pressed_keys):
                if not self._command_active:
                    self._command_active = True
                    if self.on_command_start:
                        threading.Thread(target=self.on_command_start, daemon=True).start()
                return

            # Check hold-to-talk
            if self._hold_keys.issubset(self._pressed_keys):
                if not self._hold_active and not self._command_active:
                    self._hold_active = True
                    if self.on_hold_start:
                        threading.Thread(target=self.on_hold_start, daemon=True).start()

    def _on_release(self, key):
        with self._lock:
            normalized = self._normalize_key(key)

            # Check if releasing breaks a held combo
            if self._command_active and not self._command_keys.issubset(self._pressed_keys - {normalized}):
                self._command_active = False
                if self.on_command_end:
                    threading.Thread(target=self.on_command_end, daemon=True).start()
            elif self._hold_active and not self._hold_keys.issubset(self._pressed_keys - {normalized}):
                self._hold_active = False
                if self.on_hold_end:
                    threading.Thread(target=self.on_hold_end, daemon=True).start()

            self._pressed_keys.discard(normalized)

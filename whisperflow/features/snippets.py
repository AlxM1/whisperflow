"""Voice shortcuts / snippets for frequently used text."""

import logging
import re

from ..config import load_json_file, save_json_file

logger = logging.getLogger(__name__)


class SnippetManager:
    """Manages voice shortcuts that expand trigger phrases into saved text.

    Example: "insert my calendar link" -> "https://calendly.com/username"
    """

    def __init__(self, path: str):
        self.path = path
        self._snippets: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        data = load_json_file(self.path)
        self._snippets = data.get("snippets", {})
        logger.info(f"Loaded {len(self._snippets)} snippets")

    def save(self) -> None:
        save_json_file(self.path, {"snippets": self._snippets})

    def add_snippet(self, trigger: str, content: str, description: str = "") -> None:
        """Add or update a voice snippet.

        Args:
            trigger: The voice trigger phrase (e.g., "my calendar link")
            content: The text to insert
            description: Optional description
        """
        self._snippets[trigger.lower()] = {
            "content": content,
            "description": description,
        }
        self.save()
        logger.info(f"Added snippet: '{trigger}' -> '{content[:50]}...'")

    def remove_snippet(self, trigger: str) -> None:
        """Remove a snippet."""
        key = trigger.lower()
        if key in self._snippets:
            del self._snippets[key]
            self.save()

    def get_snippets(self) -> dict[str, dict]:
        """Get all snippets."""
        return self._snippets.copy()

    def check_for_snippet(self, text: str) -> str | None:
        """Check if text contains a snippet trigger and return the expanded content.

        Returns the snippet content if a trigger is found, None otherwise.
        The trigger can appear with common prefixes like "insert", "paste", "add".
        """
        text_lower = text.lower().strip()

        # Direct match
        if text_lower in self._snippets:
            return self._snippets[text_lower]["content"]

        # Match with common prefixes
        prefixes = ["insert ", "paste ", "add ", "type ", "put ", "enter "]
        for prefix in prefixes:
            if text_lower.startswith(prefix):
                remainder = text_lower[len(prefix):]
                if remainder in self._snippets:
                    return self._snippets[remainder]["content"]

        # Fuzzy match - check if any trigger is contained in the text
        for trigger, data in self._snippets.items():
            if trigger in text_lower:
                return data["content"]

        return None

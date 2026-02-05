"""Personal dictionary for custom words, names, and technical terms."""

import logging
from pathlib import Path

from ..config import load_json_file, save_json_file

logger = logging.getLogger(__name__)


class PersonalDictionary:
    """Manages a personal dictionary of custom words for improved transcription.

    Words in the dictionary are fed to Whisper as initial_prompt hints,
    which significantly improves recognition of custom terms.
    """

    def __init__(self, path: str):
        self.path = path
        self._words: dict[str, str] = {}  # word -> category
        self._load()

    def _load(self) -> None:
        data = load_json_file(self.path)
        self._words = data.get("words", {})
        logger.info(f"Loaded {len(self._words)} dictionary words")

    def save(self) -> None:
        save_json_file(self.path, {"words": self._words})

    def add_word(self, word: str, category: str = "general") -> None:
        """Add a word to the dictionary."""
        self._words[word] = category
        self.save()
        logger.info(f"Added to dictionary: '{word}' ({category})")

    def remove_word(self, word: str) -> None:
        """Remove a word from the dictionary."""
        if word in self._words:
            del self._words[word]
            self.save()
            logger.info(f"Removed from dictionary: '{word}'")

    def get_words(self) -> dict[str, str]:
        """Get all dictionary words."""
        return self._words.copy()

    def get_whisper_prompt(self) -> str:
        """Build an initial_prompt string for Whisper from dictionary words.

        Whisper uses the initial_prompt to bias its recognition toward
        these words when they appear in speech.
        """
        if not self._words:
            return ""
        # Include words as a comma-separated list in a natural context
        words = list(self._words.keys())
        return "Context words: " + ", ".join(words) + "."

    def check_and_learn(self, original: str, corrected: str) -> list[str]:
        """Compare original transcription with corrected text to learn new words.

        Detects both new words and capitalization corrections.
        Returns list of newly learned words.
        """
        orig_tokens = original.split()
        corr_tokens = corrected.split()
        learned = []

        # Build map of lowered original words -> their forms
        orig_forms = {}
        for token in orig_tokens:
            orig_forms[token.lower()] = token

        for cword in corr_tokens:
            low = cword.lower()
            if low in self._words or cword in self._words:
                continue

            is_new_word = low not in orig_forms
            is_capitalization_change = (
                not is_new_word and orig_forms.get(low, "") != cword
            )

            if is_new_word or is_capitalization_change:
                if (
                    cword[0].isupper()  # Proper noun
                    or any(c in cword for c in '-_.')  # Technical term
                    or any(c.isupper() for c in cword[1:])  # camelCase
                ):
                    self.add_word(cword, "auto-learned")
                    learned.append(cword)

        return learned

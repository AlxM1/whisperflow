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

        Returns list of newly learned words.
        """
        orig_words = set(original.lower().split())
        corr_words = set(corrected.lower().split())

        # Words in corrected but not original might be corrections
        new_words = corr_words - orig_words
        learned = []

        for word in new_words:
            # Only learn words that look like proper nouns or technical terms
            # (capitalized in original corrected text, or contain special chars)
            for cword in corrected.split():
                if cword.lower() == word:
                    if (
                        cword[0].isupper()  # Proper noun
                        or any(c in cword for c in '-_.')  # Technical term
                        or any(c.isupper() for c in cword[1:])  # camelCase
                    ) and word not in self._words:
                        self.add_word(cword, "auto-learned")
                        learned.append(cword)
                    break

        return learned

"""Auto-editing: filler removal, punctuation, grammar, course correction."""

import logging
import re

from ..llm.base import LLMProvider
from .prompts import AUTO_EDIT_PROMPT

logger = logging.getLogger(__name__)


class AutoEditor:
    """Cleans up raw transcription text using LLM processing."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def process(self, raw_text: str) -> str:
        """Clean up raw transcription into polished text.

        Applies:
        - Filler word removal
        - Punctuation and capitalization
        - Course correction
        - Grammar fixes
        """
        if not raw_text.strip():
            return ""

        # For very short text (< 5 words), do basic cleanup without LLM
        words = raw_text.split()
        if len(words) <= 3:
            return self._basic_cleanup(raw_text)

        response = self.llm.complete(AUTO_EDIT_PROMPT, raw_text)

        if response.text:
            return response.text
        else:
            # Fallback to basic cleanup if LLM fails
            logger.warning("LLM auto-edit failed, using basic cleanup")
            return self._basic_cleanup(raw_text)

    def _basic_cleanup(self, text: str) -> str:
        """Basic text cleanup without LLM (fallback)."""
        # Remove common fillers
        fillers = [
            r'\bum\b', r'\buh\b', r'\bah\b', r'\ber\b',
            r'\byou know\b', r'\bI mean\b', r'\bbasically\b',
            r'\bliterally\b', r'\blike\s+(?=\w)',
        ]
        result = text
        for filler in fillers:
            result = re.sub(filler, '', result, flags=re.IGNORECASE)

        # Clean up whitespace
        result = re.sub(r'\s+', ' ', result).strip()

        # Capitalize first letter
        if result:
            result = result[0].upper() + result[1:]

        # Add period if missing
        if result and result[-1] not in '.!?':
            result += '.'

        return result

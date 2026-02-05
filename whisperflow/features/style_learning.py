"""Style learning: learns user's writing patterns over time."""

import json
import logging
from pathlib import Path

from ..config import load_json_file, save_json_file
from ..llm.base import LLMProvider

logger = logging.getLogger(__name__)


STYLE_LEARNING_PROMPT = """\
You are analyzing a user's writing corrections to learn their style preferences.

Given the original text (what was generated) and the corrected text (what the user changed it to), identify the style patterns:
- Preferred vocabulary choices
- Sentence length preferences
- Formality level
- Punctuation habits
- Any consistent changes

Output a brief JSON summary of observed patterns:
{
  "vocabulary": ["preferred word -> instead of word"],
  "formality": "casual/neutral/formal",
  "sentence_style": "short/medium/long",
  "notes": "any other patterns"
}
"""


class StyleLearner:
    """Learns and applies user's writing style from corrections.

    Stores examples of corrections the user makes to generated text
    and uses them as few-shot examples to guide future outputs.
    """

    def __init__(self, path: str, max_examples: int = 200):
        self.path = path
        self.max_examples = max_examples
        self._examples: list[dict] = []
        self._style_summary: dict = {}
        self._load()

    def _load(self) -> None:
        data = load_json_file(self.path)
        self._examples = data.get("examples", [])
        self._style_summary = data.get("style_summary", {})
        logger.info(f"Loaded {len(self._examples)} style examples")

    def save(self) -> None:
        save_json_file(self.path, {
            "examples": self._examples,
            "style_summary": self._style_summary,
        })

    def record_correction(self, original: str, corrected: str) -> None:
        """Record a user correction for style learning.

        Args:
            original: The auto-edited text that was generated
            corrected: What the user changed it to
        """
        if original.strip() == corrected.strip():
            return  # No actual correction

        self._examples.append({
            "original": original,
            "corrected": corrected,
        })

        # Keep within limit
        if len(self._examples) > self.max_examples:
            self._examples = self._examples[-self.max_examples:]

        self.save()
        logger.info("Recorded style correction")

    def analyze_style(self, llm: LLMProvider) -> dict:
        """Use LLM to analyze collected corrections and extract style patterns."""
        if len(self._examples) < 3:
            return {}

        # Send recent examples for analysis
        recent = self._examples[-20:]
        examples_text = "\n\n".join(
            f"Original: {ex['original']}\nCorrected: {ex['corrected']}"
            for ex in recent
        )

        response = llm.complete(STYLE_LEARNING_PROMPT, examples_text)

        try:
            self._style_summary = json.loads(response.text)
            self.save()
            return self._style_summary
        except (json.JSONDecodeError, Exception):
            logger.debug("Could not parse style analysis as JSON")
            return {}

    def get_style_prompt_addition(self) -> str:
        """Get a prompt addition based on learned style preferences.

        Returns additional instructions to append to the auto-edit prompt.
        """
        if not self._examples:
            return ""

        lines = ["\n\nUser style preferences (learned from their corrections):"]

        if self._style_summary:
            if vocab := self._style_summary.get("vocabulary"):
                lines.append(f"- Preferred vocabulary: {', '.join(vocab[:10])}")
            if formality := self._style_summary.get("formality"):
                lines.append(f"- Formality level: {formality}")
            if style := self._style_summary.get("sentence_style"):
                lines.append(f"- Sentence style: {style}")
            if notes := self._style_summary.get("notes"):
                lines.append(f"- Notes: {notes}")

        # Include a few recent examples as few-shot guidance
        recent = self._examples[-5:]
        if recent:
            lines.append("\nRecent correction examples (match this style):")
            for ex in recent:
                lines.append(f"  Before: {ex['original'][:100]}")
                lines.append(f"  After:  {ex['corrected'][:100]}")

        return "\n".join(lines)

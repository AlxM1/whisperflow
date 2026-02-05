"""Tone matching: adjusts text formality based on active application."""

import logging
from typing import Any

from ..llm.base import LLMProvider
from ..input.context import WindowContext
from .prompts import TONE_PROMPTS, CONTEXT_TONE_MAP, AUTO_EDIT_PROMPT

logger = logging.getLogger(__name__)


class ToneMatcher:
    """Adjusts transcription tone and formatting based on app context."""

    def __init__(self, llm: LLMProvider, app_tones: dict[str, str] | None = None):
        self.llm = llm
        # User-configurable app tone overrides
        self.app_tones = app_tones or {}

    def get_tone(self, context: WindowContext) -> str:
        """Determine the appropriate tone for the current context."""
        detected_app = context.detected_app

        # Check user overrides first
        for app_pattern, tone in self.app_tones.items():
            if app_pattern.lower() in detected_app.lower():
                return tone

        # Fall back to default context map
        return CONTEXT_TONE_MAP.get(detected_app, "professional")

    def apply_tone(self, text: str, context: WindowContext) -> str:
        """Apply tone-appropriate formatting to text.

        Args:
            text: Already auto-edited text
            context: Active window context

        Returns:
            Text with appropriate tone applied
        """
        tone = self.get_tone(context)

        # For neutral/basic text, the auto-editor already did enough
        if tone not in TONE_PROMPTS:
            return text

        tone_instruction = TONE_PROMPTS[tone]

        system_prompt = (
            f"{tone_instruction}\n\n"
            "Rules:\n"
            "- Output ONLY the formatted text, no explanations\n"
            "- Do NOT change the core meaning\n"
            "- Do NOT add information that isn't present\n"
            "- Keep the text natural, not robotic\n"
            "- If the text is already in the right tone, return it unchanged"
        )

        response = self.llm.complete(system_prompt, text)

        if response.text:
            return response.text
        return text

    def build_combined_prompt(self, context: WindowContext) -> str:
        """Build a combined system prompt for auto-edit + tone matching in one LLM call.

        This is more efficient than two separate calls.
        """
        tone = self.get_tone(context)
        tone_instruction = TONE_PROMPTS.get(tone, "")

        if tone_instruction:
            return f"{AUTO_EDIT_PROMPT}\n\nAdditional tone guidance:\n{tone_instruction}"
        return AUTO_EDIT_PROMPT

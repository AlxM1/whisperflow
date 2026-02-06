"""Main text processing pipeline - orchestrates all processing steps."""

import logging
from dataclasses import dataclass

from ..llm.base import LLMProvider
from ..input.context import WindowContext, get_active_window, get_selected_text
from ..features.dictionary import PersonalDictionary
from ..features.snippets import SnippetManager
from ..features.style_learning import StyleLearner
from .auto_edit import AutoEditor
from .command_mode import CommandMode
from .tone import ToneMatcher

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from the processing pipeline."""
    original_text: str
    processed_text: str
    mode: str  # "dictation", "command", "question", "snippet"
    context: WindowContext
    language: str
    duration: float


class ProcessingPipeline:
    """Main pipeline that orchestrates all text processing.

    Flow:
    1. Raw transcription comes in
    2. Check for snippet triggers
    3. In command mode: classify intent and execute
    4. In dictation mode:
       a. Auto-edit (filler removal, punctuation, course correction)
       b. Apply tone matching based on active app
       c. Apply style preferences
    """

    def __init__(
        self,
        llm: LLMProvider,
        dictionary: PersonalDictionary,
        snippets: SnippetManager,
        style_learner: StyleLearner,
        config: dict,
    ):
        self.llm = llm
        self.dictionary = dictionary
        self.snippets = snippets
        self.style_learner = style_learner
        self.config = config

        self.auto_editor = AutoEditor(llm)
        self.command_mode = CommandMode(llm)
        self.tone_matcher = ToneMatcher(
            llm,
            app_tones=config.get("context", {}).get("app_tones", {}),
        )

        self._auto_edit_enabled = config.get("processing", {}).get("auto_edit", True)
        self._context_aware = config.get("processing", {}).get("context_aware", True)
        self._tone_matching = config.get("processing", {}).get("tone_matching", True)

    def process_dictation(
        self,
        raw_text: str,
        language: str = "",
        duration: float = 0.0,
    ) -> PipelineResult:
        """Process a dictation transcription through the full pipeline.

        Args:
            raw_text: Raw transcription from Whisper
            language: Detected language
            duration: Audio duration in seconds

        Returns:
            PipelineResult with processed text and metadata
        """
        # Get active window context
        context = WindowContext("", "", "")
        if self._context_aware:
            context = get_active_window()
            logger.info(f"Context: {context.detected_app} ({context.app_name})")

        # Step 1: Check for snippet triggers
        snippet_result = self.snippets.check_for_snippet(raw_text)
        if snippet_result is not None:
            logger.info(f"Snippet triggered: '{raw_text}' -> '{snippet_result[:50]}'")
            return PipelineResult(
                original_text=raw_text,
                processed_text=snippet_result,
                mode="snippet",
                context=context,
                language=language,
                duration=duration,
            )

        # Step 2: Auto-edit (with optional tone + style in single LLM call)
        if self._auto_edit_enabled:
            # Build the combined prompt: auto-edit base + tone + style
            if self._tone_matching and context.app_name:
                prompt = self.tone_matcher.build_combined_prompt(context)
            else:
                from .prompts import AUTO_EDIT_PROMPT
                prompt = AUTO_EDIT_PROMPT

            # Append style hints if available (single LLM call, not a second one)
            style_addition = self.style_learner.get_style_prompt_addition()
            if style_addition and len(raw_text.split()) > 5:
                prompt += style_addition

            response = self.llm.complete(prompt, raw_text)
            processed = response.text if response.text else raw_text
        elif self._tone_matching and context.app_name:
            # Tone matching still applies even when auto-edit is off
            processed = self.tone_matcher.apply_tone(raw_text, context)
        else:
            processed = raw_text

        return PipelineResult(
            original_text=raw_text,
            processed_text=processed,
            mode="dictation",
            context=context,
            language=language,
            duration=duration,
        )

    def process_command(
        self,
        raw_text: str,
        language: str = "",
        duration: float = 0.0,
    ) -> PipelineResult:
        """Process a command mode transcription.

        In command mode, the spoken text is interpreted as a command to
        edit currently selected text, or as a question for the AI assistant.
        """
        context = get_active_window() if self._context_aware else WindowContext("", "", "")

        # Classify the intent
        intent = self.command_mode.classify_intent(raw_text)
        logger.info(f"Command mode intent: {intent}")

        if intent == "question":
            answer = self.command_mode.answer_question(raw_text)
            return PipelineResult(
                original_text=raw_text,
                processed_text=answer,
                mode="question",
                context=context,
                language=language,
                duration=duration,
            )

        if intent == "command":
            # Get selected text from the active window
            selected = get_selected_text()
            if selected:
                result = self.command_mode.execute_command(raw_text, selected)
                return PipelineResult(
                    original_text=raw_text,
                    processed_text=result,
                    mode="command",
                    context=context,
                    language=language,
                    duration=duration,
                )
            else:
                logger.info("Command mode but no text selected, treating as dictation")

        # Fall back to dictation processing
        return self.process_dictation(raw_text, language, duration)

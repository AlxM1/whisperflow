"""Command Mode: voice commands to edit existing text."""

import logging

from ..llm.base import LLMProvider
from .prompts import COMMAND_DETECTION_PROMPT, COMMAND_EXECUTION_PROMPT, AI_ASSISTANT_PROMPT

logger = logging.getLogger(__name__)

# Keywords that strongly indicate a command (even without selected text)
COMMAND_KEYWORDS = [
    "translate", "summarize", "summarise", "rewrite", "rephrase",
    "make this", "make it", "turn this", "convert this",
    "fix the", "fix this", "correct the", "correct this",
    "bold", "italic", "bullet", "bullets", "numbered list",
    "shorten", "expand", "simplify", "formalize",
    "delete", "remove",
]

# Keywords indicating AI assistant query
ASSISTANT_KEYWORDS = [
    "what is", "what are", "what's", "how do", "how does", "how to",
    "why is", "why does", "when did", "when does", "where is",
    "tell me", "explain", "define", "look up", "search for",
    "calculate", "what time", "who is", "who was",
]


class CommandMode:
    """Handles voice commands to edit text or query AI."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def classify_intent(self, text: str) -> str:
        """Classify if the spoken text is a command, dictation, or question.

        Returns: 'command', 'dictation', or 'question'
        """
        text_lower = text.lower().strip()

        # Check for explicit assistant queries
        for kw in ASSISTANT_KEYWORDS:
            if text_lower.startswith(kw):
                return "question"

        # Check for explicit command keywords
        for kw in COMMAND_KEYWORDS:
            if kw in text_lower:
                return "command"

        # Use LLM for ambiguous cases
        response = self.llm.complete(COMMAND_DETECTION_PROMPT, text)
        result = response.text.strip().lower()

        if "command" in result:
            return "command"
        return "dictation"

    def execute_command(self, command: str, selected_text: str) -> str:
        """Execute a voice command on the selected text.

        Args:
            command: The voice command (e.g., "make this more formal")
            selected_text: The text to modify

        Returns:
            The modified text
        """
        if not selected_text:
            logger.warning("Command mode activated but no text selected")
            return ""

        prompt = f"Command: {command}\n\nText to modify:\n{selected_text}"
        response = self.llm.complete(COMMAND_EXECUTION_PROMPT, prompt)

        if response.text:
            logger.info(f"Command executed: '{command}' on {len(selected_text)} chars")
            return response.text
        else:
            logger.warning("Command execution failed")
            return selected_text

    def answer_question(self, question: str) -> str:
        """Answer a question using the AI assistant.

        Args:
            question: The question to answer

        Returns:
            The answer text
        """
        response = self.llm.complete(AI_ASSISTANT_PROMPT, question)
        if response.text:
            return response.text
        return ""

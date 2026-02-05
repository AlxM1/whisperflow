"""Base interface for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from an LLM."""
    text: str
    model: str
    usage_tokens: int = 0


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(self, system_prompt: str, user_message: str) -> LLMResponse:
        """Generate a completion.

        Args:
            system_prompt: System instructions
            user_message: User message to process

        Returns:
            LLMResponse with the generated text
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available."""
        ...

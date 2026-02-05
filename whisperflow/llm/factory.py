"""Factory for creating LLM providers from config."""

import logging
from typing import Any

from .base import LLMProvider
from .ollama import OllamaProvider
from .openai_api import OpenAIProvider
from .anthropic_api import AnthropicProvider

logger = logging.getLogger(__name__)


class NoopProvider(LLMProvider):
    """No-op LLM provider that passes through text unchanged."""

    def complete(self, system_prompt: str, user_message: str) -> Any:
        from .base import LLMResponse
        return LLMResponse(text=user_message, model="noop")

    def is_available(self) -> bool:
        return True


def create_llm_provider(config: dict) -> LLMProvider:
    """Create an LLM provider from configuration."""
    provider_name = config.get("provider", "none")

    if provider_name == "ollama":
        ollama_cfg = config.get("ollama", {})
        provider = OllamaProvider(
            url=ollama_cfg.get("url", "http://localhost:11434"),
            model=ollama_cfg.get("model", "llama3.1:8b"),
        )
    elif provider_name == "openai":
        openai_cfg = config.get("openai", {})
        provider = OpenAIProvider(
            api_key=openai_cfg.get("api_key", ""),
            model=openai_cfg.get("model", "gpt-4o-mini"),
        )
    elif provider_name == "anthropic":
        anthropic_cfg = config.get("anthropic", {})
        provider = AnthropicProvider(
            api_key=anthropic_cfg.get("api_key", ""),
            model=anthropic_cfg.get("model", "claude-sonnet-4-20250514"),
        )
    else:
        logger.info("No LLM provider configured, using passthrough")
        return NoopProvider()

    if provider.is_available():
        logger.info(f"LLM provider: {provider_name}")
        return provider
    else:
        logger.warning(f"LLM provider '{provider_name}' not available, falling back to passthrough")
        return NoopProvider()

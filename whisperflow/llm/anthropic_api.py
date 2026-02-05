"""Anthropic API LLM provider (BYOK - Bring Your Own Key)."""

import logging

from .base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """LLM provider using Anthropic API with user's own API key."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
        return self._client

    def complete(self, system_prompt: str, user_message: str) -> LLMResponse:
        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            text = response.content[0].text if response.content else ""
            tokens = (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0
            return LLMResponse(text=text.strip(), model=self.model, usage_tokens=tokens)
        except Exception as e:
            logger.error(f"Anthropic request failed: {e}")
            return LLMResponse(text="", model=self.model)

    def is_available(self) -> bool:
        return bool(self.api_key)

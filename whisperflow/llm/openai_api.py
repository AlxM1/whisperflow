"""OpenAI API LLM provider (BYOK - Bring Your Own Key)."""

import logging

from .base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """LLM provider using OpenAI API with user's own API key."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def complete(self, system_prompt: str, user_message: str) -> LLMResponse:
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            text = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0
            return LLMResponse(text=text.strip(), model=self.model, usage_tokens=tokens)
        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
            return LLMResponse(text="", model=self.model)

    def is_available(self) -> bool:
        return bool(self.api_key)

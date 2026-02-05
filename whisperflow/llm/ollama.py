"""Ollama LLM provider for local inference."""

import logging

import requests

from .base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """LLM provider using Ollama for local model inference."""

    def __init__(self, url: str = "http://localhost:11434", model: str = "llama3.1:8b"):
        self.url = url.rstrip("/")
        self.model = model

    def complete(self, system_prompt: str, user_message: str) -> LLMResponse:
        try:
            resp = requests.post(
                f"{self.url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 2048,
                    },
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            tokens = data.get("eval_count", 0)
            return LLMResponse(text=content.strip(), model=self.model, usage_tokens=tokens)
        except requests.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            return LLMResponse(text="", model=self.model)

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.url}/api/tags", timeout=3)
            return resp.status_code == 200
        except requests.RequestException:
            return False

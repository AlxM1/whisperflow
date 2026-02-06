"""Ollama LLM provider for local inference."""

import logging

import requests

from .base import LLMProvider, LLMResponse
from ..retry import retry_with_backoff

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """LLM provider using Ollama for local model inference."""

    def __init__(self, url: str = "http://localhost:11434", model: str = "llama3.1:8b"):
        self.url = url.rstrip("/")
        self.model = model

    def complete(self, system_prompt: str, user_message: str) -> LLMResponse:
        def _do_request():
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
            return resp

        try:
            resp = retry_with_backoff(
                _do_request,
                max_retries=2,
                base_delay=1.0,
                exceptions=(requests.ConnectionError, requests.Timeout),
            )
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
            if resp.status_code != 200:
                return False
            # Verify the configured model is actually available
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            if self.model not in model_names:
                base_names = [n.split(":")[0] for n in model_names]
                model_base = self.model.split(":")[0]
                if model_base not in base_names:
                    logger.warning(
                        f"Ollama model '{self.model}' not found. "
                        f"Available: {', '.join(model_names) or 'none'}"
                    )
                    return False
            return True
        except requests.RequestException:
            return False

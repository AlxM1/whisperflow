"""Remote speech-to-text via GPU server API."""

import logging
import time

import requests

from .base import TranscriptionBackend, TranscriptionResult
from ..retry import retry_with_backoff

logger = logging.getLogger(__name__)


class RemoteWhisperBackend(TranscriptionBackend):
    """Transcription via remote GPU server (e.g., RTX 5090 on Windows PC)."""

    def __init__(self, server_url: str = "http://192.168.1.100:8765", timeout: int = 60):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def transcribe(
        self,
        audio_data: bytes,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        url = f"{self.server_url}/transcribe"

        files = {"audio": ("recording.wav", audio_data, "audio/wav")}
        data = {}
        if language and language != "auto":
            data["language"] = language
        if initial_prompt:
            data["initial_prompt"] = initial_prompt

        start = time.time()

        def _do_request():
            resp = requests.post(url, files=files, data=data, timeout=self.timeout)
            resp.raise_for_status()
            return resp

        try:
            resp = retry_with_backoff(
                _do_request,
                max_retries=2,
                base_delay=1.0,
                exceptions=(requests.ConnectionError, requests.Timeout),
            )
        except requests.RequestException as e:
            logger.error(f"Remote transcription failed: {e}")
            return TranscriptionResult(text="", language="", confidence=0.0, duration=0.0)

        result = resp.json()
        elapsed = time.time() - start
        logger.info(
            f"Remote transcription completed in {elapsed:.1f}s "
            f"(lang={result.get('language', '?')})"
        )

        return TranscriptionResult(
            text=result.get("text", ""),
            language=result.get("language", ""),
            confidence=result.get("confidence", 0.0),
            duration=result.get("duration", 0.0),
        )

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

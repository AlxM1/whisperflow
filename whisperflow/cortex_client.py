"""Cortex task activity reporting client.

Reports WhisperFlow task activity to Cortex for system monitoring and analytics.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)


class CortexClient:
    """Client for reporting task activity to Cortex."""

    def __init__(
        self,
        cortex_url: str = "http://cortex:3011",
        service_name: str = "whisperflow",
        enabled: bool = True,
        timeout: int = 5,
    ):
        """Initialize Cortex client.

        Args:
            cortex_url: Base URL of Cortex service
            service_name: Service identifier for task reports
            enabled: Whether Cortex reporting is enabled
            timeout: HTTP request timeout in seconds
        """
        self.cortex_url = cortex_url.rstrip("/")
        self.service_name = service_name
        self.enabled = enabled
        self.timeout = timeout
        self._session = None

    def _get_session(self) -> requests.Session:
        """Get or create HTTP session with connection pooling."""
        if self._session is None:
            self._session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=2,
                pool_maxsize=5,
                max_retries=0,
            )
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
        return self._session

    def _report_async(self, endpoint: str, payload: dict) -> None:
        """Report task in background thread (non-blocking)."""
        if not self.enabled:
            return

        def _do_report():
            try:
                url = f"{self.cortex_url}{endpoint}"
                session = self._get_session()
                response = session.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"},
                )
                if response.status_code not in (200, 201, 204):
                    logger.warning(
                        f"Cortex report failed: {response.status_code} {response.text[:100]}"
                    )
            except requests.Timeout:
                logger.debug(f"Cortex report timeout (service may be unavailable)")
            except requests.RequestException as e:
                logger.debug(f"Cortex report failed: {e}")
            except Exception:
                logger.exception("Unexpected error reporting to Cortex")

        thread = threading.Thread(target=_do_report, daemon=True)
        thread.start()

    def report_transcription(
        self,
        mode: str,
        duration: float,
        language: str = "",
        backend: str = "local",
        success: bool = True,
        error: str = "",
        metadata: dict | None = None,
    ) -> None:
        """Report a transcription task to Cortex.

        Args:
            mode: Transcription mode (dictation, command, question, transcribe)
            duration: Audio duration in seconds
            language: Detected language (ISO code)
            backend: Transcription backend used (local, remote)
            success: Whether transcription succeeded
            error: Error message if failed
            metadata: Additional metadata (model, processing_time, etc.)
        """
        payload = {
            "service": self.service_name,
            "task_type": f"transcription.{mode}",
            "timestamp": datetime.utcnow().isoformat(),
            "status": "completed" if success else "failed",
            "metrics": {
                "audio_duration_seconds": round(duration, 2),
                "language": language or "unknown",
                "backend": backend,
            },
            "metadata": metadata or {},
        }

        if error:
            payload["error"] = error

        self._report_async("/api/tasks", payload)

    def report_llm_processing(
        self,
        operation: str,
        model: str,
        input_length: int,
        output_length: int = 0,
        processing_time: float = 0.0,
        success: bool = True,
        error: str = "",
    ) -> None:
        """Report LLM processing task to Cortex.

        Args:
            operation: LLM operation (auto_edit, tone_match, command, question)
            model: LLM model name
            input_length: Input text length in characters
            output_length: Output text length in characters
            processing_time: Processing time in seconds
            success: Whether processing succeeded
            error: Error message if failed
        """
        payload = {
            "service": self.service_name,
            "task_type": f"llm.{operation}",
            "timestamp": datetime.utcnow().isoformat(),
            "status": "completed" if success else "failed",
            "metrics": {
                "model": model,
                "input_length": input_length,
                "output_length": output_length,
                "processing_time_seconds": round(processing_time, 3),
            },
        }

        if error:
            payload["error"] = error

        self._report_async("/api/tasks", payload)

    def report_health_check(
        self,
        component: str,
        healthy: bool,
        latency_ms: float = 0.0,
        details: dict | None = None,
    ) -> None:
        """Report component health status to Cortex.

        Args:
            component: Component name (transcriber, llm, vad, etc.)
            healthy: Whether component is healthy
            latency_ms: Component latency in milliseconds
            details: Additional health check details
        """
        payload = {
            "service": self.service_name,
            "task_type": f"health.{component}",
            "timestamp": datetime.utcnow().isoformat(),
            "status": "healthy" if healthy else "unhealthy",
            "metrics": {
                "latency_ms": round(latency_ms, 1),
            },
            "metadata": details or {},
        }

        self._report_async("/api/tasks", payload)

    def report_error(
        self,
        error_type: str,
        message: str,
        stack_trace: str = "",
        context: dict | None = None,
    ) -> None:
        """Report an error to Cortex.

        Args:
            error_type: Error type/category
            message: Error message
            stack_trace: Full stack trace
            context: Contextual information about the error
        """
        payload = {
            "service": self.service_name,
            "task_type": f"error.{error_type}",
            "timestamp": datetime.utcnow().isoformat(),
            "status": "failed",
            "error": message,
            "metadata": {
                "stack_trace": stack_trace,
                **(context or {}),
            },
        }

        self._report_async("/api/tasks", payload)

    def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            self._session.close()
            self._session = None


# Global Cortex client instance (singleton)
_cortex_client: CortexClient | None = None


def get_cortex_client() -> CortexClient:
    """Get or create the global Cortex client instance."""
    global _cortex_client
    if _cortex_client is None:
        cortex_url = os.environ.get("CORTEX_URL", "http://cortex:3011")
        enabled = os.environ.get("CORTEX_ENABLED", "true").lower() in ("true", "1", "yes")
        _cortex_client = CortexClient(
            cortex_url=cortex_url,
            service_name="whisperflow",
            enabled=enabled,
        )
        if enabled:
            logger.info(f"Cortex reporting enabled: {cortex_url}")
        else:
            logger.info("Cortex reporting disabled")
    return _cortex_client

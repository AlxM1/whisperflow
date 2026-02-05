"""Base interface for speech-to-text transcription."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    """Result of a transcription."""
    text: str
    language: str
    confidence: float
    duration: float  # audio duration in seconds

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class TranscriptionBackend(ABC):
    """Abstract base class for transcription backends."""

    @abstractmethod
    def transcribe(
        self,
        audio_data: bytes,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio data to text.

        Args:
            audio_data: WAV audio bytes
            language: Optional language code (None for auto-detect)
            initial_prompt: Optional prompt to guide transcription (for dictionary words)

        Returns:
            TranscriptionResult with the transcribed text
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available and ready."""
        ...

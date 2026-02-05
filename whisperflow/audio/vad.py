"""Voice Activity Detection using Silero VAD."""

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Silero VAD expects 16kHz mono audio in chunks of 512 samples (32ms)
SILERO_CHUNK_SIZE = 512
SILERO_SAMPLE_RATE = 16000


class VoiceActivityDetector:
    """Detects voice activity in audio using Silero VAD."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._model = None
        self._h = None
        self._c = None

    def load(self) -> None:
        """Load the Silero VAD model."""
        logger.info("Loading Silero VAD model...")
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._model.eval()
        self.reset()
        logger.info("Silero VAD model loaded")

    def reset(self) -> None:
        """Reset internal state for new audio stream."""
        if self._model is not None:
            self._model.reset_states()

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Check if an audio chunk contains speech.

        Args:
            audio_chunk: float32 numpy array of audio samples at 16kHz

        Returns:
            True if speech is detected above threshold
        """
        if self._model is None:
            self.load()

        tensor = torch.from_numpy(audio_chunk).float()
        if len(tensor) < SILERO_CHUNK_SIZE:
            tensor = torch.nn.functional.pad(tensor, (0, SILERO_CHUNK_SIZE - len(tensor)))

        confidence = self._model(tensor, SILERO_SAMPLE_RATE).item()
        return confidence > self.threshold

    def get_speech_probability(self, audio_chunk: np.ndarray) -> float:
        """Get the probability that audio contains speech."""
        if self._model is None:
            self.load()

        tensor = torch.from_numpy(audio_chunk).float()
        if len(tensor) < SILERO_CHUNK_SIZE:
            tensor = torch.nn.functional.pad(tensor, (0, SILERO_CHUNK_SIZE - len(tensor)))

        return self._model(tensor, SILERO_SAMPLE_RATE).item()

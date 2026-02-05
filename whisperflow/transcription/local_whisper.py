"""Local speech-to-text using faster-whisper."""

import io
import logging
import time

import numpy as np

from .base import TranscriptionBackend, TranscriptionResult

logger = logging.getLogger(__name__)


class LocalWhisperBackend(TranscriptionBackend):
    """Transcription using faster-whisper running locally."""

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "float16",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load_model(self) -> None:
        """Lazy-load the whisper model."""
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        logger.info(f"Loading faster-whisper model: {self.model_size} (device={self.device})")
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("Model loaded successfully")

    def transcribe(
        self,
        audio_data: bytes,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        self._load_model()

        # Convert WAV bytes to numpy float32
        import wave

        buf = io.BytesIO(audio_data)
        with wave.open(buf, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            sample_rate = wf.getframerate()
            duration = wf.getnframes() / sample_rate

        audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        lang = language if language and language != "auto" else None

        start = time.time()
        segments, info = self._model.transcribe(
            audio_np,
            language=lang,
            initial_prompt=initial_prompt,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text)

        text = " ".join(text_parts).strip()
        elapsed = time.time() - start

        logger.info(
            f"Transcribed {duration:.1f}s audio in {elapsed:.1f}s "
            f"(lang={info.language}, prob={info.language_probability:.2f})"
        )

        return TranscriptionResult(
            text=text,
            language=info.language,
            confidence=info.language_probability,
            duration=duration,
        )

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

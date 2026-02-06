"""Audio capture module - records microphone input for dictation."""

import io
import logging
import threading
import wave
from collections import deque
from typing import Callable

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# Maximum recording buffer: 5 minutes at 16kHz = ~9.6M samples
MAX_BUFFER_FRAMES = 3000  # 3000 * 1600 samples/frame = 300s at 100ms blocks


class AudioCapture:
    """Captures audio from the microphone with hold-to-talk semantics."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "float32",
        on_audio_ready: Callable[[bytes], None] | None = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.on_audio_ready = on_audio_ready

        self._recording = False
        self._frames: deque[np.ndarray] = deque(maxlen=MAX_BUFFER_FRAMES)
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    def start_recording(self) -> None:
        """Begin capturing audio from microphone."""
        with self._lock:
            if self._recording:
                return
            self._frames.clear()
            self._recording = True
            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=self.dtype,
                    callback=self._audio_callback,
                    blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
                )
                self._stream.start()
                logger.info("Recording started")
            except Exception as e:
                self._recording = False
                self._stream = None
                logger.error(f"Failed to start recording: {e}")
                raise

    def stop_recording(self) -> bytes:
        """Stop capturing and return the recorded audio as WAV bytes."""
        with self._lock:
            if not self._recording:
                return b""
            self._recording = False
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            # Snapshot frames under lock to avoid race with _audio_callback
            frames = list(self._frames)
            self._frames.clear()

        audio_data = np.concatenate(frames) if frames else np.array([])

        if len(audio_data) == 0:
            logger.warning("No audio captured")
            return b""

        wav_bytes = self._to_wav(audio_data)
        logger.info(f"Recording stopped, captured {len(audio_data) / self.sample_rate:.1f}s")

        if self.on_audio_ready:
            self.on_audio_ready(wav_bytes)

        return wav_bytes

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def duration(self) -> float:
        """Current recording duration in seconds."""
        if not self._frames:
            return 0.0
        total_samples = sum(len(f) for f in self._frames)
        return total_samples / self.sample_rate

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.warning(f"Audio status: {status}")
        if self._recording:
            # deque with maxlen handles overflow automatically
            self._frames.append(indata[:, 0].copy() if indata.ndim > 1 else indata.copy())

    def _to_wav(self, audio_data: np.ndarray) -> bytes:
        """Convert float32 numpy array to WAV bytes."""
        # Convert float32 [-1, 1] to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return buf.getvalue()

    def get_audio_level(self) -> float:
        """Get current audio input level (RMS) for UI visualization."""
        if not self._frames:
            return 0.0
        recent = self._frames[-1] if self._frames else np.array([0.0])
        return float(np.sqrt(np.mean(recent**2)))

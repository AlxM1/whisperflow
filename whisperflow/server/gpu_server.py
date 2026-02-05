"""GPU inference server for running on Windows PC with RTX 5090.

This server exposes faster-whisper transcription as a REST API.
Run this on the Windows machine with the GPU, and point the
Linux client at it via config.

Usage:
    pip install faster-whisper fastapi uvicorn python-multipart
    python -m whisperflow.server.gpu_server --host 0.0.0.0 --port 8765
"""

import argparse
import io
import logging
import sys
import tempfile
import time
import wave

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_app():
    """Create the FastAPI application."""
    from fastapi import FastAPI, File, Form, UploadFile

    app = FastAPI(title="WhisperFlow GPU Server", version="0.1.0")

    # Lazy-load model on first request
    _state = {"model": None}

    def get_model():
        if _state["model"] is None:
            from faster_whisper import WhisperModel
            logger.info("Loading faster-whisper model (large-v3-turbo, CUDA)...")
            _state["model"] = WhisperModel(
                "large-v3-turbo",
                device="cuda",
                compute_type="float16",
            )
            logger.info("Model loaded successfully")
        return _state["model"]

    @app.get("/health")
    async def health():
        return {"status": "ok", "gpu": True}

    @app.post("/transcribe")
    async def transcribe(
        audio: UploadFile = File(...),
        language: str = Form(default=""),
        initial_prompt: str = Form(default=""),
    ):
        audio_bytes = await audio.read()

        # Parse WAV
        buf = io.BytesIO(audio_bytes)
        with wave.open(buf, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            sample_rate = wf.getframerate()
            duration = wf.getnframes() / sample_rate

        audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        model = get_model()
        lang = language if language else None
        prompt = initial_prompt if initial_prompt else None

        start = time.time()
        segments, info = model.transcribe(
            audio_np,
            language=lang,
            initial_prompt=prompt,
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
            f"(lang={info.language}, speed={duration/elapsed:.1f}x)"
        )

        return {
            "text": text,
            "language": info.language,
            "confidence": info.language_probability,
            "duration": duration,
            "processing_time": elapsed,
        }

    return app


def main():
    parser = argparse.ArgumentParser(description="WhisperFlow GPU Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8765, help="Port number")
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper model size")
    args = parser.parse_args()

    import uvicorn

    app = create_app()
    logger.info(f"Starting GPU server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

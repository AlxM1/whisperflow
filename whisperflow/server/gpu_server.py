"""GPU inference server for running on Windows PC with RTX 5090.

This server exposes faster-whisper transcription as a REST API.
Run this on the Windows machine with the GPU, and point the
Linux client at it via config.

Usage:
    pip install faster-whisper fastapi uvicorn python-multipart
    python -m whisperflow.server.gpu_server --host 0.0.0.0 --port 8765

Authentication:
    Set WHISPERFLOW_API_KEY env var on both server and client.
    If not set, authentication is disabled (LAN-only use).
"""

import argparse
import asyncio
import io
import logging
import os
import secrets
import sys
import time
import wave

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Max upload: 50MB (roughly 10 minutes of 16kHz 16-bit audio)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# Semaphore to serialize model access (one transcription at a time)
_transcribe_semaphore = asyncio.Semaphore(1)


def create_app(api_key: str = "", model_name: str = "large-v3-turbo"):
    """Create the FastAPI application."""
    from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile

    app = FastAPI(title="WhisperFlow GPU Server", version="0.2.0")

    # Lazy-load model on first request
    _state = {"model": None}

    def get_model():
        if _state["model"] is None:
            from faster_whisper import WhisperModel
            logger.info(f"Loading faster-whisper model ({model_name}, CUDA)...")
            _state["model"] = WhisperModel(
                model_name,
                device="cuda",
                compute_type="float16",
            )
            logger.info("Model loaded successfully")
        return _state["model"]

    def _check_auth(authorization: str | None):
        """Verify API key if authentication is enabled."""
        if not api_key:
            return  # Auth disabled
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        # Accept "Bearer <key>" or just "<key>"
        token = authorization.removeprefix("Bearer ").strip()
        if not secrets.compare_digest(token, api_key):
            raise HTTPException(status_code=403, detail="Invalid API key")

    @app.get("/health")
    async def health(authorization: str | None = Header(default=None)):
        _check_auth(authorization)
        return {
            "status": "ok",
            "gpu": True,
            "model": model_name,
            "auth_required": bool(api_key),
        }

    @app.post("/transcribe")
    async def transcribe(
        audio: UploadFile = File(...),
        language: str = Form(default=""),
        initial_prompt: str = Form(default=""),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(authorization)

        audio_bytes = await audio.read()

        # Enforce file size limit
        if len(audio_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file too large ({len(audio_bytes)} bytes, max {MAX_UPLOAD_BYTES})",
            )

        # Parse WAV
        try:
            buf = io.BytesIO(audio_bytes)
            with wave.open(buf, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                sample_rate = wf.getframerate()
                duration = wf.getnframes() / sample_rate
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid WAV file: {e}")

        audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        lang = language if language else None
        prompt = initial_prompt if initial_prompt else None

        # Serialize model access to prevent concurrent GPU usage
        async with _transcribe_semaphore:
            model = get_model()

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
    parser.add_argument(
        "--api-key", default="",
        help="API key for authentication (or set WHISPERFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("WHISPERFLOW_DATA_DIR", ""),
        help="Override data directory for all storage",
    )
    args = parser.parse_args()

    if args.data_dir:
        from whisperflow.config import set_data_dir
        set_data_dir(args.data_dir)

    api_key = args.api_key or os.environ.get("WHISPERFLOW_API_KEY", "")
    if api_key:
        logger.info("Authentication enabled")
    else:
        logger.warning("No API key set — server is open to all network requests")

    import uvicorn

    app = create_app(api_key=api_key, model_name=args.model)
    logger.info(f"Starting GPU server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

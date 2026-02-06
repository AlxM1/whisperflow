"""WhisperFlow Web Server - serves PWA mobile app + REST API.

Allows using WhisperFlow from any device (phone, tablet, remote PC)
by providing a web-based interface that captures audio and processes
it through the full WhisperFlow pipeline.

Usage:
    whisperflow-web --host 0.0.0.0 --port 8766

For mobile microphone access, HTTPS is required. See docs for setup.
"""

import argparse
import io
import logging
import os
import secrets
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB


def _convert_to_wav(audio_bytes: bytes, content_type: str = "") -> bytes:
    """Convert browser audio (WebM/Opus, OGG, MP4) to 16kHz mono WAV for Whisper.

    Uses ffmpeg for reliable conversion. Falls back to returning raw bytes
    if the input is already WAV or ffmpeg is not available.
    """
    # If already WAV, return as-is
    if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        return audio_bytes

    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg not found — cannot convert browser audio to WAV. "
                       "Install with: sudo apt install ffmpeg")
        raise RuntimeError("ffmpeg is required for browser audio conversion")

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path.rsplit(".", 1)[0] + ".wav"

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", tmp_in_path,
                "-ar", "16000",     # 16kHz sample rate
                "-ac", "1",         # mono
                "-sample_fmt", "s16",  # 16-bit signed int
                "-f", "wav",
                tmp_out_path,
            ],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            logger.error(f"ffmpeg conversion failed: {result.stderr.decode(errors='replace')}")
            raise RuntimeError("Audio conversion failed")

        with open(tmp_out_path, "rb") as f:
            return f.read()
    finally:
        for p in (tmp_in_path, tmp_out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def create_app(api_key: str = ""):
    """Create the FastAPI application with API + static PWA serving."""
    from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, JSONResponse

    app = FastAPI(title="WhisperFlow Web", version="0.1.0")

    # Lazy-loaded components
    _state = {
        "config": None,
        "transcriber": None,
        "llm": None,
        "pipeline": None,
        "dictionary": None,
        "snippets": None,
        "style_learner": None,
        "history": None,
    }

    def _init_components():
        """Initialize WhisperFlow components on first request."""
        if _state["config"] is not None:
            return

        from whisperflow.config import load_config
        from whisperflow.llm.factory import create_llm_provider
        from whisperflow.features.dictionary import PersonalDictionary
        from whisperflow.features.snippets import SnippetManager
        from whisperflow.features.style_learning import StyleLearner
        from whisperflow.features.history import TranscriptionHistory
        from whisperflow.processing.pipeline import ProcessingPipeline

        config = load_config()
        _state["config"] = config

        # Transcription backend
        trans_cfg = config["transcription"]
        if trans_cfg["backend"] == "remote":
            from whisperflow.transcription.remote_whisper import RemoteWhisperBackend
            backend = RemoteWhisperBackend(server_url=trans_cfg["remote_url"])
            if backend.is_available():
                _state["transcriber"] = backend
                logger.info(f"Using remote transcription: {trans_cfg['remote_url']}")

        if _state["transcriber"] is None:
            from whisperflow.transcription.local_whisper import LocalWhisperBackend
            _state["transcriber"] = LocalWhisperBackend(
                model_size=trans_cfg["model_size"],
                device=trans_cfg["device"],
                compute_type=trans_cfg["compute_type"],
            )

        _state["llm"] = create_llm_provider(config["llm"])
        _state["dictionary"] = PersonalDictionary(config["dictionary"]["path"])
        _state["snippets"] = SnippetManager(config["snippets"]["path"])
        _state["style_learner"] = StyleLearner(
            config["style"]["path"],
            max_examples=config["style"]["max_examples"],
        )
        _state["history"] = TranscriptionHistory(
            config["history"]["path"],
            max_entries=config["history"]["max_entries"],
        )
        _state["pipeline"] = ProcessingPipeline(
            llm=_state["llm"],
            dictionary=_state["dictionary"],
            snippets=_state["snippets"],
            style_learner=_state["style_learner"],
            config=config,
        )
        logger.info("WhisperFlow components initialized")

    def _check_auth(authorization: str | None):
        if not api_key:
            return
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        token = authorization.removeprefix("Bearer ").strip()
        if not secrets.compare_digest(token, api_key):
            raise HTTPException(status_code=403, detail="Invalid API key")

    # -- API Endpoints --

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "auth_required": bool(api_key)}

    @app.post("/api/dictate")
    async def dictate(
        audio: UploadFile = File(...),
        language: str = Form(default=""),
        authorization: str | None = Header(default=None),
    ):
        """Process audio through the full dictation pipeline."""
        _check_auth(authorization)
        _init_components()

        audio_bytes = await audio.read()
        if len(audio_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Audio too large")

        # Convert browser audio (WebM/Opus) to WAV for Whisper
        try:
            wav_bytes = _convert_to_wav(audio_bytes, audio.content_type or "")
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))

        # Transcribe
        initial_prompt = _state["dictionary"].get_whisper_prompt()
        lang = language if language and language != "auto" else None

        start = time.time()
        result = _state["transcriber"].transcribe(
            wav_bytes, language=lang, initial_prompt=initial_prompt or None,
        )

        if result.is_empty:
            return {"text": "", "raw_text": "", "mode": "empty", "duration": 0}

        # Process through pipeline
        pipeline_result = _state["pipeline"].process_dictation(
            result.text, result.language, result.duration,
        )

        # Save to history
        config = _state["config"]
        if config["history"]["enabled"]:
            _state["history"].add(
                raw_text=result.text,
                processed_text=pipeline_result.processed_text,
                app_context="mobile",
                mode=pipeline_result.mode,
                language=result.language,
                duration=result.duration,
            )

        # Auto-learn dictionary
        if config["dictionary"].get("auto_learn", True):
            _state["dictionary"].check_and_learn(result.text, pipeline_result.processed_text)

        elapsed = time.time() - start
        return {
            "text": pipeline_result.processed_text,
            "raw_text": result.text,
            "mode": pipeline_result.mode,
            "language": result.language,
            "duration": result.duration,
            "processing_time": round(elapsed, 2),
        }

    @app.post("/api/command")
    async def command(
        audio: UploadFile = File(...),
        selected_text: str = Form(default=""),
        language: str = Form(default=""),
        authorization: str | None = Header(default=None),
    ):
        """Process audio as a command (edit selected text or ask question)."""
        _check_auth(authorization)
        _init_components()

        audio_bytes = await audio.read()
        if len(audio_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Audio too large")

        # Convert browser audio (WebM/Opus) to WAV for Whisper
        try:
            wav_bytes = _convert_to_wav(audio_bytes, audio.content_type or "")
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))

        initial_prompt = _state["dictionary"].get_whisper_prompt()
        lang = language if language and language != "auto" else None

        result = _state["transcriber"].transcribe(
            wav_bytes, language=lang, initial_prompt=initial_prompt or None,
        )

        if result.is_empty:
            return {"text": "", "raw_text": "", "mode": "empty"}

        # For mobile, we pass selected_text directly instead of xclip
        from whisperflow.processing.command_mode import CommandMode
        command_mode = CommandMode(_state["llm"])

        intent = command_mode.classify_intent(result.text)

        if intent == "question":
            answer = command_mode.answer_question(result.text)
            return {
                "text": answer,
                "raw_text": result.text,
                "mode": "question",
                "language": result.language,
            }

        if intent == "command" and selected_text:
            edited = command_mode.execute_command(result.text, selected_text)
            return {
                "text": edited,
                "raw_text": result.text,
                "mode": "command",
                "language": result.language,
            }

        # Fall back to dictation
        pipeline_result = _state["pipeline"].process_dictation(
            result.text, result.language, result.duration,
        )
        return {
            "text": pipeline_result.processed_text,
            "raw_text": result.text,
            "mode": "dictation",
            "language": result.language,
        }

    @app.get("/api/history")
    async def get_history(
        limit: int = 50,
        authorization: str | None = Header(default=None),
    ):
        _check_auth(authorization)
        _init_components()
        entries = _state["history"].get_recent(limit)
        return {"entries": entries}

    @app.post("/api/transcribe")
    async def transcribe(
        audio: UploadFile = File(...),
        language: str = Form(default=""),
        authorization: str | None = Header(default=None),
    ):
        """Raw speech-to-text: no LLM, no auto-edit, no pipeline. Just STT."""
        _check_auth(authorization)
        _init_components()

        audio_bytes = await audio.read()
        if len(audio_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Audio too large")

        try:
            wav_bytes = _convert_to_wav(audio_bytes, audio.content_type or "")
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))

        initial_prompt = _state["dictionary"].get_whisper_prompt()
        lang = language if language and language != "auto" else None

        start = time.time()
        result = _state["transcriber"].transcribe(
            wav_bytes, language=lang, initial_prompt=initial_prompt or None,
        )
        elapsed = time.time() - start

        if result.is_empty:
            return {"text": "", "language": "", "duration": 0, "processing_time": 0}

        # Save to history
        config = _state["config"]
        if config["history"]["enabled"]:
            _state["history"].add(
                raw_text=result.text,
                processed_text=result.text,
                app_context="mobile-transcribe",
                mode="transcribe",
                language=result.language,
                duration=result.duration,
            )

        return {
            "text": result.text,
            "language": result.language,
            "confidence": result.confidence,
            "duration": result.duration,
            "processing_time": round(elapsed, 2),
        }

    @app.get("/api/snippets")
    async def get_snippets(authorization: str | None = Header(default=None)):
        _check_auth(authorization)
        _init_components()
        return {"snippets": _state["snippets"].get_snippets()}

    @app.post("/api/snippets")
    async def add_snippet(
        trigger: str = Form(...),
        content: str = Form(...),
        description: str = Form(default=""),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(authorization)
        _init_components()
        _state["snippets"].add_snippet(trigger, content, description)
        return {"ok": True}

    @app.delete("/api/snippets/{trigger}")
    async def delete_snippet(trigger: str, authorization: str | None = Header(default=None)):
        _check_auth(authorization)
        _init_components()
        _state["snippets"].remove_snippet(trigger)
        return {"ok": True}

    @app.get("/api/dictionary")
    async def get_dictionary(authorization: str | None = Header(default=None)):
        _check_auth(authorization)
        _init_components()
        return {"words": _state["dictionary"].get_words()}

    @app.post("/api/dictionary")
    async def add_word(
        word: str = Form(...),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(authorization)
        _init_components()
        _state["dictionary"].add_word(word)
        return {"ok": True}

    @app.delete("/api/dictionary/{word}")
    async def remove_word(word: str, authorization: str | None = Header(default=None)):
        _check_auth(authorization)
        _init_components()
        _state["dictionary"].remove_word(word)
        return {"ok": True}

    @app.get("/api/history/search")
    async def search_history(
        q: str = "",
        limit: int = 50,
        authorization: str | None = Header(default=None),
    ):
        _check_auth(authorization)
        _init_components()
        if not q:
            return {"entries": []}
        entries = _state["history"].search(q, limit)
        return {"entries": entries}

    # -- PWA Static Files --

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/manifest.json")
    async def manifest():
        return FileResponse(STATIC_DIR / "manifest.json", media_type="application/manifest+json")

    @app.get("/sw.js")
    async def service_worker():
        return FileResponse(STATIC_DIR / "sw.js", media_type="application/javascript")

    # Mount static files last
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


def main():
    parser = argparse.ArgumentParser(description="WhisperFlow Web Server (Mobile PWA)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8766, help="Port number")
    parser.add_argument(
        "--api-key", default="",
        help="API key for authentication (or set WHISPERFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--ssl-cert", default="",
        help="Path to SSL certificate (required for mobile mic access)",
    )
    parser.add_argument(
        "--ssl-key", default="",
        help="Path to SSL private key",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("WHISPERFLOW_API_KEY", "")
    if api_key:
        logger.info("Authentication enabled")
    else:
        logger.warning("No API key set -- server is open to all network requests")

    import uvicorn

    app = create_app(api_key=api_key)

    ssl_kwargs = {}
    if args.ssl_cert and args.ssl_key:
        ssl_kwargs["ssl_certfile"] = args.ssl_cert
        ssl_kwargs["ssl_keyfile"] = args.ssl_key
        logger.info("HTTPS enabled")
    else:
        logger.warning(
            "Running without HTTPS. Mobile microphone access requires HTTPS. "
            "Generate a self-signed cert: "
            "openssl req -x509 -newkey rsa:4096 -nodes -keyout key.pem -out cert.pem -days 365"
        )

    logger.info(f"Starting WhisperFlow Web on {args.host}:{args.port}")
    logger.info(f"Open on your phone: {'https' if ssl_kwargs else 'http'}://<your-ip>:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, **ssl_kwargs)


if __name__ == "__main__":
    main()

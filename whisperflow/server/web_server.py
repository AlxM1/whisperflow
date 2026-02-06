"""WhisperFlow Web Server - serves PWA mobile app + REST API.

Allows using WhisperFlow from any device (phone, tablet, remote PC)
by providing a web-based interface that captures audio and processes
it through the full WhisperFlow pipeline.

Usage:
    whisperflow-web --host 0.0.0.0 --port 8766

For mobile microphone access, HTTPS is required. See docs for setup.
"""

import argparse
import collections
import io
import json
import logging
import os
import secrets
import shutil
import subprocess
import tempfile
import time
import wave
import zipfile
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB

# -- Simple in-memory rate limiter --
_rate_buckets: dict[str, collections.deque] = {}
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 60  # requests per window


def _check_rate_limit(client_ip: str) -> bool:
    """Returns True if the request should be rate-limited."""
    now = time.time()
    if client_ip not in _rate_buckets:
        _rate_buckets[client_ip] = collections.deque()
    bucket = _rate_buckets[client_ip]
    # Remove expired entries
    while bucket and bucket[0] < now - RATE_LIMIT_WINDOW:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_MAX:
        return True
    bucket.append(now)
    return False


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

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as StarletteRequest
    from starlette.responses import JSONResponse as StarletteJSONResponse

    app = FastAPI(title="WhisperFlow Web", version="0.1.0")

    # Rate limiting middleware
    class RateLimitMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: StarletteRequest, call_next):
            if request.url.path.startswith("/api/"):
                client_ip = request.client.host if request.client else "unknown"
                if _check_rate_limit(client_ip):
                    return StarletteJSONResponse(
                        {"detail": "Rate limit exceeded. Try again later."},
                        status_code=429,
                    )
            return await call_next(request)

    app.add_middleware(RateLimitMiddleware)

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
            remote_key = trans_cfg.get("remote_api_key", "") or os.environ.get("WHISPERFLOW_API_KEY", "")
            backend = RemoteWhisperBackend(server_url=trans_cfg["remote_url"], api_key=remote_key)
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

        # Save audio if configured
        audio_file = ""
        config = _state["config"]
        if config.get("history", {}).get("save_audio", False):
            audio_dir = Path(config["history"]["path"]).parent / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio_file = str(audio_dir / f"{int(time.time() * 1000)}.wav")
            Path(audio_file).write_bytes(wav_bytes)

        # Save to history
        if config["history"]["enabled"]:
            _state["history"].add(
                raw_text=result.text,
                processed_text=pipeline_result.processed_text,
                app_context="mobile",
                mode=pipeline_result.mode,
                language=result.language,
                duration=result.duration,
                audio_path=audio_file,
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
            "has_audio": bool(audio_file),
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

        # Save audio if configured
        audio_file = ""
        config = _state["config"]
        if config.get("history", {}).get("save_audio", False):
            audio_dir = Path(config["history"]["path"]).parent / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio_file = str(audio_dir / f"{int(time.time() * 1000)}.wav")
            Path(audio_file).write_bytes(wav_bytes)

        # Save to history
        if config["history"]["enabled"]:
            _state["history"].add(
                raw_text=result.text,
                processed_text=result.text,
                app_context="mobile-transcribe",
                mode="transcribe",
                language=result.language,
                duration=result.duration,
                audio_path=audio_file,
            )

        return {
            "text": result.text,
            "language": result.language,
            "confidence": result.confidence,
            "duration": result.duration,
            "processing_time": round(elapsed, 2),
            "has_audio": bool(audio_file),
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
        if not q:
            return {"entries": []}
        _init_components()
        entries = _state["history"].search(q, limit)
        return {"entries": entries}

    # -- History Export --

    @app.get("/api/history/export")
    async def export_history(
        format: str = "json",
        authorization: str | None = Header(default=None),
    ):
        from fastapi.responses import Response
        _check_auth(authorization)
        _init_components()
        if format == "csv":
            data = _state["history"].export_csv()
            return Response(content=data, media_type="text/csv",
                            headers={"Content-Disposition": "attachment; filename=whisperflow_history.csv"})
        else:
            data = _state["history"].export_json()
            return Response(content=data, media_type="application/json",
                            headers={"Content-Disposition": "attachment; filename=whisperflow_history.json"})

    # -- Bulk Dictionary Import --

    @app.post("/api/dictionary/import")
    async def import_dictionary(
        file: UploadFile = File(...),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(authorization)
        _init_components()
        content = await file.read()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=422, detail="File must be UTF-8 text")
        # Support JSON ({"words": {"word": "category"}}) or plain text (one word per line)
        imported = 0
        if text.strip().startswith("{"):
            data = json.loads(text)
            words = data.get("words", data)
            for word, cat in words.items():
                if word.strip():
                    _state["dictionary"].add_word(word.strip(), cat if isinstance(cat, str) else "imported")
                    imported += 1
        else:
            for line in text.splitlines():
                word = line.strip()
                if word and not word.startswith("#"):
                    _state["dictionary"].add_word(word, "imported")
                    imported += 1
        return {"ok": True, "imported": imported}

    # -- Audio Storage & Playback --

    @app.get("/api/history/{entry_id}/audio")
    async def get_audio(entry_id: int, authorization: str | None = Header(default=None)):
        _check_auth(authorization)
        _init_components()
        entry = _state["history"].get_entry(entry_id)
        if not entry or not entry.audio_path:
            raise HTTPException(status_code=404, detail="Audio not found")
        audio_path = Path(entry.audio_path)
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="Audio file missing")
        return FileResponse(audio_path, media_type="audio/wav")

    # -- Usage Stats --

    @app.get("/api/stats")
    async def get_stats(authorization: str | None = Header(default=None)):
        _check_auth(authorization)
        _init_components()
        stats = _state["history"].get_stats()
        stats["dictionary_words"] = len(_state["dictionary"].get_words())
        stats["snippets_count"] = len(_state["snippets"].get_snippets())
        return stats

    # -- Language Memory --

    @app.get("/api/settings/language")
    async def get_language_setting(authorization: str | None = Header(default=None)):
        _check_auth(authorization)
        _init_components()
        # Return the last detected language from recent history
        entries = _state["history"].get_recent(1)
        last_lang = entries[0].language if entries else ""
        return {"language": last_lang}

    # -- Backup & Restore --

    @app.get("/api/backup")
    async def backup(authorization: str | None = Header(default=None)):
        from fastapi.responses import Response
        _check_auth(authorization)
        _init_components()
        config = _state["config"]

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # Dictionary
            dict_path = Path(config["dictionary"]["path"])
            if dict_path.exists():
                zf.write(dict_path, "dictionary.json")
            # Snippets
            snip_path = Path(config["snippets"]["path"])
            if snip_path.exists():
                zf.write(snip_path, "snippets.json")
            # Style examples
            style_path = Path(config["style"]["path"])
            if style_path.exists():
                zf.write(style_path, "style_examples.json")
            # History export as JSON
            zf.writestr("history.json", _state["history"].export_json())
            # Config
            from whisperflow.config import CONFIG_FILE
            if CONFIG_FILE.exists():
                zf.write(CONFIG_FILE, "config.yaml")

        return Response(
            content=buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=whisperflow_backup.zip"},
        )

    @app.post("/api/restore")
    async def restore(
        file: UploadFile = File(...),
        authorization: str | None = Header(default=None),
    ):
        _check_auth(authorization)
        _init_components()
        config = _state["config"]

        content = await file.read()
        restored = []
        try:
            with zipfile.ZipFile(io.BytesIO(content), "r") as zf:
                names = zf.namelist()
                if "dictionary.json" in names:
                    data = json.loads(zf.read("dictionary.json"))
                    words = data.get("words", data)
                    for word, cat in words.items():
                        _state["dictionary"].add_word(word, cat if isinstance(cat, str) else "general")
                    restored.append("dictionary")
                if "snippets.json" in names:
                    snip_path = Path(config["snippets"]["path"])
                    snip_path.write_bytes(zf.read("snippets.json"))
                    _state["snippets"]._load()
                    restored.append("snippets")
                if "style_examples.json" in names:
                    style_path = Path(config["style"]["path"])
                    style_path.write_bytes(zf.read("style_examples.json"))
                    restored.append("style_examples")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid backup file: {e}")

        return {"ok": True, "restored": restored}

    # -- WebSocket Streaming STT --

    @app.websocket("/ws/stream")
    async def websocket_stream(websocket):
        from starlette.websockets import WebSocket, WebSocketDisconnect
        await websocket.accept()

        # Auth check via first message
        if api_key:
            try:
                auth_msg = await websocket.receive_json()
                token = auth_msg.get("api_key", "")
                if not secrets.compare_digest(token, api_key):
                    await websocket.send_json({"error": "Invalid API key"})
                    await websocket.close()
                    return
            except Exception:
                await websocket.close()
                return

        _init_components()
        audio_chunks = []
        language = "auto"

        try:
            while True:
                msg = await websocket.receive()

                if msg.get("type") == "websocket.receive":
                    if "bytes" in msg and msg["bytes"]:
                        audio_chunks.append(msg["bytes"])
                    elif "text" in msg:
                        data = json.loads(msg["text"])
                        if data.get("action") == "stop":
                            # Combine all chunks and transcribe
                            if audio_chunks:
                                combined = b"".join(audio_chunks)
                                try:
                                    wav_bytes = _convert_to_wav(combined)
                                except RuntimeError:
                                    await websocket.send_json({"error": "Audio conversion failed"})
                                    audio_chunks.clear()
                                    continue

                                initial_prompt = _state["dictionary"].get_whisper_prompt()
                                lang = language if language != "auto" else None
                                result = _state["transcriber"].transcribe(
                                    wav_bytes, language=lang,
                                    initial_prompt=initial_prompt or None,
                                )
                                await websocket.send_json({
                                    "text": result.text,
                                    "language": result.language,
                                    "confidence": result.confidence,
                                    "duration": result.duration,
                                    "final": True,
                                })
                            audio_chunks.clear()
                        elif data.get("action") == "config":
                            language = data.get("language", "auto")
                        elif data.get("action") == "clear":
                            audio_chunks.clear()
        except Exception:
            pass  # WebSocket disconnected

    # -- Digital Asset Links (Android TWA verification) --

    @app.get("/.well-known/assetlinks.json")
    async def asset_links():
        """Serve Digital Asset Links for Android TWA verification.

        The fingerprint is populated from WHISPERFLOW_APK_FINGERPRINT env var
        or returns a placeholder that must be configured before TWA verification works.
        """
        fingerprint = os.environ.get(
            "WHISPERFLOW_APK_FINGERPRINT",
            "AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99",
        )
        return JSONResponse([{
            "relation": ["delegate_permission/common.handle_all_urls"],
            "target": {
                "namespace": "android_app",
                "package_name": "com.whisperflow.twa",
                "sha256_cert_fingerprints": [fingerprint],
            },
        }])

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
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("WHISPERFLOW_DATA_DIR", ""),
        help="Override data directory for all storage (config, history, dictionary, etc.)",
    )
    args = parser.parse_args()

    if args.data_dir:
        from whisperflow.config import set_data_dir
        set_data_dir(args.data_dir)

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

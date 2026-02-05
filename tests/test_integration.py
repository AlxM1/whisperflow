"""Integration tests - test component interactions with mocked externals."""

import io
import json
import os
import sys
import tempfile
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Detect missing system deps to skip tests cleanly
_has_portaudio = False
try:
    import sounddevice
    _has_portaudio = True
except OSError:
    pass

_has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

_has_pynput = False
try:
    from pynput import keyboard as _kb
    _has_pynput = True
except (ImportError, Exception):
    pass

needs_portaudio = pytest.mark.skipif(not _has_portaudio, reason="PortAudio not available")
needs_display = pytest.mark.skipif(not _has_display, reason="No display server (X11/Wayland)")
needs_pynput = pytest.mark.skipif(not _has_pynput, reason="pynput needs X11 display")


# ============================================================================
# LLM Factory
# ============================================================================

class TestLLMFactory:
    def test_noop_provider(self):
        from whisperflow.llm.factory import NoopProvider
        provider = NoopProvider()
        assert provider.is_available()
        response = provider.complete("system", "hello world")
        assert response.text == "hello world"
        assert response.model == "noop"

    def test_factory_creates_noop_for_none(self):
        from whisperflow.llm.factory import create_llm_provider, NoopProvider
        provider = create_llm_provider({"provider": "none"})
        assert isinstance(provider, NoopProvider)

    def test_factory_creates_noop_for_empty_config(self):
        from whisperflow.llm.factory import create_llm_provider, NoopProvider
        provider = create_llm_provider({})
        assert isinstance(provider, NoopProvider)

    def test_factory_ollama_unavailable_falls_back(self):
        from whisperflow.llm.factory import create_llm_provider, NoopProvider
        config = {
            "provider": "ollama",
            "ollama": {"url": "http://localhost:99999", "model": "test"},
        }
        provider = create_llm_provider(config)
        assert isinstance(provider, NoopProvider)

    def test_factory_openai_no_key_falls_back(self):
        from whisperflow.llm.factory import create_llm_provider, NoopProvider
        config = {
            "provider": "openai",
            "openai": {"api_key": "", "model": "gpt-4o-mini"},
        }
        provider = create_llm_provider(config)
        assert isinstance(provider, NoopProvider)

    def test_factory_anthropic_no_key_falls_back(self):
        from whisperflow.llm.factory import create_llm_provider, NoopProvider
        config = {
            "provider": "anthropic",
            "anthropic": {"api_key": "", "model": "claude-sonnet-4-20250514"},
        }
        provider = create_llm_provider(config)
        assert isinstance(provider, NoopProvider)

    def test_openai_provider_with_key_is_available(self):
        from whisperflow.llm.openai_api import OpenAIProvider
        provider = OpenAIProvider(api_key="sk-test-key")
        assert provider.is_available()

    def test_anthropic_provider_with_key_is_available(self):
        from whisperflow.llm.anthropic_api import AnthropicProvider
        provider = AnthropicProvider(api_key="sk-ant-test-key")
        assert provider.is_available()


# ============================================================================
# Audio Capture
# ============================================================================

@needs_portaudio
class TestAudioCapture:
    """Tests for AudioCapture - uses numpy directly to avoid sounddevice import."""

    def _make_capture(self):
        """Create AudioCapture instance without importing sounddevice at module level."""
        from collections import deque
        capture = MagicMock()
        capture.channels = 1
        capture.sample_rate = 16000
        capture._recording = False
        capture._frames = deque()

        # Bind real methods
        from whisperflow.audio import capture as cap_module
        capture._to_wav = lambda audio_data: cap_module.AudioCapture._to_wav(capture, audio_data)
        capture._audio_callback = lambda indata, frames, time_info, status: cap_module.AudioCapture._audio_callback(capture, indata, frames, time_info, status)
        return capture

    def test_wav_conversion(self):
        """Test that _to_wav produces valid WAV bytes."""
        capture = self._make_capture()

        t = np.linspace(0, 1, 16000, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t)

        wav_bytes = capture._to_wav(audio)

        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert wf.getsampwidth() == 2
            assert wf.getnframes() == 16000

    def test_wav_roundtrip(self):
        """Test WAV encode -> decode produces reasonable audio."""
        capture = self._make_capture()

        original = np.random.randn(8000).astype(np.float32) * 0.5
        wav_bytes = capture._to_wav(original)

        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
        decoded = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        assert np.allclose(original, decoded, atol=1e-3)

    def test_audio_callback_appends_frames(self):
        """Audio callback should append frames when recording."""
        capture = self._make_capture()
        capture._recording = True

        data = np.random.randn(1600).astype(np.float32)
        capture._audio_callback(data, 1600, None, None)

        assert len(capture._frames) == 1
        np.testing.assert_array_equal(capture._frames[0], data)

    def test_audio_callback_ignores_when_not_recording(self):
        """Audio callback should ignore data when not recording."""
        capture = self._make_capture()
        capture._recording = False

        data = np.random.randn(1600).astype(np.float32)
        capture._audio_callback(data, 1600, None, None)

        assert len(capture._frames) == 0

    @needs_portaudio
    def test_stop_without_start_returns_empty(self):
        from whisperflow.audio.capture import AudioCapture
        capture = AudioCapture()
        result = capture.stop_recording()
        assert result == b""

    @needs_portaudio
    def test_is_recording_property(self):
        from whisperflow.audio.capture import AudioCapture
        capture = AudioCapture()
        assert not capture.is_recording


# ============================================================================
# Hotkey Manager
# ============================================================================

class TestHotkeyManager:
    @needs_pynput
    def test_parse_hotkey(self):
        from whisperflow.input.hotkey import parse_hotkey
        from pynput import keyboard

        keys = parse_hotkey("ctrl_l+space")
        assert keyboard.Key.ctrl_l in keys
        assert keyboard.Key.space in keys
        assert len(keys) == 2

    @needs_pynput
    def test_parse_hotkey_three_keys(self):
        from whisperflow.input.hotkey import parse_hotkey
        from pynput import keyboard

        keys = parse_hotkey("ctrl_l+shift+space")
        assert keyboard.Key.ctrl_l in keys
        assert keyboard.Key.shift_l in keys
        assert keyboard.Key.space in keys
        assert len(keys) == 3

    @needs_pynput
    def test_parse_single_char_key(self):
        from whisperflow.input.hotkey import parse_hotkey
        from pynput import keyboard

        keys = parse_hotkey("ctrl+w")
        assert keyboard.Key.ctrl_l in keys
        assert len(keys) == 2

    @needs_pynput
    def test_manager_init(self):
        from whisperflow.input.hotkey import HotkeyManager

        mgr = HotkeyManager(
            hold_key="ctrl_l+space",
            command_key="ctrl_l+shift+space",
        )
        assert not mgr._hold_active
        assert not mgr._command_active

    @needs_pynput
    def test_manager_start_stop(self):
        from whisperflow.input.hotkey import HotkeyManager

        with patch("whisperflow.input.hotkey.keyboard.Listener"):
            mgr = HotkeyManager()
            mgr.start()
            assert mgr._listener is not None
            mgr.stop()
            assert mgr._listener is None


# ============================================================================
# Text Injection
# ============================================================================

class TestTextInjection:
    def test_inject_empty_text(self):
        from whisperflow.input.injection import TextInjector
        injector = TextInjector()
        assert not injector.inject("")

    @patch("whisperflow.input.injection.subprocess")
    def test_inject_x11(self, mock_subprocess):
        from whisperflow.input.injection import TextInjector

        mock_subprocess.run.return_value = MagicMock(stdout="")
        mock_subprocess.Popen.return_value = MagicMock()
        mock_subprocess.Popen.return_value.communicate.return_value = (b"", b"")

        injector = TextInjector(method="xdotool", delay_ms=0)
        result = injector.inject("Hello World")
        assert result is True

    @patch("whisperflow.input.injection.subprocess")
    def test_inject_wayland(self, mock_subprocess):
        from whisperflow.input.injection import TextInjector

        mock_subprocess.Popen.return_value = MagicMock()
        mock_subprocess.Popen.return_value.communicate.return_value = (b"", b"")
        mock_subprocess.run.return_value = MagicMock()

        injector = TextInjector(method="wl-copy", delay_ms=0)
        result = injector.inject("Hello World")
        assert result is True


# ============================================================================
# Tone Matching
# ============================================================================

class TestToneMatching:
    def test_get_tone_for_email(self):
        from whisperflow.processing.tone import ToneMatcher
        from whisperflow.input.context import WindowContext
        from whisperflow.llm.factory import NoopProvider

        matcher = ToneMatcher(NoopProvider())
        ctx = WindowContext("thunderbird", "Inbox", "Thunderbird")
        assert matcher.get_tone(ctx) == "professional"

    def test_get_tone_for_slack(self):
        from whisperflow.processing.tone import ToneMatcher
        from whisperflow.input.context import WindowContext
        from whisperflow.llm.factory import NoopProvider

        matcher = ToneMatcher(NoopProvider())
        ctx = WindowContext("slack", "general - Slack", "Slack")
        assert matcher.get_tone(ctx) == "casual"

    def test_get_tone_for_code(self):
        from whisperflow.processing.tone import ToneMatcher
        from whisperflow.input.context import WindowContext
        from whisperflow.llm.factory import NoopProvider

        matcher = ToneMatcher(NoopProvider())
        ctx = WindowContext("code", "main.py - VSCode", "Code")
        assert matcher.get_tone(ctx) == "technical"

    def test_user_tone_override(self):
        from whisperflow.processing.tone import ToneMatcher
        from whisperflow.input.context import WindowContext
        from whisperflow.llm.factory import NoopProvider

        matcher = ToneMatcher(NoopProvider(), app_tones={"slack": "formal"})
        ctx = WindowContext("slack", "general - Slack", "Slack")
        assert matcher.get_tone(ctx) == "formal"

    def test_build_combined_prompt(self):
        from whisperflow.processing.tone import ToneMatcher
        from whisperflow.input.context import WindowContext
        from whisperflow.llm.factory import NoopProvider

        matcher = ToneMatcher(NoopProvider())
        ctx = WindowContext("slack", "general - Slack", "Slack")
        prompt = matcher.build_combined_prompt(ctx)
        assert "casual" in prompt.lower()

    def test_apply_tone_noop_passthrough(self):
        from whisperflow.processing.tone import ToneMatcher
        from whisperflow.input.context import WindowContext
        from whisperflow.llm.factory import NoopProvider

        matcher = ToneMatcher(NoopProvider())
        ctx = WindowContext("slack", "general - Slack", "Slack")
        result = matcher.apply_tone("Hello there!", ctx)
        assert result == "Hello there!"


# ============================================================================
# Config Management
# ============================================================================

class TestConfig:
    def test_deep_merge(self):
        from whisperflow.config import _deep_merge
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}, "e": 5}
        result = _deep_merge(base, override)
        assert result["a"] == 1
        assert result["b"]["c"] == 99
        assert result["b"]["d"] == 3
        assert result["e"] == 5

    def test_deep_merge_empty_override(self):
        from whisperflow.config import _deep_merge
        base = {"a": 1}
        result = _deep_merge(base, {})
        assert result == {"a": 1}

    def test_load_json_file_missing(self):
        from whisperflow.config import load_json_file
        result = load_json_file("/nonexistent/path.json")
        assert result == {}

    def test_save_and_load_json(self):
        from whisperflow.config import save_json_file, load_json_file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_json_file(path, {"key": "value"})
            result = load_json_file(path)
            assert result == {"key": "value"}
        finally:
            os.unlink(path)

    def test_load_default_config(self):
        """Default config YAML should load without errors and contain expected sections."""
        from whisperflow.config import load_config
        with patch.dict(os.environ, {"WHISPERFLOW_CONFIG_DIR": tempfile.mkdtemp()}):
            config = load_config()
            assert "audio" in config
            assert "hotkey" in config
            assert "transcription" in config
            assert "llm" in config
            assert "processing" in config
            assert "context" in config
            assert "dictionary" in config
            assert "snippets" in config
            assert "history" in config
            assert "ui" in config
            assert "injection" in config

    def test_config_default_values(self):
        """Verify key default values in config."""
        from whisperflow.config import load_config
        with patch.dict(os.environ, {"WHISPERFLOW_CONFIG_DIR": tempfile.mkdtemp()}):
            config = load_config()
            assert config["audio"]["sample_rate"] == 16000
            assert config["transcription"]["model_size"] == "large-v3-turbo"
            assert config["llm"]["provider"] == "ollama"
            assert config["processing"]["auto_edit"] is True
            assert config["injection"]["method"] == "xdotool"


# ============================================================================
# Processing Pipeline (Integration)
# ============================================================================

class TestProcessingPipeline:
    """Integration test: process_dictation with NoopProvider and mocked context."""

    def _make_pipeline(self):
        from whisperflow.llm.factory import NoopProvider
        from whisperflow.features.dictionary import PersonalDictionary
        from whisperflow.features.snippets import SnippetManager
        from whisperflow.features.style_learning import StyleLearner
        from whisperflow.processing.pipeline import ProcessingPipeline

        self._tmpdir = tempfile.mkdtemp()
        dict_path = os.path.join(self._tmpdir, "dict.json")
        snip_path = os.path.join(self._tmpdir, "snip.json")
        style_path = os.path.join(self._tmpdir, "style.json")

        for p in (dict_path, snip_path, style_path):
            with open(p, "w") as f:
                f.write("{}")

        config = {
            "processing": {
                "auto_edit": True,
                "context_aware": False,
                "tone_matching": True,
                "course_correction": True,
            },
            "context": {"app_tones": {}},
        }

        return ProcessingPipeline(
            llm=NoopProvider(),
            dictionary=PersonalDictionary(dict_path),
            snippets=SnippetManager(snip_path),
            style_learner=StyleLearner(style_path),
            config=config,
        )

    def test_dictation_basic(self):
        pipeline = self._make_pipeline()
        result = pipeline.process_dictation("hello this is a test sentence for the pipeline")
        assert result.mode == "dictation"
        assert result.processed_text
        assert result.original_text == "hello this is a test sentence for the pipeline"

    def test_dictation_snippet_triggers(self):
        pipeline = self._make_pipeline()
        pipeline.snippets.add_snippet("my email", "user@example.com")
        result = pipeline.process_dictation("my email")
        assert result.mode == "snippet"
        assert result.processed_text == "user@example.com"

    def test_dictation_snippet_with_prefix(self):
        pipeline = self._make_pipeline()
        pipeline.snippets.add_snippet("my email", "user@example.com")
        result = pipeline.process_dictation("insert my email")
        assert result.mode == "snippet"
        assert result.processed_text == "user@example.com"

    @patch("whisperflow.processing.pipeline.get_active_window")
    @patch("whisperflow.processing.pipeline.get_selected_text")
    def test_command_question(self, mock_selected, mock_window):
        from whisperflow.input.context import WindowContext
        mock_window.return_value = WindowContext("firefox", "Test", "Firefox")
        mock_selected.return_value = ""

        pipeline = self._make_pipeline()
        pipeline._context_aware = True
        result = pipeline.process_command("what is the capital of France")
        assert result.mode == "question"
        assert result.processed_text

    @patch("whisperflow.processing.pipeline.get_active_window")
    @patch("whisperflow.processing.pipeline.get_selected_text")
    def test_command_with_selected_text(self, mock_selected, mock_window):
        from whisperflow.input.context import WindowContext
        mock_window.return_value = WindowContext("code", "main.py", "Code")
        mock_selected.return_value = "this is some selected text to edit"

        pipeline = self._make_pipeline()
        pipeline._context_aware = True
        result = pipeline.process_command("make this more formal")
        assert result.mode == "command"
        assert result.processed_text

    @patch("whisperflow.processing.pipeline.get_active_window")
    @patch("whisperflow.processing.pipeline.get_selected_text")
    def test_command_no_selected_falls_to_dictation(self, mock_selected, mock_window):
        from whisperflow.input.context import WindowContext
        mock_window.return_value = WindowContext("code", "main.py", "Code")
        mock_selected.return_value = ""

        pipeline = self._make_pipeline()
        pipeline._context_aware = True
        result = pipeline.process_command("make this more formal")
        assert result.mode == "dictation"

    def test_dictation_auto_edit_disabled(self):
        pipeline = self._make_pipeline()
        pipeline._auto_edit_enabled = False
        result = pipeline.process_dictation("um uh hello world test text here")
        assert result.processed_text == "um uh hello world test text here"

    def test_pipeline_result_has_language_and_duration(self):
        pipeline = self._make_pipeline()
        result = pipeline.process_dictation("test", language="en", duration=1.5)
        assert result.language == "en"
        assert result.duration == 1.5


# ============================================================================
# Transcription Backends
# ============================================================================

class TestTranscriptionResult:
    def test_empty_check(self):
        from whisperflow.transcription.base import TranscriptionResult
        empty = TranscriptionResult(text="", language="en", confidence=0.9, duration=1.0)
        assert empty.is_empty
        whitespace = TranscriptionResult(text="  ", language="en", confidence=0.9, duration=1.0)
        assert whitespace.is_empty
        non_empty = TranscriptionResult(text="hello", language="en", confidence=0.9, duration=1.0)
        assert not non_empty.is_empty

    def test_local_whisper_not_available_without_package(self):
        from whisperflow.transcription.local_whisper import LocalWhisperBackend
        backend = LocalWhisperBackend()
        available = backend.is_available()
        assert isinstance(available, bool)

    def test_remote_whisper_unavailable(self):
        from whisperflow.transcription.remote_whisper import RemoteWhisperBackend
        backend = RemoteWhisperBackend(server_url="http://localhost:99999")
        assert not backend.is_available()

    def test_remote_whisper_transcribe_failure(self):
        from whisperflow.transcription.remote_whisper import RemoteWhisperBackend
        backend = RemoteWhisperBackend(server_url="http://localhost:99999")
        result = backend.transcribe(b"fake audio data")
        assert result.text == ""
        assert result.confidence == 0.0


# ============================================================================
# GPU Server API
# ============================================================================

class TestGPUServer:
    def test_create_app(self):
        from whisperflow.server.gpu_server import create_app
        app = create_app()
        assert app.title == "WhisperFlow GPU Server"

    def test_health_endpoint(self):
        from whisperflow.server.gpu_server import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "gpu": True}


# ============================================================================
# Module Import Smoke Tests (skip modules requiring system deps)
# ============================================================================

class TestModuleImports:
    """Verify all modules can be imported without errors."""

    def test_import_config(self):
        import whisperflow.config

    @needs_portaudio
    def test_import_audio_capture(self):
        import whisperflow.audio.capture

    def test_import_transcription_base(self):
        import whisperflow.transcription.base

    def test_import_transcription_local(self):
        import whisperflow.transcription.local_whisper

    def test_import_transcription_remote(self):
        import whisperflow.transcription.remote_whisper

    def test_import_llm_base(self):
        import whisperflow.llm.base

    def test_import_llm_factory(self):
        import whisperflow.llm.factory

    def test_import_llm_ollama(self):
        import whisperflow.llm.ollama

    def test_import_llm_openai(self):
        import whisperflow.llm.openai_api

    def test_import_llm_anthropic(self):
        import whisperflow.llm.anthropic_api

    def test_import_processing_pipeline(self):
        import whisperflow.processing.pipeline

    def test_import_processing_auto_edit(self):
        import whisperflow.processing.auto_edit

    def test_import_processing_command_mode(self):
        import whisperflow.processing.command_mode

    def test_import_processing_tone(self):
        import whisperflow.processing.tone

    def test_import_processing_prompts(self):
        import whisperflow.processing.prompts

    def test_import_input_context(self):
        import whisperflow.input.context

    @needs_pynput
    def test_import_input_hotkey(self):
        import whisperflow.input.hotkey

    def test_import_input_injection(self):
        import whisperflow.input.injection

    def test_import_features_dictionary(self):
        import whisperflow.features.dictionary

    def test_import_features_snippets(self):
        import whisperflow.features.snippets

    def test_import_features_history(self):
        import whisperflow.features.history

    def test_import_features_style_learning(self):
        import whisperflow.features.style_learning

    def test_import_server(self):
        import whisperflow.server.gpu_server

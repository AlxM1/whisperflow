"""Integration tests - test component interactions with mocked externals."""

import io
import json
import os
import sys
import tempfile
import wave
from pathlib import Path
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

    @patch("whisperflow.input.injection.CLIPBOARD_RESTORE_DELAY", 0)
    @patch("whisperflow.input.injection.subprocess")
    def test_inject_x11(self, mock_subprocess):
        from whisperflow.input.injection import TextInjector

        mock_subprocess.run.return_value = MagicMock(stdout="", returncode=0)
        mock_subprocess.Popen.return_value = MagicMock()
        mock_subprocess.Popen.return_value.communicate.return_value = (b"", b"")

        injector = TextInjector(method="xdotool", delay_ms=0)
        result = injector.inject("Hello World")
        assert result is True

    @patch("whisperflow.input.injection.CLIPBOARD_RESTORE_DELAY", 0)
    @patch("whisperflow.input.injection.subprocess")
    def test_inject_wayland(self, mock_subprocess):
        from whisperflow.input.injection import TextInjector

        mock_subprocess.Popen.return_value = MagicMock()
        mock_subprocess.Popen.return_value.communicate.return_value = (b"", b"")
        mock_subprocess.run.return_value = MagicMock(stdout="", returncode=0)

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
        import whisperflow.config as cfg_mod
        from whisperflow.config import load_config
        tmpdir = Path(tempfile.mkdtemp())
        with patch.object(cfg_mod, "CONFIG_DIR", tmpdir), \
             patch.object(cfg_mod, "CONFIG_FILE", tmpdir / "config.yaml"):
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
        import whisperflow.config as cfg_mod
        from whisperflow.config import load_config
        tmpdir = Path(tempfile.mkdtemp())
        with patch.object(cfg_mod, "CONFIG_DIR", tmpdir), \
             patch.object(cfg_mod, "CONFIG_FILE", tmpdir / "config.yaml"):
            config = load_config()
            assert config["audio"]["sample_rate"] == 16000
            assert config["transcription"]["model_size"] == "large-v3-turbo"
            assert config["llm"]["provider"] == "ollama"
            assert config["processing"]["auto_edit"] is True
            assert config["injection"]["method"] == "auto"

    def test_set_data_dir_overrides_paths(self):
        """set_data_dir should redirect all storage paths to the custom directory."""
        import whisperflow.config as cfg_mod
        from whisperflow.config import set_data_dir, load_config

        tmpdir = Path(tempfile.mkdtemp())
        custom_dir = tmpdir / "my_data"

        # Save originals to restore after test
        orig_config_dir = cfg_mod.CONFIG_DIR
        orig_config_file = cfg_mod.CONFIG_FILE
        try:
            set_data_dir(str(custom_dir))
            assert cfg_mod.CONFIG_DIR == custom_dir
            assert cfg_mod.CONFIG_FILE == custom_dir / "config.yaml"

            config = load_config()
            # All data paths should point to the custom directory
            assert config["dictionary"]["path"] == str(custom_dir / "dictionary.json")
            assert config["snippets"]["path"] == str(custom_dir / "snippets.json")
            assert config["style"]["path"] == str(custom_dir / "style_examples.json")
            assert config["history"]["path"] == str(custom_dir / "history.db")

            # Verify the directory was created
            assert custom_dir.exists()
        finally:
            cfg_mod.CONFIG_DIR = orig_config_dir
            cfg_mod.CONFIG_FILE = orig_config_file

    def test_set_data_dir_with_tilde(self):
        """set_data_dir should expand ~ in paths."""
        import whisperflow.config as cfg_mod
        from whisperflow.config import set_data_dir

        orig_config_dir = cfg_mod.CONFIG_DIR
        orig_config_file = cfg_mod.CONFIG_FILE
        try:
            set_data_dir("~/whisperflow_test_data")
            assert "~" not in str(cfg_mod.CONFIG_DIR)
            assert cfg_mod.CONFIG_DIR.is_absolute()
        finally:
            cfg_mod.CONFIG_DIR = orig_config_dir
            cfg_mod.CONFIG_FILE = orig_config_file

    def test_custom_path_not_rewritten(self):
        """Paths that don't match the default prefix should just be expanded."""
        import whisperflow.config as cfg_mod
        from whisperflow.config import load_config

        tmpdir = Path(tempfile.mkdtemp())
        custom_dir = tmpdir / "data"
        custom_dir.mkdir()

        # Write a user config with a custom path
        config_file = custom_dir / "config.yaml"
        import yaml
        config_file.write_text(yaml.dump({
            "dictionary": {"path": "/opt/whisperflow/my_dict.json"},
        }))

        orig_config_dir = cfg_mod.CONFIG_DIR
        orig_config_file = cfg_mod.CONFIG_FILE
        try:
            cfg_mod.CONFIG_DIR = custom_dir
            cfg_mod.CONFIG_FILE = config_file
            config = load_config()
            # Custom absolute path should be preserved
            assert config["dictionary"]["path"] == "/opt/whisperflow/my_dict.json"
        finally:
            cfg_mod.CONFIG_DIR = orig_config_dir
            cfg_mod.CONFIG_FILE = orig_config_file


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

    def test_remote_whisper_sends_auth_header(self):
        """Remote backend should send Authorization header when api_key is set."""
        from whisperflow.transcription.remote_whisper import RemoteWhisperBackend
        backend = RemoteWhisperBackend(server_url="http://localhost:99999", api_key="test-key")
        assert backend.api_key == "test-key"

        # Verify auth header is sent in is_available check
        with patch("whisperflow.transcription.remote_whisper.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            backend.is_available()
            mock_get.assert_called_once()
            _, kwargs = mock_get.call_args
            assert kwargs["headers"]["Authorization"] == "Bearer test-key"

    def test_remote_whisper_no_auth_without_key(self):
        """Remote backend should not send auth header when api_key is empty."""
        from whisperflow.transcription.remote_whisper import RemoteWhisperBackend
        backend = RemoteWhisperBackend(server_url="http://localhost:99999", api_key="")

        with patch("whisperflow.transcription.remote_whisper.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            backend.is_available()
            _, kwargs = mock_get.call_args
            assert "Authorization" not in kwargs.get("headers", {})


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
        data = response.json()
        assert data["status"] == "ok"
        assert data["gpu"] is True
        assert data["auth_required"] is False

    def test_health_endpoint_auth_required(self):
        from whisperflow.server.gpu_server import create_app
        from fastapi.testclient import TestClient

        app = create_app(api_key="test-secret-key")
        client = TestClient(app)
        # Without auth
        response = client.get("/health")
        assert response.status_code == 401
        # With correct auth
        response = client.get("/health", headers={"Authorization": "Bearer test-secret-key"})
        assert response.status_code == 200
        assert response.json()["auth_required"] is True
        # With wrong auth
        response = client.get("/health", headers={"Authorization": "Bearer wrong-key"})
        assert response.status_code == 403


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

    def test_import_retry(self):
        import whisperflow.retry


# ============================================================================
# Retry with Backoff
# ============================================================================

class TestRetryWithBackoff:
    def test_succeeds_immediately(self):
        from whisperflow.retry import retry_with_backoff
        result = retry_with_backoff(lambda: 42, max_retries=3)
        assert result == 42

    def test_retries_on_failure_then_succeeds(self):
        from whisperflow.retry import retry_with_backoff
        calls = [0]
        def flaky():
            calls[0] += 1
            if calls[0] < 3:
                raise ConnectionError("fail")
            return "ok"
        result = retry_with_backoff(flaky, max_retries=3, base_delay=0.01,
                                     exceptions=(ConnectionError,))
        assert result == "ok"
        assert calls[0] == 3

    def test_raises_after_max_retries(self):
        from whisperflow.retry import retry_with_backoff
        def always_fail():
            raise ConnectionError("down")
        with pytest.raises(ConnectionError):
            retry_with_backoff(always_fail, max_retries=2, base_delay=0.01,
                               exceptions=(ConnectionError,))

    def test_does_not_retry_unmatched_exception(self):
        from whisperflow.retry import retry_with_backoff
        def bad():
            raise ValueError("wrong type")
        with pytest.raises(ValueError):
            retry_with_backoff(bad, max_retries=3, base_delay=0.01,
                               exceptions=(ConnectionError,))


# ============================================================================
# Config Validation
# ============================================================================

class TestConfigValidation:
    def test_valid_config_no_warnings(self):
        from whisperflow.config import validate_config, load_config
        with patch.dict(os.environ, {"WHISPERFLOW_CONFIG_DIR": tempfile.mkdtemp()}):
            config = load_config()
            warnings = validate_config(config)
            assert len(warnings) == 0

    def test_invalid_backend(self):
        from whisperflow.config import validate_config
        config = {
            "transcription": {"backend": "invalid", "device": "auto", "compute_type": "float16"},
            "llm": {"provider": "ollama"},
            "injection": {"method": "auto"},
            "audio": {"vad_threshold": 0.5, "sample_rate": 16000},
            "ui": {"overlay_position": "top-right"},
            "hotkey": {"hold_key": "ctrl_l+space", "command_key": "ctrl_l+shift+space",
                       "whisper_mode_key": "ctrl_l+alt+w"},
        }
        warnings = validate_config(config)
        assert any("transcription.backend" in w for w in warnings)

    def test_invalid_vad_threshold(self):
        from whisperflow.config import validate_config
        config = {
            "transcription": {"backend": "local", "device": "auto", "compute_type": "float16"},
            "llm": {"provider": "ollama"},
            "injection": {"method": "auto"},
            "audio": {"vad_threshold": 2.0, "sample_rate": 16000},
            "ui": {"overlay_position": "top-right"},
            "hotkey": {"hold_key": "ctrl_l+space", "command_key": "ctrl_l+shift+space",
                       "whisper_mode_key": "ctrl_l+alt+w"},
        }
        warnings = validate_config(config)
        assert any("vad_threshold" in w for w in warnings)


# ============================================================================
# Injection Auto-Detection
# ============================================================================

class TestInjectionAutoDetect:
    def test_auto_detects_x11(self):
        from whisperflow.input.injection import TextInjector
        with patch.dict(os.environ, {"DISPLAY": ":0"}, clear=False):
            os.environ.pop("WAYLAND_DISPLAY", None)
            injector = TextInjector(method="auto")
            assert injector.method == "x11"

    def test_auto_detects_wayland(self):
        from whisperflow.input.injection import TextInjector
        with patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"}, clear=False):
            injector = TextInjector(method="auto")
            assert injector.method == "wayland"

    def test_wl_copy_maps_to_wayland(self):
        from whisperflow.input.injection import TextInjector
        injector = TextInjector(method="wl-copy")
        assert injector.method == "wayland"


# ============================================================================
# Pipeline Tone With Auto-Edit Disabled
# ============================================================================

class TestPipelineToneIndependent:
    """Verify tone matching works even when auto_edit is disabled."""

    def test_tone_applies_without_auto_edit(self):
        from whisperflow.llm.factory import NoopProvider
        from whisperflow.features.dictionary import PersonalDictionary
        from whisperflow.features.snippets import SnippetManager
        from whisperflow.features.style_learning import StyleLearner
        from whisperflow.processing.pipeline import ProcessingPipeline
        from whisperflow.input.context import WindowContext

        tmpdir = tempfile.mkdtemp()
        config = {
            "processing": {"auto_edit": False, "context_aware": True, "tone_matching": True},
            "context": {"app_tones": {"slack": "casual"}},
        }
        pipeline = ProcessingPipeline(
            llm=NoopProvider(),
            dictionary=PersonalDictionary(os.path.join(tmpdir, "d.json")),
            snippets=SnippetManager(os.path.join(tmpdir, "s.json")),
            style_learner=StyleLearner(os.path.join(tmpdir, "st.json")),
            config=config,
        )
        # Mock the active window to be slack
        with patch("whisperflow.processing.pipeline.get_active_window",
                    return_value=WindowContext("slack", "general", "Slack")):
            result = pipeline.process_dictation("hello there", "", 0.0)
            # With NoopProvider, tone_matcher.apply_tone returns text unchanged
            assert result.processed_text == "hello there"
            assert result.mode == "dictation"


# ============================================================================
# Web Server (PWA Mobile App)
# ============================================================================

class TestWebServer:
    """Tests for the WhisperFlow web server (mobile PWA backend)."""

    def test_create_app(self):
        from whisperflow.server.web_server import create_app
        app = create_app()
        assert app.title == "WhisperFlow Web"

    def test_health_endpoint_no_auth(self):
        from whisperflow.server.web_server import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["auth_required"] is False

    def test_health_endpoint_with_auth(self):
        from whisperflow.server.web_server import create_app
        from fastapi.testclient import TestClient

        app = create_app(api_key="my-secret")
        client = TestClient(app)
        # Without auth
        resp = client.get("/api/health")
        assert resp.status_code == 200  # health is public
        assert resp.json()["auth_required"] is True

    def test_dictate_requires_auth(self):
        from whisperflow.server.web_server import create_app
        from fastapi.testclient import TestClient

        app = create_app(api_key="my-secret")
        client = TestClient(app)
        resp = client.post("/api/dictate", files={"audio": ("test.wav", b"fake", "audio/wav")},
                           data={"language": ""})
        assert resp.status_code == 401

    def test_dictate_wrong_auth(self):
        from whisperflow.server.web_server import create_app
        from fastapi.testclient import TestClient

        app = create_app(api_key="my-secret")
        client = TestClient(app)
        resp = client.post("/api/dictate",
                           files={"audio": ("test.wav", b"fake", "audio/wav")},
                           data={"language": ""},
                           headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 403

    def test_dictate_correct_auth_processes_audio(self):
        """With correct auth, dictate processes audio through mocked pipeline."""
        from whisperflow.server.web_server import create_app
        from fastapi.testclient import TestClient
        from whisperflow.transcription.base import TranscriptionResult

        app = create_app(api_key="test-key")
        client = TestClient(app)
        wav_bytes = _make_wav_bytes(duration_s=0.5)

        # Mock the transcription result
        mock_result = TranscriptionResult(
            text="hello world", language="en", confidence=0.95, duration=0.5,
        )
        mock_pipeline_result = MagicMock()
        mock_pipeline_result.processed_text = "Hello world."
        mock_pipeline_result.mode = "dictation"

        with patch("whisperflow.server.web_server._convert_to_wav", return_value=wav_bytes):
            # Patch _init_components to set up mocked state
            def _mock_init(state):
                def init():
                    if state["config"] is not None:
                        return
                    state["config"] = {
                        "history": {"enabled": False},
                        "dictionary": {"auto_learn": False},
                    }
                    state["transcriber"] = MagicMock()
                    state["transcriber"].transcribe.return_value = mock_result
                    state["dictionary"] = MagicMock()
                    state["dictionary"].get_whisper_prompt.return_value = ""
                    state["pipeline"] = MagicMock()
                    state["pipeline"].process_dictation.return_value = mock_pipeline_result
                return init

            # Access the _state dict through the closure
            # We need to find and replace _init_components
            # Instead, patch at the module level and control the state dict
            import whisperflow.server.web_server as ws_mod

            # The _state dict is created inside create_app, so we need
            # a different approach: just set _state values before the call
            # Simplest: re-create app and intercept
            app2 = create_app(api_key="test-key")
            client2 = TestClient(app2)

            # Inject mocked components by calling dictate and mocking load_config
            with patch("whisperflow.config.load_config") as mock_lc, \
                 patch("whisperflow.llm.factory.create_llm_provider") as mock_llm, \
                 patch("whisperflow.features.dictionary.PersonalDictionary") as mock_dict_cls, \
                 patch("whisperflow.features.snippets.SnippetManager") as mock_snip, \
                 patch("whisperflow.features.style_learning.StyleLearner") as mock_style, \
                 patch("whisperflow.features.history.TranscriptionHistory") as mock_hist, \
                 patch("whisperflow.processing.pipeline.ProcessingPipeline") as mock_pipe_cls, \
                 patch("whisperflow.transcription.local_whisper.LocalWhisperBackend") as mock_whisper:

                mock_lc.return_value = {
                    "transcription": {"backend": "local", "model_size": "tiny",
                                      "device": "cpu", "compute_type": "float32"},
                    "llm": {"provider": "none"},
                    "dictionary": {"path": "/tmp/d.json", "auto_learn": False},
                    "snippets": {"path": "/tmp/s.json"},
                    "style": {"path": "/tmp/st.json", "max_examples": 10},
                    "history": {"path": "/tmp/h.db", "max_entries": 100, "enabled": False},
                }
                mock_whisper_inst = MagicMock()
                mock_whisper_inst.transcribe.return_value = mock_result
                mock_whisper.return_value = mock_whisper_inst

                mock_dict_inst = MagicMock()
                mock_dict_inst.get_whisper_prompt.return_value = ""
                mock_dict_cls.return_value = mock_dict_inst

                mock_pipe_inst = MagicMock()
                mock_pipe_inst.process_dictation.return_value = mock_pipeline_result
                mock_pipe_cls.return_value = mock_pipe_inst

                resp = client2.post(
                    "/api/dictate",
                    files={"audio": ("test.wav", wav_bytes, "audio/wav")},
                    data={"language": ""},
                    headers={"Authorization": "Bearer test-key"},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["text"] == "Hello world."
                assert data["raw_text"] == "hello world"
                assert data["mode"] == "dictation"

    def test_convert_to_wav_passthrough(self):
        """WAV input should pass through without conversion."""
        from whisperflow.server.web_server import _convert_to_wav
        wav_bytes = _make_wav_bytes(duration_s=0.1)
        result = _convert_to_wav(wav_bytes)
        assert result == wav_bytes

    def test_convert_to_wav_detects_wav_header(self):
        """Should detect RIFF/WAVE header and skip conversion."""
        from whisperflow.server.web_server import _convert_to_wav
        wav_bytes = _make_wav_bytes(duration_s=0.1)
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"
        result = _convert_to_wav(wav_bytes)
        assert result[:4] == b"RIFF"

    def test_convert_to_wav_non_wav_needs_ffmpeg(self):
        """Non-WAV input should use ffmpeg for conversion."""
        from whisperflow.server.web_server import _convert_to_wav
        fake_webm = b"\x1a\x45\xdf\xa3" + b"\x00" * 100  # WebM magic bytes

        with patch("whisperflow.server.web_server.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="ffmpeg"):
                _convert_to_wav(fake_webm)

    def test_convert_to_wav_ffmpeg_conversion(self):
        """Test ffmpeg-based conversion with mocked subprocess."""
        from whisperflow.server.web_server import _convert_to_wav
        fake_webm = b"\x1a\x45\xdf\xa3" + b"\x00" * 100
        wav_output = _make_wav_bytes(duration_s=0.1)

        with patch("whisperflow.server.web_server.shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("whisperflow.server.web_server.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                # Mock the file read after conversion
                with patch("builtins.open", side_effect=[
                    # First open: write temp in file (handled by NamedTemporaryFile)
                    # We need to mock the final read
                    MagicMock(),
                ]):
                    with patch("whisperflow.server.web_server.tempfile.NamedTemporaryFile") as mock_tmp:
                        mock_tmp_instance = MagicMock()
                        mock_tmp_instance.__enter__ = MagicMock(return_value=mock_tmp_instance)
                        mock_tmp_instance.__exit__ = MagicMock(return_value=False)
                        mock_tmp_instance.name = "/tmp/test_input.webm"
                        mock_tmp.return_value = mock_tmp_instance

                        # Mock reading the output file
                        mock_open = MagicMock()
                        mock_file = MagicMock()
                        mock_file.read.return_value = wav_output
                        mock_file.__enter__ = MagicMock(return_value=mock_file)
                        mock_file.__exit__ = MagicMock(return_value=False)

                        with patch("builtins.open", return_value=mock_file):
                            with patch("os.unlink"):
                                result = _convert_to_wav(fake_webm)
                                assert result == wav_output
                                # Verify ffmpeg was called with right args
                                mock_run.assert_called_once()
                                call_args = mock_run.call_args[0][0]
                                assert call_args[0] == "ffmpeg"
                                assert "-ar" in call_args
                                assert "16000" in call_args

    def test_static_files_exist(self):
        """Verify all required PWA static files exist."""
        from whisperflow.server.web_server import STATIC_DIR
        assert (STATIC_DIR / "index.html").exists()
        assert (STATIC_DIR / "style.css").exists()
        assert (STATIC_DIR / "app.js").exists()
        assert (STATIC_DIR / "manifest.json").exists()
        assert (STATIC_DIR / "sw.js").exists()
        assert (STATIC_DIR / "icon-192.png").exists()
        assert (STATIC_DIR / "icon-512.png").exists()

    def test_service_worker_caches_icons(self):
        """Service worker should cache all PWA icons for offline use."""
        from whisperflow.server.web_server import STATIC_DIR
        sw_content = (STATIC_DIR / "sw.js").read_text()
        assert "icon-192.png" in sw_content
        assert "icon-512.png" in sw_content

    def test_index_endpoint(self):
        from whisperflow.server.web_server import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "WhisperFlow" in resp.text

    def test_manifest_endpoint(self):
        from whisperflow.server.web_server import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)
        resp = client.get("/manifest.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "WhisperFlow"

    def test_service_worker_endpoint(self):
        from whisperflow.server.web_server import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)
        resp = client.get("/sw.js")
        assert resp.status_code == 200
        assert "caches" in resp.text or "cache" in resp.text

    def test_transcribe_requires_auth(self):
        from whisperflow.server.web_server import create_app
        from fastapi.testclient import TestClient

        app = create_app(api_key="secret")
        client = TestClient(app)
        resp = client.post("/api/transcribe",
                           files={"audio": ("test.wav", b"fake", "audio/wav")},
                           data={"language": ""})
        assert resp.status_code == 401

    def test_snippets_crud_no_auth(self):
        """Snippets CRUD works without auth when no key is set."""
        from whisperflow.server.web_server import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        # Mock components init
        with patch("whisperflow.config.load_config") as mock_lc, \
             patch("whisperflow.llm.factory.create_llm_provider"), \
             patch("whisperflow.features.dictionary.PersonalDictionary"), \
             patch("whisperflow.features.snippets.SnippetManager") as mock_snip_cls, \
             patch("whisperflow.features.style_learning.StyleLearner"), \
             patch("whisperflow.features.history.TranscriptionHistory"), \
             patch("whisperflow.processing.pipeline.ProcessingPipeline"), \
             patch("whisperflow.transcription.local_whisper.LocalWhisperBackend"):
            mock_lc.return_value = {
                "transcription": {"backend": "local", "model_size": "tiny",
                                  "device": "cpu", "compute_type": "float32"},
                "llm": {"provider": "none"},
                "dictionary": {"path": "/tmp/d.json", "auto_learn": False},
                "snippets": {"path": "/tmp/s.json"},
                "style": {"path": "/tmp/st.json", "max_examples": 10},
                "history": {"path": "/tmp/h.db", "max_entries": 100, "enabled": False},
            }
            mock_snip_inst = MagicMock()
            mock_snip_inst.get_snippets.return_value = {"my email": {"content": "user@test.com"}}
            mock_snip_cls.return_value = mock_snip_inst

            resp = client.get("/api/snippets")
            assert resp.status_code == 200
            data = resp.json()
            assert "my email" in data["snippets"]

    def test_dictionary_crud_no_auth(self):
        """Dictionary CRUD works without auth when no key is set."""
        from whisperflow.server.web_server import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        with patch("whisperflow.config.load_config") as mock_lc, \
             patch("whisperflow.llm.factory.create_llm_provider"), \
             patch("whisperflow.features.dictionary.PersonalDictionary") as mock_dict_cls, \
             patch("whisperflow.features.snippets.SnippetManager"), \
             patch("whisperflow.features.style_learning.StyleLearner"), \
             patch("whisperflow.features.history.TranscriptionHistory"), \
             patch("whisperflow.processing.pipeline.ProcessingPipeline"), \
             patch("whisperflow.transcription.local_whisper.LocalWhisperBackend"):
            mock_lc.return_value = {
                "transcription": {"backend": "local", "model_size": "tiny",
                                  "device": "cpu", "compute_type": "float32"},
                "llm": {"provider": "none"},
                "dictionary": {"path": "/tmp/d.json", "auto_learn": False},
                "snippets": {"path": "/tmp/s.json"},
                "style": {"path": "/tmp/st.json", "max_examples": 10},
                "history": {"path": "/tmp/h.db", "max_entries": 100, "enabled": False},
            }
            mock_dict_inst = MagicMock()
            mock_dict_inst.get_words.return_value = {"WhisperFlow": "product"}
            mock_dict_cls.return_value = mock_dict_inst

            resp = client.get("/api/dictionary")
            assert resp.status_code == 200
            data = resp.json()
            assert "WhisperFlow" in data["words"]

    def test_history_search_endpoint(self):
        """History search endpoint returns filtered results."""
        from whisperflow.server.web_server import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        with patch("whisperflow.config.load_config") as mock_lc, \
             patch("whisperflow.llm.factory.create_llm_provider"), \
             patch("whisperflow.features.dictionary.PersonalDictionary"), \
             patch("whisperflow.features.snippets.SnippetManager"), \
             patch("whisperflow.features.style_learning.StyleLearner"), \
             patch("whisperflow.features.history.TranscriptionHistory") as mock_hist_cls, \
             patch("whisperflow.processing.pipeline.ProcessingPipeline"), \
             patch("whisperflow.transcription.local_whisper.LocalWhisperBackend"):
            mock_lc.return_value = {
                "transcription": {"backend": "local", "model_size": "tiny",
                                  "device": "cpu", "compute_type": "float32"},
                "llm": {"provider": "none"},
                "dictionary": {"path": "/tmp/d.json", "auto_learn": False},
                "snippets": {"path": "/tmp/s.json"},
                "style": {"path": "/tmp/st.json", "max_examples": 10},
                "history": {"path": "/tmp/h.db", "max_entries": 100, "enabled": False},
            }
            mock_hist_inst = MagicMock()
            mock_hist_inst.search.return_value = [{"raw_text": "hello", "processed_text": "Hello."}]
            mock_hist_cls.return_value = mock_hist_inst

            resp = client.get("/api/history/search?q=hello")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["entries"]) == 1

    def test_history_search_empty_query(self):
        from whisperflow.server.web_server import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/history/search?q=")
        assert resp.status_code == 200
        assert resp.json()["entries"] == []

    def test_new_tabs_in_index(self):
        """Verify new tabs appear in the HTML."""
        from whisperflow.server.web_server import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)
        resp = client.get("/")
        assert "Transcribe" in resp.text
        assert "Snippets" in resp.text
        assert "Dictionary" in resp.text
        assert "theme-toggle" in resp.text

    def test_command_requires_auth(self):
        from whisperflow.server.web_server import create_app
        from fastapi.testclient import TestClient

        app = create_app(api_key="cmd-secret")
        client = TestClient(app)
        resp = client.post("/api/command",
                           files={"audio": ("test.wav", b"fake", "audio/wav")},
                           data={"language": "", "selected_text": ""})
        assert resp.status_code == 401

    def test_command_endpoint_processes_audio(self):
        """Command endpoint processes audio through mocked pipeline."""
        from whisperflow.server.web_server import create_app
        from fastapi.testclient import TestClient
        from whisperflow.transcription.base import TranscriptionResult

        app = create_app()
        client = TestClient(app)
        wav_bytes = _make_wav_bytes(duration_s=0.5)

        mock_result = TranscriptionResult(
            text="what is Python", language="en", confidence=0.95, duration=0.5,
        )

        with patch("whisperflow.server.web_server._convert_to_wav", return_value=wav_bytes), \
             patch("whisperflow.config.load_config") as mock_lc, \
             patch("whisperflow.llm.factory.create_llm_provider"), \
             patch("whisperflow.features.dictionary.PersonalDictionary") as mock_dict_cls, \
             patch("whisperflow.features.snippets.SnippetManager"), \
             patch("whisperflow.features.style_learning.StyleLearner"), \
             patch("whisperflow.features.history.TranscriptionHistory"), \
             patch("whisperflow.processing.pipeline.ProcessingPipeline"), \
             patch("whisperflow.transcription.local_whisper.LocalWhisperBackend") as mock_whisper:

            mock_lc.return_value = {
                "transcription": {"backend": "local", "model_size": "tiny",
                                  "device": "cpu", "compute_type": "float32"},
                "llm": {"provider": "none"},
                "dictionary": {"path": "/tmp/d.json", "auto_learn": False},
                "snippets": {"path": "/tmp/s.json"},
                "style": {"path": "/tmp/st.json", "max_examples": 10},
                "history": {"path": "/tmp/h.db", "max_entries": 100, "enabled": False},
            }
            mock_whisper_inst = MagicMock()
            mock_whisper_inst.transcribe.return_value = mock_result
            mock_whisper.return_value = mock_whisper_inst

            mock_dict_inst = MagicMock()
            mock_dict_inst.get_whisper_prompt.return_value = ""
            mock_dict_cls.return_value = mock_dict_inst

            # Mock CommandMode to return a question answer
            with patch("whisperflow.processing.command_mode.CommandMode") as mock_cmd:
                mock_cmd_inst = MagicMock()
                mock_cmd_inst.classify_intent.return_value = "question"
                mock_cmd_inst.answer_question.return_value = "Python is a programming language"
                mock_cmd.return_value = mock_cmd_inst

                resp = client.post(
                    "/api/command",
                    files={"audio": ("test.wav", wav_bytes, "audio/wav")},
                    data={"language": "", "selected_text": ""},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["mode"] == "question"
                assert data["text"] == "Python is a programming language"

    def test_stats_tab_in_index(self):
        from whisperflow.server.web_server import create_app
        from fastapi.testclient import TestClient
        app = create_app()
        client = TestClient(app)
        resp = client.get("/")
        assert "Stats" in resp.text
        assert "stats-panel" in resp.text
        assert "backup-btn" in resp.text

    def test_history_export_empty(self):
        """History export returns empty when no entries."""
        from whisperflow.server.web_server import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        with patch("whisperflow.config.load_config") as mock_lc, \
             patch("whisperflow.llm.factory.create_llm_provider"), \
             patch("whisperflow.features.dictionary.PersonalDictionary"), \
             patch("whisperflow.features.snippets.SnippetManager"), \
             patch("whisperflow.features.style_learning.StyleLearner"), \
             patch("whisperflow.features.history.TranscriptionHistory") as mock_hist_cls, \
             patch("whisperflow.processing.pipeline.ProcessingPipeline"), \
             patch("whisperflow.transcription.local_whisper.LocalWhisperBackend"):
            mock_lc.return_value = {
                "transcription": {"backend": "local", "model_size": "tiny",
                                  "device": "cpu", "compute_type": "float32"},
                "llm": {"provider": "none"},
                "dictionary": {"path": "/tmp/d.json", "auto_learn": False},
                "snippets": {"path": "/tmp/s.json"},
                "style": {"path": "/tmp/st.json", "max_examples": 10},
                "history": {"path": "/tmp/h.db", "max_entries": 100, "enabled": False},
            }
            mock_hist = MagicMock()
            mock_hist.export_json.return_value = "[]"
            mock_hist.export_csv.return_value = "id,timestamp\n"
            mock_hist_cls.return_value = mock_hist

            resp = client.get("/api/history/export?format=json")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "application/json"

            resp = client.get("/api/history/export?format=csv")
            assert resp.status_code == 200
            assert "text/csv" in resp.headers["content-type"]

    def test_stats_endpoint(self):
        """Stats endpoint returns expected fields."""
        from whisperflow.server.web_server import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        with patch("whisperflow.config.load_config") as mock_lc, \
             patch("whisperflow.llm.factory.create_llm_provider"), \
             patch("whisperflow.features.dictionary.PersonalDictionary") as mock_dict_cls, \
             patch("whisperflow.features.snippets.SnippetManager") as mock_snip_cls, \
             patch("whisperflow.features.style_learning.StyleLearner"), \
             patch("whisperflow.features.history.TranscriptionHistory") as mock_hist_cls, \
             patch("whisperflow.processing.pipeline.ProcessingPipeline"), \
             patch("whisperflow.transcription.local_whisper.LocalWhisperBackend"):
            mock_lc.return_value = {
                "transcription": {"backend": "local", "model_size": "tiny",
                                  "device": "cpu", "compute_type": "float32"},
                "llm": {"provider": "none"},
                "dictionary": {"path": "/tmp/d.json", "auto_learn": False},
                "snippets": {"path": "/tmp/s.json"},
                "style": {"path": "/tmp/st.json", "max_examples": 10},
                "history": {"path": "/tmp/h.db", "max_entries": 100, "enabled": False},
            }
            mock_hist = MagicMock()
            mock_hist.get_stats.return_value = {
                "total_transcriptions": 42, "total_duration_seconds": 120.5,
                "total_words": 1000, "modes": {"dictation": 30, "command": 12},
                "languages": {"en": 35, "es": 7},
                "last_7_days": {"transcriptions": 10, "duration_seconds": 50.0},
            }
            mock_hist_cls.return_value = mock_hist
            mock_dict_cls.return_value.get_words.return_value = {"test": "general"}
            mock_snip_cls.return_value.get_snippets.return_value = {"hi": "hello"}

            resp = client.get("/api/stats")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_transcriptions"] == 42
            assert data["dictionary_words"] == 1
            assert data["snippets_count"] == 1

    def test_rate_limit_middleware(self):
        """Rate limiter returns 429 after exceeding limit."""
        from whisperflow.server.web_server import _rate_buckets, _check_rate_limit
        _rate_buckets.clear()
        # Simulate many requests
        for _ in range(60):
            assert not _check_rate_limit("test-ip")
        # Next one should be limited
        assert _check_rate_limit("test-ip")
        _rate_buckets.clear()

    def test_import_web_server(self):
        import whisperflow.server.web_server


class TestHistoryAudioPath:
    def test_history_entry_has_audio_path(self):
        from whisperflow.features.history import HistoryEntry
        entry = HistoryEntry(
            id=1, timestamp=0, raw_text="test", processed_text="test",
            app_context="", mode="dictation", language="en", duration=1.0,
            audio_path="/tmp/test.wav",
        )
        assert entry.audio_path == "/tmp/test.wav"

    def test_history_stats(self):
        import tempfile
        from whisperflow.features.history import TranscriptionHistory
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            h = TranscriptionHistory(path, max_entries=100)
            h.add("hello world", "Hello world.", mode="dictation", language="en", duration=2.0)
            h.add("command test", "command result", mode="command", language="en", duration=1.5)
            stats = h.get_stats()
            assert stats["total_transcriptions"] == 2
            assert stats["total_duration_seconds"] == 3.5
            assert stats["total_words"] == 4
            assert stats["modes"]["dictation"] == 1
            assert stats["modes"]["command"] == 1
        finally:
            import os
            os.unlink(path)

    def test_history_export_csv(self):
        import tempfile
        from whisperflow.features.history import TranscriptionHistory
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            h = TranscriptionHistory(path, max_entries=100)
            h.add("hello", "Hello.", mode="dictation", language="en", duration=1.0)
            csv_data = h.export_csv()
            assert "hello" in csv_data
            assert "Hello." in csv_data
            assert csv_data.startswith("id,timestamp,raw_text")
        finally:
            import os
            os.unlink(path)

    def test_history_export_json(self):
        import tempfile
        import json as _json
        from whisperflow.features.history import TranscriptionHistory
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            h = TranscriptionHistory(path, max_entries=100)
            h.add("hello", "Hello.", mode="dictation", language="en", duration=1.0)
            json_data = h.export_json()
            entries = _json.loads(json_data)
            assert len(entries) == 1
            assert entries[0]["raw_text"] == "hello"
        finally:
            import os
            os.unlink(path)


def _make_wav_bytes(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Create a valid WAV file with silence for testing."""
    num_samples = int(duration_s * sample_rate)
    samples = np.zeros(num_samples, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()

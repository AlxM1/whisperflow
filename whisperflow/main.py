"""WhisperFlow - Main application entry point.

A local AI-powered voice dictation tool with auto-editing,
context awareness, command mode, and more.
"""

import logging
import os
import signal
import sys
import threading
import time

from .config import load_config, save_config
from .audio.capture import AudioCapture
from .transcription.base import TranscriptionResult
from .llm.factory import create_llm_provider, NoopProvider
from .features.dictionary import PersonalDictionary
from .features.snippets import SnippetManager
from .features.style_learning import StyleLearner
from .features.history import TranscriptionHistory
from .processing.pipeline import ProcessingPipeline
from .input.injection import TextInjector
from .input.hotkey import HotkeyManager
from .cortex_client import get_cortex_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Transcription timeout in seconds
TRANSCRIPTION_TIMEOUT = 60


class WhisperFlowApp:
    """Main application class orchestrating all components."""

    def __init__(self):
        self.config = load_config()
        self._whisper_mode = False
        self._command_mode_active = False
        self._running = False
        self._vad = None

        # Initialize Cortex client
        self.cortex = get_cortex_client()

        # Initialize components
        logger.info("Initializing WhisperFlow...")

        # Audio
        self.audio = AudioCapture(
            sample_rate=self.config["audio"]["sample_rate"],
            channels=self.config["audio"]["channels"],
        )

        # VAD (Voice Activity Detection) — optional, loaded lazily
        self._vad_threshold = self.config["audio"].get("vad_threshold", 0.5)
        self._min_speech_duration = self.config["audio"].get("min_speech_duration", 0.3)

        # Transcription backend
        self.transcriber = self._create_transcriber()

        # LLM
        self.llm = create_llm_provider(self.config["llm"])
        if isinstance(self.llm, NoopProvider) and self.config["llm"].get("provider", "none") != "none":
            logger.warning(
                "LLM provider '%s' is not available — AI features (auto-edit, "
                "tone matching, command mode) will pass text through unchanged",
                self.config["llm"].get("provider"),
            )

        # Features
        self.dictionary = PersonalDictionary(self.config["dictionary"]["path"])
        self.snippets = SnippetManager(self.config["snippets"]["path"])
        self.style_learner = StyleLearner(
            self.config["style"]["path"],
            max_examples=self.config["style"]["max_examples"],
        )
        self.history = TranscriptionHistory(
            self.config["history"]["path"],
            max_entries=self.config["history"]["max_entries"],
        )

        # Processing pipeline
        self.pipeline = ProcessingPipeline(
            llm=self.llm,
            dictionary=self.dictionary,
            snippets=self.snippets,
            style_learner=self.style_learner,
            config=self.config,
        )

        # Text injection
        self.injector = TextInjector(
            method=self.config["injection"]["method"],
            delay_ms=self.config["injection"]["delay_ms"],
        )

        # Hotkey manager
        self.hotkey = HotkeyManager(
            hold_key=self.config["hotkey"]["hold_key"],
            command_key=self.config["hotkey"]["command_key"],
            whisper_mode_key=self.config["hotkey"]["whisper_mode_key"],
            on_hold_start=self._on_hold_start,
            on_hold_end=self._on_hold_end,
            on_command_start=self._on_command_start,
            on_command_end=self._on_command_end,
            on_whisper_toggle=self._on_whisper_toggle,
        )

        # UI components (lazy-loaded)
        self._tray = None
        self._overlay = None
        self._qt_app = None

        logger.info("WhisperFlow initialized")

    def _create_transcriber(self):
        """Create the appropriate transcription backend."""
        trans_cfg = self.config["transcription"]

        if trans_cfg["backend"] == "remote":
            from .transcription.remote_whisper import RemoteWhisperBackend
            api_key = trans_cfg.get("remote_api_key", "") or os.environ.get("WHISPERFLOW_API_KEY", "")
            backend = RemoteWhisperBackend(server_url=trans_cfg["remote_url"], api_key=api_key)
            if backend.is_available():
                logger.info(f"Using remote transcription: {trans_cfg['remote_url']}")
                return backend
            logger.warning("Remote backend unavailable, falling back to local")

        from .transcription.local_whisper import LocalWhisperBackend
        return LocalWhisperBackend(
            model_size=trans_cfg["model_size"],
            device=trans_cfg["device"],
            compute_type=trans_cfg["compute_type"],
        )

    # -- Hotkey callbacks --

    def _on_hold_start(self):
        """Called when hold-to-talk key is pressed."""
        logger.info("Hold-to-talk: START")
        try:
            self.audio.start_recording()
        except Exception:
            if self._overlay:
                self._overlay.set_mode("error", "Microphone error")
            return
        if self._overlay:
            self._overlay.set_mode("recording")
        if self._tray:
            self._tray.set_recording(True)

    def _on_hold_end(self):
        """Called when hold-to-talk key is released."""
        logger.info("Hold-to-talk: END")
        audio_data = self.audio.stop_recording()
        if self._tray:
            self._tray.set_recording(False)

        if not audio_data:
            if self._overlay:
                self._overlay.set_mode("idle")
            return

        # Check for speech using VAD before burning transcription resources
        if not self._check_vad(audio_data):
            logger.info("No speech detected by VAD, skipping transcription")
            if self._overlay:
                self._overlay.set_mode("idle")
            return

        if self._overlay:
            self._overlay.set_mode("processing")

        # Process in background thread
        threading.Thread(
            target=self._process_audio,
            args=(audio_data, False),
            daemon=True,
        ).start()

    def _on_command_start(self):
        """Called when command mode key is pressed."""
        logger.info("Command mode: START")
        self._command_mode_active = True
        try:
            self.audio.start_recording()
        except Exception:
            self._command_mode_active = False
            if self._overlay:
                self._overlay.set_mode("error", "Microphone error")
            return
        if self._overlay:
            self._overlay.set_mode("command")

    def _on_command_end(self):
        """Called when command mode key is released."""
        logger.info("Command mode: END")
        audio_data = self.audio.stop_recording()
        self._command_mode_active = False

        if not audio_data:
            if self._overlay:
                self._overlay.set_mode("idle")
            return

        if not self._check_vad(audio_data):
            logger.info("No speech detected by VAD, skipping transcription")
            if self._overlay:
                self._overlay.set_mode("idle")
            return

        if self._overlay:
            self._overlay.set_mode("processing")

        threading.Thread(
            target=self._process_audio,
            args=(audio_data, True),
            daemon=True,
        ).start()

    def _on_whisper_toggle(self):
        """Toggle whisper mode (for quiet environments)."""
        self._whisper_mode = not self._whisper_mode
        state = "ON" if self._whisper_mode else "OFF"
        logger.info(f"Whisper mode: {state}")
        if self._tray:
            self._tray.show_notification("WhisperFlow", f"Whisper mode: {state}")

    # -- VAD --

    def _get_vad(self):
        """Lazy-load VAD model on first use."""
        if self._vad is None:
            try:
                from .audio.vad import VoiceActivityDetector
                self._vad = VoiceActivityDetector(threshold=self._vad_threshold)
                self._vad.load()
            except Exception as e:
                logger.warning(f"VAD unavailable, skipping speech detection: {e}")
                self._vad = False  # sentinel: tried and failed
        return self._vad

    def _check_vad(self, wav_bytes: bytes) -> bool:
        """Check if audio contains speech using VAD. Returns True if speech found or VAD unavailable."""
        vad = self._get_vad()
        if not vad:
            return True  # VAD not available, assume speech present

        try:
            import io
            import wave
            import numpy as np

            buf = io.BytesIO(wav_bytes)
            with wave.open(buf, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                sample_rate = wf.getframerate()

            audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

            # Check if total duration is below minimum
            duration = len(audio_np) / sample_rate
            if duration < self._min_speech_duration:
                return False

            # Check chunks for speech (512 samples = 32ms at 16kHz)
            chunk_size = 512
            speech_frames = 0
            total_frames = 0
            vad.reset()
            for i in range(0, len(audio_np) - chunk_size, chunk_size):
                chunk = audio_np[i:i + chunk_size]
                total_frames += 1
                if vad.is_speech(chunk):
                    speech_frames += 1

            if total_frames == 0:
                return False

            speech_ratio = speech_frames / total_frames
            has_speech = speech_ratio > 0.1  # At least 10% of frames have speech
            logger.debug(f"VAD: {speech_frames}/{total_frames} frames with speech ({speech_ratio:.1%})")
            return has_speech

        except Exception as e:
            logger.warning(f"VAD check failed: {e}")
            return True  # Fail open

    # -- Audio processing --

    def _process_audio(self, audio_data: bytes, is_command: bool):
        """Process captured audio through the full pipeline."""
        transcription_start = time.time()
        try:
            # Step 1: Transcribe (with timeout)
            initial_prompt = self.dictionary.get_whisper_prompt()
            lang_cfg = self.config["transcription"]["language"]
            language = lang_cfg if lang_cfg != "auto" else None

            result = self._transcribe_with_timeout(
                audio_data, language, initial_prompt or None
            )

            if result is None:
                logger.error("Transcription timed out")
                self.cortex.report_transcription(
                    mode="command" if is_command else "dictation",
                    duration=0.0,
                    backend=self.config["transcription"]["backend"],
                    success=False,
                    error="Transcription timeout",
                )
                if self._overlay:
                    self._overlay.set_mode("error", "Transcription timeout")
                return

            if result.is_empty:
                logger.info("Empty transcription, skipping")
                if self._overlay:
                    self._overlay.set_mode("idle")
                return

            logger.info(f"Raw transcription: '{result.text}'")

            # Step 2: Process through pipeline
            if is_command:
                pipeline_result = self.pipeline.process_command(
                    result.text, result.language, result.duration
                )
            else:
                pipeline_result = self.pipeline.process_dictation(
                    result.text, result.language, result.duration
                )

            logger.info(f"Processed text ({pipeline_result.mode}): '{pipeline_result.processed_text}'")

            # Step 3: Inject text into active app
            if pipeline_result.processed_text:
                success = self.injector.inject(pipeline_result.processed_text)
                if not success:
                    logger.error("Text injection failed")
                    if self._overlay:
                        self._overlay.set_mode("error", "Injection failed")
                    return

            # Step 4: Save to history
            if self.config["history"]["enabled"]:
                self.history.add(
                    raw_text=result.text,
                    processed_text=pipeline_result.processed_text,
                    app_context=pipeline_result.context.detected_app,
                    mode=pipeline_result.mode,
                    language=result.language,
                    duration=result.duration,
                )

            # Step 5: Auto-learn dictionary from transcription vs processed
            if self.config["dictionary"].get("auto_learn", True):
                self.dictionary.check_and_learn(result.text, pipeline_result.processed_text)

            # Step 6: Report to Cortex
            transcription_elapsed = time.time() - transcription_start
            self.cortex.report_transcription(
                mode=pipeline_result.mode,
                duration=result.duration,
                language=result.language,
                backend=self.config["transcription"]["backend"],
                success=True,
                metadata={
                    "model": self.config["transcription"]["model_size"],
                    "processing_time": round(transcription_elapsed, 2),
                    "app_context": pipeline_result.context.detected_app,
                    "text_length": len(pipeline_result.processed_text),
                },
            )

        except Exception as e:
            logger.exception("Error processing audio")
            self.cortex.report_error(
                error_type="processing",
                message=str(e),
                stack_trace="",
                context={"is_command": is_command},
            )
            if self._overlay:
                self._overlay.set_mode("error", "Processing error")
                return
        finally:
            # Only reset to idle if we didn't set error above
            if self._overlay and self._overlay._mode != "error":
                self._overlay.set_mode("idle")

    def _transcribe_with_timeout(self, audio_data, language, initial_prompt):
        """Run transcription with a timeout. Returns None on timeout."""
        result_holder = [None]
        error_holder = [None]

        def _do_transcribe():
            try:
                result_holder[0] = self.transcriber.transcribe(
                    audio_data,
                    language=language,
                    initial_prompt=initial_prompt,
                )
            except Exception as e:
                error_holder[0] = e

        t = threading.Thread(target=_do_transcribe, daemon=True)
        t.start()
        t.join(timeout=TRANSCRIPTION_TIMEOUT)

        if t.is_alive():
            logger.error(f"Transcription exceeded {TRANSCRIPTION_TIMEOUT}s timeout")
            return None

        if error_holder[0]:
            raise error_holder[0]

        return result_holder[0]

    # -- Settings UI callbacks --

    def _show_settings(self):
        from .ui.tray import SettingsDialog
        dialog = SettingsDialog(self.config)
        if dialog.exec():
            self.config = dialog.get_config()
            save_config(self.config)
            logger.info("Settings saved")

    def _show_dictionary(self):
        from .ui.tray import DictionaryDialog
        dialog = DictionaryDialog(self.dictionary)
        dialog.exec()

    def _show_snippets(self):
        from .ui.tray import SnippetsDialog
        dialog = SnippetsDialog(self.snippets)
        dialog.exec()

    def _show_history(self):
        from .ui.tray import HistoryDialog
        dialog = HistoryDialog(self.history)
        dialog.exec()

    # -- App lifecycle --

    def run(self):
        """Run the application."""
        self._running = True

        # Try to start with GUI
        try:
            from PyQt6.QtWidgets import QApplication
            from .ui.tray import SystemTray
            from .ui.overlay import RecordingOverlay

            self._qt_app = QApplication(sys.argv)
            self._qt_app.setQuitOnLastWindowClosed(False)

            # System tray
            self._tray = SystemTray(self)
            self._tray.show_settings.connect(self._show_settings)
            self._tray.show_dictionary.connect(self._show_dictionary)
            self._tray.show_snippets.connect(self._show_snippets)
            self._tray.show_history.connect(self._show_history)
            self._tray.quit_app.connect(self.quit)

            # Recording overlay
            if self.config["ui"].get("show_overlay", True):
                self._overlay = RecordingOverlay(
                    position=self.config["ui"].get("overlay_position", "top-right")
                )

            # Start hotkey listener
            self.hotkey.start()

            logger.info("WhisperFlow is running. Hold Ctrl+Space to dictate.")
            if self._tray:
                self._tray.show_notification(
                    "WhisperFlow",
                    "Voice dictation ready. Hold Ctrl+Space to speak."
                )

            # Handle SIGINT gracefully
            signal.signal(signal.SIGINT, lambda *_: self.quit())

            sys.exit(self._qt_app.exec())

        except ImportError:
            # Headless mode (no PyQt6)
            logger.info("Running in headless mode (no PyQt6)")
            self.hotkey.start()

            logger.info("WhisperFlow is running. Hold Ctrl+Space to dictate. Ctrl+C to quit.")
            try:
                signal.pause()
            except KeyboardInterrupt:
                self.quit()

    def quit(self):
        """Shut down the application."""
        logger.info("Shutting down WhisperFlow...")
        self._running = False
        self.hotkey.stop()
        if self._qt_app:
            self._qt_app.quit()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="WhisperFlow - AI voice dictation")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("WHISPERFLOW_DATA_DIR", ""),
        help="Override data directory for all storage (config, history, dictionary, etc.). "
             "Defaults to ~/.config/whisperflow/ or WHISPERFLOW_DATA_DIR env var.",
    )
    args = parser.parse_args()

    if args.data_dir:
        from .config import set_data_dir
        set_data_dir(args.data_dir)

    app = WhisperFlowApp()
    app.run()


if __name__ == "__main__":
    main()

"""WhisperFlow - Main application entry point.

A local AI-powered voice dictation tool with auto-editing,
context awareness, command mode, and more.
"""

import logging
import signal
import sys
import threading

from .config import load_config, save_config
from .audio.capture import AudioCapture
from .transcription.base import TranscriptionResult
from .llm.factory import create_llm_provider
from .features.dictionary import PersonalDictionary
from .features.snippets import SnippetManager
from .features.style_learning import StyleLearner
from .features.history import TranscriptionHistory
from .processing.pipeline import ProcessingPipeline
from .input.injection import TextInjector
from .input.hotkey import HotkeyManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class WhisperFlowApp:
    """Main application class orchestrating all components."""

    def __init__(self):
        self.config = load_config()
        self._whisper_mode = False
        self._command_mode_active = False
        self._running = False

        # Initialize components
        logger.info("Initializing WhisperFlow...")

        # Audio
        self.audio = AudioCapture(
            sample_rate=self.config["audio"]["sample_rate"],
            channels=self.config["audio"]["channels"],
        )

        # Transcription backend
        self.transcriber = self._create_transcriber()

        # LLM
        self.llm = create_llm_provider(self.config["llm"])

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
            backend = RemoteWhisperBackend(server_url=trans_cfg["remote_url"])
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
        self.audio.start_recording()
        if self._overlay:
            self._overlay.set_mode("recording")
        if self._tray:
            self._tray.set_recording(True)

    def _on_hold_end(self):
        """Called when hold-to-talk key is released."""
        logger.info("Hold-to-talk: END")
        audio_data = self.audio.stop_recording()
        if self._overlay:
            self._overlay.set_mode("processing")
        if self._tray:
            self._tray.set_recording(False)

        if not audio_data:
            if self._overlay:
                self._overlay.set_mode("idle")
            return

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
        self.audio.start_recording()
        if self._overlay:
            self._overlay.set_mode("command")

    def _on_command_end(self):
        """Called when command mode key is released."""
        logger.info("Command mode: END")
        audio_data = self.audio.stop_recording()
        self._command_mode_active = False
        if self._overlay:
            self._overlay.set_mode("processing")

        if not audio_data:
            if self._overlay:
                self._overlay.set_mode("idle")
            return

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

    # -- Audio processing --

    def _process_audio(self, audio_data: bytes, is_command: bool):
        """Process captured audio through the full pipeline."""
        try:
            # Step 1: Transcribe
            initial_prompt = self.dictionary.get_whisper_prompt()
            lang_cfg = self.config["transcription"]["language"]
            language = lang_cfg if lang_cfg != "auto" else None

            result = self.transcriber.transcribe(
                audio_data,
                language=language,
                initial_prompt=initial_prompt or None,
            )

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
                self.injector.inject(pipeline_result.processed_text)

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

        except Exception:
            logger.exception("Error processing audio")
        finally:
            if self._overlay:
                self._overlay.set_mode("idle")

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
    app = WhisperFlowApp()
    app.run()


if __name__ == "__main__":
    main()

"""System tray icon and menu for WhisperFlow."""

import logging
import sys

logger = logging.getLogger(__name__)

_qt_available = False
try:
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
    from PyQt6.QtGui import QIcon, QAction, QPixmap, QPainter, QColor, QFont
    from PyQt6.QtWidgets import (
        QApplication, QSystemTrayIcon, QMenu, QWidget,
        QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QDialog, QTextEdit, QLineEdit, QComboBox, QCheckBox,
        QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
        QFormLayout, QSpinBox, QGroupBox,
    )
    _qt_available = True
except ImportError:
    logger.warning("PyQt6 not available - tray icon disabled")


if _qt_available:
    def _create_icon(color: str = "#44aa44") -> QIcon:
        """Create a simple colored circle icon."""
        pixmap = QPixmap(64, 64)
        pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(8, 8, 48, 48)
        # "W" letter
        painter.setPen(QColor("white"))
        painter.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "W")
        painter.end()
        return QIcon(pixmap)

    class SettingsDialog(QDialog):
        """Settings dialog for WhisperFlow configuration."""

        def __init__(self, config: dict, parent=None):
            super().__init__(parent)
            self.config = config
            self.setWindowTitle("WhisperFlow Settings")
            self.setMinimumSize(600, 500)

            layout = QVBoxLayout(self)
            tabs = QTabWidget()
            layout.addWidget(tabs)

            # General tab
            general = QWidget()
            general_layout = QFormLayout(general)

            self.hotkey_edit = QLineEdit(config.get("hotkey", {}).get("hold_key", "ctrl_l+space"))
            general_layout.addRow("Hold-to-talk hotkey:", self.hotkey_edit)

            self.command_key_edit = QLineEdit(config.get("hotkey", {}).get("command_key", "ctrl_l+shift+space"))
            general_layout.addRow("Command mode hotkey:", self.command_key_edit)

            self.overlay_check = QCheckBox()
            self.overlay_check.setChecked(config.get("ui", {}).get("show_overlay", True))
            general_layout.addRow("Show overlay:", self.overlay_check)

            tabs.addTab(general, "General")

            # Transcription tab
            transcription = QWidget()
            trans_layout = QFormLayout(transcription)

            self.backend_combo = QComboBox()
            self.backend_combo.addItems(["local", "remote"])
            self.backend_combo.setCurrentText(config.get("transcription", {}).get("backend", "local"))
            trans_layout.addRow("Backend:", self.backend_combo)

            self.model_combo = QComboBox()
            self.model_combo.addItems(["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"])
            self.model_combo.setCurrentText(config.get("transcription", {}).get("model_size", "large-v3-turbo"))
            trans_layout.addRow("Model:", self.model_combo)

            self.language_edit = QLineEdit(config.get("transcription", {}).get("language", "auto"))
            trans_layout.addRow("Language (auto or ISO code):", self.language_edit)

            self.remote_url_edit = QLineEdit(config.get("transcription", {}).get("remote_url", ""))
            trans_layout.addRow("Remote GPU server URL:", self.remote_url_edit)

            tabs.addTab(transcription, "Transcription")

            # LLM tab
            llm_tab = QWidget()
            llm_layout = QFormLayout(llm_tab)

            self.llm_provider_combo = QComboBox()
            self.llm_provider_combo.addItems(["ollama", "openai", "anthropic", "none"])
            self.llm_provider_combo.setCurrentText(config.get("llm", {}).get("provider", "ollama"))
            llm_layout.addRow("Provider:", self.llm_provider_combo)

            self.ollama_url_edit = QLineEdit(config.get("llm", {}).get("ollama", {}).get("url", "http://localhost:11434"))
            llm_layout.addRow("Ollama URL:", self.ollama_url_edit)

            self.ollama_model_edit = QLineEdit(config.get("llm", {}).get("ollama", {}).get("model", "llama3.1:8b"))
            llm_layout.addRow("Ollama model:", self.ollama_model_edit)

            self.openai_key_edit = QLineEdit(config.get("llm", {}).get("openai", {}).get("api_key", ""))
            self.openai_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
            llm_layout.addRow("OpenAI API key:", self.openai_key_edit)

            self.anthropic_key_edit = QLineEdit(config.get("llm", {}).get("anthropic", {}).get("api_key", ""))
            self.anthropic_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
            llm_layout.addRow("Anthropic API key:", self.anthropic_key_edit)

            tabs.addTab(llm_tab, "LLM")

            # Processing tab
            proc_tab = QWidget()
            proc_layout = QFormLayout(proc_tab)

            self.auto_edit_check = QCheckBox()
            self.auto_edit_check.setChecked(config.get("processing", {}).get("auto_edit", True))
            proc_layout.addRow("Auto-edit (filler removal, punctuation):", self.auto_edit_check)

            self.course_correction_check = QCheckBox()
            self.course_correction_check.setChecked(config.get("processing", {}).get("course_correction", True))
            proc_layout.addRow("Course correction:", self.course_correction_check)

            self.context_check = QCheckBox()
            self.context_check.setChecked(config.get("processing", {}).get("context_aware", True))
            proc_layout.addRow("Context-aware formatting:", self.context_check)

            self.tone_check = QCheckBox()
            self.tone_check.setChecked(config.get("processing", {}).get("tone_matching", True))
            proc_layout.addRow("Tone matching:", self.tone_check)

            tabs.addTab(proc_tab, "Processing")

            # Buttons
            btn_layout = QHBoxLayout()
            save_btn = QPushButton("Save")
            save_btn.clicked.connect(self.accept)
            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(self.reject)
            btn_layout.addStretch()
            btn_layout.addWidget(save_btn)
            btn_layout.addWidget(cancel_btn)
            layout.addLayout(btn_layout)

        def get_config(self) -> dict:
            """Get updated config from dialog values."""
            self.config["hotkey"]["hold_key"] = self.hotkey_edit.text()
            self.config["hotkey"]["command_key"] = self.command_key_edit.text()
            self.config["ui"]["show_overlay"] = self.overlay_check.isChecked()
            self.config["transcription"]["backend"] = self.backend_combo.currentText()
            self.config["transcription"]["model_size"] = self.model_combo.currentText()
            self.config["transcription"]["language"] = self.language_edit.text()
            self.config["transcription"]["remote_url"] = self.remote_url_edit.text()
            self.config["llm"]["provider"] = self.llm_provider_combo.currentText()
            self.config["llm"]["ollama"]["url"] = self.ollama_url_edit.text()
            self.config["llm"]["ollama"]["model"] = self.ollama_model_edit.text()
            self.config["llm"]["openai"]["api_key"] = self.openai_key_edit.text()
            self.config["llm"]["anthropic"]["api_key"] = self.anthropic_key_edit.text()
            self.config["processing"]["auto_edit"] = self.auto_edit_check.isChecked()
            self.config["processing"]["course_correction"] = self.course_correction_check.isChecked()
            self.config["processing"]["context_aware"] = self.context_check.isChecked()
            self.config["processing"]["tone_matching"] = self.tone_check.isChecked()
            return self.config

    class DictionaryDialog(QDialog):
        """Dialog for managing personal dictionary."""

        def __init__(self, dictionary, parent=None):
            super().__init__(parent)
            self.dictionary = dictionary
            self.setWindowTitle("Personal Dictionary")
            self.setMinimumSize(500, 400)

            layout = QVBoxLayout(self)

            # Add word
            add_layout = QHBoxLayout()
            self.word_edit = QLineEdit()
            self.word_edit.setPlaceholderText("Add a word...")
            add_layout.addWidget(self.word_edit)
            add_btn = QPushButton("Add")
            add_btn.clicked.connect(self._add_word)
            add_layout.addWidget(add_btn)
            layout.addLayout(add_layout)

            # Word table
            self.table = QTableWidget()
            self.table.setColumnCount(2)
            self.table.setHorizontalHeaderLabels(["Word", "Category"])
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            layout.addWidget(self.table)

            # Delete button
            del_btn = QPushButton("Delete Selected")
            del_btn.clicked.connect(self._delete_word)
            layout.addWidget(del_btn)

            self._refresh_table()

        def _refresh_table(self):
            words = self.dictionary.get_words()
            self.table.setRowCount(len(words))
            for i, (word, cat) in enumerate(words.items()):
                self.table.setItem(i, 0, QTableWidgetItem(word))
                self.table.setItem(i, 1, QTableWidgetItem(cat))

        def _add_word(self):
            word = self.word_edit.text().strip()
            if word:
                self.dictionary.add_word(word)
                self.word_edit.clear()
                self._refresh_table()

        def _delete_word(self):
            row = self.table.currentRow()
            if row >= 0:
                word = self.table.item(row, 0).text()
                self.dictionary.remove_word(word)
                self._refresh_table()

    class SnippetsDialog(QDialog):
        """Dialog for managing voice snippets."""

        def __init__(self, snippets, parent=None):
            super().__init__(parent)
            self.snippets = snippets
            self.setWindowTitle("Voice Snippets")
            self.setMinimumSize(600, 400)

            layout = QVBoxLayout(self)

            # Add snippet
            add_group = QGroupBox("Add Snippet")
            add_layout = QFormLayout(add_group)
            self.trigger_edit = QLineEdit()
            self.trigger_edit.setPlaceholderText("Voice trigger (e.g., 'my calendar link')")
            add_layout.addRow("Trigger:", self.trigger_edit)
            self.content_edit = QTextEdit()
            self.content_edit.setMaximumHeight(80)
            self.content_edit.setPlaceholderText("Content to insert...")
            add_layout.addRow("Content:", self.content_edit)
            add_btn = QPushButton("Add Snippet")
            add_btn.clicked.connect(self._add_snippet)
            add_layout.addRow(add_btn)
            layout.addWidget(add_group)

            # Snippets table
            self.table = QTableWidget()
            self.table.setColumnCount(2)
            self.table.setHorizontalHeaderLabels(["Trigger", "Content"])
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            layout.addWidget(self.table)

            del_btn = QPushButton("Delete Selected")
            del_btn.clicked.connect(self._delete_snippet)
            layout.addWidget(del_btn)

            self._refresh_table()

        def _refresh_table(self):
            snippets = self.snippets.get_snippets()
            self.table.setRowCount(len(snippets))
            for i, (trigger, data) in enumerate(snippets.items()):
                self.table.setItem(i, 0, QTableWidgetItem(trigger))
                self.table.setItem(i, 1, QTableWidgetItem(data.get("content", "")))

        def _add_snippet(self):
            trigger = self.trigger_edit.text().strip()
            content = self.content_edit.toPlainText().strip()
            if trigger and content:
                self.snippets.add_snippet(trigger, content)
                self.trigger_edit.clear()
                self.content_edit.clear()
                self._refresh_table()

        def _delete_snippet(self):
            row = self.table.currentRow()
            if row >= 0:
                trigger = self.table.item(row, 0).text()
                self.snippets.remove_snippet(trigger)
                self._refresh_table()

    class HistoryDialog(QDialog):
        """Dialog showing recent transcription history."""

        def __init__(self, history, parent=None):
            super().__init__(parent)
            self.history = history
            self.setWindowTitle("Transcription History")
            self.setMinimumSize(700, 500)

            layout = QVBoxLayout(self)

            # Search
            search_layout = QHBoxLayout()
            self.search_edit = QLineEdit()
            self.search_edit.setPlaceholderText("Search history...")
            self.search_edit.textChanged.connect(self._search)
            search_layout.addWidget(self.search_edit)
            layout.addLayout(search_layout)

            # History table
            self.table = QTableWidget()
            self.table.setColumnCount(4)
            self.table.setHorizontalHeaderLabels(["Time", "App", "Mode", "Text"])
            self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
            layout.addWidget(self.table)

            self._load_entries(self.history.get_recent())

        def _load_entries(self, entries):
            import datetime
            self.table.setRowCount(len(entries))
            for i, entry in enumerate(entries):
                dt = datetime.datetime.fromtimestamp(entry.timestamp)
                self.table.setItem(i, 0, QTableWidgetItem(dt.strftime("%Y-%m-%d %H:%M")))
                self.table.setItem(i, 1, QTableWidgetItem(entry.app_context))
                self.table.setItem(i, 2, QTableWidgetItem(entry.mode))
                self.table.setItem(i, 3, QTableWidgetItem(entry.processed_text[:100]))

        def _search(self, query):
            if query:
                entries = self.history.search(query)
            else:
                entries = self.history.get_recent()
            self._load_entries(entries)

    class SystemTray(QObject):
        """System tray icon with menu for WhisperFlow."""

        show_settings = pyqtSignal()
        show_dictionary = pyqtSignal()
        show_snippets = pyqtSignal()
        show_history = pyqtSignal()
        toggle_whisper_mode = pyqtSignal()
        quit_app = pyqtSignal()

        def __init__(self, app_ref):
            super().__init__()
            self.app_ref = app_ref
            self._whisper_mode = False

            self.tray = QSystemTrayIcon()
            self.tray.setIcon(_create_icon())
            self.tray.setToolTip("WhisperFlow - Voice Dictation")

            menu = QMenu()

            status_action = menu.addAction("WhisperFlow v0.1.0")
            status_action.setEnabled(False)
            menu.addSeparator()

            settings_action = menu.addAction("Settings...")
            settings_action.triggered.connect(self.show_settings.emit)

            dict_action = menu.addAction("Personal Dictionary...")
            dict_action.triggered.connect(self.show_dictionary.emit)

            snippets_action = menu.addAction("Voice Snippets...")
            snippets_action.triggered.connect(self.show_snippets.emit)

            history_action = menu.addAction("History...")
            history_action.triggered.connect(self.show_history.emit)

            menu.addSeparator()

            self._whisper_action = menu.addAction("Whisper Mode: OFF")
            self._whisper_action.triggered.connect(self._toggle_whisper)

            menu.addSeparator()

            quit_action = menu.addAction("Quit")
            quit_action.triggered.connect(self.quit_app.emit)

            self.tray.setContextMenu(menu)
            self.tray.show()

        def _toggle_whisper(self):
            self._whisper_mode = not self._whisper_mode
            state = "ON" if self._whisper_mode else "OFF"
            self._whisper_action.setText(f"Whisper Mode: {state}")
            self.toggle_whisper_mode.emit()

        def set_recording(self, recording: bool):
            """Update tray icon to reflect recording state."""
            if recording:
                self.tray.setIcon(_create_icon("#ff4444"))
                self.tray.setToolTip("WhisperFlow - Recording...")
            else:
                self.tray.setIcon(_create_icon("#44aa44"))
                self.tray.setToolTip("WhisperFlow - Voice Dictation")

        def show_notification(self, title: str, message: str):
            """Show a system notification."""
            self.tray.showMessage(title, message, QSystemTrayIcon.MessageIcon.Information, 3000)

else:
    class SystemTray:
        """Stub when PyQt6 not available."""
        def __init__(self, app_ref):
            logger.warning("PyQt6 not available - running in headless mode")
        def set_recording(self, recording: bool): pass
        def show_notification(self, title: str, message: str):
            logger.info(f"[{title}] {message}")

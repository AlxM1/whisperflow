"""Recording overlay indicator - shows when dictation is active."""

import logging
import sys

logger = logging.getLogger(__name__)

_qt_available = False
try:
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve
    from PyQt6.QtGui import QColor, QPainter, QFont
    from PyQt6.QtWidgets import QWidget, QApplication, QLabel, QHBoxLayout
    _qt_available = True
except ImportError:
    pass


if _qt_available:
    class RecordingOverlay(QWidget):
        """Floating overlay that shows recording state."""

        def __init__(self, position: str = "top-right"):
            super().__init__()
            self._position = position
            self._mode = "idle"  # idle, recording, processing, command
            self._audio_level = 0.0
            self._pulse_opacity = 1.0

            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
                | Qt.WindowType.WindowTransparentForInput
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
            self.setFixedSize(200, 44)

            # Layout
            layout = QHBoxLayout(self)
            layout.setContentsMargins(12, 8, 12, 8)

            self._dot = QLabel("●")
            self._dot.setFont(QFont("", 14))
            layout.addWidget(self._dot)

            self._label = QLabel("WhisperFlow")
            self._label.setFont(QFont("", 11))
            self._label.setStyleSheet("color: white;")
            layout.addWidget(self._label)

            # Pulse animation timer
            self._pulse_timer = QTimer()
            self._pulse_timer.timeout.connect(self._pulse)
            self._pulse_direction = -0.05

            self._position_window()

        def _position_window(self):
            """Position the overlay on screen."""
            screen = QApplication.primaryScreen()
            if not screen:
                return
            geom = screen.availableGeometry()
            w, h = self.width(), self.height()
            margin = 20

            positions = {
                "top-right": (geom.right() - w - margin, geom.top() + margin),
                "top-left": (geom.left() + margin, geom.top() + margin),
                "bottom-right": (geom.right() - w - margin, geom.bottom() - h - margin),
                "bottom-left": (geom.left() + margin, geom.bottom() - h - margin),
                "center": (geom.center().x() - w // 2, geom.top() + margin),
            }
            x, y = positions.get(self._position, positions["top-right"])
            self.move(x, y)

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Background
            colors = {
                "idle": QColor(40, 40, 40, 180),
                "recording": QColor(200, 40, 40, 200),
                "processing": QColor(40, 120, 200, 200),
                "command": QColor(160, 80, 200, 200),
            }
            bg = colors.get(self._mode, colors["idle"])
            painter.setBrush(bg)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(self.rect(), 12, 12)

        def set_mode(self, mode: str, label: str = ""):
            """Set the overlay mode: idle, recording, processing, command."""
            self._mode = mode
            labels = {
                "idle": "WhisperFlow",
                "recording": "🎤 Listening...",
                "processing": "⚡ Processing...",
                "command": "🔧 Command Mode",
            }
            self._label.setText(label or labels.get(mode, "WhisperFlow"))

            dot_colors = {
                "idle": "gray",
                "recording": "#ff4444",
                "processing": "#4488ff",
                "command": "#aa55dd",
            }
            self._dot.setStyleSheet(f"color: {dot_colors.get(mode, 'gray')};")

            if mode == "recording":
                self._pulse_timer.start(50)
            else:
                self._pulse_timer.stop()

            self.update()
            if mode != "idle":
                self.show()
            else:
                self.hide()

        def set_audio_level(self, level: float):
            """Update audio level for visual feedback (0.0 to 1.0)."""
            self._audio_level = min(1.0, level)

        def _pulse(self):
            self._pulse_opacity += self._pulse_direction
            if self._pulse_opacity <= 0.5 or self._pulse_opacity >= 1.0:
                self._pulse_direction *= -1
            opacity_hex = hex(int(self._pulse_opacity * 255))[2:].zfill(2)
            self._dot.setStyleSheet(f"color: #ff4444{opacity_hex};")

else:
    class RecordingOverlay:
        """Stub overlay when PyQt6 is not available."""
        def __init__(self, position: str = "top-right"):
            logger.warning("PyQt6 not available, recording overlay disabled")
        def set_mode(self, mode: str, label: str = ""): pass
        def set_audio_level(self, level: float): pass
        def show(self): pass
        def hide(self): pass

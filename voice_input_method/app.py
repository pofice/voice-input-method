"""Main application window — thin UI shell.

All business logic lives in VoiceEngine (engine.py).
All dependency assembly lives in factory.py.
Hotkey listening lives in hotkey.py.
This file only handles PySide6 widgets and signals.
"""

import time
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QTextEdit
from PySide6.QtGui import QMouseEvent, QIcon
from PySide6.QtCore import Qt, QEvent, Signal, QPointF

from .config import Config, resolve_resource_path
from .factory import create_engine, create_indicator
from .hotkey import CombinedHotkeyListener


class InputButton(QPushButton):
    """Custom button with press/hover visual feedback and simulated press/release."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.isPressed = False
        self.pressed.connect(self._on_pressed)
        self.released.connect(self._on_released)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _on_pressed(self):
        self.setStyleSheet("background-color: rgba(90, 133, 15, 0.8);")

    def _on_released(self):
        if self.underMouse():
            self.setStyleSheet("background-color: rgba(100, 145, 40, 1);")
        else:
            self.setStyleSheet("background-color: rgba(90, 133, 15, 1);")

    def enterEvent(self, event):
        self.setStyleSheet("background-color: rgba(100, 145, 40, 1);")

    def leaveEvent(self, event):
        self._on_released()

    def simulatePress(self):
        if not self.isPressed:
            center = QPointF(self.rect().center())
            event = QMouseEvent(
                QEvent.Type.MouseButtonPress, center, center,
                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            QApplication.postEvent(self, event)
            self.isPressed = True

    def simulateRelease(self):
        if self.isPressed:
            center = QPointF(self.rect().center())
            event = QMouseEvent(
                QEvent.Type.MouseButtonRelease, center, center,
                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            QApplication.postEvent(self, event)
            self.isPressed = False


class MainWindow(QWidget):
    """Thin UI shell — delegates everything to VoiceEngine."""

    transcription_ready = Signal(str)
    partial_text_ready = Signal(str)

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # --- Create engine via factory (GUI-free assembly) ---
        self.engine = create_engine(
            config,
            on_partial=lambda text: self.partial_text_ready.emit(text),
            on_result=lambda text: self.transcription_ready.emit(text),
        )

        # Signals → UI updates
        self.transcription_ready.connect(self._on_transcription)
        self.partial_text_ready.connect(self._on_partial_text)

        # Build UI
        self._build_ui()

        # Start engine (loads models, starts audio)
        print("Loading ASR model...")
        warmup_path = resolve_resource_path(config, "warmup_file")
        self.engine.start()
        self.engine.warmup(str(warmup_path))
        print("Models ready.")

        # Recording indicator (macOS: native AppKit, others: no-op)
        self._indicator = create_indicator(config.platform)

        # Combined hotkey listener (single pynput Listener for both modes)
        self._long_record_start = None
        self._pending_long_save_duration = None
        self._hotkey = CombinedHotkeyListener(
            hold_hotkey=config.hotkey,
            hold_on_press=lambda: self.button.simulatePress(),
            hold_on_release=lambda: self.button.simulateRelease(),
            toggle_hotkey=config.toggle_hotkey or None,
            toggle_on_start=self._on_long_record_start,
            toggle_on_stop=self._on_long_record_stop,
        )
        self._hotkey.start()

        # Start hotword file watcher (Qt-dependent, belongs in UI layer)
        if self.engine.hotword_provider and hasattr(self.engine.hotword_provider, "start_watching"):
            self.engine.hotword_provider.start_watching()

        # Drag state
        self._drag_position = None

    def _build_ui(self):
        self.setWindowOpacity(self.config.window_opacity)


        icon_path = resolve_resource_path(self.config, "icon_file")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.setWindowTitle(self.config.window_title)
        self.resize(self.config.window_width, self.config.window_height)

        # Text area
        self.textEdit = QTextEdit(self)
        self.textEdit.move(0, 0)
        self.textEdit.resize(self.width(), self.height() - 32)

        # Input button
        self.button = InputButton(self)
        self.button.setText("长按输入")
        self.button.resize(self.width() // 2, 32)
        self.button.move(0, self.height() - 32)
        self.button.pressed.connect(self._on_start_recording)
        self.button.released.connect(self._on_stop_recording)

        # Convert button
        self.convertButton = InputButton(self)
        self.convertButton.setText("繁简转换")
        self.convertButton.resize(self.width() // 2, 32)
        self.convertButton.move(self.width() // 2, self.height() - 32)
        self.convertButton.released.connect(self._convert_text)


    # --- Recording lifecycle ---

    def _on_start_recording(self):
        self.engine.start_recording()
        self._indicator.show("dot")

    def _on_stop_recording(self):
        self.engine.stop_recording()
        self._indicator.hide()

    # --- Long recording (toggle hotkey) ---

    def _on_long_record_start(self):
        self._long_record_start = time.time()
        self.engine.start_recording()
        self._indicator.show("ring")

    def _on_long_record_stop(self):
        duration = int(time.time() - self._long_record_start) if self._long_record_start else 0
        self._long_record_start = None
        self.engine.stop_recording()
        self._indicator.hide()
        # Save transcription to file after result callback fires
        self._pending_long_save_duration = duration

    def _on_transcription(self, text: str):
        self.textEdit.setText(text)
        # Save long recording result to file
        if hasattr(self, "_pending_long_save_duration") and self._pending_long_save_duration is not None:
            self._save_long_recording(text, self._pending_long_save_duration)
            self._pending_long_save_duration = None

    def _save_long_recording(self, text: str, duration: int):
        """Save transcription to ~/voice-recordings/YYYY-MM-DD_HH-MM-SS_XXs.txt"""
        if not text.strip():
            return
        save_dir = Path.home() / "voice-recordings"
        save_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{ts}_{duration}s.txt"
        filepath = save_dir / filename
        filepath.write_text(text, encoding="utf-8")
        print(f"Long recording saved: {filepath}")

    # --- UI callbacks ---

    def _on_partial_text(self, text: str):
        self.textEdit.setText(text)

    def _convert_text(self):
        text = self.textEdit.toPlainText()
        self.engine.convert_chinese_async(text)

    # --- Window drag ---

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_position is not None:
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()

    def resizeEvent(self, event):
        btn_h = self.button.height()
        self.textEdit.resize(self.width(), self.height() - btn_h)
        self.button.resize(self.width() // 2, btn_h)
        self.convertButton.resize(self.width() // 2, btn_h)
        self.button.move(0, self.height() - btn_h)
        self.convertButton.move(self.width() // 2, self.height() - btn_h)

    def closeEvent(self, event):
        self._indicator.shutdown()
        self.engine.shutdown()
        self._hotkey.stop()
        if self.engine.hotword_provider and hasattr(self.engine.hotword_provider, "stop_watching"):
            self.engine.hotword_provider.stop_watching()
        super().closeEvent(event)

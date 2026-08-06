"""Main application window — thin UI shell.

All business logic lives in VoiceEngine (engine.py).
All dependency assembly lives in factory.py.
Hotkey listening lives in hotkey.py.
This file only handles PySide6 widgets and signals.
"""

import time

from PySide6.QtCore import QEvent, QPointF, Qt, Signal
from PySide6.QtGui import QIcon, QMouseEvent
from PySide6.QtWidgets import QApplication, QMenu, QPushButton, QTextEdit, QWidget

from .config import Config, resolve_resource_path
from .engine import archive_recording_audio, archive_recording_text
from .factory import create_engine, create_hotkey_listener, create_indicator


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

        # Switch to specified audio device if set via --device
        device_index = getattr(config, "_device_index", None)
        if device_index is not None:
            self.engine.recorder.switch_device(device_index)
            print(f"Using audio device: {device_index}")

        # Recording indicator (macOS: native AppKit, others: no-op)
        self._indicator = create_indicator(config.platform)

        # Hotkey listener — backend chosen by factory (evdev on Wayland, else pynput)
        self._long_record_start = None
        self._pending_long_save_base = None
        self._hotkey = create_hotkey_listener(
            config,
            hold_on_press=lambda: self.button.simulatePress(),
            hold_on_release=lambda: self.button.simulateRelease(),
            toggle_on_start=self._on_long_record_start,
            toggle_on_stop=self._on_long_record_stop,
        )
        try:
            self._hotkey.start()
        except PermissionError as e:
            # evdev needs the `input` group; the GUI button still works without it
            print(f"[hotkey] 全局热键不可用: {e}")

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

        # Button bar height and circle size
        self._btn_h = 32
        self._circle_size = 28

        # Input button (left)
        self.button = InputButton(self)
        self.button.setText("长按输入")
        self.button.pressed.connect(self._on_start_recording)
        self.button.released.connect(self._on_stop_recording)

        # Long recording button (center, circular)
        self.longRecordButton = QPushButton("\u25CF", self)
        self.longRecordButton.setCursor(Qt.CursorShape.PointingHandCursor)
        self.longRecordButton.setFixedSize(self._circle_size, self._circle_size)
        self._long_recording_active = False
        self._update_long_record_style()
        self.longRecordButton.clicked.connect(self._on_long_record_button)

        # Convert button (right)
        self.convertButton = InputButton(self)
        self.convertButton.setText("繁简转换")
        self.convertButton.released.connect(self._convert_text)

        self._layout_buttons()


    def _layout_buttons(self):
        """Position the three buttons at the bottom of the window."""
        w = self.width()
        h = self._btn_h
        circle = self._circle_size
        side_w = (w - circle) // 2

        self.button.resize(side_w, h)
        self.button.move(0, self.height() - h)

        # Center the circle vertically and horizontally between the two buttons
        self.longRecordButton.move(side_w + (circle - self._circle_size) // 2,
                                    self.height() - h + (h - circle) // 2)

        self.convertButton.resize(w - side_w - circle, h)
        self.convertButton.move(side_w + circle, self.height() - h)

    def _update_long_record_style(self):
        r = self._circle_size // 2
        common = f"margin: 0px; padding: 0px; border: none; border-radius: {r}px; color: white; font-size: 10px;"
        if self._long_recording_active:
            self.longRecordButton.setStyleSheet(
                f"QPushButton {{ {common} background-color: #e53935; }}"
                f"QPushButton:hover {{ background-color: #ef5350; }}"
                f"QPushButton:pressed {{ background-color: #c62828; }}"
            )
        else:
            self.longRecordButton.setStyleSheet(
                f"QPushButton {{ {common} background-color: rgba(90, 133, 15, 1); }}"
                f"QPushButton:hover {{ background-color: rgba(100, 145, 40, 1); }}"
                f"QPushButton:pressed {{ background-color: rgba(80, 120, 10, 1); }}"
            )

    def _on_long_record_button(self):
        """GUI button click toggles long recording."""
        if not self._long_recording_active:
            self._on_long_record_start()
        else:
            self._on_long_record_stop()

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
        self._long_recording_active = True
        self._update_long_record_style()
        self.engine.start_recording()
        self._indicator.show("ring")

    def _on_long_record_stop(self):
        duration = int(time.time() - self._long_record_start) if self._long_record_start else 0
        self._long_record_start = None
        self._long_recording_active = False
        self._update_long_record_style()
        self.engine.stop_recording()
        self._indicator.hide()
        # Archive the audio NOW — before transcription, which can fail
        # (remote backend + network down) and must never lose the recording.
        self._pending_long_save_base = archive_recording_audio(
            self.engine._audio_path, duration
        )
        if self._pending_long_save_base is not None:
            print(f"Long recording audio saved: {self._pending_long_save_base}.wav")

    def _on_transcription(self, text: str):
        self.textEdit.setText(text)
        # Pair the transcript with the already-archived audio
        if getattr(self, "_pending_long_save_base", None) is not None:
            archive_recording_text(self._pending_long_save_base, text)
            print(f"Long recording transcript saved: {self._pending_long_save_base}.txt")
            self._pending_long_save_base = None

    # --- UI callbacks ---

    def _on_partial_text(self, text: str):
        self.textEdit.setText(text)

    def _convert_text(self):
        text = self.textEdit.toPlainText()
        self.engine.convert_chinese_async(text)

    # --- Context menu (right-click) ---

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        # Audio device submenu
        device_menu = menu.addMenu("切换麦克风")
        devices = self.engine.recorder.refresh_and_list_devices()
        for dev in devices:
            action = device_menu.addAction(dev["name"])
            action.setData(dev["index"])
            action.triggered.connect(lambda checked, idx=dev["index"], name=dev["name"]: self._switch_device(idx, name))
        menu.exec(event.globalPos())

    def _switch_device(self, device_index: int, name: str):
        self.engine.recorder.switch_device(device_index)
        print(f"Switched to: {name}")

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
        self.textEdit.resize(self.width(), self.height() - self._btn_h)
        self._layout_buttons()

    def closeEvent(self, event):
        self._indicator.shutdown()
        self.engine.shutdown()
        self._hotkey.stop()
        if self.engine.hotword_provider and hasattr(self.engine.hotword_provider, "stop_watching"):
            self.engine.hotword_provider.stop_watching()
        super().closeEvent(event)

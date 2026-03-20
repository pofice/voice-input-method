"""Main application window - unified across all platforms."""

import os
import threading
import tempfile

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QTextEdit, QCheckBox
from PyQt5.QtGui import QMouseEvent, QIcon
from PyQt5.QtCore import Qt, QEvent, QTimer, pyqtSignal
from pynput import keyboard

from .config import Config, resolve_resource_path
from .audio import AudioRecorder
from .recognition import SpeechRecognizer
from .text_processing import clean_spaces, convert_chinese_numbers, ChineseConverter
from .hotwords import HotwordManager
from .platform import get_backend


# Map config hotkey names to pynput Key objects
HOTKEY_MAP = {
    "scroll_lock": keyboard.Key.scroll_lock,
    "pause": keyboard.Key.pause,
    "f6": keyboard.Key.f6,
    "f7": keyboard.Key.f7,
    "f8": keyboard.Key.f8,
    "f9": keyboard.Key.f9,
    "f10": keyboard.Key.f10,
    "f11": keyboard.Key.f11,
    "f12": keyboard.Key.f12,
}


class InputButton(QPushButton):
    """Custom button with press/hover visual feedback and simulated press/release."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.isPressed = False
        self.pressed.connect(self._on_pressed)
        self.released.connect(self._on_released)
        self.setCursor(Qt.PointingHandCursor)

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
            event = QMouseEvent(
                QEvent.MouseButtonPress, self.rect().center(),
                Qt.LeftButton, Qt.LeftButton, Qt.NoModifier,
            )
            QApplication.postEvent(self, event)
            self.isPressed = True

    def simulateRelease(self):
        if self.isPressed:
            event = QMouseEvent(
                QEvent.MouseButtonRelease, self.rect().center(),
                Qt.LeftButton, Qt.LeftButton, Qt.NoModifier,
            )
            QApplication.postEvent(self, event)
            self.isPressed = False


class MainWindow(QWidget):
    """Main application window."""

    transcription_ready = pyqtSignal(str)
    text_ready = pyqtSignal(str)

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Platform backend
        self.backend = get_backend(config.platform)
        missing = self.backend.check_permissions()
        for msg in missing:
            print(f"WARNING: {msg}")

        # Audio recorder
        self.recorder = AudioRecorder(config.sample_rate, config.channels)

        # Temp file for audio
        self._audio_path = os.path.join(tempfile.gettempdir(), "voice_input_audio.wav")

        # ASR model
        self.recognizer = SpeechRecognizer(
            model_type=config.model_type,
            model_dir=config.model_dir,
            quantize=config.quantize,
        )

        # Text processing
        self.chinese_converter: ChineseConverter | None = None
        if config.enable_traditional_chinese:
            lib_path = resolve_resource_path(config, "library_file")
            self.chinese_converter = ChineseConverter(lib_path)

        # Hotwords
        self.hotword_manager: HotwordManager | None = None
        if config.enable_hotwords:
            hw_path = resolve_resource_path(config, "hotwords_file")
            self.hotword_manager = HotwordManager(hw_path)

        # Signals
        self.transcription_ready.connect(self._on_transcription)
        self.text_ready.connect(self._on_text_update)

        # Threading
        self._convert_lock = threading.Lock()

        # Build UI
        self._build_ui()

        # Load model
        print("Loading ASR model...")
        self.recognizer.load()
        warmup_path = resolve_resource_path(config, "warmup_file")
        hotwords = self.hotword_manager.hotwords_str if self.hotword_manager else ""
        self.recognizer.warmup(str(warmup_path), hotwords)
        print("Model ready.")

        # Start audio
        self.recorder.start()

        # Start hotkey listener
        self._setup_hotkey()

        # Start hotword file watcher
        if self.hotword_manager:
            self.hotword_manager.start_watching()

        # Drag state
        self._drag_position = None

    def _build_ui(self):
        self.setWindowOpacity(self.config.window_opacity)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

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
        self.button.pressed.connect(self._start_recording)
        self.button.released.connect(self._stop_recording)

        # Convert button
        self.convertButton = InputButton(self)
        self.convertButton.setText("繁简转换")
        self.convertButton.resize(self.width() // 2, 32)
        self.convertButton.move(self.width() // 2, self.height() - 32)
        self.convertButton.released.connect(self._convert_text)

        # Number conversion checkbox
        if self.config.enable_number_conversion:
            self.number_checkbox = QCheckBox("阿拉伯数字", self)
            self.number_checkbox.setChecked(False)
            self.number_checkbox.move(10, self.height() - 50)
            self.number_checkbox.resize(180, 20)
        else:
            self.number_checkbox = None

    def _setup_hotkey(self):
        hotkey = HOTKEY_MAP.get(self.config.hotkey, keyboard.Key.scroll_lock)

        def on_press(key):
            try:
                if key == hotkey and not self.button.isPressed:
                    self.button.simulatePress()
            except AttributeError:
                pass

        def on_release(key):
            if key == hotkey and self.button.isPressed:
                self.button.simulateRelease()

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def _start_recording(self):
        self.recorder.start_recording()

    def _stop_recording(self):
        self.recorder.stop_recording(self._audio_path)
        threading.Thread(target=self._transcribe, daemon=True).start()

    def _transcribe(self):
        hotwords = self.hotword_manager.hotwords_str if self.hotword_manager else ""
        text = self.recognizer.transcribe(self._audio_path, hotwords)
        if text:
            text = clean_spaces(text)
            self.transcription_ready.emit(text)

    def _on_transcription(self, text: str):
        if self.number_checkbox and self.number_checkbox.isChecked():
            text = convert_chinese_numbers(text)
        self.textEdit.setText(text)
        self.backend.paste_text(text)

    def _convert_text(self):
        if self.chinese_converter:
            threading.Thread(target=self._convert_text_thread, daemon=True).start()

    def _convert_text_thread(self):
        with self._convert_lock:
            text = self.textEdit.toPlainText()
            if not text:
                return
            converted = self.chinese_converter.convert(text)
            self.text_ready.emit(converted)
            clipboard = QApplication.clipboard()
            clipboard.setText(converted)

    def _on_text_update(self, text: str):
        self.textEdit.setText(text)

    # --- Window drag support ---

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_position is not None:
            self.move(event.globalPos() - self._drag_position)
            event.accept()

    # --- Responsive layout ---

    def resizeEvent(self, event):
        btn_h = self.button.height()
        self.textEdit.resize(self.width(), self.height() - btn_h)
        self.button.resize(self.width() // 2, btn_h)
        self.convertButton.resize(self.width() // 2, btn_h)
        self.button.move(0, self.height() - btn_h)
        self.convertButton.move(self.width() // 2, self.height() - btn_h)
        if self.number_checkbox:
            self.number_checkbox.resize(self.width() // 2, self.number_checkbox.height())
            self.number_checkbox.move(
                (self.width() - self.number_checkbox.width()) // 2,
                self.height() - 50,
            )

    def closeEvent(self, event):
        self.recorder.stop()
        if hasattr(self, "_listener"):
            self._listener.stop()
        if self.hotword_manager:
            self.hotword_manager.stop_watching()
        super().closeEvent(event)

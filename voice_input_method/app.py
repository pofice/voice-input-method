"""Main application window - thin UI shell delegating to VoiceEngine."""

from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QTextEdit, QCheckBox
from PySide6.QtGui import QMouseEvent, QIcon
from PySide6.QtCore import Qt, QEvent, Signal, QPointF
from pynput import keyboard

from .config import Config, resolve_resource_path
from .audio import AudioRecorder
from .recognition import SpeechRecognizer, StreamingRecognizer
from .text_processing import ChineseConverter
from .hotwords import HotwordManager
from .platform import get_backend
from .engine import VoiceEngine, EngineConfig


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
    """Thin UI shell — all business logic lives in VoiceEngine."""

    transcription_ready = Signal(str)
    partial_text_ready = Signal(str)
    text_ready = Signal(str)

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # --- Assemble dependencies for VoiceEngine ---
        backend = get_backend(config.platform)
        missing = backend.check_permissions()
        for msg in missing:
            print(f"WARNING: {msg}")

        streaming_recognizer: StreamingRecognizer | None = None
        if config.streaming:
            chunk_size = config.chunk_size or [5, 10, 5]
            streaming_recognizer = StreamingRecognizer(
                model_dir=config.streaming_model_dir,
                quantize=config.quantize,
                chunk_size=chunk_size,
            )

        engine_config = EngineConfig(
            streaming=config.streaming,
            two_pass=config.two_pass,
            enable_number_conversion=config.enable_number_conversion,
            enable_traditional_chinese=config.enable_traditional_chinese,
            chunk_size=config.chunk_size or [5, 10, 5],
        )

        chinese_converter: ChineseConverter | None = None
        if config.enable_traditional_chinese:
            lib_path = resolve_resource_path(config, "library_file")
            chinese_converter = ChineseConverter(lib_path)

        self.hotword_manager: HotwordManager | None = None
        if config.enable_hotwords:
            hw_path = resolve_resource_path(config, "hotwords_file")
            self.hotword_manager = HotwordManager(hw_path)

        # Audio recorder — wired to engine's chunk callback if streaming
        chunk_samples = streaming_recognizer.step_samples if streaming_recognizer else 0
        self.recorder = AudioRecorder(
            sample_rate=config.sample_rate,
            channels=config.channels,
            on_chunk=None,  # will be set after engine is created
            chunk_samples=chunk_samples,
        )

        self.engine = VoiceEngine(
            config=engine_config,
            recorder=self.recorder,
            recognizer=SpeechRecognizer(
                model_type=config.model_type,
                model_dir=config.model_dir,
                quantize=config.quantize,
            ),
            paster=backend,
            streaming_recognizer=streaming_recognizer,
            hotword_provider=self.hotword_manager,
            chinese_converter=chinese_converter,
            on_partial=lambda text: self.partial_text_ready.emit(text),
            on_result=lambda text: self.transcription_ready.emit(text),
        )

        # Wire streaming chunk callback now that engine exists
        if config.streaming:
            self.recorder._on_chunk = self.engine.on_audio_chunk

        # Signals → UI updates
        self.transcription_ready.connect(self._on_transcription)
        self.partial_text_ready.connect(self._on_partial_text)
        self.text_ready.connect(self._on_text_update)

        # Build UI
        self._build_ui()

        # Start engine (loads models, starts audio)
        print("Loading ASR model...")
        warmup_path = resolve_resource_path(config, "warmup_file")
        self.engine.start()
        self.engine.warmup(str(warmup_path))
        print("Models ready.")

        # Start hotkey listener
        self._setup_hotkey()

        # Start hotword file watcher
        if self.hotword_manager:
            self.hotword_manager.start_watching()

        # Drag state
        self._drag_position = None

    def _build_ui(self):
        self.setWindowOpacity(self.config.window_opacity)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)

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

    # --- Delegate to engine ---

    def _start_recording(self):
        self.engine.start_recording()

    def _stop_recording(self):
        self.engine.stop_recording()

    # --- UI callbacks ---

    def _on_partial_text(self, text: str):
        self.textEdit.setText(text)

    def _on_transcription(self, text: str):
        self.textEdit.setText(text)

    def _convert_text(self):
        import threading
        threading.Thread(target=self._convert_text_thread, daemon=True).start()

    def _convert_text_thread(self):
        text = self.textEdit.toPlainText()
        converted = self.engine.convert_chinese(text)
        if converted:
            self.text_ready.emit(converted)
            clipboard = QApplication.clipboard()
            clipboard.setText(converted)

    def _on_text_update(self, text: str):
        self.textEdit.setText(text)

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
        if self.number_checkbox:
            self.number_checkbox.resize(self.width() // 2, self.number_checkbox.height())
            self.number_checkbox.move(
                (self.width() - self.number_checkbox.width()) // 2,
                self.height() - 50,
            )

    def closeEvent(self, event):
        self.engine.shutdown()
        if hasattr(self, "_listener"):
            self._listener.stop()
        if self.hotword_manager:
            self.hotword_manager.stop_watching()
        super().closeEvent(event)

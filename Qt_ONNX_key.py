from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QTextEdit
from PyQt5.QtGui import QMouseEvent, QIcon
from PyQt5.QtCore import QProcess, Qt, QEvent, QTimer, pyqtSignal
from funasr_onnx import Paraformer
from pynput import keyboard
from pynput.keyboard import Controller, Key
import threading

class MyButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.isPressed = False

    def simulatePress(self):
        if not self.isPressed:
            pressEvent = QMouseEvent(QEvent.MouseButtonPress, self.rect().center(), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
            QApplication.postEvent(self, pressEvent)
            self.isPressed = True

    def simulateRelease(self):
        if self.isPressed:
            releaseEvent = QMouseEvent(QEvent.MouseButtonRelease, self.rect().center(), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
            QApplication.postEvent(self, releaseEvent)
            self.isPressed = False

class MyWindow(QWidget):
    # Define a signal
    transcription_ready = pyqtSignal(str)
    def __init__(self):
        super().__init__()

        # Connect the signal to a slot
        self.transcription_ready.connect(self.update_transcription)

        # 窗口置顶
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        # 设置窗口图标
        self.setWindowIcon(QIcon('./icon.png'))

        # 设置窗口的标题
        self.setWindowTitle('RTX-IM')

        # 设置窗口的大小
        self.resize(200, 100)

        # 在窗口中间添加一个按钮，大小是50×50
        self.button = MyButton(self)
        self.button.setText("输入")
        self.button.resize(200, 32)

        # 添加一个文本编辑框
        self.textEdit = QTextEdit(self)
        self.textEdit.move(0, 0)
        self.textEdit.resize(self.width(), self.height() - self.button.height())

        # 使按钮居中
        self.button.move(int((self.width() - self.button.width()) / 2), int(self.height() - self.button.height()))

        # 创建一个进程对象用于录音
        self.recorder = QProcess()

        # 初始化模型
        model_dir = "/home/pofice/.cache/modelscope/hub/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/"
        self.model = Paraformer(model_dir, batch_size=1, quantize=True)

        # 当按钮被按下时开始录音
        self.button.pressed.connect(self.startRecording)
        self.button.released.connect(self.stopRecording)

        # Setup global hotkey
        self.setup_hotkey()

        self.timer = QTimer()
        self.timer.timeout.connect(self.simulatePress)

        # Initialize isF6Pressed attribute
        self.isF6Pressed = False

        # 添加一个定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.simulatePress)

    def setup_hotkey(self):
        def on_press(key):
            try:
                if key == keyboard.Key.scroll_lock and not self.button.isPressed:  # Change here
                    self.button.simulatePress()  # 按下Scroll Lock时模拟鼠标按下事件
            except AttributeError:
                pass

        def on_release(key):
            if key == keyboard.Key.scroll_lock and self.button.isPressed:  # Change here
                self.button.simulateRelease()  # 释放Scroll Lock时模拟鼠标释放事件

        # Collect events until released
        self.listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release)
        self.listener.start()

    def startRecording(self):
        self.recorder.start("arecord", ["-f", "cd", "audio.wav"])
        print("Recording started...")
        self.isRecording = True  # 添加一个状态变量，表示正在录音

    def stopRecording(self):
        self.recorder.terminate()
        self.recorder.waitForFinished()
        print("Recording stopped")
        self.isRecording = False  # 录音结束，改变状态变量的值
        self.transcribe_audio()

    def simulatePress(self):
        if not self.button.isPressed and not self.isRecording:  # 在模拟鼠标按下事件时，检查是否正在录音
            self.button.simulatePress()

    def simulateRelease(self):
        if self.button.isPressed and not self.isRecording:  # 在模拟鼠标释放事件时，检查是否正在录音
            self.button.simulateRelease()

    def transcribe_audio(self):
        # 创建一个新的线程来执行转录的任务
        thread = threading.Thread(target=self.transcribe_audio_thread)
        thread.start()

    def transcribe_audio_thread(self):
        wav_path = ['./audio.wav']
        result = self.model(wav_path)
        print("Transcription: ", result)
        if result and 'preds' in result[0]:
            transcription = result[0]['preds'][0]
            # Emit the signal with the transcription
            self.transcription_ready.emit(transcription)

    def update_transcription(self, transcription):
        self.textEdit.setText(transcription)
        # 将文本复制到剪贴板
        clipboard = QApplication.clipboard()
        clipboard.setText(transcription)

        # 模拟按下Ctrl+V
        keyboard = Controller()
        with keyboard.pressed(Key.ctrl):  # 直接使用Key.ctrl
            keyboard.press('v')
            keyboard.release('v')

    # 让窗口可以拖拽
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragPosition = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.dragPosition)
            event.accept()

    # 让布局自适应大小
    def resizeEvent(self, event):
        # 让文本编辑框的大小自适应窗口大小
        self.textEdit.resize(self.width(), self.height() - self.button.height())
        # 让按钮的宽度自适应窗口大小，高度保持不变
        self.button.resize(self.width(), self.button.height())
        # 让按钮居中
        self.button.move(0, int(self.height() - self.button.height()))

if __name__ == "__main__":
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec_()
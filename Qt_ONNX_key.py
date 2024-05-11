from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QTextEdit
from PyQt5.QtGui import QMouseEvent, QKeyEvent
from PyQt5.QtCore import QProcess, Qt, QEvent, QTimer
from funasr_onnx import Paraformer
from pynput import keyboard
import os
import zhconv

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
    def __init__(self):
        super().__init__()

        # 窗口置顶
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        # 设置窗口的标题
        self.setWindowTitle('RTX-IM')

        # 设置窗口的大小
        self.resize(200, 100)

        # 在窗口中间添加一个按钮，大小是50×50
        self.button = MyButton(self)
        self.button.setText("输入")
        self.button.resize(100, 50)

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
                if key == keyboard.Key.f6 and not self.button.isPressed:
                    self.button.simulatePress()  # 按下F6时模拟鼠标按下事件
                    self.timer.start(100)  # 启动定时器，每100毫秒模拟一次按钮按下事件
            except AttributeError:
                pass

        def on_release(key):
            if key == keyboard.Key.f6 and self.button.isPressed:
                self.button.simulateRelease()  # 释放F6时模拟鼠标释放事件
                self.timer.stop()  # 停止定时器

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
        wav_path = ['./audio.wav']
        result = self.model(wav_path)
        print("Transcription: ", result)
        if result and 'preds' in result[0]:
            transcription = result[0]['preds'][0]
            self.textEdit.setText(transcription)
            # 将文本复制到剪贴板
            clipboard = QApplication.clipboard()
            clipboard.setText(transcription)

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
        self.textEdit.resize(self.width(), self.height() - self.button.height())
        self.button.move(int((self.width() - self.button.width()) / 2), int(self.height() - self.button.height()))

if __name__ == "__main__":
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec_()
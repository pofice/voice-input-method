from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QTextEdit
from PyQt5.QtCore import QProcess, Qt
import os
import zhconv

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        # 窗口置顶
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        # 设置窗口的标题
        self.setWindowTitle('RTX语音输入法')

        # 设置窗口的大小
        self.resize(250, 200)

        # 在窗口中间添加一个按钮，大小是50×50
        self.button = QPushButton(self)
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

        # 当按钮被按下时开始录音
        self.button.pressed.connect(self.startRecording)
        self.button.released.connect(self.stopRecording)

    def startRecording(self):
        self.recorder.start("arecord", ["-f", "cd", "audio.wav"])
        print("Recording started...")

    def stopRecording(self):
        self.recorder.terminate()
        self.recorder.waitForFinished()
        print("Recording stopped")
        try:
            # 中文或英语
            os.system("whisper --language Chinese audio.wav > text.txt")

            with open("text.txt", "r") as file:
                lines = file.readlines()
            text = ''.join([line.split('] ')[-1] for line in lines if '] ' in line])

            # 将繁体字转换为简体字
            text = zhconv.convert(text, 'zh-cn')

            self.textEdit.insertPlainText(text)

            # 将文本复制到剪贴板
            clipboard = QApplication.clipboard()
            clipboard.setText(text)

        except FileNotFoundError:
            print("Could not find audio.wav or text.txt")
        finally:
            if os.path.exists("text.txt"):
                os.remove("text.txt")

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

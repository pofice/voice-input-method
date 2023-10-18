# 导入PyQt5模块
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QTextEdit
from PyQt5.QtCore import QProcess
import os

import zhconv

# 按键触发库（不推荐）
#import keyboard
#import time

# 创建一个应用程序对象
app = QApplication([])

# 创建一个主窗口对象
window = QWidget()

# 设置窗口的标题
window.setWindowTitle('RTX语音输入法')

# 设置窗口的大小
window.resize(250, 200)

# 在窗口中间添加一个按钮，大小是50×50
button = QPushButton(window)
button.setText("输入")
button.resize(100, 50)

# 添加一个文本编辑框
textEdit = QTextEdit(window)
textEdit.move(0, 0)
textEdit.resize(window.width(), window.height() - button.height())

# 使按钮居中
button.move(int((window.width() - button.width()) / 2), int(window.height() - button.height()))

# 创建一个进程对象用于录音
recorder = QProcess()

 # 当按钮被按下时开始录音
def startRecording():
     recorder.start("arecord", ["-f", "cd", "audio.wav"])
     print("Recording started...")

def stopRecording():
    recorder.terminate()
    recorder.waitForFinished()
    print("Recording stopped")
    try:
        # 自动检测语言（不推荐）
        #os.system("whisper audio.wav > text.txt")

        # 中文或英语
        os.system("whisper --language Chinese audio.wav > text.txt")

        with open("text.txt", "r") as file:
            lines = file.readlines()
        text = ''.join([line.split('] ')[-1] for line in lines if '] ' in line])

        # 将繁体字转换为简体字
        text = zhconv.convert(text, 'zh-cn')

        textEdit.insertPlainText(text)

        # 将文本复制到剪贴板
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

    except FileNotFoundError:
        print("Could not find audio.wav or text.txt")
    finally:
        if os.path.exists("text.txt"):
            os.remove("text.txt")

button.pressed.connect(startRecording)
button.released.connect(stopRecording)

# 按键触发（不推荐）
#keyboard.on_press_key("F6", startRecording)
#keyboard.on_release_key("F6", stopRecording)

# 显示窗口
window.show()

# 进入应用程序的主循环
app.exec_()

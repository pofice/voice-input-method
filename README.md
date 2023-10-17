# linux-voice-input-method
这是在linux本地部署的基于Whisper的语音输入法

长按即可输入，它会自动把输出文本复制到剪贴板

要求有桌面环境，最好是kde
# 安装
无需创建虚拟环境，直接安装即可
```sh
pip install -U openai-whisper
```
装好依赖后即可运行
```sh
python qt.py
```

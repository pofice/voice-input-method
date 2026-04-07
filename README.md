# Voice Input Method

基于 FunASR + PySide6 的低延迟中文语音输入法。GUI 与内核完全解耦，支持 AI 自动化测试。

![Demo](demo/rtxim.png)

## 架构

```
config.yaml
    ↓
factory.py ──→ VoiceEngine (engine.py)     ← 核心，零 GUI 依赖
                ├── AudioRecorder           ← 录音（sounddevice）
                ├── SpeechRecognizer        ← 离线 ASR（FunASR Paraformer）
                ├── StreamingRecognizer     ← 流式 ASR（Paraformer-online）
                ├── PlatformBackend         ← 文本粘贴（平台相关）
                ├── HotwordManager          ← 热词加载
                └── ChineseConverter        ← 繁简转换（opencc）

app.py ─── 纯 UI 壳（PySide6），可替换
hotkey.py ─ 全局热键（pynput），可替换
```

所有依赖通过 Protocol 接口注入，每个模块可独立 mock 测试。

## 安装

Python 3.10+。

```shell
# 基础安装
pip install .

# 开发环境（含 pytest）
pip install -e ".[dev]"

# 可选功能
pip install ".[number]"         # 中文数字转阿拉伯
pip install ".[macos]"          # macOS 权限检测
pip install ".[integration]"    # 集成测试依赖（模型下载等）
```

## 模型下载

```shell
pip install modelscope

# 离线模型（必需，~250MB）
modelscope download --model iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch

# 流式模型（可选，启用 streaming 时需要，~160MB）
modelscope download --model damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online
```

在 `config.yaml` 中设置 `model_dir` 和 `streaming_model_dir`。留空则使用 modelscope 默认缓存路径。

## 运行

```shell
# 使用默认配置
voice-input

# 指定配置
voice-input --config /path/to/config.yaml

# 直接运行模块
python -m voice_input_method
```

全局热键默认 **Scroll Lock**，长按录音，松开识别。可在 `config.yaml` 中修改。

## 配置

复制 `config.yaml` 修改。核心选项：

```yaml
# ASR 模型
model_type: seaco_paraformer    # "paraformer" 或 "seaco_paraformer"（支持热词）
model_dir: ""                    # 模型目录，留空用默认缓存

# 流式识别
streaming: false                 # true 启用实时出字
streaming_model_dir: ""          # 流式模型目录
two_pass: false                  # true 启用流式+离线修正

# 热键
hotkey: scroll_lock              # scroll_lock/pause/f6-f12

# 平台
platform: ""                     # 留空自动检测，或 x11/wayland/windows/macos
```

完整配置见 [config.yaml](config.yaml)。

### 识别模式

| 模式 | 配置 | 延迟 | 准确率 |
|------|------|------|--------|
| 离线 | `streaming: false` | 录完后识别 | 最高 |
| 纯流式 | `streaming: true` | 实时出字 | 较高 |
| 2pass | `streaming: true, two_pass: true` | 实时+修正 | 高（推荐） |

## 测试

```shell
# 安装开发依赖
pip install -e ".[dev]"

# 运行单元测试（52 个，~2s，无模型/GUI/硬件依赖）
pytest tests/ -m "not integration"

# 运行集成测试（10 个，~20s，需下载 ASR 模型）
pip install ".[integration]"
pytest tests/ -m integration

# 全部测试
pytest tests/ -v

# 带覆盖率
pytest tests/ --cov=voice_input_method --cov-report=term-missing
```

### AI 测试示例

核心流水线可以用 mock 完整测试，不需要 GUI、麦克风或 ASR 模型：

```python
from voice_input_method.engine import VoiceEngine, EngineConfig
from tests.mocks import MockRecorder, MockRecognizer, MockPaster

engine = VoiceEngine(
    config=EngineConfig(),
    recorder=MockRecorder(),
    recognizer=MockRecognizer(text="测试结果"),
    paster=MockPaster(),
)
engine.start()
engine.start_recording()
engine.stop_recording()
# MockPaster.pasted == ["测试结果"]
```

也可以用工厂函数从 Config 组装完整 engine（不含 GUI）：

```python
from voice_input_method.config import load_config
from voice_input_method.factory import create_engine

config = load_config("config.yaml")
engine = create_engine(config, on_result=lambda text: print(text))
engine.start()
```

## 项目结构

```
voice_input_method/
├── engine.py           # 核心流水线（录音→识别→后处理→粘贴）
├── protocols.py        # Protocol 接口定义（AudioSource, Recognizer, TextPaster 等）
├── factory.py          # 工厂函数：Config → VoiceEngine（零 GUI 依赖）
├── app.py              # PySide6 UI 壳
├── hotkey.py           # 全局热键监听（pynput，lazy import）
├── audio.py            # 录音模块（sounddevice，自动设备检测）
├── recognition.py      # ASR 封装（离线 Paraformer + 流式 Paraformer-online）
├── text_processing.py  # 文本后处理（空格清理、数字转换、繁简转换）
├── hotwords.py         # 热词加载（纯 Python，Qt 文件监控可选）
├── config.py           # YAML 配置加载
├── platform/           # 平台后端（文本粘贴）
│   ├── base.py         # PlatformBackend ABC
│   ├── x11.py          # Ctrl+V
│   ├── wayland.py      # xdotool
│   ├── windows.py      # Ctrl+V
│   └── macos.py        # Cmd+V + 权限检测
└── resources/          # 内置资源文件

tests/
├── mocks.py            # Mock 对象库（MockRecorder, MockRecognizer 等）
├── conftest.py         # 共享 fixtures
├── fixtures/           # 测试音频（TTS 合成中文语音）
├── test_engine.py      # 核心流水线测试
├── test_integration.py # 真实 ASR 模型集成测试
├── test_audio.py       # 音频重采样测试
├── test_config.py      # 配置加载测试
├── test_hotwords.py    # 热词管理测试
├── test_hotkey.py      # 热键模块测试
├── test_factory.py     # 工厂函数测试
├── test_protocols.py   # Protocol 契约测试
└── test_text_processing.py  # 文本处理测试
```

## 平台支持

| 平台 | 输入方式 | 备注 |
|------|----------|------|
| Linux X11 | 剪贴板 + Ctrl+V | 默认 |
| Linux Wayland | xdotool 逐字输入 | 需安装 xdotool |
| Windows | 剪贴板 + Ctrl+V | |
| macOS | 剪贴板 + Cmd+V | 需授予辅助功能和麦克风权限 |

## 自定义热词

编辑 `hotwords.txt`，每行一个词。运行时修改自动热重载。

```shell
# 从 rime-ice 提取用户词库作为热词
python tools/rime_ice2hotwords.py /path/to/rime_ice.userdb.txt -o hotwords.txt
```

## 致谢

- [FunASR](https://github.com/alibaba-damo-academy/FunASR) - 阿里达摩院语音识别框架
- [PySide6](https://doc.qt.io/qtforpython-6/) - Qt for Python (LGPL)
- [rime-ice](https://github.com/iDvel/rime-ice) - Rime 输入法配置

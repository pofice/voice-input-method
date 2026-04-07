# Voice Input Method

基于 FunASR + PySide6 的低延迟中文语音输入法。GUI 与内核完全解耦，可被 AI agent 完整测试和扩展。

![Demo](demo/rtxim.png)

---

## 给 AI agent 的导读

这个 README 不试图复述代码细节（那些会过时）。下面这张表告诉你：**什么问题去看哪个文件**。读代码永远是事实的唯一来源。

| 想知道什么 | 去看哪里 |
|----------|---------|
| 整体架构如何串起来 | `voice_input_method/factory.py` — 一个文件看完所有依赖装配 |
| 核心业务流水线（录音→识别→后处理→粘贴） | `voice_input_method/engine.py` 的 `VoiceEngine` 类 |
| 各组件的接口契约 | `voice_input_method/protocols.py` — 5 个 Protocol 定义 |
| ASR 模型怎么调用 | `voice_input_method/recognition.py` |
| 录音怎么做 | `voice_input_method/audio.py` |
| 文本后处理（数字转换、繁简、热词） | `voice_input_method/text_processing.py`、`voice_input_method/hotwords.py` |
| 命令行入口 / 各命令选项 | `voice_input_method/cli.py`，或运行 `voice-input-cli --help` / `... transcribe --help` |
| GUI 怎么和 engine 交互 | `voice_input_method/app.py` — 这是一个薄壳，业务逻辑全在 `engine` 里 |
| 怎么写一个 mock 来测试 | `tests/mocks.py`，对照 `protocols.py` 实现就行 |
| 完整 mock 测试示例 | `tests/test_engine.py` |
| 用真实模型的端到端测试 | `tests/test_integration.py` |
| 配置项有哪些 | `voice_input_method/config.py` 的 `Config` dataclass + `config.yaml` |
| 平台后端如何加新平台 | `voice_input_method/platform/base.py` 的 `PlatformBackend` ABC，然后参考 `x11.py`/`macos.py` 等 |
| 支持的全局热键 | `voice_input_method/hotkey.py` 的 `_HOTKEY_NAMES` |

**架构原则**（这一段不会变，可以信赖）：
- 依赖只能从外向内：GUI/CLI → factory → engine → protocols → 具体实现
- core 层（`engine.py`、`protocols.py`、`text_processing.py`、`config.py`）从不 import GUI
- 所有外部依赖（音频硬件、ASR 模型、UI、键盘、剪贴板）都是可替换的端口适配器
- 任何核心流水线变更都应该可以用 `tests/mocks.py` 完整测试，无需 GUI/麦克风/模型

**入口点**（`pyproject.toml` `[project.scripts]` 里的真相）：
- `voice-input` → `voice_input_method.__main__:main`（GUI 模式）
- `voice-input-cli` → `voice_input_method.cli:main`（headless CLI，AI 用这个）

---

## 安装

需要 Python 3.10+。

```shell
pip install .                    # 基础安装（含 cn2an/jieba/funasr-onnx 等核心依赖）
pip install -e ".[dev]"          # 开发环境（pytest + ruff）
pip install ".[macos]"           # macOS 权限检测（需 PyObjC）
pip install ".[integration]"     # 集成测试需要的额外依赖
```

具体的 extras 组合请直接看 `pyproject.toml` 的 `[project.optional-dependencies]`。

## 模型下载

```shell
pip install modelscope

# 离线模型（CLI 默认会用这个 ID 自动下载）
modelscope download --model damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx

# 流式模型（启用 streaming 时需要）
modelscope download --model damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx
```

模型大小约 250MB / 160MB。在 `config.yaml` 设置 `model_dir`/`streaming_model_dir`，留空则用 modelscope 默认缓存。

CLI 默认模型 ID 在 `voice_input_method/cli.py` 的 `DEFAULT_OFFLINE_MODEL` / `DEFAULT_STREAMING_MODEL`，**这两个常量是真相**。

## 运行

### GUI 模式

```shell
voice-input
voice-input --config /path/to/config.yaml
```

热键长按录音、松开识别。默认热键和所有可选键见 `voice_input_method/hotkey.py`。

### CLI 模式（AI / 脚本 / CI）

```shell
voice-input-cli --help                          # 顶层帮助
voice-input-cli transcribe --help               # 单文件转写所有选项
voice-input-cli batch --help                    # 批量转写所有选项
voice-input-cli info                            # 版本和默认模型 ID

voice-input-cli transcribe input.wav            # 最简用法
voice-input-cli transcribe input.wav --json     # 结构化输出（含耗时）
```

CLI 完全 headless：吃 WAV 文件吐文字，不需要 GUI/麦克风/键盘。结构化 JSON 输出适合 AI agent 拿来判断改动有没有效果。

## 测试

```shell
pytest tests/ -m "not integration"   # 单元测试，~2s，无外部依赖
pytest tests/ -m integration         # 集成测试，需先下载模型
pytest tests/                        # 全部
pytest tests/ --cov=voice_input_method
```

测试分两层：
- **单元测试**：用 `tests/mocks.py` 的 mock 对象，跑 engine 的所有路径，零外部依赖
- **集成测试**：`tests/test_integration.py` 用真实 Paraformer ONNX 模型 + 真实中文语音音频（`tests/fixtures/`）

### AI 怎么用 mock 测试核心流水线

最小例子（详见 `tests/test_engine.py`）：

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
```

要从 `Config` 直接装配真实 engine（不要 GUI），用 `voice_input_method.factory.create_engine`，详见该文件。

## 平台

| 平台 | 输入方式 | 注意事项 |
|------|---------|---------|
| Linux X11 | 剪贴板 + Ctrl+V | 默认 |
| Linux Wayland | xdotool 逐字输入 | 需安装 xdotool |
| Windows | 剪贴板 + Ctrl+V | |
| macOS | 剪贴板 + Cmd+V | 需授予辅助功能和麦克风权限 |

各平台的具体实现在 `voice_input_method/platform/{x11,wayland,windows,macos}.py`，每个文件不到 30 行。加新平台只需要继承 `PlatformBackend` 并在 `platform/__init__.py:get_backend()` 注册。

## 自定义热词

编辑 `hotwords.txt`，每行一个词，运行时修改自动热重载。规则见 `voice_input_method/hotwords.py` 的 `HotwordManager`。

```shell
python tools/rime_ice2hotwords.py /path/to/rime_ice.userdb.txt -o hotwords.txt
```

## 致谢

- [FunASR](https://github.com/alibaba-damo-academy/FunASR) — 阿里达摩院语音识别框架
- [PySide6](https://doc.qt.io/qtforpython-6/) — Qt for Python (LGPL)
- [rime-ice](https://github.com/iDvel/rime-ice) — Rime 输入法配置

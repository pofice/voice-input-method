# AI 开发规范

本文件是给 AI agent（Claude Code 等）的开发指引。人类开发者也适用。

## 架构约束

**依赖方向（不可违反）**：

```
GUI (app.py)  /  CLI (cli.py)
        ↓
    factory.py
        ↓
    engine.py
        ↓
    protocols.py
        ↓
audio / recognition/ (funasr, sherpa-sensevoice, sherpa-nano) / indicator / hotwords / platform / text_processing
```

- 依赖只能向下，不能向上
- `engine.py`、`protocols.py`、`text_processing.py`、`config.py` 禁止 import GUI（PySide6）
- 新增模块必须先在 `protocols.py` 定义 Protocol 接口，再写实现
- `factory.py` 是唯一的装配点，负责实例化所有依赖并注入 engine

**依赖管理**：

- 新增的 Python 包必须加到 `pyproject.toml` 的 `dependencies`
- 禁止用 `try: import xxx` 做优雅降级来代替声明依赖——用户 `pip install` 后功能必须完整可用
- 平台相关的可选依赖放 `[project.optional-dependencies]`（如 `macos` extras）

**测试要求**：

- 新功能必须有对应的单元测试
- 新的 Protocol 必须在 `tests/mocks.py` 添加 Mock 实现
- 新的 Mock 必须在 `tests/test_protocols.py` 添加 Protocol 契约测试
- 测试不依赖 GUI、麦克风、ASR 模型——用 mock

## 变更同步清单

每次修改代码后，逐项检查：

- [ ] `README.md` 导读表——新增、删除或重命名了文件？更新表格
- [ ] `README.md` 功能表——新增功能、删除功能、改了默认值？更新表格
- [ ] `config.yaml`——和 `config.py` 的 `Config` dataclass 字段一致？新字段要加示例和注释
- [ ] `pyproject.toml`——新依赖加了？版本号需要 bump？
- [ ] `tests/mocks.py`——新 Protocol 需要新 Mock？
- [ ] `tests/test_protocols.py`——新 Mock 需要契约测试？

原则：**README 是索引不是复述**。告诉读者去哪个文件找答案，不要把代码细节抄进 README。

## 提交规范

格式：`<type>(scope): 中文描述`

```
feat(indicator): 新增 macOS 原生录音指示器
fix(deps): 将 noisereduce 加入核心依赖
docs(README): 更新功能表和导读表
refactor(hotkey): 合并两个热键监听器为 CombinedHotkeyListener
test(indicator): 添加 NullIndicator 和工厂函数测试
chore(ci): 添加 GitHub Actions 单元测试
perf(engine): 启动时预加载 jieba 词典
```

- type 用英文（feat/fix/docs/refactor/test/chore/perf）
- scope 可选，用英文模块名
- 描述用中文，简明扼要
- 每个功能点一个 commit，不要混多个不相关改动

## 切换识别后端

项目支持三个识别后端，由 `config.recognizer_backend` 选择：

| backend | 模型 | 体积 | 特点 | 依赖 |
|---------|------|------|------|------|
| `funasr`（默认） | SeacoParaformer | ~370MB | 支持热词，streaming，funasr_onnx | 核心依赖 |
| `sherpa-sensevoice` | SenseVoice-Small | ~229MB (int8) | 内置标点+ITN，多语言，快 | `pip install -e ".[sherpa]"` |
| `sherpa-nano` | Fun-ASR-Nano (LLM) | ~800MB (int8) | 方言/口音/噪声最强 | `pip install -e ".[sherpa]"` |

### CLI 方式（一次性切换）

```shell
# funasr（默认，不传 --backend 也行）
voice-input-cli transcribe input.wav

# sherpa-sensevoice
voice-input-cli transcribe input.wav \
  --backend sherpa-sensevoice \
  --sensevoice-model /path/to/model.int8.onnx \
  --sensevoice-tokens /path/to/tokens.txt

# sherpa-nano
voice-input-cli transcribe input.wav \
  --backend sherpa-nano \
  --nano-model-dir /path/to/sherpa-onnx-funasr-nano-int8-2025-12-30
```

同样的参数也适用于 `batch` 和 `doctor` 子命令。

### config.yaml 方式（GUI 常驻）

```yaml
recognizer_backend: sherpa-nano     # funasr / sherpa-sensevoice / sherpa-nano

# sherpa-sensevoice 字段（只在该后端下生效）
sensevoice_model_path: ""
sensevoice_tokens_path: ""
sensevoice_language: "zh"

# sherpa-nano 字段（只在该后端下生效）
nano_model_dir: ""
```

切完 config.yaml 直接 `voice-input` 启动即可。

### 模型下载

- **sherpa-sensevoice**: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
- **sherpa-nano**: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2

### 注意事项

- `nano_model_dir` 是**目录路径**，目录内必须有 `encoder_adaptor.int8.onnx`、`llm.int8.onnx`、`embedding.int8.onnx`、`Qwen3-0.6B/`
- sherpa 后端缺必填字段会立刻抛 `ConfigError`（见 `factory._create_recognizer`），不会等 sherpa-onnx 爆 FileNotFoundError
- `streaming: true` 只对 funasr 生效；配合 sherpa 后端会发 `RuntimeWarning` 并自动禁用
- 新增后端的步骤：`recognition/` 下加新模块（实现 `Recognizer` Protocol）→ `factory._create_recognizer` 加分支 → `config.py` 加字段 → `config.yaml` 加示例 → `CLAUDE.md` 本表更新

## 快速上手

```shell
# 部署 + 验证
git clone https://github.com/pofice/voice-input-method.git
cd voice-input-method
pip install -e ".[dev]"
voice-input-cli doctor

# 跑测试
pytest tests/ -m "not integration"

# 理解架构
# 从 factory.py 开始读——一个文件看完所有依赖装配
```

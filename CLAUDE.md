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

## 识别后端约定

> 用户操作（CLI 命令、config.yaml 字段、模型下载）见 README《识别后端》章节。本节只写 AI 改代码时的契约。

**契约**：

- 所有后端都必须实现 `protocols.Recognizer`（load / warmup / transcribe）
- `factory._create_recognizer` 是唯一的实例化入口，禁止在 `cli.py` / `app.py` / 测试外硬编码具体后端类
- 缺必填配置时必须抛 `factory.ConfigError` 并指出具体字段，**不允许**让 sherpa-onnx / funasr-onnx 自己爆 FileNotFoundError
- `streaming: true` 仅对 `funasr` 后端有效；其他后端组合时 factory 必须发 `RuntimeWarning` 并禁用 streaming，不允许静默失败

**新增后端的标准步骤**：

1. `recognition/{backend_name}.py` — 实现 `Recognizer` Protocol（参考 `sherpa_sensevoice.py`）
2. `config.py` — 在 `Config` dataclass 加该后端的必填字段（`{backend}_model_path` 等）
3. `factory._create_recognizer` — 加 `if backend == "..."` 分支，**先**校验必填字段，缺失抛 `ConfigError`
4. `config.yaml` — 加字段示例和注释
5. `cli.py` — 在 `add_backend_args` 注册对应的 `--{backend}-*` 参数，并在 `_build_config` 里映射到 Config
6. `README.md` — 后端表格 + CLI 命令 + config.yaml 示例 + 模型下载链接
7. `tests/` — mock 测试覆盖 ConfigError 路径

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

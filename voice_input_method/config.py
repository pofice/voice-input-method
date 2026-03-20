"""Configuration management."""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Config:
    # Model
    model_type: str = "seaco_paraformer"  # "paraformer" or "seaco_paraformer"
    model_dir: str = ""
    quantize: bool = True

    # Audio
    sample_rate: int = 44100
    channels: int = 2

    # Hotkey
    hotkey: str = "scroll_lock"

    # UI
    window_title: str = "Rtxime"
    window_width: int = 200
    window_height: int = 100
    window_opacity: float = 0.8

    # Streaming
    streaming: bool = False  # Enable real-time streaming recognition
    streaming_model_dir: str = ""  # Path to online model (required if streaming=True)
    chunk_size: list = None  # [left, body, right] in frames, default [5, 10, 5]
    two_pass: bool = False  # 2pass mode: stream + offline correction

    # Features
    enable_hotwords: bool = True
    enable_number_conversion: bool = False
    enable_traditional_chinese: bool = True

    # Platform override (auto-detected if empty)
    platform: str = ""

    # Paths (resolved relative to config file or package)
    hotwords_file: str = "hotwords.txt"
    library_file: str = "library.txt"
    warmup_file: str = "warmup.wav"
    icon_file: str = "icon.png"
    style_file: str = "style.css"


def detect_platform() -> str:
    if sys.platform == "darwin":
        return "macos"
    elif sys.platform == "win32":
        return "windows"
    elif sys.platform.startswith("linux"):
        session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
        if session_type == "wayland":
            return "wayland"
        return "x11"
    return "x11"


def load_config(config_path: str | None = None) -> Config:
    """Load configuration from YAML file, falling back to defaults."""
    config = Config()

    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)

    if not config.platform:
        config.platform = detect_platform()

    return config


def resolve_resource_path(config: Config, attr: str) -> Path:
    """Resolve a resource file path. Checks CWD first, then package resources dir."""
    filename = getattr(config, attr)
    # Check CWD
    cwd_path = Path.cwd() / filename
    if cwd_path.exists():
        return cwd_path
    # Check package resources
    pkg_path = Path(__file__).parent / "resources" / filename
    if pkg_path.exists():
        return pkg_path
    # Fall back to CWD path (will fail later with a clear error)
    return cwd_path

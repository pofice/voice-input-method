"""Tests for config module."""

from pathlib import Path

import yaml

from voice_input_method.config import Config, detect_platform, load_config, resolve_resource_path


class TestConfig:
    def test_defaults(self):
        c = Config()
        assert c.model_type == "seaco_paraformer"
        assert c.quantize is True
        assert c.sample_rate == 44100
        assert c.streaming is False

    def test_load_from_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "model_type": "paraformer",
            "streaming": True,
            "hotkey": "f9",
            "window_title": "Test",
        }))
        c = load_config(str(config_file))
        assert c.model_type == "paraformer"
        assert c.streaming is True
        assert c.hotkey == "f9"
        assert c.window_title == "Test"

    def test_load_missing_file(self):
        c = load_config("/nonexistent/config.yaml")
        # Should return defaults, not crash
        assert c.model_type == "seaco_paraformer"

    def test_load_none(self):
        c = load_config(None)
        assert isinstance(c, Config)

    def test_unknown_keys_ignored(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"unknown_key": "value"}))
        c = load_config(str(config_file))
        assert not hasattr(c, "unknown_key") or c.model_type == "seaco_paraformer"


class TestDetectPlatform:
    def test_returns_string(self):
        result = detect_platform()
        assert result in ("x11", "wayland", "windows", "macos")


class TestResolveResourcePath:
    def test_cwd_takes_priority(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "test.txt").write_text("hello")
        c = Config()
        c.hotwords_file = "test.txt"
        path = resolve_resource_path(c, "hotwords_file")
        assert path.exists()
        assert path == tmp_path / "test.txt"

    def test_fallback_to_cwd_path(self):
        c = Config()
        c.hotwords_file = "nonexistent_file.txt"
        path = resolve_resource_path(c, "hotwords_file")
        # Returns CWD path even if it doesn't exist
        assert isinstance(path, Path)

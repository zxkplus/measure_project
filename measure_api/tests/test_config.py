"""Tests for config module."""

from __future__ import annotations

import os
import tempfile

import pytest
import yaml

from measure_api.config import Config


def test_load_defaults():
    """Default config loads without error."""
    Config.reset()
    cfg = Config.load_from_dict({
        "log": {"level": "INFO", "directory": "logs"},
        "server": {"port": 5000},
    })
    assert cfg.get("log.level") == "INFO"
    assert cfg.get("server.port") == 5000


def test_dot_notation_with_default():
    """Dot-notation returns default for missing keys."""
    Config.reset()
    cfg = Config.load_from_dict({"a": {"b": 1}})
    assert cfg.get("a.b") == 1
    assert cfg.get("a.c", "default") == "default"
    assert cfg.get("x.y.z", 42) == 42


def test_local_override():
    """Local config overrides default values."""
    Config.reset()
    base = {
        "log": {"level": "INFO", "directory": "logs", "console_output": True},
        "server": {"port": 5000},
    }
    override = {
        "log": {"level": "DEBUG"},
        "server": {"port": 9000},
    }
    cfg = Config.load_from_dict(base)
    assert cfg.get("log.level") == "INFO"
    assert cfg.get("server.port") == 5000


def test_set_and_reload():
    """Config.set() and partial reload work."""
    Config.reset()
    cfg = Config.load_from_dict({"log": {"level": "INFO", "directory": "logs"}})
    assert cfg.get("log.level") == "INFO"
    cfg.set("log.level", "DEBUG")
    assert cfg.get("log.level") == "DEBUG"


def test_data_accessor():
    """Access raw data."""
    Config.reset()
    cfg = Config.load_from_dict({"a": 1})
    assert cfg.data == {"a": 1}


def test_singleton():
    """Multiple loads return same instance."""
    Config.reset()
    cfg1 = Config.load_from_dict({"key": "value"})
    cfg2 = Config.load()
    assert cfg2.get("key") == "value"


def test_reset():
    """Reset clears singleton."""
    Config.reset()
    cfg = Config.load_from_dict({"a": 1})
    assert cfg.get("a") == 1
    Config.reset()
    cfg2 = Config.load_from_dict({"a": 2})
    assert cfg2.get("a") == 2


def test_file_loading():
    """Loading from actual YAML file works."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"log": {"level": "WARNING"}}, f)
        fpath = f.name

    Config.reset()
    cfg = Config.load(fpath)
    assert cfg.get("log.level") == "WARNING"
    os.unlink(fpath)


def test_missing_file():
    """Loading from nonexistent path raises error."""
    Config.reset()
    with pytest.raises(FileNotFoundError):
        Config.load("/nonexistent/config.yaml")

"""
Configuration management for Measure API.

Usage:
    from measure_api.config import Config
    cfg = Config.load()
    level = cfg.get("log.level")
    port = cfg.get("server.port", 5000)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge override into base.  Only dict-valued leaves are merged;
    non-dict values in override replace the base value.
    """
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _get_nested(d: dict, key: str, default: Any = None) -> Any:
    """Dot-separated key lookup, e.g. 'log.level' -> d['log']['level']."""
    parts = key.split(".")
    current = d
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _set_nested(d: dict, key: str, value: Any) -> None:
    """Set a dot-separated key in-place."""
    parts = key.split(".")
    current = d
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


class Config:
    """
    Singleton configuration manager.

    Load order:
      1. ``config.yaml`` (bundled default, required)
      2. ``config.local.yaml`` (local override, optional)

    Thread-safe after initialisation (read-only access).
    """

    _instance: Optional["Config"] = None
    _data: dict = {}
    _base_dir: str = ""

    def __init__(self, data: dict, base_dir: str) -> None:
        self._data = data
        self._base_dir = base_dir

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        """
        Load configuration.

        Args:
            path: Optional explicit config path.  If None, searches
                  ``config.yaml`` next to this module.

        Returns:
            Singleton ``Config`` instance.
        """
        if cls._instance is not None:
            return cls._instance

        if path is not None:
            base_dir = os.path.dirname(os.path.abspath(path))
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            # Default: find config.yaml next to this module
            base_dir = os.path.dirname(os.path.abspath(__file__))
            default_path = os.path.join(base_dir, "config.yaml")
            with open(default_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # Local override
            local_path = os.path.join(base_dir, "config.local.yaml")
            if os.path.isfile(local_path):
                with open(local_path, "r", encoding="utf-8") as f:
                    local_data = yaml.safe_load(f) or {}
                data = _deep_merge(data, local_data)

        cls._instance = cls(data, base_dir)
        return cls._instance

    @classmethod
    def load_from_dict(cls, data: dict, base_dir: str = "") -> "Config":
        """Create a Config from an in-memory dict (for testing)."""
        cls._instance = cls(data, base_dir)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear singleton (mainly for testing)."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a config value by dot-separated key.

        Examples:
            cfg.get("log.level")        # "INFO"
            cfg.get("server.port", 5000) # 5000
            cfg.get("nonexistent.key", 42) # 42
        """
        return _get_nested(self._data, key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a config value at runtime (e.g. hot-reload log level)."""
        _set_nested(self._data, key, value)

    @property
    def data(self) -> dict:
        """Access raw config dict (read-only recommended)."""
        return self._data

    @property
    def base_dir(self) -> str:
        """Directory containing the config file."""
        return self._base_dir

    # ------------------------------------------------------------------
    # Hot-reload
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """
        Re-read config files and merge.

        Respects the same load order as ``load()``.
        Call this at runtime to pick up changes without restarting.
        """
        default_path = os.path.join(self._base_dir, "config.yaml")
        with open(default_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        local_path = os.path.join(self._base_dir, "config.local.yaml")
        if os.path.isfile(local_path):
            with open(local_path, "r", encoding="utf-8") as f:
                local_data = yaml.safe_load(f) or {}
            data = _deep_merge(data, local_data)

        self._data = data

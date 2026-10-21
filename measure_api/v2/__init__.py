"""Headless FastAPI measurement service public exports."""

from typing import Any

__all__ = ["create_app", "ServiceSettings"]


def __getattr__(name: str) -> Any:
    """Load the public service factory without importing legacy GUI factories."""
    if name == "create_app":
        from measure_api.v2.app import create_app

        return create_app
    if name == "ServiceSettings":
        from measure_api.v2.models import ServiceSettings

        return ServiceSettings
    raise AttributeError(name)

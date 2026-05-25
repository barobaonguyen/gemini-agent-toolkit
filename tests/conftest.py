from __future__ import annotations

from typing import Any


def config_value(config: Any, name: str) -> Any:
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name)


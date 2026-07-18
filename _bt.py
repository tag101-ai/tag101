"""Small compatibility layer around Bittensor imports.

Most unit tests exercise the off-chain pieces and should not need a live
Bittensor install. Runtime entry points call :func:`require_bittensor` before
touching chain objects.
"""

from __future__ import annotations

import os
from types import ModuleType
from typing import Any


try:  # pragma: no cover - depends on optional runtime dependency
    os.environ.setdefault("BT_NO_PARSE_CLI_ARGS", "1")
    import bittensor as bt  # type: ignore
except Exception:  # pragma: no cover - exercised when dependency is absent
    bt = None  # type: ignore


def require_bittensor() -> Any:
    if bt is None:
        raise RuntimeError(
            "bittensor is required for miner/validator runtime. "
            "Install the project with `pip install -e .` first."
        )
    return bt


def bittensor_attr(lower_name: str, upper_name: str | None = None) -> Any:
    module = require_bittensor()
    for name in (upper_name, lower_name):
        if name and hasattr(module, name):
            candidate = getattr(module, name)
            if not isinstance(candidate, ModuleType):
                return candidate
    submodule = getattr(module, lower_name, None)
    if isinstance(submodule, ModuleType):
        for name in (upper_name, lower_name.capitalize()):
            if name and hasattr(submodule, name):
                return getattr(submodule, name)
    raise AttributeError(f"bittensor has no {lower_name!r} constructor")

"""Standard-library logging setup for the neuron runtime.

bittensor 11 removed ``bt.logging``; the SDK now emits through the stdlib
``logging`` module under the ``bittensor`` namespace and leaves handler/level
configuration to the application. These helpers install that configuration.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

_CONFIGURED = False


def configure_logging(config: Any = None) -> None:
    """Install stream/file handlers and set levels from the neuron config (idempotent)."""
    global _CONFIGURED

    debug = False
    trace = False
    log_dir: str | None = None
    logging_cfg = getattr(config, "logging", None) if config is not None else None
    if logging_cfg is not None:
        debug = bool(getattr(logging_cfg, "debug", False))
        trace = bool(getattr(logging_cfg, "trace", False))
        log_dir = getattr(logging_cfg, "logging_dir", None)

    app_level = logging.DEBUG if (debug or trace) else logging.INFO

    if not _CONFIGURED:
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        root = logging.getLogger()
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        root.addHandler(stream)
        if log_dir:
            try:
                path = Path(os.path.expanduser(str(log_dir)))
                path.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(path / "tag101.log")
                file_handler.setFormatter(formatter)
                root.addHandler(file_handler)
            except OSError:
                pass
        _CONFIGURED = True

    logging.getLogger("tag101").setLevel(app_level)
    logging.getLogger("bittensor").setLevel(logging.DEBUG if trace else logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return the ``tag101.<name>`` logger."""
    return logging.getLogger(f"tag101.{name}")

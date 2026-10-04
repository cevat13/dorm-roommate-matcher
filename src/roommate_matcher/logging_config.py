"""Logging setup shared by the CLI, the API and the UI."""

from __future__ import annotations

import logging
import sys
from typing import Final

from roommate_matcher.config import get_settings

_CONFIGURED: dict[str, bool] = {"done": False}

_PLAIN_FORMAT: Final = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_JSON_FORMAT: Final = (
    '{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
)


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger once per process.

    Args:
        level: Optional override for the configured log level.
    """
    if _CONFIGURED["done"]:
        return
    settings = get_settings()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_JSON_FORMAT if settings.log_json else _PLAIN_FORMAT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level or settings.log_level)
    _CONFIGURED["done"] = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: Logger name, conventionally ``__name__``.

    Returns:
        A logger attached to the configured root handler.
    """
    configure_logging()
    return logging.getLogger(name)

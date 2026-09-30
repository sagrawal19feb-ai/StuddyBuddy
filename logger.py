"""
logger.py — Professional logging subsystem for StudyBuddy
=============================================================

Three separate log streams are maintained so that different audiences
(humans, developers, the chat transcript) never pollute each other:

  * ``logs/debug.log``   — verbose developer diagnostics (DEBUG level)
  * ``logs/error.log``   — warnings and errors only (WARNING +)
  * ``logs/chat.log``    — a plain transcript of every user/bot exchange

Design
------
There is a single root logger named ``studybuddy``. All three file handlers
are attached to it; each handler filters by its own level, so a single
``log.info(...)`` call lands in ``debug.log`` while a ``log.warning(...)``
call lands in *both* ``debug.log`` and ``error.log``.

Module code obtains a namespaced child logger via :func:`get_logger`
(``studybuddy.<module>``); the standard ``logging`` hierarchy makes every
record propagate to the root where the handlers live. Handlers rotate so
log files can never grow without bound.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import (
    CHAT_LOG_FILE,
    DEBUG_LOG_FILE,
    ERROR_LOG_FILE,
    ensure_directory_structure,
)

# --- colorama: make ANSI codes safe on Windows ---------------------------------
try:
    from colorama import just_fix_windows_console, Fore, Style

    just_fix_windows_console()
except ImportError:  # pragma: no cover - defensive
    class Fore:  # type: ignore[no-redef]
        RED = GREEN = YELLOW = CYAN = MAGENTA = ""
        LIGHTBLACK_EX = LIGHTRED_EX = LIGHTCYAN_EX = ""

    class Style:  # type: ignore[no-redef]
        RESET_ALL = DIM = ""


# --- Level colours used by the console formatter ---------------------------------
_LEVEL_COLOURS: dict[int, str] = {
    logging.DEBUG: Fore.LIGHTBLACK_EX,
    logging.INFO: Fore.CYAN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.MAGENTA,
}

_ROOT_NAME = "studybuddy"
_CHAT_NAME = f"{_ROOT_NAME}.chat"

_configured = False  # module-level guard so handlers are attached only once


class ColourFormatter(logging.Formatter):
    """Log formatter that colours the level name on interactive terminals."""

    def format(self, record: logging.LogRecord) -> str:
        colour = _LEVEL_COLOURS.get(record.levelno, "")
        original = record.levelname
        record.levelname = f"{colour}{original:<8}{Style.RESET_ALL}"
        message = super().format(record)
        record.levelname = original  # restore for other formatters sharing the record
        return message


def _file_handler(path: Path, level: int, fmt: str) -> logging.Handler:
    """Build a rotating file handler for one log stream."""
    handler = RotatingFileHandler(
        path,
        maxBytes=2 * 1024 * 1024,  # 2 MB
        backupCount=3,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))
    return handler


def _configure_root_logger() -> None:
    """Attach the debug/error/console handlers to the root ``studybuddy`` logger."""
    global _configured
    if _configured:
        return
    ensure_directory_structure()

    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(logging.DEBUG)
    root.propagate = False  # never leak records to the global root logger

    root.addHandler(_file_handler(
        DEBUG_LOG_FILE, logging.DEBUG,
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    ))
    root.addHandler(_file_handler(
        ERROR_LOG_FILE, logging.WARNING,
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    ))
    # Console handler: surface warnings and errors directly to stderr.
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.WARNING)
    console.setFormatter(ColourFormatter(
        "%(levelname)-8s | %(name)s | %(message)s"
    ))
    root.addHandler(console)
    _configured = True


def _chat_logger() -> logging.Logger:
    """The dedicated chat-transcript logger (plain text, INFO only)."""
    ensure_directory_structure()
    chat = logging.getLogger(_CHAT_NAME)
    chat.setLevel(logging.INFO)
    chat.propagate = False
    if not chat.handlers:
        chat.addHandler(_file_handler(
            CHAT_LOG_FILE, logging.INFO,
            "%(asctime)s | %(message)s",
        ))
    return chat


# ---------------------------------------------------------------------------
# Module-level facade — the rest of the codebase uses these two functions
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger wired into the StudyBuddy log files.

    The returned object is a standard :class:`logging.Logger` child of the
    ``studybuddy`` root, so callers can use ``logger.info(...)``,
    ``logger.warning(...)`` etc. as usual.
    """
    _configure_root_logger()
    logger = logging.getLogger(f"{_ROOT_NAME}.{name}")
    logger.setLevel(logging.DEBUG)
    return logger


def log_chat(role: str, message: str) -> None:
    """Append a single exchange to the plain-text chat transcript."""
    _chat_logger().info("%-5s | %s", role.upper(), message)

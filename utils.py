"""
utils.py — Shared helper functions for StudyBuddy
=====================================================

Small, dependency-free utilities used across the whole application:

  * JSON reading / writing with atomic saves and friendly error handling
  * Date & time formatting
  * Random selection that avoids immediate repetition
  * Weighted random choice
  * Text helpers (pluralisation, sanitisation, deduplication)

Every function here is intentionally pure and unit-testable.
"""

from __future__ import annotations

import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, TypeVar

from config import Config, build_config

T = TypeVar("T")

# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------
def load_json(path: Path | str, default: Any = None) -> Any:
    """Safely load a JSON file.

    If the file is missing or malformed, the caller-provided ``default`` is
    returned instead of raising — a corrupt data file must never crash the
    chatbot, it should just fall back to sensible defaults.

    Parameters
    ----------
    path : Path | str
        Location of the JSON file.
    default : Any
        Value returned when the file cannot be read.

    Returns
    -------
    Any
        Parsed JSON content or ``default``.
    """
    file_path = Path(path)
    if not file_path.exists():
        return default
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return default


def save_json(path: Path | str, data: Any) -> bool:
    """Atomically write ``data`` to a JSON file.

    The file is first written to a temporary name in the same directory and
    then renamed over the target. This guarantees that a crash mid-write can
    never leave a half-written (corrupt) JSON file behind.

    Parameters
    ----------
    path : Path | str
        Destination file path.
    data : Any
        JSON-serialisable content.

    Returns
    -------
    bool
        ``True`` on success, ``False`` on failure.
    """
    file_path = Path(path)
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
        temp_path.replace(file_path)
        return True
    except (OSError, TypeError):
        # Don't leave a half-written temp file behind on failure.
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False


# ---------------------------------------------------------------------------
# Date & time helpers
# ---------------------------------------------------------------------------
def now() -> datetime:
    """Return the current local datetime."""
    return datetime.now()


def date_str(date: Optional[datetime] = None) -> str:
    """Human-friendly date, e.g. ``2026-08-05``."""
    return (date or now()).strftime("%Y-%m-%d")


def time_str(date: Optional[datetime] = None) -> str:
    """Human-friendly time, e.g. ``14:32``."""
    return (date or now()).strftime("%H:%M")


def timestamp_str(date: Optional[datetime] = None) -> str:
    """Full timestamp for logs, e.g. ``2026-08-05 14:32:07``."""
    return (date or now()).strftime("%Y-%m-%d %H:%M:%S")


def hour_of_day(date: Optional[datetime] = None) -> int:
    """Return the current hour (0-23)."""
    return (date or now()).hour


def greeting_for_hour(hour: int) -> str:
    """Return a time-of-day greeting word for the given hour (0-23)."""
    if hour < 5:
        return "Good night"
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    return "Good evening"


# ---------------------------------------------------------------------------
# Random selection helpers
# ---------------------------------------------------------------------------
class NonRepeatingRandom:
    """Random picker that never returns the same item twice in a row.

    Useful for study tips and motivational quotes where an immediate repeat
    of the same line would make the chatbot feel scripted.

    Example
    -------
    >>> picker = NonRepeatingRandom(["a", "b", "c"])
    >>> picker.pick()   # any of a/b/c
    >>> picker.pick()   # guaranteed to differ from the previous pick
    """

    def __init__(self, items: Sequence[T], rng: Optional[random.Random] = None) -> None:
        self._items = list(items)
        self._rng = rng or random
        self._last_index: Optional[int] = None

    def pick(self) -> Optional[T]:
        """Return a random item, avoiding the previous one if possible."""
        if not self._items:
            return None
        if len(self._items) == 1:
            return self._items[0]
        candidates = [
            i for i in range(len(self._items))
            if i != self._last_index
        ]
        chosen = self._rng.choice(candidates)
        self._last_index = chosen
        return self._items[chosen]


def random_choice(items: Sequence[T]) -> T:
    """Random choice with a safe fallback for empty sequences."""
    if not items:
        raise ValueError("random_choice() called with an empty sequence")
    return random.choice(list(items))


def weighted_pick(
    items: Sequence[T],
    weights: Sequence[float],
    rng: Optional[random.Random] = None,
) -> T:
    """Pick an item from ``items`` proportionally to ``weights``.

    Raises
    ------
    ValueError
        If the sequences are empty or of mismatched length.
    """
    if not items or len(items) != len(weights):
        raise ValueError("items and weights must be non-empty and of equal length")
    total = sum(weights)
    if total <= 0:
        return items[0]
    pick = (rng or random).uniform(0.0, total)
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += weight
        if pick <= cumulative:
            return item
    return items[-1]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def pluralise(count: int, singular: str, plural: Optional[str] = None) -> str:
    """Return ``singular`` or ``plural`` depending on ``count``."""
    if count == 1:
        return singular
    return plural if plural else f"{singular}s"


def sanitise_filename(name: str) -> str:
    """Convert arbitrary text into a safe file name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._")
    return cleaned or "untitled"


def dedupe(sequence: Iterable[T]) -> list[T]:
    """Remove duplicates while preserving first-seen order."""
    seen: set[T] = set()
    result: list[T] = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def clamp(value: float, low: float, high: float) -> float:
    """Constrain ``value`` to the inclusive range ``[low, high]``."""
    return max(low, min(high, value))


def strip_filler(text: str, fillers: Sequence[str]) -> str:
    """Remove leading/trailing filler phrases (case-insensitive)."""
    lowered = text.lower().strip()
    for filler in fillers:
        if lowered.startswith(filler):
            text = text[len(filler):].strip()
            lowered = text.lower().strip()
        if lowered.endswith(filler):
            text = text[: -len(filler)].strip()
            lowered = text.lower().strip()
    return text.strip()


def progress_percent(done: int, total: int) -> float:
    """Return the completion percentage, safely handling ``total == 0``."""
    if total <= 0:
        return 0.0
    return round(clamp(done / total * 100.0, 0.0, 100.0), 1)


# ---------------------------------------------------------------------------
# Application-level helpers
# ---------------------------------------------------------------------------
def load_config() -> Config:
    """Load the application configuration (cached per process)."""
    return build_config()


def load_asset_text(relative_path: Path | str) -> str:
    """Read a text asset (e.g. the ASCII logo) relative to the project root."""
    asset = Path(__file__).resolve().parent / relative_path
    try:
        return asset.read_text(encoding="utf-8")
    except OSError:
        return ""


def is_yes(text: str) -> bool:
    """Best-effort detection of an affirmative answer."""
    return text.strip().lower() in {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "definitely", "of course"}


def ordinal(n: int) -> str:
    """Return the ordinal form of a number: 1 -> '1st', 3 -> '3rd', 11 -> '11th'."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def is_no(text: str) -> bool:
    """Best-effort detection of a negative answer."""
    return text.strip().lower() in {"no", "n", "nope", "nah", "not really", "no thanks"}

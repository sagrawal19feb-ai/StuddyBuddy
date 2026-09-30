"""
config.py — Central configuration for StudyBuddy
====================================================

This module is the single source of truth for:

  * Application metadata (name, version, author)
  * Filesystem layout (data / user_data / logs / assets)
  * User settings (theme, name, preferred hours, notifications, ...)
  * Environment variables loaded from a ``.env`` file (``python-dotenv``)

Keeping every path and default in one place means the rest of the codebase
never has to guess where a file lives — it always asks :class:`Config`.

Design notes
------------
* Pure module-level functions are used where no state is required.
* :class:`Config` is a small value object; the heavier read/write behaviour
  (profile, progress, settings) lives in :mod:`storage`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at runtime but required by requirements.txt
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - defensive fallback
    load_dotenv = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------
APP_NAME: str = "StudyBuddy"
APP_SLUG: str = "studybuddy"  # safe, lowercase slug
APP_TAGLINE: str = "Your personal study companion — learn smarter, not harder"
APP_AUTHOR: str = "StudyBuddy Team"
APP_YEAR: str = "2026"

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
# ``BASE_DIR`` is the directory that contains this file (project root).
BASE_DIR: Path = Path(__file__).resolve().parent

ASSETS_DIR: Path = BASE_DIR / "assets"
DATA_DIR: Path = BASE_DIR / "data"
USER_DATA_DIR: Path = BASE_DIR / "user_data"
CHAT_HISTORY_DIR: Path = USER_DATA_DIR / "chat_history"
LOGS_DIR: Path = BASE_DIR / "logs"

# Well-known data files
RESPONSES_FILE: Path = DATA_DIR / "responses.json"
INTENTS_FILE: Path = DATA_DIR / "intents.json"
STUDY_TIPS_FILE: Path = DATA_DIR / "study_tips.json"
MOTIVATION_FILE: Path = DATA_DIR / "motivation.json"
QUIZ_FILE: Path = DATA_DIR / "quiz.json"
KNOWLEDGE_FILE: Path = DATA_DIR / "knowledge.json"
SUBJECTS_FILE: Path = DATA_DIR / "subjects.json"

# User-specific files
PROFILE_FILE: Path = USER_DATA_DIR / "profile.json"
PROGRESS_FILE: Path = USER_DATA_DIR / "progress.json"
SETTINGS_FILE: Path = USER_DATA_DIR / "settings.json"

# Log files
ERROR_LOG_FILE: Path = LOGS_DIR / "error.log"
DEBUG_LOG_FILE: Path = LOGS_DIR / "debug.log"
CHAT_LOG_FILE: Path = LOGS_DIR / "chat.log"

# Presentation assets
LOGO_FILE: Path = ASSETS_DIR / "logo.txt"

# ---------------------------------------------------------------------------
# Default user settings (merged with settings.json + .env overrides)
# ---------------------------------------------------------------------------
DEFAULT_SETTINGS: dict[str, object] = {
    "username": "Student",
    "theme": "blue",            # see ui.py THEMES for valid keys
    "preferred_study_hours": 4, # hours per day the user can usually study
    "preferred_subjects": [],   # list[str] e.g. ["Mathematics", "Physics"]
    "notifications": True,      # show reminders/status on startup
    "quiz_question_count": 5,   # default questions per quiz
    "search_results": 6,        # default number of web results to show
}

# ---------------------------------------------------------------------------
# Runtime / engine tuning
# ---------------------------------------------------------------------------
RUNTIME: dict[str, object] = {
    # Intent detection
    "fuzzy_threshold": 78.0,     # rapidfuzz score (%) below which a match is rejected
    "typo_correction_window": 3, # max edit distance for a single-word typo fix
    # Web search
    "search_timeout": 12,        # seconds before an HTTP request is abandoned
    "search_user_agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "search_max_results": 8,
    # Study planner
    "default_days": 7,           # schedule horizon in days
    "min_block_minutes": 25,     # shortest study block
    "max_block_minutes": 120,    # longest study block
}


# ---------------------------------------------------------------------------
# Environment variables (.env support)
# ---------------------------------------------------------------------------
def load_environment() -> None:
    """Load environment variables from a ``.env`` file at the project root.

    Keys are read in the following priority order (highest wins):
    settings.json  >  environment variable  >  default.
    """
    if load_dotenv is None:
        return
    env_file = BASE_DIR / ".env"
    load_dotenv(dotenv_path=env_file, override=False)


@dataclass
class Config:
    """Runtime configuration object.

    Reads environment variables once and exposes settings that may have been
    overridden through ``.env`` (e.g. ``STUDYBUDDY_SEARCH_MAX_RESULTS=10``).
    """

    settings: dict[str, object] = field(default_factory=lambda: dict(DEFAULT_SETTINGS))
    runtime: dict[str, object] = field(default_factory=lambda: dict(RUNTIME))

    # -- convenience accessors -------------------------------------------------
    @property
    def username(self) -> str:
        return str(self.settings.get("username", "Student"))

    @property
    def theme(self) -> str:
        return str(self.settings.get("theme", "blue"))

    @property
    def notifications(self) -> bool:
        return bool(self.settings.get("notifications", True))

    def get(self, key: str, default: object = None) -> object:
        """Return a setting value (falling back to the default)."""
        return self.settings.get(key, DEFAULT_SETTINGS.get(key, default))

    def get_runtime(self, key: str, default: object = None) -> object:
        """Return a runtime tuning value (falling back to the default)."""
        return self.runtime.get(key, RUNTIME.get(key, default))

    # -- .env overrides -------------------------------------------------------
    def apply_env_overrides(self) -> None:
        """Apply environment variables on top of the current settings.

        Environment variables are optional; every setting already has a
        sensible default so StudyBuddy runs fine without a ``.env`` file.
        """
        mappings = {
            "STUDYBUDDY_USERNAME": "username",
            "STUDYBUDDY_THEME": "theme",
            "STUDYBUDDY_HOURS": "preferred_study_hours",
            "STUDYBUDDY_QUIZ_QUESTIONS": "quiz_question_count",
            "STUDYBUDDY_SEARCH_RESULTS": "search_results",
        }
        for env_key, setting_key in mappings.items():
            value = os.getenv(env_key)
            if value is None:
                continue
            try:
                current = self.settings.get(setting_key)
                if isinstance(current, bool):
                    self.settings[setting_key] = value.strip().lower() in {"1", "true", "yes", "on"}
                elif isinstance(current, int):
                    self.settings[setting_key] = int(value)
                else:
                    self.settings[setting_key] = value
            except ValueError:
                # A malformed env value must never crash the app.
                continue


# ---------------------------------------------------------------------------
# Module-level helpers used across the application
# ---------------------------------------------------------------------------
def ensure_directory_structure() -> None:
    """Create every directory the application needs, if missing."""
    for directory in (ASSETS_DIR, DATA_DIR, USER_DATA_DIR, CHAT_HISTORY_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def build_config() -> Config:
    """Factory that assembles a fully initialised :class:`Config`.

    Order of operations:
      1. Load ``.env`` into the environment (if present).
      2. Start with default settings.
      3. Apply environment overrides.
      4. Make sure the directory layout exists.

    Returns
    -------
    Config
        A ready-to-use configuration object.
    """
    load_environment()
    cfg = Config()
    cfg.apply_env_overrides()
    ensure_directory_structure()
    return cfg

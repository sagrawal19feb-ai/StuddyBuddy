"""
storage.py — Persistent storage manager for StudyBuddy
==========================================================

Everything the chatbot needs to remember lives on disk as human-readable
JSON inside ``user_data/``:

  * ``profile.json``   — who the user is (name, class, favourite/weak subjects)
  * ``progress.json``  — study hours, quiz results, topics covered, goals
  * ``settings.json``  — preferences (theme, hours, subjects, notifications)
  * ``chat_history/``  — one JSON file per calendar day with every exchange

The :class:`StorageManager` is the only class allowed to touch these files.
It uses atomic writes (see :func:`utils.save_json`) so a crash can never
corrupt the user's data, and it merges with defaults so missing keys are
always filled in.
"""

from __future__ import annotations

import copy
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import utils
from config import (
    CHAT_HISTORY_DIR,
    PROFILE_FILE,
    PROGRESS_FILE,
    SETTINGS_FILE,
    DEFAULT_SETTINGS,
)

# ---------------------------------------------------------------------------
# Default shapes for persisted documents
# ---------------------------------------------------------------------------
DEFAULT_PROFILE: dict[str, Any] = {
    "name": "",
    "class": "",
    "favourite_subjects": [],
    "weak_subjects": [],
    "first_seen": "",
    "last_seen": "",
    "last_conversation": [],
    "recent_searches": [],
    "last_topic": "",          # the last subject/topic discussed (continuity)
    "custom_facts": {},        # facts the user taught the bot: {topic: {fact, created}}
}

DEFAULT_PROGRESS: dict[str, Any] = {
    "total_study_hours": 0.0,
    "sessions": 0,
    "quizzes_completed": 0,
    "quiz_history": [],
    "avg_quiz_score": 0.0,
    "topics_covered": [],
    "goals_completed": [],
    "flashcards_reviewed": 0,
    "cards_mastered": 0,
    "subject_accuracy": {},   # {subject: {"answered": int, "correct": int}}
    "daily_log": {},          # {date: {"hours": float, "quizzes": int}}
}


class StorageManager:
    """Reads and writes all persistent StudyBuddy data."""

    def __init__(self) -> None:
        self._profile: dict[str, Any] = {}
        self._progress: dict[str, Any] = {}
        self._settings: dict[str, Any] = {}
        self.load_all()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_all(self) -> None:
        """Load every persisted document from disk (with defaults)."""
        self._profile = self._merge_defaults(
            utils.load_json(PROFILE_FILE, {}), DEFAULT_PROFILE
        )
        self._progress = self._merge_defaults(
            utils.load_json(PROGRESS_FILE, {}), DEFAULT_PROGRESS
        )
        self._settings = self._merge_defaults(
            utils.load_json(SETTINGS_FILE, {}), DEFAULT_SETTINGS
        )

    def save_all(self) -> None:
        """Persist every in-memory document back to disk."""
        self.save_profile()
        self.save_progress()
        self.save_settings()

    def reset_all(self) -> None:
        """Erase every piece of user data and reload defaults.

        Deletes ``profile.json``, ``progress.json``, ``settings.json`` and
        all daily chat-history files, then resets the in-memory state to
        fresh defaults. Logs are kept (they are a transcript, not user data).
        """
        for path in (PROFILE_FILE, PROGRESS_FILE, SETTINGS_FILE):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        if CHAT_HISTORY_DIR.exists():
            for history_file in CHAT_HISTORY_DIR.glob("*.json"):
                try:
                    history_file.unlink(missing_ok=True)
                except OSError:
                    pass
        self.load_all()

    # -- profile -----------------------------------------------------------
    def save_profile(self) -> bool:
        self._profile["last_seen"] = utils.timestamp_str()
        return utils.save_json(PROFILE_FILE, self._profile)

    def get_profile(self) -> dict[str, Any]:
        return self._profile

    def update_profile(self, **fields: Any) -> None:
        """Update profile fields in place and persist the change."""
        self._profile.update({k: v for k, v in fields.items() if v is not None})
        self.save_profile()

    def register_first_visit(self) -> None:
        """Record the first time the user opened the app."""
        if not self._profile.get("first_seen"):
            self._profile["first_seen"] = utils.timestamp_str()

    def is_returning_user(self) -> bool:
        """True once the profile has a stored name."""
        return bool(self._profile.get("name"))

    # -- progress ----------------------------------------------------------
    def save_progress(self) -> bool:
        return utils.save_json(PROGRESS_FILE, self._progress)

    def get_progress(self) -> dict[str, Any]:
        return self._progress

    def update_progress(self, **fields: Any) -> None:
        self._progress.update({k: v for k, v in fields.items() if v is not None})
        self.save_progress()

    def add_study_hours(self, hours: float) -> None:
        """Accumulate study time and update today's daily log."""
        today = utils.date_str()
        self._progress["total_study_hours"] = round(
            float(self._progress.get("total_study_hours", 0.0)) + hours, 2
        )
        self._progress["sessions"] = int(self._progress.get("sessions", 0)) + 1
        daily = self._progress.setdefault("daily_log", {})
        entry = daily.setdefault(today, {"hours": 0.0, "quizzes": 0})
        entry["hours"] = round(float(entry.get("hours", 0.0)) + hours, 2)
        self.save_progress()

    def add_covered_topic(self, topic: str) -> None:
        topics = self._progress.get("topics_covered", [])
        if topic and topic not in topics:
            topics.append(topic)
            self._progress["topics_covered"] = topics
            self.save_progress()

    def record_quiz(self, subject: str, score: float, total: int,
                    weak_topics: list[str]) -> None:
        """Store one quiz result and refresh aggregate statistics."""
        progress = self._progress
        progress["quizzes_completed"] = int(progress.get("quizzes_completed", 0)) + 1
        history = progress.setdefault("quiz_history", [])
        history.append({
            "date": utils.timestamp_str(),
            "subject": subject,
            "score": score,
            "total": total,
            "percent": round(score / total * 100, 1) if total else 0.0,
            "weak_topics": weak_topics,
        })
        # Keep the last 100 results so the history file stays tidy.
        progress["quiz_history"] = history[-100:]

        scores = [entry["percent"] for entry in history]
        progress["avg_quiz_score"] = round(sum(scores) / len(scores), 1) if scores else 0.0

        today = utils.date_str()
        daily = progress.setdefault("daily_log", {})
        entry = daily.setdefault(today, {"hours": 0.0, "quizzes": 0})
        entry["quizzes"] = int(entry.get("quizzes", 0)) + 1

        for topic in weak_topics:
            self.add_covered_topic(f"Needs revision: {topic}")
        self.save_progress()

    def record_subject_answer(self, subject: str, correct: bool) -> None:
        """Track per-subject accuracy for weak-topic analysis."""
        accuracy = self._progress.setdefault("subject_accuracy", {})
        entry = accuracy.setdefault(subject, {"answered": 0, "correct": 0})
        entry["answered"] = int(entry.get("answered", 0)) + 1
        entry["correct"] = int(entry.get("correct", 0)) + (1 if correct else 0)
        self.save_progress()

    def complete_goal(self, goal: str) -> None:
        goals = self._progress.get("goals_completed", [])
        if goal and goal not in goals:
            goals.append(goal)
            self._progress["goals_completed"] = goals
            self.save_progress()

    def record_flashcard_review(self, reviewed: int, mastered: int) -> None:
        progress = self._progress
        progress["flashcards_reviewed"] = int(progress.get("flashcards_reviewed", 0)) + reviewed
        progress["cards_mastered"] = int(progress.get("cards_mastered", 0)) + mastered
        self.save_progress()

    def streak_days(self) -> int:
        """Count consecutive days with study activity (today or yesterday start).

        Uses the ``daily_log`` map (date -> activity). A streak of 0 means
        no recent activity; it is not reset by a single missed day, only by
        two or more consecutive inactive days.
        """
        daily = self._progress.get("daily_log", {})
        if not daily:
            return 0
        active_days = set(daily.keys())
        today = date.today()
        cursor = today if today.isoformat() in active_days else today - timedelta(days=1)
        streak = 0
        while cursor.isoformat() in active_days:
            streak += 1
            cursor -= timedelta(days=1)
        return streak

    # -- settings ----------------------------------------------------------
    def save_settings(self) -> bool:
        return utils.save_json(SETTINGS_FILE, self._settings)

    def get_settings(self) -> dict[str, Any]:
        return self._settings

    def update_settings(self, **fields: Any) -> None:
        self._settings.update({k: v for k, v in fields.items() if v is not None})
        self.save_settings()

    # -- chat history ------------------------------------------------------
    def today_chat_file(self) -> Path:
        """Path of today's chat-history JSON file."""
        return CHAT_HISTORY_DIR / f"chat-{utils.date_str()}.json"

    def load_chat_history(self, date: Optional[str] = None) -> list[dict[str, Any]]:
        """Load the conversation log for ``date`` (default: today)."""
        target = CHAT_HISTORY_DIR / f"chat-{date or utils.date_str()}.json"
        return utils.load_json(target, [])

    def append_chat(self, role: str, message: str) -> None:
        """Append one turn to today's chat-history file."""
        file_path = self.today_chat_file()
        history = utils.load_json(file_path, [])
        history.append({
            "timestamp": utils.timestamp_str(),
            "role": role,
            "message": message,
        })
        utils.save_json(file_path, history[-2000:])

    def list_chat_history_files(self) -> list[str]:
        """Return the file names of all stored daily chat logs."""
        if not CHAT_HISTORY_DIR.exists():
            return []
        return sorted(
            p.name for p in CHAT_HISTORY_DIR.iterdir()
            if p.suffix == ".json"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_defaults(data: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
        """Return ``defaults`` updated with any top-level keys in ``data``.

        ``defaults`` is DEEP-COPIED: nested containers (e.g. ``custom_facts``,
        ``recent_searches``) must never be shared with the module-level
        default objects. A shallow copy would let a mutation like
        ``profile["custom_facts"][key] = ...`` permanently pollute the
        defaults and "leak" into every later load — including after a reset.
        """
        merged = copy.deepcopy(defaults)
        merged.update({k: v for k, v in data.items() if v is not None})
        return merged

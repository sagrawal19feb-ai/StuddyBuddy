"""
planner.py — Daily routine planner for StudyBuddy
====================================================

Builds a complete, balanced day plan: morning routine, study blocks,
breaks, revision, exercise and sleep. Unlike the weekly :mod:`scheduler`,
this module optimises a *single* day around the student's wake-up time and
available study hours.

Design
------
The day is built as a sequence of *anchored* events. Study blocks start
after the morning routine and are interleaved with real breaks; lunch and
dinner are anchored to realistic clock times (12:30 and 19:30) and any gap
between the morning session and lunch is filled with a sensible
"independent practice" slot, so a 7 AM riser never gets "lunch" at 9:50 AM.

The plan is returned as structured data so the UI can render it as a Rich
table; a plain-text version is also provided for copy/paste.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from logger import get_logger

log = get_logger("planner")

# Clock anchors (minutes from midnight).
LUNCH_ANCHOR = 12 * 60 + 30     # 12:30
LUNCH_MINUTES = 45
DINNER_ANCHOR = 19 * 60 + 30    # 19:30
DINNER_MINUTES = 40


@dataclass
class DailyBlock:
    """One entry in the daily routine."""

    time: str
    label: str
    kind: str = "activity"  # activity | study | break | meal | exercise | sleep

    def as_row(self) -> list[str]:
        return [self.time, self.label]


@dataclass
class DailyPlan:
    """A complete single-day routine."""

    wake_time: str
    sleep_time: str
    blocks: list[DailyBlock] = field(default_factory=list)

    def render_plain(self) -> str:
        """Render the routine as a simple two-column table."""
        try:
            from tabulate import tabulate
            return tabulate([b.as_row() for b in self.blocks],
                            headers=["Time", "Activity"])
        except ImportError:  # pragma: no cover
            return "\n".join(f"{b.time}  {b.label}" for b in self.blocks)


class DailyPlanner:
    """Generates balanced daily routines."""

    def __init__(self) -> None:
        self.defaults = {
            "wake": "06:30",
            "sleep": "22:30",
            "study_hours": 4,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def create_plan(
        self,
        study_hours: Optional[float] = None,
        wake_time: Optional[str] = None,
        exercise: bool = True,
        sleep_time: Optional[str] = None,
    ) -> DailyPlan:
        """Build a daily routine.

        Parameters
        ----------
        study_hours : float | None
            Hours dedicated to study today (default from settings).
        wake_time : str | None
            "HH:MM" wake-up time (default 06:30).
        exercise : bool
            Whether to include an exercise slot.
        sleep_time : str | None
            "HH:MM" bedtime (default 22:30).
        """
        study_hours = float(study_hours or self.defaults["study_hours"])
        study_hours = max(1.0, min(12.0, study_hours))
        wake = wake_time or self.defaults["wake"]
        sleep = sleep_time or self.defaults["sleep"]

        blocks: list[DailyBlock] = []
        cursor = self._to_minutes(wake)

        # --- morning routine -------------------------------------------------
        cursor = self._add(blocks, cursor, 20, "Wake up & freshen up")
        cursor = self._add(
            blocks, cursor, 10,
            "Light exercise / stretching" if exercise else "Stretch & breathe",
            kind="exercise" if exercise else "activity",
        )
        cursor = self._add(blocks, cursor, 25, "Healthy breakfast", kind="meal")
        cursor = self._add(blocks, cursor, 10, "Plan the day & set 3 goals")

        # --- study blocks (50 min each + 5 min breaks) -------------------------
        # Number of 50-minute blocks is derived from the requested study hours
        # so the plan never silently doubles or halves the user's target
        # (2 minimum, 6 maximum — a realistic cap for one day).
        n_blocks = max(2, min(6, round(study_hours * 60 / 50)))

        cursor = self._add(blocks, cursor, 50, "📚 Morning study block #1", kind="study")
        cursor = self._add(blocks, cursor, 5, "Short break (5 min)", kind="break")
        cursor = self._add(blocks, cursor, 50, "📚 Morning study block #2", kind="study")

        # --- gap before the 12:30 lunch anchor -----------------------------------
        if cursor < LUNCH_ANCHOR:
            gap = LUNCH_ANCHOR - cursor
            if gap <= 20:
                cursor = self._add(blocks, cursor, gap, "Relax before lunch", kind="break")
            elif gap <= 90:
                cursor = self._add(blocks, cursor, gap, "Independent practice (optional)",
                                   kind="study")
            else:
                cursor = self._add(blocks, cursor, 15, "Snack & stretch", kind="break")
                cursor = self._add(blocks, cursor, LUNCH_ANCHOR - cursor,
                                   "🎯 Free time & rest (flexible)", kind="break")

        # --- lunch anchored at 12:30 ----------------------------------------------
        cursor = self._add(blocks, cursor, LUNCH_MINUTES, "🍱 Lunch break", kind="meal")

        # --- remaining afternoon/evening study blocks -------------------------------
        for i in range(3, n_blocks + 1):
            cursor = self._add(blocks, cursor, 50, f"📚 Study block #{i}", kind="study")
            if i < n_blocks:
                cursor = self._add(blocks, cursor, 10, "Break — walk around, hydrate", kind="break")

        if exercise:
            cursor = self._add(blocks, cursor, 30, "🏃 Exercise / sports", kind="exercise")

        # --- dinner anchored at 19:30 -------------------------------------------------
        if cursor < DINNER_ANCHOR - 10:
            cursor = self._add(blocks, cursor, 15, "Recharge break", kind="break")
        if cursor < DINNER_ANCHOR:
            cursor = self._add(blocks, cursor, DINNER_ANCHOR - cursor,
                               "🎯 Free time & rest (flexible)", kind="break")
        cursor = self._add(blocks, cursor, DINNER_MINUTES, "🍽 Dinner & family time", kind="meal")

        # --- evening wind-down -----------------------------------------------------
        cursor = self._add(blocks, cursor, 25, "✅ Revision: recap today's learning", kind="study")
        cursor = self._add(blocks, cursor, 20, "🧘 Relax, read, plan tomorrow")

        sleep_minutes = self._to_minutes(sleep)
        if cursor < sleep_minutes:
            cursor = self._add(blocks, cursor, sleep_minutes - cursor,
                               "Wind-down / leisure", kind="break")
        cursor = self._add(blocks, cursor, 0, "🛌 Sleep time", kind="sleep")

        log.info("Daily plan generated: wake=%s sleep=%s study=%.1fh", wake, sleep, study_hours)
        return DailyPlan(wake_time=wake, sleep_time=sleep, blocks=blocks)

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _add(blocks: list[DailyBlock], cursor: int, minutes: int,
             label: str, kind: str = "activity") -> int:
        """Append a block and return the new cursor position."""
        if minutes < 0:
            minutes = 0
        blocks.append(DailyBlock(
            time=DailyPlanner._fmt(cursor),
            label=label,
            kind=kind,
        ))
        return cursor + minutes

    @staticmethod
    def _to_minutes(hhmm: str) -> int:
        try:
            hours, minutes = hhmm.strip().split(":")
            return int(hours) * 60 + int(minutes)
        except (ValueError, AttributeError):
            return 6 * 60 + 30

    @staticmethod
    def _fmt(minutes: int) -> str:
        minutes = minutes % (24 * 60)
        return f"{minutes // 60:02d}:{minutes % 60:02d}"

    @staticmethod
    def validate_time(hhmm: str) -> bool:
        """Validate a 'HH:MM' string."""
        try:
            hours, minutes = hhmm.strip().split(":")
            return 0 <= int(hours) <= 23 and 0 <= int(minutes) <= 59
        except (ValueError, AttributeError):
            return False

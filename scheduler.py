"""
scheduler.py — Study schedule generator for StudyBuddy
=========================================================

Turns simple answers — *how many hours you can study, which subjects, how
hard they are, and when the exam is* — into a personalised, day-by-day
timetable.

The algorithm:

  1. Every subject gets a **weight** = base difficulty + exam proximity bonus
     (+ a small boost for subjects the user marked as weak/favourite).
  2. The weekly hours are split across subjects proportionally to weight.
  3. Each day's study time is sliced into focused blocks (25-50 minutes)
     separated by short breaks, with a long lunch break, a revision slot and
     an exercise slot woven in.

The result is returned both as structured data (for Rich tables in the UI)
and as plain text (via ``tabulate``) so it can be copied or saved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import utils
from logger import get_logger

log = get_logger("scheduler")

DIFFICULTY_WEIGHT = {"easy": 0.8, "medium": 1.0, "hard": 1.3}
DEFAULT_DIFFICULTY = "medium"


@dataclass
class StudyBlock:
    """One contiguous activity inside a scheduled day."""

    start: str           # "08:00"
    end: str             # "08:30"
    duration_min: int
    label: str           # "Mathematics — Algebra"
    kind: str = "study"  # study | break | meal | revision | exercise | review

    def as_row(self) -> list[str]:
        return [self.start, self.end, f"{self.duration_min} min", self.label]


@dataclass
class DayPlan:
    """The schedule for a single day."""

    day_index: int
    date_label: str
    blocks: list[StudyBlock] = field(default_factory=list)

    def study_minutes(self) -> int:
        return sum(b.duration_min for b in self.blocks if b.kind in {"study", "revision"})

    def as_table(self) -> list[list[str]]:
        return [block.as_row() for block in self.blocks]


@dataclass
class StudySchedule:
    """A complete weekly study timetable."""

    days: list[DayPlan]
    subject_distribution: dict[str, float]  # subject -> hours/week
    total_hours_per_day: float
    notes: list[str] = field(default_factory=list)

    def total_hours(self) -> float:
        return round(sum(day.study_minutes() for day in self.days) / 60.0, 1)

    def render_plain(self) -> str:
        """Render the whole schedule as plain text (tabulate-style)."""
        try:
            from tabulate import tabulate
        except ImportError:  # pragma: no cover
            return self._render_fallback()
        sections: list[str] = []
        for day in self.days:
            header = f"{day.date_label}  (study {day.study_minutes()} min)"
            sections.append(header)
            sections.append(tabulate(day.as_table(), headers=["Start", "End", "Duration", "Activity"]))
            sections.append("")
        return "\n".join(sections)

    def _render_fallback(self) -> str:
        """Very small fallback if tabulate is unavailable."""
        lines = []
        for day in self.days:
            lines.append(f"{day.date_label}  (study {day.study_minutes()} min)")
            for block in day.blocks:
                lines.append(f"  {block.start}-{block.end}  {block.label}")
            lines.append("")
        return "\n".join(lines)


class StudyPlanner:
    """Generates personalised study schedules."""

    def __init__(self) -> None:
        self.known_subjects: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def create_schedule(
        self,
        hours_per_day: float,
        subjects: list[str],
        difficulties: Optional[dict[str, str]] = None,
        weak_subjects: Optional[list[str]] = None,
        exam_date: Optional[str] = None,
        days: int = 7,
        start_hour: int = 8,
    ) -> StudySchedule:
        """Build a personalised schedule.

        Parameters
        ----------
        hours_per_day : float
            How many hours the student can study each day.
        subjects : list[str]
            The subjects to include.
        difficulties : dict[str, str] | None
            Optional mapping subject -> "easy" | "medium" | "hard".
        weak_subjects : list[str] | None
            Subjects the student finds hard (get extra time).
        exam_date : str | None
            ISO date of the nearest exam, used to boost urgency.
        days : int
            Number of days to plan (default 7).
        start_hour : int
            Hour of day study blocks start (24h, default 8).
        """
        difficulties = difficulties or {}
        weak_subjects = [s.lower() for s in (weak_subjects or [])]
        subjects = utils.dedupe(subjects)
        if not subjects:
            subjects = ["General Studies"]

        # Guard against absurd inputs: hours/day clamped to a realistic range
        # and days to 1-30, so the schedule can never explode into nonsense.
        try:
            hours_per_day = utils.clamp(float(hours_per_day or 0), 0.5, 16.0)
        except (TypeError, ValueError):
            hours_per_day = 4.0
        try:
            days = int(utils.clamp(int(days or 7), 1, 30))
        except (TypeError, ValueError):
            days = 7

        weights = self._subject_weights(subjects, difficulties, weak_subjects, exam_date)
        distribution = self._distribute(subjects, weights, hours_per_day, days)

        plans: list[DayPlan] = []
        today = date.today()
        for day_offset in range(days):
            day_date = today + timedelta(days=day_offset)
            label = (
                f"{day_date.strftime('%A')} {day_date.strftime('%d %b')}"
                + ("  (exam week ⚡)" if self._days_until(exam_date) == day_offset else "")
            )
            plan = self._build_day(
                day_index=day_offset,
                date_label=label,
                hours=hours_per_day,
                subjects=subjects,
                distribution=distribution,
                start_hour=start_hour,
            )
            plans.append(plan)

        notes = self._build_notes(subjects, hours_per_day, exam_date)
        return StudySchedule(
            days=plans,
            subject_distribution=distribution,
            total_hours_per_day=hours_per_day,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Weighting & distribution
    # ------------------------------------------------------------------
    def _subject_weights(
        self,
        subjects: list[str],
        difficulties: dict[str, str],
        weak: list[str],
        exam_date: Optional[str],
    ) -> list[float]:
        """Compute a weight per subject (difficulty + urgency + weakness)."""
        days_until_exam = self._days_until(exam_date)
        weights: list[float] = []
        for subject in subjects:
            diff = difficulties.get(subject, DEFAULT_DIFFICULTY)
            weight = DIFFICULTY_WEIGHT.get(diff.lower(), 1.0)
            if subject.lower() in weak:
                weight *= 1.25  # weak subjects deserve extra practice
            if days_until_exam is not None and 0 <= days_until_exam <= 14:
                # Subjects with an imminent exam get a big urgency boost.
                urgency = 1.0 + (14 - days_until_exam) * 0.03
                weight *= urgency
            weights.append(round(weight, 3))
        return weights

    @staticmethod
    def _days_until(exam_date: Optional[str]) -> Optional[int]:
        """Number of days until the exam (None if not given/invalid)."""
        if not exam_date:
            return None
        try:
            target = datetime.strptime(exam_date.strip(), "%Y-%m-%d").date()
        except ValueError:
            return None
        return (target - date.today()).days

    @staticmethod
    def _distribute(subjects: list[str], weights: list[float],
                    hours_per_day: float, days: int) -> dict[str, float]:
        """Split the weekly hours among subjects by weight."""
        total_weight = sum(weights)
        weekly_hours = hours_per_day * days
        distribution: dict[str, float] = {}
        for subject, weight in zip(subjects, weights):
            share = weekly_hours * weight / total_weight
            distribution[subject] = round(share, 2)
        return distribution

    # ------------------------------------------------------------------
    # Day construction
    # ------------------------------------------------------------------
    def _build_day(
        self,
        day_index: int,
        date_label: str,
        hours: float,
        subjects: list[str],
        distribution: dict[str, float],
        start_hour: int,
    ) -> DayPlan:
        """Assemble one day: study blocks, breaks, revision, exercise."""
        plan = DayPlan(day_index=day_index, date_label=date_label)
        cursor = start_hour * 60  # minutes from midnight

        # Morning warm-up (light review) — always helps.
        plan.blocks.append(StudyBlock(
            *self._slot(cursor, 15), label="Warm-up: review yesterday's notes", kind="review"
        ))
        cursor = self._advance(cursor, 15)

        # Distribute today's study minutes across the subjects.
        today_minutes = max(25, int(hours * 60))
        subject_share: dict[str, float] = {}
        total_share = sum(distribution.values())
        for subject in subjects:
            subject_share[subject] = distribution.get(subject, 0) * today_minutes / max(total_share, 1)

        blocks = self._study_blocks(subject_share, cursor)
        for block in blocks:
            plan.blocks.append(block)
            cursor = self._advance(cursor, block.duration_min)

        # Short break after the main study blocks.
        plan.blocks.append(StudyBlock(*self._slot(cursor, 10), label="Refresh break", kind="break"))
        cursor = self._advance(cursor, 10)

        # Lunch / long break (only for full-day plans).
        if hours >= 3:
            plan.blocks.append(StudyBlock(*self._slot(cursor, 45), label="Lunch break", kind="meal"))
            cursor = self._advance(cursor, 45)

        # Revision slot — reinforce what was studied today.
        plan.blocks.append(StudyBlock(
            *self._slot(cursor, 20), label="Revision: key points of the day", kind="revision"
        ))
        cursor = self._advance(cursor, 20)

        # Exercise — study breaks shouldn't be sedentary.
        plan.blocks.append(StudyBlock(*self._slot(cursor, 20), label="Exercise / walk", kind="exercise"))
        return plan

    def _study_blocks(self, subject_share: dict[str, float], cursor: int) -> list[StudyBlock]:
        """Chop each subject's daily minutes into 25-50 min pomodoro blocks."""
        blocks: list[StudyBlock] = []
        for subject, minutes in subject_share.items():
            remaining = int(minutes)
            while remaining >= 25:
                block_len = min(50, remaining)
                blocks.append(StudyBlock(
                    *self._slot(cursor, block_len), label=f"Study: {subject}", kind="study"
                ))
                cursor = self._advance(cursor, block_len)
                remaining -= block_len
                if remaining >= 25:
                    # 5-minute break between pomodoros.
                    blocks.append(StudyBlock(*self._slot(cursor, 5), label="Micro-break", kind="break"))
                    cursor = self._advance(cursor, 5)
        return blocks

    # -- time helpers ------------------------------------------------------
    @staticmethod
    def _slot(start_minutes: int, duration: int) -> tuple[str, str, int]:
        end_minutes = start_minutes + duration
        fmt = lambda m: f"{m // 60:02d}:{m % 60:02d}"
        return fmt(start_minutes), fmt(end_minutes), duration

    @staticmethod
    def _advance(cursor: int, duration: int) -> int:
        return cursor + duration

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------
    @staticmethod
    def _build_notes(subjects: list[str], hours_per_day: float, exam_date: Optional[str]) -> list[str]:
        notes = [
            f"Total planned study time: {hours_per_day * 7:.1f} hours over 7 days.",
            "Pomodoro technique is built in: 25-50 min focus + 5 min break.",
            "Adjust the plan if a day feels heavy — consistency beats intensity.",
        ]
        if exam_date:
            try:
                target = datetime.strptime(exam_date, "%Y-%m-%d").date()
                days_left = (target - date.today()).days
                if days_left >= 0:
                    notes.append(f"Exam in {days_left} days — revision gets priority as the date nears.")
            except ValueError:
                notes.append("Exam date not recognised — treated as no exam.")
        return notes

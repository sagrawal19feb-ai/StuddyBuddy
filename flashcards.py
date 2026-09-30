"""
flashcards.py — Flashcard engine for StudyBuddy
==================================================

Flashcards are built from the same JSON question bank as quizzes, plus the
general-knowledge facts — every card is a question with a revealed answer.
Features:

  * randomised card order,
  * self-graded review (the student marks a card as known or not),
  * missed cards return up to a fixed cap, then are flagged for practice,
  * a short session report at the end,
  * progress tracking (cards reviewed / mastered).

Like :mod:`quiz_engine`, the loop is a send-capable generator so the UI
layer controls pacing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator, Optional

from knowledge import KnowledgeBase
from logger import get_logger
from storage import StorageManager

log = get_logger("flashcards")


@dataclass
class Flashcard:
    """A single card: question -> answer (plus optional topic tag)."""

    question: str
    answer: str
    topic: str = "General"
    source: str = "quiz"      # "quiz" or "knowledge"
    misses: int = 0           # how many times the student graded it missed

    def reveal(self) -> str:
        """Format the card with its answer visible."""
        return f"**{self.question}**\n\nAnswer: **{self.answer}**"


# A card is repeated after a miss, but only this many times per session —
# beyond that it is set aside and flagged as "needs practice" so the session
# can never loop forever on a single card.
MAX_REPEATS_PER_CARD = 3


@dataclass
class FlashcardReport:
    """End-of-session summary."""

    reviewed: int
    mastered: int
    repeated_rounds: int
    needs_practice: list[str] = field(default_factory=list)

    def render(self) -> str:
        if self.reviewed == 0:
            return "No cards were reviewed this session. Let's try again soon! 📚"
        mastered_pct = round(self.mastered / self.reviewed * 100) if self.reviewed else 0
        lines = [
            f"📇 Flashcard session complete! You reviewed **{self.reviewed}** cards "
            f"and mastered **{self.mastered}** ({mastered_pct}%).",
        ]
        if self.needs_practice:
            lines.append("")
            lines.append("📌 Cards that still need practice:")
            for question in self.needs_practice[:6]:
                lines.append(f"  • {question}")
            if len(self.needs_practice) > 6:
                lines.append(f"  … and {len(self.needs_practice) - 6} more")
            lines.append("")
            lines.append("Review them in your next session — short daily reviews beat "
                         "long cramming sessions!")
        else:
            lines.append("")
            lines.append("No cards left in the practice pile — nicely done! "
                         "Keep the momentum going.")
        return "\n".join(lines)


class FlashcardEngine:
    """Builds and runs flashcard sessions."""

    def __init__(self, knowledge: KnowledgeBase, storage: StorageManager) -> None:
        self.knowledge = knowledge
        self.storage = storage
        self._rng = random

    # ------------------------------------------------------------------
    # Card construction
    # ------------------------------------------------------------------
    def build_deck(self, subject: Optional[str] = None, limit: int = 10) -> list[Flashcard]:
        """Assemble a shuffled deck from quiz questions + knowledge facts."""
        cards: list[Flashcard] = []

        # 1) From the quiz bank (question + correct option + explanation)
        bank = self.knowledge.questions_for(subject or None, None)
        if not bank and subject:
            # The user gave a topic name ("Algebra") rather than a subject.
            bank = [
                q for q in self.knowledge.questions_for(None, None)
                if q.get("topic", "").lower() == subject.lower()
            ]
        for item in bank:
            options = item.get("options", [])
            answer = item.get("answer", "")
            if answer in options:
                cards.append(Flashcard(
                    question=item.get("question", "?"),
                    answer=answer,
                    topic=item.get("topic", "General"),
                    source="quiz",
                ))

        # 2) From the knowledge base
        for fact in self.knowledge.knowledge_facts():
            cards.append(Flashcard(
                question=fact.get("question", fact.get("fact", "?")),
                answer=fact.get("answer", fact.get("fact", "")),
                topic=fact.get("category", "General"),
                source="knowledge",
            ))

        if not cards:
            cards.append(Flashcard(
                question="What does 'Pomodoro' mean in study technique?",
                answer="A 25-minute focused study block followed by a short break.",
                topic="Study Skills",
                source="knowledge",
            ))

        self._rng.shuffle(cards)
        return cards[:max(1, limit)]

    # ------------------------------------------------------------------
    # Session loop
    # ------------------------------------------------------------------
    def run(self, subject: Optional[str] = None,
            limit: int = 10) -> Iterator[Flashcard | FlashcardReport]:
        """Run a flashcard session as a send-capable generator.

        The UI yields each :class:`Flashcard`, asks the student to grade it
        ("k" for known, "m" for missed), and sends the grade back.

        **Repeat policy (anti-infinite-loop):** a missed card returns for
        another attempt, but each card is asked at most ``MAX_REPEATS_PER_CARD``
        times (its first appearance + misses). After that it is set aside and
        listed in the report as "needs practice", so a session always ends.
        The student may also end any time by sending "quit".
        """
        deck = self.build_deck(subject, limit)
        yield from self.run_cards(deck)

    def run_cards(self, cards: list[Flashcard]) -> Iterator[Flashcard | FlashcardReport]:
        """Run a session over an explicit set of cards.

        Used by the chatbot to turn *missed quiz questions* into a targeted
        flashcard review — the deck is whatever the caller provides instead
        of being rebuilt from the knowledge base.
        """
        if not cards:
            yield FlashcardReport(0, 0, 0)
            return

        mastered = 0
        reviewed = 0
        rounds = 0
        needs_practice: list[str] = []
        current = list(cards)

        while current:
            rounds += 1
            missed: list[Flashcard] = []
            for card in current:
                grade = yield card            # UI sends "k" / "m" / "quit"
                reviewed += 1
                if grade is None or str(grade).strip().lower() in {"quit", "q", "exit"}:
                    yield FlashcardReport(reviewed, mastered, rounds, needs_practice)
                    return
                if str(grade).strip().lower() in {"k", "known", "yes", "y", "correct"}:
                    mastered += 1
                else:
                    card.misses += 1
                    if card.misses >= MAX_REPEATS_PER_CARD:
                        # Give up on this card for today — flag it, don't loop.
                        needs_practice.append(card.question)
                    else:
                        missed.append(card)

            if missed:
                self._rng.shuffle(missed)
                current = missed  # repeat only the missed cards (up to the cap)
            else:
                current = []

        report = FlashcardReport(reviewed, mastered, rounds, needs_practice)
        self.storage.record_flashcard_review(reviewed, mastered)
        log.info("Flashcard session: reviewed=%d mastered=%d rounds=%d needs_practice=%d",
                 reviewed, mastered, rounds, len(needs_practice))
        yield report

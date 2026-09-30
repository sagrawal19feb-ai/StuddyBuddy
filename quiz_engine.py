"""
quiz_engine.py — Quiz engine for StudyBuddy
==============================================

Conducts multiple-choice quizzes from the JSON question bank
(``data/quiz.json``) with:

  * subject and difficulty selection,
  * randomised question order and shuffled options,
  * live scoring,
  * weak-topic analysis (topics where accuracy drops below a threshold),
  * a post-quiz performance report with a progress bar,
  * per-subject accuracy tracking for long-term weak-subject detection.

The engine is a generator-style wizard: :meth:`run` yields *steps* — each
step is either a ``Question`` to present or the final ``QuizReport`` — so
the UI layer can interleave rendering and user input naturally.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Iterator, Optional

import utils
from knowledge import KnowledgeBase
from logger import get_logger
from storage import StorageManager

log = get_logger("quiz_engine")

WEAK_THRESHOLD_PERCENT = 60.0  # below this, a topic counts as "weak"


@dataclass
class Question:
    """A single MCQ ready to display."""

    text: str
    options: list[str]          # e.g. ["Photosynthesis", "Respiration", ...]
    correct_index: int          # 0-based index into options
    subject: str
    topic: str
    difficulty: str
    explanation: str = ""

    def prompt(self) -> str:
        """Render the question with lettered options."""
        letters = string.ascii_lowercase[: len(self.options)]
        lines = [f"**{self.text}**", ""]
        lines.extend(f"  {letter}) {option}" for letter, option in zip(letters, self.options))
        lines.append("")
        lines.append("(Type the letter of your answer, or 'quit' to stop the quiz.)")
        return "\n".join(lines)


@dataclass
class QuizReport:
    """Final statistics after a quiz."""

    subject: str
    correct: int
    total: int
    percent: float
    weak_topics: list[str] = field(default_factory=list)
    message: str = ""
    # The questions the user got wrong: [{"question": ..., "answer": ...}]
    missed: list[dict[str, str]] = field(default_factory=list)

    def render(self) -> str:
        """Human-readable summary of the quiz performance."""
        grade = "🌟 Excellent!" if self.percent >= 80 else (
            "👍 Good effort!" if self.percent >= 60 else "💪 Keep practicing!"
        )
        lines = [
            f"{grade} You scored **{self.correct}/{self.total}** "
            f"({self.percent:.1f}%) in {self.subject}.",
            "",
        ]
        if self.weak_topics:
            lines.append("📉 Topics needing revision:")
            lines.extend(f"  • {topic}" for topic in self.weak_topics)
            lines.append(
                "Try our flashcards, study tips, or a web search on these topics. "
                "Consistent small practice is the key!"
            )
        else:
            lines.append("🎉 No weak topics detected — you've mastered this set. "
                         "Want to try the next difficulty?")
        return "\n".join(lines)


class QuizEngine:
    """Runs interactive MCQ quizzes."""

    def __init__(self, knowledge: KnowledgeBase, storage: StorageManager) -> None:
        self.knowledge = knowledge
        self.storage = storage
        self._rng = random

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def available_subjects(self) -> list[str]:
        return self.knowledge.quiz_subjects()

    def available_difficulties(self) -> list[str]:
        return ["easy", "medium", "hard"]

    def select_subject(self, subject: Optional[str]) -> str:
        """Resolve the quiz subject (defaults to 'all subjects')."""
        if not subject or subject.strip().lower() in {"all", "any", "random", ""}:
            return "All Subjects"
        return subject.strip()

    def select_difficulty(self, difficulty: Optional[str]) -> str:
        if not difficulty or difficulty.strip().lower() in {"all", "any", "random", ""}:
            return "all"
        return difficulty.strip().lower()

    # ------------------------------------------------------------------
    # Question building
    # ------------------------------------------------------------------
    def build_questions(self, subject: str, difficulty: str,
                        count: Optional[int] = None) -> list[Question]:
        """Assemble a randomised question set.

        ``subject`` may be a bank subject ("Mathematics"), a generic label
        ("All Subjects"), or a topic name ("Algebra") — topics are matched
        against the question's ``topic`` field.

        When the hand-written bank runs short of ``count`` questions, the
        engine **auto-generates extra MCQs from the knowledge base** (see
        :meth:`_generate_fact_questions`), so quizzes can never run out of
        questions — StudyBuddy can quiz on any topic.
        """
        bank = self.knowledge.questions_for(
            None if subject == "All Subjects" else subject,
            None if difficulty == "all" else difficulty,
        )
        # Topic-level fallback: the user asked for a topic, not a subject.
        if not bank and subject not in {"All Subjects", "all"}:
            bank = [
                q for q in self.knowledge.questions_for(None, None)
                if q.get("topic", "").lower() == subject.lower()
            ]
            if difficulty != "all":
                bank = [q for q in bank if q.get("difficulty", "").lower() == difficulty]
        if not bank:
            bank = self.knowledge.questions_for(None, None)

        limit = int(count or self.storage.get_settings().get("quiz_question_count", 5))
        limit = max(1, min(limit, 20))  # hard cap keeps sessions reasonable

        chosen = self._rng.sample(bank, k=min(limit, len(bank))) if bank else []
        questions: list[Question] = []
        seen_texts: set[str] = set()
        for item in chosen:
            options = list(item.get("options", []))
            correct = item.get("answer")
            if correct not in options:
                continue
            self._rng.shuffle(options)
            question = Question(
                text=item.get("question", "?"),
                options=options,
                correct_index=options.index(correct),
                subject=item.get("subject", "General"),
                topic=item.get("topic", "General"),
                difficulty=item.get("difficulty", "medium"),
                explanation=item.get("explanation", ""),
            )
            if question.text not in seen_texts:
                seen_texts.add(question.text)
                questions.append(question)

        # Top up with knowledge-fact questions when the bank is short.
        if len(questions) < limit:
            needed = limit - len(questions)
            for generated in self._generate_fact_questions(subject, difficulty, needed):
                if generated.text not in seen_texts:
                    seen_texts.add(generated.text)
                    questions.append(generated)

        return questions[:limit]

    # ------------------------------------------------------------------
    # Knowledge-fact question generation
    # ------------------------------------------------------------------
    def _generate_fact_questions(self, subject: str, difficulty: str,
                                 needed: int) -> list[Question]:
        """Auto-build MCQs from knowledge facts with short answers.

        The correct option is the fact's answer (trimmed to its first
        clause); distractors are answers from other facts in the same
        category. This gives StudyBuddy an effectively unlimited question
        pool without hand-writing every question.
        """
        if needed <= 0:
            return []

        target_category = None
        for category in self.knowledge.knowledge_categories():
            if category.lower() == subject.lower():
                target_category = category
                break

        pool = self.knowledge.facts_for_quiz(target_category)
        if len(pool) < 5:  # not enough options in this category — widen
            pool = self.knowledge.facts_for_quiz(None)

        picked = self._rng.sample(pool, k=min(needed, len(pool)))
        questions: list[Question] = []
        for fact in picked:
            answer = self._short_answer(fact)
            if not answer:
                continue
            distractors = self._pick_distractors(pool, fact, answer, 3)
            if len(distractors) < 3:
                continue
            options = distractors + [answer]
            self._rng.shuffle(options)
            questions.append(Question(
                text=fact.get("question", "?"),
                options=options,
                correct_index=options.index(answer),
                subject=fact.get("category", "General Knowledge"),
                topic=fact.get("category", "General Knowledge"),
                difficulty=difficulty if difficulty != "all" else "medium",
                explanation=fact.get("answer", ""),
            ))
        return questions

    @staticmethod
    def _short_answer(fact: dict[str, object]) -> str:
        """Return a compact MCQ option from a fact's full answer.

        Takes the first clause/sentence and enforces a 2-48 character range;
        facts with long discursive answers are skipped automatically.
        """
        answer = str(fact.get("answer", "")).strip()
        for separator in (". ", " — ", " – ", ";"):
            index = answer.find(separator)
            if index != -1:
                answer = answer[:index]
        answer = answer.strip(" ,.;—–:")
        if not (2 <= len(answer) <= 48):
            return ""
        return answer

    def _pick_distractors(self, pool: list[dict[str, object]], fact: dict[str, object],
                          correct: str, count: int) -> list[str]:
        """Pick ``count`` wrong answers from other facts (deduped)."""
        candidates: list[str] = []
        seen: set[str] = set()
        for other in pool:
            if other is fact:
                continue
            candidate = self._short_answer(other)
            if candidate and candidate.lower() != correct.lower() and candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
        self._rng.shuffle(candidates)
        return candidates[:count]

    # ------------------------------------------------------------------
    # The quiz loop (generator of steps)
    # ------------------------------------------------------------------
    def run(self, subject: str = "All Subjects", difficulty: str = "all",
            count: Optional[int] = None) -> Iterator[Question | QuizReport]:
        """Yield questions one by one, then the final report.

        This is a **send-capable generator**: the UI layer displays a
        :class:`Question`, collects the user's answer, and feeds it back
        with ``generator.send(answer_text)``. The next step (question or
        final report) is then returned.

        Usage pattern (implemented by chatbot.py)::

            steps = quiz_engine.run(subject, difficulty)
            step = next(steps)                       # first question
            while True:
                if isinstance(step, QuizReport):
                    break
                answer = ui.ask(...)                 # user's letter or 'quit'
                if answer is None:
                    break
                try:
                    step = steps.send(answer)
                except StopIteration:
                    break
        """
        questions = self.build_questions(subject, difficulty, count)
        if not questions:
            yield QuizReport(
                subject=subject, correct=0, total=0, percent=0.0,
                message="No questions available for that selection.",
            )
            return

        correct_count = 0
        answered: list[dict[str, object]] = []
        missed: list[dict[str, str]] = []

        for question in questions:
            # `answer` receives whatever the UI sends back via generator.send()
            answer = yield question
            correct = self._is_correct(answer, question)
            if correct:
                correct_count += 1
            else:
                missed.append({
                    "question": question.text,
                    "answer": question.options[question.correct_index],
                })
            answered.append({
                "topic": question.topic,
                "correct": correct,
                "difficulty": question.difficulty,
            })
            self.storage.record_subject_answer(question.subject, correct)

        percent = utils.progress_percent(correct_count, len(questions))
        weak_topics = self._analyse_weak_topics(answered)
        report = QuizReport(
            subject=subject,
            correct=correct_count,
            total=len(questions),
            percent=percent,
            weak_topics=weak_topics,
            missed=missed,
        )
        self.storage.record_quiz(subject, correct_count, len(questions), weak_topics)
        log.info("Quiz finished: subject=%s score=%d/%d missed=%d",
                 subject, correct_count, len(questions), len(missed))
        yield report

    @staticmethod
    def _is_correct(answer: object, question: Question) -> bool:
        """Interpret a user answer (letter, word, or index) as correct/incorrect.

        Numeric answers are ambiguous: "2" could mean option index 2 OR the
        option text "2" (when the options themselves are numbers). Both
        interpretations are accepted, so numeric quizzes never mark a right
        text answer as wrong.
        """
        if answer is None:
            return False
        text = str(answer).strip().lower()
        if not text:
            return False
        # Letter form: "b", "b)", "(b)"
        letter = text.replace("(", "").replace(")", "").strip()
        if len(letter) == 1 and letter in string.ascii_lowercase:
            return (ord(letter) - ord("a")) == question.correct_index
        # Number form: "2" can be an index OR the literal option text.
        if text.isdigit():
            if (int(text) - 1) == question.correct_index:
                return True
            correct_text = question.options[question.correct_index].lower()
            return text == correct_text
        # Spoken numbers: "four" can be index 4 OR the option text "4".
        spoken = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }
        if text in spoken:
            if (spoken[text] - 1) == question.correct_index:
                return True
            correct_text = question.options[question.correct_index].lower()
            return correct_text.isdigit() and spoken[text] == int(correct_text)
        # Full option text form (exact match)
        return text in {opt.lower() for opt in question.options} and \
            question.options[question.correct_index].lower() == text


    # ------------------------------------------------------------------
    # Weak topic analysis
    # ------------------------------------------------------------------
    @staticmethod
    def _analyse_weak_topics(answered: list[dict[str, object]]) -> list[str]:
        """Return topics where the user's accuracy is below the threshold."""
        per_topic: dict[str, dict[str, int]] = {}
        for entry in answered:
            topic = str(entry.get("topic") or "General")
            stats = per_topic.setdefault(topic, {"correct": 0, "total": 0})
            stats["total"] += 1
            if entry.get("correct"):
                stats["correct"] += 1
        weak = [
            topic for topic, stats in per_topic.items()
            if stats["total"] > 0
            and utils.progress_percent(stats["correct"], stats["total"]) < WEAK_THRESHOLD_PERCENT
        ]
        return utils.dedupe(weak)

    # ------------------------------------------------------------------
    # Long-term weak subjects (from accumulated accuracy)
    # ------------------------------------------------------------------
    def weak_subjects_from_history(self, minimum_answers: int = 3) -> list[str]:
        """Detect chronically weak subjects from stored per-subject accuracy."""
        accuracy = self.storage.get_progress().get("subject_accuracy", {})
        weak: list[str] = []
        for subject, stats in accuracy.items():
            if stats.get("answered", 0) >= minimum_answers:
                percent = utils.progress_percent(stats.get("correct", 0), stats.get("answered", 0))
                if percent < WEAK_THRESHOLD_PERCENT:
                    weak.append(f"{subject} ({percent:.0f}%)")
        return weak

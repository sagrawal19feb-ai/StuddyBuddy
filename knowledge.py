"""
knowledge.py — Knowledge base loader for StudyBuddy
=======================================================

The chatbot's "brain data" lives in JSON files under ``data/``:

  * ``responses.json``   — templated replies for every intent (many variants)
  * ``intents.json``     — intent definitions (keywords, examples)
  * ``study_tips.json``  — the study-tips database
  * ``motivation.json``  — the motivational-quote database
  * ``quiz.json``        — the MCQ question bank
  * ``knowledge.json``   — general-knowledge Q&A facts
  * ``subjects.json``    — subject metadata (topics, difficulty hints)

:class:`KnowledgeBase` loads all of these once at startup and exposes typed
accessors. If a data file is missing or corrupted the application logs the
problem and keeps running with an empty-but-valid database, so a broken
download can never brick the whole chatbot.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import utils
from config import (
    INTENTS_FILE,
    KNOWLEDGE_FILE,
    MOTIVATION_FILE,
    QUIZ_FILE,
    RESPONSES_FILE,
    STUDY_TIPS_FILE,
    SUBJECTS_FILE,
)
from logger import get_logger

log = get_logger("knowledge")


class KnowledgeBase:
    """Central repository of every static piece of chatbot knowledge."""

    def __init__(self) -> None:
        self.responses: dict[str, list[str]] = {}
        self.intents: list[dict[str, Any]] = []
        self.study_tips: dict[str, list[str]] = {}
        self.motivation: list[str] = []
        self.quiz: dict[str, Any] = {}
        self.knowledge: list[dict[str, Any]] = []
        self.subjects: dict[str, Any] = {}
        self.load_all()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_all(self) -> None:
        """Load every knowledge file. Failures degrade gracefully."""
        self.responses = self._load(RESPONSES_FILE, {}, "responses")
        raw_intents = self._load(INTENTS_FILE, [], "intents")
        self.intents = (
            raw_intents.get("intents", [])
            if isinstance(raw_intents, dict)
            else raw_intents
        )
        raw_tips = self._load(STUDY_TIPS_FILE, {}, "study tips")
        self.study_tips = {
            key: value for key, value in raw_tips.items()
            if key.lower() not in self._META_KEYS and isinstance(value, list)
        }
        raw_motivation = self._load(MOTIVATION_FILE, [], "motivation")
        self.motivation = (
            raw_motivation.get("quotes", [])
            if isinstance(raw_motivation, dict)
            else raw_motivation
        )
        self.quiz = self._load(QUIZ_FILE, {}, "quiz bank")
        raw_knowledge = self._load(KNOWLEDGE_FILE, [], "knowledge base")
        self.knowledge = (
            raw_knowledge.get("facts", [])
            if isinstance(raw_knowledge, dict)
            else raw_knowledge
        )
        self.subjects = self._load(SUBJECTS_FILE, {}, "subjects")
        self._keyword_freq: dict[str, int] = self._compute_keyword_frequency()
        self._summary()

    @staticmethod
    def _load(path, default, label: str):
        data = utils.load_json(path, default)
        if data == default or data in ({}, []):
            log.warning("Knowledge file '%s' is empty or missing; using defaults.", label)
        return data

    def _compute_keyword_frequency(self) -> dict[str, int]:
        """Count how many facts share each keyword (for distinctiveness).

        Distinctiveness is judged on whole-word containment: "prime" counts
        against every fact whose keywords mention prime numbers, prime
        factorization, or prime ministers. A word shared by many facts (e.g.
        "capital") is too generic to be decisive on its own; a word in one or
        two facts is a strong signal for the topic.
        """
        freq: dict[str, int] = {}
        for fact in self.knowledge:
            seen: set[str] = set()
            for keyword in fact.get("keywords", []):
                for word in re.findall(r"[a-z0-9]+", keyword.lower()):
                    if len(word) >= 4 and word not in seen:
                        seen.add(word)
                        freq[word] = freq.get(word, 0) + 1
        return freq

    def _keyword_frequency(self, keyword: str) -> int:
        # Unknown keywords (e.g. short distinctive terms like "pi" that were
        # filtered out of the frequency map) are treated as rare/specific.
        return self._keyword_freq.get(keyword.lower(), 1)

    def _fact_haystacks(self) -> list[str]:
        """Cache of every fact's searchable text (question + keywords)."""
        if not hasattr(self, "_haystacks"):
            self._haystacks = [
                " ".join([
                    fact.get("question", ""),
                    " ".join(fact.get("keywords", [])),
                ]).lower()
                for fact in self.knowledge
            ]
        return self._haystacks

    def _summary(self) -> None:
        log.info(
            "Knowledge base loaded: %d response keys, %d intents, %d tips, "
            "%d quotes, %d quiz entries, %d knowledge facts, %d subjects",
            len(self.responses), len(self.intents),
            sum(len(v) for v in self.study_tips.values()),
            len(self.motivation), len(self.quiz.get("questions", [])),
            len(self.knowledge), len(self.subjects),
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def response_variants(self, key: str) -> list[str]:
        """Return all reply variants for an intent key (empty if unknown)."""
        return list(self.responses.get(key, []))

    def intent_definitions(self) -> list[dict[str, Any]]:
        return self.intents

    def intent_names(self) -> list[str]:
        return [entry.get("name", "") for entry in self.intents if entry.get("name")]

    def tip_categories(self) -> list[str]:
        return list(self.study_tips.keys())

    def tips_for(self, category: str) -> list[str]:
        return list(self.study_tips.get(category, []))

    def all_tips(self) -> list[str]:
        """Flatten every tip across categories (for random display)."""
        return [tip for tips in self.study_tips.values() for tip in tips]

    def motivation_quotes(self) -> list[str]:
        return list(self.motivation)

    def quiz_data(self) -> dict[str, Any]:
        return self.quiz

    def quiz_subjects(self) -> list[str]:
        """Return subject keys that actually have questions in the bank."""
        questions = self.quiz.get("questions", [])
        return utils.dedupe(q.get("subject", "") for q in questions if q.get("subject"))

    def questions_for(self, subject: Optional[str] = None,
                      difficulty: Optional[str] = None) -> list[dict[str, Any]]:
        """Filter the question bank by subject and/or difficulty."""
        questions = self.quiz.get("questions", [])
        if subject and subject != "all":
            questions = [q for q in questions if q.get("subject", "").lower() == subject.lower()]
        if difficulty and difficulty != "all":
            questions = [q for q in questions if q.get("difficulty", "").lower() == difficulty.lower()]
        return questions

    def knowledge_facts(self, category: Optional[str] = None) -> list[dict[str, Any]]:
        if category:
            return [f for f in self.knowledge if f.get("category") == category]
        return self.knowledge

    def knowledge_categories(self) -> list[str]:
        return utils.dedupe(f.get("category", "General") for f in self.knowledge)

    # Metadata keys inside subject files that are not subjects themselves.
    _META_KEYS = {"description", "_comment", "meta"}

    def _subject_entries(self) -> list[tuple[str, dict[str, Any]]]:
        """Yield (name, info) pairs for real subjects, skipping metadata keys."""
        for key, value in self.subjects.items():
            if key.lower() in self._META_KEYS or not isinstance(value, dict):
                continue
            yield key, value

    def subject_names(self) -> list[str]:
        """Return the display names of every known subject."""
        return [name for name, _ in self._subject_entries()]

    def subject_topics(self, subject: str) -> list[str]:
        """Return the topic list for a subject (case-insensitive lookup)."""
        for key, value in self._subject_entries():
            if key.lower() == subject.lower():
                return list(value.get("topics", []))
        return []

    def subject_info(self, subject: str) -> dict[str, Any]:
        for key, value in self._subject_entries():
            if key.lower() == subject.lower():
                return value
        return {}

    def find_knowledge(self, query: str) -> Optional[dict[str, Any]]:
        """Best-effort lookup of a knowledge fact that answers ``query``.

        Two matching strategies are combined:

        1. **Keyword scoring** — count how many of the fact's keywords appear
           in the query (longer keywords count for more).
        2. **Fuzzy matching** — if keywords score poorly, compare the query
           against each fact's question with ``rapidfuzz`` partial ratio, so
           paraphrases ("powerhouse of the cell") and small typos still hit
           the right fact.

        A match is returned only when it is confident enough; otherwise
        ``None`` so the caller can fall back to search or a generic reply.
        """
        fact, _keyword_based, _strength = self.knowledge_match(query)
        return fact

    def find_knowledge_strength(self, query: str) -> tuple[Optional[dict[str, Any]], bool]:
        """Legacy wrapper: returns ``(fact, keyword_based)``.

        See :meth:`knowledge_match` for the full-strength version.
        """
        fact, keyword_based, _strength = self.knowledge_match(query)
        return fact, keyword_based

    def knowledge_match(self, query: str) -> tuple[Optional[dict[str, Any]], bool, float]:
        """Find the best knowledge fact for ``query`` and rate the match.

        Returns
        -------
        (fact, keyword_based, strength)
            ``keyword_based`` is ``True`` when at least one of the fact's
            keywords appeared in the query (a strong, specific match).
            ``strength`` is a weighted evidence score: multi-word keywords
            count 2.0, single distinctive keywords count 1.0 (so a single
            generic word like "study" scores only ~1.0 and cannot trigger
            the intent shortcut, while "photosynthesis" + "plants make food"
            scores 3.0+).

        The fuzzy path is deliberately conservative: it only fires when the
        query contains a **distinctive word** that actually appears in the
        candidate fact, preventing template false-positives where
        "what is the capital of Brazil" would otherwise fuzzy-match
        "what is the capital of Japan?" just because the sentence shapes
        are similar.
        """
        from rapidfuzz import fuzz
        lowered = query.lower()
        significant = self._significant_words(query)  # rarest first
        rarest = significant[0] if significant else None
        # If the query's most specific word appears in NO fact, the topic is
        # simply not covered — do not guess via secondary/common words.
        rarest_covered = (
            rarest is None
            or self._keyword_frequency(rarest) > 0
            or any(rarest in h for h in self._fact_haystacks())
        )
        best: Optional[dict[str, Any]] = None
        best_score = 0.0
        best_strength = 0.0
        keyword_based = False
        for fact in self.knowledge:
            score = 0.0
            strength = 0.0
            hit = False
            for keyword in fact.get("keywords", []):
                key = keyword.lower()
                # Whole-word match only — "pi" must NOT match inside "capital",
                # and "api" must not match inside "capital" either.
                if not re.search(rf"\b{re.escape(key)}\b", lowered):
                    continue
                if " " in key:
                    # Multi-word keywords are specific phrases: strong evidence.
                    hit = True
                    score += 1.0 + len(key) / 40.0
                    strength += 2.0
                elif rarest_covered and rarest is not None and key == rarest:
                    # A single-word keyword is decisive only when it IS the
                    # query's most specific (rarest) word — so "sigma rule"
                    # can never match a DNA fact via the incidental word
                    # "rule"; "gravity" still matches the gravity fact.
                    hit = True
                    score += 1.0 + len(key) / 40.0
                    strength += 1.0
                elif rarest is None and self._keyword_frequency(key) <= 1:
                    # No significant words in the query ("what is pi") — only
                    # accept genuinely distinctive single keywords.
                    hit = True
                    score += 1.0 + len(key) / 40.0
                    strength += 1.0
            if hit and score > best_score:
                best_score = score
                best = fact
                best_strength = strength
                keyword_based = True

        # Fuzzy fallback: paraphrases and typos that keywords missed.
        if best_score < 1.0:
            # The RAREST significant query word must appear in the fact.
            # A common word like "capital" is not evidence of a match; a rare
            # word like "brazil" is. If the rarest word is absent from the
            # whole knowledge base (frequency 0), the topic is simply not
            # covered — no fuzzy guessing.
            if significant:
                decisive = significant[0]
                haystacks = self._fact_haystacks()
                for index, fact in enumerate(self.knowledge):
                    if decisive not in haystacks[index]:
                        continue
                    fuzzy = float(fuzz.partial_ratio(lowered, fact.get("question", "").lower()))
                    if fuzzy >= 75.0 and fuzzy > best_score:
                        best_score = fuzzy
                        best = fact
                        best_strength = 1.5
                        keyword_based = False

        if best_score > 0:
            return best, keyword_based, best_strength
        return None, False, 0.0

    def _significant_words(self, query: str) -> list[str]:
        """Distinctive words in the query, rarest first.

        Stopwords and very common words are dropped; the remaining words are
        ranked by how rarely they appear across the knowledge base, so the
        rarest word (e.g. "brazil") is tried first as the decisive match.
        """
        import re
        stopwords = {
            "what", "whats", "which", "where", "when", "who", "whom",
            "why", "how", "does", "do", "did", "is", "are", "was", "were",
            "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
            "with", "from", "by", "that", "this", "these", "those", "it", "its",
            "can", "could", "would", "should", "about", "between", "difference",
            "mean", "means", "meaning", "define", "called", "known", "tell",
            "please", "explain", "give", "show", "want", "need", "you", "your",
            # generic attribute words that describe the QUESTION, not the
            # topic: "what is the VALUE of pi" -> the topic is pi, not value.
            "value", "values", "symbol", "symbols", "formula", "formulas",
            "definition", "example", "examples", "full", "stand", "stands",
            "name", "names", "part", "role", "function", "working", "works",
            "effect", "effects", "process", "uses", "use", "used", "called",
        }
        words = {
            w for w in re.findall(r"[a-zA-Z]{4,}", query.lower())
            if w not in stopwords
        }
        def frequency(word: str) -> int:
            return sum(1 for h in self._fact_haystacks() if word in h)
        return sorted(words, key=frequency)

    def related_facts(self, fact: dict[str, Any], limit: int = 2,
                      exclude: Optional[list[dict[str, Any]]] = None) -> list[dict[str, Any]]:
        """Return other facts from the same category, for follow-up reading.

        Parameters
        ----------
        fact : dict
            The fact that was just answered.
        limit : int
            How many related facts to return (default 2).
        exclude : list[dict] | None
            Additional facts to skip (e.g. ones already shown).
        """
        import random
        category = fact.get("category", "General")
        excluded_ids = {id(fact)}
        for item in exclude or []:
            excluded_ids.add(id(item))
        candidates = [
            f for f in self.knowledge
            if f.get("category") == category and id(f) not in excluded_ids
        ]
        random.shuffle(candidates)
        return candidates[:limit]

    def facts_for_quiz(self, category: Optional[str] = None) -> list[dict[str, Any]]:
        """Return facts whose answers are short enough to use as MCQ options.

        The quiz engine uses these to auto-generate questions, so StudyBuddy
        can quiz on any topic even without hand-written MCQs.
        """
        def short(fact: dict[str, Any]) -> bool:
            answer = fact.get("answer", "").strip()
            return 2 <= len(answer) <= 48
        facts = self.knowledge_facts(category)
        return [f for f in facts if short(f)]

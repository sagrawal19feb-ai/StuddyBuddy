"""
intents.py — Intent detection engine for StudyBuddy
======================================================

This module decides *what the user means*. The pipeline is:

    1. Expand contractions ("don't" → "do not", "what's" → "what is")
       so every intent keyword matches spoken/written English uniformly.
    2. Normalise the input (lowercase, strip punctuation, collapse spaces).
    3. Correct obvious typos using rapidfuzz against a vocabulary built
       from every intent keyword and all subject names.
    4. Score the input against every intent using a combination of:
         - exact substring matches for multi-word keywords,
         - fuzzy (Levenshtein) similarity for single keywords, with a
           length guard so "you" never fuzzy-matches the keyword "yo",
         - similarity to the hand-written *example* phrases per intent
           (token-set ratio, order-insensitive),
         - a specificity bonus and evidence accumulation for tie-breaking.
    5. Return the best intent with a confidence score, or ``"unknown"``.

Negation guard
--------------
Messages that explicitly decline something ("no quiz", "i don't want a
test") are routed to ``"unknown"`` instead of triggering the declined
intent — the bot should not push a quiz on someone who just said no.

Intents are defined declaratively in ``data/intents.json`` so new behaviours
can be added without touching Python code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from rapidfuzz import fuzz, process

from config import RUNTIME
from knowledge import KnowledgeBase
from logger import get_logger

log = get_logger("intents")

# Intent names handled implicitly by active wizards.
WIZARD_INTENTS = {"quiz", "flashcards", "schedule", "daily_plan"}

# ---------------------------------------------------------------------------
# Contraction expansion — applied BEFORE matching so "i dont understand"
# becomes "i do not understand" and matches the same keywords.
# ---------------------------------------------------------------------------
_CONTRACTION_MAP: dict[str, str] = {
    "dont": "do not", "don't": "do not", "do not": "do not",
    "cant": "can not", "can't": "can not", "cannot": "can not",
    "wont": "will not", "won't": "will not",
    "didnt": "did not", "didn't": "did not",
    "isnt": "is not", "isn't": "is not",
    "arent": "are not", "aren't": "are not",
    "wasnt": "was not", "wasn't": "was not",
    "werent": "were not", "weren't": "were not",
    "hasnt": "has not", "hasn't": "has not",
    "havent": "have not", "haven't": "have not",
    "im": "i am", "i'm": "i am", "i am": "i am",
    "ive": "i have", "i've": "i have",
    "id": "i would", "i'd": "i would",
    "id like": "i would like", "i'd like": "i would like",
    "youre": "you are", "you're": "you are",
    "whats": "what is", "what's": "what is", "what is": "what is",
    "wheres": "where is", "where's": "where is",
    "whos": "who is", "who's": "who is",
    "hows": "how is", "how's": "how is",
    "its": "it is", "it's": "it is",
    "doesnt": "does not", "doesn't": "does not",
    "wanna": "want to", "gonna": "going to", "gotta": "got to",
    # informal / texting abbreviations (safe: never applied to math input)
    "idk": "i do not know", "dunno": "do not know",
    "bout": "about", "gimme": "give me", "lemme": "let me",
    "u": "you", "ur": "your", "r": "are", "plz": "please",
    "cya": "see you", "wht": "what", "whts": "what is",
    "cause": "because", "cuz": "because", "tho": "though",
    "kinda": "kind of", "hru": "how are you",
    "y": "why", "n": "no", "ya": "you", "ye": "yes",
}

# Phrases that indicate the user is explicitly declining something, so the
# detected intent should NOT fire. Kept narrow on purpose: "i do not
# understand chemistry" must NOT be treated as a refusal.
_NEGATION_HINTS = (
    "no thanks", "nah", "nope", "not interested", "not now", "not today",
    "skip it", "skip that", "never mind",
    "do not want", "dont want", "don't want", "do not wish", "dont wish",
    "i do not want", "i dont want", "i don't want",
    "i do not need", "i dont need", "i don't need",
    "no more", "stop",
)


def _expand_contractions(text: str) -> str:
    """Replace common contractions/spellings with canonical forms.

    Token-level replacement: "dont" → "do not", "whats" → "what is", etc.
    Plain tokens and punctuation around them are preserved.
    """
    tokens = text.split()
    expanded: list[str] = []
    for token in tokens:
        stripped = token.strip(".,!?;:()\"'’")
        replacement = _CONTRACTION_MAP.get(stripped.lower())
        if replacement:
            # Preserve trailing punctuation from the original token.
            suffix = token[len(stripped):] if token.endswith(stripped) else ""
            prefix = token[:len(token) - len(stripped)] if token.startswith(stripped) else ""
            expanded.append(prefix + replacement + suffix)
        else:
            expanded.append(token)
    return " ".join(expanded)


@dataclass
class IntentResult:
    """The outcome of intent detection for one user message."""

    name: str                 # canonical intent name
    confidence: float         # 0.0 - 1.0
    matched_keyword: str      # the keyword/example that produced the match
    topic: Optional[str] = None   # detected subject/topic, if any


class IntentDetector:
    """Detects the user's intent using fuzzy matching and topic detection."""

    def __init__(self, knowledge: KnowledgeBase, threshold: Optional[float] = None) -> None:
        self.knowledge = knowledge
        self.threshold = float(
            threshold if threshold is not None
            else RUNTIME.get("fuzzy_threshold", 78.0)
        )
        self._intent_defs: list[dict[str, Any]] = knowledge.intent_definitions()
        self._vocabulary: set[str] = set()
        self._subject_names: list[str] = []
        self._examples: dict[str, list[str]] = {}
        self._build_index()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        """Pre-compute keyword, example and vocabulary structures."""
        for definition in self._intent_defs:
            name = definition.get("name", "")
            keywords = definition.get("keywords", [])
            self._examples[name] = [
                self._normalise(ex) for ex in definition.get("examples", [])
            ]
            for keyword in keywords:
                self._vocabulary.add(keyword.lower())
                for word in keyword.lower().split():
                    if len(word) >= 3:
                        self._vocabulary.add(word)
        self._subject_names = [name.lower() for name in self.knowledge.subject_names()]
        for name in self.knowledge.subject_names():
            self._vocabulary.add(name.lower())
            for topic in self.knowledge.subject_topics(name):
                self._vocabulary.add(topic.lower())
        log.debug("Intent index built: %d intents, %d vocabulary terms, %d example sets",
                  len(self._intent_defs), len(self._vocabulary), len(self._examples))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, text: str) -> IntentResult:
        """Detect the intent behind ``text``.

        Returns
        -------
        IntentResult
            The best-matching intent and confidence. If nothing reaches the
            confidence threshold the intent is ``"unknown"``.
        """
        raw = text.strip()
        if not raw:
            return IntentResult("unknown", 0.0, "")

        expanded = _expand_contractions(raw)
        corrected = self.correct_typos(expanded)
        normalised = self._normalise(corrected)

        # Negation guard: if the message explicitly declines a bot service
        # ("no quiz", "i don't want a test"), never fire that intent.
        if self._is_negation(expanded):
            # ...but "i don't want to study / learn" is a DEMOTIVATION
            # statement, which is exactly the motivation intent.
            if re.search(r"\b(study|learn|revise|revision|exam|homework)\b",
                         expanded.lower()):
                return IntentResult("motivation", 0.6, "demotivation")
            return IntentResult("unknown", 0.0, "")

        topic = self.detect_topic(corrected)
        best_name, best_score, best_keyword = self._score_all(normalised)

        # A strong topic mention with weak intent evidence still suggests the
        # user wants study help on that topic.
        if topic and best_score < 0.35:
            return IntentResult("study_help", 0.5, topic, topic)

        # Knowledge shortcut: "explain gravity" / "what is X" are better
        # served by a direct knowledge-base answer when a fact genuinely
        # matches. Requires a whole-word keyword hit (keyword_based) — the
        # junk-generic-keyword false positives were eliminated from the data,
        # so a single distinctive keyword (e.g. "photosynthesis") is enough.
        if best_name in {"study_help", "knowledge"}:
            fact, keyword_based, strength = self.knowledge.knowledge_match(corrected)
            if fact and keyword_based and strength >= 1.0 and best_score >= 80.0:
                return IntentResult("knowledge", 1.0, "knowledge-fact", topic)

        confidence = min(1.0, best_score / 100.0)
        return IntentResult(best_name, confidence, best_keyword, topic)

    def _is_negation(self, text: str) -> bool:
        """True when the message reads like a refusal/decline.

        Only explicit refusal phrases count ("no thanks", "i do not want a
        quiz"). Study phrases that merely contain "not" ("i do not
        understand chemistry") are never treated as refusals.
        """
        lowered = text.lower().strip()
        if any(hint in lowered for hint in _NEGATION_HINTS):
            return True
        # Short "no ..." refusals ("no quiz today", "no tests",
        # "no flashcards right now") — but not study phrases like
        # "no idea how to solve this problem" (too long to be a bare refusal).
        return lowered.startswith("no ") and len(lowered) <= 24

    def correct_typos(self, text: str) -> str:
        """Fix misspelled words using fuzzy matching against the vocabulary.

        Only tokens whose best match scores high enough are replaced, so
        legitimate words are never mangled.
        """
        tokens = re_words(text)
        corrected: list[str] = []
        for token in tokens:
            lowered = token.lower()
            if lowered in self._vocabulary or not lowered.isalpha() or len(token) <= 2:
                corrected.append(token)
                continue
            best = process.extractOne(
                lowered,
                list(self._vocabulary),
                scorer=fuzz.ratio,
                score_cutoff=88,
            )
            if best is not None:
                corrected.append(best[0])
            else:
                corrected.append(token)
        return " ".join(corrected)

    def detect_topic(self, text: str) -> Optional[str]:
        """Return the subject/topic mentioned in ``text`` (or ``None``).

        Uses fuzzy matching against subject names and their topic lists so
        both "maths" and "math" resolve to the canonical subject.
        """
        lowered = text.lower()

        # 1) Exact / substring match against canonical subject names
        for name in self._subject_names:
            if name in lowered:
                return self.knowledge.subject_names()[self._subject_names.index(name)]

        # 2) Fuzzy match against subject names (handles "mathematics"->"maths")
        match = process.extractOne(
            lowered, self._subject_names, scorer=fuzz.partial_ratio
        )
        if match is not None:
            best_name, best_score = match[0], match[1]
            if best_score >= 75:
                return self.knowledge.subject_names()[self._subject_names.index(best_name)]

        # 3) Fuzzy match against known topics (e.g. "algebra")
        all_topics: list[str] = []
        topic_owner: dict[str, str] = {}
        for name in self.knowledge.subject_names():
            for topic in self.knowledge.subject_topics(name):
                key = topic.lower()
                all_topics.append(key)
                topic_owner.setdefault(key, name)
        if all_topics:
            match = process.extractOne(
                lowered, all_topics, scorer=fuzz.partial_ratio
            )
            if match is not None and match[1] >= 78:
                best_topic = match[0]
                return topic_owner.get(best_topic, best_topic.capitalize())
        return None

    # ------------------------------------------------------------------
    # Scoring internals
    # ------------------------------------------------------------------
    def _score_all(self, normalised: str) -> tuple[str, float, str]:
        """Score the query against every intent.

        Combines keyword matching with example-phrase matching. The winner
        is chosen by score, with ties broken by *evidence* (the sum of the
        top keyword scores), so a broad intent with many weak hits can't
        beat a focused intent with one strong hit.
        """
        best_name = "unknown"
        best_score = 0.0
        best_keyword = ""
        best_evidence = 0.0
        best_specificity = 0  # words in the best matched keyword/example

        for definition in self._intent_defs:
            name = definition.get("name", "")
            keyword_scores = [
                self._score_keyword(normalised, keyword)
                for keyword in definition.get("keywords", [])
            ]
            top_keyword = max(keyword_scores, default=0.0)
            top_kw = ""
            if top_keyword > 0:
                # Among keywords tied at the top score, prefer the LONGEST —
                # "my progress" is more specific evidence than "progress".
                tied = [
                    definition.get("keywords", [])[i]
                    for i, s in enumerate(keyword_scores) if s == top_keyword
                ]
                top_kw = max(tied, key=lambda k: len(k.split()))
            # Evidence = sum of the top 3 keyword scores.
            evidence = sum(sorted(keyword_scores, reverse=True)[:3])

            # Example-phrase matching (token-set ratio is order-insensitive).
            example_score = 0.0
            example_text = ""
            for example in self._examples.get(name, []):
                s = self._fuzzy_phrase(normalised, example)
                if s > example_score:
                    example_score = s
                    example_text = example

            if example_score * 0.92 > top_keyword:
                score = example_score * 0.92
                specificity = len(example_text.split())
            else:
                score = top_keyword
                specificity = len(top_kw.split()) if top_kw else 0

            # Prefer: higher score; on a tie, the MORE SPECIFIC (longer)
            # matched keyword — "i need help" beats "help", "study plan"
            # beats "study", "best way to study" beats "what is the".
            # Last resort: total evidence.
            if (score > best_score
                    or (score == best_score and specificity > best_specificity)
                    or (score == best_score and specificity == best_specificity
                        and evidence > best_evidence)):
                best_score = score
                best_name = name
                best_keyword = top_kw or example_text
                best_evidence = evidence
                best_specificity = specificity

        if best_score < self.threshold:
            return "unknown", best_score, best_keyword
        return best_name, best_score, best_keyword

    def _score_keyword(self, normalised: str, keyword: str) -> float:
        """Score how well ``normalised`` matches ``keyword`` (0-100)."""
        kw = keyword.lower().strip()
        if not kw:
            return 0.0
        # Multi-word phrases: substring presence is a strong signal; else an
        # order-insensitive fuzzy match catches rephrasings.
        if len(kw.split()) > 1:
            if kw in normalised:
                return 100.0
            return self._fuzzy_phrase(normalised, kw)
        # Single words: exact token match, else a guarded fuzzy match.
        # The length guard stops "you" from fuzzy-matching the keyword "yo",
        # and the high cutoff stops junk matches on common short words.
        if kw in normalised.split():
            return 100.0
        if len(kw) < 3:
            return 0.0
        token_score = max(
            (fuzz.ratio(token, kw) for token in normalised.split() if len(token) >= 3),
            default=0.0,
        )
        return float(token_score) if token_score >= 85 else 0.0

    @staticmethod
    def _fuzzy_phrase(query: str, phrase: str) -> float:
        """Order-insensitive fuzzy similarity between two phrases.

        Uses ``fuzz.token_ratio`` but *penalises one-sided containment*:
        rapidfuzz scores "how are you" vs "how are you built" as a perfect
        100 because one token set is a subset of the other. That overstates
        the match — the user never said "built". When the score is perfect
        but the token sets differ, the match is capped at 85 so an exact
        keyword hit (checked separately) always wins.
        """
        score = float(fuzz.token_ratio(query, phrase))
        if score >= 99.0 and set(query.split()) != set(phrase.split()):
            return 85.0
        return score

    @staticmethod
    def _normalise(text: str) -> str:
        """Lowercase, expand contractions, strip punctuation, collapse spaces."""
        expanded = _expand_contractions(text)
        lowered = expanded.lower()
        cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in lowered)
        return " ".join(cleaned.split())


def re_words(text: str) -> list[str]:
    """Split text into word tokens."""
    return [token for token in text.split() if token.strip()]

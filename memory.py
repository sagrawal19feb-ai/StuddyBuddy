"""
memory.py — Conversation memory for StudyBuddy
==================================================

Unlike a plain stateless chatbot, StudyBuddy keeps a working memory of the
current session so conversations flow naturally:

  * the user's name, class and favourite subjects (loaded from the profile)
  * the most recent topic/subject being discussed
  * the last intent that was handled (used to build follow-up questions)
  * a *pending action* — e.g. after asking "Which topic?" the chatbot waits
    for the answer before routing it to the right module
  * a small recent-turn buffer for light anaphora resolution ("that", "it")

:class:`ConversationMemory` is deliberately in-memory for the session; long
term persistence lives in :mod:`storage`.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Optional

import utils
from storage import StorageManager

# Pronouns that usually point back at the most recent topic.
_REFERENTIAL_WORDS = {"it", "that", "this", "them", "they", "the topic", "the subject"}

# Multi-word intent keywords that contain a referential word. When one of
# these appears in a message, the referential word is part of a fixed
# command phrase ("save this fact", "remember that", "forget that") and must
# NOT be rewritten to the last topic — otherwise "save this fact for later"
# would become "save Atomic Structure fact for later".
def _protected_reference_phrases(intents: list[dict[str, Any]]) -> frozenset[str]:
    phrases: set[str] = set()
    for definition in intents:
        for keyword in definition.get("keywords", []):
            lowered = keyword.lower()
            if len(lowered.split()) >= 2 and any(
                word in lowered for word in _REFERENTIAL_WORDS
            ):
                phrases.add(lowered)
    return frozenset(phrases)


class ConversationMemory:
    """Session-scoped context holder for natural-feeling conversation."""

    def __init__(self, storage: StorageManager) -> None:
        self.storage = storage
        self._protected_refs: frozenset[str] = frozenset()
        self.recent_turns: deque[dict[str, str]] = deque(maxlen=6)
        self.last_topic: Optional[str] = None
        self.last_intent: Optional[str] = None
        self.last_action: Optional[dict[str, str]] = None   # e.g. {"intent": "quiz", "detail": "Mathematics"}
        self.recent_topics: deque[str] = deque(maxlen=5)    # rolling topic history
        self.distraction_streak: int = 0                    # consecutive distraction attempts
        self.pending_action: Optional[str] = None   # e.g. "study_help_topic"
        self.pending_context: dict[str, Any] = {}   # extra state for the pending action
        self.session_start: str = ""
        self.turns_this_session: int = 0

    # ------------------------------------------------------------------
    # Profile-backed facts
    # ------------------------------------------------------------------
    @property
    def user_name(self) -> str:
        return str(self.storage.get_profile().get("name") or "Friend")

    @property
    def user_class(self) -> str:
        return str(self.storage.get_profile().get("class") or "")

    @property
    def favourite_subjects(self) -> list[str]:
        return list(self.storage.get_profile().get("favourite_subjects", []))

    @property
    def weak_subjects(self) -> list[str]:
        return list(self.storage.get_profile().get("weak_subjects", []))

    def set_protected_phrases(self, intents: list[dict[str, Any]]) -> None:
        """Register the fixed command phrases that must not be rewritten.

        Called at startup with the intent definitions, so reference
        resolution never mangles commands like "save this fact".
        """
        self._protected_refs = _protected_reference_phrases(intents)

    # ------------------------------------------------------------------
    # Turn tracking
    # ------------------------------------------------------------------
    def record_turn(self, user_text: str, bot_text: str) -> None:
        """Store one exchange in the rolling buffer."""
        self.recent_turns.append({"user": user_text, "bot": bot_text})
        self.turns_this_session += 1

    def resolve_reference(self, text: str) -> str:
        """Replace "it/that/this" with the last topic when unambiguous.

        Example
        -------
        >>> memory.last_topic = "Algebra"
        >>> memory.resolve_reference("Give me notes for that")
        'Give me notes for Algebra'

        Fixed command phrases ("save this fact", "remember that") are
        protected: their referential word is part of the command, not a
        pointer to the topic.
        """
        if not self.last_topic:
            return text
        lowered = text.lower().strip()
        for phrase in self._protected_refs:
            if phrase not in lowered:
                continue
            last_word = phrase.split()[-1]
            # A phrase whose LAST word is the reference ("tell me more about
            # it") is a genuine pointer when the message also ends on it —
            # resolve it. Otherwise ("save this fact", "remember that X")
            # the reference is part of a fixed command — leave it alone.
            if (last_word in _REFERENTIAL_WORDS
                    and re.search(rf"\b{re.escape(last_word)}\b\s*[.!?]*$", lowered)):
                continue
            return text
        for word in sorted(_REFERENTIAL_WORDS, key=len, reverse=True):
            pattern = rf"\b{re.escape(word)}\b"
            if re.search(pattern, lowered):
                return re.sub(pattern, self.last_topic, text, flags=re.IGNORECASE)
        return text

    # ------------------------------------------------------------------
    # Pending-action workflow
    # ------------------------------------------------------------------
    def set_pending(self, action: Optional[str], **context: Any) -> None:
        """Queue an expected reply from the user (or clear it with ``None``)."""
        self.pending_action = action
        self.pending_context = context

    def consume_pending(self) -> Optional[str]:
        """Return and clear the pending action (used after a successful reply)."""
        action = self.pending_action
        self.pending_action = None
        self.pending_context = {}
        return action

    # ------------------------------------------------------------------
    # Topic & intent tracking
    # ------------------------------------------------------------------
    def set_topic(self, topic: str) -> None:
        """Remember the subject/topic currently being discussed.

        Also persists it in the profile so a later session can pick up
        where the user left off, and keeps a rolling topic history so the
        bot can recall what was covered earlier in the conversation.
        """
        topic = topic.strip()
        if topic and topic.lower() != (self.last_topic or "").lower():
            self.last_topic = topic
            self.recent_topics.append(topic)
            self.storage.update_profile(last_topic=topic)

    def set_intent(self, intent: str) -> None:
        self.last_intent = intent

    def record_action(self, intent: str, detail: str = "") -> None:
        """Remember the most recent thing the bot did, for context.

        Used for contextual follow-ups (\"you just did a quiz on Mathematics
        — want to review the missed ones?\") and the \"what were we doing?\"
        recap.
        """
        self.last_action = {"intent": intent, "detail": detail}
        self.last_intent = intent

    def recent_topics_list(self) -> list[str]:
        """Return the topics discussed this session, oldest first."""
        return list(self.recent_topics)

    def is_weak_subject(self, topic: str) -> bool:
        """True when ``topic`` matches a subject the user finds tricky.

        Handles parent/child relationships between knowledge categories and
        the subject names users actually type: a fact categorised under
        "Science" counts as weak when the user listed "Physics" or
        "Chemistry" as weak, and "Computer Science" covers "Programming".
        """
        topic_lower = topic.lower()
        if any(topic_lower == weak.lower() for weak in self.weak_subjects):
            return True
        # Category -> child topics map (category: children the user may say).
        children = {
            "science": ["physics", "chemistry", "biology", "environmental science"],
            "computer science": ["programming", "coding", "computer", "python", "java"],
            "mathematics": ["maths", "math", "algebra", "calculus", "geometry", "trigonometry"],
            "history": ["indian history", "world history"],
            "geography": ["indian geography", "world geography"],
        }
        weak_lower = [w.lower() for w in self.weak_subjects]
        # 1. Topic is a parent category and a weak subject is one of its children.
        for weak in weak_lower:
            if weak in children.get(topic_lower, []):
                return True
        # 2. Topic is a child and a weak subject is its parent category.
        for weak in weak_lower:
            if topic_lower in children.get(weak, []):
                return True
        return False

    # ------------------------------------------------------------------
    # Custom facts (things the user teaches the bot)
    # ------------------------------------------------------------------
    def add_custom_fact(self, topic: str, fact: str) -> None:
        """Store a user-taught fact: e.g. 'mitochondria' -> '...'."""
        topic = topic.strip()
        fact = fact.strip()
        if not topic or not fact:
            return
        profile = self.storage.get_profile()
        facts = profile.setdefault("custom_facts", {})
        facts[topic.lower()] = {
            "topic": topic,
            "fact": fact,
            "created": utils.timestamp_str(),
        }
        self.storage.update_profile(custom_facts=facts)

    def forget_custom_fact(self, topic: str) -> bool:
        """Remove a stored fact; returns True if something was removed."""
        profile = self.storage.get_profile()
        facts = profile.setdefault("custom_facts", {})
        key = topic.lower()
        removed = False
        for stored_key in list(facts):
            if stored_key == key or key in stored_key or stored_key in key:
                del facts[stored_key]
                removed = True
        if removed:
            self.storage.update_profile(custom_facts=facts)
        return removed

    def find_custom_fact(self, query: str) -> dict[str, str] | None:
        """Return a stored fact whose topic matches the query, if any."""
        profile = self.storage.get_profile()
        facts = profile.get("custom_facts", {})
        if not facts:
            return None
        lowered = query.lower()
        for key, entry in facts.items():
            if key in lowered:
                return entry
        # Fallback: match on significant topic words (>=4 chars).
        query_words = {w for w in lowered.split() if len(w) >= 4}
        for key, entry in facts.items():
            topic_words = {w for w in key.split() if len(w) >= 4}
            if topic_words and topic_words & query_words:
                return entry
        return None

    def list_custom_facts(self) -> list[dict[str, str]]:
        profile = self.storage.get_profile()
        return list(profile.get("custom_facts", {}).values())

    # ------------------------------------------------------------------
    # Recent-search memory
    # ------------------------------------------------------------------
    def add_recent_search(self, query: str) -> None:
        """Remember a web search (max 12, newest first)."""
        profile = self.storage.get_profile()
        searches = profile.get("recent_searches", [])
        if query in searches:
            searches.remove(query)
        searches.insert(0, query)
        profile["recent_searches"] = searches[:12]
        self.storage.update_profile(recent_searches=searches[:12])

    def recent_searches(self, limit: int = 5) -> list[str]:
        return list(self.storage.get_profile().get("recent_searches", []))[:limit]

    # ------------------------------------------------------------------
    # Profile updates
    # ------------------------------------------------------------------
    def update_profile(self, **fields: Any) -> None:
        self.storage.update_profile(**fields)

    # ------------------------------------------------------------------
    # Snapshot / restore (used by the profile on startup)
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable snapshot for debugging."""
        return {
            "last_topic": self.last_topic,
            "last_intent": self.last_intent,
            "last_action": self.last_action,
            "recent_topics": self.recent_topics_list(),
            "pending_action": self.pending_action,
            "turns_this_session": self.turns_this_session,
        }

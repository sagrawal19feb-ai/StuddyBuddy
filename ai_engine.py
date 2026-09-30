"""
ai_engine.py — Response generation engine for StudyBuddy
===========================================================

This module decides *what the chatbot says*:

  * Every reply is chosen at random from many hand-written variants stored
    in ``data/responses.json``, and the same variant is never used twice in
    a row (templates keep a per-key memory).
  * Replies are personalised with the user's name and current time of day.
  * Follow-up questions are generated contextually — after "I need help in
    maths" the bot lists Algebra/Calculus/... and remembers it is waiting
    for a topic.
  * Structured replies (lists, tips, quotes) are wrapped into nice panels by
    the UI layer — the engine returns plain text plus a ``style`` hint.

Design: this engine is rule/template based (fully offline, deterministic
quality, zero API cost). The module is structured so a neural/LLM backend
could be dropped in later behind the same :meth:`generate` interface.
"""

from __future__ import annotations

import random
from typing import Optional

import utils
from knowledge import KnowledgeBase
from logger import get_logger
from memory import ConversationMemory

log = get_logger("ai_engine")


class ResponseGenerator:
    """Generates varied, context-aware chatbot replies."""

    # Style hints consumed by ui.py to choose panel colours.
    STYLE_INFO = "info"
    STYLE_SUCCESS = "success"
    STYLE_WARNING = "warning"
    STYLE_ERROR = "error"
    STYLE_MOTIVATION = "motivation"
    STYLE_LIST = "list"

    def __init__(self, knowledge: KnowledgeBase, memory: ConversationMemory) -> None:
        self.knowledge = knowledge
        self.memory = memory
        self._last_used: dict[str, str] = {}   # key -> last variant text

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------
    def generate(self, intent: str, **context: object) -> tuple[str, str]:
        """Return ``(message, style)`` for the given intent.

        Parameters
        ----------
        intent : str
            Canonical intent name (see ``data/intents.json``).
        context : dict
            Extra data used to fill placeholders in the reply templates,
            e.g. ``name=..., topic=...``.

        Returns
        -------
        tuple[str, str]
            The message text and a style hint for the UI layer.
        """
        variants = self.knowledge.response_variants(intent)
        if not variants:
            variants = self.knowledge.response_variants("fallback")
        template = self._pick_variant(intent, variants)
        message = self._fill_template(template, context)
        return message, self.STYLE_INFO

    def _pick_variant(self, key: str, variants: list[str]) -> str:
        """Pick a variant avoiding an immediate repeat of the last one."""
        if not variants:
            return ""
        last = self._last_used.get(key)
        pool = [v for v in variants if v != last] or variants
        chosen = random.choice(pool)
        self._last_used[key] = chosen
        return chosen

    def _fill_template(self, template: str, context: dict[str, object]) -> str:
        """Substitute ``{placeholder}`` tokens with context values."""
        if not template:
            return ""
        placeholders = {
            "name": self.memory.user_name,
            "topic": context.get("topic") or self.memory.last_topic or "that topic",
            "greeting": utils.greeting_for_hour(utils.hour_of_day()),
        }
        # Accept strings AND numbers (e.g. {n} = a count) as placeholder
        # values; anything else is ignored.
        placeholders.update({
            k: v for k, v in context.items()
            if isinstance(v, (str, int, float))
        })
        try:
            return template.format(**placeholders)
        except (KeyError, IndexError, ValueError):
            return template  # never crash on a malformed template

    # ------------------------------------------------------------------
    # Curated content (study tips / motivation)
    # ------------------------------------------------------------------
    def study_tip(self, category: Optional[str] = None) -> str:
        """Return a random study tip (optionally filtered by category)."""
        if category:
            tips = self.knowledge.tips_for(category)
            if not tips:
                category = None
        if not category:
            tips = self.knowledge.all_tips()
        if not tips:
            return "Tip: break your study into short, focused Pomodoro sessions."
        return random.choice(tips)

    def motivation_quote(self) -> str:
        """Return a random motivational quote."""
        quotes = self.knowledge.motivation_quotes()
        if not quotes:
            return (
                "“You don't have to be great to start, but you have to start "
                "to be great.” — Zig Ziglar"
            )
        return random.choice(quotes)

    # ------------------------------------------------------------------
    # Follow-up generation
    # ------------------------------------------------------------------
    def topic_follow_up(self) -> str:
        """Build the "Which topic?" question with a menu of options."""
        topic = self.memory.pending_context.get("topic") or self.memory.last_topic or ""
        options = self.knowledge.subject_topics(topic)
        if not options:
            return (
                "Sure! Which topic or subject would you like to work on? "
                "Just type it, e.g. *Algebra*, *Photosynthesis*, *Loops in Python*."
            )
        lines = ["Sure! Which topic would you like to explore?"]
        lines.append("")
        lines.extend(f"  {i + 1}. {item}" for i, item in enumerate(options[:6]))
        lines.append("")
        lines.append("(Type the topic name or its number — or 'none' to skip.)")
        return "\n".join(lines)

    def study_help_reply(self, topic: str) -> str:
        """Build a rich reply after the user picks a topic to study."""
        topics = self.knowledge.subject_topics(topic)
        info = self.knowledge.subject_info(topic)

        lines = [
            f"Great choice — **{topic}** is a fantastic area to master! 🎯",
            "",
        ]
        if topics:
            lines.append("Here are the key topics we can cover:")
            lines.extend(f"  • {item}" for item in topics[:8])
            lines.append("")
        difficulty = info.get("difficulty")
        if difficulty:
            lines.append(f"Difficulty rating: {difficulty}/5")
        lines.append(
            "I can give you **notes**, run a **quick quiz**, make **flashcards**, "
            "or **search the web** for resources. What would you like first?"
        )
        return "\n".join(lines)

    def search_intro(self, query: str) -> str:
        """Announce a web search that is about to happen."""
        return f"Searching the web for **“{query}”** — one moment… 🔍"

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def onboarding_question(self) -> str:
        """Ask for the user's name on first run."""
        return (
            "👋 Welcome to StudyBuddy! I noticed this is your first time here.\n"
            "Before we dive in — **what's your name?**"
        )

    def welcome_back(self, last_seen: str) -> str:
        """Greet a returning user warmly."""
        return (
            f"{utils.greeting_for_hour(utils.hour_of_day())}, {self.memory.user_name}! 👋\n"
            f"Welcome back — great to see you again. Last time you were here: {last_seen or 'recently'}.\n"
            "How can I help you today? Study tips, a schedule, a quiz, or something else?"
        )

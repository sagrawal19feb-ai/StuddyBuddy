"""
chatbot.py — The StudyBuddy chat engine (orchestrator)
=========================================================

This is the brain of the application. It owns every component — knowledge
base, storage, memory, intent detector, response generator, search engine,
quiz/flashcard/scheduler/planner engines and the UI — and routes each user
message to the right module.

Conversation states
-------------------
``idle``                     — normal chat
``onboarding``               — first-run interview (name, class, subjects)
``topic_choice``             — waiting for the user to pick a study topic
``study_help_action``        — waiting for what to do with the chosen topic
``quiz`` / ``flashcards``    — active wizard (runs to completion)

Wizards (quiz, flashcards, scheduler, daily planner, settings, onboarding)
run inside :meth:`handle_message` using the injected UI to ask questions, so
the caller simply feeds each user message in.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import utils
from ai_engine import ResponseGenerator
from calculator import MathSolver
from config import Config
from flashcards import Flashcard, FlashcardEngine, FlashcardReport
from intents import IntentDetector
from knowledge import KnowledgeBase
from logger import get_logger, log_chat
from memory import ConversationMemory
from planner import DailyPlanner
from quiz_engine import QuizEngine, QuizReport
from scheduler import StudyPlanner
from search_engine import SearchEngine
from storage import StorageManager
from ui import ConsoleUI
from validation import canonical_subjects, normalise_class, resolve_subjects

log = get_logger("chatbot")

# Slash-command aliases handled without intent detection.
COMMAND_ALIASES: dict[str, str] = {
    "quit": "quit", "exit": "quit", "q": "quit", "bye": "quit",
    "help": "help", "menu": "help",
    "progress": "progress", "stats": "progress",
    "profile": "profile", "whoami": "profile",
    "settings": "settings", "prefs": "settings",
    "history": "history", "logs": "history",
    "search": "search",
    "clear": "clear", "cls": "clear",
}

EXIT_PHRASES = {"quit", "exit", "bye", "goodbye", "see you", "good night", "see ya"}

# Matches bare "open 2", "open top" etc. (also handled as /open commands).
_RE_OPEN = re.compile(r"^open\s+(\d+|top)$")

# Question starters stripped before checking whether a question has a real
# topic ("what is X" -> "X"). Used by _is_topicless_question. Longest phrases
# come first so "what is" wins over "what".
_QUESTION_STARTER = re.compile(
    r"^(what is|what are|what was|what were|what does|what do|what did|"
    r"whats|why is|why do|why does|how is|how are|how do|who is|who was|"
    r"when is|when was|where is|where are|which is|"
    r"what|why|how|who|when|where|which|is|are|was|were|does|do|did|"
    r"can|could)\b\s*",
    re.I,
)

# Words that make a question topic-less: after removing the starter there is
# no real subject to look up ("what's new", "what is this", "what is going
# on") — a web search would be pointless. Includes leftover auxiliary verbs
# ("is", "are") that survive the starter strip.
_TOPICLESS_WORDS = frozenset({
    "new", "this", "that", "it", "there", "here", "up", "on", "going",
    "happening", "happen", "next", "else", "something", "anything",
    "about", "more", "thing", "things", "one", "now", "today", "then",
    "is", "are", "was", "were", "do", "does", "did", "am", "be", "been",
    "will", "would", "should", "could", "can", "have", "has", "had",
})

# Off-topic phrases that would distract the user from studying. When one of
# these appears in a chatty message the bot does NOT engage — it gently
# redirects back to study. Deliberately phrase-specific ("cricket score",
# not "cricket") so legitimate study topics like "help with cricket rules"
# still work through the study_help flow.
_DISTRACTION_PHRASES = frozenset([
    # Multi-word phrases ONLY. Single words live in _DISTRACTION_WORDS and
    # go through proportion-aware detection, so a rich study question like
    # "what causes weather patterns" is never flagged just for containing
    # the word "weather".
    "recommend a movie", "what is the weather", "weather today",
    "favourite movie", "favorite movie", "favourite colour", "favorite color",
    "favourite food", "favorite food", "favourite song", "favorite song",
    "horoscope today", "palm reading", "daily horoscope",
    "dank memes", "celebrity news", "korean drama",
    "live score", "match score", "cricket score", "ipl score", "world cup score",
    "tell me a story", "story for me", "bedtime story", "tell me a joke",
    "sing for me", "sing a song", "dance for me", "play a song", "play music",
    "play some music", "play me a song", "put on some music", "play a song for me",
    "play a game", "play a game with me", "lets play a game", "let us play a game",
    "lets play", "play games", "playing games", "play fortnite", "play pubg",
    "play minecraft", "free fire", "youtube shorts", "music video",
    "netflix and chill", "prime video", "hotstar",
])

# Phrases that CONTAIN a distraction word but are actually legitimate study
# topics ("game theory", "weather patterns", "film studies"). When one of
# these appears, the message is study content — never a distraction.
_STUDY_PHRASES = frozenset([
    "game theory", "game design", "game development", "game programming",
    "music theory", "music history", "sound waves", "music production",
    "weather patterns", "weather and climate", "weather forecasting",
    "climate change", "climate science", "atmospheric science",
    "film studies", "movie analysis", "film analysis", "cinema history",
    "sports science", "sports medicine", "sports psychology",
    "story writing", "storytelling", "drama in literature",
    "literature", "poetry", "theatre studies", "theater studies",
    "media studies", "journalism",
])

# Question/stop words not worth counting when judging distraction weight.
_DISTRACTION_STOPWORDS = frozenset({
    "what", "why", "how", "who", "when", "where", "which", "does", "is",
    "are", "was", "were", "the", "and", "for", "with", "from", "into",
    "this", "that", "your", "you", "can", "could", "would", "should",
    "do", "did", "will", "have", "has", "had", "not", "but", "about",
    "its", "it", "me", "my", "i", "a", "an", "of", "to", "on", "in",
})


class StudyBuddyChatbot:
    """Full conversational engine for StudyBuddy."""

    def __init__(self, config: Config, ui: ConsoleUI) -> None:
        self.config = config
        self.ui = ui
        self.running = True
        self.state = "idle"

        # Core subsystems
        self.storage = StorageManager()
        self.knowledge = KnowledgeBase()
        self.memory = ConversationMemory(self.storage)
        self.memory.set_protected_phrases(self.knowledge.intent_definitions())
        self.intent_detector = IntentDetector(self.knowledge)
        self.generator = ResponseGenerator(self.knowledge, self.memory)

        # Feature engines
        self.search = SearchEngine(self.knowledge)
        self.quiz = QuizEngine(self.knowledge, self.storage)
        self.flashcards = FlashcardEngine(self.knowledge, self.storage)
        self.planner = StudyPlanner()
        self.daily = DailyPlanner()
        self.calculator = MathSolver()

        self._apply_settings_to_ui()
        self._refresh_weak_subjects()

    # ------------------------------------------------------------------
    # Startup helpers
    # ------------------------------------------------------------------
    def _apply_settings_to_ui(self) -> None:
        settings = self.storage.get_settings()
        self.ui.set_theme(str(settings.get("theme", "blue")))

    def _refresh_weak_subjects(self) -> None:
        """Sync stored weak subjects from quiz history accuracy."""
        detected = self.quiz.weak_subjects_from_history()
        if detected:
            self.storage.update_profile(weak_subjects=detected)

    # ------------------------------------------------------------------
    # Main entry: the interactive loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Start the conversational loop (blocking)."""
        self.start_session()
        while self.running:
            try:
                text = self.ui.ask("You")
                if self.ui.eof:
                    self._quit()          # stdin closed (piped input / demo ended)
                    break
                if not text:
                    continue
                self.handle_message(text)
            except KeyboardInterrupt:
                self._quit()
            except EOFError:
                self._quit()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def start_session(self) -> None:
        """Welcome the user (new or returning) and begin the session."""
        self.storage.register_first_visit()
        if not self.storage.is_returning_user():
            self.ui.print_logo()
            self.state = "onboarding"
            self._run_onboarding()
        else:
            self.ui.print_logo()
            profile = self.storage.get_profile()
            last_seen = profile.get("last_seen", "")
            message = self.generator.welcome_back(last_seen)
            # Pick up where the user left off — topic continuity.
            last_topic = profile.get("last_topic", "")
            if last_topic:
                message += f"\n\nWe were last working on **{last_topic}** — "
                message += "want to continue, or try something new?"
            # Context-aware recap line: streak + quiz average + goals.
            progress = self.storage.get_progress()
            streak = self.storage.streak_days()
            avg = float(progress.get("avg_quiz_score", 0))
            goals = len(progress.get("goals_completed", []))
            recaps = []
            if streak:
                recaps.append(f"🔥 {streak}-day streak")
            if avg:
                recaps.append(f"{avg:.1f}% avg quiz score")
            if goals:
                recaps.append(f"{goals} goal(s) done")
            if recaps:
                message += "\n\n" + " · ".join(recaps) + " — nice work so far!"
            self.ui.print_bot(message, "info")
            self._save_exchange("bot", message)
            self._show_startup_hint()

    def _show_startup_hint(self) -> None:
        hint = (
            "Try: `study tips`, `motivate me`, `make a schedule`, `quiz`, "
            "`flashcards`, or `search Python loops`. Type `help` for everything."
        )
        self.ui.console.print(hint, style=self.ui.style("muted"))
        if bool(self.storage.get_settings().get("notifications", True)):
            self._show_notification()
        self.ui.divider()

    def _show_notification(self) -> None:
        """Show a lightweight status reminder when notifications are on."""
        progress = self.storage.get_progress()
        hours = float(progress.get("total_study_hours", 0))
        goals = len(progress.get("goals_completed", []))
        quizzes = int(progress.get("quizzes_completed", 0))
        if hours or goals or quizzes:
            parts: list[str] = []
            if hours:
                parts.append(f"studied {hours:.1f}h")
            if goals:
                parts.append(f"completed {goals} goal(s)")
            if quizzes:
                parts.append(f"finished {quizzes} quiz(zes)")
            self.ui.console.print(
                f"🔔 Quick status: {', '.join(parts)} so far. Keep the streak alive!",
                style=self.ui.style("muted"),
            )

    def _quit(self) -> None:
        """Graceful shutdown."""
        self.running = False
        farewell = self.generator.generate("farewell")
        self.ui.print_bot(farewell[0], "info")
        self._save_exchange("user", "quit")
        self._save_exchange("bot", farewell[0])
        log.info("Session ended by user.")
        self.storage.save_all()

    # ------------------------------------------------------------------
    # Message routing
    # ------------------------------------------------------------------
    def handle_message(self, raw_text: str) -> str:
        """Process one user message and return the primary bot reply text."""
        text = raw_text.strip()
        if not text:
            return ""

        # 1. Slash commands & explicit exit phrases first.
        if text.startswith("/"):
            return self._handle_command(text[1:].strip())

        lowered = text.lower().strip()
        if lowered in EXIT_PHRASES or lowered.startswith("quit "):
            self._quit()
            return ""

        # 1b. Bare "open 2" / "open top" — open a previous search result
        #     without needing the slash prefix.
        open_match = _RE_OPEN.match(lowered)
        if open_match and self.search.last_results:
            self.ui.print_user(text)
            return self._handle_open(open_match.group(1) or "top")

        # 2. Cross-message conversation states (topic choice flows). Note that
        #    quiz/flashcard wizards run synchronously inside one message, so
        #    no separate state branch is needed for them.
        if self.state == "topic_choice":
            return self._handle_topic_choice(text)
        if self.state == "study_help_action":
            return self._handle_study_help_action(text)

        # 3. Normal intent detection.
        return self._handle_idle(text)

    # ------------------------------------------------------------------
    # Idle-state routing
    # ------------------------------------------------------------------
    def _handle_idle(self, text: str) -> str:
        self.ui.print_user(text)
        self._save_exchange("user", text)

        resolved = self.memory.resolve_reference(text)

        # 0. Arithmetic first: "what is 12*8" / "15% of 200" / "sqrt 144".
        solved = self.calculator.solve(resolved)
        if solved:
            reply, style = self.generator.generate("calculate", result=solved)
            self.ui.print_bot(reply, style)
            self._save_exchange("bot", f"Calculation: {solved}")
            return reply

        result = self.intent_detector.detect(resolved)
        self.memory.set_intent(result.name)

        # Focus guard: never engage with off-topic chatter. The guard checks
        # BOTH the raw message and the typo-corrected version, so a misspelled
        # distraction ("wheather", "moive") is caught too — otherwise a typo'd
        # "what is the wheather" would be answered from the web instead of
        # being redirected back to study.
        corrected = self.intent_detector.correct_typos(resolved)
        if result.name in {"fun", "smalltalk", "knowledge", "unknown", "distraction"}:
            if self._is_distraction(resolved) or self._is_distraction(corrected):
                self.memory.set_intent("distraction")
                return self._redirect_to_study()

        reply, style = self.generator.generate("thinking")
        self.ui.loading(reply, duration=0.7)

        log.debug("Intent=%s confidence=%.2f keyword=%r", result.name, result.confidence,
                  result.matched_keyword)

        handlers = {
            "greeting": self._handle_greeting,
            "farewell": self._handle_farewell,
            "thanks": self._handle_thanks,
            "help": self._handle_help,
            "capabilities": self._handle_capabilities,
            "identity": self._handle_identity,
            "smalltalk": self._handle_smalltalk,
            "study_tips": self._handle_study_tips,
            "motivation": self._handle_motivation,
            "study_help": lambda t: self._handle_study_help(t, result),
            "schedule": self._handle_schedule_wizard,
            "daily_plan": self._handle_daily_plan_wizard,
            "quiz": self._handle_quiz_wizard,
            "flashcards": self._handle_flashcard_wizard,
            "knowledge": self._handle_knowledge,
            "search": self._handle_search,
            "calculate": self._handle_calculate,
            "remember": self._handle_remember,
            "study_method": self._handle_study_method,
            "elaborate": self._handle_elaborate,
            "context": self._handle_context,
            "fun": self._handle_fun,
            "distraction": self._handle_distraction,
            "reset": self._handle_reset,
            "progress": self._handle_progress,
            "settings": self._handle_settings_wizard,
            "profile": self._handle_profile,
            "history": self._handle_history,
        }

        handler = handlers.get(result.name)
        if handler is None:
            return self._handle_unknown(text)

        try:
            return handler(text)
        except Exception as exc:  # never let one bad turn kill the session
            log.error("Handler %s crashed: %s", result.name, exc)
            self.state = "idle"   # recover from any interrupted wizard state
            self.memory.consume_pending()
            self.ui.print_error("Something went wrong processing that — please try again.")
            return ""

    # ------------------------------------------------------------------
    # Intent handlers (idle)
    # ------------------------------------------------------------------
    def _handle_greeting(self, text: str) -> str:
        hour = utils.hour_of_day()
        name = self.memory.user_name
        reply, style = self.generator.generate(
            "greeting", name=name, greeting=utils.greeting_for_hour(hour)
        )
        # Context awareness: a greeting right after studying a topic can
        # continue it naturally ("hey, still on Algebra?").
        if self.memory.last_topic and self.memory.turns_this_session > 1:
            reply += f"\nStill on **{self.memory.last_topic}**? "
            reply += "Say `quiz` or `flashcards` to keep the momentum!"
        self.ui.print_bot(reply, style)
        follow = self._occasional_followup()
        if follow:
            self.ui.console.print(follow, style=self.ui.style("muted"))
        self._save_exchange("bot", reply)
        return reply

    def _handle_farewell(self, text: str) -> str:
        self._quit()
        return ""

    def _handle_thanks(self, text: str) -> str:
        reply, style = self.generator.generate("thanks")
        self.ui.print_bot(reply, style)
        self._save_exchange("bot", reply)
        return reply

    def _handle_help(self, text: str) -> str:
        commands = [
            ("study tips", "Get a random, research-backed study tip"),
            ("motivate me", "Boost your mood with an encouraging quote"),
            ("help with <subject>", "Deep-dive into a subject/topic"),
            ("make a schedule", "Personalised weekly study timetable"),
            ("daily plan", "A balanced plan for today"),
            ("quiz", "MCQ quiz with weak-topic analysis"),
            ("flashcards", "Flashcard revision session"),
            ("search <topic>", "Search the web for study material"),
            ("<any question>", "If it's not in my knowledge, I'll answer it from the web automatically"),
            ("what is <calculation>", "Solve maths instantly (e.g. `what is 12*8`)"),
            ("remember that …", "Teach me a fact I'll recall later"),
            ("best way to study <subject>", "Get a study-method recommendation"),
            ("tell me more", "Dig deeper into the current topic (web search)"),
            ("what were we doing", "Context recap — last action, topics, weak areas"),
            ("progress", "Your study stats, streak & achievements"),
            ("settings", "Customise theme, name, preferences"),
            ("history", "Show today's conversation history"),
            ("profile", "Show your study profile"),
            ("/open <number>", "Open a search result in your browser"),
            ("/clear", "Clear the terminal"),
            ("/reset", "Erase all data and start fresh (asks for confirmation)"),
            ("quit", "End the session"),
        ]
        self.ui.print_help(commands)
        reply = "Here's everything I can do — just type a command or ask naturally! 😊"
        self._save_exchange("bot", reply)
        return reply

    def _handle_capabilities(self, text: str) -> str:
        features = [
            "🧠 Intent detection with typo correction",
            "🧮 Instant maths solver (e.g. `what is 15% of 200`)",
            "📚 170+ knowledge facts + auto-generated quiz questions",
            "🗓  Personalised weekly study schedules",
            "🌅 Daily routine planner",
            "📝 MCQ quizzes with weak-topic analysis",
            "🗃  Flashcards with spaced repetition",
            "🎯 Flashcards built from your missed quiz questions",
            "🧠 Study memory — teach me facts with `remember that …`",
            "🎯 Study-method recommendations per subject",
            "🔎 Answers anything outside my knowledge by extracting from the web",
            "🔍 Deeper dives with `tell me more` + live web search",
            "🧭 Context awareness — remembers what you were studying, recaps "
            "on demand, and nudges your weak areas",
            "🎯 Focus mode — off-topic chatter gets redirected back to studying",
            "📊 Progress tracking, streaks & achievements",
            "💾 Automatic chat history & daily logs",
            "🎨 Themed, professional console UI",
        ]
        self.ui.print_panel("\n".join(f"  {f}" for f in features),
                            title="✨ What I can do", style="info")
        self._save_exchange("bot", "capabilities")
        return "That's my toolkit — pick anything and let's get started!"

    def _handle_identity(self, text: str) -> str:
        reply, style = self.generator.generate("identity")
        self.ui.print_bot(reply, style)
        self._save_exchange("bot", reply)
        return reply

    def _handle_smalltalk(self, text: str) -> str:
        """Chatty "how are you?" — a brief warm ack, then back to study."""
        return self._redirect_to_study()

    def _handle_study_tips(self, text: str) -> str:
        category = self._tip_category(text)
        self.memory.record_action("study_tips", category or "")
        tip = self.generator.study_tip(category)
        self.ui.print_bot(f"💡 **Study Tip**\n{tip}", "success")
        more = self.ui.ask("Want another tip? (y/n)")
        if utils.is_yes(more):
            tip2 = self.generator.study_tip(category)
            self.ui.print_bot(f"💡 **Another one!**\n{tip2}", "success")
            self._save_exchange("bot", f"Study tip: {tip}\nStudy tip: {tip2}")
            return tip2
        self._save_exchange("bot", f"Study tip: {tip}")
        return tip

    def _handle_motivation(self, text: str) -> str:
        self.memory.record_action("motivation", "")
        quote = self.generator.motivation_quote()
        self.ui.print_bot(f"🌟 **Keep going, {self.memory.user_name}!**\n\n{quote}", "motivation")
        self._save_exchange("bot", f"Motivation: {quote}")
        return quote

    def _handle_study_help(self, text: str, result: Any) -> str:
        """Route 'I need help in <subject>' into a topic-choice flow."""
        topic = result.topic
        if topic:
            self.memory.set_topic(topic)
            self.memory.record_action("study_help", topic)
        self.memory.set_pending("topic_choice", topic=topic)
        self.state = "topic_choice"
        question = self.generator.topic_follow_up()
        self.ui.print_bot(question, "info")
        self._save_exchange("bot", question)
        return question

    def _handle_knowledge(self, text: str) -> str:
        # 1. Facts the user taught the bot take priority.
        custom = self.memory.find_custom_fact(text)
        if custom:
            answer = custom.get("fact", "")
            self.ui.print_bot(
                f"🧠 **From your study memory** — {custom.get('topic', '')}\n\n{answer}",
                "info",
            )
            self._save_exchange("bot", f"Custom fact: {answer}")
            return answer

        # 2. Topic-less questions ("what's new", "what is this", "what is
        #    going on") have nothing to look up — answer politely instead of
        #    firing a pointless web search.
        if self._is_topicless_question(text):
            return self._respond_topicless(text)

        # 3. Everything else: answer LIVE from the web with the extracted
        #    text shown inline. The offline knowledge base is only used as
        #    an emergency fallback when the web is unreachable.
        return self._answer_from_web(text)

    def _handle_search(self, text: str) -> str:
        return self._run_search(text)

    def _handle_fun(self, text: str) -> str:
        """Jokes/games/riddles — fun is a distraction, so redirect to study."""
        return self._redirect_to_study()

    def _handle_distraction(self, text: str) -> str:
        """Explicitly off-topic requests (weather, movies, gossip...)."""
        return self._redirect_to_study()

    def _handle_reset(self, text: str) -> str:
        """Erase all saved data and start fresh (with confirmation).

        Wipes profile, progress, settings and chat history, then runs the
        onboarding interview again so the user is fully set up. Also clears
        the in-session memory and the web-answer cache.
        """
        confirm = self.ui.ask(
            "⚠️ This will erase your profile, progress, settings and chat "
            "history permanently. Are you sure? (y/n)"
        )
        if not utils.is_yes(confirm):
            reply = "Okay — nothing was changed. Your data is safe! 🙂"
            self.ui.print_bot(reply, "info")
            self._save_exchange("bot", reply)
            return reply

        self.storage.reset_all()
        # Fresh session memory + empty caches.
        self.memory = ConversationMemory(self.storage)
        self.memory.set_protected_phrases(self.knowledge.intent_definitions())
        self.search.last_results = []
        self.search._answer_cache.clear()
        self.generator._last_used.clear()
        self.state = "idle"
        self.memory.consume_pending()

        self.ui.print_bot(
            "✅ **All data has been reset.** You're starting completely fresh — "
            "let's set up your profile again!", "success",
        )
        self._save_exchange("bot", "Reset complete")
        self._run_onboarding()
        log.info("User reset all data.")
        return "Reset complete"

    def _handle_calculate(self, text: str) -> str:
        """Answer an arithmetic query (also caught pre-intent in _handle_idle)."""
        solved = self.calculator.solve(text)
        if not solved:
            reply, style = self.generator.generate("calculate_none")
            self.ui.print_bot(reply, "warning")
            self._save_exchange("bot", reply)
            return reply
        reply, style = self.generator.generate("calculate", result=solved)
        self.ui.print_bot(reply, style)
        self._save_exchange("bot", f"Calculation: {solved}")
        return reply

    def _handle_remember(self, text: str) -> str:
        """Let the user teach the bot facts it recalls later.

        Examples
        --------
        "remember that mitochondria is the powerhouse of the cell"
        "what do you remember?"
        "forget mitochondria"
        """
        lowered = text.lower().strip()

        # 1. Listing stored facts.
        if any(phrase in lowered for phrase in ("what do you remember", "list your memories",
                                                "what facts do you know", "show my facts")):
            facts = self.memory.list_custom_facts()
            if not facts:
                reply = self.generator.generate("remember_empty")[0]
                self.ui.print_bot(reply, "info")
                self._save_exchange("bot", reply)
                return reply
            rows = [[f.get("topic", ""), f.get("fact", "")[:70]] for f in facts]
            self.ui.print_table(["Topic", "What I remember"], rows,
                                title="🧠 My Study Memory")
            reply = f"I'm remembering {len(facts)} fact(s) for you."
            self._save_exchange("bot", reply)
            return reply

        # 2. Forgetting a fact.
        if lowered.startswith("forget"):
            topic = lowered.replace("forget", "").replace("that", "").strip().title()
            if topic and self.memory.forget_custom_fact(topic):
                reply = self.generator.generate("remember_forgot")[0]
            else:
                reply = "I couldn't find that in my memory — but I've forgotten nothing else! 🙂"
            self.ui.print_bot(reply, "info")
            self._save_exchange("bot", reply)
            return reply

        # 3. Saving a new fact — only when the message actually starts with a
        #    save command, so "i remember studying this" never gets stored.
        save_start = ("please remember", "remember that", "remember this", "remember:",
                      "note that", "note:", "take note", "save this fact", "memorize this")
        if not lowered.startswith(save_start):
            reply = (
                "I can remember facts for you! Try: `remember that mitochondria is "
                "the powerhouse of the cell` — then ask me about it anytime. 🧠"
            )
            self.ui.print_bot(reply, "info")
            self._save_exchange("bot", reply)
            return reply

        body = re.sub(
            r"^(please\s+)?(remember|note|take note of|save this fact|memorize this for me)"
            r"(?:[:,]|\s+(?:that|this))?\s*",
            "", text, flags=re.IGNORECASE
        ).strip()
        # Reject junk bodies ("for later", "this", "that") so we never save
        # a meaningless fact.
        if (not body or len(body) < 5 or body.lower() in
                {"for later", "this", "that", "it", "for me", "please", "down"}):
            reply = "What should I remember? Try: `remember that mitochondria is the powerhouse of the cell`"
            self.ui.print_bot(reply, "info")
            return reply

        # Extract a topic from "<something> is <fact>" when possible.
        topic = body
        match = re.match(r"^(.+?)\s+(?:is|are|was|were|means)\s+(.+)$", body, re.IGNORECASE)
        if match:
            topic = match.group(1).strip().title()
        self.memory.add_custom_fact(topic, body)
        reply = self.generator.generate("remember_saved", fact=body)[0]
        self.ui.print_bot(reply, "success")
        self.storage.add_covered_topic(f"Remembered: {topic}")
        self._save_exchange("bot", f"Saved fact ({topic}): {body}")
        return reply

    def _handle_study_method(self, text: str) -> str:
        """Recommend a study technique, ideally tailored to the topic."""
        topic = self.intent_detector.detect_topic(text)
        method = ""
        if topic:
            method = self.knowledge.subject_info(topic).get("method", "")
            self.memory.set_topic(topic)
            self.memory.record_action("study_method", topic)
        if method:
            reply = self.generator.generate(
                "study_method", topic=topic, method=method
            )[0]
        else:
            reply = self.generator.generate("study_method_generic")[0]
        self.ui.print_bot(reply, "success")
        self._save_exchange("bot", f"Study method recommendation: {reply}")
        return reply

    def _handle_elaborate(self, text: str) -> str:
        """Deep-dive: search the web for more detail on the current topic.

        Works with the last discussed topic ("tell me more" right after a
        knowledge answer) or an explicit topic in the message ("tell me more
        about gravity").
        """
        topic = self.intent_detector.detect_topic(text)
        if not topic:
            topic = self.memory.last_topic
        if topic:
            self.memory.set_topic(topic)
            self.ui.print_bot(f"Let me find deeper details on **{topic}** for you. 🔍", "info")
            return self._run_search(f"search {topic} detailed explanation")

        reply = (
            "Happy to elaborate! Which topic? You can say `tell me more about "
            "<topic>` — e.g. `tell me more about photosynthesis`."
        )
        self.ui.print_bot(reply, "info")
        self._save_exchange("bot", reply)
        return reply

    def _handle_context(self, text: str) -> str:
        """Recap the conversation: what we did, topics covered, and where
        the user stands — a context-aware "where were we?" answer.

        Combines in-session memory (last action, recent topics) with
        persistent progress (streak, quiz average, goals).
        """
        lines = ["🧭 **Here's where we stand:**", ""]

        # 1. This session's activity.
        topics = self.memory.recent_topics_list()
        last = self.memory.last_action
        if last and last.get("intent"):
            intent = last["intent"]
            detail = last.get("detail") or ""
            label = {
                "quiz": "a **quiz**" + (f" on {detail}" if detail else ""),
                "flashcards": "**flashcards**" + (f" on {detail}" if detail else ""),
                "knowledge": f"read about **{detail or 'a topic'}**",
                "web_answer": f"looked up **{detail}** from the web",
                "search": f"searched for **{detail}**",
                "study_help": f"started studying **{detail or 'a topic'}**",
                "study_method": f"got a study method for **{detail}**",
                "schedule": f"made a schedule ({detail})",
                "daily_plan": "planned your day",
                "study_tips": "read study tips",
                "motivation": "got a motivation boost",
            }.get(intent)
            if label:
                lines.append(f"• Last thing we did: you {label}.")
        else:
            lines.append("• No activity yet this session.")

        if topics:
            lines.append(f"• Topics we've covered this session: **{', '.join(topics)}**.")
        else:
            profile_topic = self.storage.get_profile().get("last_topic", "")
            if profile_topic:
                lines.append(f"• Last session you were on **{profile_topic}**.")

        # 2. Persistent progress.
        progress = self.storage.get_progress()
        streak = self.storage.streak_days()
        avg = float(progress.get("avg_quiz_score", 0))
        quizzes = int(progress.get("quizzes_completed", 0))
        if streak:
            lines.append(f"• 🔥 Streak: **{streak} day(s)**.")
        if quizzes:
            lines.append(f"• Quiz average: **{avg:.1f}%** across {quizzes} quiz(zes).")

        # 3. Smart suggestion based on where the user left off.
        weak = self.memory.weak_subjects
        suggestion = "Say `quiz`, `flashcards`, or `search <topic>` to keep going. 🎯"
        if weak:
            suggestion = f"Your weak area is **{weak[0]}** — want to `quiz` or `search` on it?"
        lines.append("")
        lines.append(suggestion)

        reply = "\n".join(lines)
        self.ui.print_bot(reply, "info")
        self.memory.record_action("context", "")
        self._save_exchange("bot", reply)
        return reply

    def _handle_unknown(self, text: str) -> str:
        """Fallback: custom facts → knowledge base → search offer → redirect.

        Every path keeps the user studying: facts are answered, real
        questions can be searched, and everything else is redirected
        instead of answered with filler.
        """
        # 1. Custom facts the user taught the bot.
        custom = self.memory.find_custom_fact(text)
        if custom:
            answer = custom.get("fact", "")
            self.ui.print_bot(
                f"🧠 **From your study memory** — {custom.get('topic', '')}\n\n{answer}",
                "info",
            )
            self._save_exchange("bot", f"Custom fact: {answer}")
            return answer

        # 2. Genuine study-style question → answer LIVE from the web (with
        #    the extracted text inline). The offline knowledge base is only
        #    used as an emergency fallback when the web is unreachable.
        if re.match(r"^(what|why|how|who|when|where|which|can|could|is|are|do|does)\b",
                    text.lower()):
            if self._is_topicless_question(text):
                return self._respond_topicless(text)
            return self._answer_from_web(text)

        # 3. If the message looks like a real topic the user wants to know
        #    about (e.g. "sigma", "sigma male"), answer it from the web —
        #    don't lecture about focus. Only gibberish and off-topic chatter
        #    get the study redirect.
        if self._looks_like_real_query(text):
            return self._answer_from_web(text)

        # 4. Gibberish / off-topic — refocus on study.
        return self._redirect_to_study()

    # ------------------------------------------------------------------
    # Search flow
    # ------------------------------------------------------------------
    def _run_search(self, text: str) -> str:
        query = self.search.build_query(text)
        # Context awareness: a bare "search" / "search more" with no topic
        # should continue the current topic, not search for the generic
        # fallback "study resources".
        if query == "study resources" and self.memory.last_topic:
            query = self.memory.last_topic
        self.memory.add_recent_search(query)
        self.memory.record_action("search", query)
        self.storage.add_covered_topic(f"Searched: {query}")

        intro = self.generator.search_intro(query)
        self.ui.print_bot(intro, "info")
        self.ui.loading("Searching educational sites", duration=1.0)

        results = self.search.search(query)
        if not results:
            message = "No results could be fetched right now — please check your connection."
            self.ui.print_bot(message, "warning")
            return message

        formatted = self.search.format_results(results, query)
        self.ui.print_bot(formatted, "list")
        self._save_exchange("bot", f"Search results for: {query}")

        answer = self.ui.ask("Open the top result in your browser? (y/n)")
        if utils.is_yes(answer) and results:
            opened = self.search.open_in_browser(results[0].url)
            if opened:
                self.ui.console.print("Opened in your default browser 🌐",
                                      style=self.ui.style("success"))
        return f"Search complete: {query}"

    def _answer_from_web(self, text: str) -> str:
        """Answer a question LIVE from the web, showing the extracted text.

        Flow:
          1. Announce the lookup, search for the query.
          2. Fetch the top pages and pull the best paragraphs
             (see :meth:`SearchEngine.extract_brief`).
          3. Present the extracted paragraphs inline — a real answer, not
             just a list of links — with the source title and URL.
          4. If the web is unreachable, fall back to the offline knowledge
             base, then to the results list.
        """
        query = self.search.build_query(text)
        # Context awareness: bare queries continue the current topic.
        if query == "study resources" and self.memory.last_topic:
            query = self.memory.last_topic
        self.memory.add_recent_search(query)
        self.memory.record_action("web_answer", query)
        self.storage.add_covered_topic(f"Searched: {query}")

        self.ui.print_bot("Let me look that up online for you… 🔍", "info")
        self.ui.loading("Fetching the answer from the web", duration=1.0)

        results = self.search.search(query)
        if not results:
            # Offline fallback: use the local knowledge base if it covers it.
            return self._offline_fallback_answer(text, query)

        extracted = self.search.extract_brief(query, results, max_passages=2)
        if extracted and extracted.get("passages"):
            title = extracted.get("title", "")
            url = extracted.get("url", "")
            passages = extracted["passages"]
            lines = [f"📡 **From the web** — {title}", ""]
            for i, passage in enumerate(passages, start=1):
                lines.append(passage)
                lines.append("")
            lines.append(f"Source: {url}")
            self.ui.print_bot("\n".join(lines), "info")
            self._save_exchange("bot", f"Web answer for {query}: {passages[0][:200]}")
            # Offer the full results list as a follow-up.
            more = self.ui.ask("Want the full list of results? (y/n)")
            if utils.is_yes(more):
                formatted = self.search.format_results(results, query)
                self.ui.print_bot(formatted, "list")
            return f"Web answer: {passages[0][:120]}"

        # Nothing extractable — show the results list instead.
        self.ui.print_bot(
            "I couldn't extract a clean paragraph from those pages, but here are "
            "the top results:", "info",
        )
        formatted = self.search.format_results(results, query)
        self.ui.print_bot(formatted, "list")
        self._save_exchange("bot", f"Search results for: {query}")
        return f"Search complete: {query}"

    def _offline_fallback_answer(self, text: str, query: str) -> str:
        """Emergency answer when the web is unreachable — local knowledge base.

        Shows a clear note that the answer is from the stored local database
        (offline mode) rather than live from the web.
        """
        fact = self.knowledge.find_knowledge(text)
        if fact:
            answer = fact.get("answer") or fact.get("fact", "")
            category = fact.get("category", "General")
            self.memory.set_topic(category)
            self.memory.record_action("knowledge", category)
            self.ui.print_bot(
                f"📖 **{category}** (offline answer — web unavailable)\n\n{answer}",
                "info",
            )
            self._save_exchange("bot", f"Offline knowledge ({category}): {answer}")
            return answer
        message = "I couldn't reach the web right now — please check your connection."
        self.ui.print_bot(message, "warning")
        self._save_exchange("bot", message)
        return message

    def _handle_command(self, command: str) -> str:
        """Handle slash-commands like /help, /open 3, /search python loops."""
        parts = command.split(maxsplit=1)
        name = parts[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else ""

        mapping = {
            "help": self._handle_help, "menu": self._handle_help,
            "progress": self._handle_progress, "stats": self._handle_progress,
            "profile": self._handle_profile, "whoami": self._handle_profile,
            "settings": self._handle_settings_wizard, "prefs": self._handle_settings_wizard,
            "history": self._handle_history, "logs": self._handle_history,
            "clear": self._handle_clear, "cls": self._handle_clear,
            "reset": self._handle_reset, "wipe": self._handle_reset,
        }
        if name in mapping:
            return mapping[name](command)

        if name == "quit":
            self._quit()
            return ""
        if name == "search":
            if not argument:
                reply = "Usage: /search <topic>  — e.g. /search biology notes"
                self.ui.print_bot(reply, "warning")
                return reply
            return self._run_search(argument)
        if name == "open":
            return self._handle_open(argument)
        if name == "theme":
            return self._handle_theme_command(argument)

        reply = f"Unknown command `/{name}`. Type `help` to see all commands."
        self.ui.print_bot(reply, "warning")
        return reply

    def _handle_open(self, argument: str) -> str:
        """Open a search result by number ('open 3') or the top one."""
        if not self.search.last_results:
            reply = "No recent search results to open. Try `search <topic>` first."
            self.ui.print_bot(reply, "warning")
            return reply
        if argument.lower() == "top" or not argument:
            target = 1
        elif argument.isdigit():
            target = int(argument)
        else:
            reply = "Usage: open <number> (e.g. `open 2`) or `open top`."
            self.ui.print_bot(reply, "warning")
            return reply
        opened = self.search.open_result(target)
        reply = (
            f"Opening result #{target} in your browser… 🌐"
            if opened else "Sorry, I couldn't open the browser."
        )
        self.ui.print_bot(reply, "success" if opened else "error")
        return reply

    def _handle_theme_command(self, argument: str) -> str:
        from ui import THEMES
        if argument and argument in THEMES:
            self.storage.update_settings(theme=argument)
            self._apply_settings_to_ui()
            reply = f"Theme switched to **{argument}** 🎨"
            self.ui.print_bot(reply, "success")
            return reply
        reply = f"Available themes: {', '.join(THEMES)}. Try `/theme blue`."
        self.ui.print_bot(reply, "info")
        return reply

    def _handle_clear(self, text: str) -> str:
        self.ui.clear()
        self.ui.print_logo()
        return ""

    # ------------------------------------------------------------------
    # Progress / profile / history
    # ------------------------------------------------------------------
    def _handle_progress(self, text: str) -> str:
        progress = self.storage.get_progress()
        total_hours = float(progress.get("total_study_hours", 0))
        quizzes = int(progress.get("quizzes_completed", 0))
        avg = float(progress.get("avg_quiz_score", 0))
        topics = list(progress.get("topics_covered", []))
        goals = list(progress.get("goals_completed", []))
        flashcards = int(progress.get("flashcards_reviewed", 0))

        goal_hours = 50.0  # e.g. a "50 hours this term" goal
        self.ui.print_panel(
            f"**Study Dashboard** for {self.memory.user_name}", title="📊 Progress",
            style="info",
        )
        self.ui.print_progress_bar(utils.progress_percent(total_hours, goal_hours),
                                   label="Total study hours")
        self.ui.print_progress_bar(min(100.0, avg), label="Average quiz score")
        self.ui.print_progress_bar(utils.progress_percent(len(topics), 20),
                                   label="Topics covered")

        streak = self.storage.streak_days()
        table_rows = [
            ["Study hours", f"{total_hours:.1f} h"],
            ["Sessions", str(progress.get("sessions", 0))],
            ["Quizzes completed", str(quizzes)],
            ["Average quiz score", f"{avg:.1f}%"],
            ["Flashcards reviewed", str(flashcards)],
            ["Goals completed", str(len(goals))],
            ["🔥 Study streak", f"{streak} day(s)"],
        ]
        self.ui.print_table(["Metric", "Value"], table_rows, title="Statistics")
        if streak >= 3:
            self.ui.console.print(
                f"🔥 {streak}-day streak — you're building a great habit, "
                f"{self.memory.user_name}! Keep it going.",
                style=self.ui.style("success"),
            )

        if topics:
            self.ui.print_panel(
                "**Recently covered:** " + ", ".join(topics[-6:]), title="🧠 Topics",
                style="success",
            )
        if not total_hours and not quizzes:
            self.ui.print_bot(
                "You haven't logged any study time yet. Try a **schedule**, **quiz** "
                "or **flashcards** — I'll track everything automatically! 🚀",
                "info",
            )
        reply = f"You've studied {total_hours:.1f} hours across {progress.get('sessions', 0)} sessions."
        self._save_exchange("bot", reply)
        return reply

    def _handle_profile(self, text: str) -> str:
        profile = self.storage.get_profile()
        rows = [
            ["Name", profile.get("name") or "—"],
            ["Class", profile.get("class") or "—"],
            ["Favourite subjects", ", ".join(profile.get("favourite_subjects", [])) or "—"],
            ["Weak subjects", ", ".join(profile.get("weak_subjects", [])) or "—"],
            ["First seen", profile.get("first_seen") or "—"],
            ["Last seen", profile.get("last_seen") or "—"],
            ["Recent searches", ", ".join(profile.get("recent_searches", [])[:3]) or "—"],
        ]
        self.ui.print_table(["Field", "Value"], rows, title="👤 My Profile")
        reply = "That's everything I know about you — type `settings` to change it."
        self._save_exchange("bot", reply)
        return reply

    def _handle_history(self, text: str) -> str:
        history = self.storage.load_chat_history()
        if not history:
            reply = "No conversation history for today yet."
            self.ui.print_bot(reply, "info")
            return reply
        rows = [[entry.get("timestamp", ""), entry.get("role", ""), entry.get("message", "")[:60]]
                for entry in history[-12:]]
        self.ui.print_table(["Time", "Role", "Message"], rows,
                            title=f"Today's Conversation ({utils.date_str()})")
        files = self.storage.list_chat_history_files()
        if files:
            self.ui.console.print(f"Stored daily logs: {', '.join(files)}",
                                  style=self.ui.style("muted"))
        reply = f"Showing the last {len(rows)} messages."
        self._save_exchange("bot", reply)
        return reply

    # ------------------------------------------------------------------
    # Topic-choice flow (state machine)
    # ------------------------------------------------------------------
    def _handle_topic_choice(self, text: str) -> str:
        self.ui.print_user(text)
        self._save_exchange("user", text)

        if text.lower() in {"none", "skip", "back", "no"}:
            self.state = "idle"
            self.memory.consume_pending()
            reply = self.generator.generate("fallback")[0]
            self.ui.print_bot(reply, "info")
            return reply

        context_topic = self.memory.pending_context.get("topic") or ""
        topic = self._resolve_topic_choice(text, context_topic)

        if not topic:
            self.ui.print_bot(
                "Hmm, I couldn't quite catch that topic. Try the number from the list "
                "or type the name directly — e.g. *Algebra*.", "warning",
            )
            return ""

        self.memory.set_topic(topic)
        self.storage.add_covered_topic(topic)
        self.state = "study_help_action"
        self.memory.set_pending("study_help_action", topic=topic)

        reply = self.generator.study_help_reply(topic)
        self.ui.print_bot(reply, "info")
        self._save_exchange("bot", f"Study help for: {topic}")
        return reply

    def _resolve_topic_choice(self, text: str, context_topic: str) -> str:
        """Map a number or name to a concrete topic."""
        options = self.knowledge.subject_topics(context_topic) if context_topic else []
        if text.strip().isdigit() and options:
            index = int(text.strip()) - 1
            if 0 <= index < len(options):
                return options[index]
        topic = self.intent_detector.detect_topic(text)
        if topic:
            return topic
        # Accept free text as a topic directly.
        return text.strip().title()

    def _handle_study_help_action(self, text: str) -> str:
        self.ui.print_user(text)
        self._save_exchange("user", text)
        topic = self.memory.pending_context.get("topic") or self.memory.last_topic or ""
        choice = text.strip().lower()

        self.state = "idle"
        self.memory.consume_pending()

        actions = {
            "1": "notes", "notes": "notes", "overview": "notes",
            "2": "quiz", "quiz": "quiz",
            "3": "flashcards", "cards": "flashcards",
            "4": "search", "search": "search", "web": "search",
            "5": "tips", "tips": "tips", "tip": "tips",
        }

        action = actions.get(choice)
        if not action:
            # Maybe they typed an actual subject name.
            detected = self.intent_detector.detect_topic(text)
            if detected:
                self.memory.set_topic(detected)
                self.state = "study_help_action"
                self.memory.set_pending("study_help_action", topic=detected)
                reply = self.generator.study_help_reply(detected)
                self.ui.print_bot(reply, "info")
                self._save_exchange("bot", f"Study help for: {detected}")
                return reply
            self.ui.print_bot(
                "Sure! You can say **notes**, **quiz**, **flashcards**, **search**, "
                "or **tips**. Which one?", "info",
            )
            return ""

        return self._dispatch_topic_action(action, topic)

    def _dispatch_topic_action(self, action: str, topic: str) -> str:
        """Run the chosen study action for the active topic."""
        if action == "notes":
            fact = self.knowledge.find_knowledge(topic)
            if fact:
                answer = fact.get("answer") or fact.get("fact", "")
                self.ui.print_bot(f"📖 {answer}", "info")
                self._save_exchange("bot", f"Notes on {topic}: {answer}")
                return answer
            return self._run_search(f"search notes for {topic}")
        if action == "quiz":
            return self._run_quiz(subject=topic)
        if action == "flashcards":
            return self._run_flashcards(subject=topic)
        if action == "search":
            return self._run_search(f"search {topic}")
        if action == "tips":
            tip = self.generator.study_tip()
            self.ui.print_bot(f"💡 **Study Tip for {topic}**\n{tip}", "success")
            self._save_exchange("bot", f"Study tip: {tip}")
            return tip
        return ""

    # ------------------------------------------------------------------
    # Wizards: quiz, flashcards, schedule, daily plan, settings, onboarding
    # ------------------------------------------------------------------
    def _handle_quiz_wizard(self, text: str) -> str:
        subject = self._choose_subject()
        if not subject:
            return self._wizard_cancelled()
        difficulty = self._choose_difficulty()
        if not difficulty:
            return self._wizard_cancelled()
        return self._run_quiz(subject=subject, difficulty=difficulty)

    def _choose_subject(self) -> str:
        subjects = self.quiz.available_subjects() or ["General"]
        options = {s: "Practice questions" for s in subjects}
        options = {"All Subjects": "Mix of everything"} | options
        choice = self.ui.print_menu("Choose a quiz subject:", options)
        return choice or ""

    def _choose_difficulty(self) -> str:
        options = {"easy": "Warm-up questions", "medium": "Balanced challenge",
                   "hard": "Tough exam-style questions"}
        choice = self.ui.print_menu("Choose difficulty:", options)
        return choice or ""

    def _run_quiz(self, subject: str = "All Subjects",
                  difficulty: str = "all") -> str:
        self.state = "quiz"
        self.memory.set_pending("quiz")
        self.memory.record_action("quiz", subject)
        count = int(self.storage.get_settings().get("quiz_question_count", 5))
        self.ui.loading("Preparing questions", duration=0.8)

        steps = self.quiz.run(subject, difficulty, count)
        try:
            step = next(steps)
        except StopIteration:
            self.state = "idle"
            return ""

        first = True
        while True:
            if isinstance(step, QuizReport):
                self.state = "idle"
                self.memory.consume_pending()
                if step.message:
                    self.ui.print_bot(step.message, "warning")
                    return step.message
                self.ui.print_bot(step.render(), "success" if step.percent >= 60 else "warning")
                self._save_exchange("bot", f"Quiz report: {step.correct}/{step.total}")
                self.memory.record_action("quiz", subject)
                # Smarter follow-up: turn wrong answers into flashcards.
                if step.missed:
                    offer = self.ui.ask(
                        f"Want me to turn the {len(step.missed)} you missed into flashcards? (y/n)"
                    )
                    if utils.is_yes(offer):
                        cards = [
                            Flashcard(question=item["question"], answer=item["answer"],
                                      topic=subject, source="quiz-missed")
                            for item in step.missed
                        ]
                        self._run_flashcards(cards=cards)
                # Context awareness: if the quiz exposed weak topics (but the
                # user got no specific misses to convert), offer material.
                elif step.weak_topics:
                    offer = self.ui.ask(
                        "Want me to search the web for revision material on "
                        f"**{', '.join(step.weak_topics[:3])}**? (y/n)"
                    )
                    if utils.is_yes(offer):
                        return self._run_search(
                            f"search revision notes for {step.weak_topics[0]}"
                        )
                return f"Quiz done: {step.correct}/{step.total}"
            if first:
                self.ui.print_bot("Let's go! Answer each question by typing the letter. 🎯", "info")
                first = False
            self.ui.print_bot(step.prompt(), "list")
            answer = self.ui.ask("Your answer")
            if self.ui.eof:  # stdin closed — end the quiz gracefully
                self.state = "idle"
                self.memory.consume_pending()
                return ""
            if answer.lower() in {"quit", "q", "exit", "stop"}:
                self.state = "idle"
                self.memory.consume_pending()
                self.ui.print_bot("Quiz stopped — no pressure, we can resume anytime! 🙌", "info")
                return ""
            if not answer:
                continue
            try:
                step = steps.send(answer)
            except StopIteration:
                self.state = "idle"
                self.memory.consume_pending()
                break

        self.state = "idle"
        return ""

    def _handle_flashcard_wizard(self, text: str) -> str:
        return self._run_flashcards()

    def _run_flashcards(self, subject: Optional[str] = None,
                        cards: Optional[list[Flashcard]] = None) -> str:
        """Run a flashcard session.

        Parameters
        ----------
        subject : str | None
            Subject/topic to build cards for (default: mixed deck).
        cards : list[Flashcard] | None
            Explicit deck (e.g. cards built from missed quiz questions).
            When given, ``subject`` is ignored.
        """
        self.state = "flashcards"
        self.memory.set_pending("flashcards")
        self.memory.record_action("flashcards", subject or ("missed-question deck" if cards else "mixed"))
        limit = max(5, int(self.storage.get_settings().get("quiz_question_count", 5)))
        if cards:
            steps = self.flashcards.run_cards(cards)
        else:
            steps = self.flashcards.run(subject, limit)
        try:
            step = next(steps)
        except StopIteration:
            self.state = "idle"
            return ""

        if cards:
            self.ui.print_bot(
                "Here are **flashcards from your missed questions** — this time, "
                "let's nail them! Grade each: **k** = knew it, **m** = missed it. 🎯",
                "info",
            )
        else:
            self.ui.print_bot(
                "Flashcard time! Read the question, try to recall the answer, "
                "then grade yourself: **k** = knew it, **m** = missed it. 🗃️\n"
                "A missed card comes back **up to 3 times** — after that it's "
                "flagged for practice. Type `quit` anytime to end.", "info",
            )
        while True:
            if isinstance(step, FlashcardReport):
                self.state = "idle"
                self.memory.consume_pending()
                self.ui.print_bot(step.render(), "success")
                self._save_exchange("bot", f"Flashcard report: {step.reviewed} reviewed")
                return f"Flashcards done: {step.reviewed} reviewed"
            # Friendly cue when a previously-missed card comes back.
            if step.misses > 0:
                self.ui.console.print(
                    f"🔄 This one again (missed {step.misses}x) — give it a "
                    f"proper read before grading.",
                    style=self.ui.style("muted"),
                )
            self.ui.print_bot(step.reveal(), "list")
            grade = self.ui.ask("Grade it (k = knew / m = missed)")
            if self.ui.eof:  # stdin closed — end the session gracefully
                self.state = "idle"
                self.memory.consume_pending()
                return ""
            if not grade:
                continue
            if grade.lower() in {"quit", "q", "exit", "stop"}:
                self.state = "idle"
                self.memory.consume_pending()
                self.ui.print_bot("Flashcard session stopped — see you next time! 👋", "info")
                return ""
            if grade.lower() in {"m", "missed", "no", "n"}:
                self.ui.console.print(
                    "🔄 Noted — this card returns for another try (max 3).",
                    style=self.ui.style("muted"),
                )
            try:
                step = steps.send(grade.lower())
            except StopIteration:
                self.state = "idle"
                self.memory.consume_pending()
                break
        self.state = "idle"
        return ""

    def _handle_schedule_wizard(self, text: str) -> str:
        self.ui.print_bot(
            "Let's build your personalised study schedule! 🗓️ I'll ask a few "
            "quick questions.", "info",
        )

        hours_raw = self.ui.ask("How many hours can you study per day? (e.g. 4)")
        try:
            hours = float(hours_raw or 4)
            hours = max(1.0, min(12.0, hours))
        except ValueError:
            self.ui.print_bot("No problem — I'll assume **4 hours** per day.", "info")
            hours = 4.0

        subjects = []
        while True:
            subjects_raw = self.ui.ask("Which subjects? (comma-separated, e.g. Maths, Physics)")
            if not subjects_raw.strip():
                break
            subjects, invalid = resolve_subjects(subjects_raw, self.knowledge)
            if subjects:
                if invalid:
                    self.ui.print_bot(
                        f"Ignored **{', '.join(invalid)}** — not subjects I know. "
                        f"Using: {', '.join(subjects)}.", "info",
                    )
                break
            self.ui.print_bot(
                f"I don't recognise **{', '.join(invalid)}** as subjects. "
                f"Valid examples: Maths, Physics, Chemistry, Biology, History...",
                "warning",
            )
        if not subjects:
            subjects = ["Mathematics", "Science"]
        self.memory.set_topic(subjects[0])

        difficulties: dict[str, str] = {}
        weak = [s.lower() for s in self.memory.weak_subjects]
        for subject in subjects:
            answer = self.ui.ask(f"How hard is {subject} for you? (easy/medium/hard)")
            if answer.strip().lower() in {"easy", "medium", "hard"}:
                difficulties[subject] = answer.strip().lower()
                if answer.strip().lower() == "hard" and subject.lower() not in weak:
                    weak.append(subject.lower())

        exam_raw = self.ui.ask("Any exam date? (YYYY-MM-DD, or press Enter to skip)")
        exam_date = exam_raw if exam_raw and self._valid_date(exam_raw) else None

        days_raw = self.ui.ask("Plan for how many days? (default 7)")
        try:
            days = max(1, min(21, int(days_raw or 7)))
        except ValueError:
            days = 7

        self.ui.loading("Designing your timetable", duration=1.0)
        schedule = self.planner.create_schedule(
            hours_per_day=hours,
            subjects=subjects,
            difficulties=difficulties,
            weak_subjects=weak,
            exam_date=exam_date,
            days=days,
        )

        self.ui.print_bot(
            f"Here's your **{days}-day study plan** — {schedule.total_hours()} study hours "
            f"total. 📅", "success",
        )
        for day in schedule.days:
            self.ui.print_table(
                ["Start", "End", "Duration", "Activity"],
                day.as_table(),
                title=day.date_label,
            )
        for note in schedule.notes:
            self.ui.console.print(f"• {note}", style=self.ui.style("muted"))
        if schedule.subject_distribution:
            self.ui.print_table(
                ["Subject", "Hours / week"],
                [[k, f"{v:.1f} h"] for k, v in schedule.subject_distribution.items()],
                title="Time distribution",
            )

        self.memory.record_action("schedule", ", ".join(subjects))
        self.storage.add_study_hours(min(hours, 1.0) / 2)  # light credit for planning effort
        self.storage.complete_goal(f"Created a {days}-day study schedule")
        self._save_exchange("bot", f"Study schedule generated for {days} days")
        return f"Schedule ready: {days} days, {subjects}"

    def _handle_daily_plan_wizard(self, text: str) -> str:
        self.ui.print_bot(
            "Let's design your perfect day! 🌅 Answer a couple of quick questions.",
            "info",
        )
        wake = self.ui.ask("What time do you wake up? (HH:MM, default 06:30)")
        if not self.daily.validate_time(wake):
            wake = "06:30"
            self.ui.console.print("Using default wake time **06:30**.", style=self.ui.style("muted"))
        hours_raw = self.ui.ask("How many hours can you study today? (default 4)")
        try:
            hours = max(1.0, min(12.0, float(hours_raw or 4)))
        except ValueError:
            hours = 4.0
        exercise = self.ui.ask_yes_no("Include an exercise slot?")

        plan = self.daily.create_plan(study_hours=hours, wake_time=wake, exercise=exercise)
        self.ui.print_bot("Here's your balanced day plan! ⚖️", "success")
        self.ui.print_table(["Time", "Activity"],
                            [b.as_row() for b in plan.blocks],
                            title=f"Daily Plan — wake {plan.wake_time}")
        self.memory.record_action("daily_plan", f"wake {plan.wake_time}")
        self.storage.complete_goal("Created a daily routine plan")
        self._save_exchange("bot", "Daily routine plan generated")
        return f"Daily plan generated (wake {plan.wake_time})"

    def _handle_settings_wizard(self, text: str) -> str:
        settings = self.storage.get_settings()
        options = {
            "theme": f"Change colour theme (current: {settings.get('theme', 'blue')})",
            "name": f"Change username (current: {settings.get('username')})",
            "hours": f"Preferred study hours/day (current: {settings.get('preferred_study_hours')})",
            "subjects": f"Preferred subjects (current: {', '.join(settings.get('preferred_subjects', [])) or 'none'})",
            "questions": f"Quiz questions per quiz (current: {settings.get('quiz_question_count')})",
            "notifications": f"Notifications (current: {'on' if settings.get('notifications') else 'off'})",
            "done": "Finish settings",
        }
        while True:
            choice = self.ui.print_menu("⚙️ Settings menu:", options)
            if not choice or choice == "done":
                self.ui.print_bot("Settings saved! ✅", "success")
                self._save_exchange("bot", "Settings updated")
                return "Settings saved."

            if choice == "theme":
                from ui import THEMES
                theme_choice = self.ui.ask(f"Theme? ({', '.join(THEMES)})")
                if theme_choice in THEMES:
                    self.storage.update_settings(theme=theme_choice)
                    self._apply_settings_to_ui()
                    self.ui.print_bot(f"Theme switched to **{theme_choice}** 🎨", "success")
            elif choice == "name":
                new_name = self.ui.ask("What should I call you?")
                if new_name:
                    self.storage.update_settings(username=new_name)
                    self.storage.update_profile(name=new_name)
                    self.ui.print_bot(f"Username set to **{new_name}** ✏️", "success")
            elif choice == "hours":
                try:
                    hours = max(1, min(16, int(self.ui.ask("Preferred study hours per day?"))))
                    self.storage.update_settings(preferred_study_hours=hours)
                    self.ui.print_bot(f"Preferred study hours set to **{hours} h/day** ⏰", "success")
                except ValueError:
                    self.ui.print_bot("That wasn't a valid number — nothing changed.", "warning")
            elif choice == "subjects":
                raw = self.ui.ask("Preferred subjects (comma-separated)?")
                prefs, prefs_invalid = resolve_subjects(raw, self.knowledge)
                if prefs:
                    self.storage.update_settings(preferred_subjects=prefs)
                    self.storage.update_profile(favourite_subjects=prefs)
                    note = f"Preferred subjects saved: {', '.join(prefs)} ✅"
                    if prefs_invalid:
                        note += f" (ignored: {', '.join(prefs_invalid)})"
                    self.ui.print_bot(note, "success")
                else:
                    self.ui.print_bot(
                        "No valid subjects entered — nothing changed. "
                        "Try Maths, Physics, Chemistry, Computer Science...", "warning",
                    )
            elif choice == "questions":
                try:
                    count = max(1, min(20, int(self.ui.ask("Questions per quiz (1-20)?"))))
                    self.storage.update_settings(quiz_question_count=count)
                    self.ui.print_bot(f"Quiz length set to **{count} questions** 📝", "success")
                except ValueError:
                    self.ui.print_bot("That wasn't a valid number — nothing changed.", "warning")
            elif choice == "notifications":
                current = bool(settings.get("notifications", True))
                self.storage.update_settings(notifications=not current)
                state = "on 🔔" if not current else "off 🔕"
                self.ui.print_bot(f"Notifications turned **{state}**", "success")
            settings = self.storage.get_settings()

    def _run_onboarding(self) -> None:
        """First-run interview to build the user profile."""
        self.ui.print_bot(self.generator.onboarding_question(), "info")
        name = self.ui.ask("Your name")
        if not name:
            name = "Student"
        self.storage.update_profile(name=name)

        # Class/grade — reject nonsense like "69" and re-ask.
        klass = ""
        while True:
            klass_raw = self.ui.ask(
                "Which class/grade are you in? (1-12, or college/degree; optional)"
            )
            if not klass_raw.strip():
                break
            klass = normalise_class(klass_raw)
            if klass:
                break
            self.ui.print_bot(
                f"Hmm, **{klass_raw.strip()}** doesn't look like a valid class. "
                "Try a number from 1 to 12, or say `college`/`degree`/`masters` "
                "— or press Enter to skip.", "warning",
            )
        self.storage.update_profile(**{"class": klass})

        # Favourite subjects — only known subjects are kept.
        fav = []
        while True:
            fav_raw = self.ui.ask("Favourite subjects? (e.g. Maths, Physics; optional)")
            if not fav_raw.strip():
                break
            fav, fav_invalid = resolve_subjects(fav_raw, self.knowledge)
            if fav:
                break
            if fav_invalid:
                self.ui.print_bot(
                    f"I don't recognise **{', '.join(fav_invalid)}** as a subject. "
                    f"Valid subjects: {', '.join(canonical_subjects(self.knowledge))}",
                    "warning",
                )
            else:
                self.ui.print_bot(
                    "No valid subjects found. Try names like Maths, Physics, "
                    "Chemistry, Biology, Computer Science, History...", "warning",
                )
        self.storage.update_profile(favourite_subjects=fav)

        # Weak subjects — only known subjects are kept.
        weak = []
        while True:
            weak_raw = self.ui.ask("Subjects you find tricky? (e.g. Chemistry; optional)")
            if not weak_raw.strip():
                break
            weak, weak_invalid = resolve_subjects(weak_raw, self.knowledge)
            if weak:
                break
            if weak_invalid:
                self.ui.print_bot(
                    f"I don't recognise **{', '.join(weak_invalid)}** as a subject. "
                    f"Valid subjects: {', '.join(canonical_subjects(self.knowledge))}",
                    "warning",
                )
            else:
                self.ui.print_bot(
                    "No valid subjects found. Try names like Maths, Physics, "
                    "Chemistry, Biology, History...", "warning",
                )
        self.storage.update_profile(weak_subjects=weak)

        self.memory.update_profile(name=name)
        self.state = "idle"
        welcome = (
            f"🎉 All set, **{name}**! Your profile is saved — I'll remember it next time.\n"
            f"Favourite subjects: {', '.join(fav) if fav else 'not set yet'}\n"
            f"Weak areas: {', '.join(weak) if weak else 'none noted — great!'}\n\n"
            "What shall we study today?"
        )
        self.ui.print_bot(welcome, "success")
        self._save_exchange("bot", welcome)
        self.storage.complete_goal("Completed first-run onboarding")
        log.info("Onboarding complete for %s", name)

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _wizard_cancelled(self) -> str:
        reply = "No problem — whenever you're ready, just say the word! 😊"
        self.ui.print_bot(reply, "info")
        self._save_exchange("bot", reply)
        return reply

    def _tip_category(self, text: str) -> Optional[str]:
        """Map user text to a study-tip category if one is mentioned."""
        lowered = text.lower()
        categories = self.knowledge.tip_categories()
        for category in categories:
            if category.lower() in lowered:
                return category
        return None

    # Common keyboard-mash tokens that must never be web-searched.
    _GIBBERISH_TOKENS = frozenset({
        "qwerty", "qwert", "qwertyuiop", "asdf", "asdfg", "asdfgh", "asdfghjkl",
        "zxcv", "zxcvbn", "zxcvbnm", "qaz", "wsx", "edc", "rfv", "tgb", "yhn",
        "ujm", "poiuy", "lkjh", "mnbvc", "qwertz", "aaaa", "eeee", "iiii",
    })

    def _looks_like_real_query(self, text: str) -> bool:
        """True when the message contains real, meaningful words.

        Used to decide between answering from the web and redirecting to
        study: a real word like "sigma" deserves an answer, while gibberish
        ("asdfghjkl zxcvbnm", "qwerty") gets the focus redirect. Heuristic:
        at least one content word of 3+ letters has a vowel that is NOT its
        first character AND is not a known keyboard-mash token.
        """
        tokens = re.findall(r"[a-zA-Z]{3,}", text)
        if not tokens:
            return False
        for token in tokens:
            lowered = token.lower()
            if lowered in self._GIBBERISH_TOKENS:
                continue
            if any(v in lowered[1:] for v in "aeiou"):
                return True
        return False

    def _is_topicless_question(self, text: str) -> bool:
        """True when a question has no real topic to look up.

        "what's new", "what is this", "what is that", "what is going on" —
        after stripping the question starter there is no meaningful subject,
        so a web search would be pointless. Returns False for real questions
        like "what is photosynthesis" or "why is the sky blue".
        """
        cleaned = _QUESTION_STARTER.sub("", text.strip()).strip(" ?!.,;:")
        if not cleaned:
            return True
        # Numbers suggest a calculation ("what is 12*8") — not a vague
        # question; let the calculator / normal flow handle it.
        if re.search(r"\d", cleaned):
            return False
        words = [w for w in re.findall(r"[a-z]+", cleaned.lower()) if len(w) > 1]
        if not words:
            return True
        return all(w in _TOPICLESS_WORDS for w in words)

    def _respond_topicless(self, text: str) -> str:
        """Answer vague/basic questions politely — no web search.

        "what's new" reads as small talk, "what is this/that" asks about the
        app itself, and anything else just invites the user to name a topic.
        """
        lowered = text.lower().strip()
        if "new" in lowered:
            reply = (
                f"Nothing new on my end, {self.memory.user_name}! 🤖 I'm still "
                "your study companion — tips, quizzes, schedules and web "
                "research are all ready. What shall we study?"
            )
        elif "this" in lowered or "that" in lowered:
            reply = (
                "This is **StudyBuddy** — your personal study companion! 🎓 "
                "I help with study tips, schedules, quizzes, flashcards, "
                "motivation and web research. Type `help` to see everything."
            )
        else:
            reply = (
                "Happy to help! 😊 What would you like to know about? Try a "
                "topic like `what is photosynthesis`, or ask for `study tips`, "
                "`quiz` or `make a schedule`."
            )
        self.ui.print_bot(reply, "info")
        self._save_exchange("bot", reply)
        return reply

    def _valid_date(self, value: str) -> bool:
        try:
            import datetime
            datetime.datetime.strptime(value.strip(), "%Y-%m-%d")
            return True
        except ValueError:
            return False

    def _occasional_followup(self) -> str:
        """Sometimes add a contextual follow-up to keep the conversation flowing.

        When the bot knows what the user was last doing, the follow-up is
        tailored to that context instead of being generic.
        """
        import random
        if random.random() > 0.35:
            return ""
        last = self.memory.last_action
        topic = self.memory.last_topic
        if last and last.get("intent"):
            intent = last["intent"]
            detail = last.get("detail") or ""
            context_options = {
                "quiz": [
                    f"Nice — want to revise **{detail or 'that'}** with flashcards?",
                    "Want to try another quiz on a different subject?",
                ],
                "flashcards": [
                    "Good revision! Fancy a quick quiz to test it?",
                    "Want to search for more material on that topic?",
                ],
                "knowledge": [
                    f"Want to go deeper on **{detail or topic or 'that'}**? Say `tell me more`.",
                    f"Quiz yourself on **{detail or topic or 'that'}** with `quiz`.",
                ],
                "web_answer": [
                    f"Want me to quiz you on **{detail or 'that'}**?",
                    "Want the full list of results? Say `open top`.",
                ],
                "search": [
                    f"Found anything useful on **{detail or 'that'}**? Want a quiz on it?",
                ],
                "study_help": [
                    f"Want to keep going with **{detail or 'that'}**?",
                ],
            }
            if intent in context_options:
                return random.choice(context_options[intent])
        options = [
            "Want me to plan your study session for today? Just say `daily plan`. 🌅",
            "Feeling stuck on a topic? Try `search <topic>` for resources. 🔎",
            "Quick revision? Say `quiz` or `flashcards` anytime. 🎯",
        ]
        return random.choice(options)

    # ------------------------------------------------------------------
    # Focus mode: keep the user on task
    # ------------------------------------------------------------------
    # Single-word distraction topics used for typo-tolerant fuzzy matching.
    _DISTRACTION_WORDS = frozenset({
        "weather", "forecast", "movie", "film", "netflix", "song", "sing", "music",
        "joke", "meme", "gossip", "drama", "horoscope", "astrology", "game",
        "instagram", "tiktok", "whatsapp", "facebook", "snapchat", "twitter",
        "reels", "story", "riddle", "gaming", "anime", "kpop", "celebrity",
        "celeb", "fashion", "shopping", "outfit", "makeup", "haircut", "hairstyle",
        "dating", "crush", "girlfriend", "boyfriend", "streaming", "binge",
        "dance", "fortnite", "pubg", "minecraft", "trending", "status",
    })

    _study_terms_cache: Optional[frozenset[str]] = None

    def _study_terms(self) -> frozenset[str]:
        """Lowercased subject names + topics — legitimate study vocabulary."""
        if self._study_terms_cache is None:
            terms: set[str] = set()
            for name in self.knowledge.subject_names():
                terms.add(name.lower())
                terms.update(t.lower() for t in self.knowledge.subject_topics(name))
            self._study_terms_cache = frozenset(terms)
        return self._study_terms_cache

    def _is_distraction(self, text: str) -> bool:
        """True when the message is genuinely off-topic chatter.

        Three-tier detection:

        1. **Study-phrase whitelist** — "game theory", "weather patterns",
           "film studies" contain distraction words but are real study
           topics; they are never flagged.
        2. **Explicit phrases** ("recommend a movie", "cricket score",
           "play a game", "tiktok") always count — they are unambiguous
           entertainment/social requests, including typos of them.
        3. **Single distraction words** ("weather", "game", "song") only
           count for short, bare messages with no study content — a rich
           question like "what causes weather patterns" or a message that
           mentions a real subject/topic is answered, not lectured.
        """
        from rapidfuzz import process, fuzz
        lowered = text.lower().strip()

        # Tier 1: legitimate study phrases containing distraction words.
        if any(phrase in lowered for phrase in _STUDY_PHRASES):
            return False

        # Tier 2: explicit multi-word phrases (exact, then typo-corrected).
        if any(phrase in lowered for phrase in _DISTRACTION_PHRASES):
            return True
        corrected = self.intent_detector.correct_typos(lowered)
        if corrected != lowered and any(phrase in corrected for phrase in _DISTRACTION_PHRASES):
            return True

        # Tier 3: single distraction words, proportion-aware.
        tokens = [
            t for t in re.findall(r"[a-z]+", lowered)
            if len(t) >= 4 and t not in _DISTRACTION_STOPWORDS
        ]
        if not tokens:
            return False
        # Any genuine study content (subject/topic mentioned) -> not a
        # distraction, no matter how many distraction words appear.
        if any(t in self._study_terms() for t in tokens):
            return False
        words = list(self._DISTRACTION_WORDS)
        matched = sum(
            1 for token in tokens
            if process.extractOne(token, words, scorer=fuzz.ratio, score_cutoff=82) is not None
        )
        if matched == 0:
            return False
        # Primarily-a-distraction: short message, distraction words >= half.
        return matched / len(tokens) >= 0.5 and len(tokens) <= 5

    def _study_suggestion(self) -> str:
        """Pick a concrete, personalised study action to offer.

        Suggestions are noun phrases so they read naturally inside the
        redirect templates ("How about a quick quiz?").
        """
        import random
        weak = self.memory.weak_subjects
        favourite = self.memory.favourite_subjects
        options = [
            "a quick quiz (`quiz`)",
            "some flashcards (`flashcards`)",
            "a fresh study tip (`study tips`)",
            "a weekly study schedule (`make a schedule`)",
            "a plan for today (`daily plan`)",
            "a web search for study material (`search <topic>`)",
        ]
        if weak:
            options.insert(0, f"a quiz on {weak[0]} — your tricky area")
        if favourite:
            options.append(f"revising your favourite subject, {favourite[0]}")
        return random.choice(options)

    def _redirect_to_study(self) -> str:
        """Reply that steers the user back to studying (never engages).

        Uses the focus-redirect templates (varied, never repeated back to
        back) with a concrete, personalised suggestion attached. After a few
        distraction attempts in a row, the reply turns firmer and calls out
        the pattern — "even smarter" at keeping the user on task.
        """
        self.memory.distraction_streak += 1
        suggestion = self._study_suggestion()

        if self.memory.distraction_streak >= 3:
            weak = self.memory.weak_subjects[0] if self.memory.weak_subjects else "your studies"
            reply, style = self.generator.generate(
                "focus_redirect_firm",
                suggestion=suggestion,
                n=utils.ordinal(self.memory.distraction_streak),
                weak=weak,
            )
        else:
            reply, style = self.generator.generate(
                "focus_redirect", suggestion=suggestion
            )
        self.ui.print_bot(reply, "info")
        self._save_exchange("bot", reply)
        return reply

    def _save_exchange(self, role: str, message: str) -> None:
        """Persist one exchange to JSON history + plain-text chat log."""
        if not message:
            return
        self.storage.append_chat(role, message)
        log_chat(role, message)



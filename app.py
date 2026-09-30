"""
app.py — StudyBuddy entry point
==================================

Run with::

    python app.py

Responsibilities
----------------
1. Build the configuration (env + defaults + directory layout).
2. Initialise the UI and the chatbot engine.
3. Enter the interactive chat loop with graceful shutdown handling.
4. Make sure every crash is logged to ``logs/error.log`` instead of
   killing the terminal with a traceback.

A quick non-interactive health check can be run with ``--check``::

    python app.py --check
"""

from __future__ import annotations

import argparse
import sys
from config import APP_NAME, build_config
from logger import get_logger
from ui import ConsoleUI

log = get_logger("app")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="app.py",
        description=f"{APP_NAME} — your study companion.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run a quick health check (imports + data files) and exit.",
    )
    parser.add_argument(
        "--theme",
        default=None,
        help="Force a colour theme (overrides saved settings).",
    )
    return parser.parse_args(argv)


def run_health_check() -> int:
    """Verify that every subsystem can be constructed without error."""
    from chatbot import StudyBuddyChatbot

    config = build_config()
    ui = ConsoleUI(theme=str(config.theme), interactive=False)

    try:
        chatbot = StudyBuddyChatbot(config, ui)
    except Exception as exc:  # pragma: no cover - defensive
        log.error("Health check FAILED: %s", exc)
        print(f"❌ Health check failed: {exc}")
        return 1

    intent = chatbot.intent_detector.detect("i need help in maths")
    print(f"✔ Intents module OK (sample intent: {intent.name})")
    print(f"✔ Knowledge base OK ({len(chatbot.knowledge.intent_names())} intents, "
          f"{len(chatbot.knowledge.all_tips())} tips, "
          f"{len(chatbot.knowledge.quiz_subjects())} quiz subjects)")
    print(f"✔ Storage OK (profile: {bool(chatbot.storage.is_returning_user())})")
    print("✅ All systems nominal — StudyBuddy is ready to run.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Application entry point."""
    args = parse_args(argv)

    # 1. Load configuration (env, defaults, directories)
    config = build_config()
    log.info("%s starting", APP_NAME)

    if args.check:
        return run_health_check()

    # 2. Initialise UI + chatbot
    ui = ConsoleUI(theme=str(config.theme))
    if args.theme:
        ui.set_theme(args.theme)

    try:
        from chatbot import StudyBuddyChatbot
        chatbot = StudyBuddyChatbot(config, ui)
    except Exception as exc:
        log.critical("Fatal startup error: %s", exc)
        ui.print_error(f"Could not start {APP_NAME}: {exc}")
        return 1

    # 3. Interactive loop
    try:
        chatbot.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # never exit with an unlogged traceback
        log.error("Unhandled exception in main loop: %s", exc)
        ui.print_error(f"Unexpected error: {exc}")
        return 1

    log.info("%s exiting cleanly", APP_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
ui.py — Console user interface for StudyBuddy
================================================

Everything the user *sees* lives here. Built on top of the ``rich`` library
this module provides:

  * the ASCII-art welcome banner,
  * panels, tables, progress bars and status indicators,
  * a loading animation with a spinner thread,
  * a themed colour palette (swappable via settings),
  * safe ``input()`` wrappers that work on real terminals *and* when stdin
    is piped (e.g. for automated demos/tests).

The UI layer never contains business logic — it only renders what the
chatbot asks it to render.
"""

from __future__ import annotations

import builtins
import re
import time
from typing import Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import utils
from config import LOGO_FILE, APP_NAME, APP_TAGLINE

# ---------------------------------------------------------------------------
# Theme palettes (keyed by settings.theme)
# ---------------------------------------------------------------------------
THEMES: dict[str, dict[str, str]] = {
    "blue": {
        "primary": "bold cyan", "secondary": "dim cyan",
        "accent": "magenta", "panel_border": "cyan",
        "success": "bold green", "warning": "bold yellow", "error": "bold red",
        "info": "cyan", "prompt": "bold cyan", "muted": "dim",
    },
    "green": {
        "primary": "bold green", "secondary": "dim green",
        "accent": "yellow", "panel_border": "green",
        "success": "bold green", "warning": "bold yellow", "error": "bold red",
        "info": "green", "prompt": "bold green", "muted": "dim",
    },
    "purple": {
        "primary": "bold magenta", "secondary": "dim magenta",
        "accent": "cyan", "panel_border": "magenta",
        "success": "bold green", "warning": "bold yellow", "error": "bold red",
        "info": "magenta", "prompt": "bold magenta", "muted": "dim",
    },
    "solarized": {
        "primary": "bold #268bd2", "secondary": "dim #586e75",
        "accent": "#d33682", "panel_border": "#268bd2",
        "success": "bold #859900", "warning": "bold #b58900", "error": "bold #dc322f",
        "info": "#268bd2", "prompt": "bold #268bd2", "muted": "dim #657b83",
    },
    "monokai": {
        "primary": "bold #66d9ef", "secondary": "dim #75715e",
        "accent": "#f92672", "panel_border": "#66d9ef",
        "success": "bold #a6e22e", "warning": "bold #e6db74", "error": "bold #f92672",
        "info": "#66d9ef", "prompt": "bold #a6e22e", "muted": "dim #75715e",
    },
}

STYLE_HINT_MAP: dict[str, str] = {
    "info": "info", "success": "success", "warning": "warning",
    "error": "error", "motivation": "accent", "list": "info",
}

# Matches Markdown-style bold (**text**) so content writers can use familiar
# syntax; it is converted to Rich markup ([bold]text[/]) before rendering.
#
# The match is deliberately conservative so code such as "print(2 ** 3)" is
# never mangled: an opener must start at a word boundary (start-of-string or
# whitespace) and be followed by a non-space; a closer must be preceded by a
# non-space and followed by whitespace, end-of-string or punctuation.
_MD_BOLD_PATTERN = re.compile(
    r"(\A|\s)\*\*(?!\s)(?P<content>.+?)(?<!\s)\*\*(?=\s|$|[.,!?;:)%)\"'”])",
    re.DOTALL,
)


def md_to_rich(text: str) -> str:
    """Convert ``**bold**`` Markdown to Rich ``[bold]...[/]`` markup.

    The rest of the string is left untouched so proper Rich markup (if any)
    still works. Content is escaped first so stray characters can never
    crash the renderer.
    """
    escaped = text.replace("[", "\\[").replace("]", "\\]")
    return _MD_BOLD_PATTERN.sub(r"\1[bold]\g<content>[/]", escaped)


class ConsoleUI:
    """High-level console façade used by the chatbot and app."""

    def __init__(self, theme: str = "blue", interactive: bool = True) -> None:
        self.interactive = interactive
        self.eof = False  # becomes True when stdin reaches EOF
        self.console = Console(highlight=False, safe_box=False)
        self.set_theme(theme)

    # ------------------------------------------------------------------
    # Theme handling
    # ------------------------------------------------------------------
    def set_theme(self, theme: str) -> None:
        """Switch the active colour palette."""
        palette = THEMES.get(theme, THEMES["blue"])
        self.theme = theme
        self._palette = palette

    def style(self, key: str) -> str:
        """Return the style string for a semantic key."""
        return self._palette.get(key, self._palette["info"])

    # ------------------------------------------------------------------
    # Banner & welcome
    # ------------------------------------------------------------------
    def print_logo(self) -> None:
        """Render the ASCII logo centred inside a panel."""
        logo = utils.load_asset_text(LOGO_FILE)
        if not logo:
            logo = APP_NAME
        title = Text(f" {APP_NAME} ", style=self.style("accent"))
        panel = Panel(
            logo,
            title=title,
            subtitle=Text(APP_TAGLINE, style=self.style("muted")),
            border_style=self.style("panel_border"),
            box=box.ROUNDED,
            expand=False,
        )
        self.console.print(panel)
        self.console.print()

    def print_welcome(self, message: str) -> None:
        """Show the welcome panel for a new or returning user."""
        self.print_panel(message, title="👋 Welcome", style="info")

    # ------------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------------
    def print_panel(self, message: str, title: Optional[str] = None,
                    style: str = "info", markdown: bool = True) -> None:
        """Render ``message`` inside a themed panel.

        By default ``**bold**`` Markdown is converted to Rich markup; pass
        ``markdown=False`` when the message already contains Rich markup
        (e.g. the menu builder).
        """
        border = self.style(STYLE_HINT_MAP.get(style, style if style in self._palette else "info"))
        body = md_to_rich(message) if (message and markdown) else (message or "")
        panel = Panel(
            Text.from_markup(body) if body else "",
            title=title,
            border_style=border,
            box=box.ROUNDED,
        )
        self.console.print(panel)

    def print_bot(self, message: str, style: str = "info") -> None:
        """Print a chatbot reply with a small "StudyBuddy" tag.

        ``**bold**`` Markdown is converted to Rich markup automatically.
        """
        border = self.style(STYLE_HINT_MAP.get(style, "info"))
        panel = Panel(
            Text.from_markup(md_to_rich(message)) if message else "",
            title=f"[{border}]📘 StudyBuddy[/]",
            border_style=border,
            box=box.ROUNDED,
        )
        self.console.print(panel)

    def print_user(self, message: str) -> None:
        """Echo the user's own message in a slim, muted panel."""
        self.console.print(
            Panel(
                Text(message, style=self.style("secondary")),
                title="You",
                border_style=self.style("muted"),
                box=box.SIMPLE,
            )
        )

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    def print_table(self, headers: list[str], rows: list[list[str]],
                    title: Optional[str] = None) -> None:
        """Render a table with the active theme."""
        table = Table(
            title=title,
            title_style=self.style("primary"),
            header_style=self.style("primary"),
            border_style=self.style("panel_border"),
            box=box.SIMPLE_HEAVY,
        )
        for header in headers:
            table.add_column(header)
        for row in rows:
            table.add_row(*[str(cell) for cell in row])
        self.console.print(table)

    # ------------------------------------------------------------------
    # Progress & status
    # ------------------------------------------------------------------
    def print_progress_bar(self, percent: float, label: str = "Progress",
                           width: int = 30) -> None:
        """Render a simple inline progress bar (e.g. study progress)."""
        percent = utils.clamp(percent, 0.0, 100.0)
        filled = int(width * percent / 100)
        bar = "█" * filled + "░" * (width - filled)
        colour = "green" if percent >= 60 else ("yellow" if percent >= 30 else "red")
        text = Text()
        text.append(f" {label}: ", style=self.style("primary"))
        text.append(bar, style=f"bold {colour}")
        text.append(f" {percent:.0f}%", style=self.style("muted"))
        self.console.print(text)

    def loading(self, message: str = "Thinking", duration: float = 1.2) -> None:
        """Show a short spinner animation with a status message.

        Uses a background thread so the console stays responsive.
        """
        if not self.interactive or duration <= 0:
            return
        with self.console.status(
            f"[{self.style('primary')}]{message}…",
            spinner="dots",
        ):
            time.sleep(duration)

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def ask(self, prompt: str = "") -> str:
        """Prompt the user for text input.

        Works both interactively and when stdin is piped (tests/demos),
        because it falls back to plain ``input()``.

        Returns an empty string when the user just pressed Enter, and an
        empty string on EOF as well — callers that need to detect EOF should
        check ``self.eof`` afterwards.
        """
        styled = f"[{self.style('prompt')}]{prompt}[/]" if prompt else ""
        suffix = "➤ "
        try:
            self.console.print(styled, end="")
            self.console.print(suffix, end="", style=self.style("accent"))
            self.console.file.flush()
        except OSError:
            pass

        try:
            value = builtins.input()
            self.eof = False
        except (EOFError, KeyboardInterrupt):
            self.eof = True
            value = ""
        return value.strip()

    # ------------------------------------------------------------------
    # Menus
    # ------------------------------------------------------------------
    def print_menu(self, title: str, options: dict[str, str]) -> str:
        """Render a numbered menu and return the user's choice key.

        Parameters
        ----------
        title : str
            Menu title.
        options : dict[str, str]
            Mapping of option key -> option description (display order kept).
        """
        lines = [title, ""]
        for index, (key, description) in enumerate(options.items(), start=1):
            lines.append(f"  [bold]{index}[/]. [info]{key}[/] — {description}")
        lines.append("")
        lines.append("  Type a number (or command name) to choose, or 'back' to cancel.")
        self.print_panel("\n".join(lines), title="📋 Menu", style="info", markdown=False)

        while True:
            answer = self.ask("Your choice")
            if self.eof:                     # stdin closed — bail out
                return ""
            if answer.lower() in {"back", "cancel", "exit", "quit", "b"}:
                return ""
            if answer.isdigit():
                index = int(answer)
                if 1 <= index <= len(options):
                    return list(options.keys())[index - 1]
            lowered = answer.lower()
            for key in options:
                if lowered == key or lowered in key.split():
                    return key
            self.console.print(
                f"[{self.style('warning')}]Please pick a valid option.[/]"
            )

    def print_help(self, commands: list[tuple[str, str]]) -> None:
        """Render the command reference table."""
        self.print_table(
            ["Command", "What it does"],
            [[cmd, desc] for cmd, desc in commands],
            title="Command Reference",
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def status_line(self, left: str, right: str = "") -> None:
        """Print a slim status/divider line."""
        self.console.rule(Text(left, style=self.style("muted")), style=self.style("muted"))
        if right:
            self.console.print(right, style=self.style("muted"))

    def divider(self) -> None:
        self.console.rule(style=self.style("muted"))

    def print_error(self, message: str) -> None:
        self.print_panel(f"⚠️  {message}", title="Error", style="error")

    def clear(self) -> None:
        """Clear the terminal (no-op when output is piped)."""
        if self.console.is_terminal:
            self.console.clear()

    def ask_yes_no(self, question: str) -> bool:
        """Ask a yes/no question; returns True for affirmative answers."""
        answer = self.ask(f"{question} (y/n)")
        return utils.is_yes(answer)

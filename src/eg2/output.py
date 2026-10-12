"""Terminal output helpers for the lesson series.

Design rule: **silence is the enemy.** A learner should always be able to see
what just happened, so these helpers print by default.

Standard library only. Nothing in here may import torch, numpy, or
sentence-transformers, or the environment doctor would fail on exactly the
machines that need diagnosing.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Iterable, Sequence


def _force_utf8(stream) -> None:
    """Stop a stray non-ASCII character from crashing the whole lesson.

    Windows consoles still default to cp1252, where box-drawing characters are
    undefined -- so printing one raises UnicodeEncodeError and takes the lesson
    down. Reconfiguring to UTF-8 with a replacement fallback means a cosmetic
    problem can never become a fatal one.
    """
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


for _stream in (sys.stdout, sys.stderr):
    _force_utf8(_stream)


def _supports(ch: str) -> bool:
    """Can the current stdout actually encode this character?"""
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        ch.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


#: Box-drawing characters degrade to ASCII rather than crashing.
RULE_CHAR = "─" if _supports("─") else "-"
BAR_CHAR = "═" if _supports("═") else "="

# --- Colour handling ------------------------------------------------------

_COLOURS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
}


def _enable_windows_vt() -> bool:
    """Ask the Windows console to interpret ANSI escape codes.

    Without this, every colour literal prints as raw escape garbage on cmd.exe.
    Returns True if VT processing is believed to be active.
    """
    if sys.platform != "win32":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        # -11 == STD_OUTPUT_HANDLE, 0x0004 == ENABLE_VIRTUAL_TERMINAL_PROCESSING
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def colour_enabled(stream=None) -> bool:
    """Decide whether to emit ANSI colour codes."""
    if os.environ.get("NO_COLOR"):
        return False
    stream = stream or sys.stdout
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    return _enable_windows_vt()


_COLOUR_ON = colour_enabled()


def paint(text: str, style: str) -> str:
    """Wrap `text` in a colour style, or return it unchanged if colour is off."""
    if not _COLOUR_ON or style not in _COLOURS:
        return text
    return f"{_COLOURS[style]}{text}{_COLOURS['reset']}"


def width(default: int = 78) -> int:
    """Terminal width, clamped so output stays readable in narrow windows."""
    return max(60, min(shutil.get_terminal_size((default, 24)).columns, 100))


# --- Structural elements --------------------------------------------------


def rule(char: str = RULE_CHAR) -> str:
    """A full-width horizontal rule."""
    return paint(char * width(), "dim")


def banner(title: str, subtitle: str = "") -> None:
    """Opening banner: what this lesson is, in one line."""
    bar = BAR_CHAR * width()
    print()
    print(paint(bar, "cyan"))
    print(paint(f"  {title}", "bold"))
    if subtitle:
        print(paint(f"  {subtitle}", "dim"))
    print(paint(bar, "cyan"))


def section(title: str, number: int | None = None) -> None:
    """Start a new major section of a lesson."""
    label = f"{number}. {title}" if number is not None else title
    print()
    print()
    print(paint(f"{label}", "bold"))
    print(rule())


def step(text: str, number: int | None = None, total: int | None = None) -> None:
    """A numbered action the lesson is about to take."""
    if number is not None and total is not None:
        prefix = paint(f"[{number}/{total}]", "blue")
    elif number is not None:
        prefix = paint(f"[{number}]", "blue")
    else:
        prefix = paint("[ ]", "blue")
    print(f"{prefix} {text}")


def explain(text: str) -> None:
    """One-line narration of what just happened or why it matters."""
    for line in text.splitlines():
        print(f"    {paint('->', 'dim')} {line}")


def note(text: str) -> None:
    """Neutral asides the learner should not skip."""
    for line in text.splitlines():
        print(f"  {paint('note:', 'blue')} {line}")


def warn(text: str) -> None:
    """Something that will bite later if ignored."""
    for line in text.splitlines():
        print(f"  {paint('warning:', 'yellow')} {line}")


def error(text: str) -> None:
    for line in text.splitlines():
        print(f"  {paint('error:', 'red')} {line}")


def ok(text: str) -> None:
    print(f"  {paint('ok:', 'green')} {text}")


def bullet(items: Iterable[str], indent: int = 2) -> None:
    """A plain bulleted list."""
    pad = " " * indent
    for item in items:
        print(f"{pad}{paint('*', 'dim')} {item}")


def kv(pairs: Sequence[tuple[str, object]], indent: int = 2) -> None:
    """Aligned key/value block -- the workhorse for 'read your output' tables."""
    if not pairs:
        return
    pad = " " * indent
    key_width = max(len(str(key)) for key, _ in pairs)
    for key, value in pairs:
        print(f"{pad}{paint(str(key).ljust(key_width), 'dim')}  {value}")


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    """Minimal fixed-width table. Numbers right-aligned, text left-aligned."""
    if not rows:
        return
    columns = len(headers)
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i in range(columns):
            widths[i] = max(widths[i], len(str(row[i])))

    def render(cells: Sequence[object]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            text = str(cell)
            parts.append(text.rjust(widths[i]) if _is_number(text) else text.ljust(widths[i]))
        return "  ".join(parts).rstrip()

    print()
    print("  " + paint(render(headers), "bold"))
    print("  " + paint("  ".join("-" * w for w in widths), "dim"))
    for row in rows:
        print("  " + render(row))


def _is_number(text: str) -> bool:
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def checkpoint(question: str, answer: str) -> None:
    """Self-check question with its answer.

    The answer is printed rather than hidden behind an HTML <details> block --
    this runs in a terminal, where that markup renders as literal text. Read the
    question out loud and commit to an answer before looking down.
    """
    print()
    print(paint("  checkpoint", "bold"))
    print(f"  Q. {question}")
    print()
    for line in answer.splitlines():
        print(f"    {paint('A.', 'green')} {line}")
    print()


def footer(text: str) -> None:
    """Closing line, plus what to run next."""
    print()
    print(rule())
    print(paint(f"  {text}", "dim"))
    print()
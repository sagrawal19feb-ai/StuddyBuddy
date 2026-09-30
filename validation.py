"""
validation.py — Input validation for StudyBuddy
================================================

Rejects nonsense values so the profile and schedules stay clean:

  * **Class / grade** — only realistic school classes (1-12) or common
    higher-education labels ("college", "degree", "masters"...). A joke like
    "69", "0" or "100" is rejected and re-asked.
  * **Subjects** — every subject must resolve to a known subject (with
    aliases and fuzzy matching, so "chem" → Chemistry, "cs" → Computer
    Science). Unknown entries are dropped and reported, never stored.

These validators are used EVERYWHERE class/subjects are collected —
onboarding, the schedule wizard and the settings menu — so the data is
always clean, not just at first input.
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Class / grade validation
# ---------------------------------------------------------------------------
VALID_CLASS_NUMBERS = range(1, 13)  # school classes 1-12

# Common higher-education / generic labels that are acceptable answers,
# mapped to their display form.
VALID_CLASS_WORDS: dict[str, str] = {
    "college": "College", "university": "University", "degree": "Degree",
    "undergraduate": "Undergraduate", "ug": "Undergraduate",
    "bachelors": "Bachelors", "bachelor": "Bachelors",
    "postgraduate": "Postgraduate", "pg": "Postgraduate",
    "masters": "Masters", "master": "Masters",
    "diploma": "Diploma", "engineering": "Engineering",
    "btech": "B.Tech", "bsc": "BSc", "bcom": "B.Com", "ba": "BA",
    "graduation": "Graduation", "school": "School",
}


def normalise_class(value: str) -> str:
    """Return a clean class value, or ``""`` when the input is invalid.

    Accepts:
      * "10", "10th", "class 10", "ten"  -> "10"
      * "college", "degree", "masters"   -> the word (nicely formatted)

    Rejects:
      * "69", "0", "100", "-5", "twenty-seven thousand"  -> ""
    """
    text = (value or "").strip()
    if not text:
        return ""
    lowered = text.lower()

    # A minus sign means a negative number — always invalid.
    if "-" in text:
        return ""

    # Numeric: strip "class", ordinal suffixes, keep digits.
    digits = re.sub(r"[^0-9]", "", text)
    if digits:
        number = int(digits)
        if number in VALID_CLASS_NUMBERS:
            return str(number)
        return ""  # out-of-range number (69, 0, 99, ...) is invalid

    # Word form ("ten" -> 10) for common numbers.
    word_numbers = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12,
    }
    if lowered in word_numbers:
        return str(word_numbers[lowered])

    # Higher-education / generic labels.
    for word, display in VALID_CLASS_WORDS.items():
        if lowered == word or lowered.startswith(word):
            return display

    return ""


# ---------------------------------------------------------------------------
# Subject validation
# ---------------------------------------------------------------------------
# Common short forms / alternate names -> canonical subject.
SUBJECT_ALIASES: dict[str, str] = {
    "math": "Mathematics", "maths": "Mathematics", "mathematics": "Mathematics",
    "chem": "Chemistry", "chemistry": "Chemistry",
    "bio": "Biology", "biology": "Biology",
    "phys": "Physics", "physics": "Physics",
    "science": "Science", "environmental science": "Environment",
    "cs": "Computer Science", "comp sci": "Computer Science",
    "computer": "Computer Science", "computer science": "Computer Science",
    "computers": "Computer Science", "coding": "Programming",
    "programming": "Programming", "computer programming": "Programming",
    "history": "History", "indian history": "Indian History",
    "world history": "World History", "modern history": "World History",
    "geography": "Geography", "geo": "Geography",
    "english": "English", "eng": "English",
    "gk": "General Knowledge", "general knowledge": "General Knowledge",
    "general awareness": "General Knowledge", "current affairs": "General Knowledge",
    "economics": "Economics", "eco": "Economics", "economy": "Economics",
    "civics": "Civics", "polity": "Civics", "political science": "Civics",
    "environment": "Environment", "environmental studies": "Environment",
    "evs": "Environment", "environmental science": "Environment",
    "technology": "Technology", "tech": "Technology", "computers and technology": "Technology",
    "space": "Space", "astronomy": "Space", "space science": "Space",
    "human body": "Human Body", "human biology": "Human Body",
    "body": "Human Body", "human physiology": "Human Body",
    "english literature": "English", "literature": "English",
}


def canonical_subjects(knowledge) -> list[str]:
    """Return the full list of known subjects (names + quiz + categories)."""
    subjects = set()
    subjects.update(knowledge.subject_names())
    subjects.update(knowledge.quiz_subjects())
    subjects.update(knowledge.knowledge_categories())
    return sorted(subjects)


def resolve_subject(entry: str, known: list[str]) -> Optional[str]:
    """Map a user entry to a canonical subject name, or ``None`` if invalid.

    Exact alias lookup first, then fuzzy matching against the known subject
    list so typos/short forms still resolve ("chemistri" -> Chemistry).
    """
    text = entry.strip().lower()
    if not text:
        return None

    # Exact alias (also match "class 10 physics" style leftovers).
    if text in SUBJECT_ALIASES:
        return SUBJECT_ALIASES[text]
    for alias, canonical in SUBJECT_ALIASES.items():
        if text == alias or text.startswith(alias + " ") or text.endswith(" " + alias):
            return canonical

    # Direct hit on a known subject.
    lowered_known = [k.lower() for k in known]
    if text in lowered_known:
        return known[lowered_known.index(text)]

    # Fuzzy fallback (partial ratio, order-insensitive, >= 78).
    from rapidfuzz import fuzz
    best_name, best_score = "", 0.0
    for subject in known:
        score = fuzz.partial_ratio(text, subject.lower())
        if score > best_score:
            best_score = score
            best_name = subject
    if best_score >= 78:
        return best_name
    return None


def resolve_subjects(raw: str, knowledge) -> tuple[list[str], list[str]]:
    """Split a comma-separated subject string into (valid, invalid).

    Returns canonical subject names; the invalid entries are reported so the
    caller can tell the user what was dropped.
    """
    known = canonical_subjects(knowledge)
    valid: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        resolved = resolve_subject(entry, known)
        if resolved and resolved.lower() not in seen:
            seen.add(resolved.lower())
            valid.append(resolved)
        elif not resolved:
            invalid.append(entry)
    return valid, invalid

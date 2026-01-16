"""
BIBLOS v2 - Input Validation Utilities

Provides centralized input validation for verse IDs, book codes,
and other common inputs across the system.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple
from functools import lru_cache


# Valid book codes (3-letter abbreviations)
VALID_BOOK_CODES = frozenset([
    # Old Testament
    "GEN", "EXO", "LEV", "NUM", "DEU",
    "JOS", "JDG", "RUT", "1SA", "2SA",
    "1KI", "2KI", "1CH", "2CH", "EZR",
    "NEH", "EST", "JOB", "PSA", "PRO",
    "ECC", "SNG", "ISA", "JER", "LAM",
    "EZK", "DAN", "HOS", "JOL", "AMO",
    "OBA", "JON", "MIC", "NAH", "HAB",
    "ZEP", "HAG", "ZEC", "MAL",
    # New Testament
    "MAT", "MRK", "LUK", "JHN", "ACT",
    "ROM", "1CO", "2CO", "GAL", "EPH",
    "PHP", "COL", "1TH", "2TH", "1TI",
    "2TI", "TIT", "PHM", "HEB", "JAS",
    "1PE", "2PE", "1JN", "2JN", "3JN",
    "JUD", "REV",
    # Deuterocanonical (LXX additions)
    "TOB", "JDT", "ESG", "WIS", "SIR",
    "BAR", "LJE", "S3Y", "SUS", "BEL",
    "1MA", "2MA", "1ES", "PRM", "PS2",
    "ODE", "PSS", "3MA", "4MA"
])

# Compiled regex for verse ID validation
# Format: BOOK.CHAPTER.VERSE or BOOK.CHAPTER.VERSE-VERSE
VERSE_ID_PATTERN = re.compile(
    r'^([1-4]?[A-Z]{2,3})\.(\d{1,3})\.(\d{1,3})(?:-(\d{1,3}))?$'
)

# Maximum valid chapter numbers per book (approximate)
MAX_CHAPTERS = {
    "PSA": 150, "ISA": 66, "JER": 52, "EZK": 48, "GEN": 50,
    "EXO": 40, "LEV": 27, "NUM": 36, "DEU": 34, "MAT": 28,
    "MRK": 16, "LUK": 24, "JHN": 21, "ACT": 28, "REV": 22,
}
DEFAULT_MAX_CHAPTER = 50

# Maximum verse numbers per chapter (approximate)
DEFAULT_MAX_VERSE = 180  # Psalm 119 has 176 verses


class VerseIdValidationError(ValueError):
    """Exception raised for invalid verse ID format."""

    def __init__(self, verse_id: str, reason: str):
        self.verse_id = verse_id
        self.reason = reason
        super().__init__(f"Invalid verse ID '{verse_id}': {reason}")


@lru_cache(maxsize=10000)
def validate_verse_id(verse_id: str, strict: bool = True) -> Tuple[str, int, int]:
    """
    Validate a verse ID and return parsed components.

    Args:
        verse_id: The verse ID to validate (e.g., "GEN.1.1", "PSA.119.176")
        strict: If True, validates book code against known books

    Returns:
        Tuple of (book_code, chapter, verse)

    Raises:
        VerseIdValidationError: If the verse ID is invalid
    """
    if not verse_id:
        raise VerseIdValidationError(verse_id, "Verse ID cannot be empty")

    if not isinstance(verse_id, str):
        raise VerseIdValidationError(str(verse_id), "Verse ID must be a string")

    # Normalize whitespace
    verse_id = verse_id.strip()

    # Check format
    match = VERSE_ID_PATTERN.match(verse_id)
    if not match:
        raise VerseIdValidationError(
            verse_id,
            "Invalid format. Expected BOOK.CHAPTER.VERSE (e.g., GEN.1.1)"
        )

    book = match.group(1)
    chapter = int(match.group(2))
    verse = int(match.group(3))

    # Validate book code
    if strict and book not in VALID_BOOK_CODES:
        raise VerseIdValidationError(
            verse_id,
            f"Unknown book code '{book}'. Valid codes include: GEN, EXO, MAT, etc."
        )

    # Validate chapter range
    max_chapter = MAX_CHAPTERS.get(book, DEFAULT_MAX_CHAPTER)
    if chapter < 1 or chapter > max_chapter:
        raise VerseIdValidationError(
            verse_id,
            f"Chapter {chapter} out of range for {book} (1-{max_chapter})"
        )

    # Validate verse range
    if verse < 1 or verse > DEFAULT_MAX_VERSE:
        raise VerseIdValidationError(
            verse_id,
            f"Verse {verse} out of range (1-{DEFAULT_MAX_VERSE})"
        )

    return book, chapter, verse


def is_valid_verse_id(verse_id: str, strict: bool = False) -> bool:
    """
    Check if a verse ID is valid without raising an exception.

    Args:
        verse_id: The verse ID to check
        strict: If True, validates book code against known books

    Returns:
        True if valid, False otherwise
    """
    try:
        validate_verse_id(verse_id, strict=strict)
        return True
    except (VerseIdValidationError, TypeError):
        return False


def normalize_verse_id(verse_id: str) -> str:
    """
    Normalize a verse ID to standard format.

    - Strips whitespace
    - Uppercases book code
    - Validates format

    Args:
        verse_id: The verse ID to normalize

    Returns:
        Normalized verse ID

    Raises:
        VerseIdValidationError: If the verse ID is invalid
    """
    if not verse_id:
        raise VerseIdValidationError(verse_id, "Verse ID cannot be empty")

    # Strip and uppercase
    normalized = verse_id.strip().upper()

    # Validate (also ensures format is correct)
    validate_verse_id(normalized, strict=False)

    return normalized


def validate_book_code(book_code: str) -> str:
    """
    Validate and normalize a book code.

    Args:
        book_code: The book code to validate (e.g., "gen", "GEN")

    Returns:
        Normalized (uppercase) book code

    Raises:
        ValueError: If the book code is invalid
    """
    if not book_code:
        raise ValueError("Book code cannot be empty")

    normalized = book_code.strip().upper()

    if normalized not in VALID_BOOK_CODES:
        raise ValueError(
            f"Invalid book code '{normalized}'. "
            f"Valid codes include: GEN, EXO, LEV, NUM, DEU, MAT, MRK, LUK, JHN, etc."
        )

    return normalized


def parse_verse_range(verse_range: str) -> Tuple[str, str]:
    """
    Parse a verse range string into start and end verse IDs.

    Args:
        verse_range: Range like "GEN.1.1-5" or "GEN.1.1-GEN.2.3"

    Returns:
        Tuple of (start_verse_id, end_verse_id)

    Raises:
        VerseIdValidationError: If the range is invalid
    """
    if "-" not in verse_range:
        # Single verse
        validate_verse_id(verse_range)
        return verse_range, verse_range

    # Check if it's a simple verse range (GEN.1.1-5)
    if verse_range.count(".") == 2:
        base, end_verse = verse_range.rsplit("-", 1)
        if end_verse.isdigit():
            book, chapter, start_verse = base.rsplit(".", 2)
            start_id = f"{book}.{chapter}.{start_verse}"
            end_id = f"{book}.{chapter}.{end_verse}"
            validate_verse_id(start_id)
            validate_verse_id(end_id)
            return start_id, end_id

    # Full range (GEN.1.1-GEN.2.3)
    parts = verse_range.split("-", 1)
    if len(parts) != 2:
        raise VerseIdValidationError(verse_range, "Invalid verse range format")

    start_id, end_id = parts
    validate_verse_id(start_id)
    validate_verse_id(end_id)

    return start_id, end_id

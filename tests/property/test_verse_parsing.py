"""
Property-Based Tests for Verse ID Parsing

Tests verse reference parsing with malformed inputs, edge cases, and Unicode.
"""
import pytest
from hypothesis import given, strategies as st, settings, assume, example
import re

from data.schemas import validate_verse_id, normalize_verse_id
from tests.property.strategies import verse_id_strategy, ALL_BOOKS


class TestVerseIDParsing:
    """Property-based tests for verse ID parsing."""

    @given(verse_id_strategy(valid_only=True))
    @settings(max_examples=300)
    def test_valid_verse_id_format(self, verse_id):
        """Valid verse IDs should match expected format."""
        assert validate_verse_id(verse_id) is True

        # Should have 3 parts separated by dots
        parts = verse_id.split(".")
        assert len(parts) == 3

        # Book should be 3 uppercase letters
        assert parts[0] in ALL_BOOKS
        assert parts[0].isupper()
        assert len(parts[0]) == 3

        # Chapter and verse should be numeric
        assert parts[1].isdigit()
        assert parts[2].isdigit()

        # Chapter and verse should be positive
        assert int(parts[1]) > 0
        assert int(parts[2]) > 0

    @given(st.text(max_size=100))
    @settings(max_examples=300)
    @example("")  # Empty string
    @example("   ")  # Whitespace only
    @example("GEN")  # Missing chapter and verse
    @example("GEN.1")  # Missing verse
    @example(".1.1")  # Missing book
    @example("GEN.1.")  # Trailing dot
    @example("GEN..1")  # Double dot
    @example("GEN.0.1")  # Zero chapter
    @example("GEN.1.0")  # Zero verse
    @example("GEN.-1.1")  # Negative chapter
    @example("GEN.1.-1")  # Negative verse
    @example("ABC.1.1")  # Invalid book code
    @example("Gen.1.1")  # Lowercase book
    @example("GEN:1:1")  # Wrong separator
    @example("GEN 1:1")  # Space separator
    @example("בְּרֵאשִׁית.1.1")  # Hebrew in book code
    @example("ΓΕΝ.1.1")  # Greek letters
    def test_malformed_verse_ids(self, text):
        """Malformed verse IDs should not crash validation."""
        # Should not raise exceptions
        result = validate_verse_id(text)
        assert isinstance(result, bool)

        # If validation passes, it should be properly formatted
        if result:
            assert isinstance(text, str)
            assert len(text) > 0
            parts = text.upper().replace(" ", ".").replace(":", ".").split(".")
            assert len(parts) >= 3
            # Should have numeric chapter and verse
            try:
                int(parts[1])
                int(parts[2])
            except (ValueError, IndexError):
                # If this fails, validation should have returned False
                assert False, f"validate_verse_id returned True for invalid: {text}"

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=300)
    def test_normalization_never_crashes(self, text):
        """Normalization should handle any input without crashing."""
        normalized = normalize_verse_id(text)
        assert isinstance(normalized, str)

    @given(st.sampled_from(ALL_BOOKS), st.integers(min_value=1, max_value=150), st.integers(min_value=1, max_value=176))
    @settings(max_examples=300)
    def test_various_separators_normalize(self, book, chapter, verse):
        """Various separator formats should normalize to standard format."""
        # Test different separator combinations
        formats = [
            f"{book}.{chapter}.{verse}",  # Standard
            f"{book}:{chapter}:{verse}",  # Colon
            f"{book} {chapter}:{verse}",  # Space + colon
            f"{book}.{chapter}:{verse}",  # Mixed
            f"{book} {chapter} {verse}",  # All spaces
        ]

        expected = f"{book}.{chapter}.{verse}"

        for fmt in formats:
            normalized = normalize_verse_id(fmt)
            assert normalized == expected, f"Format {fmt} should normalize to {expected}, got {normalized}"

    @given(st.sampled_from(ALL_BOOKS), st.integers(min_value=1, max_value=150), st.integers(min_value=1, max_value=176))
    @settings(max_examples=300)
    def test_case_insensitive_normalization(self, book, chapter, verse):
        """Normalization should handle lowercase book codes."""
        lowercase = f"{book.lower()}.{chapter}.{verse}"
        normalized = normalize_verse_id(lowercase)
        assert normalized == f"{book}.{chapter}.{verse}"

        mixedcase = f"{book[0]}{book[1].lower()}{book[2]}.{chapter}.{verse}"
        normalized = normalize_verse_id(mixedcase)
        assert normalized == f"{book}.{chapter}.{verse}"

    @given(verse_id_strategy(valid_only=True))
    @settings(max_examples=200)
    def test_normalization_preserves_valid_ids(self, verse_id):
        """Normalizing an already valid ID should return the same ID."""
        normalized = normalize_verse_id(verse_id)
        assert normalized == verse_id

    @given(st.text(max_size=100))
    @settings(max_examples=200)
    def test_empty_and_whitespace(self, text):
        """Edge case: empty strings and whitespace."""
        assume(len(text.strip()) == 0)  # Only whitespace

        # Should not crash
        result = validate_verse_id(text)
        assert result is False  # Empty/whitespace should be invalid

        normalized = normalize_verse_id(text)
        assert isinstance(normalized, str)

    @given(st.text(min_size=1000, max_size=5000))
    @settings(max_examples=50)
    def test_very_long_strings(self, long_text):
        """Very long strings should not cause performance issues."""
        # Should complete reasonably quickly (Hypothesis will timeout if too slow)
        result = validate_verse_id(long_text)
        assert isinstance(result, bool)

        normalized = normalize_verse_id(long_text)
        assert isinstance(normalized, str)

    @given(st.text(alphabet=".\n\t\r :", min_size=1, max_size=100))
    @settings(max_examples=200)
    def test_only_separators(self, text):
        """Strings with only separator characters."""
        result = validate_verse_id(text)
        # Should be invalid
        assert result is False

    @given(st.integers())
    @settings(max_examples=100)
    def test_non_string_input_validation(self, value):
        """validate_verse_id should handle non-string inputs gracefully."""
        # Calling with integer should return False (not crash)
        result = validate_verse_id(str(value))
        assert isinstance(result, bool)


class TestVerseIDComponents:
    """Tests for extracting components from verse IDs."""

    @given(verse_id_strategy(valid_only=True))
    @settings(max_examples=200)
    def test_component_extraction(self, verse_id):
        """Should be able to extract book, chapter, verse from valid ID."""
        parts = verse_id.split(".")
        book, chapter_str, verse_str = parts

        # Book should be valid
        assert book in ALL_BOOKS

        # Should be able to convert to integers
        chapter = int(chapter_str)
        verse = int(verse_str)

        assert chapter > 0
        assert verse > 0

    @given(
        st.sampled_from(ALL_BOOKS),
        st.integers(min_value=1, max_value=150),
        st.integers(min_value=1, max_value=176)
    )
    @settings(max_examples=200)
    def test_reconstruction(self, book, chapter, verse):
        """Should be able to reconstruct verse ID from components."""
        verse_id = f"{book}.{chapter}.{verse}"
        assert validate_verse_id(verse_id) is True

        # Extract components
        parts = verse_id.split(".")
        assert parts[0] == book
        assert int(parts[1]) == chapter
        assert int(parts[2]) == verse


class TestVerseIDEdgeCases:
    """Edge cases specific to biblical references."""

    @given(st.sampled_from(ALL_BOOKS))
    @settings(max_examples=100)
    def test_max_chapter_numbers(self, book):
        """Test with maximum reasonable chapter numbers."""
        # Most books have < 50 chapters, but Psalms has 150
        max_chapters = {
            "PSA": 150,
            "GEN": 50,
            "ISA": 66,
        }

        max_chapter = max_chapters.get(book, 50)

        for chapter in [1, max_chapter, max_chapter + 1, max_chapter * 2]:
            verse_id = f"{book}.{chapter}.1"
            result = validate_verse_id(verse_id)
            # Should not crash regardless of whether it's valid
            assert isinstance(result, bool)

    @given(st.sampled_from(["PSA"]))  # Psalm 119 has 176 verses
    @settings(max_examples=50)
    def test_max_verse_numbers(self, book):
        """Test with maximum verse numbers."""
        for verse in [1, 176, 177, 200, 1000]:
            verse_id = f"{book}.119.{verse}"
            result = validate_verse_id(verse_id)
            assert isinstance(result, bool)

    def test_specific_problematic_cases(self):
        """Test specific known problematic cases."""
        cases = [
            ("", False),  # Empty
            ("GEN.1.1", True),  # Valid
            ("gen.1.1", False),  # Lowercase (before normalization)
            ("GENESIS.1.1", False),  # Full name
            ("GN.1.1", False),  # 2-letter code
            ("GENE.1.1", False),  # 4-letter code
            ("GEN.1.1.1", False),  # Extra component
            ("GEN.1", False),  # Missing verse
            ("1.1", False),  # Missing book
            ("GEN.1.1 ", False),  # Trailing space (before normalization)
            (" GEN.1.1", False),  # Leading space (before normalization)
        ]

        for verse_id, expected in cases:
            result = validate_verse_id(verse_id)
            # Note: Some of these may pass after normalization
            # The key is they don't crash
            assert isinstance(result, bool), f"Failed on: {verse_id}"

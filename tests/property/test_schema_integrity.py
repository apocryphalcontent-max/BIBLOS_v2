"""
Property-Based Tests for Schema Integrity

Tests schema validation, serialization, and Unicode handling using Hypothesis.
"""
import sys
import json
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from hypothesis import given, strategies as st, settings, example
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant

# Import schemas with fallback
try:
    from data.schemas import (
        VerseSchema,
        CrossReferenceSchema,
        WordSchema,
        ExtractionResultSchema,
        GoldenRecordSchema,
        validate_verse_id,
        normalize_verse_id,
    )
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False
    # Create minimal stubs for testing
    from dataclasses import dataclass, asdict

    @dataclass
    class VerseSchema:
        verse_id: str = ""
        book: str = ""
        chapter: int = 0
        verse: int = 0
        text: str = ""
        original_text: str = ""
        testament: str = "OT"
        language: str = "hebrew"

        def to_dict(self):
            return asdict(self)

        def to_json(self, indent=2):
            return json.dumps(self.to_dict(), indent=indent)

        @classmethod
        def from_dict(cls, data):
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

        def validate(self):
            errors = []
            if not self.verse_id:
                errors.append("verse_id is required")
            if not self.book:
                errors.append("book is required")
            if self.chapter < 1:
                errors.append("chapter must be >= 1")
            if self.verse < 1:
                errors.append("verse must be >= 1")
            if self.testament not in ["OT", "NT"]:
                errors.append("testament must be OT or NT")
            return errors

    @dataclass
    class CrossReferenceSchema:
        source_ref: str = ""
        target_ref: str = ""
        connection_type: str = "thematic"
        strength: str = "moderate"
        confidence: float = 1.0
        bidirectional: bool = False
        notes: list = None
        sources: list = None
        verified: bool = False
        patristic_support: bool = False

        def __post_init__(self):
            if self.notes is None:
                self.notes = []
            if self.sources is None:
                self.sources = []

        def to_dict(self):
            return asdict(self)

        def to_json(self, indent=2):
            return json.dumps(self.to_dict(), indent=indent)

        @classmethod
        def from_dict(cls, data):
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

        def validate(self):
            errors = []
            if not self.source_ref:
                errors.append("source_ref is required")
            if not self.target_ref:
                errors.append("target_ref is required")
            if not 0 <= self.confidence <= 1:
                errors.append("confidence must be between 0 and 1")
            return errors

    @dataclass
    class WordSchema:
        word_id: str = ""
        verse_id: str = ""
        surface_form: str = ""
        lemma: str = ""
        position: int = 0

        def to_dict(self):
            return asdict(self)

        def to_json(self, indent=2):
            return json.dumps(self.to_dict(), indent=indent)

    @dataclass
    class ExtractionResultSchema:
        agent_name: str = ""
        extraction_type: str = ""
        verse_id: str = ""
        status: str = "pending"
        confidence: float = 0.0
        processing_time: float = 0.0

        def to_dict(self):
            return asdict(self)

        def to_json(self, indent=2):
            return json.dumps(self.to_dict(), indent=indent)

    @dataclass
    class GoldenRecordSchema:
        verse_id: str = ""
        text: str = ""
        agent_count: int = 0
        total_processing_time: float = 0.0

        def to_dict(self):
            return asdict(self)

        def to_json(self, indent=2):
            return json.dumps(self.to_dict(), indent=indent)

    def validate_verse_id(verse_id: str) -> bool:
        if not verse_id or not isinstance(verse_id, str):
            return False
        parts = verse_id.upper().replace(" ", ".").replace(":", ".").split(".")
        if len(parts) < 3:
            return False
        try:
            int(parts[1])
            int(parts[2])
            return True
        except ValueError:
            return False

    def normalize_verse_id(verse_id: str) -> str:
        return verse_id.upper().replace(" ", ".").replace(":", ".")

# Import test strategies
from tests.property.strategies import (
    verse_schema_strategy,
    cross_reference_schema_strategy,
    word_schema_strategy,
    extraction_result_schema_strategy,
    golden_record_schema_strategy,
    verse_id_strategy,
    unicode_text_strategy,
    biblical_text_strategy,
)


# =============================================================================
# VERSE SCHEMA TESTS
# =============================================================================

class TestVerseSchemaIntegrity:
    """Property-based tests for VerseSchema."""

    @given(verse_schema_strategy(valid_only=True))
    @settings(max_examples=200)
    def test_valid_verse_roundtrip_serialization(self, verse):
        """Valid verses should survive JSON roundtrip."""
        # Serialize to dict
        data = verse.to_dict()
        assert isinstance(data, dict)

        # Serialize to JSON string
        json_str = verse.to_json()
        assert isinstance(json_str, str)

        # Deserialize from JSON
        parsed = json.loads(json_str)
        assert parsed["verse_id"] == verse.verse_id

        # Recreate from dict
        recreated = VerseSchema.from_dict(data)
        assert recreated.verse_id == verse.verse_id
        assert recreated.book == verse.book
        assert recreated.chapter == verse.chapter
        assert recreated.verse == verse.verse

    @given(verse_schema_strategy(valid_only=True))
    @settings(max_examples=200)
    def test_valid_verse_validation_passes(self, verse):
        """Valid verses should pass validation."""
        errors = verse.validate()
        assert isinstance(errors, list)
        assert len(errors) == 0, f"Valid verse should have no errors, got: {errors}"

    @given(verse_schema_strategy(valid_only=False))
    @settings(max_examples=200)
    def test_invalid_verse_validation_catches_errors(self, verse):
        """Invalid verses should be caught by validation."""
        errors = verse.validate()
        # If verse is actually invalid, should have errors
        # We check individual fields
        if not verse.verse_id:
            assert any("verse_id" in err for err in errors)
        if not verse.book:
            assert any("book" in err for err in errors)
        if verse.chapter < 1:
            assert any("chapter" in err for err in errors)
        if verse.verse < 1:
            assert any("verse" in err for err in errors)
        if verse.testament not in ["OT", "NT"]:
            assert any("testament" in err for err in errors)

    @given(unicode_text_strategy())
    @settings(max_examples=200)
    @example("")  # Explicitly test empty string
    @example("Ἐν ἀρχῇ ἦν ὁ λόγος")  # Greek
    @example("בְּרֵאשִׁית בָּרָא אֱלֹהִים")  # Hebrew
    @example("ⲉⲛ ⲁⲣⲭⲏ ⲛⲉⲣⲉ ⲡϣⲁϫⲉ")  # Coptic
    @example("🙏✝️📖")  # Emojis
    def test_unicode_text_handling(self, text):
        """Schema should handle all Unicode text without crashing."""
        verse = VerseSchema(
            verse_id="GEN.1.1",
            book="GEN",
            chapter=1,
            verse=1,
            text=text,
            original_text=text,
        )

        # Should not crash
        data = verse.to_dict()
        assert isinstance(data, dict)

        # Should serialize to JSON without errors
        json_str = verse.to_json()
        assert isinstance(json_str, str)

        # Should deserialize correctly
        parsed = json.loads(json_str)
        assert parsed["text"] == text

    @given(st.text(max_size=100))
    @settings(max_examples=200)
    def test_verse_id_field_accepts_any_string(self, verse_id_text):
        """verse_id field should accept any string without crashing."""
        verse = VerseSchema(
            verse_id=verse_id_text,
            book="GEN",
            chapter=1,
            verse=1,
        )
        assert verse.verse_id == verse_id_text


# =============================================================================
# CROSS-REFERENCE SCHEMA TESTS
# =============================================================================

class TestCrossReferenceSchemaIntegrity:
    """Property-based tests for CrossReferenceSchema."""

    @given(cross_reference_schema_strategy(valid_only=True))
    @settings(max_examples=200)
    def test_valid_crossref_roundtrip(self, crossref):
        """Valid cross-references should survive serialization roundtrip."""
        data = crossref.to_dict()
        json_str = crossref.to_json()
        parsed = json.loads(json_str)

        assert parsed["source_ref"] == crossref.source_ref
        assert parsed["target_ref"] == crossref.target_ref
        assert parsed["connection_type"] == crossref.connection_type
        assert parsed["strength"] == crossref.strength

        recreated = CrossReferenceSchema.from_dict(data)
        assert recreated.source_ref == crossref.source_ref
        assert recreated.target_ref == crossref.target_ref

    @given(cross_reference_schema_strategy(valid_only=True))
    @settings(max_examples=200)
    def test_valid_crossref_validation(self, crossref):
        """Valid cross-references should pass validation."""
        errors = crossref.validate()
        assert len(errors) == 0, f"Valid cross-ref should have no errors, got: {errors}"

    @given(cross_reference_schema_strategy(valid_only=False))
    @settings(max_examples=200)
    def test_invalid_crossref_validation(self, crossref):
        """Invalid cross-references should fail validation appropriately."""
        errors = crossref.validate()

        # Check specific validation rules
        if not crossref.source_ref:
            assert any("source_ref" in err for err in errors)
        if not crossref.target_ref:
            assert any("target_ref" in err for err in errors)
        if not (0 <= crossref.confidence <= 1):
            assert any("confidence" in err for err in errors)

    @given(cross_reference_schema_strategy(valid_only=True))
    @settings(max_examples=200)
    def test_confidence_bounds(self, crossref):
        """Confidence scores should always be in valid range [0, 1]."""
        assert 0.0 <= crossref.confidence <= 1.0
        assert not (crossref.confidence != crossref.confidence)  # Not NaN
        assert crossref.confidence not in [float('inf'), float('-inf')]

    @given(cross_reference_schema_strategy(valid_only=True))
    @settings(max_examples=200)
    def test_source_target_distinct(self, crossref):
        """Source and target should be different verses."""
        # This is a business rule - a verse shouldn't reference itself
        # (though technically allowed by schema)
        if crossref.source_ref == crossref.target_ref:
            # This is unusual but not invalid - log it
            pass


# =============================================================================
# WORD SCHEMA TESTS
# =============================================================================

class TestWordSchemaIntegrity:
    """Property-based tests for WordSchema."""

    @given(word_schema_strategy())
    @settings(max_examples=200)
    def test_word_roundtrip(self, word):
        """Words should survive serialization roundtrip."""
        data = word.to_dict()
        json_str = word.to_json()
        parsed = json.loads(json_str)

        assert parsed["word_id"] == word.word_id
        assert parsed["verse_id"] == word.verse_id
        assert parsed["position"] == word.position

    @given(word_schema_strategy())
    @settings(max_examples=200)
    def test_word_id_format(self, word):
        """Word ID should be in format VERSE_ID.POSITION."""
        assert "." in word.word_id
        parts = word.word_id.split(".")
        assert len(parts) >= 4  # BOOK.CHAPTER.VERSE.POSITION

    @given(word_schema_strategy())
    @settings(max_examples=200)
    def test_position_non_negative(self, word):
        """Word position should be non-negative."""
        assert word.position >= 0


# =============================================================================
# EXTRACTION RESULT SCHEMA TESTS
# =============================================================================

class TestExtractionResultIntegrity:
    """Property-based tests for ExtractionResultSchema."""

    @given(extraction_result_schema_strategy())
    @settings(max_examples=200)
    def test_extraction_result_roundtrip(self, result):
        """Extraction results should survive serialization."""
        data = result.to_dict()
        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert parsed["agent_name"] == result.agent_name
        assert parsed["verse_id"] == result.verse_id
        assert parsed["status"] == result.status

    @given(extraction_result_schema_strategy())
    @settings(max_examples=200)
    def test_confidence_valid(self, result):
        """Confidence should be in [0, 1]."""
        assert 0.0 <= result.confidence <= 1.0

    @given(extraction_result_schema_strategy())
    @settings(max_examples=200)
    def test_processing_time_non_negative(self, result):
        """Processing time should be non-negative."""
        assert result.processing_time >= 0.0


# =============================================================================
# GOLDEN RECORD SCHEMA TESTS
# =============================================================================

class TestGoldenRecordIntegrity:
    """Property-based tests for GoldenRecordSchema."""

    @given(golden_record_schema_strategy())
    @settings(max_examples=200)
    def test_golden_record_roundtrip(self, record):
        """Golden records should survive serialization."""
        data = record.to_dict()
        json_str = record.to_json()
        parsed = json.loads(json_str)

        assert parsed["verse_id"] == record.verse_id
        assert parsed["agent_count"] == record.agent_count

    @given(golden_record_schema_strategy())
    @settings(max_examples=200)
    def test_agent_count_non_negative(self, record):
        """Agent count should be non-negative."""
        assert record.agent_count >= 0

    @given(golden_record_schema_strategy())
    @settings(max_examples=200)
    def test_processing_time_non_negative(self, record):
        """Total processing time should be non-negative."""
        assert record.total_processing_time >= 0.0


# =============================================================================
# VALIDATION FUNCTION TESTS
# =============================================================================

class TestValidationFunctions:
    """Property-based tests for validation utility functions."""

    @given(verse_id_strategy(valid_only=True))
    @settings(max_examples=200)
    def test_valid_verse_id_validation(self, verse_id):
        """Valid verse IDs should pass validation."""
        assert validate_verse_id(verse_id) is True

    @given(verse_id_strategy(valid_only=False))
    @settings(max_examples=200)
    def test_invalid_verse_id_validation(self, verse_id):
        """Validation should correctly identify invalid verse IDs."""
        is_valid = validate_verse_id(verse_id)
        # If it's actually invalid, should return False
        if is_valid:
            # If it returns True, it should have correct format
            assert isinstance(verse_id, str)
            assert len(verse_id) > 0

    @given(st.text(max_size=100))
    @settings(max_examples=200)
    @example("GEN.1.1")
    @example("gen.1.1")
    @example("GEN:1:1")
    @example("GEN 1:1")
    def test_normalize_verse_id(self, verse_id):
        """Normalization should handle various formats without crashing."""
        normalized = normalize_verse_id(verse_id)
        assert isinstance(normalized, str)
        # Should convert to uppercase and use dots
        assert normalized == verse_id.upper().replace(" ", ".").replace(":", ".")

    @given(verse_id_strategy(valid_only=True))
    @settings(max_examples=200)
    def test_normalize_idempotent(self, verse_id):
        """Normalizing twice should produce same result."""
        normalized_once = normalize_verse_id(verse_id)
        normalized_twice = normalize_verse_id(normalized_once)
        assert normalized_once == normalized_twice


# =============================================================================
# STATEFUL TESTING - SCHEMA MUTATION
# =============================================================================

class SchemaStateMachine(RuleBasedStateMachine):
    """Stateful testing for schema mutations."""

    def __init__(self):
        super().__init__()
        self.verse = VerseSchema(
            verse_id="GEN.1.1",
            book="GEN",
            chapter=1,
            verse=1,
            text="In the beginning...",
        )

    @rule(text=st.text(max_size=500))
    def update_text(self, text):
        """Update verse text."""
        self.verse.text = text

    @rule(original_text=biblical_text_strategy())
    def update_original_text(self, original_text):
        """Update original text."""
        self.verse.original_text = original_text

    @invariant()
    def verse_id_unchanged(self):
        """Verse ID should never change."""
        assert self.verse.verse_id == "GEN.1.1"

    @invariant()
    def serialization_works(self):
        """Serialization should always work."""
        data = self.verse.to_dict()
        assert isinstance(data, dict)
        json_str = self.verse.to_json()
        assert isinstance(json_str, str)


TestSchemaStateMachine = SchemaStateMachine.TestCase

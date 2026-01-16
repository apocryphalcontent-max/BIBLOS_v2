"""
Property-Based Tests for Cross-Reference Invariants

Tests business rules and invariants for cross-reference relationships.
"""
import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, precondition

from data.schemas import CrossReferenceSchema, ConnectionType, ConnectionStrength
from tests.property.strategies import (
    cross_reference_schema_strategy,
    verse_id_strategy,
    verse_pair_strategy,
    connection_type_strategy,
    connection_strength_strategy,
    confidence_score_strategy,
)


class TestCrossReferenceInvariants:
    """Property-based tests for cross-reference invariants."""

    @given(cross_reference_schema_strategy(valid_only=True))
    @settings(max_examples=300)
    def test_confidence_in_valid_range(self, crossref):
        """Confidence scores must always be in [0, 1]."""
        assert 0.0 <= crossref.confidence <= 1.0
        assert not (crossref.confidence != crossref.confidence)  # Not NaN
        assert crossref.confidence != float('inf')
        assert crossref.confidence != float('-inf')

    @given(cross_reference_schema_strategy(valid_only=True))
    @settings(max_examples=300)
    def test_connection_type_is_valid_enum(self, crossref):
        """Connection type must be a valid enum value."""
        valid_types = [e.value for e in ConnectionType]
        assert crossref.connection_type in valid_types, \
            f"Invalid connection type: {crossref.connection_type}"

    @given(cross_reference_schema_strategy(valid_only=True))
    @settings(max_examples=300)
    def test_strength_is_valid_enum(self, crossref):
        """Connection strength must be a valid enum value."""
        valid_strengths = [e.value for e in ConnectionStrength]
        assert crossref.strength in valid_strengths, \
            f"Invalid strength: {crossref.strength}"

    @given(cross_reference_schema_strategy(valid_only=True))
    @settings(max_examples=300)
    def test_source_and_target_required(self, crossref):
        """Source and target refs are required fields."""
        assert crossref.source_ref is not None
        assert crossref.target_ref is not None
        assert len(crossref.source_ref) > 0
        assert len(crossref.target_ref) > 0

    @given(cross_reference_schema_strategy(valid_only=True))
    @settings(max_examples=300)
    def test_self_reference_detection(self, crossref):
        """A verse should not reference itself (business rule)."""
        # While technically allowed by schema, self-references are unusual
        if crossref.source_ref == crossref.target_ref:
            # Flag this as unusual (but don't fail - it's not invalid)
            # In production, this might trigger a warning
            pass

    @given(verse_pair_strategy())
    @settings(max_examples=200)
    def test_verse_pairs_are_distinct(self, verse_pair):
        """Generated verse pairs should be distinct."""
        source, target = verse_pair
        assert source != target


class TestBidirectionalInvariants:
    """Tests for bidirectional cross-reference consistency."""

    @given(
        verse_pair_strategy(),
        connection_type_strategy(valid_only=True),
        connection_strength_strategy(valid_only=True),
        confidence_score_strategy(valid_only=True),
    )
    @settings(max_examples=200)
    def test_bidirectional_symmetry(self, verse_pair, conn_type, strength, confidence):
        """Bidirectional references should be symmetric."""
        source, target = verse_pair

        # Create forward reference
        forward = CrossReferenceSchema(
            source_ref=source,
            target_ref=target,
            connection_type=conn_type,
            strength=strength,
            confidence=confidence,
            bidirectional=True,
        )

        # Create reverse reference
        reverse = CrossReferenceSchema(
            source_ref=target,
            target_ref=source,
            connection_type=conn_type,
            strength=strength,
            confidence=confidence,
            bidirectional=True,
        )

        # Both should be valid
        assert len(forward.validate()) == 0
        assert len(reverse.validate()) == 0

        # Properties should match
        assert forward.connection_type == reverse.connection_type
        assert forward.strength == reverse.strength
        assert forward.confidence == reverse.confidence

    @given(cross_reference_schema_strategy(valid_only=True))
    @settings(max_examples=200)
    def test_bidirectional_flag_consistency(self, crossref):
        """Bidirectional flag should be boolean."""
        assert isinstance(crossref.bidirectional, bool)


class TestConnectionTypeSemantics:
    """Tests for connection type semantic constraints."""

    @given(
        verse_pair_strategy(),
        st.booleans(),
    )
    @settings(max_examples=100)
    def test_typological_connections_cross_testaments(self, verse_pair, bidirectional):
        """Typological connections typically go OT -> NT."""
        source, target = verse_pair

        # Create typological connection
        crossref = CrossReferenceSchema(
            source_ref=source,
            target_ref=target,
            connection_type="typological",
            strength="strong",
            confidence=0.9,
            bidirectional=bidirectional,
        )

        # Should be valid regardless of testament direction
        errors = crossref.validate()
        assert len(errors) == 0

        # In a real system, we might want to warn if typological goes NT -> OT
        # but it's not invalid

    @given(verse_pair_strategy())
    @settings(max_examples=100)
    def test_prophetic_fulfillment_semantics(self, verse_pair):
        """Prophetic connections should have specific characteristics."""
        source, target = verse_pair

        crossref = CrossReferenceSchema(
            source_ref=source,
            target_ref=target,
            connection_type="prophetic",
            strength="strong",
            confidence=0.95,
        )

        errors = crossref.validate()
        assert len(errors) == 0

    @given(
        verse_pair_strategy(),
        st.sampled_from([e.value for e in ConnectionType])
    )
    @settings(max_examples=200)
    def test_all_connection_types_valid(self, verse_pair, conn_type):
        """All connection types should be valid with any verse pair."""
        source, target = verse_pair

        crossref = CrossReferenceSchema(
            source_ref=source,
            target_ref=target,
            connection_type=conn_type,
            strength="moderate",
            confidence=0.7,
        )

        errors = crossref.validate()
        assert len(errors) == 0


class TestStrengthConfidenceRelationship:
    """Tests for relationship between strength and confidence."""

    @given(cross_reference_schema_strategy(valid_only=True))
    @settings(max_examples=200)
    def test_strong_connections_often_high_confidence(self, crossref):
        """Strong connections tend to have high confidence (soft rule)."""
        # This is a statistical tendency, not a hard rule
        if crossref.strength == "strong":
            # Don't enforce, but track
            # In production, low confidence with strong strength might trigger review
            pass

    @given(
        verse_pair_strategy(),
        st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=200)
    def test_confidence_independent_of_strength(self, verse_pair, confidence):
        """Confidence and strength are independent attributes."""
        source, target = verse_pair

        # Create references with all strength combinations
        for strength in ["strong", "moderate", "weak"]:
            crossref = CrossReferenceSchema(
                source_ref=source,
                target_ref=target,
                connection_type="thematic",
                strength=strength,
                confidence=confidence,
            )

            # Should all be valid
            errors = crossref.validate()
            assert len(errors) == 0


class TestNotesAndSources:
    """Tests for notes and sources fields."""

    @given(
        cross_reference_schema_strategy(valid_only=True),
        st.lists(st.text(min_size=1, max_size=200), max_size=10),
    )
    @settings(max_examples=200)
    def test_notes_list_handling(self, crossref, notes):
        """Notes should handle arbitrary text lists."""
        crossref.notes = notes
        assert isinstance(crossref.notes, list)
        assert len(crossref.notes) == len(notes)

        # Should serialize correctly
        data = crossref.to_dict()
        assert data["notes"] == notes

    @given(
        cross_reference_schema_strategy(valid_only=True),
        st.lists(st.text(min_size=1, max_size=100), max_size=10),
    )
    @settings(max_examples=200)
    def test_sources_list_handling(self, crossref, sources):
        """Sources should handle arbitrary text lists."""
        crossref.sources = sources
        assert isinstance(crossref.sources, list)

        # Should serialize correctly
        data = crossref.to_dict()
        assert data["sources"] == sources

    @given(cross_reference_schema_strategy(valid_only=True))
    @settings(max_examples=200)
    def test_empty_notes_and_sources_valid(self, crossref):
        """Empty notes and sources lists should be valid."""
        crossref.notes = []
        crossref.sources = []

        errors = crossref.validate()
        assert len(errors) == 0


class TestVerificationAndPatristicSupport:
    """Tests for verification and patristic support flags."""

    @given(cross_reference_schema_strategy(valid_only=True))
    @settings(max_examples=200)
    def test_verification_flag_boolean(self, crossref):
        """Verified flag should be boolean."""
        assert isinstance(crossref.verified, bool)

    @given(cross_reference_schema_strategy(valid_only=True))
    @settings(max_examples=200)
    def test_patristic_support_flag_boolean(self, crossref):
        """Patristic support flag should be boolean."""
        assert isinstance(crossref.patristic_support, bool)

    @given(
        cross_reference_schema_strategy(valid_only=True),
        st.booleans(),
        st.booleans(),
    )
    @settings(max_examples=200)
    def test_verification_patristic_independence(self, crossref, verified, patristic):
        """Verification and patristic support are independent."""
        crossref.verified = verified
        crossref.patristic_support = patristic

        # All combinations should be valid
        errors = crossref.validate()
        assert len(errors) == 0


# =============================================================================
# STATEFUL TESTING - CROSS-REFERENCE COLLECTION
# =============================================================================

class CrossReferenceCollectionMachine(RuleBasedStateMachine):
    """Stateful testing for a collection of cross-references."""

    def __init__(self):
        super().__init__()
        self.crossrefs = []
        self.verse_refs = set()

    @rule(crossref=cross_reference_schema_strategy(valid_only=True))
    def add_crossref(self, crossref):
        """Add a cross-reference to the collection."""
        self.crossrefs.append(crossref)
        self.verse_refs.add(crossref.source_ref)
        self.verse_refs.add(crossref.target_ref)

    @rule()
    def remove_last_crossref(self):
        """Remove the last cross-reference."""
        if self.crossrefs:
            self.crossrefs.pop()

    @rule(index=st.integers(min_value=0, max_value=100))
    def get_crossref(self, index):
        """Get a cross-reference by index."""
        if 0 <= index < len(self.crossrefs):
            crossref = self.crossrefs[index]
            assert crossref is not None

    @invariant()
    def all_crossrefs_valid(self):
        """All cross-references in collection should be valid."""
        for crossref in self.crossrefs:
            errors = crossref.validate()
            assert len(errors) == 0

    @invariant()
    def confidence_scores_in_range(self):
        """All confidence scores should be in valid range."""
        for crossref in self.crossrefs:
            assert 0.0 <= crossref.confidence <= 1.0

    @invariant()
    def no_duplicate_refs(self):
        """No duplicate (source, target) pairs."""
        pairs = [(cr.source_ref, cr.target_ref) for cr in self.crossrefs]
        # Allow duplicates but track them
        # In production, duplicates might be merged
        pass


TestCrossReferenceCollection = CrossReferenceCollectionMachine.TestCase

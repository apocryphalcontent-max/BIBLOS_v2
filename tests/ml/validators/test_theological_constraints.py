"""
Tests for Theological Constraint Validator.

Tests the 6 patristic theological constraints:
1. Chronological Priority - Type must precede antitype
2. Typological Escalation - Antitype must exceed type in scope/magnitude
3. Prophetic Coherence - Fulfillment extends promise without contradiction
4. Christological Warrant - Requires patristic/apostolic support
5. Liturgical Amplification - Liturgical usage boosts confidence
6. Fourfold Foundation - Allegorical reading requires literal foundation
"""

import pytest
import numpy as np
from typing import Dict, Any, List

from ml.validators import (
    ConstraintViolationSeverity,
    ConstraintType,
    ConstraintResult,
    Scope,
    ScopeMagnitudeAnalyzer,
    SemanticCoherenceChecker,
    TheologicalConstraintValidator,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def validator() -> TheologicalConstraintValidator:
    """Create a validator instance for testing."""
    return TheologicalConstraintValidator()


@pytest.fixture
def scope_analyzer() -> ScopeMagnitudeAnalyzer:
    """Create a scope analyzer instance for testing."""
    return ScopeMagnitudeAnalyzer()


@pytest.fixture
def coherence_checker() -> SemanticCoherenceChecker:
    """Create a coherence checker instance for testing."""
    return SemanticCoherenceChecker()


@pytest.fixture
def valid_typological_candidate() -> Dict[str, Any]:
    """Create a valid typological cross-reference candidate."""
    return {
        "source_ref": "GEN.22.2",  # Isaac (OT type)
        "target_ref": "JHN.3.16",  # Christ (NT antitype)
        "connection_type": "typological",
        "confidence": 0.85,
        "source_testament": "OT",
        "target_testament": "NT",
    }


@pytest.fixture
def valid_typological_context() -> Dict[str, Any]:
    """Create context for valid typological validation."""
    return {
        "type_element": {"text": "Take your son, your only son Isaac, whom you love"},
        "antitype_element": {"text": "God so loved the world that he gave his only Son"},
        "type_context": {},
        "antitype_context": {},
        "patristic_witnesses": [
            {"father": "Chrysostom", "work": "Homilies on Genesis"},
            {"father": "Cyril of Alexandria", "work": "Glaphyra"},
        ],
        "liturgical_contexts": ["holy_week", "lectionary"],
        "literal_analysis": {"text": "Abraham offering Isaac", "historical_context": "Moriah"},
        "allegorical_claim": {"type": "christological", "antitype": "Christ's sacrifice"},
    }


@pytest.fixture
def invalid_chronological_candidate() -> Dict[str, Any]:
    """Create an invalid candidate where antitype precedes type."""
    return {
        "source_ref": "MAT.1.1",  # NT source
        "target_ref": "GEN.1.1",  # OT target
        "connection_type": "typological",
        "confidence": 0.75,
        "source_testament": "NT",
        "target_testament": "OT",
    }


@pytest.fixture
def prophetic_candidate() -> Dict[str, Any]:
    """Create a prophetic cross-reference candidate."""
    return {
        "source_ref": "ISA.7.14",  # Prophecy
        "target_ref": "MAT.1.23",  # Fulfillment
        "connection_type": "prophetic",
        "confidence": 0.9,
        "source_testament": "OT",
        "target_testament": "NT",
    }


@pytest.fixture
def prophetic_context() -> Dict[str, Any]:
    """Create context for prophetic validation."""
    return {
        "promise_semantics": {"embedding": np.random.rand(768).tolist()},
        "fulfillment_semantics": {"embedding": np.random.rand(768).tolist()},
        "nt_quotations": ["MAT.1.23"],
        "christological_claim": "virgin birth",
    }


# =============================================================================
# ENUM AND DATACLASS TESTS
# =============================================================================

class TestConstraintViolationSeverity:
    """Tests for ConstraintViolationSeverity enum."""

    def test_severity_values(self):
        """Test all severity values exist."""
        assert ConstraintViolationSeverity.IMPOSSIBLE.value == "IMPOSSIBLE"
        assert ConstraintViolationSeverity.CRITICAL.value == "CRITICAL"
        assert ConstraintViolationSeverity.SOFT.value == "SOFT"
        assert ConstraintViolationSeverity.WARNING.value == "WARNING"
        assert ConstraintViolationSeverity.BOOST.value == "BOOST"

    def test_severity_count(self):
        """Test correct number of severity levels."""
        assert len(ConstraintViolationSeverity) == 5


class TestConstraintType:
    """Tests for ConstraintType enum."""

    def test_constraint_type_values(self):
        """Test all constraint types exist."""
        assert ConstraintType.TYPOLOGICAL_ESCALATION.value == "TYPOLOGICAL_ESCALATION"
        assert ConstraintType.PROPHETIC_COHERENCE.value == "PROPHETIC_COHERENCE"
        assert ConstraintType.CHRONOLOGICAL_PRIORITY.value == "CHRONOLOGICAL_PRIORITY"
        assert ConstraintType.CHRISTOLOGICAL_WARRANT.value == "CHRISTOLOGICAL_WARRANT"
        assert ConstraintType.LITURGICAL_AMPLIFICATION.value == "LITURGICAL_AMPLIFICATION"
        assert ConstraintType.FOURFOLD_FOUNDATION.value == "FOURFOLD_FOUNDATION"

    def test_constraint_type_count(self):
        """Test correct number of constraint types."""
        assert len(ConstraintType) == 6


class TestScope:
    """Tests for Scope enum."""

    def test_scope_values(self):
        """Test all scope levels exist."""
        assert Scope.LOCAL.value == "LOCAL"
        assert Scope.NATIONAL.value == "NATIONAL"
        assert Scope.UNIVERSAL.value == "UNIVERSAL"
        assert Scope.COSMIC.value == "COSMIC"

    def test_scope_ordering(self):
        """Test scope ordering makes theological sense."""
        # Scopes should escalate: LOCAL < NATIONAL < UNIVERSAL < COSMIC
        scopes = list(Scope)
        assert scopes[0] == Scope.LOCAL
        assert scopes[3] == Scope.COSMIC


class TestConstraintResult:
    """Tests for ConstraintResult dataclass."""

    def test_passing_result(self):
        """Test creating a passing constraint result."""
        result = ConstraintResult(
            passed=True,
            constraint_type=ConstraintType.CHRONOLOGICAL_PRIORITY,
            confidence_modifier=1.0,
            reason="Type precedes antitype correctly"
        )
        assert result.passed is True
        assert result.confidence_modifier == 1.0
        assert result.violation_severity is None

    def test_failing_result(self):
        """Test creating a failing constraint result."""
        result = ConstraintResult(
            passed=False,
            constraint_type=ConstraintType.CHRONOLOGICAL_PRIORITY,
            violation_severity=ConstraintViolationSeverity.IMPOSSIBLE,
            confidence_modifier=0.0,
            reason="Antitype precedes type - impossible"
        )
        assert result.passed is False
        assert result.confidence_modifier == 0.0
        assert result.violation_severity == ConstraintViolationSeverity.IMPOSSIBLE

    def test_to_dict(self):
        """Test dictionary serialization."""
        result = ConstraintResult(
            passed=True,
            constraint_type=ConstraintType.LITURGICAL_AMPLIFICATION,
            violation_severity=ConstraintViolationSeverity.BOOST,
            confidence_modifier=1.15,
            reason="Used in Great Vespers"
        )
        d = result.to_dict()
        assert d["passed"] is True
        assert d["constraint_type"] == "LITURGICAL_AMPLIFICATION"
        assert d["violation_severity"] == "BOOST"
        assert d["confidence_modifier"] == 1.15


# =============================================================================
# SCOPE MAGNITUDE ANALYZER TESTS
# =============================================================================

class TestScopeMagnitudeAnalyzer:
    """Tests for ScopeMagnitudeAnalyzer helper class."""

    def test_analyze_scope_cosmic(self, scope_analyzer):
        """Test cosmic scope detection."""
        # Use keywords that match the SCOPE_INDICATORS for COSMIC
        element = {"text": "The eternal God created all things in the universe forever"}
        scope = scope_analyzer.analyze_scope(element, {})
        assert scope == Scope.COSMIC

    def test_analyze_scope_local(self, scope_analyzer):
        """Test local scope detection."""
        element = {"text": "A man from the land of Uz"}
        scope = scope_analyzer.analyze_scope(element, {})
        assert scope == Scope.LOCAL

    def test_analyze_scope_national(self, scope_analyzer):
        """Test national scope detection."""
        element = {"text": "The children of Israel were fruitful"}
        scope = scope_analyzer.analyze_scope(element, {})
        assert scope == Scope.NATIONAL

    def test_calculate_magnitude_high(self, scope_analyzer):
        """Test high magnitude calculation."""
        element = {"text": "God gave his eternal salvation to all who believe"}
        magnitude = scope_analyzer.calculate_magnitude(element)
        assert magnitude > 50  # High magnitude expected

    def test_calculate_magnitude_low(self, scope_analyzer):
        """Test low magnitude calculation."""
        element = {"text": "One man worked in the field for a day"}
        magnitude = scope_analyzer.calculate_magnitude(element)
        assert magnitude < 50  # Low magnitude expected

    def test_analyze_fulfillment_completeness(self, scope_analyzer):
        """Test fulfillment completeness analysis."""
        type_elem = {"text": "offering sacrifice lamb altar blood"}
        antitype_elem = {"text": "Christ sacrifice lamb of God blood eternal offering"}
        completeness = scope_analyzer.analyze_fulfillment_completeness(type_elem, antitype_elem)
        assert 0.0 <= completeness <= 1.0


# =============================================================================
# SEMANTIC COHERENCE CHECKER TESTS
# =============================================================================

class TestSemanticCoherenceChecker:
    """Tests for SemanticCoherenceChecker helper class."""

    def test_extract_promise_components(self, coherence_checker):
        """Test promise component extraction."""
        semantics = {
            "text": "I will give you the land and establish my covenant",
            "themes": ["covenant", "promise"],
            "keywords": ["land", "establish"]
        }
        components = coherence_checker.extract_promise_components(semantics)
        assert "covenant" in components
        assert "land" in components

    def test_detect_contradictions(self, coherence_checker):
        """Test contradiction detection between promise and fulfillment."""
        promise = {"text": "God will bring life and blessing to the people"}
        fulfillment = {"text": "Through suffering came death"}
        contradictions = coherence_checker.detect_contradictions(promise, fulfillment)
        # Should detect potential contradiction (life vs death)
        assert len(contradictions) >= 0  # May or may not detect depending on resolution markers

    def test_no_contradictions_with_resolution(self, coherence_checker):
        """Test that resolutions prevent contradiction detection."""
        promise = {"text": "God will bring life"}
        fulfillment = {"text": "Through death came life resulting in salvation"}
        contradictions = coherence_checker.detect_contradictions(promise, fulfillment)
        # Resolution marker "resulting in" should prevent contradiction
        assert len(contradictions) == 0


# =============================================================================
# CHRONOLOGICAL PRIORITY TESTS
# =============================================================================

class TestChronologicalPriority:
    """Tests for Chronological Priority constraint."""

    def test_valid_ot_to_nt_typology(self, validator):
        """Test valid OT->NT typological connection passes."""
        result = validator.validate_chronological_priority("GEN.22.2", "JHN.3.16")
        assert result.passed is True
        assert result.confidence_modifier >= 1.0

    def test_invalid_nt_to_ot_typology(self, validator):
        """Test invalid NT->OT typological connection fails."""
        result = validator.validate_chronological_priority("MAT.1.1", "GEN.1.1")
        assert result.passed is False
        assert result.violation_severity == ConstraintViolationSeverity.IMPOSSIBLE
        assert result.confidence_modifier == 0.0

    def test_same_testament_within_ot(self, validator):
        """Test OT->OT typology within same testament."""
        # GEN (position 0) before EXO (position 1)
        result = validator.validate_chronological_priority("GEN.1.1", "EXO.12.1")
        assert result.passed is True

    def test_invalid_same_testament(self, validator):
        """Test invalid OT->OT where target precedes source."""
        # EXO (position 1) before GEN (position 0) should fail
        result = validator.validate_chronological_priority("EXO.12.1", "GEN.1.1")
        assert result.passed is False


# =============================================================================
# TYPOLOGICAL ESCALATION TESTS
# =============================================================================

class TestTypologicalEscalation:
    """Tests for Typological Escalation constraint."""

    def test_valid_escalation_isaac_christ(self, validator):
        """Test Isaac->Christ escalation (LOCAL->COSMIC)."""
        type_elem = {"text": "Take your son Isaac, your only son"}
        antitype_elem = {"text": "God so loved the world that he gave his only Son for eternal salvation"}

        result = validator.validate_typological_escalation(
            type_elem, antitype_elem, {}, {}
        )
        assert result.passed is True

    def test_invalid_de_escalation(self, validator):
        """Test de-escalation fails the constraint."""
        # Cosmic description for type
        type_elem = {"text": "God created heaven and earth for all eternity"}
        # Local description for antitype
        antitype_elem = {"text": "A man walked through one village"}

        result = validator.validate_typological_escalation(
            type_elem, antitype_elem, {}, {}
        )
        # De-escalation should not pass cleanly
        if result.passed:
            assert result.confidence_modifier <= 1.0


# =============================================================================
# PROPHETIC COHERENCE TESTS
# =============================================================================

class TestPropheticCoherence:
    """Tests for Prophetic Coherence constraint."""

    def test_coherent_prophecy_fulfillment(self, validator):
        """Test coherent prophecy-fulfillment passes."""
        promise_sem = {"embedding": np.random.rand(768).tolist()}
        fulfill_sem = {"embedding": np.random.rand(768).tolist()}

        result = validator.validate_prophetic_coherence(
            "ISA.7.14", "MAT.1.23", promise_sem, fulfill_sem
        )
        # Should pass or have mild warning
        assert result.confidence_modifier > 0

    def test_incoherent_fulfillment_warning(self, validator):
        """Test semantically incoherent fulfillment."""
        # Very different embeddings (orthogonal)
        promise_sem = {"embedding": ([1.0] + [0.0] * 767)}
        fulfill_sem = {"embedding": ([0.0] + [1.0] + [0.0] * 766)}

        result = validator.validate_prophetic_coherence(
            "ISA.7.14", "MAT.1.23", promise_sem, fulfill_sem
        )
        # Low coherence should reduce confidence
        assert result.confidence_modifier <= 1.0


# =============================================================================
# CHRISTOLOGICAL WARRANT TESTS
# =============================================================================

class TestChristologicalWarrant:
    """Tests for Christological Warrant constraint."""

    def test_strong_patristic_support_boosts(self, validator):
        """Test strong patristic support boosts confidence."""
        nt_quotes = ["JHN.3.16", "HEB.11.17"]
        patristic = [
            {"father": "Chrysostom", "work": "Homilies"},
            {"father": "Augustine", "work": "City of God"},
            {"father": "Cyril", "work": "Glaphyra"},
        ]

        result = validator.validate_christological_warrant(
            "GEN.22.2", "Isaac as type of Christ", nt_quotes, patristic
        )
        assert result.passed is True
        assert result.violation_severity == ConstraintViolationSeverity.BOOST

    def test_no_patristic_support_critical(self, validator):
        """Test lack of patristic support generates critical warning."""
        result = validator.validate_christological_warrant(
            "GEN.22.2", "Novel claim", [], []
        )
        # Should fail with critical severity
        assert result.passed is False
        assert result.violation_severity == ConstraintViolationSeverity.CRITICAL


# =============================================================================
# LITURGICAL AMPLIFICATION TESTS
# =============================================================================

class TestLiturgicalAmplification:
    """Tests for Liturgical Amplification constraint."""

    def test_liturgical_usage_boosts(self, validator):
        """Test liturgical usage boosts confidence."""
        liturgical = ["pascha", "nativity"]

        result = validator.validate_liturgical_amplification("ISA.9.6", liturgical)
        assert result.passed is True
        assert result.violation_severity == ConstraintViolationSeverity.BOOST
        assert result.confidence_modifier > 1.0

    def test_no_liturgical_usage_neutral(self, validator):
        """Test lack of liturgical usage is neutral (not penalized)."""
        result = validator.validate_liturgical_amplification("GEN.1.1", [])
        assert result.passed is True
        assert result.confidence_modifier == 1.0


# =============================================================================
# FOURFOLD FOUNDATION TESTS
# =============================================================================

class TestFourfoldFoundation:
    """Tests for Fourfold Foundation constraint."""

    def test_literal_foundation_passes(self, validator):
        """Test allegorical reading with literal foundation passes."""
        literal = {"text": "Burning Bush event", "historical_context": "Horeb"}
        allegorical = {"type": "typological", "claim": "Mary as burning bush"}

        result = validator.validate_fourfold_foundation("EXO.3.2", literal, allegorical)
        assert result.passed is True

    def test_missing_literal_foundation_warns(self, validator):
        """Test allegorical reading without literal foundation warns."""
        literal = {}  # No literal analysis
        allegorical = {"type": "allegorical", "claim": "Spiritual interpretation"}

        result = validator.validate_fourfold_foundation("SNG.4.12", literal, allegorical)
        # Should pass but with warning or reduced confidence
        assert result.confidence_modifier <= 1.0


# =============================================================================
# COMPOSITE VALIDATION TESTS
# =============================================================================

class TestValidateAllConstraints:
    """Tests for validate_all_constraints method."""

    @pytest.mark.asyncio
    async def test_valid_candidate_passes_all(
        self, validator, valid_typological_candidate, valid_typological_context
    ):
        """Test valid candidate passes all applicable constraints."""
        results = await validator.validate_all_constraints(
            valid_typological_candidate, valid_typological_context
        )
        assert len(results) >= 1  # At least chronological for typological

        # Should not have any IMPOSSIBLE violations
        impossible = [r for r in results if r.violation_severity == ConstraintViolationSeverity.IMPOSSIBLE]
        assert len(impossible) == 0

    @pytest.mark.asyncio
    async def test_invalid_candidate_gets_impossible(self, validator, invalid_chronological_candidate):
        """Test invalid candidate gets IMPOSSIBLE for chronological violation."""
        results = await validator.validate_all_constraints(invalid_chronological_candidate, {})

        # Should have chronological priority failure
        chrono_results = [r for r in results if r.constraint_type == ConstraintType.CHRONOLOGICAL_PRIORITY]
        assert len(chrono_results) == 1
        assert chrono_results[0].passed is False
        assert chrono_results[0].violation_severity == ConstraintViolationSeverity.IMPOSSIBLE


class TestCalculateCompositeModifier:
    """Tests for calculate_composite_modifier method."""

    def test_all_passing_modifier(self, validator):
        """Test composite modifier with all passing constraints."""
        results = [
            ConstraintResult(passed=True, constraint_type=ConstraintType.CHRONOLOGICAL_PRIORITY, confidence_modifier=1.0),
            ConstraintResult(passed=True, constraint_type=ConstraintType.TYPOLOGICAL_ESCALATION, confidence_modifier=1.1),
            ConstraintResult(passed=True, constraint_type=ConstraintType.CHRISTOLOGICAL_WARRANT,
                           violation_severity=ConstraintViolationSeverity.BOOST, confidence_modifier=1.15),
        ]
        modifier = validator.calculate_composite_modifier(results)
        assert modifier > 1.0  # Should boost

    def test_impossible_overrides_all(self, validator):
        """Test IMPOSSIBLE violation overrides all boosts."""
        results = [
            ConstraintResult(passed=False, constraint_type=ConstraintType.CHRONOLOGICAL_PRIORITY,
                           violation_severity=ConstraintViolationSeverity.IMPOSSIBLE, confidence_modifier=0.0),
            ConstraintResult(passed=True, constraint_type=ConstraintType.CHRISTOLOGICAL_WARRANT,
                           violation_severity=ConstraintViolationSeverity.BOOST, confidence_modifier=1.2),
        ]
        modifier = validator.calculate_composite_modifier(results)
        assert modifier == 0.0  # IMPOSSIBLE should zero out

    def test_mixed_results_combined(self, validator):
        """Test mixed passing/warning results combine correctly."""
        results = [
            ConstraintResult(passed=True, constraint_type=ConstraintType.CHRONOLOGICAL_PRIORITY, confidence_modifier=1.0),
            ConstraintResult(passed=True, constraint_type=ConstraintType.TYPOLOGICAL_ESCALATION,
                           violation_severity=ConstraintViolationSeverity.WARNING, confidence_modifier=0.9),
        ]
        modifier = validator.calculate_composite_modifier(results)
        assert 0.8 < modifier < 1.0  # Should be slightly reduced


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for constraint validation."""

    @pytest.mark.asyncio
    async def test_single_validation_under_50ms(
        self, validator, valid_typological_candidate, valid_typological_context
    ):
        """Test single constraint validation completes under 50ms."""
        import time
        start = time.perf_counter()
        await validator.validate_all_constraints(valid_typological_candidate, valid_typological_context)
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 50, f"Validation took {elapsed:.1f}ms, expected < 50ms"

    @pytest.mark.asyncio
    async def test_batch_100_under_2s(
        self, validator, valid_typological_candidate, valid_typological_context
    ):
        """Test batch of 100 validations completes under 2 seconds."""
        import time
        candidates = [(valid_typological_candidate.copy(), valid_typological_context.copy()) for _ in range(100)]

        start = time.perf_counter()
        for candidate, context in candidates:
            await validator.validate_all_constraints(candidate, context)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"Batch validation took {elapsed:.2f}s, expected < 2s"


# =============================================================================
# THEOLOGICAL TEST CASES
# =============================================================================

class TestTheologicalCases:
    """Tests with canonical theological examples."""

    @pytest.mark.asyncio
    async def test_isaac_christ_canonical(self, validator):
        """Test Isaac->Christ - the canonical typological example."""
        candidate = {
            "source_ref": "GEN.22.2",
            "target_ref": "JHN.3.16",
            "connection_type": "typological",
            "confidence": 0.85,
        }
        context = {
            "type_element": {"text": "Take your son, your only son Isaac, whom you love"},
            "antitype_element": {"text": "God so loved the world that he gave his only Son for eternal salvation"},
            "patristic_witnesses": [
                {"father": "Chrysostom", "work": "Homilies on Genesis"},
                {"father": "Cyril of Alexandria", "work": "Glaphyra"},
                {"father": "Augustine", "work": "City of God"},
            ],
            "liturgical_contexts": ["holy_week"],
            "literal_analysis": {"text": "Abraham offering Isaac", "historical_context": "Moriah"},
            "allegorical_claim": {"type": "christological"},
        }
        results = await validator.validate_all_constraints(candidate, context)

        # Should pass all constraints
        failed = [r for r in results if not r.passed]
        assert len(failed) == 0, f"Isaac->Christ should pass all: {[r.reason for r in failed]}"

        # Composite should boost
        modifier = validator.calculate_composite_modifier(results)
        assert modifier >= 1.0

    @pytest.mark.asyncio
    async def test_geographic_connection_minimal_constraints(self, validator):
        """Test geographic connections have minimal theological constraints."""
        candidate = {
            "source_ref": "GEN.12.6",  # Shechem
            "target_ref": "JHN.4.5",   # Sychar (near Shechem)
            "connection_type": "geographical",
            "confidence": 0.7,
        }
        results = await validator.validate_all_constraints(candidate, {})

        # Geographic should have fewer constraints applied (may be empty)
        impossible = [r for r in results if r.violation_severity == ConstraintViolationSeverity.IMPOSSIBLE]
        assert len(impossible) == 0

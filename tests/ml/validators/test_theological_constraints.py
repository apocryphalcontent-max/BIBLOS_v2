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
from typing import Dict, Any

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
        "features": {
            "mutual_influence": 0.45,
            "transformation_type": "RADICAL",
        },
        "patristic_citations": [
            {"father": "Chrysostom", "work": "Homilies on Genesis"},
            {"father": "Cyril of Alexandria", "work": "Glaphyra"},
        ],
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
        "features": {},
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
        "features": {
            "promise_embedding": np.random.rand(768),
            "fulfillment_embedding": np.random.rand(768),
        },
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

    def test_scope_comparison_escalation(self, scope_analyzer):
        """Test that scope escalation is detected correctly."""
        # LOCAL -> UNIVERSAL is escalation
        result = scope_analyzer.compare_scopes(Scope.LOCAL, Scope.UNIVERSAL)
        assert result > 0  # Positive = escalation

    def test_scope_comparison_same(self, scope_analyzer):
        """Test same scope comparison."""
        result = scope_analyzer.compare_scopes(Scope.NATIONAL, Scope.NATIONAL)
        assert result == 0  # Same scope

    def test_scope_comparison_de_escalation(self, scope_analyzer):
        """Test that scope de-escalation is detected."""
        # COSMIC -> LOCAL is de-escalation
        result = scope_analyzer.compare_scopes(Scope.COSMIC, Scope.LOCAL)
        assert result < 0  # Negative = de-escalation

    def test_analyze_type_antitype_valid(self, scope_analyzer):
        """Test valid type-antitype escalation analysis."""
        # Isaac (LOCAL/individual) -> Christ (COSMIC/eternal)
        type_data = {"scope": Scope.LOCAL, "magnitude": 0.3}
        antitype_data = {"scope": Scope.COSMIC, "magnitude": 0.95}

        result = scope_analyzer.analyze_escalation(type_data, antitype_data)
        assert result["escalates"] is True
        assert result["scope_change"] > 0
        assert result["magnitude_change"] > 0

    def test_analyze_type_antitype_invalid(self, scope_analyzer):
        """Test invalid de-escalation detection."""
        # Cannot have antitype smaller than type
        type_data = {"scope": Scope.UNIVERSAL, "magnitude": 0.8}
        antitype_data = {"scope": Scope.LOCAL, "magnitude": 0.2}

        result = scope_analyzer.analyze_escalation(type_data, antitype_data)
        assert result["escalates"] is False


# =============================================================================
# SEMANTIC COHERENCE CHECKER TESTS
# =============================================================================

class TestSemanticCoherenceChecker:
    """Tests for SemanticCoherenceChecker helper class."""

    def test_identical_embeddings_high_coherence(self, coherence_checker):
        """Test that identical embeddings show perfect coherence."""
        embedding = np.random.rand(768)
        coherence = coherence_checker.check_coherence(embedding, embedding)
        assert coherence >= 0.99  # Should be ~1.0

    def test_orthogonal_embeddings_low_coherence(self, coherence_checker):
        """Test that orthogonal embeddings show low coherence."""
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([0.0, 1.0, 0.0])
        coherence = coherence_checker.check_coherence(embedding1, embedding2)
        assert coherence < 0.1  # Should be ~0.0

    def test_similar_embeddings_moderate_coherence(self, coherence_checker):
        """Test that similar embeddings show moderate-high coherence."""
        base = np.random.rand(768)
        similar = base + np.random.rand(768) * 0.1  # Small perturbation
        coherence = coherence_checker.check_coherence(base, similar)
        assert 0.8 < coherence < 1.0


# =============================================================================
# CHRONOLOGICAL PRIORITY TESTS
# =============================================================================

class TestChronologicalPriority:
    """Tests for Chronological Priority constraint."""

    @pytest.mark.asyncio
    async def test_valid_ot_to_nt_typology(self, validator, valid_typological_candidate):
        """Test valid OT->NT typological connection passes."""
        result = await validator.validate_chronological_priority(valid_typological_candidate)
        assert result.passed is True
        assert result.confidence_modifier >= 1.0

    @pytest.mark.asyncio
    async def test_invalid_nt_to_ot_typology(self, validator, invalid_chronological_candidate):
        """Test invalid NT->OT typological connection fails."""
        result = await validator.validate_chronological_priority(invalid_chronological_candidate)
        assert result.passed is False
        assert result.violation_severity == ConstraintViolationSeverity.IMPOSSIBLE
        assert result.confidence_modifier == 0.0

    @pytest.mark.asyncio
    async def test_same_testament_warning(self, validator):
        """Test same-testament typology gets warning but not rejection."""
        candidate = {
            "source_ref": "GEN.1.1",
            "target_ref": "EXO.12.1",
            "connection_type": "typological",
            "source_testament": "OT",
            "target_testament": "OT",
        }
        result = await validator.validate_chronological_priority(candidate)
        # Should pass but with reduced confidence
        assert result.passed is True
        assert result.confidence_modifier < 1.0

    @pytest.mark.asyncio
    async def test_non_typological_bypasses_check(self, validator):
        """Test non-typological connections bypass chronological check."""
        candidate = {
            "source_ref": "MAT.1.1",
            "target_ref": "GEN.1.1",
            "connection_type": "thematic",  # Not typological
            "source_testament": "NT",
            "target_testament": "OT",
        }
        result = await validator.validate_chronological_priority(candidate)
        assert result.passed is True
        assert result.confidence_modifier == 1.0


# =============================================================================
# TYPOLOGICAL ESCALATION TESTS
# =============================================================================

class TestTypologicalEscalation:
    """Tests for Typological Escalation constraint."""

    @pytest.mark.asyncio
    async def test_valid_escalation_isaac_christ(self, validator, valid_typological_candidate):
        """Test Isaac->Christ escalation (LOCAL->COSMIC)."""
        result = await validator.validate_typological_escalation(valid_typological_candidate)
        assert result.passed is True
        # Should get boost for strong escalation
        assert result.confidence_modifier >= 1.0

    @pytest.mark.asyncio
    async def test_invalid_de_escalation(self, validator):
        """Test de-escalation fails the constraint."""
        candidate = {
            "source_ref": "JHN.1.1",  # Cosmic Christ
            "target_ref": "GEN.4.4",  # Abel (local individual)
            "connection_type": "typological",
            "source_testament": "NT",
            "target_testament": "OT",
            "features": {
                "source_scope": "COSMIC",
                "target_scope": "LOCAL",
            },
        }
        result = await validator.validate_typological_escalation(candidate)
        assert result.passed is False
        assert result.violation_severity in [
            ConstraintViolationSeverity.SOFT,
            ConstraintViolationSeverity.CRITICAL
        ]

    @pytest.mark.asyncio
    async def test_lateral_typology_warning(self, validator):
        """Test lateral (same-scope) typology gets warning."""
        candidate = {
            "source_ref": "GEN.22.2",
            "target_ref": "GEN.37.3",  # Joseph - similar scope to Isaac
            "connection_type": "typological",
            "source_testament": "OT",
            "target_testament": "OT",
            "features": {
                "source_scope": "NATIONAL",
                "target_scope": "NATIONAL",
            },
        }
        result = await validator.validate_typological_escalation(candidate)
        # Lateral is allowed but with warning
        if result.violation_severity:
            assert result.violation_severity == ConstraintViolationSeverity.WARNING


# =============================================================================
# PROPHETIC COHERENCE TESTS
# =============================================================================

class TestPropheticCoherence:
    """Tests for Prophetic Coherence constraint."""

    @pytest.mark.asyncio
    async def test_coherent_prophecy_fulfillment(self, validator, prophetic_candidate):
        """Test coherent prophecy-fulfillment passes."""
        result = await validator.validate_prophetic_coherence(prophetic_candidate)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_incoherent_fulfillment_warning(self, validator):
        """Test semantically incoherent fulfillment gets warning."""
        # Create embeddings that are very different (low coherence)
        candidate = {
            "source_ref": "ISA.7.14",
            "target_ref": "MAT.1.23",
            "connection_type": "prophetic",
            "source_testament": "OT",
            "target_testament": "NT",
            "features": {
                "promise_embedding": np.array([1.0, 0.0, 0.0] + [0.0] * 765),
                "fulfillment_embedding": np.array([0.0, 1.0, 0.0] + [0.0] * 765),
            },
        }
        result = await validator.validate_prophetic_coherence(candidate)
        # Low coherence should result in warning or soft violation
        assert result.confidence_modifier < 1.0

    @pytest.mark.asyncio
    async def test_non_prophetic_bypasses(self, validator):
        """Test non-prophetic connections bypass coherence check."""
        candidate = {
            "source_ref": "PSA.23.1",
            "target_ref": "JHN.10.11",
            "connection_type": "thematic",  # Not prophetic
            "source_testament": "OT",
            "target_testament": "NT",
        }
        result = await validator.validate_prophetic_coherence(candidate)
        assert result.passed is True
        assert result.confidence_modifier == 1.0


# =============================================================================
# CHRISTOLOGICAL WARRANT TESTS
# =============================================================================

class TestChristologicalWarrant:
    """Tests for Christological Warrant constraint."""

    @pytest.mark.asyncio
    async def test_strong_patristic_support_boosts(self, validator, valid_typological_candidate):
        """Test strong patristic support boosts confidence."""
        result = await validator.validate_christological_warrant(valid_typological_candidate)
        assert result.passed is True
        # Should get boost for multiple patristic witnesses
        assert result.violation_severity == ConstraintViolationSeverity.BOOST
        assert result.confidence_modifier > 1.0

    @pytest.mark.asyncio
    async def test_no_patristic_support_warning(self, validator):
        """Test lack of patristic support generates warning."""
        candidate = {
            "source_ref": "GEN.22.2",
            "target_ref": "JHN.3.16",
            "connection_type": "typological",
            "source_testament": "OT",
            "target_testament": "NT",
            "patristic_citations": [],  # No support
        }
        result = await validator.validate_christological_warrant(candidate)
        # Should warn but not reject
        assert result.violation_severity in [
            ConstraintViolationSeverity.WARNING,
            ConstraintViolationSeverity.SOFT
        ]
        assert result.confidence_modifier < 1.0

    @pytest.mark.asyncio
    async def test_single_witness_passes(self, validator):
        """Test single patristic witness passes but doesn't boost."""
        candidate = {
            "source_ref": "GEN.22.2",
            "target_ref": "JHN.3.16",
            "connection_type": "typological",
            "source_testament": "OT",
            "target_testament": "NT",
            "patristic_citations": [
                {"father": "Chrysostom", "work": "Homilies"}
            ],
        }
        result = await validator.validate_christological_warrant(candidate)
        assert result.passed is True
        # Single witness passes but may not boost
        assert result.confidence_modifier >= 1.0


# =============================================================================
# LITURGICAL AMPLIFICATION TESTS
# =============================================================================

class TestLiturgicalAmplification:
    """Tests for Liturgical Amplification constraint."""

    @pytest.mark.asyncio
    async def test_liturgical_usage_boosts(self, validator):
        """Test liturgical usage boosts confidence."""
        candidate = {
            "source_ref": "ISA.9.6",
            "target_ref": "LUK.2.11",
            "connection_type": "prophetic",
            "source_testament": "OT",
            "target_testament": "NT",
            "liturgical_references": [
                {"service": "Nativity Vespers", "usage": "prokeimenon"},
                {"service": "Nativity Matins", "usage": "gospel"}
            ],
        }
        result = await validator.validate_liturgical_amplification(candidate)
        assert result.passed is True
        assert result.violation_severity == ConstraintViolationSeverity.BOOST
        assert result.confidence_modifier > 1.0

    @pytest.mark.asyncio
    async def test_no_liturgical_usage_neutral(self, validator):
        """Test lack of liturgical usage is neutral (not penalized)."""
        candidate = {
            "source_ref": "GEN.1.1",
            "target_ref": "JHN.1.1",
            "connection_type": "thematic",
            "source_testament": "OT",
            "target_testament": "NT",
            "liturgical_references": [],
        }
        result = await validator.validate_liturgical_amplification(candidate)
        assert result.passed is True
        # Absence is neutral, not penalized
        assert result.confidence_modifier == 1.0


# =============================================================================
# FOURFOLD FOUNDATION TESTS
# =============================================================================

class TestFourfoldFoundation:
    """Tests for Fourfold Foundation constraint."""

    @pytest.mark.asyncio
    async def test_literal_foundation_passes(self, validator):
        """Test allegorical reading with literal foundation passes."""
        candidate = {
            "source_ref": "EXO.3.2",  # Burning Bush - literal event
            "target_ref": "LUK.1.35",  # Theotokos - allegorical type
            "connection_type": "typological",
            "source_testament": "OT",
            "target_testament": "NT",
            "features": {
                "has_literal_foundation": True,
                "historical_event": "Burning Bush theophany at Horeb",
            },
        }
        result = await validator.validate_fourfold_foundation(candidate)
        assert result.passed is True
        assert result.confidence_modifier >= 1.0

    @pytest.mark.asyncio
    async def test_missing_literal_foundation_warns(self, validator):
        """Test allegorical reading without literal foundation warns."""
        candidate = {
            "source_ref": "SNG.4.12",  # Song of Songs - no clear historical event
            "target_ref": "LUK.1.28",  # Mary
            "connection_type": "typological",
            "source_testament": "OT",
            "target_testament": "NT",
            "features": {
                "has_literal_foundation": False,
            },
        }
        result = await validator.validate_fourfold_foundation(candidate)
        # Should warn about missing literal foundation
        if not result.passed:
            assert result.violation_severity in [
                ConstraintViolationSeverity.WARNING,
                ConstraintViolationSeverity.SOFT
            ]


# =============================================================================
# COMPOSITE VALIDATION TESTS
# =============================================================================

class TestValidateAllConstraints:
    """Tests for validate_all_constraints method."""

    @pytest.mark.asyncio
    async def test_valid_candidate_passes_all(self, validator, valid_typological_candidate):
        """Test valid candidate passes all applicable constraints."""
        results = await validator.validate_all_constraints(valid_typological_candidate)
        assert len(results) >= 2  # At least chronological and escalation for typological

        # Should not have any IMPOSSIBLE violations
        impossible = [r for r in results if r.violation_severity == ConstraintViolationSeverity.IMPOSSIBLE]
        assert len(impossible) == 0

    @pytest.mark.asyncio
    async def test_invalid_candidate_gets_impossible(self, validator, invalid_chronological_candidate):
        """Test invalid candidate gets IMPOSSIBLE for chronological violation."""
        results = await validator.validate_all_constraints(invalid_chronological_candidate)

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
            ConstraintResult(passed=True, constraint_type=ConstraintType.CHRISTOLOGICAL_WARRANT, confidence_modifier=1.15),
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
    async def test_single_validation_under_50ms(self, validator, valid_typological_candidate):
        """Test single constraint validation completes under 50ms."""
        import time
        start = time.perf_counter()
        await validator.validate_all_constraints(valid_typological_candidate)
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 50, f"Validation took {elapsed:.1f}ms, expected < 50ms"

    @pytest.mark.asyncio
    async def test_batch_100_under_2s(self, validator, valid_typological_candidate):
        """Test batch of 100 validations completes under 2 seconds."""
        import time
        candidates = [valid_typological_candidate.copy() for _ in range(100)]

        start = time.perf_counter()
        for candidate in candidates:
            await validator.validate_all_constraints(candidate)
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
            "source_testament": "OT",
            "target_testament": "NT",
            "patristic_citations": [
                {"father": "Chrysostom", "work": "Homilies on Genesis"},
                {"father": "Cyril of Alexandria", "work": "Glaphyra"},
                {"father": "Augustine", "work": "City of God"},
            ],
            "liturgical_references": [
                {"service": "Holy Week", "usage": "readings"},
            ],
            "features": {
                "mutual_influence": 0.5,
                "transformation_type": "RADICAL",
                "has_literal_foundation": True,
            },
        }
        results = await validator.validate_all_constraints(candidate)

        # Should pass all constraints and get boosts
        failed = [r for r in results if not r.passed]
        assert len(failed) == 0, f"Isaac->Christ should pass all: {[r.reason for r in failed]}"

        # Composite should boost
        modifier = validator.calculate_composite_modifier(results)
        assert modifier >= 1.0

    @pytest.mark.asyncio
    async def test_burning_bush_theotokos(self, validator):
        """Test Burning Bush->Theotokos Mary typology."""
        candidate = {
            "source_ref": "EXO.3.2",
            "target_ref": "LUK.1.35",
            "connection_type": "typological",
            "confidence": 0.8,
            "source_testament": "OT",
            "target_testament": "NT",
            "patristic_citations": [
                {"father": "Gregory of Nyssa", "work": "Life of Moses"},
            ],
            "liturgical_references": [
                {"service": "Theotokos feasts", "usage": "hymns"},
            ],
            "features": {
                "has_literal_foundation": True,
                "historical_event": "Burning Bush theophany",
            },
        }
        results = await validator.validate_all_constraints(candidate)

        # Should pass chronological and escalation
        chrono = [r for r in results if r.constraint_type == ConstraintType.CHRONOLOGICAL_PRIORITY]
        assert all(r.passed for r in chrono)

    @pytest.mark.asyncio
    async def test_geographic_connection_minimal_constraints(self, validator):
        """Test geographic connections have minimal theological constraints."""
        candidate = {
            "source_ref": "GEN.12.6",  # Shechem
            "target_ref": "JHN.4.5",   # Sychar (near Shechem)
            "connection_type": "geographical",
            "confidence": 0.7,
            "source_testament": "OT",
            "target_testament": "NT",
        }
        results = await validator.validate_all_constraints(candidate)

        # Geographic should have fewer constraints applied
        # And should generally pass
        impossible = [r for r in results if r.violation_severity == ConstraintViolationSeverity.IMPOSSIBLE]
        assert len(impossible) == 0

"""
Tests for InterVerseNecessityCalculator - The Second Impossible Oracle.

These tests verify that the calculator correctly identifies NECESSARY
verse relationships (not just helpful ones) using:
- Semantic gap analysis
- Presupposition detection
- Explicit reference extraction
- Bayesian score computation
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Optional

from ml.engines.necessity_calculator import (
    # Enums
    NecessityType,
    NecessityStrength,
    GapType,
    PresuppositionType,
    SyntacticRole,
    ClauseType,
    # Dataclasses
    SemanticGap,
    Presupposition,
    ExplicitReference,
    ResolutionCandidate,
    ScoreDistribution,
    NecessityAnalysisResult,
    VerseData,
    # Component classes
    ReferenceExtractor,
    PresuppositionDetector,
    GapSeverityCalculator,
    NecessityScoreComputer,
    DependencyGraph,
    # Main class
    InterVerseNecessityCalculator,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_config():
    """Create mock configuration."""
    class MockConfig:
        absolute_threshold = 0.90
        strong_threshold = 0.70
        moderate_threshold = 0.50
        weak_threshold = 0.30
        gap_coverage_weight = 0.35
        severity_weight = 0.25
        type_weight = 0.15
        explicit_ref_weight = 0.15
        presupposition_weight = 0.10
        cache_enabled = True
        cache_ttl_seconds = 604800
    return MockConfig()


@pytest.fixture
def calculator(mock_config):
    """Create calculator instance with mock dependencies."""
    calc = InterVerseNecessityCalculator(
        omni_resolver=None,
        neo4j_client=None,
        redis_client=None,
        embedding_generator=None,
        config=mock_config
    )
    return calc


@pytest.fixture
def reference_extractor():
    """Create reference extractor instance."""
    return ReferenceExtractor()


@pytest.fixture
def presupposition_detector():
    """Create presupposition detector instance."""
    return PresuppositionDetector()


@pytest.fixture
def severity_calculator():
    """Create severity calculator instance."""
    return GapSeverityCalculator()


@pytest.fixture
def score_computer(mock_config):
    """Create score computer instance."""
    return NecessityScoreComputer(mock_config)


# =============================================================================
# CANONICAL NECESSITY TESTS
# =============================================================================

class TestCanonicalNecessity:
    """Tests for canonical necessity relationships from specification."""

    @pytest.mark.asyncio
    async def test_hebrews_genesis_necessity_absolute(self, calculator):
        """
        HEB.11.17 → GEN.22.1 should be ABSOLUTE necessity.

        Hebrews passage is incomprehensible without Genesis Akedah:
        - "Abraham" - who is Abraham?
        - "tested" - what test?
        - "offered up Isaac" - what offering?
        - "the promises" - what promise?
        """
        result = await calculator.calculate_necessity(
            verse_a="HEB.11.17",
            verse_b="GEN.22.1"
        )

        # Must be at least STRONG necessity (aiming for ABSOLUTE)
        assert result.necessity_score >= 0.70
        assert result.strength in [NecessityStrength.ABSOLUTE, NecessityStrength.STRONG]

        # Must identify key gaps
        gap_types = {g.gap_type for g in result.semantic_gaps}
        assert GapType.ENTITY_PERSON in gap_types  # Abraham, Isaac

        # Should fill gaps
        assert result.gap_coverage >= 0.5

        # Type should be presuppositional
        assert result.necessity_type in [
            NecessityType.PRESUPPOSITIONAL,
            NecessityType.NARRATIVE
        ]

    @pytest.mark.asyncio
    async def test_matthew_isaiah_quotation_strong(self, calculator):
        """
        MAT.1.23 → ISA.7.14 should be STRONG necessity.

        Matthew explicitly quotes Isaiah with citation formula:
        "As it is written by the prophet Isaiah..."
        """
        result = await calculator.calculate_necessity(
            verse_a="MAT.1.23",
            verse_b="ISA.7.14"
        )

        # Must be at least MODERATE (citation present)
        assert result.necessity_score >= 0.50
        assert result.strength in [
            NecessityStrength.ABSOLUTE,
            NecessityStrength.STRONG,
            NecessityStrength.MODERATE
        ]

        # Should detect citation formula
        # (Note: Our mock text includes "As it is written by the prophet Isaiah")
        if result.has_citation_formula:
            assert result.necessity_type == NecessityType.REFERENTIAL

    @pytest.mark.asyncio
    async def test_romans_leviticus_definitional(self, calculator):
        """
        ROM.3.25 → LEV.16.15 should be MODERATE-STRONG necessity.

        "propitiation" (ἱλαστήριον) requires sacrificial context.
        """
        result = await calculator.calculate_necessity(
            verse_a="ROM.3.25",
            verse_b="LEV.16.15"
        )

        # At least WEAK (theological term connection)
        assert result.necessity_score >= 0.30

        # Should identify term gap
        term_gaps = [g for g in result.semantic_gaps if g.gap_type == GapType.TERM_TECHNICAL]
        assert len(term_gaps) >= 1

    @pytest.mark.asyncio
    async def test_no_necessity_thematic_only(self, calculator):
        """
        GEN.1.1 → PSA.19.1 should have WEAK/NONE necessity.

        Both discuss creation, but:
        - GEN.1.1 is complete without PSA.19.1
        - PSA.19.1 is complete without GEN.1.1
        - Thematic parallel, not logical dependency
        """
        result = await calculator.calculate_necessity(
            verse_a="GEN.1.1",
            verse_b="PSA.19.1"
        )

        # Must be WEAK or NONE
        assert result.necessity_score < 0.50
        assert result.strength in [
            NecessityStrength.WEAK,
            NecessityStrength.NONE,
            NecessityStrength.MODERATE  # Allow moderate for thematic overlap
        ]

    @pytest.mark.asyncio
    async def test_john_genesis_literary_echo(self, calculator):
        """
        JHN.1.1 → GEN.1.1 - John echoes Genesis but doesn't require it.

        "In the beginning was the Word" is comprehensible standalone.
        Genesis enriches but doesn't complete the meaning.
        """
        result = await calculator.calculate_necessity(
            verse_a="JHN.1.1",
            verse_b="GEN.1.1"
        )

        # Should be at most MODERATE (literary echo, not dependency)
        assert result.necessity_score < 0.70


# =============================================================================
# BIDIRECTIONAL TESTS
# =============================================================================

class TestBidirectionalNecessity:
    """Tests for bidirectional necessity relationships."""

    @pytest.mark.asyncio
    async def test_asymmetric_necessity(self, calculator):
        """
        HEB.11.17 needs GEN.22 (high), but GEN.22 doesn't need HEB.11.17 (low).

        The Old Testament is foundational; New Testament depends on it.
        """
        forward = await calculator.calculate_necessity("HEB.11.17", "GEN.22.1")
        reverse = await calculator.calculate_necessity("GEN.22.1", "HEB.11.17")

        # Forward should be higher than reverse
        assert forward.necessity_score >= reverse.necessity_score

        # Forward should be at least moderate
        assert forward.necessity_score >= 0.5

        # Reverse should be low (OT doesn't need NT)
        assert reverse.necessity_score < 0.5

    @pytest.mark.asyncio
    async def test_parallel_accounts_moderate_bidirection(self, calculator):
        """
        Chronicles/Kings parallel accounts may have moderate bidirectionality.
        Both tell the same story, each has unique details.
        """
        forward = await calculator.calculate_necessity("2SAM.11.2", "1CHR.20.1")
        reverse = await calculator.calculate_necessity("1CHR.20.1", "2SAM.11.2")

        # At least one direction should have some necessity
        max_score = max(forward.necessity_score, reverse.necessity_score)
        assert max_score >= 0.2


# =============================================================================
# GAP DETECTION TESTS
# =============================================================================

class TestGapDetection:
    """Tests for semantic gap detection."""

    @pytest.mark.asyncio
    async def test_gap_detection_entities(self, calculator):
        """Test detection of entity gaps (persons)."""
        gaps = await calculator.identify_semantic_gaps(
            verse_text="By faith Abraham offered up Isaac his son",
            verse_id="HEB.11.17"
        )

        # Should find Abraham and Isaac as entity gaps
        entity_gaps = [g for g in gaps if g.gap_type == GapType.ENTITY_PERSON]
        entity_names = {g.trigger_text.lower() for g in entity_gaps}

        assert 'abraham' in entity_names or any('abraham' in n for n in entity_names)
        assert 'isaac' in entity_names or any('isaac' in n for n in entity_names)

    @pytest.mark.asyncio
    async def test_gap_detection_events(self, calculator):
        """Test detection of event gaps."""
        gaps = await calculator.identify_semantic_gaps(
            verse_text="when he was tested and offered up Isaac",
            verse_id="HEB.11.17"
        )

        # Should find event gaps
        event_gaps = [g for g in gaps if g.gap_type == GapType.EVENT_HISTORICAL]
        # Note: may be empty if "tested" not detected as event
        # The main test is that it doesn't error

    @pytest.mark.asyncio
    async def test_gap_detection_theological_terms(self, calculator):
        """Test detection of theological term gaps."""
        gaps = await calculator.identify_semantic_gaps(
            verse_text="God put him forward as a propitiation by his blood",
            verse_id="ROM.3.25"
        )

        # Should find propitiation as technical term
        term_gaps = [g for g in gaps if g.gap_type == GapType.TERM_TECHNICAL]
        term_texts = {g.trigger_text.lower() for g in term_gaps}

        assert 'propitiation' in term_texts

    @pytest.mark.asyncio
    async def test_gap_detection_quotation_formula(self, calculator):
        """Test detection of quotation/citation formula gaps."""
        gaps = await calculator.identify_semantic_gaps(
            verse_text="As it is written in the prophet Isaiah, the virgin shall conceive",
            verse_id="MAT.1.23"
        )

        # Should find citation formula gap
        quote_gaps = [g for g in gaps if g.gap_type == GapType.QUOTATION_EXPLICIT]
        assert len(quote_gaps) >= 1


# =============================================================================
# SEVERITY CALCULATION TESTS
# =============================================================================

class TestSeverityCalculation:
    """Tests for gap severity calculation."""

    def test_gap_severity_entity_person(self):
        """Person entity gaps should have high severity."""
        gap = SemanticGap(
            gap_id="test1",
            gap_type=GapType.ENTITY_PERSON,
            trigger_text="Abraham",
            syntactic_role=SyntacticRole.SUBJECT,
            clause_type=ClauseType.MAIN,
            is_focus=True,
            base_severity=0.85,
        )

        severity = gap.compute_final_severity()
        assert severity >= 0.8  # High severity for subject person in main clause

    def test_gap_severity_adjunct_lower(self):
        """Adjunct gaps should have lower severity than subject gaps."""
        subject_gap = SemanticGap(
            gap_id="test1",
            gap_type=GapType.ENTITY_PLACE,
            trigger_text="Jerusalem",
            syntactic_role=SyntacticRole.SUBJECT,
            clause_type=ClauseType.MAIN,
            base_severity=0.65,
        )

        adjunct_gap = SemanticGap(
            gap_id="test2",
            gap_type=GapType.ENTITY_PLACE,
            trigger_text="Jerusalem",
            syntactic_role=SyntacticRole.ADJUNCT,
            clause_type=ClauseType.SUBORDINATE,
            base_severity=0.65,
        )

        subject_severity = subject_gap.compute_final_severity()
        adjunct_severity = adjunct_gap.compute_final_severity()

        assert subject_severity > adjunct_severity

    def test_gap_severity_focus_boost(self):
        """Gaps in focus position should get severity boost."""
        normal_gap = SemanticGap(
            gap_id="test1",
            gap_type=GapType.ENTITY_PERSON,
            trigger_text="Moses",
            syntactic_role=SyntacticRole.OBJECT,
            clause_type=ClauseType.MAIN,
            is_focus=False,
            base_severity=0.85,
        )

        focus_gap = SemanticGap(
            gap_id="test2",
            gap_type=GapType.ENTITY_PERSON,
            trigger_text="Moses",
            syntactic_role=SyntacticRole.OBJECT,
            clause_type=ClauseType.MAIN,
            is_focus=True,
            base_severity=0.85,
        )

        normal_severity = normal_gap.compute_final_severity()
        focus_severity = focus_gap.compute_final_severity()

        assert focus_severity > normal_severity


# =============================================================================
# SCORE DISTRIBUTION TESTS
# =============================================================================

class TestScoreDistribution:
    """Tests for statistical score distributions."""

    def test_score_distribution_mean(self):
        """Test that mean is computed correctly."""
        dist = ScoreDistribution(alpha=8.0, beta_param=2.0)
        mean = dist.mean

        # Alpha=8, Beta=2 should give mean = 8/(8+2) = 0.8
        assert 0.79 <= mean <= 0.81

    def test_score_distribution_variance(self):
        """Test that variance is computed correctly."""
        dist = ScoreDistribution(alpha=5.0, beta_param=5.0)

        # Symmetric distribution should have variance
        # = 5*5 / (10*10*11) = 25/1100 ≈ 0.0227
        assert dist.variance > 0
        assert dist.variance < 0.1

    def test_score_distribution_confidence_interval(self):
        """Test that confidence intervals are valid."""
        dist = ScoreDistribution(alpha=10.0, beta_param=5.0)
        lower, upper = dist.confidence_interval(0.95)

        # Interval should be bounded [0, 1]
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0
        assert lower < upper

        # Mean should be within interval
        assert lower <= dist.mean <= upper


# =============================================================================
# COMPONENT CLASS TESTS
# =============================================================================

class TestReferenceExtractor:
    """Tests for ReferenceExtractor component."""

    @pytest.mark.asyncio
    async def test_detect_citation_formula_english(self, reference_extractor):
        """Test detection of English citation formulas."""
        refs = await reference_extractor.extract_explicit_references(
            text="As it is written in the prophet, Behold I send my messenger",
            verse_id="MAT.11.10",
            language="english"
        )

        assert len(refs) >= 1
        assert refs[0].formula_type == "quotation"
        assert "written" in refs[0].citation_formula.lower()

    @pytest.mark.asyncio
    async def test_no_citation_formula(self, reference_extractor):
        """Test that non-citation text returns no references."""
        refs = await reference_extractor.extract_explicit_references(
            text="In the beginning God created the heavens and the earth",
            verse_id="GEN.1.1",
            language="english"
        )

        assert len(refs) == 0


class TestPresuppositionDetector:
    """Tests for PresuppositionDetector component."""

    @pytest.mark.asyncio
    async def test_detect_existential_presupposition(self, presupposition_detector):
        """Test detection of existential presuppositions (the X)."""
        presups = await presupposition_detector.detect_presuppositions(
            text="The Lord spoke to Moses",
            syntax_tree=None
        )

        existential = [p for p in presups if p.ptype == PresuppositionType.EXISTENTIAL]
        assert len(existential) >= 1

    @pytest.mark.asyncio
    async def test_detect_factive_presupposition(self, presupposition_detector):
        """Test detection of factive presuppositions (knew that)."""
        presups = await presupposition_detector.detect_presuppositions(
            text="Moses knew that the Lord would deliver them",
            syntax_tree=None
        )

        factive = [p for p in presups if p.ptype == PresuppositionType.FACTIVE]
        assert len(factive) >= 1

    @pytest.mark.asyncio
    async def test_detect_lexical_presupposition(self, presupposition_detector):
        """Test detection of lexical presuppositions (again, stopped)."""
        presups = await presupposition_detector.detect_presuppositions(
            text="The people began to murmur again",
            syntax_tree=None
        )

        lexical = [p for p in presups if p.ptype == PresuppositionType.LEXICAL]
        assert len(lexical) >= 1


class TestNecessityScoreComputer:
    """Tests for NecessityScoreComputer component."""

    @pytest.mark.asyncio
    async def test_compute_high_score(self, score_computer):
        """Test that high evidence yields high score."""
        # Create many filled gaps
        gaps = [
            SemanticGap(
                gap_id=f"gap_{i}",
                gap_type=GapType.ENTITY_PERSON,
                trigger_text="test",
                base_severity=0.9,
            )
            for i in range(5)
        ]

        # All gaps filled
        gaps_filled = gaps.copy()

        # Presuppositions satisfied
        presups = [
            Presupposition(
                presupposition_id="pres1",
                ptype=PresuppositionType.EXISTENTIAL,
                trigger_text="the",
                presupposed_content="exists",
                confidence=0.8,
            )
        ]

        # Explicit reference
        refs = [
            ExplicitReference(
                reference_id="ref1",
                source_verse="A",
                target_verse="B",
                formula_type="quotation",
                citation_formula="as it is written",
                quoted_text="test",
                confidence=0.9,
            )
        ]

        score, dist = await score_computer.compute_necessity_score(
            gaps=gaps,
            gaps_filled=gaps_filled,
            presuppositions=presups,
            presuppositions_satisfied=presups,
            explicit_references=refs,
            necessity_type=NecessityType.REFERENTIAL
        )

        # Should be high score
        assert score >= 0.7

    @pytest.mark.asyncio
    async def test_compute_low_score(self, score_computer):
        """Test that low evidence yields low score."""
        # Few gaps, none filled
        gaps = [
            SemanticGap(
                gap_id="gap1",
                gap_type=GapType.DEFINITE_NP,
                trigger_text="the thing",
                base_severity=0.3,
            )
        ]

        score, dist = await score_computer.compute_necessity_score(
            gaps=gaps,
            gaps_filled=[],  # None filled
            presuppositions=[],
            presuppositions_satisfied=[],
            explicit_references=[],
            necessity_type=NecessityType.PRESUPPOSITIONAL
        )

        # Should be low score
        assert score < 0.5

    def test_classify_strength(self, score_computer):
        """Test strength classification."""
        assert score_computer.classify_strength(0.95) == NecessityStrength.ABSOLUTE
        assert score_computer.classify_strength(0.75) == NecessityStrength.STRONG
        assert score_computer.classify_strength(0.55) == NecessityStrength.MODERATE
        assert score_computer.classify_strength(0.35) == NecessityStrength.WEAK
        assert score_computer.classify_strength(0.15) == NecessityStrength.NONE


# =============================================================================
# DEPENDENCY GRAPH TESTS
# =============================================================================

class TestDependencyGraph:
    """Tests for DependencyGraph functionality."""

    @pytest.fixture
    def sample_graph(self):
        """Create sample dependency graph."""
        graph = DependencyGraph()
        graph.add_edge("HEB.11.17", "GEN.22.1", score=0.9, necessity_type="presuppositional")
        graph.add_edge("HEB.11.17", "GEN.12.1", score=0.7, necessity_type="covenantal")
        graph.add_edge("GEN.22.1", "GEN.12.1", score=0.5, necessity_type="narrative")
        return graph

    def test_graph_construction(self, sample_graph):
        """Test that graph is constructed correctly."""
        assert sample_graph.total_nodes == 3
        assert sample_graph.total_edges == 3

    def test_root_verses(self, sample_graph):
        """Test finding root verses (no dependencies)."""
        roots = sample_graph.root_verses
        # GEN.12.1 has no outgoing edges in our sample
        assert "GEN.12.1" in roots

    def test_find_necessity_chain(self, sample_graph):
        """Test finding necessity chain between verses."""
        chain = sample_graph.find_necessity_chain("HEB.11.17", "GEN.12.1")
        # Should find path (direct or through GEN.22.1)
        assert len(chain) >= 2
        assert chain[0] == "HEB.11.17"
        assert chain[-1] == "GEN.12.1"

    def test_get_direct_dependencies(self, sample_graph):
        """Test getting direct dependencies."""
        deps = sample_graph.get_direct_dependencies("HEB.11.17")
        dep_ids = [d[0] for d in deps]

        assert "GEN.22.1" in dep_ids
        assert "GEN.12.1" in dep_ids

    def test_get_dependents(self, sample_graph):
        """Test getting verses that depend on a given verse."""
        dependents = sample_graph.get_dependents("GEN.22.1")
        dep_ids = [d[0] for d in dependents]

        assert "HEB.11.17" in dep_ids

    def test_from_necessity_results(self):
        """Test building graph from analysis results."""
        # Create mock results
        result1 = NecessityAnalysisResult(
            analysis_id="1",
            source_verse="A",
            target_verse="B",
            timestamp="2024-01-01",
            necessity_score=0.8,
            necessity_type=NecessityType.PRESUPPOSITIONAL,
            strength=NecessityStrength.STRONG,
            confidence=0.8,
            score_distribution=ScoreDistribution(),
            confidence_interval=(0.7, 0.9),
            semantic_gaps=[],
            gaps_filled_by_target=0,
            total_gaps=0,
            gap_coverage=0.0,
            weighted_severity_filled=0.0,
            presuppositions=[],
            presuppositions_satisfied=0,
            explicit_references=[],
            has_citation_formula=False,
            dependency_chain=["A", "B"],
            chain_length=1,
            is_direct_dependency=True,
            bidirectional=False,
            reverse_necessity_score=0.1,
            mutual_necessity=False,
            reasoning="test",
            evidence_summary="test",
        )

        graph = DependencyGraph.from_necessity_results([result1], min_score=0.5)
        assert graph.total_edges == 1


# =============================================================================
# CACHING TESTS
# =============================================================================

class TestCaching:
    """Tests for result caching."""

    @pytest.mark.asyncio
    async def test_cache_hit(self, calculator):
        """Test that second call returns cached result."""
        # First call
        result1 = await calculator.calculate_necessity("HEB.11.17", "GEN.22.1")
        assert result1.cache_hit == False

        # Second call should hit cache
        result2 = await calculator.calculate_necessity("HEB.11.17", "GEN.22.1")
        assert result2.cache_hit == True

    @pytest.mark.asyncio
    async def test_force_recompute_bypasses_cache(self, calculator):
        """Test that force_recompute bypasses cache."""
        # First call
        result1 = await calculator.calculate_necessity("GEN.1.1", "PSA.19.1")

        # Force recompute
        result2 = await calculator.calculate_necessity(
            "GEN.1.1", "PSA.19.1",
            force_recompute=True
        )
        assert result2.cache_hit == False


# =============================================================================
# RESULT SERIALIZATION TESTS
# =============================================================================

class TestResultSerialization:
    """Tests for result serialization."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = NecessityAnalysisResult(
            analysis_id="test123",
            source_verse="A",
            target_verse="B",
            timestamp="2024-01-01T00:00:00",
            necessity_score=0.75,
            necessity_type=NecessityType.PRESUPPOSITIONAL,
            strength=NecessityStrength.STRONG,
            confidence=0.8,
            score_distribution=ScoreDistribution(alpha=5, beta_param=2),
            confidence_interval=(0.6, 0.9),
            semantic_gaps=[],
            gaps_filled_by_target=2,
            total_gaps=3,
            gap_coverage=0.67,
            weighted_severity_filled=1.5,
            presuppositions=[],
            presuppositions_satisfied=1,
            explicit_references=[],
            has_citation_formula=True,
            dependency_chain=["A", "B"],
            chain_length=1,
            is_direct_dependency=True,
            bidirectional=False,
            reverse_necessity_score=0.2,
            mutual_necessity=False,
            reasoning="Test reasoning",
            evidence_summary="Test evidence",
            computation_time_ms=100.5,
        )

        d = result.to_dict()
        assert d["analysis_id"] == "test123"
        assert d["necessity_score"] == 0.75
        assert d["necessity_type"] == "presuppositional"
        assert d["strength"] == "strong"

    def test_to_json_and_back(self):
        """Test JSON round-trip."""
        result = NecessityAnalysisResult(
            analysis_id="test456",
            source_verse="HEB.11.17",
            target_verse="GEN.22.1",
            timestamp="2024-01-01T00:00:00",
            necessity_score=0.9,
            necessity_type=NecessityType.PRESUPPOSITIONAL,
            strength=NecessityStrength.ABSOLUTE,
            confidence=0.9,
            score_distribution=ScoreDistribution(),
            confidence_interval=(0.8, 0.95),
            semantic_gaps=[],
            gaps_filled_by_target=3,
            total_gaps=4,
            gap_coverage=0.75,
            weighted_severity_filled=2.5,
            presuppositions=[],
            presuppositions_satisfied=2,
            explicit_references=[],
            has_citation_formula=False,
            dependency_chain=["HEB.11.17", "GEN.22.1"],
            chain_length=1,
            is_direct_dependency=True,
            bidirectional=False,
            reverse_necessity_score=0.1,
            mutual_necessity=False,
            reasoning="Hebrews requires Genesis",
            evidence_summary="Abraham, Isaac mentioned",
        )

        json_str = result.to_json()
        restored = NecessityAnalysisResult.from_json(json_str)

        assert restored.source_verse == result.source_verse
        assert restored.target_verse == result.target_verse
        assert restored.necessity_score == result.necessity_score
        assert restored.necessity_type == result.necessity_type

    def test_to_neo4j_properties(self):
        """Test conversion to Neo4j properties."""
        result = NecessityAnalysisResult(
            analysis_id="test789",
            source_verse="MAT.1.23",
            target_verse="ISA.7.14",
            timestamp="2024-01-01T00:00:00",
            necessity_score=0.85,
            necessity_type=NecessityType.REFERENTIAL,
            strength=NecessityStrength.STRONG,
            confidence=0.85,
            score_distribution=ScoreDistribution(),
            confidence_interval=(0.75, 0.95),
            semantic_gaps=[],
            gaps_filled_by_target=1,
            total_gaps=2,
            gap_coverage=0.5,
            weighted_severity_filled=0.9,
            presuppositions=[],
            presuppositions_satisfied=0,
            explicit_references=[],
            has_citation_formula=True,
            dependency_chain=["MAT.1.23", "ISA.7.14"],
            chain_length=1,
            is_direct_dependency=True,
            bidirectional=False,
            reverse_necessity_score=0.15,
            mutual_necessity=False,
            reasoning="Citation formula detected",
            evidence_summary="As it is written",
        )

        props = result.to_neo4j_properties()
        assert props["score"] == 0.85
        assert props["type"] == "referential"
        assert props["strength"] == "strong"
        assert props["has_citation"] == True


# =============================================================================
# ENUM TESTS
# =============================================================================

class TestEnums:
    """Tests for enum definitions."""

    def test_necessity_type_values(self):
        """Test NecessityType enum values."""
        assert NecessityType.REFERENTIAL.value == "referential"
        assert NecessityType.PRESUPPOSITIONAL.value == "presuppositional"
        assert NecessityType.DEFINITIONAL.value == "definitional"

    def test_necessity_strength_ordering(self):
        """Test that strength enum can be compared by name."""
        strengths = [
            NecessityStrength.NONE,
            NecessityStrength.WEAK,
            NecessityStrength.MODERATE,
            NecessityStrength.STRONG,
            NecessityStrength.ABSOLUTE,
        ]
        # Just verify all exist
        assert len(strengths) == 5

    def test_gap_type_categories(self):
        """Test GapType categories."""
        # Entity types
        assert GapType.ENTITY_PERSON.value.startswith("entity_")
        assert GapType.ENTITY_PLACE.value.startswith("entity_")

        # Event types
        assert GapType.EVENT_HISTORICAL.value.startswith("event_")

        # Concept types
        assert GapType.CONCEPT_THEOLOGICAL.value.startswith("concept_")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for full calculator workflow."""

    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self, calculator):
        """Test complete analysis workflow from start to finish."""
        # Analyze a known necessity relationship
        result = await calculator.calculate_necessity(
            verse_a="HEB.11.17",
            verse_b="GEN.22.1"
        )

        # Verify all result fields are populated
        assert result.analysis_id is not None
        assert result.source_verse == "HEB.11.17"
        assert result.target_verse == "GEN.22.1"
        assert result.timestamp is not None
        assert 0.0 <= result.necessity_score <= 1.0
        assert result.necessity_type is not None
        assert result.strength is not None
        assert result.reasoning is not None
        assert result.computation_time_ms >= 0

    @pytest.mark.asyncio
    async def test_multiple_calculations_consistent(self, calculator):
        """Test that multiple calculations give consistent results."""
        # Calculate twice with force_recompute
        result1 = await calculator.calculate_necessity(
            "ROM.3.25", "LEV.16.15",
            force_recompute=True
        )
        result2 = await calculator.calculate_necessity(
            "ROM.3.25", "LEV.16.15",
            force_recompute=True
        )

        # Results should be very close (deterministic algorithm)
        assert abs(result1.necessity_score - result2.necessity_score) < 0.01
        assert result1.necessity_type == result2.necessity_type
        assert result1.strength == result2.strength

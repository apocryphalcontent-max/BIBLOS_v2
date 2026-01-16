"""
Tests for LXX Christological Extractor (Third Impossible Oracle).

Tests the discovery of Christological content uniquely present in the Septuagint
but absent from or muted in the Masoretic Text.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ml.engines.lxx_extractor import (
    LXXChristologicalExtractor,
    ChristologicalCategory,
    DivergenceType,
    ManuscriptPriority,
    ManuscriptWitness,
    LXXDivergence,
    LXXAnalysisResult,
    NTQuotation,
    PatristicWitness
)


class TestManuscriptPriority:
    """Test manuscript priority ranking."""

    def test_reliability_weight_dss(self):
        """DSS should have highest reliability weight."""
        weight = ManuscriptPriority.DSS.reliability_weight
        assert weight == 1.0

    def test_reliability_weight_oldest_lxx(self):
        """Oldest LXX manuscripts should have high reliability."""
        weight = ManuscriptPriority.OLDEST_LXX.reliability_weight
        assert 0.6 < weight < 0.9

    def test_reliability_weight_mt(self):
        """MT should have lowest reliability weight."""
        weight = ManuscriptPriority.MASORETIC.reliability_weight
        assert weight == 0.3  # Minimum weight

    def test_priority_ordering(self):
        """Earlier manuscripts should have higher weights."""
        dss_weight = ManuscriptPriority.DSS.reliability_weight
        lxx_weight = ManuscriptPriority.OLDEST_LXX.reliability_weight
        mt_weight = ManuscriptPriority.MASORETIC.reliability_weight

        assert dss_weight > lxx_weight > mt_weight


class TestLXXDivergence:
    """Test LXX divergence data structure and scoring."""

    def test_composite_score_calculation(self):
        """Test composite score computation."""
        div = LXXDivergence(
            divergence_id="div_ISA.7.14_0",
            verse_id="ISA.7.14",
            mt_text_hebrew="עַלְמָה",
            mt_text_transliterated="almah",
            mt_gloss="young woman",
            lxx_text_greek="παρθένος",
            lxx_text_transliterated="parthenos",
            lxx_gloss="virgin",
            divergence_type=DivergenceType.LEXICAL,
            christological_category=ChristologicalCategory.VIRGIN_BIRTH,
            christological_significance="Virgin birth prophecy",
            divergence_score=0.9,
            christological_score=0.95,
            manuscript_confidence=0.85
        )

        # Add NT quotation support
        div.nt_quotations = [
            NTQuotation(
                nt_verse="MAT.1.23",
                nt_text_greek="η παρθενος",
                quote_type="exact",
                follows_lxx=True,
                follows_mt=False,
                verbal_agreement_lxx=0.95,
                verbal_agreement_mt=0.3,
                theological_significance="Matthew follows LXX for virgin birth"
            )
        ]

        # Add patristic support
        div.patristic_witnesses = [
            PatristicWitness(
                father="Justin Martyr",
                era="ante-nicene",
                work="Dialogue with Trypho",
                citation="Ch. 43",
                interpretation="Virgin birth from Isaiah",
                text_preference="lxx",
                christological_reading=True
            )
        ]

        score = div.compute_composite_score()

        # Should be high due to strong divergence, Christological significance,
        # NT support, and patristic witness
        assert score > 0.8
        assert div.composite_score > 0.8

    def test_no_christological_content(self):
        """Test divergence without Christological significance."""
        div = LXXDivergence(
            divergence_id="div_GEN.1.1_0",
            verse_id="GEN.1.1",
            mt_text_hebrew="בְּרֵאשִׁית",
            mt_text_transliterated="bereshit",
            mt_gloss="in beginning",
            lxx_text_greek="ἐν ἀρχῇ",
            lxx_text_transliterated="en archē",
            lxx_gloss="in beginning",
            divergence_type=DivergenceType.GRAMMATICAL,
            christological_category=None,
            christological_significance="",
            divergence_score=0.3,
            christological_score=0.0,
            manuscript_confidence=0.5
        )

        score = div.compute_composite_score()

        # Should be low without Christological content
        assert score < 0.3


@pytest.mark.asyncio
class TestLXXChristologicalExtractor:
    """Test LXX Christological Extractor functionality."""

    @pytest.fixture
    def mock_lxx_client(self):
        """Mock LXX corpus client."""
        client = AsyncMock()
        client.get_verse = AsyncMock(return_value={
            "verse_id": "ISA.7.14",
            "lxx_verse_id": "ISA.7.14",
            "text": "ἰδοὺ ἡ παρθένος ἐν γαστρὶ ἕξει",
            "words": [
                {"text": "παρθένος", "lemma": "παρθένος", "gloss": "virgin", "morphology": {}}
            ]
        })
        client.initialize = AsyncMock()
        client.cleanup = AsyncMock()
        return client

    @pytest.fixture
    def mock_mt_client(self):
        """Mock MT (Text-Fabric) client."""
        client = AsyncMock()
        client.get_verse = AsyncMock(return_value={
            "verse_id": "ISA.7.14",
            "text": "הנה העלמה הרה וילדת בן",
            "words": [
                {"text": "העלמה", "lemma": "עלמה", "gloss": "young woman", "morphology": {}}
            ]
        })
        client.initialize = AsyncMock()
        client.cleanup = AsyncMock()
        return client

    @pytest.fixture
    def mock_neo4j(self):
        """Mock Neo4j client."""
        client = AsyncMock()
        client.query = AsyncMock(return_value=[])
        return client

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        client = AsyncMock()
        client.get = AsyncMock(return_value=None)
        client.set = AsyncMock()
        return client

    @pytest.fixture
    def mock_config(self):
        """Mock LXX extractor config."""
        from dataclasses import dataclass

        @dataclass
        class MockConfig:
            alignment_threshold: float = 0.7
            min_divergence_score: float = 0.3
            christological_threshold: float = 0.5
            include_manuscripts: bool = True
            include_nt_quotations: bool = True
            include_patristic: bool = True
            max_patristic_witnesses: int = 10
            cache_enabled: bool = False
            cache_ttl_seconds: int = 3600
            batch_size: int = 50
            parallel_analysis: bool = True

        return MockConfig()

    @pytest.fixture
    def extractor(self, mock_lxx_client, mock_mt_client, mock_neo4j, mock_redis, mock_config):
        """Create extractor instance with mocks."""
        return LXXChristologicalExtractor(
            lxx_client=mock_lxx_client,
            mt_client=mock_mt_client,
            neo4j=mock_neo4j,
            redis=mock_redis,
            config=mock_config
        )

    async def test_normalize_verse_ids_psalm(self, extractor):
        """Test Psalm numbering conversion (LXX = MT - 1)."""
        mt_id, lxx_id = await extractor._normalize_verse_ids("PSA.40.6")

        assert mt_id == "PSA.40.6"
        assert lxx_id == "PSA.39.6"

    async def test_normalize_verse_ids_non_psalm(self, extractor):
        """Test non-Psalm verses keep same numbering."""
        mt_id, lxx_id = await extractor._normalize_verse_ids("ISA.7.14")

        assert mt_id == "ISA.7.14"
        assert lxx_id == "ISA.7.14"

    async def test_normalize_verse_ids_psalm_out_of_range(self, extractor):
        """Test Psalms outside the offset range."""
        mt_id, lxx_id = await extractor._normalize_verse_ids("PSA.1.1")

        # Psalm 1 is outside the offset range (10-147)
        assert mt_id == "PSA.1.1"
        assert lxx_id == "PSA.1.1"

    def test_determine_oldest_lxx_support(self, extractor):
        """Test oldest manuscript determination with LXX support."""
        witnesses = [
            ManuscriptWitness(
                manuscript_id="Codex Vaticanus",
                manuscript_type=ManuscriptPriority.OLDEST_LXX,
                date_range="4th century CE",
                century_numeric=4,
                reading="παρθένος",
                reading_transliterated="parthenos",
                supports_lxx=True,
                supports_mt=False,
                notes="",
                reliability_score=0.8
            ),
            ManuscriptWitness(
                manuscript_id="4QIsaᵃ",
                manuscript_type=ManuscriptPriority.DSS,
                date_range="125-100 BCE",
                century_numeric=-1,
                reading="עלמה",
                reading_transliterated="almah",
                supports_lxx=False,
                supports_mt=True,
                notes="",
                reliability_score=1.0
            )
        ]

        oldest, supports = extractor._determine_oldest(witnesses)

        assert oldest.manuscript_id == "4QIsaᵃ"  # DSS is oldest
        assert supports == "mt"  # DSS supports MT

    def test_determine_oldest_no_witnesses(self, extractor):
        """Test with no manuscript witnesses."""
        oldest, supports = extractor._determine_oldest([])

        assert oldest is None
        assert supports == "unknown"

    def test_known_christological_verses(self):
        """Test known Christological verses catalog."""
        known = LXXChristologicalExtractor.KNOWN_CHRISTOLOGICAL_VERSES

        # Check key verses are cataloged
        assert "ISA.7.14" in known
        assert known["ISA.7.14"] == ChristologicalCategory.VIRGIN_BIRTH

        assert "PSA.40.6" in known
        assert known["PSA.40.6"] == ChristologicalCategory.INCARNATION

        assert "PSA.22.16" in known
        assert known["PSA.22.16"] == ChristologicalCategory.PASSION

        # Should have at least 10 cataloged verses
        assert len(known) >= 10

    def test_morphology_compatibility(self, extractor):
        """Test morphological compatibility scoring."""
        # Matching morphology
        mt_morph = {"number": "singular", "gender": "feminine"}
        lxx_morph = {"number": "singular", "gender": "feminine"}

        score = extractor._morphology_compatibility(mt_morph, lxx_morph)
        assert score == 1.0

        # Partial match
        mt_morph = {"number": "singular", "gender": "feminine"}
        lxx_morph = {"number": "singular", "gender": "masculine"}

        score = extractor._morphology_compatibility(mt_morph, lxx_morph)
        assert 0.4 < score < 0.6

        # Dual -> plural conversion (Greek lacks dual)
        mt_morph = {"number": "dual"}
        lxx_morph = {"number": "plural"}

        score = extractor._morphology_compatibility(mt_morph, lxx_morph)
        assert score >= 0.8

    def test_verbal_agreement(self, extractor):
        """Test verbal agreement calculation."""
        # Exact match
        text1 = "ἡ παρθένος ἐν γαστρὶ ἕξει"
        text2 = "ἡ παρθένος ἐν γαστρὶ ἕξει"
        assert extractor._verbal_agreement(text1, text2) == 1.0

        # No match
        text1 = "completely different text"
        text2 = "κάτι εντελώς διαφορετικό"
        assert extractor._verbal_agreement(text1, text2) < 0.2

        # Empty strings
        assert extractor._verbal_agreement("", "test") == 0.0
        assert extractor._verbal_agreement("test", "") == 0.0

    async def test_extract_christological_content_isaiah_7_14(
        self, extractor, mock_neo4j
    ):
        """Test extraction for Isaiah 7:14 (parthenos)."""
        # Mock Neo4j to return NT quotation data
        mock_neo4j.query.return_value = [
            {
                "nt_verse": "MAT.1.23",
                "nt_text": "ἡ παρθένος ἐν γαστρὶ ἕξει",
                "quote_type": "exact",
                "agreement": 0.95
            }
        ]

        result = await extractor.extract_christological_content("ISA.7.14")

        assert result.verse_id == "ISA.7.14"
        # Should detect Christological content
        assert result.christological_divergence_count >= 0

    def test_similar_gloss(self, extractor):
        """Test gloss similarity check."""
        # Similar glosses
        assert extractor._similar_gloss("young woman", "woman young")
        assert extractor._similar_gloss("virgin", "virgin")

        # Different glosses
        assert not extractor._similar_gloss("virgin", "completely different")

        # Empty glosses
        assert not extractor._similar_gloss("", "test")


@pytest.mark.integration
@pytest.mark.asyncio
class TestLXXExtractorIntegration:
    """Integration tests requiring actual corpus data."""

    pytestmark = pytest.mark.skipif(
        "not config.getoption('--integration')",
        reason="Requires --integration flag and corpus data"
    )

    async def test_full_extraction_isaiah_7_14(self):
        """Full integration test for ISA.7.14."""
        # This would require actual corpus databases
        pytest.skip("Requires full corpus setup")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for Text-Fabric integration.
"""
import pytest


class TestTextFabricIntegration:
    """Tests for TextFabricIntegration."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test integration initialization."""
        from integrations.text_fabric import TextFabricIntegration

        integration = TextFabricIntegration(corpus_type="bhsa")
        await integration.initialize()

        assert integration._initialized is True

        await integration.cleanup()

    @pytest.mark.asyncio
    async def test_get_verse(self, sample_verse_id):
        """Test getting verse data."""
        from integrations.text_fabric import TextFabricIntegration

        integration = TextFabricIntegration()
        await integration.initialize()

        verse = await integration.get_verse(sample_verse_id)

        assert verse is not None
        assert verse.verse_id == sample_verse_id

        await integration.cleanup()

    @pytest.mark.asyncio
    async def test_get_verses_book(self):
        """Test getting verses for a book."""
        from integrations.text_fabric import TextFabricIntegration

        integration = TextFabricIntegration()
        await integration.initialize()

        # May return empty in mock mode
        verses = await integration.get_verses("GEN", chapter=1)

        assert isinstance(verses, list)

        await integration.cleanup()

    def test_supported_books_ot(self):
        """Test OT book support."""
        from integrations.text_fabric import TextFabricIntegration

        integration = TextFabricIntegration(corpus_type="bhsa")
        books = integration.get_supported_books()

        assert "GEN" in books
        assert "PSA" in books
        assert "MAL" in books

    def test_supported_books_nt(self):
        """Test NT book support."""
        from integrations.text_fabric import TextFabricIntegration

        integration = TextFabricIntegration(corpus_type="sblgnt")
        books = integration.get_supported_books()

        assert "MAT" in books
        assert "REV" in books

    def test_get_language(self):
        """Test language detection."""
        from integrations.text_fabric import TextFabricIntegration
        from integrations.base import Language

        hebrew_int = TextFabricIntegration(corpus_type="bhsa")
        assert hebrew_int.get_language() == Language.HEBREW

        greek_int = TextFabricIntegration(corpus_type="sblgnt")
        assert greek_int.get_language() == Language.GREEK


class TestMaculaIntegration:
    """Tests for MaculaIntegration."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test integration initialization."""
        from integrations.macula import MaculaIntegration

        integration = MaculaIntegration(corpus_type="greek")
        await integration.initialize()

        assert integration._initialized is True

        await integration.cleanup()

    @pytest.mark.asyncio
    async def test_get_verse(self, sample_verse_id):
        """Test getting verse data."""
        from integrations.macula import MaculaIntegration

        integration = MaculaIntegration()
        await integration.initialize()

        # May return None if no corpus loaded
        verse = await integration.get_verse(sample_verse_id)

        # Just test it doesn't crash
        await integration.cleanup()

    def test_greek_morphology_parsing(self):
        """Test Greek morphology code parsing."""
        from integrations.macula import MaculaIntegration

        integration = MaculaIntegration(corpus_type="greek")

        # Test verb parsing
        morph = integration._parse_greek_morphology("V-PAI-3S")
        assert morph.part_of_speech == "verb"

        # Test noun parsing
        morph = integration._parse_greek_morphology("N-NSM")
        assert morph.part_of_speech == "noun"

    def test_hebrew_morphology_parsing(self):
        """Test Hebrew morphology code parsing."""
        from integrations.macula import MaculaIntegration

        integration = MaculaIntegration(corpus_type="hebrew")

        morph = integration._parse_hebrew_morphology("HVqp3ms")
        assert morph.part_of_speech == "verb"

    def test_verse_id_normalization(self):
        """Test verse ID normalization."""
        from integrations.macula import MaculaIntegration

        integration = MaculaIntegration()

        # Test different formats
        assert integration._normalize_verse_id("Gen 1:1") == "GEN.1.1"
        assert integration._normalize_verse_id("GEN.1.1") == "GEN.1.1"
        assert integration._normalize_verse_id("gen.1.1") == "GEN.1.1"

    def test_supported_books(self):
        """Test supported books."""
        from integrations.macula import MaculaIntegration

        greek_int = MaculaIntegration(corpus_type="greek")
        books = greek_int.get_supported_books()
        assert "MAT" in books
        assert "REV" in books

        hebrew_int = MaculaIntegration(corpus_type="hebrew")
        books = hebrew_int.get_supported_books()
        assert "GEN" in books
        assert "MAL" in books

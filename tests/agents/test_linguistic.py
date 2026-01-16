"""
Tests for linguistic extraction agents.
"""
import pytest


class TestGrammateus:
    """Tests for GRAMMATEUS agent (Structural Analysis)."""

    @pytest.mark.asyncio
    async def test_extract_basic(self, sample_verse_id, sample_verse_text, sample_context):
        """Test basic extraction."""
        from agents.linguistic.grammateus import GramateusAgent

        agent = GramateusAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_context)

        assert result.agent_name == "grammateus"
        assert result.verse_id == sample_verse_id
        assert result.confidence > 0
        assert "word_count" in result.data or "structural" in str(result.data)

    @pytest.mark.asyncio
    async def test_validate_result(self, sample_verse_id, sample_verse_text, sample_context):
        """Test result validation."""
        from agents.linguistic.grammateus import GramateusAgent

        agent = GramateusAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_context)
        is_valid = await agent.validate(result)
        assert is_valid is True

    def test_dependencies(self):
        """Test agent dependencies."""
        from agents.linguistic.grammateus import GramateusAgent

        agent = GramateusAgent()
        deps = agent.get_dependencies()
        assert isinstance(deps, list)
        assert len(deps) == 0  # GRAMMATEUS has no dependencies


class TestMorphologos:
    """Tests for MORPHOLOGOS agent (Morphological Analysis)."""

    @pytest.mark.asyncio
    async def test_extract_basic(self, sample_verse_id, sample_verse_text, sample_linguistic_context):
        """Test basic extraction."""
        from agents.linguistic.morphologos import MorphologosAgent

        agent = MorphologosAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_linguistic_context)

        assert result.agent_name == "morphologos"
        assert result.verse_id == sample_verse_id
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_morphological_analysis(self, sample_verse_id, sample_verse_text, sample_linguistic_context):
        """Test morphological analysis output."""
        from agents.linguistic.morphologos import MorphologosAgent

        agent = MorphologosAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_linguistic_context)

        # Should contain word analyses
        data = result.data
        assert isinstance(data, dict)

    def test_dependencies(self):
        """Test agent dependencies."""
        from agents.linguistic.morphologos import MorphologosAgent

        agent = MorphologosAgent()
        deps = agent.get_dependencies()
        assert "grammateus" in deps


class TestSyntaktikos:
    """Tests for SYNTAKTIKOS agent (Syntactic Analysis)."""

    @pytest.mark.asyncio
    async def test_extract_basic(self, sample_verse_id, sample_verse_text, sample_linguistic_context):
        """Test basic extraction."""
        from agents.linguistic.syntaktikos import SyntaktikosAgent

        agent = SyntaktikosAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_linguistic_context)

        assert result.agent_name == "syntaktikos"
        assert result.verse_id == sample_verse_id

    @pytest.mark.asyncio
    async def test_clause_detection(self, sample_verse_id, sample_verse_text, sample_linguistic_context):
        """Test clause structure detection."""
        from agents.linguistic.syntaktikos import SyntaktikosAgent

        agent = SyntaktikosAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_linguistic_context)

        data = result.data
        # Should have syntactic structure info
        assert isinstance(data, dict)

    def test_dependencies(self):
        """Test agent dependencies."""
        from agents.linguistic.syntaktikos import SyntaktikosAgent

        agent = SyntaktikosAgent()
        deps = agent.get_dependencies()
        assert "grammateus" in deps
        assert "morphologos" in deps


class TestSemantikos:
    """Tests for SEMANTIKOS agent (Semantic Analysis)."""

    @pytest.mark.asyncio
    async def test_extract_basic(self, sample_verse_id, sample_verse_text, sample_linguistic_context):
        """Test basic extraction."""
        from agents.linguistic.semantikos import SemantikosAgent

        agent = SemantikosAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_linguistic_context)

        assert result.agent_name == "semantikos"
        assert result.verse_id == sample_verse_id

    @pytest.mark.asyncio
    async def test_semantic_field_detection(self, sample_verse_id, sample_verse_text, sample_linguistic_context):
        """Test semantic field detection."""
        from agents.linguistic.semantikos import SemantikosAgent

        agent = SemantikosAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_linguistic_context)

        data = result.data
        assert isinstance(data, dict)

    def test_dependencies(self):
        """Test agent dependencies."""
        from agents.linguistic.semantikos import SemantikosAgent

        agent = SemantikosAgent()
        deps = agent.get_dependencies()
        assert "grammateus" in deps

"""
Domain-Specific Embedding Generators

Each domain has a specialized embedder that generates embeddings
optimized for that particular type of semantic relationship.
"""
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class VerseContext:
    """Context information for verse embedding."""
    verse_id: str
    text: str
    testament: str  # "old" or "new"
    book: str
    chapter: int
    verse: int
    words: Optional[List[Dict[str, Any]]] = None
    cross_references: Optional[List[str]] = None
    patristic_witnesses: Optional[List[Dict[str, str]]] = None


class DomainEmbedder:
    """Base class for domain-specific embedders."""

    def __init__(self, model_name: str):
        """
        Initialize embedder.

        Args:
            model_name: SentenceTransformer model name
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed")

        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")

    def embed(self, context: VerseContext) -> np.ndarray:
        """
        Generate embedding for verse context.

        Args:
            context: Verse context with text and metadata

        Returns:
            Embedding vector
        """
        raise NotImplementedError()

    def embed_batch(self, contexts: List[VerseContext]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple verses efficiently.

        Args:
            contexts: List of verse contexts

        Returns:
            List of embedding vectors
        """
        return [self.embed(ctx) for ctx in contexts]


class SemanticEmbedder(DomainEmbedder):
    """
    Semantic domain embedder.

    Focuses on general meaning and contextual understanding.
    Uses the full verse text.
    """

    def embed(self, context: VerseContext) -> np.ndarray:
        """Generate semantic embedding from verse text."""
        return self.model.encode(context.text, convert_to_numpy=True)

    def embed_batch(self, contexts: List[VerseContext]) -> List[np.ndarray]:
        """Batch encode for efficiency."""
        texts = [ctx.text for ctx in contexts]
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


class TypologicalEmbedder(DomainEmbedder):
    """
    Typological domain embedder.

    Emphasizes type/antitype patterns by including:
    - Testament information (OT types, NT antitypes)
    - Word-level patterns
    - Structural similarities
    """

    def embed(self, context: VerseContext) -> np.ndarray:
        """Generate typological embedding."""
        # Construct typology-aware text
        testament_prefix = "[OT]" if context.testament == "old" else "[NT]"
        book_info = f"[{context.book}]"

        # Include key words if available
        key_words = ""
        if context.words:
            # Extract important words (nouns, verbs)
            important = [
                w.get('lemma', '') for w in context.words
                if w.get('part_of_speech') in ['noun', 'verb']
            ]
            key_words = " ".join(important[:5])  # Top 5 key words

        combined_text = f"{testament_prefix} {book_info} {context.text} {key_words}"
        return self.model.encode(combined_text, convert_to_numpy=True)


class PropheticEmbedder(DomainEmbedder):
    """
    Prophetic domain embedder.

    Emphasizes prophecy-fulfillment patterns by including:
    - Testament flow (prophecy -> fulfillment)
    - Predictive language markers
    - Fulfillment indicators
    """

    PROPHECY_MARKERS = [
        "shall", "will", "thus says", "behold", "prophecy",
        "foretold", "predicted", "to come"
    ]

    FULFILLMENT_MARKERS = [
        "fulfilled", "accomplished", "came to pass",
        "as it is written", "according to"
    ]

    def embed(self, context: VerseContext) -> np.ndarray:
        """Generate prophetic embedding."""
        text_lower = context.text.lower()

        # Detect prophecy vs fulfillment
        is_prophecy = any(marker in text_lower for marker in self.PROPHECY_MARKERS)
        is_fulfillment = any(marker in text_lower for marker in self.FULFILLMENT_MARKERS)

        prefix = ""
        if is_prophecy:
            prefix = "[PROPHECY]"
        if is_fulfillment:
            prefix = "[FULFILLMENT]"

        testament_info = f"[{context.testament.upper()}]"
        combined_text = f"{prefix} {testament_info} {context.text}"

        return self.model.encode(combined_text, convert_to_numpy=True)


class PatristicEmbedder(DomainEmbedder):
    """
    Patristic domain embedder.

    Emphasizes Church Father interpretations by including:
    - Patristic witnesses and their authority levels
    - Dominant interpretative traditions
    - Consensus patterns
    """

    def embed(self, context: VerseContext) -> np.ndarray:
        """Generate patristic embedding."""
        # Include patristic witness information
        patristic_context = ""
        if context.patristic_witnesses:
            fathers = [w.get('father_name', '') for w in context.patristic_witnesses[:3]]
            interpretations = [w.get('interpretation', '') for w in context.patristic_witnesses[:2]]

            patristic_context = f"[Fathers: {', '.join(fathers)}] "
            if interpretations:
                patristic_context += " ".join(interpretations[:100])  # Limit length

        combined_text = f"{context.text} {patristic_context}"
        return self.model.encode(combined_text, convert_to_numpy=True)


class LiturgicalEmbedder(DomainEmbedder):
    """
    Liturgical domain embedder.

    Emphasizes liturgical usage patterns by including:
    - Feast day associations
    - Liturgical season contexts
    - Worship themes
    """

    LITURGICAL_THEMES = {
        "christmas": ["incarnation", "birth", "nativity", "emmanuel"],
        "pascha": ["resurrection", "passover", "tomb", "risen"],
        "theophany": ["baptism", "manifestation", "jordan", "dove"],
        "pentecost": ["spirit", "tongues", "wind", "fire"],
        "transfiguration": ["glory", "mountain", "moses", "elijah"],
    }

    def embed(self, context: VerseContext) -> np.ndarray:
        """Generate liturgical embedding."""
        text_lower = context.text.lower()

        # Detect liturgical themes
        detected_themes = []
        for theme, keywords in self.LITURGICAL_THEMES.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_themes.append(theme)

        theme_prefix = ""
        if detected_themes:
            theme_prefix = f"[Themes: {', '.join(detected_themes)}]"

        combined_text = f"{theme_prefix} {context.text}"
        return self.model.encode(combined_text, convert_to_numpy=True)


class MultiDomainEmbedder:
    """
    Orchestrates all domain embedders.

    Generates embeddings across all domains for a verse.
    """

    def __init__(self):
        """Initialize all domain embedders."""
        from ml.embeddings.multi_vector_store import EmbeddingDomain

        self.embedders = {
            EmbeddingDomain.SEMANTIC: SemanticEmbedder(
                EmbeddingDomain.SEMANTIC.model_name
            ),
            EmbeddingDomain.TYPOLOGICAL: TypologicalEmbedder(
                EmbeddingDomain.TYPOLOGICAL.model_name
            ),
            EmbeddingDomain.PROPHETIC: PropheticEmbedder(
                EmbeddingDomain.PROPHETIC.model_name
            ),
            EmbeddingDomain.PATRISTIC: PatristicEmbedder(
                EmbeddingDomain.PATRISTIC.model_name
            ),
            EmbeddingDomain.LITURGICAL: LiturgicalEmbedder(
                EmbeddingDomain.LITURGICAL.model_name
            ),
        }
        logger.info("Initialized all domain embedders")

    def embed_verse(
        self,
        context: VerseContext,
        domains: Optional[List] = None
    ) -> Dict:
        """
        Generate embeddings across specified domains.

        Args:
            context: Verse context
            domains: List of domains to embed (default: all)

        Returns:
            Dict mapping domains to embeddings
        """
        from ml.embeddings.multi_vector_store import EmbeddingDomain

        if domains is None:
            domains = list(EmbeddingDomain)

        embeddings = {}
        for domain in domains:
            embedder = self.embedders.get(domain)
            if embedder:
                try:
                    embeddings[domain] = embedder.embed(context)
                except Exception as e:
                    logger.error(f"Error embedding {context.verse_id} in {domain.value}: {e}")
                    embeddings[domain] = None

        return embeddings

    def embed_batch(
        self,
        contexts: List[VerseContext],
        domains: Optional[List] = None
    ) -> List[Dict]:
        """
        Batch embed multiple verses across domains.

        Args:
            contexts: List of verse contexts
            domains: Domains to embed

        Returns:
            List of dicts mapping domains to embeddings
        """
        from ml.embeddings.multi_vector_store import EmbeddingDomain

        if domains is None:
            domains = list(EmbeddingDomain)

        # Batch by domain for efficiency
        results = [{} for _ in contexts]

        for domain in domains:
            embedder = self.embedders.get(domain)
            if not embedder:
                continue

            try:
                embeddings = embedder.embed_batch(contexts)
                for i, embedding in enumerate(embeddings):
                    results[i][domain] = embedding
            except Exception as e:
                logger.error(f"Error batch embedding in {domain.value}: {e}")
                for i in range(len(contexts)):
                    results[i][domain] = None

        return results

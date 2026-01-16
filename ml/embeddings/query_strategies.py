"""
Query Strategies for Multi-Domain Vector Search

Provides specialized query configurations for different use cases:
- Theological: Deep theological analysis with patristic emphasis
- Liturgical: Feast day and worship context queries
- Typological: Type/antitype pattern discovery
- Prophetic: Prophecy-fulfillment chain analysis
- CrossRef: Cross-reference discovery optimization
- Covenantal: Covenant structure and progression queries

Each strategy defines:
- Domain weights for hybrid search
- Aggregation method
- Score thresholds
- Result filtering
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from ml.embeddings.multi_vector_store import (
    EmbeddingDomain,
    HybridSearchConfig,
    AggregationMethod,
    SearchResult,
    MultiVectorStore,
)
from ml.embeddings.domain_embedders import VerseContext, MultiDomainEmbedder

import numpy as np

logger = logging.getLogger(__name__)


class QueryPurpose(Enum):
    """Purpose of the query, used to select optimal strategy."""
    THEOLOGICAL_ANALYSIS = "theological_analysis"
    LITURGICAL_CONTEXT = "liturgical_context"
    TYPOLOGY_DISCOVERY = "typology_discovery"
    PROPHECY_CHAIN = "prophecy_chain"
    CROSS_REFERENCE = "cross_reference"
    COVENANT_STUDY = "covenant_study"
    PATRISTIC_CONSENSUS = "patristic_consensus"
    GENERAL_SIMILARITY = "general_similarity"


@dataclass
class QueryStrategyConfig:
    """Configuration for a query strategy."""
    name: str
    description: str
    domain_weights: Dict[EmbeddingDomain, float]
    aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_SUM
    min_score: float = 0.0
    min_domains_matched: int = 1
    score_normalization: bool = True
    diversity_penalty: float = 0.0
    rrf_k: int = 60
    explain: bool = False
    post_filters: Optional[Dict[str, Any]] = None


class QueryStrategy(ABC):
    """Base class for query strategies."""

    def __init__(self, config: QueryStrategyConfig):
        self.config = config

    @abstractmethod
    def get_weights(self, context: Optional[VerseContext] = None) -> Dict[EmbeddingDomain, float]:
        """
        Get domain weights, optionally adjusted based on query context.

        Args:
            context: Optional query context for adaptive weighting

        Returns:
            Domain weight mapping
        """
        pass

    def get_search_config(self) -> HybridSearchConfig:
        """Get HybridSearchConfig for this strategy."""
        return HybridSearchConfig(
            aggregation_method=self.config.aggregation_method,
            rrf_k=self.config.rrf_k,
            min_domains_matched=self.config.min_domains_matched,
            score_normalization=self.config.score_normalization,
            diversity_penalty=self.config.diversity_penalty,
            explain=self.config.explain,
        )

    def post_process(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Apply post-processing filters to results.

        Override in subclasses for strategy-specific filtering.
        """
        if self.config.min_score > 0:
            results = [r for r in results if r.score >= self.config.min_score]
        return results


class TheologicalQueryStrategy(QueryStrategy):
    """
    Strategy for deep theological analysis.

    Emphasizes:
    - Patristic interpretations (highest weight)
    - Typological connections
    - Semantic similarity

    Use for: Theological questions, doctrinal studies, exegesis
    """

    def __init__(self):
        super().__init__(QueryStrategyConfig(
            name="theological",
            description="Deep theological analysis with patristic emphasis",
            domain_weights={
                EmbeddingDomain.PATRISTIC: 0.35,
                EmbeddingDomain.TYPOLOGICAL: 0.25,
                EmbeddingDomain.SEMANTIC: 0.20,
                EmbeddingDomain.PROPHETIC: 0.10,
                EmbeddingDomain.COVENANTAL: 0.10,
                EmbeddingDomain.LITURGICAL: 0.00,
            },
            aggregation_method=AggregationMethod.WEIGHTED_SUM,
            min_domains_matched=2,
            diversity_penalty=0.1,
        ))

    def get_weights(self, context: Optional[VerseContext] = None) -> Dict[EmbeddingDomain, float]:
        weights = self.config.domain_weights.copy()

        # Adapt based on testament
        if context and context.testament == "old":
            # OT: boost typological (finding NT fulfillments)
            weights[EmbeddingDomain.TYPOLOGICAL] *= 1.2
            weights[EmbeddingDomain.PROPHETIC] *= 1.1
        elif context and context.testament == "new":
            # NT: boost patristic (stronger interpretive tradition)
            weights[EmbeddingDomain.PATRISTIC] *= 1.2

        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}


class LiturgicalQueryStrategy(QueryStrategy):
    """
    Strategy for liturgical context queries.

    Emphasizes:
    - Liturgical associations (highest weight)
    - Semantic similarity
    - Prophetic (for feast readings)

    Use for: Feast day readings, liturgical connections, worship themes
    """

    def __init__(self):
        super().__init__(QueryStrategyConfig(
            name="liturgical",
            description="Liturgical context and feast day associations",
            domain_weights={
                EmbeddingDomain.LITURGICAL: 0.40,
                EmbeddingDomain.SEMANTIC: 0.25,
                EmbeddingDomain.PROPHETIC: 0.15,
                EmbeddingDomain.TYPOLOGICAL: 0.10,
                EmbeddingDomain.PATRISTIC: 0.10,
                EmbeddingDomain.COVENANTAL: 0.00,
            },
            aggregation_method=AggregationMethod.WEIGHTED_SUM,
            min_domains_matched=1,
        ))

    def get_weights(self, context: Optional[VerseContext] = None) -> Dict[EmbeddingDomain, float]:
        return self.config.domain_weights.copy()


class TypologicalQueryStrategy(QueryStrategy):
    """
    Strategy for type/antitype pattern discovery.

    Emphasizes:
    - Typological patterns (highest weight)
    - Cross-testament connections
    - Prophetic elements

    Use for: Finding OT types for NT antitypes, typological exegesis
    """

    def __init__(self):
        super().__init__(QueryStrategyConfig(
            name="typological",
            description="Type/antitype pattern discovery",
            domain_weights={
                EmbeddingDomain.TYPOLOGICAL: 0.45,
                EmbeddingDomain.PROPHETIC: 0.20,
                EmbeddingDomain.SEMANTIC: 0.15,
                EmbeddingDomain.COVENANTAL: 0.10,
                EmbeddingDomain.PATRISTIC: 0.10,
                EmbeddingDomain.LITURGICAL: 0.00,
            },
            aggregation_method=AggregationMethod.RRF,  # RRF better for cross-testament
            rrf_k=60,
            min_domains_matched=2,
            diversity_penalty=0.2,  # Encourage diversity across books
        ))

    def get_weights(self, context: Optional[VerseContext] = None) -> Dict[EmbeddingDomain, float]:
        weights = self.config.domain_weights.copy()

        if context:
            # Strong cross-testament preference
            if context.testament == "old":
                # Looking for NT antitypes: semantic helps bridge
                weights[EmbeddingDomain.SEMANTIC] *= 1.3
            else:
                # Looking for OT types: typological is key
                weights[EmbeddingDomain.TYPOLOGICAL] *= 1.2
                weights[EmbeddingDomain.COVENANTAL] *= 1.2

        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def post_process(self, results: List[SearchResult]) -> List[SearchResult]:
        """Filter to prefer cross-testament results."""
        results = super().post_process(results)

        # Boost cross-testament results
        for result in results:
            # This would need testament info in payload
            pass

        return results


class PropheticQueryStrategy(QueryStrategy):
    """
    Strategy for prophecy-fulfillment chain analysis.

    Emphasizes:
    - Prophetic patterns (highest weight)
    - Typological connections
    - Semantic similarity

    Use for: Finding prophecy fulfillments, messianic passages
    """

    def __init__(self):
        super().__init__(QueryStrategyConfig(
            name="prophetic",
            description="Prophecy-fulfillment chain analysis",
            domain_weights={
                EmbeddingDomain.PROPHETIC: 0.45,
                EmbeddingDomain.TYPOLOGICAL: 0.20,
                EmbeddingDomain.SEMANTIC: 0.15,
                EmbeddingDomain.PATRISTIC: 0.10,
                EmbeddingDomain.COVENANTAL: 0.10,
                EmbeddingDomain.LITURGICAL: 0.00,
            },
            aggregation_method=AggregationMethod.WEIGHTED_SUM,
            min_domains_matched=2,
        ))

    def get_weights(self, context: Optional[VerseContext] = None) -> Dict[EmbeddingDomain, float]:
        weights = self.config.domain_weights.copy()

        if context:
            text_lower = context.text.lower()

            # Boost prophetic if explicit prophecy markers present
            prophecy_markers = ["shall", "will", "says the lord", "thus"]
            if any(m in text_lower for m in prophecy_markers):
                weights[EmbeddingDomain.PROPHETIC] *= 1.3

            # Boost typological for messianic content
            messianic_markers = ["messiah", "christ", "son of david", "anointed"]
            if any(m in text_lower for m in messianic_markers):
                weights[EmbeddingDomain.TYPOLOGICAL] *= 1.2

        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}


class CrossRefDiscoveryStrategy(QueryStrategy):
    """
    Strategy for cross-reference discovery.

    Balanced approach emphasizing:
    - Semantic similarity (vocabulary/themes)
    - Typological patterns
    - Prophetic connections
    - Covenantal links

    Use for: Discovering new cross-references, reference validation
    """

    def __init__(self):
        super().__init__(QueryStrategyConfig(
            name="crossref_discovery",
            description="Cross-reference discovery optimization",
            domain_weights={
                EmbeddingDomain.SEMANTIC: 0.25,
                EmbeddingDomain.TYPOLOGICAL: 0.25,
                EmbeddingDomain.PROPHETIC: 0.20,
                EmbeddingDomain.COVENANTAL: 0.15,
                EmbeddingDomain.PATRISTIC: 0.10,
                EmbeddingDomain.LITURGICAL: 0.05,
            },
            aggregation_method=AggregationMethod.RRF,  # RRF handles different scales well
            rrf_k=60,
            min_domains_matched=3,  # Require broad support
            diversity_penalty=0.15,
        ))

    def get_weights(self, context: Optional[VerseContext] = None) -> Dict[EmbeddingDomain, float]:
        return self.config.domain_weights.copy()


class CovenantStudyStrategy(QueryStrategy):
    """
    Strategy for covenant structure and progression studies.

    Emphasizes:
    - Covenantal domain (highest weight)
    - Typological (covenant progression)
    - Semantic (vocabulary)

    Use for: Covenant theology, progressive revelation studies
    """

    def __init__(self):
        super().__init__(QueryStrategyConfig(
            name="covenant_study",
            description="Covenant structure and progression analysis",
            domain_weights={
                EmbeddingDomain.COVENANTAL: 0.45,
                EmbeddingDomain.TYPOLOGICAL: 0.20,
                EmbeddingDomain.SEMANTIC: 0.15,
                EmbeddingDomain.PROPHETIC: 0.10,
                EmbeddingDomain.PATRISTIC: 0.10,
                EmbeddingDomain.LITURGICAL: 0.00,
            },
            aggregation_method=AggregationMethod.WEIGHTED_SUM,
            min_domains_matched=2,
        ))

    def get_weights(self, context: Optional[VerseContext] = None) -> Dict[EmbeddingDomain, float]:
        weights = self.config.domain_weights.copy()

        if context:
            # Detect specific covenant context
            text_lower = context.text.lower()

            # Boost related domains based on covenant markers
            covenant_markers = {
                "abrahamic": ["abraham", "promise", "seed", "circumcision"],
                "mosaic": ["moses", "law", "sinai", "commandments"],
                "davidic": ["david", "throne", "kingdom"],
                "new": ["new covenant", "heart", "spirit"],
            }

            for covenant, markers in covenant_markers.items():
                if any(m in text_lower for m in markers):
                    weights[EmbeddingDomain.COVENANTAL] *= 1.2
                    break

        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}


class PatristicConsensusStrategy(QueryStrategy):
    """
    Strategy for finding patristic consensus interpretations.

    Emphasizes:
    - Patristic domain (dominant)
    - Theological context

    Use for: Finding verses with strong patristic witness, consensus views
    """

    def __init__(self):
        super().__init__(QueryStrategyConfig(
            name="patristic_consensus",
            description="Patristic consensus and interpretive tradition",
            domain_weights={
                EmbeddingDomain.PATRISTIC: 0.60,
                EmbeddingDomain.SEMANTIC: 0.20,
                EmbeddingDomain.TYPOLOGICAL: 0.10,
                EmbeddingDomain.LITURGICAL: 0.10,
                EmbeddingDomain.PROPHETIC: 0.00,
                EmbeddingDomain.COVENANTAL: 0.00,
            },
            aggregation_method=AggregationMethod.WEIGHTED_SUM,
            min_domains_matched=1,
        ))

    def get_weights(self, context: Optional[VerseContext] = None) -> Dict[EmbeddingDomain, float]:
        return self.config.domain_weights.copy()


class GeneralSimilarityStrategy(QueryStrategy):
    """
    Strategy for general semantic similarity.

    Balanced weights across all domains for broad discovery.

    Use for: General verse similarity, exploration
    """

    def __init__(self):
        super().__init__(QueryStrategyConfig(
            name="general",
            description="General semantic similarity across all domains",
            domain_weights={
                EmbeddingDomain.SEMANTIC: 0.25,
                EmbeddingDomain.TYPOLOGICAL: 0.15,
                EmbeddingDomain.PROPHETIC: 0.15,
                EmbeddingDomain.PATRISTIC: 0.15,
                EmbeddingDomain.LITURGICAL: 0.15,
                EmbeddingDomain.COVENANTAL: 0.15,
            },
            aggregation_method=AggregationMethod.WEIGHTED_SUM,
            min_domains_matched=1,
        ))

    def get_weights(self, context: Optional[VerseContext] = None) -> Dict[EmbeddingDomain, float]:
        return self.config.domain_weights.copy()


class StrategySelector:
    """
    Selects and manages query strategies.

    Provides strategy selection based on:
    - Explicit purpose specification
    - Automatic detection from query content
    """

    STRATEGIES = {
        QueryPurpose.THEOLOGICAL_ANALYSIS: TheologicalQueryStrategy,
        QueryPurpose.LITURGICAL_CONTEXT: LiturgicalQueryStrategy,
        QueryPurpose.TYPOLOGY_DISCOVERY: TypologicalQueryStrategy,
        QueryPurpose.PROPHECY_CHAIN: PropheticQueryStrategy,
        QueryPurpose.CROSS_REFERENCE: CrossRefDiscoveryStrategy,
        QueryPurpose.COVENANT_STUDY: CovenantStudyStrategy,
        QueryPurpose.PATRISTIC_CONSENSUS: PatristicConsensusStrategy,
        QueryPurpose.GENERAL_SIMILARITY: GeneralSimilarityStrategy,
    }

    def __init__(self):
        self._strategy_cache: Dict[QueryPurpose, QueryStrategy] = {}

    def get_strategy(self, purpose: QueryPurpose) -> QueryStrategy:
        """Get strategy for specified purpose."""
        if purpose not in self._strategy_cache:
            strategy_class = self.STRATEGIES.get(purpose, GeneralSimilarityStrategy)
            self._strategy_cache[purpose] = strategy_class()
        return self._strategy_cache[purpose]

    def detect_purpose(self, text: str) -> QueryPurpose:
        """
        Auto-detect query purpose from text content.

        Uses keyword analysis to suggest appropriate strategy.
        """
        text_lower = text.lower()

        # Check for specific indicators
        if any(word in text_lower for word in ["father", "chrysostom", "basil", "gregory", "patristic"]):
            return QueryPurpose.PATRISTIC_CONSENSUS

        if any(word in text_lower for word in ["covenant", "promise", "oath", "sign"]):
            return QueryPurpose.COVENANT_STUDY

        if any(word in text_lower for word in ["feast", "liturgy", "pascha", "nativity", "theophany"]):
            return QueryPurpose.LITURGICAL_CONTEXT

        if any(word in text_lower for word in ["prophecy", "fulfill", "foretold", "messiah"]):
            return QueryPurpose.PROPHECY_CHAIN

        if any(word in text_lower for word in ["type", "antitype", "shadow", "figure", "pattern"]):
            return QueryPurpose.TYPOLOGY_DISCOVERY

        if any(word in text_lower for word in ["cross-reference", "related", "similar", "connection"]):
            return QueryPurpose.CROSS_REFERENCE

        # Default to theological for substantive queries, general for short ones
        if len(text.split()) > 10:
            return QueryPurpose.THEOLOGICAL_ANALYSIS

        return QueryPurpose.GENERAL_SIMILARITY

    def get_all_strategies(self) -> Dict[str, QueryStrategy]:
        """Get all available strategies."""
        return {
            purpose.value: self.get_strategy(purpose)
            for purpose in QueryPurpose
        }


class StrategicVectorSearch:
    """
    High-level interface for strategic multi-domain search.

    Combines:
    - Strategy selection
    - Embedding generation
    - Vector store search
    - Result post-processing
    """

    def __init__(
        self,
        vector_store: MultiVectorStore,
        embedder: Optional[MultiDomainEmbedder] = None
    ):
        """
        Initialize strategic search.

        Args:
            vector_store: Multi-domain vector store
            embedder: Multi-domain embedder (created if not provided)
        """
        self.vector_store = vector_store
        self.embedder = embedder or MultiDomainEmbedder(lazy_load=True)
        self.selector = StrategySelector()

    async def search(
        self,
        query_text: str,
        purpose: Optional[QueryPurpose] = None,
        context: Optional[VerseContext] = None,
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform strategic search.

        Args:
            query_text: Query text
            purpose: Explicit query purpose (auto-detected if None)
            context: Additional context for adaptive weighting
            top_k: Number of results
            filter_conditions: Optional metadata filters

        Returns:
            Ranked search results
        """
        # Detect or use specified purpose
        if purpose is None:
            purpose = self.selector.detect_purpose(query_text)
            logger.info(f"Auto-detected query purpose: {purpose.value}")

        # Get strategy
        strategy = self.selector.get_strategy(purpose)

        # Create context if not provided
        if context is None:
            context = VerseContext(
                verse_id="QUERY",
                text=query_text,
                testament="old",  # Default
                book="GEN",
                chapter=0,
                verse=0
            )

        # Get adaptive weights
        weights = strategy.get_weights(context)

        # Generate embeddings for active domains
        active_domains = [d for d, w in weights.items() if w > 0]
        embeddings = self.embedder.embed_verse(context, domains=active_domains)

        # Build query vectors dict
        query_vectors = {
            domain: emb for domain, emb in embeddings.items()
            if emb is not None
        }

        # Get search config
        config = strategy.get_search_config()

        # Perform hybrid search
        results = await self.vector_store.hybrid_search(
            query_vectors=query_vectors,
            weights=weights,
            top_k=top_k * 2,  # Get extra for post-processing
            filter_conditions=filter_conditions,
            config=config
        )

        # Apply strategy-specific post-processing
        results = strategy.post_process(results)

        return results[:top_k]

    async def search_similar_to_verse(
        self,
        verse_id: str,
        purpose: Optional[QueryPurpose] = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Find verses similar to a given verse using strategic search.

        Args:
            verse_id: Source verse ID
            purpose: Query purpose (affects domain weighting)
            top_k: Number of results

        Returns:
            Similar verses
        """
        # Get strategy
        if purpose is None:
            purpose = QueryPurpose.CROSS_REFERENCE
        strategy = self.selector.get_strategy(purpose)

        weights = strategy.get_weights()
        config = strategy.get_search_config()

        # Get embeddings for source verse from each domain
        query_vectors = {}
        for domain in EmbeddingDomain:
            if weights.get(domain, 0) > 0:
                embedding = await self.vector_store.get_embedding(domain, verse_id)
                if embedding is not None:
                    query_vectors[domain] = embedding

        if not query_vectors:
            logger.warning(f"No embeddings found for {verse_id}")
            return []

        # Perform hybrid search
        results = await self.vector_store.hybrid_search(
            query_vectors=query_vectors,
            weights=weights,
            top_k=top_k + 1,  # Extra for self-exclusion
            config=config
        )

        # Remove self
        results = [r for r in results if r.verse_id != verse_id]

        return results[:top_k]

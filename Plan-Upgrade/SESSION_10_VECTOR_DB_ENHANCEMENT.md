# SESSION 10: VECTOR DB ENHANCEMENT

## Session Overview

**Objective**: Transform the vector database from a simple embedding store to a multi-vector hybrid search system with specialized embeddings for patristic interpretation, liturgical context, and typological patterns. Implement advanced retrieval strategies for the oracle engines.

**Estimated Duration**: 1 Claude session (90-120 minutes of focused implementation)

**Prerequisites**:
- Understanding of existing embedding infrastructure (`ml/embeddings/`)
- Session 01 complete (Mutual Transformation - for embedding comparison)
- Session 06 complete (Fractal Typology - for pattern embeddings)
- Familiarity with vector databases (Qdrant, Pinecone, or Weaviate)

---

## Part 1: Understanding Multi-Vector Architecture

### Core Concept
Instead of a single embedding per verse, maintain multiple specialized embedding vectors that capture different semantic dimensions:

1. **Semantic Embedding**: General meaning (existing)
2. **Patristic Embedding**: Captures how Church Fathers interpreted this text
3. **Liturgical Embedding**: Captures liturgical usage and context
4. **Typological Embedding**: Captures type-antitype pattern signatures
5. **Covenantal Embedding**: Captures position in covenant arc
6. **Prophetic Embedding**: Captures prophetic content and fulfillment markers

### Why Multi-Vector for BIBLOS

1. **Precision Search**: Query with specific theological intent
2. **Cross-Modal Matching**: Find liturgical uses of a verse, or patristic interpretations
3. **Pattern Discovery**: Similar typological patterns cluster together
4. **Semantic Decomposition**: Understand which aspect of a verse matches
5. **Oracle Support**: Each oracle engine has optimized retrieval

### Embedding Space Configuration

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

class EmbeddingDomain(Enum):
    """Specialized embedding domains for biblical data."""
    SEMANTIC = "semantic"
    PATRISTIC = "patristic"
    LITURGICAL = "liturgical"
    TYPOLOGICAL = "typological"
    COVENANTAL = "covenantal"
    PROPHETIC = "prophetic"

    @property
    def dimension(self) -> int:
        """Vector dimension for this domain."""
        return {
            EmbeddingDomain.SEMANTIC: 384,
            EmbeddingDomain.PATRISTIC: 384,
            EmbeddingDomain.LITURGICAL: 256,  # Smaller for simpler structure
            EmbeddingDomain.TYPOLOGICAL: 384,
            EmbeddingDomain.COVENANTAL: 256,
            EmbeddingDomain.PROPHETIC: 384,
        }[self]

    @property
    def model_id(self) -> str:
        """Preferred model for this domain."""
        return {
            EmbeddingDomain.SEMANTIC: "paraphrase-multilingual-MiniLM-L12-v2",
            EmbeddingDomain.PATRISTIC: "paraphrase-multilingual-MiniLM-L12-v2",
            EmbeddingDomain.LITURGICAL: "all-MiniLM-L6-v2",
            EmbeddingDomain.TYPOLOGICAL: "paraphrase-multilingual-MiniLM-L12-v2",
            EmbeddingDomain.COVENANTAL: "all-MiniLM-L6-v2",
            EmbeddingDomain.PROPHETIC: "paraphrase-multilingual-MiniLM-L12-v2",
        }[self]

    @property
    def expected_density(self) -> float:
        """Expected population density (fraction of verses with this embedding)."""
        return {
            EmbeddingDomain.SEMANTIC: 1.0,      # All verses
            EmbeddingDomain.PATRISTIC: 0.15,    # ~15% have patristic commentary
            EmbeddingDomain.LITURGICAL: 0.08,   # ~8% in liturgical use
            EmbeddingDomain.TYPOLOGICAL: 0.12,  # ~12% participate in typology
            EmbeddingDomain.COVENANTAL: 0.05,   # ~5% directly covenant-related
            EmbeddingDomain.PROPHETIC: 0.10,    # ~10% prophetic/fulfillment
        }[self]

@dataclass
class DomainStatistics:
    """Statistics for an embedding domain."""
    domain: EmbeddingDomain
    vector_count: int
    avg_norm: float
    centroid: np.ndarray
    variance: float

    def coverage(self, total_verses: int) -> float:
        """Actual coverage compared to expected."""
        expected = self.domain.expected_density * total_verses
        return self.vector_count / expected if expected > 0 else 0.0
```

### Hybrid Search Strategy

Combine multiple signals for comprehensive retrieval:
```
Final Score = α × semantic_sim
            + β × patristic_sim
            + γ × typological_sim
            + δ × keyword_match
            + ε × graph_proximity
```

Where weights (α, β, γ, δ, ε) are query-dependent.

---

## Part 2: Embedding Type Specifications

### File: `ml/embeddings/multi_vector.py`

**Embedding Types**:

#### 1. `SemanticEmbedding` (Existing, Enhanced)
```python
@dataclass
class SemanticEmbedding:
    verse_id: str
    vector: np.ndarray          # 384-dim from sentence-transformers
    model_id: str               # e.g., "paraphrase-multilingual-MiniLM-L12-v2"
    language: str               # "hebrew", "greek", "english"
    text_used: str              # Source text for embedding
    created_at: datetime

    # Quality metrics
    norm: float = field(default=0.0)
    is_normalized: bool = field(default=True)

    def __post_init__(self):
        if self.is_normalized:
            self.vector = self.vector / (np.linalg.norm(self.vector) + 1e-8)
        self.norm = np.linalg.norm(self.vector)

    def similarity(self, other: "SemanticEmbedding") -> float:
        """Cosine similarity with another embedding."""
        return float(np.dot(self.vector, other.vector))
```

#### 2. `PatristicEmbedding`
```python
@dataclass
class PatristicEmbedding:
    verse_id: str
    vector: np.ndarray          # 384-dim
    interpretation_sources: List[str]  # Father IDs who interpreted
    interpretation_summary: str  # Aggregated interpretation text
    consensus_level: float      # Agreement among Fathers (0-1)
    traditions: List[str]       # ["Eastern", "Western", etc.]
    fourfold_senses: Dict[str, float]  # {literal: 0.4, allegorical: 0.3, ...}

    # Weighting by Father authority
    authority_weighted: bool = True
    total_authority_weight: float = 0.0

    def dominant_sense(self) -> str:
        """Return the most emphasized fourfold sense."""
        if not self.fourfold_senses:
            return "literal"
        return max(self.fourfold_senses, key=self.fourfold_senses.get)

    def is_consensus(self, threshold: float = 0.7) -> bool:
        """Whether Fathers agree on interpretation."""
        return self.consensus_level >= threshold
```

#### 3. `LiturgicalEmbedding`
```python
class LiturgicalSeason(Enum):
    """Major liturgical seasons."""
    ORDINARY = "ordinary"
    ADVENT = "advent"
    NATIVITY = "nativity"
    THEOPHANY = "theophany"
    LENT = "lent"
    PASCHA = "pascha"
    PENTECOST = "pentecost"
    DORMITION = "dormition"

@dataclass
class LiturgicalEmbedding:
    verse_id: str
    vector: np.ndarray          # 256-dim
    liturgical_uses: List[str]  # ["Paschal Vigil", "Vespers", etc.]
    feast_associations: List[str]  # Feasts where read
    hymn_references: List[str]  # Hymns that quote/reference
    liturgical_weight: float    # Centrality in liturgical life
    tradition: str              # "Byzantine", "Roman", etc.

    # Seasonal context
    seasons: List[LiturgicalSeason] = field(default_factory=list)
    is_prokeimenon: bool = False
    is_alleluia_verse: bool = False
    is_communion_verse: bool = False

    def seasonal_relevance(self, season: LiturgicalSeason) -> float:
        """Relevance score for a specific season."""
        if season in self.seasons:
            return 1.0
        return 0.2  # Base relevance for universal texts
```

#### 4. `TypologicalEmbedding`
```python
@dataclass
class TypologicalEmbedding:
    verse_id: str
    vector: np.ndarray          # 384-dim
    pattern_ids: List[str]      # TypePattern IDs from Session 06
    type_role: str              # "type", "antitype", "both"
    fractal_layers: List[str]   # Active layers for this verse
    pattern_strength: float     # Composite typological strength

    # Connection details from Session 06
    correspondence_types: List[str] = field(default_factory=list)
    connected_verses: List[str] = field(default_factory=list)

    def is_hub(self, min_connections: int = 3) -> bool:
        """Whether this verse is a typological hub."""
        return len(self.connected_verses) >= min_connections

    def primary_pattern(self) -> Optional[str]:
        """Return the strongest pattern for this verse."""
        return self.pattern_ids[0] if self.pattern_ids else None
```

#### 5. `CovenantEmbedding`
```python
class CovenantRole(Enum):
    """Role within covenant structure."""
    INITIATION = "initiation"
    PROMISE = "promise"
    CONDITION = "condition"
    SIGN = "sign"
    FULFILLMENT = "fulfillment"
    RENEWAL = "renewal"

@dataclass
class CovenantEmbedding:
    verse_id: str
    vector: np.ndarray          # 256-dim
    covenant_ids: List[str]     # Covenants this verse relates to
    covenant_role: CovenantRole
    promise_elements: List[str] # ["seed", "land", "blessing"]
    arc_position: float         # Position in covenant narrative (0-1)

    # Covenant progression tracking
    progression_index: int = 0  # 0=Adamic, 1=Noahic, 2=Abrahamic, etc.
    bridges_covenants: bool = False  # True if connects multiple covenants

    def is_key_verse(self) -> bool:
        """Whether this is a key covenant verse."""
        return self.covenant_role in {CovenantRole.INITIATION, CovenantRole.PROMISE}
```

#### 6. `PropheticEmbedding`
```python
class PropheticSpecificity(Enum):
    """Level of prophetic specificity."""
    GENERAL = "general"           # General themes
    TEMPORAL = "temporal"         # Time-related predictions
    GEOGRAPHICAL = "geographical" # Location-specific
    PERSONAL = "personal"         # Person-specific (names, roles)
    CIRCUMSTANTIAL = "circumstantial"  # Specific circumstances
    MIRACULOUS = "miraculous"     # Supernatural elements

@dataclass
class PropheticEmbedding:
    verse_id: str
    vector: np.ndarray          # 384-dim
    prophecy_ids: List[str]     # Related prophecies
    prophetic_role: str         # "prophecy", "fulfillment", "neither"
    specificity_markers: List[PropheticSpecificity]
    fulfillment_confidence: float  # If fulfillment, confidence level

    # From Session 07 integration
    natural_probability: Optional[float] = None
    necessity_score: Optional[float] = None
    bayesian_strength: Optional[str] = None

    def is_messianic(self) -> bool:
        """Whether this is a messianic prophecy/fulfillment."""
        return any("messianic" in pid.lower() for pid in self.prophecy_ids)
```

---

## Part 3: Embedding Generation Pipelines

### File: `ml/embeddings/generators/`

#### Base Generator with Caching
```python
from abc import ABC, abstractmethod
from functools import lru_cache
import hashlib

class EmbeddingGenerator(ABC):
    """Base class for embedding generation with caching."""

    def __init__(self, model_id: str, cache_dir: Optional[Path] = None):
        self.model = SentenceTransformer(model_id)
        self.model_id = model_id
        self.cache_dir = cache_dir
        self._cache: Dict[str, np.ndarray] = {}

    @abstractmethod
    async def generate(self, verse_id: str, context: Dict) -> Optional[Any]:
        """Generate embedding for a verse."""
        pass

    async def batch_generate(
        self,
        verses: List[str],
        contexts: List[Dict],
        batch_size: int = 32
    ) -> List[Any]:
        """Batch generate with batching for efficiency."""
        results = []
        for i in range(0, len(verses), batch_size):
            batch_verses = verses[i:i + batch_size]
            batch_contexts = contexts[i:i + batch_size]

            batch_results = await asyncio.gather(*[
                self.generate(v, c) for v, c in zip(batch_verses, batch_contexts)
            ])
            results.extend(batch_results)

        return results

    def _cache_key(self, verse_id: str, context_hash: str) -> str:
        """Generate cache key."""
        return f"{self.model_id}:{verse_id}:{context_hash}"

    def _hash_context(self, context: Dict) -> str:
        """Hash context for cache key."""
        return hashlib.md5(str(sorted(context.items())).encode()).hexdigest()[:8]
```

#### Patristic Embedding Generator
```python
class PatristicEmbeddingGenerator(EmbeddingGenerator):
    """
    Generate embeddings from aggregated patristic interpretation.
    """

    def __init__(self, model_id: str, patristic_db, min_fathers: int = 1):
        super().__init__(model_id)
        self.patristic_db = patristic_db
        self.min_fathers = min_fathers

    async def generate(self, verse_id: str, context: Dict = None) -> Optional[PatristicEmbedding]:
        # Gather all patristic interpretations
        interpretations = await self.patristic_db.get_interpretations(verse_id)

        if len(interpretations) < self.min_fathers:
            return None  # Insufficient patristic data

        # Aggregate interpretation texts with authority weighting
        aggregated_text, total_weight = self._aggregate_interpretations(interpretations)

        # Generate embedding
        vector = self.model.encode(aggregated_text, normalize_embeddings=True)

        # Calculate consensus using pairwise similarity
        consensus = self._calculate_consensus(interpretations)

        # Identify traditions
        traditions = list(set(i.father_tradition for i in interpretations))

        # Calculate fourfold sense distribution
        fourfold = self._calculate_fourfold_distribution(interpretations)

        return PatristicEmbedding(
            verse_id=verse_id,
            vector=vector,
            interpretation_sources=[i.father_id for i in interpretations],
            interpretation_summary=aggregated_text[:500],
            consensus_level=consensus,
            traditions=traditions,
            fourfold_senses=fourfold,
            authority_weighted=True,
            total_authority_weight=total_weight
        )

    def _aggregate_interpretations(self, interpretations: List) -> Tuple[str, float]:
        """Combine interpretations with authority weighting."""
        weighted_texts = []
        total_weight = 0.0

        for interp in interpretations:
            weight = interp.father_authority_weight
            weighted_texts.append((weight, interp.interpretation))
            total_weight += weight

        # Sort by weight, take top N
        weighted_texts.sort(key=lambda x: -x[0])
        top_interpretations = [t[1] for t in weighted_texts[:5]]

        return " ".join(top_interpretations), total_weight

    def _calculate_consensus(self, interpretations: List) -> float:
        """Measure agreement among Fathers using embedding similarity."""
        if len(interpretations) < 2:
            return 1.0

        # Embed each interpretation
        vectors = [
            self.model.encode(i.interpretation, normalize_embeddings=True)
            for i in interpretations
        ]

        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sim = float(np.dot(vectors[i], vectors[j]))
                similarities.append(sim)

        return np.mean(similarities) if similarities else 1.0

    def _calculate_fourfold_distribution(self, interpretations: List) -> Dict[str, float]:
        """Calculate distribution of fourfold senses."""
        sense_counts = {"literal": 0, "allegorical": 0, "tropological": 0, "anagogical": 0}

        for interp in interpretations:
            if hasattr(interp, 'fourfold_sense') and interp.fourfold_sense:
                sense = interp.fourfold_sense.lower()
                if sense in sense_counts:
                    sense_counts[sense] += interp.father_authority_weight

        total = sum(sense_counts.values())
        if total > 0:
            return {k: v / total for k, v in sense_counts.items()}
        return {"literal": 1.0, "allegorical": 0.0, "tropological": 0.0, "anagogical": 0.0}
```

#### Typological Embedding Generator
```python
class TypologicalEmbeddingGenerator(EmbeddingGenerator):
    """
    Generate embeddings from typological pattern context.
    Integrates with Session 06 Fractal Typology Engine.
    """

    def __init__(self, model_id: str, typology_engine, neo4j_client):
        super().__init__(model_id)
        self.typology_engine = typology_engine
        self.neo4j = neo4j_client

    async def generate(self, verse_id: str, context: Dict = None) -> Optional[TypologicalEmbedding]:
        # Get typological connections for this verse from graph
        connections = await self.neo4j.execute("""
            MATCH (v:Verse {id: $verse_id})-[t:TYPIFIES]-(connected:Verse)
            RETURN type(t) AS rel_type,
                   t.pattern_id AS pattern_id,
                   t.composite_strength AS strength,
                   t.dominant_layer AS layer,
                   t.correspondence_type AS correspondence,
                   connected.id AS connected_id,
                   CASE WHEN startNode(t).id = $verse_id THEN 'type' ELSE 'antitype' END AS role
        """, verse_id=verse_id)

        if not connections:
            return None

        # Determine role
        roles = set(c["role"] for c in connections)
        if len(roles) > 1:
            role = "both"
        else:
            role = list(roles)[0]

        # Build pattern context text for embedding
        pattern_ids = list(set(c["pattern_id"] for c in connections if c["pattern_id"]))
        layers = list(set(c["layer"] for c in connections if c["layer"]))
        correspondences = list(set(c["correspondence"] for c in connections if c["correspondence"]))
        connected_verses = list(set(c["connected_id"] for c in connections))

        # Generate context text describing typological position
        context_text = self._build_typological_context(
            verse_id, role, pattern_ids, layers, connections
        )

        # Generate embedding
        vector = self.model.encode(context_text, normalize_embeddings=True)

        # Calculate average strength
        strengths = [c["strength"] for c in connections if c["strength"]]
        avg_strength = np.mean(strengths) if strengths else 0.0

        return TypologicalEmbedding(
            verse_id=verse_id,
            vector=vector,
            pattern_ids=pattern_ids,
            type_role=role,
            fractal_layers=layers,
            pattern_strength=avg_strength,
            correspondence_types=correspondences,
            connected_verses=connected_verses
        )

    def _build_typological_context(
        self,
        verse_id: str,
        role: str,
        patterns: List[str],
        layers: List[str],
        connections: List[Dict]
    ) -> str:
        """Build descriptive text for typological embedding."""
        parts = [f"This verse serves as {role} in typological pattern"]

        if patterns:
            parts.append(f"in patterns: {', '.join(patterns)}")

        if layers:
            parts.append(f"operating at layers: {', '.join(layers)}")

        # Add connection summaries
        for conn in connections[:3]:  # Top 3 connections
            parts.append(
                f"connects to {conn['connected_id']} via {conn.get('correspondence', 'unknown')} correspondence"
            )

        return ". ".join(parts)
```

---

## Part 4: Vector Database Schema

### File: `db/vector_store.py`

**Multi-Collection Architecture**:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter,
    FieldCondition, MatchValue, MatchAny, OptimizersConfigDiff,
    PayloadSchemaType, PayloadIndexParams
)

class MultiVectorStore:
    """
    Multi-collection vector store with hybrid search.
    """

    COLLECTION_CONFIG = {
        EmbeddingDomain.SEMANTIC: {
            "name": "biblos_semantic",
            "dimension": 384,
            "distance": Distance.COSINE,
            "indexed_fields": ["verse_id", "book", "testament", "language"]
        },
        EmbeddingDomain.PATRISTIC: {
            "name": "biblos_patristic",
            "dimension": 384,
            "distance": Distance.COSINE,
            "indexed_fields": ["verse_id", "traditions", "consensus_level", "dominant_sense"]
        },
        EmbeddingDomain.LITURGICAL: {
            "name": "biblos_liturgical",
            "dimension": 256,
            "distance": Distance.COSINE,
            "indexed_fields": ["verse_id", "liturgical_uses", "feast_associations", "tradition"]
        },
        EmbeddingDomain.TYPOLOGICAL: {
            "name": "biblos_typological",
            "dimension": 384,
            "distance": Distance.COSINE,
            "indexed_fields": ["verse_id", "pattern_ids", "type_role", "fractal_layers"]
        },
        EmbeddingDomain.COVENANTAL: {
            "name": "biblos_covenantal",
            "dimension": 256,
            "distance": Distance.COSINE,
            "indexed_fields": ["verse_id", "covenant_ids", "covenant_role", "promise_elements"]
        },
        EmbeddingDomain.PROPHETIC: {
            "name": "biblos_prophetic",
            "dimension": 384,
            "distance": Distance.COSINE,
            "indexed_fields": ["verse_id", "prophecy_ids", "prophetic_role", "specificity_markers"]
        }
    }

    def __init__(self, qdrant_client: QdrantClient):
        self.client = qdrant_client
        self._id_cache: Dict[str, int] = {}
        self._next_id = 0

    async def create_collections(self) -> None:
        """Create all embedding collections with appropriate configs."""
        for domain, config in self.COLLECTION_CONFIG.items():
            # Check if collection exists
            collections = await self.client.get_collections()
            if config["name"] in [c.name for c in collections.collections]:
                continue

            await self.client.create_collection(
                collection_name=config["name"],
                vectors_config=VectorParams(
                    size=config["dimension"],
                    distance=config["distance"]
                ),
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=10000
                )
            )

            # Create payload indexes
            for field in config["indexed_fields"]:
                await self.client.create_payload_index(
                    collection_name=config["name"],
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD
                )

    def _verse_to_id(self, verse_id: str) -> int:
        """Convert verse ID to numeric ID for Qdrant."""
        if verse_id not in self._id_cache:
            self._id_cache[verse_id] = self._next_id
            self._next_id += 1
        return self._id_cache[verse_id]

    async def upsert_embedding(
        self,
        domain: EmbeddingDomain,
        verse_id: str,
        vector: np.ndarray,
        payload: Dict
    ) -> None:
        """Upsert embedding to appropriate collection."""
        config = self.COLLECTION_CONFIG[domain]
        await self.client.upsert(
            collection_name=config["name"],
            points=[PointStruct(
                id=self._verse_to_id(verse_id),
                vector=vector.tolist(),
                payload={"verse_id": verse_id, **payload}
            )]
        )

    async def search(
        self,
        domain: EmbeddingDomain,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[Filter] = None,
        score_threshold: float = 0.0
    ) -> List[Dict]:
        """Search single collection with optional filtering."""
        config = self.COLLECTION_CONFIG[domain]
        results = await self.client.search(
            collection_name=config["name"],
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=filters,
            score_threshold=score_threshold
        )
        return [
            {
                "verse_id": r.payload["verse_id"],
                "score": r.score,
                "domain": domain.value,
                **r.payload
            }
            for r in results
        ]

    async def hybrid_search(
        self,
        query_vectors: Dict[EmbeddingDomain, np.ndarray],
        weights: Dict[EmbeddingDomain, float],
        top_k: int = 10,
        filters: Optional[Dict[EmbeddingDomain, Filter]] = None,
        aggregation: str = "weighted_sum"  # weighted_sum, max, rrf
    ) -> List[Dict]:
        """
        Hybrid search across multiple collections with weighted aggregation.
        """
        all_results: Dict[str, Dict] = {}

        # Search each collection in parallel
        search_tasks = []
        for domain, vector in query_vectors.items():
            weight = weights.get(domain, 0.0)
            if weight == 0:
                continue

            collection_filter = filters.get(domain) if filters else None
            search_tasks.append(
                self._search_with_weight(domain, vector, weight, top_k * 2, collection_filter)
            )

        domain_results = await asyncio.gather(*search_tasks)

        # Aggregate results
        for results, domain, weight in domain_results:
            for r in results:
                verse_id = r["verse_id"]
                if verse_id not in all_results:
                    all_results[verse_id] = {
                        "verse_id": verse_id,
                        "scores": {},
                        "weighted_total": 0.0,
                        "contributing_domains": []
                    }

                all_results[verse_id]["scores"][domain.value] = r["score"]
                all_results[verse_id]["contributing_domains"].append(domain.value)

                if aggregation == "weighted_sum":
                    all_results[verse_id]["weighted_total"] += r["score"] * weight
                elif aggregation == "max":
                    all_results[verse_id]["weighted_total"] = max(
                        all_results[verse_id]["weighted_total"],
                        r["score"] * weight
                    )

        # Apply RRF if requested
        if aggregation == "rrf":
            all_results = self._apply_rrf(all_results, query_vectors.keys())

        # Sort and return top_k
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: -x["weighted_total"]
        )[:top_k]

        return sorted_results

    def _apply_rrf(self, results: Dict, domains: List[EmbeddingDomain], k: int = 60) -> Dict:
        """Apply Reciprocal Rank Fusion for aggregation."""
        # Build per-domain rankings
        domain_rankings = {d.value: [] for d in domains}

        for verse_id, data in results.items():
            for domain_val, score in data["scores"].items():
                domain_rankings[domain_val].append((verse_id, score))

        # Sort each domain ranking
        for domain_val in domain_rankings:
            domain_rankings[domain_val].sort(key=lambda x: -x[1])

        # Calculate RRF scores
        rrf_scores = {}
        for domain_val, ranking in domain_rankings.items():
            for rank, (verse_id, _) in enumerate(ranking):
                if verse_id not in rrf_scores:
                    rrf_scores[verse_id] = 0.0
                rrf_scores[verse_id] += 1.0 / (k + rank + 1)

        # Update weighted_total with RRF scores
        for verse_id in results:
            results[verse_id]["weighted_total"] = rrf_scores.get(verse_id, 0.0)

        return results

    async def _search_with_weight(
        self,
        domain: EmbeddingDomain,
        vector: np.ndarray,
        weight: float,
        top_k: int,
        filter_: Optional[Filter]
    ) -> Tuple[List[Dict], EmbeddingDomain, float]:
        """Search wrapper that returns domain and weight with results."""
        results = await self.search(domain, vector, top_k, filter_)
        return results, domain, weight
```

---

## Part 5: Query Strategies

### File: `ml/embeddings/query_strategies.py`

**Strategy Pattern for Different Use Cases**:

```python
from abc import ABC, abstractmethod

class QueryStrategy(ABC):
    """Base query strategy."""

    @abstractmethod
    def get_weights(self) -> Dict[EmbeddingDomain, float]:
        """Return domain weights for this strategy."""
        pass

    def get_filters(self, context: Dict) -> Dict[EmbeddingDomain, Filter]:
        """Return per-domain filters."""
        return {}

    def get_aggregation(self) -> str:
        """Return aggregation method."""
        return "weighted_sum"

    @property
    def name(self) -> str:
        """Strategy name for logging."""
        return self.__class__.__name__

class TheologicalQueryStrategy(QueryStrategy):
    """
    For theological research queries.
    Emphasizes patristic and typological dimensions.
    """

    def get_weights(self) -> Dict[EmbeddingDomain, float]:
        return {
            EmbeddingDomain.SEMANTIC: 0.25,
            EmbeddingDomain.PATRISTIC: 0.35,
            EmbeddingDomain.TYPOLOGICAL: 0.25,
            EmbeddingDomain.COVENANTAL: 0.15
        }

    def get_filters(self, context: Dict) -> Dict[EmbeddingDomain, Filter]:
        filters = {}
        if "tradition" in context:
            filters[EmbeddingDomain.PATRISTIC] = Filter(
                must=[FieldCondition(
                    key="traditions",
                    match=MatchAny(any=context["tradition"] if isinstance(context["tradition"], list) else [context["tradition"]])
                )]
            )
        if "min_consensus" in context:
            filters[EmbeddingDomain.PATRISTIC] = Filter(
                must=[FieldCondition(
                    key="consensus_level",
                    range={"gte": context["min_consensus"]}
                )]
            )
        return filters

class LiturgicalQueryStrategy(QueryStrategy):
    """
    For finding liturgically related verses.
    Emphasizes liturgical embedding heavily.
    """

    def get_weights(self) -> Dict[EmbeddingDomain, float]:
        return {
            EmbeddingDomain.SEMANTIC: 0.15,
            EmbeddingDomain.LITURGICAL: 0.60,
            EmbeddingDomain.PATRISTIC: 0.25
        }

    def get_filters(self, context: Dict) -> Dict[EmbeddingDomain, Filter]:
        filters = {}
        if "feast" in context:
            filters[EmbeddingDomain.LITURGICAL] = Filter(
                must=[FieldCondition(
                    key="feast_associations",
                    match=MatchAny(any=[context["feast"]])
                )]
            )
        if "service" in context:
            filters[EmbeddingDomain.LITURGICAL] = Filter(
                must=[FieldCondition(
                    key="liturgical_uses",
                    match=MatchAny(any=[context["service"]])
                )]
            )
        return filters

class TypologicalQueryStrategy(QueryStrategy):
    """
    For discovering typological connections.
    Emphasizes typological patterns.
    """

    def get_weights(self) -> Dict[EmbeddingDomain, float]:
        return {
            EmbeddingDomain.SEMANTIC: 0.15,
            EmbeddingDomain.TYPOLOGICAL: 0.50,
            EmbeddingDomain.COVENANTAL: 0.20,
            EmbeddingDomain.PROPHETIC: 0.15
        }

    def get_filters(self, context: Dict) -> Dict[EmbeddingDomain, Filter]:
        filters = {}
        if "pattern_id" in context:
            filters[EmbeddingDomain.TYPOLOGICAL] = Filter(
                must=[FieldCondition(
                    key="pattern_ids",
                    match=MatchAny(any=[context["pattern_id"]])
                )]
            )
        if "layer" in context:
            filters[EmbeddingDomain.TYPOLOGICAL] = Filter(
                must=[FieldCondition(
                    key="fractal_layers",
                    match=MatchAny(any=[context["layer"]])
                )]
            )
        if "role" in context:
            filters[EmbeddingDomain.TYPOLOGICAL] = Filter(
                must=[FieldCondition(
                    key="type_role",
                    match=MatchValue(value=context["role"])
                )]
            )
        return filters

class PropheticQueryStrategy(QueryStrategy):
    """
    For analyzing prophetic connections.
    Emphasizes prophetic embeddings.
    """

    def get_weights(self) -> Dict[EmbeddingDomain, float]:
        return {
            EmbeddingDomain.SEMANTIC: 0.15,
            EmbeddingDomain.PROPHETIC: 0.50,
            EmbeddingDomain.TYPOLOGICAL: 0.20,
            EmbeddingDomain.COVENANTAL: 0.15
        }

    def get_filters(self, context: Dict) -> Dict[EmbeddingDomain, Filter]:
        filters = {}
        if "prophetic_role" in context:
            filters[EmbeddingDomain.PROPHETIC] = Filter(
                must=[FieldCondition(
                    key="prophetic_role",
                    match=MatchValue(value=context["prophetic_role"])
                )]
            )
        if "messianic_only" in context and context["messianic_only"]:
            filters[EmbeddingDomain.PROPHETIC] = Filter(
                must=[FieldCondition(
                    key="is_messianic",
                    match=MatchValue(value=True)
                )]
            )
        return filters

class CrossRefDiscoveryStrategy(QueryStrategy):
    """
    Balanced strategy for cross-reference discovery.
    Uses all embedding types with balanced weights.
    """

    def get_weights(self) -> Dict[EmbeddingDomain, float]:
        return {
            EmbeddingDomain.SEMANTIC: 0.25,
            EmbeddingDomain.PATRISTIC: 0.18,
            EmbeddingDomain.LITURGICAL: 0.10,
            EmbeddingDomain.TYPOLOGICAL: 0.20,
            EmbeddingDomain.COVENANTAL: 0.12,
            EmbeddingDomain.PROPHETIC: 0.15
        }

    def get_aggregation(self) -> str:
        """Use RRF for cross-ref discovery to balance diverse signals."""
        return "rrf"

# Strategy factory
STRATEGIES = {
    "theological": TheologicalQueryStrategy,
    "liturgical": LiturgicalQueryStrategy,
    "typological": TypologicalQueryStrategy,
    "prophetic": PropheticQueryStrategy,
    "cross_ref": CrossRefDiscoveryStrategy,
}

def get_strategy(name: str) -> QueryStrategy:
    """Get strategy by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]()
```

---

## Part 6: Integration with Oracle Engines

### Integration Points

**For Omni-Contextual Resolver (Session 03)**:
```python
class OmniContextualVectorIntegration:
    """Vector search integration for OmniContextual Resolver."""

    def __init__(self, vector_store: MultiVectorStore, corpus):
        self.vector_store = vector_store
        self.corpus = corpus

    async def get_occurrence_contexts(self, lemma: str) -> List[Dict]:
        """
        Get all occurrences of a lemma with rich embedding context.
        """
        # Find all verses containing this lemma
        verses = await self.corpus.get_verses_with_lemma(lemma)

        contexts = []
        for verse_id in verses:
            # Retrieve embeddings for this verse
            semantic = await self.vector_store.get_embedding(EmbeddingDomain.SEMANTIC, verse_id)
            patristic = await self.vector_store.get_embedding(EmbeddingDomain.PATRISTIC, verse_id)

            contexts.append({
                "verse_id": verse_id,
                "semantic_context": semantic,
                "patristic_interpretation": patristic.interpretation_summary if patristic else None,
                "consensus": patristic.consensus_level if patristic else 0,
                "dominant_sense": patristic.dominant_sense() if patristic else "literal"
            })

        return contexts

    async def find_similar_contexts(
        self,
        lemma: str,
        target_verse: str,
        top_k: int = 10
    ) -> List[Dict]:
        """Find verses with similar context for sense disambiguation."""
        target_embed = await self.vector_store.get_embedding(EmbeddingDomain.SEMANTIC, target_verse)
        if not target_embed:
            return []

        # Get verses with this lemma
        lemma_verses = set(await self.corpus.get_verses_with_lemma(lemma))

        # Search for similar semantic contexts
        results = await self.vector_store.search(
            domain=EmbeddingDomain.SEMANTIC,
            query_vector=target_embed.vector,
            top_k=top_k * 2
        )

        # Filter to only lemma verses
        return [r for r in results if r["verse_id"] in lemma_verses][:top_k]
```

**For Fractal Typology Engine (Session 06)**:
```python
class TypologyVectorIntegration:
    """Vector search integration for Fractal Typology Engine."""

    def __init__(self, vector_store: MultiVectorStore, neo4j_client):
        self.vector_store = vector_store
        self.neo4j = neo4j_client

    async def find_pattern_matches(
        self,
        pattern_id: str,
        source_verse: str,
        search_testament: Optional[str] = None
    ) -> List[Dict]:
        """
        Find verses matching a typological pattern.
        """
        source_embed = await self.vector_store.get_embedding(
            EmbeddingDomain.TYPOLOGICAL, source_verse
        )

        if not source_embed:
            return []

        strategy = TypologicalQueryStrategy()
        context = {"pattern_id": pattern_id}

        if search_testament:
            context["testament"] = search_testament

        query_vectors = {EmbeddingDomain.TYPOLOGICAL: source_embed.vector}

        # Also include semantic for content matching
        semantic_embed = await self.vector_store.get_embedding(
            EmbeddingDomain.SEMANTIC, source_verse
        )
        if semantic_embed:
            query_vectors[EmbeddingDomain.SEMANTIC] = semantic_embed.vector

        results = await self.vector_store.hybrid_search(
            query_vectors=query_vectors,
            weights=strategy.get_weights(),
            filters=strategy.get_filters(context),
            top_k=20
        )

        return results

    async def discover_new_patterns(
        self,
        verse_id: str,
        min_similarity: float = 0.7
    ) -> List[Dict]:
        """Discover potential new typological patterns."""
        semantic_embed = await self.vector_store.get_embedding(
            EmbeddingDomain.SEMANTIC, verse_id
        )

        if not semantic_embed:
            return []

        # Search across both testaments
        results = await self.vector_store.search(
            domain=EmbeddingDomain.SEMANTIC,
            query_vector=semantic_embed.vector,
            top_k=50,
            score_threshold=min_similarity
        )

        # Filter to opposite testament for typology candidates
        verse_testament = await self.neo4j.execute(
            "MATCH (v:Verse {id: $id}) RETURN v.testament AS testament",
            id=verse_id
        )
        current_testament = verse_testament[0]["testament"] if verse_testament else None

        candidates = []
        for r in results:
            if r["verse_id"] != verse_id:
                target_testament = r.get("testament")
                if target_testament and target_testament != current_testament:
                    candidates.append(r)

        return candidates
```

---

## Part 7: Event-Driven Updates

### Integration with Session 08 Event Sourcing

**Vector Store Event Projection**:
```python
class VectorStoreProjection(ProjectionBase):
    """
    Updates vector store based on domain events.
    """

    def __init__(
        self,
        vector_store: MultiVectorStore,
        generators: Dict[EmbeddingDomain, EmbeddingGenerator]
    ):
        super().__init__(projection_name="vector_store")
        self.vector_store = vector_store
        self.generators = generators
        self._pending_updates: Dict[str, Set[EmbeddingDomain]] = {}
        self._batch_size = 50

    async def _handle_PatristicWitnessAddedEvent(self, event):
        """Regenerate patristic embedding when new interpretation added."""
        generator = self.generators[EmbeddingDomain.PATRISTIC]
        embedding = await generator.generate(event.target_id)

        if embedding:
            await self.vector_store.upsert_embedding(
                domain=EmbeddingDomain.PATRISTIC,
                verse_id=event.target_id,
                vector=embedding.vector,
                payload={
                    "interpretation_sources": embedding.interpretation_sources,
                    "consensus_level": embedding.consensus_level,
                    "traditions": embedding.traditions,
                    "fourfold_senses": embedding.fourfold_senses,
                    "dominant_sense": embedding.dominant_sense()
                }
            )

    async def _handle_TypologicalConnectionIdentifiedEvent(self, event):
        """Update typological embeddings for both type and antitype."""
        generator = self.generators[EmbeddingDomain.TYPOLOGICAL]

        for verse_id in [event.type_ref, event.antitype_ref]:
            embedding = await generator.generate(verse_id)

            if embedding:
                await self.vector_store.upsert_embedding(
                    domain=EmbeddingDomain.TYPOLOGICAL,
                    verse_id=verse_id,
                    vector=embedding.vector,
                    payload={
                        "pattern_ids": embedding.pattern_ids,
                        "type_role": embedding.type_role,
                        "fractal_layers": embedding.fractal_layers,
                        "pattern_strength": embedding.pattern_strength,
                        "connected_verses": embedding.connected_verses
                    }
                )

    async def _handle_ProphecyFulfillmentProvedEvent(self, event):
        """Update prophetic embeddings for prophecy and fulfillment."""
        generator = self.generators[EmbeddingDomain.PROPHETIC]

        for verse_id in [event.prophecy_verse, event.fulfillment_verse]:
            embedding = await generator.generate(verse_id)

            if embedding:
                await self.vector_store.upsert_embedding(
                    domain=EmbeddingDomain.PROPHETIC,
                    verse_id=verse_id,
                    vector=embedding.vector,
                    payload={
                        "prophecy_ids": embedding.prophecy_ids,
                        "prophetic_role": embedding.prophetic_role,
                        "specificity_markers": [s.value for s in embedding.specificity_markers],
                        "fulfillment_confidence": embedding.fulfillment_confidence,
                        "necessity_score": embedding.necessity_score
                    }
                )
```

---

## Part 8: Testing Specification

### Unit Tests: `tests/ml/embeddings/test_multi_vector.py`

**Test 1: `test_semantic_embedding_generation`**
- Generate semantic embedding for verse
- Verify dimension correct (384)
- Verify normalization (unit norm)
- Verify reproducibility

**Test 2: `test_patristic_embedding_consensus`**
- Add 5 similar patristic interpretations
- Generate patristic embedding
- Verify high consensus (> 0.7)
- Add 1 contradictory interpretation
- Verify lower consensus

**Test 3: `test_typological_embedding_patterns`**
- Create typological connections in graph
- Generate typological embedding
- Verify pattern IDs included
- Verify type_role correct

**Test 4: `test_hybrid_search_weighted_sum`**
- Index embeddings across multiple domains
- Perform hybrid search with custom weights
- Verify weights applied correctly
- Verify score calculation

**Test 5: `test_hybrid_search_rrf`**
- Index embeddings across multiple domains
- Perform hybrid search with RRF aggregation
- Verify RRF scores calculated correctly

**Test 6: `test_query_strategy_filters`**
- Use theological strategy with tradition filter
- Verify only matching traditions returned
- Use liturgical strategy with feast filter
- Verify only matching feasts returned

**Test 7: `test_event_projection_updates`**
- Add patristic witness via event
- Verify embedding regenerated
- Verify vector store updated
- Verify payload correct

---

## Part 9: Configuration

### Add to `config.py`

**New Configuration Dataclass**: `MultiVectorConfig`

```python
@dataclass
class MultiVectorConfig:
    """Configuration for multi-vector embedding system."""

    # Qdrant connection
    qdrant_host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    qdrant_port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    qdrant_grpc_port: int = 6334
    qdrant_api_key: Optional[str] = field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))

    # Model configuration
    default_embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimension: int = 384
    normalize_embeddings: bool = True

    # Batch processing
    batch_size: int = 100
    max_concurrent_requests: int = 10

    # Domain enablement
    enable_patristic_embeddings: bool = True
    enable_liturgical_embeddings: bool = True
    enable_typological_embeddings: bool = True
    enable_covenantal_embeddings: bool = True
    enable_prophetic_embeddings: bool = True

    # Search configuration
    default_strategy: str = "cross_ref"
    default_top_k: int = 10
    min_score_threshold: float = 0.3

    # Caching
    cache_embeddings: bool = True
    cache_ttl_hours: int = 168  # 1 week
    cache_max_size_mb: int = 1024

    # Event projection
    enable_event_projection: bool = True
    projection_batch_size: int = 50
    projection_flush_interval_seconds: int = 10
```

---

## Part 10: Success Criteria

### Functional Requirements
- [ ] All six embedding types generating correctly
- [ ] Multi-collection vector store operational
- [ ] Hybrid search working with weighted aggregation and RRF
- [ ] Query strategies selecting correct weights
- [ ] Event-driven updates working
- [ ] Integration with oracle engines functional

### Performance Requirements
- [ ] Single embedding generation: < 100ms
- [ ] Batch embedding (100 verses): < 10 seconds
- [ ] Single collection search: < 50ms
- [ ] Hybrid search (6 collections): < 300ms
- [ ] Full population (31,000 verses × 6 types): < 4 hours

### Quality Requirements
- [ ] Patristic embeddings capture Father consensus
- [ ] Liturgical embeddings reflect actual usage
- [ ] Typological embeddings cluster similar patterns
- [ ] Hybrid search improves over single-vector search

---

## Part 11: Detailed Implementation Order

1. **Create `ml/embeddings/multi_vector.py`** with dataclasses and enums
2. **Create `ml/embeddings/generators/`** directory with base class
3. **Implement `SemanticEmbeddingGenerator`** (enhance existing)
4. **Implement `PatristicEmbeddingGenerator`** with consensus calculation
5. **Implement `LiturgicalEmbeddingGenerator`** with seasonal context
6. **Implement `TypologicalEmbeddingGenerator`** with graph integration
7. **Implement `CovenantEmbeddingGenerator`** with arc positioning
8. **Implement `PropheticEmbeddingGenerator`** with specificity markers
9. **Create `db/vector_store.py`** with `MultiVectorStore`
10. **Create `ml/embeddings/query_strategies.py`** with all strategies
11. **Implement event projection** (integrate with Session 08)
12. **Create integration points** with oracle engines
13. **Add configuration to `config.py`**
14. **Write population script**
15. **Write unit tests**
16. **Run population for all verses**
17. **Benchmark hybrid search performance**

---

## Part 12: Dependencies on Other Sessions

### Depends On
- SESSION 01: Mutual Transformation (for embedding comparison logic)
- SESSION 06: Fractal Typology (for pattern embeddings and graph data)
- SESSION 08: Event Sourcing (for projection integration)
- SESSION 09: Neo4j Graph (for typological connection queries)

### Depended On By
- SESSION 11: Pipeline Integration (uses multi-vector search for enrichment)

### External Dependencies
- Qdrant vector database
- sentence-transformers library
- Patristic interpretation database
- Liturgical usage database

---

## Session Completion Checklist

```markdown
- [ ] `ml/embeddings/multi_vector.py` with all embedding types and enums
- [ ] All six embedding generators implemented with caching
- [ ] `db/vector_store.py` with MultiVectorStore and RRF aggregation
- [ ] All query strategies implemented with filtering
- [ ] Event projection for vector updates
- [ ] Integration with Session 03 (OmniContextual)
- [ ] Integration with Session 06 (Typology)
- [ ] Integration with Session 07 (Prophetic)
- [ ] Population script created
- [ ] Configuration added to config.py
- [ ] All embeddings populated
- [ ] Unit tests passing
- [ ] Hybrid search benchmarked
- [ ] Documentation complete
```

**Next Session**: SESSION 11: Pipeline Integration & Orchestration

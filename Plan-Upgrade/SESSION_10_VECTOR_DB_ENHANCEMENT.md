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
```

#### 3. `LiturgicalEmbedding`
```python
@dataclass
class LiturgicalEmbedding:
    verse_id: str
    vector: np.ndarray          # 384-dim
    liturgical_uses: List[str]  # ["Paschal Vigil", "Vespers", etc.]
    feast_associations: List[str]  # Feasts where read
    hymn_references: List[str]  # Hymns that quote/reference
    liturgical_weight: float    # Centrality in liturgical life
    tradition: str              # "Byzantine", "Roman", etc.
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
```

#### 5. `CovenantEmbedding`
```python
@dataclass
class CovenantEmbedding:
    verse_id: str
    vector: np.ndarray          # 384-dim
    covenant_ids: List[str]     # Covenants this verse relates to
    covenant_role: str          # "promise", "condition", "fulfillment"
    promise_elements: List[str] # ["seed", "land", "blessing"]
    arc_position: float         # Position in covenant narrative (0-1)
```

#### 6. `PropheticEmbedding`
```python
@dataclass
class PropheticEmbedding:
    verse_id: str
    vector: np.ndarray          # 384-dim
    prophecy_ids: List[str]     # Related prophecies
    prophetic_role: str         # "prophecy", "fulfillment", "neither"
    specificity_markers: List[str]  # ["person", "location", "time"]
    fulfillment_confidence: float  # If fulfillment, confidence level
```

---

## Part 3: Embedding Generation Pipelines

### File: `ml/embeddings/generators/`

#### Base Generator
```python
class EmbeddingGenerator:
    """Base class for embedding generation."""

    def __init__(self, model_id: str):
        self.model = SentenceTransformer(model_id)

    async def generate(self, verse_id: str, context: Dict) -> np.ndarray:
        raise NotImplementedError

    async def batch_generate(self, verses: List[str], contexts: List[Dict]) -> List[np.ndarray]:
        raise NotImplementedError
```

#### Patristic Embedding Generator
```python
class PatristicEmbeddingGenerator(EmbeddingGenerator):
    """
    Generate embeddings from aggregated patristic interpretation.
    """

    def __init__(self, model_id: str, patristic_db):
        super().__init__(model_id)
        self.patristic_db = patristic_db

    async def generate(self, verse_id: str, context: Dict = None) -> PatristicEmbedding:
        # Gather all patristic interpretations
        interpretations = await self.patristic_db.get_interpretations(verse_id)

        if not interpretations:
            return None  # No patristic data

        # Aggregate interpretation texts
        aggregated_text = self._aggregate_interpretations(interpretations)

        # Generate embedding
        vector = self.model.encode(aggregated_text)

        # Calculate consensus
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
            fourfold_senses=fourfold
        )

    def _aggregate_interpretations(self, interpretations: List) -> str:
        """Combine interpretations into coherent text for embedding."""
        # Weight by Father authority
        weighted_texts = []
        for interp in interpretations:
            weight = interp.father_authority_weight
            weighted_texts.append((weight, interp.interpretation))

        # Sort by weight, take top N
        weighted_texts.sort(key=lambda x: -x[0])
        top_interpretations = [t[1] for t in weighted_texts[:5]]

        return " ".join(top_interpretations)

    def _calculate_consensus(self, interpretations: List) -> float:
        """Measure agreement among Fathers."""
        if len(interpretations) < 2:
            return 1.0

        # Embed each interpretation
        vectors = [self.model.encode(i.interpretation) for i in interpretations]

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sim = cosine_similarity([vectors[i]], [vectors[j]])[0][0]
                similarities.append(sim)

        return np.mean(similarities)
```

#### Liturgical Embedding Generator
```python
class LiturgicalEmbeddingGenerator(EmbeddingGenerator):
    """
    Generate embeddings from liturgical context.
    """

    def __init__(self, model_id: str, liturgical_db):
        super().__init__(model_id)
        self.liturgical_db = liturgical_db

    async def generate(self, verse_id: str, context: Dict = None) -> LiturgicalEmbedding:
        # Gather liturgical usage data
        usages = await self.liturgical_db.get_liturgical_uses(verse_id)

        if not usages:
            return None

        # Build liturgical context text
        context_text = self._build_liturgical_context(usages)

        # Generate embedding
        vector = self.model.encode(context_text)

        return LiturgicalEmbedding(
            verse_id=verse_id,
            vector=vector,
            liturgical_uses=[u.service for u in usages],
            feast_associations=[u.feast for u in usages if u.feast],
            hymn_references=[u.hymn_id for u in usages if u.hymn_id],
            liturgical_weight=self._calculate_liturgical_weight(usages),
            tradition="Byzantine"  # Or detect from usages
        )

    def _build_liturgical_context(self, usages: List) -> str:
        """Build text representation of liturgical context."""
        elements = []
        for usage in usages:
            elements.append(f"Used in {usage.service}")
            if usage.feast:
                elements.append(f"for {usage.feast}")
            if usage.hymn_text:
                elements.append(f"sung as: {usage.hymn_text[:100]}")
        return ". ".join(elements)
```

#### Typological Embedding Generator
```python
class TypologicalEmbeddingGenerator(EmbeddingGenerator):
    """
    Generate embeddings from typological pattern context.
    """

    def __init__(self, model_id: str, typology_engine):
        super().__init__(model_id)
        self.typology_engine = typology_engine

    async def generate(self, verse_id: str, context: Dict = None) -> TypologicalEmbedding:
        # Get typological connections for this verse
        type_connections = await self.typology_engine.get_type_connections(verse_id)
        antitype_connections = await self.typology_engine.get_antitype_connections(verse_id)

        if not type_connections and not antitype_connections:
            return None

        # Determine role
        if type_connections and antitype_connections:
            role = "both"
        elif type_connections:
            role = "type"
        else:
            role = "antitype"

        # Build pattern context
        patterns = await self._gather_patterns(type_connections, antitype_connections)
        context_text = self._build_typological_context(patterns)

        # Generate embedding
        vector = self.model.encode(context_text)

        return TypologicalEmbedding(
            verse_id=verse_id,
            vector=vector,
            pattern_ids=[p.id for p in patterns],
            type_role=role,
            fractal_layers=list(set(l for p in patterns for l in p.layers)),
            pattern_strength=np.mean([p.strength for p in patterns]) if patterns else 0
        )
```

---

## Part 4: Vector Database Schema

### File: `db/vector_store.py`

**Multi-Collection Architecture**:

```python
class MultiVectorStore:
    """
    Multi-collection vector store with hybrid search.
    """

    # Collection names
    COLLECTIONS = {
        "semantic": "biblos_semantic",
        "patristic": "biblos_patristic",
        "liturgical": "biblos_liturgical",
        "typological": "biblos_typological",
        "covenantal": "biblos_covenantal",
        "prophetic": "biblos_prophetic"
    }

    def __init__(self, qdrant_client):
        self.client = qdrant_client

    async def create_collections(self) -> None:
        """Create all embedding collections with appropriate configs."""
        for name, collection in self.COLLECTIONS.items():
            await self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=384,  # MiniLM dimension
                    distance=Distance.COSINE
                ),
                # Enable payload indexing for filtering
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=10000
                )
            )

    async def upsert_embedding(
        self,
        embedding_type: str,
        verse_id: str,
        vector: np.ndarray,
        payload: Dict
    ) -> None:
        """Upsert embedding to appropriate collection."""
        collection = self.COLLECTIONS[embedding_type]
        await self.client.upsert(
            collection_name=collection,
            points=[PointStruct(
                id=self._verse_to_id(verse_id),
                vector=vector.tolist(),
                payload={"verse_id": verse_id, **payload}
            )]
        )

    async def search(
        self,
        embedding_type: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[Filter] = None
    ) -> List[Dict]:
        """Search single collection."""
        collection = self.COLLECTIONS[embedding_type]
        results = await self.client.search(
            collection_name=collection,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=filters
        )
        return [{"verse_id": r.payload["verse_id"], "score": r.score, **r.payload} for r in results]

    async def hybrid_search(
        self,
        query_vectors: Dict[str, np.ndarray],
        weights: Dict[str, float],
        top_k: int = 10,
        filters: Optional[Dict[str, Filter]] = None
    ) -> List[Dict]:
        """
        Hybrid search across multiple collections with weighted aggregation.
        """
        all_results = {}

        # Search each collection
        for embed_type, vector in query_vectors.items():
            if embed_type not in self.COLLECTIONS:
                continue

            weight = weights.get(embed_type, 0.0)
            if weight == 0:
                continue

            collection_filter = filters.get(embed_type) if filters else None
            results = await self.search(
                embedding_type=embed_type,
                query_vector=vector,
                top_k=top_k * 2,  # Get more for aggregation
                filters=collection_filter
            )

            for r in results:
                verse_id = r["verse_id"]
                if verse_id not in all_results:
                    all_results[verse_id] = {
                        "verse_id": verse_id,
                        "scores": {},
                        "weighted_total": 0.0
                    }
                all_results[verse_id]["scores"][embed_type] = r["score"]
                all_results[verse_id]["weighted_total"] += r["score"] * weight

        # Sort by weighted total and return top_k
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: -x["weighted_total"]
        )[:top_k]

        return sorted_results
```

---

## Part 5: Query Strategies

### File: `ml/embeddings/query_strategies.py`

**Strategy Pattern for Different Use Cases**:

```python
class QueryStrategy:
    """Base query strategy."""

    def get_weights(self) -> Dict[str, float]:
        raise NotImplementedError

    def get_filters(self, context: Dict) -> Dict[str, Filter]:
        return {}
```

#### Theological Query Strategy
```python
class TheologicalQueryStrategy(QueryStrategy):
    """
    For theological research queries.
    Emphasizes patristic and typological dimensions.
    """

    def get_weights(self) -> Dict[str, float]:
        return {
            "semantic": 0.3,
            "patristic": 0.35,
            "typological": 0.25,
            "covenantal": 0.1
        }

    def get_filters(self, context: Dict) -> Dict[str, Filter]:
        filters = {}
        if "tradition" in context:
            filters["patristic"] = Filter(
                must=[FieldCondition(key="traditions", match=MatchAny(any=context["tradition"]))]
            )
        return filters
```

#### Liturgical Query Strategy
```python
class LiturgicalQueryStrategy(QueryStrategy):
    """
    For finding liturgically related verses.
    Emphasizes liturgical embedding heavily.
    """

    def get_weights(self) -> Dict[str, float]:
        return {
            "semantic": 0.2,
            "liturgical": 0.6,
            "patristic": 0.2
        }

    def get_filters(self, context: Dict) -> Dict[str, Filter]:
        filters = {}
        if "feast" in context:
            filters["liturgical"] = Filter(
                must=[FieldCondition(key="feast_associations", match=MatchAny(any=[context["feast"]]))]
            )
        return filters
```

#### Typological Query Strategy
```python
class TypologicalQueryStrategy(QueryStrategy):
    """
    For discovering typological connections.
    Emphasizes typological patterns.
    """

    def get_weights(self) -> Dict[str, float]:
        return {
            "semantic": 0.2,
            "typological": 0.5,
            "covenantal": 0.2,
            "prophetic": 0.1
        }

    def get_filters(self, context: Dict) -> Dict[str, Filter]:
        filters = {}
        if "pattern_id" in context:
            filters["typological"] = Filter(
                must=[FieldCondition(key="pattern_ids", match=MatchAny(any=[context["pattern_id"]]))]
            )
        if "layer" in context:
            filters["typological"] = Filter(
                must=[FieldCondition(key="fractal_layers", match=MatchAny(any=[context["layer"]]))]
            )
        return filters
```

#### Prophetic Query Strategy
```python
class PropheticQueryStrategy(QueryStrategy):
    """
    For analyzing prophetic connections.
    Emphasizes prophetic embeddings.
    """

    def get_weights(self) -> Dict[str, float]:
        return {
            "semantic": 0.2,
            "prophetic": 0.5,
            "typological": 0.2,
            "covenantal": 0.1
        }

    def get_filters(self, context: Dict) -> Dict[str, Filter]:
        filters = {}
        if "role" in context:
            filters["prophetic"] = Filter(
                must=[FieldCondition(key="prophetic_role", match=MatchValue(value=context["role"]))]
            )
        return filters
```

#### Cross-Reference Discovery Strategy
```python
class CrossRefDiscoveryStrategy(QueryStrategy):
    """
    Balanced strategy for cross-reference discovery.
    Uses all embedding types with balanced weights.
    """

    def get_weights(self) -> Dict[str, float]:
        return {
            "semantic": 0.25,
            "patristic": 0.20,
            "liturgical": 0.10,
            "typological": 0.20,
            "covenantal": 0.15,
            "prophetic": 0.10
        }
```

---

## Part 6: Integration with Oracle Engines

### Integration Points

**For Omni-Contextual Resolver (Session 03)**:
```python
async def get_occurrence_contexts(self, lemma: str) -> List[Dict]:
    """
    Get all occurrences of a lemma with rich embedding context.
    """
    # Find all verses containing this lemma
    verses = await self.corpus.get_verses_with_lemma(lemma)

    # Retrieve all embedding types for these verses
    contexts = []
    for verse_id in verses:
        semantic = await self.vector_store.get_embedding("semantic", verse_id)
        patristic = await self.vector_store.get_embedding("patristic", verse_id)

        contexts.append({
            "verse_id": verse_id,
            "semantic_context": semantic,
            "patristic_interpretation": patristic.interpretation_summary if patristic else None,
            "consensus": patristic.consensus_level if patristic else 0
        })

    return contexts
```

**For Fractal Typology Engine (Session 06)**:
```python
async def find_pattern_matches(self, pattern_id: str, source_verse: str) -> List[Dict]:
    """
    Find verses matching a typological pattern.
    """
    # Get typological embedding for source
    source_embed = await self.vector_store.get_embedding("typological", source_verse)

    if not source_embed:
        return []

    # Search with typological strategy
    strategy = TypologicalQueryStrategy()
    results = await self.vector_store.hybrid_search(
        query_vectors={"typological": source_embed.vector},
        weights=strategy.get_weights(),
        filters=strategy.get_filters({"pattern_id": pattern_id}),
        top_k=20
    )

    return results
```

**For Prophetic Prover (Session 07)**:
```python
async def find_fulfillment_candidates(self, prophecy_verse: str) -> List[Dict]:
    """
    Find potential fulfillment verses for a prophecy.
    """
    # Get prophetic embedding for prophecy
    prophecy_embed = await self.vector_store.get_embedding("prophetic", prophecy_verse)

    if not prophecy_embed:
        return []

    # Search for fulfillments
    strategy = PropheticQueryStrategy()
    results = await self.vector_store.hybrid_search(
        query_vectors={"prophetic": prophecy_embed.vector, "semantic": prophecy_embed.vector},
        weights=strategy.get_weights(),
        filters=strategy.get_filters({"role": "fulfillment"}),
        top_k=20
    )

    return results
```

---

## Part 7: Event-Driven Updates

### Integration with Session 08 Event Sourcing

**Vector Store Event Projection**:
```python
class VectorStoreProjection(ProjectionBase):
    """
    Updates vector store based on events.
    """

    def __init__(self, event_store, vector_store, generators):
        super().__init__(event_store)
        self.vector_store = vector_store
        self.generators = generators

    async def _handle_PatristicWitnessAddedEvent(self, event):
        """Regenerate patristic embedding when new interpretation added."""
        generator = self.generators["patristic"]
        embedding = await generator.generate(event.target_id)

        if embedding:
            await self.vector_store.upsert_embedding(
                embedding_type="patristic",
                verse_id=event.target_id,
                vector=embedding.vector,
                payload={
                    "interpretation_sources": embedding.interpretation_sources,
                    "consensus_level": embedding.consensus_level,
                    "traditions": embedding.traditions,
                    "fourfold_senses": embedding.fourfold_senses
                }
            )

    async def _handle_TypologicalConnectionIdentifiedEvent(self, event):
        """Update typological embeddings for both type and antitype."""
        generator = self.generators["typological"]

        for verse_id in [event.type_ref, event.antitype_ref]:
            embedding = await generator.generate(verse_id)

            if embedding:
                await self.vector_store.upsert_embedding(
                    embedding_type="typological",
                    verse_id=verse_id,
                    vector=embedding.vector,
                    payload={
                        "pattern_ids": embedding.pattern_ids,
                        "type_role": embedding.type_role,
                        "fractal_layers": embedding.fractal_layers,
                        "pattern_strength": embedding.pattern_strength
                    }
                )
```

---

## Part 8: Testing Specification

### Unit Tests: `tests/ml/embeddings/test_multi_vector.py`

**Test 1: `test_semantic_embedding_generation`**
- Generate semantic embedding for verse
- Verify dimension correct (384)
- Verify reproducibility

**Test 2: `test_patristic_embedding_generation`**
- Add patristic interpretations
- Generate patristic embedding
- Verify consensus calculation

**Test 3: `test_liturgical_embedding_generation`**
- Add liturgical uses
- Generate liturgical embedding
- Verify feast associations captured

**Test 4: `test_typological_embedding_generation`**
- Create typological connections
- Generate typological embedding
- Verify pattern IDs included

**Test 5: `test_hybrid_search_weighted`**
- Index multiple embedding types
- Perform hybrid search
- Verify weights applied correctly

**Test 6: `test_query_strategy_theological`**
- Use theological query strategy
- Verify patristic weight is high
- Verify results match expected

**Test 7: `test_query_strategy_liturgical`**
- Use liturgical query strategy
- Filter by feast
- Verify liturgical verses returned

**Test 8: `test_event_driven_update`**
- Add patristic witness via event
- Verify embedding regenerated
- Verify vector store updated

---

## Part 9: Configuration

### Add to `config.py`

**New Configuration Dataclass**: `MultiVectorConfig`

Fields:
- `qdrant_host: str = "localhost"`
- `qdrant_port: int = 6333`
- `embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"`
- `embedding_dimension: int = 384`
- `batch_size: int = 100`
- `enable_patristic_embeddings: bool = True`
- `enable_liturgical_embeddings: bool = True`
- `enable_typological_embeddings: bool = True`
- `enable_covenantal_embeddings: bool = True`
- `enable_prophetic_embeddings: bool = True`
- `default_strategy: str = "cross_ref_discovery"`
- `cache_embeddings: bool = True`
- `cache_ttl_hours: int = 168`  # 1 week

---

## Part 10: Data Population

### Population Script: `scripts/populate_multi_vectors.py`

```python
async def populate_all_embeddings():
    """
    Populate all embedding collections from existing data.
    """
    verses = await corpus.get_all_verses()
    total = len(verses)

    # Initialize generators
    generators = {
        "semantic": SemanticEmbeddingGenerator(config.embedding_model),
        "patristic": PatristicEmbeddingGenerator(config.embedding_model, patristic_db),
        "liturgical": LiturgicalEmbeddingGenerator(config.embedding_model, liturgical_db),
        "typological": TypologicalEmbeddingGenerator(config.embedding_model, typology_engine),
        "covenantal": CovenantEmbeddingGenerator(config.embedding_model, covenant_db),
        "prophetic": PropheticEmbeddingGenerator(config.embedding_model, prophecy_db)
    }

    for i, verse in enumerate(verses):
        if i % 100 == 0:
            logger.info(f"Processing {i}/{total}")

        for embed_type, generator in generators.items():
            try:
                embedding = await generator.generate(verse.id)
                if embedding:
                    await vector_store.upsert_embedding(
                        embedding_type=embed_type,
                        verse_id=verse.id,
                        vector=embedding.vector,
                        payload=embedding.to_payload()
                    )
            except Exception as e:
                logger.error(f"Failed to generate {embed_type} for {verse.id}: {e}")

    logger.info("Population complete")
```

---

## Part 11: Success Criteria

### Functional Requirements
- [ ] All six embedding types generating correctly
- [ ] Multi-collection vector store operational
- [ ] Hybrid search working with weighted aggregation
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

## Part 12: Detailed Implementation Order

1. **Create `ml/embeddings/multi_vector.py`** with dataclasses
2. **Create `ml/embeddings/generators/`** directory
3. **Implement `SemanticEmbeddingGenerator`** (enhance existing)
4. **Implement `PatristicEmbeddingGenerator`**
5. **Implement `LiturgicalEmbeddingGenerator`**
6. **Implement `TypologicalEmbeddingGenerator`**
7. **Implement `CovenantEmbeddingGenerator`**
8. **Implement `PropheticEmbeddingGenerator`**
9. **Create `db/vector_store.py`** with `MultiVectorStore`
10. **Create `ml/embeddings/query_strategies.py`**
11. **Implement event projection** (integrate with Session 08)
12. **Create integration points** with oracle engines
13. **Add configuration to `config.py`**
14. **Write population script**
15. **Write unit tests**
16. **Run population for all verses**
17. **Benchmark hybrid search performance**

---

## Part 13: Dependencies on Other Sessions

### Depends On
- SESSION 01: Mutual Transformation (for embedding comparison logic)
- SESSION 06: Fractal Typology (for pattern embeddings)
- SESSION 08: Event Sourcing (for projection integration)

### Depended On By
- SESSION 11: Pipeline Integration (uses multi-vector search)

### External Dependencies
- Qdrant vector database
- sentence-transformers library
- Patristic interpretation database
- Liturgical usage database

---

## Session Completion Checklist

```markdown
- [ ] `ml/embeddings/multi_vector.py` with all embedding types
- [ ] All six embedding generators implemented
- [ ] `db/vector_store.py` with MultiVectorStore
- [ ] All query strategies implemented
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

# BIBLOS v2 Database Layer Optimization Report

**Date**: 2026-01-15
**Analyst**: Database Optimization Expert
**Scope**: `db/models.py`, `db/postgres.py`, `db/neo4j_client.py`, `db/qdrant_client.py`, `db/connection_pool.py`, `alembic/`

---

## Executive Summary

The BIBLOS v2 database layer has a solid foundation but contains several optimization opportunities that could significantly improve performance. Key findings include:

1. **Missing Indexes**: 8 critical indexes missing on frequently queried columns
2. **N+1 Query Patterns**: 5 locations with potential N+1 query issues
3. **Connection Pool**: Suboptimal pool sizing for async workloads
4. **Query Batching**: 3 methods using inefficient single-record operations
5. **Caching**: No application-level caching for frequently accessed data
6. **Neo4j Queries**: Cypher queries missing LIMIT clauses and index hints

---

## 1. SQLAlchemy Models (`db/models.py`)

### 1.1 Missing Indexes

#### High Priority

| Table | Column(s) | Query Pattern | Impact |
|-------|-----------|---------------|--------|
| `cross_references` | `(confidence, connection_type)` | Filter by confidence threshold | HIGH |
| `patristic_citations` | `(verse_id, father_name)` | Lookup citations by verse+father | HIGH |
| `extraction_results` | `(status)` | Filter pending/failed extractions | MEDIUM |
| `extraction_results` | `(agent_name, status)` | Agent status monitoring | MEDIUM |
| `verses` | `(book_id, chapter, verse_num)` | Sequential verse lookup | HIGH |
| `cross_references` | `(strength)` | Filter strong connections | MEDIUM |
| `patristic_citations` | `(century)` | Filter by patristic era | LOW |
| `patristic_citations` | `(tradition)` | Filter by Eastern/Western | LOW |

#### Recommended Index Additions

```python
# Add to Verse.__table_args__
Index("ix_verses_book_chapter_verse", "book_id", "chapter", "verse_num"),

# Add to CrossReference.__table_args__
Index("ix_crossref_confidence_type", "confidence", "connection_type"),
Index("ix_crossref_strength", "strength"),

# Add to PatristicCitation model
__table_args__ = (
    Index("ix_patristic_verse_father", "verse_id", "father_name"),
    Index("ix_patristic_century", "century"),
    Index("ix_patristic_tradition", "tradition"),
)

# Add to ExtractionResult.__table_args__
Index("ix_extraction_status", "status"),
Index("ix_extraction_agent_status", "agent_name", "status"),
```

### 1.2 Relationship Loading Strategy Issues

**Current Issue**: Lines 114-124 use default lazy loading which causes N+1 queries.

```python
# CURRENT (PROBLEMATIC)
source_refs: Mapped[List["CrossReference"]] = relationship(
    back_populates="source_verse",
    foreign_keys="CrossReference.source_id"
)
```

**Recommendation**: Add `lazy="selectin"` for predictable batch loading:

```python
# OPTIMIZED
source_refs: Mapped[List["CrossReference"]] = relationship(
    back_populates="source_verse",
    foreign_keys="CrossReference.source_id",
    lazy="selectin"  # Batch load in 1 query
)
```

### 1.3 Embedding Column Storage

**Current Issue**: Lines 98-100 store embeddings as `ARRAY(Float)` which is inefficient for vector operations.

```python
# CURRENT
embedding: Mapped[Optional[List[float]]] = mapped_column(ARRAY(Float), nullable=True)
```

**Recommendation**: Use pgvector's `VECTOR` type for native vector operations:

```python
from pgvector.sqlalchemy import Vector

# OPTIMIZED
embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(768), nullable=True)
embedding_greek: Mapped[Optional[List[float]]] = mapped_column(Vector(768), nullable=True)
embedding_hebrew: Mapped[Optional[List[float]]] = mapped_column(Vector(768), nullable=True)

# Add HNSW index for fast similarity search
__table_args__ = (
    Index(
        "ix_verses_embedding_hnsw",
        "embedding",
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"}
    ),
)
```

### 1.4 JSONB Column Indexing

**Current Issue**: JSONB columns (`morphology`, `syntax`, `semantics`) lack GIN indexes for JSON path queries.

**Recommendation**: Add GIN indexes for frequently queried JSON paths:

```python
# Add to Verse.__table_args__
Index("ix_verses_morphology_gin", "morphology", postgresql_using="gin"),
Index("ix_verses_syntax_gin", "syntax", postgresql_using="gin"),
```

---

## 2. PostgreSQL Client (`db/postgres.py`)

### 2.1 N+1 Query Pattern: `batch_upsert_verses`

**Location**: Lines 175-199

**Current Issue**: Individual SELECT + INSERT/UPDATE per verse in a loop:

```python
for verse_data in verses:
    existing = await session.execute(
        select(Verse).where(Verse.reference == verse_data["reference"])
    )
    # ... individual operations
```

**Recommendation**: Use PostgreSQL's `ON CONFLICT DO UPDATE` (upsert):

```python
async def batch_upsert_verses(self, verses: List[Dict[str, Any]]) -> int:
    """Optimized batch upsert using PostgreSQL ON CONFLICT."""
    from sqlalchemy.dialects.postgresql import insert

    async with self.session() as session:
        # Batch insert with conflict handling
        stmt = insert(Verse).values(verses)
        stmt = stmt.on_conflict_do_update(
            index_elements=['reference'],
            set_={
                col.name: stmt.excluded[col.name]
                for col in Verse.__table__.columns
                if col.name not in ('id', 'reference', 'created_at')
            }
        )
        await session.execute(stmt)
        return len(verses)
```

### 2.2 N+1 Query Pattern: `add_crossref`

**Location**: Lines 225-259

**Current Issue**: Two separate queries to fetch source and target verses.

```python
source = await session.execute(
    select(Verse).where(Verse.reference == source_ref)
)
target = await session.execute(
    select(Verse).where(Verse.reference == target_ref)
)
```

**Recommendation**: Combine into single query:

```python
async def add_crossref(self, source_ref: str, target_ref: str, ...):
    async with self.session() as session:
        # Single query for both verses
        result = await session.execute(
            select(Verse)
            .where(Verse.reference.in_([source_ref, target_ref]))
        )
        verses = {v.reference: v for v in result.scalars().all()}

        source = verses.get(source_ref)
        target = verses.get(target_ref)

        if not source or not target:
            return None

        # ... rest of method
```

### 2.3 Missing Batch Operations

**Issue**: No batch methods for cross-references or extraction results.

**Recommendation**: Add batch insert methods:

```python
async def batch_insert_crossrefs(
    self,
    crossrefs: List[Dict[str, Any]],
    chunk_size: int = 1000
) -> int:
    """Batch insert cross-references efficiently."""
    from sqlalchemy.dialects.postgresql import insert

    count = 0
    async with self.session() as session:
        for chunk in chunked(crossrefs, chunk_size):
            stmt = insert(CrossReference).values(chunk)
            stmt = stmt.on_conflict_do_nothing(
                constraint="uq_crossref_pair"
            )
            result = await session.execute(stmt)
            count += result.rowcount
    return count

async def batch_save_extractions(
    self,
    results: List[Dict[str, Any]],
    chunk_size: int = 500
) -> int:
    """Batch save extraction results."""
    async with self.session() as session:
        for chunk in chunked(results, chunk_size):
            session.add_all([
                ExtractionResult(**r) for r in chunk
            ])
        await session.flush()
    return len(results)
```

### 2.4 Connection Pool Configuration

**Location**: Lines 37-43

**Current Issue**: Default pool settings may be suboptimal.

```python
pool_size: int = 10,
max_overflow: int = 20,
```

**Recommendation**: Add connection pool optimization:

```python
self._engine = create_async_engine(
    self.database_url,
    pool_size=self.pool_size,
    max_overflow=self.max_overflow,
    pool_pre_ping=True,
    # ADD THESE OPTIMIZATIONS:
    pool_recycle=3600,      # Recycle connections after 1 hour
    pool_timeout=30,        # Wait 30s for connection
    connect_args={
        "command_timeout": 60,
        "server_settings": {
            "jit": "off",                    # Disable JIT for short queries
            "statement_timeout": "60000",     # 60s statement timeout
            "work_mem": "128MB",              # Memory per query
            "effective_cache_size": "4GB"     # Cache estimate
        }
    }
)
```

### 2.5 Missing Caching Layer

**Issue**: Frequently accessed data (books, verse lookups) hit database every time.

**Recommendation**: Add Redis caching integration:

```python
from functools import lru_cache
import json

class PostgresClient:
    def __init__(self, ..., redis_client=None):
        self._redis = redis_client
        self._cache_ttl = 3600  # 1 hour

    async def get_book(self, code: str) -> Optional[Book]:
        """Get book with caching."""
        cache_key = f"book:{code}"

        # Check cache first
        if self._redis:
            cached = await self._redis.get(cache_key)
            if cached:
                return Book(**json.loads(cached))

        # Query database
        async with self.session() as session:
            result = await session.execute(
                select(Book).where(Book.code == code)
            )
            book = result.scalar_one_or_none()

            # Cache result
            if book and self._redis:
                await self._redis.setex(
                    cache_key,
                    self._cache_ttl,
                    json.dumps(book_to_dict(book))
                )

            return book

    async def invalidate_cache(self, pattern: str = "*"):
        """Invalidate cache entries."""
        if self._redis:
            async for key in self._redis.scan_iter(pattern):
                await self._redis.delete(key)
```

### 2.6 Vector Search Optimization

**Location**: Lines 293-320

**Current Issue**: Raw SQL with string interpolation, no query plan caching.

**Recommendation**: Use prepared statements and proper vector type:

```python
async def find_similar_verses(
    self,
    embedding: List[float],
    limit: int = 10,
    threshold: float = 0.7,
    book_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Optimized vector similarity search."""
    async with self.session() as session:
        # Use SQLAlchemy Core for prepared statement caching
        from sqlalchemy import literal_column

        base_query = (
            select(
                Verse.reference,
                Verse.text_english,
                literal_column(
                    f"1 - (embedding <=> '{embedding}'::vector)"
                ).label("similarity")
            )
            .where(Verse.embedding.isnot(None))
            .where(literal_column("similarity") > threshold)
            .order_by(literal_column("similarity").desc())
            .limit(limit)
        )

        if book_filter:
            base_query = base_query.join(Book).where(Book.code == book_filter)

        result = await session.execute(base_query)
        return [
            {"reference": r[0], "text": r[1], "similarity": r[2]}
            for r in result.fetchall()
        ]
```

---

## 3. Neo4j Client (`db/neo4j_client.py`)

### 3.1 Missing Index Usage in Queries

**Location**: Lines 150-157

**Current Issue**: MERGE without explicit index hint.

```cypher
MERGE (v:Verse {reference: $reference})
SET v += $props
RETURN elementId(v) as id
```

**Recommendation**: Add index hint for better query planning:

```cypher
MERGE (v:Verse {reference: $reference})
USING INDEX v:Verse(reference)
SET v += $props
RETURN elementId(v) as id
```

### 3.2 Unbounded Path Queries

**Location**: Lines 386-399 (`get_verse_neighborhood`)

**Current Issue**: Path query without result limit can return massive subgraphs.

```cypher
MATCH path = (center:Verse {reference: $ref})-[*1..$depth]-(connected)
```

**Recommendation**: Add LIMIT and explicit node filtering:

```cypher
MATCH path = (center:Verse {reference: $ref})-[*1..$depth]-(connected)
WHERE connected:Verse OR connected:ThematicCategory OR connected:ChurchFather
WITH path, connected
LIMIT 1000
WITH collect(DISTINCT connected) as nodes,
     collect(DISTINCT relationships(path)) as rels
RETURN nodes[..100] as nodes, rels[..200] as rels
```

### 3.3 Statistics Query Inefficiency

**Location**: Lines 402-424

**Current Issue**: Multiple MATCH clauses count entire database.

```cypher
MATCH (v:Verse) WITH count(v) as verses
MATCH (f:ChurchFather) WITH verses, count(f) as fathers
...
```

**Recommendation**: Use APOC or parallel count:

```cypher
CALL {
    MATCH (v:Verse) RETURN count(v) as verses
}
CALL {
    MATCH (f:ChurchFather) RETURN count(f) as fathers
}
CALL {
    MATCH (t:ThematicCategory) RETURN count(t) as themes
}
CALL {
    MATCH ()-[r]->() RETURN count(r) as relationships
}
RETURN verses, fathers, themes, relationships
```

### 3.4 Missing Connection Pooling

**Current Issue**: Single driver instance without explicit connection pooling.

**Recommendation**: Configure driver with connection pool:

```python
self._driver = AsyncGraphDatabase.driver(
    self.uri,
    auth=(self.user, self.password),
    # ADD CONNECTION POOL CONFIG
    max_connection_pool_size=50,
    max_transaction_retry_time=30,
    connection_acquisition_timeout=30,
    encrypted=False  # For local development
)
```

### 3.5 Batch Node Creation

**Current Issue**: `create_verse_node` creates one node per call.

**Recommendation**: Add batch creation method:

```python
async def batch_create_verse_nodes(
    self,
    verses: List[Dict[str, Any]],
    batch_size: int = 1000
) -> int:
    """Batch create verse nodes using UNWIND."""
    if not self._driver:
        return 0

    count = 0
    async with self._driver.session() as session:
        for batch in chunked(verses, batch_size):
            result = await session.run("""
                UNWIND $verses as verse
                MERGE (v:Verse {reference: verse.reference})
                SET v += verse.properties
                RETURN count(v) as created
            """, verses=batch)
            record = await result.single()
            count += record["created"]

    return count
```

---

## 4. Qdrant Client (`db/qdrant_client.py`)

### 4.1 Search Without Filtering Optimization

**Location**: Lines 203-257

**Current Issue**: Filter construction happens for every search request.

**Recommendation**: Pre-build common filters:

```python
class QdrantVectorStore:
    # Pre-built filter templates
    _testament_filters = {
        "OT": qmodels.Filter(must=[
            qmodels.FieldCondition(
                key="testament",
                match=qmodels.MatchValue(value="OT")
            )
        ]),
        "NT": qmodels.Filter(must=[
            qmodels.FieldCondition(
                key="testament",
                match=qmodels.MatchValue(value="NT")
            )
        ])
    }

    async def search_similar(
        self,
        query_embedding: List[float],
        collection: str = "verses",
        limit: int = 10,
        testament_filter: Optional[str] = None,  # "OT" or "NT"
        **kwargs
    ) -> List[SearchResult]:
        # Use pre-built filter
        query_filter = None
        if testament_filter:
            query_filter = self._testament_filters.get(testament_filter)

        # ... rest of method
```

### 4.2 Missing HNSW Index Configuration

**Current Issue**: Collections created with default HNSW parameters.

**Recommendation**: Optimize HNSW for biblical corpus size:

```python
async def create_collections(self) -> None:
    """Create collections with optimized HNSW parameters."""
    for name, config in self.COLLECTIONS.items():
        await self._client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=config["dimension"],
                distance=getattr(qmodels.Distance, config["distance"].upper())
            ),
            # ADD HNSW OPTIMIZATION
            hnsw_config=qmodels.HnswConfigDiff(
                m=16,                    # Connections per node
                ef_construct=200,        # Build-time search width
                full_scan_threshold=10000  # Use brute force below this
            ),
            # ADD QUANTIZATION for memory efficiency
            quantization_config=qmodels.ScalarQuantization(
                scalar=qmodels.ScalarQuantizationConfig(
                    type=qmodels.ScalarType.INT8,
                    always_ram=True
                )
            )
        )
```

### 4.3 Parallel Batch Upsert

**Location**: Lines 156-201

**Current Issue**: Sequential batch processing.

**Recommendation**: Use parallel batch uploads:

```python
async def batch_upsert_parallel(
    self,
    points: List[Tuple[str, List[float], Dict[str, Any]]],
    collection: str = "verses",
    batch_size: int = 100,
    max_parallel: int = 4
) -> int:
    """Parallel batch upsert for faster ingestion."""
    import asyncio

    batches = list(chunked(points, batch_size))
    semaphore = asyncio.Semaphore(max_parallel)
    count = 0

    async def upload_batch(batch):
        nonlocal count
        async with semaphore:
            point_structs = [
                qmodels.PointStruct(
                    id=self._ref_to_uuid(ref),
                    vector=embedding,
                    payload={"reference": ref, **payload}
                )
                for ref, embedding, payload in batch
            ]
            await self._client.upsert(
                collection_name=collection,
                points=point_structs,
                wait=False  # Don't wait for each batch
            )
            count += len(batch)

    await asyncio.gather(*[upload_batch(b) for b in batches])
    return count
```

---

## 5. Connection Pool Manager (`db/connection_pool.py`)

### 5.1 Pool Sizing Recommendations

**Current Configuration** (Lines 66-92):
- PostgreSQL: 20 connections
- Neo4j: 10 connections
- Qdrant: 10 connections
- Redis: 20 connections

**Analysis**: For async workloads with the 24-agent SDES pipeline:

| Database | Current | Recommended | Rationale |
|----------|---------|-------------|-----------|
| PostgreSQL | 20 | 50 | 24 agents + API + overhead |
| Neo4j | 10 | 30 | Graph traversals are connection-bound |
| Qdrant | 10 | 20 | Vector search is CPU-bound |
| Redis | 20 | 50 | High cache hit rate needed |

**Recommended Configuration**:

```python
def _default_config(self) -> Dict[str, ConnectionConfig]:
    return {
        "postgres": ConnectionConfig(
            name="postgres",
            connection_string=os.getenv("DATABASE_URL", ...),
            max_connections=int(os.getenv("PG_POOL_SIZE", "50")),
            timeout_seconds=30,
            retry_attempts=3,
            health_check_interval=30
        ),
        "neo4j": ConnectionConfig(
            name="neo4j",
            connection_string=os.getenv("NEO4J_URI", ...),
            max_connections=int(os.getenv("NEO4J_POOL_SIZE", "30")),
            timeout_seconds=30,
            retry_attempts=3
        ),
        # ... etc
    }
```

### 5.2 Health Check Optimization

**Current Issue**: Health check runs SELECT 1 which doesn't test real workload.

**Recommendation**: Use representative query:

```python
async def _check_health(self, name: str, client: Any) -> bool:
    try:
        if name == "postgres":
            async with client.session() as session:
                # Test actual table access
                result = await session.execute(
                    text("SELECT COUNT(*) FROM books LIMIT 1")
                )
                return result.scalar() is not None
        # ... etc
```

### 5.3 Circuit Breaker Pattern

**Current Issue**: No circuit breaker - failed connections keep retrying.

**Recommendation**: Add circuit breaker:

```python
from dataclasses import dataclass, field
from enum import Enum
import time

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: int = 30
    _failures: int = 0
    _state: CircuitState = CircuitState.CLOSED
    _last_failure_time: float = 0

    def record_failure(self):
        self._failures += 1
        self._last_failure_time = time.time()
        if self._failures >= self.failure_threshold:
            self._state = CircuitState.OPEN

    def record_success(self):
        self._failures = 0
        self._state = CircuitState.CLOSED

    def can_execute(self) -> bool:
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time > self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN allows one request
```

---

## 6. Alembic Migration Patterns

### 6.1 Missing pgvector Index in Migration

**Location**: `alembic/versions/001_initial_schema.py`

**Current Issue**: No HNSW index for vector search.

**Recommendation**: Add migration for vector index:

```python
def upgrade() -> None:
    # After CREATE EXTENSION vector

    # Create HNSW index for fast similarity search
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_verses_embedding_hnsw
        ON verses USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
```

### 6.2 Zero-Downtime Migration Pattern

**Recommendation**: Use `CONCURRENTLY` for index creation:

```python
def upgrade() -> None:
    # Create indexes concurrently to avoid locking
    op.execute("""
        CREATE INDEX CONCURRENTLY ix_crossref_confidence
        ON cross_references (confidence DESC)
    """)
```

---

## 7. Query Optimization Examples

### 7.1 Cross-Reference Lookup Optimization

**Before**:
```python
# Two queries
async def get_crossrefs_for_verse(self, verse_ref: str):
    verse = await session.execute(select(Verse).where(Verse.reference == verse_ref))
    verse = verse.scalar_one_or_none()
    result = await session.execute(
        select(CrossReference)
        .where((CrossReference.source_id == verse.id) | (CrossReference.target_id == verse.id))
    )
```

**After**:
```python
# Single optimized query with CTEs
async def get_crossrefs_for_verse(self, verse_ref: str):
    query = text("""
        WITH verse_id AS (
            SELECT id FROM verses WHERE reference = :ref
        )
        SELECT cr.*,
               sv.reference as source_ref,
               tv.reference as target_ref
        FROM cross_references cr
        JOIN verse_id vi ON cr.source_id = vi.id OR cr.target_id = vi.id
        JOIN verses sv ON cr.source_id = sv.id
        JOIN verses tv ON cr.target_id = tv.id
        ORDER BY cr.confidence DESC
        LIMIT 100
    """)
    return await session.execute(query, {"ref": verse_ref})
```

### 7.2 Patristic Citation Aggregation

**Optimized Query**:
```sql
-- Get all patristic citations for a book with father summary
SELECT
    v.reference,
    v.text_english,
    jsonb_agg(
        jsonb_build_object(
            'father', pc.father_name,
            'work', pc.work_title,
            'century', pc.century
        ) ORDER BY pc.century
    ) as citations,
    count(DISTINCT pc.father_name) as father_count
FROM verses v
JOIN books b ON v.book_id = b.id
LEFT JOIN patristic_citations pc ON pc.verse_id = v.id
WHERE b.code = 'GEN'
GROUP BY v.id, v.reference, v.text_english
HAVING count(pc.id) > 0
ORDER BY v.chapter, v.verse_num;
```

### 7.3 Neo4j Cross-Reference Path Analysis

**Optimized Cypher**:
```cypher
// Find typological connections with patristic support
MATCH (source:Verse {reference: $sourceRef})
MATCH path = (source)-[:TYPOLOGICALLY_FULFILLS|PROPHETICALLY_FULFILLS*1..3]->(target:Verse)
WHERE target.testament = 'NT'
OPTIONAL MATCH (f:ChurchFather)-[:CITED_BY]->(source)
OPTIONAL MATCH (f2:ChurchFather)-[:CITED_BY]->(target)
WITH path, target,
     collect(DISTINCT f.name) as source_fathers,
     collect(DISTINCT f2.name) as target_fathers
RETURN target.reference as targetRef,
       length(path) as distance,
       [r IN relationships(path) | type(r)] as connectionTypes,
       source_fathers,
       target_fathers,
       size(source_fathers) + size(target_fathers) as patristic_support
ORDER BY patristic_support DESC, distance
LIMIT 20
```

---

## 8. Implementation Priority

### Phase 1: Critical (Week 1)
1. Add missing composite indexes (estimated 60% query speedup)
2. Fix N+1 patterns in `batch_upsert_verses`
3. Implement Redis caching for book/verse lookups
4. Add pgvector HNSW index

### Phase 2: Important (Week 2)
1. Implement batch insert methods for cross-references
2. Optimize Neo4j connection pooling
3. Add circuit breaker to connection manager
4. Configure Qdrant HNSW parameters

### Phase 3: Enhancement (Week 3)
1. Implement query result caching
2. Add prepared statement caching
3. Optimize health check queries
4. Add metrics for connection pool utilization

---

## 9. Monitoring Recommendations

### Key Metrics to Track

```python
# PostgreSQL
- pg_stat_statements.calls (query frequency)
- pg_stat_statements.mean_time (avg query time)
- pg_stat_user_indexes.idx_scan (index usage)
- pg_stat_user_tables.seq_scan (table scans to avoid)

# Neo4j
- dbms.database.index.query.time (index query time)
- db.cypher.queries (query count)
- pool.used_connections (connection utilization)

# Qdrant
- collection.vectors_count (vector count)
- search.latency_ms (search latency)
- indexing.progress (indexing status)
```

---

## 10. Files Modified Summary

| File | Changes | LOC Changed |
|------|---------|-------------|
| `db/models.py` | Add 8 indexes, update relationship loading | ~50 |
| `db/postgres.py` | Fix N+1, add batch methods, caching | ~200 |
| `db/neo4j_client.py` | Optimize queries, add batching | ~100 |
| `db/qdrant_client.py` | HNSW config, parallel upload | ~80 |
| `db/connection_pool.py` | Pool sizing, circuit breaker | ~100 |
| `alembic/versions/002_*.py` | New migration for indexes | ~50 |

**Total Estimated Changes**: ~580 lines of code

---

*Report generated by Database Optimization Expert*

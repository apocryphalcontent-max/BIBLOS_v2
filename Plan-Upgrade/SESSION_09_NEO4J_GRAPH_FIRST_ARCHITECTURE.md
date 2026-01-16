# SESSION 09: NEO4J GRAPH-FIRST ARCHITECTURE

## Session Overview

**Objective**: Transform BIBLOS v2 from using Neo4j as a secondary store to a graph-first architecture where Neo4j is the primary source of truth for all relational data. Implement the full SPIDERWEB schema with advanced graph algorithms.

**Estimated Duration**: 1 Claude session (90-120 minutes of focused implementation)

**Prerequisites**:
- Understanding of existing `db/neo4j_optimized.py`
- Session 08 complete (Event Sourcing - for projection integration)
- Familiarity with Neo4j Cypher query language
- Understanding of graph algorithms (PageRank, community detection)

---

## Part 1: Understanding Graph-First Architecture

### Core Concept
In a graph-first architecture, the graph database is not just for querying relationships - it IS the primary data model. All biblical data is inherently relational (verses connect to verses, words to meanings, types to antitypes), making a graph the natural representation.

### Why Graph-First for BIBLOS

1. **Natural Data Model**: Scripture IS a network (cross-references, typology, quotations)
2. **Path Queries**: Find paths between any two verses through multiple relationship types
3. **Pattern Matching**: Discover structural patterns across the entire canon
4. **Graph Algorithms**: PageRank for verse centrality, community detection for thematic clusters
5. **Real-Time Updates**: Add new connections without schema changes

### SPIDERWEB Schema Vision

The complete interconnected web of biblical data:
```
Verse ─── CROSS_REFERENCES ──→ Verse
  │                              │
  ├── HAS_WORD ──→ Word ←── HAS_WORD ──┤
  │                  │                  │
  │         HAS_LEMMA ↓                 │
  │                Lemma                │
  │                  │                  │
  │         HAS_MEANING ↓               │
  │               Meaning               │
  │                                     │
  ├── TYPIFIES ──────────────→ Verse    │
  │                              │      │
  ├── FULFILLS ──────────────→ Verse    │
  │                              │      │
  └── QUOTED_IN ─────────────→ Verse    │
                                 │      │
Father ── INTERPRETS ───────────┘      │
   │                                    │
   └── CITES ──────────────────────────┘
```

---

## Part 2: Complete Node Schema

### File: `db/neo4j_schema.py`

**Node Types**:

#### 1. `Verse` Node
```cypher
CREATE (v:Verse {
    id: "GEN.1.1",           // Primary key: BOOK.CHAPTER.VERSE
    book: "GEN",
    book_full: "Genesis",
    chapter: 1,
    verse: 1,
    testament: "OT",          // OT or NT

    // Text content
    text_hebrew: "בְּרֵאשִׁית בָּרָא אֱלֹהִים...",
    text_greek: "Ἐν ἀρχῇ ἐποίησεν ὁ θεὸς...",  // LXX for OT, NT Greek for NT
    text_english: "In the beginning...",

    // Computed properties
    word_count: 7,
    centrality_score: 0.0,    // PageRank, updated by algorithms
    community_id: 0,          // Community detection result
    processing_status: "completed",

    // Timestamps
    created_at: datetime(),
    updated_at: datetime()
})
```

#### 2. `Word` Node
```cypher
CREATE (w:Word {
    id: "GEN.1.1.1",          // verse_id + position
    verse_id: "GEN.1.1",
    position: 1,               // Word position in verse
    surface_form: "בְּרֵאשִׁית",
    transliteration: "bereshit",
    lemma_id: "H7225",         // Strong's number
    part_of_speech: "noun",
    morphology: "Ncfsa",       // Morphological code
    gloss: "beginning"
})
```

#### 3. `Lemma` Node
```cypher
CREATE (l:Lemma {
    id: "H7225",               // Strong's number
    language: "hebrew",
    lemma: "רֵאשִׁית",
    transliteration: "reshith",
    gloss: "beginning, first",
    occurrence_count: 51,
    semantic_domain: "time",

    // From OmniContextual Resolver (Session 03)
    polysemous: true,
    meaning_count: 3
})
```

#### 4. `Meaning` Node
```cypher
CREATE (m:Meaning {
    id: "H7225.1",
    lemma_id: "H7225",
    meaning: "beginning (temporal)",
    definition: "The first point in time of something",
    usage_count: 42,
    theological_weight: 0.8,
    example_verses: ["GEN.1.1", "PRO.8.22"]
})
```

#### 5. `Father` Node (Patristic Sources)
```cypher
CREATE (f:Father {
    id: "chrysostom",
    name: "John Chrysostom",
    name_greek: "Ἰωάννης ὁ Χρυσόστομος",
    tradition: "Eastern",
    era: "4th-5th century",
    birth_year: 349,
    death_year: 407,
    notable_works: ["Homilies on Genesis", "Homilies on Matthew"],
    authority_weight: 0.95    // For constraint validation
})
```

#### 6. `Work` Node (Patristic Works)
```cypher
CREATE (w:Work {
    id: "chrysostom_gen_hom",
    father_id: "chrysostom",
    title: "Homilies on Genesis",
    title_original: "Εἰς τὴν Γένεσιν ὁμιλίαι",
    type: "homily",
    book_count: 67,
    approximate_date: 388
})
```

#### 7. `Prophecy` Node
```cypher
CREATE (p:Prophecy {
    id: "virgin_birth",
    name: "Virgin Birth Prophecy",
    prophecy_verse: "ISA.7.14",
    natural_probability: 1e-10,
    independence_level: "fully_independent",
    category: "messianic"
})
```

#### 8. `Covenant` Node
```cypher
CREATE (c:Covenant {
    id: "abrahamic",
    name: "Abrahamic Covenant",
    initiation_verse: "GEN.12.1-3",
    key_promises: ["land", "seed", "blessing"],
    conditional: false,
    status: "eternal"
})
```

#### 9. `TypePattern` Node
```cypher
CREATE (tp:TypePattern {
    id: "sacrificial_lamb",
    name: "Sacrificial Lamb Pattern",
    layers: ["WORD", "PHRASE", "PERICOPE"],
    canonical_type: "GEN.22.8",
    canonical_antitype: "JHN.1.29"
})
```

---

## Part 3: Complete Relationship Schema

### Relationship Types

#### 1. Cross-Reference Relationships
```cypher
// Basic cross-reference
CREATE (a:Verse)-[:CROSS_REFERENCES {
    id: "GEN.1.1:JHN.1.1",
    connection_type: "thematic",      // thematic, verbal, typological, etc.
    confidence: 0.92,
    discovery_method: "gnn",
    mutual_influence: 0.75,           // From Session 01
    necessity_score: 0.65,            // From Session 04
    validated: true,
    validator: "theologos",
    created_at: datetime()
}]->(b:Verse)
```

#### 2. Typological Relationships
```cypher
// Type-antitype connection
CREATE (type:Verse)-[:TYPIFIES {
    pattern_id: "sacrificial_lamb",
    fractal_depth: 4,                 // From Session 06
    composite_strength: 0.87,
    dominant_layer: "PERICOPE",
    relation: "prefiguration"
}]->(antitype:Verse)
```

#### 3. Prophetic Relationships
```cypher
// Prophecy-fulfillment
CREATE (prophecy:Verse)-[:FULFILLED_IN {
    prophecy_id: "virgin_birth",
    fulfillment_type: "explicit",
    necessity_score: 0.95,
    lxx_support: true,
    manuscript_confidence: 0.98
}]->(fulfillment:Verse)
```

#### 4. Quotation Relationships
```cypher
// NT quoting OT
CREATE (ot:Verse)-[:QUOTED_IN {
    quote_type: "exact",              // exact, adapted, allusion
    follows_lxx: true,
    follows_mt: false,
    introduced_by: "as it is written"
}]->(nt:Verse)
```

#### 5. Linguistic Relationships
```cypher
// Verse contains words
CREATE (v:Verse)-[:HAS_WORD {position: 1}]->(w:Word)

// Word has lemma
CREATE (w:Word)-[:HAS_LEMMA]->(l:Lemma)

// Lemma has meanings
CREATE (l:Lemma)-[:HAS_MEANING]->(m:Meaning)

// Words share lemma (implicit via shared Lemma node)
```

#### 6. Patristic Relationships
```cypher
// Father interprets verse
CREATE (f:Father)-[:INTERPRETS {
    work_id: "chrysostom_gen_hom",
    homily: 2,
    interpretation: "Chrysostom sees in 'beginning'...",
    fourfold_sense: "literal",
    citation: "PG 53.23"
}]->(v:Verse)

// Father cites another Father
CREATE (f1:Father)-[:CITES {
    context: "agreement",
    work_id: "augustine_civ_dei"
}]->(f2:Father)
```

#### 7. Covenant Relationships
```cypher
// Verse belongs to covenant arc
CREATE (v:Verse)-[:IN_COVENANT {
    role: "promise",                  // promise, condition, fulfillment
    promise_element: "seed"
}]->(c:Covenant)

// Covenant builds on prior covenant
CREATE (c1:Covenant)-[:BUILDS_ON]->(c2:Covenant)
```

---

## Part 4: Graph Algorithm Integration

### File: `db/neo4j_algorithms.py`

**PageRank for Verse Centrality**:
```python
async def calculate_verse_centrality(self) -> None:
    """
    Calculate PageRank centrality for all verses based on cross-references.
    High centrality = verse is highly connected (hub verse).
    """
    await self.execute("""
        CALL gds.pageRank.write({
            nodeProjection: 'Verse',
            relationshipProjection: {
                CROSS_REFERENCES: {
                    type: 'CROSS_REFERENCES',
                    properties: ['confidence']
                }
            },
            relationshipWeightProperty: 'confidence',
            writeProperty: 'centrality_score',
            dampingFactor: 0.85,
            maxIterations: 20
        })
    """)
```

**Community Detection for Thematic Clusters**:
```python
async def detect_thematic_communities(self) -> Dict[int, List[str]]:
    """
    Detect communities of verses that are densely interconnected.
    These represent thematic clusters.
    """
    result = await self.execute("""
        CALL gds.louvain.stream({
            nodeProjection: 'Verse',
            relationshipProjection: 'CROSS_REFERENCES',
            relationshipWeightProperty: 'confidence'
        })
        YIELD nodeId, communityId
        RETURN gds.util.asNode(nodeId).id AS verse_id, communityId
        ORDER BY communityId
    """)

    communities = {}
    for record in result:
        community_id = record["communityId"]
        verse_id = record["verse_id"]
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(verse_id)

    return communities
```

**Shortest Path for Connection Discovery**:
```python
async def find_connection_path(
    self,
    verse_a: str,
    verse_b: str,
    max_hops: int = 5
) -> List[Dict]:
    """
    Find shortest path between two verses through any relationship type.
    """
    result = await self.execute("""
        MATCH path = shortestPath(
            (a:Verse {id: $verse_a})-[*..{max_hops}]-(b:Verse {id: $verse_b})
        )
        RETURN path,
               [r IN relationships(path) | type(r)] AS rel_types,
               [n IN nodes(path) | n.id] AS verse_ids,
               length(path) AS hops
    """, verse_a=verse_a, verse_b=verse_b, max_hops=max_hops)

    return result
```

**Pattern Matching for Typological Chains**:
```python
async def find_typological_chain(
    self,
    start_verse: str,
    pattern_id: str
) -> List[Dict]:
    """
    Find chain of typological connections following a pattern.
    """
    result = await self.execute("""
        MATCH chain = (start:Verse {id: $start_verse})
            -[:TYPIFIES*1..10]->(end:Verse)
        WHERE ALL(r IN relationships(chain) WHERE r.pattern_id = $pattern_id)
        RETURN [n IN nodes(chain) | n.id] AS verses,
               [r IN relationships(chain) | r.composite_strength] AS strengths,
               length(chain) AS depth
        ORDER BY depth DESC
        LIMIT 10
    """, start_verse=start_verse, pattern_id=pattern_id)

    return result
```

**Centrality-Weighted Cross-Reference Discovery**:
```python
async def discover_central_connections(
    self,
    verse_id: str,
    top_k: int = 10
) -> List[Dict]:
    """
    Find cross-references weighted by target verse centrality.
    Prioritizes connections to hub verses.
    """
    result = await self.execute("""
        MATCH (source:Verse {id: $verse_id})-[r:CROSS_REFERENCES]->(target:Verse)
        RETURN target.id AS target_verse,
               r.confidence AS confidence,
               target.centrality_score AS centrality,
               r.confidence * target.centrality_score AS weighted_score
        ORDER BY weighted_score DESC
        LIMIT $top_k
    """, verse_id=verse_id, top_k=top_k)

    return result
```

---

## Part 5: Graph-First Query Patterns

### Common Query Patterns

**Pattern 1: Theological Web Around a Verse**
```cypher
// Get complete context web for a verse
MATCH (v:Verse {id: $verse_id})
OPTIONAL MATCH (v)-[cr:CROSS_REFERENCES]-(connected:Verse)
OPTIONAL MATCH (v)-[:HAS_WORD]->(w:Word)-[:HAS_LEMMA]->(l:Lemma)
OPTIONAL MATCH (f:Father)-[i:INTERPRETS]->(v)
OPTIONAL MATCH (v)-[:TYPIFIES|FULFILLED_IN]->(related:Verse)
OPTIONAL MATCH (v)-[:IN_COVENANT]->(c:Covenant)
RETURN v, collect(DISTINCT connected) AS cross_refs,
       collect(DISTINCT l) AS lemmas,
       collect(DISTINCT {father: f, interpretation: i}) AS patristic,
       collect(DISTINCT related) AS typological,
       collect(DISTINCT c) AS covenants
```

**Pattern 2: Multi-Hop Intertextual Path**
```cypher
// Find how verse A connects to verse B through intermediate verses
MATCH path = (a:Verse {id: $verse_a})-[:CROSS_REFERENCES*1..4]-(b:Verse {id: $verse_b})
WITH path, [r IN relationships(path) | r.confidence] AS confidences
WITH path, reduce(s = 1.0, c IN confidences | s * c) AS path_confidence
RETURN [n IN nodes(path) | n.id] AS verses,
       [r IN relationships(path) | r.connection_type] AS types,
       path_confidence
ORDER BY path_confidence DESC
LIMIT 5
```

**Pattern 3: Lemma-Based Verbal Connections**
```cypher
// Find verses sharing significant lemmas (verbal parallels)
MATCH (v1:Verse {id: $verse_id})-[:HAS_WORD]->(w1:Word)-[:HAS_LEMMA]->(l:Lemma)
WHERE l.occurrence_count < 100  // Rare enough to be significant
MATCH (v2:Verse)-[:HAS_WORD]->(w2:Word)-[:HAS_LEMMA]->(l)
WHERE v2.id <> $verse_id
WITH v2, collect(DISTINCT l.lemma) AS shared_lemmas, count(DISTINCT l) AS lemma_count
WHERE lemma_count >= 2
RETURN v2.id AS verse_id, shared_lemmas, lemma_count
ORDER BY lemma_count DESC
LIMIT 20
```

**Pattern 4: Patristic Consensus Query**
```cypher
// Find patristic consensus on a verse
MATCH (f:Father)-[i:INTERPRETS]->(v:Verse {id: $verse_id})
WITH v, collect({
    father: f.name,
    tradition: f.tradition,
    interpretation: i.interpretation,
    sense: i.fourfold_sense
}) AS interpretations
RETURN v.id,
       size(interpretations) AS father_count,
       [i IN interpretations WHERE i.tradition = "Eastern"] AS eastern,
       [i IN interpretations WHERE i.tradition = "Western"] AS western
```

**Pattern 5: Covenant Arc Tracing**
```cypher
// Trace covenant promises to fulfillments
MATCH (promise:Verse)-[:IN_COVENANT {role: "promise"}]->(c:Covenant {id: $covenant_id})
MATCH (fulfillment:Verse)-[:IN_COVENANT {role: "fulfillment"}]->(c)
OPTIONAL MATCH (promise)-[t:TYPIFIES]->(fulfillment)
RETURN promise.id AS promise_verse,
       fulfillment.id AS fulfillment_verse,
       c.key_promises AS promises,
       t.composite_strength AS typological_strength
ORDER BY typological_strength DESC
```

---

## Part 6: Event Projection Integration

### Integration with Session 08 Event Sourcing

**Neo4j Projection Handler**:
```python
class Neo4jEventProjection(ProjectionBase):
    """
    Projects events to Neo4j graph in real-time.
    """

    async def _handle_CrossReferenceDiscoveredEvent(self, event):
        await self.neo4j.execute("""
            MERGE (s:Verse {id: $source_ref})
            ON CREATE SET s.created_at = datetime()
            MERGE (t:Verse {id: $target_ref})
            ON CREATE SET t.created_at = datetime()
            CREATE (s)-[:CROSS_REFERENCES {
                id: $id,
                connection_type: $type,
                confidence: $confidence,
                discovery_method: $method,
                created_at: datetime($timestamp)
            }]->(t)
        """, source_ref=event.source_ref, target_ref=event.target_ref,
             id=event.aggregate_id, type=event.connection_type,
             confidence=event.initial_confidence, method=event.discovery_method,
             timestamp=event.timestamp.isoformat())

    async def _handle_CrossReferenceRefinedEvent(self, event):
        # Update relationship properties
        for key, value in event.new_values.items():
            await self.neo4j.execute(f"""
                MATCH ()-[r:CROSS_REFERENCES {{id: $id}}]->()
                SET r.{key} = $value,
                    r.updated_at = datetime()
            """, id=event.cross_ref_id, value=value)

    async def _handle_TypologicalConnectionIdentifiedEvent(self, event):
        await self.neo4j.execute("""
            MATCH (t:Verse {id: $type_ref})
            MATCH (a:Verse {id: $antitype_ref})
            CREATE (t)-[:TYPIFIES {
                pattern_id: $pattern,
                fractal_depth: $depth,
                composite_strength: $strength,
                dominant_layer: $layer,
                created_at: datetime()
            }]->(a)
        """, type_ref=event.type_ref, antitype_ref=event.antitype_ref,
             pattern=event.pattern_id, depth=event.fractal_depth,
             strength=event.composite_strength, layer=event.dominant_layer)
```

---

## Part 7: Schema Migration

### Migration Script: `db/migrations/neo4j_graph_first.py`

**Phase 1: Create Constraints and Indexes**
```cypher
// Unique constraints
CREATE CONSTRAINT verse_id IF NOT EXISTS FOR (v:Verse) REQUIRE v.id IS UNIQUE;
CREATE CONSTRAINT lemma_id IF NOT EXISTS FOR (l:Lemma) REQUIRE l.id IS UNIQUE;
CREATE CONSTRAINT father_id IF NOT EXISTS FOR (f:Father) REQUIRE f.id IS UNIQUE;
CREATE CONSTRAINT covenant_id IF NOT EXISTS FOR (c:Covenant) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT prophecy_id IF NOT EXISTS FOR (p:Prophecy) REQUIRE p.id IS UNIQUE;

// Performance indexes
CREATE INDEX verse_book IF NOT EXISTS FOR (v:Verse) ON (v.book);
CREATE INDEX verse_testament IF NOT EXISTS FOR (v:Verse) ON (v.testament);
CREATE INDEX verse_centrality IF NOT EXISTS FOR (v:Verse) ON (v.centrality_score);
CREATE INDEX lemma_language IF NOT EXISTS FOR (l:Lemma) ON (l.language);
CREATE INDEX father_tradition IF NOT EXISTS FOR (f:Father) ON (f.tradition);

// Full-text indexes for search
CREATE FULLTEXT INDEX verse_text IF NOT EXISTS FOR (v:Verse) ON EACH [v.text_english, v.text_hebrew, v.text_greek];
```

**Phase 2: Migrate Existing Data**
```python
async def migrate_verses_to_graph():
    """Migrate all verses from PostgreSQL to Neo4j."""
    async for verse in postgres.get_all_verses():
        await neo4j.execute("""
            MERGE (v:Verse {id: $id})
            SET v.book = $book,
                v.chapter = $chapter,
                v.verse = $verse_num,
                v.testament = $testament,
                v.text_hebrew = $hebrew,
                v.text_greek = $greek,
                v.text_english = $english,
                v.word_count = $word_count,
                v.created_at = datetime($created)
        """, id=verse.id, book=verse.book, chapter=verse.chapter,
             verse_num=verse.verse, testament=verse.testament,
             hebrew=verse.text_hebrew, greek=verse.text_greek,
             english=verse.text_english, word_count=verse.word_count,
             created=verse.created_at.isoformat())

async def migrate_cross_references():
    """Migrate cross-references to graph relationships."""
    async for ref in postgres.get_all_cross_references():
        await neo4j.execute("""
            MATCH (s:Verse {id: $source})
            MATCH (t:Verse {id: $target})
            CREATE (s)-[:CROSS_REFERENCES {
                id: $id,
                connection_type: $type,
                confidence: $confidence,
                mutual_influence: $mutual,
                necessity_score: $necessity,
                validated: $validated
            }]->(t)
        """, source=ref.source_ref, target=ref.target_ref,
             id=ref.id, type=ref.connection_type,
             confidence=ref.confidence, mutual=ref.mutual_influence_score,
             necessity=ref.necessity_score, validated=ref.validated)
```

---

## Part 8: Testing Specification

### Unit Tests: `tests/db/test_neo4j_graph_first.py`

**Test 1: `test_verse_creation`**
- Create verse node
- Verify all properties set
- Check indexes work

**Test 2: `test_cross_reference_creation`**
- Create two verses
- Create cross-reference relationship
- Verify bidirectional query works

**Test 3: `test_pagerank_calculation`**
- Create network of verses
- Run PageRank
- Verify hub verses have high centrality

**Test 4: `test_community_detection`**
- Create clustered verse network
- Run Louvain
- Verify clusters detected correctly

**Test 5: `test_shortest_path`**
- Create path between distant verses
- Query shortest path
- Verify correct path returned

**Test 6: `test_typological_chain`**
- Create multi-hop typological chain
- Query chain by pattern
- Verify complete chain returned

**Test 7: `test_event_projection`**
- Emit cross-reference event
- Verify graph updated
- Check all properties projected

**Test 8: `test_patristic_consensus`**
- Add multiple Father interpretations
- Query consensus
- Verify tradition grouping

---

## Part 9: Configuration

### Add to `config.py`

**New Configuration Dataclass**: `Neo4jGraphFirstConfig`

Fields:
- `uri: str = "bolt://localhost:7687"`
- `username: str = "neo4j"`
- `password: str = ""`  # From environment
- `database: str = "biblos"`
- `max_connection_pool_size: int = 50`
- `connection_timeout_seconds: int = 30`
- `enable_gds_algorithms: bool = True`  # Graph Data Science
- `pagerank_damping: float = 0.85`
- `pagerank_iterations: int = 20`
- `community_resolution: float = 1.0`  # Louvain resolution
- `auto_calculate_centrality: bool = True`
- `centrality_recalc_interval_hours: int = 24`

---

## Part 10: Performance Optimization

### Query Optimization Strategies

**1. Use Relationship Indexes**:
```cypher
// Create relationship property index for frequent queries
CREATE INDEX cross_ref_type FOR ()-[r:CROSS_REFERENCES]-() ON (r.connection_type)
```

**2. Batch Operations**:
```python
async def batch_create_relationships(self, relationships: List[Dict]):
    """Create many relationships efficiently."""
    await self.execute("""
        UNWIND $rels AS rel
        MATCH (s:Verse {id: rel.source})
        MATCH (t:Verse {id: rel.target})
        CREATE (s)-[:CROSS_REFERENCES {
            id: rel.id,
            connection_type: rel.type,
            confidence: rel.confidence
        }]->(t)
    """, rels=relationships)
```

**3. Projection for Algorithms**:
```cypher
// Create in-memory graph projection for faster algorithms
CALL gds.graph.project(
    'biblos-graph',
    'Verse',
    'CROSS_REFERENCES',
    {
        relationshipProperties: ['confidence']
    }
)
```

**4. Caching Hot Queries**:
```python
# Cache frequently accessed verse webs
@cached(ttl=3600)
async def get_verse_web(self, verse_id: str) -> Dict:
    return await self._query_verse_web(verse_id)
```

---

## Part 11: Success Criteria

### Functional Requirements
- [ ] All node types created with proper schemas
- [ ] All relationship types implemented
- [ ] Graph algorithms (PageRank, Louvain) working
- [ ] Path finding queries functional
- [ ] Event projection updating graph in real-time
- [ ] Migration from PostgreSQL complete

### Performance Requirements
- [ ] Single verse query: < 50ms
- [ ] Web query (verse + connections): < 200ms
- [ ] Shortest path (5 hops): < 500ms
- [ ] PageRank calculation: < 5 minutes
- [ ] Community detection: < 10 minutes
- [ ] Batch insert (1000 relationships): < 10 seconds

### Data Integrity
- [ ] All verses from corpus present
- [ ] All cross-references migrated
- [ ] Relationship properties preserved
- [ ] No orphan nodes

---

## Part 12: Detailed Implementation Order

1. **Update `db/neo4j_schema.py`** with complete node/relationship types
2. **Create constraint/index migration script**
3. **Implement `Neo4jGraphClient`** with all CRUD operations
4. **Create `db/neo4j_algorithms.py`** with GDS integrations
5. **Implement query patterns** for common use cases
6. **Create event projection handler** (integrate with Session 08)
7. **Write migration script** from PostgreSQL
8. **Run migration** for all existing data
9. **Calculate initial PageRank** and community detection
10. **Add configuration to `config.py`**
11. **Write unit tests**
12. **Performance benchmark** critical queries
13. **Document query patterns**

---

## Part 13: Dependencies on Other Sessions

### Depends On
- SESSION 08: Event Sourcing (for projection integration)

### Depended On By
- SESSION 10: Vector DB Enhancement (uses graph for context)
- SESSION 11: Pipeline Integration (queries graph for enrichment)

### External Dependencies
- Neo4j 5.x with Graph Data Science library
- APOC procedures for utility functions

---

## Session Completion Checklist

```markdown
- [ ] `db/neo4j_schema.py` with complete schema
- [ ] `db/neo4j_algorithms.py` implemented
- [ ] `db/neo4j_graph_client.py` with all operations
- [ ] Constraint/index migration complete
- [ ] Data migration from PostgreSQL complete
- [ ] Event projection integrated (Session 08)
- [ ] PageRank calculation working
- [ ] Community detection working
- [ ] Path finding queries working
- [ ] Configuration added to config.py
- [ ] Unit tests passing
- [ ] Performance benchmarks met
- [ ] Documentation complete
```

**Next Session**: SESSION 10: Vector DB Enhancement

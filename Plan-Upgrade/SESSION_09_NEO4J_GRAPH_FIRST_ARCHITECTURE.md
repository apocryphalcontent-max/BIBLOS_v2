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

### Graph Topology Metrics

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class GraphMetricType(Enum):
    """Metrics computed over the SPIDERWEB graph."""
    DEGREE_CENTRALITY = "degree_centrality"
    BETWEENNESS_CENTRALITY = "betweenness_centrality"
    PAGERANK = "pagerank"
    CLUSTERING_COEFFICIENT = "clustering_coefficient"
    COMMUNITY_ID = "community_id"

    @property
    def computation_cost(self) -> str:
        """GDS algorithm complexity class."""
        return {
            GraphMetricType.DEGREE_CENTRALITY: "O(V+E)",
            GraphMetricType.BETWEENNESS_CENTRALITY: "O(V*E)",
            GraphMetricType.PAGERANK: "O(iterations*(V+E))",
            GraphMetricType.CLUSTERING_COEFFICIENT: "O(V*d²)",
            GraphMetricType.COMMUNITY_ID: "O(V+E*log(V))",
        }[self]

@dataclass
class GraphStatistics:
    """Global statistics for the SPIDERWEB graph."""
    node_count: int
    edge_count: int
    avg_degree: float
    density: float
    diameter: Optional[int]  # None if disconnected
    connected_components: int
    largest_component_size: int

    @property
    def is_sparse(self) -> bool:
        """Biblical graphs are inherently sparse."""
        return self.density < 0.01

    def coverage_ratio(self) -> float:
        """Fraction of verses in largest component."""
        return self.largest_component_size / self.node_count if self.node_count else 0.0
```

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

**Node Label Configuration**:

```python
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

class NodeLabel(Enum):
    """All node labels in the SPIDERWEB schema."""
    VERSE = "Verse"
    WORD = "Word"
    LEMMA = "Lemma"
    MEANING = "Meaning"
    FATHER = "Father"
    WORK = "Work"
    PROPHECY = "Prophecy"
    COVENANT = "Covenant"
    TYPE_PATTERN = "TypePattern"
    SEMANTIC_DOMAIN = "SemanticDomain"
    LITURGICAL_CONTEXT = "LiturgicalContext"

    @property
    def primary_key(self) -> str:
        """Primary key field for each node type."""
        return {
            NodeLabel.VERSE: "id",
            NodeLabel.WORD: "id",
            NodeLabel.LEMMA: "id",
            NodeLabel.MEANING: "id",
            NodeLabel.FATHER: "id",
            NodeLabel.WORK: "id",
            NodeLabel.PROPHECY: "id",
            NodeLabel.COVENANT: "id",
            NodeLabel.TYPE_PATTERN: "id",
            NodeLabel.SEMANTIC_DOMAIN: "domain_id",
            NodeLabel.LITURGICAL_CONTEXT: "context_id",
        }[self]

    @property
    def required_indexes(self) -> List[str]:
        """Fields requiring secondary indexes."""
        return {
            NodeLabel.VERSE: ["book", "testament", "centrality_score", "community_id"],
            NodeLabel.WORD: ["verse_id", "lemma_id"],
            NodeLabel.LEMMA: ["language", "semantic_domain"],
            NodeLabel.FATHER: ["tradition", "era"],
        }.get(self, [])

@dataclass
class NodeSchema:
    """Schema definition for a node type."""
    label: NodeLabel
    properties: Dict[str, type]
    required_properties: List[str]
    computed_properties: List[str] = field(default_factory=list)
    full_text_fields: List[str] = field(default_factory=list)

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate data against schema."""
        for prop in self.required_properties:
            if prop not in data or data[prop] is None:
                return False
        return True
```

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
    betweenness_score: 0.0,   // Betweenness centrality
    community_id: 0,          // Community detection result
    hub_score: 0.0,           // HITS hub score
    authority_score: 0.0,     // HITS authority score
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
    gloss: "beginning",

    // Syntactic role from SDES
    syntactic_role: "subject",
    clause_type: "main",
    discourse_function: "topic"
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
    meaning_count: 3,

    // Distribution metrics
    ot_occurrences: 51,
    nt_occurrences: 0,  // Greek equivalent tracked separately
    book_distribution: ["GEN", "EXO", "LEV", "NUM", "DEU", "JER", "PRO"]
})
```

#### 4. `Meaning` Node with Disambiguation
```cypher
CREATE (m:Meaning {
    id: "H7225.1",
    lemma_id: "H7225",
    meaning: "beginning (temporal)",
    definition: "The first point in time of something",
    usage_count: 42,
    theological_weight: 0.8,
    example_verses: ["GEN.1.1", "PRO.8.22"],

    // Disambiguation features from Session 03
    syntactic_patterns: ["construct_chain", "absolute"],
    collocations: ["H3117", "H8141"],  // Common co-occurring lemmas
    semantic_field: "temporal_sequence"
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
    authority_weight: 0.95,    // For constraint validation

    // Citation graph metrics
    citation_count: 0,         // Times cited by other Fathers
    influence_score: 0.0       // Computed from citation graph
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
    approximate_date: 388,

    // Coverage statistics
    verses_covered: 1200,
    ot_coverage: 0.04,
    genre: "expository"
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
    category: "messianic",

    // From Session 07
    specificity_factors: ["person_name", "biological_miracle"],
    compound_probability: 1e-15,
    bayesian_strength: "decisive"
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
    status: "eternal",

    // Covenant arc tracking
    promise_verses: ["GEN.12.1-3", "GEN.15.1-21", "GEN.17.1-14"],
    fulfillment_verses: ["GAL.3.16", "ROM.4.13-16"],
    progression_order: 2  // After Noahic, before Mosaic
})
```

#### 9. `TypePattern` Node
```cypher
CREATE (tp:TypePattern {
    id: "sacrificial_lamb",
    name: "Sacrificial Lamb Pattern",
    layers: ["WORD", "PHRASE", "PERICOPE"],
    canonical_type: "GEN.22.8",
    canonical_antitype: "JHN.1.29",

    // From Session 06
    keywords_hebrew: ["H3532", "H2076"],  // lamb, sacrifice
    keywords_greek: ["G286", "G2380"],
    correspondence_points: 7,
    fractal_depth: 5
})
```

---

## Part 3: Complete Relationship Schema

### Relationship Type Configuration

```python
class RelationshipType(Enum):
    """All relationship types in SPIDERWEB."""
    CROSS_REFERENCES = "CROSS_REFERENCES"
    TYPIFIES = "TYPIFIES"
    FULFILLED_IN = "FULFILLED_IN"
    QUOTED_IN = "QUOTED_IN"
    HAS_WORD = "HAS_WORD"
    HAS_LEMMA = "HAS_LEMMA"
    HAS_MEANING = "HAS_MEANING"
    INTERPRETS = "INTERPRETS"
    CITES = "CITES"
    IN_COVENANT = "IN_COVENANT"
    BUILDS_ON = "BUILDS_ON"
    IN_SEMANTIC_DOMAIN = "IN_SEMANTIC_DOMAIN"

    @property
    def is_weighted(self) -> bool:
        """Whether this relationship carries a weight/confidence."""
        return self in {
            RelationshipType.CROSS_REFERENCES,
            RelationshipType.TYPIFIES,
            RelationshipType.FULFILLED_IN,
            RelationshipType.QUOTED_IN,
        }

    @property
    def weight_property(self) -> Optional[str]:
        """Name of the weight property if weighted."""
        if not self.is_weighted:
            return None
        return {
            RelationshipType.CROSS_REFERENCES: "confidence",
            RelationshipType.TYPIFIES: "composite_strength",
            RelationshipType.FULFILLED_IN: "necessity_score",
            RelationshipType.QUOTED_IN: "match_confidence",
        }[self]
```

### Relationship Types

#### 1. Cross-Reference Relationships
```cypher
// Basic cross-reference with full provenance
CREATE (a:Verse)-[:CROSS_REFERENCES {
    id: "GEN.1.1:JHN.1.1",
    connection_type: "thematic",      // thematic, verbal, typological, etc.
    confidence: 0.92,
    discovery_method: "gnn",
    mutual_influence: 0.75,           // From Session 01
    necessity_score: 0.65,            // From Session 04
    validated: true,
    validator: "theologos",
    validation_timestamp: datetime(),

    // Multi-dimensional scoring
    semantic_similarity: 0.88,
    structural_similarity: 0.72,
    theological_alignment: 0.95,

    created_at: datetime()
}]->(b:Verse)
```

#### 2. Typological Relationships
```cypher
// Type-antitype connection with fractal detail
CREATE (type:Verse)-[:TYPIFIES {
    pattern_id: "sacrificial_lamb",
    fractal_depth: 4,                 // From Session 06
    composite_strength: 0.87,
    dominant_layer: "PERICOPE",
    relation: "prefiguration",

    // Layer-wise breakdown
    word_layer_score: 0.75,
    phrase_layer_score: 0.82,
    verse_layer_score: 0.88,
    pericope_layer_score: 0.91,

    correspondence_type: "symbolic"   // From Session 06
}]->(antitype:Verse)
```

#### 3. Prophetic Relationships
```cypher
// Prophecy-fulfillment with evidence
CREATE (prophecy:Verse)-[:FULFILLED_IN {
    prophecy_id: "virgin_birth",
    fulfillment_type: "explicit",
    necessity_score: 0.95,
    lxx_support: true,
    manuscript_confidence: 0.98,

    // Citation formula
    introduction_formula: "as_it_is_written",
    quotation_style: "exact_lxx"
}]->(fulfillment:Verse)
```

#### 4. Quotation Relationships
```cypher
// NT quoting OT with textual analysis
CREATE (ot:Verse)-[:QUOTED_IN {
    quote_type: "exact",              // exact, adapted, allusion
    follows_lxx: true,
    follows_mt: false,
    introduced_by: "as it is written",

    // Text-critical detail
    variant_readings: ["Vaticanus", "Sinaiticus"],
    agreement_percentage_lxx: 0.95,
    agreement_percentage_mt: 0.72,

    // Hermeneutical method
    interpretation_method: "typological"  // literal, typological, allegorical
}]->(nt:Verse)
```

#### 5. Linguistic Relationships
```cypher
// Verse contains words with positional data
CREATE (v:Verse)-[:HAS_WORD {
    position: 1,
    clause_id: "GEN.1.1.c1",
    syntactic_relation: "subject"
}]->(w:Word)

// Word has lemma
CREATE (w:Word)-[:HAS_LEMMA]->(l:Lemma)

// Lemma has meanings with context
CREATE (l:Lemma)-[:HAS_MEANING {
    frequency: 0.82,           // How often this meaning occurs
    primary: true
}]->(m:Meaning)
```

#### 6. Patristic Relationships
```cypher
// Father interprets verse with full citation
CREATE (f:Father)-[:INTERPRETS {
    work_id: "chrysostom_gen_hom",
    homily: 2,
    interpretation: "Chrysostom sees in 'beginning'...",
    fourfold_sense: "literal",
    citation: "PG 53.23",

    // Interpretation metadata
    interpretation_length: 450,
    quotes_verse_directly: true,
    references_other_fathers: ["origen"],
    liturgical_context: "lectionary_reading"
}]->(v:Verse)

// Father cites another Father with context
CREATE (f1:Father)-[:CITES {
    context: "agreement",             // agreement, refutation, elaboration
    work_id: "augustine_civ_dei",
    book: 11,
    chapter: 6,
    nature: "explicit"                // explicit, implicit, allusion
}]->(f2:Father)
```

#### 7. Covenant Relationships
```cypher
// Verse belongs to covenant arc with role
CREATE (v:Verse)-[:IN_COVENANT {
    role: "promise",                  // promise, condition, fulfillment
    promise_element: "seed",
    centrality_in_covenant: 0.85,     // How central to covenant theology
    explicit_covenant_mention: true
}]->(c:Covenant)

// Covenant builds on prior covenant
CREATE (c1:Covenant)-[:BUILDS_ON {
    continuity_type: "expansion",     // expansion, elaboration, transformation
    shared_promises: ["blessing"],
    new_elements: ["priesthood"]
}]->(c2:Covenant)
```

---

## Part 4: Graph Algorithm Integration

### File: `db/neo4j_algorithms.py`

**Algorithm Configuration**:

```python
@dataclass
class AlgorithmConfig:
    """Configuration for graph algorithms."""
    pagerank_damping: float = 0.85
    pagerank_iterations: int = 20
    louvain_resolution: float = 1.0
    louvain_max_levels: int = 10
    path_max_hops: int = 5
    batch_size: int = 1000

    # Algorithm projection settings
    relationship_orientation: str = "NATURAL"  # NATURAL, REVERSE, UNDIRECTED
    concurrent_algorithms: bool = True

ALGORITHM_CONFIG = AlgorithmConfig()
```

**PageRank for Verse Centrality**:
```python
async def calculate_verse_centrality(
    self,
    relationship_types: List[str] = None,
    write_property: str = "centrality_score"
) -> Dict[str, float]:
    """
    Calculate PageRank centrality for all verses based on cross-references.
    High centrality = verse is highly connected (hub verse).

    Returns dict of top 100 verses by centrality.
    """
    rel_types = relationship_types or ["CROSS_REFERENCES", "TYPIFIES", "FULFILLED_IN"]
    rel_projection = ", ".join([
        f"'{rt}': {{type: '{rt}', properties: ['confidence']}}"
        for rt in rel_types
    ])

    await self.execute(f"""
        CALL gds.pageRank.write({{
            nodeProjection: 'Verse',
            relationshipProjection: {{{rel_projection}}},
            relationshipWeightProperty: 'confidence',
            writeProperty: '{write_property}',
            dampingFactor: {ALGORITHM_CONFIG.pagerank_damping},
            maxIterations: {ALGORITHM_CONFIG.pagerank_iterations},
            concurrency: 4
        }})
    """)

    # Return top verses
    result = await self.execute(f"""
        MATCH (v:Verse)
        WHERE v.{write_property} > 0
        RETURN v.id AS verse_id, v.{write_property} AS centrality
        ORDER BY centrality DESC
        LIMIT 100
    """)

    return {r["verse_id"]: r["centrality"] for r in result}
```

**Community Detection for Thematic Clusters**:
```python
async def detect_thematic_communities(
    self,
    min_community_size: int = 5,
    include_singleton: bool = False
) -> Dict[int, List[str]]:
    """
    Detect communities of verses that are densely interconnected.
    These represent thematic clusters.

    Returns community_id -> list of verse_ids mapping.
    """
    result = await self.execute(f"""
        CALL gds.louvain.stream({{
            nodeProjection: 'Verse',
            relationshipProjection: 'CROSS_REFERENCES',
            relationshipWeightProperty: 'confidence',
            maxLevels: {ALGORITHM_CONFIG.louvain_max_levels},
            resolution: {ALGORITHM_CONFIG.louvain_resolution}
        }})
        YIELD nodeId, communityId
        WITH gds.util.asNode(nodeId).id AS verse_id, communityId
        WITH communityId, collect(verse_id) AS verses
        WHERE size(verses) >= {min_community_size}
        RETURN communityId, verses
        ORDER BY size(verses) DESC
    """)

    communities = {}
    for record in result:
        community_id = record["communityId"]
        verses = record["verses"]
        communities[community_id] = verses

    # Write community IDs back to nodes
    for community_id, verses in communities.items():
        await self.execute("""
            MATCH (v:Verse)
            WHERE v.id IN $verses
            SET v.community_id = $community_id
        """, verses=verses, community_id=community_id)

    return communities
```

**Shortest Path for Connection Discovery**:
```python
async def find_connection_path(
    self,
    verse_a: str,
    verse_b: str,
    max_hops: int = None,
    relationship_types: List[str] = None
) -> List[Dict]:
    """
    Find shortest path between two verses through specified relationship types.
    Returns path details including all intermediate verses and relationship types.
    """
    max_hops = max_hops or ALGORITHM_CONFIG.path_max_hops
    rel_filter = "|".join(relationship_types) if relationship_types else "*"

    result = await self.execute(f"""
        MATCH path = shortestPath(
            (a:Verse {{id: $verse_a}})-[:{rel_filter}*..{max_hops}]-(b:Verse {{id: $verse_b}})
        )
        WITH path,
             [r IN relationships(path) | type(r)] AS rel_types,
             [n IN nodes(path) | n.id] AS verse_ids,
             [r IN relationships(path) | coalesce(r.confidence, r.composite_strength, 1.0)] AS weights,
             length(path) AS hops
        RETURN verse_ids, rel_types, weights, hops,
               reduce(product = 1.0, w IN weights | product * w) AS path_confidence
        ORDER BY path_confidence DESC
    """, verse_a=verse_a, verse_b=verse_b)

    return [
        {
            "verses": r["verse_ids"],
            "relationship_types": r["rel_types"],
            "weights": r["weights"],
            "hops": r["hops"],
            "path_confidence": r["path_confidence"]
        }
        for r in result
    ]
```

**Pattern Matching for Typological Chains**:
```python
async def find_typological_chain(
    self,
    start_verse: str,
    pattern_id: str,
    max_depth: int = 10
) -> List[Dict]:
    """
    Find chain of typological connections following a pattern.
    Traces how a type develops through multiple fulfillments.
    """
    result = await self.execute("""
        MATCH chain = (start:Verse {id: $start_verse})
            -[:TYPIFIES*1..{max_depth}]->(end:Verse)
        WHERE ALL(r IN relationships(chain) WHERE r.pattern_id = $pattern_id)
        WITH chain,
             [n IN nodes(chain) | {id: n.id, testament: n.testament}] AS verses,
             [r IN relationships(chain) | {
                 strength: r.composite_strength,
                 layer: r.dominant_layer,
                 relation: r.relation
             }] AS connections,
             length(chain) AS depth
        RETURN verses, connections, depth,
               reduce(s = 1.0, c IN connections | s * c.strength) AS chain_confidence
        ORDER BY chain_confidence DESC
        LIMIT 10
    """, start_verse=start_verse, pattern_id=pattern_id, max_depth=max_depth)

    return list(result)
```

**Centrality-Weighted Cross-Reference Discovery**:
```python
async def discover_central_connections(
    self,
    verse_id: str,
    top_k: int = 10,
    min_confidence: float = 0.5
) -> List[Dict]:
    """
    Find cross-references weighted by target verse centrality.
    Prioritizes connections to hub verses (theologically significant nodes).
    """
    result = await self.execute("""
        MATCH (source:Verse {id: $verse_id})-[r:CROSS_REFERENCES]->(target:Verse)
        WHERE r.confidence >= $min_confidence
        WITH target, r,
             r.confidence AS confidence,
             target.centrality_score AS centrality,
             target.community_id AS community
        WITH target, r, confidence, centrality, community,
             confidence * (1 + log(1 + coalesce(centrality, 0))) AS weighted_score
        RETURN target.id AS target_verse,
               target.book AS book,
               confidence,
               centrality,
               community,
               weighted_score,
               r.connection_type AS connection_type
        ORDER BY weighted_score DESC
        LIMIT $top_k
    """, verse_id=verse_id, top_k=top_k, min_confidence=min_confidence)

    return list(result)
```

**HITS Algorithm for Hub/Authority Detection**:
```python
async def calculate_hits_scores(self) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate HITS (Hyperlink-Induced Topic Search) scores.
    Hub = verse that points to many authorities (OT cross-ref sources)
    Authority = verse pointed to by many hubs (key NT fulfillment verses)
    """
    await self.execute("""
        CALL gds.alpha.hits.write({
            nodeProjection: 'Verse',
            relationshipProjection: {
                REFS: {
                    type: 'CROSS_REFERENCES',
                    orientation: 'NATURAL'
                }
            },
            hitsIterations: 20,
            authProperty: 'authority_score',
            hubProperty: 'hub_score'
        })
    """)

    # Return top hubs and authorities
    hubs = await self.execute("""
        MATCH (v:Verse) WHERE v.hub_score > 0
        RETURN v.id AS verse_id, v.hub_score AS score
        ORDER BY score DESC LIMIT 50
    """)

    authorities = await self.execute("""
        MATCH (v:Verse) WHERE v.authority_score > 0
        RETURN v.id AS verse_id, v.authority_score AS score
        ORDER BY score DESC LIMIT 50
    """)

    return (
        {r["verse_id"]: r["score"] for r in hubs},
        {r["verse_id"]: r["score"] for r in authorities}
    )
```

---

## Part 5: Graph-First Query Patterns

### Common Query Patterns

**Pattern 1: Theological Web Around a Verse**
```cypher
// Get complete context web for a verse with depth control
MATCH (v:Verse {id: $verse_id})

// Cross-references with confidence threshold
OPTIONAL MATCH (v)-[cr:CROSS_REFERENCES]-(connected:Verse)
WHERE cr.confidence >= 0.5

// Linguistic structure
OPTIONAL MATCH (v)-[:HAS_WORD]->(w:Word)-[:HAS_LEMMA]->(l:Lemma)
OPTIONAL MATCH (l)-[:HAS_MEANING]->(m:Meaning)

// Patristic interpretation
OPTIONAL MATCH (f:Father)-[i:INTERPRETS]->(v)

// Typological connections
OPTIONAL MATCH (v)-[t:TYPIFIES|FULFILLED_IN]-(related:Verse)

// Covenant context
OPTIONAL MATCH (v)-[ic:IN_COVENANT]->(c:Covenant)

RETURN v,
       collect(DISTINCT {verse: connected, confidence: cr.confidence, type: cr.connection_type}) AS cross_refs,
       collect(DISTINCT {lemma: l, meaning: m}) AS linguistic,
       collect(DISTINCT {father: f.name, interpretation: i.interpretation, sense: i.fourfold_sense}) AS patristic,
       collect(DISTINCT {verse: related, relation: type(t)}) AS typological,
       collect(DISTINCT {covenant: c.name, role: ic.role}) AS covenants
```

**Pattern 2: Multi-Hop Intertextual Path with Confidence Decay**
```cypher
// Find how verse A connects to verse B with path confidence calculation
MATCH path = (a:Verse {id: $verse_a})-[:CROSS_REFERENCES*1..4]-(b:Verse {id: $verse_b})
WITH path,
     [r IN relationships(path) | r.confidence] AS confidences,
     [n IN nodes(path) | n.id] AS verse_ids,
     [r IN relationships(path) | r.connection_type] AS types

// Calculate path confidence with decay factor
WITH path, verse_ids, types, confidences,
     reduce(s = 1.0, c IN confidences | s * c * 0.95) AS path_confidence,
     size(verse_ids) AS hop_count

WHERE path_confidence > 0.1

RETURN verse_ids,
       types,
       confidences,
       path_confidence,
       hop_count
ORDER BY path_confidence DESC
LIMIT 5
```

**Pattern 3: Lemma-Based Verbal Connections with Rarity Weighting**
```cypher
// Find verses sharing significant lemmas (verbal parallels)
MATCH (v1:Verse {id: $verse_id})-[:HAS_WORD]->(w1:Word)-[:HAS_LEMMA]->(l:Lemma)
WHERE l.occurrence_count < 100  // Rare enough to be significant

MATCH (v2:Verse)-[:HAS_WORD]->(w2:Word)-[:HAS_LEMMA]->(l)
WHERE v2.id <> $verse_id

WITH v2, l,
     1.0 / log(l.occurrence_count + 1) AS rarity_weight

WITH v2,
     collect(DISTINCT {lemma: l.lemma, gloss: l.gloss, weight: rarity_weight}) AS shared_lemmas,
     sum(rarity_weight) AS total_weight,
     count(DISTINCT l) AS lemma_count

WHERE lemma_count >= 2

RETURN v2.id AS verse_id,
       v2.text_english AS text,
       shared_lemmas,
       lemma_count,
       total_weight
ORDER BY total_weight DESC
LIMIT 20
```

**Pattern 4: Patristic Consensus Query**
```cypher
// Find patristic consensus on a verse with tradition weighting
MATCH (f:Father)-[i:INTERPRETS]->(v:Verse {id: $verse_id})

WITH v, f, i,
     CASE f.tradition
         WHEN "Eastern" THEN 1.0
         WHEN "Western" THEN 0.9
         WHEN "Syriac" THEN 0.85
         ELSE 0.8
     END AS tradition_weight

WITH v, collect({
    father: f.name,
    tradition: f.tradition,
    era: f.era,
    interpretation: i.interpretation,
    sense: i.fourfold_sense,
    weight: tradition_weight * f.authority_weight
}) AS interpretations

RETURN v.id,
       size(interpretations) AS father_count,
       [i IN interpretations WHERE i.tradition = "Eastern"] AS eastern,
       [i IN interpretations WHERE i.tradition = "Western"] AS western,
       [i IN interpretations | i.sense] AS senses_used,
       reduce(s = 0.0, i IN interpretations | s + i.weight) AS total_authority_weight
```

**Pattern 5: Covenant Arc Tracing**
```cypher
// Trace covenant promises to fulfillments with typological overlay
MATCH (promise:Verse)-[ip:IN_COVENANT {role: "promise"}]->(c:Covenant {id: $covenant_id})
MATCH (fulfillment:Verse)-[if:IN_COVENANT {role: "fulfillment"}]->(c)

OPTIONAL MATCH (promise)-[t:TYPIFIES]->(fulfillment)
OPTIONAL MATCH (promise)-[f:FULFILLED_IN]->(fulfillment)

WITH promise, fulfillment, c, t, f,
     coalesce(t.composite_strength, 0) + coalesce(f.necessity_score, 0) AS connection_strength

RETURN promise.id AS promise_verse,
       promise.text_english AS promise_text,
       fulfillment.id AS fulfillment_verse,
       fulfillment.text_english AS fulfillment_text,
       c.key_promises AS promises,
       connection_strength,
       CASE WHEN t IS NOT NULL THEN "typological" ELSE NULL END AS has_typology,
       CASE WHEN f IS NOT NULL THEN "prophetic" ELSE NULL END AS has_prophecy
ORDER BY connection_strength DESC
```

---

## Part 6: Event Projection Integration

### Integration with Session 08 Event Sourcing

**Neo4j Projection Handler**:
```python
class Neo4jEventProjection(ProjectionBase):
    """
    Projects events to Neo4j graph in real-time.
    Maintains consistency between event store and graph.
    """

    def __init__(self, neo4j_client: Neo4jGraphClient):
        super().__init__(projection_name="neo4j_graph")
        self.neo4j = neo4j_client
        self._batch_buffer: List[EventBase] = []
        self._batch_size = 100

    async def _handle_CrossReferenceDiscoveredEvent(self, event: CrossReferenceDiscoveredEvent):
        """Project new cross-reference to graph."""
        await self.neo4j.execute("""
            MERGE (s:Verse {id: $source_ref})
            ON CREATE SET s.created_at = datetime(), s.processing_status = 'pending'
            MERGE (t:Verse {id: $target_ref})
            ON CREATE SET t.created_at = datetime(), t.processing_status = 'pending'
            CREATE (s)-[:CROSS_REFERENCES {
                id: $id,
                connection_type: $type,
                confidence: $confidence,
                discovery_method: $method,
                correlation_id: $correlation_id,
                created_at: datetime($timestamp)
            }]->(t)
        """,
            source_ref=event.source_ref,
            target_ref=event.target_ref,
            id=event.aggregate_id,
            type=event.connection_type,
            confidence=event.initial_confidence,
            method=event.discovery_method,
            correlation_id=event.correlation_id,
            timestamp=event.timestamp.isoformat()
        )

        # Invalidate centrality cache for affected vertices
        await self._invalidate_centrality_cache([event.source_ref, event.target_ref])

    async def _handle_CrossReferenceRefinedEvent(self, event: CrossReferenceRefinedEvent):
        """Update relationship properties on refinement."""
        set_clauses = ", ".join([
            f"r.{key} = ${key}" for key in event.new_values.keys()
        ])

        await self.neo4j.execute(f"""
            MATCH ()-[r:CROSS_REFERENCES {{id: $id}}]->()
            SET {set_clauses},
                r.updated_at = datetime(),
                r.version = coalesce(r.version, 0) + 1
        """, id=event.cross_ref_id, **event.new_values)

    async def _handle_TypologicalConnectionIdentifiedEvent(self, event: TypologicalConnectionIdentifiedEvent):
        """Project typological connection to graph."""
        await self.neo4j.execute("""
            MATCH (t:Verse {id: $type_ref})
            MATCH (a:Verse {id: $antitype_ref})
            CREATE (t)-[:TYPIFIES {
                pattern_id: $pattern,
                fractal_depth: $depth,
                composite_strength: $strength,
                dominant_layer: $layer,
                relation: $relation,
                correlation_id: $correlation_id,
                created_at: datetime()
            }]->(a)
        """,
            type_ref=event.type_ref,
            antitype_ref=event.antitype_ref,
            pattern=event.pattern_id,
            depth=event.fractal_depth,
            strength=event.composite_strength,
            layer=event.dominant_layer,
            relation=event.relation_type,
            correlation_id=event.correlation_id
        )

    async def _handle_ProphecyFulfillmentProvedEvent(self, event: ProphecyFulfillmentProvedEvent):
        """Project prophecy fulfillment to graph."""
        await self.neo4j.execute("""
            MERGE (p:Prophecy {id: $prophecy_id})
            ON CREATE SET
                p.name = $prophecy_name,
                p.category = $category,
                p.created_at = datetime()

            WITH p
            MATCH (ov:Verse {id: $prophecy_verse})
            MATCH (nv:Verse {id: $fulfillment_verse})

            CREATE (ov)-[:FULFILLED_IN {
                prophecy_id: $prophecy_id,
                fulfillment_type: $fulfillment_type,
                necessity_score: $necessity_score,
                bayesian_strength: $strength,
                lxx_support: $lxx_support,
                created_at: datetime()
            }]->(nv)
        """,
            prophecy_id=event.prophecy_id,
            prophecy_name=event.prophecy_name,
            category=event.category,
            prophecy_verse=event.prophecy_verse,
            fulfillment_verse=event.fulfillment_verse,
            fulfillment_type=event.fulfillment_type,
            necessity_score=event.necessity_score,
            strength=event.bayesian_strength,
            lxx_support=event.lxx_support
        )

    async def _invalidate_centrality_cache(self, verse_ids: List[str]):
        """Mark verses for centrality recalculation."""
        await self.neo4j.execute("""
            MATCH (v:Verse)
            WHERE v.id IN $verse_ids
            SET v.centrality_stale = true
        """, verse_ids=verse_ids)
```

---

## Part 7: Schema Migration

### Migration Script: `db/migrations/neo4j_graph_first.py`

**Phase 1: Create Constraints and Indexes**
```cypher
// Unique constraints - primary keys
CREATE CONSTRAINT verse_id IF NOT EXISTS FOR (v:Verse) REQUIRE v.id IS UNIQUE;
CREATE CONSTRAINT word_id IF NOT EXISTS FOR (w:Word) REQUIRE w.id IS UNIQUE;
CREATE CONSTRAINT lemma_id IF NOT EXISTS FOR (l:Lemma) REQUIRE l.id IS UNIQUE;
CREATE CONSTRAINT meaning_id IF NOT EXISTS FOR (m:Meaning) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT father_id IF NOT EXISTS FOR (f:Father) REQUIRE f.id IS UNIQUE;
CREATE CONSTRAINT work_id IF NOT EXISTS FOR (w:Work) REQUIRE w.id IS UNIQUE;
CREATE CONSTRAINT covenant_id IF NOT EXISTS FOR (c:Covenant) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT prophecy_id IF NOT EXISTS FOR (p:Prophecy) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT pattern_id IF NOT EXISTS FOR (tp:TypePattern) REQUIRE tp.id IS UNIQUE;

// Performance indexes - frequently queried fields
CREATE INDEX verse_book IF NOT EXISTS FOR (v:Verse) ON (v.book);
CREATE INDEX verse_testament IF NOT EXISTS FOR (v:Verse) ON (v.testament);
CREATE INDEX verse_chapter IF NOT EXISTS FOR (v:Verse) ON (v.book, v.chapter);
CREATE INDEX verse_centrality IF NOT EXISTS FOR (v:Verse) ON (v.centrality_score);
CREATE INDEX verse_community IF NOT EXISTS FOR (v:Verse) ON (v.community_id);
CREATE INDEX word_verse IF NOT EXISTS FOR (w:Word) ON (w.verse_id);
CREATE INDEX word_lemma IF NOT EXISTS FOR (w:Word) ON (w.lemma_id);
CREATE INDEX lemma_language IF NOT EXISTS FOR (l:Lemma) ON (l.language);
CREATE INDEX lemma_domain IF NOT EXISTS FOR (l:Lemma) ON (l.semantic_domain);
CREATE INDEX father_tradition IF NOT EXISTS FOR (f:Father) ON (f.tradition);
CREATE INDEX father_era IF NOT EXISTS FOR (f:Father) ON (f.era);

// Relationship property indexes for weighted queries
CREATE INDEX cross_ref_type FOR ()-[r:CROSS_REFERENCES]-() ON (r.connection_type);
CREATE INDEX cross_ref_confidence FOR ()-[r:CROSS_REFERENCES]-() ON (r.confidence);
CREATE INDEX typifies_pattern FOR ()-[r:TYPIFIES]-() ON (r.pattern_id);

// Full-text indexes for search
CREATE FULLTEXT INDEX verse_text IF NOT EXISTS
FOR (v:Verse) ON EACH [v.text_english, v.text_hebrew, v.text_greek];

CREATE FULLTEXT INDEX father_name IF NOT EXISTS
FOR (f:Father) ON EACH [f.name, f.name_greek];

CREATE FULLTEXT INDEX interpretation_text IF NOT EXISTS
FOR ()-[i:INTERPRETS]-() ON EACH [i.interpretation];
```

**Phase 2: Migrate Existing Data**
```python
async def migrate_verses_to_graph(
    postgres: PostgresClient,
    neo4j: Neo4jGraphClient,
    batch_size: int = 500
) -> int:
    """Migrate all verses from PostgreSQL to Neo4j."""
    migrated = 0
    batch = []

    async for verse in postgres.get_all_verses():
        batch.append({
            "id": verse.id,
            "book": verse.book,
            "book_full": verse.book_full,
            "chapter": verse.chapter,
            "verse_num": verse.verse,
            "testament": verse.testament,
            "hebrew": verse.text_hebrew,
            "greek": verse.text_greek,
            "english": verse.text_english,
            "word_count": verse.word_count,
            "created": verse.created_at.isoformat()
        })

        if len(batch) >= batch_size:
            await neo4j.execute("""
                UNWIND $batch AS v
                MERGE (verse:Verse {id: v.id})
                SET verse.book = v.book,
                    verse.book_full = v.book_full,
                    verse.chapter = v.chapter,
                    verse.verse = v.verse_num,
                    verse.testament = v.testament,
                    verse.text_hebrew = v.hebrew,
                    verse.text_greek = v.greek,
                    verse.text_english = v.english,
                    verse.word_count = v.word_count,
                    verse.created_at = datetime(v.created),
                    verse.centrality_score = 0.0,
                    verse.community_id = -1,
                    verse.processing_status = 'migrated'
            """, batch=batch)
            migrated += len(batch)
            batch = []

    if batch:
        await neo4j.execute("""...""", batch=batch)
        migrated += len(batch)

    return migrated

async def migrate_cross_references(
    postgres: PostgresClient,
    neo4j: Neo4jGraphClient,
    batch_size: int = 1000
) -> int:
    """Migrate cross-references to graph relationships."""
    migrated = 0
    batch = []

    async for ref in postgres.get_all_cross_references():
        batch.append({
            "source": ref.source_ref,
            "target": ref.target_ref,
            "id": str(ref.id),
            "type": ref.connection_type,
            "confidence": ref.confidence,
            "mutual": ref.mutual_influence_score,
            "necessity": ref.necessity_score,
            "validated": ref.validated,
            "method": ref.discovery_method or "migration"
        })

        if len(batch) >= batch_size:
            await neo4j.execute("""
                UNWIND $batch AS r
                MATCH (s:Verse {id: r.source})
                MATCH (t:Verse {id: r.target})
                CREATE (s)-[:CROSS_REFERENCES {
                    id: r.id,
                    connection_type: r.type,
                    confidence: r.confidence,
                    mutual_influence: r.mutual,
                    necessity_score: r.necessity,
                    validated: r.validated,
                    discovery_method: r.method,
                    migrated_at: datetime()
                }]->(t)
            """, batch=batch)
            migrated += len(batch)
            batch = []

    if batch:
        await neo4j.execute("""...""", batch=batch)
        migrated += len(batch)

    return migrated
```

---

## Part 8: Testing Specification

### Unit Tests: `tests/db/test_neo4j_graph_first.py`

**Test 1: `test_verse_creation_with_indexes`**
- Create verse node with all properties
- Verify all properties set correctly
- Check that indexes are being used (EXPLAIN query)

**Test 2: `test_cross_reference_creation_bidirectional`**
- Create two verses
- Create cross-reference relationship
- Verify forward query works
- Verify reverse query works (undirected matching)

**Test 3: `test_pagerank_calculation`**
- Create star network (hub verse with 10 connections)
- Create linear chain (5 verses in sequence)
- Run PageRank
- Verify hub verse has highest centrality
- Verify chain endpoints have lower centrality than middle

**Test 4: `test_community_detection_clusters`**
- Create two dense clusters with weak inter-cluster link
- Run Louvain
- Verify two distinct communities detected
- Verify community IDs written to nodes

**Test 5: `test_shortest_path_multi_hop`**
- Create path: A -> B -> C -> D -> E
- Query shortest path A to E
- Verify returns 4-hop path
- Verify intermediate nodes correct

**Test 6: `test_typological_chain_by_pattern`**
- Create chain: Gen22 -> Isa53 -> John1 -> Rev5 (sacrificial_lamb)
- Query chain by pattern
- Verify complete chain returned in order
- Verify chain confidence calculated correctly

**Test 7: `test_event_projection_creates_relationship`**
- Emit CrossReferenceDiscoveredEvent
- Verify graph updated with new relationship
- Check all properties projected correctly
- Verify correlation_id preserved

**Test 8: `test_patristic_consensus_grouping`**
- Add 3 Eastern + 2 Western Father interpretations
- Query consensus
- Verify tradition grouping correct
- Verify authority weights summed

---

## Part 9: Configuration

### Add to `config.py`

**New Configuration Dataclass**: `Neo4jGraphFirstConfig`

```python
@dataclass
class Neo4jGraphFirstConfig:
    """Configuration for graph-first Neo4j architecture."""

    # Connection settings
    uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    username: str = field(default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j"))
    password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    database: str = "biblos"

    # Connection pool
    max_connection_pool_size: int = 50
    connection_timeout_seconds: int = 30
    max_transaction_retry_time_seconds: int = 30

    # Graph Data Science
    enable_gds_algorithms: bool = True
    gds_tier: str = "enterprise"  # community, standard, enterprise

    # PageRank settings
    pagerank_damping: float = 0.85
    pagerank_iterations: int = 20
    pagerank_tolerance: float = 0.0001

    # Community detection
    community_resolution: float = 1.0
    community_max_levels: int = 10

    # Automated maintenance
    auto_calculate_centrality: bool = True
    centrality_recalc_interval_hours: int = 24
    auto_detect_communities: bool = True
    community_recalc_interval_hours: int = 48

    # Query optimization
    query_cache_size: int = 1000
    query_cache_ttl_seconds: int = 3600
    enable_query_logging: bool = False
    slow_query_threshold_ms: int = 1000

    # Projection management
    enable_event_projection: bool = True
    projection_batch_size: int = 100
    projection_flush_interval_seconds: int = 5
```

---

## Part 10: Performance Optimization

### Query Optimization Strategies

**1. Use Relationship Indexes**:
```cypher
// Create relationship property index for frequent queries
CREATE INDEX cross_ref_type FOR ()-[r:CROSS_REFERENCES]-() ON (r.connection_type);
CREATE INDEX cross_ref_confidence FOR ()-[r:CROSS_REFERENCES]-() ON (r.confidence);
CREATE INDEX typifies_strength FOR ()-[r:TYPIFIES]-() ON (r.composite_strength);
```

**2. Batch Operations with Transaction Management**:
```python
async def batch_create_relationships(
    self,
    relationships: List[Dict],
    batch_size: int = 500
) -> int:
    """Create many relationships efficiently with batching."""
    created = 0

    for i in range(0, len(relationships), batch_size):
        batch = relationships[i:i + batch_size]
        await self.execute("""
            UNWIND $rels AS rel
            MATCH (s:Verse {id: rel.source})
            MATCH (t:Verse {id: rel.target})
            CREATE (s)-[:CROSS_REFERENCES {
                id: rel.id,
                connection_type: rel.type,
                confidence: rel.confidence,
                created_at: datetime()
            }]->(t)
        """, rels=batch)
        created += len(batch)

    return created
```

**3. Named Graph Projection for Algorithms**:
```python
async def create_algorithm_projection(
    self,
    projection_name: str = "biblos-graph"
) -> None:
    """Create in-memory graph projection for faster algorithms."""

    # Drop existing projection if present
    await self.execute(f"""
        CALL gds.graph.exists('{projection_name}')
        YIELD exists
        CALL apoc.do.when(exists,
            "CALL gds.graph.drop('{projection_name}') YIELD graphName RETURN graphName",
            "RETURN null AS graphName",
            {{}}
        ) YIELD value
        RETURN value
    """)

    # Create fresh projection
    await self.execute(f"""
        CALL gds.graph.project(
            '{projection_name}',
            'Verse',
            {{
                CROSS_REFERENCES: {{
                    type: 'CROSS_REFERENCES',
                    properties: ['confidence'],
                    orientation: 'UNDIRECTED'
                }},
                TYPIFIES: {{
                    type: 'TYPIFIES',
                    properties: ['composite_strength'],
                    orientation: 'NATURAL'
                }}
            }}
        )
    """)
```

**4. Caching Hot Queries**:
```python
from functools import lru_cache
from datetime import datetime, timedelta

class CachedGraphClient:
    """Graph client with query result caching."""

    def __init__(self, client: Neo4jGraphClient, cache_ttl: int = 3600):
        self._client = client
        self._cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[Any, datetime]] = {}

    async def get_verse_web(self, verse_id: str) -> Dict:
        """Get verse web with caching."""
        cache_key = f"verse_web:{verse_id}"

        if cache_key in self._cache:
            result, cached_at = self._cache[cache_key]
            if datetime.utcnow() - cached_at < timedelta(seconds=self._cache_ttl):
                return result

        result = await self._client._query_verse_web(verse_id)
        self._cache[cache_key] = (result, datetime.utcnow())
        return result

    def invalidate_verse(self, verse_id: str):
        """Invalidate cache for a specific verse."""
        keys_to_remove = [k for k in self._cache if verse_id in k]
        for key in keys_to_remove:
            del self._cache[key]
```

---

## Part 11: Success Criteria

### Functional Requirements
- [ ] All 9 node types created with proper schemas
- [ ] All 7 core relationship types implemented
- [ ] Graph algorithms (PageRank, Louvain, HITS) working
- [ ] Path finding queries functional
- [ ] Event projection updating graph in real-time
- [ ] Migration from PostgreSQL complete

### Performance Requirements
- [ ] Single verse query: < 50ms
- [ ] Web query (verse + connections): < 200ms
- [ ] Shortest path (5 hops): < 500ms
- [ ] PageRank calculation (31,000 verses): < 5 minutes
- [ ] Community detection: < 10 minutes
- [ ] Batch insert (1000 relationships): < 10 seconds

### Data Integrity
- [ ] All verses from corpus present (~31,000)
- [ ] All cross-references migrated
- [ ] Relationship properties preserved
- [ ] No orphan nodes
- [ ] Constraint violations: 0

---

## Part 12: Detailed Implementation Order

1. **Update `db/neo4j_schema.py`** with complete node/relationship types and enums
2. **Create constraint/index migration script** with all indexes
3. **Implement `Neo4jGraphClient`** with all CRUD operations
4. **Create `db/neo4j_algorithms.py`** with GDS integrations (PageRank, Louvain, HITS)
5. **Implement query patterns** for common use cases
6. **Create event projection handler** (integrate with Session 08)
7. **Write migration script** from PostgreSQL with batching
8. **Run migration** for all existing data
9. **Calculate initial PageRank** and community detection
10. **Add configuration to `config.py`**
11. **Write unit tests** for all components
12. **Performance benchmark** critical queries
13. **Document query patterns** for API layer

---

## Part 13: Dependencies on Other Sessions

### Depends On
- SESSION 08: Event Sourcing (for projection integration and event consumption)

### Depended On By
- SESSION 10: Vector DB Enhancement (uses graph for context and neighbor retrieval)
- SESSION 11: Pipeline Integration (queries graph for enrichment data)

### External Dependencies
- Neo4j 5.x with Graph Data Science library installed
- APOC procedures for utility functions
- At least 8GB heap for algorithm projections

---

## Session Completion Checklist

```markdown
- [ ] `db/neo4j_schema.py` with complete schema enums and dataclasses
- [ ] `db/neo4j_algorithms.py` implemented with PageRank, Louvain, HITS
- [ ] `db/neo4j_graph_client.py` with all CRUD and query operations
- [ ] Constraint/index migration complete
- [ ] Data migration from PostgreSQL complete
- [ ] Event projection integrated (Session 08)
- [ ] PageRank calculation working
- [ ] Community detection working
- [ ] Path finding queries working
- [ ] Configuration added to config.py
- [ ] Unit tests passing
- [ ] Performance benchmarks met
- [ ] Query pattern documentation complete
```

**Next Session**: SESSION 10: Vector DB Enhancement

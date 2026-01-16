# SESSION 04: INTER-VERSE NECESSITY CALCULATOR

## Session Overview

**Objective**: Implement the `InterVerseNecessityCalculator` engine that determines whether verse B is NECESSARY to understand verse A (not just helpful). This is the second of the Five Impossible Oracles.

**Estimated Duration**: 1 Claude session (60-90 minutes of focused implementation)

**Prerequisites**:
- Session 01 complete (Mutual Transformation Metric)
- Session 03 complete (Omni-Contextual Resolver)
- Understanding of propositional logic (implications, necessities)
- Familiarity with dependency graph concepts

---

## Part 1: Understanding the Oracle Concept

### Core Capability
Determine if verse B is **logically necessary** to understand verse A:
1. A cannot be fully understood without B
2. B provides information that A presupposes
3. Removing B leaves A with semantic gaps

### Necessity vs. Helpfulness Distinction

**Helpful Connection** (Standard Cross-Reference):
- GEN.1.1 → JHN.1.1 - "Beginning" echoes, enriches understanding
- Understanding GEN.1.1 is possible without JHN.1.1

**Necessary Connection** (What This Engine Detects):
- HEB.11.17-19 → GEN.22 - Hebrews REQUIRES Genesis Isaac narrative
- "Abraham reasoned that God could raise the dead" makes NO SENSE without the Akedah
- This is a logical dependency, not just a thematic echo

### Canonical Example: HEB.11.17-19 ↔ GEN.22.1-14
```
HEB.11.17: "By faith Abraham, when he was tested, offered up Isaac"
HEB.11.18: "of whom it was said, 'In Isaac your seed shall be called'"
HEB.11.19: "concluding that God was able to raise him up, even from the dead"

Without GEN.22:
- Who is Abraham? Unknown
- What test? Unknown
- Why offer Isaac? Incomprehensible
- What seed promise? Missing
- Resurrection reasoning? Groundless

Necessity Score: 0.95 (near-absolute dependency)
```

### Necessity Types

1. **REFERENTIAL_NECESSITY**: Verse A explicitly refers to content in B
2. **PRESUPPOSITIONAL_NECESSITY**: Verse A presupposes facts established in B
3. **ARGUMENTATIVE_NECESSITY**: Verse A's argument depends on B's content
4. **DEFINITIONAL_NECESSITY**: Key terms in A are defined/explained in B
5. **NARRATIVE_NECESSITY**: Story in A requires prior narrative in B

---

## Part 2: File Creation Specification

### File: `ml/engines/necessity_calculator.py`

**Location**: `ml/engines/` (directory created in Session 03)

**Dependencies to Import**:
- `dataclasses` for result schemas
- `typing` for comprehensive type hints
- `logging` for analysis logging
- `numpy` for semantic calculations
- `networkx` for dependency graph operations
- Access to OmniContextualResolver (Session 03)
- Access to verse/reference database

**Classes to Define**:

#### 1. `NecessityType` (Enum)
```python
class NecessityType(Enum):
    REFERENTIAL = "referential"        # Explicit reference
    PRESUPPOSITIONAL = "presuppositional"  # Assumed knowledge
    ARGUMENTATIVE = "argumentative"    # Logical argument dependency
    DEFINITIONAL = "definitional"      # Term definition
    NARRATIVE = "narrative"            # Story sequence
    COVENANTAL = "covenantal"          # Covenant promise chain
    PROPHETIC = "prophetic"            # Prophecy-fulfillment link
```

#### 2. `NecessityStrength` (Enum)
```python
class NecessityStrength(Enum):
    ABSOLUTE = "absolute"      # >0.9: Cannot understand A without B
    STRONG = "strong"          # 0.7-0.9: Severe comprehension loss
    MODERATE = "moderate"      # 0.5-0.7: Significant context loss
    WEAK = "weak"              # 0.3-0.5: Helpful but not required
    NONE = "none"              # <0.3: No necessity relationship
```

#### 3. `SemanticGap` (Dataclass)
Fields:
- `gap_type: str` - Type of gap (entity, event, concept, term)
- `description: str` - What information is missing
- `source_text: str` - Text in A that creates the gap
- `resolution_source: str` - Verse B that fills the gap
- `severity: float` - How critical this gap is (0-1)

#### 4. `NecessityAnalysisResult` (Dataclass)
Fields:
- `source_verse: str` - Verse A (the dependent verse)
- `target_verse: str` - Verse B (the required verse)
- `necessity_score: float` - Overall necessity (0-1)
- `necessity_type: NecessityType` - Primary type of necessity
- `strength: NecessityStrength` - Classification
- `semantic_gaps: List[SemanticGap]` - All identified gaps
- `presuppositions: List[str]` - Presupposed facts
- `explicit_references: List[str]` - Direct textual references
- `dependency_chain: List[str]` - Full chain of dependencies
- `reasoning: str` - Human-readable explanation
- `bidirectional: bool` - Does B also need A?
- `confidence: float` - Confidence in analysis

#### 5. `DependencyGraph` (Dataclass)
Fields:
- `nodes: Dict[str, Dict]` - Verse nodes with metadata
- `edges: List[Tuple[str, str, float]]` - Directed edges with necessity scores
- `root_verses: List[str]` - Foundational verses (no dependencies)
- `terminal_verses: List[str]` - End verses (not depended upon)
- `cycles: List[List[str]]` - Circular dependencies (mutual necessity)

#### 6. `InterVerseNecessityCalculator` (Main Class)

**Constructor**:
- Accept OmniContextualResolver reference
- Accept verse database reference
- Accept configuration

**Methods**:

##### `async def calculate_necessity(self, verse_a, verse_b) -> NecessityAnalysisResult`
Main entry point:
1. Extract verse_a content and analyze structure
2. Identify all references/presuppositions in verse_a
3. Check if verse_b fulfills any identified needs
4. Calculate necessity score based on gap analysis
5. Classify necessity type and strength
6. Return complete result

##### `async def identify_semantic_gaps(self, verse_text, verse_id) -> List[SemanticGap]`
- Parse verse for undefined entities (names, places, events)
- Identify assumed knowledge markers ("as it is written", "remember")
- Detect incomplete references ("the prophet", "that day")
- Find technical terms requiring definition
- Return list of gaps with severity ratings

##### `async def find_gap_fillers(self, gaps: List[SemanticGap]) -> Dict[str, List[str]]`
- For each gap, search corpus for resolving verses
- Rank by resolution completeness
- Return mapping of gap → candidate resolver verses

##### `async def extract_presuppositions(self, verse_text) -> List[str]`
- Identify implicit assumptions in the verse
- Mark discourse markers indicating prior knowledge
- Extract references to prior events/statements
- Return list of presupposed facts

##### `async def detect_explicit_references(self, verse_text) -> List[str]`
- Find citation formulas ("it is written", "scripture says")
- Identify quotation patterns
- Match to source verses
- Return list of verse IDs referenced

##### `async def calculate_gap_severity(self, gap: SemanticGap, verse_context: str) -> float`
- Measure how critical this gap is for understanding
- Higher severity if gap affects main clause
- Lower if gap is in subordinate clause
- Return 0-1 severity score

##### `async def compute_necessity_score(self, gaps_filled: int, total_gaps: int, gap_severities: List[float]) -> float`
- Base score from gap coverage
- Weight by severity
- Adjust for necessity type
- Return composite score

##### `async def build_dependency_graph(self, verse_ids: List[str]) -> DependencyGraph`
- Calculate necessity between all pairs
- Build directed graph
- Identify strongly connected components
- Return complete graph structure

##### `async def find_necessity_chain(self, verse_a: str, verse_b: str) -> List[str]`
- Find shortest path of necessity from A to B
- May pass through intermediate verses
- Return ordered chain

---

## Part 3: Necessity Detection Algorithms

### Algorithm 1: Reference Extraction

**Pattern Matching for Citations**:
```python
CITATION_PATTERNS = [
    r"as it is written",
    r"scripture says",
    r"according to",
    r"the prophet\s+\w+\s+said",
    r"Moses\s+(?:said|wrote|commanded)",
    r"David\s+(?:said|wrote|spoke)",
    r"it is written in the (?:law|prophets)",
    r"καθὼς γέγραπται",  # Greek: "as it is written"
    r"כַּכָּתוּב",  # Hebrew: "as it is written"
]
```

**Named Entity Tracking**:
```python
async def track_entity_introduction(self, entity_name: str) -> str:
    """Find where an entity is first introduced in canon."""
    # Search for first occurrence with definitional context
    # Abraham first introduced in GEN.11.26
    # Moses first introduced in EXO.2.10
    first_occurrence = await self.find_first_mention(entity_name)
    return first_occurrence
```

### Algorithm 2: Presupposition Detection

**Presupposition Markers**:
```python
PRESUPPOSITION_MARKERS = {
    "definite_article": r"\bthe\s+(?!LORD)\w+",  # "the prophet" assumes known
    "possessive": r"\bhis\s+\w+|\bher\s+\w+",    # "his son" assumes known
    "temporal_back_reference": r"when\s+\w+\s+had",  # refers to past event
    "result_marker": r"therefore|thus|so\s+then",  # assumes prior reasoning
    "continuation": r"and\s+(?:he|she|they|it)\s+\w+",  # continues narrative
}
```

### Algorithm 3: Gap Severity Calculation

```python
def calculate_severity(self, gap: SemanticGap, syntax_tree) -> float:
    """
    Determine how critical a semantic gap is.
    """
    severity = 0.5  # Base severity

    # Higher if in main clause
    if gap.location in syntax_tree.main_clause:
        severity += 0.3

    # Higher if subject or object
    if gap.syntactic_role in ['subject', 'direct_object']:
        severity += 0.2

    # Lower if in conditional/hypothetical
    if gap.location in syntax_tree.conditional_clauses:
        severity -= 0.2

    # Higher if theological term
    if gap.gap_type == 'theological_concept':
        severity += 0.15

    return min(1.0, max(0.0, severity))
```

### Algorithm 4: Necessity Score Computation

```python
async def compute_final_score(
    self,
    gaps_filled: int,
    total_gaps: int,
    gap_severities: List[float],
    necessity_type: NecessityType
) -> float:
    """
    Compute overall necessity score using weighted combination.
    """
    if total_gaps == 0:
        return 0.0

    # Base coverage score
    coverage = gaps_filled / total_gaps

    # Weighted by severity
    if gap_severities:
        severity_weight = sum(gap_severities) / len(gap_severities)
    else:
        severity_weight = 0.5

    # Type multiplier
    type_multipliers = {
        NecessityType.REFERENTIAL: 1.0,
        NecessityType.PRESUPPOSITIONAL: 0.9,
        NecessityType.ARGUMENTATIVE: 0.95,
        NecessityType.DEFINITIONAL: 0.85,
        NecessityType.NARRATIVE: 0.8,
        NecessityType.COVENANTAL: 0.95,
        NecessityType.PROPHETIC: 0.9,
    }

    type_mult = type_multipliers.get(necessity_type, 0.8)

    # Final score
    score = coverage * severity_weight * type_mult
    return min(1.0, max(0.0, score))
```

---

## Part 4: Integration Points

### Integration 1: OmniContextualResolver

**Dependency**: Use resolved word meanings to detect semantic gaps

**Integration**:
```python
async def analyze_with_resolved_meanings(self, verse_id: str):
    # Get resolved meanings for polysemous words
    resolved = await self.omni_resolver.resolve_verse_meanings(verse_id)

    # Check if resolution required external verses
    for word_result in resolved:
        if word_result.required_context_verses:
            # These create necessity relationships
            for ctx_verse in word_result.required_context_verses:
                yield NecessityRelation(verse_id, ctx_verse, "definitional")
```

### Integration 2: Neo4j Graph Database

**Usage**: Store and query necessity relationships

**Schema Extension**:
```cypher
// Add NECESSITATES relationship type
CREATE (a:Verse)-[:NECESSITATES {
    score: 0.95,
    type: "presuppositional",
    gaps_filled: 5,
    primary_gap: "Abraham identity"
}]->(b:Verse)

// Query necessity chains
MATCH path = (start:Verse {id: 'HEB.11.17'})-[:NECESSITATES*]->(end:Verse)
RETURN path
ORDER BY length(path)
```

### Integration 3: Cross-Reference Pipeline

**Location**: `ml/inference/pipeline.py`

**Modification**:
1. After cross-reference discovery, check necessity for strong connections
2. Add necessity_score to CrossReferenceSchema
3. Flag relationships with necessity_score > 0.7 as "essential"

**Hook Point**:
```python
# In _classify_candidates() after scoring
if candidate.confidence > 0.8:
    necessity = await self.necessity_calculator.calculate_necessity(
        source_verse, candidate.target_verse
    )
    candidate.features["necessity_score"] = necessity.necessity_score
    candidate.features["necessity_type"] = necessity.necessity_type.value
```

### Integration 4: Data Schemas

**Add to `data/schemas.py`**:
```python
@dataclass
class CrossReferenceSchema:
    # ... existing fields ...

    # New necessity fields
    necessity_score: float = 0.0
    necessity_type: str = "none"
    semantic_gaps_filled: int = 0
    is_essential: bool = False  # True if necessity_score > 0.7
```

---

## Part 5: Testing Specification

### Unit Tests: `tests/ml/engines/test_necessity_calculator.py`

**Test 1: `test_hebrews_genesis_necessity`**
- Input: HEB.11.17-19 → GEN.22.1-14
- Expected: necessity_score > 0.9
- Type: PRESUPPOSITIONAL or NARRATIVE
- Strength: ABSOLUTE
- Gaps: Abraham identity, Isaac identity, test event, seed promise

**Test 2: `test_matthew_isaiah_quotation`**
- Input: MAT.1.23 → ISA.7.14
- Expected: necessity_score > 0.85
- Type: REFERENTIAL (explicit "as it is written")
- Strength: STRONG

**Test 3: `test_no_necessity_thematic`**
- Input: GEN.1.1 → PSA.19.1 (thematic, not necessary)
- Expected: necessity_score < 0.3
- Strength: NONE or WEAK
- Both verses comprehensible independently

**Test 4: `test_definitional_necessity`**
- Input: ROM.3.25 (propitiation) → LEV.16 (Day of Atonement)
- Expected: necessity_score > 0.7
- Type: DEFINITIONAL
- Gap: "propitiation" term requires sacrificial context

**Test 5: `test_narrative_necessity`**
- Input: 1KI.18.36 → GEN.12.1-3 (God of Abraham)
- Expected: necessity_score > 0.6
- Type: COVENANTAL
- Gap: "God of Abraham" requires covenant history

**Test 6: `test_gap_detection`**
- Input: HEB.11.17
- Expected gaps: Abraham (entity), Isaac (entity), testing (event), promise (concept)
- Each gap should have severity > 0.5

**Test 7: `test_dependency_graph_construction`**
- Build graph for Hebrews 11 faith heroes
- Verify directed edges to OT source passages
- Verify no cycles (one-way dependency)

**Test 8: `test_bidirectional_necessity`**
- Input: GEN.22 ↔ HEB.11.17
- Expected: HEB needs GEN (high), GEN does NOT need HEB (low)
- Verify asymmetric relationship

---

## Part 6: Configuration

### Add to `config.py`

**New Configuration Dataclass**: `NecessityCalculatorConfig`

Fields:
- `absolute_threshold: float = 0.9` - Score for ABSOLUTE classification
- `strong_threshold: float = 0.7` - Score for STRONG classification
- `moderate_threshold: float = 0.5` - Score for MODERATE classification
- `weak_threshold: float = 0.3` - Score for WEAK classification
- `gap_severity_weight: float = 0.4` - Weight of severity in final score
- `type_weight: float = 0.2` - Weight of necessity type in final score
- `coverage_weight: float = 0.4` - Weight of gap coverage in final score
- `max_chain_depth: int = 10` - Maximum necessity chain length
- `cache_necessity_results: bool = True` - Cache computed necessities
- `parallel_gap_detection: bool = True` - Parallelize gap detection

---

## Part 7: Caching Strategy

### Performance Optimization

**Level 1: Gap Analysis Cache**
- Key: `gaps:{verse_id}`
- Value: List of SemanticGap objects
- TTL: 1 week (verse content doesn't change)

**Level 2: Necessity Result Cache**
- Key: `necessity:{verse_a}:{verse_b}`
- Value: NecessityAnalysisResult
- TTL: 1 week

**Level 3: Dependency Graph Cache**
- Key: `dep_graph:{book_id}` or `dep_graph:{chapter_range}`
- Value: Serialized DependencyGraph
- TTL: 1 day

**Implementation in Redis**:
```python
async def cache_necessity(self, verse_a: str, verse_b: str, result: NecessityAnalysisResult):
    key = f"biblos:necessity:{verse_a}:{verse_b}"
    await self.redis.setex(key, 604800, result.to_json())

async def get_cached_necessity(self, verse_a: str, verse_b: str) -> Optional[NecessityAnalysisResult]:
    key = f"biblos:necessity:{verse_a}:{verse_b}"
    data = await self.redis.get(key)
    return NecessityAnalysisResult.from_json(data) if data else None
```

---

## Part 8: Plugins/Tools to Use

### During Implementation
- **sequential-thinking MCP**: Use for complex necessity logic design
- **memory MCP**: Store presupposition patterns and gap types
- **context7 MCP**: Reference networkx documentation for graph operations

### Corpus Access
- **Text-Fabric integration**: For Hebrew syntactic analysis (presupposition detection)
- **Macula integration**: For Greek discourse markers

### Testing Commands
```bash
# Run unit tests
pytest tests/ml/engines/test_necessity_calculator.py -v

# Run with theological test cases
pytest tests/ml/engines/test_necessity_calculator.py -k "hebrews or matthew" -v

# Performance testing
pytest tests/ml/engines/test_necessity_calculator.py -k "performance" --benchmark

# Graph construction tests
pytest tests/ml/engines/test_necessity_calculator.py -k "graph" -v
```

---

## Part 9: Success Criteria

### Functional Requirements
- [ ] Correctly identifies explicit quotations (citation formulas)
- [ ] Detects presuppositional relationships
- [ ] Calculates gap severity accurately
- [ ] Produces valid necessity scores
- [ ] Builds correct dependency graphs
- [ ] Distinguishes necessary from merely helpful connections

### Theological Accuracy
- [ ] HEB.11.17-19 → GEN.22: ABSOLUTE necessity
- [ ] MAT.1.23 → ISA.7.14: STRONG necessity (quotation)
- [ ] ROM.3.25 → LEV.16: MODERATE-STRONG (definitional)
- [ ] Thematic-only connections: WEAK or NONE

### Performance Requirements
- [ ] Single pair calculation: < 500ms (cached)
- [ ] First-time calculation: < 3 seconds
- [ ] Graph construction (100 verses): < 30 seconds
- [ ] Batch processing: > 50 pairs/second

---

## Part 10: Detailed Implementation Order

1. **Create enums**: `NecessityType`, `NecessityStrength`
2. **Create dataclasses**: `SemanticGap`, `NecessityAnalysisResult`, `DependencyGraph`
3. **Implement `identify_semantic_gaps()`** - core gap detection
4. **Implement `extract_presuppositions()`** - presupposition markers
5. **Implement `detect_explicit_references()`** - citation pattern matching
6. **Implement `calculate_gap_severity()`** - severity scoring
7. **Implement `find_gap_fillers()`** - resolve gaps to source verses
8. **Implement `compute_necessity_score()`** - combine metrics
9. **Implement main `calculate_necessity()`** - orchestrate analysis
10. **Implement `build_dependency_graph()`** - graph construction
11. **Add caching layer** in Redis client
12. **Add configuration to `config.py`**
13. **Integrate with pipeline** for automatic necessity scoring
14. **Write and run unit tests**
15. **Perform theological validation** with canonical test cases

---

## Part 11: Dependencies on Other Sessions

### Depends On
- SESSION 01: Mutual Transformation Metric (for semantic shift analysis)
- SESSION 03: Omni-Contextual Resolver (for word meaning resolution)

### Depended On By
- SESSION 06: Fractal Typology Engine (uses necessity for layer connections)
- SESSION 07: Prophetic Necessity Prover (builds on necessity calculations)
- SESSION 11: Pipeline Integration (orchestrates necessity scoring)

### External Dependencies
- NetworkX for graph algorithms
- Verse database with full text access
- Syntactic analysis data for presupposition detection

---

## Session Completion Checklist

```markdown
- [ ] `ml/engines/necessity_calculator.py` implemented
- [ ] All enums and dataclasses defined
- [ ] Gap detection working for entities, events, concepts
- [ ] Presupposition extraction functional
- [ ] Citation pattern matching accurate
- [ ] Necessity scoring produces valid results
- [ ] Dependency graph construction working
- [ ] Caching layer integrated
- [ ] Configuration added to config.py
- [ ] HEB.11.17 → GEN.22 test passing (ABSOLUTE)
- [ ] MAT.1.23 → ISA.7.14 test passing (STRONG)
- [ ] Thematic-only connections correctly classified (WEAK/NONE)
- [ ] Performance tests passing
- [ ] Documentation complete
```

**Next Session**: SESSION 05: LXX Christological Extractor

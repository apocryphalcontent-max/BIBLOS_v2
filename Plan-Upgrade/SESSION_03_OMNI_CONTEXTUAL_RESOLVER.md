# SESSION 03: OMNI-CONTEXTUAL RESOLVER ENGINE

## Session Overview

**Objective**: Implement the `OmniContextualResolver` engine that determines absolute word meaning via eliminative reasoning across all biblical occurrences. This is the first of the Five Impossible Oracles.

**Estimated Duration**: 1 Claude session (60-90 minutes of focused implementation)

**Prerequisites**:
- Access to Greek/Hebrew corpus data (macula-hebrew, macula-greek, BHSA, other configured earliest transcripts)
- Understanding of lemma-based word searching
- Familiarity with semantic range analysis

---

## Part 1: Understanding the Oracle Concept

### Core Capability
Given any word in any verse, determine its ABSOLUTE meaning by:
1. Finding ALL occurrences of that word across the entire canon
2. Extracting the semantic range from all contexts
3. Eliminating impossible meanings for THIS specific verse
4. Concluding with the only possible meaning(s)

### Canonical Example: רוּחַ (ruach) in GEN.1.2
- Occurs 389 times in OT
- Semantic range: wind, breath, spirit, Spirit (divine)
- In GEN.1.2 context:
  - "wind" eliminated: no physical source described
  - "breath" eliminated: "merachefet" (hovering) requires agency, not passive air
  - Conclusion: "Divine Spirit" (Third Person of Trinity)
- Confidence: Very high due to eliminative reasoning

### Why This Is "Impossible"
No human scholar can reasonably hold all 389 occurrences in mind simultaneously. The system can, and can perform rigorous eliminative logic across all of them. This is superhuman contextual analysis.

---

## Part 2: File Creation Specification

### File: `ml/engines/omnicontext_resolver.py`

**Location**: Create new directory `ml/engines/` if it doesn't exist

**Dependencies to Import**:
- `dataclasses` for result schemas
- `typing` for comprehensive type hints
- `logging` for analysis logging
- `numpy` for semantic field vectors
- Access to corpus clients (Text-Fabric, Macula integration)
- Access to embedding model for semantic similarity

**Classes to Define**:

#### 1. `EliminationReason` (Enum)
- `GRAMMATICAL_INCOMPATIBILITY` - Grammar rules out this meaning
- `CONTEXTUAL_IMPOSSIBILITY` - Context excludes this meaning
- `SEMANTIC_CONTRADICTION` - Semantic field conflict
- `THEOLOGICAL_IMPOSSIBILITY` - Orthodox theology excludes this
- `SYNTACTIC_CONSTRAINT` - Syntax rules out this meaning
- `COLLOCATIONAL_VIOLATION` - Word associations exclude this

#### 2. `EliminationStep` (Dataclass)
Fields:
- `meaning: str` - The meaning being evaluated
- `eliminated: bool` - Was it eliminated?
- `reason: EliminationReason` - Why eliminated (if eliminated)
- `explanation: str` - Human-readable explanation
- `evidence_verses: List[str]` - Verses supporting this elimination
- `confidence: float` - Confidence in this elimination step

#### 3. `SemanticFieldEntry` (Dataclass)
Fields:
- `lemma: str` - The word lemma
- `meaning: str` - One possible meaning
- `occurrence_count: int` - How many times in this meaning
- `primary_contexts: List[str]` - Sample verse IDs
- `semantic_neighbors: List[str]` - Related lemmas in this meaning
- `theological_weight: float` - Importance in theological discourse

#### 4. `AbsoluteMeaningResult` (Dataclass)
Fields:
- `word: str` - The word analyzed
- `verse_id: str` - The verse context
- `primary_meaning: str` - Determined absolute meaning
- `confidence: float` - Overall confidence (0-1)
- `reasoning_chain: List[EliminationStep]` - All elimination steps
- `eliminated_alternatives: Dict[str, str]` - Meaning → reason eliminated
- `remaining_candidates: List[str]` - Meanings not eliminated (usually 1)
- `semantic_field_map: Dict[str, SemanticFieldEntry]` - All meanings mapped
- `total_occurrences: int` - How many times word appears in canon
- `analysis_coverage: float` - % of occurrences analyzed

#### 5. `OmniContextualResolver` (Main Class)

**Constructor**:
- Accept corpus client reference
- Accept embedding model reference
- Accept configuration

**Class Attributes**:
```python
# Hebrew words with significant semantic range
POLYSEMOUS_HEBREW = {
    "רוּחַ": ["wind", "breath", "spirit", "Spirit"],
    "נֶפֶשׁ": ["soul", "life", "person", "throat", "desire"],
    "לֵב": ["heart", "mind", "understanding", "will"],
    "בָּרָא": ["create", "make", "shape"],
    "כָּבוֹד": ["glory", "weight", "honor", "wealth"],
    # ... extensive list
}

# Greek words with significant semantic range
POLYSEMOUS_GREEK = {
    "λόγος": ["word", "speech", "reason", "account", "Word"],
    "πνεῦμα": ["spirit", "Spirit", "wind", "breath"],
    "σάρξ": ["flesh", "body", "human nature", "sinful nature"],
    "ψυχή": ["soul", "life", "self", "person"],
    # ... extensive list
}
```

**Methods**:

##### `async def resolve_absolute_meaning(self, word, verse_id, language) -> AbsoluteMeaningResult`
Main entry point:
1. Get all occurrences of this word
2. Extract semantic range from all contexts
3. Get current verse context
4. Eliminate impossible meanings systematically
5. Rank remaining by parallel support
6. Map semantic resonances
7. Return complete result

##### `async def get_all_occurrences(self, word, language) -> List[Dict]`
- Query corpus for all instances of this lemma
- Return list with verse_id, context, morphology

##### `async def extract_semantic_range(self, occurrences) -> List[SemanticFieldEntry]`
- Cluster occurrences by meaning
- Use embedding similarity to group
- Return semantic field with all meanings

##### `async def get_verse_context(self, verse_id) -> Dict`
- Get full context for this verse
- Include surrounding verses
- Include syntactic analysis
- Include morphological analysis

##### `async def check_contextual_compatibility(self, meaning, verse_context, grammatical_constraints) -> CompatibilityResult`
- Test if this meaning can work in this context
- Check grammatical agreement
- Check semantic coherence
- Check collocational patterns
- Return compatible: bool, impossibility_reason: Optional[str]

##### `async def rank_by_parallel_support(self, remaining_meanings, verse_id) -> List[Tuple[str, float]]`
- For each remaining meaning, find supporting parallels
- Calculate support score based on:
  - Number of exact parallel constructions
  - Theological significance of parallels
  - Patristic usage patterns
- Return ranked list

##### `async def map_semantic_field(self, word, primary_meaning) -> Dict[str, SemanticFieldEntry]`
- Build complete map of all meanings
- Calculate theological weight for each
- Identify resonances across canon

##### `def parse_grammatical_constraints(self, verse_id) -> Dict`
- Extract from morphological analysis:
  - Part of speech requirements
  - Gender/number agreement
  - Case requirements (for Greek)
  - State requirements (for Hebrew)
  - Verb form constraints

---

## Part 3: Elimination Logic System

### Grammatical Elimination

**Rule**: If morphological analysis constrains meaning, eliminate incompatible readings

**Example**: Hebrew verb form may exclude nominal meanings

**Implementation**:
```python
def eliminate_by_grammar(self, meaning, morph_analysis) -> Optional[EliminationStep]:
    if meaning.part_of_speech != morph_analysis.required_pos:
        return EliminationStep(
            meaning=meaning,
            eliminated=True,
            reason=EliminationReason.GRAMMATICAL_INCOMPATIBILITY,
            explanation=f"Meaning requires {meaning.part_of_speech}, but grammar requires {morph_analysis.required_pos}",
            confidence=0.95
        )
    return None
```

### Contextual Elimination

**Rule**: If surrounding context excludes a meaning, eliminate it

**Example**: "ruach" as "wind" requires physical source; GEN.1.2 has none

**Implementation**:
```python
async def eliminate_by_context(self, meaning, context) -> Optional[EliminationStep]:
    contextual_requirements = self.meaning_requirements.get(meaning, [])
    for requirement in contextual_requirements:
        if not self.check_context_has(context, requirement):
            return EliminationStep(
                meaning=meaning,
                eliminated=True,
                reason=EliminationReason.CONTEXTUAL_IMPOSSIBILITY,
                explanation=f"Context lacks required element: {requirement}",
                confidence=0.85
            )
    return None
```

### Semantic Elimination

**Rule**: If semantic field creates contradictions, eliminate

**Example**: "death" meaning in context affirming "life"

**Implementation**:
```python
async def eliminate_by_semantic_field(self, meaning, context) -> Optional[EliminationStep]:
    meaning_field = self.get_semantic_field(meaning)
    context_field = self.extract_semantic_field(context)
    if self.fields_contradict(meaning_field, context_field):
        return EliminationStep(
            meaning=meaning,
            eliminated=True,
            reason=EliminationReason.SEMANTIC_CONTRADICTION,
            explanation=f"Semantic field of '{meaning}' contradicts context field",
            confidence=0.80
        )
    return None
```

### Theological Elimination

**Rule**: If meaning contradicts Orthodox theology, eliminate (with care)

**Example**: Modalist reading of πνεῦμα in Trinitarian context

**Implementation**:
```python
async def eliminate_by_theology(self, meaning, context) -> Optional[EliminationStep]:
    if self.is_trinitarian_context(context):
        if meaning in self.MODALIST_READINGS:
            return EliminationStep(
                meaning=meaning,
                eliminated=True,
                reason=EliminationReason.THEOLOGICAL_IMPOSSIBILITY,
                explanation="Reading incompatible with Trinitarian theology established by context",
                confidence=0.90
            )
    return None
```

---

## Part 4: Integration Points

### Integration 1: Corpus Access

**Required Interfaces**:
- `integrations/text_fabric.py` - Access to BHSA Hebrew corpus
- `integrations/macula.py` - Access to Macula Greek/Hebrew data
- `db/postgres_client.py` - Cached occurrence data

**Data Requirements**:
- Complete lemma index for Hebrew OT (8,679 unique lemmas)
- Complete lemma index for Greek NT (5,461 unique lemmas)
- Occurrence counts per lemma
- Morphological coding per occurrence

### Integration 2: Embedding Model

**Required Interface**:
- `ml/embeddings/sentence_embeddings.py` - For semantic similarity
- Need word-level embeddings OR sentence embeddings with word extraction

**Usage**:
- Cluster occurrences by semantic similarity
- Detect semantic field boundaries
- Measure contextual fit

### Integration 3: Pipeline Integration

**Location**: Can be invoked by agents or pipeline phases

**Invocation Points**:
- `agents/linguistic/semantikos.py` - Semantic analysis agent
- `agents/theological/theologos.py` - Theological analysis agent
- `pipeline/phases/linguistic.py` - Linguistic phase

**Data Flow**:
```
verse_processing → word_extraction → omni_resolver.resolve() → semantic_enrichment
```

---

## Part 5: Testing Specification

### Unit Tests: `tests/ml/engines/test_omnicontext_resolver.py`

**Test 1: `test_ruach_genesis_1_2`**
- Input: "רוּחַ", GEN.1.2
- Expected: Primary meaning = "Spirit" (Divine)
- Eliminated: "wind" (no source), "breath" (requires subject)
- Confidence: > 0.85

**Test 2: `test_logos_john_1_1`**
- Input: "λόγος", JHN.1.1
- Expected: Primary meaning = "Word" (Divine Person)
- Eliminated: "word" (lowercase, mere speech)
- Evidence: Unique theological context

**Test 3: `test_nephesh_basic`**
- Input: "נֶפֶשׁ", GEN.2.7
- Expected: Primary meaning = "living being" or "soul"
- Semantic range correctly mapped

**Test 4: `test_elimination_chain`**
- Verify elimination steps are recorded correctly
- Verify evidence verses are cited
- Verify confidence decreases with fewer eliminations

**Test 5: `test_single_meaning_word`**
- Input: Word with only one meaning
- Expected: No elimination needed, high confidence

**Test 6: `test_semantic_field_mapping`**
- Verify complete semantic field returned
- Verify theological weights calculated
- Verify resonances identified

**Test 7: `test_performance_large_occurrence_count`**
- Input: Common word (e.g., "and" - thousands of occurrences)
- Expected: Complete within 5 seconds
- Verify sampling strategy works for very common words

---

## Part 6: Configuration

### Add to `config.py`

**New Configuration Dataclass**: `OmniContextualConfig`

Fields:
- `max_occurrences_full_analysis: int = 500` # Above this, use sampling
- `sample_size_large_words: int = 200` # Sample size for common words
- `elimination_confidence_threshold: float = 0.7` # Min confidence to eliminate
- `semantic_similarity_threshold: float = 0.8` # For clustering meanings
- `parallel_support_weight: float = 0.3` # Weight in final scoring
- `theological_weight_multiplier: float = 1.2` # Boost for theological terms
- `cache_semantic_ranges: bool = True` # Cache analyzed semantic ranges
- `cache_ttl_hours: int = 168` # One week cache TTL

---

## Part 7: Caching Strategy

### Performance Optimization

Given that words are analyzed repeatedly, implement multi-level caching:

**Level 1: Occurrence Cache**
- Key: `lemma:{language}`
- Value: List of all occurrences
- TTL: 1 week (corpus doesn't change)

**Level 2: Semantic Range Cache**
- Key: `semantic_range:{lemma}:{language}`
- Value: Clustered meanings with counts
- TTL: 1 week

**Level 3: Resolution Cache**
- Key: `resolution:{word}:{verse_id}`
- Value: AbsoluteMeaningResult
- TTL: 1 day (may be refined by learning)

**Implementation Location**: `db/redis_client.py`

```python
async def cache_semantic_range(self, lemma, language, range_data):
    key = f"biblos:semantic_range:{lemma}:{language}"
    await self.redis.setex(key, 604800, json.dumps(range_data))

async def get_cached_semantic_range(self, lemma, language) -> Optional[Dict]:
    key = f"biblos:semantic_range:{lemma}:{language}"
    data = await self.redis.get(key)
    return json.loads(data) if data else None
```

---

## Part 8: Plugins/Tools to Use

### During Implementation
- **sequential-thinking MCP**: Use for elimination logic design
- **memory MCP**: Store polysemous word lists and semantic ranges
- **context7 MCP**: Reference for numpy clustering, cosine similarity

### Corpus Access
- **Text-Fabric integration**: For BHSA Hebrew data
- **Macula integration**: For Greek/Hebrew morphology

### Testing Commands
```bash
# Run unit tests
pytest tests/ml/engines/test_omnicontext_resolver.py -v

# Run with specific theological test cases
pytest tests/ml/engines/test_omnicontext_resolver.py -k "ruach or logos" -v

# Performance testing
pytest tests/ml/engines/test_omnicontext_resolver.py -k "performance" --benchmark
```

---

## Part 9: Success Criteria

### Functional Requirements
- [ ] Retrieves all occurrences for any lemma
- [ ] Correctly clusters into semantic meanings
- [ ] Elimination logic produces valid reasoning chains
- [ ] Final meaning determination is accurate for test cases
- [ ] Caching improves repeat query performance

### Theological Accuracy
- [ ] רוּחַ in GEN.1.2 resolves to "Spirit"
- [ ] λόγος in JHN.1.1 resolves to "Word" (divine)
- [ ] Elimination reasoning matches patristic interpretation

### Performance Requirements
- [ ] Single word resolution: < 2 seconds (cached)
- [ ] First-time resolution: < 10 seconds
- [ ] Common word (>500 occurrences): < 5 seconds with sampling

---

## Part 10: Detailed Implementation Order

1. **Create directory structure**: `mkdir -p ml/engines`
2. **Create `__init__.py`** with exports
3. **Implement enums and dataclasses** first
4. **Implement `get_all_occurrences`** - depends on corpus integration
5. **Implement `extract_semantic_range`** - clustering logic
6. **Implement individual elimination methods**:
   - `eliminate_by_grammar`
   - `eliminate_by_context`
   - `eliminate_by_semantic_field`
   - `eliminate_by_theology`
7. **Implement `check_contextual_compatibility`** - combines eliminations
8. **Implement `rank_by_parallel_support`** - final ranking
9. **Implement `map_semantic_field`** - complete mapping
10. **Implement main `resolve_absolute_meaning`** - orchestrates everything
11. **Add caching layer** in Redis client
12. **Add configuration to `config.py`**
13. **Write and run unit tests**
14. **Perform theological validation** with known test cases

---

## Part 11: Dependencies on Other Sessions

### Depends On
- None (this is an independent oracle engine)

### Depended On By
- SESSION 04: Inter-Verse Necessity Calculator (uses resolved meanings)
- SESSION 06: Fractal Typology Engine (uses semantic analysis)
- SESSION 11: Pipeline Integration (orchestrates oracle engines)

### External Dependencies
- Corpus data must be accessible via Text-Fabric or Macula
- Embedding model must be available for semantic clustering

---

## Session Completion Checklist

```markdown
- [ ] `ml/engines/__init__.py` created
- [ ] `ml/engines/omnicontext_resolver.py` implemented
- [ ] All dataclasses and enums defined
- [ ] Elimination logic working for all 4 types
- [ ] Semantic range extraction functional
- [ ] Parallel support ranking implemented
- [ ] Caching layer integrated
- [ ] Configuration added to config.py
- [ ] רוּחַ GEN.1.2 test passing
- [ ] λόγος JHN.1.1 test passing
- [ ] Performance tests passing
- [ ] Documentation complete
```

**Next Session**: SESSION 04: Inter-Verse Necessity Calculator

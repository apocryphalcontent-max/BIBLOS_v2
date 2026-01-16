# SESSION 01: MUTUAL TRANSFORMATION METRIC IMPLEMENTATION

## Session Overview

**Objective**: Implement the `MutualTransformationMetric` class that measures bidirectional semantic shift between connected verses. This is the foundational metric that powers the "verses redefine each other" capability.

**Estimated Duration**: 1 Claude session (45-60 minutes of focused implementation)

**Prerequisites**:
- Familiarity with existing `ml/models/gnn_discovery.py` (GATv2Conv architecture)
- Understanding of `ml/inference/pipeline.py` structure
- Access to embedding generation utilities

---

## Part 1: Understanding the Theological Principle

### Core Concept
When two verses are genuinely connected, their meanings mutually transform. The Burning Bush (Exodus 3:2) and the Theotokos Mary both gain deeper significance when understood in relation to each other:
- The bush becomes a type of Mary (contains divine fire without being consumed)
- Mary's virginal motherhood is illuminated as the antitype fulfillment

### Mathematical Representation
Given two verses A and B with embeddings before and after GNN refinement:
- `source_shift = 1 - cosine_similarity(A_before, A_after)`
- `target_shift = 1 - cosine_similarity(B_before, B_after)`
- `mutual_influence = 2 * (source_shift * target_shift) / (source_shift + target_shift + ε)`

The harmonic mean ensures BOTH verses must shift for high mutual influence. One-sided influence scores low.

---

## Part 2: File Creation Specification

### File: `ml/metrics/mutual_transformation.py`

**Location**: Create new directory `ml/metrics/` if it doesn't exist

**Dependencies to Import**:
- `numpy` for vector operations
- `scipy.spatial.distance` for cosine distance
- `dataclasses` for result schema
- `enum` for transformation type classification
- `typing` for type hints

**Classes to Define**:

#### 1. `TransformationType` (Enum)
- `RADICAL` - mutual_influence > 0.4 (e.g., Isaac → Christ)
- `MODERATE` - mutual_influence between 0.2-0.4 (e.g., Temple → Church)
- `MINIMAL` - mutual_influence < 0.2 (e.g., geographic location match)

#### 2. `MutualTransformationScore` (Dataclass)
Fields:
- `source_shift: float` - Cosine distance for source verse (0-1)
- `target_shift: float` - Cosine distance for target verse (0-1)
- `mutual_influence: float` - Harmonic mean (0-1)
- `transformation_type: TransformationType` - Classification
- `source_delta_vector: np.ndarray` - Actual embedding change vector
- `target_delta_vector: np.ndarray` - Actual embedding change vector
- `directionality: float` - Asymmetry measure (which verse influenced more)

#### 3. `MutualTransformationMetric` (Main Class)

**Methods**:

##### `async def measure_transformation(self, source_verse, target_verse, source_before, source_after, target_before, target_after) -> MutualTransformationScore`
- Calculate cosine distances
- Compute harmonic mean with epsilon protection (1e-10)
- Classify transformation type based on thresholds
- Return complete score object

##### `def calculate_directionality(self, source_shift, target_shift) -> float`
- Returns -1 to 1 where:
  - -1 = source completely dominated
  - 0 = perfectly mutual
  - 1 = target completely dominated
- Formula: `(target_shift - source_shift) / (target_shift + source_shift + ε)`

##### `def extract_semantic_components(self, delta_vector, vocabulary_index) -> Dict[str, float]`
- Decompose the delta vector into interpretable semantic dimensions
- Map to theological categories (christological, soteriological, eschatological, etc.)
- Return dictionary of component contributions

##### `async def measure_batch(self, pairs: List[Tuple]) -> List[MutualTransformationScore]`
- Batch processing for efficiency
- Vectorized cosine similarity calculations
- Returns list of scores in same order as input pairs

---

## Part 3: Integration Points

### Integration 1: `ml/inference/pipeline.py`

**Location**: Within `_classify_candidates()` method (approximately line 768-793)

**Modification Strategy**:
1. BEFORE calling `_refine_with_gnn()`, capture baseline embeddings for all candidates
2. Store in dictionary keyed by verse_id
3. AFTER GNN refinement, extract updated embeddings from GNN model
4. Invoke `MutualTransformationMetric.measure_transformation()` for each candidate
5. Store results in `candidate.features["mutual_influence"]` and related fields

**Specific Hook Point**:
- Look for the existing `if self.config.use_gnn_refinement:` block
- Insert embedding capture logic before the GNN call
- Insert measurement logic after the GNN call returns

**Data Flow**:
```
candidates → capture_embeddings_before → gnn_refine → capture_embeddings_after → measure_transformation → update_candidate_features
```

### Integration 2: `data/schemas.py`

**Location**: `CrossReferenceSchema` class (already exists with mutual_influence fields)

**Verification**: Confirm these fields exist (they should from original plan):
- `mutual_influence_score: float = 0.0`
- `source_semantic_shift: float = 0.0`
- `target_semantic_shift: float = 0.0`
- `transformation_type: str = "MINIMAL"`

If not present, add them to the schema.

### Integration 3: `ml/inference/postprocessor.py`

**Location**: In `_apply_final_scoring()` method

**Modification**:
- Add mutual_influence as a factor in final confidence calculation
- High mutual_influence (>0.4) should boost confidence by 10-15%
- Zero mutual_influence should not penalize (neutral factor)

---

## Part 4: GNN Embedding Extraction

### Understanding the Existing GNN Architecture

The `CrossRefGNN` class in `gnn_discovery.py` uses GATv2Conv layers. To extract embeddings:

**Current Forward Method Structure**:
```
forward(x, edge_index) → hidden representations through GAT layers
```

**Required Modification/Access Point**:
- Need method to get node embedding by verse_id after a forward pass
- May need to cache final hidden states in a dictionary
- Access via `gnn_model.get_node_embedding(verse_id)`

**Implementation Approach**:
1. Add `self._node_embeddings: Dict[str, np.ndarray] = {}` to GNN class
2. Populate during forward pass
3. Add `get_node_embedding(verse_id) -> np.ndarray` accessor method
4. Alternative: Extract from the node features tensor using verse-to-index mapping

---

## Part 5: Testing Specification

### Unit Tests to Create: `tests/ml/metrics/test_mutual_transformation.py`

**Test 1: `test_identical_embeddings_zero_shift`**
- Input: Same embedding for before/after
- Expected: source_shift = 0, target_shift = 0, mutual_influence = 0
- Type: MINIMAL

**Test 2: `test_orthogonal_embeddings_max_shift`**
- Input: Orthogonal embeddings (cosine = 0)
- Expected: source_shift = 1, mutual_influence high
- Type: RADICAL

**Test 3: `test_asymmetric_shift`**
- Input: Large source shift, small target shift
- Expected: Low mutual_influence (harmonic mean penalizes asymmetry)
- Directionality should reflect the asymmetry

**Test 4: `test_classification_thresholds`**
- Verify RADICAL > 0.4, MODERATE 0.2-0.4, MINIMAL < 0.2

**Test 5: `test_batch_processing`**
- Process 100 pairs in batch
- Verify results match individual processing
- Measure performance improvement

**Test 6: `test_theological_test_case_isaac_christ`**
- Use real embeddings for GEN.22.2 (Isaac) and JHN.3.16 (Christ)
- Verify high mutual_influence (both verses should shift significantly)
- This is the canonical test case for the feature

---

## Part 6: Configuration

### Add to `config.py`

**New Configuration Dataclass**: `MutualTransformationConfig`

Fields:
- `radical_threshold: float = 0.4`
- `moderate_threshold: float = 0.2`
- `directionality_weight: float = 0.1` (for scoring adjustment)
- `enable_semantic_decomposition: bool = True`
- `cache_embeddings: bool = True`

**Integration**: Add to main `BiblosConfig` as `mutual_transformation: MutualTransformationConfig`

---

## Part 7: Plugins/Tools to Use

### During Implementation
- **sequential-thinking MCP**: Use for complex logic design decisions
- **memory MCP**: Store decisions about threshold values and rationale
- **context7 MCP**: Look up numpy/scipy documentation for cosine similarity

### Testing Commands
```bash
# Run unit tests
pytest tests/ml/metrics/test_mutual_transformation.py -v

# Run with coverage
pytest tests/ml/metrics/test_mutual_transformation.py --cov=ml.metrics.mutual_transformation

# Performance benchmark
pytest tests/ml/metrics/test_mutual_transformation.py -k "performance" --benchmark
```

---

## Part 8: Success Criteria

### Functional Requirements
- [ ] `MutualTransformationMetric` class created and functional
- [ ] Cosine distance calculations mathematically correct
- [ ] Harmonic mean produces expected values for test cases
- [ ] Classification thresholds correctly applied
- [ ] Integration with pipeline captures embeddings before/after GNN

### Performance Requirements
- [ ] Single pair measurement: < 10ms
- [ ] Batch of 100 pairs: < 500ms
- [ ] Memory overhead: < 50MB for embedding cache

### Theological Validation
- [ ] Isaac/Christ test case produces RADICAL classification
- [ ] Burning Bush/Theotokos test case produces MODERATE or RADICAL
- [ ] Geographic-only connections produce MINIMAL

---

## Part 9: Detailed Implementation Order

1. **Create directory structure**: `mkdir -p ml/metrics`
2. **Create `__init__.py`** in `ml/metrics/` with appropriate exports
3. **Implement `MutualTransformationScore` dataclass** first (simple, no dependencies)
4. **Implement `TransformationType` enum**
5. **Implement core `measure_transformation()` method** with hardcoded thresholds
6. **Add configuration integration** to make thresholds configurable
7. **Implement batch processing** for efficiency
8. **Add GNN embedding accessor** method to `CrossRefGNN` class
9. **Integrate with `_classify_candidates()`** in pipeline
10. **Write and run unit tests**
11. **Run integration test** with sample verse pairs
12. **Document** with docstrings and type hints

---

## Part 10: Dependencies on Other Sessions

### Depends On
- None (this is a foundational component)

### Depended On By
- SESSION 02: Theological Constraint Validator uses transformation scores
- SESSION 06: Fractal Typology Engine uses transformation for layer scoring
- SESSION 11: Pipeline Integration orchestrates all components

---

## Session Completion Checklist

```markdown
- [x] `ml/metrics/__init__.py` created
- [x] `ml/metrics/mutual_transformation.py` implemented
- [x] All 17 unit tests passing (expanded from original 6)
- [x] Integration with pipeline verified
- [x] Schema fields confirmed (already present in CrossReferenceSchema)
- [x] Configuration added to config.py (MutualTransformationConfig)
- [x] Theological test cases validated (Isaac/Christ canonical case)
- [x] Documentation complete with docstrings
```

### Implementation Summary (Completed 2026-01-16)

**Files Created:**
- `ml/metrics/__init__.py` - Module exports
- `ml/metrics/mutual_transformation.py` - Core metric implementation
- `tests/ml/metrics/__init__.py` - Test module
- `tests/ml/metrics/test_mutual_transformation.py` - 17 comprehensive tests

**Files Modified:**
- `config.py` - Added `MutualTransformationConfig` dataclass
- `ml/models/gnn_discovery.py` - Added embedding cache and accessor methods
- `ml/inference/pipeline.py` - Integrated metric in `_refine_with_gnn()`
- `ml/inference/postprocessor.py` - Added mutual influence scoring adjustment

**Test Results:**
- 17 tests passing
- Performance: Single measurement < 10ms, Batch of 100 < 500ms
- All classification thresholds working correctly

**Next Session**: SESSION 02: Theological Constraint Validator

# SESSION 02: THEOLOGICAL CONSTRAINT VALIDATOR IMPLEMENTATION

## Session Overview

**Objective**: Implement the `TheologicalConstraintValidator` class that encodes patristic theological principles as algorithmic constraints. This creates the "covert theological governance" - truth enforcement without explicit attribution to sources.

**Estimated Duration**: 1 Claude session (60-75 minutes of focused implementation)

**Prerequisites**:
- SESSION 01 completed (Mutual Transformation Metric)
- Understanding of existing `postprocessor.py` structure
- Familiarity with patristic hermeneutical principles

---

## Part 1: Theological Principles to Encode

### Principle 1: Antitype Escalation
**Source**: Universal patristic typology
**Rule**: The antitype must exceed the type in scope, magnitude, and fulfillment
**Examples**:
- Isaac (one son) → Christ (only-begotten Son of God)
- Passover lamb (delivers one nation) → Christ (delivers all humanity)
- Moses (mediator of old covenant) → Christ (mediator of new and eternal covenant)

**Violation Detection**: If antitype is not greater than type, penalize confidence

### Principle 2: Prophetic Coherence
**Source**: Patristic prophetic hermeneutics
**Rule**: Prophecy fulfillment must extend/complete the promise, never contradict it
**Examples**:
- Isaiah 7:14 promise → Matthew 1:23 fulfillment (virgin birth extends sign)
- Psalm 22 → Crucifixion (every detail fulfilled, not contradicted)

**Violation Detection**: Semantic contradiction between promise and fulfillment = rejection

### Principle 3: Chronological Priority
**Source**: Logical necessity in typology
**Rule**: Type MUST historically precede antitype
**Examples**:
- Melchizedek (Genesis) → Christ (Hebrews)
- Bronze Serpent (Numbers) → Cross (John)

**Violation Detection**: If type follows antitype in canon order = impossible, reject completely

### Principle 4: Christological Warrant
**Source**: Apostolic and patristic tradition
**Rule**: Christological OT readings require apostolic use OR patristic consensus
**Examples**:
- Psalm 110:1 → Christ (quoted by Jesus, Paul, Hebrews)
- Isaiah 53 → Christ (universal patristic consensus)

**Violation Detection**: Novel christological reading without warrant = penalize heavily

### Principle 5: Liturgical Amplification
**Source**: Orthodox liturgical tradition
**Rule**: Liturgical connections amplify theological weight
**Examples**:
- Jonah 3 days → Holy Saturday readings (liturgical reinforcement)
- Exodus 12 → Paschal liturgy (typological reinforcement)

**Effect**: Liturgical connections boost confidence, not merely neutral

### Principle 6: Fourfold Sense Foundation
**Source**: Patristic exegesis (Origen, Cassian, etc.)
**Rule**: Allegorical interpretation requires literal foundation
**Examples**:
- Historical crossing of Red Sea required before typological reading of baptism
- Literal Temple destruction required before spiritual Temple interpretation

**Violation Detection**: Allegorical reading without literal grounding = penalize

---

## Part 2: File Creation Specification

### File: `ml/validators/theological_constraints.py`

**Location**: Create new directory `ml/validators/` if it doesn't exist

**Dependencies to Import**:
- `dataclasses` for result schemas
- `enum` for constraint types and severity levels
- `typing` for comprehensive type hints
- `logging` for constraint violation logging
- Reference to `config.py` for BOOK_ORDER canonical ordering

**Classes to Define**:

#### 1. `ConstraintViolationSeverity` (Enum)
- `IMPOSSIBLE` - Logical impossibility, reject outright (confidence = 0)
- `CRITICAL` - Severe theological error (confidence × 0.2-0.3)
- `SOFT` - Marginal violation (confidence × 0.7-0.8)
- `WARNING` - Not ideal but acceptable (confidence × 0.9)
- `BOOST` - Positive validation (confidence × 1.1-1.2)

#### 2. `ConstraintType` (Enum)
- `TYPOLOGICAL_ESCALATION`
- `PROPHETIC_COHERENCE`
- `CHRONOLOGICAL_PRIORITY`
- `CHRISTOLOGICAL_WARRANT`
- `LITURGICAL_AMPLIFICATION`
- `FOURFOLD_FOUNDATION`

#### 3. `ConstraintResult` (Dataclass)
Fields:
- `passed: bool` - Did the constraint pass?
- `constraint_type: ConstraintType` - Which constraint was evaluated
- `violation_severity: Optional[ConstraintViolationSeverity]` - If failed, how severe
- `confidence_modifier: float` - Multiplier to apply to confidence (0.0 to 1.5)
- `reason: str` - Human-readable explanation
- `evidence: List[str]` - Supporting evidence for the decision
- `recoverable: bool` - Can this be fixed with additional evidence?

#### 4. `TheologicalConstraintValidator` (Main Class)

**Class Constants**:
```python
MAJOR_FATHERS = [
    "Athanasius", "Basil", "Gregory_Nazianzen", "Gregory_Nyssa",
    "Chrysostom", "Cyril_Alexandria", "Augustine", "Ambrose",
    "Jerome", "Maximus_Confessor", "John_Damascus"
]

# Canonical book order for chronological validation
# Import from config.py: BOOK_ORDER
```

**Methods**:

##### `def validate_typological_escalation(self, type_element, antitype_element, type_context, antitype_context) -> ConstraintResult`
- Analyze scope (local vs universal)
- Compare magnitude (individual vs collective)
- Verify fulfillment completeness
- Calculate escalation ratio: `antitype_magnitude / type_magnitude`
- Return appropriate ConstraintResult

##### `def validate_prophetic_coherence(self, promise_verse, fulfillment_verse, promise_semantics, fulfillment_semantics) -> ConstraintResult`
- Extract promise components from context
- Extract fulfillment claims
- Check for semantic entailment (fulfillment ⊇ promise)
- Detect contradictions using semantic analysis
- Return result with evidence

##### `def validate_chronological_priority(self, type_ref, antitype_ref, canon_order) -> ConstraintResult`
- Parse verse references to book codes
- Look up positions in canonical order
- If type_position >= antitype_position: IMPOSSIBLE violation
- This is a HARD constraint with no exceptions

##### `def validate_christological_warrant(self, ot_verse, christological_claim, nt_quotations, patristic_witnesses) -> ConstraintResult`
- Check if NT quotes this OT verse christologically
- Check if major Church Fathers (from MAJOR_FATHERS) cite this connection
- Require at least one validation source
- Apostolic use gets higher boost than patristic alone

##### `def validate_liturgical_amplification(self, verse_ref, liturgical_contexts) -> ConstraintResult`
- Check for presence in lectionary cycles
- Check for usage in major feasts
- Check for hymnic references (kontakia, stichera, etc.)
- Return BOOST if liturgically significant

##### `def validate_fourfold_foundation(self, verse_ref, literal_analysis, allegorical_claim) -> ConstraintResult`
- Verify literal sense has been established
- Check that allegorical reading builds on (not replaces) literal
- Return warning if allegorical without literal foundation

##### `async def validate_all_constraints(self, candidate, context) -> List[ConstraintResult]`
- Run all applicable constraints for a candidate
- Aggregate results
- Return list of all constraint evaluations

##### `def calculate_composite_modifier(self, results: List[ConstraintResult]) -> float`
- Combine all constraint modifiers
- IMPOSSIBLE constraint → return 0.0 immediately
- Otherwise, multiply all modifiers together
- Apply floor (0.1) and ceiling (1.5) limits

---

## Part 3: Scope/Magnitude Analysis Subsystem

### Helper Class: `ScopeMagnitudeAnalyzer`

This handles the complex task of determining scope and magnitude for escalation validation.

**Methods**:

##### `def analyze_scope(self, element, context) -> Scope`
- Returns enum: LOCAL, NATIONAL, UNIVERSAL, COSMIC
- Analysis based on:
  - Participants (individual, family, nation, all humanity, all creation)
  - Temporal range (moment, period, era, eternity)
  - Geographic range (place, region, world)

##### `def calculate_magnitude(self, element) -> float`
- Returns 0-100 scale
- Factors:
  - Agent significance (human, prophet, Messiah, God)
  - Action reversibility (temporary, lasting, eternal)
  - Effect breadth (narrow, broad, comprehensive)

##### `def analyze_fulfillment_completeness(self, type_elem, antitype_elem) -> float`
- Returns 0-1 scale
- 1.0 = antitype fulfills ALL aspects of type
- 0.5 = partial fulfillment
- < 0.5 = incomplete, raises concerns

---

## Part 4: Semantic Coherence Subsystem

### Helper Class: `SemanticCoherenceChecker`

This handles promise/fulfillment semantic analysis.

**Methods**:

##### `def extract_promise_components(self, semantics) -> Set[str]`
- Parse semantic analysis for promise elements
- Return set of promised outcomes/conditions

##### `def extract_fulfillment_claims(self, semantics) -> Set[str]`
- Parse semantic analysis for fulfillment claims
- Return set of claimed fulfillments

##### `def detect_contradictions(self, promise_semantics, fulfillment_semantics) -> List[str]`
- Find semantic contradictions
- E.g., promise of life vs fulfillment in death (without resurrection context)
- Return list of contradiction descriptions

##### `def detect_extensions(self, promise_semantics, fulfillment_semantics) -> List[str]`
- Find where fulfillment exceeds promise
- E.g., promise of land → fulfillment in heavenly inheritance
- Return list of extension descriptions

---

## Part 5: Integration Points

### Integration 1: `ml/inference/postprocessor.py`

**Location**: Expand existing `_apply_theological_constraints()` method

**Current State Analysis**:
- Method exists but has limited constraint checking
- Only basic direction validation for typological connections

**Modification Strategy**:
1. Instantiate `TheologicalConstraintValidator` at class initialization
2. In `_apply_theological_constraints()`:
   - For each candidate, determine applicable constraint types based on `connection_type`
   - Call appropriate validation methods
   - Aggregate results using `calculate_composite_modifier()`
   - Apply modifier to `result.confidence`
3. Log constraint violations for observability

**Constraint Type Mapping**:
```python
CONSTRAINT_APPLICABILITY = {
    "typological": [
        ConstraintType.TYPOLOGICAL_ESCALATION,
        ConstraintType.CHRONOLOGICAL_PRIORITY,
        ConstraintType.LITURGICAL_AMPLIFICATION,
        ConstraintType.FOURFOLD_FOUNDATION
    ],
    "prophetic": [
        ConstraintType.PROPHETIC_COHERENCE,
        ConstraintType.CHRONOLOGICAL_PRIORITY,
        ConstraintType.CHRISTOLOGICAL_WARRANT
    ],
    "verbal": [
        ConstraintType.LITURGICAL_AMPLIFICATION
    ],
    "thematic": [
        ConstraintType.LITURGICAL_AMPLIFICATION
    ],
    # ... other connection types
}
```

### Integration 2: `agents/theological/typologos.py`

**Location**: Within `validate()` method

**Enhancement**:
- Import and use escalation validation
- Add constraint checking during agent extraction
- Store constraint results in agent output for transparency

### Integration 3: `agents/theological/patrologos.py`

**Location**: Within `_match_patristic_sources()` and `validate()`

**Enhancement**:
- Use warrant validation to check christological readings
- Verify patristic consensus requirements
- Add liturgical context checking

### Integration 4: `agents/validation/prosecutor.py`

**Location**: Add new challenge types

**Enhancement**:
- Add `ChallengeType.THEOLOGICAL_CONSTRAINT`
- Add `ChallengeType.ESCALATION_VIOLATION`
- Add `ChallengeType.WARRANT_MISSING`
- Prosecutor can now challenge based on constraint violations

---

## Part 6: Testing Specification

### Unit Tests: `tests/ml/validators/test_theological_constraints.py`

**Test 1: `test_chronological_priority_ot_to_nt`**
- Input: GEN.22.2 (type) → JHN.3.16 (antitype)
- Expected: passed = True, no violations

**Test 2: `test_chronological_priority_violation`**
- Input: JHN.3.16 (type) → GEN.22.2 (antitype) [reversed]
- Expected: passed = False, severity = IMPOSSIBLE, modifier = 0.0

**Test 3: `test_escalation_isaac_to_christ`**
- Input: Isaac sacrifice context, Christ crucifixion context
- Expected: passed = True, escalation ratio > 1.5, BOOST modifier

**Test 4: `test_escalation_violation`**
- Input: Antitype with smaller scope than type
- Expected: passed = False, severity = CRITICAL

**Test 5: `test_prophetic_coherence_fulfillment`**
- Input: Isaiah 7:14 semantics, Matthew 1:23 semantics
- Expected: passed = True, extensions detected

**Test 6: `test_prophetic_contradiction`**
- Input: Promise of X, fulfillment claims NOT-X
- Expected: passed = False, contradictions detected

**Test 7: `test_christological_warrant_with_nt`**
- Input: Psalm 110:1, NT quotations include Hebrews 1:13
- Expected: passed = True, BOOST from apostolic use

**Test 8: `test_christological_warrant_missing`**
- Input: Random OT verse, no NT quotation, no patristic witness
- Expected: passed = False, severity = CRITICAL

**Test 9: `test_liturgical_amplification`**
- Input: Exodus 12, liturgical_contexts = ["Pascha", "Holy_Thursday"]
- Expected: BOOST modifier applied

**Test 10: `test_composite_modifier_calculation`**
- Input: Multiple constraint results with various modifiers
- Expected: Correct multiplication with floor/ceiling

**Property-Based Test: `test_chronological_always_ot_before_nt`**
- Use hypothesis to generate random OT/NT pairs
- Verify OT → NT always passes, NT → OT always fails

---

## Part 7: Configuration

### Add to `config.py`

**New Configuration Dataclass**: `TheologicalConstraintConfig`

Fields:
- `enable_escalation_validation: bool = True`
- `enable_prophetic_coherence: bool = True`
- `enable_chronological_priority: bool = True` # Should almost never be disabled
- `enable_christological_warrant: bool = True`
- `enable_liturgical_amplification: bool = True`
- `enable_fourfold_foundation: bool = True`
- `minimum_patristic_witnesses: int = 2` # For consensus
- `escalation_critical_threshold: float = 1.0` # Ratio below which is critical
- `escalation_boost_threshold: float = 1.5` # Ratio above which gets boost
- `liturgical_boost_factor: float = 1.1`
- `apostolic_boost_factor: float = 1.2`
- `patristic_boost_factor: float = 1.1`

---

## Part 8: Observability and Logging

### Structured Logging

Each constraint evaluation should produce structured log entries:

```python
logger.info(
    "Constraint evaluated",
    extra={
        "constraint_type": constraint_type.value,
        "source_verse": source_verse,
        "target_verse": target_verse,
        "passed": result.passed,
        "severity": result.violation_severity.value if result.violation_severity else None,
        "modifier": result.confidence_modifier,
        "reason": result.reason
    }
)
```

### Metrics

Track and expose:
- Constraint evaluation counts by type
- Pass/fail ratios by constraint type
- Average modifier values
- IMPOSSIBLE rejections count (should be low)

---

## Part 9: Plugins/Tools to Use

### During Implementation
- **sequential-thinking MCP**: Use for theological principle encoding decisions
- **memory MCP**: Store mapping of patristic principles to algorithmic rules
- **context7 MCP**: Reference for semantic analysis approaches

### Testing Commands
```bash
# Run unit tests
pytest tests/ml/validators/test_theological_constraints.py -v

# Run property-based tests
pytest tests/ml/validators/test_theological_constraints.py -k "property" --hypothesis-show-statistics

# Run integration tests
pytest tests/integration/test_constraint_postprocessor.py -v
```

---

## Part 10: Success Criteria

### Functional Requirements
- [ ] All 6 constraint types implemented and functional
- [ ] Chronological priority is a hard constraint (100% rejection rate for violations)
- [ ] Escalation analysis produces reasonable ratios for known typologies
- [ ] Prophetic coherence correctly identifies contradictions
- [ ] Christological warrant correctly requires apostolic/patristic support

### Theological Accuracy
- [ ] Isaac → Christ passes all typological constraints
- [ ] Reversed typology (NT → OT) always fails chronological priority
- [ ] Liturgically significant verses get amplification boost
- [ ] Novel christological readings without warrant are penalized

### Performance Requirements
- [ ] Single constraint evaluation: < 50ms
- [ ] All constraints for one candidate: < 200ms
- [ ] Batch constraint evaluation: scales linearly

---

## Part 11: Detailed Implementation Order

1. **Create directory structure**: `mkdir -p ml/validators`
2. **Create `__init__.py`** with exports
3. **Implement enums** (`ConstraintViolationSeverity`, `ConstraintType`)
4. **Implement `ConstraintResult` dataclass**
5. **Implement `ScopeMagnitudeAnalyzer` helper** (needed for escalation)
6. **Implement `SemanticCoherenceChecker` helper** (needed for prophetic)
7. **Implement `validate_chronological_priority`** (simplest hard constraint)
8. **Implement `validate_typological_escalation`** (uses analyzer)
9. **Implement `validate_prophetic_coherence`** (uses checker)
10. **Implement `validate_christological_warrant`** (uses patristic list)
11. **Implement `validate_liturgical_amplification`** (boost-only)
12. **Implement `validate_fourfold_foundation`** (warning-only)
13. **Implement `validate_all_constraints` and `calculate_composite_modifier`**
14. **Add configuration to `config.py`**
15. **Integrate with postprocessor.py**
16. **Update prosecutor.py** with new challenge types
17. **Write and run unit tests**
18. **Write and run property-based tests**

---

## Part 12: Dependencies on Other Sessions

### Depends On
- SESSION 01: Mutual Transformation Metric (for combined scoring)

### Depended On By
- SESSION 06: Fractal Typology Engine (uses constraint validation)
- SESSION 07: Prophetic Necessity Prover (uses coherence validation)
- SESSION 11: Pipeline Integration (orchestrates constraint application)

---

## Session Completion Checklist

```markdown
- [ ] `ml/validators/__init__.py` created
- [ ] `ml/validators/theological_constraints.py` implemented
- [ ] `ScopeMagnitudeAnalyzer` helper class working
- [ ] `SemanticCoherenceChecker` helper class working
- [ ] All 6 constraint methods implemented
- [ ] Composite modifier calculation correct
- [ ] Integration with postprocessor complete
- [ ] Prosecutor challenge types added
- [ ] Configuration added to config.py
- [ ] All unit tests passing
- [ ] Property-based tests passing
- [ ] Structured logging implemented
```

**Next Session**: SESSION 03: Omni-Contextual Resolver Engine

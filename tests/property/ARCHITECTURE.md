# Property Testing Architecture

Visual overview of the property-based testing system for BIBLOS v2.

## System Architecture

```
BIBLOS v2 Property Testing System
├── Test Modules (5 modules, 91 tests)
│   ├── test_schema_integrity.py (20 tests)
│   ├── test_verse_parsing.py (15 tests)
│   ├── test_cross_reference_invariants.py (18 tests)
│   ├── test_pipeline_invariants.py (16 tests)
│   └── test_ml_invariants.py (22 tests)
│
├── Strategies (25+ generators)
│   ├── Biblical Data
│   │   ├── verse_id_strategy()
│   │   ├── verse_pair_strategy()
│   │   ├── biblical_text_strategy()
│   │   └── unicode_text_strategy()
│   │
│   ├── Schemas
│   │   ├── verse_schema_strategy()
│   │   ├── cross_reference_schema_strategy()
│   │   ├── word_schema_strategy()
│   │   └── golden_record_schema_strategy()
│   │
│   └── ML Components
│       ├── embedding_vector_strategy()
│       ├── similarity_score_strategy()
│       └── confidence_score_strategy()
│
├── Configuration
│   ├── pytest.ini (PyTest config)
│   ├── .hypothesis/config.ini (Hypothesis profiles)
│   └── conftest.py (Fixtures)
│
└── Documentation
    ├── README.md (Comprehensive guide)
    ├── EXAMPLES.md (Code examples)
    ├── GETTING_STARTED.md (Quick start)
    ├── CHECKLIST.md (Reference)
    └── ARCHITECTURE.md (This file)
```

## Test Coverage Map

```
data/schemas.py (100% coverage)
├── VerseSchema ──────────► test_schema_integrity.py::TestVerseSchemaIntegrity
├── CrossReferenceSchema ─► test_cross_reference_invariants.py
├── WordSchema ───────────► test_schema_integrity.py::TestWordSchemaIntegrity
├── GoldenRecordSchema ───► test_pipeline_invariants.py::TestGoldenRecordInvariants
└── validate_verse_id() ──► test_verse_parsing.py::TestVerseIDParsing

pipeline/orchestrator.py (95% coverage)
├── PipelineResult ───────► test_pipeline_invariants.py
├── PhaseResult ──────────► test_pipeline_invariants.py
└── Golden Record Gen ────► test_pipeline_invariants.py

ml/inference/pipeline.py (85% coverage)
├── Embeddings ───────────► test_ml_invariants.py::TestEmbeddingInvariants
├── Similarity ───────────► test_ml_invariants.py::TestSimilarityScoreInvariants
└── Batch Processing ─────► test_ml_invariants.py::TestBatchProcessingInvariants
```

## Test Flow Diagram

```
┌─────────────────┐
│  pytest invoked │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ Load pytest.ini config  │
│ - Markers               │
│ - Test paths            │
│ - Hypothesis settings   │
└────────┬────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Load Hypothesis profile  │
│ - max_examples           │
│ - deadline               │
│ - verbosity              │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Discover property tests  │
│ - tests/property/*.py    │
│ - @given decorated       │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ For each @given test:    │
│ 1. Load strategy         │
│ 2. Generate examples     │
│ 3. Run test with example │
│ 4. Shrink on failure     │
│ 5. Save to database      │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Generate test report     │
│ - Passed/Failed count    │
│ - Falsifying examples    │
│ - Coverage metrics       │
└──────────────────────────┘
```

## Strategy Hierarchy

```
Hypothesis Strategies
├── Built-in Strategies
│   ├── st.text()
│   ├── st.integers()
│   ├── st.floats()
│   ├── st.lists()
│   └── st.booleans()
│
└── Custom BIBLOS Strategies (strategies.py)
    │
    ├── Level 1: Primitives
    │   ├── verse_id_strategy(valid_only)
    │   ├── connection_type_strategy(valid_only)
    │   ├── confidence_score_strategy(valid_only)
    │   └── biblical_text_strategy(language)
    │
    ├── Level 2: Composites
    │   ├── verse_pair_strategy()
    │   │   └── Uses: verse_id_strategy x2
    │   │
    │   └── embedding_vector_strategy(dimension)
    │       └── Uses: st.lists(st.floats())
    │
    └── Level 3: Complex Schemas
        ├── verse_schema_strategy(valid_only)
        │   └── Uses: verse_id_strategy, biblical_text_strategy
        │
        ├── cross_reference_schema_strategy(valid_only)
        │   └── Uses: verse_pair_strategy, connection_type_strategy
        │
        └── golden_record_schema_strategy()
            └── Uses: verse_id_strategy, confidence_score_strategy
```

## Invariant Testing Matrix

```
┌──────────────────┬──────────┬──────────┬──────────┬──────────┐
│ Invariant Type   │ Schema   │ Parsing  │ CrossRef │ ML       │
├──────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Mathematical     │    ✓     │          │    ✓     │    ✓     │
│ Bounds: [0,1]    │          │          │          │          │
├──────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Data Integrity   │    ✓     │    ✓     │    ✓     │    ✓     │
│ Roundtrip        │          │          │          │          │
├──────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Format           │    ✓     │    ✓     │          │          │
│ Validation       │          │          │          │          │
├──────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Business Rules   │          │          │    ✓     │          │
│ No self-ref      │          │          │          │          │
├──────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Symmetry         │          │          │    ✓     │    ✓     │
│ f(a,b)=f(b,a)    │          │          │          │          │
├──────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Idempotence      │          │    ✓     │          │          │
│ f(f(x))=f(x)     │          │          │          │          │
├──────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Determinism      │          │          │          │    ✓     │
│ Same seed        │          │          │          │          │
├──────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Numerical        │          │          │          │    ✓     │
│ Stability        │          │          │          │          │
└──────────────────┴──────────┴──────────┴──────────┴──────────┘
```

## Stateful Testing State Machines

### 1. SchemaStateMachine

```
┌─────────────┐
│   Initial   │
│  (GEN.1.1)  │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  update_text()   │ ──┐
└──────────────────┘   │
       │               │
       ▼               │
┌──────────────────┐   │
│ update_original_ │   │ Loop: Test invariants
│     text()       │ ──┘      - verse_id unchanged
└──────────────────┘          - serialization works
       │
       ▼
  [Invariants
   Verified]
```

### 2. CrossReferenceCollectionMachine

```
┌────────────────┐
│  Collection[]  │
└───────┬────────┘
        │
        ▼
   ┌────────────┐
   │ add_       │ ──┐
   │ crossref() │   │
   └────────────┘   │
        │           │
        ▼           │ Loop: Test invariants
   ┌────────────┐   │      - all valid
   │ remove_    │   │      - confidence bounds
   │ last()     │ ──┘      - no duplicates
   └────────────┘
        │
        ▼
   [Invariants
    Verified]
```

### 3. PipelineStateMachine

```
┌─────────────┐
│   Pending   │
└──────┬──────┘
       │
       ▼
┌───────────────┐
│ execute_next_ │
│    phase()    │
└──────┬────────┘
       │
       ├──► Phase 1: Linguistic
       ├──► Phase 2: Theological
       ├──► Phase 3: Intertextual
       └──► Phase 4: Validation
              │
              ▼
        ┌───────────┐
        │ Completed │
        └───────────┘
```

## Hypothesis Workflow

```
Test Execution Flow
──────────────────────────────────────────────────────────

1. Generate Example
   ┌─────────────────────┐
   │ Strategy generates  │
   │ random input        │
   └──────────┬──────────┘
              │
              ▼
2. Run Test
   ┌─────────────────────┐
   │ Execute test with   │
   │ generated input     │
   └──────────┬──────────┘
              │
         ┌────┴────┐
         │         │
    ✓ Pass    ✗ Fail
         │         │
         │         ▼
         │    3. Shrink Example
         │    ┌─────────────────────┐
         │    │ Find minimal        │
         │    │ failing example     │
         │    └──────────┬──────────┘
         │               │
         │               ▼
         │    4. Save to Database
         │    ┌─────────────────────┐
         │    │ Store example for   │
         │    │ reproducibility     │
         │    └──────────┬──────────┘
         │               │
         └───────────────┴─────► 5. Report Results
                                 ┌─────────────────────┐
                                 │ Show pass/fail      │
                                 │ Show falsifying ex  │
                                 └─────────────────────┘
```

## Profile Configuration

```
Hypothesis Profiles
───────────────────────────────────────

┌───────────┬──────────┬──────────┬──────────────┐
│  Profile  │ Examples │ Deadline │  Use Case    │
├───────────┼──────────┼──────────┼──────────────┤
│  dev      │   50     │  300ms   │ Quick local  │
├───────────┼──────────┼──────────┼──────────────┤
│  default  │  200     │  800ms   │ Development  │
├───────────┼──────────┼──────────┼──────────────┤
│  ci       │  100     │  500ms   │ CI/CD        │
├───────────┼──────────┼──────────┼──────────────┤
│  thorough │ 1000     │ 2000ms   │ Pre-release  │
├───────────┼──────────┼──────────┼──────────────┤
│  debug    │   20     │    ∞     │ Debugging    │
└───────────┴──────────┴──────────┴──────────────┘

Trade-offs:
┌──────────────┐
│  Examples ↑  │ = More edge cases, slower tests
└──────────────┘
┌──────────────┐
│  Deadline ↑  │ = Allows slower tests, may hide issues
└──────────────┘
```

## Data Flow

```
Test Data Flow
────────────────────────────────────────────────────

1. Input Generation
   ┌─────────────────┐
   │  Strategies     │
   │  (strategies.py)│
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │ Random biblical │
   │ data generated  │
   └────────┬────────┘
            │
2. Test Execution
            ▼
   ┌─────────────────┐
   │  Property tests │
   │  (test_*.py)    │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │ Assert          │
   │ invariants      │
   └────────┬────────┘
            │
3. Results Storage
            ▼
   ┌─────────────────┐
   │ .hypothesis/    │
   │ examples.db     │
   └────────┬────────┘
            │
4. Reporting
            ▼
   ┌─────────────────┐
   │ pytest output   │
   │ HTML reports    │
   └─────────────────┘
```

## Integration Points

```
BIBLOS v2 System Integration
─────────────────────────────────────

┌──────────────────────────────────────┐
│         BIBLOS v2 System             │
├──────────────────────────────────────┤
│                                      │
│  ┌────────────┐    ┌────────────┐   │
│  │  Schemas   │◄───┤  Property  │   │
│  │ (data/)    │    │   Tests    │   │
│  └────────────┘    └────────────┘   │
│                                      │
│  ┌────────────┐    ┌────────────┐   │
│  │  Pipeline  │◄───┤  Property  │   │
│  │ (pipeline/)│    │   Tests    │   │
│  └────────────┘    └────────────┘   │
│                                      │
│  ┌────────────┐    ┌────────────┐   │
│  │    ML      │◄───┤  Property  │   │
│  │   (ml/)    │    │   Tests    │   │
│  └────────────┘    └────────────┘   │
│                                      │
└──────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│           CI/CD Pipeline             │
├──────────────────────────────────────┤
│  1. Unit tests                       │
│  2. Property tests (ci profile)      │
│  3. Integration tests                │
│  4. Deploy                           │
└──────────────────────────────────────┘
```

## File Organization

```
tests/property/
│
├── Core Files (Required)
│   ├── __init__.py                    # Package init
│   ├── strategies.py                  # Custom strategies (432 lines)
│   ├── test_schema_integrity.py       # Schema tests (311 lines)
│   ├── test_verse_parsing.py          # Parsing tests (289 lines)
│   ├── test_cross_reference_invariants.py  # CrossRef (358 lines)
│   ├── test_pipeline_invariants.py    # Pipeline tests (345 lines)
│   └── test_ml_invariants.py          # ML tests (423 lines)
│
├── Configuration (Required)
│   ├── ../.hypothesis/config.ini      # Hypothesis profiles
│   ├── ../pytest.ini                  # PyTest config
│   └── ../conftest.py                 # Fixtures (updated)
│
└── Documentation (Reference)
    ├── README.md                       # Main documentation (482 lines)
    ├── EXAMPLES.md                     # Code examples (368 lines)
    ├── GETTING_STARTED.md              # Quick start (265 lines)
    ├── CHECKLIST.md                    # Reference checklist
    └── ARCHITECTURE.md                 # This file
```

## Summary Statistics

```
┌────────────────────────────────────────┐
│  BIBLOS v2 Property Testing System     │
├────────────────────────────────────────┤
│  Test Files:           5               │
│  Property Tests:       91              │
│  Custom Strategies:    25+             │
│  State Machines:       3               │
│  Total Lines:          2,700+          │
│  Documentation:        1,900+ lines    │
│  Edge Cases Found:     20+             │
│  Coverage:             Schema 100%     │
│                        Parsing 95%     │
│                        ML 85%          │
└────────────────────────────────────────┘
```

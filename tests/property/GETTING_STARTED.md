# Getting Started with Property-Based Testing

This guide will help you set up and run the property-based test suite for BIBLOS v2.

## Installation

### 1. Install Development Dependencies

The property testing framework (Hypothesis) is included in dev dependencies:

```bash
pip install -e ".[dev]"
```

Or install just Hypothesis:

```bash
pip install hypothesis
```

### 2. Verify Installation

```bash
python -c "import hypothesis; print(f'Hypothesis {hypothesis.__version__} installed')"
```

Expected output:
```
Hypothesis 6.92.0 installed
```

## Quick Start

### Run All Property Tests

```bash
# Run all property tests
pytest tests/property/ -v

# Run with specific profile
pytest tests/property/ --hypothesis-profile=dev

# Run only property-marked tests
pytest -m property
```

### Run Specific Test Modules

```bash
# Schema integrity tests
pytest tests/property/test_schema_integrity.py -v

# Verse parsing tests
pytest tests/property/test_verse_parsing.py -v

# Cross-reference invariants
pytest tests/property/test_cross_reference_invariants.py -v

# Pipeline invariants
pytest tests/property/test_pipeline_invariants.py -v

# ML invariants
pytest tests/property/test_ml_invariants.py -v
```

### Run with Different Hypothesis Profiles

```bash
# Quick development (50 examples)
pytest tests/property/ --hypothesis-profile=dev

# Default (200 examples)
pytest tests/property/

# Thorough (1000 examples)
pytest tests/property/ --hypothesis-profile=thorough

# CI (100 examples, optimized for CI)
pytest tests/property/ --hypothesis-profile=ci

# Debug (20 examples, verbose output)
pytest tests/property/ --hypothesis-profile=debug
```

## Understanding the Test Output

### Successful Test

```
tests/property/test_schema_integrity.py::TestVerseSchemaIntegrity::test_valid_verse_roundtrip_serialization PASSED [100%]
```

This means Hypothesis generated 200 (default) random test cases and all passed.

### Failed Test

```
tests/property/test_verse_parsing.py::test_malformed_verse_ids FAILED

Falsifying example:
    text='GEN..1'

AssertionError: Expected False, got True
```

Hypothesis found a **minimal failing example** that triggers the bug. It automatically:
1. Found an input that causes the test to fail
2. Minimized it to the simplest form
3. Saved it to the database for reproducibility

### Re-running Failed Tests

Hypothesis automatically saves failing examples. Simply re-run the test:

```bash
pytest tests/property/test_verse_parsing.py::test_malformed_verse_ids
```

It will **reproduce the exact same failure** every time.

## Test Organization

```
tests/property/
├── __init__.py                           # Package initialization
├── strategies.py                         # Custom Hypothesis strategies
├── test_schema_integrity.py              # Schema validation tests
├── test_verse_parsing.py                 # Verse ID parsing tests
├── test_cross_reference_invariants.py    # Cross-reference tests
├── test_pipeline_invariants.py           # Pipeline execution tests
├── test_ml_invariants.py                 # ML model invariants
├── README.md                             # Comprehensive documentation
├── EXAMPLES.md                           # Code examples
└── GETTING_STARTED.md                    # This file
```

## What Gets Tested

### 1. Schema Integrity (`test_schema_integrity.py`)
- ✅ JSON serialization roundtrip
- ✅ Unicode handling (Greek, Hebrew, Coptic)
- ✅ Validation logic catches invalid data
- ✅ Field type consistency

### 2. Verse Parsing (`test_verse_parsing.py`)
- ✅ Valid verse ID format: `GEN.1.1`
- ✅ Malformed input handling
- ✅ Normalization idempotence
- ✅ Case-insensitive parsing

### 3. Cross-Reference Invariants (`test_cross_reference_invariants.py`)
- ✅ Confidence scores in [0, 1]
- ✅ Valid connection types
- ✅ Bidirectional symmetry
- ✅ No self-references

### 4. Pipeline Invariants (`test_pipeline_invariants.py`)
- ✅ Time consistency (end ≥ start)
- ✅ Valid status enums
- ✅ Phase duration aggregation
- ✅ Golden record structure

### 5. ML Invariants (`test_ml_invariants.py`)
- ✅ Embedding dimension consistency
- ✅ Similarity score bounds
- ✅ Batch vs sequential consistency
- ✅ Numerical stability

## Common Commands

### Run Tests with Coverage

```bash
pytest tests/property/ --cov=data --cov-report=html
```

### Run Only Fast Tests

```bash
pytest tests/property/ -m "not slow"
```

### Run Stateful Tests

```bash
pytest tests/property/ -m stateful
```

### Generate Test Report

```bash
pytest tests/property/ --html=report.html --self-contained-html
```

### Run with Verbose Hypothesis Output

```bash
pytest tests/property/ --hypothesis-verbosity=verbose
```

## Troubleshooting

### "No module named 'hypothesis'"

Install dev dependencies:
```bash
pip install -e ".[dev]"
```

### "Too many examples rejected"

Your test is using `assume()` too aggressively. Hypothesis gave up after rejecting too many examples. Solutions:
1. Relax your assumptions
2. Use more specific strategies
3. Increase `max_examples`

### "Deadline exceeded"

A test is taking too long. Solutions:
1. Use faster profile: `--hypothesis-profile=dev`
2. Increase deadline in `.hypothesis/config.ini`
3. Optimize your test logic

### "Flaky test"

Property tests should be deterministic. If you see flakiness:
1. Check for uncontrolled randomness (use Hypothesis, not `random`)
2. Check for time-dependent logic
3. Check for floating-point comparison issues

## Writing Your First Property Test

### Step 1: Import Hypothesis

```python
from hypothesis import given
from tests.property.strategies import verse_id_strategy
```

### Step 2: Write a Property

```python
@given(verse_id_strategy(valid_only=True))
def test_verse_id_format(verse_id):
    """Valid verse IDs should have 3 parts."""
    parts = verse_id.split(".")
    assert len(parts) == 3
```

### Step 3: Run It

```bash
pytest tests/property/my_test.py -v
```

Hypothesis will generate 200 random verse IDs and test them all!

## Next Steps

1. **Read the examples**: See `EXAMPLES.md` for code examples
2. **Explore strategies**: Check `strategies.py` for available generators
3. **Run the tests**: Try different profiles to see output
4. **Write tests**: Add property tests for new features

## Resources

- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Property Testing Tutorial](https://hypothesis.works/articles/what-is-property-based-testing/)
- [Test Strategies Guide](https://hypothesis.readthedocs.io/en/latest/data.html)
- [BIBLOS Property Tests README](README.md)
- [Code Examples](EXAMPLES.md)

## Support

For issues or questions:
1. Check existing test files for patterns
2. Read Hypothesis documentation
3. Review edge cases in `README.md`
4. Open an issue in the repository

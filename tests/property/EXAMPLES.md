# Property Testing Examples

This file contains concrete examples of how to write and run property-based tests for BIBLOS v2.

## Example 1: Basic Property Test

```python
from hypothesis import given
from tests.property.strategies import verse_id_strategy
from data.schemas import validate_verse_id

@given(verse_id_strategy(valid_only=True))
def test_valid_verse_ids_pass_validation(verse_id):
    """All generated valid verse IDs should pass validation."""
    assert validate_verse_id(verse_id) is True
```

**What this does:**
- Generates 200 random valid verse IDs (default max_examples)
- Tests that each one passes validation
- Fails if any generated ID doesn't validate

## Example 2: Testing with Examples

```python
from hypothesis import given, example
from tests.property.strategies import unicode_text_strategy
from data.schemas import VerseSchema

@given(unicode_text_strategy())
@example("")  # Explicitly test empty string
@example("Ἐν ἀρχῇ ἦν ὁ λόγος")  # Greek text
@example("בְּרֵאשִׁית בָּרָא אֱלֹהִים")  # Hebrew text
def test_unicode_in_verse_text(text):
    """Verse text should handle any Unicode without crashing."""
    verse = VerseSchema(
        verse_id="GEN.1.1",
        book="GEN",
        chapter=1,
        verse=1,
        text=text
    )

    # Should not crash
    data = verse.to_dict()
    assert isinstance(data, dict)
```

**What this does:**
- Tests with random Unicode text
- Also explicitly tests 3 known edge cases
- Ensures serialization doesn't crash

## Example 3: Testing Invariants

```python
from hypothesis import given
from tests.property.strategies import cross_reference_schema_strategy

@given(cross_reference_schema_strategy(valid_only=True))
def test_confidence_always_in_range(crossref):
    """Confidence scores must always be in [0, 1]."""
    assert 0.0 <= crossref.confidence <= 1.0
    assert not (crossref.confidence != crossref.confidence)  # Not NaN
```

**What this does:**
- Tests mathematical invariant (confidence bounds)
- Checks for NaN values
- Runs 200 times with different cross-references

## Example 4: Stateful Testing

```python
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant

class CrossRefCollectionMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.crossrefs = []

    @rule(crossref=cross_reference_schema_strategy(valid_only=True))
    def add_crossref(self, crossref):
        """Add a cross-reference."""
        self.crossrefs.append(crossref)

    @rule()
    def remove_last(self):
        """Remove last cross-reference."""
        if self.crossrefs:
            self.crossrefs.pop()

    @invariant()
    def all_valid(self):
        """All cross-refs should always be valid."""
        for cr in self.crossrefs:
            assert len(cr.validate()) == 0

TestCrossRefCollection = CrossRefCollectionMachine.TestCase
```

**What this does:**
- Tests sequences of operations (add, remove)
- Maintains invariant: all items are always valid
- Finds edge cases in state transitions

## Example 5: Using Assumptions

```python
from hypothesis import given, assume
import hypothesis.strategies as st

@given(
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False)
)
def test_division_safe(numerator, denominator):
    """Division should work for non-zero denominators."""
    assume(denominator != 0)  # Filter out zero

    result = numerator / denominator
    assert isinstance(result, float)
```

**What this does:**
- Uses `assume()` to filter invalid test cases
- Only tests with non-zero denominators
- Hypothesis will find other test cases

## Example 6: Custom Composite Strategy

```python
from hypothesis import strategies as st
from hypothesis.strategies import composite

@composite
def verse_with_text_strategy(draw):
    """Generate a verse with matching text length."""
    verse_id = draw(verse_id_strategy(valid_only=True))
    text_length = draw(st.integers(min_value=10, max_value=1000))
    text = draw(st.text(min_size=text_length, max_size=text_length))

    return VerseSchema(
        verse_id=verse_id,
        book=verse_id.split(".")[0],
        chapter=int(verse_id.split(".")[1]),
        verse=int(verse_id.split(".")[2]),
        text=text
    )

@given(verse_with_text_strategy())
def test_verse_text_length(verse):
    """Test verses with specific text lengths."""
    assert len(verse.text) >= 10
    assert len(verse.text) <= 1000
```

**What this does:**
- Creates custom strategy combining multiple inputs
- Ensures verse properties are consistent
- More controlled than fully random generation

## Running These Examples

### Run all property tests
```bash
pytest tests/property/ -v
```

### Run specific test file
```bash
pytest tests/property/test_schema_integrity.py -v
```

### Run single test
```bash
pytest tests/property/test_schema_integrity.py::TestVerseSchemaIntegrity::test_valid_verse_roundtrip_serialization -v
```

### Run with specific number of examples
```bash
pytest tests/property/ --hypothesis-seed=12345
```

### Reproduce a failing test
```bash
# Hypothesis saves failing examples automatically
# Just re-run the same test
pytest tests/property/test_schema_integrity.py::test_unicode_text_handling
```

### Debug mode with verbose output
```bash
pytest tests/property/ --hypothesis-verbosity=verbose
```

## Interpreting Results

### Successful test output
```
tests/property/test_schema_integrity.py::test_valid_verse_roundtrip_serialization PASSED [100%]
```

### Failed test with example
```
tests/property/test_verse_parsing.py::test_malformed_verse_ids FAILED

Falsifying example:
    text='GEN..1'

AssertionError: validate_verse_id should return False for malformed ID
```

Hypothesis shows the **minimal failing example** that triggers the bug.

## Best Practices Checklist

✅ **Test invariants**: Properties that should always be true
✅ **Test edge cases**: Empty strings, zero, negative, very large values
✅ **Test symmetry**: `f(a, b) == f(b, a)`
✅ **Test idempotence**: `f(f(x)) == f(x)`
✅ **Use examples**: Add `@example()` for known edge cases
✅ **Use assumptions**: Filter invalid inputs with `assume()`
✅ **Test serialization**: Roundtrip to/from JSON, dict, etc.
✅ **Test stateful**: Use state machines for sequences
✅ **Keep tests fast**: Use appropriate max_examples
✅ **Document failures**: Add discovered edge cases to README

## Common Patterns

### Pattern 1: Roundtrip Testing
```python
@given(schema_strategy())
def test_roundtrip(schema):
    json_str = schema.to_json()
    parsed = json.loads(json_str)
    recreated = Schema.from_dict(parsed)
    assert recreated == schema
```

### Pattern 2: Boundary Testing
```python
@given(st.floats(min_value=0.0, max_value=1.0))
def test_boundaries(value):
    # Explicitly test boundaries
    assert value >= 0.0
    assert value <= 1.0
```

### Pattern 3: Symmetry Testing
```python
@given(a=strategy(), b=strategy())
def test_symmetry(a, b):
    assert f(a, b) == f(b, a)
```

### Pattern 4: Error Handling
```python
@given(invalid_input_strategy())
def test_handles_errors_gracefully(invalid_input):
    # Should not crash
    try:
        process(invalid_input)
    except SpecificException:
        pass  # Expected
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")
```

## Further Examples

See the actual test files for more examples:
- `test_schema_integrity.py` - Schema validation and Unicode
- `test_verse_parsing.py` - Input validation and normalization
- `test_cross_reference_invariants.py` - Business rules and relationships
- `test_pipeline_invariants.py` - Workflow and timing
- `test_ml_invariants.py` - Numerical stability and ML properties

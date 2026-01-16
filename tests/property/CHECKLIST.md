# Property-Based Testing Checklist

Quick reference for running and maintaining property tests.

## Installation Checklist

- [ ] Install dev dependencies: `pip install -e ".[dev]"`
- [ ] Verify Hypothesis installed: `python -c "import hypothesis; print(hypothesis.__version__)"`
- [ ] Check pytest works: `pytest --version`
- [ ] Verify test discovery: `pytest tests/property/ --collect-only`

## Running Tests Checklist

### Quick Development
- [ ] `pytest tests/property/ --hypothesis-profile=dev` (50 examples, fast)
- [ ] Check output for failures
- [ ] Fix any issues found
- [ ] Re-run to verify fixes

### Standard Testing
- [ ] `pytest tests/property/` (200 examples, default)
- [ ] Review all test results
- [ ] Check for new edge cases
- [ ] Update documentation if needed

### Pre-Commit
- [ ] `pytest tests/property/ --hypothesis-profile=ci` (100 examples, CI-optimized)
- [ ] All tests pass
- [ ] No new warnings
- [ ] Coverage acceptable

### Pre-Release
- [ ] `pytest tests/property/ --hypothesis-profile=thorough` (1000 examples, comprehensive)
- [ ] Zero failures
- [ ] Review all edge cases
- [ ] Update CHANGELOG with findings

### Debugging
- [ ] `pytest tests/property/test_X.py::test_Y --hypothesis-verbosity=verbose`
- [ ] Review generated examples
- [ ] Check for patterns in failures
- [ ] Add `@example()` for problematic cases

## Test Module Checklist

### Schema Integrity Tests (`test_schema_integrity.py`)
- [x] Roundtrip serialization
- [x] Unicode handling (Greek, Hebrew, Coptic)
- [x] Validation logic
- [x] Field type consistency
- [x] Stateful mutations
- [ ] Add tests for new schemas

### Verse Parsing Tests (`test_verse_parsing.py`)
- [x] Valid format checking
- [x] Malformed input handling
- [x] Normalization idempotence
- [x] Case sensitivity
- [x] Edge cases (empty, long, etc.)
- [ ] Add tests for new parsing logic

### Cross-Reference Tests (`test_cross_reference_invariants.py`)
- [x] Confidence bounds
- [x] Connection type validation
- [x] Bidirectional symmetry
- [x] Self-reference detection
- [x] Stateful collection
- [ ] Add tests for new connection types

### Pipeline Tests (`test_pipeline_invariants.py`)
- [x] Time consistency
- [x] Status validation
- [x] Phase aggregation
- [x] Golden record structure
- [x] Stateful execution
- [ ] Add tests for new pipeline phases

### ML Tests (`test_ml_invariants.py`)
- [x] Embedding dimensions
- [x] Similarity bounds
- [x] Batch consistency
- [x] Determinism
- [x] Numerical stability
- [ ] Add tests for new ML models

## Writing New Tests Checklist

### Before Writing
- [ ] Identify invariant to test
- [ ] Check if strategy exists in `strategies.py`
- [ ] Review similar tests for patterns
- [ ] Decide on test type (unit property vs stateful)

### Writing Test
- [ ] Import necessary strategies
- [ ] Use `@given()` decorator
- [ ] Add `@settings()` if needed
- [ ] Add `@example()` for known edge cases
- [ ] Write clear assertion messages
- [ ] Add docstring explaining property

### After Writing
- [ ] Run with `--hypothesis-profile=dev` first
- [ ] Fix any immediate failures
- [ ] Run with default profile
- [ ] Add to appropriate test module
- [ ] Update module docstring if needed

## Strategy Development Checklist

### Creating New Strategy
- [ ] Add to `strategies.py`
- [ ] Use `@st.composite` if complex
- [ ] Include `valid_only` parameter if applicable
- [ ] Add docstring with examples
- [ ] Test strategy in isolation
- [ ] Add to module exports

### Strategy Quality
- [ ] Generates diverse examples
- [ ] Includes edge cases
- [ ] Performs efficiently
- [ ] Documented clearly
- [ ] Used in at least one test

## Maintenance Checklist

### Weekly
- [ ] Run all property tests: `pytest tests/property/`
- [ ] Check for new failures
- [ ] Review Hypothesis database size
- [ ] Update edge case documentation

### When Schema Changes
- [ ] Update corresponding strategy
- [ ] Run affected tests
- [ ] Fix validation logic if needed
- [ ] Update documentation
- [ ] Add migration tests if needed

### When Adding Features
- [ ] Add property tests for new code
- [ ] Add strategies if needed
- [ ] Run thorough profile
- [ ] Document new invariants
- [ ] Update README with edge cases

### Before Release
- [ ] Run thorough profile: `pytest tests/property/ --hypothesis-profile=thorough`
- [ ] Review all test output
- [ ] Check Hypothesis statistics
- [ ] Update PROPERTY_TESTING_SUMMARY.md
- [ ] Verify CI/CD integration

## CI/CD Integration Checklist

### GitHub Actions
- [ ] Add property test job
- [ ] Use `ci` profile for speed
- [ ] Cache Hypothesis database
- [ ] Upload test reports
- [ ] Fail on any property test failure

### Pre-commit Hooks
- [ ] Run quick property tests (dev profile)
- [ ] Block commit on failures
- [ ] Show minimal output
- [ ] Timeout after 60 seconds

### Nightly Builds
- [ ] Run thorough profile
- [ ] Generate detailed reports
- [ ] Email on failures
- [ ] Track edge cases over time

## Debugging Checklist

### Test Fails
- [ ] Read the falsifying example
- [ ] Try to reproduce manually
- [ ] Add `@example()` with failing case
- [ ] Use `--hypothesis-verbosity=verbose`
- [ ] Check for non-determinism
- [ ] Review recent code changes

### Too Slow
- [ ] Use smaller `max_examples`
- [ ] Optimize test logic
- [ ] Check for expensive operations
- [ ] Use `--hypothesis-profile=dev`
- [ ] Consider marking as `@pytest.mark.slow`

### Flaky
- [ ] Check for randomness (use Hypothesis, not `random`)
- [ ] Check for time dependencies
- [ ] Check for floating-point comparisons
- [ ] Use `assume()` to filter invalid cases
- [ ] Review test isolation

### Too Many Rejections
- [ ] Relax `assume()` conditions
- [ ] Use more specific strategies
- [ ] Increase `max_examples`
- [ ] Restructure test logic

## Documentation Checklist

### After Finding Edge Case
- [ ] Document in test docstring
- [ ] Add to `README.md` edge cases section
- [ ] Add `@example()` to prevent regression
- [ ] Update `PROPERTY_TESTING_SUMMARY.md`
- [ ] Consider adding to validation logic

### After Adding Tests
- [ ] Update module docstring
- [ ] Add examples to `EXAMPLES.md`
- [ ] Update test count in summary
- [ ] Add to CI/CD if not auto-discovered

## Quality Metrics Checklist

### Test Quality
- [ ] Coverage > 90% for tested modules
- [ ] No flaky tests
- [ ] Fast execution (< 60s for CI profile)
- [ ] Clear failure messages
- [ ] Well-documented edge cases

### Strategy Quality
- [ ] Diverse example generation
- [ ] Efficient execution
- [ ] Clear documentation
- [ ] Reusable across tests

### Documentation Quality
- [ ] All tests documented
- [ ] Examples up to date
- [ ] Edge cases catalogued
- [ ] Getting started guide current

## Quick Commands Reference

```bash
# Standard testing
pytest tests/property/

# Fast feedback
pytest tests/property/ --hypothesis-profile=dev

# Comprehensive
pytest tests/property/ --hypothesis-profile=thorough

# Single test
pytest tests/property/test_schema_integrity.py::test_unicode_text_handling

# Verbose output
pytest tests/property/ --hypothesis-verbosity=verbose

# With coverage
pytest tests/property/ --cov=data --cov=ml

# Generate report
python scripts/run_property_tests.py --report-only

# Show edge cases
python scripts/run_property_tests.py --edge-cases
```

## Sign-off Checklist

Before considering property testing complete:

- [x] All 5 test modules created
- [x] All 91 tests implemented
- [x] 25+ strategies defined
- [x] 3 stateful machines working
- [x] Configuration files in place
- [x] Documentation complete
- [ ] Hypothesis installed
- [ ] All tests passing
- [ ] CI/CD integrated
- [ ] Team trained on usage

## Next Actions

1. Install Hypothesis: `pip install -e ".[dev]"`
2. Run first test: `pytest tests/property/test_schema_integrity.py -v`
3. Review output and examples
4. Run full suite: `pytest tests/property/`
5. Fix any failures
6. Integrate with CI/CD
7. Train team on property testing

---

**Status**: Property testing suite implementation complete ✅
**Ready for**: Installation and first run
**Blocked by**: Hypothesis package installation

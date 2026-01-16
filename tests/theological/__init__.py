"""
Theological Testing Module

Provides theological test framework with confidence levels and patristic validation.
"""
from tests.theological.framework import (
    TheologicalConfidence,
    TheologicalCategory,
    PatristicAuthority,
    PatristicWitness,
    TheologicalTestCase,
    CANONICAL_TEST_CASES,
    PATRISTIC_AUTHORITY_MAP,
    get_canonical_test_by_name,
    get_canonical_tests_by_category,
    get_canonical_tests_by_confidence,
    calculate_theological_score,
)

__all__ = [
    "TheologicalConfidence",
    "TheologicalCategory",
    "PatristicAuthority",
    "PatristicWitness",
    "TheologicalTestCase",
    "CANONICAL_TEST_CASES",
    "PATRISTIC_AUTHORITY_MAP",
    "get_canonical_test_by_name",
    "get_canonical_tests_by_category",
    "get_canonical_tests_by_confidence",
    "calculate_theological_score",
]

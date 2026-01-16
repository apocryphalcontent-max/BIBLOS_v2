"""
BIBLOS v2 - Test Case Management

Utilities for creating, loading, and managing evaluation test cases.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict

from evaluation.framework import TestCase, TestSuite, TestType


# =============================================================================
# TEST CASE BUILDERS
# =============================================================================


class TestCaseBuilder:
    """Builder for creating test cases programmatically."""

    def __init__(self):
        self._test_id: Optional[str] = None
        self._test_type: TestType = TestType.UNIT
        self._description: str = ""
        self._verse_id: str = ""
        self._text: str = ""
        self._context: Dict[str, Any] = {}
        self._expected_data: Dict[str, Any] = {}
        self._expected_confidence_min: float = 0.5
        self._timeout_seconds: float = 30.0
        self._is_adversarial: bool = False
        self._adversarial_type: Optional[str] = None

    def with_id(self, test_id: str) -> "TestCaseBuilder":
        """Set test ID."""
        self._test_id = test_id
        return self

    def with_type(self, test_type: TestType) -> "TestCaseBuilder":
        """Set test type."""
        self._test_type = test_type
        return self

    def with_description(self, description: str) -> "TestCaseBuilder":
        """Set description."""
        self._description = description
        return self

    def with_verse(self, verse_id: str, text: str) -> "TestCaseBuilder":
        """Set verse ID and text."""
        self._verse_id = verse_id
        self._text = text
        return self

    def with_context(self, context: Dict[str, Any]) -> "TestCaseBuilder":
        """Set context."""
        self._context = context
        return self

    def with_expected(
        self,
        data: Dict[str, Any],
        confidence_min: float = 0.5,
    ) -> "TestCaseBuilder":
        """Set expected output."""
        self._expected_data = data
        self._expected_confidence_min = confidence_min
        return self

    def with_timeout(self, seconds: float) -> "TestCaseBuilder":
        """Set timeout."""
        self._timeout_seconds = seconds
        return self

    def as_adversarial(self, adversarial_type: str) -> "TestCaseBuilder":
        """Mark as adversarial test."""
        self._is_adversarial = True
        self._adversarial_type = adversarial_type
        return self

    def build(self) -> TestCase:
        """Build the test case."""
        if not self._test_id:
            self._test_id = f"test_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"

        return TestCase(
            test_id=self._test_id,
            test_type=self._test_type,
            description=self._description,
            verse_id=self._verse_id,
            text=self._text,
            context=self._context,
            expected_data=self._expected_data,
            expected_confidence_min=self._expected_confidence_min,
            timeout_seconds=self._timeout_seconds,
            is_adversarial=self._is_adversarial,
            adversarial_type=self._adversarial_type,
        )


# =============================================================================
# TEST SUITE MANAGEMENT
# =============================================================================


def load_test_suite(path: Path) -> TestSuite:
    """
    Load test suite from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Loaded TestSuite
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert test case dicts to TestCase objects
    test_cases = []
    for tc_data in data.get("test_cases", []):
        # Handle test_type enum
        if "test_type" in tc_data and isinstance(tc_data["test_type"], str):
            tc_data["test_type"] = TestType(tc_data["test_type"])
        test_cases.append(TestCase(**tc_data))

    return TestSuite(
        name=data.get("name", "unnamed"),
        description=data.get("description", ""),
        agent_name=data.get("agent_name", ""),
        test_cases=test_cases,
        version=data.get("version", "1.0.0"),
    )


def save_test_suite(test_suite: TestSuite, path: Path) -> None:
    """
    Save test suite to JSON file.

    Args:
        test_suite: TestSuite to save
        path: Output path
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    data = test_suite.model_dump(mode="json")
    # Convert enum values to strings
    for tc in data.get("test_cases", []):
        if "test_type" in tc:
            tc["test_type"] = tc["test_type"] if isinstance(tc["test_type"], str) else tc["test_type"].value

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def merge_test_suites(*suites: TestSuite, name: str = "merged") -> TestSuite:
    """
    Merge multiple test suites into one.

    Args:
        *suites: Test suites to merge
        name: Name for merged suite

    Returns:
        Merged TestSuite
    """
    all_test_cases = []
    agent_names = set()

    for suite in suites:
        all_test_cases.extend(suite.test_cases)
        agent_names.add(suite.agent_name)

    return TestSuite(
        name=name,
        description=f"Merged from {len(suites)} test suites",
        agent_name=", ".join(agent_names),
        test_cases=all_test_cases,
    )


# =============================================================================
# PREDEFINED TEST CASES
# =============================================================================


def create_grammateus_test_suite() -> TestSuite:
    """Create test suite for GRAMMATEUS agent."""
    test_cases = [
        TestCaseBuilder()
        .with_id("grammateus_001")
        .with_type(TestType.UNIT)
        .with_description("Basic narrative text analysis")
        .with_verse(
            "GEN.1.1",
            "In the beginning God created the heavens and the earth."
        )
        .with_context({"book": "GEN", "chapter": 1, "testament": "OT"})
        .with_expected({
            "word_count": 10,
            "text_type": "narrative",
            "has_direct_speech": False,
        }, confidence_min=0.7)
        .build(),

        TestCaseBuilder()
        .with_id("grammateus_002")
        .with_type(TestType.UNIT)
        .with_description("Direct speech detection")
        .with_verse(
            "GEN.3.1",
            'Now the serpent was more crafty than any of the wild animals the LORD God had made. He said to the woman, "Did God really say?"'
        )
        .with_context({"book": "GEN", "chapter": 3, "testament": "OT"})
        .with_expected({
            "has_direct_speech": True,
            "text_type": "narrative",
        }, confidence_min=0.7)
        .build(),

        TestCaseBuilder()
        .with_id("grammateus_003")
        .with_type(TestType.UNIT)
        .with_description("Poetry detection (Psalm)")
        .with_verse(
            "PSA.23.1",
            "The LORD is my shepherd, I lack nothing."
        )
        .with_context({"book": "PSA", "chapter": 23, "testament": "OT"})
        .with_expected({
            "text_type": "poetry",
        }, confidence_min=0.6)
        .build(),

        TestCaseBuilder()
        .with_id("grammateus_004")
        .with_type(TestType.UNIT)
        .with_description("Prophecy detection")
        .with_verse(
            "ISA.7.14",
            "Therefore the Lord himself will give you a sign: The virgin will conceive and give birth to a son, and will call him Immanuel."
        )
        .with_context({"book": "ISA", "chapter": 7, "testament": "OT"})
        .with_expected({
            "text_type": "prophecy",
            "has_quotation": False,
        }, confidence_min=0.6)
        .build(),

        TestCaseBuilder()
        .with_id("grammateus_005")
        .with_type(TestType.REGRESSION)
        .with_description("NT quotation of OT")
        .with_verse(
            "MAT.1.23",
            '"The virgin will conceive and give birth to a son, and they will call him Immanuel" (which means "God with us").'
        )
        .with_context({"book": "MAT", "chapter": 1, "testament": "NT"})
        .with_expected({
            "has_quotation": True,
        }, confidence_min=0.7)
        .build(),
    ]

    return TestSuite(
        name="grammateus_tests",
        description="Test suite for GRAMMATEUS textual analysis agent",
        agent_name="grammateus",
        test_cases=test_cases,
    )


def create_patrologos_test_suite() -> TestSuite:
    """Create test suite for PATROLOGOS agent."""
    test_cases = [
        TestCaseBuilder()
        .with_id("patrologos_001")
        .with_type(TestType.UNIT)
        .with_description("Christological theme detection")
        .with_verse(
            "JHN.1.1",
            "In the beginning was the Word, and the Word was with God, and the Word was God."
        )
        .with_context({"book": "JHN", "chapter": 1, "testament": "NT"})
        .with_expected({
            "themes": ["christology"],
            "doctrinal_significance": {"level": "high"},
        }, confidence_min=0.6)
        .build(),

        TestCaseBuilder()
        .with_id("patrologos_002")
        .with_type(TestType.UNIT)
        .with_description("Trinitarian theme detection")
        .with_verse(
            "MAT.28.19",
            "Therefore go and make disciples of all nations, baptizing them in the name of the Father and of the Son and of the Holy Spirit."
        )
        .with_context({"book": "MAT", "chapter": 28, "testament": "NT"})
        .with_expected({
            "themes": ["trinity"],
            "doctrinal_significance": {"level": "high"},
        }, confidence_min=0.6)
        .build(),

        TestCaseBuilder()
        .with_id("patrologos_003")
        .with_type(TestType.UNIT)
        .with_description("Soteriological theme detection")
        .with_verse(
            "ROM.3.24",
            "and all are justified freely by his grace through the redemption that came by Christ Jesus."
        )
        .with_context({"book": "ROM", "chapter": 3, "testament": "NT"})
        .with_expected({
            "themes": ["soteriology"],
        }, confidence_min=0.5)
        .build(),

        TestCaseBuilder()
        .with_id("patrologos_004")
        .with_type(TestType.REGRESSION)
        .with_description("Multiple themes detection")
        .with_verse(
            "JHN.3.16",
            "For God so loved the world that he gave his one and only Son, that whoever believes in him shall not perish but have eternal life."
        )
        .with_context({"book": "JHN", "chapter": 3, "testament": "NT"})
        .with_expected({
            "themes": ["christology", "soteriology"],
            "doctrinal_significance": {"level": "high"},
        }, confidence_min=0.7)
        .build(),
    ]

    return TestSuite(
        name="patrologos_tests",
        description="Test suite for PATROLOGOS patristic analysis agent",
        agent_name="patrologos",
        test_cases=test_cases,
    )


def create_syndesmos_test_suite() -> TestSuite:
    """Create test suite for SYNDESMOS cross-reference agent."""
    test_cases = [
        TestCaseBuilder()
        .with_id("syndesmos_001")
        .with_type(TestType.UNIT)
        .with_description("Typological connection: Isaac/Christ")
        .with_verse(
            "GEN.22.2",
            "Then God said, 'Take your son, your only son, whom you love--Isaac--and go to the region of Moriah. Sacrifice him there as a burnt offering.'"
        )
        .with_context({
            "book": "GEN", "chapter": 22, "testament": "OT",
            "known_types": ["sacrifice", "only_son"],
        })
        .with_expected({
            "cross_references": [
                {"target_ref": "JHN.3.16", "connection_type": "typological"},
            ],
        }, confidence_min=0.5)
        .build(),

        TestCaseBuilder()
        .with_id("syndesmos_002")
        .with_type(TestType.UNIT)
        .with_description("Verbal parallel: 'In the beginning'")
        .with_verse(
            "GEN.1.1",
            "In the beginning God created the heavens and the earth."
        )
        .with_context({"book": "GEN", "chapter": 1, "testament": "OT"})
        .with_expected({
            "cross_references": [
                {"target_ref": "JHN.1.1", "connection_type": "verbal"},
            ],
        }, confidence_min=0.5)
        .build(),

        TestCaseBuilder()
        .with_id("syndesmos_003")
        .with_type(TestType.UNIT)
        .with_description("Prophetic fulfillment")
        .with_verse(
            "ISA.53.5",
            "But he was pierced for our transgressions, he was crushed for our iniquities."
        )
        .with_context({"book": "ISA", "chapter": 53, "testament": "OT"})
        .with_expected({
            "cross_references": [
                {"connection_type": "prophetic"},
            ],
        }, confidence_min=0.5)
        .build(),

        TestCaseBuilder()
        .with_id("syndesmos_004")
        .with_type(TestType.REGRESSION)
        .with_description("Multiple cross-references")
        .with_verse(
            "HEB.11.17",
            "By faith Abraham, when God tested him, offered Isaac as a sacrifice."
        )
        .with_context({"book": "HEB", "chapter": 11, "testament": "NT"})
        .with_expected({
            "cross_references": [
                {"target_ref": "GEN.22.2", "connection_type": "historical"},
            ],
        }, confidence_min=0.6)
        .build(),
    ]

    return TestSuite(
        name="syndesmos_tests",
        description="Test suite for SYNDESMOS cross-reference discovery agent",
        agent_name="syndesmos",
        test_cases=test_cases,
    )


def create_adversarial_test_suite() -> TestSuite:
    """Create adversarial test suite for robustness testing."""
    test_cases = [
        # Empty/minimal input
        TestCaseBuilder()
        .with_id("adversarial_001")
        .with_type(TestType.ADVERSARIAL)
        .with_description("Empty text handling")
        .with_verse("GEN.1.1", "")
        .with_context({})
        .as_adversarial("empty_input")
        .with_expected({}, confidence_min=0.0)
        .with_timeout(5.0)
        .build(),

        # Very long text
        TestCaseBuilder()
        .with_id("adversarial_002")
        .with_type(TestType.ADVERSARIAL)
        .with_description("Very long text handling")
        .with_verse("PSA.119.1", "A " * 10000)  # 10k word repetition
        .with_context({"book": "PSA"})
        .as_adversarial("long_input")
        .with_expected({}, confidence_min=0.0)
        .with_timeout(60.0)
        .build(),

        # Special characters
        TestCaseBuilder()
        .with_id("adversarial_003")
        .with_type(TestType.ADVERSARIAL)
        .with_description("Special characters handling")
        .with_verse(
            "GEN.1.1",
            "In the beginning <script>alert('xss')</script> God created."
        )
        .with_context({})
        .as_adversarial("special_chars")
        .with_expected({}, confidence_min=0.0)
        .build(),

        # Non-ASCII (Hebrew/Greek)
        TestCaseBuilder()
        .with_id("adversarial_004")
        .with_type(TestType.ADVERSARIAL)
        .with_description("Non-ASCII text (Hebrew) handling")
        .with_verse(
            "GEN.1.1",
            "BQRASEN BQRASEN BQRASEN"
        )
        .with_context({"language": "hebrew"})
        .as_adversarial("unicode")
        .with_expected({}, confidence_min=0.3)
        .build(),

        # Invalid verse reference
        TestCaseBuilder()
        .with_id("adversarial_005")
        .with_type(TestType.ADVERSARIAL)
        .with_description("Invalid verse reference handling")
        .with_verse("INVALID.999.999", "Some text here.")
        .with_context({})
        .as_adversarial("invalid_ref")
        .with_expected({}, confidence_min=0.0)
        .build(),
    ]

    return TestSuite(
        name="adversarial_tests",
        description="Adversarial test suite for robustness testing",
        agent_name="all",
        test_cases=test_cases,
    )


# =============================================================================
# TEST SUITE REGISTRY
# =============================================================================


TEST_SUITE_REGISTRY: Dict[str, callable] = {
    "grammateus": create_grammateus_test_suite,
    "patrologos": create_patrologos_test_suite,
    "syndesmos": create_syndesmos_test_suite,
    "adversarial": create_adversarial_test_suite,
}


def get_test_suite(name: str) -> TestSuite:
    """
    Get a predefined test suite by name.

    Args:
        name: Name of the test suite

    Returns:
        TestSuite instance
    """
    if name not in TEST_SUITE_REGISTRY:
        raise ValueError(
            f"Unknown test suite: {name}. "
            f"Available: {list(TEST_SUITE_REGISTRY.keys())}"
        )

    return TEST_SUITE_REGISTRY[name]()


def list_available_test_suites() -> List[str]:
    """List all available test suite names."""
    return list(TEST_SUITE_REGISTRY.keys())

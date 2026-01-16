"""
BIBLOS v2 - Test Configuration

Pytest fixtures and configuration for all tests.
"""
import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any, Generator
import json


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_verse_id() -> str:
    """Sample verse ID for testing."""
    return "GEN.1.1"


@pytest.fixture
def sample_verse_text() -> str:
    """Sample verse text for testing."""
    return "In the beginning God created the heaven and the earth."


@pytest.fixture
def sample_hebrew_text() -> str:
    """Sample Hebrew text for testing."""
    return "בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ"


@pytest.fixture
def sample_greek_text() -> str:
    """Sample Greek text for testing."""
    return "Ἐν ἀρχῇ ἦν ὁ λόγος καὶ ὁ λόγος ἦν πρὸς τὸν θεόν"


@pytest.fixture
def sample_context() -> Dict[str, Any]:
    """Sample pipeline context."""
    return {
        "verse_id": "GEN.1.1",
        "text": "In the beginning God created the heaven and the earth.",
        "metadata": {
            "book": "GEN",
            "chapter": 1,
            "verse": 1,
            "testament": "OT"
        },
        "agent_results": {},
        "phase_results": {}
    }


@pytest.fixture
def sample_linguistic_context(sample_context) -> Dict[str, Any]:
    """Context with linguistic results."""
    return {
        **sample_context,
        "linguistic_results": {
            "structural": {
                "sentence_count": 1,
                "word_count": 10,
                "clause_types": ["declarative"]
            },
            "morphological": {
                "word_analyses": [
                    {"word": "In", "pos": "preposition"},
                    {"word": "beginning", "pos": "noun"},
                    {"word": "God", "pos": "noun"},
                    {"word": "created", "pos": "verb"}
                ]
            }
        },
        "agent_results": {
            "grammateus": {
                "data": {"word_count": 10, "sentence_count": 1},
                "confidence": 0.95
            },
            "morphologos": {
                "data": {"analyses": []},
                "confidence": 0.9
            }
        }
    }


@pytest.fixture
def sample_theological_context(sample_linguistic_context) -> Dict[str, Any]:
    """Context with theological results."""
    return {
        **sample_linguistic_context,
        "theological_results": {
            "patristic": {
                "citations": [
                    {"father": "Augustine", "work": "Confessions"}
                ]
            },
            "typological": {
                "connections": []
            }
        },
        "agent_results": {
            **sample_linguistic_context["agent_results"],
            "patrologos": {
                "data": {"citations": []},
                "confidence": 0.85
            },
            "theologos": {
                "data": {"themes": ["creation"]},
                "confidence": 0.9
            }
        }
    }


@pytest.fixture
def sample_cross_reference() -> Dict[str, Any]:
    """Sample cross-reference data."""
    return {
        "source_ref": "GEN.1.1",
        "target_ref": "JHN.1.1",
        "connection_type": "typological",
        "strength": "strong",
        "confidence": 0.85,
        "notes": ["In the beginning - verbal parallel"]
    }


@pytest.fixture
def sample_golden_record() -> Dict[str, Any]:
    """Sample golden record."""
    return {
        "verse_id": "GEN.1.1",
        "text": "In the beginning God created the heaven and the earth.",
        "data": {
            "structural": {"word_count": 10},
            "morphological": {"analyses": []},
            "patristic": {"citations": []},
            "cross_references": []
        },
        "created_at": "2024-01-01T00:00:00Z",
        "version": "2.0.0"
    }


@pytest.fixture
def tmp_data_dir(tmp_path) -> Path:
    """Temporary data directory for tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def mock_corpus_data(tmp_data_dir) -> Path:
    """Create mock corpus data files."""
    corpus_dir = tmp_data_dir / "corpus"
    corpus_dir.mkdir()

    # Create sample verse file
    verse_data = {
        "verse_id": "GEN.1.1",
        "text": "In the beginning God created the heaven and the earth.",
        "words": [
            {"surface": "In", "lemma": "in", "morph": "P"},
            {"surface": "beginning", "lemma": "beginning", "morph": "N-NSF"}
        ]
    }

    with open(corpus_dir / "GEN.json", "w") as f:
        json.dump([verse_data], f)

    return corpus_dir


# Property test fixtures
@pytest.fixture
def hypothesis_settings_profile():
    """Get current Hypothesis settings profile."""
    from hypothesis import settings
    return settings.default


# Theological Test Fixtures
@pytest.fixture
def mock_patristic_witnesses():
    """Mock patristic witness data for testing."""
    return [
        {
            "father_name": "Basil the Great",
            "authority": "ecumenical_father",
            "quote": "'In the beginning' signifies the co-eternal Son through whom all was made",
            "source_work": "Hexaemeron, Homily 1",
            "verse_reference": "GEN.1.1"
        },
        {
            "father_name": "John Chrysostom",
            "authority": "ecumenical_father",
            "quote": "The Word present in Genesis is the same Logos incarnate in John",
            "source_work": "Homilies on Genesis",
            "verse_reference": "GEN.1.1"
        },
        {
            "father_name": "Justin Martyr",
            "authority": "major_father",
            "quote": "The virgin shall conceive refers to Mary, mother of Christ",
            "source_work": "Dialogue with Trypho, 84",
            "verse_reference": "ISA.7.14"
        }
    ]


# Test data validation helpers
def assert_valid_verse_id(verse_id: str):
    """Assert that verse ID follows valid format."""
    parts = verse_id.split('.')
    assert len(parts) == 3, f"Invalid verse ID format: {verse_id}"
    assert len(parts[0]) == 3, f"Invalid book code: {parts[0]}"
    assert parts[1].isdigit(), f"Invalid chapter: {parts[1]}"
    assert parts[2].isdigit(), f"Invalid verse: {parts[2]}"


def assert_valid_confidence(confidence: float):
    """Assert that confidence score is in valid range."""
    assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of range [0, 1]"


def assert_valid_connection_type(connection_type: str):
    """Assert that connection type is valid."""
    valid_types = {
        "typological", "prophetic", "verbal", "thematic",
        "conceptual", "historical", "liturgical", "narrative",
        "genealogical", "geographical"
    }
    assert connection_type in valid_types, f"Invalid connection type: {connection_type}"


# Markers
def pytest_configure(config):
    """Configure pytest markers."""
    # Existing markers
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "db: marks database tests")
    config.addinivalue_line("markers", "ml: marks ML tests")
    config.addinivalue_line("markers", "property: marks property-based tests using Hypothesis")
    config.addinivalue_line("markers", "stateful: marks stateful property tests")

    # Theological confidence level markers
    config.addinivalue_line("markers", "dogmatic: Tests at DOGMATIC confidence level (100% pass rate required)")
    config.addinivalue_line("markers", "consensus: Tests at CONSENSUS confidence level (98% pass rate required)")
    config.addinivalue_line("markers", "traditional: Tests at TRADITIONAL confidence level (90% pass rate required)")
    config.addinivalue_line("markers", "scholarly: Tests at SCHOLARLY confidence level (85% pass rate required)")
    config.addinivalue_line("markers", "exploratory: Tests at EXPLORATORY confidence level (75% pass rate required)")

    # Additional test category markers
    config.addinivalue_line("markers", "e2e: End-to-end tests covering complete workflows")
    config.addinivalue_line("markers", "performance: Performance and SLO tests")

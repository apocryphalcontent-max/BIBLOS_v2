"""
BIBLOS v2 - Multi-Agent Extraction System (SDES)
Scripture Data Extraction System with 24 specialized agents.

Agent Categories:
- Linguistic: morphology, syntax, semantics, phonology, discourse
- Theological: patristic, typological, doctrinal, liturgical
- Intertextual: cross-references, parallels, allusions
- Validation: correction, verification, adversarial testing
- Scrapers: content ingestion with garbage filtering
"""

from agents.base import (
    BaseExtractionAgent,
    AgentConfig,
    ExtractionResult,
    ExtractionType,
    ProcessingStatus,
)
from agents.registry import AgentRegistry
from agents.orchestrator import AgentOrchestrator

# Scraper agents for content population
from agents.scrapers import (
    BaseScraperAgent,
    ScraperConfig,
    ScraperResult,
    ContentQuality,
    GarbageFilter,
    PatristicScraperAgent,
    TextCleanerAgent,
)

__version__ = "2.0.0"
__all__ = [
    # Core extraction agents
    "BaseExtractionAgent",
    "AgentConfig",
    "ExtractionResult",
    "ExtractionType",
    "ProcessingStatus",
    "AgentRegistry",
    "AgentOrchestrator",
    # Scraper agents
    "BaseScraperAgent",
    "ScraperConfig",
    "ScraperResult",
    "ContentQuality",
    "GarbageFilter",
    "PatristicScraperAgent",
    "TextCleanerAgent",
]

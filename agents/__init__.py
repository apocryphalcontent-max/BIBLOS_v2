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
# Use LangGraphOrchestrator as the canonical orchestrator, with AgentOrchestrator alias
from agents.langgraph_orchestrator import LangGraphOrchestrator, OrchestrationConfig
AgentOrchestrator = LangGraphOrchestrator  # Backwards compatibility alias

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
    # Orchestrators
    "LangGraphOrchestrator",
    "OrchestrationConfig",
    "AgentOrchestrator",  # Backwards compatibility alias
    # Scraper agents
    "BaseScraperAgent",
    "ScraperConfig",
    "ScraperResult",
    "ContentQuality",
    "GarbageFilter",
    "PatristicScraperAgent",
    "TextCleanerAgent",
]

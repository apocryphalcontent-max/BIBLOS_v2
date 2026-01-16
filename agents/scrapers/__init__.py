"""
BIBLOS v2 - Scraper Agents

Specialized agents for scraping, ingesting, and filtering patristic
and theological source materials.
"""
from agents.scrapers.base_scraper import (
    BaseScraperAgent,
    ScraperConfig,
    ScraperResult,
    ContentQuality,
    GarbageFilter,
)
from agents.scrapers.patristic_scraper import PatristicScraperAgent
from agents.scrapers.text_cleaner import TextCleanerAgent


__all__ = [
    "BaseScraperAgent",
    "ScraperConfig",
    "ScraperResult",
    "ContentQuality",
    "GarbageFilter",
    "PatristicScraperAgent",
    "TextCleanerAgent",
]

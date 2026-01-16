"""
BIBLOS v2 - External Integrations

Provides integration with external biblical text corpora:
- Text-Fabric: Hebrew Bible (BHSA) and Greek NT
- Macula: Greek and Hebrew morphological data
"""
from integrations.base import (
    BaseCorpusIntegration,
    VerseData,
    WordData,
    MorphologyData
)
from integrations.text_fabric import TextFabricIntegration
from integrations.macula import MaculaIntegration

__all__ = [
    "BaseCorpusIntegration",
    "VerseData",
    "WordData",
    "MorphologyData",
    "TextFabricIntegration",
    "MaculaIntegration"
]

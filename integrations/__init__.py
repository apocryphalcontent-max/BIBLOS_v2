"""
BIBLOS v2 - External Integrations

Provides integration with external biblical text corpora:
- Text-Fabric: Hebrew Bible (BHSA) and Greek NT
- Macula: Greek and Hebrew morphological data
- LXX Corpus: Septuagint with morphology and variants
"""
from integrations.base import (
    BaseCorpusIntegration,
    VerseData,
    WordData,
    MorphologyData
)
from integrations.text_fabric import TextFabricIntegration
from integrations.macula import MaculaIntegration
from integrations.lxx_corpus import (
    LXXCorpusClient,
    LXXWord,
    LXXVerse
)

__all__ = [
    "BaseCorpusIntegration",
    "VerseData",
    "WordData",
    "MorphologyData",
    "TextFabricIntegration",
    "MaculaIntegration",
    "LXXCorpusClient",
    "LXXWord",
    "LXXVerse"
]

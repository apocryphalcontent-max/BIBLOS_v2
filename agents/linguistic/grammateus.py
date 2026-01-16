"""
BIBLOS v2 - GRAMMATEUS Agent

Primary textual analysis coordinator for biblical texts.
Coordinates and synthesizes output from other linguistic agents.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import re

from agents.base import (
    BaseExtractionAgent,
    AgentConfig,
    ExtractionResult,
    ExtractionType,
    ProcessingStatus
)


@dataclass
class TextualFeatures:
    """Extracted textual features."""
    word_count: int
    unique_words: int
    avg_word_length: float
    sentence_count: int
    has_direct_speech: bool
    has_quotation: bool
    rhetorical_devices: List[str]
    text_type: str  # narrative, discourse, poetry, prophecy, law


class GramateusAgent(BaseExtractionAgent):
    """
    GRAMMATEUS - Master textual analysis agent.

    Responsibilities:
    - Text segmentation and tokenization
    - Genre/type classification
    - Rhetorical device identification
    - Coordination of linguistic analysis
    - Synthesis of agent outputs
    """

    # Rhetorical patterns
    RHETORICAL_PATTERNS = {
        "chiasm": r"(\w+).*?(\w+).*?\2.*?\1",  # A-B-B'-A' pattern
        "inclusio": r"^(.{10,}).*\1$",  # Same at start and end
        "anaphora": r"(\b\w+\b)(?:\s+\S+){0,5}\s+\1",  # Word repetition
        "parallelism": r"(.+?[,;:])\s*(.+?[,;:])",  # Parallel clauses
    }

    # Text type indicators
    TEXT_TYPE_KEYWORDS = {
        "narrative": ["and", "then", "after", "when", "said", "went"],
        "discourse": ["therefore", "thus", "for", "because", "but"],
        "poetry": ["selah", "psalm", "song", "praise"],
        "prophecy": ["oracle", "vision", "says the lord", "behold"],
        "law": ["shall", "must", "commandment", "statute", "ordinance"]
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="grammateus",
                extraction_type=ExtractionType.STRUCTURAL,
                batch_size=500,
                min_confidence=0.6
            )
        super().__init__(config)
        self.logger = logging.getLogger("biblos.agents.grammateus")

    async def extract(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> ExtractionResult:
        """
        Extract textual features from verse.

        Args:
            verse_id: Canonical verse ID (e.g., GEN.1.1)
            text: Verse text
            context: Additional context (book info, surrounding verses)

        Returns:
            ExtractionResult with textual analysis
        """
        features = self._analyze_text(text, context)
        rhetorical = self._identify_rhetorical_devices(text)
        text_type = self._classify_text_type(text, context)

        data = {
            "word_count": features.word_count,
            "unique_words": features.unique_words,
            "avg_word_length": features.avg_word_length,
            "sentence_count": features.sentence_count,
            "has_direct_speech": features.has_direct_speech,
            "has_quotation": features.has_quotation,
            "rhetorical_devices": rhetorical,
            "text_type": text_type,
            "tokens": self._tokenize(text),
            "clauses": self._segment_clauses(text)
        }

        confidence = self._calculate_confidence(data)

        return ExtractionResult(
            agent_name=self.config.name,
            extraction_type=self.config.extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.COMPLETED,
            data=data,
            confidence=confidence
        )

    def _analyze_text(self, text: str, context: Dict[str, Any]) -> TextualFeatures:
        """Analyze basic textual features."""
        words = self._tokenize(text)
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        return TextualFeatures(
            word_count=len(words),
            unique_words=len(set(words)),
            avg_word_length=sum(len(w) for w in words) / max(1, len(words)),
            sentence_count=len(sentences),
            has_direct_speech='"' in text or "'" in text or ":" in text,
            has_quotation=self._detect_quotation(text, context),
            rhetorical_devices=[],
            text_type=""
        )

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Handle Greek, Hebrew, and Latin characters
        pattern = r'[\w\u0370-\u03FF\u1F00-\u1FFF\u0590-\u05FF]+'
        return re.findall(pattern, text.lower())

    def _segment_clauses(self, text: str) -> List[str]:
        """Segment text into clauses."""
        # Split on clause boundaries
        clauses = re.split(r'[,;:.]|\band\b|\bbut\b|\bfor\b', text)
        return [c.strip() for c in clauses if c.strip()]

    def _identify_rhetorical_devices(self, text: str) -> List[str]:
        """Identify rhetorical devices in text."""
        devices = []

        for device, pattern in self.RHETORICAL_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                devices.append(device)

        # Check for repetition
        words = self._tokenize(text)
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        if any(count >= 3 for count in word_counts.values()):
            devices.append("repetition")

        return devices

    def _classify_text_type(self, text: str, context: Dict[str, Any]) -> str:
        """Classify the type of text."""
        text_lower = text.lower()
        scores = {}

        for text_type, keywords in self.TEXT_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[text_type] = score

        # Consider book context
        book = context.get("book", "")
        if book in ["PSA", "PRO", "SNG", "LAM"]:
            scores["poetry"] = scores.get("poetry", 0) + 3
        elif book in ["ISA", "JER", "EZK", "DAN", "HOS", "JOL", "AMO"]:
            scores["prophecy"] = scores.get("prophecy", 0) + 2
        elif book in ["LEV", "DEU"]:
            scores["law"] = scores.get("law", 0) + 2

        if scores:
            return max(scores.keys(), key=lambda k: scores[k])
        return "narrative"

    def _detect_quotation(self, text: str, context: Dict[str, Any]) -> bool:
        """Detect if verse contains an OT quotation in NT context."""
        # Check for quotation markers
        if any(marker in text.lower() for marker in [
            "it is written", "scripture says", "as it says",
            "according to", "fulfilled", "spoken by"
        ]):
            return True

        # Check context for quotation info
        return context.get("is_quotation", False)

    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence score for extraction."""
        confidence = 0.7  # Base confidence

        # Adjust based on data quality
        if data["word_count"] > 0:
            confidence += 0.1
        if data["rhetorical_devices"]:
            confidence += 0.1
        if data["text_type"] != "narrative":  # More specific classification
            confidence += 0.1

        return min(1.0, confidence)

    async def validate(self, result: ExtractionResult) -> bool:
        """Validate extraction result."""
        data = result.data

        # Check required fields
        if not all(key in data for key in ["word_count", "text_type", "tokens"]):
            return False

        # Check data consistency
        if data["word_count"] != len(data.get("tokens", [])):
            return False

        return True

    def get_dependencies(self) -> List[str]:
        """Get agent dependencies."""
        return []  # GRAMMATEUS is the coordinator, no dependencies

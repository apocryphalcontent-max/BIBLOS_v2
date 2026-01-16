"""
BIBLOS v2 - TOPOS Agent

Common topic and motif analysis for biblical texts.
"""
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

from agents.base import (
    BaseExtractionAgent,
    AgentConfig,
    ExtractionResult,
    ExtractionType,
    ProcessingStatus
)


class MotifCategory(Enum):
    """Categories of biblical motifs."""
    DIVINE_ACTION = "divine_action"  # God's actions
    HUMAN_RESPONSE = "human_response"  # Human responses
    COSMIC = "cosmic"  # Cosmic/creation
    CULTIC = "cultic"  # Worship/ritual
    SOCIAL = "social"  # Social structures
    ESCHATOLOGICAL = "eschatological"  # End times
    CHRISTOLOGICAL = "christological"  # Christ-centered


class TopicType(Enum):
    """Types of biblical topics."""
    THEME = "theme"  # Major theme
    MOTIF = "motif"  # Recurring pattern
    TOPOS = "topos"  # Common topic
    FORMULA = "formula"  # Set phrase
    IMAGE = "image"  # Imagery


@dataclass
class BiblicalMotif:
    """A biblical motif or topic."""
    name: str
    category: MotifCategory
    topic_type: TopicType
    keywords: List[str]
    occurrences: List[str]
    theological_significance: str
    frequency: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "topic_type": self.topic_type.value,
            "keywords": self.keywords,
            "occurrences": self.occurrences,
            "theological_significance": self.theological_significance,
            "frequency": self.frequency
        }


class ToposAgent(BaseExtractionAgent):
    """
    TOPOS - Motif/topic analysis agent.

    Performs:
    - Motif identification
    - Topic classification
    - Pattern tracking
    - Theme mapping
    - Imagery analysis
    """

    # Biblical motifs and their characteristics
    BIBLICAL_MOTIFS = {
        "exodus": {
            "category": MotifCategory.DIVINE_ACTION,
            "keywords": ["exodus", "deliver", "egypt", "bondage", "freedom", "redeem"],
            "significance": "Divine deliverance from oppression"
        },
        "covenant": {
            "category": MotifCategory.DIVINE_ACTION,
            "keywords": ["covenant", "promise", "oath", "sworn", "faithful", "steadfast"],
            "significance": "God's binding relationship with His people"
        },
        "wilderness": {
            "category": MotifCategory.HUMAN_RESPONSE,
            "keywords": ["wilderness", "desert", "wander", "forty", "testing", "manna"],
            "significance": "Testing and formation in isolation"
        },
        "shepherd": {
            "category": MotifCategory.DIVINE_ACTION,
            "keywords": ["shepherd", "flock", "pasture", "sheep", "lead", "guide"],
            "significance": "Divine care and guidance"
        },
        "vineyard": {
            "category": MotifCategory.DIVINE_ACTION,
            "keywords": ["vineyard", "vine", "grapes", "fruit", "branches", "prune"],
            "significance": "God's people under cultivation"
        },
        "temple": {
            "category": MotifCategory.CULTIC,
            "keywords": ["temple", "house", "sanctuary", "dwelling", "presence", "glory"],
            "significance": "Divine presence among humanity"
        },
        "sacrifice": {
            "category": MotifCategory.CULTIC,
            "keywords": ["sacrifice", "offering", "blood", "altar", "atonement", "lamb"],
            "significance": "Substitutionary atonement"
        },
        "kingdom": {
            "category": MotifCategory.ESCHATOLOGICAL,
            "keywords": ["kingdom", "reign", "throne", "king", "rule", "dominion"],
            "significance": "Divine sovereignty and rule"
        },
        "day_of_lord": {
            "category": MotifCategory.ESCHATOLOGICAL,
            "keywords": ["day", "lord", "coming", "judgment", "wrath", "salvation"],
            "significance": "Eschatological intervention"
        },
        "remnant": {
            "category": MotifCategory.HUMAN_RESPONSE,
            "keywords": ["remnant", "few", "survive", "preserve", "remain", "faithful"],
            "significance": "Faithful minority preserved"
        },
        "new_creation": {
            "category": MotifCategory.COSMIC,
            "keywords": ["new", "heaven", "earth", "create", "renew", "restore"],
            "significance": "Cosmic renewal and restoration"
        },
        "light_darkness": {
            "category": MotifCategory.COSMIC,
            "keywords": ["light", "darkness", "shine", "blind", "see", "illuminate"],
            "significance": "Revelation vs ignorance"
        },
        "water": {
            "category": MotifCategory.COSMIC,
            "keywords": ["water", "sea", "river", "flood", "thirst", "drink"],
            "significance": "Life, chaos, and cleansing"
        },
        "bread": {
            "category": MotifCategory.CULTIC,
            "keywords": ["bread", "eat", "food", "hunger", "satisfy", "feast"],
            "significance": "Sustenance and communion"
        },
        "servant": {
            "category": MotifCategory.CHRISTOLOGICAL,
            "keywords": ["servant", "slave", "humble", "obey", "submit", "service"],
            "significance": "Humble obedience and mission"
        },
        "son": {
            "category": MotifCategory.CHRISTOLOGICAL,
            "keywords": ["son", "heir", "firstborn", "beloved", "only"],
            "significance": "Divine sonship and inheritance"
        },
        "bride_bridegroom": {
            "category": MotifCategory.SOCIAL,
            "keywords": ["bride", "bridegroom", "wedding", "marriage", "husband", "wife"],
            "significance": "Covenant relationship imagery"
        },
        "warfare": {
            "category": MotifCategory.COSMIC,
            "keywords": ["war", "battle", "fight", "enemy", "victory", "conquer"],
            "significance": "Spiritual conflict"
        }
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="topos",
                extraction_type=ExtractionType.INTERTEXTUAL,
                batch_size=200,
                min_confidence=0.6
            )
        super().__init__(config)
        self.logger = logging.getLogger("biblos.agents.topos")

    async def extract(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> ExtractionResult:
        """Extract motif/topic analysis."""
        # Identify motifs
        motifs = self._identify_motifs(text, verse_id)

        # Classify topics
        topics = self._classify_topics(motifs)

        # Analyze patterns
        patterns = self._analyze_patterns(motifs, context)

        # Map themes
        themes = self._map_themes(motifs)

        # Extract imagery
        imagery = self._extract_imagery(text)

        data = {
            "motifs": [m.to_dict() for m in motifs],
            "topics": topics,
            "patterns": patterns,
            "themes": themes,
            "imagery": imagery,
            "motif_density": self._calculate_density(motifs),
            "dominant_category": self._find_dominant_category(motifs)
        }

        confidence = self._calculate_confidence(motifs)

        return ExtractionResult(
            agent_name=self.config.name,
            extraction_type=self.config.extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.COMPLETED,
            data=data,
            confidence=confidence
        )

    def _identify_motifs(
        self,
        text: str,
        verse_id: str
    ) -> List[BiblicalMotif]:
        """Identify biblical motifs in text."""
        motifs = []
        text_lower = text.lower()

        for motif_name, info in self.BIBLICAL_MOTIFS.items():
            matching_keywords = [
                kw for kw in info["keywords"]
                if kw in text_lower
            ]

            if matching_keywords:
                motifs.append(BiblicalMotif(
                    name=motif_name,
                    category=info["category"],
                    topic_type=TopicType.MOTIF,
                    keywords=matching_keywords,
                    occurrences=[verse_id],
                    theological_significance=info["significance"],
                    frequency=len(matching_keywords) / len(info["keywords"])
                ))

        return motifs

    def _classify_topics(
        self,
        motifs: List[BiblicalMotif]
    ) -> Dict[str, Any]:
        """Classify topics from identified motifs."""
        by_category: Dict[str, List[str]] = {}

        for motif in motifs:
            cat = motif.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(motif.name)

        return {
            "by_category": by_category,
            "total_motifs": len(motifs),
            "unique_categories": list(by_category.keys())
        }

    def _analyze_patterns(
        self,
        motifs: List[BiblicalMotif],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze patterns in motifs."""
        if not motifs:
            return {
                "recurring": [],
                "dominant": None,
                "co_occurrence": []
            }

        # Find recurring motifs (based on frequency)
        recurring = [
            m.name for m in motifs
            if m.frequency > 0.3
        ]

        # Find dominant motif
        dominant = max(motifs, key=lambda m: m.frequency) if motifs else None

        # Find co-occurring motifs
        co_occurrence = []
        if len(motifs) >= 2:
            co_occurrence = [
                {"motif1": motifs[i].name, "motif2": motifs[j].name}
                for i in range(len(motifs))
                for j in range(i + 1, len(motifs))
            ]

        return {
            "recurring": recurring,
            "dominant": dominant.name if dominant else None,
            "co_occurrence": co_occurrence[:5]  # Limit to 5
        }

    def _map_themes(
        self,
        motifs: List[BiblicalMotif]
    ) -> List[Dict[str, Any]]:
        """Map motifs to broader themes."""
        theme_mapping = {
            MotifCategory.DIVINE_ACTION: "Divine Intervention",
            MotifCategory.HUMAN_RESPONSE: "Human Faith and Failure",
            MotifCategory.COSMIC: "Creation and Cosmos",
            MotifCategory.CULTIC: "Worship and Sacrifice",
            MotifCategory.SOCIAL: "Community and Covenant",
            MotifCategory.ESCHATOLOGICAL: "End Times and Fulfillment",
            MotifCategory.CHRISTOLOGICAL: "Christ and Messianic Hope"
        }

        themes = []
        seen_themes: Set[str] = set()

        for motif in motifs:
            theme = theme_mapping.get(motif.category, "General")
            if theme not in seen_themes:
                themes.append({
                    "theme": theme,
                    "motifs": [m.name for m in motifs if m.category == motif.category],
                    "category": motif.category.value
                })
                seen_themes.add(theme)

        return themes

    def _extract_imagery(self, text: str) -> List[Dict[str, str]]:
        """Extract imagery from text."""
        imagery = []
        text_lower = text.lower()

        # Image categories and their keywords
        image_types = {
            "natural": ["mountain", "sea", "river", "tree", "rock", "fire", "wind"],
            "agricultural": ["seed", "harvest", "vineyard", "fruit", "wheat", "field"],
            "architectural": ["house", "temple", "city", "gate", "foundation", "wall"],
            "royal": ["throne", "crown", "scepter", "palace", "kingdom"],
            "domestic": ["bread", "wine", "oil", "lamp", "door", "garment"],
            "martial": ["sword", "shield", "armor", "battle", "victory"]
        }

        for img_type, keywords in image_types.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if matches:
                imagery.append({
                    "type": img_type,
                    "images": matches
                })

        return imagery

    def _calculate_density(
        self,
        motifs: List[BiblicalMotif]
    ) -> float:
        """Calculate motif density."""
        if not motifs:
            return 0.0

        # Base density on count and frequency
        base = len(motifs) * 0.15
        freq_bonus = sum(m.frequency for m in motifs) * 0.1

        return min(1.0, base + freq_bonus + 0.2)

    def _find_dominant_category(
        self,
        motifs: List[BiblicalMotif]
    ) -> Optional[str]:
        """Find the dominant motif category."""
        if not motifs:
            return None

        categories = [m.category.value for m in motifs]
        category_counts: Dict[str, int] = {}
        for c in categories:
            category_counts[c] = category_counts.get(c, 0) + 1

        return max(category_counts, key=category_counts.get) if category_counts else None

    def _calculate_confidence(
        self,
        motifs: List[BiblicalMotif]
    ) -> float:
        """Calculate confidence score."""
        confidence = 0.5

        if motifs:
            confidence += 0.2
            # Boost for high-frequency motifs
            high_freq = sum(1 for m in motifs if m.frequency > 0.5)
            confidence += high_freq * 0.1

        return min(1.0, confidence)

    async def validate(self, result: ExtractionResult) -> bool:
        """Validate extraction result."""
        data = result.data
        return "motifs" in data and "themes" in data

    def get_dependencies(self) -> List[str]:
        """Get agent dependencies."""
        return ["grammateus", "semantikos"]

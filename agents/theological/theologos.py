"""
BIBLOS v2 - THEOLOGOS Agent

Systematic theology extraction from biblical texts.
"""
from typing import Dict, List, Any, Optional
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


class DoctrinalCategory(Enum):
    """Major doctrinal categories."""
    THEOLOGY_PROPER = "theology_proper"  # Doctrine of God
    CHRISTOLOGY = "christology"  # Doctrine of Christ
    PNEUMATOLOGY = "pneumatology"  # Doctrine of Holy Spirit
    ANTHROPOLOGY = "anthropology"  # Doctrine of humanity
    HAMARTIOLOGY = "hamartiology"  # Doctrine of sin
    SOTERIOLOGY = "soteriology"  # Doctrine of salvation
    ECCLESIOLOGY = "ecclesiology"  # Doctrine of church
    ESCHATOLOGY = "eschatology"  # Doctrine of last things
    ANGELOLOGY = "angelology"  # Doctrine of angels
    BIBLIOLOGY = "bibliology"  # Doctrine of Scripture


class TheologicalWeight(Enum):
    """Weight of theological significance."""
    FOUNDATIONAL = "foundational"  # Core doctrinal verse
    SUPPORTING = "supporting"  # Supports doctrine
    ILLUSTRATIVE = "illustrative"  # Illustrates doctrine
    CONTEXTUAL = "contextual"  # Provides context


@dataclass
class DoctrinalAssertion:
    """A doctrinal assertion extracted from text."""
    category: DoctrinalCategory
    assertion: str
    weight: TheologicalWeight
    related_doctrines: List[str]
    creedal_connection: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "assertion": self.assertion,
            "weight": self.weight.value,
            "related_doctrines": self.related_doctrines,
            "creedal_connection": self.creedal_connection
        }


class TheologosAgent(BaseExtractionAgent):
    """
    THEOLOGOS - Systematic theology agent.

    Performs:
    - Doctrinal category classification
    - Theological assertion extraction
    - Creedal mapping
    - Dogmatic significance assessment
    - Theological term analysis
    """

    # Key terms for each doctrinal category
    CATEGORY_TERMS = {
        DoctrinalCategory.THEOLOGY_PROPER: [
            "god", "lord", "almighty", "eternal", "omnipotent", "omniscient",
            "creator", "father", "holy", "righteous", "merciful", "sovereign"
        ],
        DoctrinalCategory.CHRISTOLOGY: [
            "christ", "jesus", "son", "logos", "word", "messiah", "lord",
            "savior", "lamb", "king", "priest", "mediator", "incarnation"
        ],
        DoctrinalCategory.PNEUMATOLOGY: [
            "spirit", "holy spirit", "paraclete", "comforter", "advocate",
            "breath", "wind", "anointing", "gifts", "fruit"
        ],
        DoctrinalCategory.ANTHROPOLOGY: [
            "man", "human", "adam", "image", "likeness", "soul", "body",
            "flesh", "heart", "mind", "spirit", "created"
        ],
        DoctrinalCategory.HAMARTIOLOGY: [
            "sin", "transgression", "iniquity", "fall", "death", "curse",
            "wicked", "evil", "unrighteousness", "guilt", "corruption"
        ],
        DoctrinalCategory.SOTERIOLOGY: [
            "save", "salvation", "redeem", "redemption", "justify", "sanctify",
            "grace", "faith", "atonement", "forgive", "reconcile", "ransom"
        ],
        DoctrinalCategory.ECCLESIOLOGY: [
            "church", "assembly", "body", "bride", "flock", "temple",
            "baptism", "communion", "eucharist", "bishop", "deacon", "elder"
        ],
        DoctrinalCategory.ESCHATOLOGY: [
            "kingdom", "heaven", "judgment", "resurrection", "parousia",
            "eternal", "life", "death", "hell", "paradise", "new"
        ],
        DoctrinalCategory.ANGELOLOGY: [
            "angel", "seraph", "cherub", "archangel", "michael", "gabriel",
            "demon", "satan", "devil", "principalities", "powers"
        ],
        DoctrinalCategory.BIBLIOLOGY: [
            "scripture", "word", "law", "commandment", "prophecy",
            "inspiration", "revelation", "written", "testimony"
        ]
    }

    # Creedal connections
    CREEDAL_PHRASES = {
        "maker of heaven and earth": "Nicene Creed - Creator",
        "only begotten son": "Nicene Creed - Christology",
        "begotten not made": "Nicene Creed - Christology",
        "consubstantial": "Nicene Creed - Trinity",
        "proceeds from the father": "Nicene Creed - Pneumatology",
        "one baptism": "Nicene Creed - Ecclesiology",
        "resurrection of the dead": "Nicene Creed - Eschatology",
        "life of the world to come": "Nicene Creed - Eschatology"
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="theologos",
                extraction_type=ExtractionType.THEOLOGICAL,
                batch_size=200,
                min_confidence=0.6
            )
        super().__init__(config)
        self.logger = logging.getLogger("biblos.agents.theologos")

    async def extract(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> ExtractionResult:
        """Extract systematic theology from verse."""
        # Classify doctrinal categories
        categories = self._classify_categories(text)

        # Extract doctrinal assertions
        assertions = self._extract_assertions(text, categories)

        # Map to creeds
        creedal = self._map_to_creeds(text)

        # Analyze theological terms
        terms = self._analyze_terms(text)

        # Assess significance
        significance = self._assess_significance(categories, assertions)

        data = {
            "categories": [c.value for c in categories],
            "assertions": [a.to_dict() for a in assertions],
            "creedal_connections": creedal,
            "theological_terms": terms,
            "significance": significance,
            "trinitarian_content": self._check_trinitarian(text),
            "soteriological_content": self._check_soteriological(text)
        }

        confidence = self._calculate_confidence(categories, assertions)

        return ExtractionResult(
            agent_name=self.config.name,
            extraction_type=self.config.extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.COMPLETED,
            data=data,
            confidence=confidence
        )

    def _classify_categories(self, text: str) -> List[DoctrinalCategory]:
        """Classify text into doctrinal categories."""
        categories = []
        text_lower = text.lower()

        for category, terms in self.CATEGORY_TERMS.items():
            if any(term in text_lower for term in terms):
                categories.append(category)

        return categories if categories else [DoctrinalCategory.THEOLOGY_PROPER]

    def _extract_assertions(
        self,
        text: str,
        categories: List[DoctrinalCategory]
    ) -> List[DoctrinalAssertion]:
        """Extract doctrinal assertions from text."""
        assertions = []
        text_lower = text.lower()

        for category in categories:
            # Generate assertion based on category
            assertion_text = self._generate_assertion(text, category)
            weight = self._determine_weight(text, category)
            related = self._find_related_doctrines(category)
            creedal = self._find_creedal_connection(text)

            assertions.append(DoctrinalAssertion(
                category=category,
                assertion=assertion_text,
                weight=weight,
                related_doctrines=related,
                creedal_connection=creedal
            ))

        return assertions

    def _generate_assertion(
        self,
        text: str,
        category: DoctrinalCategory
    ) -> str:
        """Generate a doctrinal assertion summary."""
        # Simplified assertion generation
        category_assertions = {
            DoctrinalCategory.THEOLOGY_PROPER: "Reveals attributes of God",
            DoctrinalCategory.CHRISTOLOGY: "Teaches about Christ",
            DoctrinalCategory.PNEUMATOLOGY: "Speaks of the Holy Spirit",
            DoctrinalCategory.ANTHROPOLOGY: "Addresses human nature",
            DoctrinalCategory.HAMARTIOLOGY: "Concerns sin and its effects",
            DoctrinalCategory.SOTERIOLOGY: "Relates to salvation",
            DoctrinalCategory.ECCLESIOLOGY: "Pertains to the Church",
            DoctrinalCategory.ESCHATOLOGY: "Concerns last things",
            DoctrinalCategory.ANGELOLOGY: "Involves spiritual beings",
            DoctrinalCategory.BIBLIOLOGY: "Concerns Scripture itself"
        }
        return category_assertions.get(category, "General theological content")

    def _determine_weight(
        self,
        text: str,
        category: DoctrinalCategory
    ) -> TheologicalWeight:
        """Determine theological weight of text."""
        text_lower = text.lower()

        # Check for foundational indicators
        foundational_markers = ["is", "am", "are", "shall be", "must"]
        supporting_markers = ["therefore", "for", "because", "so that"]

        if any(marker in text_lower for marker in foundational_markers):
            return TheologicalWeight.FOUNDATIONAL
        elif any(marker in text_lower for marker in supporting_markers):
            return TheologicalWeight.SUPPORTING

        return TheologicalWeight.CONTEXTUAL

    def _find_related_doctrines(
        self,
        category: DoctrinalCategory
    ) -> List[str]:
        """Find related doctrinal categories."""
        relations = {
            DoctrinalCategory.THEOLOGY_PROPER: ["christology", "pneumatology"],
            DoctrinalCategory.CHRISTOLOGY: ["soteriology", "ecclesiology"],
            DoctrinalCategory.PNEUMATOLOGY: ["ecclesiology", "soteriology"],
            DoctrinalCategory.ANTHROPOLOGY: ["hamartiology", "soteriology"],
            DoctrinalCategory.HAMARTIOLOGY: ["soteriology", "anthropology"],
            DoctrinalCategory.SOTERIOLOGY: ["christology", "ecclesiology"],
            DoctrinalCategory.ECCLESIOLOGY: ["soteriology", "eschatology"],
            DoctrinalCategory.ESCHATOLOGY: ["soteriology", "christology"],
            DoctrinalCategory.ANGELOLOGY: ["eschatology", "hamartiology"],
            DoctrinalCategory.BIBLIOLOGY: ["theology_proper", "christology"]
        }
        return relations.get(category, [])

    def _find_creedal_connection(self, text: str) -> Optional[str]:
        """Find creedal connections in text."""
        text_lower = text.lower()

        for phrase, creed in self.CREEDAL_PHRASES.items():
            if phrase in text_lower:
                return creed

        return None

    def _map_to_creeds(self, text: str) -> List[Dict[str, str]]:
        """Map text to creedal statements."""
        connections = []
        text_lower = text.lower()

        for phrase, creed in self.CREEDAL_PHRASES.items():
            if phrase in text_lower:
                connections.append({
                    "phrase": phrase,
                    "creed": creed
                })

        return connections

    def _analyze_terms(self, text: str) -> List[Dict[str, Any]]:
        """Analyze theological terms in text."""
        terms = []
        text_lower = text.lower()

        # Find significant theological terms
        all_terms = set()
        for category_terms in self.CATEGORY_TERMS.values():
            all_terms.update(category_terms)

        for term in all_terms:
            if term in text_lower:
                terms.append({
                    "term": term,
                    "significance": "high" if term in [
                        "god", "christ", "spirit", "salvation", "grace"
                    ] else "standard"
                })

        return terms

    def _assess_significance(
        self,
        categories: List[DoctrinalCategory],
        assertions: List[DoctrinalAssertion]
    ) -> Dict[str, Any]:
        """Assess overall theological significance."""
        high_significance_categories = {
            DoctrinalCategory.THEOLOGY_PROPER,
            DoctrinalCategory.CHRISTOLOGY,
            DoctrinalCategory.SOTERIOLOGY
        }

        is_high = any(c in high_significance_categories for c in categories)
        foundational_count = sum(
            1 for a in assertions
            if a.weight == TheologicalWeight.FOUNDATIONAL
        )

        return {
            "level": "high" if is_high else "standard",
            "foundational_assertions": foundational_count,
            "category_count": len(categories),
            "dogmatic_importance": is_high and foundational_count > 0
        }

    def _check_trinitarian(self, text: str) -> Dict[str, Any]:
        """Check for Trinitarian content."""
        text_lower = text.lower()

        father_terms = ["father", "god"]
        son_terms = ["son", "christ", "jesus", "word"]
        spirit_terms = ["spirit", "holy spirit"]

        has_father = any(t in text_lower for t in father_terms)
        has_son = any(t in text_lower for t in son_terms)
        has_spirit = any(t in text_lower for t in spirit_terms)

        persons_count = sum([has_father, has_son, has_spirit])

        return {
            "is_trinitarian": persons_count >= 2,
            "persons_referenced": persons_count,
            "has_father": has_father,
            "has_son": has_son,
            "has_spirit": has_spirit
        }

    def _check_soteriological(self, text: str) -> Dict[str, Any]:
        """Check for soteriological content."""
        text_lower = text.lower()

        soteriological_terms = [
            "save", "salvation", "redeem", "justify", "sanctify",
            "grace", "faith", "forgive", "reconcile"
        ]

        found_terms = [t for t in soteriological_terms if t in text_lower]

        return {
            "has_soteriological_content": len(found_terms) > 0,
            "terms_found": found_terms,
            "density": len(found_terms) / len(soteriological_terms)
        }

    def _calculate_confidence(
        self,
        categories: List[DoctrinalCategory],
        assertions: List[DoctrinalAssertion]
    ) -> float:
        """Calculate confidence score."""
        confidence = 0.5

        if categories:
            confidence += 0.2

        if assertions:
            confidence += 0.2
            # Boost for creedal connections
            creedal_count = sum(
                1 for a in assertions if a.creedal_connection
            )
            confidence += creedal_count * 0.05

        return min(1.0, confidence)

    async def validate(self, result: ExtractionResult) -> bool:
        """Validate extraction result."""
        data = result.data
        return "categories" in data and "assertions" in data

    def get_dependencies(self) -> List[str]:
        """Get agent dependencies."""
        return ["grammateus", "semantikos"]

"""
BIBLOS v2 - PATROLOGOS Agent

Patristic interpretation analysis for biblical texts.
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


class PatristicSchool(Enum):
    """Major patristic schools of interpretation."""
    ALEXANDRIAN = "alexandrian"  # Allegorical emphasis
    ANTIOCHENE = "antiochene"  # Literal/historical emphasis
    CAPPADOCIAN = "cappadocian"  # Balanced approach
    LATIN_WEST = "latin_west"  # Western Fathers
    SYRIAN = "syrian"  # Syriac tradition
    DESERT = "desert"  # Desert Fathers


class InterpretiveSense(Enum):
    """Fourfold sense of Scripture."""
    LITERAL = "literal"  # Historical/grammatical
    ALLEGORICAL = "allegorical"  # Christological/typological
    TROPOLOGICAL = "tropological"  # Moral/ethical
    ANAGOGICAL = "anagogical"  # Eschatological/heavenly


@dataclass
class PatristicReference:
    """Reference to a patristic source."""
    father: str
    work: str
    citation: str
    school: PatristicSchool
    sense: InterpretiveSense
    relevance_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "father": self.father,
            "work": self.work,
            "citation": self.citation,
            "school": self.school.value,
            "sense": self.sense.value,
            "relevance_score": self.relevance_score
        }


class PatrologosAgent(BaseExtractionAgent):
    """
    PATROLOGOS - Patristic interpretation agent.

    Performs:
    - Patristic citation matching
    - School identification
    - Fourfold sense classification
    - Consensus patrum analysis
    - Theological theme extraction
    """

    # Major Church Fathers and their characteristics
    FATHERS = {
        "chrysostom": {
            "name": "John Chrysostom",
            "school": PatristicSchool.ANTIOCHENE,
            "emphasis": ["moral", "practical", "homiletical"],
            "era": "4th century"
        },
        "origen": {
            "name": "Origen of Alexandria",
            "school": PatristicSchool.ALEXANDRIAN,
            "emphasis": ["allegorical", "spiritual", "philosophical"],
            "era": "3rd century"
        },
        "augustine": {
            "name": "Augustine of Hippo",
            "school": PatristicSchool.LATIN_WEST,
            "emphasis": ["theological", "philosophical", "pastoral"],
            "era": "4th-5th century"
        },
        "basil": {
            "name": "Basil the Great",
            "school": PatristicSchool.CAPPADOCIAN,
            "emphasis": ["ascetical", "cosmological", "liturgical"],
            "era": "4th century"
        },
        "gregory_nyssa": {
            "name": "Gregory of Nyssa",
            "school": PatristicSchool.CAPPADOCIAN,
            "emphasis": ["mystical", "philosophical", "allegorical"],
            "era": "4th century"
        },
        "gregory_nazianzen": {
            "name": "Gregory the Theologian",
            "school": PatristicSchool.CAPPADOCIAN,
            "emphasis": ["trinitarian", "christological", "rhetorical"],
            "era": "4th century"
        },
        "athanasius": {
            "name": "Athanasius of Alexandria",
            "school": PatristicSchool.ALEXANDRIAN,
            "emphasis": ["christological", "soteriological", "polemical"],
            "era": "4th century"
        },
        "cyril_alexandria": {
            "name": "Cyril of Alexandria",
            "school": PatristicSchool.ALEXANDRIAN,
            "emphasis": ["christological", "typological", "eucharistic"],
            "era": "5th century"
        },
        "jerome": {
            "name": "Jerome",
            "school": PatristicSchool.LATIN_WEST,
            "emphasis": ["textual", "philological", "historical"],
            "era": "4th-5th century"
        },
        "ephrem": {
            "name": "Ephrem the Syrian",
            "school": PatristicSchool.SYRIAN,
            "emphasis": ["poetic", "typological", "mariological"],
            "era": "4th century"
        }
    }

    # Theological keywords for matching
    THEOLOGICAL_THEMES = {
        "christology": ["christ", "son", "logos", "incarnation", "messiah"],
        "trinity": ["father", "son", "spirit", "trinity", "consubstantial"],
        "soteriology": ["salvation", "redemption", "atonement", "grace"],
        "ecclesiology": ["church", "body", "bride", "assembly"],
        "eschatology": ["kingdom", "judgment", "resurrection", "eternal"],
        "pneumatology": ["spirit", "paraclete", "gifts", "indwelling"],
        "anthropology": ["image", "likeness", "soul", "body", "flesh"],
        "mariology": ["mary", "virgin", "theotokos", "mother"]
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="patrologos",
                extraction_type=ExtractionType.THEOLOGICAL,
                batch_size=200,
                min_confidence=0.6
            )
        super().__init__(config)
        self.logger = logging.getLogger("biblos.agents.patrologos")

    async def extract(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> ExtractionResult:
        """Extract patristic analysis from verse."""
        # Identify theological themes
        themes = self._identify_themes(text)

        # Match potential patristic interpretations
        references = self._match_patristic_sources(text, themes, context)

        # Classify interpretive senses
        senses = self._classify_senses(text, context)

        # Identify consensus patrum
        consensus = self._analyze_consensus(references)

        data = {
            "themes": themes,
            "patristic_references": [r.to_dict() for r in references],
            "interpretive_senses": [s.value for s in senses],
            "consensus": consensus,
            "primary_school": self._determine_primary_school(references),
            "liturgical_usage": self._check_liturgical_usage(verse_id, context),
            "doctrinal_significance": self._assess_doctrinal_significance(themes)
        }

        confidence = self._calculate_confidence(references, themes)

        return ExtractionResult(
            agent_name=self.config.name,
            extraction_type=self.config.extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.COMPLETED,
            data=data,
            confidence=confidence
        )

    def _identify_themes(self, text: str) -> List[str]:
        """Identify theological themes in text."""
        themes = []
        text_lower = text.lower()

        for theme, keywords in self.THEOLOGICAL_THEMES.items():
            if any(kw in text_lower for kw in keywords):
                themes.append(theme)

        return themes

    def _match_patristic_sources(
        self,
        text: str,
        themes: List[str],
        context: Dict[str, Any]
    ) -> List[PatristicReference]:
        """Match text to potential patristic sources."""
        references = []
        text_lower = text.lower()

        # Get existing patristic citations from context
        existing_citations = context.get("patristic_citations", [])

        for citation in existing_citations:
            father_key = citation.get("father", "").lower().replace(" ", "_")
            if father_key in self.FATHERS:
                father_info = self.FATHERS[father_key]
                references.append(PatristicReference(
                    father=father_info["name"],
                    work=citation.get("work", "Unknown"),
                    citation=citation.get("text", ""),
                    school=father_info["school"],
                    sense=self._determine_sense(citation.get("text", "")),
                    relevance_score=citation.get("relevance", 0.7)
                ))

        # Suggest likely commentators based on themes
        for theme in themes:
            suggested = self._suggest_fathers_for_theme(theme)
            for father_key in suggested:
                if father_key in self.FATHERS:
                    father_info = self.FATHERS[father_key]
                    # Don't duplicate
                    if not any(r.father == father_info["name"] for r in references):
                        references.append(PatristicReference(
                            father=father_info["name"],
                            work="(suggested)",
                            citation="",
                            school=father_info["school"],
                            sense=InterpretiveSense.LITERAL,
                            relevance_score=0.5
                        ))

        return references

    def _suggest_fathers_for_theme(self, theme: str) -> List[str]:
        """Suggest Church Fathers for a given theme."""
        theme_fathers = {
            "christology": ["cyril_alexandria", "athanasius", "gregory_nazianzen"],
            "trinity": ["gregory_nazianzen", "basil", "augustine"],
            "soteriology": ["athanasius", "augustine", "chrysostom"],
            "ecclesiology": ["cyprian", "augustine", "chrysostom"],
            "eschatology": ["origen", "augustine", "ephrem"],
            "pneumatology": ["basil", "cyril_alexandria", "augustine"],
            "anthropology": ["gregory_nyssa", "origen", "augustine"],
            "mariology": ["ephrem", "cyril_alexandria", "john_damascene"]
        }
        return theme_fathers.get(theme, ["chrysostom", "augustine"])

    def _determine_sense(self, text: str) -> InterpretiveSense:
        """Determine the interpretive sense of a text."""
        text_lower = text.lower()

        allegorical_markers = ["signifies", "represents", "typifies", "symbol"]
        moral_markers = ["we should", "we must", "teaches us", "virtue", "sin"]
        eschatological_markers = ["heaven", "eternal", "kingdom", "final"]

        if any(m in text_lower for m in allegorical_markers):
            return InterpretiveSense.ALLEGORICAL
        if any(m in text_lower for m in moral_markers):
            return InterpretiveSense.TROPOLOGICAL
        if any(m in text_lower for m in eschatological_markers):
            return InterpretiveSense.ANAGOGICAL

        return InterpretiveSense.LITERAL

    def _classify_senses(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> List[InterpretiveSense]:
        """Classify all applicable interpretive senses."""
        senses = [InterpretiveSense.LITERAL]  # Always has literal sense
        text_lower = text.lower()

        # Check for typological/allegorical potential
        if context.get("has_type", False) or any(
            word in text_lower for word in ["like", "as", "image", "shadow"]
        ):
            senses.append(InterpretiveSense.ALLEGORICAL)

        # Check for moral teaching
        if any(word in text_lower for word in [
            "commandment", "shall", "should", "righteous", "wicked"
        ]):
            senses.append(InterpretiveSense.TROPOLOGICAL)

        # Check for eschatological content
        if any(word in text_lower for word in [
            "kingdom", "eternal", "heaven", "judgment", "resurrection"
        ]):
            senses.append(InterpretiveSense.ANAGOGICAL)

        return senses

    def _analyze_consensus(
        self,
        references: List[PatristicReference]
    ) -> Dict[str, Any]:
        """Analyze consensus among Church Fathers."""
        if not references:
            return {"has_consensus": False, "schools_represented": []}

        schools = [r.school.value for r in references]
        senses = [r.sense.value for r in references]

        return {
            "has_consensus": len(set(senses)) == 1,
            "schools_represented": list(set(schools)),
            "dominant_sense": max(set(senses), key=senses.count) if senses else None,
            "father_count": len(references)
        }

    def _determine_primary_school(
        self,
        references: List[PatristicReference]
    ) -> Optional[str]:
        """Determine the primary patristic school."""
        if not references:
            return None

        schools = [r.school.value for r in references]
        return max(set(schools), key=schools.count)

    def _check_liturgical_usage(
        self,
        verse_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check for liturgical usage of the verse."""
        return {
            "in_lectionary": context.get("lectionary_usage", False),
            "liturgical_occasions": context.get("liturgical_occasions", []),
            "hymnic_references": context.get("hymnic_refs", [])
        }

    def _assess_doctrinal_significance(
        self,
        themes: List[str]
    ) -> Dict[str, Any]:
        """Assess the doctrinal significance of the verse."""
        high_significance = ["trinity", "christology", "soteriology"]
        medium_significance = ["ecclesiology", "pneumatology", "mariology"]

        level = "standard"
        if any(t in high_significance for t in themes):
            level = "high"
        elif any(t in medium_significance for t in themes):
            level = "medium"

        return {
            "level": level,
            "themes": themes,
            "ecumenical_relevance": any(t in ["trinity", "christology"] for t in themes)
        }

    def _calculate_confidence(
        self,
        references: List[PatristicReference],
        themes: List[str]
    ) -> float:
        """Calculate confidence score."""
        confidence = 0.5

        if references:
            confidence += 0.2
            # Boost for actual citations vs suggestions
            actual_refs = [r for r in references if r.citation]
            confidence += len(actual_refs) * 0.05

        if themes:
            confidence += 0.1

        return min(1.0, confidence)

    async def validate(self, result: ExtractionResult) -> bool:
        """Validate extraction result."""
        data = result.data
        return "themes" in data and "patristic_references" in data

    def get_dependencies(self) -> List[str]:
        """Get agent dependencies."""
        return ["grammateus", "semantikos"]

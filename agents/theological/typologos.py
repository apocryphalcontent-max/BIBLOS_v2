"""
BIBLOS v2 - TYPOLOGOS Agent

Typological analysis agent for identifying OT types and NT antitypes.
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


class TypeCategory(Enum):
    """Categories of biblical types."""
    PERSONAL = "personal"  # Adam, Moses, David, etc.
    INSTITUTIONAL = "institutional"  # Temple, priesthood, sacrifices
    EVENTIVE = "eventive"  # Exodus, flood, wilderness
    MATERIAL = "material"  # Ark, manna, bronze serpent
    TEMPORAL = "temporal"  # Sabbath, feasts, Jubilee


class TypeRelation(Enum):
    """Nature of type-antitype relationship."""
    CORRESPONDENCE = "correspondence"  # Direct parallel
    ESCALATION = "escalation"  # Greater fulfillment
    CONTRAST = "contrast"  # Antithetical fulfillment
    RECAPITULATION = "recapitulation"  # Pattern repetition


@dataclass
class TypeConnection:
    """A typological connection between passages."""
    type_ref: str  # OT reference
    antitype_ref: str  # NT reference
    category: TypeCategory
    relation: TypeRelation
    type_element: str  # What in OT
    antitype_element: str  # What in NT
    patristic_witness: List[str]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type_ref": self.type_ref,
            "antitype_ref": self.antitype_ref,
            "category": self.category.value,
            "relation": self.relation.value,
            "type_element": self.type_element,
            "antitype_element": self.antitype_element,
            "patristic_witness": self.patristic_witness,
            "confidence": self.confidence
        }


class TypologosAgent(BaseExtractionAgent):
    """
    TYPOLOGOS - Typological analysis agent.

    Performs:
    - Type identification in OT texts
    - Antitype matching in NT texts
    - Typological pattern analysis
    - Patristic typological witness
    - Escalation/fulfillment mapping
    """

    # Major typological figures
    TYPE_FIGURES = {
        "adam": {
            "category": TypeCategory.PERSONAL,
            "antitype": "Christ",
            "relation": TypeRelation.CONTRAST,
            "keywords": ["adam", "first man", "garden"],
            "nt_refs": ["ROM.5.14", "1CO.15.22", "1CO.15.45"],
            "patristic": ["Irenaeus", "Cyril of Alexandria"]
        },
        "moses": {
            "category": TypeCategory.PERSONAL,
            "antitype": "Christ",
            "relation": TypeRelation.ESCALATION,
            "keywords": ["moses", "prophet", "lawgiver", "mediator"],
            "nt_refs": ["HEB.3.2-6", "JHN.1.17", "ACT.3.22"],
            "patristic": ["Origen", "Gregory of Nyssa"]
        },
        "david": {
            "category": TypeCategory.PERSONAL,
            "antitype": "Christ",
            "relation": TypeRelation.ESCALATION,
            "keywords": ["david", "king", "shepherd", "son of"],
            "nt_refs": ["MAT.1.1", "ACT.2.25-31", "REV.22.16"],
            "patristic": ["Augustine", "Chrysostom"]
        },
        "melchizedek": {
            "category": TypeCategory.PERSONAL,
            "antitype": "Christ",
            "relation": TypeRelation.CORRESPONDENCE,
            "keywords": ["melchizedek", "priest", "king", "salem"],
            "nt_refs": ["HEB.5.6", "HEB.7.1-17"],
            "patristic": ["Cyril of Alexandria", "Theodoret"]
        },
        "isaac": {
            "category": TypeCategory.PERSONAL,
            "antitype": "Christ",
            "relation": TypeRelation.CORRESPONDENCE,
            "keywords": ["isaac", "only son", "sacrifice", "moriah"],
            "nt_refs": ["GAL.4.28", "HEB.11.17-19", "JHN.3.16"],
            "patristic": ["Origen", "Ephrem"]
        },
        "joseph": {
            "category": TypeCategory.PERSONAL,
            "antitype": "Christ",
            "relation": TypeRelation.CORRESPONDENCE,
            "keywords": ["joseph", "sold", "exalted", "savior"],
            "nt_refs": ["ACT.7.9-14"],
            "patristic": ["Ambrose", "Ephrem"]
        },
        "passover_lamb": {
            "category": TypeCategory.INSTITUTIONAL,
            "antitype": "Christ",
            "relation": TypeRelation.ESCALATION,
            "keywords": ["passover", "lamb", "blood", "doorpost"],
            "nt_refs": ["JHN.1.29", "1CO.5.7", "1PE.1.19"],
            "patristic": ["Cyril of Alexandria", "Melito of Sardis"]
        },
        "temple": {
            "category": TypeCategory.INSTITUTIONAL,
            "antitype": "Christ/Church",
            "relation": TypeRelation.ESCALATION,
            "keywords": ["temple", "house", "dwelling", "sanctuary"],
            "nt_refs": ["JHN.2.19-21", "1CO.3.16", "EPH.2.21"],
            "patristic": ["Origen", "Cyril of Jerusalem"]
        },
        "tabernacle": {
            "category": TypeCategory.INSTITUTIONAL,
            "antitype": "Christ",
            "relation": TypeRelation.CORRESPONDENCE,
            "keywords": ["tabernacle", "tent", "dwelling", "glory"],
            "nt_refs": ["JHN.1.14", "HEB.8.2", "HEB.9.11"],
            "patristic": ["Gregory of Nyssa", "Cyril of Alexandria"]
        },
        "exodus": {
            "category": TypeCategory.EVENTIVE,
            "antitype": "Salvation in Christ",
            "relation": TypeRelation.ESCALATION,
            "keywords": ["exodus", "deliver", "egypt", "redeem"],
            "nt_refs": ["LUK.9.31", "1CO.10.1-4", "HEB.3.16"],
            "patristic": ["Origen", "Gregory of Nyssa"]
        },
        "flood": {
            "category": TypeCategory.EVENTIVE,
            "antitype": "Baptism/Judgment",
            "relation": TypeRelation.CORRESPONDENCE,
            "keywords": ["flood", "water", "ark", "noah"],
            "nt_refs": ["1PE.3.20-21", "2PE.2.5", "MAT.24.37-39"],
            "patristic": ["Justin Martyr", "Cyprian"]
        },
        "manna": {
            "category": TypeCategory.MATERIAL,
            "antitype": "Christ/Eucharist",
            "relation": TypeRelation.ESCALATION,
            "keywords": ["manna", "bread", "heaven", "wilderness"],
            "nt_refs": ["JHN.6.31-35", "JHN.6.48-51", "REV.2.17"],
            "patristic": ["Origen", "Cyril of Alexandria"]
        },
        "bronze_serpent": {
            "category": TypeCategory.MATERIAL,
            "antitype": "Christ on Cross",
            "relation": TypeRelation.CORRESPONDENCE,
            "keywords": ["serpent", "pole", "look", "live"],
            "nt_refs": ["JHN.3.14-15"],
            "patristic": ["Justin Martyr", "Chrysostom"]
        },
        "sabbath": {
            "category": TypeCategory.TEMPORAL,
            "antitype": "Eternal Rest",
            "relation": TypeRelation.ESCALATION,
            "keywords": ["sabbath", "rest", "seventh", "cease"],
            "nt_refs": ["HEB.4.9-11", "COL.2.16-17"],
            "patristic": ["Origen", "Augustine"]
        }
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="typologos",
                extraction_type=ExtractionType.TYPOLOGICAL,
                batch_size=200,
                min_confidence=0.65
            )
        super().__init__(config)
        self.logger = logging.getLogger("biblos.agents.typologos")

    async def extract(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> ExtractionResult:
        """Extract typological analysis from verse."""
        is_ot = self._is_old_testament(verse_id)

        # Identify types or antitypes
        if is_ot:
            types = self._identify_types(text, verse_id)
            connections = self._build_type_connections(types, verse_id)
        else:
            antitypes = self._identify_antitypes(text, verse_id)
            connections = self._build_antitype_connections(antitypes, verse_id)

        # Analyze typological patterns
        patterns = self._analyze_patterns(text, connections)

        # Check for explicit NT quotations/allusions
        explicit_refs = self._find_explicit_references(text, context)

        data = {
            "is_ot": is_ot,
            "connections": [c.to_dict() for c in connections],
            "patterns": patterns,
            "explicit_references": explicit_refs,
            "typological_density": self._calculate_density(connections),
            "categories": list(set(c.category.value for c in connections)),
            "christological_focus": self._assess_christological_focus(connections)
        }

        confidence = self._calculate_confidence(connections, explicit_refs)

        return ExtractionResult(
            agent_name=self.config.name,
            extraction_type=self.config.extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.COMPLETED,
            data=data,
            confidence=confidence
        )

    def _is_old_testament(self, verse_id: str) -> bool:
        """Check if verse is from Old Testament."""
        ot_books = {
            "GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT",
            "1SA", "2SA", "1KI", "2KI", "1CH", "2CH", "EZR", "NEH",
            "EST", "JOB", "PSA", "PRO", "ECC", "SNG", "ISA", "JER",
            "LAM", "EZK", "DAN", "HOS", "JOL", "AMO", "OBA", "JON",
            "MIC", "NAM", "HAB", "ZEP", "HAG", "ZEC", "MAL"
        }
        book = verse_id.split(".")[0]
        return book in ot_books

    def _identify_types(
        self,
        text: str,
        verse_id: str
    ) -> List[Dict[str, Any]]:
        """Identify typological figures in OT text."""
        types = []
        text_lower = text.lower()

        for type_key, type_info in self.TYPE_FIGURES.items():
            for keyword in type_info["keywords"]:
                if keyword in text_lower:
                    types.append({
                        "type_key": type_key,
                        "info": type_info,
                        "keyword_match": keyword
                    })
                    break

        return types

    def _identify_antitypes(
        self,
        text: str,
        verse_id: str
    ) -> List[Dict[str, Any]]:
        """Identify antitype references in NT text."""
        antitypes = []
        text_lower = text.lower()

        for type_key, type_info in self.TYPE_FIGURES.items():
            # Check if this verse is a known antitype reference
            if verse_id in type_info.get("nt_refs", []):
                antitypes.append({
                    "type_key": type_key,
                    "info": type_info,
                    "is_explicit": True
                })
            # Also check keywords in NT context
            elif any(kw in text_lower for kw in type_info["keywords"]):
                antitypes.append({
                    "type_key": type_key,
                    "info": type_info,
                    "is_explicit": False
                })

        return antitypes

    def _build_type_connections(
        self,
        types: List[Dict[str, Any]],
        verse_id: str
    ) -> List[TypeConnection]:
        """Build type connections for OT verse."""
        connections = []

        for type_data in types:
            info = type_data["info"]
            for nt_ref in info.get("nt_refs", []):
                connections.append(TypeConnection(
                    type_ref=verse_id,
                    antitype_ref=nt_ref,
                    category=info["category"],
                    relation=info["relation"],
                    type_element=type_data["type_key"],
                    antitype_element=info["antitype"],
                    patristic_witness=info.get("patristic", []),
                    confidence=0.8
                ))

        return connections

    def _build_antitype_connections(
        self,
        antitypes: List[Dict[str, Any]],
        verse_id: str
    ) -> List[TypeConnection]:
        """Build antitype connections for NT verse."""
        connections = []

        for antitype_data in antitypes:
            info = antitype_data["info"]
            confidence = 0.9 if antitype_data["is_explicit"] else 0.6

            connections.append(TypeConnection(
                type_ref="(OT source)",
                antitype_ref=verse_id,
                category=info["category"],
                relation=info["relation"],
                type_element=antitype_data["type_key"],
                antitype_element=info["antitype"],
                patristic_witness=info.get("patristic", []),
                confidence=confidence
            ))

        return connections

    def _analyze_patterns(
        self,
        text: str,
        connections: List[TypeConnection]
    ) -> Dict[str, Any]:
        """Analyze typological patterns."""
        if not connections:
            return {"pattern_count": 0, "dominant_category": None}

        categories = [c.category.value for c in connections]
        relations = [c.relation.value for c in connections]

        return {
            "pattern_count": len(connections),
            "dominant_category": max(set(categories), key=categories.count),
            "dominant_relation": max(set(relations), key=relations.count),
            "unique_categories": list(set(categories)),
            "has_escalation": TypeRelation.ESCALATION.value in relations
        }

    def _find_explicit_references(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find explicit typological references."""
        refs = []
        text_lower = text.lower()

        # Markers of explicit typological reference
        markers = [
            "as it is written", "fulfillment", "fulfilled",
            "according to", "as moses", "as david", "like"
        ]

        for marker in markers:
            if marker in text_lower:
                refs.append({
                    "marker": marker,
                    "type": "explicit"
                })

        # Check context for known OT quotations
        if context.get("ot_quotations"):
            for quote in context["ot_quotations"]:
                refs.append({
                    "source": quote.get("source"),
                    "type": "quotation"
                })

        return refs

    def _calculate_density(
        self,
        connections: List[TypeConnection]
    ) -> float:
        """Calculate typological density."""
        if not connections:
            return 0.0

        # Higher density with more connections and diverse categories
        base = len(connections) * 0.2
        categories = len(set(c.category for c in connections))
        diversity_bonus = categories * 0.1

        return min(1.0, base + diversity_bonus)

    def _assess_christological_focus(
        self,
        connections: List[TypeConnection]
    ) -> Dict[str, Any]:
        """Assess Christological focus of typology."""
        if not connections:
            return {"is_christological": False, "focus": None}

        christ_types = [c for c in connections if "Christ" in c.antitype_element]

        return {
            "is_christological": len(christ_types) > 0,
            "focus": "Christological" if christ_types else "Other",
            "christ_connection_count": len(christ_types)
        }

    def _calculate_confidence(
        self,
        connections: List[TypeConnection],
        explicit_refs: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score."""
        confidence = 0.5

        if connections:
            avg_conf = sum(c.confidence for c in connections) / len(connections)
            confidence += avg_conf * 0.3

        if explicit_refs:
            confidence += 0.2

        return min(1.0, confidence)

    async def validate(self, result: ExtractionResult) -> bool:
        """Validate extraction result."""
        data = result.data
        return "connections" in data and "is_ot" in data

    def get_dependencies(self) -> List[str]:
        """Get agent dependencies."""
        return ["grammateus", "semantikos", "patrologos"]
